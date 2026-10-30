"""
validation_analysis.py -- analysis of the validation session run for the ICCE review (points 1, 2, 5, 6, 7, 10).

Usage:
    python validation_analysis.py --validation benchmark_validation.csv ^
        --batches validation_trace_batches.csv --frames validation_trace_frames.csv --hwinfo hwinfo_validation.CSV ^
        --main benchmark_steady_state.csv --main-hwinfo HWiN_sensor_log.CSV ^
        --control benchmark_smt_control.csv --control-hwinfo HWiN_sensor_log_smt.CSV --out out

The validation session ran 16 of the 180 configurations x 9 conditions x 3 trials (432 tests) with the main
session's calibration. Every window closed only when the original rule (>= 1,000 game frames and >= 50 batches) AND
20 s had both been met, and every game frame and AI batch was traced, so each test is analysed over two windows
rebuilt from the traces:
  * orig    -- up to the time the original rule was met (the end of frame 1,000 and of batch 50, or 2.5 s for the
               no-AI controls): comparable with the main and control sessions (replication, point 10);
  * equal20 -- the first 20 s of every window: equal exposure for every condition (point 7). Used for the effects.
Frames and batches that end after the window's end are left out, as in the harness.

Conditions and matched controls (same block):
  E_CORE, P_SHARED, UNPINNED, E_CORE_1LP     vs BASELINE (LP 1 worker running)
  P_SMT, E_CORE_SMT_IDLE, P_SHARED_SMT_IDLE  vs SMT_IDLE (LP 1 worker parked): the same background work (point 1)
Batch timing (point 6): release = scheduled start of the update period, start delay = start - release, batch
latency = end - start, deadline = release + 50 ms. A missed deadline is a late start when the batch itself took at
most 50 ms, and a long batch otherwise. Residence (points 2 and 5): the LP the AI thread was on just before and just
after each batch. On-CPU share: thread cycles over batch time (about 1 when the thread ran the whole batch).
Confidence intervals: 95% percentile bootstrap of the median over the 16 configurations (2,000 resamples).
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from audit_analysis import BLOCK, CONFIG, cluster_boot_median, load_bench, load_hwinfo, refresh_label, session_stats

PERIOD_MS = 50.0
EQUAL_S = 20.0
CONTROL_OF = {"E_CORE": "BASELINE", "P_SHARED": "BASELINE", "UNPINNED": "BASELINE", "E_CORE_1LP": "BASELINE",
              "P_SMT": "SMT_IDLE", "E_CORE_SMT_IDLE": "SMT_IDLE", "P_SHARED_SMT_IDLE": "SMT_IDLE"}
AI_CONDS = list(CONTROL_OF)
GAME_LP, SIB_LP, HOUSE_LP = 0, 1, 19


def p99(v):
    """Nearest-rank 99th percentile, as in the harness."""
    v = np.sort(np.asarray(v, float))
    n = len(v)
    return v[min(max((n * 99 + 99) // 100, 1), n) - 1] if n else np.nan


def lp_class(lp):
    return np.select([lp == GAME_LP, lp == SIB_LP, (lp >= 2) & (lp <= 11), (lp >= 12) & (lp <= 18), lp == HOUSE_LP],
                     ["LP 0 (game)", "LP 1 (sibling)", "LP 2-11 (P, workers)", "LP 12-18 (E)", "LP 19 (E, telemetry)"],
                     "other")


# ----------------------------------------------------------------------------------------------------------
# Windows rebuilt from the traces
# ----------------------------------------------------------------------------------------------------------
def load_traces(frames_path, batches_path, tests):
    f = pd.read_csv(frames_path)
    b = pd.read_csv(batches_path)
    keep = set(tests.Trace_ID)
    f, b = f[f.Trace_ID.isin(keep)].copy(), b[b.Trace_ID.isin(keep)].copy()
    f["dur"] = (f.End_ns - f.Start_ns) / 1e6
    b["lat"] = (b.End_ns - b.Start_ns) / 1e6
    b["delay"] = (b.Start_ns - b.Release_ns) / 1e6
    b["late_start_miss"] = (b.Missed == 1) & (b.lat <= PERIOD_MS)
    b["oncpu"] = b.Thread_Cycles / TSC_HZ / (b.lat / 1000)
    return f, b


def window_ends(tests, f, b):
    """End of each window in ns from the window start: 'orig' (original rule) and 'equal20'."""
    f1000 = f[f.Frame == 999].set_index("Trace_ID").End_ns
    b50 = b[b.Batch == 49].set_index("Trace_ID").End_ns
    t = tests.set_index("Trace_ID")
    has_ai = t.AI_Batches.notna()
    orig = pd.Series(np.where(has_ai, np.maximum(f1000.reindex(t.index), b50.reindex(t.index)),
                              np.maximum(f1000.reindex(t.index), 2.5e9)), index=t.index)
    return {"orig": orig, "equal20": pd.Series(EQUAL_S * 1e9, index=t.index)}


def window_stats(tests, f, b, end):
    fe = f[f.End_ns <= f.Trace_ID.map(end)]
    be = b[b.End_ns <= b.Trace_ID.map(end)]
    g = fe.groupby("Trace_ID").dur
    out = pd.DataFrame({"window_s": end / 1e9, "frames": g.size(), "game_mean": g.mean(), "game_p99": g.apply(p99)})
    a = be.groupby("Trace_ID")
    ai = pd.DataFrame({"updates": a.size(), "ai_mean": a.lat.mean(), "ai_p99": a.lat.apply(p99),
                       "missed": a.Missed.sum(), "late_start_misses": a.late_start_miss.sum(),
                       "delay_median": a.delay.median(), "delay_p99": a.delay.apply(p99),
                       "busy_ms": a.lat.sum(), "cycles": a.Thread_Cycles.sum()})
    out = out.join(ai)
    out["oncpu_share"] = out.cycles / TSC_HZ / (out.busy_ms / 1000)
    t = tests.set_index("Trace_ID")[BLOCK + ["Condition", "mflop_batch", "Block_Pos", "ts0"]]
    return t.join(out).reset_index()


def wide(ws):
    w = ws.pivot_table(index=BLOCK, columns="Condition",
                       values=["game_mean", "game_p99", "ai_mean", "ai_p99", "missed", "updates", "mflop_batch"])
    w.columns = [f"{a}|{c}" for a, c in w.columns]
    w = w.reset_index()
    w["mflop"] = w["mflop_batch|BASELINE"] if "mflop_batch|BASELINE" in w else w.filter(like="mflop_batch").max(axis=1)
    return w


def clusters_of(df):
    return df[CONFIG].astype(int).astype(str).agg("|".join, axis=1).values


def config_p(v, df):
    m = pd.Series(np.asarray(v, float)).groupby(clusters_of(df)).median().dropna()
    return float(stats.wilcoxon(m).pvalue) if (m != 0).any() else 1.0


# ----------------------------------------------------------------------------------------------------------
# Tables
# ----------------------------------------------------------------------------------------------------------
def quality_table(tests, ws_orig, ws_eq):
    rows = []
    for cond, g in tests.groupby("Condition"):
        o = ws_orig[ws_orig.Condition == cond]
        e = ws_eq[ws_eq.Condition == cond]
        rows.append({"condition": cond, "tests": len(g), "drift_flags": int(g.Drift_Flag.sum()),
                     "truncated": int(g.Truncated.sum()), "pre_test_frame_ms": round(g.Pre_Test_Frame_ms.median(), 3),
                     "window_s_median": round(g.Measure_Window_s.median(), 2), "window_s_max": round(g.Measure_Window_s.max(), 2),
                     "orig_window_s_median": round(o.window_s.median(), 2), "orig_frames_median": int(o.frames.median()),
                     "equal20_frames_median": int(e.frames.median()),
                     "equal20_updates_median": int(e.updates.median()) if e.updates.notna().any() else np.nan,
                     "equal20_updates_min": int(e.updates.min()) if e.updates.notna().any() else np.nan})
    return pd.DataFrame(rows)


def operating_point_table(sessions):
    rows = []
    for name, b, h in sessions:
        s = session_stats(b, h)
        t0, t1 = b.ts0.min(), b.ts1.max()
        hw = h[(h.ts >= t0) & (h.ts <= t1)]
        base = b[b.Condition == "BASELINE"]
        slope = stats.linregress((base.ts0 - t0).dt.total_seconds() / 3600, base.Game_Avg_ms).slope if len(base) > 2 else np.nan
        span_h = (t1 - t0).total_seconds() / 3600
        rows.append({"session": name, "tests": len(b), "hours": round(span_h, 2),
                     "pkg_temp_mean_c": round(s["pkg_temp_mean"], 1), "pkg_temp_max_c": round(s["pkg_temp_max"], 0),
                     "pkg_throttle_pct": round(s["throttle_share"], 1), "pcore0_throttle_pct": round(s["pcore0_throttle_share"], 1),
                     "pkg_power_w": round(s["pkg_power"], 1), "pcore0_clock_mhz": round(s["pcore_clock"], 0),
                     "pl1_w": s["pl1"], "pcore0_clock_refresh": refresh_label(hw, "P-core 0 Clock [MHz]"),
                     "baseline_mean_frame_ms": round(base.Game_Avg_ms.median(), 3) if len(base) else np.nan,
                     "baseline_drift_ms": round(slope * span_h, 3) if np.isfinite(slope) else np.nan,
                     "rapl_pkg_w_baseline": round(base.Mean_RAPL_Package_W.median(), 2) if len(base) else np.nan,
                     "procperf_p_baseline": round(base.Mean_ProcPerf_P_Pct.median(), 1) if len(base) else np.nan})
    return pd.DataFrame(rows)


def effects_table(ws):
    """Equal 20-s windows: each AI condition against its matched control in the same block."""
    w = wide(ws)
    cl = clusters_of(w)
    rows = []
    for cond in AI_CONDS:
        ctl = CONTROL_OF[cond]
        row = {"condition": cond, "control": ctl}
        for key in ["game_mean", "game_p99"]:
            x, c = w[f"{key}|{cond}"], w[f"{key}|{ctl}"]
            row[f"{key}_ms"] = round(x.median(), 2)
            row[f"{key}_control_ms"] = round(c.median(), 2)
            m, lo, hi = cluster_boot_median(x - c, cl)
            row.update({f"{key}_change_ms": round(m, 3), f"{key}_ci_lo": round(lo, 3), f"{key}_ci_hi": round(hi, 3),
                        f"{key}_change_pct": round(((x / c - 1) * 100).median(), 2), f"{key}_p": config_p(x - c, w)})
        rows.append(row)
    return pd.DataFrame(rows)


def same_work_table(ws):
    """Point 1: with the LP 1 worker parked in all three, the placements do the same total work, so their measured
    frame times can be ranked directly. Per-block differences between placements."""
    w = wide(ws)
    cl = clusters_of(w)
    pairs = [("E_CORE_SMT_IDLE", "P_SMT"), ("P_SHARED_SMT_IDLE", "P_SMT"), ("P_SHARED_SMT_IDLE", "E_CORE_SMT_IDLE"),
             ("E_CORE", "UNPINNED"), ("E_CORE_1LP", "E_CORE")]
    rows = []
    for a, z in pairs:
        for key in ["game_mean", "game_p99", "ai_mean", "ai_p99"]:
            d = w[f"{key}|{a}"] - w[f"{key}|{z}"]
            m, lo, hi = cluster_boot_median(d, cl)
            rows.append({"a": a, "b": z, "measure": key, "a_ms": round(w[f"{key}|{a}"].median(), 2),
                         "b_ms": round(w[f"{key}|{z}"].median(), 2), "a_minus_b_ms": round(m, 3), "ci_lo": round(lo, 3),
                         "ci_hi": round(hi, 3), "a_lower_blocks_pct": round((d < 0).mean() * 100, 1), "p": config_p(d, w)})
    return pd.DataFrame(rows)


def deadline_table(ws, window):
    rows = []
    for cond in AI_CONDS:
        a = ws[ws.Condition == cond]
        c = a.groupby(CONFIG).agg(mflop=("mflop_batch", "first"), p99=("ai_p99", "median"), misses=("missed", "sum"),
                                  late=("late_start_misses", "sum"), updates=("updates", "sum"))
        c["fit"], c["met"] = c.p99 <= PERIOD_MS, c.misses == 0
        margin = c.p99[~c.met].min()
        rows.append({"window": window, "condition": cond, "configs": len(c), "configs_fit": int(c.fit.sum()),
                     "configs_met": int(c.met.sum()), "fit_but_missed": int((c.fit & ~c.met).sum()),
                     "lowest_p99_with_miss_ms": round(margin, 2) if np.isfinite(margin) else np.nan,
                     "met_updates_median": int(c[c.met].updates.median()) if c.met.any() else 0,
                     "met_updates_min": int(c[c.met].updates.min()) if c.met.any() else 0,
                     "tests_with_miss_pct": round((a.missed > 0).mean() * 100, 1),
                     "misses": int(c.misses.sum()), "late_start_share_of_misses_pct":
                         round(c.late.sum() / c.misses.sum() * 100, 1) if c.misses.sum() else np.nan})
    return pd.DataFrame(rows)


def per_config_table(ws_eq, ws_orig):
    """Per configuration and AI condition: P99 batch latency and missed deadlines over the three trials, in both windows."""
    parts = []
    for name, ws in [("equal20", ws_eq), ("orig", ws_orig)]:
        a = ws[ws.Condition.isin(AI_CONDS)]
        c = a.groupby(CONFIG + ["Condition"]).agg(mflop=("mflop_batch", "first"), p99=("ai_p99", "median"),
                                                  delay_p99=("delay_p99", "median"), misses=("missed", "sum"),
                                                  late=("late_start_misses", "sum"), updates=("updates", "sum"))
        c.columns = [f"{x}_{name}" if x != "mflop" else x for x in c.columns]
        parts.append(c)
    t = parts[0].join(parts[1].drop(columns="mflop")).reset_index()
    return t.round(2).sort_values(["Condition", "mflop"])


def timing_table(b, tests):
    """Point 6: per AI condition, start delay, batch latency and on-CPU share over the batches of the given windows."""
    x = b.merge(tests[["Trace_ID", "Condition", "mflop_batch"]], on="Trace_ID")
    rows = []
    for cond in AI_CONDS:
        for band, sel in [("all", x.Condition == cond), ("<100 MFLOP", (x.Condition == cond) & (x.mflop_batch < 100)),
                          (">=1000 MFLOP", (x.Condition == cond) & (x.mflop_batch >= 1000))]:
            g = x[sel]
            if g.empty:
                continue
            miss = g[g.Missed == 1]
            rows.append({"condition": cond, "work": band, "batches": len(g),
                         "delay_median_ms": round(g.delay.median(), 3), "delay_p99_ms": round(p99(g.delay), 2),
                         "delay_over_5ms_pct": round((g.delay > 5).mean() * 100, 2),
                         "latency_median_ms": round(g.lat.median(), 2), "oncpu_share_median": round(g.oncpu.median(), 3),
                         "missed": len(miss), "late_start_misses": int(miss.late_start_miss.sum()),
                         "late_start_share_pct": round(miss.late_start_miss.mean() * 100, 1) if len(miss) else np.nan,
                         "missed_delay_median_ms": round(miss.delay.median(), 2) if len(miss) else np.nan})
    return pd.DataFrame(rows)


def residence_table(b, tests):
    """Points 2 and 5: where the AI thread ran (LP just before each batch), and batches that ended on another LP."""
    x = b.merge(tests[["Trace_ID", "Condition"]], on="Trace_ID")
    x["cls"] = lp_class(x.LP_Start.values)
    rows = []
    for cond in AI_CONDS:
        g = x[x.Condition == cond]
        share = g.cls.value_counts(normalize=True) * 100
        per_test = g.groupby("Trace_ID").LP_Start.nunique()
        prev = g.groupby("Trace_ID").LP_End.shift()
        moved_between = (g.LP_Start != prev) & prev.notna()
        row = {"condition": cond, "batches": len(g), "lps_seen": " ".join(map(str, sorted(g.LP_Start.unique()))),
               "distinct_lps_per_test_median": int(per_test.median()),
               "lp_changed_within_batch_pct": round((g.LP_Start != g.LP_End).mean() * 100, 2),
               "lp_changed_between_batches_pct": round(moved_between.mean() * 100, 2)}
        for c in ["LP 0 (game)", "LP 1 (sibling)", "LP 2-11 (P, workers)", "LP 12-18 (E)", "LP 19 (E, telemetry)"]:
            row[c] = round(share.get(c, 0.0), 2)
        rows.append(row)
    return pd.DataFrame(rows)


def unpinned_tests_table(b, tests, ws_eq):
    """UNPINNED per test: share of batches that started on a P-core LP, with the test's cost against its BASELINE."""
    x = b.merge(tests[["Trace_ID", "Condition"]], on="Trace_ID")
    u = x[x.Condition == "UNPINNED"]
    p_share = u.assign(p=u.LP_Start <= 11, lp0=u.LP_Start == GAME_LP).groupby("Trace_ID")[["p", "lp0"]].mean() * 100
    w = ws_eq.set_index("Trace_ID")
    t = w[w.Condition == "UNPINNED"][BLOCK + ["mflop_batch", "game_mean", "game_p99", "ai_mean", "missed"]].join(p_share)
    base = ws_eq[ws_eq.Condition == "BASELINE"].set_index(BLOCK)[["game_mean", "game_p99"]]
    t = t.join(base, on=BLOCK, rsuffix="_base")
    t["mean_change_pct"] = (t.game_mean / t.game_mean_base - 1) * 100
    t["p99_change_pct"] = (t.game_p99 / t.game_p99_base - 1) * 100
    return t.reset_index().round(3).sort_values("p", ascending=False)


def replication_table(ws_orig, main, control):
    """Point 10: validation session (original-rule windows) against the main and control sessions, per configuration
    (median of three trials) and per block-level cost."""
    k = CONFIG
    v = ws_orig.groupby(k + ["Condition"])[["game_mean", "game_p99", "ai_mean", "ai_p99"]].median()
    rows = []
    for session, b, conds in [("main", main, ["BASELINE", "E_CORE", "P_SHARED", "P_SMT"]), ("control", control, ["SMT_IDLE", "P_SMT"])]:
        e = b.rename(columns={"Game_Avg_ms": "game_mean", "Game_99th_ms": "game_p99", "AI_Avg_ms": "ai_mean", "AI_99th_ms": "ai_p99"})
        e = e.groupby(k + ["Condition"])[["game_mean", "game_p99", "ai_mean", "ai_p99"]].median()
        for cond in conds:
            vv = v.xs(cond, level="Condition")
            ee = e.xs(cond, level="Condition").reindex(vv.index)
            for m in ["game_mean", "game_p99", "ai_mean", "ai_p99"]:
                if vv[m].isna().all():
                    continue
                d = (vv[m] / ee[m] - 1) * 100
                rows.append({"earlier_session": session, "condition": cond, "measure": m, "configs": int(d.notna().sum()),
                             "median_diff_pct": round(d.median(), 2), "min_pct": round(d.min(), 2), "max_pct": round(d.max(), 2),
                             "within_5pct": int((d.abs() <= 5).sum())})
    # costs: placement minus control per configuration, both sessions
    def cost(b, cond, ctl, cols=("game_mean", "game_p99")):
        x = b[b.Condition == cond].set_index(BLOCK)
        c = b[b.Condition == ctl].set_index(BLOCK).reindex(x.index)
        return {m: (x[m] - c[m]).groupby(level=list(range(4))).median() for m in cols}
    vr = ws_orig.rename(columns={})
    er = main.rename(columns={"Game_Avg_ms": "game_mean", "Game_99th_ms": "game_p99"})
    cr = control.rename(columns={"Game_Avg_ms": "game_mean", "Game_99th_ms": "game_p99"})
    for cond, ctl, earlier in [("E_CORE", "BASELINE", er), ("P_SHARED", "BASELINE", er), ("P_SMT", "SMT_IDLE", cr)]:
        cv, ce = cost(vr, cond, ctl), cost(earlier, cond, ctl)
        for m in ["game_mean", "game_p99"]:
            a, z = cv[m], ce[m].reindex(cv[m].index)
            rows.append({"earlier_session": "main" if earlier is er else "control", "condition": f"{cond} - {ctl}",
                         "measure": f"{m}_cost_ms", "configs": int(a.notna().sum()),
                         "median_diff_pct": np.nan, "min_pct": np.nan, "max_pct": np.nan, "within_5pct": np.nan,
                         "validation_cost_median_ms": round(a.median(), 3), "earlier_cost_median_ms": round(z.median(), 3),
                         "cost_spearman_rho": round(stats.spearmanr(a, z).correlation, 3),
                         "same_sign_configs": int((np.sign(a) == np.sign(z)).sum())})
    return pd.DataFrame(rows)


def hwinfo_table(tests, h):
    """HWiNFO sensors (every 2 s, all fresh) averaged over each test window (samples in (start + 2 s, end]), then the
    median by condition and the paired change of each AI condition against its control in the same block."""
    cols = {"pkg_power_w": "CPU Package Power [W]", "pkg_temp_c": "CPU Package [°C].1", "pcore0_clock_mhz": "P-core 0 Clock [MHz]",
            "lp0_usage_pct": "P-core 0 T0 Usage [%]", "lp1_usage_pct": "P-core 0 T1 Usage [%]"}
    H = pd.DataFrame({"ts": h.ts})
    for k, c in cols.items():
        H[k] = pd.to_numeric(h[c], errors="coerce")
    ecl = [c for c in h.columns if c.startswith("E-core") and c.endswith(" Clock [MHz]") and "Effective" not in c]
    H["ecore_clock_mhz"] = h[ecl].apply(pd.to_numeric, errors="coerce").mean(axis=1)
    i0 = np.searchsorted(H.ts.values, tests.ts0.values + np.timedelta64(2000, "ms"))
    i1 = np.searchsorted(H.ts.values, tests.ts1.values, side="right")
    per = pd.DataFrame([H.iloc[a:z].drop(columns="ts").mean().to_dict() for a, z in zip(i0, i1)])
    per["samples"] = i1 - i0
    per = pd.concat([tests[BLOCK + ["Condition"]].reset_index(drop=True), per], axis=1)
    rows = []
    for cond, g in per.groupby("Condition"):
        row = {"condition": cond, "control": CONTROL_OF.get(cond, ""), "samples_per_test": int(g.samples.median())}
        row.update({k: round(g[k].median(), 2) for k in list(cols) + ["ecore_clock_mhz"]})
        if cond in CONTROL_OF:
            x = g.set_index(BLOCK)
            c = per[per.Condition == CONTROL_OF[cond]].set_index(BLOCK).reindex(x.index)
            cl = ["|".join(map(str, k[:4])) for k in x.index]
            for k in ["pcore0_clock_mhz", "pkg_power_w"]:
                d = x[k] - c[k]
                m, lo, hi = cluster_boot_median(d.values, cl)
                row.update({f"d_{k}": round(m, 2), f"d_{k}_ci_lo": round(lo, 2), f"d_{k}_ci_hi": round(hi, 2),
                            f"d_{k}_pct": round(((x[k] / c[k] - 1) * 100).median(), 2)})
        rows.append(row)
    return pd.DataFrame(rows)


def facts_table(t, tests, ws_orig, main, control):
    """Single figures quoted in the paper's validation text. Same columns as the audit's facts.csv."""
    F = []

    def add(fid, raw, fmt, desc):
        F.append({"id": fid, "value": fmt.format(raw), "raw": raw, "description": desc})

    q, op, eff, sw = t["quality"], t["operating_point"].set_index("session"), t["effects_equal20"].set_index("condition"), t["same_work"]
    dl, tm, rs, rep, hw = t["deadlines"], t["timing"], t["residence"].set_index("condition"), t["replication"], t["hwinfo"].set_index("condition")
    add("tests", len(tests), "{:d}", "Validation tests")
    add("configs", tests.groupby(CONFIG).ngroups, "{:d}", "Configurations")
    add("hours", op.loc["validation", "hours"], "{:.1f}", "Session length (h)")
    add("days_after_main", (tests.ts0.min() - main.ts0.min()).days, "{:d}", "Days between the main and validation sessions")
    add("drift_flags", int(q.drift_flags.sum()), "{:d}", "Drift flags")
    add("pkg_temp", op.loc["validation", "pkg_temp_mean_c"], "{:.1f}", "Package temperature, mean during tests (°C)")
    add("pkg_throttle", op.loc["validation", "pkg_throttle_pct"], "{:.1f}", "HWiNFO rows with package thermal throttling (%)")
    add("pkg_power", op.loc["validation", "pkg_power_w"], "{:.1f}", "HWiNFO package power, mean during tests (W)")
    add("baseline_frame", op.loc["validation", "baseline_mean_frame_ms"], "{:.2f}", "BASELINE mean frame, median over tests (ms)")
    add("baseline_frame_main", op.loc["main", "baseline_mean_frame_ms"], "{:.2f}", "Main session BASELINE mean frame, median over tests (ms)")
    fm = rep[rep.measure == "game_mean"]
    add("rep_frame_median_lo", fm.median_diff_pct.min(), "{:.1f}", "Replication: smallest per-condition median difference of configuration mean frames (%)")
    add("rep_frame_median_hi", fm.median_diff_pct.max(), "{:.1f}", "Replication: largest per-condition median difference of configuration mean frames (%)")
    add("rep_frame_absmax", np.maximum(fm.max_pct.abs(), fm.min_pct.abs()).max(), "{:.1f}", "Replication: largest absolute configuration difference in mean frame (%)")
    for cond in ["E_CORE - BASELINE", "P_SMT - SMT_IDLE", "P_SHARED - BASELINE"]:
        r = rep[(rep.condition == cond) & (rep.measure == "game_mean_cost_ms")].iloc[0]
        tag = cond.split(" ")[0].lower()
        add(f"rep_cost_{tag}", r.validation_cost_median_ms, "{:.2f}", f"{cond}: validation cost, median of configuration medians (ms, original windows)")
        add(f"rep_cost_{tag}_earlier", r.earlier_cost_median_ms, "{:.2f}", f"{cond}: earlier-session cost, same configurations (ms)")
        add(f"rep_rho_{tag}", r.cost_spearman_rho, "{:.3f}", f"{cond}: Spearman rho of the configuration costs between sessions")
    # same configurations fit / met on the original windows as in the earlier sessions
    for cond, b in [("E_CORE", main), ("P_SMT", main), ("P_SHARED", main)]:
        v = ws_orig[ws_orig.Condition == cond].groupby(CONFIG).agg(p99=("ai_p99", "median"), mis=("missed", "sum"))
        e = b[(b.Condition == cond) & b.set_index(CONFIG).index.isin(v.index)].groupby(CONFIG).agg(p99=("AI_99th_ms", "median"), mis=("AI_Missed_Deadlines", "sum"))
        same_fit = set(v[v.p99 <= PERIOD_MS].index) == set(e[e.p99 <= PERIOD_MS].index)
        same_met = set(v[v.mis == 0].index) == set(e[e.mis == 0].index)
        add(f"fit_{cond}", int((v.p99 <= PERIOD_MS).sum()), "{:d}", f"{cond}: configurations fitting within 50 ms (original windows)")
        add(f"met_{cond}", int((v.mis == 0).sum()), "{:d}", f"{cond}: configurations with no missed deadline (original windows)")
        add(f"same_sets_{cond}", int(same_fit and same_met), "{:d}", f"{cond}: 1 if the same configurations fit and met every deadline as in the main session")
    for cond in ["E_CORE", "P_SMT", "P_SHARED"]:
        d = dl[dl.condition == cond].set_index("window")
        add(f"met_equal20_{cond}", int(d.loc["equal20", "configs_met"]), "{:d}", f"{cond}: configurations with no missed deadline (equal 20-s windows)")
        add(f"met_updates_orig_{cond}", int(d.loc["orig", "met_updates_median"]), "{:d}", f"{cond}: updates over three trials of configurations without a miss (original windows), median")
    for cond in ["E_CORE_SMT_IDLE", "P_SMT", "P_SHARED_SMT_IDLE", "SMT_IDLE"]:
        r = eff.loc[cond] if cond in eff.index else eff[eff.control == cond].iloc[0]
        key = "game_mean_control_ms" if cond == "SMT_IDLE" else "game_mean_ms"
        add(f"frame_{cond}", r[key], "{:.2f}", f"{cond}: mean frame, median of blocks (ms, equal 20-s windows)")
        if cond != "SMT_IDLE":
            add(f"p99_{cond}", r.game_p99_ms, "{:.2f}", f"{cond}: P99 frame, median of blocks (ms)")
    r = sw[(sw.a == "E_CORE_SMT_IDLE") & (sw.b == "P_SMT") & (sw.measure == "game_mean")].iloc[0]
    add("ecore_lower_than_smt_blocks", r.a_lower_blocks_pct, "{:.0f}", "Blocks whose E_CORE_SMT_IDLE mean frame was below P_SMT's (%)")
    r = sw[(sw.a == "P_SHARED_SMT_IDLE") & (sw.b == "E_CORE_SMT_IDLE") & (sw.measure == "game_mean")].iloc[0]
    add("ecore_lower_than_shared_blocks", 100 - r.a_lower_blocks_pct, "{:.0f}", "Blocks whose E_CORE_SMT_IDLE mean frame was below P_SHARED_SMT_IDLE's (%)")
    u = rs.loc["UNPINNED"]
    add("unpinned_ecore_pct", u["LP 12-18 (E)"] + u["LP 19 (E, telemetry)"], "{:.1f}", "UNPINNED batches starting on an E-core LP (%)")
    add("unpinned_lp19_pct", u["LP 19 (E, telemetry)"], "{:.0f}", "UNPINNED batches starting on LP 19 (%)")
    r = sw[(sw.a == "E_CORE") & (sw.b == "UNPINNED") & (sw.measure == "game_mean")].iloc[0]
    add("unpinned_diff_ci_absmax", max(abs(r.ci_lo), abs(r.ci_hi)), "{:.2f}", "|95% CI bound| of E_CORE minus UNPINNED mean frame (ms)")
    r = sw[(sw.a == "E_CORE_1LP") & (sw.b == "E_CORE") & (sw.measure == "ai_p99")].iloc[0]
    add("onelp_p99_increase", r.a_minus_b_ms, "{:.1f}", "E_CORE_1LP minus E_CORE P99 batch latency, median of blocks (ms)")
    r = sw[(sw.a == "E_CORE_1LP") & (sw.b == "E_CORE") & (sw.measure == "game_mean")].iloc[0]
    add("onelp_frame_diff_ci_absmax", max(abs(r.ci_lo), abs(r.ci_hi)), "{:.2f}", "|95% CI bound| of E_CORE_1LP minus E_CORE mean frame (ms)")
    d = dl[(dl.condition == "E_CORE_1LP") & (dl.window == "equal20")].iloc[0]
    add("onelp_lowest_p99_with_miss", d.lowest_p99_with_miss_ms, "{:.0f}", "E_CORE_1LP: lowest configuration P99 with a missed deadline (ms)")
    e = dl[(dl.condition == "E_CORE") & (dl.window == "equal20")].iloc[0]
    add("onelp_more_misses_pct", (d.misses / e.misses - 1) * 100, "{:.0f}", "E_CORE_1LP missed deadlines over E_CORE's, minus 1 (%, equal 20-s windows)")
    add("pcore_clock_validation_ghz", op.loc["validation", "pcore0_clock_mhz"] / 1000, "{:.2f}", "HWiNFO P-core 0 clock, median during tests (GHz)")
    add("ecore_lp_change_between_pct", rs.loc["E_CORE", "lp_changed_between_batches_pct"], "{:.0f}", "E_CORE: batches starting on another LP than the previous batch ended on (%)")
    for cond in ["E_CORE", "P_SMT"]:
        r = tm[(tm.condition == cond) & (tm.work == "all")].iloc[0]
        add(f"delay_median_{cond}", r.delay_median_ms, "{:.2f}", f"{cond}: start delay, median over batches (ms)")
        add(f"delay_p99_{cond}", r.delay_p99_ms, "{:.2f}", f"{cond}: start delay, P99 over batches (ms)")
    r = tm[(tm.condition == "E_CORE") & (tm.work == "all")].iloc[0]
    add("ecore_long_batch_miss_pct", 100 - r.late_start_share_pct, "{:.0f}", "E_CORE: missed deadlines whose batch took more than 50 ms (%)")
    r = tm[(tm.condition == "P_SHARED") & (tm.work == "<100 MFLOP")].iloc[0]
    add("shared_light_delay_median", r.delay_median_ms, "{:.1f}", "P_SHARED below 100 MFLOP: start delay, median (ms)")
    add("shared_light_delay_p99", r.delay_p99_ms, "{:.1f}", "P_SHARED below 100 MFLOP: start delay, P99 (ms)")
    add("shared_light_misses", int(r.missed), "{:d}", "P_SHARED below 100 MFLOP: missed deadlines")
    add("shared_light_late_misses", int(r.late_start_misses), "{:d}", "P_SHARED below 100 MFLOP: missed deadlines whose batch took at most 50 ms")
    add("shared_light_oncpu", r.oncpu_share_median * 100, "{:.1f}", "P_SHARED below 100 MFLOP: on-CPU share of batch latency, median (%)")
    r = tm[(tm.condition == "P_SHARED") & (tm.work == ">=1000 MFLOP")].iloc[0]
    add("shared_heavy_oncpu", r.oncpu_share_median * 100, "{:.0f}", "P_SHARED at or above 1,000 MFLOP: on-CPU share, median (%)")
    e = hw.loc["E_CORE"]
    add("ecore_clock_drop_mhz", -e.d_pcore0_clock_mhz, "{:.0f}", "P-core 0 clock, BASELINE minus E_CORE, median of blocks (MHz)")
    add("ecore_clock_drop_ci_lo", -e.d_pcore0_clock_mhz_ci_hi, "{:.0f}", "Its 95% CI, lower bound (MHz)")
    add("ecore_clock_drop_ci_hi", -e.d_pcore0_clock_mhz_ci_lo, "{:.0f}", "Its 95% CI, upper bound (MHz)")
    add("ecore_clock_drop_pct", -e.d_pcore0_clock_mhz_pct, "{:.1f}", "Same, median of the paired % changes (%)")
    add("ecore_cost_pct", eff.loc["E_CORE", "game_mean_change_pct"], "{:.1f}", "E_CORE vs BASELINE mean-frame change, median (%, equal 20-s windows)")
    add("smt_clock_change_mhz", hw.loc["P_SMT", "d_pcore0_clock_mhz"], "{:.0f}", "P-core 0 clock, P_SMT minus SMT_IDLE, median of blocks (MHz)")
    return pd.DataFrame(F)


TSC_HZ = np.nan


def main_cli():
    global TSC_HZ
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for a in ["validation", "batches", "frames", "hwinfo", "main", "main-hwinfo", "control", "control-hwinfo"]:
        ap.add_argument(f"--{a}", required=True)
    ap.add_argument("--tsc-hz", type=float, default=2918398932.0, help="tsc_hz from the session's run_metadata.txt")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    TSC_HZ = a.tsc_hz
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    tests = load_bench(a.validation)
    main, control = load_bench(a.main), load_bench(a.control)
    hv, hm, hc = load_hwinfo(a.hwinfo), load_hwinfo(a.main_hwinfo), load_hwinfo(a.control_hwinfo)
    f, b = load_traces(a.frames, a.batches, tests)
    ends = window_ends(tests, f, b)
    ws_orig = window_stats(tests, f, b, ends["orig"])
    ws_eq = window_stats(tests, f, b, ends["equal20"])
    b20 = b[b.End_ns <= EQUAL_S * 1e9]  # batch-level tables use the same equal 20-s windows as the effects

    tables = [
        ("quality", quality_table(tests, ws_orig, ws_eq)),
        ("operating_point", operating_point_table([("main", main, hm), ("control", control, hc), ("validation", tests, hv)])),
        ("effects_equal20", effects_table(ws_eq)),
        ("effects_orig", effects_table(ws_orig)),
        ("same_work", same_work_table(ws_eq)),
        ("deadlines", pd.concat([deadline_table(ws_orig, "orig"), deadline_table(ws_eq, "equal20")], ignore_index=True)),
        ("per_config", per_config_table(ws_eq, ws_orig)),
        ("timing", timing_table(b20, tests)),
        ("residence", residence_table(b20, tests)),
        ("unpinned_tests", unpinned_tests_table(b20, tests, ws_eq)),
        ("replication", replication_table(ws_orig, main, control)),
        ("hwinfo", hwinfo_table(tests, hv)),
    ]
    tables.append(("facts", facts_table(dict(tables), tests, ws_orig, main, control)))
    for name, df in tables:
        df.to_csv(out / f"{name}.csv", index=False)
        print(f"wrote {name}.csv ({len(df)} rows)")


if __name__ == "__main__":
    main_cli()
