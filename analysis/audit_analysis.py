"""
audit_analysis.py -- audit and analysis of the hybrid-core AI placement benchmark (two sessions).

Usage:
    python audit_analysis.py --main benchmark_steady_state.csv --main-hwinfo HWiN_sensor_log.CSV ^
                             --control benchmark_smt_control.csv --control-hwinfo HWiN_sensor_log_smt.CSV ^
                             --out out

Inputs
  --main            CSV of the main session (BASELINE, P_SHARED, P_SMT, E_CORE), written by
                    steady_state_compute_bound.cpp.
  --main-hwinfo     HWiNFO64 sensor CSV logged during the main session.
  --control         CSV of the control session (P_SMT and SMT_IDLE), same grid and calibration.
  --control-hwinfo  HWiNFO64 sensor CSV logged during the control session.

Output: small CSV tables in --out (each documented in its function), used by the audit report, the IEEE paper
(paper_facts.csv holds the paper's single figures that no other table does) and make_paper_figures.py.

Design notes
  * Every placement comparison is PAIRED WITHIN A BLOCK. A block is one (inputs, hidden, layers, agents,
    trial) combination whose conditions ran back to back in random order, so slow drift affects a placement
    and its control almost equally (drift_order.csv measures how much).
  * E_CORE and P_SHARED are compared with BASELINE of the same block (main session).
  * P_SMT parks the worker on the game LP's SMT sibling, so its matched control is SMT_IDLE (same worker
    set, no AI) of the same block in the control session. The main session's P_SMT is used for the
    cross-session replication check and, with E_CORE and P_SHARED, for AI latency (same session).
  * Confidence intervals: 95% percentile bootstrap of the median, resampling the 180 model
    configurations (clusters of 3 trials), 2,000 resamples, fixed seed.
  * HWiNFO per-core sensors were starved in the main session (changed only every ~80-110 s, median, under load);
    in the control session every sensor refreshed every 2 s, so its per-core data are used per test there.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

BLOCK = ["Inputs", "Hidden_Size", "Layers", "Agent_Density", "Trial"]
CONFIG = ["Inputs", "Hidden_Size", "Layers", "Agent_Density"]
AI_CONDS = ["E_CORE", "P_SMT", "P_SHARED"]
CONTROL_OF = {"E_CORE": "BASELINE", "P_SHARED": "BASELINE", "P_SMT": "SMT_IDLE"}
BOOT_N, BOOT_SEED = 2000, 1337


# ----------------------------------------------------------------------------------------------------------
# Loading
# ----------------------------------------------------------------------------------------------------------
def gru_params(I, H, L):
    """Parameter count of an exported model (see load_bench)."""
    return 3 * H * (I + H + 2) + (L - 1) * 3 * H * (2 * H + 2) + 3 * H + 3


def load_bench(path):
    b = pd.read_csv(path)
    b["ts0"] = pd.to_datetime(b.Timestamp_Start.str[:23])
    b["ts1"] = pd.to_datetime(b.Timestamp_End.str[:23])
    b = b.sort_values("ts0").reset_index(drop=True)
    # GRU size and work per AI batch (multiply-add counted as 2 FLOPs; sequence length 10; PyTorch has two
    # bias vectors per gate set; FC head maps hidden -> 3 outputs at the last step only).
    I, H, L, A = b.Inputs, b.Hidden_Size, b.Layers, b.Agent_Density
    b["params"] = gru_params(I, H, L)
    b["mflop_batch"] = (10 * (2 * 3 * H * (I + H) + (L - 1) * 2 * 3 * H * (2 * H)) + 2 * 3 * H) * A / 1e6
    return b


def load_hwinfo(path):
    h = pd.read_csv(path, encoding="latin-1", skipfooter=2, engine="python")  # last 2 rows repeat the header
    h.columns = [c.replace("Â°", "°") for c in h.columns]
    h["ts"] = pd.to_datetime(h.Date + " " + h.Time, format="%d.%m.%Y %H:%M:%S.%f")
    return h.copy()


def wide_by_block(main, control):
    """One row per block. Main-session conditions keep their names; control-session P_SMT becomes 'P_SMT2'."""
    vals = ["Game_Avg_ms", "Game_Median_ms", "Game_99th_ms", "Game_Max_ms", "Pre_Test_Frame_ms", "AI_Avg_ms",
            "AI_99th_ms", "AI_Duty_Cycle", "AI_Missed_Deadlines", "AI_Batches", "Mean_RAPL_Package_W",
            "Mean_ProcPerf_P_Pct", "mflop_batch"]
    c = control.assign(Condition=control.Condition.replace({"P_SMT": "P_SMT2"}))
    both = pd.concat([main, c], ignore_index=True)
    w = both.pivot_table(index=BLOCK, columns="Condition", values=vals)
    w.columns = [f"{a}|{cond}" for a, cond in w.columns]
    w = w.reset_index()
    w["mflop"] = w["mflop_batch|E_CORE"]
    return w


def change_pct(w, metric, cond):
    """Per-block % change of a placement against its matched control (P_SMT uses the control session)."""
    if cond == "P_SMT":
        return (w[f"{metric}|P_SMT2"] / w[f"{metric}|SMT_IDLE"] - 1) * 100
    return (w[f"{metric}|{cond}"] / w[f"{metric}|BASELINE"] - 1) * 100


def delta(w, metric, cond):
    if cond == "P_SMT":
        return w[f"{metric}|P_SMT2"] - w[f"{metric}|SMT_IDLE"]
    return w[f"{metric}|{cond}"] - w[f"{metric}|BASELINE"]


def cluster_boot_median(values, clusters):
    """Median and 95% CI, resampling clusters (model configurations) with replacement."""
    df = pd.DataFrame({"v": np.asarray(values, float), "c": clusters}).dropna()
    groups = [g.v.values for _, g in df.groupby("c")]
    rng = np.random.default_rng(BOOT_SEED)
    meds = np.empty(BOOT_N)
    for i in range(BOOT_N):
        pick = rng.integers(0, len(groups), len(groups))
        meds[i] = np.median(np.concatenate([groups[j] for j in pick]))
    return float(np.median(df.v)), float(np.percentile(meds, 2.5)), float(np.percentile(meds, 97.5))


def config_wilcoxon_p(values, w):
    """Two-sided Wilcoxon signed-rank p over the 180 configuration medians (the 3 trials of a configuration are
    not independent, so blocks are not tested one by one)."""
    df = pd.DataFrame({"v": np.asarray(values, float), **{c: w[c].values for c in CONFIG}})
    return float(stats.wilcoxon(df.groupby(CONFIG).v.median().dropna()).pvalue)


# ----------------------------------------------------------------------------------------------------------
# Tables
# ----------------------------------------------------------------------------------------------------------
def hourly_table(b, h):
    """Main session, per hour: BASELINE frame, pre-test frame (non-SMT conditions), BASELINE RAPL package power
    and P-core % performance (benchmark PDH counters), HWiNFO package temperature (fresh sensor)."""
    t0 = b.ts0.min()
    b = b.assign(hour=((b.ts0 - t0).dt.total_seconds() // 3600).astype(int))
    hw = h[(h.ts >= t0) & (h.ts <= b.ts1.max())].copy()
    hw["hour"] = ((hw.ts - t0).dt.total_seconds() // 3600).astype(int)
    hw["pkg_t"] = pd.to_numeric(hw["CPU Package [°C].1"], errors="coerce")
    rows = []
    for k in sorted(b.hour.unique()):
        s = b[b.hour == k]
        base = s[s.Condition == "BASELINE"]
        pre = s[s.Condition != "P_SMT"]
        rows.append({
            "hour": k + 1,
            "start_time": (t0 + pd.Timedelta(hours=int(k))).strftime("%H:%M"),
            "baseline_frame_ms": round(base.Game_Avg_ms.median(), 3),
            "pretest_frame_ms": round(pre.Pre_Test_Frame_ms.median(), 3),
            "baseline_rapl_w": round(base.Mean_RAPL_Package_W.mean(), 2),
            "baseline_pcore_perf_pct": round(base.Mean_ProcPerf_P_Pct.mean(), 2),
            "hwinfo_pkg_temp_c": round(hw[hw.hour == k].pkg_t.mean(), 2),
            "baseline_tests": len(base),
        })
    return pd.DataFrame(rows)


def impact_table(w):
    """Game frame-time change by placement against its matched control, binned by AI work per batch.
    Medians and quartiles over blocks; mean vs mean and P99 vs P99."""
    edges = [0, 10, 30, 100, 300, 1000, 3000, 10000, 1e9]
    labels = ["<10", "10-30", "30-100", "100-300", "300-1k", "1k-3k", "3k-10k", ">10k"]
    w = w.assign(bin=pd.cut(w.mflop, edges, labels=labels, right=False))
    rows = []
    for order, lab in enumerate(labels):
        s = w[w.bin == lab]
        for cond in AI_CONDS:
            m = change_pct(s, "Game_Avg_ms", cond)
            p = change_pct(s, "Game_99th_ms", cond)
            rows.append({
                "bin_order": order, "mflop_bin": lab, "mflop_median": round(s.mflop.median(), 2),
                "condition": cond, "control": CONTROL_OF[cond], "blocks": len(s),
                "mean_change_pct": round(m.median(), 3), "mean_change_q1": round(m.quantile(.25), 3),
                "mean_change_q3": round(m.quantile(.75), 3),
                "p99_change_pct": round(p.median(), 3), "p99_change_q1": round(p.quantile(.25), 3),
                "p99_change_q3": round(p.quantile(.75), 3),
            })
    return pd.DataFrame(rows)


def ai_latency_table(b):
    """Main session: one row per model configuration and AI placement, median over the 3 trials."""
    ai = b[b.Condition.isin(AI_CONDS)]
    g = ai.groupby(CONFIG + ["Condition"]).agg(
        mflop_batch=("mflop_batch", "first"), params=("params", "first"),
        ai_mean_ms=("AI_Avg_ms", "median"), ai_p99_ms=("AI_99th_ms", "median"),
        duty_cycle=("AI_Duty_Cycle", "median")).reset_index()
    g["config"] = ("in" + g.Inputs.astype(str) + " H" + g.Hidden_Size.astype(str) + " L" + g.Layers.astype(str)
                   + " x" + g.Agent_Density.astype(str))
    g = g.rename(columns={"Condition": "condition", "Inputs": "inputs", "Hidden_Size": "hidden",
                          "Layers": "layers", "Agent_Density": "agents"})
    g["mflop_batch"] = g.mflop_batch.round(3)
    for c in ["ai_mean_ms", "ai_p99_ms"]:
        g[c] = g[c].round(3)
    g["duty_cycle"] = g.duty_cycle.round(4)
    return g[["config", "condition", "inputs", "hidden", "layers", "agents", "params", "mflop_batch",
              "ai_mean_ms", "ai_p99_ms", "duty_cycle"]]


def ecore_dose_table(w):
    """Main session: E_CORE minus BASELINE in the same block, binned by the E-core AI duty cycle."""
    edges = [0, .10, .25, .50, .75, .99, 1.01]
    labels = ["0-10%", "10-25%", "25-50%", "50-75%", "75-99%", "Saturated"]
    w = w.assign(bin=pd.cut(w["AI_Duty_Cycle|E_CORE"], edges, labels=labels))
    rows = []
    for order, lab in enumerate(labels):
        s = w[w.bin == lab]
        rows.append({
            "bin_order": order, "duty_bin": lab, "blocks": len(s),
            "d_package_w": round(delta(s, "Mean_RAPL_Package_W", "E_CORE").median(), 3),
            "d_pcore_perf_pts": round(delta(s, "Mean_ProcPerf_P_Pct", "E_CORE").median(), 3),
            "d_game_frame_pct": round(change_pct(s, "Game_Avg_ms", "E_CORE").median(), 3),
        })
    return pd.DataFrame(rows)


def psmt_duty_table(w):
    """Control session: P_SMT and SMT_IDLE game frame statistics by P_SMT AI duty cycle, plus the main
    session's BASELINE mean frame of the same blocks for reference."""
    edges = [0, .02, .10, .25, .50, .75, .99, 1.01]
    labels = ["<2%", "2-10%", "10-25%", "25-50%", "50-75%", "75-99%", "Saturated"]
    w = w.assign(bin=pd.cut(w["AI_Duty_Cycle|P_SMT2"], edges, labels=labels))
    rows = []
    for order, lab in enumerate(labels):
        s = w[w.bin == lab]
        rows.append({
            "bin_order": order, "duty_bin": lab, "blocks": len(s),
            "idle_median_ms": round(s["Game_Median_ms|SMT_IDLE"].median(), 3),
            "idle_mean_ms": round(s["Game_Avg_ms|SMT_IDLE"].median(), 3),
            "idle_p99_ms": round(s["Game_99th_ms|SMT_IDLE"].median(), 3),
            "smt_median_ms": round(s["Game_Median_ms|P_SMT2"].median(), 3),
            "smt_mean_ms": round(s["Game_Avg_ms|P_SMT2"].median(), 3),
            "smt_p99_ms": round(s["Game_99th_ms|P_SMT2"].median(), 3),
            "mean_change_pct": round(change_pct(s, "Game_Avg_ms", "P_SMT").median(), 3),
            "p99_change_pct": round(change_pct(s, "Game_99th_ms", "P_SMT").median(), 3),
            "baseline_mean_ms": round(s["Game_Avg_ms|BASELINE"].median(), 3),
        })
    return pd.DataFrame(rows)


def shared_stall_table(w):
    """Main session: P_SHARED P99 frame minus BASELINE median frame, binned by the AI batch time measured on a
    P-core thread in the same block's P_SMT test (a proxy for the batch's CPU time)."""
    edges = [0, 2, 5, 10, 20, 30, 40, 60, 100, 1e9]
    labels = ["<2", "2-5", "5-10", "10-20", "20-30", "30-40", "40-60", "60-100", ">100"]
    w = w.assign(bin=pd.cut(w["AI_Avg_ms|P_SMT"], edges, labels=labels, right=False))
    rows = []
    for order, lab in enumerate(labels):
        s = w[w.bin == lab]
        rows.append({
            "bin_order": order, "ai_batch_ms_bin": lab, "blocks": len(s),
            "p99_stall_ms": round((s["Game_99th_ms|P_SHARED"] - s["Game_Median_ms|BASELINE"]).median(), 2),
            "max_frame_ms": round(s["Game_Max_ms|P_SHARED"].median(), 2),
        })
    return pd.DataFrame(rows)


def refresh_label(sub, col):
    span = (sub.ts.max() - sub.ts.min()).total_seconds() if len(sub) > 1 else 0
    v = sub[col].astype(str)
    changes = sub.ts[v != v.shift()].iloc[1:]  # the first row is not a change
    gaps = changes.diff().dt.total_seconds().dropna()
    if len(changes) == 0:
        return f"no update in {span:.0f} s"
    if len(gaps) == 0:
        return f"1 update in {span:.0f} s"
    return f"{gaps.median():.0f} s" if gaps.median() >= 10 else f"{gaps.median():.1f} s"


def hwinfo_refresh_table(sessions):
    """Median seconds between value changes of selected HWiNFO sensors, per session and phase."""
    sensors = [("CPU package power", "CPU Package Power [W]"), ("P-core 0 clock", "P-core 0 Clock [MHz]"),
               ("E-core effective clock", "E-core 6 T0 Effective Clock [MHz]"),
               ("CPU package temperature", "CPU Package [°C].1")]
    rows = []
    order = 0
    for sname, b, h in sessions:
        t0, t1 = b.ts0.min(), b.ts1.max()
        phases = [("Idle before run", h.ts.min(), t0 - pd.Timedelta(seconds=184)),
                  ("Soak", t0 - pd.Timedelta(seconds=183), t0),
                  ("Tests, first hour", t0, t0 + pd.Timedelta(hours=1)),
                  ("Tests, last hour", t1 - pd.Timedelta(hours=1), t1),
                  ("Tests, all", t0, t1),
                  ("Idle after run", t1 + pd.Timedelta(seconds=60), h.ts.max())]
        for pname, a, z in phases:
            sub = h[(h.ts >= a) & (h.ts <= z)]
            row = {"phase_order": order, "phase": f"{sname}: {pname}", "samples": len(sub)}
            for label, col in sensors:
                row[label] = refresh_label(sub, col)
            rows.append(row)
            order += 1
    return pd.DataFrame(rows)


def control_hw_table(control, hc):
    """Control session, HWiNFO per-core data averaged over each test window (samples whose timestamps fall
    in (start + 2 s, end]), then by condition."""
    cols = {"lp0_usage_pct": "P-core 0 T0 Usage [%]", "lp1_usage_pct": "P-core 0 T1 Usage [%]",
            "lp1_eff_clock_mhz": "P-core 0 T1 Effective Clock [MHz]", "pcore0_clock_mhz": "P-core 0 Clock [MHz]",
            "package_power_w": "CPU Package Power [W]", "package_temp_c": "CPU Package [°C].1"}
    H = pd.DataFrame({"ts": hc.ts})
    for k, c in cols.items():
        H[k] = pd.to_numeric(hc[c], errors="coerce")
    i0 = np.searchsorted(H.ts.values, control.ts0.values + np.timedelta64(2000, "ms"))
    i1 = np.searchsorted(H.ts.values, control.ts1.values, side="right")
    per_test = pd.DataFrame([H.iloc[a:z].drop(columns="ts").mean().to_dict() for a, z in zip(i0, i1)])
    per_test["condition"] = control.Condition.values
    t = per_test.groupby("condition").mean().round(2).reset_index()
    return t


def replication_table(main, control):
    """P_SMT measured in both sessions: per block, control-session value vs main-session value."""
    a = main[main.Condition == "P_SMT"].set_index(BLOCK)
    c = control[control.Condition == "P_SMT"].set_index(BLOCK)
    rows = []
    for metric, label in [("Game_Avg_ms", "Game mean frame"), ("Game_Median_ms", "Game median frame"),
                          ("Game_99th_ms", "Game P99 frame"), ("AI_Avg_ms", "AI mean batch"), ("AI_99th_ms", "AI P99 batch")]:
        d = (c[metric] / a[metric] - 1) * 100
        rows.append({"metric": label, "median_diff_pct": round(d.median(), 2), "q1_pct": round(d.quantile(.25), 2),
                     "q3_pct": round(d.quantile(.75), 2), "q05_pct": round(d.quantile(.05), 2),
                     "q95_pct": round(d.quantile(.95), 2), "min_pct": round(d.min(), 2), "max_pct": round(d.max(), 2),
                     "within_5pct_share": round((d.abs() <= 5).mean() * 100, 1),
                     "spearman_rho": round(stats.spearmanr(c[metric], a[metric].reindex(c.index)).correlation, 4)})
    return pd.DataFrame(rows)


def absolute_table(main, control):
    """Directly measured medians over the tests of each condition and session (no placement minus control): game mean
    and P99 frame, window length, AI updates (batches) per test, and missed deadlines per scheduled update. The window
    closes after 1,000 game frames and 50 batches, so its length, and the number of updates a test observes, follows
    the game's frame time."""
    rows = []
    for session, b in [("main", main), ("control", control)]:
        for cond, g in b.groupby("Condition"):
            row = {"session": session, "condition": cond, "tests": len(g),
                   "game_mean_ms": round(g.Game_Avg_ms.median(), 2), "game_p99_ms": round(g.Game_99th_ms.median(), 2),
                   "window_s": round(g.Measure_Window_s.median(), 2)}
            if cond in AI_CONDS:
                missed = g.AI_Missed_Deadlines / g.AI_Batches * 100
                row.update({"updates_median": int(g.AI_Batches.median()), "updates_min": int(g.AI_Batches.min()),
                            "updates_max": int(g.AI_Batches.max()), "missed_per_update_median_pct": round(missed.median(), 1),
                            "tests_with_missed_pct": round((g.AI_Missed_Deadlines > 0).mean() * 100, 1)})
            rows.append(row)
    return pd.DataFrame(rows)


def architecture_table(w):
    """Robustness check of the clustering: the 180 configurations are 60 architectures (I, H, L) at three agent
    counts. Resampling the 60 architectures, each with all its agent counts and trials, and testing the 60
    architecture medians, beside the configuration-level results of effects_summary.csv."""
    arch = w[["Inputs", "Hidden_Size", "Layers"]].astype(str).agg("|".join, axis=1)
    rows = []
    for cond in AI_CONDS:
        for key, metric in [("mean", "Game_Avg_ms"), ("p99", "Game_99th_ms")]:
            for unit, v in [("ms", delta(w, metric, cond)), ("pct", change_pct(w, metric, cond))]:
                m, lo, hi = cluster_boot_median(v, arch)
                p = stats.wilcoxon(pd.Series(np.asarray(v, float)).groupby(arch.values).median().dropna()).pvalue
                rows.append({"placement": cond, "measure": f"{key}_frame_change_{unit}", "median": round(m, 3),
                             "arch_ci_lo": round(lo, 3), "arch_ci_hi": round(hi, 3), "arch_wilcoxon_p": float(p)})
    return pd.DataFrame(rows)


def effects_table(w, main):
    """Headline effect of each placement against its matched control, with 95% cluster-bootstrap CIs."""
    clusters = w[CONFIG].astype(str).agg("|".join, axis=1)
    rows = []
    for cond in AI_CONDS:
        row = {"placement": cond, "control": CONTROL_OF[cond], "session": "control" if cond == "P_SMT" else "main"}
        for key, metric in [("mean", "Game_Avg_ms"), ("p99", "Game_99th_ms")]:
            m, lo, hi = cluster_boot_median(change_pct(w, metric, cond), clusters)
            row.update({f"{key}_change_pct": round(m, 2), f"{key}_ci_lo": round(lo, 2), f"{key}_ci_hi": round(hi, 2)})
        m, lo, hi = cluster_boot_median(delta(w, "Mean_RAPL_Package_W", cond), clusters)
        row.update({"power_change_w": round(m, 2), "power_ci_lo": round(lo, 2), "power_ci_hi": round(hi, 2)})
        # % Processor Performance is an unweighted mean over the P-core LPs. For P_SMT vs SMT_IDLE it mostly
        # records LP 1 going from idle to busy, not a clock change (HWiNFO shows the same P-core 0 clock; see
        # control_hwinfo.csv), so it is left blank there.
        if cond == "P_SMT":
            row.update({"pcore_perf_change_pts": np.nan, "pcore_perf_ci_lo": np.nan, "pcore_perf_ci_hi": np.nan})
        else:
            m, lo, hi = cluster_boot_median(delta(w, "Mean_ProcPerf_P_Pct", cond), clusters)
            row.update({"pcore_perf_change_pts": round(m, 2), "pcore_perf_ci_lo": round(lo, 2), "pcore_perf_ci_hi": round(hi, 2)})
        ai = main[main.Condition == cond]
        row["ai_p99_within_tick_pct"] = round((ai.AI_99th_ms <= 50).mean() * 100, 1)
        row["ai_batch_vs_p_smt"] = round((w[f"AI_Avg_ms|{cond}"] / w["AI_Avg_ms|P_SMT"]).median(), 2)
        row["missed_tick_share_pct"] = round((w[f"AI_Missed_Deadlines|{cond}"] / w[f"AI_Batches|{cond}"]).median() * 100, 1)
        row["mean_ci"] = f"{row['mean_ci_lo']:.2f} to {row['mean_ci_hi']:.2f}" if cond == "E_CORE" else f"{row['mean_ci_lo']:.1f} to {row['mean_ci_hi']:.1f}"
        row["p99_ci"] = f"{row['p99_ci_lo']:.2f} to {row['p99_ci_hi']:.2f}" if cond == "E_CORE" else f"{row['p99_ci_lo']:.1f} to {row['p99_ci_hi']:.1f}"
        # Absolute changes (ms), the share of blocks with a slower mean frame, and configuration-level tests
        for key, metric in [("mean", "Game_Avg_ms"), ("p99", "Game_99th_ms")]:
            m, lo, hi = cluster_boot_median(delta(w, metric, cond), clusters)
            row.update({f"{key}_change_ms": round(m, 3), f"{key}_ms_ci_lo": round(lo, 3), f"{key}_ms_ci_hi": round(hi, 3)})
        row["blocks_slower_pct"] = round((change_pct(w, "Game_Avg_ms", cond) > 0).mean() * 100, 1)
        tests = [("mean_pct", change_pct(w, "Game_Avg_ms", cond)), ("mean_ms", delta(w, "Game_Avg_ms", cond)),
                 ("p99_pct", change_pct(w, "Game_99th_ms", cond)), ("p99_ms", delta(w, "Game_99th_ms", cond)),
                 ("power", delta(w, "Mean_RAPL_Package_W", cond))]
        if cond != "P_SMT":
            tests.append(("pcore_perf", delta(w, "Mean_ProcPerf_P_Pct", cond)))
        for key, v in tests:
            row[f"{key}_wilcoxon_p"] = config_wilcoxon_p(v, w)
        rows.append(row)
    return pd.DataFrame(rows)


def deadline_table(main):
    """Main session, per AI placement: P99 batch time within the 50 ms tick versus actual deadline success. A
    configuration fits when its median P99 batch time over the 3 trials is at most 50 ms, and meets every deadline
    when none of its trials missed a tick. Every configuration below the smallest non-fitting work per batch fits
    and none above the largest fitting one does; in between (the band) the two mix."""
    rows = []
    for cond in AI_CONDS:
        a = main[main.Condition == cond].assign(missed=lambda d: d.AI_Missed_Deadlines / d.AI_Batches)
        c = a.groupby(CONFIG).agg(mflop=("mflop_batch", "first"), p99=("AI_99th_ms", "median"),
                                  misses=("AI_Missed_Deadlines", "sum"), max_missed=("missed", "max"),
                                  updates=("AI_Batches", "sum"))
        c["fit"], c["met"] = c.p99 <= 50, c.misses == 0
        # Margin: the lowest median P99 among configurations that missed a deadline; every configuration below it met
        # every deadline. Exposure: the updates (batches) observed over the 3 trials of each configuration that met
        # every deadline.
        margin = c.p99[~c.met].min()
        lo, hi = c.mflop[~c.fit].min(), c.mflop[c.fit].max()
        band, below, fm = c[(c.mflop >= lo) & (c.mflop <= hi)], c[c.mflop < lo], c[c.fit & ~c.met]
        fm_tests = a.set_index(CONFIG).loc[fm.index]
        rows.append({
            "condition": cond, "configs": len(c), "configs_fit": int(c.fit.sum()),
            "configs_met_every_deadline": int(c.met.sum()), "fit_configs_missing_ticks": len(fm),
            "fit_missed_share_max_pct": round(fm.max_missed.max() * 100, 1) if len(fm) else 0.0,
            "fit_missed_tests_without_long_batch": int(((fm_tests.AI_Missed_Deadlines > 0) & (fm_tests.AI_Max_ms <= 50)).sum()),
            "all_fit_below_mflop": round(lo, 1), "none_fit_above_mflop": round(hi, 1),
            "band_configs": len(band), "band_fit": int(band.fit.sum()),
            "configs_below": len(below), "configs_below_met_every_deadline": int(below.met.sum()),
            "tests_missed_all_batches_under_50ms": int(((a.AI_Missed_Deadlines > 0) & (a.AI_Max_ms < 50)).sum()),
            "tests_with_missed_tick_pct": round((a.AI_Missed_Deadlines > 0).mean() * 100, 1),
            "missed_share_median_pct": round(a.missed.median() * 100, 1),
            "light_missed_share_median_pct": round(a[a.mflop_batch < 100].missed.median() * 100, 1),
            "lowest_p99_with_miss_ms": round(margin, 2), "configs_below_margin": int((c.p99 < margin).sum()),
            "configs_below_margin_met": int(c[c.p99 < margin].met.sum()),
            "fit_missed_p99_min_ms": round(fm.p99.min(), 1) if len(fm) else np.nan,
            "fit_missed_p99_max_ms": round(fm.p99.max(), 1) if len(fm) else np.nan,
            "met_updates_median": int(c[c.met].updates.median()) if c.met.any() else 0,
            "met_updates_min": int(c[c.met].updates.min()) if c.met.any() else 0,
            "met_updates_max": int(c[c.met].updates.max()) if c.met.any() else 0,
        })
    return pd.DataFrame(rows)


def drift_order_table(main, control):
    """Slow drift and the pairing that limits it, per placement: least-squares rise of the control's mean frame over
    its session, the time between a placement and its control inside the block (and the drift over the longest such
    gap), and the placement's mean-frame change split by whether it ran before or after its control."""
    rows = []
    for cond in AI_CONDS:
        ctl = CONTROL_OF[cond]
        b = control if cond == "P_SMT" else main
        c = b[b.Condition == ctl]
        slope = stats.linregress((c.ts0 - b.ts0.min()).dt.total_seconds() / 3600, c.Game_Avg_ms).slope  # ms per h
        span_h = (b.ts1.max() - b.ts0.min()).total_seconds() / 3600
        x = b[b.Condition == cond].set_index(BLOCK)
        y = c.set_index(BLOCK).reindex(x.index)
        gap_s = (x.ts0 - y.ts0).abs().dt.total_seconds()
        chg = (x.Game_Avg_ms / y.Game_Avg_ms - 1) * 100
        first = x.Block_Pos < y.Block_Pos
        clusters = pd.Series(["|".join(map(str, k[:4])) for k in x.index], index=x.index)
        row = {"condition": cond, "control": ctl, "session": "control" if cond == "P_SMT" else "main",
               "session_hours": round(span_h, 2), "control_drift_ms": round(slope * span_h, 3),
               "control_drift_pct": round(slope * span_h / c.Game_Avg_ms.median() * 100, 2),
               "gap_median_s": round(gap_s.median(), 1), "gap_max_s": round(gap_s.max(), 1),
               "drift_over_max_gap_us": round(slope * gap_s.max() / 3600 * 1000, 2)}
        for tag, sel in [("before", first), ("after", ~first)]:
            m, lo, hi = cluster_boot_median(chg[sel].values, clusters[sel].values)
            row.update({f"{tag}_blocks": int(sel.sum()), f"{tag}_change_pct": round(m, 2),
                        f"{tag}_ci_lo": round(lo, 2), f"{tag}_ci_hi": round(hi, 2)})
        rows.append(row)
    return pd.DataFrame(rows)


def ecore_weight_table(w):
    """Main session, blocks whose E-core AI never idled (duty cycle >= 0.99): E_CORE minus BASELINE by the model's
    weight size (4-byte parameters, MiB). Last-level-cache or memory contention would grow with the weights; the
    largest group exceeds the 24-MiB last-level cache."""
    s = w[w["AI_Duty_Cycle|E_CORE"] >= 0.99]
    s = s.assign(mib=gru_params(s.Inputs, s.Hidden_Size, s.Layers) * 4 / 2**20)
    s = s.assign(bin=pd.cut(s.mib, [0, 7, 25, 1e9], labels=["<7 MiB", "7-25 MiB", ">25 MiB"]))
    rows = []
    for order, lab in enumerate(s.bin.cat.categories):
        g = s[s.bin == lab]
        rows.append({"bin_order": order, "weights_bin": lab, "blocks": len(g), "mib_min": round(g.mib.min(), 1),
                     "mib_max": round(g.mib.max(), 1),
                     "d_game_frame_pct": round(change_pct(g, "Game_Avg_ms", "E_CORE").median(), 2),
                     "d_package_w": round(delta(g, "Mean_RAPL_Package_W", "E_CORE").median(), 2)})
    return pd.DataFrame(rows)


def throughput_table(ai_lat):
    """Main session: median inference throughput (work per batch over mean batch time, GFLOP/s) across the model
    configurations of each hidden width, per AI placement."""
    t = ai_lat.assign(gflops=ai_lat.mflop_batch / ai_lat.ai_mean_ms)
    return t.pivot_table(index="hidden", columns="condition", values="gflops", aggfunc="median").round(2).reset_index()


def paper_facts_table(main, control, w):
    """Single figures quoted in the IEEE paper that no other table holds. Same columns as facts.csv."""
    F = []

    def add(fid, raw, fmt, desc):
        F.append({"id": fid, "value": fmt.format(raw), "raw": raw, "description": desc})

    work = main.groupby(CONFIG).mflop_batch.first()
    add("mflop_min", work.min(), "{:.1f}", "Smallest work per batch over the 180 configurations (MFLOP)")
    add("mflop_max", work.max(), "{:,.0f}", "Largest work per batch over the 180 configurations (MFLOP)")
    for tag, b in [("main", main), ("control", control)]:
        add(f"telemetry_ms_{tag}", (b.Measure_Window_s * 1000 / b.Telemetry_Samples).median(), "{:.0f}",
            f"{tag.capitalize()} session: PDH telemetry interval, median over tests (ms)")
    ec = change_pct(w, "Game_Avg_ms", "E_CORE").sort_values()
    top = w.loc[ec.index[-1]]
    add("ecore_outlier_pct", ec.iloc[-1], "{:.1f}", "Largest E_CORE vs BASELINE mean-frame change of any block (%)")
    add("ecore_outlier_block", f"I={top.Inputs:.0f}, H={top.Hidden_Size:.0f}, L={top.Layers:.0f}, "
        f"A={top.Agent_Density:.0f}, trial {top.Trial:.0f}", "{}", "Block of that change")
    add("ecore_outlier_duty_pct", top["AI_Duty_Cycle|E_CORE"] * 100, "{:.0f}", "E-core AI duty cycle in that block (%)")
    add("ecore_second_pct", ec.iloc[-2], "{:.1f}", "Second-largest E_CORE vs BASELINE mean-frame change (%)")
    ai = main[main.Condition.isin(AI_CONDS)]
    few = ai[ai.AI_Batches < 100]
    add("few_batch_tests", len(few), "{:d}", "Main-session AI tests with fewer than 100 batches (nearest-rank P99 = maximum)")
    add("few_batch_min_mflop", few.mflop_batch.min(), "{:,.0f}", "Smallest work per batch among those tests (MFLOP)")
    add("few_batch_min_p99_ms", few.AI_99th_ms.min(), "{:.1f}", "Smallest P99 batch time among those tests (ms)")
    add("few_batch_p99_is_max", int((few.AI_99th_ms == few.AI_Max_ms).sum()), "{:d}", "Of those tests, P99 equals the maximum")
    for tag, b, cond in [("idle", control, "SMT_IDLE"), ("base", main, "BASELINE")]:
        a = b[b.Condition == cond]
        add(f"{tag}_p99_over_median_pct", ((a.Game_99th_ms / a.Game_Median_ms - 1) * 100).median(), "{:.1f}",
            f"{cond}: P99 frame above the median frame of the same test, median over tests (%)")
    add("base_pcore_perf", main[main.Condition == "BASELINE"].Mean_ProcPerf_P_Pct.median(), "{:.1f}",
        "BASELINE P-core % Processor Performance, median over tests")
    light = w[w.mflop < 300]
    pct = change_pct(light, "Game_Avg_ms", "P_SMT")
    add("smt_light_blocks", len(light), "{:d}", "Blocks below 300 MFLOP per batch")
    add("smt_light_ms", delta(light, "Game_Avg_ms", "P_SMT").median(), "{:.2f}", "P_SMT vs SMT_IDLE mean frame below 300 MFLOP, median (ms)")
    add("smt_light_pct", pct.median(), "{:.1f}", "Same, median change (%)")
    add("smt_light_under5_pct", (pct < 5).mean() * 100, "{:.0f}", "Share of those blocks with a change under 5% (%)")
    heavy = w[w.mflop > 3000]
    duty = pd.concat([heavy["AI_Duty_Cycle|E_CORE"], heavy["AI_Duty_Cycle|P_SMT2"], heavy["AI_Duty_Cycle|P_SHARED"]])
    add("heavy_min_duty_pct", duty.min() * 100, "{:.1f}",
        "Lowest AI duty cycle of any placement above 3,000 MFLOP per batch (P_SMT from the control session) (%)")
    add("smt_over_base_pct", (w["Game_Avg_ms|P_SMT2"] > w["Game_Avg_ms|BASELINE"]).mean() * 100, "{:.0f}",
        "Blocks whose control-session P_SMT mean frame exceeded the main-session BASELINE mean frame (%)")
    for tag, b in [("main", main), ("control", control)]:
        top_c = b.Peak_ACPI_ThermalZone_C.max()
        add(f"acpi_peak_top_{tag}", int((b.Peak_ACPI_ThermalZone_C == top_c).sum()), "{:,d}",
            f"{tag.capitalize()} session: tests whose ACPI thermal-zone peak equals the session maximum ({top_c:.2f} °C), of {len(b):,}")
    return pd.DataFrame(F)


def effects_latex(eff):
    names = {"E_CORE": "E-cores (E\\_CORE)", "P_SMT": "SMT sibling (P\\_SMT)", "P_SHARED": "Game's LP (P\\_SHARED)"}
    ctrl = {"BASELINE": "BASELINE", "SMT_IDLE": "SMT\\_IDLE"}
    lines = [
        "% Generated by audit_analysis.py. Medians over 540 paired blocks; 95% cluster-bootstrap CIs (180 configurations).",
        "% Requires \\usepackage{booktabs} in the preamble.",
        "\\begin{table}[t]", "\\centering", "\\caption{Game and AI cost of each AI placement}", "\\label{tab:placement}",
        "\\setlength{\\tabcolsep}{3pt}", "\\begin{tabular}{@{}lcccc@{}}", "\\toprule",
        "Placement & Control & Mean frame & P99 frame & AI P99 $\\le$ 50\\,ms \\\\",
        "          &         & change (\\%) & change (\\%) & (\\% of tests) \\\\", "\\midrule",
    ]
    for _, r in eff.iterrows():
        lines.append(f"{names[r.placement]} & {ctrl[r.control]} & "
                     f"{r.mean_change_pct:+.1f} [{r.mean_ci_lo:+.1f}, {r.mean_ci_hi:+.1f}] & "
                     f"{r.p99_change_pct:+.1f} [{r.p99_ci_lo:+.1f}, {r.p99_ci_hi:+.1f}] & "
                     f"{r.ai_p99_within_tick_pct:.0f} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}", ""]
    return "\n".join(lines)


def session_stats(b, h):
    t0, t1 = b.ts0.min(), b.ts1.max()
    hw = h[(h.ts >= t0) & (h.ts <= t1)]
    num = lambda c: pd.to_numeric(hw[c], errors="coerce")
    return {
        "pkg_temp_mean": num("CPU Package [°C].1").mean(), "pkg_temp_max": num("CPU Package [°C].1").max(),
        "throttle_share": (hw["Package/Ring Thermal Throttling [Yes/No]"] == "Yes").mean() * 100,
        # The game's own core: its per-core thermal-throttling flag (fresh every 2 s in the control session)
        "pcore0_throttle_share": (hw["P-core 0 Thermal Throttling [Yes/No]"] == "Yes").mean() * 100,
        "pl1": num("PL1 Power Limit (Static) [W]").median(), "pkg_power": num("CPU Package Power [W]").mean(),
        "pcore_clock": num("P-core 0 Clock [MHz]").median(),
    }


def facts_table(main, control, hm, hc, w, hourly, ecore, psmt, stall, ai_lat, refresh, rep, eff, chw):
    """Single figures quoted in the audit text. 'value' is the display string; 'raw' the unrounded value."""
    F = []

    def add(fid, raw, fmt, desc):
        F.append({"id": fid, "value": fmt.format(raw), "raw": raw, "description": desc})

    sm, sc = session_stats(main, hm), session_stats(control, hc)
    nonsmt = main[main.Condition != "P_SMT"]
    base = main[main.Condition == "BASELINE"]
    add("pretest_mean", nonsmt.Pre_Test_Frame_ms.mean(), "{:.2f}", "Main session: mean pre-test frame (ms), non-SMT tests")
    add("pretest_sd", nonsmt.Pre_Test_Frame_ms.std(), "{:.2f}", "Main session: SD of pre-test frame (ms), non-SMT tests")
    add("pretest_max_dev", (nonsmt.Pre_Test_Frame_ms / 16.0 - 1).abs().max() * 100, "{:.1f}",
        "Main session: largest pre-test deviation from the 16 ms target (%)")
    slope = stats.linregress((base.ts0 - main.ts0.min()).dt.total_seconds() / 3600, base.Game_Avg_ms).slope
    span_h = (main.ts1.max() - main.ts0.min()).total_seconds() / 3600
    add("drift_ms", slope * span_h, "{:.1f}", "Main session: fitted BASELINE frame-time rise over the session (ms)")
    add("drift_pct", slope * span_h / base.Game_Avg_ms.median() * 100, "{:.1f}", "Same, as % of the median")
    add("rapl_drop", hourly.baseline_rapl_w.iloc[0] - hourly.baseline_rapl_w.iloc[-1], "{:.1f}",
        "Main session: BASELINE package power, first hour minus last hour (W)")
    add("session_hours", span_h, "{:.0f}", "Main session: duration of the test phase (h)")
    add("control_hours", (control.ts1.max() - control.ts0.min()).total_seconds() / 3600, "{:.1f}",
        "Control session: duration of the test phase (h)")
    add("pkg_temp_mean", sm["pkg_temp_mean"], "{:.1f}", "Main session: HWiNFO package temperature, mean during tests (°C)")
    add("pkg_temp_max", sm["pkg_temp_max"], "{:.0f}", "Main session: HWiNFO package temperature, max during tests (°C)")
    add("throttle_share", sm["throttle_share"], "{:.1f}", "Main session: HWiNFO rows with package thermal throttling = Yes (%)")
    add("pl_limit", sm["pl1"], "{:.0f}", "PL1 and PL2 power limits reported by HWiNFO (W)")
    add("rapl_base", base.Mean_RAPL_Package_W.median(), "{:.1f}", "Main session: BASELINE package power, median of per-test RAPL means (W)")
    add("pcore_clock_ghz", sm["pcore_clock"] / 1000, "{:.2f}", "Main session: HWiNFO P-core 0 clock, median during tests (GHz)")
    add("c_pkg_temp_mean", sc["pkg_temp_mean"], "{:.1f}", "Control session: HWiNFO package temperature, mean during tests (°C)")
    add("c_throttle_share", sc["throttle_share"], "{:.0f}", "Control session: HWiNFO rows with package thermal throttling = Yes (%)")
    add("pcore0_throttle_share", sm["pcore0_throttle_share"], "{:.1f}", "Main session: HWiNFO rows with P-core 0 thermal throttling = Yes (%)")
    add("c_pcore0_throttle_share", sc["pcore0_throttle_share"], "{:.1f}", "Control session: HWiNFO rows with P-core 0 thermal throttling = Yes (%)")
    add("c_pkg_power", sc["pkg_power"], "{:.1f}", "Control session: HWiNFO package power, mean during tests (W)")
    add("c_pcore_clock_ghz", sc["pcore_clock"] / 1000, "{:.2f}", "Control session: HWiNFO P-core 0 clock, median during tests (GHz)")

    prev = main.Condition.shift()
    B = main.assign(prev=prev)
    B = B[B.Condition == "BASELINE"]
    add("carryover_p", stats.kruskal(*[g.Game_Avg_ms.values for _, g in B.groupby("prev")]).pvalue, "{:.2f}",
        "Main session: Kruskal-Wallis p, BASELINE frame time by preceding condition")
    g = main.groupby(CONFIG + ["Condition"])
    cv = (g.Game_Avg_ms.std() / g.Game_Avg_ms.mean() * 100)
    add("cv_game_max_cond", cv.groupby(level="Condition").median().max(), "{:.1f}",
        "Main session: largest per-condition median trial-to-trial CV of mean game frame (%)")
    cva = (g.AI_Avg_ms.std() / g.AI_Avg_ms.mean() * 100).dropna()
    add("cv_ai_max_cond", cva.groupby(level="Condition").median().max(), "{:.1f}",
        "Main session: largest per-condition median trial-to-trial CV of mean AI batch time (%)")

    E = eff.set_index("placement")
    for cond, tag in [("E_CORE", "ecore"), ("P_SHARED", "shared"), ("P_SMT", "smt")]:
        r = E.loc[cond]
        add(f"{tag}_mean_pct", r.mean_change_pct, "{:.2f}" if tag == "ecore" else "{:.1f}",
            f"{cond} vs {r.control}: median change of mean game frame (%)")
        add(f"{tag}_mean_lo", r.mean_ci_lo, "{:.2f}" if tag == "ecore" else "{:.1f}", f"{cond}: 95% CI low (%)")
        add(f"{tag}_mean_hi", r.mean_ci_hi, "{:.2f}" if tag == "ecore" else "{:.1f}", f"{cond}: 95% CI high (%)")
        add(f"{tag}_p99_pct", r.p99_change_pct, "{:.1f}" if tag != "shared" else "{:.0f}",
            f"{cond} vs {r.control}: median change of P99 game frame (%)")
        add(f"{tag}_fit", r.ai_p99_within_tick_pct, "{:.0f}", f"{cond} tests whose AI P99 batch time fits the 50 ms tick (%)")
    add("ecore_blocks_slower", (change_pct(w, "Game_Avg_ms", "E_CORE") > 0).mean() * 100, "{:.0f}",
        "Blocks where E_CORE mean frame > BASELINE (%)")
    add("ecore_rapl", E.loc["E_CORE"].power_change_w, "{:.1f}", "E_CORE minus BASELINE package power, median (W)")
    add("ecore_perf", -E.loc["E_CORE"].pcore_perf_change_pts, "{:.1f}", "Drop in P-core % performance under E_CORE, median (points)")
    add("ecore_sat_pct", ecore[ecore.duty_bin == "Saturated"].d_game_frame_pct.iloc[0], "{:.1f}",
        "E_CORE frame change when the E-core AI is saturated (%)")
    sat = w[w["AI_Duty_Cycle|P_SHARED"] > 0.99]
    add("shared_sat_mean", sat["Game_Avg_ms|P_SHARED"].median(), "{:.1f}", "P_SHARED mean frame when AI saturates LP0 (ms)")
    add("shared_stall_plateau", stall[stall.bin_order >= 5].p99_stall_ms.median(), "{:.0f}",
        "P_SHARED P99 stall above BASELINE median once AI batches exceed ~30 ms (ms)")
    light = w[w["AI_Avg_ms|P_SMT"] < 2]
    add("shared_light_p99_pct", change_pct(light, "Game_99th_ms", "P_SHARED").median(), "{:.0f}",
        "P_SHARED P99 change for the lightest models (AI batch < 2 ms) (%)")

    # SMT sibling (control session)
    sat_s = psmt[psmt.duty_bin == "Saturated"].iloc[0]
    low_s = psmt[psmt.duty_bin == "<2%"].iloc[0]
    add("smt_sat_mean_pct", sat_s.mean_change_pct, "{:.0f}", "P_SMT vs SMT_IDLE mean frame when the AI saturates LP 1 (%)")
    add("smt_sat_p99_pct", sat_s.p99_change_pct, "{:.0f}", "P_SMT vs SMT_IDLE P99 frame when the AI saturates LP 1 (%)")
    add("smt_sat_frame", sat_s.smt_mean_ms, "{:.1f}", "P_SMT mean frame when the AI saturates LP 1 (ms)")
    add("smt_low_mean_pct", low_s.mean_change_pct, "{:.1f}", "P_SMT vs SMT_IDLE mean frame at AI duty < 2% (%)")
    add("idle_median_ms", w["Game_Median_ms|SMT_IDLE"].median(), "{:.1f}", "SMT_IDLE median frame (ms)")
    add("idle_p99_ms", w["Game_99th_ms|SMT_IDLE"].median(), "{:.1f}", "SMT_IDLE P99 frame (ms)")
    add("base_frame_ms", w["Game_Avg_ms|BASELINE"].median(), "{:.1f}", "BASELINE mean frame, median (ms)")
    add("base_p99_ms", w["Game_99th_ms|BASELINE"].median(), "{:.1f}", "BASELINE P99 frame, median (ms)")
    add("smt_worker_cost", ((w["Game_Avg_ms|BASELINE"] / w["Game_Avg_ms|SMT_IDLE"]) - 1).median() * 100, "{:.0f}",
        "BASELINE mean frame vs SMT_IDLE mean frame of the same block: cost of a physics worker on the sibling (%)")
    ch = chw.set_index("condition")
    add("idle_lp1_usage", ch.loc["SMT_IDLE"].lp1_usage_pct, "{:.1f}", "Control session: HWiNFO LP 1 usage during SMT_IDLE tests (%)")
    add("smt_lp1_usage", ch.loc["P_SMT"].lp1_usage_pct, "{:.0f}", "Control session: HWiNFO LP 1 usage during P_SMT tests (%)")
    add("smt_power_w", E.loc["P_SMT"].power_change_w, "{:.1f}", "P_SMT minus SMT_IDLE package power, median (W)")

    # Replication
    R = rep.set_index("metric")
    add("rep_game_diff", R.loc["Game mean frame"].median_diff_pct, "{:.1f}", "P_SMT game mean frame, control vs main session, median difference (%)")
    add("rep_ai_diff", R.loc["AI mean batch"].median_diff_pct, "{:.1f}", "P_SMT AI mean batch, control vs main session, median difference (%)")
    add("rep_game_rho", R.loc["Game mean frame"].spearman_rho, "{:.2f}", "P_SMT game mean frame, Spearman rho between sessions")
    add("rep_ai_rho", R.loc["AI mean batch"].spearman_rho, "{:.3f}", "P_SMT AI mean batch, Spearman rho between sessions")

    # AI latency
    piv = ai_lat.pivot_table(index="config", columns="condition", values="ai_mean_ms")
    ratio = piv["E_CORE"] / piv["P_SMT"]
    add("e_over_p_min", ratio.min(), "{:.1f}", "E-core / P-core-sibling AI batch time, minimum ratio")
    add("e_over_p_max", ratio.max(), "{:.1f}", "E-core / P-core-sibling AI batch time, maximum ratio")
    big = ai_lat[ai_lat.hidden == 512]
    gfl = big.mflop_batch / big.ai_mean_ms
    add("gflops_e", gfl[big.condition == "E_CORE"].median(), "{:.0f}", "E-core inference throughput at hidden 512 (GFLOP/s)")
    add("gflops_p", gfl[big.condition == "P_SMT"].median(), "{:.0f}", "P-core-sibling inference throughput at hidden 512 (GFLOP/s)")
    add("heaviest_e_s", ai_lat[ai_lat.condition == "E_CORE"].ai_mean_ms.max() / 1000, "{:.1f}",
        "Heaviest configuration, E-core mean AI batch time (s)")
    light_ai = ai_lat[ai_lat.mflop_batch < 100]
    add("shared_ai_floor_ms", light_ai[light_ai.condition == "P_SHARED"].ai_p99_ms.median(), "{:.0f}",
        "P_SHARED AI batch P99 for models under 100 MFLOP per batch, median (ms)")
    add("smt_ai_light_ms", light_ai[light_ai.condition == "P_SMT"].ai_p99_ms.median(), "{:.1f}",
        "P_SMT AI batch P99 for models under 100 MFLOP per batch, median (ms)")

    # HWiNFO refresh
    vals = []
    for ph in ["Main: Tests, first hour", "Main: Tests, last hour"]:
        r = refresh[refresh.phase == ph].iloc[0]
        vals += [float(str(r["CPU package power"]).split()[0]), float(str(r["P-core 0 clock"]).split()[0])]
    add("hw_refresh_min", min(vals), "{:.0f}",
        "Main session: shortest median refresh of package power and the P-core 0 clock in the first or last test hour (s)")
    add("hw_refresh_max", max(vals), "{:.0f}",
        "Main session: longest median refresh of package power and the P-core 0 clock in the first or last test hour (s)")
    r = refresh[refresh.phase == "Main: Tests, all"].iloc[0]
    vals = [float(str(r[c]).split()[0]) for c in ["CPU package power", "P-core 0 clock", "E-core effective clock"]]
    add("hw_refresh_session_min", min(vals), "{:.0f}",
        "Main session: shortest median refresh over the whole test phase of package power, P-core 0 clock and E-core effective clock (s)")
    add("hw_refresh_session_max", max(vals), "{:.0f}", "Same, longest (s)")

    # ACPI thermal zone (not the CPU package sensor)
    both = pd.concat([main, control])
    add("acpi_mean_min", both.Mean_ACPI_ThermalZone_C.min(), "{:.2f}", "ACPI thermal zone, lowest per-test mean, both sessions (°C)")
    add("acpi_mean_max", both.Mean_ACPI_ThermalZone_C.max(), "{:.2f}", "ACPI thermal zone, highest per-test mean, both sessions (°C)")
    return pd.DataFrame(F)


def checks_table(main, control, w, facts):
    f = dict(zip(facts.id, facts.value))
    key = ["Inputs", "Hidden_Size", "Layers", "Agent_Density", "Condition", "Trial"]

    def runs(b):
        k = b[BLOCK].astype(str).agg("|".join, axis=1)
        return int((k != k.shift()).sum())

    ai = main[main.Condition != "BASELINE"]
    cs = ai.groupby(CONFIG).AI_Output_Checksum.nunique()
    cm = ai.groupby(CONFIG).AI_Output_Checksum.first()
    cc = control[control.Condition == "P_SMT"].groupby(CONFIG).AI_Output_Checksum.first()
    same_ck = int((cc == cm.reindex(cc.index)).sum())
    rows = [
        ("Completeness", "Pass",
         f"Main: {len(main):,} rows in {main.groupby(BLOCK).ngroups} blocks of {main.Condition.nunique()} conditions. "
         f"Control: {len(control):,} rows in {control.groupby(BLOCK).ngroups} blocks of {control.Condition.nunique()}. "
         f"{main.duplicated(key).sum() + control.duplicated(key).sum()} duplicate keys; every block ran contiguously "
         f"({runs(main)} and {runs(control)} block runs)."),
        ("Placement and affinity", "Pass",
         "AI masks 0x1 (P_SHARED), 0x2 (P_SMT), 0x7F000 (E_CORE) and 0x0 for the no-AI controls; workers on LP 1-11, "
         "with LP 1's worker parked in P_SMT and SMT_IDLE; telemetry on LP 19; every affinity call verified."),
        ("Calibration and drift", "Pass",
         f"Both sessions used loop count {main.Calibrated_Loop_Count.iloc[0]:,}. Main pre-test frame {f['pretest_mean']} ± "
         f"{f['pretest_sd']} ms, largest deviation {f['pretest_max_dev']}% against the 15% tolerance; "
         f"{int(main.Drift_Flag.sum() + control.Drift_Flag.sum())} drift flags in {len(main) + len(control):,} tests."),
        ("Model execution", "Pass",
         f"{(ai.Batch_Mode == 'batched').sum() + (control.Batch_Mode == 'batched').sum():,} of "
         f"{len(ai) + (control.Condition == 'P_SMT').sum():,} AI tests ran batched. One output checksum per model "
         f"configuration across placements and trials ({(cs == 1).sum()} of {len(cs)}), identical in the control "
         f"session ({same_ck} of {len(cc)})."),
        ("Sample sizes", "Pass",
         f"At least {min(main.Game_Frames.min(), control.Game_Frames.min()):,} game frames and "
         f"{int(min(ai.AI_Batches.min(), control[control.Condition == 'P_SMT'].AI_Batches.min()))} AI batches per test; "
         f"{int(main.Truncated.sum() + control.Truncated.sum())} tests hit the 300 s cap."),
        ("Repeatability", "Pass",
         f"Median trial-to-trial variation is at most {f['cv_game_max_cond']}% for mean game frame time and "
         f"{f['cv_ai_max_cond']}% for mean AI batch time in any condition."),
        ("Cross-session replication", "Pass",
         f"P_SMT, measured in both sessions, differs by a median {f['rep_game_diff']}% in game frame time "
         f"(Spearman {f['rep_game_rho']}) and {f['rep_ai_diff']}% in AI batch time (Spearman {f['rep_ai_rho']})."),
        ("Order and carry-over", "Pass",
         f"BASELINE frame time does not depend on which condition ran before it (Kruskal-Wallis p = {f['carryover_p']})."),
        ("P_SMT control", "Pass",
         f"Resolved by the control session: P_SMT is now compared with SMT_IDLE in the same block. The idle sibling "
         f"alone gives a {f['idle_p99_ms']} ms P99 against a {f['idle_median_ms']} ms median, which is why BASELINE "
         "could not serve as P_SMT's control."),
        ("Thermal operating point", "Disclose",
         f"Both sessions sat at the thermal limit: package {f['pkg_temp_mean']} and {f['c_pkg_temp_mean']} °C on average, "
         f"throttling flag set in {f['throttle_share']}% and {f['c_throttle_share']}% of HWiNFO rows, about "
         f"{f['rapl_base']} W with PL1 and PL2 at {f['pl_limit']} W."),
        ("Slow drift", "Disclose",
         f"Main-session BASELINE frame time rose about {f['drift_ms']} ms ({f['drift_pct']}%) over {f['session_hours']} h "
         f"while package power fell {f['rapl_drop']} W at constant temperature. Pairing within blocks limits its effect "
         "(drift_order.csv)."),
        ("HWiNFO", "Disclose",
         f"Main session: package power and per-core clocks changed only every {f['hw_refresh_session_min']}-"
         f"{f['hw_refresh_session_max']} s (median over the test phase; {f['hw_refresh_min']}-{f['hw_refresh_max']} s in "
         "its first and last hour), so they are used at session level only. Control session: every sensor refreshed "
         "every 2 s and is used per test."),
        ("PDH telemetry columns", "Disclose",
         f"PowerMeter reads {max(main.Peak_PowerMeter_W.max(), control.Peak_PowerMeter_W.max()):.0f} W in every test, and "
         "the ACPI thermal zone, which is not the CPU package sensor, changes only in 1-K steps (per-test means "
         f"{f['acpi_mean_min']}-{f['acpi_mean_max']} °C); drop both. RAPL package power is valid per test. % Processor "
         "Performance is a mean over P-core LPs, so compare it only between conditions with the same busy LPs (not P_SMT "
         "against SMT_IDLE)."),
        ("Tail metrics", "Disclose",
         "With 1,000 frames the P99.9 value is the second-largest frame. Use P99 as the main tail metric."),
    ]
    return pd.DataFrame(rows, columns=["check", "status", "evidence"])


def main_cli():
    ap = argparse.ArgumentParser(description="Audit and analysis of the hybrid-core AI placement benchmark.")
    ap.add_argument("--main", required=True, help="main-session CSV (benchmark_steady_state.csv)")
    ap.add_argument("--main-hwinfo", required=True, help="HWiNFO CSV of the main session")
    ap.add_argument("--control", required=True, help="control-session CSV (benchmark_smt_control.csv)")
    ap.add_argument("--control-hwinfo", required=True, help="HWiNFO CSV of the control session")
    ap.add_argument("--out", default="out", help="output folder (default: out)")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    main = load_bench(a.main)
    control = load_bench(a.control)
    hm, hc = load_hwinfo(a.main_hwinfo), load_hwinfo(a.control_hwinfo)
    w = wide_by_block(main, control)

    hourly = hourly_table(main, hm)
    impact = impact_table(w)
    ai_lat = ai_latency_table(main)
    ecore = ecore_dose_table(w)
    psmt = psmt_duty_table(w)
    stall = shared_stall_table(w)
    refresh = hwinfo_refresh_table([("Main", main, hm), ("Control", control, hc)])
    chw = control_hw_table(control, hc)
    rep = replication_table(main, control)
    eff = effects_table(w, main)
    facts = facts_table(main, control, hm, hc, w, hourly, ecore, psmt, stall, ai_lat, refresh, rep, eff, chw)
    checks = checks_table(main, control, w, facts)
    deadlines = deadline_table(main)
    drift = drift_order_table(main, control)
    weights = ecore_weight_table(w)
    throughput = throughput_table(ai_lat)
    paper_facts = paper_facts_table(main, control, w)
    absolute = absolute_table(main, control)
    architecture = architecture_table(w)

    tables = [("absolute", absolute), ("architecture", architecture),("hourly", hourly), ("impact", impact), ("ai_latency", ai_lat), ("ecore_dose", ecore),
              ("psmt_duty", psmt), ("shared_stall", stall), ("hwinfo_refresh", refresh), ("control_hwinfo", chw),
              ("replication", rep), ("effects_summary", eff), ("facts", facts), ("checks", checks),
              ("deadlines", deadlines), ("drift_order", drift), ("ecore_weights", weights), ("throughput", throughput),
              ("paper_facts", paper_facts)]
    for name, df in tables:
        df.to_csv(out / f"{name}.csv", index=False)
        print(f"wrote {name}.csv ({len(df)} rows)")
    (out / "effects_table.tex").write_text(effects_latex(eff), encoding="utf-8")
    print("wrote effects_table.tex")


if __name__ == "__main__":
    main_cli()
