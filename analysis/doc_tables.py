"""
doc_tables.py -- the extra tables quoted in the experiment record (the editable document), computed from the
same four input files as audit_analysis.py and with its helpers. Keep this file next to audit_analysis.py.

Usage:
    python doc_tables.py --main benchmark_steady_state.csv --main-hwinfo HWiN_sensor_log.CSV ^
                         --control benchmark_smt_control.csv --control-hwinfo HWiN_sensor_log_smt.CSV ^
                         --out out\\doc

Writes (all small CSV files):
  effect_tests.csv     per placement against its matched control: paired Wilcoxon signed-rank p-values and the
                       share of blocks in which the placement was worse
  ai_summary.csv       AI behaviour per placement: P99 within the 50 ms tick, missed ticks, saturation, batch time
  samples.csv          sample sizes, window lengths and pre-test frames per session and condition
  order_balance.csv    how often each condition ran at each position inside its block
  operating_point.csv  HWiNFO readings over each session's test phase: temperatures, limit flags, power, clocks
  configs.csv          per-configuration results for all 180 model configurations (the record's appendix)
  doc_facts.csv        single figures quoted in the record that no other table holds
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from audit_analysis import (AI_CONDS, BLOCK, CONFIG, CONTROL_OF, ai_latency_table, change_pct, delta,
                            load_bench, load_hwinfo, wide_by_block)


def effect_tests(w):
    rows = []
    for cond in AI_CONDS:
        r = {"placement": cond, "control": CONTROL_OF[cond], "blocks": int(change_pct(w, "Game_Avg_ms", cond).notna().sum())}
        for key, metric in [("mean", "Game_Avg_ms"), ("p99", "Game_99th_ms")]:
            x = change_pct(w, metric, cond).dropna()
            r[f"{key}_slower_pct"] = round((x > 0).mean() * 100, 1)
            r[f"{key}_wilcoxon_p"] = stats.wilcoxon(x).pvalue
        d = delta(w, "Mean_RAPL_Package_W", cond).dropna()
        r["power_higher_pct"] = round((d > 0).mean() * 100, 1)
        r["power_wilcoxon_p"] = stats.wilcoxon(d).pvalue
        if cond == "P_SMT":  # % Processor Performance is not comparable between P_SMT and SMT_IDLE
            r["pcore_perf_lower_pct"], r["pcore_perf_wilcoxon_p"] = np.nan, np.nan
        else:
            d = delta(w, "Mean_ProcPerf_P_Pct", cond).dropna()
            r["pcore_perf_lower_pct"] = round((d < 0).mean() * 100, 1)
            r["pcore_perf_wilcoxon_p"] = stats.wilcoxon(d).pvalue
        rows.append(r)
    t = pd.DataFrame(rows)
    for c in [c for c in t.columns if c.endswith("_p")]:
        t[c] = t[c].map(lambda v: "" if pd.isna(v) else f"{v:.2e}")
    return t


def ai_summary(main, control, ai_lat):
    rows = []
    sessions = [("main", main, c) for c in AI_CONDS] + [("control", control, "P_SMT")]
    for sess, b, cond in sessions:
        a = b[b.Condition == cond]
        cfg = ai_lat[ai_lat.condition == cond] if sess == "main" else None
        rows.append({
            "session": sess, "condition": cond, "tests": len(a),
            "p99_within_tick_pct": round((a.AI_99th_ms <= 50).mean() * 100, 1),
            "configs_p99_within_tick": int((cfg.ai_p99_ms <= 50).sum()) if cfg is not None else "",
            "tests_with_missed_tick_pct": round((a.AI_Missed_Deadlines > 0).mean() * 100, 1),
            "missed_share_median_pct": round((a.AI_Missed_Deadlines / a.AI_Batches).median() * 100, 1),
            "missed_share_mean_pct": round((a.AI_Missed_Deadlines / a.AI_Batches).mean() * 100, 1),
            "saturated_pct": round((a.AI_Duty_Cycle > 0.99).mean() * 100, 1),
            "duty_median_pct": round(a.AI_Duty_Cycle.median() * 100, 1),
            "batch_mean_ms_min": round(a.AI_Avg_ms.min(), 3), "batch_mean_ms_median": round(a.AI_Avg_ms.median(), 2),
            "batch_mean_ms_max": round(a.AI_Avg_ms.max(), 1),
            "batch_p99_ms_median": round(a.AI_99th_ms.median(), 2),
            "agents_per_s_median": round(a.AI_Agents_Per_Sec.median(), 0),
        })
    return pd.DataFrame(rows)


def samples(main, control):
    rows = []
    for sess, b in [("main", main), ("control", control)]:
        gaps = (b.ts0.shift(-1) - b.ts1).dt.total_seconds().dropna()
        for cond, a in b.groupby("Condition", sort=False):
            ai = a.AI_Batches.dropna()
            rows.append({
                "session": sess, "condition": cond, "tests": len(a),
                "frames_min": int(a.Game_Frames.min()), "frames_median": int(a.Game_Frames.median()),
                "frames_max": int(a.Game_Frames.max()),
                "batches_min": int(ai.min()) if len(ai) else "", "batches_median": int(ai.median()) if len(ai) else "",
                "batches_max": int(ai.max()) if len(ai) else "",
                "window_s_min": round(a.Measure_Window_s.min(), 1), "window_s_median": round(a.Measure_Window_s.median(), 1),
                "window_s_max": round(a.Measure_Window_s.max(), 1),
                "telemetry_samples_median": int(a.Telemetry_Samples.median()),
                "pretest_ms_mean": round(a.Pre_Test_Frame_ms.mean(), 3), "pretest_ms_sd": round(a.Pre_Test_Frame_ms.std(), 3),
                "drift_ref_ms": round(a.Drift_Ref_ms.iloc[0], 4),
                "pretest_max_dev_pct": round(((a.Pre_Test_Frame_ms / a.Drift_Ref_ms) - 1).abs().max() * 100, 1),
                "session_gap_s_median": round(gaps.median(), 2),
                "game_frames_total": int(a.Game_Frames.sum()), "ai_batches_total": int(ai.sum()) if len(ai) else "",
            })
    return pd.DataFrame(rows)


def order_balance(main, control):
    rows = []
    for sess, b in [("main", main), ("control", control)]:
        ct = pd.crosstab(b.Condition, b.Block_Pos)
        for cond, r in ct.iterrows():
            rows.append({"session": sess, "condition": cond, **{f"pos_{int(k)}": int(v) for k, v in r.items()}})
    return pd.DataFrame(rows)


YES_NO = {
    "package_thermal_throttling": "Package/Ring Thermal Throttling [Yes/No]",
    "core_thermal_throttling": "Core Thermal Throttling (avg) [Yes/No]",
    "package_power_limit_exceeded": "Package/Ring Power Limit Exceeded [Yes/No]",
    "ia_prochot": "IA: PROCHOT [Yes/No]",
    "ia_thermal_event": "IA: Thermal Event [Yes/No]",
    "ia_running_avg_thermal_limit": "IA: Running Average Thermal Limit [Yes/No]",
    "ia_vr_tdc": "IA: VR TDC [Yes/No]",
    "ia_edp_other": "IA: Electrical Design Point/Other (ICCmax PL4 SVID DDR RAPL) [Yes/No]",
    "ia_pl1": "IA: Package-Level RAPL/PBM PL1 [Yes/No]",
    "ia_pl2_pl3": "IA: Package-Level RAPL/PBM PL2 PL3 [Yes/No]",
    "ia_max_turbo_limit": "IA: Max Turbo Limit [Yes/No]",
    "ia_turbo_attenuation": "IA: Turbo Attenuation (MCT) [Yes/No]",
}
NUMERIC = {
    "package_temp_c": "CPU Package [°C].1",
    "core_distance_to_tjmax_c": "Core Distance to TjMAX (avg) [°C]",
    "package_power_w": "CPU Package Power [W]",
    "ia_cores_power_w": "IA Cores Power [W]",
    "gt_cores_power_w": "GT Cores Power [W]",
    "system_agent_power_w": "System Agent Power [W]",
    "pl1_static_w": "PL1 Power Limit (Static) [W]",
    "pl1_dynamic_w": "PL1 Power Limit (Dynamic) [W]",
    "pl2_static_w": "PL2 Power Limit (Static) [W]",
    "pl2_dynamic_w": "PL2 Power Limit (Dynamic) [W]",
    "ring_clock_mhz": "Ring/LLC Clock [MHz]",
    "memory_clock_mhz": "Memory Clock [MHz]",
    "total_cpu_usage_pct": "Total CPU Usage [%]",
}


def operating_point(sessions):
    rows = []
    for sess, b, h in sessions:
        hw = h[(h.ts >= b.ts0.min()) & (h.ts <= b.ts1.max())]
        r = {"session": sess, "hwinfo_rows": len(hw)}
        for k, c in NUMERIC.items():
            v = pd.to_numeric(hw[c], errors="coerce")
            r[f"{k}_mean"] = round(v.mean(), 2)
            r[f"{k}_min"] = round(v.min(), 2)
            r[f"{k}_max"] = round(v.max(), 2)
        for k, c in YES_NO.items():
            r[f"{k}_yes_pct"] = round((hw[c] == "Yes").mean() * 100, 1)
        pc = [f"P-core {i} Clock [MHz]" for i in range(6)]
        ec = [f"E-core {i} Clock [MHz]" for i in range(6, 14)]
        r["pcore_clock_mhz_median"] = round(np.nanmedian(hw[pc].apply(pd.to_numeric, errors="coerce").values), 0)
        r["ecore_clock_mhz_median"] = round(np.nanmedian(hw[ec].apply(pd.to_numeric, errors="coerce").values), 0)
        rows.append(r)
    return pd.DataFrame(rows).set_index("session").T.reset_index().rename(columns={"index": "measure"})


def configs(main, w):
    ai = ai_latency_table(main)
    p_mean = ai.pivot_table(index=["inputs", "hidden", "layers", "agents"], columns="condition", values="ai_mean_ms")
    p_p99 = ai.pivot_table(index=["inputs", "hidden", "layers", "agents"], columns="condition", values="ai_p99_ms")
    first = ai.groupby(["inputs", "hidden", "layers", "agents"])[["params", "mflop_batch"]].first()
    w = w.copy()
    for cond in AI_CONDS:
        w[f"mean_{cond}"] = change_pct(w, "Game_Avg_ms", cond)
        w[f"p99_{cond}"] = change_pct(w, "Game_99th_ms", cond)
    g = w.groupby(CONFIG)[[f"{k}_{c}" for k in ("mean", "p99") for c in AI_CONDS]].median()
    g.index = g.index.set_names(["inputs", "hidden", "layers", "agents"])
    t = first.join(p_mean.add_prefix("ai_mean_ms_")).join(p_p99.add_prefix("ai_p99_ms_"))
    t = t.join(g.add_prefix("game_").add_suffix("_change_pct"))
    t = t.reset_index().sort_values(["mflop_batch", "inputs", "hidden", "layers", "agents"]).reset_index(drop=True)
    for c in t.columns:
        if c.startswith("ai_") or c.startswith("game_"):
            t[c] = t[c].round(2)
    t["mflop_batch"] = t.mflop_batch.round(1)
    return t


def doc_facts(main, control, w):
    F = []

    def add(fid, value, desc):
        F.append({"id": fid, "value": value, "description": desc})

    keys = lambda b: b[BLOCK].astype(str).agg("|".join, axis=1)
    km, kc = keys(main), keys(control)
    om, oc = km[km != km.shift()].tolist(), kc[kc != kc.shift()].tolist()
    add("same_block_order", om == oc, "The control session ran its 540 blocks in the same order as the main session")
    add("main_frames_total", int(main.Game_Frames.sum()), "Main session: game frames measured, all tests")
    add("control_frames_total", int(control.Game_Frames.sum()), "Control session: game frames measured, all tests")
    add("main_batches_total", int(main.AI_Batches.sum()), "Main session: AI batches measured, all tests")
    add("control_batches_total", int(control.AI_Batches.sum()), "Control session: AI batches measured, all tests")
    sat = w[w["AI_Duty_Cycle|P_SHARED"] > 0.99]
    add("shared_sat_blocks", len(sat), "Blocks in which the P_SHARED AI thread was busy over 99% of the window")
    add("shared_sat_base_ms", round(sat["Game_Avg_ms|BASELINE"].median(), 2), "BASELINE mean frame of those blocks, median (ms)")
    add("idle_vs_base_p99_pct", round(((w["Game_99th_ms|BASELINE"] / w["Game_99th_ms|SMT_IDLE"]) - 1).median() * 100, 1),
        "BASELINE P99 frame vs SMT_IDLE P99 frame of the same block, median change (%)")
    add("idle_mean_ms", round(w["Game_Avg_ms|SMT_IDLE"].median(), 2), "SMT_IDLE mean frame, median (ms)")
    add("smt_cfg_over_baseline", int((w["Game_Avg_ms|P_SMT2"] > w["Game_Avg_ms|BASELINE"]).sum()),
        "Blocks in which the P_SMT mean frame (control session) exceeded the BASELINE mean frame of the same block")
    return pd.DataFrame(F)


def main_cli():
    ap = argparse.ArgumentParser(description="Extra tables quoted in the experiment record.")
    ap.add_argument("--main", required=True)
    ap.add_argument("--main-hwinfo", required=True)
    ap.add_argument("--control", required=True)
    ap.add_argument("--control-hwinfo", required=True)
    ap.add_argument("--out", default="out/doc")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    main, control = load_bench(a.main), load_bench(a.control)
    hm, hc = load_hwinfo(a.main_hwinfo), load_hwinfo(a.control_hwinfo)
    w = wide_by_block(main, control)
    tables = [("effect_tests", effect_tests(w)), ("ai_summary", ai_summary(main, control, ai_latency_table(main))),
              ("samples", samples(main, control)), ("order_balance", order_balance(main, control)),
              ("operating_point", operating_point([("main", main, hm), ("control", control, hc)])),
              ("configs", configs(main, w)), ("doc_facts", doc_facts(main, control, w))]
    for name, df in tables:
        df.to_csv(out / f"{name}.csv", index=False)
        print(f"wrote {name}.csv ({len(df)} rows)")


if __name__ == "__main__":
    main_cli()
