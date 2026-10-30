"""
make_thesis_figures.py -- thesis-size figures (6.5 in text width) from the tables written by audit_analysis.py.

Usage:
    python make_thesis_figures.py --tables out --figs thesis_figures

Writes thesis_fig1 .. thesis_fig8 as PNG (300 dpi) and PDF. Same palette and marker shapes as
make_paper_figures.py, so every series stays readable in grayscale print. Figure titles are not drawn
inside the images; they go above each figure in the manuscript (APA 7).
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Patch, Rectangle
from matplotlib.ticker import FuncFormatter, NullFormatter

COLOR = {"E_CORE": "#2a78d6", "P_SMT": "#eb6834", "P_SHARED": "#1baf7a"}
MARKER = {"E_CORE": "o", "P_SMT": "s", "P_SHARED": "^"}
LABEL = {"E_CORE": "E-cores", "P_SMT": "SMT sibling", "P_SHARED": "Shared LP"}
INK, SECONDARY, MUTED, GRID, AXIS = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
GAME, WORKER, TELEM = "#3d3c38", "#d6d5cc", "#efeee8"
DASH = (0, (4, 3))
MINUS = "−"
EN = "–"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Liberation Sans", "Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 9, "axes.labelsize": 9, "axes.titlesize": 9, "xtick.labelsize": 8, "ytick.labelsize": 8,
    "legend.fontsize": 8, "pdf.fonttype": 42, "ps.fonttype": 42,
    "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": AXIS, "axes.linewidth": 0.7,
    "axes.labelcolor": INK, "text.color": INK, "xtick.color": SECONDARY, "ytick.color": SECONDARY,
    "xtick.major.width": 0.7, "ytick.major.width": 0.7, "xtick.major.size": 3, "ytick.major.size": 3,
    "axes.grid": True, "axes.grid.axis": "y", "grid.color": GRID, "grid.linewidth": 0.6,
    "lines.linewidth": 1.5, "lines.markersize": 5, "legend.frameon": False,
    "savefig.dpi": 300, "savefig.bbox": "tight", "savefig.pad_inches": 0.03,
})


def signed(fmt, v):
    return fmt.format(v).replace("-", MINUS)


def minus_axis(axis):
    axis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}".replace("-", MINUS)))


def plain_log_axis(axis, compact=False):
    def f(v, _):
        if compact and v >= 1000:
            return f"{v / 1000:g}k"
        return f"{v:,.0f}" if v >= 1 else f"{v:g}"
    axis.set_major_formatter(FuncFormatter(f))
    axis.set_minor_formatter(NullFormatter())


def save(fig, figs, name):
    fig.savefig(figs / f"{name}.png")
    fig.savefig(figs / f"{name}.pdf")
    plt.close(fig)
    print(f"wrote {name}")


# ---------------------------------------------------------------- Figure 1: thread placement
def fig1_placement(figs):
    conds = ["BASELINE", "P_SHARED", "P_SMT", "SMT_IDLE", "E_CORE"]
    sub = {"BASELINE": "no AI", "P_SHARED": "AI on LP 0", "P_SMT": "AI on LP 1", "SMT_IDLE": "no AI, LP 1 idle",
           "E_CORE": "AI on E-cores"}

    def xpos(lp):
        if lp <= 11:
            return lp + (lp // 2) * 0.18
        base = 11 + 5 * 0.18 + 1 + 0.55
        return base + (lp - 12) + (0.35 if lp >= 16 else 0)

    fig, ax = plt.subplots(figsize=(6.5, 3.05))
    ax.set_axis_off()
    w, h = 0.9, 0.78
    rows = {c: len(conds) - 1 - i for i, c in enumerate(conds)}

    def cell(lp, row, face, edge=None, ls="-", text=None, tcolor=INK, hatch=None):
        x = xpos(lp)
        ax.add_patch(Rectangle((x, row), w, h, facecolor=face, edgecolor=edge or face, linewidth=0.7,
                               linestyle=ls, hatch=hatch))
        if text:
            ax.text(x + w / 2, row + h / 2, text, ha="center", va="center", fontsize=7.5, color=tcolor,
                    fontweight="bold")

    for c in conds:
        r = rows[c]
        # LP 0: game (P_SHARED: game and AI share it)
        if c == "P_SHARED":
            x = xpos(0)
            ax.add_patch(Rectangle((x, r + h / 2), w, h / 2, facecolor=GAME, edgecolor=GAME, linewidth=0.7))
            ax.add_patch(Rectangle((x, r), w, h / 2, facecolor=COLOR["P_SHARED"], edgecolor=COLOR["P_SHARED"],
                                   linewidth=0.7))
            ax.text(x + w / 2, r + 3 * h / 4, "G", ha="center", va="center", fontsize=6.5, color="white",
                    fontweight="bold")
            ax.text(x + w / 2, r + h / 4, "AI", ha="center", va="center", fontsize=6.5, color="white",
                    fontweight="bold")
        else:
            cell(0, r, GAME, text="G", tcolor="white")
        # LP 1
        if c == "P_SMT":
            cell(1, r, COLOR["P_SMT"], text="AI", tcolor="white")
        elif c == "SMT_IDLE":
            cell(1, r, "white", edge=SECONDARY, ls=(0, (2, 1.5)))
        else:
            cell(1, r, WORKER)
        for lp in range(2, 12):
            cell(lp, r, WORKER)
        if c == "E_CORE":
            ax.add_patch(Rectangle((xpos(12), r), xpos(18) + w - xpos(12), h, facecolor="#cfe0f6",
                                   edgecolor=COLOR["E_CORE"], linewidth=0.9))
        else:
            for lp in range(12, 19):
                cell(lp, r, "white", edge=AXIS)
        cell(19, r, TELEM, edge=MUTED, text="T", tcolor=SECONDARY)
        if c == "E_CORE":
            x0, x1 = xpos(12), xpos(18) + w
            ax.text((x0 + x1) / 2, r + h / 2, "one AI thread; Windows\nchooses among LP 12–18", ha="center",
                    va="center", fontsize=7.2, color="#123f73", fontweight="bold", linespacing=1.15)
        ax.text(-0.35, r + h * 0.62, c, ha="right", va="center", fontsize=8.5, fontweight="bold", color=INK)
        ax.text(-0.35, r + h * 0.20, sub[c], ha="right", va="center", fontsize=7.5, color=SECONDARY)

    top = len(conds) - 1 + h
    for lp in range(20):
        ax.text(xpos(lp) + w / 2, top + 0.12, str(lp), ha="center", va="bottom", fontsize=7.5, color=SECONDARY)
    groups = [(0, 1, "P-core 0", 0.62), (2, 11, "P-cores 1–5 (two LPs each)", 0.62),
              (12, 15, "cluster 1", 0.62), (16, 19, "cluster 2", 0.62), (12, 19, "E-cores (one LP each)", 1.15)]
    for a, b, name, dy in groups:
        x0, x1 = xpos(a), xpos(b) + w
        y = top + dy
        ax.plot([x0, x1], [y, y], color=AXIS, linewidth=0.9, solid_capstyle="butt")
        ax.text((x0 + x1) / 2, y + 0.08, name, ha="center", va="bottom", fontsize=7.5, color=INK)
    ax.text(-0.35, top + 0.12, "Logical processor", ha="right", va="bottom", fontsize=7.5, color=SECONDARY)

    handles = [Patch(facecolor=GAME, edgecolor=GAME, label="G  Game thread"),
               Patch(facecolor=WORKER, edgecolor=WORKER, label="Worker thread"),
               Patch(facecolor=COLOR["P_SHARED"], edgecolor=COLOR["P_SHARED"], label="AI thread (shared LP)"),
               Patch(facecolor=COLOR["P_SMT"], edgecolor=COLOR["P_SMT"], label="AI thread (sibling)"),
               Patch(facecolor="#cfe0f6", edgecolor=COLOR["E_CORE"], label="AI thread may run here"),
               Patch(facecolor="white", edgecolor=SECONDARY, linestyle=(0, (2, 1.5)), label="Worker parked, LP idle"),
               Patch(facecolor="white", edgecolor=AXIS, label="Idle E-core"),
               Patch(facecolor=TELEM, edgecolor=MUTED, label="T  Telemetry and control")]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.02), ncol=4, fontsize=7.5,
              handlelength=1.4, columnspacing=1.2, handletextpad=0.5)
    ax.set_xlim(-5.2, xpos(19) + w + 0.1)
    ax.set_ylim(-0.15, top + 1.6)
    save(fig, figs, "thesis_fig1_placement")


def paper_fig1_placement(figs):
    """Figure 1 drawn at the IEEE column width (3.5 in) so that its text prints at 7 pt or more without scaling;
    the E_CORE row marks the allowed set of its single AI thread."""
    conds = ["BASELINE", "P_SHARED", "P_SMT", "SMT_IDLE", "E_CORE"]
    sub = {"BASELINE": "no AI", "P_SHARED": "AI on LP 0", "P_SMT": "AI on LP 1", "SMT_IDLE": "no AI, LP 1 idle",
           "E_CORE": "AI on LP 12–18"}

    def xpos(lp):
        if lp <= 11:
            return lp + (lp // 2) * 0.22
        return 11 + 5 * 0.22 + 1.6 + (lp - 12)

    fig, ax = plt.subplots(figsize=(3.5, 1.45))
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)  # the drawing fills the column width at 1:1
    ax.set_axis_off()
    w, h = 0.86, 0.74
    rows = {c: len(conds) - 1 - i for i, c in enumerate(conds)}

    def cell(lp, row, face, edge=None, ls="-", text=None, tcolor=INK):
        x = xpos(lp)
        ax.add_patch(Rectangle((x, row), w, h, facecolor=face, edgecolor=edge or face, linewidth=0.6, linestyle=ls))
        if text:
            ax.text(x + w / 2, row + h / 2, text, ha="center", va="center", fontsize=7, color=tcolor, fontweight="bold")

    for c in conds:
        r = rows[c]
        if c == "P_SHARED":
            x = xpos(0)
            ax.add_patch(Rectangle((x, r + h / 2), w, h / 2, facecolor=GAME, edgecolor=GAME, linewidth=0.6))
            ax.add_patch(Rectangle((x, r), w, h / 2, facecolor=COLOR["P_SHARED"], edgecolor=COLOR["P_SHARED"],
                                   linewidth=0.6))
        else:
            cell(0, r, GAME, text="G", tcolor="white")
        if c == "P_SMT":
            cell(1, r, COLOR["P_SMT"], text="AI", tcolor="white")
        elif c == "SMT_IDLE":
            cell(1, r, "white", edge=SECONDARY, ls=(0, (2, 1.5)))
        else:
            cell(1, r, WORKER)
        for lp in range(2, 12):
            cell(lp, r, WORKER)
        if c == "E_CORE":
            x0, x1 = xpos(12), xpos(18) + w
            ax.add_patch(Rectangle((x0, r), x1 - x0, h, facecolor="#cfe0f6", edgecolor=COLOR["E_CORE"], linewidth=0.9))
            ax.text((x0 + x1) / 2, r + h / 2, "1 AI thread", ha="center", va="center", fontsize=7,
                    color="#123f73", fontweight="bold")
        else:
            for lp in range(12, 19):
                cell(lp, r, "white", edge=AXIS)
        cell(19, r, TELEM, edge=MUTED, text="T", tcolor=SECONDARY)
        ax.text(-0.4, r + h * 0.78, c, ha="right", va="center", fontsize=7, fontweight="bold", color=INK,
                family="monospace")
        ax.text(-0.4, r + h * 0.10, sub[c], ha="right", va="center", fontsize=7, color=SECONDARY)

    top = len(conds) - 1 + h
    # LP numbers: single LPs where a condition differs, ranges elsewhere (20 two-digit labels do not fit at 7 pt)
    for a, b in [(0, 0), (1, 1), (2, 11), (12, 18), (19, 19)]:
        label = str(a) if a == b else f"{a}–{b}"
        ax.text((xpos(a) + xpos(b) + w) / 2, top + 0.1, label, ha="center", va="bottom", fontsize=7, color=SECONDARY)
    for a, b, name in [(0, 1, "P-core 0"), (2, 11, "P-cores 1–5"), (12, 19, "E-cores")]:
        x0, x1 = xpos(a), xpos(b) + w
        y = top + 0.68
        ax.plot([x0, x1], [y, y], color=AXIS, linewidth=0.8, solid_capstyle="butt")
        ax.text((x0 + x1) / 2, y + 0.07, name, ha="center", va="bottom", fontsize=7, color=INK)
    ax.text(-0.4, top + 0.1, "LP", ha="right", va="bottom", fontsize=7, color=SECONDARY)

    handles = [Patch(facecolor=GAME, edgecolor=GAME, label="G  Game thread"),
               Patch(facecolor=WORKER, edgecolor=WORKER, label="Worker thread"),
               Patch(facecolor=COLOR["P_SHARED"], edgecolor=COLOR["P_SHARED"], label="AI thread, shared LP"),
               Patch(facecolor=COLOR["P_SMT"], edgecolor=COLOR["P_SMT"], label="AI thread, sibling"),
               Patch(facecolor="#cfe0f6", edgecolor=COLOR["E_CORE"], label="LPs allowed to the AI thread"),
               Patch(facecolor="white", edgecolor=SECONDARY, linestyle=(0, (2, 1.5)), label="Worker parked, LP idle"),
               Patch(facecolor="white", edgecolor=AXIS, label="Idle E-core LP"),
               Patch(facecolor=TELEM, edgecolor=MUTED, label="T  Telemetry and control")]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.44, 0.03), ncol=2, fontsize=7,
              handlelength=1.2, columnspacing=1.0, handletextpad=0.4, labelspacing=0.3)
    ax.set_xlim(-6.6, xpos(19) + w + 0.05)
    ax.set_ylim(-0.1, top + 1.25)
    save(fig, figs, "paper_fig1_placement")


# ---------------------------------------------------------------- Figure 2: procedure
def fig2_procedure(figs):
    fig, ax = plt.subplots(figsize=(6.5, 6.0))
    ax.set_axis_off()
    session = [
        ("Launcher", "checks administrator rights, AC power and\nmodel files; stops SysMain and Windows\nUpdate if they are running"),
        ("Process set-up", "HIGH_PRIORITY_CLASS; EcoQoS throttling\noff; high-resolution waitable timer;\none ONNX Runtime environment"),
        ("Topology and workers", "game LP 0, sibling LP 1 and control LP 19\nread from Windows; 11 physics workers\nstart on LP 1\u201311 and run all session"),
        ("Soak", "180 s of back-to-back game frames\nwith every worker running"),
        ("Calibration", "main: frame loop scaled to 16 ms\n(338,571 iterations); control: loaded\nfrom calibrated_loops.txt"),
        ("Plan", "540 blocks (180 configurations \u00d7 3 trials)\nin shuffled order; conditions shuffled\ninside each block (seed 1337)"),
    ]
    test = [
        ("(a) Worker set", "LP 1's worker parked for P_SMT and\nSMT_IDLE; otherwise it keeps running"),
        ("(b) Pre-test check", "median of 60 frames on LP 0; drift flag\nif more than 15% from the reference"),
        ("(c) Launch and warm-up", "game keeps LP 0 busy; AI thread pins,\nopens the model, runs 5 warm-up batches"),
        ("(d) Measurement window", "every frame timed; one AI batch per\n50 ms tick; counters every 100 ms"),
        ("(e) Stop rule", "\u22651,000 frames and \u226550 AI batches\n(controls: \u22652.5 s); cap 300 s"),
        ("(f) Record", "one 49-column CSV row, flushed to disk;\nAC power checked again"),
    ]
    bw, bh, gap = 4.6, 1.62, 0.32
    xs, xt = 0.0, 5.35
    ytop = 10.0

    def column(items, x, face, title):
        ax.text(x + bw / 2, ytop + 0.55, title, ha="center", va="bottom", fontsize=9.5, fontweight="bold",
                color=INK)
        ys = []
        for i, (head, body) in enumerate(items):
            y = ytop - (i + 1) * bh - i * gap
            ys.append(y)
            ax.add_patch(FancyBboxPatch((x, y), bw, bh, boxstyle="round,pad=0.02,rounding_size=0.12",
                                        facecolor=face, edgecolor=AXIS, linewidth=0.8))
            ax.text(x + 0.16, y + bh - 0.16, head, ha="left", va="top", fontsize=8.5, fontweight="bold", color=INK)
            ax.text(x + 0.16, y + bh - 0.50, body, ha="left", va="top", fontsize=7.4, color=SECONDARY,
                    linespacing=1.3)
            if i:
                ax.add_patch(FancyArrowPatch((x + bw / 2, y + bh + gap - 0.01), (x + bw / 2, y + bh + 0.02),
                                             arrowstyle="-|>", mutation_scale=8, color=MUTED, linewidth=0.9))
        return ys

    ys = column(session, xs, "#f6f5f1", "Once per session")
    yt = column(test, xt, "#eef4fc", "For every test")
    xm = (xs + bw + xt) / 2
    ax.plot([xs + bw, xm, xm], [ys[-1] + bh / 2, ys[-1] + bh / 2, yt[0] + bh / 2], color=MUTED, linewidth=1.0)
    ax.add_patch(FancyArrowPatch((xm, yt[0] + bh / 2), (xt + 0.02, yt[0] + bh / 2), arrowstyle="-|>",
                                 mutation_scale=10, color=MUTED, linewidth=1.0))
    xr = xt + bw + 0.25
    ax.plot([xt + bw, xr, xr], [yt[-1] + bh / 2, yt[-1] + bh / 2, yt[0] + bh / 2], color=COLOR["E_CORE"],
            linewidth=1.0)
    ax.add_patch(FancyArrowPatch((xr, yt[0] + bh / 2), (xt + bw + 0.02, yt[0] + bh / 2), arrowstyle="-|>",
                                 mutation_scale=9, color=COLOR["E_CORE"], linewidth=1.0))
    ax.text(xr + 0.1, (yt[0] + yt[-1] + bh) / 2, "next test (2,160 main; 1,080 control)", rotation=90,
            ha="left", va="center", fontsize=7.5, color=COLOR["E_CORE"])
    ax.set_xlim(-0.05, xr + 0.45)
    ax.set_ylim(ys[-1] - 0.1, ytop + 1.0)
    save(fig, figs, "thesis_fig2_procedure")


# ---------------------------------------------------------------- Figure 3: operating point
def fig3_operating_point(hrs, figs):
    hrs = hrs.sort_values("hour")
    x = hrs.hour.values
    labels = hrs.start_time.values
    panels = [
        ("hwinfo_pkg_temp_c", "(a) CPU package temperature (°C)", (90, 100.8), 100, "Tj,max 100 °C"),
        ("baseline_rapl_w", "(b) Package power, RAPL (W)", (43.5, 47.0), None, None),
        ("baseline_pcore_perf_pct", "(c) P-core % Processor Performance", (106.5, 111.0), None, None),
        ("baseline_frame_ms", "(d) Game mean frame time (ms)", (15.6, 16.6), 16.0, "16 ms target"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(6.5, 4.0), sharex=True)
    for ax, (col, title, ylim, ref, reflab) in zip(axes.flat, panels):
        ax.plot(x, hrs[col], color=GAME, marker="o", markersize=4, markeredgecolor="white", markeredgewidth=0.5)
        if ref is not None:
            ax.axhline(ref, color=MUTED, linestyle=DASH, linewidth=0.9)
            ax.text(x[-1] + 0.3, ref, reflab, ha="right", va="bottom", fontsize=7.5, color=SECONDARY)
        ax.set_ylim(*ylim)
        ax.set_title(title, loc="left", fontsize=9)
    for ax in axes[1]:
        ax.set_xticks(x[::2])
        ax.set_xticklabels(labels[::2])
        ax.set_xlabel("Start of session hour (CDT, 11 Oct 2026)")
    fig.tight_layout(h_pad=1.2, w_pad=1.5)
    save(fig, figs, "thesis_fig3_operating_point")


# ---------------------------------------------------------------- Figure 4: game cost vs work
def fig4_game_cost(t, figs, height=2.9, name="thesis_fig4_game_cost"):
    fig, axes = plt.subplots(1, 2, figsize=(6.5, height))
    for ax, (key, title) in zip(axes, [("mean", "(a) Mean frame time change (%)"),
                                       ("p99", "(b) P99 frame time change (%)")]):
        for cond in ["P_SHARED", "P_SMT", "E_CORE"]:
            d = t[t.condition == cond].sort_values("bin_order")
            ax.fill_between(d.mflop_median, d[f"{key}_change_q1"], d[f"{key}_change_q3"], color=COLOR[cond],
                            alpha=0.15, linewidth=0)
            ax.plot(d.mflop_median, d[f"{key}_change_pct"], color=COLOR[cond], marker=MARKER[cond],
                    markeredgecolor="white", markeredgewidth=0.5)
        ax.axhline(0, color=AXIS, linewidth=0.7, zorder=0)
        ax.set_xscale("log")
        plain_log_axis(ax.xaxis, compact=True)
        minus_axis(ax.yaxis)
        ax.set_title(title, loc="left", fontsize=9)
        ax.set_xlabel("AI work per batch (MFLOP)")
        # Inset: the E-core series on its own scale (the shared-LP scale compresses it to the zero line)
        d = t[t.condition == "E_CORE"].sort_values("bin_order")
        ins = ax.inset_axes([0.13, 0.50, 0.42, 0.40])
        ins.fill_between(d.mflop_median, d[f"{key}_change_q1"], d[f"{key}_change_q3"], color=COLOR["E_CORE"],
                         alpha=0.15, linewidth=0)
        ins.plot(d.mflop_median, d[f"{key}_change_pct"], color=COLOR["E_CORE"], marker=MARKER["E_CORE"],
                 markersize=3.5, markeredgecolor="white", markeredgewidth=0.4, linewidth=1.2)
        ins.axhline(0, color=AXIS, linewidth=0.6, zorder=0)
        ins.set_xscale("log")
        ins.set_xticks([10, 100, 1000, 10000])
        plain_log_axis(ins.xaxis, compact=True)
        minus_axis(ins.yaxis)
        ins.tick_params(labelsize=7.5, length=2, pad=1.5)
        ins.set_title("E-cores only (%)", loc="left", fontsize=8, pad=2, color=COLOR["E_CORE"])
        for side in ("top", "right"):
            ins.spines[side].set_visible(True)
        ins.set_facecolor("white")
    handles = [Line2D([], [], color=COLOR[c], marker=MARKER[c], markeredgecolor="white", markeredgewidth=0.5,
                      label=f"{LABEL[c]} ({c})") for c in ["P_SHARED", "P_SMT", "E_CORE"]]
    fig.legend(handles=handles, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.06), handlelength=1.8,
               columnspacing=1.4)
    fig.tight_layout(w_pad=2.0)
    save(fig, figs, name)


# ---------------------------------------------------------------- Figure 5: AI latency
def fig5_ai_latency(a, figs, height=3.0, name="thesis_fig5_ai_latency"):
    fig, axes = plt.subplots(1, 2, figsize=(6.5, height), gridspec_kw={"width_ratios": [1.25, 1]})
    ax = axes[0]
    for cond in ["P_SHARED", "E_CORE", "P_SMT"]:
        d = a[a.condition == cond]
        ax.scatter(d.mflop_batch, d.ai_p99_ms, s=11, marker=MARKER[cond], color=COLOR[cond], edgecolors="white",
                   linewidths=0.3, zorder=3)
    ax.axhline(50, color=MUTED, linewidth=0.9, linestyle=DASH, zorder=2)
    ax.text(a.mflop_batch.min() * 1.05, 50 * 1.2, "50-ms update period", color=SECONDARY, fontsize=7.5)
    ax.set_xscale("log")
    ax.set_yscale("log")
    plain_log_axis(ax.xaxis, compact=True)
    plain_log_axis(ax.yaxis)
    ax.grid(True, which="major", axis="both")
    ax.set_xlabel("AI work per batch (MFLOP)")
    ax.set_title("(a) P99 batch latency (ms)", loc="left", fontsize=9)

    ax = axes[1]
    a = a.assign(gflops=a.mflop_batch / a.ai_mean_ms)
    g = a.groupby(["condition", "hidden"]).gflops.median().unstack(0)
    for cond in ["P_SMT", "E_CORE", "P_SHARED"]:
        ls = DASH if cond == "P_SHARED" else "-"
        ax.plot(g.index, g[cond], color=COLOR[cond], marker=MARKER[cond], markeredgecolor="white",
                markeredgewidth=0.5, linestyle=ls, zorder=4 if cond == "P_SHARED" else 3,
                markersize=6 if cond == "P_SHARED" else 5)
    ax.set_xscale("log", base=2)
    ax.set_xticks(list(g.index))
    ax.set_xticklabels([str(v) for v in g.index])
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_ylim(0, g.values.max() * 1.15)
    ax.set_xlabel("GRU hidden size")
    ax.set_title("(b) Inference throughput (GFLOP/s)", loc="left", fontsize=9)
    handles = [Line2D([], [], color=COLOR[c], marker=MARKER[c], markeredgecolor="white", markeredgewidth=0.5,
                      label=f"{LABEL[c]} ({c})") for c in ["P_SHARED", "P_SMT", "E_CORE"]]
    fig.legend(handles=handles, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.06), handlelength=1.8,
               columnspacing=1.4)
    fig.tight_layout(w_pad=2.0)
    save(fig, figs, name)


# ---------------------------------------------------------------- Figure 6: E-core dose response
def fig6_ecore_dose(e, figs):
    e = e.sort_values("bin_order")
    ticks = ["<10", f"10{EN}25", f"25{EN}50", f"50{EN}75", f"75{EN}99", "100 (never idle)"]
    panels = [("d_package_w", "(a) Package power change (W)", "{:+.2f}"),
              ("d_pcore_perf_pts", "(b) P-core % Processor Performance change (points)", "{:+.2f}"),
              ("d_game_frame_pct", "(c) Game mean frame time change (%)", "{:+.2f}")]
    fig, axes = plt.subplots(3, 1, figsize=(6.5, 5.0), sharex=True)
    x = np.arange(len(e))
    for ax, (col, title, fmt) in zip(axes, panels):
        v = e[col].values
        ax.bar(x, v, width=0.55, color=COLOR["E_CORE"], linewidth=0, zorder=3)
        ax.axhline(0, color=AXIS, linewidth=0.7, zorder=4)
        ax.set_title(title, loc="left", fontsize=9)
        minus_axis(ax.yaxis)
        lo, hi = min(0, v.min()), max(0, v.max())
        pad = (hi - lo) * 0.25
        ax.set_ylim(lo - (pad if lo < 0 else 0), hi + (pad if hi > 0 else 0))
        for xi, vi in zip(x, v):
            ax.annotate(signed(fmt, vi), (xi, vi), xytext=(0, 2 if vi >= 0 else -2), textcoords="offset points",
                        ha="center", va="bottom" if vi >= 0 else "top", fontsize=7.5, color=SECONDARY)
    axes[-1].set_xticks(x)
    axes[-1].set_xticklabels(ticks)
    axes[-1].set_xlabel("E-core AI duty cycle (%)")
    fig.tight_layout(h_pad=0.8)
    save(fig, figs, "thesis_fig6_ecore_dose")


# ---------------------------------------------------------------- Figure 7: SMT sibling contention
def fig7_smt(p, figs):
    p = p.sort_values("bin_order")
    ticks = ["<2", f"2{EN}10", f"10{EN}25", f"25{EN}50", f"50{EN}75", f"75{EN}99", "100"]
    x = np.arange(len(p))
    fig, axes = plt.subplots(1, 2, figsize=(6.5, 2.9), sharey=True)
    ax = axes[0]
    ax.plot(x, p.baseline_mean_ms, color=MUTED, linewidth=1.0, linestyle=DASH)
    ax.plot(x, p.idle_mean_ms, color=SECONDARY, marker="o", markeredgecolor="white", markeredgewidth=0.5)
    ax.plot(x, p.smt_mean_ms, color=COLOR["P_SMT"], marker="s", markeredgecolor="white", markeredgewidth=0.5)
    ax.text(0, p.baseline_mean_ms.iloc[0] + 0.45, "BASELINE (worker on LP 1)", fontsize=7.5, color=SECONDARY)
    ax.set_title("(a) Game mean frame time (ms)", loc="left", fontsize=9)
    ax = axes[1]
    ax.plot(x, p.idle_p99_ms, color=SECONDARY, marker="o", markeredgecolor="white", markeredgewidth=0.5)
    ax.plot(x, p.smt_p99_ms, color=COLOR["P_SMT"], marker="s", markeredgecolor="white", markeredgewidth=0.5)
    ax.set_title("(b) Game P99 frame time (ms)", loc="left", fontsize=9)
    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels(ticks, fontsize=7.5)
        ax.set_xlabel("AI duty cycle on the SMT sibling (%)")
        ax.set_ylim(0, 22)
    handles = [Line2D([], [], color=COLOR["P_SMT"], marker="s", markeredgecolor="white", markeredgewidth=0.5,
                      label="P_SMT (AI on LP 1)"),
               Line2D([], [], color=SECONDARY, marker="o", markeredgecolor="white", markeredgewidth=0.5,
                      label="SMT_IDLE (LP 1 idle)"),
               Line2D([], [], color=MUTED, linestyle=DASH, linewidth=1.0, label="BASELINE mean (main session)")]
    fig.legend(handles=handles, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.06), handlelength=2.0,
               columnspacing=1.4)
    fig.tight_layout(w_pad=1.5)
    save(fig, figs, "thesis_fig7_smt")


# ---------------------------------------------------------------- Figure 8: same-LP stall
def fig8_stall(s, figs):
    s = s.sort_values("bin_order")
    ticks = ["<2", f"2{EN}5", f"5{EN}10", f"10{EN}20", f"20{EN}30", f"30{EN}40", f"40{EN}60", f"60{EN}100", ">100"]
    x = np.arange(len(s))
    fig, ax = plt.subplots(figsize=(6.0, 2.8))
    plateau = float(np.median(s.p99_stall_ms.values[-4:]))
    ax.axhline(plateau, color=MUTED, linestyle=DASH, linewidth=0.9)
    ax.text(0, plateau + 1.5, f"plateau {plateau:.1f} ms (median of the last four bands)", fontsize=7.5,
            color=SECONDARY)
    ax.plot(x, s.p99_stall_ms, color=COLOR["P_SHARED"], marker=MARKER["P_SHARED"], markeredgecolor="white",
            markeredgewidth=0.5, markersize=6)
    for xi, vi in zip(x, s.p99_stall_ms.values):
        if xi < 2:
            ax.annotate(f"{vi:.1f}", (xi, vi), xytext=(0, 7), textcoords="offset points", ha="center",
                        va="bottom", fontsize=7.5, color=SECONDARY)
        elif xi < 5:
            ax.annotate(f"{vi:.1f}", (xi, vi), xytext=(6, -3), textcoords="offset points", ha="left", va="top",
                        fontsize=7.5, color=SECONDARY)
    ax.set_xticks(x)
    ax.set_xticklabels(ticks)
    ax.set_ylim(0, 70)
    ax.set_xlabel("AI batch run time on a P-core thread (ms)")
    ax.set_ylabel("P99 frame above BASELINE\nmedian frame (ms)")
    fig.tight_layout()
    save(fig, figs, "thesis_fig8_stall")


# ---------------------------------------------------------------- Figure 6 (compact): E-core dose response, one row
def fig6c_ecore_compact(e, figs):
    e = e.sort_values("bin_order")
    ticks = ["<10", f"10{EN}25", f"25{EN}50", f"50{EN}75", f"75{EN}99", "100"]
    panels = [("d_package_w", "(a) Package power (W)", "{:+.2f}"),
              ("d_pcore_perf_pts", "(b) P-core performance (points)", "{:+.2f}"),
              ("d_game_frame_pct", "(c) Game mean frame time (%)", "{:+.2f}")]
    fig, axes = plt.subplots(1, 3, figsize=(6.5, 2.45))
    x = np.arange(len(e))
    for ax, (col, title, fmt) in zip(axes, panels):
        v = e[col].values
        ax.bar(x, v, width=0.62, color=COLOR["E_CORE"], linewidth=0, zorder=3)
        ax.axhline(0, color=AXIS, linewidth=0.7, zorder=4)
        ax.set_xticks(x)
        ax.set_xticklabels(ticks, rotation=40, ha="right", rotation_mode="anchor", fontsize=8)
        ax.set_title(title, loc="left", fontsize=8.5)
        minus_axis(ax.yaxis)
        lo, hi = min(0, v.min()), max(0, v.max())
        pad = (hi - lo) * 0.22
        ax.set_ylim(lo - (pad if lo < 0 else 0), hi + (pad if hi > 0 else 0))
        for xi in (0, len(v) - 1):
            vi = v[xi]
            ax.annotate(signed(fmt, vi), (xi, vi), xytext=(0, 2 if vi >= 0 else -2), textcoords="offset points",
                        ha="center", va="bottom" if vi >= 0 else "top", fontsize=7.5, color=SECONDARY)
    axes[1].set_xlabel("E-core AI duty cycle (%)")
    fig.tight_layout(w_pad=1.4)
    save(fig, figs, "paper_fig4_ecore_compact")


def main():
    ap = argparse.ArgumentParser(description="Thesis figures from audit_analysis.py tables.")
    ap.add_argument("--tables", default="out")
    ap.add_argument("--figs", default="thesis_figures")
    args = ap.parse_args()
    tables, figs = Path(args.tables), Path(args.figs)
    figs.mkdir(parents=True, exist_ok=True)
    fig1_placement(figs)
    paper_fig1_placement(figs)
    fig2_procedure(figs)
    fig3_operating_point(pd.read_csv(tables / "hourly.csv"), figs)
    fig4_game_cost(pd.read_csv(tables / "impact.csv"), figs)
    fig5_ai_latency(pd.read_csv(tables / "ai_latency.csv"), figs)
    # IEEE paper (Figs. 2 and 3): the same plots, shorter, to fit six pages with the validation session
    fig4_game_cost(pd.read_csv(tables / "impact.csv"), figs, height=2.4, name="paper_fig2_game_cost")
    fig5_ai_latency(pd.read_csv(tables / "ai_latency.csv"), figs, height=2.5, name="paper_fig3_ai_latency")
    fig6_ecore_dose(pd.read_csv(tables / "ecore_dose.csv"), figs)
    fig7_smt(pd.read_csv(tables / "psmt_duty.csv"), figs)
    fig8_stall(pd.read_csv(tables / "shared_stall.csv"), figs)
    paper_fig_ecore_dose(pd.read_csv(tables / "ecore_dose.csv"), figs)
    fig6c_ecore_compact(pd.read_csv(tables / "ecore_dose.csv"), figs)



# ---------------------------------------------------------------- journal-paper version of Figure 6 (compact)
def paper_fig_ecore_dose(e, figs):
    """E-core thermal coupling in one row of three panels for the journal paper (all text at least 8 pt).
    Horizontal bars keep the duty-cycle band labels readable without rotation."""
    e = e.sort_values("bin_order")
    labels = ["<10%", f"10{EN}25%", f"25{EN}50%", f"50{EN}75%", f"75{EN}99%", "100%"]
    panels = [("d_package_w", "(a) Package power (W)", "{:+.2f}"),
              ("d_pcore_perf_pts", "(b) P-core performance (points)", "{:+.2f}"),
              ("d_game_frame_pct", "(c) Game mean frame time (%)", "{:+.2f}")]
    fig, axes = plt.subplots(1, 3, figsize=(6.5, 2.35), sharey=True)
    y = np.arange(len(e))
    for ax, (col, title, fmt) in zip(axes, panels):
        v = e[col].values
        ax.barh(y, v, height=0.62, color=COLOR["E_CORE"], linewidth=0, zorder=3)
        ax.axvline(0, color=AXIS, linewidth=0.7, zorder=4)
        ax.grid(True, axis="x")
        ax.grid(False, axis="y")
        ax.tick_params(axis="y", length=0, labelsize=8)
        ax.tick_params(axis="x", labelsize=8)
        ax.set_title(title, loc="left", fontsize=9)
        ax.xaxis.set_major_formatter(FuncFormatter(lambda val, _: f"{val:g}".replace("-", MINUS)))
        lo, hi = min(0, v.min()), max(0, v.max())
        pad = (hi - lo) * 0.30
        ax.set_xlim(lo - (pad if lo < 0 else 0), hi + (pad if hi > 0 else 0))
        for yi, vi in zip(y, v):
            ax.annotate(signed(fmt, vi), (vi, yi), xytext=(3 if vi >= 0 else -3, 0), textcoords="offset points",
                        ha="left" if vi >= 0 else "right", va="center", fontsize=7.5 if False else 8,
                        color=SECONDARY)
    axes[0].set_yticks(y)
    axes[0].set_yticklabels(labels)
    axes[0].invert_yaxis()
    axes[0].set_ylabel("E-core AI duty cycle", fontsize=9)
    fig.tight_layout(w_pad=1.2)
    save(fig, figs, "paper_fig4_ecore_dose")


if __name__ == "__main__":
    main()
