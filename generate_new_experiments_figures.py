"""
New Figure 11 for FedTwin-CL: the compression-matched uplink comparison and
the capacity sweep, from the 2026-09-04/05 follow-up experiments
(EXPERIMENT_PROCEDURE.md Section 8). Same palette/rcParams as
generate_real_result_figures.py / generate_real_sweep_figures.py, reused
unchanged so this figure reads as part of the same visual system.

Panel A: cumulative fleet uplink (log scale), dense FedAvg vs.
compression-matched FedAvg vs. FedTwin-CL, both real benchmarks tested --
the single most important new finding this pass (masking's fair marginal
contribution is ~3-4x, not the ~70-100x the original dense-only comparison
implied).
Panel B: capacity sweep -- current-task-metric gap to FedAvg (FedAvg minus
FedTwin-CL) as a function of backbone width (1x/2x/4x), both benchmarks,
showing the gap narrows on Fed-Twin-FEMTO but not on Fed-Twin-CMAPSS.
"""
import json
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

FIGURES_DIR = os.path.join(os.path.dirname(__file__), "figures")
REAL_DIR = os.path.join(os.path.dirname(__file__), "real_data")
os.makedirs(FIGURES_DIR, exist_ok=True)

with open(os.path.join(REAL_DIR, "compression_matched_real_results.json")) as f:
    COMPRESSION = json.load(f)
with open(os.path.join(REAL_DIR, "capacity_sweep_real_results.json")) as f:
    CAPACITY = json.load(f)

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 10,
    "axes.titlesize": 10.5,
    "axes.labelsize": 9.5,
    "legend.fontsize": 8.5,
    "xtick.labelsize": 8.5,
    "ytick.labelsize": 8.5,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_MUTED = "#898781"
GRID = "#e1e0d9"
CAT = {
    "blue": "#2a78d6", "green": "#008300", "magenta": "#e87ba4", "yellow": "#eda100",
    "aqua": "#1baf7a", "orange": "#eb6834", "violet": "#4a3aa7", "red": "#e34948",
}


def _clean(ax):
    ax.set_facecolor(SURFACE)
    ax.grid(color=GRID, lw=0.9, zorder=0)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def fig11():
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(12.5, 5.0))
    fig.patch.set_facecolor(SURFACE)

    # Panel A: compression-matched uplink
    benches = ["cmapss", "femto"]
    labels = ["Fed-Twin-CMAPSS", "Fed-Twin-FEMTO"]
    dense = []
    matched = []
    fedtwincl = []
    for b in benches:
        existing = COMPRESSION[b]["existing_methods_uplink"]
        recs = COMPRESSION[b]["records"]
        dense.append(existing["fedavg"]["total_uplink_bytes"] / 1e6)
        matched.append(sum(r["uplink_bytes_compressed"] for r in recs) / 1e6)
        fedtwincl.append(existing["fedtwin-cl"]["total_uplink_bytes"] / 1e6)

    x = np.arange(len(benches))
    w = 0.25
    axA.bar(x - w, dense, width=w, color=CAT["red"], label="Dense FedAvg", zorder=3)
    axA.bar(x, matched, width=w, color=CAT["yellow"], label="FedAvg + top-k + quant (NEW)", zorder=3)
    axA.bar(x + w, fedtwincl, width=w, color=CAT["blue"], label="FedTwin-CL (masked + compressed)", zorder=3)
    axA.set_yscale("log")
    axA.set_xticks(x)
    axA.set_xticklabels(labels)
    axA.set_ylabel("Cumulative fleet uplink, MB (log scale)")
    axA.set_title("A. Compression-matched uplink comparison")
    axA.legend(frameon=False, loc="upper right")
    _clean(axA)

    # Panel B: capacity sweep gap-to-FedAvg
    widths = [1, 2, 4]
    for b, label, color in zip(benches, labels, [CAT["orange"], CAT["aqua"]]):
        recs = CAPACITY[b]
        gaps = []
        for wm in widths:
            fedavg = np.mean([r["current_task_metric"] for r in recs
                               if r["method"] == "fedavg" and r["width_mult"] == wm and r.get("is_summary_record")])
            ftcl = np.mean([r["current_task_metric"] for r in recs
                             if r["method"] == "fedtwin-cl" and r["width_mult"] == wm and r.get("is_summary_record")])
            gaps.append(fedavg - ftcl)
        axB.plot(widths, gaps, marker="o", color=color, lw=2, label=label, zorder=3)
    axB.axhline(0, color=INK_MUTED, lw=1, ls="--", zorder=2)
    axB.set_xticks(widths)
    axB.set_xticklabels(["1x\n(512 ch)", "2x\n(1024 ch)", "4x\n(2048 ch)"])
    axB.set_xlabel("Backbone width")
    axB.set_ylabel("Gap to FedAvg (FedAvg − FedTwin-CL current-task metric)")
    axB.set_title("B. Capacity sweep: does width close the gap?")
    axB.legend(frameon=False, loc="upper right")
    _clean(axB)

    fig.suptitle("Figure 11. Compression-matched uplink and capacity-sweep gap, real data (2026-09 follow-up)",
                  y=1.02, fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "figure11_compression_capacity.png"), facecolor=SURFACE)
    plt.close(fig)
    print("Wrote figure11_compression_capacity.png")


if __name__ == "__main__":
    fig11()
