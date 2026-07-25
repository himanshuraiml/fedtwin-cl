"""
Generates Figures 5-10 for FedTwin-CL from the SPIKE PILOT's real, executed
output (code/pilot_results_spike.json, code/spike_extended_results.json).

These are real numbers from real code -- NOT fabricated -- but the code ran
on a synthetic generator standing in for the three real benchmark datasets
(NASA C-MAPSS, FEMTO-ST PRONOSTIA, MIMII), which have not been acquired
yet. See EXPERIMENT_PROCEDURE.md Section 1 for the full distinction. Do
not read these figures as the real dataset-backed pilot.

Same palette and rcParams as generate_figures.py (figures 1-4), reused
unchanged so all ten figures in the manuscript read as one visual system.
"""

import json
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

FIGURES_DIR = os.path.join(os.path.dirname(__file__), "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)

with open(os.path.join(os.path.dirname(__file__), "code", "pilot_results_spike.json")) as f:
    RECORDS = json.load(f)
with open(os.path.join(os.path.dirname(__file__), "code", "spike_extended_results.json")) as f:
    EXTENDED = json.load(f)

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
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
CAT = {
    "blue": "#2a78d6", "green": "#008300", "magenta": "#e87ba4", "yellow": "#eda100",
    "aqua": "#1baf7a", "orange": "#eb6834", "violet": "#4a3aa7", "red": "#e34948",
}
STATUS = {"good": "#0ca30c", "warning": "#fab219", "serious": "#ec835a", "critical": "#d03b3b"}

METHOD_ORDER = ["fedavg", "fedavg-ewc", "fedcat-external", "fedtwin-cl-no-ctfr", "fedtwin-cl"]
METHOD_COLOR = {
    "fedavg": INK_MUTED, "fedavg-ewc": CAT["yellow"], "fedcat-external": CAT["orange"],
    "fedtwin-cl-no-ctfr": CAT["aqua"], "fedtwin-cl": CAT["blue"], "centralized": INK,
}
BENCHMARK_ORDER = ["fed-twin-cmapss", "fed-twin-femto", "fed-twin-mimii"]
DEVICE_ORDER = ["remote-asset", "sensor-edge", "line-gateway", "plant-fog"]


def _clean(ax):
    ax.set_facecolor(SURFACE)
    ax.grid(color=GRID, lw=0.9, zorder=0)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def fig5_fps_vs_round():
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2), sharey=False)
    fig.patch.set_facecolor(SURFACE)
    for ax, bench in zip(axes, BENCHMARK_ORDER):
        _clean(ax)
        for m in METHOD_ORDER:
            rows = [r for r in RECORDS if r.get("benchmark") == bench and r.get("method") == m
                    and "raw_metric" in r]
            by_round = {}
            for r in rows:
                by_round.setdefault(r["round"], []).append(r["raw_metric"])
            rounds = sorted(by_round)
            means = [np.mean(by_round[rd]) for rd in rounds]
            ax.plot(rounds, means, color=METHOD_COLOR[m], lw=1.9, label=m, zorder=3)
        ax.set_title(bench, fontsize=9.5)
        ax.set_xlabel("Communication round")
    axes[0].set_ylabel("Current-task metric (higher is better)")
    axes[-1].legend(frameon=False, fontsize=7.4, loc="lower right")
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "fig5_fps_vs_round.png"))
    plt.close(fig)


def fig6_uplink_traffic(bench="fed-twin-femto"):
    totals = {}
    for m in METHOD_ORDER:
        rows = [r for r in RECORDS if r.get("benchmark") == bench and r.get("method") == m
                and r.get("round") == 20 and "uplink_bytes" in r]
        totals[m] = sum(r["uplink_bytes"] for r in rows) / 1e6

    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    fig.patch.set_facecolor(SURFACE)
    _clean(ax)
    methods = [m for m in METHOD_ORDER if m in totals]
    vals = [totals[m] for m in methods]
    ax.bar(methods, vals, color=[METHOD_COLOR[m] for m in methods], zorder=3)
    ax.set_ylabel("Cumulative fleet uplink (MB)")
    ax.set_yscale("log")
    ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "fig6_uplink_traffic.png"))
    plt.close(fig)


def fig7_energy_by_device_class(bench="fed-twin-femto", methods=("fedavg", "fedtwin-cl")):
    fig, ax = plt.subplots(figsize=(8.2, 4.6))
    fig.patch.set_facecolor(SURFACE)
    _clean(ax)
    x = np.arange(len(DEVICE_ORDER))
    width = 0.35
    for i, m in enumerate(methods):
        vals = []
        for dc in DEVICE_ORDER:
            rows = [r["tx_energy_mj"] for r in RECORDS
                    if r.get("benchmark") == bench and r.get("method") == m and r.get("device_class") == dc
                    and r.get("tx_energy_mj", 0) > 0]
            vals.append(np.mean(rows) if rows else 0.0)
        ax.bar(x + (i - 0.5) * width, vals, width=width, label=m, color=METHOD_COLOR[m], zorder=3)
    ax.set_xticks(x)
    ax.set_xticklabels(DEVICE_ORDER, rotation=15)
    ax.set_ylabel("Mean per-round transmission energy (mJ)")
    ax.set_yscale("log")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "fig7_energy_by_device_class.png"))
    plt.close(fig)


def fig8_dtts_sensitivity():
    sweep = EXTENDED["dtts_sweep"]
    lambdas = sorted(float(k) for k in sweep)
    fps = [sweep[str(l) if str(l) in sweep else l]["fleet_fps"] for l in lambdas]
    timing = [sweep[str(l) if str(l) in sweep else l]["mean_abs_timing_error"] for l in lambdas]
    commits = [sweep[str(l) if str(l) in sweep else l]["mean_tasks_committed"] for l in lambdas]

    fig, ax1 = plt.subplots(figsize=(7.8, 5.0))
    fig.patch.set_facecolor(SURFACE)
    _clean(ax1)
    ax1.plot(lambdas, fps, color=CAT["blue"], lw=2.0, marker="o", zorder=3)
    ax1.set_xlabel(r"$\lambda_{PH}$")
    ax1.set_ylabel("Fleet FPS", color=CAT["blue"])

    ax2 = ax1.twinx()
    ax2.plot(lambdas, timing, color=STATUS["critical"], lw=2.0, marker="s", zorder=3)
    ax2.set_ylabel("Mean |commit-point timing error| (rounds)", color=STATUS["critical"])
    ax2.grid(False)

    for l, c in zip(lambdas, commits):
        ax1.annotate(f"{c:.2f} tasks/site", (l, sweep[str(l) if str(l) in sweep else l]["fleet_fps"]),
                     xytext=(0, -14), textcoords="offset points", ha="center", fontsize=7, color=INK_MUTED)

    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "fig8_dtts_sensitivity.png"))
    plt.close(fig)


def fig9_ctfr_summary():
    leadtime = EXTENDED["ctfr_leadtime"]
    fig, ax = plt.subplots(figsize=(8.0, 4.6))
    fig.patch.set_facecolor(SURFACE)
    _clean(ax)
    x = np.arange(len(BENCHMARK_ORDER))
    flags = [leadtime[b]["n_flags_raised"] for b in BENCHMARK_ORDER]
    confirmed = [leadtime[b]["n_confirmed"] for b in BENCHMARK_ORDER]
    width = 0.35
    ax.bar(x - width / 2, flags, width=width, label="Flags raised", color=CAT["magenta"], zorder=3)
    ax.bar(x + width / 2, confirmed, width=width, label="Confirmed (matching later commit)",
           color=STATUS["good"], zorder=3)
    ax.set_xticks(x)
    ax.set_xticklabels(BENCHMARK_ORDER)
    ax.set_ylabel("CTFR flag count")
    for xi, f, c in zip(x, flags, confirmed):
        ax.text(xi + width / 2, c + max(flags) * 0.02, str(c), ha="center", fontsize=8, color=STATUS["good"])
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "fig9_early_warning_leadtime.png"))
    plt.close(fig)


def fig10_heterogeneity_sensitivity():
    # Real-data finding: at this configuration (S0=0.45, [smin,smax]=[0.25,0.70],
    # CMAX ratio spread 0.4-1.6x mean), sparsity for the narrowest and widest
    # device classes is already pinned to the smin/smax clip bound regardless
    # of gamma for most of the run (rho_global stays low until late), so
    # gamma shows negligible effect in this configuration -- see
    # SPIKE_RESULTS.md Section on Fig 10 for the full diagnosis. Plotted
    # here honestly (near-flat lines), not smoothed into a nicer story.
    gammas = [0.0, 0.2, 0.4, 0.6, 0.8]
    fps_by_dc = {
        "remote-asset": [-1.3128, -1.3128, -1.3123, -1.3123, -1.3123],
        "sensor-edge": [-1.7212, -1.7212, -1.7347, -1.7347, -1.7347],
        "line-gateway": [-1.4571, -1.4571, -1.4566, -1.4566, -1.4566],
        "plant-fog": [-1.8233, -1.8233, -1.8225, -1.8225, -1.8225],
    }
    fig, ax = plt.subplots(figsize=(7.8, 4.8))
    fig.patch.set_facecolor(SURFACE)
    _clean(ax)
    colors = [CAT["blue"], CAT["green"], CAT["aqua"], CAT["violet"]]
    for dc, color in zip(DEVICE_ORDER, colors):
        ax.plot(gammas, fps_by_dc[dc], color=color, lw=2.0, marker="o", label=dc, zorder=3)
    ax.set_xlabel(r"Heterogeneity scaling $\gamma$ (Eq. 2)")
    ax.set_ylabel("FPS by device class")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "fig10_heterogeneity_sensitivity.png"))
    plt.close(fig)


if __name__ == "__main__":
    fig5_fps_vs_round()
    fig6_uplink_traffic()
    fig7_energy_by_device_class()
    fig8_dtts_sensitivity()
    fig9_ctfr_summary()
    fig10_heterogeneity_sensitivity()
    print(f"Wrote figures 5-10 to {FIGURES_DIR}")
