"""
Regenerates Figures 5-7 for FedTwin-CL from the REAL dataset pilot output
(real_data/cmapss_real_results.json, real_data/femto_real_results.json,
real_data/mimii_real_results.json) -- real NASA C-MAPSS turbofan telemetry,
real IEEE PHM 2012 / FEMTO-ST PRONOSTIA bearing vibration, and real MIMII
(valve, 6dB+0dB) acoustic clips, run through the TCN-Nano-lite PyTorch
pipelines in real_data/real_pipeline_*.py. This supersedes the earlier
synthetic-spike versions of Figures 5-7 (generate_spike_result_figures.py),
which are now kept only as the source for Figures 8-10 (DTTS sensitivity,
CTFR lead-time, heterogeneity sweep -- parameter sweeps not yet re-run on
real data, see manuscript Section 7.5-7.7).

Same palette and rcParams as generate_figures.py / generate_spike_result_figures.py,
reused unchanged so all ten figures in the manuscript read as one visual system.
"""

import json
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

FIGURES_DIR = os.path.join(os.path.dirname(__file__), "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)
REAL_DIR = os.path.join(os.path.dirname(__file__), "real_data")

RECORDS = []
for fname in ("cmapss_real_results.json", "femto_real_results.json", "mimii_real_results.json"):
    with open(os.path.join(REAL_DIR, fname)) as f:
        RECORDS += json.load(f)

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

METHOD_ORDER = ["fedavg", "fedavg-ewc", "fedcat-external", "fedtwin-cl-no-ctfr", "fedtwin-cl"]
METHOD_LABEL = {
    "fedavg": "FedAvg", "fedavg-ewc": "FedAvg+EWC", "fedcat-external": "FedCAT (oracle)",
    "fedtwin-cl-no-ctfr": "FedTwin-CL w/o CTFR", "fedtwin-cl": "FedTwin-CL (Ours)",
}
METHOD_COLOR = {
    "fedavg": INK_MUTED, "fedavg-ewc": CAT["yellow"], "fedcat-external": CAT["orange"],
    "fedtwin-cl-no-ctfr": CAT["aqua"], "fedtwin-cl": CAT["blue"],
}
BENCHMARK_ORDER = ["fed-twin-cmapss-real", "fed-twin-femto-real", "fed-twin-mimii-real"]
BENCHMARK_LABEL = {
    "fed-twin-cmapss-real": "Fed-Twin-CMAPSS (real)",
    "fed-twin-femto-real": "Fed-Twin-FEMTO (real)",
    "fed-twin-mimii-real": "Fed-Twin-MIMII (real)",
}
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
            ax.plot(rounds, means, color=METHOD_COLOR[m], lw=1.9, label=METHOD_LABEL[m], zorder=3)
        ax.set_title(BENCHMARK_LABEL[bench], fontsize=9.5)
        ax.set_xlabel("Communication round")
    axes[0].set_ylabel("Current-task metric (higher is better)")
    axes[-1].legend(frameon=False, fontsize=7.4, loc="lower right")
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "fig5_fps_vs_round.png"))
    plt.close(fig)


def fig6_uplink_traffic():
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.4), sharey=False)
    fig.patch.set_facecolor(SURFACE)
    for ax, bench in zip(axes, BENCHMARK_ORDER):
        _clean(ax)
        totals = {}
        for m in METHOD_ORDER:
            rows = [r for r in RECORDS if r.get("benchmark") == bench and r.get("method") == m
                    and "uplink_bytes" in r]
            if not rows:
                continue
            last_round = max(r["round"] for r in rows)
            totals[m] = sum(r["uplink_bytes"] for r in rows if r["round"] == last_round) / 1e6
        methods = [m for m in METHOD_ORDER if m in totals]
        vals = [totals[m] for m in methods]
        ax.bar([METHOD_LABEL[m] for m in methods], vals, color=[METHOD_COLOR[m] for m in methods], zorder=3)
        ax.set_yscale("log")
        ax.set_title(BENCHMARK_LABEL[bench], fontsize=9.5)
        ax.tick_params(axis="x", rotation=25)
    axes[0].set_ylabel("Cumulative fleet uplink (MB)")
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "fig6_uplink_traffic.png"))
    plt.close(fig)


def fig7_energy_by_device_class(methods=("fedavg", "fedtwin-cl")):
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.4), sharey=False)
    fig.patch.set_facecolor(SURFACE)
    x = np.arange(len(DEVICE_ORDER))
    width = 0.35
    for ax, bench in zip(axes, BENCHMARK_ORDER):
        _clean(ax)
        for i, m in enumerate(methods):
            vals = []
            for dc in DEVICE_ORDER:
                rows = [r["tx_energy_mj"] for r in RECORDS
                        if r.get("benchmark") == bench and r.get("method") == m and r.get("device_class") == dc
                        and r.get("tx_energy_mj", 0) > 0]
                vals.append(np.mean(rows) if rows else 0.0)
            ax.bar(x + (i - 0.5) * width, vals, width=width, label=METHOD_LABEL[m], color=METHOD_COLOR[m], zorder=3)
        ax.set_xticks(x)
        ax.set_xticklabels(DEVICE_ORDER, rotation=20)
        ax.set_yscale("log")
        ax.set_title(BENCHMARK_LABEL[bench], fontsize=9.5)
    axes[0].set_ylabel("Mean per-round transmission energy (mJ)")
    axes[-1].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "fig7_energy_by_device_class.png"))
    plt.close(fig)


if __name__ == "__main__":
    fig5_fps_vs_round()
    fig6_uplink_traffic()
    fig7_energy_by_device_class()
    print(f"Wrote real-data Figures 5-7 to {FIGURES_DIR}")
