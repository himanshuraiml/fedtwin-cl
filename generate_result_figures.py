"""
Result-figure generation script for FedTwin-CL -- Figures 5-10 in
FedTwin-CL_Manuscript.md (all currently marked [RESULT PENDING]).

This script is NOT run yet and produces NOTHING until a real pilot output
file exists. Unlike Paper_13's generate_result_figures.py, there is no
existing FedTwin-CL codebase to import a metrics module from: this paper
retargets the FedCAT engine (Paper_4) at three new benchmarks, and that
retargeting (DTTS, CTFR, the RUL/anomaly-detection heads) has not been
implemented yet. The expected input schema is therefore specified directly
in this file's docstring and loader, so that once the pilot exists, wiring
it in is a matter of matching field names, not redesigning the charts.

Expected input: a JSON file at pilot_results.json, a flat list of records,
one per (site, method, benchmark, round) observation, with fields:
  benchmark          str   "fed-twin-cmapss" | "fed-twin-femto" | "fed-twin-mimii"
  method             str   "fedavg" | "fedprox" | "fedavg-ewc" |
                            "fedcat-external" | "fedtwin-cl-no-ctfr" |
                            "fedtwin-cl" | "centralized"
  site_id            int
  device_class       str   "remote-asset" | "sensor-edge" | "line-gateway" | "plant-fog"
  round              int
  fps                float   Fleet Performance Score, manuscript Eq. 7
  fbwt               float   backward transfer in FPS units, or null
  uplink_bytes       int     cumulative bytes uploaded by this site through this round
  tx_energy_mj       float   this round's transmission energy for this site
  lambda_ph          float   DTTS threshold active for this record (Section 6.5 sweep)
  commit_timing_error float  cycles between DTTS commit and reference stage boundary, or null
  ctfr_flag_raised   bool
  ctfr_lead_time     float   or null if no flag / not yet confirmed
  ctfr_flag_confirmed bool

Produces (once pilot_results.json exists), matching the manuscript's
numbering:
  fig5_fps_vs_round.png           - Section 7.1 / Table 3
  fig6_uplink_traffic.png         - Section 7.2 / Table 4
  fig7_energy_by_device_class.png - Section 7.2
  fig8_dtts_sensitivity.png       - Section 7.3 / Table 5
  fig9_early_warning_leadtime.png - Section 7.4 / Table 6
  fig10_heterogeneity_sensitivity.png - Section 7.5

Same palette and rcParams as generate_figures.py, reused unchanged so all
ten figures in the manuscript read as one visual system.
"""

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

FIGURES_DIR = os.path.join(os.path.dirname(__file__), "figures")
RESULTS_JSON = os.path.join(os.path.dirname(__file__), "pilot_results.json")

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

METHOD_ORDER = [
    "fedavg", "fedprox", "fedavg-ewc", "fedcat-external",
    "fedtwin-cl-no-ctfr", "fedtwin-cl", "centralized",
]
METHOD_COLOR = {
    "fedavg": INK_MUTED, "fedprox": INK_MUTED, "fedavg-ewc": CAT["yellow"],
    "fedcat-external": CAT["orange"], "fedtwin-cl-no-ctfr": CAT["aqua"],
    "fedtwin-cl": CAT["blue"], "centralized": INK,
}
BENCHMARK_ORDER = ["fed-twin-cmapss", "fed-twin-femto", "fed-twin-mimii"]
DEVICE_ORDER = ["remote-asset", "sensor-edge", "line-gateway", "plant-fog"]


def load_records(path):
    with open(path) as f:
        return json.load(f)


def _clean_axes(ax):
    ax.set_facecolor(SURFACE)
    ax.grid(color=GRID, lw=0.9, zorder=0)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)


def fig5_fps_vs_round(records):
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2), sharey=True)
    fig.patch.set_facecolor(SURFACE)
    for ax, bench in zip(axes, BENCHMARK_ORDER):
        _clean_axes(ax)
        for method in METHOD_ORDER:
            rows = [r for r in records if r["benchmark"] == bench and r["method"] == method
                    and not r.get("aux", False)]
            if not rows:
                continue
            by_round = {}
            for r in rows:
                by_round.setdefault(r["round"], []).append(r["fps"])
            rounds = sorted(by_round)
            means = [np.mean(by_round[rd]) for rd in rounds]
            ax.plot(rounds, means, color=METHOD_COLOR[method], lw=1.9, label=method, zorder=3)
        ax.set_title(bench, fontsize=9.5)
        ax.set_xlabel("Communication round")
    axes[0].set_ylabel("Fleet Performance Score")
    axes[0].set_ylim(0, 1)
    axes[-1].legend(frameon=False, fontsize=7.6, loc="lower right")
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "fig5_fps_vs_round.png"))
    plt.close(fig)


def fig6_uplink_traffic(records, benchmark="fed-twin-femto"):
    totals = {}
    for method in METHOD_ORDER:
        rows = [r for r in records if r["benchmark"] == benchmark and r["method"] == method
                and not r.get("aux", False)]
        if not rows:
            continue
        last_round = max(r["round"] for r in rows)
        totals[method] = sum(r["uplink_bytes"] for r in rows if r["round"] == last_round) / 1e6

    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    fig.patch.set_facecolor(SURFACE)
    _clean_axes(ax)
    methods = [m for m in METHOD_ORDER if m in totals]
    vals = [totals[m] for m in methods]
    ax.bar(methods, vals, color=[METHOD_COLOR[m] for m in methods], zorder=3)
    ax.set_ylabel("Cumulative fleet uplink (MB)")
    ax.tick_params(axis="x", rotation=25)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "fig6_uplink_traffic.png"))
    plt.close(fig)


def fig7_energy_by_device_class(records, methods=("fedavg", "fedtwin-cl"), benchmark="fed-twin-femto"):
    fig, ax = plt.subplots(figsize=(8.2, 4.6))
    fig.patch.set_facecolor(SURFACE)
    _clean_axes(ax)
    x = np.arange(len(DEVICE_ORDER))
    width = 0.35
    for i, method in enumerate(methods):
        vals = []
        for dc in DEVICE_ORDER:
            rows = [r["tx_energy_mj"] for r in records
                    if r["method"] == method and r["device_class"] == dc
                    and r["benchmark"] == benchmark and r["tx_energy_mj"] > 0]
            vals.append(np.mean(rows) if rows else 0.0)
        ax.bar(x + (i - 0.5) * width, vals, width=width, label=method,
               color=METHOD_COLOR.get(method, CAT["blue"]), zorder=3)
    ax.set_xticks(x)
    ax.set_xticklabels(DEVICE_ORDER, rotation=15)
    ax.set_ylabel("Mean per-round transmission energy (mJ)")
    ax.set_yscale("log")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "fig7_energy_by_device_class.png"))
    plt.close(fig)


def fig8_dtts_sensitivity(records, benchmark="fed-twin-femto", method="fedtwin-cl"):
    rows = [r for r in records if r["benchmark"] == benchmark and r["method"] == method
            and r.get("commit_timing_error") is not None]
    by_lambda = {}
    for r in rows:
        by_lambda.setdefault(r["lambda_ph"], {"fps": [], "err": []})
        by_lambda[r["lambda_ph"]]["fps"].append(r["fps"])
        by_lambda[r["lambda_ph"]]["err"].append(r["commit_timing_error"])
    lambdas = sorted(by_lambda)

    fig, ax1 = plt.subplots(figsize=(7.5, 4.8))
    fig.patch.set_facecolor(SURFACE)
    _clean_axes(ax1)
    fps_means = [np.mean(by_lambda[l]["fps"]) for l in lambdas]
    ax1.plot(lambdas, fps_means, color=CAT["blue"], lw=2.0, marker="o", zorder=3, label="Fleet FPS")
    ax1.set_xlabel(r"$\lambda_{PH}$")
    ax1.set_ylabel("Fleet FPS", color=CAT["blue"])
    ax1.set_ylim(0, 1)

    ax2 = ax1.twinx()
    err_means = [np.mean(by_lambda[l]["err"]) for l in lambdas]
    ax2.plot(lambdas, err_means, color=STATUS["critical"], lw=2.0, marker="s", zorder=3,
             label="Commit-point timing error")
    ax2.set_ylabel("Timing error (cycles)", color=STATUS["critical"])
    ax2.grid(False)

    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "fig8_dtts_sensitivity.png"))
    plt.close(fig)


def fig9_early_warning_leadtime(records):
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 4.2), sharey=True)
    fig.patch.set_facecolor(SURFACE)
    for ax, bench in zip(axes, BENCHMARK_ORDER):
        _clean_axes(ax)
        lead_times = [r["ctfr_lead_time"] for r in records
                      if r["benchmark"] == bench and r.get("ctfr_flag_confirmed")
                      and r.get("ctfr_lead_time") is not None]
        if lead_times:
            ax.hist(lead_times, bins=15, color=STATUS["good"], edgecolor=SURFACE, zorder=3)
        ax.axvline(0, color=INK_MUTED, lw=1.2, ls=(0, (3, 2)))
        ax.set_title(bench, fontsize=9.3)
        ax.set_xlabel("Early-warning lead time")
    axes[0].set_ylabel("Confirmed flags")
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "fig9_early_warning_leadtime.png"))
    plt.close(fig)


def fig10_heterogeneity_sensitivity(records, benchmark="fed-twin-femto", method="fedtwin-cl"):
    fig, ax = plt.subplots(figsize=(7.8, 4.8))
    fig.patch.set_facecolor(SURFACE)
    _clean_axes(ax)
    gammas = sorted({r.get("gamma") for r in records if r.get("gamma") is not None})
    for dc, color in zip(DEVICE_ORDER, [CAT["blue"], CAT["green"], CAT["aqua"], CAT["violet"]]):
        vals = []
        for g in gammas:
            rows = [r["fps"] for r in records
                    if r["benchmark"] == benchmark and r["method"] == method
                    and r["device_class"] == dc and r.get("gamma") == g
                    and r.get("aux", False)]
            vals.append(np.mean(rows) if rows else np.nan)
        ax.plot(gammas, vals, color=color, lw=2.0, marker="o", label=dc, zorder=3)
    ax.set_xlabel(r"Heterogeneity scaling $\gamma$ (Eq. 2)")
    ax.set_ylabel("FPS by device class")
    ax.set_ylim(0, 1)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "fig10_heterogeneity_sensitivity.png"))
    plt.close(fig)


if __name__ == "__main__":
    if not os.path.exists(RESULTS_JSON):
        print(
            f"No pilot results found at {RESULTS_JSON}.\n"
            "This script is a stub: it does nothing until the FedTwin-CL pilot "
            "(DTTS + CTFR implemented on top of the FedCAT engine, run across "
            "the three benchmarks of Section 6) exports its output to that "
            "path. See the module docstring for the expected record schema."
        )
        raise SystemExit(0)

    records = load_records(RESULTS_JSON)
    fig5_fps_vs_round(records)
    fig6_uplink_traffic(records)
    fig7_energy_by_device_class(records)
    fig8_dtts_sensitivity(records)
    fig9_early_warning_leadtime(records)
    fig10_heterogeneity_sensitivity(records)
    print(f"Wrote result figures to {FIGURES_DIR}")
