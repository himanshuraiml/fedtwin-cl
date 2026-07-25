"""
Regenerates Figures 8-10 for FedTwin-CL from the REAL sweep results
(real_data/dtts_sweep_real_results.json, real_data/ctfr_leadtime_real_results.json,
real_data/gamma_sweep_real_results.json), replacing the earlier synthetic-spike
versions now that all three parameter sweeps have been re-run on real data.

Per-figure real-vs-synthetic comparison policy (decided from the actual
numbers, not assumed in advance):
- Figure 8 (DTTS sweep): REAL vs SYNTHETIC side-by-side panels. The real
  sweep shows a genuinely different pattern (near-flat commit count and
  noisy, non-monotonic timing error vs. the synthetic sweep's clean
  monotonic tradeoff), and the two sweeps operate over non-overlapping
  lambda_PH ranges (real: 0.05-0.70, synthetic: 0.3-1.5), so an overlay on
  one shared axis would misrepresent rather than clarify the comparison.
- Figure 9 (CTFR): REAL ONLY. Real confirmed precision (0/287, 5/128,
  0/71) matches the synthetic finding closely (both near-zero across all
  three benchmarks) -- no meaningful deviation, so only the real numbers
  are plotted, exactly replacing the synthetic version.
- Figure 10 (heterogeneity/gamma): REAL ONLY, at a deliberately
  capacity-constrained registry width (see real_gamma_sweep.py docstring).
  The original synthetic Fig 10 was measured at a different, uncorrected
  width regime that Section 7.7's own diagnosis identified as the reason
  no gamma-sensitivity was visible; plotting it beside this corrected-width
  real result on the same axes would conflate "real vs. synthetic" with
  "wrong width vs. right width" as confounded variables, so only the
  corrected, real result is shown, with the earlier flat result kept as
  textual context rather than an axis overlay.

Same palette and rcParams as the other figure-generation scripts.
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

with open(os.path.join(REAL_DIR, "dtts_sweep_real_results.json")) as f:
    DTTS_REAL = json.load(f)
with open(os.path.join(REAL_DIR, "ctfr_leadtime_real_results.json")) as f:
    CTFR_REAL = json.load(f)
with open(os.path.join(REAL_DIR, "gamma_sweep_real_results.json")) as f:
    GAMMA_REAL = json.load(f)

# Synthetic spike numbers (code/spike_extended_results.json), reused only for Figure 8's comparison panel.
DTTS_SYNTHETIC = {
    0.3: {"mean_tasks_committed": 3.75, "mean_abs_timing_error": 2.21, "fleet_fps": -0.631, "fleet_fbwt": -0.057},
    0.5: {"mean_tasks_committed": 3.54, "mean_abs_timing_error": 1.87, "fleet_fps": -0.669, "fleet_fbwt": -0.122},
    0.7: {"mean_tasks_committed": 3.25, "mean_abs_timing_error": 1.48, "fleet_fps": -0.729, "fleet_fbwt": -0.254},
    1.0: {"mean_tasks_committed": 2.50, "mean_abs_timing_error": 1.24, "fleet_fps": -0.923, "fleet_fbwt": -0.459},
    1.5: {"mean_tasks_committed": 0.88, "mean_abs_timing_error": 1.19, "fleet_fps": -1.471, "fleet_fbwt": -1.162},
}

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
STATUS = {"good": "#0ca30c", "warning": "#fab219", "serious": "#ec835a", "critical": "#d03b3b"}
DEVICE_ORDER = ["remote-asset", "sensor-edge", "line-gateway", "plant-fog"]
BENCHMARK_ORDER = ["fed-twin-cmapss-real", "fed-twin-femto-real", "fed-twin-mimii-real"]
BENCHMARK_LABEL = {
    "fed-twin-cmapss-real": "Fed-Twin-CMAPSS", "fed-twin-femto-real": "Fed-Twin-FEMTO",
    "fed-twin-mimii-real": "Fed-Twin-MIMII",
}


def _clean(ax):
    ax.set_facecolor(SURFACE)
    ax.grid(color=GRID, lw=0.9, zorder=0)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def fig8_dtts_sensitivity():
    fig, (ax_r1, ax_s1) = plt.subplots(1, 2, figsize=(12.5, 5.0))
    fig.patch.set_facecolor(SURFACE)

    # --- Real panel ---
    _clean(ax_r1)
    lambdas_r = sorted(float(k) for k in DTTS_REAL)
    fps_r = [DTTS_REAL[str(l) if str(l) in DTTS_REAL else l]["fleet_fps"] for l in lambdas_r]
    timing_r = [DTTS_REAL[str(l) if str(l) in DTTS_REAL else l]["mean_abs_timing_error"] for l in lambdas_r]
    commits_r = [DTTS_REAL[str(l) if str(l) in DTTS_REAL else l]["mean_tasks_committed"] for l in lambdas_r]
    ax_r1.plot(lambdas_r, fps_r, color=CAT["blue"], lw=2.0, marker="o", zorder=3)
    ax_r1.set_xlabel(r"$\lambda_{PH}$ (real, calibrated range)")
    ax_r1.set_ylabel("Fleet metric (real)", color=CAT["blue"])
    ax_r2 = ax_r1.twinx()
    ax_r2.plot(lambdas_r, timing_r, color=STATUS["critical"], lw=2.0, marker="s", zorder=3)
    ax_r2.set_ylabel("Mean |timing error| (rounds)", color=STATUS["critical"])
    ax_r2.grid(False)
    for l, c, y in zip(lambdas_r, commits_r, fps_r):
        ax_r1.annotate(f"{c:.2f}", (l, y), xytext=(0, -13), textcoords="offset points",
                        ha="center", fontsize=6.8, color=INK_MUTED)
    ax_r1.set_title("Real Fed-Twin-FEMTO", fontsize=9.5)

    # --- Synthetic panel ---
    _clean(ax_s1)
    lambdas_s = sorted(DTTS_SYNTHETIC)
    fps_s = [DTTS_SYNTHETIC[l]["fleet_fps"] for l in lambdas_s]
    timing_s = [DTTS_SYNTHETIC[l]["mean_abs_timing_error"] for l in lambdas_s]
    commits_s = [DTTS_SYNTHETIC[l]["mean_tasks_committed"] for l in lambdas_s]
    ax_s1.plot(lambdas_s, fps_s, color=CAT["blue"], lw=2.0, marker="o", zorder=3)
    ax_s1.set_xlabel(r"$\lambda_{PH}$ (synthetic, calibrated range)")
    ax_s2 = ax_s1.twinx()
    ax_s2.plot(lambdas_s, timing_s, color=STATUS["critical"], lw=2.0, marker="s", zorder=3)
    ax_s2.set_ylabel("Mean |timing error| (rounds)", color=STATUS["critical"])
    ax_s2.grid(False)
    for l, c, y in zip(lambdas_s, commits_s, fps_s):
        ax_s1.annotate(f"{c:.2f}", (l, y), xytext=(0, -13), textcoords="offset points",
                        ha="center", fontsize=6.8, color=INK_MUTED)
    ax_s1.set_title("Synthetic Fed-Twin-FEMTO (supplementary)", fontsize=9.5)

    fig.suptitle("DTTS segmentation quality vs. threshold: real data does not reproduce the synthetic sweep's clean tradeoff",
                 fontsize=9, y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "fig8_dtts_sensitivity.png"))
    plt.close(fig)


def fig9_ctfr_summary():
    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    fig.patch.set_facecolor(SURFACE)
    _clean(ax)
    x = np.arange(len(BENCHMARK_ORDER))
    flags = [CTFR_REAL[b]["n_flags_raised"] for b in BENCHMARK_ORDER]
    confirmed = [CTFR_REAL[b]["n_confirmed"] for b in BENCHMARK_ORDER]
    width = 0.35
    ax.bar(x - width / 2, flags, width=width, label="Flags raised", color=CAT["magenta"], zorder=3)
    ax.bar(x + width / 2, confirmed, width=width, label="Confirmed (matching later own commit)",
           color=STATUS["good"], zorder=3)
    ax.set_xticks(x)
    ax.set_xticklabels([BENCHMARK_LABEL[b] for b in BENCHMARK_ORDER])
    ax.set_ylabel("CTFR flag count (real data)")
    for xi, f, c in zip(x, flags, confirmed):
        ax.text(xi + width / 2, c + max(flags) * 0.02, str(c), ha="center", fontsize=8, color=STATUS["good"])
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "fig9_early_warning_leadtime.png"))
    plt.close(fig)


def fig10_heterogeneity_sensitivity():
    fps = GAMMA_REAL["fps_by_gamma_device"]
    gammas = sorted(float(g) for g in fps)
    fig, ax = plt.subplots(figsize=(7.8, 4.8))
    fig.patch.set_facecolor(SURFACE)
    _clean(ax)
    colors = [CAT["blue"], CAT["green"], CAT["aqua"], CAT["violet"]]
    for dc, color in zip(DEVICE_ORDER, colors):
        vals = [fps[str(g) if str(g) in fps else g][dc] for g in gammas]
        ax.plot(gammas, vals, color=color, lw=2.0, marker="o", label=dc, zorder=3)
    ax.set_xlabel(r"Heterogeneity scaling $\gamma$ (Eq. 2), capacity-constrained registry width")
    ax.set_ylabel("FPS by device class (real Fed-Twin-FEMTO)")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "fig10_heterogeneity_sensitivity.png"))
    plt.close(fig)


if __name__ == "__main__":
    fig8_dtts_sensitivity()
    fig9_ctfr_summary()
    fig10_heterogeneity_sensitivity()
    print(f"Wrote real-data Figures 8-10 to {FIGURES_DIR}")
