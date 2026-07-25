"""
Figure generation script for:
  FedTwin-CL: Federated Continual Learning for Industrial IoT Digital Twins
  under Progressive Concept Drift

Generates the four CONCEPTUAL / DESIGN figures the manuscript can support
before the live pilot run described in Section 6 completes -- schematic
illustrations of the fleet architecture, the two new mechanisms (DTTS,
CTFR), and the three-benchmark drift-injection design. Figures 5-10 in the
manuscript (RESULT PENDING) are NOT produced by this script -- they require
real pilot output and belong in generate_result_figures.py once that data
exists.

Palette: the dataviz-skill reference default (validated categorical order,
CVD-safe; see references/palette.md), reused unchanged from Paper_4 and
Paper_13 so all figures across this portfolio read as one visual system.

Produces 4 PNG files in figures/:
  fig1_system_architecture.png   - Fleet of digital twins + server registries
  fig2_dtts_mechanism.png        - Page-Hinkley drift detection and task commit
  fig3_ctfr_mechanism.png        - Cross-twin signature broadcast and early warning
  fig4_benchmark_construction.png- C-MAPSS / FEMTO / MIMII drift-injection design

No captions or titles are baked into the images; captions live in the
manuscript, immediately after each [INSERT FIGURE N] placeholder.
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle
from matplotlib.lines import Line2D

FIGURES_DIR = os.path.join(os.path.dirname(__file__), "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)

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

# ── dataviz-skill reference palette (validated, reused unchanged) ──────────
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"

CAT = {
    "blue": "#2a78d6",
    "green": "#008300",
    "magenta": "#e87ba4",
    "yellow": "#eda100",
    "aqua": "#1baf7a",
    "orange": "#eb6834",
    "violet": "#4a3aa7",
    "red": "#e34948",
}
SEQ_BLUE = {  # ordinal-safe steps (>=250)
    250: "#86b6ef", 350: "#5598e7", 450: "#2a78d6", 550: "#1c5cab", 650: "#104281",
}
STATUS = {"good": "#0ca30c", "warning": "#fab219", "serious": "#ec835a", "critical": "#d03b3b"}


def new_ax(figsize, xlim, ylim):
    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.axis("off")
    return fig, ax


def box(ax, x, y, w, h, text, fc=SURFACE, ec=INK_SECONDARY, tc=INK, fs=9, lw=1.3, weight="normal"):
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.02,rounding_size=0.06",
        linewidth=lw, edgecolor=ec, facecolor=fc, zorder=3,
    ))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
             fontsize=fs, color=tc, weight=weight, zorder=4, wrap=True)


def arrow(ax, xy_from, xy_to, color=INK_SECONDARY, lw=1.6, style="-|>", connectionstyle="arc3,rad=0.0"):
    ax.add_patch(FancyArrowPatch(
        xy_from, xy_to, arrowstyle=style, mutation_scale=11,
        linewidth=lw, color=color, connectionstyle=connectionstyle, zorder=2,
    ))


# ═══════════════════════════════════════════════════════════════════════════
# Figure 1 — Fleet system architecture: sites, twins, server registries
# ═══════════════════════════════════════════════════════════════════════════
def fig1():
    fig, ax = new_ax((13.0, 10.2), (-0.6, 17.2), (-1.7, 13.4))

    site_specs = [
        ("Remote Asset Node", CAT["blue"], 9.6),
        ("Sensor-Edge Node", CAT["green"], 7.1),
        ("Line Gateway", CAT["aqua"], 4.6),
        ("Plant Fog Node", CAT["violet"], 2.1),
    ]
    site_x, site_w, site_h = 0.3, 3.4, 1.7

    for label, color, y in site_specs:
        box(ax, site_x, y, site_w, site_h, f"{label}\n(digital twin)",
            fc=color, tc=SURFACE, ec="none", fs=8.6, weight="bold")
        box(ax, site_x + site_w + 0.5, y, 3.4, site_h,
            "DTTS\n(Page-Hinkley on\nhealth indicator)",
            fc=SURFACE, ec=color, tc=INK, fs=8.0, lw=1.6)
        arrow(ax, (site_x + site_w, y + site_h / 2), (site_x + site_w + 0.5, y + site_h / 2), lw=1.3)
        arrow(ax, (site_x + site_w + 0.5, y + site_h * 0.3), (site_x + site_w, y + site_h * 0.3),
              color=INK_MUTED, lw=1.0, connectionstyle="arc3,rad=0.3")

    dtts_right = site_x + site_w + 0.5 + 3.4
    server_x, server_y, server_w, server_h = 11.6, 1.6, 4.8, 8.6
    box(ax, server_x, server_y, server_w, server_h,
        "SERVER\n\nGlobal Mask Registry\n$\\mathcal{M}^{\\mathrm{global}}$ (Eq. 4, bitwise OR)\n\nCTFR Signature Registry\n$\\{\\sigma^{(n,k)}\\}$ (Section 4.7)",
        fc=INK, tc=SURFACE, ec="none", fs=8.8)

    for label, color, y in site_specs:
        arrow(ax, (dtts_right, y + site_h * 0.75), (server_x, server_y + server_h * 0.62),
              color=color, lw=1.6, connectionstyle=f"arc3,rad={0.10 if y > 5.5 else -0.10}")
        arrow(ax, (server_x, server_y + server_h * 0.38), (dtts_right, y + site_h * 0.25),
              color=INK_MUTED, lw=1.1, connectionstyle=f"arc3,rad={-0.10 if y > 5.5 else 0.10}")

    ax.text(server_x + server_w / 2, server_y + server_h + 0.3,
             "broadcast: shared backbone $\\theta$ + $\\mathcal{M}^{\\mathrm{global}}$ + signature registry",
             ha="center", va="bottom", fontsize=7.8, color=INK_SECONDARY, style="italic")
    ax.text(server_x + server_w / 2, server_y - 0.3,
             "upload: sparse quantized delta + task mask (if committing) + signature (if committing)",
             ha="center", va="top", fontsize=7.8, color=INK_SECONDARY, style="italic")

    # CTFR cross-site relevance-scoring flow (Eq. 6), dashed, from server down to one
    # representative receiving site -- the comparison itself runs locally at that site,
    # not at the server; the server only broadcasts the signature registry it was drawn from.
    ctfr_y_top = 12.6
    box(ax, 2.3, ctfr_y_top - 0.55, 9.0, 0.95,
        "CTFR: a receiving site compares its own current trajectory, locally,\nagainst the broadcast signature registry (Eq. 6) and raises an\nearly-warning flag before its own DTTS statistic independently triggers",
        fc="#fdf0f4", ec=CAT["magenta"], tc=CAT["magenta"], fs=7.6, lw=1.4)
    ax.plot([server_x + server_w * 0.3, server_x + server_w * 0.3, 2.3 + 9.0 * 0.5],
            [server_y + server_h, ctfr_y_top + 0.4, ctfr_y_top + 0.4],
            color=CAT["magenta"], lw=1.2, ls=(0, (3, 2)), zorder=1)
    ax.plot([2.3 + 9.0 * 0.15, 2.3 + 9.0 * 0.15], [ctfr_y_top - 0.55, site_specs[1][2] + site_h],
            color=CAT["magenta"], lw=1.2, ls=(0, (3, 2)), zorder=1)
    arrow(ax, (2.3 + 9.0 * 0.15, site_specs[1][2] + site_h + 0.5), (site_x + site_w + 0.5, site_specs[1][2] + site_h),
          color=CAT["magenta"], lw=1.2, style="-|>", connectionstyle="arc3,rad=-0.15")

    # Legend
    ax.add_patch(FancyBboxPatch((0.1, -1.55), 16.9, 1.55, boxstyle="round,pad=0.02,rounding_size=0.05",
                                 linewidth=1.0, edgecolor=GRID, facecolor="#ffffff", zorder=2))
    leg_y1 = 0.35
    leg_items = [(color, f"Data + mask upload ({label})") for label, color, _ in site_specs]
    lx = 0.5
    for color, text in leg_items:
        ax.plot([lx, lx + 0.5], [leg_y1, leg_y1], color=color, lw=2.2, zorder=3)
        ax.text(lx + 0.65, leg_y1, text, ha="left", va="center", fontsize=7.2, color=INK_SECONDARY, zorder=3)
        lx += 4.1

    leg_y2 = -0.35
    ax.plot([0.5, 1.0], [leg_y2, leg_y2], color=INK_MUTED, lw=2.2, zorder=3)
    ax.text(1.15, leg_y2, "Broadcast / download", ha="left", va="center", fontsize=7.2, color=INK_SECONDARY, zorder=3)
    ax.plot([5.5, 6.0], [leg_y2, leg_y2], color=CAT["magenta"], lw=1.6, ls=(0, (3, 2)), zorder=3)
    ax.text(6.15, leg_y2, "CTFR relevance-scoring comparison (runs locally at receiving site)",
             ha="left", va="center", fontsize=7.2, color=INK_SECONDARY, zorder=3)

    ax.text(0.4, -1.15,
             "Raw sensor windows never leave a site; only sparse quantized deltas, task masks, and bounded-dimension signatures are uploaded to the server.",
             ha="left", va="center", fontsize=7.7, color=INK_MUTED, style="italic")

    fig.savefig(os.path.join(FIGURES_DIR, "fig1_system_architecture.png"))
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════
# Figure 2 — DTTS: Page-Hinkley drift detection and task-boundary commit
# ═══════════════════════════════════════════════════════════════════════════
def fig2():
    rng = np.random.default_rng(7)
    t = np.linspace(0, 100, 400)

    def stage_curve(t, breaks, levels, noise_scale=0.035):
        y = np.zeros_like(t)
        for i in range(len(breaks) - 1):
            mask = (t >= breaks[i]) & (t < breaks[i + 1])
            local = np.linspace(0, 1, mask.sum())
            y[mask] = levels[i] + (levels[i + 1] - levels[i]) * local ** 1.6
        y += rng.normal(0, noise_scale, size=t.shape)
        return y

    breaks = [0, 38, 74, 100]
    levels = [0.08, 0.08, 0.34, 0.95]
    h = stage_curve(t, breaks, levels)

    # Page-Hinkley-style cumulative statistic (illustrative, not literally recomputed from h)
    delta = 0.01
    hbar = np.cumsum(h) / (np.arange(len(h)) + 1)
    m = np.cumsum(h - hbar - delta)
    ph = m - np.minimum.accumulate(m)
    lam = 0.9 * ph.max() * 0.62  # illustrative threshold line

    fig, axes = plt.subplots(2, 1, figsize=(10.5, 6.0), sharex=True,
                              gridspec_kw={"height_ratios": [1.3, 1]})
    fig.patch.set_facecolor(SURFACE)

    ax = axes[0]
    ax.set_facecolor(SURFACE)
    ax.plot(t, h, color=CAT["blue"], lw=1.8, zorder=3)
    ax.set_ylabel("Health indicator $h_t$")
    ax.set_ylim(-0.05, 1.05)
    ax.grid(color=GRID, lw=0.9, zorder=0)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    stage_labels = ["Early wear", "Degrading", "Near failure"]
    stage_mid = [(breaks[i] + breaks[i + 1]) / 2 for i in range(3)]
    for x, lbl in zip(stage_mid, stage_labels):
        ax.text(x, 1.0, lbl, ha="center", va="bottom", fontsize=8.4, color=INK_SECONDARY)

    ax = axes[1]
    ax.set_facecolor(SURFACE)
    ax.plot(t, ph, color=CAT["orange"], lw=1.8, zorder=3)
    ax.axhline(lam, color=STATUS["critical"], lw=1.3, ls=(0, (4, 2)), zorder=2)
    ax.text(2, lam + 0.4, r"$\lambda_{PH}$", color=STATUS["critical"], fontsize=9.5, va="bottom")
    ax.set_ylabel(r"$PH_t$ (Eq. 3)")
    ax.set_xlabel("Operating time (site-local)")
    ax.grid(color=GRID, lw=0.9, zorder=0)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)

    commit_pts = []
    crossed = False
    for xi, yi in zip(t, ph):
        if yi > lam and not crossed:
            commit_pts.append(xi)
            crossed = True
        if yi < lam * 0.4:
            crossed = False

    for ax_i in axes:
        for cx in commit_pts:
            ax_i.axvline(cx, color=INK_MUTED, lw=1.0, ls=(0, (1, 2)), zorder=1)
    for cx in commit_pts:
        axes[1].plot([cx], [lam], marker="o", markersize=7, color=STATUS["critical"], zorder=4)
        axes[1].annotate("task\ncommit", (cx, lam), xytext=(cx + 2, lam + ph.max() * 0.22),
                          fontsize=7.4, color=STATUS["critical"], ha="left")

    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "fig2_dtts_mechanism.png"))
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════
# Figure 3 — CTFR: cross-twin signature broadcast and early-warning lead time
# ═══════════════════════════════════════════════════════════════════════════
def fig3():
    fig, ax = new_ax((11.2, 6.2), (-0.6, 15.6), (-0.6, 8.6))

    # Site A timeline (top): commits a fault, produces signature
    ay = 6.4
    ax.plot([0.4, 14.8], [ay, ay], color=AXIS, lw=1.4, zorder=1)
    ax.text(-0.2, ay, "Site A", ha="right", va="center", fontsize=9.5, color=INK_SECONDARY, weight="bold")
    ax.add_patch(Circle((6.0, ay), 0.22, facecolor=CAT["red"], edgecolor="none", zorder=3))
    ax.text(6.0, ay + 0.55, "Site A commits fault mask\n+ signature σ^(A,k) (Eq. 6)", ha="center", va="bottom",
            fontsize=8.0, color=CAT["red"])
    arrow(ax, (6.0, ay - 0.25), (6.0, ay - 1.15), color=CAT["magenta"], lw=1.6)
    box(ax, 4.6, ay - 2.05, 2.8, 0.9, "broadcast to fleet\n(server registry)", fc=CAT["magenta"], tc=SURFACE, ec="none", fs=7.8)

    # Site B timeline (bottom): CTFR flag before own DTTS trigger
    by = 1.6
    ax.plot([0.4, 14.8], [by, by], color=AXIS, lw=1.4, zorder=1)
    ax.text(-0.2, by, "Site B", ha="right", va="center", fontsize=9.5, color=INK_SECONDARY, weight="bold")

    flag_x = 8.4
    owntrigger_x = 12.6
    arrow(ax, (5.6, by + 1.55), (flag_x, by + 0.28), color=CAT["magenta"], lw=1.4, connectionstyle="arc3,rad=-0.15")
    ax.add_patch(Circle((flag_x, by), 0.2, facecolor=CAT["magenta"], edgecolor="none", zorder=3))
    ax.text(flag_x, by + 0.55, "CTFR flag raised\n(cos-sim > τ_CTFR, Eq. 6)", ha="center", va="bottom",
            fontsize=8.0, color=CAT["magenta"])

    ax.add_patch(Circle((owntrigger_x, by), 0.2, facecolor=INK_MUTED, edgecolor="none", zorder=3))
    ax.text(owntrigger_x, by + 0.55, "Site B's own DTTS\nwould have triggered here", ha="center", va="bottom",
            fontsize=8.0, color=INK_MUTED)

    ax.annotate("", xy=(owntrigger_x, by - 0.55), xytext=(flag_x, by - 0.55),
                arrowprops=dict(arrowstyle="<->", color=STATUS["good"], lw=1.6))
    ax.text((flag_x + owntrigger_x) / 2, by - 0.95, "Early-Warning Lead Time\n(Section 4.7, Table 6)",
            ha="center", va="top", fontsize=8.2, color=STATUS["good"], weight="bold")

    ax.text(7.6, -0.3,
            "Site B's raw sensor data never leaves Site B; only σ^(A,k) (bounded-dimension, Proposition 5.5)\n"
            "crosses the network, and the similarity comparison (Eq. 6) runs locally at Site B.",
            ha="center", va="center", fontsize=8.0, color=INK_MUTED)

    fig.savefig(os.path.join(FIGURES_DIR, "fig3_ctfr_mechanism.png"))
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════
# Figure 4 — Three-benchmark drift-injection design
# ═══════════════════════════════════════════════════════════════════════════
def fig4():
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.6))
    fig.patch.set_facecolor(SURFACE)

    panels = [
        ("Fed-Twin-CMAPSS", "NASA C-MAPSS turbofan [10]",
         ["Nominal", "Degrading", "Near end-of-life"], "operating-condition\nswitch (FD002/FD004)\n= sensor recalibration event",
         CAT["blue"]),
        ("Fed-Twin-FEMTO", "IEEE PHM 2012 / PRONOSTIA [11]",
         ["Early wear", "Degrading", "Near failure"], "progressive\nbearing wear\n(all sites)",
         CAT["green"]),
        ("Fed-Twin-MIMII", "MIMII acoustic [12]",
         ["Normal (SNR a)", "Normal (SNR b)", "Anomalous"], "SNR / gain switch\nmid-stream\n= sensor recalibration event",
         CAT["aqua"]),
    ]

    for ax, (title, source, stages, event, color) in zip(axes, panels):
        ax.set_facecolor(SURFACE)
        ax.set_xlim(-0.5, 10.5)
        ax.set_ylim(-2.4, 3.2)
        ax.axis("off")
        ax.set_title(f"{title}\n{source}", fontsize=9.3, color=INK, pad=8)

        seg_w = 10.0 / len(stages)
        ramp = [SEQ_BLUE[250], SEQ_BLUE[450], SEQ_BLUE[650]]
        for i, (stage, fc) in enumerate(zip(stages, ramp)):
            x0 = i * seg_w
            ax.add_patch(FancyBboxPatch((x0, 0.4), seg_w - 0.15, 1.1, boxstyle="round,pad=0.02,rounding_size=0.05",
                                          linewidth=0, facecolor=fc, zorder=3))
            tc = SURFACE if i >= 1 else INK
            ax.text(x0 + (seg_w - 0.15) / 2, 0.95, stage, ha="center", va="center", fontsize=7.6, color=tc, zorder=4)
            if i > 0:
                ax.plot([x0, x0], [0.4, 1.5], color=SURFACE, lw=2.2, zorder=5)
                ax.plot([x0], [1.9], marker="v", markersize=6, color=INK_MUTED, zorder=5)

        ax.annotate("", xy=(0, -0.15), xytext=(10, -0.15),
                     arrowprops=dict(arrowstyle="-|>", color=AXIS, lw=1.3))
        ax.text(5, -0.55, "DTTS commit points determined online (Eq. 4);\nstage bins shown here are the evaluation reference only",
                 ha="center", va="top", fontsize=7.2, color=INK_MUTED)

        ax.add_patch(FancyBboxPatch((1.2, -2.2), 7.6, 1.15, boxstyle="round,pad=0.03,rounding_size=0.08",
                                      linewidth=1.2, edgecolor=color, facecolor=SURFACE, zorder=3))
        ax.text(5, -1.6, event, ha="center", va="center", fontsize=7.8, color=INK, zorder=4)

    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "fig4_benchmark_construction.png"))
    plt.close(fig)


if __name__ == "__main__":
    fig1()
    fig2()
    fig3()
    fig4()
    print(f"Wrote 4 figures to {FIGURES_DIR}")
