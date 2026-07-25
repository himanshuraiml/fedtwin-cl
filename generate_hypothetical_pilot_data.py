"""
Generates a HYPOTHETICAL pilot_results.json for FedTwin-CL.

This is NOT real experimental output. No dataset acquisition, DTTS/CTFR
implementation, or pilot run has happened yet (see the manuscript status
note). The numbers here are illustrative placeholders, hand-picked to be
internally consistent with (a) the theoretical bounds derived in Section 5
(near-zero forgetting for any mask-registry method, a >90% communication
reduction consistent with Theorem 5.3's inherited bound, the largest energy
reduction at the weakest device tier) and (b) the qualitative ordering
FedCAT's real results established on its own synthetic benchmarks (Table 2
of Paper_4). They exist so the manuscript can be read end-to-end in a
submission-shaped form while the real pilot is pending, and MUST be
replaced wholesale once real pilot output exists -- do not tune these
numbers to "look better," and do not cite them outside this project.

Run this, then `generate_result_figures.py`, to (re)produce Figures 5-10
and Tables 3-6 exactly as embedded in FedTwin-CL_Manuscript.md.
"""

import json
import os
import numpy as np

OUT_PATH = os.path.join(os.path.dirname(__file__), "pilot_results.json")
rng = np.random.default_rng(42)

BENCHMARKS = ["fed-twin-cmapss", "fed-twin-femto", "fed-twin-mimii"]
METHODS_COMM = ["fedavg", "fedprox", "fedavg-ewc", "fedcat-external", "fedtwin-cl-no-ctfr", "fedtwin-cl"]
METHODS_ALL = METHODS_COMM + ["centralized"]
ROUNDS = list(range(1, 21))

# 24-site fleet, device-class composition reused unchanged from FedCAT (Table 1)
DEVICE_CLASSES = (
    ["remote-asset"] * 6 + ["sensor-edge"] * 10 + ["line-gateway"] * 6 + ["plant-fog"] * 2
)
E_TX = {"remote-asset": 1.8, "sensor-edge": 0.09, "line-gateway": 0.09, "plant-fog": 0.012}  # mJ/B
CMAX = {"remote-asset": 32, "sensor-edge": 64, "line-gateway": 128, "plant-fog": 96}
CMAX_MEAN = np.mean(list(CMAX.values()))
DEVICE_OFFSET = {"remote-asset": -0.025, "sensor-edge": -0.010, "line-gateway": 0.005, "plant-fog": 0.015}

# Target final-round FPS and FBWT per (benchmark, method) -- Table 3
FINAL_FPS = {
    "fed-twin-cmapss": {
        "fedavg": 0.583, "fedprox": 0.612, "fedavg-ewc": 0.771,
        "fedcat-external": 0.892, "fedtwin-cl-no-ctfr": 0.879, "fedtwin-cl": 0.881,
        "centralized": 0.918,
    },
    "fed-twin-femto": {
        "fedavg": 0.549, "fedprox": 0.578, "fedavg-ewc": 0.744,
        "fedcat-external": 0.869, "fedtwin-cl-no-ctfr": 0.851, "fedtwin-cl": 0.855,
        "centralized": 0.901,
    },
    "fed-twin-mimii": {
        "fedavg": 0.601, "fedprox": 0.634, "fedavg-ewc": 0.782,
        "fedcat-external": 0.905, "fedtwin-cl-no-ctfr": 0.891, "fedtwin-cl": 0.897,
        "centralized": 0.929,
    },
}
FINAL_FBWT = {
    "fed-twin-cmapss": {"fedavg": -0.243, "fedprox": -0.211, "fedavg-ewc": -0.079,
                         "fedcat-external": -0.006, "fedtwin-cl-no-ctfr": -0.009, "fedtwin-cl": -0.008},
    "fed-twin-femto": {"fedavg": -0.268, "fedprox": -0.232, "fedavg-ewc": -0.091,
                        "fedcat-external": -0.007, "fedtwin-cl-no-ctfr": -0.011, "fedtwin-cl": -0.010},
    "fed-twin-mimii": {"fedavg": -0.219, "fedprox": -0.187, "fedavg-ewc": -0.068,
                        "fedcat-external": -0.005, "fedtwin-cl-no-ctfr": -0.008, "fedtwin-cl": -0.006},
}
# convergence speed (smaller tau = faster) and starting FPS per method
TAU = {"fedavg": 9.0, "fedprox": 8.0, "fedavg-ewc": 6.5,
       "fedcat-external": 4.2, "fedtwin-cl-no-ctfr": 4.6, "fedtwin-cl": 4.4, "centralized": 3.0}
START_FPS = 0.14

# Table 4 -- cumulative fleet uplink over 20 rounds, Fed-Twin-FEMTO (bytes)
TOTAL_UPLINK_MB = {
    "fedavg": 92.4, "fedprox": 92.4, "fedavg-ewc": 92.4,
    "fedcat-external": 6.10, "fedtwin-cl-no-ctfr": 6.24, "fedtwin-cl": 6.31,
}

# Table 5 -- DTTS sensitivity (Fed-Twin-FEMTO, fedtwin-cl, final round)
LAMBDA_PH_SWEEP = {
    5: {"fps": 0.842, "fbwt": -0.014, "timing_error": 14.2, "tasks_per_site": 5.8},
    10: {"fps": 0.867, "fbwt": -0.011, "timing_error": 7.6, "tasks_per_site": 4.1},
    20: {"fps": 0.855, "fbwt": -0.010, "timing_error": 4.3, "tasks_per_site": 2.9},
    40: {"fps": 0.849, "fbwt": -0.010, "timing_error": 11.8, "tasks_per_site": 1.7},
}

# Table 6 -- early-warning lead time (per benchmark)
LEAD_TIME = {
    "fed-twin-cmapss": {"mean": 3.4, "sd": 1.6, "precision": 0.78, "recall": 0.61, "n_confirmed": 42},
    "fed-twin-femto": {"mean": 2.1, "sd": 1.1, "precision": 0.74, "recall": 0.58, "n_confirmed": 35},
    "fed-twin-mimii": {"mean": 1.6, "sd": 0.9, "precision": 0.81, "recall": 0.53, "n_confirmed": 29},
}

# Section 7.5 -- heterogeneity sensitivity (Fed-Twin-FEMTO, fedtwin-cl, round 20)
GAMMA_SWEEP = [0.0, 0.2, 0.4, 0.6, 0.8]
GAMMA_DEVICE_FPS = {
    "remote-asset": [0.712, 0.771, 0.818, 0.846, 0.851],
    "sensor-edge":  [0.831, 0.841, 0.849, 0.853, 0.850],
    "line-gateway": [0.869, 0.865, 0.861, 0.858, 0.849],
    "plant-fog":    [0.881, 0.874, 0.867, 0.860, 0.848],
}


def curve(target, method, r):
    tau = TAU[method]
    return target - (target - START_FPS) * np.exp(-r / tau)


records = []

# ── Convergence + communication + energy records ───────────────────────────
for bench in BENCHMARKS:
    for method in METHODS_ALL:
        target = FINAL_FPS[bench][method]
        for site_id, dc in enumerate(DEVICE_CLASSES):
            offset = DEVICE_OFFSET[dc] if method != "centralized" else 0.0
            cum_bytes = 0
            for r in ROUNDS:
                fps = float(np.clip(curve(target, method, r) + offset + rng.normal(0, 0.006), 0, 1))
                fbwt = None
                if method != "centralized" and r >= 15:
                    fbwt = float(FINAL_FBWT[bench][method] + rng.normal(0, 0.003))

                round_bytes = 0
                tx_energy = 0.0
                if method in METHODS_COMM and bench == "fed-twin-femto":
                    per_site_mean_kb = (TOTAL_UPLINK_MB[method] * 1000) / 20 / 24
                    scale = CMAX[dc] / CMAX_MEAN
                    round_bytes = int(per_site_mean_kb * 1000 * scale * (1 + rng.normal(0, 0.05)))
                    tx_energy = round_bytes * E_TX[dc]
                    cum_bytes += round_bytes

                records.append({
                    "benchmark": bench, "method": method, "site_id": site_id,
                    "device_class": dc, "round": r, "fps": fps, "fbwt": fbwt,
                    "uplink_bytes": cum_bytes, "tx_energy_mj": tx_energy,
                    "lambda_ph": 20.0, "gamma": 0.6 if method == "fedtwin-cl" else None,
                    "commit_timing_error": None, "aux": False,
                    "ctfr_flag_raised": False, "ctfr_lead_time": None, "ctfr_flag_confirmed": False,
                })

# ── DTTS sensitivity records (Fed-Twin-FEMTO, fedtwin-cl, final round) ─────
# aux=True throughout: these are a separate lambda_ph sweep, not part of the
# main round-by-round convergence curve at the lambda_ph=20 operating point,
# even where lambda_ph happens to equal 20 here too.
for lam, vals in LAMBDA_PH_SWEEP.items():
    for site_id, dc in enumerate(DEVICE_CLASSES):
        fps = float(np.clip(vals["fps"] + DEVICE_OFFSET[dc] + rng.normal(0, 0.006), 0, 1))
        err = float(max(0.0, vals["timing_error"] + rng.normal(0, 1.2)))
        records.append({
            "benchmark": "fed-twin-femto", "method": "fedtwin-cl", "site_id": site_id,
            "device_class": dc, "round": 20, "fps": fps, "fbwt": vals["fbwt"],
            "uplink_bytes": 0, "tx_energy_mj": 0.0,
            "lambda_ph": float(lam), "gamma": None,
            "commit_timing_error": err, "aux": True,
            "ctfr_flag_raised": False, "ctfr_lead_time": None, "ctfr_flag_confirmed": False,
        })

# ── Heterogeneity sensitivity records (Fed-Twin-FEMTO, fedtwin-cl) ─────────
for gi, g in enumerate(GAMMA_SWEEP):
    for dc in ["remote-asset", "sensor-edge", "line-gateway", "plant-fog"]:
        base = GAMMA_DEVICE_FPS[dc][gi]
        for rep in range(3):  # a few site replicates per device class per gamma
            fps = float(np.clip(base + rng.normal(0, 0.005), 0, 1))
            records.append({
                "benchmark": "fed-twin-femto", "method": "fedtwin-cl", "site_id": 1000 + gi * 10 + rep,
                "device_class": dc, "round": 20, "fps": fps, "fbwt": None,
                "uplink_bytes": 0, "tx_energy_mj": 0.0,
                "lambda_ph": 20.0, "gamma": g,
                "commit_timing_error": None, "aux": True,
                "ctfr_flag_raised": False, "ctfr_lead_time": None, "ctfr_flag_confirmed": False,
            })

# ── CTFR early-warning lead-time records ────────────────────────────────────
for bench in BENCHMARKS:
    stats = LEAD_TIME[bench]
    for i in range(stats["n_confirmed"]):
        lt = float(max(0.1, rng.normal(stats["mean"], stats["sd"])))
        records.append({
            "benchmark": bench, "method": "fedtwin-cl", "site_id": 2000 + i,
            "device_class": DEVICE_CLASSES[i % 24], "round": int(rng.integers(3, 20)),
            "fps": None, "fbwt": None, "uplink_bytes": 0, "tx_energy_mj": 0.0,
            "lambda_ph": 20.0, "gamma": None, "commit_timing_error": None, "aux": True,
            "ctfr_flag_raised": True, "ctfr_lead_time": lt, "ctfr_flag_confirmed": True,
        })
    n_unconfirmed = int(stats["n_confirmed"] * (1 - stats["precision"]) / stats["precision"])
    for i in range(n_unconfirmed):
        records.append({
            "benchmark": bench, "method": "fedtwin-cl", "site_id": 3000 + i,
            "device_class": DEVICE_CLASSES[i % 24], "round": int(rng.integers(3, 20)),
            "fps": None, "fbwt": None, "uplink_bytes": 0, "tx_energy_mj": 0.0,
            "lambda_ph": 20.0, "gamma": None, "commit_timing_error": None, "aux": True,
            "ctfr_flag_raised": True, "ctfr_lead_time": None, "ctfr_flag_confirmed": False,
        })

with open(OUT_PATH, "w") as f:
    json.dump(records, f)

print(f"Wrote {len(records)} HYPOTHETICAL records to {OUT_PATH}")
print("These are illustrative placeholders only -- replace with real pilot output before submission.")
