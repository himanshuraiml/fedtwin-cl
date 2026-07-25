"""
Real preprocessing for NASA C-MAPSS -> Fed-Twin-CMAPSS benchmark.

Loads the actual train_FD001.txt (single condition, 100 engines) and
train_FD002.txt (six conditions, 260 engines) files, builds a per-engine
sequence of (sensor window -> capped RUL) pairs, and assigns 24 engines to
the manuscript's 24 simulated sites: 16 from FD001 (single, stable
operating condition) and 8 from FD002 (six operating conditions per
engine, standing in for the FD002/FD004 "operating-condition-switch"
recalibration-event sites in the manuscript's benchmark design).

Standard C-MAPSS preprocessing choices (used almost universally in the
literature since Heimes 2008 and Saxena et al. 2008 themselves):
  - RUL capped at 125 cycles (piecewise-linear RUL): early-life RUL labels
    are otherwise enormous and uninformative, since degradation has not
    yet started.
  - 14 of 21 sensors retained; the other 7 are dropped because they are
    exactly or near-constant in FD001 and carry no signal (a real,
    well-known property of this dataset, not an artifact of this project).
  - Per-sensor z-score normalization using TRAINING-set statistics only.

Health indicator used to drive DTTS: elapsed-life fraction
h_t = cycle / max_cycle(engine). This is computed directly from the data
(not from a model), monotonic in [0, 1], and is the real-data instantiation
of the "rolling health-indicator statistic" Section 4.3 of the manuscript
describes -- see EXPERIMENT_PROCEDURE.md for why this specific choice was
made over a model-residual-based indicator (simplicity, given the spike's
goal is validating the mask-registry/DTTS/CTFR mechanism on real signal
statistics, not building a state-of-the-art RUL health indicator).

Ground-truth STAGE (used only for measuring forgetting, never given to
DTTS): a 3-way quantile split of h_t per engine (nominal / degrading /
near-end-of-life), matching the manuscript's Fed-Twin-CMAPSS description.
"""
import os
import numpy as np

DATA_DIR = os.path.join(os.path.dirname(__file__), "cmapss",
                         "6. Turbofan Engine Degradation Simulation Data Set")
OUT_DIR = os.path.join(os.path.dirname(__file__), "processed")
os.makedirs(OUT_DIR, exist_ok=True)

COLS = ["unit", "cycle", "op1", "op2", "op3"] + [f"s{i}" for i in range(1, 22)]

# Sensors that are (near-)constant in FD001 and carry no signal -- standard
# exclusion since Heimes (2008) and used in nearly all subsequent work.
DROP_SENSORS = ["s1", "s5", "s6", "s10", "s16", "s18", "s19"]
KEEP_SENSORS = [c for c in COLS if c.startswith("s") and c not in DROP_SENSORS]
RUL_CAP = 125
WINDOW = 8  # cycles per input window, for a real TCN's temporal receptive field


def load_fd(name):
    path = os.path.join(DATA_DIR, f"train_{name}.txt")
    rows = []
    with open(path) as f:
        for line in f:
            parts = line.split()
            if not parts:
                continue
            rows.append([float(x) for x in parts])
    arr = np.array(rows, dtype=np.float64)
    return arr  # shape (n_rows, 26)


def build_engine_dict(arr):
    """Returns {unit_id: {"cycle": [...], "sensors": (T, n_keep), "op": (T,3)}}"""
    col_idx = {c: i for i, c in enumerate(COLS)}
    engines = {}
    units = np.unique(arr[:, col_idx["unit"]]).astype(int)
    for u in units:
        mask = arr[:, col_idx["unit"]] == u
        rows = arr[mask]
        rows = rows[np.argsort(rows[:, col_idx["cycle"]])]
        cycle = rows[:, col_idx["cycle"]]
        sensors = rows[:, [col_idx[c] for c in KEEP_SENSORS]]
        op = rows[:, [col_idx["op1"], col_idx["op2"], col_idx["op3"]]]
        engines[int(u)] = {"cycle": cycle, "sensors": sensors, "op": op}
    return engines


def normalize_sensors(engines, mean=None, std=None):
    """Single-condition normalization (FD001/FD003): one global mean/std."""
    if mean is None:
        all_sensors = np.concatenate([e["sensors"] for e in engines.values()], axis=0)
        mean = all_sensors.mean(axis=0)
        std = all_sensors.std(axis=0)
        std[std < 1e-6] = 1.0
    for e in engines.values():
        e["sensors"] = (e["sensors"] - mean) / std
    return mean, std


def normalize_sensors_by_condition(engines, n_regimes=6):
    """Per-operating-condition normalization (FD002/FD004): raw sensor
    readings differ drastically across the six operating regimes, so a
    single global mean/std (correct for FD001's one stable condition)
    leaves some regimes wildly out of the normalized range the model was
    ever trained on -- a well-documented requirement for these two
    sub-datasets in the C-MAPSS literature, not a project-specific choice.
    Clusters every row's (op1, op2, op3) into `n_regimes` operating
    conditions and z-scores each regime's sensor readings using that
    regime's OWN statistics (fit on this engine set, not reused from
    FD001/FD003)."""
    from sklearn.cluster import KMeans
    all_ops = np.concatenate([e["op"] for e in engines.values()], axis=0)
    km = KMeans(n_clusters=n_regimes, n_init=10, random_state=0).fit(all_ops)

    all_sensors = np.concatenate([e["sensors"] for e in engines.values()], axis=0)
    all_labels = km.predict(all_ops)
    regime_mean = np.zeros((n_regimes, all_sensors.shape[1]))
    regime_std = np.ones((n_regimes, all_sensors.shape[1]))
    for r in range(n_regimes):
        sel = all_sensors[all_labels == r]
        if len(sel) > 1:
            regime_mean[r] = sel.mean(axis=0)
            s = sel.std(axis=0)
            s[s < 1e-6] = 1.0
            regime_std[r] = s

    for e in engines.values():
        labels = km.predict(e["op"])
        e["sensors"] = (e["sensors"] - regime_mean[labels]) / regime_std[labels]
        e["regime"] = labels
    return km, regime_mean, regime_std


def build_site_record(engine, engine_id, source, n_stages=3):
    cycle = engine["cycle"]
    sensors = engine["sensors"]
    T = len(cycle)
    max_cycle = cycle[-1]

    h = cycle / max_cycle  # elapsed-life fraction, health indicator
    rul = np.clip(max_cycle - cycle, 0, RUL_CAP)
    rul_norm = rul / RUL_CAP  # normalize target to [0,1] for stable training

    # ground-truth stage (evaluation only): quantile bins of h
    edges = np.quantile(h, np.linspace(0, 1, n_stages + 1))
    stage = np.clip(np.digitize(h, edges[1:-1]), 0, n_stages - 1)

    # windows: for t >= WINDOW-1, input is sensors[t-WINDOW+1 : t+1]
    windows, targets, hs, stages, cycles = [], [], [], [], []
    for t in range(WINDOW - 1, T):
        windows.append(sensors[t - WINDOW + 1: t + 1])
        targets.append(rul_norm[t])
        hs.append(h[t])
        stages.append(stage[t])
        cycles.append(cycle[t])

    return {
        "engine_id": int(engine_id),
        "source": source,  # "FD001" or "FD002"
        "windows": np.stack(windows).astype(np.float32),      # (N, WINDOW, n_sensors)
        "rul_norm": np.array(targets, dtype=np.float32),        # (N,)
        "health": np.array(hs, dtype=np.float32),               # (N,)
        "stage": np.array(stages, dtype=np.int64),              # (N,)
        "cycle": np.array(cycles, dtype=np.float32),
        "op": engine["op"][WINDOW - 1:].astype(np.float32),
    }


if __name__ == "__main__":
    print("Loading FD001 (single condition)...")
    fd001 = load_fd("FD001")
    eng001 = build_engine_dict(fd001)
    mean, std = normalize_sensors(eng001)
    print(f"  {len(eng001)} engines, sensor dim {len(KEEP_SENSORS)}")

    print("Loading FD002 (six conditions)...")
    fd002 = load_fd("FD002")
    eng002 = build_engine_dict(fd002)
    # FD002 needs PER-CONDITION normalization, not FD001's global stats: its
    # six operating regimes produce raw sensor ranges that differ by orders
    # of magnitude, so reusing FD001's single-condition mean/std leaves most
    # of FD002 far outside the normalized range any model ever trains on
    # (confirmed empirically: doing this naively produced catastrophic
    # train/eval blowups for every FD002-sourced site, traced back to this
    # exact preprocessing step, not a training-loop bug).
    normalize_sensors_by_condition(eng002, n_regimes=6)
    print(f"  {len(eng002)} engines")

    rng = np.random.default_rng(1)
    fd001_ids = rng.choice(sorted(eng001.keys()), size=16, replace=False)
    fd002_ids = rng.choice(sorted(eng002.keys()), size=8, replace=False)
    print(f"Selected 16 FD001 engines: {sorted(fd001_ids)}")
    print(f"Selected 8 FD002 engines (recalibration sites): {sorted(fd002_ids)}")

    sites = []
    for eid in sorted(fd001_ids):
        sites.append(build_site_record(eng001[eid], eid, "FD001"))
    for eid in sorted(fd002_ids):
        sites.append(build_site_record(eng002[eid], eid, "FD002"))

    lengths = [s["windows"].shape[0] for s in sites]
    print(f"Site sequence lengths: min={min(lengths)} max={max(lengths)} mean={np.mean(lengths):.1f}")

    np.savez(os.path.join(OUT_DIR, "cmapss_sites.npz"),
             sites=np.array(sites, dtype=object), keep_sensors=KEEP_SENSORS,
             mean=mean, std=std, allow_pickle=True)
    print(f"Saved {len(sites)} sites to {OUT_DIR}/cmapss_sites.npz")
