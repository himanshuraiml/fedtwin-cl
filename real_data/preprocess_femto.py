"""
Real preprocessing for IEEE PHM 2012 / FEMTO-ST PRONOSTIA bearing dataset
-> Fed-Twin-FEMTO benchmark.

Uses all 17 real run-to-failure bearings with revealed ground truth: the 6
in Learning_set (Bearing1_1, 1_2, 2_1, 2_2, 3_1, 3_2) plus the 11 in
Full_Test_Set (Bearing1_3..1_7, 2_3..2_7, 3_3), spanning the three real
operating conditions (load/speed combinations) of the original challenge.
24 simulated sites are assigned from this pool of 17 (7 sites reuse a
bearing already used elsewhere, unavoidable since only 17 real run-to-
failure traces exist in the standard dataset -- see EXPERIMENT_PROCEDURE.md
for the same reuse note).

Each bearing's data is a sequence of ~2560-sample snapshot files (0.1s at
25.6kHz), recorded periodically until failure. Rather than feed the raw
waveform into the network, each snapshot is reduced to a small set of
STANDARD time-domain vibration features (RMS, kurtosis, peak, crest factor,
std, for both horizontal and vertical channels = 10 features), a common,
well-established practice in bearing prognostics feature engineering. This
keeps the exact same "WINDOW x features -> Conv1d -> FC" TCN-Nano-lite
architecture and mask-registry/DTTS/CTFR code already validated on
C-MAPSS, fed by a different real signal's derived features instead of
raw multivariate telemetry.

Health indicator: rolling RMS of the horizontal channel (Section 4.3 of
the manuscript names this exact statistic for Fed-Twin-FEMTO), normalized
per bearing to [0, 1] by its own observed min/max -- the real,
signal-derived instantiation of "progressive bearing wear," not a
simulated ramp.
"""
import os
import glob
import numpy as np
import pandas as pd

BASE = os.path.join(os.path.dirname(__file__), "femto", "10. FEMTO Bearing")
LEARNING_DIR = os.path.join(BASE, "Learning_set")
FULLTEST_DIR = os.path.join(BASE, "Full_Test_Set")
OUT_DIR = os.path.join(os.path.dirname(__file__), "processed")
os.makedirs(OUT_DIR, exist_ok=True)

RUL_CAP_SNAPSHOTS = 50  # cap RUL at 50 snapshots (~500s) before normalizing, matches C-MAPSS convention
WINDOW = 8  # snapshots per input window, matching C-MAPSS's WINDOW for architecture reuse

BEARINGS = (
    [(LEARNING_DIR, f"Bearing1_{i}") for i in (1, 2)]
    + [(FULLTEST_DIR, f"Bearing1_{i}") for i in (3, 4, 5, 6, 7)]
    + [(LEARNING_DIR, f"Bearing2_{i}") for i in (1, 2)]
    + [(FULLTEST_DIR, f"Bearing2_{i}") for i in (3, 4, 5, 6, 7)]
    + [(LEARNING_DIR, f"Bearing3_{i}") for i in (1, 2)]
    + [(FULLTEST_DIR, "Bearing3_3")]
)


def read_snapshot(path):
    # 6 columns, no header: hour, min, sec, microsec-ish counter, horiz accel, vert accel
    df = pd.read_csv(path, header=None)
    if df.shape[1] < 6:
        # a few files in this dataset are ';'-delimited instead of ','
        df = pd.read_csv(path, header=None, sep=";")
    horiz = df.iloc[:, 4].to_numpy(dtype=np.float64)
    vert = df.iloc[:, 5].to_numpy(dtype=np.float64)
    return horiz, vert


def features_for_snapshot(horiz, vert):
    def stats(x):
        rms = np.sqrt(np.mean(x ** 2))
        peak = np.max(np.abs(x))
        std = np.std(x)
        mean4 = np.mean((x - x.mean()) ** 4)
        kurt = mean4 / (std ** 4 + 1e-9)
        crest = peak / (rms + 1e-9)
        return [rms, peak, std, kurt, crest]
    return np.array(stats(horiz) + stats(vert), dtype=np.float32)  # 10 features


def build_bearing_record(folder, name, site_id, n_stages=3):
    files = sorted(glob.glob(os.path.join(folder, name, "acc_*.csv")))
    feats, rms_h = [], []
    for fp in files:
        try:
            horiz, vert = read_snapshot(fp)
        except Exception:
            continue
        f = features_for_snapshot(horiz, vert)
        feats.append(f)
        rms_h.append(f[0])  # horizontal RMS = health-indicator raw signal

    feats = np.stack(feats)  # (T, 10)
    rms_h = np.array(rms_h)
    T = len(rms_h)

    # health indicator: ROLLING (smoothed) horizontal RMS, per-bearing
    # min-max normalized. The manuscript names this exact statistic
    # ("rolling RMS-of-vibration-envelope") for a reason: raw per-snapshot
    # RMS on real accelerometer data is noisy enough that the instantaneous
    # signal crosses quantile-stage boundaries back and forth many times
    # before settling (confirmed empirically: this produced 2-3x more
    # ground-truth "stage" transitions than the at-most-2 a monotonic
    # 3-stage split should ever give, since quantile bins computed on an
    # oscillating signal are themselves revisited repeatedly). A rolling
    # mean is the actual "rolling" statistic, not a cosmetic smoothing
    # choice, and gives a health indicator that only crosses each stage
    # boundary once, matching what "progressive bearing wear" is supposed
    # to mean.
    # pandas' rolling().mean() with min_periods=1 shrinks the window near
    # the boundaries instead of mixing in implicit zeros the way
    # np.convolve(..., mode="same") does -- that zero-padding version was
    # tried first and confirmed (by inspection) to fabricate a sharp fake
    # "recovery" over the last ~7 samples of every sequence, exactly where
    # a real bearing is failing fastest and the health indicator matters
    # most, which was corrupting the ground-truth stage assignment near
    # the end of the run for every site.
    smooth_window = 15
    rms_h_smooth = pd.Series(rms_h).rolling(window=smooth_window, min_periods=1, center=True).mean().to_numpy()
    h = (rms_h_smooth - rms_h_smooth.min()) / (rms_h_smooth.max() - rms_h_smooth.min() + 1e-9)

    # per-feature z-score using THIS bearing's own statistics (each bearing
    # run at a fixed condition, so unlike C-MAPSS FD002 there is no
    # within-bearing regime shift to worry about)
    mean, std = feats.mean(axis=0), feats.std(axis=0)
    std[std < 1e-6] = 1.0
    feats_norm = (feats - mean) / std

    rul = np.clip(T - 1 - np.arange(T), 0, RUL_CAP_SNAPSHOTS)
    rul_norm = rul / RUL_CAP_SNAPSHOTS

    # Ground-truth stage by TIME tercile (equal-DURATION segments), not by
    # health-indicator VALUE quantile. The two are not the same when h
    # rises slowly at first and steeply near failure (confirmed on this
    # data): a value-quantile split can give one stage a tiny fraction of
    # real elapsed time, small enough that a 20-round schedule's chunking
    # skips over it inside a single round without ever training on it,
    # so it gets "protected" using untrained initial weights (a
    # meaningless zero-forgetting result, not a real one). An equal-
    # duration split guarantees every stage gets a proportional share of
    # the round schedule, mirroring how the C-MAPSS elapsed-life-fraction
    # health indicator already naturally divides into roughly equal
    # calendar-time thirds.
    stage = np.clip((np.arange(T) * n_stages) // T, 0, n_stages - 1)

    windows, targets, hs, stages, idxs = [], [], [], [], []
    for t in range(WINDOW - 1, T):
        windows.append(feats_norm[t - WINDOW + 1: t + 1])
        targets.append(rul_norm[t])
        hs.append(h[t])
        stages.append(stage[t])
        idxs.append(t)

    return {
        "engine_id": site_id, "source": name,
        "windows": np.stack(windows).astype(np.float32),
        "rul_norm": np.array(targets, dtype=np.float32),
        "health": np.array(hs, dtype=np.float32),
        "stage": np.array(stages, dtype=np.int64),
        "cycle": np.array(idxs, dtype=np.float32),
        "op": np.zeros((len(idxs), 3), dtype=np.float32),  # no operating-condition drift for FEMTO
    }


if __name__ == "__main__":
    rng = np.random.default_rng(1)
    assignment = list(range(len(BEARINGS)))
    extra = list(rng.choice(len(BEARINGS), size=24 - len(BEARINGS), replace=False))
    assignment += extra
    rng.shuffle(assignment)

    sites = []
    for site_id, bidx in enumerate(assignment):
        folder, name = BEARINGS[bidx]
        print(f"site {site_id}: {name} ({'Learning' if folder == LEARNING_DIR else 'FullTest'})")
        sites.append(build_bearing_record(folder, name, site_id))

    lengths = [s["windows"].shape[0] for s in sites]
    print(f"Site sequence lengths: min={min(lengths)} max={max(lengths)} mean={np.mean(lengths):.1f}")

    np.savez(os.path.join(OUT_DIR, "femto_sites.npz"),
              sites=np.array(sites, dtype=object), allow_pickle=True)
    print(f"Saved {len(sites)} sites to {OUT_DIR}/femto_sites.npz")
