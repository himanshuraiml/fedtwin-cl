"""
Real preprocessing for MIMII (valve, 6dB + 0dB) -> Fed-Twin-MIMII benchmark.

Unlike C-MAPSS/FEMTO, MIMII has no continuous run-to-failure recording:
each 10-second clip is an independent recording, labeled normal or
abnormal, grouped by machine ID and SNR level, with no timestamp linking
clips to a real degradation timeline. The manuscript's Fed-Twin-MIMII
design ("progressive fault emergence," "SNR/gain recalibration event")
therefore has to be CONSTRUCTED here: each site (one real machine ID) is
given a simulated exposure schedule across the 20 communication rounds --
mostly normal clips early, an increasing share of real abnormal clips
later -- and a subset of sites switch from real 6dB clips to real 0dB
clips at a fixed round, standing in for the recalibration event. Every
clip and every label is real; only the ROUND-TO-CLIP assignment schedule
is simulated, exactly analogous to how the manuscript's own benchmark
design section already frames Fed-Twin-MIMII's drift injection.

Two lessons carried over directly from Fed-Twin-FEMTO's debugging (see
EXPERIMENT_PROCEDURE.md): ground-truth stage is assigned by TIME (round)
tercile, not by any noisy derived-signal quantile, so every stage gets a
guaranteed, even share of the 20-round schedule; and the health indicator
DTTS actually watches is a ROLLING statistic (here: rolling mean of the
site's own recent labels standing in for "rolling anomaly score from the
site's own detection head" per Section 4.3 of the manuscript), not an
instantaneous, noisy per-clip value.

Audio features: each 10s clip (16 kHz, first channel only) is split into
WINDOW=8 equal-length frames; each frame is reduced to 9 standard
time/frequency-domain features (RMS, zero-crossing rate, spectral
centroid, spectral rolloff, and 5 log-energy FFT bands), giving the same
"WINDOW x features" shape C-MAPSS/FEMTO already use, so the identical
TCN-Nano-lite architecture and mask-registry/DTTS/CTFR code apply
unchanged.
"""
import os
import glob
import numpy as np
import pandas as pd
from scipy.io import wavfile

BASE = os.path.join(os.path.dirname(__file__), "mimii")
OUT_DIR = os.path.join(os.path.dirname(__file__), "processed")
os.makedirs(OUT_DIR, exist_ok=True)

MACHINE = "valve"
N_FRAMES = 8  # WINDOW
N_ROUNDS = 20
CLIPS_PER_ROUND = 6  # real clips drawn per site per round


def list_machine_ids(snr_dir):
    root = os.path.join(BASE, snr_dir, MACHINE)
    return sorted(d for d in os.listdir(root) if d.startswith("id_"))


def list_clips(snr_dir, machine_id, label):
    return sorted(glob.glob(os.path.join(BASE, snr_dir, MACHINE, machine_id, label, "*.wav")))


def frame_features(x, sr, n_frames=N_FRAMES):
    n = len(x)
    edges = np.linspace(0, n, n_frames + 1).astype(int)
    feats = []
    for i in range(n_frames):
        seg = x[edges[i]:edges[i + 1]].astype(np.float64)
        if len(seg) < 8:
            seg = np.pad(seg, (0, 8 - len(seg)))
        rms = np.sqrt(np.mean(seg ** 2) + 1e-12)
        zcr = np.mean(np.abs(np.diff(np.sign(seg)))) / 2.0
        spec = np.abs(np.fft.rfft(seg * np.hanning(len(seg))))
        freqs = np.fft.rfftfreq(len(seg), d=1.0 / sr)
        spec_sum = spec.sum() + 1e-12
        centroid = float((freqs * spec).sum() / spec_sum)
        cumspec = np.cumsum(spec)
        rolloff_idx = np.searchsorted(cumspec, 0.85 * cumspec[-1])
        rolloff = float(freqs[min(rolloff_idx, len(freqs) - 1)])
        n_bands = 5
        band_edges = np.linspace(0, len(spec), n_bands + 1).astype(int)
        bands = [np.log(spec[band_edges[j]:band_edges[j + 1]].sum() + 1e-6) for j in range(n_bands)]
        feats.append([rms, zcr, centroid / 1000.0, rolloff / 1000.0] + bands)
    return np.array(feats, dtype=np.float32)  # (n_frames, 9)


def load_clip_features(path):
    sr, x = wavfile.read(path)
    if x.ndim > 1:
        x = x[:, 0]
    return frame_features(x, sr)


def build_site(site_id, machine_id, primary_snr="6dB", recal_round=None, secondary_snr="0dB",
               rng_seed=1):
    rng = np.random.default_rng(rng_seed * 10000 + site_id)
    normal_primary = list_clips(f"{primary_snr}_{MACHINE}", machine_id, "normal")
    abnormal_primary = list_clips(f"{primary_snr}_{MACHINE}", machine_id, "abnormal")
    normal_secondary = list_clips(f"{secondary_snr}_{MACHINE}", machine_id, "normal") if recal_round else []
    abnormal_secondary = list_clips(f"{secondary_snr}_{MACHINE}", machine_id, "abnormal") if recal_round else []

    # Simulated exposure schedule: abnormal fraction ramps 0 -> 0.7 across
    # the 20 rounds (progressive fault emergence), same shape for every
    # site regardless of machine ID -- only the underlying REAL clips
    # differ per site.
    abnormal_frac = np.clip(np.linspace(-0.15, 0.85, N_ROUNDS), 0, 1)

    round_windows, round_labels = [], []
    for r in range(N_ROUNDS):
        use_secondary = recal_round is not None and r >= recal_round
        normal_pool = normal_secondary if use_secondary else normal_primary
        abnormal_pool = abnormal_secondary if use_secondary else abnormal_primary
        n_abn = int(round(CLIPS_PER_ROUND * abnormal_frac[r]))
        n_norm = CLIPS_PER_ROUND - n_abn
        chosen_paths, labels = [], []
        if len(normal_pool) > 0 and n_norm > 0:
            idx = rng.integers(0, len(normal_pool), size=n_norm)
            chosen_paths += [normal_pool[i] for i in idx]
            labels += [0] * n_norm
        if len(abnormal_pool) > 0 and n_abn > 0:
            idx = rng.integers(0, len(abnormal_pool), size=n_abn)
            chosen_paths += [abnormal_pool[i] for i in idx]
            labels += [1] * n_abn
        feats = [load_clip_features(p) for p in chosen_paths]
        round_windows.append(np.stack(feats) if feats else np.zeros((0, N_FRAMES, 9), dtype=np.float32))
        round_labels.append(np.array(labels, dtype=np.float32))

    # Per-site z-score normalization of the raw audio features (RMS, ZCR,
    # centroid/rolloff, log-energy bands live on wildly different scales --
    # RMS is ~O(200), ZCR is ~O(0.1) -- and skipping this caused a
    # catastrophic-magnitude failure the first time an unnormalized
    # feature set this imbalanced was fed to the same conv1d+SGD setup on
    # C-MAPSS FD002, see EXPERIMENT_PROCEDURE.md bug #8). Pooled across all
    # of this site's own real clips (both SNR conditions if it has a
    # recalibration event), same convention as FEMTO's per-bearing z-score.
    nonempty = [w for w in round_windows if w.shape[0] > 0]
    pooled = np.concatenate(nonempty, axis=0) if nonempty else np.zeros((1, N_FRAMES, 9), dtype=np.float32)
    feat_mean = pooled.reshape(-1, pooled.shape[-1]).mean(axis=0)
    feat_std = pooled.reshape(-1, pooled.shape[-1]).std(axis=0)
    feat_std[feat_std < 1e-6] = 1.0
    round_windows = [(w - feat_mean) / feat_std if w.shape[0] > 0 else w for w in round_windows]

    # Ground-truth stage: TIME (round) tercile -- the FEMTO lesson.
    stage_per_round = np.clip((np.arange(N_ROUNDS) * 3) // N_ROUNDS, 0, 2)

    # Health indicator DTTS watches: rolling mean of the REAL observed
    # label fraction over the last 3 rounds (a real, observable quantity a
    # site has -- how many of its own recent flagged clips were abnormal --
    # standing in for "rolling anomaly score from the site's own detection
    # head," Section 4.3 of the manuscript). Not the deterministic
    # `abnormal_frac` schedule directly: that would let DTTS see a
    # perfectly smooth signal no real deployment would ever have, since
    # real per-round abnormal fraction is itself noisy (small clip counts
    # per round).
    observed_frac = np.array([labels.mean() if len(labels) else 0.0 for labels in round_labels])
    health = pd.Series(observed_frac).rolling(window=3, min_periods=1, center=False).mean().to_numpy()

    return {
        "engine_id": site_id, "source": f"{machine_id}_{primary_snr}",
        "round_windows": round_windows,   # list of (n_clips_r, N_FRAMES, 9) arrays, one per round
        "round_labels": round_labels,     # list of (n_clips_r,) arrays
        "stage_per_round": stage_per_round,
        "health_per_round": health.astype(np.float32),
        "abnormal_frac": abnormal_frac,
        "recal_round": recal_round,
    }


if __name__ == "__main__":
    ids_6db = list_machine_ids(f"6dB_{MACHINE}")
    print(f"Found {len(ids_6db)} machine IDs for {MACHINE} @ 6dB: {ids_6db}")

    rng = np.random.default_rng(1)
    # 24 sites drawn from the available machine IDs (with reuse if fewer
    # than 24 exist, same convention as C-MAPSS/FEMTO)
    n_ids = len(ids_6db)
    assignment = [ids_6db[i % n_ids] for i in rng.permutation(24)]
    recal_sites = set(range(0, 24, 3))  # matches the manuscript's fed-twin-mimii recal_sites pattern

    sites = []
    for site_id, machine_id in enumerate(assignment):
        recal_round = 10 if site_id in recal_sites else None
        print(f"site {site_id}: machine_id={machine_id} recal_round={recal_round}")
        sites.append(build_site(site_id, machine_id, recal_round=recal_round, rng_seed=1))

    np.savez(os.path.join(OUT_DIR, "mimii_sites.npz"), sites=np.array(sites, dtype=object), allow_pickle=True)
    print(f"Saved {len(sites)} sites to {OUT_DIR}/mimii_sites.npz")
