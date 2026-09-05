"""
Alternative drift detector head-to-head, real Fed-Twin-FEMTO data.

Reviewer gap item 5: implement a second real task-boundary detector and
compare it against the existing Page-Hinkley DTTS on the same benchmark
the threshold-sensitivity study (real_dtts_sweep.py, manuscript Table 4)
already uses, in the same format (commit count, timing error, downstream
fleet metric/FBWT).

KSWIN (Kolmogorov-Smirnov Windowing) is implemented from scratch here
rather than pulling in `river` or `scikit-multiflow` as a new project
dependency -- the algorithm is simple enough (two-sample KS test between a
sliding window's most recent sub-window and a random sample of its older
portion) that a minimal, from-scratch version carries less integration
risk than a new external package, and this project's own convention
(EXPERIMENT_PROCEDURE.md) already prefers from-scratch reimplementation
for exactly this reason. Only `scipy.stats.ks_2samp` is used, and scipy is
already a dependency (preprocess_mimii.py imports scipy.io.wavfile).
"""
import os
import json
import numpy as np
import torch
from scipy.stats import ks_2samp

from real_pipeline_femto import (
    DEVICE, DEVICE_CLASSES, CMAX, C_SHARED, DATA_PATH, N_ROUNDS, LOCAL_STEPS, LR,
    init_backbone, clone_theta, loss_fn, eval_metric, zero_grad, clip_grads,
    mask_grad, sparsity_target, RealSiteStream,
)


class KSWIN:
    """Minimal from-scratch KSWIN: maintains a sliding window of the last
    `window_size` health-indicator readings; on each update, compares the
    most recent `stat_size` readings against a random sample of the same
    size drawn from the rest of the window via a two-sample KS test. A
    p-value at or below `alpha` signals a distribution shift; the window is
    then reset to just the most recent sub-window (river's own convention),
    so the detector treats the point right after a confirmed shift as the
    start of a fresh baseline."""

    def __init__(self, alpha=0.01, window_size=60, stat_size=15, seed=1):
        self.alpha = alpha
        self.window_size = window_size
        self.stat_size = stat_size
        self.window = []
        self.rng = np.random.default_rng(seed)

    def update(self, x):
        self.window.append(float(x))
        if len(self.window) < self.window_size:
            return False
        most_recent = self.window[-self.stat_size:]
        rest = self.window[:-self.stat_size]
        if len(rest) < self.stat_size:
            return False
        idx = self.rng.choice(len(rest), size=self.stat_size, replace=False)
        r_sample = [rest[j] for j in idx]
        _, p = ks_2samp(r_sample, most_recent)
        if p <= self.alpha:
            self.window = list(most_recent)
            return True
        # keep window bounded even without a trigger
        if len(self.window) > self.window_size:
            self.window = self.window[-self.window_size:]
        return False


def run_detector_real(detector_name, detector_factory, seed=1):
    data = np.load(DATA_PATH, allow_pickle=True)
    sites_data = data["sites"]
    n_sensors = sites_data[0]["windows"].shape[2]
    window = sites_data[0]["windows"].shape[1]
    n_sites = len(sites_data)

    torch.manual_seed(seed * 1000)
    theta_global = init_backbone(n_sensors, window, seed=seed * 1000)
    streams = [RealSiteStream(sites_data[i], site_id=i) for i in range(n_sites)]
    thetas = [clone_theta(theta_global) for _ in range(n_sites)]
    detectors = [detector_factory(site_id=i) for i in range(n_sites)]
    global_registry = torch.zeros(C_SHARED, dtype=torch.bool, device=DEVICE)
    commit_count = [0] * n_sites
    task_masks = [dict() for _ in range(n_sites)]
    task_eval_at_commit = [dict() for _ in range(n_sites)]
    true_stage_prev = [0] * n_sites
    commit_events = [[] for _ in range(n_sites)]
    stage_end_round = [dict() for _ in range(n_sites)]

    for r in range(N_ROUNDS):
        rho_global = global_registry.float().mean().item()
        for i, dc in enumerate(DEVICE_CLASSES):
            cmax = CMAX[dc]
            site_free = (~global_registry).clone()
            site_free[cmax:] = False
            theta = thetas[i]
            stream = streams[i]
            theta_before = clone_theta(theta)

            h, true_stage_now = stream.advance_round()
            triggered = detectors[i].update(h)
            old_stage = true_stage_prev[i]
            stage_changed = true_stage_now != old_stage
            true_stage_prev[i] = true_stage_now

            stage_already_measured = old_stage in task_eval_at_commit[i]
            newly_committed = None
            if triggered and site_free.any() and not stage_already_measured:
                s_n = sparsity_target(dc, rho_global)
                free_idx = torch.where(site_free)[0]
                with torch.no_grad():
                    scores = (theta_before["conv_w"][free_idx].abs().sum(dim=(1, 2))
                              + theta_before["fc_w"][free_idx].abs())
                n_commit = max(1, int(np.ceil(s_n * len(free_idx))))
                top = free_idx[torch.argsort(-scores)[:n_commit]]
                newly_committed = torch.zeros(C_SHARED, dtype=torch.bool, device=DEVICE)
                newly_committed[top] = True
                prev = task_masks[i].get(old_stage, torch.zeros(C_SHARED, dtype=torch.bool, device=DEVICE))
                task_masks[i][old_stage] = prev | newly_committed
                commit_events[i].append((r, old_stage))

            if stage_changed:
                stage_end_round[i][old_stage] = r
                if old_stage not in task_eval_at_commit[i]:
                    Xc, yc = stream.eval_on_stage(old_stage)
                    m = task_masks[i].get(old_stage)
                    task_eval_at_commit[i][old_stage] = eval_metric(theta_before, Xc.to(DEVICE), yc.to(DEVICE), m)

            if newly_committed is not None:
                site_free = site_free & ~newly_committed
                top = torch.where(newly_committed)[0]
                with torch.no_grad():
                    theta_global["conv_w"][top] = theta_before["conv_w"][top]
                    theta_global["conv_b"][top] = theta_before["conv_b"][top]
                    theta_global["fc_w"][top] = theta_before["fc_w"][top]
                    global_registry[top] = True
                commit_count[i] += 1

            site_free_t = site_free.float()
            for _ in range(LOCAL_STEPS):
                Xb, yb = stream.minibatch()
                zero_grad(theta)
                l = loss_fn(theta, Xb.to(DEVICE), yb.to(DEVICE))
                l.backward()
                clip_grads(theta)
                mask_grad(theta, site_free_t)
                with torch.no_grad():
                    for k in ("conv_w", "conv_b", "fc_w"):
                        theta[k] -= LR * theta[k].grad
                zero_grad(theta)

        with torch.no_grad():
            for i in range(n_sites):
                thetas[i]["conv_w"][global_registry] = theta_global["conv_w"][global_registry]
                thetas[i]["conv_b"][global_registry] = theta_global["conv_b"][global_registry]
                thetas[i]["fc_w"][global_registry] = theta_global["fc_w"][global_registry]

    timing_errors = []
    fps_all, fbwt_all = [], []
    protected_fbwt, missed_fbwt = [], []
    for i in range(n_sites):
        for commit_round, old_stage in commit_events[i]:
            if old_stage in stage_end_round[i]:
                timing_errors.append(commit_round - stage_end_round[i][old_stage])
        fps_vals, fbwt_vals = [], []
        for stage_idx, at_commit in task_eval_at_commit[i].items():
            Xc, yc = streams[i].eval_on_stage(stage_idx)
            m = task_masks[i].get(stage_idx)
            final = eval_metric(thetas[i], Xc.to(DEVICE), yc.to(DEVICE), m)
            fps_vals.append(final)
            fbwt = final - at_commit
            fbwt_vals.append(fbwt)
            protected = bool(m is not None and m.any().item())
            (protected_fbwt if protected else missed_fbwt).append(fbwt)
        if fps_vals:
            fps_all.append(np.mean(fps_vals))
            fbwt_all.append(np.mean(fbwt_vals))

    result = {
        "detector": detector_name,
        "total_commits": int(sum(commit_count)),
        "mean_tasks_committed": float(np.mean(commit_count)),
        "mean_abs_timing_error": float(np.mean(np.abs(timing_errors))) if timing_errors else None,
        "n_timing_samples": len(timing_errors),
        "fleet_fps": float(np.mean(fps_all)) if fps_all else None,
        "fleet_fbwt": float(np.mean(fbwt_all)) if fbwt_all else None,
        "protected_n": len(protected_fbwt),
        "protected_mean_fbwt": float(np.mean(protected_fbwt)) if protected_fbwt else None,
        "missed_n": len(missed_fbwt),
        "missed_mean_fbwt": float(np.mean(missed_fbwt)) if missed_fbwt else None,
    }
    print(f"{detector_name}: {result}")
    return result


if __name__ == "__main__":
    from real_pipeline_femto import PageHinkley, LAM_PH, DELTA_PH

    results = {}
    results["page-hinkley"] = run_detector_real(
        "page-hinkley", lambda site_id: PageHinkley(delta=DELTA_PH, lam=LAM_PH))
    # window_size must be well under N_ROUNDS=20 (one detector.update() call
    # per site per round): a window_size of 60, a reasonable default for a
    # long streaming setting, would never fill within this pilot's 20-round
    # horizon and KSWIN would trivially never trigger -- caught before
    # running, not after a suspicious zero-commit result (EXPERIMENT_
    # PROCEDURE.md's own "check this list first" convention applied
    # proactively). window_size=10, stat_size=3.
    #
    # A second, independent surprise surfaced even after fixing that: at
    # stat_size=3 the two-sample KS test's minimum achievable p-value is
    # exactly 0.1 (confirmed directly: ks_2samp([0,0,0],[1,1,1]) already
    # returns p=0.0999...), a property of the exact small-sample KS null
    # distribution, not a bug. alpha=0.01 and alpha=0.05 are BELOW that
    # floor and can therefore never trigger, regardless of how sharply the
    # two windows differ -- KSWIN's own textbook-default alpha values are
    # simply incompatible with this benchmark's tiny (20-round) per-site
    # horizon and its correspondingly small feasible stat_size. Recorded as
    # EXPERIMENT_PROCEDURE.md bug/finding #16. We therefore run and report
    # BOTH: the naive default (alpha=0.01, which never fires, a genuine
    # real-data result in its own right) and a calibrated operating point
    # (alpha=0.10, the minimum value that can ever fire at this
    # window/stat_size) side by side.
    results["kswin_alpha0.01_naive"] = run_detector_real(
        "kswin_alpha0.01_naive", lambda site_id: KSWIN(alpha=0.01, window_size=10, stat_size=3, seed=1000 + site_id))
    results["kswin_alpha0.10_calibrated"] = run_detector_real(
        "kswin_alpha0.10_calibrated", lambda site_id: KSWIN(alpha=0.10, window_size=10, stat_size=3, seed=1000 + site_id))

    out_path = os.path.join(os.path.dirname(__file__), "alt_detector_real_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nWrote {out_path}")
