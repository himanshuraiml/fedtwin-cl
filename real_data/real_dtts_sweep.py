"""
Real-data DTTS lambda_PH sensitivity sweep (manuscript Table 5 / Figure 8),
on real Fed-Twin-FEMTO data -- the same benchmark the synthetic version
(code/spike_extended.py::dtts_sweep) used, for a direct comparison.

Self-contained loop, reusing validated primitives imported from
real_pipeline_femto.py, following the same convention spike_extended.py
used relative to fedtwin_harness.py: don't touch the validated main run_*
path, duplicate the round loop here with extra bookkeeping instead.
"""
import os
import json
import numpy as np
import torch

from real_pipeline_femto import (
    DEVICE, DEVICE_CLASSES, CMAX, C_SHARED, DATA_PATH, N_ROUNDS, LOCAL_STEPS, LR,
    init_backbone, clone_theta, forward, loss_fn, eval_metric, zero_grad, clip_grads,
    mask_grad, sparsity_target, PageHinkley, RealSiteStream,
)

LAMBDAS = (0.05, 0.08, 0.12, 0.15, 0.20, 0.30, 0.50, 0.70)


def dtts_sweep_real(lambdas=LAMBDAS, seed=1):
    data = np.load(DATA_PATH, allow_pickle=True)
    sites_data = data["sites"]
    n_sensors = sites_data[0]["windows"].shape[2]
    window = sites_data[0]["windows"].shape[1]
    n_sites = len(sites_data)

    results = {}
    for lam in lambdas:
        torch.manual_seed(seed * 1000)
        theta_global = init_backbone(n_sensors, window, seed=seed * 1000)
        streams = [RealSiteStream(sites_data[i], site_id=i) for i in range(n_sites)]
        thetas = [clone_theta(theta_global) for _ in range(n_sites)]
        phs = [PageHinkley(lam=lam) for _ in range(n_sites)]
        global_registry = torch.zeros(C_SHARED, dtype=torch.bool, device=DEVICE)
        commit_count = [0] * n_sites
        task_masks = [dict() for _ in range(n_sites)]
        task_eval_at_commit = [dict() for _ in range(n_sites)]
        true_stage_prev = [0] * n_sites
        commit_events = [[] for _ in range(n_sites)]     # (round, old_stage)
        stage_end_round = [dict() for _ in range(n_sites)]  # old_stage -> round it ended

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
                triggered = phs[i].update(h)
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
                fbwt_vals.append(final - at_commit)
            if fps_vals:
                fps_all.append(np.mean(fps_vals))
                fbwt_all.append(np.mean(fbwt_vals))

        results[lam] = {
            "mean_tasks_committed": float(np.mean(commit_count)),
            "mean_abs_timing_error": float(np.mean(np.abs(timing_errors))) if timing_errors else None,
            "n_timing_samples": len(timing_errors),
            "fleet_fps": float(np.mean(fps_all)) if fps_all else None,
            "fleet_fbwt": float(np.mean(fbwt_all)) if fbwt_all else None,
        }
        print(f"lambda_PH={lam}: {results[lam]}")

    return results


if __name__ == "__main__":
    res = dtts_sweep_real()
    out_path = os.path.join(os.path.dirname(__file__), "dtts_sweep_real_results.json")
    with open(out_path, "w") as f:
        json.dump(res, f, indent=2)
    print(f"\nWrote {out_path}")
