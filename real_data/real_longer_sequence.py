"""
Longer, finer-grained progressive drift sequence, real Fed-Twin-FEMTO
(reviewer gap item 6): 8 ground-truth stages instead of 3
(femto_sites_longseq.npz, preprocess_femto_longseq.py) and 50 communication
rounds instead of 20, run for fedavg, fedcat-external (oracle), and
fedtwin-cl (DTTS). Reuses real_pipeline_femto.py's validated primitives
(backbone, mask/commit logic, DTTS) unchanged; only N_ROUNDS, N_STAGES, and
the data file differ from the main pilot.

Tracks, per round: DTTS commit events, global registry saturation
(rho_global), and fleet current-task metric -- the three things Section
5.3's fleet-capacity claim needs tested under real, not synthetic,
pressure.
"""
import os
import json
import numpy as np
import torch

from real_pipeline_femto import (
    DEVICE, DEVICE_CLASSES, CMAX, C_SHARED, N_ROUNDS as N_ROUNDS_ORIG, LOCAL_STEPS, LR,
    init_backbone, clone_theta, loss_fn, eval_metric, zero_grad, clip_grads,
    mask_grad, sparsity_target, PageHinkley, RealSiteStream, LAM_PH, DELTA_PH,
)

DATA_PATH_LONG = os.path.join(os.path.dirname(__file__), "processed", "femto_sites_longseq.npz")
N_ROUNDS_LONG = 50
METHODS = ["fedavg", "fedcat-external", "fedtwin-cl"]


def run_long(methods=METHODS, n_rounds=N_ROUNDS_LONG, seed=1):
    data = np.load(DATA_PATH_LONG, allow_pickle=True)
    sites_data = data["sites"]
    n_sensors = sites_data[0]["windows"].shape[2]
    window = sites_data[0]["windows"].shape[1]
    n_sites = len(sites_data)
    print(f"{n_sites} real sites (long/fine variant), n_sensors={n_sensors}, window={window}, "
          f"n_rounds={n_rounds}, n_stages={len(set(np.concatenate([s['stage'] for s in sites_data]).tolist()))}")

    all_out = {}
    for method in methods:
        torch.manual_seed(seed * 1000 + METHODS.index(method))
        theta_global = init_backbone(n_sensors, window, seed=seed * 1000 + METHODS.index(method))
        streams = [RealSiteStream(sites_data[i], n_rounds=n_rounds, site_id=i) for i in range(n_sites)]
        thetas = [clone_theta(theta_global) for _ in range(n_sites)]
        phs = [PageHinkley(delta=DELTA_PH, lam=LAM_PH) for _ in range(n_sites)]
        global_registry = torch.zeros(C_SHARED, dtype=torch.bool, device=DEVICE)
        commit_count = [0] * n_sites
        task_masks = [dict() for _ in range(n_sites)]
        task_eval_at_commit = [dict() for _ in range(n_sites)]
        true_stage_prev = [0] * n_sites
        max_stage_reached = [0] * n_sites

        use_masking = method in ("fedcat-external", "fedtwin-cl")
        boundary_source = "oracle" if method == "fedcat-external" else "dtts"

        round_trace = []  # per-round: rho_global, total commits so far, mean fleet current-task metric
        for r in range(n_rounds):
            rho_global = global_registry.float().mean().item()
            for i, dc in enumerate(DEVICE_CLASSES):
                cmax = CMAX[dc]
                site_free = (~global_registry).clone()
                site_free[cmax:] = False
                theta = thetas[i]
                stream = streams[i]
                theta_before = clone_theta(theta)

                h, true_stage_now = stream.advance_round()
                if boundary_source == "dtts":
                    triggered = phs[i].update(h)
                else:
                    triggered = true_stage_now > max_stage_reached[i]
                old_stage = true_stage_prev[i]
                stage_changed = true_stage_now != old_stage
                true_stage_prev[i] = true_stage_now
                max_stage_reached[i] = max(max_stage_reached[i], true_stage_now)

                stage_already_measured = old_stage in task_eval_at_commit[i]
                newly_committed = None
                if use_masking and triggered and site_free.any() and not stage_already_measured:
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

                if stage_changed and old_stage not in task_eval_at_commit[i]:
                    Xc, yc = stream.eval_on_stage(old_stage)
                    m = task_masks[i].get(old_stage) if use_masking else None
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
                    if use_masking:
                        mask_grad(theta, site_free_t)
                    with torch.no_grad():
                        update_keys = ("conv_w", "conv_b", "fc_w") if use_masking else theta.keys()
                        for k in update_keys:
                            theta[k] -= LR * theta[k].grad
                    zero_grad(theta)

            if use_masking:
                with torch.no_grad():
                    for i in range(n_sites):
                        thetas[i]["conv_w"][global_registry] = theta_global["conv_w"][global_registry]
                        thetas[i]["conv_b"][global_registry] = theta_global["conv_b"][global_registry]
                        thetas[i]["fc_w"][global_registry] = theta_global["fc_w"][global_registry]
            else:
                with torch.no_grad():
                    for k in theta_global:
                        theta_global[k] = sum(thetas[i][k] for i in range(n_sites)) / n_sites
                    for i in range(n_sites):
                        thetas[i] = clone_theta(theta_global)

            cur_metrics = []
            for i, dc in enumerate(DEVICE_CLASSES):
                Xc, yc = streams[i].eval_on_stage(true_stage_prev[i])
                cur_metrics.append(eval_metric(thetas[i], Xc.to(DEVICE), yc.to(DEVICE)))
            round_trace.append({
                "round": r + 1,
                "rho_global": global_registry.float().mean().item(),
                "total_commits_so_far": int(sum(commit_count)),
                "mean_fleet_current_task_metric": float(np.mean(cur_metrics)),
            })

        fps_vals_all, fbwt_vals_all = [], []
        for i, dc in enumerate(DEVICE_CLASSES):
            for stage_idx, at_commit in task_eval_at_commit[i].items():
                Xc, yc = streams[i].eval_on_stage(stage_idx)
                m = task_masks[i].get(stage_idx) if use_masking else None
                final = eval_metric(thetas[i], Xc.to(DEVICE), yc.to(DEVICE), m)
                fps_vals_all.append(final)
                fbwt_vals_all.append(final - at_commit)

        all_out[method] = {
            "round_trace": round_trace,
            "final_rho_global": global_registry.float().mean().item(),
            "total_commits": int(sum(commit_count)),
            "mean_commits_per_site": float(np.mean(commit_count)),
            "fleet_fps_final": float(np.mean(fps_vals_all)) if fps_vals_all else None,
            "fleet_fbwt_final": float(np.mean(fbwt_vals_all)) if fbwt_vals_all else None,
            "n_tasks_evaluated": len(fps_vals_all),
        }
        print(f"method={method}: total_commits={all_out[method]['total_commits']} "
              f"final_rho_global={all_out[method]['final_rho_global']:.3f} "
              f"fleet_fps={all_out[method]['fleet_fps_final']:.4f} "
              f"fleet_fbwt={all_out[method]['fleet_fbwt_final']:.4f}")

    return all_out


if __name__ == "__main__":
    out = run_long()
    out_path = os.path.join(os.path.dirname(__file__), "longer_sequence_real_results.json")
    with open(out_path, "w") as f:
        json.dump(out, f)
    print(f"\nWrote {out_path}")
