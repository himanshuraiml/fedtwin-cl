"""
Real-data heterogeneity/gamma sensitivity sweep (manuscript Figure 10), on
real Fed-Twin-FEMTO data -- the real-data analogue of
code/spike_extended.py::gamma_sweep.

The synthetic version's main run used a shared registry width with
generous headroom, which the manuscript's own honest diagnosis (Section
7.7) already identified as the reason global capacity pressure
(rho_global) never grew large enough for gamma's multiplicative term in
Eq. 2 to matter. This real sweep instead uses a deliberately small,
capacity-constrained registry width (C_SHARED / 8, with CMAX values scaled
by the same factor to preserve the device-class heterogeneity ratios),
exactly the fix the synthetic diagnosis called for, applied here directly
on real data rather than repeating the same non-diagnostic configuration.
"""
import os
import json
import numpy as np
import torch

from real_pipeline_femto import (
    DEVICE, DEVICE_CLASSES, DATA_PATH, N_ROUNDS, LOCAL_STEPS, LR, S0,
    init_backbone, clone_theta, loss_fn, eval_metric, zero_grad, clip_grads,
    mask_grad, PageHinkley, RealSiteStream,
)

SCALE = 0.125
C_SHARED_SMALL = 64
CMAX_SMALL = {"remote-asset": 16, "sensor-edge": 32, "line-gateway": 64, "plant-fog": 48}
GAMMAS = (0.0, 0.2, 0.4, 0.6, 0.8)
LAM_PH = 0.15


def gamma_sweep_real(gammas=GAMMAS, seed=1):
    data = np.load(DATA_PATH, allow_pickle=True)
    sites_data = data["sites"]
    n_sensors = sites_data[0]["windows"].shape[2]
    window = sites_data[0]["windows"].shape[1]
    n_sites = len(sites_data)

    results = {g: {dc: [] for dc in set(DEVICE_CLASSES)} for g in gammas}
    rho_trace = {g: [] for g in gammas}

    for gamma in gammas:
        torch.manual_seed(seed * 1000)
        theta_global = init_backbone(n_sensors, window, width=C_SHARED_SMALL, seed=seed * 1000)
        streams = [RealSiteStream(sites_data[i], site_id=i) for i in range(n_sites)]
        thetas = [clone_theta(theta_global) for _ in range(n_sites)]
        phs = [PageHinkley(lam=LAM_PH) for _ in range(n_sites)]
        global_registry = torch.zeros(C_SHARED_SMALL, dtype=torch.bool, device=DEVICE)
        task_masks = [dict() for _ in range(n_sites)]
        task_eval_at_commit = [dict() for _ in range(n_sites)]
        true_stage_prev = [0] * n_sites

        for r in range(N_ROUNDS):
            rho_global = global_registry.float().mean().item()
            rho_trace[gamma].append(rho_global)
            for i, dc in enumerate(DEVICE_CLASSES):
                cmax = CMAX_SMALL[dc]
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
                    # sparsity_target inlined here with CMAX_SMALL instead of
                    # the module's CMAX, since this sweep runs at a scaled-down
                    # registry width (see module docstring).
                    cmax_mean = np.mean(list(CMAX_SMALL.values()))
                    s_n = float(np.clip(S0 * (CMAX_SMALL[dc] / cmax_mean) * (1 + gamma * rho_global), 0.25, 0.70))
                    free_idx = torch.where(site_free)[0]
                    with torch.no_grad():
                        scores = (theta_before["conv_w"][free_idx].abs().sum(dim=(1, 2))
                                  + theta_before["fc_w"][free_idx].abs())
                    n_commit = max(1, int(np.ceil(s_n * len(free_idx))))
                    top = free_idx[torch.argsort(-scores)[:n_commit]]
                    newly_committed = torch.zeros(C_SHARED_SMALL, dtype=torch.bool, device=DEVICE)
                    newly_committed[top] = True
                    prev = task_masks[i].get(old_stage, torch.zeros(C_SHARED_SMALL, dtype=torch.bool, device=DEVICE))
                    task_masks[i][old_stage] = prev | newly_committed

                if stage_changed and old_stage not in task_eval_at_commit[i]:
                    Xc, yc = stream.eval_on_stage(old_stage)
                    m_ = task_masks[i].get(old_stage)
                    task_eval_at_commit[i][old_stage] = eval_metric(theta_before, Xc.to(DEVICE), yc.to(DEVICE), m_)

                if newly_committed is not None:
                    site_free = site_free & ~newly_committed
                    top = torch.where(newly_committed)[0]
                    with torch.no_grad():
                        theta_global["conv_w"][top] = theta_before["conv_w"][top]
                        theta_global["conv_b"][top] = theta_before["conv_b"][top]
                        theta_global["fc_w"][top] = theta_before["fc_w"][top]
                        global_registry[top] = True

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

        for i, dc in enumerate(DEVICE_CLASSES):
            fps_vals = []
            for stage_idx in task_eval_at_commit[i]:
                Xc, yc = streams[i].eval_on_stage(stage_idx)
                m_ = task_masks[i].get(stage_idx)
                fps_vals.append(eval_metric(thetas[i], Xc.to(DEVICE), yc.to(DEVICE), m_))
            if fps_vals:
                results[gamma][dc].append(np.mean(fps_vals))

        print(f"gamma={gamma}: final rho_global={global_registry.float().mean().item():.3f}, "
              f"mean rho over run={np.mean(rho_trace[gamma]):.3f}")

    out = {g: {dc: (float(np.mean(v)) if v else None) for dc, v in dcs.items()} for g, dcs in results.items()}
    for g, dcs in out.items():
        print(f"gamma={g}: {dcs}")
    return out, {g: float(np.mean(v)) for g, v in rho_trace.items()}


if __name__ == "__main__":
    res, rho = gamma_sweep_real()
    out_path = os.path.join(os.path.dirname(__file__), "gamma_sweep_real_results.json")
    with open(out_path, "w") as f:
        json.dump({"fps_by_gamma_device": res, "mean_rho_global_by_gamma": rho}, f, indent=2)
    print(f"\nWrote {out_path}")
