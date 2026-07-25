"""
Real-data CTFR confirmed-precision / lead-time analysis (manuscript Table 6
/ Figure 9), on all three real benchmarks -- the real-data analogue of
code/spike_extended.py::ctfr_leadtime.

Real data has no distinct "fault type" taxonomy (unlike the synthetic
generator's fault library); the ground-truth stage index (0/1/2) is used
as the confirmation key instead: a CTFR flag is "confirmed" if the flagged
site later commits its OWN mask for the SAME stage index the matched
signature came from, exactly mirroring the synthetic version's "same fault
type" check with "same stage index" as the real-data equivalent.

Self-contained loop per benchmark, reusing validated primitives imported
from each real_pipeline_*.py, following the same convention as the
synthetic sweep: don't touch the validated main run_* path.
"""
import os
import sys
import json
import importlib
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))


def _run_one(module_name, benchmark_tag, seed=1):
    m = importlib.import_module(module_name)
    DEVICE, DEVICE_CLASSES, CMAX, C_SHARED = m.DEVICE, m.DEVICE_CLASSES, m.CMAX, m.C_SHARED
    N_ROUNDS, LOCAL_STEPS, LR = m.N_ROUNDS, m.LOCAL_STEPS, m.LR
    TAU_CTFR = m.TAU_CTFR
    data = np.load(m.DATA_PATH, allow_pickle=True)
    sites_data = data["sites"]
    n_sites = len(sites_data)
    is_mimii = "mimii" in module_name

    if is_mimii:
        n_features = sites_data[0]["round_windows"][1].shape[-1]
        n_frames = sites_data[0]["round_windows"][1].shape[1]
        streams = [m.MimiiSiteStream(sites_data[i], site_id=i) for i in range(n_sites)]
        theta_global = m.init_backbone(n_features, n_frames, seed=seed * 1000)
    else:
        n_sensors = sites_data[0]["windows"].shape[2]
        window = sites_data[0]["windows"].shape[1]
        streams = [m.RealSiteStream(sites_data[i], site_id=i) for i in range(n_sites)]
        theta_global = m.init_backbone(n_sensors, window, seed=seed * 1000)

    torch.manual_seed(seed * 1000)
    thetas = [m.clone_theta(theta_global) for _ in range(n_sites)]
    phs = [m.PageHinkley() for _ in range(n_sites)]
    global_registry = torch.zeros(C_SHARED, dtype=torch.bool, device=DEVICE)
    signature_registry = []          # (site_id, stage_idx, sig)
    own_commit_log = [[] for _ in range(n_sites)]   # (round, stage_idx)
    ctfr_flags = []
    h_history = [[] for _ in range(n_sites)]
    task_masks = [dict() for _ in range(n_sites)]
    task_eval_at_commit = [dict() for _ in range(n_sites)]
    true_stage_prev = [0] * n_sites
    max_stage_reached = [0] * n_sites

    for r in range(N_ROUNDS):
        rho_global = global_registry.float().mean().item()
        for i, dc in enumerate(DEVICE_CLASSES):
            cmax = CMAX[dc]
            site_free = (~global_registry).clone()
            site_free[cmax:] = False
            theta = thetas[i]
            stream = streams[i]
            theta_before = m.clone_theta(theta)

            h, true_stage_now = stream.advance_round()
            h_history[i].append(h)
            triggered = phs[i].update(h)
            old_stage = true_stage_prev[i]
            stage_changed = true_stage_now != old_stage
            true_stage_prev[i] = true_stage_now
            max_stage_reached[i] = max(max_stage_reached[i], true_stage_now)

            stage_already_measured = old_stage in task_eval_at_commit[i]
            newly_committed = None
            if triggered and site_free.any() and not stage_already_measured:
                s_n = m.sparsity_target(dc, rho_global)
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
                m_ = task_masks[i].get(old_stage)
                val = m.eval_metric(theta_before, Xc.to(DEVICE), yc.to(DEVICE), m_)
                if val is not None:
                    task_eval_at_commit[i][old_stage] = val

            if newly_committed is not None:
                site_free = site_free & ~newly_committed
                top = torch.where(newly_committed)[0]
                with torch.no_grad():
                    theta_global["conv_w"][top] = theta_before["conv_w"][top]
                    theta_global["conv_b"][top] = theta_before["conv_b"][top]
                    theta_global["fc_w"][top] = theta_before["fc_w"][top]
                    global_registry[top] = True
                own_commit_log[i].append((r, old_stage))
                sig = m.signature(h_history[i][-4:])
                signature_registry.append((i, old_stage, sig))

            site_free_t = site_free.float()
            for _ in range(LOCAL_STEPS):
                Xb, yb = stream.minibatch()
                if is_mimii and len(yb) == 0:
                    break
                m.zero_grad(theta)
                l = m.loss_fn(theta, Xb.to(DEVICE), yb.to(DEVICE))
                l.backward()
                m.clip_grads(theta)
                m.mask_grad(theta, site_free_t)
                with torch.no_grad():
                    for k in ("conv_w", "conv_b", "fc_w"):
                        theta[k] -= LR * theta[k].grad
                m.zero_grad(theta)

            if len(h_history[i]) >= 3:
                phi_now = m.signature(h_history[i][-4:])
                best_sim, best_match = 0.0, None
                for (sid, stage_idx, sig) in signature_registry:
                    if sid == i:
                        continue
                    sim = m.cos_sim(phi_now, sig)
                    if sim > best_sim:
                        best_sim, best_match = sim, (sid, stage_idx)
                if best_sim > TAU_CTFR:
                    ctfr_flags.append({"site": i, "round": r, "sim": best_sim, "matched_stage": best_match[1]})

        with torch.no_grad():
            for i in range(n_sites):
                thetas[i]["conv_w"][global_registry] = theta_global["conv_w"][global_registry]
                thetas[i]["conv_b"][global_registry] = theta_global["conv_b"][global_registry]
                thetas[i]["fc_w"][global_registry] = theta_global["fc_w"][global_registry]

    lead_times, n_confirmed = [], 0
    for f in ctfr_flags:
        i, r, matched_stage = f["site"], f["round"], f["matched_stage"]
        later = [cr for (cr, st) in own_commit_log[i] if cr > r and st == matched_stage]
        if later:
            n_confirmed += 1
            lead_times.append(min(later) - r)

    n_raised = len(ctfr_flags)
    result = {
        "n_flags_raised": n_raised,
        "n_confirmed": n_confirmed,
        "precision": (n_confirmed / n_raised) if n_raised else None,
        "mean_lead_time": float(np.mean(lead_times)) if lead_times else None,
        "std_lead_time": float(np.std(lead_times)) if lead_times else None,
        "lead_times": lead_times,
    }
    print(f"{benchmark_tag}: {n_raised} flags, {n_confirmed} confirmed, "
          f"precision={result['precision']}, mean_lead_time={result['mean_lead_time']}")
    return result


if __name__ == "__main__":
    out = {}
    for module_name, tag in [
        ("real_pipeline_cmapss", "fed-twin-cmapss-real"),
        ("real_pipeline_femto", "fed-twin-femto-real"),
        ("real_pipeline_mimii", "fed-twin-mimii-real"),
    ]:
        out[tag] = _run_one(module_name, tag)

    out_path = os.path.join(os.path.dirname(__file__), "ctfr_leadtime_real_results.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nWrote {out_path}")
