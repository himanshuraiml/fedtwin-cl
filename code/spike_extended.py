"""
Extended spike analysis: DTTS lambda_PH sweep (Table 5), CTFR lead-time
analysis (Table 6), and heterogeneity/gamma sweep (Fig 10).

Reuses the validated primitives from fedtwin_harness.py; each sweep is a
self-contained loop (like zero_forgetting_check / centralized_upper_bound)
rather than a modification of the core run_benchmark, to avoid risking the
already-validated main path.
"""
import numpy as np
from fedtwin_harness import (
    DEVICE_CLASSES, CMAX, C_SHARED, INPUT_DIM,
    init_backbone, forward, backward, mask_grads, clip_grads,
    make_fault_library, SiteStream, PageHinkley, sparsity_target,
    signature, cos_sim, _eval_metric, METHODS,
)

RNG_SEED = 1


# ─────────────────────────────────────────────────────────────────────────
# Table 5: DTTS lambda_PH sensitivity, with REAL commit-timing error
# ─────────────────────────────────────────────────────────────────────────
def dtts_sweep(benchmark_name="fed-twin-femto", mode="regression", stage_len=5,
               n_rounds=20, local_steps=40, lr=0.01, s0=0.45, gamma=0.6,
               lambdas=(0.3, 0.5, 0.7, 1.0, 1.5), rng_seed=RNG_SEED):
    results = {}
    for lam in lambdas:
        fault_lib = make_fault_library(np.random.default_rng(rng_seed))
        tasks_committed, timing_errors, fps_all, fbwt_all = [], [], [], []

        for site_id, dc in enumerate(DEVICE_CLASSES):
            rng = np.random.default_rng(rng_seed * 1000 + 3)  # fixed init, isolate lambda's effect
            site_rng = np.random.default_rng(rng_seed * 7919 + site_id)
            stream = SiteStream(mode, fault_lib, stage_len, site_rng, site_id=site_id)
            theta = init_backbone(rng)
            theta_global = init_backbone(rng)
            global_registry = np.zeros(C_SHARED, dtype=bool)
            ph = PageHinkley(delta=0.01, lam=lam)
            true_stage_prev = 0
            task_masks, task_eval_at_commit, stage_end_round = {}, {}, {}
            commit_events = []  # (round, old_stage)
            n_commits = 0

            for r in range(n_rounds):
                dcmax = CMAX[dc]
                site_free_mask = ~global_registry.copy()
                site_free_mask[dcmax:] = False
                theta_before = {k: v.copy() for k, v in theta.items()}
                h, true_stage_now, _ = stream.advance_round()
                trig, _ = ph.update(h)
                old_stage = true_stage_prev
                stage_changed = true_stage_now != old_stage
                true_stage_prev = true_stage_now

                newly_committed = None
                if trig and site_free_mask.any():
                    s_n = sparsity_target(dc, global_registry.mean(), s0=s0, gamma=gamma)
                    free_idx = np.where(site_free_mask)[0]
                    scores = np.abs(theta_before["W1"][:, free_idx]).sum(0) + np.abs(theta_before["W2"][free_idx])
                    n_commit = max(1, int(np.ceil(s_n * len(free_idx))))
                    top = free_idx[np.argsort(-scores)[:n_commit]]
                    newly_committed = np.zeros(C_SHARED, dtype=bool)
                    newly_committed[top] = True
                    task_masks[old_stage] = task_masks.get(old_stage, np.zeros(C_SHARED, dtype=bool)) | newly_committed
                    commit_events.append((r, old_stage))
                    n_commits += 1

                if stage_changed:
                    stage_end_round[old_stage] = r
                    if old_stage not in task_eval_at_commit:
                        Xc, yc = stream.eval_on_stage(old_stage, n=200)
                        m = task_masks.get(old_stage)
                        task_eval_at_commit[old_stage] = _eval_metric(theta_before, Xc, yc, mode, task_mask=m)

                if newly_committed is not None:
                    site_free_mask = site_free_mask & ~newly_committed
                    top = np.where(newly_committed)[0]
                    theta_global["W1"][:, top] = theta_before["W1"][:, top]
                    theta_global["b1"][top] = theta_before["b1"][top]
                    theta_global["W2"][top] = theta_before["W2"][top]
                    global_registry |= newly_committed

                for _ in range(local_steps):
                    X, y = stream.minibatch(n=32)
                    _, cache = forward(theta, X, mode)
                    g = clip_grads(backward(theta, cache, y, mode))
                    g = mask_grads(g, site_free_mask)
                    for k in ("W1", "b1", "W2"):
                        theta[k] = theta[k] - lr * g[k]

                theta["W1"][:, global_registry] = theta_global["W1"][:, global_registry]
                theta["b1"][global_registry] = theta_global["b1"][global_registry]
                theta["W2"][global_registry] = theta_global["W2"][global_registry]

            # commit-point timing error: for each commit tagged to old_stage,
            # compare to the ground-truth round that stage actually ended
            # (only defined if that boundary occurred within the run).
            for commit_round, old_stage in commit_events:
                if old_stage in stage_end_round:
                    timing_errors.append(commit_round - stage_end_round[old_stage])

            tasks_committed.append(n_commits)

            fbwt_vals, fps_vals = [], []
            for stage_idx, at_commit in task_eval_at_commit.items():
                Xf, yf = stream.eval_on_stage(stage_idx, n=200)
                m = task_masks.get(stage_idx)
                final = _eval_metric(theta, Xf, yf, mode, task_mask=m)
                fps_vals.append(final)
                fbwt_vals.append(final - at_commit)
            if fps_vals:
                fps_all.append(np.mean(fps_vals))
                fbwt_all.append(np.mean(fbwt_vals))

        results[lam] = {
            "mean_tasks_committed": float(np.mean(tasks_committed)),
            "mean_abs_timing_error": float(np.mean(np.abs(timing_errors))) if timing_errors else None,
            "n_timing_samples": len(timing_errors),
            "fleet_fps": float(np.mean(fps_all)),
            "fleet_fbwt": float(np.mean(fbwt_all)),
        }
    return results


# ─────────────────────────────────────────────────────────────────────────
# Table 6: CTFR lead-time, with real confirmation against each site's own
# later commit history
# ─────────────────────────────────────────────────────────────────────────
def ctfr_leadtime(benchmark_name, mode, stage_len, recal_sites=None,
                   n_rounds=20, local_steps=40, lr=0.01, s0=0.45, gamma=0.6,
                   lam_ph=0.7, tau_ctfr=0.9, rng_seed=RNG_SEED):
    fault_lib = make_fault_library(np.random.default_rng(rng_seed))
    n_sites = len(DEVICE_CLASSES)
    recal_sites = recal_sites or []

    rng = np.random.default_rng(rng_seed * 1000 + METHODS.index("fedtwin-cl"))
    streams, thetas, phs, h_history = [], [], [], []
    theta_global = init_backbone(rng)
    global_registry = np.zeros(C_SHARED, dtype=bool)
    signature_registry = []
    own_commit_log = [[] for _ in range(n_sites)]  # [(round, fault_type_id), ...]
    ctfr_flags = []

    for i, dc in enumerate(DEVICE_CLASSES):
        site_rng = np.random.default_rng(rng_seed * 7919 + i)
        recal = 10 if i in recal_sites else None
        streams.append(SiteStream(mode, fault_lib, stage_len, site_rng, recal_step=recal, site_id=i))
        thetas.append({k: v.copy() for k, v in theta_global.items()})
        phs.append(PageHinkley(delta=0.01, lam=lam_ph))
        h_history.append([])

    true_stage_prev = [0] * n_sites
    for r in range(n_rounds):
        rho_global = global_registry.mean()
        uploads_masks = []
        for i, dc in enumerate(DEVICE_CLASSES):
            cmax = CMAX[dc]
            site_free_mask = ~global_registry.copy()
            site_free_mask[cmax:] = False
            theta = thetas[i]
            stream = streams[i]
            theta_before = {k: v.copy() for k, v in theta.items()}
            h, true_stage_now, fault_type_id = stream.advance_round()
            h_history[i].append(h)
            trig, _ = phs[i].update(h)
            old_stage = true_stage_prev[i]
            true_stage_prev[i] = true_stage_now

            newly_committed = None
            if trig and site_free_mask.any():
                s_n = sparsity_target(dc, rho_global, s0=s0, gamma=gamma)
                free_idx = np.where(site_free_mask)[0]
                scores = np.abs(theta_before["W1"][:, free_idx]).sum(0) + np.abs(theta_before["W2"][free_idx])
                n_commit = max(1, int(np.ceil(s_n * len(free_idx))))
                top = free_idx[np.argsort(-scores)[:n_commit]]
                newly_committed = np.zeros(C_SHARED, dtype=bool)
                newly_committed[top] = True

            if newly_committed is not None:
                site_free_mask = site_free_mask & ~newly_committed
                top = np.where(newly_committed)[0]
                theta_global["W1"][:, top] = theta_before["W1"][:, top]
                theta_global["b1"][top] = theta_before["b1"][top]
                theta_global["W2"][top] = theta_before["W2"][top]
                global_registry |= newly_committed
                old_fault_id = int(stream.stage_order[old_stage])
                own_commit_log[i].append((r, old_fault_id))
                sig = signature(h_history[i][-4:])
                signature_registry.append((i, old_fault_id, sig))

            for _ in range(local_steps):
                X, y = stream.minibatch(n=32)
                _, cache = forward(theta, X, mode)
                g = clip_grads(backward(theta, cache, y, mode))
                g = mask_grads(g, site_free_mask)
                for k in ("W1", "b1", "W2"):
                    theta[k] = theta[k] - lr * g[k]

            if len(h_history[i]) >= 3:
                phi_now = signature(h_history[i][-4:])
                best_sim, best_match = 0.0, None
                for (sid, ftype, sig) in signature_registry:
                    if sid == i:
                        continue
                    sim = cos_sim(phi_now, sig)
                    if sim > best_sim:
                        best_sim, best_match = sim, (sid, ftype)
                if best_sim > tau_ctfr:
                    ctfr_flags.append({"site": i, "round": r, "sim": best_sim, "matched_fault": best_match[1]})

            uploads_masks.append(site_free_mask)

        for i in range(n_sites):
            thetas[i]["W1"][:, global_registry] = theta_global["W1"][:, global_registry]
            thetas[i]["b1"][global_registry] = theta_global["b1"][global_registry]
            thetas[i]["W2"][global_registry] = theta_global["W2"][global_registry]

    # Confirm each flag against the flagged site's OWN later commit history:
    # a flag is confirmed if that site later commits a task with the SAME
    # fault type the flag matched, and lead time = that commit's round
    # minus the flag's round.
    lead_times, n_confirmed, n_raised = [], 0, len(ctfr_flags)
    for f in ctfr_flags:
        i, r, matched_fault = f["site"], f["round"], f["matched_fault"]
        later = [cr for (cr, ft) in own_commit_log[i] if cr > r and ft == matched_fault]
        if later:
            n_confirmed += 1
            lead_times.append(min(later) - r)

    return {
        "n_flags_raised": n_raised,
        "n_confirmed": n_confirmed,
        "precision": (n_confirmed / n_raised) if n_raised else None,
        "mean_lead_time": float(np.mean(lead_times)) if lead_times else None,
        "std_lead_time": float(np.std(lead_times)) if lead_times else None,
        "lead_times": lead_times,
    }


# ─────────────────────────────────────────────────────────────────────────
# Fig 10: heterogeneity sensitivity (gamma sweep), FPS by device class
# ─────────────────────────────────────────────────────────────────────────
def gamma_sweep(benchmark_name="fed-twin-femto", mode="regression", stage_len=5,
                 n_rounds=20, local_steps=40, lr=0.01, s0=0.45,
                 gammas=(0.0, 0.2, 0.4, 0.6, 0.8), rng_seed=RNG_SEED):
    results = {g: {dc: [] for dc in set(DEVICE_CLASSES)} for g in gammas}
    for gamma in gammas:
        fault_lib = make_fault_library(np.random.default_rng(rng_seed))
        for site_id, dc in enumerate(DEVICE_CLASSES):
            rng = np.random.default_rng(rng_seed * 1000 + 3)
            site_rng = np.random.default_rng(rng_seed * 7919 + site_id)
            stream = SiteStream(mode, fault_lib, stage_len, site_rng, site_id=site_id)
            theta = init_backbone(rng)
            theta_global = init_backbone(rng)
            global_registry = np.zeros(C_SHARED, dtype=bool)
            ph = PageHinkley(delta=0.01, lam=0.7)
            true_stage_prev = 0
            task_masks, task_eval_at_commit = {}, {}

            for r in range(n_rounds):
                dcmax = CMAX[dc]
                site_free_mask = ~global_registry.copy()
                site_free_mask[dcmax:] = False
                theta_before = {k: v.copy() for k, v in theta.items()}
                h, true_stage_now, _ = stream.advance_round()
                trig, _ = ph.update(h)
                old_stage = true_stage_prev
                stage_changed = true_stage_now != old_stage
                true_stage_prev = true_stage_now

                newly_committed = None
                if trig and site_free_mask.any():
                    s_n = sparsity_target(dc, global_registry.mean(), s0=s0, gamma=gamma)
                    free_idx = np.where(site_free_mask)[0]
                    scores = np.abs(theta_before["W1"][:, free_idx]).sum(0) + np.abs(theta_before["W2"][free_idx])
                    n_commit = max(1, int(np.ceil(s_n * len(free_idx))))
                    top = free_idx[np.argsort(-scores)[:n_commit]]
                    newly_committed = np.zeros(C_SHARED, dtype=bool)
                    newly_committed[top] = True
                    task_masks[old_stage] = task_masks.get(old_stage, np.zeros(C_SHARED, dtype=bool)) | newly_committed

                if stage_changed and old_stage not in task_eval_at_commit:
                    Xc, yc = stream.eval_on_stage(old_stage, n=200)
                    m = task_masks.get(old_stage)
                    task_eval_at_commit[old_stage] = _eval_metric(theta_before, Xc, yc, mode, task_mask=m)

                if newly_committed is not None:
                    site_free_mask = site_free_mask & ~newly_committed
                    top = np.where(newly_committed)[0]
                    theta_global["W1"][:, top] = theta_before["W1"][:, top]
                    theta_global["b1"][top] = theta_before["b1"][top]
                    theta_global["W2"][top] = theta_before["W2"][top]
                    global_registry |= newly_committed

                for _ in range(local_steps):
                    X, y = stream.minibatch(n=32)
                    _, cache = forward(theta, X, mode)
                    g = clip_grads(backward(theta, cache, y, mode))
                    g = mask_grads(g, site_free_mask)
                    for k in ("W1", "b1", "W2"):
                        theta[k] = theta[k] - lr * g[k]

                theta["W1"][:, global_registry] = theta_global["W1"][:, global_registry]
                theta["b1"][global_registry] = theta_global["b1"][global_registry]
                theta["W2"][global_registry] = theta_global["W2"][global_registry]

            fps_vals = []
            for stage_idx in task_eval_at_commit:
                Xf, yf = stream.eval_on_stage(stage_idx, n=200)
                m = task_masks.get(stage_idx)
                fps_vals.append(_eval_metric(theta, Xf, yf, mode, task_mask=m))
            if fps_vals:
                results[gamma][dc].append(np.mean(fps_vals))

    return {g: {dc: float(np.mean(v)) for dc, v in dcs.items()} for g, dcs in results.items()}


if __name__ == "__main__":
    print("=== DTTS sweep (fed-twin-femto) ===")
    dtts = dtts_sweep()
    for lam, res in dtts.items():
        print(f"lambda_PH={lam}: {res}")

    print("\n=== CTFR lead-time (all 3 benchmarks) ===")
    for bench, mode, stage_len, recal in [
        ("fed-twin-cmapss", "regression", 6, [6, 7, 8, 9, 10]),
        ("fed-twin-femto", "regression", 5, None),
        ("fed-twin-mimii", "classification", 6, list(range(0, 24, 3))),
    ]:
        res = ctfr_leadtime(bench, mode, stage_len, recal_sites=recal)
        print(f"{bench}: {res['n_flags_raised']} flags, {res['n_confirmed']} confirmed, "
              f"precision={res['precision']}, mean_lead_time={res['mean_lead_time']}, std={res['std_lead_time']}")

    print("\n=== Gamma / heterogeneity sweep (fed-twin-femto) ===")
    gsweep = gamma_sweep()
    for g, dcs in gsweep.items():
        print(f"gamma={g}: {dcs}")
