"""
Capacity sweep on Fed-Twin-CMAPSS and Fed-Twin-FEMTO: re-runs FedAvg,
Oracle-Boundary Engine ("fedcat-external"), and full FedTwin-CL at the
pilot's current backbone width (C_SHARED=512, 1x) plus 2x (1024) and 4x
(2048) the conv channel count, scaling each device class's CMAX
proportionally (a genuinely bigger per-device capacity budget, not just a
bigger shared pool the per-device ceiling still throttles away).

Reviewer question this answers directly: is Table 1's FedTwin-CL-under-
FedAvg absolute-metric gap on these two benchmarks a capacity-budget
artifact of "TCN-Nano-lite" (a single-layer, 512-channel backbone)? If the
gap narrows/closes as width grows, yes; if it persists, the paper needs to
say so plainly rather than attribute it to width alone.

This duplicates each pipeline's core round loop (rather than importing
run_cmapss/run_femto, whose CMAX/C_SHARED are read from module-level
constants baked into those functions' own closures) so that width and CMAX
can be swept as real parameters without monkeypatching module globals that
some call sites capture at function-definition time (e.g.
init_backbone's default `width=C_SHARED` argument).
"""
import os
import sys
import json
import argparse
import importlib
import numpy as np
import torch

WIDTH_MULTS = [1, 2, 4]
N_STAGES_DEFAULT = 3

BENCH_CONFIG = {
    "cmapss": dict(module="real_pipeline_cmapss", label="fed-twin-cmapss-real", oracle_mode="simple"),
    "femto": dict(module="real_pipeline_femto", label="fed-twin-femto-real", oracle_mode="max_stage"),
}


def load_pipeline(name):
    sys.path.insert(0, os.path.dirname(__file__))
    cfg = BENCH_CONFIG[name]
    P = importlib.import_module(cfg["module"])
    return P, cfg


def run_with_width(P, cfg, methods, width_mult, seed=1):
    label = cfg["label"]
    oracle_mode = cfg["oracle_mode"]
    width = P.C_SHARED * width_mult
    cmax = {k: int(v * width_mult) for k, v in P.CMAX.items()}

    data = np.load(P.DATA_PATH, allow_pickle=True)
    sites_data = data["sites"]
    n_sensors = sites_data[0]["windows"].shape[2]
    window = sites_data[0]["windows"].shape[1]
    n_sites = len(sites_data)

    records = []
    for method in methods:
        method_idx = P.METHODS.index(method) if method in P.METHODS else 99
        torch.manual_seed(seed * 1000 + method_idx + width_mult * 7)
        theta_global = P.init_backbone(n_sensors, window, width=width, seed=seed * 1000 + method_idx + width_mult * 7)
        streams = [P.RealSiteStream(sites_data[i], site_id=i) for i in range(n_sites)]
        thetas = [P.clone_theta(theta_global) for _ in range(n_sites)]
        phs = [P.PageHinkley() for _ in range(n_sites)]
        global_registry = torch.zeros(width, dtype=torch.bool, device=P.DEVICE)
        commit_count = [0] * n_sites
        task_masks = [dict() for _ in range(n_sites)]
        task_eval_at_commit = [dict() for _ in range(n_sites)]
        true_stage_prev = [0] * n_sites
        max_stage_reached = [0] * n_sites

        use_masking = method in ("fedcat-external", "fedtwin-cl-no-ctfr", "fedtwin-cl")
        boundary_source = "oracle" if method == "fedcat-external" else "dtts"

        for r in range(P.N_ROUNDS):
            rho_global = global_registry.float().mean().item()
            for i, dc in enumerate(P.DEVICE_CLASSES):
                cmax_i = cmax[dc]
                site_free = (~global_registry).clone()
                site_free[cmax_i:] = False
                theta = thetas[i]
                stream = streams[i]
                theta_before = P.clone_theta(theta)

                h, true_stage_now = stream.advance_round()
                if boundary_source == "dtts":
                    triggered = phs[i].update(h)
                elif oracle_mode == "simple":
                    triggered = true_stage_now != true_stage_prev[i]
                else:  # max_stage (femto): only trigger on a genuinely new milestone
                    triggered = true_stage_now > max_stage_reached[i]
                old_stage = true_stage_prev[i]
                stage_changed = true_stage_now != old_stage
                true_stage_prev[i] = true_stage_now
                max_stage_reached[i] = max(max_stage_reached[i], true_stage_now)

                stage_already_measured = old_stage in task_eval_at_commit[i]
                newly_committed = None
                if use_masking and triggered and site_free.any() and not stage_already_measured:
                    s_n = P.sparsity_target(dc, rho_global)
                    # sparsity_target's s0/gamma/smin/smax are ratios of CMAX, so
                    # re-deriving cmax_mean from the SCALED cmax dict keeps the
                    # per-tier sparsity fraction comparable across widths.
                    cmax_mean = np.mean(list(cmax.values()))
                    s_n = float(np.clip(P.S0 * (cmax[dc] / cmax_mean) * (1 + P.GAMMA * rho_global), P.SMIN, P.SMAX))
                    free_idx = torch.where(site_free)[0]
                    with torch.no_grad():
                        scores = (theta_before["conv_w"][free_idx].abs().sum(dim=(1, 2))
                                  + theta_before["fc_w"][free_idx].abs())
                    n_commit = max(1, int(np.ceil(s_n * len(free_idx))))
                    top = free_idx[torch.argsort(-scores)[:n_commit]]
                    newly_committed = torch.zeros(width, dtype=torch.bool, device=P.DEVICE)
                    newly_committed[top] = True
                    prev = task_masks[i].get(old_stage, torch.zeros(width, dtype=torch.bool, device=P.DEVICE))
                    task_masks[i][old_stage] = prev | newly_committed

                if stage_changed and old_stage not in task_eval_at_commit[i]:
                    Xc, yc = stream.eval_on_stage(old_stage)
                    m = task_masks[i].get(old_stage) if use_masking else None
                    task_eval_at_commit[i][old_stage] = P.eval_metric(theta_before, Xc.to(P.DEVICE), yc.to(P.DEVICE), m)

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
                for _ in range(P.LOCAL_STEPS):
                    Xb, yb = stream.minibatch()
                    P.zero_grad(theta)
                    l = P.loss_fn(theta, Xb.to(P.DEVICE), yb.to(P.DEVICE))
                    l.backward()
                    P.clip_grads(theta)
                    if use_masking:
                        P.mask_grad(theta, site_free_t)
                    with torch.no_grad():
                        update_keys = ("conv_w", "conv_b", "fc_w") if use_masking else theta.keys()
                        for k in update_keys:
                            theta[k] -= P.LR * theta[k].grad
                    P.zero_grad(theta)

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
                        thetas[i] = P.clone_theta(theta_global)

        fps_vals_by_site = []
        for i, dc in enumerate(P.DEVICE_CLASSES):
            fps_vals = []
            for stage_idx, at_commit in task_eval_at_commit[i].items():
                Xc, yc = streams[i].eval_on_stage(stage_idx)
                m = task_masks[i].get(stage_idx) if use_masking else None
                final = P.eval_metric(thetas[i], Xc.to(P.DEVICE), yc.to(P.DEVICE), m)
                fps_vals.append(final)
            Xc, yc = streams[i].eval_on_stage(true_stage_prev[i])
            current_task_metric = P.eval_metric(thetas[i], Xc.to(P.DEVICE), yc.to(P.DEVICE))
            fps_vals_by_site.append(float(np.mean(fps_vals)) if fps_vals else None)
            records.append({"benchmark": label, "method": method, "width_mult": width_mult,
                             "site_id": i, "device_class": dc,
                             "fleet_perf_score": fps_vals_by_site[-1],
                             "current_task_metric": current_task_metric,
                             "commit_count": commit_count[i], "is_summary_record": True})
        mean_fps = np.mean([v for v in fps_vals_by_site if v is not None])
        mean_cur = np.mean([r["current_task_metric"] for r in records
                             if r.get("is_summary_record") and r["method"] == method and r["width_mult"] == width_mult])
        print(f"    width={width_mult}x ({width} ch) method={method}: "
              f"mean_FPS={mean_fps:.4f} mean_current_task_metric={mean_cur:.4f} "
              f"total_commits={sum(commit_count)}")
    return records


def run_benchmark(name, methods=("fedavg", "fedcat-external", "fedtwin-cl")):
    P, cfg = load_pipeline(name)
    all_recs = []
    print(f"=== {name} ===")
    for width_mult in WIDTH_MULTS:
        all_recs += run_with_width(P, cfg, methods, width_mult)
    return all_recs


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--benchmarks", nargs="+", default=["cmapss", "femto"])
    args = ap.parse_args()

    out = {}
    for name in args.benchmarks:
        out[name] = run_benchmark(name)

    out_path = os.path.join(os.path.dirname(__file__), "capacity_sweep_real_results.json")
    existing = {}
    if os.path.exists(out_path):
        with open(out_path) as f:
            existing = json.load(f)
    existing.update(out)
    with open(out_path, "w") as f:
        json.dump(existing, f)
    print(f"\nWrote {out_path}")

    print("\n=== Gap-to-FedAvg summary (mean current-task metric, FedAvg - FedTwin-CL) ===")
    for name in args.benchmarks:
        recs = out[name]
        for width_mult in WIDTH_MULTS:
            byM = {}
            for m in ("fedavg", "fedcat-external", "fedtwin-cl"):
                vals = [r["current_task_metric"] for r in recs
                        if r["method"] == m and r["width_mult"] == width_mult and r.get("is_summary_record")]
                byM[m] = np.mean(vals) if vals else None
            gap = byM["fedavg"] - byM["fedtwin-cl"] if byM["fedavg"] is not None and byM["fedtwin-cl"] is not None else None
            print(f"{name} width={width_mult}x: fedavg={byM['fedavg']:.4f} "
                  f"fedcat-external={byM['fedcat-external']:.4f} fedtwin-cl={byM['fedtwin-cl']:.4f} "
                  f"gap(fedavg-fedtwincl)={gap:+.4f}")
