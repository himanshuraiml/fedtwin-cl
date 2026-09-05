"""
Real-data FedProx baseline + Centralized Task-Isolated upper bound, for all
real FedTwin-CL pipelines (Fed-Twin-CMAPSS, Fed-Twin-FEMTO, Fed-Twin-MIMII).

Closes reviewer gap items 1 and 2 (EXPERIMENT_PROCEDURE.md camera-ready list,
2026-09-04 follow-up): "5 of 7 methods have real-data numbers" -> FedProx and
Centralized Task-Isolated are the two missing ones, and Centralized is also
the reference point Eq. 7 (Fleet Performance Score, Psi=1 anchor) needs.

Both baselines are implemented generically against an already-existing real
pipeline module (real_pipeline_cmapss / real_pipeline_femto / real_pipeline_mimii)
so they reuse the exact same backbone, forward pass, loss, clipping, and
per-site data-stream/eval-on-stage logic already validated and bug-fixed
there (EXPERIMENT_PROCEDURE.md bugs 1-14) -- no architecture code is
reimplemented or duplicated.

FedProx: identical to the existing "fedavg" branch of each pipeline (dense,
full-network FedAvg aggregation every round, no masking, no drift detector
needed for the update rule itself) except every local SGD step adds the
proximal-term gradient mu/2 * ||theta - theta_global_at_round_start||^2,
i.e. grad += mu * (theta - theta_global_at_round_start), computed against
the snapshot each site started the round from (which, for a dense
FedAvg-style method, IS theta_global from the end of the previous round --
verified directly rather than assumed, since it is what the aggregation
step at the end of every prior round already set every site's theta to).
mu=0.01 (a standard small-drift-penalty default from the original FedProx
paper's own recommended range, not swept here) is used for all three
benchmarks; see EXPERIMENT_PROCEDURE.md for why a full mu-sweep was judged
out of scope this pass.

Centralized Task-Isolated: for every (site, ground-truth stage) pair, train
ONE fresh model from scratch using ONLY that site's own real samples
belonging to that stage (an 80/20 within-stage train/eval split), with NO
federation (no cross-site communication at all, ever) and NO masking. This
is the forgetting-free, communication-free upper bound: each task gets its
own dedicated, never-touched-again model, so Fleet Backward Transfer is
*exactly* 0.0 by construction (not measured -- guaranteed by the protocol,
the same way Table 2's PROTECTED rows are guaranteed by Theorem 1, just for
a different reason: no shared parameters exist to be overwritten later).
"""
import os
import sys
import json
import argparse
import importlib
import numpy as np
import torch

MU_PROX = 0.01
# 300 steps was tried first and found to badly undertrain the Centralized
# Task-Isolated model (train-set metric itself still far from converged,
# confirmed by direct inspection: train_metric=-0.239 at step 300 vs
# -0.000 at step 3000 on a representative site/stage). An undertrained
# "upper bound" is not a real upper bound -- it silently produced a floor
# (persistence predictor) that beat the supposed ceiling on both real
# regression benchmarks, which is a training-budget artifact, not a
# genuine finding about task-isolated training being weak. Recorded as
# EXPERIMENT_PROCEDURE.md bug/finding #15.
CENTRALIZED_STEPS = 3000
N_STAGES = 3

BENCHMARKS = {
    "cmapss": ("real_pipeline_cmapss", "fed-twin-cmapss-real"),
    "femto": ("real_pipeline_femto", "fed-twin-femto-real"),
    "mimii": ("real_pipeline_mimii", "fed-twin-mimii-real"),
}


def load_pipeline(name):
    modname, label = BENCHMARKS[name]
    sys.path.insert(0, os.path.dirname(__file__))
    P = importlib.import_module(modname)
    return P, label


def stream_cls(P):
    """cmapss/femto name their per-site stream class RealSiteStream; mimii
    names it MimiiSiteStream. Both expose the identical
    advance_round()/minibatch()/eval_on_stage() interface, so every loop
    below is written against that interface and works unmodified for
    either name."""
    return getattr(P, "RealSiteStream", None) or getattr(P, "MimiiSiteStream")


def is_mimii(label):
    return "mimii" in label


# ── FedProx ──────────────────────────────────────────────────────────────
def run_fedprox(P, label, seed=1, mu=MU_PROX):
    data = np.load(P.DATA_PATH, allow_pickle=True)
    sites_data = data["sites"]
    if is_mimii(label):
        any_w = sites_data[0]["round_windows"][1]  # round 0 can be empty; round 1 is not
        n_sensors, window = any_w.shape[-1], any_w.shape[1]
    else:
        n_sensors = sites_data[0]["windows"].shape[2]
        window = sites_data[0]["windows"].shape[1]
    n_sites = len(sites_data)
    method = "fedprox"

    seed_idx = 100  # distinct offset so this never collides with an existing METHODS index's init seed
    torch.manual_seed(seed * 1000 + seed_idx)
    theta_global = P.init_backbone(n_sensors, window, seed=seed * 1000 + seed_idx)
    SC = stream_cls(P)
    streams = [SC(sites_data[i], site_id=i) for i in range(n_sites)]
    thetas = [P.clone_theta(theta_global) for _ in range(n_sites)]
    task_eval_at_commit = [dict() for _ in range(n_sites)]
    true_stage_prev = [0] * n_sites
    cum_bytes = [0] * n_sites
    records = []

    for r in range(P.N_ROUNDS):
        uploads = []
        for i, dc in enumerate(P.DEVICE_CLASSES):
            theta = thetas[i]
            stream = streams[i]
            theta_before = P.clone_theta(theta)
            # theta at round start == theta_global from the previous round's
            # aggregation (dense FedAvg-style broadcast, verified by
            # construction: every site is reset to theta_global at the end
            # of every round below), so this snapshot IS the correct FedProx
            # anchor point.
            global_snapshot = {k: v.detach().clone() for k, v in theta.items()}

            h, true_stage_now = stream.advance_round()
            old_stage = true_stage_prev[i]
            stage_changed = true_stage_now != old_stage
            true_stage_prev[i] = true_stage_now
            if stage_changed and old_stage not in task_eval_at_commit[i]:
                Xc, yc = stream.eval_on_stage(old_stage)
                val = P.eval_metric(theta_before, Xc.to(P.DEVICE), yc.to(P.DEVICE))
                if val is not None:  # MIMII's AUC is undefined for a single-class batch
                    task_eval_at_commit[i][old_stage] = val

            Xb, yb = stream.minibatch()
            if len(yb) > 0:
                for _ in range(P.LOCAL_STEPS):
                    P.zero_grad(theta)
                    l = P.loss_fn(theta, Xb.to(P.DEVICE), yb.to(P.DEVICE))
                    l.backward()
                    # Proximal term added BEFORE clipping, exactly the same
                    # ordering fix as EWC's penalty gradient (bug #9) -- an
                    # unbounded prox term on real, less-uniform gradients risks
                    # the identical NaN-then-fleet-wide-contamination failure
                    # mode FedAvg-EWC hit before that fix.
                    with torch.no_grad():
                        for k in theta:
                            theta[k].grad += mu * (theta[k] - global_snapshot[k])
                    P.clip_grads(theta)
                    with torch.no_grad():
                        for k in theta:
                            theta[k] -= P.LR * theta[k].grad
                    P.zero_grad(theta)

            n_free_params = sum(v.numel() for v in theta.values())
            pbytes = int(n_free_params * 32 / 8)
            cum_bytes[i] += pbytes
            uploads.append({"i": i, "energy": pbytes * P.E_TX[dc]})

        with torch.no_grad():
            for k in theta_global:
                theta_global[k] = sum(thetas[i][k] for i in range(n_sites)) / n_sites
            for i in range(n_sites):
                thetas[i] = P.clone_theta(theta_global)

        for i, dc in enumerate(P.DEVICE_CLASSES):
            Xc, yc = streams[i].eval_on_stage(true_stage_prev[i])
            metric = P.eval_metric(thetas[i], Xc.to(P.DEVICE), yc.to(P.DEVICE))
            if metric is None:
                continue
            records.append({"benchmark": label, "method": method, "site_id": i,
                             "device_class": dc, "round": r + 1, "raw_metric": metric,
                             "uplink_bytes": cum_bytes[i], "tx_energy_mj": uploads[i]["energy"],
                             "commit_count": 0})

    for i, dc in enumerate(P.DEVICE_CLASSES):
        fps_vals, fbwt_vals = [], []
        for stage_idx, at_commit in task_eval_at_commit[i].items():
            Xc, yc = streams[i].eval_on_stage(stage_idx)
            final = P.eval_metric(thetas[i], Xc.to(P.DEVICE), yc.to(P.DEVICE))
            if final is None:
                continue
            fps_vals.append(final)
            fbwt = final - at_commit
            fbwt_vals.append(fbwt)
            records.append({"benchmark": label, "method": method, "site_id": i,
                             "device_class": dc, "round": P.N_ROUNDS, "stage_idx": int(stage_idx),
                             "task_fbwt": fbwt, "task_protected": False, "is_task_record": True})
        records.append({"benchmark": label, "method": method, "site_id": i, "device_class": dc,
                         "round": P.N_ROUNDS,
                         "fleet_perf_score": float(np.mean(fps_vals)) if fps_vals else None,
                         "fleet_backward_transfer": float(np.mean(fbwt_vals)) if fbwt_vals else None,
                         "n_tasks_seen": len(fps_vals), "commit_count": 0,
                         "uplink_bytes": cum_bytes[i], "is_summary_record": True})
    print(f"  [{label}] fedprox done (mu={mu})")
    return records


# ── Centralized Task-Isolated ────────────────────────────────────────────
def _full_population_per_stage(P, label, rec):
    """Returns {stage_idx: (windows, targets)} using every real sample the
    site ever had for that ground-truth stage -- true "full local data",
    not a resampled/capped batch. cmapss/femto store one big
    (windows, rul_norm, stage) array per site, so this indexes it directly.
    MIMII stores per-round clip batches instead; its own eval_on_stage()
    already concatenates every real round tagged with a given stage with no
    sampling or cap, so it IS the full population for that benchmark and is
    reused directly instead of re-deriving the same pooling logic twice."""
    if is_mimii(label):
        stages = sorted(set(int(s) for s in rec["stage_per_round"]))
        out = {}
        SC = stream_cls(P)
        stream = SC(rec, site_id=0)
        for s in stages:
            w, y = stream.eval_on_stage(s)
            if len(y) > 0:
                out[s] = (w, y)
        return out
    else:
        windows_all = torch.tensor(rec["windows"])
        target_all = torch.tensor(rec["rul_norm"])
        stage_all = np.asarray(rec["stage"])
        out = {}
        for s in sorted(set(stage_all.tolist())):
            idx = np.where(stage_all == s)[0]
            if len(idx) > 0:
                out[int(s)] = (windows_all[idx], target_all[idx])
        return out


def run_centralized(P, label, seed=1, steps=CENTRALIZED_STEPS, n_stages=N_STAGES):
    data = np.load(P.DATA_PATH, allow_pickle=True)
    sites_data = data["sites"]
    n_sensors = sites_data[0]["windows"].shape[2] if not is_mimii(label) else None
    window = sites_data[0]["windows"].shape[1] if not is_mimii(label) else None
    n_sites = len(sites_data)
    method = "centralized"
    records = []

    for i, dc in enumerate(P.DEVICE_CLASSES):
        rec = sites_data[i]
        by_stage = _full_population_per_stage(P, label, rec)
        if is_mimii(label) and n_sensors is None:
            any_w = next(iter(by_stage.values()))[0]
            n_sensors, window = any_w.shape[-1], any_w.shape[1]
        fps_vals = []
        for stage_idx, (Xall, Yall) in by_stage.items():
            Xall = torch.as_tensor(Xall)
            Yall = torch.as_tensor(Yall)
            if len(Xall) < 5 or len(set(Yall.tolist())) < (2 if is_mimii(label) else 1):
                continue
            rng = np.random.default_rng(1000 * i + stage_idx)
            perm = rng.permutation(len(Xall))
            n_eval = max(1, int(0.2 * len(perm)))
            eval_idx, train_idx = perm[:n_eval], perm[n_eval:]
            if len(train_idx) == 0:
                train_idx = perm
            # For MIMII, guarantee the held-out eval split still has both
            # classes represented (AUC is undefined for a single-class
            # batch) by falling back to a stratified split when a plain
            # random split happens to put every abnormal clip on one side --
            # a real risk here given how few real abnormal clips exist for
            # some site/stage pairs.
            if is_mimii(label) and len(set(Yall[eval_idx].tolist())) < 2:
                pos = np.where(Yall.numpy() == 1)[0]
                neg = np.where(Yall.numpy() == 0)[0]
                if len(pos) >= 1 and len(neg) >= 1:
                    n_eval_pos = max(1, int(0.2 * len(pos)))
                    n_eval_neg = max(1, int(0.2 * len(neg)))
                    eval_idx = np.concatenate([rng.permutation(pos)[:n_eval_pos],
                                                rng.permutation(neg)[:n_eval_neg]])
                    train_idx = np.array([j for j in range(len(Yall)) if j not in set(eval_idx.tolist())])
                    if len(train_idx) == 0:
                        train_idx = np.arange(len(Yall))
                else:
                    continue  # genuinely only one class ever observed for this stage; skip, matches pipeline's own None convention

            theta = P.init_backbone(n_sensors, window, seed=seed * 1000 + i * 100 + int(stage_idx))
            Xtr, ytr = Xall[train_idx], Yall[train_idx]
            for _ in range(steps):
                bidx = np.random.randint(0, len(train_idx), size=min(P.BATCH if hasattr(P, "BATCH") else 16, len(train_idx)))
                Xb, yb = Xtr[bidx].to(P.DEVICE), ytr[bidx].to(P.DEVICE)
                P.zero_grad(theta)
                l = P.loss_fn(theta, Xb, yb)
                l.backward()
                P.clip_grads(theta)
                with torch.no_grad():
                    for k in theta:
                        theta[k] -= P.LR * theta[k].grad
                P.zero_grad(theta)

            Xev = Xall[eval_idx].to(P.DEVICE)
            yev = Yall[eval_idx].to(P.DEVICE)
            final = P.eval_metric(theta, Xev, yev)
            if final is None:
                continue
            fps_vals.append(final)
            # Forgetting-free by construction: this task's model is never
            # touched again by any other task's training, so FBWT is
            # guaranteed exactly 0.0, not separately measured -- the
            # no-shared-parameters analogue of Theorem 1's frozen-mask
            # guarantee, for a different structural reason.
            records.append({"benchmark": label, "method": method, "site_id": i,
                             "device_class": dc, "round": None, "stage_idx": int(stage_idx),
                             "task_fbwt": 0.0, "task_protected": True, "is_task_record": True,
                             "raw_metric": final, "n_train": int(len(train_idx)), "n_eval": int(len(eval_idx))})
        records.append({"benchmark": label, "method": method, "site_id": i, "device_class": dc,
                         "round": None,
                         "fleet_perf_score": float(np.mean(fps_vals)) if fps_vals else None,
                         "fleet_backward_transfer": 0.0, "n_tasks_seen": len(fps_vals),
                         "commit_count": 0, "uplink_bytes": 0, "is_summary_record": True})
    print(f"  [{label}] centralized done ({n_stages} stages/site, {steps} steps/task)")
    return records


def persistence_floor(rul_norm):
    """Trivial-baseline floor for regression benchmarks: predict the
    previous observed target value (a persistence predictor), scored with
    the same -MSE convention as eval_metric."""
    y = np.asarray(rul_norm, dtype=np.float64)
    y_pred = np.concatenate([[y[0]], y[:-1]])
    return -float(np.mean((y - y_pred) ** 2))


def compute_psi_anchors(P, label, sites_data, centralized_records):
    """Eq. 7/14 normalization anchors: ceiling = mean Centralized
    Task-Isolated fleet_perf_score across sites (Psi=1); floor = mean
    persistence-predictor (trivial baseline) raw metric across sites
    (Psi=0). MIMII (classification/AUC) uses chance-level AUC=0.5 as the
    floor instead of a persistence predictor, matching the manuscript's
    stated Eq. 14 convention."""
    ceiling_vals = [r["fleet_perf_score"] for r in centralized_records
                    if r.get("is_summary_record") and r["fleet_perf_score"] is not None]
    ceiling = float(np.mean(ceiling_vals))
    if label == "fed-twin-mimii-real":
        floor = 0.5  # chance-level AUC
    else:
        floor_vals = [persistence_floor(s["rul_norm"]) for s in sites_data]
        floor = float(np.mean(floor_vals))
    return floor, ceiling


def psi(metric, floor, ceiling):
    if ceiling == floor:
        return None
    return float(np.clip((metric - floor) / (ceiling - floor), 0.0, 1.0))


def run_benchmark(name):
    P, label = load_pipeline(name)
    recs = []
    recs += run_fedprox(P, label)
    cent_recs = run_centralized(P, label)
    recs += cent_recs

    data = np.load(P.DATA_PATH, allow_pickle=True)
    sites_data = data["sites"]
    floor, ceiling = compute_psi_anchors(P, label, sites_data, cent_recs)
    print(f"  [{label}] Psi anchors: floor={floor:.4f} ceiling={ceiling:.4f}")
    for r in recs:
        if r.get("is_summary_record") and r.get("fleet_perf_score") is not None:
            r["fps_normalized"] = psi(r["fleet_perf_score"], floor, ceiling)
    recs.append({"benchmark": label, "is_psi_anchor_record": True, "psi_floor": floor, "psi_ceiling": ceiling})
    return recs


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--benchmarks", nargs="+", default=["cmapss", "femto"],
                     choices=list(BENCHMARKS.keys()))
    args = ap.parse_args()

    all_records = {}
    for name in args.benchmarks:
        print(f"=== {name} ===")
        all_records[name] = run_benchmark(name)

    out_path = os.path.join(os.path.dirname(__file__), "fedprox_centralized_real_results.json")
    existing = {}
    if os.path.exists(out_path):
        with open(out_path) as f:
            existing = json.load(f)
    existing.update(all_records)
    with open(out_path, "w") as f:
        json.dump(existing, f)
    print(f"\nWrote {out_path} (benchmarks now present: {list(existing.keys())})")

    for name in args.benchmarks:
        recs = all_records[name]
        print(f"\n--- {name} summary ---")
        for m in ("fedprox", "centralized"):
            summ = [r for r in recs if r.get("method") == m and r.get("is_summary_record")]
            if not summ:
                continue
            fps = np.mean([r["fleet_perf_score"] for r in summ])
            fbwt = np.mean([r["fleet_backward_transfer"] for r in summ])
            fps_n = [r.get("fps_normalized") for r in summ if r.get("fps_normalized") is not None]
            fps_n_str = f" FPS_norm={np.mean(fps_n):.4f}" if fps_n else ""
            print(f"{m}: FPS={fps:.4f} FBWT={fbwt:+.4f}{fps_n_str}")
