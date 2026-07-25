"""
Real-data FedTwin-CL pipeline on MIMII (Fed-Twin-MIMII benchmark).

Same TCN-Nano-lite architecture and mask-registry/DTTS/CTFR mechanics as
real_pipeline_cmapss.py / real_pipeline_femto.py, adapted for MIMII's
different data shape: a variable number of real audio clips per round
(rather than one fixed-size window per round) and a binary classification
head (rather than RUL regression). All three FEMTO-learned fixes are
applied from the start here (not discovered again the hard way):
time-tercile ground-truth stages, milestone-based oracle triggering, and
freezing a stage's mask once its eval-at-commit snapshot is recorded.
"""
import os
import json
import numpy as np
import torch
import torch.nn.functional as F

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

DATA_PATH = os.path.join(os.path.dirname(__file__), "processed", "mimii_sites.npz")
DEVICE_CLASSES = ["remote-asset"] * 6 + ["sensor-edge"] * 10 + ["line-gateway"] * 6 + ["plant-fog"] * 2
CMAX = {"remote-asset": 128, "sensor-edge": 256, "line-gateway": 512, "plant-fog": 384}
C_SHARED = 512
E_TX = {"remote-asset": 1.8, "sensor-edge": 0.09, "line-gateway": 0.09, "plant-fog": 0.012}

N_ROUNDS = 20
LOCAL_STEPS = 8  # fewer than C-MAPSS/FEMTO: each round has only ~6 real clips available, not a resampleable stream
LR = 0.01
S0, GAMMA, SMIN, SMAX = 0.45, 0.6, 0.25, 0.70
KAPPA, B_Q = 0.5, 4
LAM_PH, DELTA_PH = 0.15, 0.01  # FEMTO-calibrated starting point; re-swept below if needed
TAU_CTFR = 0.9
EWC_LAMBDA = 1.0

METHODS = ["fedavg", "fedavg-ewc", "fedcat-external", "fedtwin-cl-no-ctfr", "fedtwin-cl"]


def init_backbone(n_features, n_frames, width=C_SHARED, seed=0):
    g = torch.Generator().manual_seed(seed)
    conv_w = torch.randn(width, n_features, n_frames, generator=g) / np.sqrt(n_features * n_frames)
    conv_b = torch.zeros(width)
    fc_w = torch.randn(width, generator=g) / np.sqrt(width)
    fc_b = torch.zeros(1)
    theta = {"conv_w": conv_w.clone().to(DEVICE).requires_grad_(True),
             "conv_b": conv_b.clone().to(DEVICE).requires_grad_(True),
             "fc_w": fc_w.clone().to(DEVICE).requires_grad_(True),
             "fc_b": fc_b.clone().to(DEVICE).requires_grad_(True)}
    return theta


def clone_theta(theta):
    return {k: v.detach().clone().requires_grad_(True) for k, v in theta.items()}


def forward(theta, X, task_mask=None):
    """X: (batch, n_frames, n_features) -> (batch, n_features, n_frames)."""
    Xc = X.permute(0, 2, 1)
    z1 = F.conv1d(Xc, theta["conv_w"], theta["conv_b"])
    a1 = F.relu(z1).squeeze(-1)
    if task_mask is not None:
        a1 = a1 * task_mask.unsqueeze(0)
    logit = a1 @ theta["fc_w"] + theta["fc_b"]
    return logit, a1


def loss_fn(theta, X, y, task_mask=None):
    logit, _ = forward(theta, X, task_mask=task_mask)
    return F.binary_cross_entropy_with_logits(logit.squeeze(-1), y)


def auc_score(y_true, y_score):
    order = np.argsort(y_score)
    y_true = np.asarray(y_true)[order]
    n_pos, n_neg = y_true.sum(), len(y_true) - y_true.sum()
    if n_pos == 0 or n_neg == 0:
        return 0.5
    ranks = np.arange(1, len(y_true) + 1)
    sum_ranks_pos = ranks[y_true.astype(bool)].sum()
    return (sum_ranks_pos - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def eval_metric(theta, X, y, task_mask=None):
    if len(y) == 0:
        return None
    with torch.no_grad():
        logit, _ = forward(theta, X, task_mask=task_mask)
        prob = torch.sigmoid(logit.squeeze(-1)).cpu().numpy()
    y_np = y.cpu().numpy() if torch.is_tensor(y) else y
    if len(set(y_np.tolist())) < 2:
        return None  # AUC undefined for a single-class batch
    return float(auc_score(y_np, prob))


def zero_grad(theta):
    for v in theta.values():
        if v.grad is not None:
            v.grad = None


def clip_grads(theta, max_norm=5.0):
    total = torch.sqrt(sum((v.grad ** 2).sum() for v in theta.values() if v.grad is not None))
    if total > max_norm and total > 0:
        scale = max_norm / total
        for v in theta.values():
            if v.grad is not None:
                v.grad.mul_(scale)


def mask_grad(theta, free_mask_t):
    theta["conv_w"].grad *= free_mask_t.view(-1, 1, 1)
    theta["conv_b"].grad *= free_mask_t
    theta["fc_w"].grad *= free_mask_t


def sparsity_target(dc, rho_global, s0=S0, gamma=GAMMA, smin=SMIN, smax=SMAX):
    cmax_mean = np.mean(list(CMAX.values()))
    s = s0 * (CMAX[dc] / cmax_mean) * (1 + gamma * rho_global)
    return float(np.clip(s, smin, smax))


class PageHinkley:
    def __init__(self, delta=DELTA_PH, lam=LAM_PH):
        self.delta, self.lam = delta, lam
        self.reset()

    def reset(self):
        self.n, self.mean, self.m, self.m_min = 0, 0.0, 0.0, 0.0

    def update(self, h):
        self.n += 1
        self.mean += (h - self.mean) / self.n
        self.m += h - self.mean - self.delta
        self.m_min = min(self.m_min, self.m)
        ph = self.m - self.m_min
        trig = ph > self.lam
        if trig:
            self.reset()
        return trig


def signature(h_trace):
    h = np.array(h_trace) if len(h_trace) > 0 else np.array([0.0])
    t = np.linspace(0, 1, len(h))
    slope = np.polyfit(t, h, 1)[0] if len(h) > 1 else 0.0
    curv = np.polyfit(t, h, 2)[0] if len(h) > 2 else 0.0
    return np.array([h.mean(), h.std(), slope, curv, h.min(), h.max()])


def cos_sim(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(np.dot(a, b) / (na * nb)) if na > 0 and nb > 0 else 0.0


def payload_bytes(n_free_params, kappa=KAPPA, b_q=B_Q):
    return int(np.ceil(kappa * n_free_params) * b_q / 8)


class MimiiSiteStream:
    """Wraps one real MIMII site's precomputed per-round clip batches."""

    def __init__(self, site_record, site_id=0):
        self.rec = site_record
        self.site_id = site_id
        self.r = -1

    def advance_round(self):
        self.r += 1
        h = float(self.rec["health_per_round"][self.r])
        stage = int(self.rec["stage_per_round"][self.r])
        return h, stage

    def minibatch(self):
        w = self.rec["round_windows"][self.r]
        y = self.rec["round_labels"][self.r]
        return torch.tensor(w), torch.tensor(y)

    def eval_on_stage(self, stage_idx, seed_bump=0):
        """All real clips from all rounds tagged with this ground-truth
        stage, pooled (deterministic, no sampling needed: MIMII rounds are
        already small, real clip counts)."""
        idxs = [r for r in range(len(self.rec["stage_per_round"])) if self.rec["stage_per_round"][r] == stage_idx]
        ws = np.concatenate([self.rec["round_windows"][r] for r in idxs], axis=0) if idxs else np.zeros((0, 8, 9), dtype=np.float32)
        ys = np.concatenate([self.rec["round_labels"][r] for r in idxs], axis=0) if idxs else np.zeros((0,), dtype=np.float32)
        return torch.tensor(ws), torch.tensor(ys)


def run_mimii(methods=METHODS, seed=1):
    data = np.load(DATA_PATH, allow_pickle=True)
    sites_data = data["sites"]
    n_sites = len(sites_data)
    n_features = sites_data[0]["round_windows"][0].shape[-1] if sites_data[0]["round_windows"][0].shape[0] > 0 \
        else sites_data[0]["round_windows"][1].shape[-1]
    n_frames = sites_data[0]["round_windows"][1].shape[1]
    print(f"{n_sites} real sites, n_features={n_features}, n_frames={n_frames}")

    records = []
    for method in methods:
        torch.manual_seed(seed * 1000 + METHODS.index(method))
        theta_global = init_backbone(n_features, n_frames, seed=seed * 1000 + METHODS.index(method))
        streams = [MimiiSiteStream(sites_data[i], site_id=i) for i in range(n_sites)]
        thetas = [clone_theta(theta_global) for _ in range(n_sites)]
        phs = [PageHinkley() for _ in range(n_sites)]
        global_registry = torch.zeros(C_SHARED, dtype=torch.bool, device=DEVICE)
        signature_registry = []
        ctfr_flags = []
        h_history = [[] for _ in range(n_sites)]
        commit_count = [0] * n_sites
        task_masks = [dict() for _ in range(n_sites)]
        task_eval_at_commit = [dict() for _ in range(n_sites)]
        true_stage_prev = [0] * n_sites
        max_stage_reached = [0] * n_sites
        cum_bytes = [0] * n_sites

        use_masking = method in ("fedcat-external", "fedtwin-cl-no-ctfr", "fedtwin-cl")
        use_ewc = method == "fedavg-ewc"
        use_ctfr = method == "fedtwin-cl"
        boundary_source = "oracle" if method == "fedcat-external" else "dtts"
        ewc_star = [None] * n_sites
        ewc_fisher = [None] * n_sites

        for r in range(N_ROUNDS):
            rho_global = global_registry.float().mean().item()
            uploads = []
            for i, dc in enumerate(DEVICE_CLASSES):
                cmax = CMAX[dc]
                site_free = (~global_registry).clone()
                site_free[cmax:] = False
                theta = thetas[i]
                stream = streams[i]
                theta_before = clone_theta(theta)

                h, true_stage_now = stream.advance_round()
                h_history[i].append(h)
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
                    val = eval_metric(theta_before, Xc.to(DEVICE), yc.to(DEVICE), m)
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
                    commit_count[i] += 1
                    if use_ctfr:
                        sig = signature(h_history[i][-4:])
                        signature_registry.append((i, sig))
                elif use_ewc and triggered and not stage_already_measured:
                    ewc_star[i] = clone_theta(theta_before)
                    Xc, yc = stream.eval_on_stage(old_stage)
                    if len(yc) > 0:
                        zero_grad(theta_before)
                        l = loss_fn(theta_before, Xc.to(DEVICE), yc.to(DEVICE))
                        l.backward()
                        clip_grads(theta_before)
                        ewc_fisher[i] = {k: (v.grad ** 2).detach() for k, v in theta_before.items()}
                        commit_count[i] += 1

                Xb, yb = stream.minibatch()
                if len(yb) > 0:
                    site_free_t = site_free.float()
                    for _ in range(LOCAL_STEPS):
                        zero_grad(theta)
                        l = loss_fn(theta, Xb.to(DEVICE), yb.to(DEVICE))
                        l.backward()
                        with torch.no_grad():
                            if use_ewc and ewc_star[i] is not None:
                                for k in theta:
                                    theta[k].grad += EWC_LAMBDA * ewc_fisher[i][k] * (theta[k] - ewc_star[i][k])
                        clip_grads(theta)
                        if use_masking:
                            mask_grad(theta, site_free_t)
                        with torch.no_grad():
                            update_keys = ("conv_w", "conv_b", "fc_w") if use_masking else theta.keys()
                            for k in update_keys:
                                theta[k] -= LR * theta[k].grad
                        zero_grad(theta)

                if use_ctfr and len(h_history[i]) >= 3:
                    phi_now = signature(h_history[i][-4:])
                    best_sim, best_match = 0.0, None
                    for (sid, sig) in signature_registry:
                        if sid == i:
                            continue
                        sim = cos_sim(phi_now, sig)
                        if sim > best_sim:
                            best_sim, best_match = sim, sid
                    if best_sim > TAU_CTFR:
                        ctfr_flags.append({"site": i, "round": r, "sim": best_sim, "match": best_match})

                if use_masking:
                    n_free_params = int(site_free.sum().item()) * (n_features * n_frames + 2)
                    pbytes = payload_bytes(n_free_params)
                else:
                    n_free_params = sum(v.numel() for v in theta.values())
                    pbytes = int(n_free_params * 32 / 8)
                cum_bytes[i] += pbytes
                uploads.append({"i": i, "energy": pbytes * E_TX[dc]})

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

            for i, dc in enumerate(DEVICE_CLASSES):
                Xc, yc = streams[i].eval_on_stage(true_stage_prev[i])
                metric = eval_metric(thetas[i], Xc.to(DEVICE), yc.to(DEVICE))
                if metric is not None:
                    records.append({"benchmark": "fed-twin-mimii-real", "method": method, "site_id": i,
                                     "device_class": dc, "round": r + 1, "raw_metric": metric,
                                     "uplink_bytes": cum_bytes[i], "tx_energy_mj": uploads[i]["energy"],
                                     "commit_count": commit_count[i]})

        for i, dc in enumerate(DEVICE_CLASSES):
            fps_vals, fbwt_vals = [], []
            for stage_idx, at_commit in task_eval_at_commit[i].items():
                Xc, yc = streams[i].eval_on_stage(stage_idx)
                m = task_masks[i].get(stage_idx) if use_masking else None
                final = eval_metric(thetas[i], Xc.to(DEVICE), yc.to(DEVICE), m)
                if final is None:
                    continue
                fps_vals.append(final)
                fbwt = final - at_commit
                fbwt_vals.append(fbwt)
                protected = bool(use_masking and m is not None and m.any().item())
                records.append({"benchmark": "fed-twin-mimii-real", "method": method, "site_id": i,
                                 "device_class": dc, "round": N_ROUNDS, "stage_idx": int(stage_idx),
                                 "task_fbwt": fbwt, "task_protected": protected, "is_task_record": True})
            records.append({"benchmark": "fed-twin-mimii-real", "method": method, "site_id": i,
                             "device_class": dc, "round": N_ROUNDS,
                             "fleet_perf_score": float(np.mean(fps_vals)) if fps_vals else None,
                             "fleet_backward_transfer": float(np.mean(fbwt_vals)) if fbwt_vals else None,
                             "n_tasks_seen": len(fps_vals), "commit_count": commit_count[i],
                             "is_summary_record": True})
        for f in ctfr_flags:
            records.append({"benchmark": "fed-twin-mimii-real", "method": method, "site_id": f["site"],
                             "round": f["round"], "ctfr_event": True, "ctfr_sim": f["sim"], "ctfr_match": f["match"]})
        print(f"  method={method} done, {sum(commit_count)} total commits")

    return records


if __name__ == "__main__":
    recs = run_mimii()
    out_path = os.path.join(os.path.dirname(__file__), "mimii_real_results.json")
    with open(out_path, "w") as f:
        json.dump(recs, f)
    print(f"Wrote {len(recs)} records to {out_path}")

    print()
    for m in METHODS:
        summ = [r for r in recs if r.get("method") == m and r.get("is_summary_record")]
        fps_vals = [r["fleet_perf_score"] for r in summ if r["fleet_perf_score"] is not None]
        fbwt_vals = [r["fleet_backward_transfer"] for r in summ if r["fleet_backward_transfer"] is not None]
        commits = np.mean([r["commit_count"] for r in summ])
        fps_str = f"{np.mean(fps_vals):.4f}" if fps_vals else "n/a"
        fbwt_str = f"{np.mean(fbwt_vals):+.4f}" if fbwt_vals else "n/a"
        print(f"{m}: FPS(AUC)={fps_str} FBWT={fbwt_str} mean_commits={commits:.2f}")

    print("\n=== Zero-forgetting validation: protected vs. missed tasks (real MIMII) ===")
    for m in ("fedcat-external", "fedtwin-cl-no-ctfr", "fedtwin-cl"):
        task_recs = [r for r in recs if r.get("method") == m and r.get("is_task_record")]
        protected = [r["task_fbwt"] for r in task_recs if r["task_protected"]]
        missed = [r["task_fbwt"] for r in task_recs if not r["task_protected"]]
        p_str = f"n={len(protected)} mean_fbwt={np.mean(protected):+.6f}" if protected else "n=0"
        ms_str = f"n={len(missed)} mean_fbwt={np.mean(missed):+.6f}" if missed else "n=0"
        print(f"{m}: PROTECTED [{p_str}]  MISSED [{ms_str}]")
