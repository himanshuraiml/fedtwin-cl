"""
Real-data FedTwin-CL pipeline on NASA C-MAPSS (Fed-Twin-CMAPSS benchmark).

Architecture: "TCN-Nano-lite", a deliberate simplification of the
manuscript's TCN-Nano (Section 6.2) to a SINGLE dilated causal Conv1d layer
whose kernel spans the entire input window, followed directly by the
regression head. This makes it structurally isomorphic to the validated
synthetic-spike MLP (conv layer's output channels play exactly the role
the MLP's hidden units played), so the mask-registry / DTTS / CTFR logic
already validated in code/fedtwin_harness.py carries over with minimal new
risk. A full multi-layer TCN-Nano, where channel masking must be threaded
through cross-layer channel mixing, is future work (EXPERIMENT_PROCEDURE.md
Section 4.3) -- not attempted here.

All bug fixes from the synthetic spike are carried over unchanged:
commit-time scoring uses pre-training weights, evaluation of a committed
task uses ONLY that task's own masked channels, the output bias is frozen
for masking methods, DTTS-triggered masks are accumulated (not overwritten)
on over-segmentation, and per-method random seeds are deterministic.
"""
import os
import json
import numpy as np
import torch
import torch.nn.functional as F

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
torch.set_grad_enabled(True)

DATA_PATH = os.path.join(os.path.dirname(__file__), "processed", "femto_sites.npz")
DEVICE_CLASSES = ["remote-asset"] * 6 + ["sensor-edge"] * 10 + ["line-gateway"] * 6 + ["plant-fog"] * 2
CMAX = {"remote-asset": 128, "sensor-edge": 256, "line-gateway": 512, "plant-fog": 384}
C_SHARED = 512
E_TX = {"remote-asset": 1.8, "sensor-edge": 0.09, "line-gateway": 0.09, "plant-fog": 0.012}

N_ROUNDS = 20
LOCAL_STEPS = 40
BATCH = 16
LR = 0.01
S0, GAMMA, SMIN, SMAX = 0.45, 0.6, 0.25, 0.70
KAPPA, B_Q = 0.5, 4
# Real vibration-derived RMS is noisier than C-MAPSS's elapsed-life-fraction
# health indicator, so the lambda_PH calibrated there (0.7) under-triggers
# heavily here; re-swept directly on this data (see real_data session notes).
LAM_PH, DELTA_PH = 0.15, 0.01
TAU_CTFR = 0.9
EWC_LAMBDA = 1.0

METHODS = ["fedavg", "fedavg-ewc", "fedcat-external", "fedtwin-cl-no-ctfr", "fedtwin-cl"]


# ── Backbone: TCN-Nano-lite ────────────────────────────────────────────────
def init_backbone(n_sensors, window, width=C_SHARED, seed=0):
    g = torch.Generator().manual_seed(seed)
    conv_w = torch.randn(width, n_sensors, window, generator=g) / np.sqrt(n_sensors * window)
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
    """X: (batch, window, n_sensors) -> (batch, n_sensors, window) for conv1d."""
    Xc = X.permute(0, 2, 1)
    z1 = F.conv1d(Xc, theta["conv_w"], theta["conv_b"])  # (batch, C, 1)
    a1 = F.relu(z1).squeeze(-1)  # (batch, C)
    if task_mask is not None:
        a1 = a1 * task_mask.unsqueeze(0)
    out = a1 @ theta["fc_w"] + theta["fc_b"]
    return out, a1


def loss_fn(theta, X, y, task_mask=None):
    pred, _ = forward(theta, X, task_mask=task_mask)
    return F.mse_loss(pred.squeeze(-1), y)


def eval_metric(theta, X, y, task_mask=None):
    with torch.no_grad():
        return -loss_fn(theta, X, y, task_mask=task_mask).item()


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


class RealSiteStream:
    """Wraps one real, preprocessed C-MAPSS engine's sequence, replaying it
    across N_ROUNDS communication rounds (the engine's own real cycle
    range divided into N_ROUNDS contiguous chunks)."""

    def __init__(self, site_record, n_rounds=N_ROUNDS, site_id=0):
        self.windows = torch.tensor(site_record["windows"])
        self.rul = torch.tensor(site_record["rul_norm"])
        self.health = site_record["health"]
        self.stage = site_record["stage"]
        self.site_id = site_id
        self.n_rounds = n_rounds
        N = len(self.rul)
        edges = np.linspace(0, N, n_rounds + 1).astype(int)
        self.round_ranges = [(edges[i], max(edges[i] + 1, edges[i + 1])) for i in range(n_rounds)]
        self.round_ranges[-1] = (self.round_ranges[-1][0], N)
        self.r = 0

    def advance_round(self):
        lo, hi = self.round_ranges[self.r]
        self._lo, self._hi = lo, hi
        h = float(self.health[hi - 1])
        stage = int(self.stage[hi - 1])
        self.r += 1
        return h, stage

    def minibatch(self, n=BATCH, rng=None):
        lo, hi = self._lo, self._hi
        idx = np.random.randint(lo, hi, size=min(n, hi - lo)) if hi > lo else np.array([lo])
        return self.windows[idx], self.rul[idx]

    def eval_on_stage(self, stage_idx, n=200):
        idx = np.where(self.stage == stage_idx)[0]
        if len(idx) == 0:
            idx = np.array([0])
        sel = np.random.default_rng(self.site_id * 1000 + stage_idx).choice(
            idx, size=min(n, len(idx)), replace=len(idx) < n)
        return self.windows[sel], self.rul[sel]


def run_cmapss(methods=METHODS, seed=1):
    data = np.load(DATA_PATH, allow_pickle=True)
    sites_data = data["sites"]
    n_sensors = sites_data[0]["windows"].shape[2]
    window = sites_data[0]["windows"].shape[1]
    n_sites = len(sites_data)
    print(f"{n_sites} real sites, n_sensors={n_sensors}, window={window}")

    records = []
    for method in methods:
        torch.manual_seed(seed * 1000 + METHODS.index(method))
        theta_global = init_backbone(n_sensors, window, seed=seed * 1000 + METHODS.index(method))
        streams = [RealSiteStream(sites_data[i], site_id=i) for i in range(n_sites)]
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
                triggered = False
                if boundary_source == "dtts":
                    triggered = phs[i].update(h)
                else:
                    # "Oracle" means perfect knowledge of the true DEGRADATION
                    # MILESTONE reached so far, not perfect knowledge of a
                    # noisy instantaneous reading -- real vibration RMS,
                    # even smoothed, still crosses quantile-stage boundaries
                    # back and forth many times (confirmed: ~53 raw
                    # transitions/site on this data). Triggering oracle on
                    # every such blip exhausts a site's mask-registry
                    # capacity on spurious commits within the first few
                    # rounds, starving every genuine later transition. An
                    # oracle that already knows the bearing reached stage 2
                    # does not "forget" that just because one noisy reading
                    # dipped back toward stage 1.
                    triggered = true_stage_now > max_stage_reached[i]
                old_stage = true_stage_prev[i]
                stage_changed = true_stage_now != old_stage
                true_stage_prev[i] = true_stage_now
                max_stage_reached[i] = max(max_stage_reached[i], true_stage_now)

                newly_committed = None
                # Real, noisy vibration-derived health indicators (unlike
                # the smooth synthetic ramps and C-MAPSS's near-monotonic
                # elapsed-life fraction) can re-enter the SAME nominal
                # ground-truth stage value later in the run after a
                # transient dip. If a mask for that stage_idx already had
                # its eval-at-commit snapshot recorded, growing it further
                # would make the later "final" evaluation use a bigger/
                # different mask than the snapshot did, contaminating the
                # forgetting measurement with a mask-CHANGE effect rather
                # than genuine weight drift. Once a stage's snapshot is
                # locked in, its mask is frozen too, even if the detector
                # fires again for the same nominal value.
                stage_already_measured = old_stage in task_eval_at_commit[i]
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
                    if use_ctfr:
                        sig = signature(h_history[i][-4:])
                        signature_registry.append((i, sig))
                elif use_ewc and triggered:
                    ewc_star[i] = clone_theta(theta_before)
                    Xc, yc = stream.eval_on_stage(old_stage, n=64)
                    zero_grad(theta_before)
                    l = loss_fn(theta_before, Xc.to(DEVICE), yc.to(DEVICE))
                    l.backward()
                    clip_grads(theta_before)  # bound the calibration gradient before squaring into Fisher
                    ewc_fisher[i] = {k: (v.grad ** 2).detach() for k, v in theta_before.items()}
                    commit_count[i] += 1

                site_free_t = site_free.float()
                for _ in range(LOCAL_STEPS):
                    Xb, yb = stream.minibatch()
                    zero_grad(theta)
                    l = loss_fn(theta, Xb.to(DEVICE), yb.to(DEVICE))
                    l.backward()
                    # EWC penalty gradient must be added BEFORE clipping, not
                    # after: clipping only the task-loss gradient leaves the
                    # penalty term completely unbounded, and on real data
                    # (larger, less uniform gradients than the synthetic
                    # spike's Gaussian inputs) that term can grow without
                    # limit as training moves away from the EWC anchor,
                    # eventually overflowing to NaN and, via FedAvg-style
                    # full-network averaging, contaminating every other
                    # site's model in the very next round.
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
                    n_free_params = int(site_free.sum().item()) * (n_sensors * window + 2)
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
                records.append({"benchmark": "fed-twin-femto-real", "method": method, "site_id": i,
                                 "device_class": dc, "round": r + 1, "raw_metric": metric,
                                 "uplink_bytes": cum_bytes[i], "tx_energy_mj": uploads[i]["energy"],
                                 "commit_count": commit_count[i]})

        for i, dc in enumerate(DEVICE_CLASSES):
            fps_vals, fbwt_vals = [], []
            for stage_idx, at_commit in task_eval_at_commit[i].items():
                Xc, yc = streams[i].eval_on_stage(stage_idx)
                m = task_masks[i].get(stage_idx) if use_masking else None
                final = eval_metric(thetas[i], Xc.to(DEVICE), yc.to(DEVICE), m)
                fps_vals.append(final)
                fbwt = final - at_commit
                fbwt_vals.append(fbwt)
                # Per-task protected-vs-missed tag, matching the synthetic
                # spike's zero-forgetting validation (Section 8 of
                # colab_pilot_spike.ipynb): a task is "protected" only if it
                # actually received a DTTS/oracle-triggered mask; averaging
                # protected and missed tasks together in one per-site FBWT
                # (as the summary record below still does, for comparability
                # with Table 3's schema) hides this cleaner signal.
                protected = bool(use_masking and m is not None and m.any().item())
                records.append({"benchmark": "fed-twin-femto-real", "method": method, "site_id": i,
                                 "device_class": dc, "round": N_ROUNDS, "stage_idx": int(stage_idx),
                                 "task_fbwt": fbwt, "task_protected": protected,
                                 "is_task_record": True})
            records.append({"benchmark": "fed-twin-femto-real", "method": method, "site_id": i,
                             "device_class": dc, "round": N_ROUNDS,
                             "fleet_perf_score": float(np.mean(fps_vals)) if fps_vals else None,
                             "fleet_backward_transfer": float(np.mean(fbwt_vals)) if fbwt_vals else None,
                             "n_tasks_seen": len(fps_vals), "commit_count": commit_count[i],
                             "is_summary_record": True})
        for f in ctfr_flags:
            records.append({"benchmark": "fed-twin-femto-real", "method": method, "site_id": f["site"],
                             "round": f["round"], "ctfr_event": True, "ctfr_sim": f["sim"], "ctfr_match": f["match"]})
        print(f"  method={method} done, {sum(commit_count)} total commits")

    return records


if __name__ == "__main__":
    recs = run_cmapss()
    out_path = os.path.join(os.path.dirname(__file__), "femto_real_results.json")
    with open(out_path, "w") as f:
        json.dump(recs, f)
    print(f"Wrote {len(recs)} records to {out_path}")

    print()
    for m in METHODS:
        summ = [r for r in recs if r.get("method") == m and r.get("is_summary_record")]
        fps = np.mean([r["fleet_perf_score"] for r in summ])
        fbwt = np.mean([r["fleet_backward_transfer"] for r in summ])
        commits = np.mean([r["commit_count"] for r in summ])
        print(f"{m}: FPS={fps:.4f} FBWT={fbwt:+.4f} mean_commits={commits:.2f}")

    print("\n=== Zero-forgetting validation: protected vs. missed tasks (real C-MAPSS) ===")
    for m in ("fedcat-external", "fedtwin-cl-no-ctfr", "fedtwin-cl"):
        task_recs = [r for r in recs if r.get("method") == m and r.get("is_task_record")]
        protected = [r["task_fbwt"] for r in task_recs if r["task_protected"]]
        missed = [r["task_fbwt"] for r in task_recs if not r["task_protected"]]
        p_str = f"n={len(protected)} mean_fbwt={np.mean(protected):+.6f}" if protected else "n=0"
        ms_str = f"n={len(missed)} mean_fbwt={np.mean(missed):+.6f}" if missed else "n=0"
        print(f"{m}: PROTECTED [{p_str}]  MISSED [{ms_str}]")
