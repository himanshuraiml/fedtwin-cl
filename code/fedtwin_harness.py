"""
FedTwin-CL spike harness -- full round loop, all method variants.
"""
import numpy as np

DEVICE_CLASSES = ["remote-asset"] * 6 + ["sensor-edge"] * 10 + ["line-gateway"] * 6 + ["plant-fog"] * 2
# Table 1 ratios (32:64:128:96) scaled up 32x: 24 sites each committing a
# task's worth of filters several times over a 20-round run need a shared
# registry with real headroom, or capacity exhaustion (Section 5.3) swamps
# every other effect and silently prevents most commits from ever happening.
# A dedicated, deliberately SMALL-width run is used separately for the
# capacity/saturation sensitivity study (Section 7.3-style), where
# exhaustion is the point rather than a confound.
CMAX = {"remote-asset": 1024, "sensor-edge": 2048, "line-gateway": 4096, "plant-fog": 3072}
E_TX = {"remote-asset": 1.8, "sensor-edge": 0.09, "line-gateway": 0.09, "plant-fog": 0.012}
C_SHARED = 4096
INPUT_DIM = 8
N_FAULT_TYPES = 4


def init_backbone(rng, width=C_SHARED, input_dim=INPUT_DIM):
    return {"W1": rng.normal(0, 1 / np.sqrt(input_dim), size=(input_dim, width)), "b1": np.zeros(width),
            "W2": rng.normal(0, 1 / np.sqrt(width), size=(width,)), "b2": np.zeros(1)}


def forward(theta, X, mode="regression", task_mask=None):
    """`task_mask`, if given, is the SPECIFIC set of committed filters for
    the task being evaluated (Section 4.9 / Eq. 7 of [5]): only those
    hidden units contribute to the output, exactly like real masked
    inference. Without it, every unit in the (possibly still-changing,
    still-being-trained-for-later-tasks) shared backbone contributes,
    which is what makes an *unmasked* eval of an old, "frozen" task look
    like it forgot even though its own committed columns never moved --
    the other, later-repurposed columns are still part of the forward
    pass. This is the mechanism Theorem 5.1's guarantee is actually
    about, and skipping it here would silently test the wrong thing."""
    z1 = X @ theta["W1"] + theta["b1"]
    a1 = np.maximum(z1, 0.0)
    if task_mask is not None:
        a1 = a1 * task_mask[None, :]
    z2 = a1 @ theta["W2"] + theta["b2"]
    if mode == "classification":
        return 1 / (1 + np.exp(-z2)), (X, z1, a1)
    return z2, (X, z1, a1)


def backward(theta, cache, y, mode="regression"):
    X, z1, a1 = cache
    n = X.shape[0]
    if mode == "classification":
        p = 1 / (1 + np.exp(-(a1 @ theta["W2"] + theta["b2"])))
        dz2 = (p - y) / n
    else:
        pred = a1 @ theta["W2"] + theta["b2"]
        dz2 = 2 * (pred - y.ravel()) / n
    gW2 = a1.T @ dz2
    gb2 = dz2.sum(axis=0, keepdims=True)
    da1 = np.outer(dz2, theta["W2"])
    dz1 = da1 * (z1 > 0)
    gW1 = X.T @ dz1
    gb1 = dz1.sum(axis=0)
    return {"W1": gW1, "b1": gb1, "W2": gW2, "b2": gb2}


def mask_grads(grads, free_mask):
    g = {k: v.copy() for k, v in grads.items()}
    g["W1"] *= free_mask[None, :]
    g["b1"] *= free_mask
    g["W2"] *= free_mask
    return g


def clip_grads(grads, max_norm=5.0):
    total = np.sqrt(sum(np.sum(v ** 2) for v in grads.values()))
    if total > max_norm and total > 0:
        scale = max_norm / total
        return {k: v * scale for k, v in grads.items()}
    return grads


def loss_fn(theta, X, y, mode="regression", task_mask=None):
    pred, _ = forward(theta, X, mode, task_mask=task_mask)
    if mode == "classification":
        eps = 1e-7
        return -np.mean(y * np.log(pred + eps) + (1 - y) * np.log(1 - pred + eps))
    return np.mean((pred.ravel() - y.ravel()) ** 2)


def auc_score(y_true, y_score):
    order = np.argsort(y_score)
    y_true = np.asarray(y_true)[order]
    n_pos, n_neg = y_true.sum(), len(y_true) - y_true.sum()
    if n_pos == 0 or n_neg == 0:
        return 0.5
    ranks = np.arange(1, len(y_true) + 1)
    sum_ranks_pos = ranks[y_true.astype(bool)].sum()
    return (sum_ranks_pos - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def make_fault_library(rng, n_types=N_FAULT_TYPES, input_dim=INPUT_DIM):
    lib = []
    for _ in range(n_types):
        A = rng.normal(0, 1.0, size=(2 * input_dim,))
        shape = rng.choice(["linear", "quadratic", "plateau"])
        rate = rng.uniform(0.8, 1.3)
        lib.append({"A": A, "shape": shape, "rate": rate})
    return lib


def phi(X):
    return np.concatenate([X, X ** 2], axis=-1)


def health_curve(t_frac, shape, rate):
    t = np.clip(t_frac * rate, 0, 1)
    if shape == "linear":
        return t
    if shape == "quadratic":
        return t ** 2
    return 1 - np.exp(-3 * t)


class SiteStream:
    """One round == one tick of `t` (one health-indicator observation, one
    possible task-boundary event). Within a round, `minibatch()` may be
    called many times for SGD without advancing `t`, always sampling from
    whichever fault is current *for this round* -- this keeps a round's
    inner local-training loop from silently crossing several true stage
    boundaries that the once-per-round boundary check would then miss."""

    def __init__(self, mode, fault_lib, stage_len, rng, recal_step=None, n_stages=10, site_id=0):
        self.mode, self.fault_lib, self.stage_len, self.rng = mode, fault_lib, stage_len, rng
        self.stage_order = rng.integers(0, len(fault_lib), size=n_stages)
        self.t = 0
        self.recal_step = recal_step
        self.site_id = site_id  # for deterministic eval-batch seeding, not id(self)
        self._round_fault = None
        self._round_h = None
        self._round_stage_idx = None

    def stage_index_at(self, t):
        return min(t // self.stage_len, len(self.stage_order) - 1)

    def stage_index(self):
        return self.stage_index_at(self.t)

    def fault_at(self, stage_idx):
        return self.fault_lib[self.stage_order[stage_idx]]

    def advance_round(self):
        """Call exactly once per communication round. Advances stream time,
        fixes the fault/health-indicator this round's minibatches will use,
        and returns (h, stage_idx, fault_type_id)."""
        stage_idx = self.stage_index()
        fault = self.fault_at(stage_idx)
        t_in_stage = (self.t % self.stage_len) / self.stage_len
        h = health_curve(t_in_stage, fault["shape"], fault["rate"]) + self.rng.normal(0, 0.02)
        if self.recal_step is not None and self.t == self.recal_step:
            h = max(0.0, h - 0.4)
        self._round_fault, self._round_h, self._round_stage_idx = fault, h, stage_idx
        self.t += 1
        return float(np.clip(h, 0, 1)), stage_idx, int(self.stage_order[stage_idx])

    def minibatch(self, n=32):
        """Fresh (X, y) pair from THIS round's fixed fault, callable many
        times per round without advancing time."""
        fault = self._round_fault
        X = self.rng.normal(0, 1.0, size=(n, INPUT_DIM))
        z = phi(X) @ fault["A"]
        z = (z - z.mean()) / (z.std() + 1e-6)
        if self.mode == "classification":
            # y depends on X (via z, this fault's own mapping), NOT on the
            # scalar health indicator h alone -- h is a single number shared
            # by the whole batch and carries no per-sample information, so a
            # label built only from h would be unlearnable from X (AUC stuck
            # at chance regardless of training). A median-split of z gives a
            # balanced, genuinely X-dependent binary label tied to the
            # CURRENT fault's own decision boundary, with h staying purely a
            # separate drift-detection signal for DTTS (Section 4.3), not
            # the classification target itself -- realistic, since a
            # machine's raw acoustic sample and its slow-moving health trend
            # are related but not the same signal.
            y = (z > 0).astype(float)
            y = np.where(self.rng.uniform(size=n) < 0.05, 1 - y, y)  # label noise
        else:
            y = z + self.rng.normal(0, 0.05, size=z.shape)
        return X, y

    def eval_on_stage(self, stage_idx, n=200, seed_bump=0):
        """Deterministic held-out eval batch for a SPECIFIC (possibly past)
        stage's fault mapping, independent of current stream time. Seeded
        from `site_id` (not Python's per-process-randomized `hash()` on
        `id(self)`), so results are reproducible across runs/processes."""
        fault = self.fault_at(stage_idx)
        rng2 = np.random.default_rng((self.site_id, stage_idx, seed_bump))
        X = rng2.normal(0, 1.0, size=(n, INPUT_DIM))
        z = phi(X) @ fault["A"]
        z = (z - z.mean()) / (z.std() + 1e-6)
        if self.mode == "classification":
            y = (z > 0).astype(float)  # same X-dependent rule as minibatch(), noise-free for eval
        else:
            y = z
        return X, y


class PageHinkley:
    def __init__(self, delta=0.01, lam=1.2):
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
        triggered = ph > self.lam
        if triggered:
            self.reset()
        return triggered, ph


def sparsity_target(device_class, rho_global, s0=0.45, gamma=0.6, smin=0.25, smax=0.70):
    cmax_mean = np.mean(list(CMAX.values()))
    s = s0 * (CMAX[device_class] / cmax_mean) * (1 + gamma * rho_global)
    return float(np.clip(s, smin, smax))


def signature(h_trace):
    h = np.array(h_trace) if len(h_trace) > 0 else np.array([0.0])
    t = np.linspace(0, 1, len(h))
    slope = np.polyfit(t, h, 1)[0] if len(h) > 1 else 0.0
    curv = np.polyfit(t, h, 2)[0] if len(h) > 2 else 0.0
    return np.array([h.mean(), h.std(), slope, curv, h.min(), h.max()])


def cos_sim(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(np.dot(a, b) / (na * nb)) if na > 0 and nb > 0 else 0.0


def payload_bytes(n_params_free, kappa=0.5, b_q=4):
    return int(np.ceil(kappa * n_params_free) * b_q / 8)


METHODS = ["fedavg", "fedavg-ewc", "fedcat-external", "fedtwin-cl-no-ctfr", "fedtwin-cl"]


def _eval_metric(theta, X, y, mode, task_mask=None):
    """Higher is always better: negative MSE for regression, AUC for
    classification -- matches the direction convention used throughout.
    `task_mask`, if given, restricts inference to a specific committed
    task's own filters (Eq. 7 of [5]); see the note on `forward` above."""
    if mode == "classification":
        pred, _ = forward(theta, X, mode, task_mask=task_mask)
        return auc_score(y, pred.ravel())
    return -loss_fn(theta, X, y, mode, task_mask=task_mask)


def run_benchmark(benchmark_name, mode, n_rounds, local_steps, rng_seed,
                   recal_sites=None, stage_len=6, lam_ph=0.7, delta_ph=0.01,
                   tau_ctfr=0.9, kappa=0.5, b_q=4, s0=0.45, gamma=0.6, lr=0.01,
                   ewc_lambda=1.0):
    """Runs all methods on one benchmark, returns a list of flat records
    matching generate_result_figures.py's schema (plus a few extra fields).

    `stage_len` is in ROUNDS (one stream tick == one round, via
    `stream.advance_round()`), NOT in local SGD steps."""
    master_rng = np.random.default_rng(rng_seed)
    fault_lib = make_fault_library(master_rng)
    n_sites = len(DEVICE_CLASSES)
    recal_sites = recal_sites or []

    records = []

    for method in METHODS:
        # Python's built-in hash() is randomized per-process (PYTHONHASHSEED)
        # unless disabled -- using it here would make every method's
        # backbone initialization silently non-reproducible run-to-run, AND
        # give methods uncontrolled, unrelated initial weights instead of a
        # fair comparison. METHODS.index() is deterministic and stable.
        rng = np.random.default_rng(rng_seed * 1000 + METHODS.index(method))
        streams, thetas, phs, h_history, commit_history = [], [], [], [], []
        theta_global = init_backbone(rng)
        global_registry = np.zeros(C_SHARED, dtype=bool)  # True = frozen fleet-wide
        signature_registry = []  # list of (site_id, task_idx, fault_type_id, sig)
        ctfr_flags = []
        # ground-truth (evaluation-only) bookkeeping for real FPS / FBWT:
        task_eval_at_commit = [dict() for _ in range(n_sites)]  # {stage_idx: metric}
        task_masks = [dict() for _ in range(n_sites)]  # {stage_idx: committed-filter boolean mask}

        for i, dc in enumerate(DEVICE_CLASSES):
            site_rng = np.random.default_rng(rng_seed * 7919 + i)
            recal = 10 if i in recal_sites else None
            streams.append(SiteStream(mode, fault_lib, stage_len, site_rng, recal_step=recal, site_id=i))
            thetas.append({k: v.copy() for k, v in theta_global.items()})
            phs.append(PageHinkley(delta=delta_ph, lam=lam_ph))
            h_history.append([])
            commit_history.append(0)

        use_masking = method in ("fedcat-external", "fedtwin-cl-no-ctfr", "fedtwin-cl")
        use_ewc = method == "fedavg-ewc"
        use_ctfr = method == "fedtwin-cl"
        boundary_source = "oracle" if method == "fedcat-external" else "dtts"

        ewc_star = [None] * n_sites
        ewc_fisher = [None] * n_sites

        cum_bytes = [0] * n_sites
        true_stage_prev = [0] * n_sites

        for r in range(n_rounds):
            rho_global = global_registry.mean()
            uploads = []

            for i, dc in enumerate(DEVICE_CLASSES):
                cmax = CMAX[dc]
                site_free_mask = ~global_registry.copy()
                site_free_mask[cmax:] = False  # width truncation

                theta = thetas[i]
                stream = streams[i]
                theta_before = {k: v.copy() for k, v in theta.items()}

                # -- advance stream ONE tick this round, fixing the fault --
                h, true_stage_now, fault_type_id = stream.advance_round()
                h_history[i].append(h)

                # -- task-boundary decision (DTTS or oracle ground truth),
                #    evaluated BEFORE this round's local training, so any
                #    commit below scores/protects the model as it stood at
                #    the END of the OLD task, not after new-task data has
                #    already started pulling weights away from it. --
                triggered = False
                if boundary_source == "dtts":
                    trig, _ = phs[i].update(h)
                    triggered = trig
                elif boundary_source == "oracle":
                    triggered = true_stage_now != true_stage_prev[i]
                old_stage = true_stage_prev[i]
                stage_changed = true_stage_now != old_stage  # GROUND TRUTH, independent of DTTS/oracle noise
                true_stage_prev[i] = true_stage_now

                # -- mask commit: compute WHICH filters would be committed
                #    for the OLD stage (if triggered), BEFORE snapshotting
                #    eval-at-commit, so the snapshot and the later final-
                #    round eval use the exact same task-specific mask (Eq. 7
                #    of [5]) -- an apples-to-apples comparison. Without this,
                #    "eval-at-commit" would score the OLD task through the
                #    FULL, still-changing network (since no mask exists
                #    yet), while "eval-at-final" would score it through only
                #    its own frozen slice, making every masking method look
                #    like it forgot even though its committed filters never
                #    moved -- the two numbers would not be measuring the
                #    same thing. --
                newly_committed = None
                if use_masking and triggered and site_free_mask.any():
                    s_n = sparsity_target(dc, rho_global, s0=s0, gamma=gamma)
                    free_idx = np.where(site_free_mask)[0]
                    scores = (np.abs(theta_before["W1"][:, free_idx]).sum(0)
                              + np.abs(theta_before["W2"][free_idx]))
                    n_commit = max(1, int(np.ceil(s_n * len(free_idx))))
                    top = free_idx[np.argsort(-scores)[:n_commit]]
                    newly_committed = np.zeros(C_SHARED, dtype=bool)
                    newly_committed[top] = True
                    # ACCUMULATE (OR), don't overwrite: DTTS can trigger
                    # more than once while the true stage is still
                    # `old_stage` (over-segmentation, Remark 5.1). Each such
                    # commit protects a genuinely different slice of columns
                    # for what is, in ground truth, still the same task;
                    # overwriting the tracked mask on a later over-trigger
                    # would silently drop the earlier commit's columns from
                    # evaluation even though they remain frozen in the
                    # registry, undercounting how much of that task the
                    # mechanism actually protected.
                    task_masks[i][old_stage] = task_masks[i].get(
                        old_stage, np.zeros(C_SHARED, dtype=bool)) | newly_committed

                # -- record eval-at-commit for the OLD stage on every GROUND
                #    TRUTH transition (not on `triggered`, which for DTTS can
                #    fire early/late/multiple-times within one true stage --
                #    using the algorithm's own noisy trigger here would let a
                #    premature DTTS commit snapshot an under-trained model
                #    and then credit the rest of that same true stage's
                #    training as "recovery," contaminating the forgetting
                #    measurement with a bookkeeping artifact rather than a
                #    real effect). theta_before is "freshly finished
                #    training on it," the correct snapshot point. Masking
                #    methods use their just-computed task mask (if any was
                #    committed this exact round) or fall back to unmasked
                #    if this stage never got a dedicated commit (an
                #    under-segmentation artifact, correctly left unmasked
                #    since nothing protected it). --
                if stage_changed and old_stage not in task_eval_at_commit[i]:
                    Xc, yc = stream.eval_on_stage(old_stage, n=200)
                    m = task_masks[i].get(old_stage) if use_masking else None
                    task_eval_at_commit[i][old_stage] = _eval_metric(theta_before, Xc, yc, mode, task_mask=m)

                if newly_committed is not None:
                    site_free_mask = site_free_mask & ~newly_committed  # frozen for THIS round's training too

                    # Commit = DIRECT WRITE of this site's just-finished,
                    # well-trained values into theta_global at exactly the
                    # committed columns (not a delta-averaged update onto a
                    # stale baseline: theta_global was never kept in sync
                    # with a site's private free-space training, since that
                    # space is intentionally un-pooled -- see note below --
                    # so "add my delta to theta_global's old value" would add
                    # a real change on top of an unrelated, meaningless
                    # baseline). Registry updated immediately, not deferred,
                    # so a later site THIS SAME round already sees these
                    # columns as taken and cannot also claim them.
                    top = np.where(newly_committed)[0]
                    theta_global["W1"][:, top] = theta_before["W1"][:, top]
                    theta_global["b1"][top] = theta_before["b1"][top]
                    theta_global["W2"][top] = theta_before["W2"][top]
                    global_registry |= newly_committed

                    commit_history[i] += 1
                    if use_ctfr:
                        # Same window length (last 4 observations) as the
                        # QUERY signature below -- comparing a signature
                        # built from a whole task's history against one
                        # built from a short recent snippet would compare
                        # different kinds of summaries and inflate spurious
                        # similarity.
                        sig = signature(h_history[i][-4:])
                        old_fault_id = int(stream.stage_order[old_stage])
                        signature_registry.append((i, commit_history[i], old_fault_id, sig))
                elif use_ewc and triggered:
                    ewc_star[i] = {k: v.copy() for k, v in theta_before.items()}
                    # diagonal Fisher proxy: mean squared gradient of the OLD
                    # task, evaluated at theta_before on a fresh OLD-task
                    # calibration batch (standard cheap EWC approximation).
                    Xc, yc = stream.eval_on_stage(old_stage, n=128, seed_bump=999)
                    _, cache_c = forward(theta_before, Xc, mode)
                    gc = backward(theta_before, cache_c, yc, mode)
                    ewc_fisher[i] = {k: (gc[k] ** 2) for k in theta_before}
                    commit_history[i] += 1

                # -- local training on this round's fixed (possibly new) task --
                for _ in range(local_steps):
                    X, y = stream.minibatch(n=32)
                    _, cache = forward(theta, X, mode)
                    g = backward(theta, cache, y, mode)
                    g = clip_grads(g)
                    if use_masking:
                        g = mask_grads(g, site_free_mask)
                    l2 = None
                    if use_ewc and ewc_star[i] is not None:
                        l2 = {k: ewc_lambda * ewc_fisher[i][k] * (theta[k] - ewc_star[i][k])
                              for k in theta}
                    # "b2" is a single shared output bias, not tied to any
                    # specific filter/task -- for masking methods it is left
                    # fixed at its zero init and NEVER trained, since an
                    # unmasked shared bias would silently drift with every
                    # later task's training and contaminate every earlier,
                    # otherwise perfectly-frozen task's masked output (this
                    # was the actual remaining source of nonzero "forgetting"
                    # even after per-task masked inference was added: W1/
                    # b1/W2 were provably frozen bit-for-bit, but b2 was
                    # not). Non-masking baselines have no per-task isolation
                    # to protect in the first place, so b2 trains normally.
                    update_keys = ("W1", "b1", "W2") if use_masking else theta.keys()
                    for k in update_keys:
                        step = g[k] + (l2[k] if l2 is not None else 0.0)
                        theta[k] = theta[k] - lr * step

                # -- CTFR relevance check (against signatures from OTHER sites only) --
                if use_ctfr and len(h_history[i]) >= 3:
                    phi_now = signature(h_history[i][-4:])
                    best_sim, best_match = 0.0, None
                    for (sid, tidx, ftype, sig) in signature_registry:
                        if sid == i:
                            continue
                        sim = cos_sim(phi_now, sig)
                        if sim > best_sim:
                            best_sim, best_match = sim, (sid, tidx, ftype)
                    if best_sim > tau_ctfr:
                        ctfr_flags.append({"site": i, "round": r, "sim": best_sim,
                                            "match": best_match, "own_stage_at_flag": true_stage_now})

                # -- compression / payload accounting (cost model only; see
                #    note above on why free-space columns are not literally
                #    pooled across sites in this synthetic multi-unrelated-
                #    task construction -- the BYTES this would cost if they
                #    were transmitted, per Eq. 6/18 of the manuscript, are
                #    still charged here, since that cost is real regardless
                #    of whether cross-site averaging happens on the far end) --
                delta = {k: theta[k] - theta_before[k] for k in theta}
                if use_masking:
                    n_free_params = int(site_free_mask.sum()) * (INPUT_DIM + 2)
                else:
                    n_free_params = theta["W1"].size + theta["b1"].size + theta["W2"].size + theta["b2"].size
                pbytes = payload_bytes(n_free_params, kappa=kappa, b_q=b_q) if use_masking \
                    else int(n_free_params * 32 / 8)
                cum_bytes[i] += pbytes
                tx_energy = pbytes * E_TX[dc]

                uploads.append({"site": i, "delta": delta, "w": 32,
                                 "bytes": pbytes, "cum_bytes": cum_bytes[i], "energy": tx_energy})

            # -- end-of-round sync --
            if use_masking:
                # Commits already wrote directly into theta_global and OR'd
                # into global_registry immediately when they happened above
                # (so same-round collisions across sites are impossible: a
                # later site in this round's loop already sees an updated
                # registry). Here we just push the current frozen state out
                # to every site, catching sites processed BEFORE a later
                # site's commit this same round.
                for i in range(n_sites):
                    thetas[i]["W1"][:, global_registry] = theta_global["W1"][:, global_registry]
                    thetas[i]["b1"][global_registry] = theta_global["b1"][global_registry]
                    thetas[i]["W2"][global_registry] = theta_global["W2"][global_registry]
            else:
                # Standard dense FedAvg: every site trains the SAME shared
                # model, so a plain sample-weighted average of full deltas
                # is the correct (and, per Table 2/3, deliberately weak)
                # baseline behavior.
                tot_w = sum(u["w"] for u in uploads)
                accum = {k: sum(u["delta"][k] * u["w"] for u in uploads) / tot_w for k in theta_global}
                for k in theta_global:
                    theta_global[k] += accum[k]
                for i in range(n_sites):
                    thetas[i] = {k: v.copy() for k, v in theta_global.items()}

            # -- per-round record: CURRENT task's metric (drives the
            #    convergence-vs-round figure; forgetting is NOT visible
            #    here by construction, see the end-of-run FPS/FBWT block
            #    below for that) --
            for i, dc in enumerate(DEVICE_CLASSES):
                Xe, ye = streams[i].eval_on_stage(true_stage_prev[i], n=200, seed_bump=r)
                metric = _eval_metric(thetas[i], Xe, ye, mode)
                records.append({
                    "benchmark": benchmark_name, "method": method, "site_id": i,
                    "device_class": dc, "round": r + 1, "raw_metric": metric,
                    "uplink_bytes": cum_bytes[i], "tx_energy_mj": uploads[i]["energy"],
                    "lambda_ph": lam_ph, "commit_count": commit_history[i],
                    "rho_global": float(global_registry.mean()),
                })

        # -- end-of-run FPS / FBWT: re-evaluate the FINAL model on EVERY
        #    stage each site ever passed through, compared against that
        #    stage's eval-at-commit snapshot. This is the metric that can
        #    actually show catastrophic forgetting (or its absence). --
        for i, dc in enumerate(DEVICE_CLASSES):
            fbwt_vals, fps_vals = [], []
            for stage_idx, metric_at_commit in task_eval_at_commit[i].items():
                Xf, yf = streams[i].eval_on_stage(stage_idx, n=200)
                m = task_masks[i].get(stage_idx) if use_masking else None
                metric_final = _eval_metric(thetas[i], Xf, yf, mode, task_mask=m)
                fps_vals.append(metric_final)
                fbwt_vals.append(metric_final - metric_at_commit)
            records.append({
                "benchmark": benchmark_name, "method": method, "site_id": i,
                "device_class": dc, "round": n_rounds, "fleet_perf_score": float(np.mean(fps_vals)),
                "fleet_backward_transfer": float(np.mean(fbwt_vals)), "n_tasks_seen": len(fps_vals),
                "final_uplink_bytes": cum_bytes[i], "commit_count": commit_history[i],
                "is_summary_record": True,
            })

        for f in ctfr_flags:
            records.append({
                "benchmark": benchmark_name, "method": method, "site_id": f["site"],
                "device_class": DEVICE_CLASSES[f["site"]], "round": f["round"],
                "ctfr_event": True, "ctfr_sim": f["sim"], "ctfr_match": f["match"],
            })

    return records


def centralized_upper_bound(benchmark_name, mode, stage_len, rng_seed, recal_sites=None,
                             steps=400, lr=0.05, width=256):
    """Centralized Task-Isolated baseline (Table 3's upper-bound row): one
    fresh, full-capacity, single-task model per (site, stage) pair the main
    run would have encountered, no federation and no capacity sharing.
    Independent of run_benchmark's round loop (no masking, no aggregation),
    since isolation is the whole point -- run separately, not folded into
    the main method comparison."""
    master_rng = np.random.default_rng(rng_seed)
    fault_lib = make_fault_library(master_rng)
    records = []
    for i, dc in enumerate(DEVICE_CLASSES):
        site_rng = np.random.default_rng(rng_seed * 7919 + i)
        recal = 10 if (recal_sites and i in recal_sites) else None
        stream = SiteStream(mode, fault_lib, stage_len, site_rng, recal_step=recal, site_id=i)
        # advance through the same 20-round schedule so stage_order draws match
        for _ in range(20):
            stream.advance_round()
        n_stages_seen = stream.stage_index() + 1

        metrics = []
        for stage_idx in range(n_stages_seen):
            rng = np.random.default_rng(rng_seed * 500 + i * 20 + stage_idx)
            theta = init_backbone(rng, width=width)
            fault = stream.fault_at(stage_idx)
            for _ in range(steps):
                X = rng.normal(0, 1.0, size=(32, INPUT_DIM))
                z = phi(X) @ fault["A"]
                z = (z - z.mean()) / (z.std() + 1e-6)
                y = (z > 0).astype(float) if mode == "classification" else z + rng.normal(0, 0.05, size=z.shape)
                _, cache = forward(theta, X, mode)
                g = clip_grads(backward(theta, cache, y, mode))
                for k in theta:
                    theta[k] = theta[k] - lr * g[k]
            Xf, yf = stream.eval_on_stage(stage_idx, n=200)
            metrics.append(_eval_metric(theta, Xf, yf, mode))

        records.append({
            "benchmark": benchmark_name, "method": "centralized", "site_id": i, "device_class": dc,
            "round": 20, "fleet_perf_score": float(np.mean(metrics)), "fleet_backward_transfer": None,
            "n_tasks_seen": len(metrics), "commit_count": None, "is_summary_record": True,
        })
    return records


BENCHMARK_CONFIGS = {
    # Multivariate telemetry RUL-style regression; a subset of sites get a
    # mid-run operating-condition switch, standing in for the FD002/FD004
    # multi-condition C-MAPSS sub-datasets and the "sensor recalibration
    # event" drift scenario from the manuscript.
    "fed-twin-cmapss": dict(mode="regression", stage_len=6, recal_sites=[6, 7, 8, 9, 10]),
    # Progressive bearing wear affects every site (no recalibration event),
    # standing in for the FEMTO-ST / PHM 2012 bearing run-to-failure design.
    "fed-twin-femto": dict(mode="regression", stage_len=5, recal_sites=None),
    # Binary anomaly detection; a third of sites get an SNR/gain
    # recalibration event mid-run, standing in for MIMII.
    "fed-twin-mimii": dict(mode="classification", stage_len=6, recal_sites=list(range(0, 24, 3))),
}


def run_all_benchmarks(n_rounds=20, local_steps=40, rng_seed=1, **kwargs):
    """Runs all three benchmark configs, all five methods PLUS the
    centralized upper bound, returns one flat list of records directly
    compatible with generate_result_figures.py's schema (after a light
    field-name adaptation, see the notebook)."""
    all_records = []
    for bench_name, cfg in BENCHMARK_CONFIGS.items():
        recs = run_benchmark(bench_name, n_rounds=n_rounds, local_steps=local_steps,
                              rng_seed=rng_seed, **{**cfg, **kwargs})
        all_records.extend(recs)
        all_records.extend(centralized_upper_bound(
            bench_name, cfg["mode"], cfg["stage_len"], rng_seed, recal_sites=cfg["recal_sites"]))
    return all_records


if __name__ == "__main__":
    recs = run_benchmark("spike-test", "regression", n_rounds=20, local_steps=40, rng_seed=1)
    print(f"{len(recs)} records")
    for m in METHODS:
        summ = [r for r in recs if r.get("method") == m and r.get("is_summary_record")]
        fps = np.mean([r["fleet_perf_score"] for r in summ])
        fbwt = np.mean([r["fleet_backward_transfer"] for r in summ])
        ntasks = np.mean([r["n_tasks_seen"] for r in summ])
        commits = np.mean([r["commit_count"] for r in summ])
        print(f"{m}: FPS={fps:.3f} FBWT={fbwt:+.4f} mean_tasks_seen={ntasks:.2f} mean_commits={commits:.2f}")
    n_flags = sum(1 for r in recs if r.get("ctfr_event"))
    print(f"CTFR flags raised: {n_flags}")
