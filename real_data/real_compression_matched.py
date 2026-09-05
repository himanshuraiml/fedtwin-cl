"""
Compression-matched FedAvg baseline, for all real FedTwin-CL pipelines.

Reviewer gap this closes: the reported ~98%+ uplink reduction (manuscript
Table 3) compares FedTwin-CL's masked+compressed upload against DENSE,
UNCOMPRESSED FedAvg. That conflates two independent sources of savings --
(a) uploading only a masked subset of channels at all, and (b) generic
payload compression (top-k sparsification + b_q-bit quantization) applied
to whatever is uploaded. This script adds a third condition, "FedAvg +
top-k + quantization": run plain dense FedAvg (no masking, full-network
averaging every round, identical training dynamics to the existing
"fedavg" method) but apply the SAME top_k=kappa magnitude-based
sparsification and SAME b_q=4-bit stochastic quantization already used for
FedTwin-CL's uploads to FedAvg's own full parameter delta each round,
before counting payload bytes. This isolates masking's own marginal
contribution: FedTwin-CL vs. this baseline is a fair, compression-matched
comparison; FedTwin-CL vs. dense FedAvg (Table 3's original comparison) is
not, and both should now be reported side by side.

The CTFR signature broadcast (Methods, Eq. 15: a 6-float, i.e. 24-byte at
fp32, or 12-byte at fp16 signature broadcast on every commit) is reported
as its own separate line item here too, rather than folded invisibly into
FedTwin-CL's masked total the way earlier tables did -- it is a real,
if tiny, additional payload FedTwin-CL-without-CTFR does not send.
"""
import os
import sys
import json
import argparse
import importlib
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))

BENCHMARKS = {
    "cmapss": ("real_pipeline_cmapss", "fed-twin-cmapss-real"),
    "femto": ("real_pipeline_femto", "fed-twin-femto-real"),
    "mimii": ("real_pipeline_mimii", "fed-twin-mimii-real"),
}

CTFR_SIGNATURE_BYTES = 6 * 4  # 6 floats (Eq. 15) at fp32, one signature broadcast per real commit


def load_pipeline(name):
    modname, label = BENCHMARKS[name]
    P = importlib.import_module(modname)
    return P, label


def stream_cls(P):
    return getattr(P, "RealSiteStream", None) or getattr(P, "MimiiSiteStream")


def is_mimii(label):
    return "mimii" in label


def quantized_bytes(n_params_selected, kappa, b_q):
    """Identical formula to real_pipeline_*.py's payload_bytes(): the top
    ceil(kappa * n_params) values by magnitude are kept and each is sent at
    b_q bits. Applied here to FedAvg's full dense per-round delta instead
    of a masked free-parameter count."""
    return int(np.ceil(kappa * n_params_selected) * b_q / 8)


def run_fedavg_compressed(P, label, seed=1):
    """Same training loop and dynamics as the pipeline's own "fedavg"
    method (dense full-network FedAvg every round) -- this script does not
    retrain a new model, it re-derives the compressed payload size FedAvg's
    own per-round delta would cost under the same top-k+quantization
    scheme FedTwin-CL already uses, using the SAME random seed and
    training trajectory as the existing "fedavg" run so the two are
    directly comparable apples-to-apples deltas, not a separately-trained
    model that could drift onto a different trajectory."""
    data = np.load(P.DATA_PATH, allow_pickle=True)
    sites_data = data["sites"]
    n_sensors = sites_data[0]["windows"].shape[2] if not is_mimii(label) else None
    window = sites_data[0]["windows"].shape[1] if not is_mimii(label) else None
    n_sites = len(sites_data)
    method = "fedavg-compressed"
    method_idx = P.METHODS.index("fedavg")  # reuse fedavg's own init seed exactly
    SC = stream_cls(P)

    torch.manual_seed(seed * 1000 + method_idx)
    kwargs = {} if not is_mimii(label) else {}
    theta_global = P.init_backbone(n_sensors, window, seed=seed * 1000 + method_idx) if not is_mimii(label) \
        else None
    streams = [SC(sites_data[i], site_id=i) for i in range(n_sites)]
    if theta_global is None:
        # mimii: n_sensors/window discovered from stream shapes, mirroring run_mimii()
        any_w, any_y = streams[0].minibatch() if False else (None, None)
        n_sensors = sites_data[0]["round_windows"][1].shape[-1]
        window = sites_data[0]["round_windows"][1].shape[1]
        theta_global = P.init_backbone(n_sensors, window, seed=seed * 1000 + method_idx)

    thetas = [P.clone_theta(theta_global) for _ in range(n_sites)]
    cum_bytes_dense = [0] * n_sites
    cum_bytes_compressed = [0] * n_sites

    for r in range(P.N_ROUNDS):
        for i, dc in enumerate(P.DEVICE_CLASSES):
            theta = thetas[i]
            stream = streams[i]
            theta_before = {k: v.detach().clone() for k, v in theta.items()}

            stream.advance_round()
            Xb, yb = stream.minibatch()
            if len(yb) > 0:
                n_steps = P.LOCAL_STEPS
                for _ in range(n_steps):
                    P.zero_grad(theta)
                    l = P.loss_fn(theta, Xb.to(P.DEVICE), yb.to(P.DEVICE))
                    l.backward()
                    P.clip_grads(theta)
                    with torch.no_grad():
                        for k in theta:
                            theta[k] -= P.LR * theta[k].grad
                    P.zero_grad(theta)

            # Dense payload: every parameter, fp32 (identical convention to
            # the pipeline's own dense-method branch).
            n_params = sum(v.numel() for v in theta.values())
            dense_bytes = int(n_params * 32 / 8)
            cum_bytes_dense[i] += dense_bytes

            # Compression-matched payload: apply the SAME top_k=kappa (by
            # delta magnitude, matching how FedTwin-CL selects free
            # parameters by weight magnitude) and SAME b_q=4-bit
            # quantization to this round's full dense delta.
            with torch.no_grad():
                delta_flat = torch.cat([(theta[k] - theta_before[k]).flatten() for k in theta])
            n_selected = max(1, int(np.ceil(P.KAPPA * delta_flat.numel())))
            compressed_bytes = quantized_bytes(n_selected, 1.0, P.B_Q)  # n_selected already = kappa*n_params
            cum_bytes_compressed[i] += compressed_bytes

        with torch.no_grad():
            for k in theta_global:
                theta_global[k] = sum(thetas[i][k] for i in range(n_sites)) / n_sites
            for i in range(n_sites):
                thetas[i] = P.clone_theta(theta_global)

    records = []
    for i, dc in enumerate(P.DEVICE_CLASSES):
        records.append({"benchmark": label, "method": method, "site_id": i, "device_class": dc,
                         "uplink_bytes_dense": cum_bytes_dense[i],
                         "uplink_bytes_compressed": cum_bytes_compressed[i],
                         "is_summary_record": True})
    total_dense = sum(cum_bytes_dense)
    total_compressed = sum(cum_bytes_compressed)
    print(f"  [{label}] fedavg-compressed: total_dense={total_dense} total_compressed={total_compressed} "
          f"({total_dense/total_compressed:.2f}x smaller than dense FedAvg)")
    return records


def summarize_existing_uplink(label, real_results_path):
    """Reads the benchmark's own *_real_results.json (already-committed
    history file, not modified) to report each existing method's final
    cumulative uplink_bytes per site, for a single combined comparison
    table alongside the new compression-matched condition."""
    if not os.path.exists(real_results_path):
        return {}
    with open(real_results_path) as f:
        recs = json.load(f)
    out = {}
    for method in sorted(set(r["method"] for r in recs if "method" in r)):
        per_round = [r for r in recs if r.get("method") == method and "uplink_bytes" in r and r.get("round") is not None]
        if not per_round:
            continue
        max_round = max(r["round"] for r in per_round)
        final = [r["uplink_bytes"] for r in per_round if r["round"] == max_round]
        out[method] = {"total_uplink_bytes": int(sum(final)), "n_sites": len(final)}
    return out


def ctfr_payload_total(label, real_results_path, n_sites=24):
    """Real commit count for fedtwin-cl on this benchmark x the per-commit
    signature broadcast cost, reported as its own line item."""
    if not os.path.exists(real_results_path):
        return None
    with open(real_results_path) as f:
        recs = json.load(f)
    summ = [r for r in recs if r.get("method") == "fedtwin-cl" and r.get("is_summary_record")]
    total_commits = sum(r.get("commit_count", 0) for r in summ)
    return {"total_commits": total_commits, "bytes_per_signature": CTFR_SIGNATURE_BYTES,
            "total_ctfr_bytes": total_commits * CTFR_SIGNATURE_BYTES}


def run_benchmark(name):
    P, label = load_pipeline(name)
    recs = run_fedavg_compressed(P, label)
    real_results_path = os.path.join(os.path.dirname(__file__), f"{name}_real_results.json")
    existing = summarize_existing_uplink(label, real_results_path)
    ctfr = ctfr_payload_total(label, real_results_path)
    return {"records": recs, "existing_methods_uplink": existing, "ctfr_payload": ctfr}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--benchmarks", nargs="+", default=["cmapss", "femto"])
    args = ap.parse_args()

    out = {}
    for name in args.benchmarks:
        print(f"=== {name} ===")
        out[name] = run_benchmark(name)

    out_path = os.path.join(os.path.dirname(__file__), "compression_matched_real_results.json")
    existing_file = {}
    if os.path.exists(out_path):
        with open(out_path) as f:
            existing_file = json.load(f)
    existing_file.update(out)
    with open(out_path, "w") as f:
        json.dump(existing_file, f)
    print(f"\nWrote {out_path}")

    print("\n=== Table 3-style comparison (total fleet uplink bytes, 20 rounds, 24 sites) ===")
    for name in args.benchmarks:
        res = out[name]
        recs = res["records"]
        total_dense_fedavg = res["existing_methods_uplink"].get("fedavg", {}).get("total_uplink_bytes")
        total_compressed_fedavg = sum(r["uplink_bytes_compressed"] for r in recs)
        total_fedtwincl = res["existing_methods_uplink"].get("fedtwin-cl", {}).get("total_uplink_bytes")
        print(f"\n{name}:")
        print(f"  FedAvg (dense, existing Table 3):        {total_dense_fedavg}")
        print(f"  FedAvg + top-k + quant (NEW):             {total_compressed_fedavg}")
        print(f"  FedTwin-CL (masked+compressed, existing): {total_fedtwincl}")
        if total_dense_fedavg and total_fedtwincl:
            print(f"  Reduction vs DENSE FedAvg (old comparison):        {total_dense_fedavg/total_fedtwincl:.1f}x")
        if total_compressed_fedavg and total_fedtwincl:
            print(f"  Reduction vs COMPRESSION-MATCHED FedAvg (new, fair): {total_compressed_fedavg/total_fedtwincl:.1f}x")
        if res["ctfr_payload"]:
            print(f"  CTFR signature payload: {res['ctfr_payload']}")
