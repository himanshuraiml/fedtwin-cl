"""
Regenerates a finer-grained, longer-horizon variant of the Fed-Twin-FEMTO
site set for the "longer, genuinely progressive drift sequence" study
(EXPERIMENT_PROCEDURE.md camera-ready item, reviewer gap item 6): 8
ground-truth stages (instead of 3) via the same TIME-tercile-style equal-
duration split (generalized from thirds to eighths), reusing
preprocess_femto.py's own bearing-loading and feature-extraction code
unchanged. Saved to a SEPARATE file (femto_sites_longseq.npz) so the
original 3-stage femto_sites.npz used by every existing result stays
untouched.
"""
import os
import numpy as np

from preprocess_femto import BEARINGS, LEARNING_DIR, OUT_DIR, build_bearing_record

N_STAGES_FINE = 8

if __name__ == "__main__":
    rng = np.random.default_rng(1)  # identical site-assignment seed/order as preprocess_femto.py
    assignment = list(range(len(BEARINGS)))
    extra = list(rng.choice(len(BEARINGS), size=24 - len(BEARINGS), replace=False))
    assignment += extra
    rng.shuffle(assignment)

    sites = []
    for site_id, bidx in enumerate(assignment):
        folder, name = BEARINGS[bidx]
        print(f"site {site_id}: {name} ({'Learning' if folder == LEARNING_DIR else 'FullTest'}), n_stages={N_STAGES_FINE}")
        sites.append(build_bearing_record(folder, name, site_id, n_stages=N_STAGES_FINE))

    lengths = [s["windows"].shape[0] for s in sites]
    print(f"Site sequence lengths: min={min(lengths)} max={max(lengths)} mean={np.mean(lengths):.1f}")

    np.savez(os.path.join(OUT_DIR, "femto_sites_longseq.npz"),
              sites=np.array(sites, dtype=object), allow_pickle=True)
    print(f"Saved {len(sites)} sites ({N_STAGES_FINE}-stage ground truth) to {OUT_DIR}/femto_sites_longseq.npz")
