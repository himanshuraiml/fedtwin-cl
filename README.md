# FedTwin-CL

Federated continual learning for industrial IoT digital twins under progressive concept drift.

FedTwin-CL is a federated continual learning engine for fleets of industrial digital twins (turbofan engines, bearing-equipped machinery, acoustic-monitored equipment) that must keep adapting to non-stationary degradation without pooling raw sensor data across sites. It combines gradient-masked structured-sparse local training with a server-maintained global frozen-mask registry, giving a proved cross-node zero-forgetting guarantee, and adds two mechanisms on top of that engine:

- **Drift-Triggered Task Segmentation (DTTS)** — a Page-Hinkley change-point test that lets each site commit task boundaries from its own health-indicator trajectory, instead of relying on an externally supplied task label.
- **Cross-Twin Fault Relevance (CTFR) scoring** — a bounded-dimension fault signature broadcast on every commit, letting other sites raise an early-warning flag before their own drift detector independently triggers, without any raw sensor data leaving its site of origin.

This repository contains the full implementation, evaluated end-to-end against three real industrial datasets: **NASA C-MAPSS** turbofan degradation, the **IEEE PHM 2012 / FEMTO-ST PRONOSTIA** bearing run-to-failure dataset, and **MIMII** industrial acoustic recordings.

## Repository layout

```
real_data/
  preprocess_cmapss.py        preprocessing for Fed-Twin-CMAPSS
  preprocess_femto.py         preprocessing for Fed-Twin-FEMTO
  preprocess_femto_longseq.py preprocessing variant for the longer/finer drift-sequence experiment
  preprocess_mimii.py         preprocessing for Fed-Twin-MIMII
  real_pipeline_cmapss.py     main pipeline: mask registry + DTTS + CTFR on Fed-Twin-CMAPSS
  real_pipeline_femto.py      main pipeline on Fed-Twin-FEMTO
  real_pipeline_mimii.py      main pipeline on Fed-Twin-MIMII
  real_fedprox_centralized.py FedProx and Centralized Task-Isolated baselines, all three benchmarks
  real_compression_matched.py compression-matched FedAvg baseline (isolates masking from generic compression)
  real_capacity_sweep.py      backbone-width capacity sweep, all three benchmarks
  real_dtts_sweep.py          DTTS threshold (lambda_PH) sensitivity sweep
  real_alt_detector.py        Page-Hinkley vs. KSWIN drift-detector comparison
  real_ctfr_leadtime.py       CTFR confirmed-precision and early-warning lead time
  real_gamma_sweep.py         heterogeneity-scaling (gamma) sensitivity sweep
  real_longer_sequence.py     longer, finer-grained drift-sequence experiment
  *_real_results.json         result files produced by the scripts above

code/
  fedtwin_harness.py          from-scratch synthetic-data implementation (validation spike, predates the real pilot)
  spike_extended.py           extended synthetic sweeps used to de-risk the real pilot

generate_*.py                 figure-generation scripts, one family per experiment batch
figures/                      generated figures
colab_pilot_spike.ipynb       notebook version of the synthetic validation spike

EXPERIMENT_PROCEDURE.md       full experimental history: what was run, in what order, and every
                               implementation bug found and fixed along the way (17 total)
```

## Requirements

- Python 3.10+
- PyTorch 2.x (CPU or MPS/CUDA)
- numpy, pandas, scipy, matplotlib

```bash
pip install torch numpy pandas scipy matplotlib
```

## Reproducing the results

### 1. Get the raw data

The raw datasets are not redistributed in this repository (large, and already freely available from their original sources):

- **NASA C-MAPSS**: NASA Prognostics Center of Excellence data repository, "Turbofan Engine Degradation Simulation Data Set." Place under `real_data/cmapss/`.
- **IEEE PHM 2012 / FEMTO-ST PRONOSTIA**: distributed via the PHM Society (all 17 public run-to-failure bearing traces, `Learning_set` + `Full_Test_Set`). Place under `real_data/femto/`.
- **MIMII**: Zenodo record [3384388](https://zenodo.org/records/3384388). This repository's experiments use the valve machine type at 6 dB and 0 dB SNR. Place under `real_data/mimii/`.

Each `preprocess_*.py` script documents its expected input directory layout in its own module docstring.

### 2. Preprocess

```bash
cd real_data
python preprocess_cmapss.py
python preprocess_femto.py
python preprocess_mimii.py
```

### 3. Run the main pipeline (mask registry + DTTS + CTFR, all baselines)

```bash
python real_pipeline_cmapss.py
python real_pipeline_femto.py
python real_pipeline_mimii.py
```

Each produces `<benchmark>_real_results.json`, reporting fleet performance, Fleet Backward Transfer, and the protected-vs-missed zero-forgetting decomposition for every method (FedAvg, FedAvg+EWC, oracle-boundary engine, FedTwin-CL without CTFR, and full FedTwin-CL).

### 4. Run the additional experiments

```bash
python real_fedprox_centralized.py --benchmarks cmapss,femto,mimii
python real_compression_matched.py --benchmarks cmapss,femto,mimii
python real_capacity_sweep.py --benchmarks cmapss,femto,mimii
python real_dtts_sweep.py
python real_alt_detector.py
python real_gamma_sweep.py
python real_longer_sequence.py
```

### 5. Regenerate figures

```bash
cd ..
python generate_real_result_figures.py
python generate_real_sweep_figures.py
python generate_new_experiments_figures.py
```

## Notes on reproducibility

`EXPERIMENT_PROCEDURE.md` documents every implementation bug found while building this pipeline (17 in total) and the empirical sanity check used to catch them: decomposing measured backward transfer by whether a task actually received a mask commit, and asserting that protected tasks show exactly zero backward transfer while missed tasks show real forgetting. A re-implementation that reproduces any of the symptoms listed there — forgetting where none should exist, a metric blowup isolated to specific sites, a suspiciously early exact-zero result, or non-reproducible runs — should consult that list first.

## License

MIT — see `LICENSE`.
