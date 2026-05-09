# Reproducibility Guide

Steps to reproduce the headline numbers reported in the paper. On a 10-core
CPU laptop the full pipeline (preprocessing + feature extraction + cross-dataset
evaluation) takes ~3 hours; with cached features the training-only step takes
~25 minutes.

---

## 1. Environment

```bash
# Python 3.9-3.12 supported. Tested on 3.10.
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Pinned versions used to produce the reported numbers:
- Python 3.10.13
- NumPy 1.26
- SciPy 1.13
- MNE-Python 1.7
- PyTorch 2.2.1
- PyTorch Geometric 2.5.0
- scikit-learn 1.4

CPU is sufficient. No GPU required.

---

## 2. Smoke test (verify your environment is correct)

```bash
cd src
python3 smoke_test.py
```

Expected output ends with `ALL CHECKS PASSED`. This runs the full pipeline on
synthetic data in under one minute and verifies every component (feature
extraction, graph construction, training, inference). If this fails, stop and
fix the environment before running the real pipeline.

---

## 3. Download the three OpenNeuro datasets

```bash
cd src
python3 download_data.py --dataset all
```

This downloads ~9.6 GB into `data/raw/{UC,UNM,Iowa}/`. Datasets:
- UC San Diego: ds002778 (50 subjects)
- UNM: ds003490 (31 subjects)
- Iowa: ds004584 (149 subjects)

**Total: 230 subjects, balanced PD/HC.**

---

## 4. Run the cross-dataset evaluation

```bash
cd src
python3 run_full_training.py
```

This runs:
1. Load all 230 subjects (~5 min)
2. Preprocess: bandpass, notch, channel harmonization, epoching (~10 min)
3. Extract 13-dim node features and 32x32 PLV connectivity (~1 hour first time, instant from cache)
4. Build PyG graphs with top-k=8 sparsification (~2 min)
5. Train and evaluate GAT and GCN encoders under strict leave-one-dataset-out (3 folds each)

Output JSON: `results/cross_dataset_full_training.json`.

**Expected headline numbers (subject-level, K=5):**

| Encoder | Mean Acc | Mean AUC | Mean F1 |
|---|---|---|---|
| GCN | 94.47% | 0.9845 | 0.9494 |
| GAT | 89.05% | 0.9606 | 0.9018 |

**Expected per-fold (GCN):**

| Train -> Test | Subjects correct | Acc | AUC |
|---|---|---|---|
| UNM + Iowa -> UC | 46 / 50 | 92.00% | 0.962 |
| UC + Iowa -> UNM | 30 / 31 | 96.77% | 1.000 |
| UC + UNM -> Iowa | 141 / 149 | 94.63% | 0.992 |

If your numbers differ by more than 1-2 pp, check:
- PyTorch version matches (2.2.x)
- `torch.use_deterministic_algorithms(True)` did not raise an error (visible in startup log)
- The same seed (42) is being set globally
- Cache fingerprint matches: see `data/processed/feature_cache_fingerprint.txt`

---

## 5. Reproducibility design

All entry-point scripts call `reproducibility.set_global_seed(42)` before any
data loading or model construction. This sets:

- Python `random.seed(42)`
- `numpy.random.seed(42)`
- `torch.manual_seed(42)` and CUDA equivalent
- `PYTHONHASHSEED=42` (so dict ordering is deterministic)
- `CUBLAS_WORKSPACE_CONFIG=:4096:8` (required for deterministic CUDA matrix ops)
- `torch.backends.cudnn.deterministic = True` and `cudnn.benchmark = False`
- `torch.use_deterministic_algorithms(True, warn_only=True)`

The feature cache is fingerprinted by preprocessing config (sample rate,
bandpass, notch, epoch length, frequency bands, channel count). Changing any
of these values prints a warning and you should clear the cache:

```bash
rm data/processed/*.npz
rm data/processed/feature_cache_fingerprint.txt
```

---

## 6. Hardware notes

- CPU: 10-core M-series Mac, ~3 hours end-to-end first run, ~25 min from cache.
- 16 GB RAM is sufficient. Peak memory ~6 GB during PLV computation.
- No GPU required. All published numbers were obtained on CPU.
- If you run on GPU, results may differ by <1 pp due to non-determinism in some
  PyG operations. Use CPU for exact reproduction of the paper's numbers.

---

## 7. Known limitations

- ICA preprocessing is **disabled by default** for speed. Enabling ICA
  (`--no-skip-ica` flag in `main.py`) adds ~1.5 hours and may shift accuracy
  by 1-2 pp. The reported headline numbers use `skip_ica=True`.
- Channel harmonization uses MNE spherical-spline interpolation when channels
  are missing in source datasets. If MNE versions differ across machines,
  interpolated values may differ by floating-point noise.
- The random initialization of GAT/GCN model parameters is seeded, but PyG's
  scatter operations on certain platforms may produce slightly different
  gradients. We have not observed this in practice on CPU.
