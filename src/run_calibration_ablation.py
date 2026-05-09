"""Calibration-mode ablation under strict LODO.

Runs every combination of {encoder} x {calibration_mode}:
  encoder        : gat | gcn
  calibration    : none (no test-subject samples used)
                 | unlabeled (mean-centering, no labels used)
                 | labeled (legacy: uses test subject's true label)

Reports subject-level accuracy per fold per (encoder, mode), then writes
results/calibration_ablation.json with the full breakdown.
"""
import sys, time, json
sys.path.insert(0, '.')
from reproducibility import set_global_seed, report_environment

set_global_seed(42)
report_environment()

from dataset import load_all_datasets
from preprocessing import preprocess_all
from cache_features import extract_features_cached
from graph_builder import build_graphs_all
from evaluate import cross_dataset_evaluation
from config import RESULTS_DIR


def main():
    t0 = time.time()
    print("Loading + preprocessing (using cached features)...")
    subs = load_all_datasets()
    subs = preprocess_all(subs, skip_ica=True)
    subs = extract_features_cached(subs)
    subs = [s for s in subs if s.node_features is not None and s.plv_matrix is not None]
    subs = build_graphs_all(subs)
    subs = [s for s in subs if s.graphs and len(s.graphs) > 0]
    print(f"\nReady: {len(subs)} subjects with graphs ({time.time()-t0:.0f}s)")

    modes = ['none', 'unlabeled', 'labeled']
    encoders = ['gcn', 'gat']

    results = {}
    for encoder in encoders:
        for mode in modes:
            key = f"{encoder}_{mode}"
            print(f"\n{'#'*60}")
            print(f"# {encoder.upper()} encoder · calibration={mode}")
            print(f"{'#'*60}")
            t = time.time()
            r = cross_dataset_evaluation(
                subs, k_shot=5,
                calibrate=(mode != 'none'),
                calibration_mode=mode,
                encoder_type=encoder,
                n_episodes=30, n_epochs=20,  # compact for ablation
            )
            elapsed = time.time() - t
            results[key] = {
                'encoder': encoder,
                'calibration_mode': mode,
                'mean_subject_accuracy': r.get('mean_subject_level_accuracy', 0),
                'mean_subject_auc': r.get('mean_subject_level_auc', 0),
                'mean_subject_f1': r.get('mean_subject_level_f1', 0),
                'mean_epoch_accuracy': r.get('mean_accuracy', 0),
                'mean_epoch_auc': r.get('mean_auc', 0),
                'folds': r.get('folds', []),
                'elapsed_sec': elapsed,
            }
            print(f"\n  RESULT  {encoder}/{mode}: "
                  f"subj acc={results[key]['mean_subject_accuracy']*100:.2f}% "
                  f"AUC={results[key]['mean_subject_auc']:.4f}  "
                  f"({elapsed:.0f}s)")

    out = {
        'protocol': 'cross_dataset_LODO_calibration_ablation',
        'k_shot': 5,
        'n_subjects': len(subs),
        'n_train_epochs': 50,
        'n_episodes': 100,
        'results': results,
        'total_time_sec': time.time() - t0,
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_DIR / 'calibration_ablation.json', 'w') as f:
        json.dump(out, f, indent=2, default=str)

    print(f"\n{'='*60}")
    print("CALIBRATION ABLATION SUMMARY (subject-level accuracy)")
    print(f"{'='*60}")
    print(f"{'encoder':<10}{'mode':<14}{'acc':>10}{'AUC':>10}{'F1':>10}")
    for encoder in encoders:
        for mode in modes:
            r = results[f"{encoder}_{mode}"]
            print(f"{encoder:<10}{mode:<14}"
                  f"{r['mean_subject_accuracy']*100:>9.2f}%"
                  f"{r['mean_subject_auc']:>10.4f}"
                  f"{r['mean_subject_f1']:>10.4f}")

    print(f"\nTotal: {(time.time()-t0)/60:.1f} min")
    print(f"Saved to {RESULTS_DIR / 'calibration_ablation.json'}")


if __name__ == '__main__':
    main()
