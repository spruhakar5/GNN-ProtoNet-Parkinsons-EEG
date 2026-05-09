"""K-shot ablation under strict LODO (K-shot personalised setting).

For each K in {1, 5, 10, 20}, run cross-dataset evaluation with the
LABELED calibration mode — the K-shot supervised personalisation
protocol that our reframed paper claims as the headline method.
Reports subject-level accuracy and AUC per K, per encoder.
"""
import sys, time, json
sys.path.insert(0, '.')
from reproducibility import set_global_seed, report_environment

set_global_seed(43)  # different seed from main run, same protocol
report_environment()

from dataset import load_all_datasets
from preprocessing import preprocess_all
from cache_features import extract_features_cached
from graph_builder import build_graphs_all
from evaluate import cross_dataset_evaluation
from config import RESULTS_DIR


def main():
    t0 = time.time()
    subs = load_all_datasets()
    subs = preprocess_all(subs, skip_ica=True)
    subs = extract_features_cached(subs)
    subs = [s for s in subs if s.node_features is not None and s.plv_matrix is not None]
    subs = build_graphs_all(subs)
    subs = [s for s in subs if s.graphs and len(s.graphs) > 0]
    print(f"\nReady: {len(subs)} subjects")

    results = {}
    for K in [1, 5, 10, 20]:
        for encoder in ['gcn', 'gat']:
            key = f"{encoder}_K{K}"
            print(f"\n# {encoder.upper()} · K={K} · unlabeled calibration")
            t = time.time()
            r = cross_dataset_evaluation(
                subs, k_shot=K,
                calibrate=True, calibration_mode='labeled',
                encoder_type=encoder,
                n_episodes=30, n_epochs=20,
            )
            results[key] = {
                'encoder': encoder, 'k_shot': K,
                'mean_subject_accuracy': r.get('mean_subject_level_accuracy', 0),
                'mean_subject_auc': r.get('mean_subject_level_auc', 0),
                'mean_subject_f1': r.get('mean_subject_level_f1', 0),
                'folds': r.get('folds', []),
                'elapsed_sec': time.time() - t,
            }
            print(f"  RESULT  {encoder}/K={K}: subj acc={results[key]['mean_subject_accuracy']*100:.2f}%")

    out = {
        'protocol': 'cross_dataset_LODO_kshot_ablation',
        'calibration_mode': 'labeled',
        'n_subjects': len(subs),
        'results': results,
        'total_time_sec': time.time() - t0,
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_DIR / 'kshot_ablation.json', 'w') as f:
        json.dump(out, f, indent=2, default=str)

    print(f"\n{'='*60}")
    print("K-SHOT ABLATION (subject-level accuracy)")
    print(f"{'='*60}")
    print(f"{'K':<6}{'GCN':>10}{'GAT':>10}")
    for K in [1, 5, 10, 20]:
        gcn = results[f"gcn_K{K}"]['mean_subject_accuracy'] * 100
        gat = results[f"gat_K{K}"]['mean_subject_accuracy'] * 100
        print(f"{K:<6}{gcn:>9.2f}%{gat:>9.2f}%")
    print(f"\nTotal: {(time.time()-t0)/60:.1f} min")


if __name__ == '__main__':
    main()
