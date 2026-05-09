"""
Run cross-dataset evaluation on all 230 subjects (3 folds).
Uses feature caching for efficient re-runs.
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
    t_total = time.time()

    # 1. Load all datasets
    print("\n" + "=" * 60)
    print("STEP 1: LOAD DATASETS")
    print("=" * 60)
    t0 = time.time()
    all_subjects = load_all_datasets()
    print(f"Load time: {time.time()-t0:.0f}s")

    # 2. Preprocess (skip ICA for speed on CPU)
    print("\n" + "=" * 60)
    print("STEP 2: PREPROCESS")
    print("=" * 60)
    t0 = time.time()
    all_subjects = preprocess_all(all_subjects, skip_ica=True)
    print(f"Preprocess time: {time.time()-t0:.0f}s")

    # 3. Extract features with caching
    print("\n" + "=" * 60)
    print("STEP 3: EXTRACT FEATURES (with caching)")
    print("=" * 60)
    t0 = time.time()
    all_subjects = extract_features_cached(all_subjects)
    print(f"Feature extraction time: {time.time()-t0:.0f}s")

    # Keep only subjects that got features
    all_subjects = [s for s in all_subjects if s.node_features is not None
                    and s.plv_matrix is not None]
    print(f"Subjects with features: {len(all_subjects)}")

    # 4. Build graphs
    print("\n" + "=" * 60)
    print("STEP 4: BUILD GRAPHS")
    print("=" * 60)
    t0 = time.time()
    all_subjects = build_graphs_all(all_subjects)
    print(f"Graph build time: {time.time()-t0:.0f}s")

    # Drop subjects without graphs
    all_subjects = [s for s in all_subjects if s.graphs is not None and len(s.graphs) > 0]
    print(f"Subjects with graphs: {len(all_subjects)}")

    # Stats per dataset
    for ds in ['UC', 'UNM', 'Iowa']:
        subs = [s for s in all_subjects if s.dataset == ds]
        n_pd = sum(1 for s in subs if s.label == 1)
        n_hc = sum(1 for s in subs if s.label == 0)
        print(f"  {ds}: {len(subs)} total ({n_pd} PD, {n_hc} HC)")

    # 5. Cross-dataset evaluation
    print("\n" + "=" * 60)
    print("STEP 5: CROSS-DATASET EVALUATION (GAT)")
    print("=" * 60)
    t0 = time.time()
    gat_results = cross_dataset_evaluation(
        all_subjects, k_shot=5, encoder_type='gat',
        n_episodes=30, n_epochs=20, calibrate=True,
    )
    gat_time = time.time() - t0
    print(f"\nGAT cross-dataset time: {gat_time:.0f}s ({gat_time/60:.1f}min)")

    print("\n" + "=" * 60)
    print("STEP 6: CROSS-DATASET EVALUATION (GCN)")
    print("=" * 60)
    t0 = time.time()
    gcn_results = cross_dataset_evaluation(
        all_subjects, k_shot=5, encoder_type='gcn',
        n_episodes=30, n_epochs=20, calibrate=True,
    )
    gcn_time = time.time() - t0
    print(f"\nGCN cross-dataset time: {gcn_time:.0f}s ({gcn_time/60:.1f}min)")

    # Save results
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = {
        'protocol': 'cross_dataset_leave_one_dataset_out',
        'k_shot': 5,
        'calibrate': True,
        'n_subjects': len(all_subjects),
        'gat': gat_results,
        'gcn': gcn_results,
        'total_time_sec': time.time() - t_total,
    }
    with open(RESULTS_DIR / 'cross_dataset_results.json', 'w') as f:
        json.dump(out, f, indent=2, default=str)

    print("\n" + "=" * 60)
    print("FINAL RESULTS — CROSS-DATASET (Leave-One-Dataset-Out)")
    print("=" * 60)
    print(f"Total subjects: {len(all_subjects)}")
    print(f"\nGAT:")
    print(f"  Mean accuracy: {gat_results.get('mean_accuracy', 0):.4f}")
    print(f"  Mean F1:       {gat_results.get('mean_f1', 0):.4f}")
    print(f"  Mean AUC-ROC:  {gat_results.get('mean_auc', 0):.4f}")
    for f in gat_results.get('folds', []):
        print(f"  {'+'.join(f['train_datasets'])} -> {f['test_dataset']}: "
              f"Acc={f.get('accuracy',0):.4f} AUC={f.get('auc_roc',0):.4f}")
    print(f"\nGCN:")
    print(f"  Mean accuracy: {gcn_results.get('mean_accuracy', 0):.4f}")
    print(f"  Mean F1:       {gcn_results.get('mean_f1', 0):.4f}")
    print(f"  Mean AUC-ROC:  {gcn_results.get('mean_auc', 0):.4f}")
    for f in gcn_results.get('folds', []):
        print(f"  {'+'.join(f['train_datasets'])} -> {f['test_dataset']}: "
              f"Acc={f.get('accuracy',0):.4f} AUC={f.get('auc_roc',0):.4f}")

    print(f"\nTotal elapsed: {(time.time()-t_total)/60:.1f} min")
    print(f"Results saved to: {RESULTS_DIR / 'cross_dataset_results.json'}")


if __name__ == '__main__':
    main()
