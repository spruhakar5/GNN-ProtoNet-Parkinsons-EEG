"""Re-run cross-dataset evaluation with full training schedule.
Uses cached features (no re-preprocess needed)."""
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

t_total = time.time()

print("Loading + preprocessing (will use cached features when available)...")
all_subjects = load_all_datasets()
all_subjects = preprocess_all(all_subjects, skip_ica=True)
all_subjects = extract_features_cached(all_subjects)
all_subjects = [s for s in all_subjects
                if s.node_features is not None and s.plv_matrix is not None]
all_subjects = build_graphs_all(all_subjects)
all_subjects = [s for s in all_subjects
                if s.graphs is not None and len(s.graphs) > 0]
print(f"\nReady: {len(all_subjects)} subjects with graphs")

# FULL training: 50 epochs x 100 episodes (vs prior 20x30)
print("\n=== GCN cross-dataset (50 ep x 100 eps) ===")
t0 = time.time()
gcn = cross_dataset_evaluation(
    all_subjects, k_shot=5, encoder_type='gcn',
    n_episodes=100, n_epochs=50, calibrate=True,
)
gcn_time = time.time() - t0

print("\n=== GAT cross-dataset (50 ep x 100 eps) ===")
t0 = time.time()
gat = cross_dataset_evaluation(
    all_subjects, k_shot=5, encoder_type='gat',
    n_episodes=100, n_epochs=50, calibrate=True,
)
gat_time = time.time() - t0

out = {
    'protocol': 'cross_dataset_LODO_full_training',
    'k_shot': 5, 'calibrate': True,
    'n_subjects': len(all_subjects),
    'n_train_epochs': 50, 'n_episodes': 100,
    'gcn': gcn, 'gat': gat,
    'gcn_train_time_sec': gcn_time, 'gat_train_time_sec': gat_time,
    'total_time_sec': time.time() - t_total,
}
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
with open(RESULTS_DIR / 'cross_dataset_full_training.json', 'w') as f:
    json.dump(out, f, indent=2, default=str)

print("\n=== FINAL ===")
for name, r in [('GCN', gcn), ('GAT', gat)]:
    print(f"\n{name} mean: Acc={r['mean_accuracy']*100:.2f}% AUC={r['mean_auc']:.3f}")
    for f in r['folds']:
        print(f"  {'+'.join(f['train_datasets'])} -> {f['test_dataset']}: "
              f"Acc={f['accuracy']*100:.2f}% AUC={f['auc_roc']:.3f}")
