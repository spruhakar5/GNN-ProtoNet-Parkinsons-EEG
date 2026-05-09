"""Ensemble GCN + GAT predictions at the subject level.
Reuses cached features and the existing trained-fold paradigm
(retrains both encoders here for full reproducibility)."""
import sys, time, json
sys.path.insert(0, '.')
import numpy as np, torch, random
from collections import defaultdict
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix

torch.manual_seed(42); np.random.seed(42); random.seed(42)

from dataset import load_all_datasets
from preprocessing import preprocess_all
from cache_features import extract_features_cached
from graph_builder import build_graphs_all
from train import create_fresh_model, train_one_fold, move_graphs_to_device
from evaluate import evaluate_subject
from config import RESULTS_DIR, DEVICE


def loso_ensemble(subjects, k_shot=5, n_episodes=100, n_epochs=50, calibrate=True):
    """Cross-dataset eval where each test subject's prediction is the mean
    of GCN and GAT softmax probabilities across query epochs."""
    datasets = ['UC', 'UNM', 'Iowa']
    fold_results = []

    for test_ds in datasets:
        train_ds = [d for d in datasets if d != test_ds]
        train_subjs = [s for s in subjects if s.dataset in train_ds]
        test_subjs = [s for s in subjects if s.dataset == test_ds]
        if not test_subjs:
            continue
        if len(set(s.label for s in train_subjs)) < 2:
            continue

        print(f"\n--- Train {'+'.join(train_ds)} ({len(train_subjs)}) -> Test {test_ds} ({len(test_subjs)}) ---")

        # Train both encoders
        gcn = create_fresh_model('gcn')
        gcn, _ = train_one_fold(gcn, train_subjs, k_shot=k_shot,
                                n_episodes=n_episodes, n_epochs=n_epochs)
        gat = create_fresh_model('gat')
        gat, _ = train_one_fold(gat, train_subjs, k_shot=k_shot,
                                n_episodes=n_episodes, n_epochs=n_epochs)

        # Per-subject ensemble prediction
        subj_true, subj_pred, subj_score = [], [], []
        for ts in test_subjs:
            _, y_true_g, y_pred_g, scores_g = evaluate_subject(
                gcn, ts, train_subjs, k_shot=k_shot, calibrate=calibrate)
            _, y_true_a, y_pred_a, scores_a = evaluate_subject(
                gat, ts, train_subjs, k_shot=k_shot, calibrate=calibrate)
            if not y_true_g or not y_true_a:
                continue
            # Average softmax probs across the two encoders, epoch-wise.
            n = min(len(scores_g), len(scores_a))
            ens_scores = [(scores_g[i] + scores_a[i]) / 2.0 for i in range(n)]
            # Subject-level: mean ensemble prob -> threshold 0.5
            mean_p = float(np.mean(ens_scores))
            pred = 1 if mean_p > 0.5 else 0
            subj_true.append(ts.label)
            subj_pred.append(pred)
            subj_score.append(mean_p)
            print(f"  {ts.subject_id} ({'PD' if ts.label==1 else 'HC'}) -> "
                  f"score={mean_p:.3f} pred={'PD' if pred==1 else 'HC'} "
                  f"({'OK' if pred==ts.label else 'WRONG'})")

        if not subj_true:
            continue
        acc = accuracy_score(subj_true, subj_pred)
        f1 = f1_score(subj_true, subj_pred, zero_division=0)
        try:
            auc = roc_auc_score(subj_true, subj_score)
        except Exception:
            auc = 0.0
        cm = confusion_matrix(subj_true, subj_pred, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
        fold_results.append({
            'train_datasets': train_ds, 'test_dataset': test_ds,
            'subject_acc': acc, 'subject_auc': auc, 'subject_f1': f1,
            'tp': int(tp), 'fp': int(fp), 'fn': int(fn), 'tn': int(tn),
            'n_correct': int(sum(1 for t, p in zip(subj_true, subj_pred) if t == p)),
            'n_total': len(subj_true),
        })
        print(f"  Fold subject-level: Acc={acc*100:.2f}% AUC={auc:.4f} F1={f1:.4f}")

    return fold_results


def main():
    t0 = time.time()
    subs = load_all_datasets()
    subs = preprocess_all(subs, skip_ica=True)
    subs = extract_features_cached(subs)
    subs = [s for s in subs if s.node_features is not None and s.plv_matrix is not None]
    subs = build_graphs_all(subs)
    subs = [s for s in subs if s.graphs and len(s.graphs) > 0]

    print(f"\n=== ENSEMBLE (GCN + GAT) cross-dataset LODO ===")
    folds = loso_ensemble(subs, k_shot=5, n_episodes=100, n_epochs=50)

    if folds:
        mean_acc = np.mean([f['subject_acc'] for f in folds])
        mean_auc = np.mean([f['subject_auc'] for f in folds])
        mean_f1 = np.mean([f['subject_f1'] for f in folds])
        out = {
            'protocol': 'cross_dataset_LODO_ensemble',
            'k_shot': 5,
            'mean_subject_accuracy': float(mean_acc),
            'mean_subject_auc': float(mean_auc),
            'mean_subject_f1': float(mean_f1),
            'folds': folds,
            'time_sec': time.time() - t0,
        }
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        with open(RESULTS_DIR / 'ensemble_results.json', 'w') as f:
            json.dump(out, f, indent=2, default=str)
        print(f"\n=== ENSEMBLE FINAL ===")
        print(f"Subject-level Acc: {mean_acc*100:.2f}%, AUC: {mean_auc:.4f}, F1: {mean_f1:.4f}")
        for f in folds:
            print(f"  {'+'.join(f['train_datasets'])} -> {f['test_dataset']}: "
                  f"Acc={f['subject_acc']*100:.2f}% ({f['n_correct']}/{f['n_total']}) "
                  f"AUC={f['subject_auc']:.4f}")


if __name__ == '__main__':
    main()
