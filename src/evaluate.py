"""
Evaluation Protocols for GNN-ProtoNet.

Protocol 1: LOSO (Leave-One-Subject-Out)
Protocol 2: Cross-Dataset Generalization
"""

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, roc_auc_score

from config import (
    N_QUERY, N_EPISODES_TRAIN, N_TRAIN_EPOCHS, DEVICE,
    ENCODER_TYPE, FREQ_BANDS,
)
from train import train_one_fold, create_fresh_model, move_graphs_to_device


def evaluate_subject(model, test_subject, train_subjects, k_shot=5,
                     calibrate=True, calibration_mode='unlabeled'):
    """
    Evaluate model on a single held-out subject.

    Parameters
    ----------
    model : GNNProtoNet (trained)
    test_subject : Subject
    train_subjects : list of Subject
    k_shot : int — calibration samples from test subject
    calibrate : bool — master switch
    calibration_mode : str — one of:
        'none'      — no calibration even if calibrate=True
        'unlabeled' — domain-adaptation mean-centering (no test labels used)
        'labeled'   — legacy: uses test subject's true label (supervised)

    Returns
    -------
    accuracy, y_true, y_pred, y_scores
    """
    model.eval()

    graphs = test_subject.graphs
    n_graphs = len(graphs)

    if n_graphs <= k_shot:
        return 0.0, [], [], []

    # Split: k_shot for calibration, rest for query
    indices = np.random.permutation(n_graphs)
    cal_indices = indices[:k_shot]
    query_indices = indices[k_shot:]

    cal_graphs = [graphs[i] for i in cal_indices]
    query_graphs_list = [graphs[i] for i in query_indices]
    query_labels_true = [test_subject.label] * len(query_indices)

    # Build support set from training subjects
    pd_train = [s for s in train_subjects if s.label == 1]
    hc_train = [s for s in train_subjects if s.label == 0]

    def get_support(subj_list, k):
        all_g = []
        for s in subj_list:
            all_g.extend(s.graphs)
        if not all_g:
            return []
        replace = len(all_g) < k
        idx = np.random.choice(len(all_g), k, replace=replace)
        return [all_g[i] for i in idx]

    hc_support = get_support(hc_train, k_shot)
    pd_support = get_support(pd_train, k_shot)
    if not hc_support or not pd_support:
        return 0.0, [], [], []
    support_graphs = hc_support + pd_support
    support_labels = torch.tensor(
        [0]*len(hc_support) + [1]*len(pd_support), dtype=torch.long
    ).to(DEVICE)

    # Move to device
    support_graphs = move_graphs_to_device(support_graphs, DEVICE)
    query_graphs_list = move_graphs_to_device(query_graphs_list, DEVICE)

    use_cal = calibrate and calibration_mode != 'none'

    with torch.no_grad():
        if use_cal:
            cal_graphs_dev = move_graphs_to_device(cal_graphs, DEVICE)
            if calibration_mode == 'labeled':
                cal_labels_t = torch.tensor(
                    [test_subject.label] * k_shot, dtype=torch.long
                ).to(DEVICE)
                log_probs, predictions = model(
                    support_graphs, support_labels, query_graphs_list,
                    cal_graphs_dev, cal_labels_t,
                )
            elif calibration_mode == 'unlabeled':
                # Pass cal graphs with None labels -> unlabeled mean-centering
                log_probs, predictions = model(
                    support_graphs, support_labels, query_graphs_list,
                    cal_graphs_dev, None,
                )
            else:
                raise ValueError(f"Unknown calibration_mode: {calibration_mode}")
        else:
            log_probs, predictions = model(
                support_graphs, support_labels, query_graphs_list,
            )

    y_pred = predictions.cpu().numpy().tolist()
    y_true = query_labels_true
    # Probability of class 1 for AUC
    probs = torch.exp(log_probs).cpu().numpy()
    y_scores = probs[:, 1].tolist() if probs.shape[1] > 1 else [0.5] * len(y_pred)

    acc = accuracy_score(y_true, y_pred)
    return acc, y_true, y_pred, y_scores


def compute_metrics(all_y_true, all_y_pred, all_y_scores):
    """Compute aggregate metrics."""
    if not all_y_true:
        return {}

    acc = accuracy_score(all_y_true, all_y_pred)
    f1 = f1_score(all_y_true, all_y_pred, average='binary', zero_division=0)

    cm = confusion_matrix(all_y_true, all_y_pred, labels=[0, 1])
    if cm.size == 4:
        tn, fp, fn, tp = cm.ravel()
    else:
        tn = fp = fn = tp = 0

    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

    try:
        auc = roc_auc_score(all_y_true, all_y_scores)
    except ValueError:
        auc = 0.0

    return {
        'accuracy': acc,
        'sensitivity': sensitivity,
        'specificity': specificity,
        'f1_score': f1,
        'auc_roc': auc,
        'confusion_matrix': cm.tolist(),
        'tn': int(tn), 'fp': int(fp), 'fn': int(fn), 'tp': int(tp),
    }


def loso_evaluation(subjects, k_shot=5, calibrate=True,
                    encoder_type=ENCODER_TYPE,
                    n_episodes=N_EPISODES_TRAIN, n_epochs=N_TRAIN_EPOCHS):
    """
    Leave-One-Subject-Out evaluation.

    For each subject: hold out as test, train on rest, evaluate.
    """
    print(f"\n{'='*60}")
    print(f"LOSO EVALUATION")
    print(f"  Subjects: {len(subjects)}, K-shot: {k_shot}")
    print(f"  Encoder: {encoder_type}, Calibration: {calibrate}")
    print(f"  Device: {DEVICE}")
    print(f"{'='*60}")

    all_y_true, all_y_pred, all_y_scores = [], [], []
    per_subject_acc = []
    subject_results = []

    for fold_idx, test_subj in enumerate(subjects):
        print(f"\n--- Fold {fold_idx+1}/{len(subjects)}: "
              f"Test={test_subj.subject_id} "
              f"({'PD' if test_subj.label==1 else 'HC'}) ---")

        train_subjs = [s for s in subjects if s.subject_id != test_subj.subject_id]

        # Check both classes exist in training
        train_labels = set(s.label for s in train_subjs)
        if len(train_labels) < 2:
            print(f"  [SKIP] Only one class in training")
            continue

        # Fresh model per fold
        model = create_fresh_model(encoder_type)
        model, losses = train_one_fold(
            model, train_subjs, k_shot=k_shot,
            n_episodes=n_episodes, n_epochs=n_epochs,
        )

        acc, y_true, y_pred, y_scores = evaluate_subject(
            model, test_subj, train_subjs,
            k_shot=k_shot, calibrate=calibrate,
        )

        all_y_true.extend(y_true)
        all_y_pred.extend(y_pred)
        all_y_scores.extend(y_scores)
        if y_true:  # Only include subjects with valid predictions
            per_subject_acc.append(acc)

        label_str = 'PD' if test_subj.label == 1 else 'HC'
        correct = sum(1 for t, p in zip(y_true, y_pred) if t == p)
        print(f"  {test_subj.subject_id} ({label_str}) -> "
              f"Acc: {acc:.4f} ({correct}/{len(y_true)})")

        subject_results.append({
            'subject_id': test_subj.subject_id,
            'dataset': test_subj.dataset,
            'true_label': test_subj.label,
            'accuracy': acc,
            'n_queries': len(y_true),
        })

    metrics = compute_metrics(all_y_true, all_y_pred, all_y_scores)
    metrics['mean_subject_accuracy'] = float(np.mean(per_subject_acc)) if per_subject_acc else 0
    metrics['std_subject_accuracy'] = float(np.std(per_subject_acc)) if per_subject_acc else 0
    metrics['per_subject'] = subject_results

    print(f"\n{'='*60}")
    print(f"LOSO RESULTS (K={k_shot}, Encoder={encoder_type})")
    print(f"{'='*60}")
    print(f"  Accuracy:    {metrics.get('accuracy', 0):.4f}")
    print(f"  Mean +/- Std: {metrics['mean_subject_accuracy']:.4f} +/- {metrics['std_subject_accuracy']:.4f}")
    print(f"  Sensitivity: {metrics.get('sensitivity', 0):.4f}")
    print(f"  Specificity: {metrics.get('specificity', 0):.4f}")
    print(f"  F1:          {metrics.get('f1_score', 0):.4f}")
    print(f"  AUC-ROC:     {metrics.get('auc_roc', 0):.4f}")

    return metrics


def cross_dataset_evaluation(subjects, k_shot=5, calibrate=True,
                              calibration_mode='unlabeled',
                              encoder_type=ENCODER_TYPE,
                              n_episodes=N_EPISODES_TRAIN, n_epochs=N_TRAIN_EPOCHS):
    """
    Cross-Dataset Generalization evaluation.

    3 folds:
      - Train UC+UNM -> Test Iowa
      - Train UC+Iowa -> Test UNM
      - Train UNM+Iowa -> Test UC
    """
    print(f"\n{'='*60}")
    print(f"CROSS-DATASET EVALUATION")
    print(f"  K-shot: {k_shot}, Encoder: {encoder_type}")
    print(f"{'='*60}")

    datasets = ['UC', 'UNM', 'Iowa']
    fold_results = []

    for test_ds in datasets:
        train_ds = [d for d in datasets if d != test_ds]

        train_subjs = [s for s in subjects if s.dataset in train_ds]
        test_subjs = [s for s in subjects if s.dataset == test_ds]

        if not train_subjs or not test_subjs:
            print(f"\n  [SKIP] {'+'.join(train_ds)} -> {test_ds}: insufficient data")
            continue

        train_labels = set(s.label for s in train_subjs)
        if len(train_labels) < 2:
            print(f"\n  [SKIP] Only one class in training for {test_ds}")
            continue

        print(f"\n--- Train: {'+'.join(train_ds)} ({len(train_subjs)}) "
              f"-> Test: {test_ds} ({len(test_subjs)}) ---")

        # Train model
        model = create_fresh_model(encoder_type)
        model, losses = train_one_fold(
            model, train_subjs, k_shot=k_shot,
            n_episodes=n_episodes, n_epochs=n_epochs,
        )

        # Evaluate on each test subject
        all_y_true, all_y_pred, all_y_scores = [], [], []
        per_subject_acc = []
        # Subject-level prediction via majority vote across query epochs
        subj_true, subj_pred, subj_score = [], [], []

        for test_subj in test_subjs:
            acc, y_true, y_pred, y_scores = evaluate_subject(
                model, test_subj, train_subjs,
                k_shot=k_shot, calibrate=calibrate,
                calibration_mode=calibration_mode,
            )
            all_y_true.extend(y_true)
            all_y_pred.extend(y_pred)
            all_y_scores.extend(y_scores)
            if y_true:
                per_subject_acc.append(acc)
                # Subject-level: majority vote on epoch predictions
                pred_label = 1 if sum(y_pred) > len(y_pred) / 2 else 0
                # Subject-level score = mean prob of class 1 across epochs
                mean_score = float(np.mean(y_scores)) if y_scores else 0.5
                subj_true.append(test_subj.label)
                subj_pred.append(pred_label)
                subj_score.append(mean_score)

            label_str = 'PD' if test_subj.label == 1 else 'HC'
            print(f"  {test_subj.subject_id} ({label_str}) -> Acc: {acc:.4f}")

        fold_metrics = compute_metrics(all_y_true, all_y_pred, all_y_scores)
        fold_metrics['train_datasets'] = train_ds
        fold_metrics['test_dataset'] = test_ds
        fold_metrics['mean_subject_accuracy'] = float(np.mean(per_subject_acc))
        # Subject-level metrics (majority vote across epochs)
        if subj_true:
            subj_metrics = compute_metrics(subj_true, subj_pred, subj_score)
            fold_metrics['subject_level_accuracy'] = subj_metrics.get('accuracy', 0)
            fold_metrics['subject_level_f1'] = subj_metrics.get('f1_score', 0)
            fold_metrics['subject_level_auc'] = subj_metrics.get('auc_roc', 0)
            fold_metrics['subject_level_sensitivity'] = subj_metrics.get('sensitivity', 0)
            fold_metrics['subject_level_specificity'] = subj_metrics.get('specificity', 0)
            fold_metrics['n_subjects_correct'] = int(sum(
                1 for t, p in zip(subj_true, subj_pred) if t == p))
            fold_metrics['n_subjects_total'] = len(subj_true)
        fold_results.append(fold_metrics)

        print(f"  Fold Accuracy: {fold_metrics.get('accuracy', 0):.4f}")

    # Aggregate across folds
    if fold_results:
        avg_acc = np.mean([f['accuracy'] for f in fold_results])
        avg_f1 = np.mean([f['f1_score'] for f in fold_results])
        avg_auc = np.mean([f.get('auc_roc', 0) for f in fold_results])
        avg_subj_acc = np.mean([f.get('subject_level_accuracy', 0) for f in fold_results])
        avg_subj_auc = np.mean([f.get('subject_level_auc', 0) for f in fold_results])
        avg_subj_f1 = np.mean([f.get('subject_level_f1', 0) for f in fold_results])
    else:
        avg_acc = avg_f1 = avg_auc = 0
        avg_subj_acc = avg_subj_auc = avg_subj_f1 = 0

    results = {
        'folds': fold_results,
        'mean_accuracy': float(avg_acc),
        'mean_f1': float(avg_f1),
        'mean_auc': float(avg_auc),
        'mean_subject_level_accuracy': float(avg_subj_acc),
        'mean_subject_level_auc': float(avg_subj_auc),
        'mean_subject_level_f1': float(avg_subj_f1),
    }

    print(f"\n{'='*60}")
    print(f"CROSS-DATASET RESULTS (K={k_shot})")
    print(f"{'='*60}")
    print(f"  Epoch-level Accuracy: {avg_acc:.4f}, AUC: {avg_auc:.4f}")
    print(f"  SUBJECT-LEVEL Accuracy: {avg_subj_acc:.4f}, AUC: {avg_subj_auc:.4f}, F1: {avg_subj_f1:.4f}")
    for f in fold_results:
        print(f"  {'+'.join(f['train_datasets'])} -> {f['test_dataset']}: "
              f"epoch={f['accuracy']:.4f} subj={f.get('subject_level_accuracy', 0):.4f} "
              f"({f.get('n_subjects_correct', 0)}/{f.get('n_subjects_total', 0)})")

    return results
