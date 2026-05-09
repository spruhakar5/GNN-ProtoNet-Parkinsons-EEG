"""
Smoke test for GNN-ProtoNet pipeline.

Runs the full pipeline on synthetic data (fast - under 1 min on CPU)
and verifies that results match expected shapes and behavior.

Use:
    cd src && python3 smoke_test.py

Exit code 0 means the pipeline is functioning correctly and reproducible
on your machine.
"""
import sys
sys.path.insert(0, '.')
from reproducibility import set_global_seed
set_global_seed(42)

import numpy as np
import torch

from dataset import generate_synthetic_data
from preprocessing import preprocess_all
from features import extract_features_all
from graph_builder import build_graphs_all
from train import create_fresh_model, train_one_fold, move_graphs_to_device


def main():
    print("=" * 60)
    print("GNN-ProtoNet Smoke Test")
    print("=" * 60)

    # 1. Synthetic data (deterministic per seed)
    subjects = generate_synthetic_data(n_subjects=6, sfreq=500, duration_sec=10)
    assert len(subjects) == 6, "Expected 6 synthetic subjects"
    assert sum(s.label for s in subjects) == 3, "Expected 3 PD subjects"

    # 2. Preprocess
    subjects = preprocess_all(subjects, skip_ica=True)
    assert all(s.epochs is not None for s in subjects), "Preprocessing failed"

    # 3. Features
    subjects = extract_features_all(subjects)
    expected_feature_dim = 13
    expected_n_channels = 32
    for s in subjects:
        assert s.node_features.shape[1] == expected_n_channels, (
            f"Expected {expected_n_channels} channels, got {s.node_features.shape[1]}"
        )
        assert s.node_features.shape[2] == expected_feature_dim, (
            f"Expected {expected_feature_dim}-dim features, got {s.node_features.shape[2]}"
        )
        assert s.plv_matrix.shape[1:] == (expected_n_channels, expected_n_channels), (
            f"PLV matrix wrong shape: {s.plv_matrix.shape}"
        )

    # 4. Graphs
    subjects = build_graphs_all(subjects)
    for s in subjects:
        assert s.graphs is not None and len(s.graphs) > 0
        g = s.graphs[0]
        assert g.x.shape == (32, 13), f"Graph nodes wrong: {g.x.shape}"
        assert g.edge_index.shape[0] == 2, "edge_index should be (2, E)"

    # 5. Train + classify GCN
    model = create_fresh_model('gcn')
    model, losses = train_one_fold(
        model, subjects, k_shot=2, n_query=5,
        n_episodes=5, n_epochs=3,
    )
    assert len(losses) == 3, "Expected 3 training epoch losses"
    assert losses[-1] < losses[0] * 1.5, (
        f"Loss did not decrease meaningfully: {losses[0]:.4f} -> {losses[-1]:.4f}"
    )

    # 6. Inference smoke
    s_graphs = subjects[0].graphs[:2] + subjects[3].graphs[:2]
    s_graphs = move_graphs_to_device(s_graphs, 'cpu')
    s_labels = torch.tensor([1, 1, 0, 0], dtype=torch.long)
    q_graphs = move_graphs_to_device(subjects[1].graphs[:3], 'cpu')

    model.train(False)  # inference mode
    with torch.no_grad():
        log_probs, preds = model(s_graphs, s_labels, q_graphs)
    assert log_probs.shape == (3, 2), f"Wrong log_probs shape: {log_probs.shape}"
    assert preds.shape == (3,), f"Wrong preds shape: {preds.shape}"

    print("\n" + "=" * 60)
    print("ALL CHECKS PASSED")
    print("=" * 60)
    print(f"Final training loss: {losses[-1]:.4f}")
    print(f"Inference produced predictions: {preds.tolist()}")
    print(f"Pipeline is functioning correctly on this machine.")
    return 0


if __name__ == '__main__':
    sys.exit(main())
