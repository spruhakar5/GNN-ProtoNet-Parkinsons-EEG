"""
GNN-ProtoNet Main Pipeline Runner.

Usage:
    python main.py                        # Synthetic data test
    python main.py --real                  # Real datasets
    python main.py --real --k_shot 5      # Specific K-shot
    python main.py --encoder gcn          # GCN ablation
    python main.py --cross-dataset        # Cross-dataset evaluation
    python main.py --ablation             # Run all ablations
"""

import argparse
import json
import time
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from config import (
    DATA_PROCESSED, RESULTS_DIR, K_SHOTS, N_EPISODES_TRAIN,
    N_TRAIN_EPOCHS, ENCODER_TYPE, TOP_K, DEVICE,
)
from dataset import load_all_datasets, generate_synthetic_data
from preprocessing import preprocess_all
from features import extract_features_all
from graph_builder import build_graphs_all
from evaluate import loso_evaluation, cross_dataset_evaluation
from visualize import generate_all_figures
from train import create_fresh_model, train_one_fold


def save_results(results, filename):
    """Save results to JSON."""
    out_dir = RESULTS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / filename
    with open(path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"Results saved: {path}")


def run_pipeline(args):
    """Run the full GNN-ProtoNet pipeline."""
    import random
    import torch
    import numpy as np
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)

    print("=" * 60)
    print("  GNN-ProtoNet: EEG Parkinson's Detection")
    print(f"  Encoder: {args.encoder}")
    print(f"  Device: {DEVICE}")
    print("=" * 60)

    # Step 1: Load data
    if args.real:
        subjects = load_all_datasets()
        if not subjects:
            print("\nNo real data found. Download datasets first or use synthetic.")
            return
    else:
        subjects = generate_synthetic_data(n_subjects=args.n_subjects)

    # Step 2: Preprocess
    subjects = preprocess_all(subjects, skip_ica=args.skip_ica)

    # Step 3: Extract features
    subjects = extract_features_all(subjects)

    # Step 4: Build graphs
    subjects = build_graphs_all(subjects, k=args.top_k)

    # Step 5: Evaluate
    all_results = {}
    k_list = [args.k_shot] if args.k_shot else K_SHOTS

    if args.cross_dataset:
        # Cross-dataset evaluation
        for k in k_list:
            print(f"\n{'#'*60}")
            print(f"# Cross-Dataset K={k}")
            print(f"{'#'*60}")
            start = time.time()
            results = cross_dataset_evaluation(
                subjects, k_shot=k, calibrate=args.calibrate,
                encoder_type=args.encoder,
                n_episodes=args.n_episodes, n_epochs=args.n_epochs,
            )
            results['elapsed_seconds'] = time.time() - start
            all_results[f'cross_dataset_k{k}'] = results
    else:
        # LOSO evaluation
        for k in k_list:
            print(f"\n{'#'*60}")
            print(f"# LOSO K={k}")
            print(f"{'#'*60}")
            start = time.time()
            results = loso_evaluation(
                subjects, k_shot=k, calibrate=args.calibrate,
                encoder_type=args.encoder,
                n_episodes=args.n_episodes, n_epochs=args.n_epochs,
            )
            results['elapsed_seconds'] = time.time() - start
            all_results[f'loso_k{k}'] = results

    # Step 6: Ablation (if requested)
    if args.ablation:
        print(f"\n{'#'*60}")
        print(f"# ABLATION STUDIES")
        print(f"{'#'*60}")

        k = args.k_shot or 5

        # GAT vs GCN
        for enc in ['gat', 'gcn']:
            key = f'ablation_{enc}_k{k}'
            if key not in all_results:
                print(f"\n--- Ablation: {enc.upper()} encoder ---")
                results = loso_evaluation(
                    subjects, k_shot=k, encoder_type=enc,
                    n_episodes=args.n_episodes, n_epochs=args.n_epochs,
                )
                all_results[key] = results

        # With vs without calibration
        print(f"\n--- Ablation: No calibration ---")
        results = loso_evaluation(
            subjects, k_shot=k, calibrate=False,
            encoder_type=args.encoder,
            n_episodes=args.n_episodes, n_epochs=args.n_epochs,
        )
        all_results[f'ablation_no_calibration_k{k}'] = results

    # Save results
    save_results(all_results, f'{args.encoder}_results.json')

    # Step 7: Generate figures
    if args.figures:
        model = create_fresh_model(args.encoder)
        # Quick train for figures
        model, losses = train_one_fold(
            model, subjects, k_shot=args.k_shot or 5,
            n_episodes=min(args.n_episodes, 50),
            n_epochs=min(args.n_epochs, 20),
        )
        generate_all_figures(model, subjects, losses=losses)

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    for key, res in all_results.items():
        if 'accuracy' in res:
            print(f"  {key}: Acc={res['accuracy']:.4f} F1={res.get('f1_score', 0):.4f}")
        elif 'mean_accuracy' in res:
            print(f"  {key}: Acc={res['mean_accuracy']:.4f} F1={res.get('mean_f1', 0):.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GNN-ProtoNet Pipeline")
    parser.add_argument('--real', action='store_true', help='Use real datasets')
    parser.add_argument('--n_subjects', type=int, default=10, help='Synthetic subjects')
    parser.add_argument('--skip-ica', dest='skip_ica', action='store_true', default=False,
                        help='Skip ICA artifact removal (faster, for testing)')
    parser.add_argument('--k_shot', type=int, default=None)
    parser.add_argument('--encoder', type=str, default=ENCODER_TYPE,
                        choices=['gat', 'gcn'])
    parser.add_argument('--top_k', type=int, default=TOP_K)
    parser.add_argument('--calibrate', action='store_true', default=True)
    parser.add_argument('--no-calibration', dest='calibrate', action='store_false')
    parser.add_argument('--cross-dataset', action='store_true', default=False)
    parser.add_argument('--ablation', action='store_true', default=False)
    parser.add_argument('--figures', action='store_true', default=False)
    parser.add_argument('--n_episodes', type=int, default=N_EPISODES_TRAIN)
    parser.add_argument('--n_epochs', type=int, default=N_TRAIN_EPOCHS)
    args = parser.parse_args()
    run_pipeline(args)
