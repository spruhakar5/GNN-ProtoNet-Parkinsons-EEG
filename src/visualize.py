"""
Visualization for GNN-ProtoNet.

Publication figures:
1. GAT attention heatmaps on electrode layout
2. t-SNE of graph embeddings (PD vs HC)
3. Prototype distance distributions
4. Training loss curves
"""

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE

from config import COMMON_CHANNELS, RESULTS_DIR, DEVICE, N_CHANNELS


# Standard 10-20 electrode positions (2D projection for plotting)
ELECTRODE_POS_2D = {
    'Fp1': (-0.3, 0.9), 'Fp2': (0.3, 0.9),
    'F7': (-0.7, 0.6), 'F3': (-0.35, 0.6), 'Fz': (0.0, 0.6),
    'F4': (0.35, 0.6), 'F8': (0.7, 0.6),
    'FT7': (-0.8, 0.35), 'FC3': (-0.4, 0.35), 'FCz': (0.0, 0.35),
    'FC4': (0.4, 0.35), 'FT8': (0.8, 0.35),
    'T7': (-0.9, 0.0), 'C3': (-0.45, 0.0), 'Cz': (0.0, 0.0),
    'C4': (0.45, 0.0), 'T8': (0.9, 0.0),
    'TP7': (-0.8, -0.35), 'CP3': (-0.4, -0.35), 'CPz': (0.0, -0.35),
    'CP4': (0.4, -0.35), 'TP8': (0.8, -0.35),
    'P7': (-0.7, -0.6), 'P3': (-0.35, -0.6), 'Pz': (0.0, -0.6),
    'P4': (0.35, -0.6), 'P8': (0.7, -0.6),
    'O1': (-0.3, -0.9), 'Oz': (0.0, -0.9), 'O2': (0.3, -0.9),
    'F9': (-0.9, 0.6), 'F10': (0.9, 0.6),
}


def plot_attention_heatmap(model, graphs, save_path=None):
    """
    Plot GAT attention weights on 10-20 electrode layout.

    Parameters
    ----------
    model : GNNProtoNet with GAT encoder
    graphs : list of PyG Data (a few representative graphs)
    save_path : str, optional
    """
    from train import move_graphs_to_device

    model.eval()
    graphs = move_graphs_to_device(graphs[:10], DEVICE)

    # Get attention weights
    with torch.no_grad():
        _, (attn_edge_index, attn_weights) = model.encode_with_attention(graphs)

    attn_ei = attn_edge_index.cpu().numpy()
    attn_w = attn_weights.cpu().numpy().flatten()

    # Build attention matrix (averaged across graphs in batch)
    attn_matrix = np.zeros((N_CHANNELS, N_CHANNELS))
    counts = np.zeros((N_CHANNELS, N_CHANNELS))

    for idx in range(attn_ei.shape[1]):
        i, j = attn_ei[0, idx] % N_CHANNELS, attn_ei[1, idx] % N_CHANNELS
        attn_matrix[i, j] += attn_w[idx]
        counts[i, j] += 1

    counts[counts == 0] = 1
    attn_matrix /= counts

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # Heatmap
    ax = axes[0]
    sns.heatmap(attn_matrix, ax=ax, cmap='Reds',
                xticklabels=COMMON_CHANNELS, yticklabels=COMMON_CHANNELS)
    ax.set_title('GAT Attention Weights', fontsize=14)
    ax.tick_params(labelsize=6)

    # Electrode layout with connections
    ax = axes[1]
    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-1.2, 1.2)
    ax.set_aspect('equal')
    ax.set_title('Attention on Electrode Layout', fontsize=14)

    # Draw head outline
    circle = plt.Circle((0, 0), 1.0, fill=False, linewidth=2, color='gray')
    ax.add_patch(circle)
    # Nose
    ax.plot([-0.1, 0, 0.1], [1.0, 1.1, 1.0], 'k-', linewidth=2)

    # Draw edges (top connections)
    threshold = np.percentile(attn_matrix[attn_matrix > 0], 90) if attn_matrix.max() > 0 else 0
    for i, ch_i in enumerate(COMMON_CHANNELS):
        for j, ch_j in enumerate(COMMON_CHANNELS):
            if i < j and attn_matrix[i, j] > threshold:
                if ch_i in ELECTRODE_POS_2D and ch_j in ELECTRODE_POS_2D:
                    x1, y1 = ELECTRODE_POS_2D[ch_i]
                    x2, y2 = ELECTRODE_POS_2D[ch_j]
                    alpha = min(attn_matrix[i, j] / (attn_matrix.max() + 1e-8), 1.0)
                    ax.plot([x1, x2], [y1, y2], 'r-', alpha=alpha, linewidth=2)

    # Draw electrodes
    for ch in COMMON_CHANNELS:
        if ch in ELECTRODE_POS_2D:
            x, y = ELECTRODE_POS_2D[ch]
            ax.plot(x, y, 'ko', markersize=8, zorder=5)
            ax.annotate(ch, (x, y), fontsize=5, ha='center', va='bottom',
                       xytext=(0, 5), textcoords='offset points')

    ax.axis('off')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  Saved attention heatmap: {save_path}")
    plt.close()


def plot_tsne_embeddings(model, subjects, save_path=None):
    """
    t-SNE visualization of graph embeddings colored by PD/HC.

    Parameters
    ----------
    model : GNNProtoNet
    subjects : list of Subject with .graphs
    save_path : str, optional
    """
    from train import move_graphs_to_device

    model.eval()
    all_embeddings = []
    all_labels = []
    all_datasets = []

    with torch.no_grad():
        for subj in subjects:
            # Sample up to 20 graphs per subject
            n_sample = min(20, len(subj.graphs))
            sample_graphs = subj.graphs[:n_sample]
            sample_graphs = move_graphs_to_device(sample_graphs, DEVICE)

            emb = model.encode(sample_graphs).cpu().numpy()
            all_embeddings.append(emb)
            all_labels.extend([subj.label] * n_sample)
            all_datasets.extend([subj.dataset] * n_sample)

    embeddings = np.concatenate(all_embeddings, axis=0)
    labels = np.array(all_labels)
    datasets = np.array(all_datasets)

    # t-SNE
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(labels)-1))
    coords = tsne.fit_transform(embeddings)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # By class
    ax = axes[0]
    for label, name, color in [(0, 'HC', '#2196F3'), (1, 'PD', '#F44336')]:
        mask = labels == label
        ax.scatter(coords[mask, 0], coords[mask, 1], c=color, label=name,
                  alpha=0.6, s=20, edgecolors='white', linewidth=0.3)
    ax.legend(fontsize=12)
    ax.set_title('t-SNE by Class (PD vs HC)', fontsize=14)
    ax.set_xlabel('t-SNE 1')
    ax.set_ylabel('t-SNE 2')

    # By dataset
    ax = axes[1]
    ds_colors = {'UC': '#4CAF50', 'UNM': '#FF9800', 'Iowa': '#9C27B0', 'synthetic': '#607D8B'}
    for ds_name, color in ds_colors.items():
        mask = datasets == ds_name
        if mask.any():
            ax.scatter(coords[mask, 0], coords[mask, 1], c=color, label=ds_name,
                      alpha=0.6, s=20, edgecolors='white', linewidth=0.3)
    ax.legend(fontsize=12)
    ax.set_title('t-SNE by Dataset', fontsize=14)
    ax.set_xlabel('t-SNE 1')
    ax.set_ylabel('t-SNE 2')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  Saved t-SNE: {save_path}")
    plt.close()


def plot_prototype_distances(model, subjects, k_shot=5, save_path=None):
    """
    Plot distribution of distances from query embeddings to PD/HC prototypes.
    """
    from train import move_graphs_to_device
    from torch_geometric.data import Batch

    model.eval()

    pd_subjects = [s for s in subjects if s.label == 1]
    hc_subjects = [s for s in subjects if s.label == 0]

    # Build support set
    support_graphs = []
    support_labels = []
    for s in hc_subjects[:3]:
        support_graphs.extend(s.graphs[:k_shot])
        support_labels.extend([0] * min(k_shot, len(s.graphs)))
    for s in pd_subjects[:3]:
        support_graphs.extend(s.graphs[:k_shot])
        support_labels.extend([1] * min(k_shot, len(s.graphs)))

    support_labels = torch.tensor(support_labels, dtype=torch.long).to(DEVICE)
    support_graphs = move_graphs_to_device(support_graphs, DEVICE)

    with torch.no_grad():
        support_emb = model.encode(support_graphs)
        prototypes = model.compute_prototypes(support_emb, support_labels)

    # Compute distances for all subjects
    pd_dists_to_pd, pd_dists_to_hc = [], []
    hc_dists_to_pd, hc_dists_to_hc = [], []

    with torch.no_grad():
        for subj in subjects:
            sample = move_graphs_to_device(subj.graphs[:30], DEVICE)
            emb = model.encode(sample)
            dists = torch.cdist(emb, prototypes).cpu().numpy()

            if subj.label == 1:  # PD
                pd_dists_to_hc.extend(dists[:, 0].tolist())
                pd_dists_to_pd.extend(dists[:, 1].tolist())
            else:  # HC
                hc_dists_to_hc.extend(dists[:, 0].tolist())
                hc_dists_to_pd.extend(dists[:, 1].tolist())

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    ax.hist(pd_dists_to_pd, bins=30, alpha=0.7, color='#F44336', label='PD to PD proto')
    ax.hist(pd_dists_to_hc, bins=30, alpha=0.7, color='#2196F3', label='PD to HC proto')
    ax.set_title('PD Subjects: Distance to Prototypes', fontsize=12)
    ax.set_xlabel('Euclidean Distance')
    ax.legend()

    ax = axes[1]
    ax.hist(hc_dists_to_hc, bins=30, alpha=0.7, color='#2196F3', label='HC to HC proto')
    ax.hist(hc_dists_to_pd, bins=30, alpha=0.7, color='#F44336', label='HC to PD proto')
    ax.set_title('HC Subjects: Distance to Prototypes', fontsize=12)
    ax.set_xlabel('Euclidean Distance')
    ax.legend()

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  Saved prototype distances: {save_path}")
    plt.close()


def plot_training_loss(losses, title="Training Loss", save_path=None):
    """Plot training loss curve."""
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(losses, linewidth=2, color='#1976D2')
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Loss', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  Saved loss curve: {save_path}")
    plt.close()


def generate_all_figures(model, subjects, losses=None, output_dir=None):
    """Generate all publication figures."""
    if output_dir is None:
        output_dir = RESULTS_DIR

    from pathlib import Path
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"GENERATING FIGURES")
    print(f"  Output: {output_dir}")
    print(f"{'='*60}")

    try:
        plot_tsne_embeddings(model, subjects,
                            save_path=str(output_dir / 'tsne_embeddings.png'))
    except Exception as e:
        print(f"  [ERROR] t-SNE: {e}")

    try:
        if model.encoder_type == 'gat':
            sample_graphs = []
            for s in subjects[:5]:
                sample_graphs.extend(s.graphs[:5])
            plot_attention_heatmap(model, sample_graphs,
                                 save_path=str(output_dir / 'attention_heatmap.png'))
    except Exception as e:
        print(f"  [ERROR] Attention heatmap: {e}")

    try:
        plot_prototype_distances(model, subjects,
                                save_path=str(output_dir / 'prototype_distances.png'))
    except Exception as e:
        print(f"  [ERROR] Prototype distances: {e}")

    if losses:
        plot_training_loss(losses,
                          save_path=str(output_dir / 'training_loss.png'))

    print(f"  Done.")
