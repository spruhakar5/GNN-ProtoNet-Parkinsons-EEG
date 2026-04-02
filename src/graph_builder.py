"""
Graph Construction for GNN-ProtoNet.

Converts node features + PLV connectivity into PyTorch Geometric graphs.
- Nodes: 32 electrodes with 13-dim feature vectors
- Edges: top-k PLV connections (k=8), weighted
"""

import numpy as np
import torch
from torch_geometric.data import Data

from config import TOP_K, N_CHANNELS


def topk_sparsify(plv_matrix, k=TOP_K):
    """
    Sparsify a PLV connectivity matrix by keeping top-k neighbors per node.

    Parameters
    ----------
    plv_matrix : (n_channels, n_channels) — symmetric PLV matrix
    k : int — number of neighbors to keep per node

    Returns
    -------
    edge_index : (2, n_edges) — COO format
    edge_weight : (n_edges,) — PLV values
    """
    n = plv_matrix.shape[0]
    # Zero out diagonal (no self-loops)
    plv = plv_matrix.copy()
    np.fill_diagonal(plv, 0)

    edges_src = []
    edges_dst = []
    weights = []

    for i in range(n):
        # Get top-k neighbors for node i
        row = plv[i]
        topk_idx = np.argsort(row)[-k:]  # indices of k largest values

        for j in topk_idx:
            if row[j] > 0:
                edges_src.append(i)
                edges_dst.append(j)
                weights.append(row[j])

    # Make undirected: deduplicate and ensure both directions exist
    edge_dict = {}
    for s, d, w in zip(edges_src, edges_dst, weights):
        key = (min(s, d), max(s, d))
        # Keep the max weight if both directions were independently selected
        if key not in edge_dict or w > edge_dict[key]:
            edge_dict[key] = w

    final_src, final_dst, final_w = [], [], []
    for (s, d), w in edge_dict.items():
        final_src.extend([s, d])
        final_dst.extend([d, s])
        final_w.extend([w, w])

    edge_index = np.array([final_src, final_dst], dtype=np.int64)
    edge_weight = np.array(final_w, dtype=np.float32)

    return edge_index, edge_weight


def build_graph(node_features, plv_matrix, label, k=TOP_K):
    """
    Build a single PyG Data object from one epoch's features.

    Parameters
    ----------
    node_features : (n_channels, n_node_features) — e.g. (32, 13)
    plv_matrix : (n_channels, n_channels) — PLV connectivity
    label : int — 0 (HC) or 1 (PD)
    k : int — top-k for sparsification

    Returns
    -------
    data : torch_geometric.data.Data
    """
    # Normalize node features (z-score per feature across nodes)
    mean = node_features.mean(axis=0, keepdims=True)
    std = node_features.std(axis=0, keepdims=True) + 1e-8
    node_features_norm = (node_features - mean) / std

    # Sparsify edges
    edge_index, edge_weight = topk_sparsify(plv_matrix, k=k)

    # Convert to tensors
    x = torch.tensor(node_features_norm, dtype=torch.float32)
    ei = torch.tensor(edge_index, dtype=torch.long)
    ew = torch.tensor(edge_weight, dtype=torch.float32)
    y = torch.tensor([label], dtype=torch.long)

    data = Data(x=x, edge_index=ei, edge_attr=ew, y=y)
    return data


def build_graphs_for_subject(subject, k=TOP_K):
    """
    Build PyG graphs for all epochs of a subject.

    Parameters
    ----------
    subject : Subject with node_features and plv_matrix populated
    k : int

    Returns
    -------
    subject : Subject with .graphs populated (list of Data)
    """
    n_epochs = subject.node_features.shape[0]
    graphs = []

    for i in range(n_epochs):
        g = build_graph(
            subject.node_features[i],
            subject.plv_matrix[i],
            subject.label,
            k=k,
        )
        graphs.append(g)

    subject.graphs = graphs
    return subject


def build_graphs_all(subjects, k=TOP_K):
    """Build graphs for all subjects."""
    print(f"\n{'='*60}")
    print(f"GRAPH CONSTRUCTION")
    print(f"  Nodes: {N_CHANNELS} electrodes")
    print(f"  Top-k: {k}")
    print(f"{'='*60}")

    for subj in subjects:
        try:
            build_graphs_for_subject(subj, k=k)
            n_edges = subj.graphs[0].edge_index.shape[1] if subj.graphs else 0
            print(f"  {subj.subject_id}: {len(subj.graphs)} graphs, "
                  f"{n_edges} edges per graph")
        except Exception as e:
            print(f"  [ERROR] {subj.subject_id}: {e}")

    return subjects
