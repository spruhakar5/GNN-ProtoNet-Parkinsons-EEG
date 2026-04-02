"""
GNN-ProtoNet: Encoder-agnostic Prototypical Network.

Combines:
- GNN encoder (GAT or GCN) for graph-level embeddings
- Prototype computation from support set
- Prototype calibration for test-subject adaptation
- Euclidean distance-based classification
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Batch

from config import (
    EMBEDDING_DIM, CALIBRATION_ALPHA, N_NODE_FEATURES,
    GAT_HEADS, GAT_HIDDEN, GCN_HIDDEN, DROPOUT,
)
try:
    from models.gat_encoder import GATEncoder
    from models.gcn_encoder import GCNEncoder
except ImportError:
    from .gat_encoder import GATEncoder
    from .gcn_encoder import GCNEncoder


class GNNProtoNet(nn.Module):
    """
    Prototypical Network with GNN encoder for EEG graph classification.
    """

    def __init__(self, encoder_type='gat', calibration_alpha=CALIBRATION_ALPHA):
        super().__init__()

        self.encoder_type = encoder_type
        self.alpha = calibration_alpha

        if encoder_type == 'gat':
            self.encoder = GATEncoder(
                in_dim=N_NODE_FEATURES,
                hidden_dim=GAT_HIDDEN,
                embed_dim=EMBEDDING_DIM,
                heads=GAT_HEADS,
                dropout=DROPOUT,
            )
        elif encoder_type == 'gcn':
            self.encoder = GCNEncoder(
                in_dim=N_NODE_FEATURES,
                hidden_dim=GCN_HIDDEN,
                embed_dim=EMBEDDING_DIM,
                dropout=DROPOUT,
            )
        else:
            raise ValueError(f"Unknown encoder type: {encoder_type}")

    def encode(self, graphs):
        """
        Encode a list of PyG Data graphs into embeddings.

        Parameters
        ----------
        graphs : list of torch_geometric.data.Data

        Returns
        -------
        embeddings : (n_graphs, EMBEDDING_DIM)
        """
        batch = Batch.from_data_list(graphs)
        embeddings = self.encoder(
            batch.x, batch.edge_index,
            edge_attr=batch.edge_attr,
            batch=batch.batch,
        )
        return embeddings

    def compute_prototypes(self, embeddings, labels):
        """
        Compute class prototypes as mean embeddings per class.

        Parameters
        ----------
        embeddings : (n_support, EMBEDDING_DIM)
        labels : (n_support,) — class labels (0 or 1)

        Returns
        -------
        prototypes : (n_classes, EMBEDDING_DIM)
        """
        classes = torch.unique(labels)
        prototypes = []
        for c in sorted(classes.tolist()):
            mask = labels == c
            prototypes.append(embeddings[mask].mean(dim=0))
        return torch.stack(prototypes)

    def calibrate_prototypes(self, prototypes, cal_embeddings, cal_labels):
        """
        Calibrate prototypes using test-subject calibration samples.

        calibrated = alpha * original + (1 - alpha) * subject_mean

        Parameters
        ----------
        prototypes : (n_classes, EMBEDDING_DIM)
        cal_embeddings : (n_cal, EMBEDDING_DIM)
        cal_labels : (n_cal,)

        Returns
        -------
        calibrated : (n_classes, EMBEDDING_DIM)
        """
        calibrated = prototypes.clone()
        n_classes = prototypes.shape[0]
        cal_classes = torch.unique(cal_labels)

        for c in sorted(cal_classes.tolist()):
            # Map class value to prototype index (prototypes ordered as [0, 1, ...])
            proto_idx = int(c)
            if proto_idx >= n_classes:
                continue
            mask = cal_labels == c
            if mask.any():
                cal_mean = cal_embeddings[mask].mean(dim=0)
                calibrated[proto_idx] = (self.alpha * prototypes[proto_idx] +
                                         (1 - self.alpha) * cal_mean)
        return calibrated

    def classify(self, query_embeddings, prototypes):
        """
        Classify by Euclidean distance to prototypes.

        Parameters
        ----------
        query_embeddings : (n_query, EMBEDDING_DIM)
        prototypes : (n_classes, EMBEDDING_DIM)

        Returns
        -------
        log_probs : (n_query, n_classes)
        predictions : (n_query,)
        """
        dists = torch.cdist(query_embeddings, prototypes)
        log_probs = F.log_softmax(-dists, dim=1)
        predictions = torch.argmin(dists, dim=1)
        return log_probs, predictions

    def forward(self, support_graphs, support_labels,
                query_graphs,
                cal_graphs=None, cal_labels=None):
        """
        Full forward: encode -> prototype -> (calibrate) -> classify.

        Parameters
        ----------
        support_graphs : list of Data
        support_labels : (n_support,)
        query_graphs : list of Data
        cal_graphs : list of Data, optional
        cal_labels : (n_cal,), optional

        Returns
        -------
        log_probs : (n_query, n_classes)
        predictions : (n_query,)
        """
        # Encode
        support_emb = self.encode(support_graphs)
        query_emb = self.encode(query_graphs)

        # Prototypes
        prototypes = self.compute_prototypes(support_emb, support_labels)

        # Calibrate if calibration data provided
        if cal_graphs is not None and cal_labels is not None:
            cal_emb = self.encode(cal_graphs)
            prototypes = self.calibrate_prototypes(prototypes, cal_emb, cal_labels)

        # Classify
        log_probs, predictions = self.classify(query_emb, prototypes)
        return log_probs, predictions

    def encode_with_attention(self, graphs):
        """
        Encode graphs and return attention weights (GAT only).
        For visualization.
        """
        if self.encoder_type != 'gat':
            raise ValueError("Attention weights only available for GAT encoder")

        batch = Batch.from_data_list(graphs)
        embedding, (attn_edge_index, attn_weights) = self.encoder.forward_with_attention(
            batch.x, batch.edge_index,
            edge_attr=batch.edge_attr,
            batch=batch.batch,
        )
        return embedding, (attn_edge_index, attn_weights)
