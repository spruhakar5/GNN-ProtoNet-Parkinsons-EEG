"""
GAT (Graph Attention Network) Encoder for EEG graphs.

Architecture:
  GAT Layer 1: in_dim -> 64, 4 heads (out: 256)
  GAT Layer 2: 256 -> 64, 4 heads (out: 256)
  GAT Layer 3: 256 -> 128, 1 head (out: 128)
  Readout: mean + max pooling (out: 256)
  MLP: 256 -> 128 (embedding dim)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool, global_max_pool


class GATEncoder(nn.Module):
    """
    Multi-layer GAT encoder with dual readout.
    Produces 128-dim graph-level embeddings.
    """

    def __init__(self, in_dim=13, hidden_dim=64, embed_dim=128,
                 heads=[4, 4, 1], dropout=0.3):
        super().__init__()

        self.dropout = dropout

        # Layer 1: in_dim -> hidden_dim * heads[0]
        self.conv1 = GATConv(in_dim, hidden_dim, heads=heads[0],
                             dropout=dropout, edge_dim=1)

        # Layer 2: hidden_dim * heads[0] -> hidden_dim * heads[1]
        self.conv2 = GATConv(hidden_dim * heads[0], hidden_dim, heads=heads[1],
                             dropout=dropout, edge_dim=1)

        # Layer 3: hidden_dim * heads[1] -> embed_dim * heads[2]
        self.conv3 = GATConv(hidden_dim * heads[1], embed_dim, heads=heads[2],
                             concat=False, dropout=dropout, edge_dim=1)

        # Batch norms
        self.bn1 = nn.BatchNorm1d(hidden_dim * heads[0])
        self.bn2 = nn.BatchNorm1d(hidden_dim * heads[1])
        self.bn3 = nn.BatchNorm1d(embed_dim)

        # MLP head: concat of mean + max pooling -> embed_dim
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.BatchNorm1d(embed_dim),
            nn.ReLU(),
        )

    def forward(self, x, edge_index, edge_attr=None, batch=None):
        """
        Parameters
        ----------
        x : (N, in_dim) — node features for all graphs in batch
        edge_index : (2, E) — edge indices
        edge_attr : (E,) or (E, 1) — edge weights (PLV)
        batch : (N,) — batch assignment vector

        Returns
        -------
        embedding : (B, 128) — graph-level embeddings
        """
        # Reshape edge_attr for GATConv
        if edge_attr is not None and edge_attr.dim() == 1:
            edge_attr = edge_attr.unsqueeze(-1)

        # Layer 1
        x = self.conv1(x, edge_index, edge_attr=edge_attr)
        x = self.bn1(x)
        x = F.elu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        # Layer 2
        x = self.conv2(x, edge_index, edge_attr=edge_attr)
        x = self.bn2(x)
        x = F.elu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        # Layer 3
        x = self.conv3(x, edge_index, edge_attr=edge_attr)
        x = self.bn3(x)
        x = F.elu(x)

        # Dual readout: mean + max pooling
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)

        x_mean = global_mean_pool(x, batch)  # (B, embed_dim)
        x_max = global_max_pool(x, batch)    # (B, embed_dim)

        # Concatenate and project
        out = torch.cat([x_mean, x_max], dim=1)  # (B, embed_dim * 2)
        embedding = self.mlp(out)                  # (B, embed_dim)

        return embedding

    def forward_with_attention(self, x, edge_index, edge_attr=None, batch=None):
        """
        Forward pass that also returns attention weights from the last layer.
        Used for visualization/interpretability.
        """
        if edge_attr is not None and edge_attr.dim() == 1:
            edge_attr = edge_attr.unsqueeze(-1)

        x = F.elu(self.bn1(self.conv1(x, edge_index, edge_attr=edge_attr)))
        x = F.dropout(x, p=self.dropout, training=self.training)

        x = F.elu(self.bn2(self.conv2(x, edge_index, edge_attr=edge_attr)))
        x = F.dropout(x, p=self.dropout, training=self.training)

        # Get attention weights from last layer
        x, (attn_edge_index, attn_weights) = self.conv3(
            x, edge_index, edge_attr=edge_attr, return_attention_weights=True
        )
        x = F.elu(self.bn3(x))

        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)

        x_mean = global_mean_pool(x, batch)
        x_max = global_max_pool(x, batch)
        out = torch.cat([x_mean, x_max], dim=1)
        embedding = self.mlp(out)

        return embedding, (attn_edge_index, attn_weights)
