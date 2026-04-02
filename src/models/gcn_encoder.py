"""
GCN (Graph Convolutional Network) Encoder — ablation baseline.

Same architecture shape as GAT but uses GCNConv instead.
No learned attention — uses PLV edge weights directly.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool, global_max_pool


class GCNEncoder(nn.Module):
    """
    Multi-layer GCN encoder with dual readout.
    Produces 128-dim graph-level embeddings.
    """

    def __init__(self, in_dim=13, hidden_dim=256, embed_dim=128, dropout=0.3):
        super().__init__()

        self.dropout = dropout

        self.conv1 = GCNConv(in_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.conv3 = GCNConv(hidden_dim, embed_dim)

        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.bn3 = nn.BatchNorm1d(embed_dim)

        self.mlp = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.BatchNorm1d(embed_dim),
            nn.ReLU(),
        )

    def forward(self, x, edge_index, edge_attr=None, batch=None):
        """
        Parameters
        ----------
        x : (N, in_dim)
        edge_index : (2, E)
        edge_attr : (E,) — edge weights used by GCNConv
        batch : (N,) — batch vector

        Returns
        -------
        embedding : (B, 128)
        """
        # GCNConv uses edge_weight parameter
        edge_weight = edge_attr

        x = F.elu(self.bn1(self.conv1(x, edge_index, edge_weight=edge_weight)))
        x = F.dropout(x, p=self.dropout, training=self.training)

        x = F.elu(self.bn2(self.conv2(x, edge_index, edge_weight=edge_weight)))
        x = F.dropout(x, p=self.dropout, training=self.training)

        x = F.elu(self.bn3(self.conv3(x, edge_index, edge_weight=edge_weight)))

        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)

        x_mean = global_mean_pool(x, batch)
        x_max = global_max_pool(x, batch)

        out = torch.cat([x_mean, x_max], dim=1)
        embedding = self.mlp(out)

        return embedding
