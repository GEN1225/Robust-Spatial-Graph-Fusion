# -*- coding: utf-8 -*-
"""
GNN Model Definitions
=====================
Includes: TKG-only, EKG-only, Fusion, and Segmented models.
All models share the same architecture parameters to ensure fair comparison.

NOTE: FusionModel and SoftGatedSegmentedModel contain proprietary core
algorithms. Only the public interfaces are provided here for reference.
The full implementations are available upon reasonable request for
academic collaboration.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, GCNConv


class TKGOnlyModel(nn.Module):
    """
    TKG-only model: uses only the spatial topology graph.
    Built with GAT (Graph Attention Network).
    """
    def __init__(self, input_dim, hidden_dim=64, heads=2, dropout=0.3):
        super().__init__()
        self.conv1 = GATConv(input_dim, hidden_dim, heads=heads, dropout=dropout, concat=True)
        self.conv2 = GATConv(hidden_dim * heads, hidden_dim, heads=1, dropout=dropout, concat=False)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, x, edge_index):
        h = F.elu(self.conv1(x, edge_index))
        h = F.elu(self.conv2(h, edge_index))
        return self.fc(h).squeeze(-1)


class EKGOnlyModel(nn.Module):
    """
    EKG-only model: uses only the ecological feature graph.
    Built with GCN (Graph Convolutional Network).
    """
    def __init__(self, input_dim, hidden_dim=64, dropout=0.3):
        super().__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, x, edge_index):
        h = F.relu(self.conv1(x, edge_index))
        h = F.relu(self.conv2(h, edge_index))
        return self.fc(h).squeeze(-1)


class FusionModel(nn.Module):
    """
    Dual-graph fusion model: TKG + EKG with cross-modal attention.

    Core innovation: novel cross-attention mechanism for fusing spatial
    topology graph (TKG) and ecological knowledge graph (EKG) embeddings.
    Full implementation is proprietary — contact authors for collaboration.

    Interface:
        forward(x, edge_index_tkg, edge_index_ekg) -> (predictions, attn_weights)
    """
    def __init__(self, input_dim, hidden_dim=64, heads=2, dropout=0.3):
        super().__init__()
        raise NotImplementedError(
            "FusionModel core algorithm is proprietary. "
            "Please contact the authors for academic collaboration."
        )

    def forward(self, x, edge_index_tkg, edge_index_ekg):
        raise NotImplementedError


class SoftGatedSegmentedModel(nn.Module):
    """
    Soft-gated segmented model with adaptive expert blending.

    Core innovation: learnable soft-gating mechanism for segment-aware
    prediction with automatic regime detection.
    Full implementation is proprietary — contact authors for collaboration.

    Interface:
        forward(x, edge_index) -> (predictions, gate_values)
    """
    def __init__(self, input_dim, hidden_dim=64, heads=2, dropout=0.3):
        super().__init__()
        raise NotImplementedError(
            "SoftGatedSegmentedModel core algorithm is proprietary. "
            "Please contact the authors for academic collaboration."
        )

    def forward(self, x, edge_index):
        raise NotImplementedError
