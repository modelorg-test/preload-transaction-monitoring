"""GCN model architecture for suspicious transaction detection.

Implements a 3-layer Graph Convolutional Network for node-level
classification (clean vs. suspicious accounts) using PyTorch Geometric.

This module defines the model architecture only — training and
inference are handled by the pipelines package.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


# ── Architecture specification ──────────────────────────────────────
# The actual GCN implementation requires torch and torch_geometric.
# We define the architecture configuration here for documentation
# and reference the PyTorch implementation below.

GCN_CONFIG = {
    "in_channels": 8,         # Number of node features
    "hidden_channels": 64,    # Hidden layer width
    "num_layers": 3,          # GCN conv layers
    "out_channels": 2,        # Binary classification
    "dropout": 0.3,
    "activation": "relu",
    "pooling": "mean",        # Global mean pooling for graph-level tasks
}


def _build_gcn():
    """Build the GCN model.

    This requires torch and torch_geometric to be installed.
    In environments without these dependencies, the architecture
    config above serves as documentation.

    Returns
    -------
    torch.nn.Module
        GCN model for node classification.

    Raises
    ------
    ImportError
        If torch or torch_geometric is not available.
    """
    import torch  # noqa: PLC0415
    import torch.nn.functional as F  # noqa: PLC0415
    from torch_geometric.nn import GCNConv  # noqa: PLC0415

    class SuspiciousActivityGCN(torch.nn.Module):
        """3-layer GCN for suspicious account detection."""

        def __init__(self) -> None:
            super().__init__()
            cfg = GCN_CONFIG
            self.conv1 = GCNConv(cfg["in_channels"], cfg["hidden_channels"])
            self.conv2 = GCNConv(cfg["hidden_channels"], cfg["hidden_channels"])
            self.conv3 = GCNConv(cfg["hidden_channels"], cfg["out_channels"])
            self.dropout = cfg["dropout"]

        def forward(self, x, edge_index):
            x = F.relu(self.conv1(x, edge_index))
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = F.relu(self.conv2(x, edge_index))
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = self.conv3(x, edge_index)
            return F.log_softmax(x, dim=1)

    return SuspiciousActivityGCN()
