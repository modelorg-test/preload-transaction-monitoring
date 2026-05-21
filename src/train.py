"""Training pipeline for the AML GCN topology model.

Converts the transaction graph into PyTorch Geometric data objects,
trains a 3-layer GCN, and registers the model with MLflow.
"""

from __future__ import annotations

import argparse
import logging

import mlflow
import mlflow.pytorch
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv

from graph_builder import build_transaction_graph, compute_node_features

logger = logging.getLogger(__name__)

MODEL_NAME = "aml-network-topology-gcn"
EXPERIMENT_NAME = "aml/gcn-topology"


class AMLGraphNet(torch.nn.Module):
    """3-layer Graph Convolutional Network for AML cluster detection."""

    def __init__(self, in_channels: int = 3, hidden: int = 64, out_channels: int = 2) -> None:
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden)
        self.conv2 = GCNConv(hidden, hidden // 2)
        self.conv3 = GCNConv(hidden // 2, out_channels)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.3, training=self.training)
        x = F.relu(self.conv2(x, edge_index))
        x = self.conv3(x, edge_index)
        return F.log_softmax(x, dim=1)


def train(transactions_path: str, labels_path: str, epochs: int = 200) -> None:
    """Run GCN training.

    Args:
        transactions_path: Parquet file of 90-day transaction ledger.
        labels_path: CSV of {account_id, is_sar_cluster} ground truth.
        epochs: Number of training epochs.
    """
    mlflow.set_experiment(EXPERIMENT_NAME)

    txns = pd.read_parquet(transactions_path)
    labels_df = pd.read_csv(labels_path).set_index("account_id")
    G = build_transaction_graph(txns)
    features_df = compute_node_features(G)
    features_df = features_df.join(labels_df, how="left")
    features_df["is_sar_cluster"] = features_df["is_sar_cluster"].fillna(0).astype(int)

    node_list = list(features_df.index)
    node_idx = {n: i for i, n in enumerate(node_list)}

    x = torch.tensor(
        features_df[["hop_count", "total_flow_usd", "network_density"]].values,
        dtype=torch.float,
    )
    y = torch.tensor(features_df["is_sar_cluster"].values, dtype=torch.long)

    edges = [(node_idx[u], node_idx[v]) for u, v in G.edges() if u in node_idx and v in node_idx]
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()

    data = Data(x=x, edge_index=edge_index, y=y)
    model = AMLGraphNet(in_channels=3)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

    with mlflow.start_run():
        model.train()
        for epoch in range(epochs):
            optimizer.zero_grad()
            out = model(data.x, data.edge_index)
            loss = F.nll_loss(out, data.y)
            loss.backward()
            optimizer.step()
            if (epoch + 1) % 50 == 0:
                logger.info("Epoch %d/%d — loss: %.4f", epoch + 1, epochs, loss.item())

        model.eval()
        with torch.no_grad():
            probs = torch.exp(model(data.x, data.edge_index))[:, 1].numpy()
        auc = roc_auc_score(data.y.numpy(), probs)
        mlflow.log_metrics({"auc_roc": auc})
        mlflow.pytorch.log_model(model, artifact_path="model", registered_model_name=MODEL_NAME)
        logger.info("Training complete — AUC: %.4f", auc)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train AML GCN topology model")
    parser.add_argument("--transactions", required=True)
    parser.add_argument("--labels", required=True)
    parser.add_argument("--epochs", type=int, default=200)
    args = parser.parse_args()
    train(args.transactions, args.labels, args.epochs)
