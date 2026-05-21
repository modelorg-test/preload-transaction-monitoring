"""Batch inference for the AML GCN topology model.

Scores overnight graph reconstructions and publishes cluster risk
scores to the investigation case management system by 06:00 UTC.
"""

from __future__ import annotations

import logging

import pandas as pd
import torch
from torch_geometric.data import Data

from graph_builder import build_transaction_graph, compute_node_features

logger = logging.getLogger(__name__)

RISK_THRESHOLD = 0.72
FATF_GREY_LIST_THRESHOLD = 0.55


def score_graph(
    model: torch.nn.Module,
    transactions: pd.DataFrame,
    high_risk_jurisdiction: bool = False,
) -> pd.DataFrame:
    """Score all nodes in the transaction graph.

    Args:
        model: Trained AMLGraphNet instance.
        transactions: 90-day transaction ledger DataFrame.
        high_risk_jurisdiction: If True, apply FATF grey-list threshold.

    Returns:
        DataFrame with account_id, cluster_risk_score, flagged columns.
    """
    G = build_transaction_graph(transactions)
    features_df = compute_node_features(G)
    node_list = list(features_df.index)
    node_idx = {n: i for i, n in enumerate(node_list)}

    x = torch.tensor(
        features_df[["hop_count", "total_flow_usd", "network_density"]].values,
        dtype=torch.float,
    )
    edges = [(node_idx[u], node_idx[v]) for u, v in G.edges() if u in node_idx and v in node_idx]
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous() if edges else torch.zeros((2, 0), dtype=torch.long)

    model.eval()
    with torch.no_grad():
        probs = torch.exp(model(x, edge_index))[:, 1].numpy()

    threshold = FATF_GREY_LIST_THRESHOLD if high_risk_jurisdiction else RISK_THRESHOLD
    results = features_df.copy()
    results["cluster_risk_score"] = probs
    results["flagged"] = probs >= threshold
    logger.info(
        "Scored %d accounts — %d flagged (threshold=%.2f)",
        len(results),
        results["flagged"].sum(),
        threshold,
    )
    return results[["cluster_risk_score", "flagged"]].reset_index()
