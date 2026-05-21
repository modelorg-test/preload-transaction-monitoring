"""Graph construction for the transaction monitoring GCN.

Builds a transaction graph from raw payment data where:
- Nodes represent accounts (customers, merchants, intermediaries)
- Edges represent transactions between accounts
- Node features include account metadata and aggregated transaction stats
- Edge features include amount, frequency, and temporal patterns
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class TransactionGraph:
    """Lightweight representation of a transaction graph.

    Attributes
    ----------
    node_features : np.ndarray
        Node feature matrix of shape (num_nodes, num_features).
    edge_index : np.ndarray
        Edge list of shape (2, num_edges).
    edge_features : np.ndarray
        Edge feature matrix of shape (num_edges, num_edge_features).
    labels : np.ndarray
        Node-level labels (0=clean, 1=suspicious).
    node_ids : list[str]
        Original account identifiers.
    """

    node_features: np.ndarray
    edge_index: np.ndarray
    edge_features: np.ndarray
    labels: np.ndarray
    node_ids: list[str]


NODE_FEATURES = [
    "account_age_days",
    "avg_daily_volume",
    "transaction_count_30d",
    "unique_counterparties_30d",
    "max_single_txn_amount",
    "country_risk_score",
    "is_pep",
    "kyc_score",
]

EDGE_FEATURES = [
    "total_amount",
    "transaction_count",
    "avg_amount",
    "std_amount",
    "days_active",
    "is_cross_border",
]


def build_graph_from_transactions(
    transactions: pd.DataFrame,
    accounts: pd.DataFrame,
) -> TransactionGraph:
    """Build a transaction graph from raw payment data.

    Parameters
    ----------
    transactions : pd.DataFrame
        Columns: sender_id, receiver_id, amount, timestamp, currency.
    accounts : pd.DataFrame
        Columns: account_id, plus NODE_FEATURES columns.

    Returns
    -------
    TransactionGraph
        Graph ready for GCN input.
    """
    # Build node index
    all_ids = sorted(set(transactions["sender_id"]) | set(transactions["receiver_id"]))
    id_to_idx = {aid: i for i, aid in enumerate(all_ids)}

    # Node features from account metadata
    account_lookup = accounts.set_index("account_id")
    node_feat_list = []
    for aid in all_ids:
        if aid in account_lookup.index:
            row = account_lookup.loc[aid]
            feats = [float(row.get(f, 0.0)) for f in NODE_FEATURES]
        else:
            feats = [0.0] * len(NODE_FEATURES)
        node_feat_list.append(feats)

    node_features = np.array(node_feat_list, dtype=np.float32)

    # Edge aggregation
    edges = transactions.groupby(["sender_id", "receiver_id"]).agg(
        total_amount=("amount", "sum"),
        transaction_count=("amount", "count"),
        avg_amount=("amount", "mean"),
        std_amount=("amount", "std"),
        first_txn=("timestamp", "min"),
        last_txn=("timestamp", "max"),
    ).reset_index()

    edges["std_amount"] = edges["std_amount"].fillna(0)
    edges["days_active"] = (
        pd.to_datetime(edges["last_txn"]) - pd.to_datetime(edges["first_txn"])
    ).dt.days.clip(lower=0)
    edges["is_cross_border"] = 0  # Placeholder

    src_indices = [id_to_idx[s] for s in edges["sender_id"]]
    dst_indices = [id_to_idx[d] for d in edges["receiver_id"]]
    edge_index = np.array([src_indices, dst_indices], dtype=np.int64)

    edge_features = edges[EDGE_FEATURES].values.astype(np.float32)

    # Labels (default all clean — would come from SAR filings)
    labels = np.zeros(len(all_ids), dtype=np.int64)

    logger.info(
        "Built graph: %d nodes, %d edges, %d node features, %d edge features",
        len(all_ids), len(edges), len(NODE_FEATURES), len(EDGE_FEATURES),
    )

    return TransactionGraph(
        node_features=node_features,
        edge_index=edge_index,
        edge_features=edge_features,
        labels=labels,
        node_ids=all_ids,
    )
