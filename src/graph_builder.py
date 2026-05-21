"""Constructs the daily transaction graph for AML topology analysis.

Ingests 90-day payment ledger snapshots and builds a NetworkX directed
graph where nodes are accounts and edges are individual transactions.
Node and edge features are computed for GCN consumption.
"""

from __future__ import annotations

import logging

import networkx as nx
import pandas as pd

logger = logging.getLogger(__name__)


def build_transaction_graph(transactions: pd.DataFrame) -> nx.DiGraph:
    """Build a directed transaction graph from a ledger DataFrame.

    Args:
        transactions: DataFrame with columns:
            sender_account, receiver_account, amount_usd, timestamp_utc.

    Returns:
        DiGraph where nodes carry total_flow_usd and clustering_coeff,
        and edges carry amount_usd and timestamp features.
    """
    G = nx.DiGraph()

    for _, row in transactions.iterrows():
        src = row["sender_account"]
        dst = row["receiver_account"]
        amount = float(row["amount_usd"])

        if not G.has_node(src):
            G.add_node(src, total_flow_usd=0.0)
        if not G.has_node(dst):
            G.add_node(dst, total_flow_usd=0.0)

        G.nodes[src]["total_flow_usd"] += amount
        G.add_edge(src, dst, amount_usd=amount, timestamp=row["timestamp_utc"])

    # Compute local clustering coefficients on the undirected projection
    undirected = G.to_undirected()
    cc = nx.clustering(undirected)
    nx.set_node_attributes(G, cc, "network_density")

    logger.info(
        "Graph built: %d nodes, %d edges",
        G.number_of_nodes(),
        G.number_of_edges(),
    )
    return G


def compute_node_features(G: nx.DiGraph) -> pd.DataFrame:
    """Extract per-node feature vectors from the transaction graph.

    Args:
        G: Transaction graph from build_transaction_graph.

    Returns:
        DataFrame indexed by account_id with columns:
        hop_count, total_flow_usd, network_density.
    """
    records = []
    for node, attrs in G.nodes(data=True):
        records.append({
            "account_id": node,
            "hop_count": G.degree(node),
            "total_flow_usd": attrs.get("total_flow_usd", 0.0),
            "network_density": attrs.get("network_density", 0.0),
        })
    return pd.DataFrame(records).set_index("account_id")
