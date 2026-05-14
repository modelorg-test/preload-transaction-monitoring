# Overview & Strategy

Graph convolutional network (GCN) evaluating ring structures and layering topology within payment networks. Identifies clusters of accounts exhibiting coordinated fund movement patterns indicative of money laundering.

## Inputs

- `hop_count`: Degree of separation in the transaction graph (int)
- `total_flow_usd`: Aggregate USD value flowing through a node (float)
- `network_density`: Local clustering coefficient of the subgraph (float)

## Outputs

- `cluster_risk_score`: Network-level suspicion score [0, 1]
- `flagged_accounts`: List of account IDs within suspicious clusters
