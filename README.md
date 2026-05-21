# Transaction Monitoring — GCN Topology Model

Graph Convolutional Network for suspicious transaction topology
detection using entity relationship analysis.

## Quick Start

```bash
pip install -r requirements.txt
python -m src.pipelines.train --data /path/to/transaction_graph.parquet
```

> **Note:** This model requires `torch` and `torch_geometric` and
> cannot be run with sklearn built-in datasets. See the training
> script for the expected data format.
