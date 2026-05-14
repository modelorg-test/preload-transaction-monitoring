# Methodology & Assumptions

A 3-layer Graph Convolutional Network operating on 1st and 2nd degree transactional hop volumes. Node embeddings are aggregated via mean-pooling and classified using a fully connected head.

The model assumes that AML typologies manifest as structural anomalies in the payment graph (e.g., fan-in/fan-out patterns, circular flows).
