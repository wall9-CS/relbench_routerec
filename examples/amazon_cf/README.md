# Rel-Amazon Seed-Time Product-CF Snapshots

This experimental pipeline augments Rel-Amazon recommendation tasks with one
seed-time-specific `product_cf` fact table. Supported tasks are:

- `user-item-purchase`
- `user-item-rate`
- `user-item-review`

Each task uses its own historical interaction filter over the `review` table,
then computes sparse binary customer-product co-occurrence with SciPy. Snapshot
metadata stores support, score, and rank for validation and analysis, but the
model-facing `product_cf` TensorFrame exposes only `__const__`.

Build snapshots:

```bash
python -m examples.build_amazon_cf_snapshots \
  --dataset rel-amazon \
  --task user-item-purchase \
  --history-days 91 \
  --min-support 3 \
  --top-l 32 \
  --alpha 0.5 \
  --output-root /data/seonghun/cf_snapshots
```

Run coverage diagnostics:

```bash
python -m examples.evaluate_amazon_cf_coverage \
  --dataset rel-amazon \
  --task user-item-purchase \
  --cf-snapshot-dir /data/seonghun/cf_snapshots/rel-amazon/user-item-purchase/window_91d_alpha_0.5_support_3_top32 \
  --splits val,test \
  --num-layers 4
```

Train the four-layer CF-augmented ID-GNN:

```bash
python -m examples.idgnn_recommendation_amazon_cf \
  --dataset rel-amazon \
  --task user-item-purchase \
  --cf-snapshot-dir /data/seonghun/cf_snapshots/rel-amazon/user-item-purchase/window_91d_alpha_0.5_support_3_top32 \
  --num_layers 4 \
  --num_neighbors 128
```

Use the same commands with `--task user-item-rate` or
`--task user-item-review` for the other recommendation tasks.
