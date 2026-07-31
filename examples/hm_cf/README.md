# Rel-HM Seed-Time CF Snapshots

This experimental pipeline augments `rel-hm/user-item-purchase` ID-GNN samples
with one seed-time-specific `article_cf` fact table. Each snapshot uses binary
customer-article interactions in `(seed_time - 8 weeks, seed_time]` by default,
computes sparse `M.T @ M` with SciPy, stores support/score/rank for analysis,
and exposes only a constant `__const__` feature to the model.

Build all default train/val/test snapshots:

```bash
python -m examples.build_hm_cf_snapshots \
  --dataset rel-hm \
  --task user-item-purchase \
  --window-weeks 8 \
  --min-support 3 \
  --top-l 32 \
  --alpha 0.5 \
  --output-root /data/seonghun/cf_snapshots
```

Build a one-seed-time pilot:

```bash
python -m examples.build_hm_cf_snapshots \
  --dataset rel-hm \
  --task user-item-purchase \
  --window-weeks 8 \
  --min-support 3 \
  --top-l 32 \
  --alpha 0.5 \
  --seed-time 2020-03-02 \
  --output-root /tmp/relbench_cf_pilot
```

Train the CF-augmented four-layer ID-GNN:

```bash
python -m examples.idgnn_recommendation_cf \
  --dataset rel-hm \
  --task user-item-purchase \
  --cf-snapshot-dir /data/seonghun/cf_snapshots/rel-hm/user-item-purchase/window_8w_alpha_0.5_support_3_top32 \
  --num_layers 4 \
  --num_neighbors 128
```

Use `--all-history` instead of `--window-weeks` for the all-history ablation.
The preprocessing command writes `manifest.json` after every successful
snapshot, skips valid existing parquet files by default, and requires
`--overwrite` to replace invalid or stale files.
