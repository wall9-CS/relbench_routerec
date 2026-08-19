# Rel-HM Seed-Time User-CF Snapshots

This experimental pipeline augments `rel-hm/user-item-purchase` ID-GNN samples
with one seed-time-specific `user_cf` fact table. Each row connects a source
customer to a top-k similar customer using binary historical purchase overlap.
The model-facing `user_cf` TensorFrame exposes only a constant `__const__`
feature.

Build default train/val/test snapshots:

```bash
python -m examples.build_hm_user_cf_snapshots \
  --dataset rel-hm \
  --task user-item-purchase \
  --window-weeks 8 \
  --min-overlap 2 \
  --top-k 32 \
  --alpha 0.5 \
  --output-root /data/seonghun/cf_snapshots
```

Train the user-CF augmented four-layer ID-GNN:

```bash
python -m examples.idgnn_recommendation_user_cf \
  --dataset rel-hm \
  --task user-item-purchase \
  --user-cf-snapshot-dir /data/seonghun/cf_snapshots/rel-hm/user-item-purchase/user_cf_window_8w_alpha_0.5_overlap_2_top32 \
  --num_layers 4 \
  --num_neighbors 128
```

Run coverage diagnostics:

```bash
python -m examples.evaluate_hm_user_cf_coverage \
  --dataset rel-hm \
  --task user-item-purchase \
  --user-cf-snapshot-dir /data/seonghun/cf_snapshots/rel-hm/user-item-purchase/user_cf_window_8w_alpha_0.5_overlap_2_top32
```

