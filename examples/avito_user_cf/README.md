# Rel-Avito Seed-Time User-CF Snapshots

This experimental pipeline augments `rel-avito/user-ad-visit` ID-GNN samples
with one seed-time-specific `user_cf` fact table. Each row connects a source
`UserInfo` user to a top-k similar user using binary historical `VisitStream`
ad-overlap. The model-facing `user_cf` TensorFrame exposes only `__const__`.

Build snapshots:

```bash
python -m examples.build_avito_user_cf_snapshots \
  --dataset rel-avito \
  --task user-ad-visit \
  --history-days 4 \
  --min-overlap 2 \
  --top-k 32 \
  --alpha 0.5 \
  --output-root /data/seonghun/cf_snapshots
```

Train:

```bash
python -m examples.idgnn_recommendation_avito_user_cf \
  --dataset rel-avito \
  --task user-ad-visit \
  --user-cf-snapshot-dir /data/seonghun/cf_snapshots/rel-avito/user-ad-visit/user_cf_window_4d_alpha_0.5_overlap_2_top32 \
  --num_layers 4 \
  --num_neighbors 128
```

