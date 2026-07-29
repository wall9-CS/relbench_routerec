# Rel-Avito Seed-Time Ad-CF Snapshots

This experimental pipeline augments `rel-avito/user-ad-visit` ID-GNN samples
with seed-time-specific direct ad-ad CF edges. Each snapshot uses binary user-ad
visits in `(seed_time - 4 days, seed_time]` by default, computes sparse
`M.T @ M`, and stores support/score/rank for analysis. At runtime, the snapshot
rows are attached as direct `AdsInfo` edges only; no intermediate `ad_cf` node
or CF score feature is exposed to the model.

Build snapshots:

```bash
python -m examples.build_avito_cf_snapshots \
  --dataset rel-avito \
  --task user-ad-visit \
  --history-days 4 \
  --min-support 3 \
  --top-l 32 \
  --alpha 0.5 \
  --output-root /data/seonghun/cf_snapshots
```

Run coverage diagnostics:

```bash
python -m examples.evaluate_avito_cf_coverage \
  --dataset rel-avito \
  --task user-ad-visit \
  --cf-snapshot-dir /data/seonghun/cf_snapshots/rel-avito/user-ad-visit/window_4d_alpha_0.5_support_3_top32 \
  --splits val,test \
  --num-layers 3
```

Train the CF-augmented ID-GNN:

```bash
python -m examples.idgnn_recommendation_avito_cf \
  --dataset rel-avito \
  --task user-ad-visit \
  --cf-snapshot-dir /data/seonghun/cf_snapshots/rel-avito/user-ad-visit/window_4d_alpha_0.5_support_3_top32 \
  --num_layers 3 \
  --num_neighbors 128
```
