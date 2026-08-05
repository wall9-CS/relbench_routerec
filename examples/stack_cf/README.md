# Rel-Stack Seed-Time Post-CF Snapshots

This experimental pipeline augments `rel-stack/user-post-comment` with one
seed-time-specific `post_cf` fact table.

Historical interactions come from non-null `comments(UserId, PostId,
CreationDate)` rows. The snapshot builder computes sparse binary user-post
co-occurrence with SciPy. Snapshot metadata stores support, score, and rank for
validation and analysis, but the model-facing `post_cf` TensorFrame exposes only
`__const__`.

Build snapshots:

```bash
python -m examples.build_stack_cf_snapshots \
  --dataset rel-stack \
  --task user-post-comment \
  --history-days 91 \
  --min-support 3 \
  --top-l 32 \
  --alpha 0.5 \
  --output-root /data/seonghun/cf_snapshots
```

Run coverage diagnostics:

```bash
python -m examples.evaluate_stack_cf_coverage \
  --dataset rel-stack \
  --task user-post-comment \
  --cf-snapshot-dir /data/seonghun/cf_snapshots/rel-stack/user-post-comment/window_91d_alpha_0.5_support_3_top32 \
  --splits val,test \
  --num-layers 4
```

Train the four-layer CF-augmented ID-GNN:

```bash
python -m examples.idgnn_recommendation_stack_cf \
  --dataset rel-stack \
  --task user-post-comment \
  --cf-snapshot-dir /data/seonghun/cf_snapshots/rel-stack/user-post-comment/window_91d_alpha_0.5_support_3_top32 \
  --num_layers 4 \
  --num_neighbors 128
```

