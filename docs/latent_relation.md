# Learned Latent Destination Relations

This implementation adds a supervised destination-to-destination relation for
RelBench recommendation tasks:

```text
learn globally -> retrieve sparse distant candidates -> reason locally with ID-GNN
```

The relation generator is trained separately from ID-GNN using only training
recommendation labels. For a query `(u, t, Y)`, the adapter builds a recent
historical destination set `H_{u,t}` from interactions with timestamp
`<= t`. A future destination `j` is scored by:

```text
s(i, j) = normalize(W_Q h_i)^T normalize(W_K h_j) / tau
g(u, j, t) = LogSumExp_{i in H_{u,t}} s(i, j)
```

The training loss is a multi-positive sampled softmax over true future task
labels and uniform random negatives. It does not use k-hop locality, CF edges,
or manually selected item attributes as supervision.

## Pipeline

1. Train the relation generator:

```bash
python -m examples.latent_relation.train \
  --dataset rel-hm \
  --task user-item-purchase \
  --checkpoint-dir outputs/latent_checkpoints \
  --history-limit 64 \
  --relation-dim 128 \
  --temperature 0.07 \
  --num-negatives 256 \
  --epochs 5
```

2. Materialize seed-time snapshots:

```bash
python -m examples.latent_relation.build_snapshots \
  --dataset rel-hm \
  --task user-item-purchase \
  --checkpoint outputs/latent_checkpoints/rel-hm_user-item-purchase_latent_relation.pt \
  --snapshot-dir outputs/latent_snapshots/rel-hm/user-item-purchase \
  --history-limit 64 \
  --top-l 32 \
  --retrieval-backend torch
```

3. Evaluate candidate locality before ID-GNN training:

```bash
python -m examples.latent_relation.evaluate_coverage \
  --dataset rel-hm \
  --task user-item-purchase \
  --snapshot-dir outputs/latent_snapshots/rel-hm/user-item-purchase \
  --history-limit 64 \
  --output-json outputs/latent_coverage_hm.json \
  --output-csv outputs/latent_coverage_hm.csv
```

4. Train ID-GNN with latent edges:

```bash
python examples/idgnn_recommendation.py \
  --dataset rel-hm \
  --task user-item-purchase \
  --augmentation latent \
  --latent-snapshot-dir outputs/latent_snapshots/rel-hm/user-item-purchase \
  --budget-mode additive \
  --num_layers 3 \
  --top-l 32
```

Equivalent Avito commands:

```bash
python -m examples.latent_relation.train \
  --dataset rel-avito \
  --task user-ad-visit \
  --checkpoint-dir outputs/latent_checkpoints

python -m examples.latent_relation.build_snapshots \
  --dataset rel-avito \
  --task user-ad-visit \
  --checkpoint outputs/latent_checkpoints/rel-avito_user-ad-visit_latent_relation.pt \
  --snapshot-dir outputs/latent_snapshots/rel-avito/user-ad-visit \
  --top-l 32

python -m examples.latent_relation.evaluate_coverage \
  --dataset rel-avito \
  --task user-ad-visit \
  --snapshot-dir outputs/latent_snapshots/rel-avito/user-ad-visit

python examples/idgnn_recommendation.py \
  --dataset rel-avito \
  --task user-ad-visit \
  --augmentation latent \
  --latent-snapshot-dir outputs/latent_snapshots/rel-avito/user-ad-visit \
  --budget-mode additive \
  --num_layers 3 \
  --top-l 32
```

## Cache Structure

Snapshots are parquet files plus `manifest.json` in the requested
`--snapshot-dir`. Each snapshot contains:

```text
src_id, dst_id, score, rank, seed_time
```

The manifest records dataset, task, history limit, relation dimension,
temperature, retrieval backend, top-L, checkpoint path/hash, training cutoff,
and software version. Existing valid snapshots are reused unless `--force` is
passed.

## Leakage Prevention

Adapters are the only dataset-specific history layer. The MVP adapters are:

- `rel-hm/user-item-purchase`: `transactions.customer_id -> article_id`, time `t_dat`
- `rel-avito/user-ad-visit`: `VisitStream.UserID -> AdID`, time `ViewDate`

History construction uses sorted interaction timestamps and
`searchsorted(..., side="right")`, so only events with `interaction_time <=
seed_time` are visible. Relation training reads only `task.get_table("train")`.
Validation and test snapshot construction use the frozen checkpoint and do not
read validation/test labels except for coverage evaluation after snapshots are
already built.

## Graph Injection

Latent snapshots are inserted as direct destination-to-destination edge types:

```text
(dst_type, "latent_relation", dst_type)
(dst_type, "rev_latent_relation", dst_type)
```

No latent node type is introduced. ID-GNN uses the existing model and
neighbor-sampling pipeline. The snapshot is selected exactly by seed time using
the same seed-time grouping utility used by the CF baseline.

## Budget Modes

`--budget-mode additive` keeps the original per-hop fanouts and samples latent
reverse edges at hop 3 on top.

`--budget-mode fixed` reserves the hop-3 fanout for latent edges and subtracts
that reserve from non-latent edge types at the same hop. PyG budgets are
per-edge-type, so this is an approximate deterministic fixed-budget allocation.

## Scalability Notes

Materialization computes destination embeddings in chunks and retrieval scans
candidate chunks. It stores only top-L buffers per query chunk and never forms a
full destination-by-destination similarity matrix. FAISS is optional and is
imported only when `--retrieval-backend faiss` is requested.

Known MVP limitations:

- Single relation head only.
- Uniform random negatives only.
- Destination encoding is table-feature based.
- Coverage diagnostics use the direct history plus latent-neighbor route, not a
full sampled PyG subgraph replay.
