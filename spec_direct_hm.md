# Specification: Direct Source-to-Item CF for Rel-HM 2-Layer ID-GNN

## 0. Scope

- **Dataset:** `rel-hm`
- **Recommendation task:** `user-item-purchase`
- **Source entity:** `customer`
- **Destination entity:** `article`
- **Existing item-CF package:** `examples/hm_cf`
- **New target package:** `examples/hm_direct_cf`
- **Goal:** materialize item-CF candidates derived from each source customer's past
  interactions as direct two-hop graph routes:

```text
customer(src) <- direct_article_cf <- article(cf candidate)
```

This is a separate experiment. Do not replace `examples/idgnn_recommendation_cf.py`
or mutate the existing `article_cf` graph semantics.

## 1. Current vs Direct Route

Current HM item-CF route needs four ID-GNN layers:

```text
customer
  <- transactions
  <- article(src historical item)
  <- article_cf
  <- article(dst CF item)
```

Direct route should need two ID-GNN layers:

```text
customer
  <- direct_article_cf
  <- article(dst CF item)
```

The direct route is still based on historical interaction nodes. The difference
is that the source customer's historical interactions are resolved during
snapshot materialization, not during NeighborLoader expansion.

## 2. Direct Snapshot Semantics

For each seed time `t`:

1. Load the existing seed-time item-CF snapshot from `examples/hm_cf`.
2. Collect source customers that appear in task rows at exactly `t`.
3. Collect each source customer's historical interacted articles from
   `transactions` with:

```text
transactions.t_dat <= t
```

4. Join historical articles to the item-CF snapshot:

```text
source customer -> historical article -> item-CF dst article
```

5. Collapse duplicate `(source customer, dst article)` candidates into one
   direct CF row.

The source-history boundary intentionally uses all history up to `seed_time`.
This matches the current four-hop graph route, where the base interaction hop is
limited by temporal sampling (`time <= input_time`), not by the item-CF rolling
window. The item-CF snapshot itself still carries its configured rolling or
all-history co-occurrence window.

Do not use future labels or interactions with `t_dat > seed_time`.

## 3. Candidate Aggregation

Intermediate joined row:

```text
src_customer_id, src_article_id, dst_article_id, cf_score, support, item_cf_rank
```

Collapse by:

```text
src_customer_id, dst_article_id
```

Aggregate diagnostics:

| Column | Meaning |
|---|---|
| `direct_score` | sum of contributing `cf_score` values |
| `max_cf_score` | maximum contributing item-CF score |
| `support_sum` | sum of contributing supports |
| `support_max` | maximum contributing support |
| `num_source_articles` | number of distinct historical source articles contributing |
| `best_src_article_id` | deterministic best historical article provenance |
| `best_item_cf_rank` | item-CF rank for `best_src_article_id -> dst_article_id` |

Pick `best_src_article_id` by:

1. `cf_score` descending
2. `support` descending
3. `item_cf_rank` ascending
4. `src_article_id` ascending

Then rank candidates per source customer by:

1. `direct_score` descending
2. `num_source_articles` descending
3. `max_cf_score` descending
4. `support_sum` descending
5. `dst_article_id` ascending

Keep at most:

```text
direct_top_k = 128
```

per source customer by default. `direct_top_k=None` may be supported for coverage
debugging, but it should not be the training default because high-history
customers can create very large direct fact tables.

Do not filter already-seen destination articles by default. The existing four-hop
route can also reach a historical article when it appears as another historical
article's CF destination. Add `--filter-seen-dst` only as an explicit ablation.

## 4. Direct Snapshot Format

Directory:

```text
<output-root>/
  rel-hm/
    user-item-purchase/
      direct_from_window_8w_alpha_0.5_support_3_itemtop32_srctop128/
        manifest.json
        direct_cf_snapshot_2020-09-16.parquet
```

Parquet schema:

| Column | Required dtype | Meaning |
|---|---:|---|
| `seed_time` | timestamp | exact task seed time |
| `src_customer_id` | int64 | source/root customer |
| `dst_article_id` | int64 | direct CF candidate article |
| `direct_score` | float32 | source-level aggregate score used for ranking |
| `max_cf_score` | float32 | best single item-CF score |
| `support_sum` | int64 | sum of item-CF support values |
| `support_max` | int64 | max item-CF support value |
| `num_source_articles` | int64 | distinct source articles contributing |
| `best_src_article_id` | int64 | provenance article for diagnostics |
| `best_item_cf_rank` | int32 | item-CF rank of best provenance edge |
| `rank` | int32 | one-based direct rank within source customer |

The model-facing TensorFrame for `direct_article_cf` exposes only:

```text
__const__ = 1.0
```

None of the diagnostic score/rank/support columns are model inputs in the first
implementation.

## 5. Manifest

`manifest.json` must include:

```json
{
  "dataset": "rel-hm",
  "task": "user-item-purchase",
  "cf_kind": "direct_source_item_from_item_cf",
  "source_entity_table": "customer",
  "source_entity_col": "customer_id",
  "dst_entity_table": "article",
  "dst_entity_col": "article_id",
  "interaction_table": "transactions",
  "interaction_time_col": "t_dat",
  "source_history_boundary": "interaction_time <= seed_time",
  "parent_item_cf_node_type": "article_cf",
  "direct_cf_node_type": "direct_article_cf",
  "direct_top_k": 128,
  "aggregate_score": "sum_cf_score",
  "filter_seen_dst": false,
  "score_used_as_model_input": false
}
```

Also copy or reference the parent item-CF manifest fields that affect candidates:

```text
window_weeks
all_history
min_support
top_l
alpha
parent_item_cf_snapshot_dir
```

Manifest compatibility must reject mismatched dataset, task, parent item-CF
configuration, `direct_top_k`, aggregation mode, and `filter_seen_dst`.

## 6. Graph Representation

Node type:

```python
DIRECT_ARTICLE_CF = "direct_article_cf"
```

Edge types:

```python
DIRECT_CF_SRC_F2P = ("direct_article_cf", "f2p_src_customer_id", "customer")
DIRECT_CF_SRC_REV = ("customer", "rev_f2p_src_customer_id", "direct_article_cf")
DIRECT_CF_DST_F2P = ("direct_article_cf", "f2p_dst_article_id", "article")
DIRECT_CF_DST_REV = ("article", "rev_f2p_dst_article_id", "direct_article_cf")
```

For every snapshot row `r`:

```text
direct_article_cf[r] -> src customer
direct_article_cf[r] -> dst article
```

Construct reverse edge stores as well because `subgraph_type="bidirectional"` is
used after sampling. Edge indices must be `torch.long` and sorted with
`sort_edge_index`.

The graph attachment function must shallow-copy `base_data`, replace only the
direct CF node/edge stores, and leave `base_data` immutable.

Required graph helpers:

```text
examples/hm_direct_cf/graph.py
  make_direct_article_cf_tensor_frame
  direct_article_cf_col_stats
  attach_direct_cf_snapshot
  build_direct_cf_schema_template
  build_direct_cf_num_neighbors
```

## 7. Fanout Schedule

Require:

```text
num_layers >= 2
```

Keep normal base graph fanouts for non-direct-CF edges:

```python
hop_fanouts = [num_neighbors // 2**i for i in range(num_layers)]
```

Set all direct-CF edge fanouts to zero, then enable only:

```python
DIRECT_CF_SRC_F2P[0] = hop_fanouts[0]
DIRECT_CF_DST_REV[1] = hop_fanouts[1]
```

For `num_layers=2` and `num_neighbors=128`:

```python
DIRECT_CF_SRC_F2P: [128, 0]
DIRECT_CF_DST_REV: [0, 64]
DIRECT_CF_SRC_REV: [0, 0]
DIRECT_CF_DST_F2P: [0, 0]
```

This samples:

```text
customer(src) -> direct_article_cf rows -> dst articles
```

It does not sample the reverse route:

```text
article -> direct_article_cf -> customer
```

## 8. CLIs

Build direct snapshots from existing item-CF snapshots:

```bash
python -m examples.build_hm_direct_cf_snapshots \
  --dataset rel-hm \
  --task user-item-purchase \
  --item-cf-snapshot-dir /data/seonghun/cf_snapshots/rel-hm/user-item-purchase/window_8w_alpha_0.5_support_3_top32 \
  --direct-top-k 128 \
  --output-root /data/seonghun/cf_snapshots
```

Coverage:

```bash
python -m examples.evaluate_hm_direct_cf_coverage \
  --dataset rel-hm \
  --task user-item-purchase \
  --direct-cf-snapshot-dir /data/seonghun/cf_snapshots/rel-hm/user-item-purchase/direct_from_window_8w_alpha_0.5_support_3_itemtop32_srctop128 \
  --splits val,test \
  --num-layers 2
```

Train 2-layer direct-CF ID-GNN:

```bash
python -m examples.idgnn_recommendation_hm_direct_cf \
  --dataset rel-hm \
  --task user-item-purchase \
  --direct-cf-snapshot-dir /data/seonghun/cf_snapshots/rel-hm/user-item-purchase/direct_from_window_8w_alpha_0.5_support_3_itemtop32_srctop128 \
  --num_layers 2 \
  --num_neighbors 128
```

The model and loss stay unchanged from the current ID-GNN recommendation script.

## 9. Coverage Diagnostics

Direct coverage is simpler than current item-CF coverage:

```text
candidate articles = direct snapshot dst_article_id values for src_customer_id
```

For each task row:

1. load the exact seed-time direct snapshot,
2. read candidates for `src_customer_id`,
3. compare candidates against the row's ground-truth `article_id` list,
4. report `coverage_rate`, `hit_rate`, `achievable_MAP`, and `coverage_gap`.

Coverage must require `num_layers >= 2`.

## 10. Tests

Use synthetic data only. Add:

```text
test/examples/test_hm_direct_cf_snapshot.py
test/examples/test_hm_direct_cf_io.py
test/examples/test_hm_direct_cf_graph.py
test/examples/test_hm_direct_cf_coverage.py
```

Required cases:

- source history uses `t_dat <= seed_time`
- no future interaction leakage
- source scope is only task source rows at seed time
- join from historical article to parent item-CF snapshot is exact
- duplicate `(src_customer_id, dst_article_id)` rows collapse deterministically
- `direct_top_k` is enforced per source customer
- default does not filter seen destination articles
- empty snapshots preserve schema
- manifest rejects incompatible parent item-CF config
- graph edge roles match the two-hop route
- base graph is not mutated by snapshot attachment
- fanout requires `num_layers >= 2`
- direct-CF reverse expansion is disabled

## 11. Acceptance Criteria

- [ ] Existing HM item-CF snapshots and training remain unchanged.
- [ ] Direct snapshots are exact per seed time.
- [ ] Direct snapshots are generated only from historical interactions and parent
      item-CF rows available at that seed time.
- [ ] `direct_article_cf` exposes only `__const__` to the model.
- [ ] `num_layers=2` can sample `customer -> direct_article_cf -> article`.
- [ ] Predictions preserve original task row order.
- [ ] Targeted direct-CF tests pass.

## 12. Out of Scope

- Recomputing item-CF scores.
- Feeding `direct_score` or `cf_score` as model features.
- Combining old four-hop `article_cf` and new direct-CF in one training script.
- Direct source-to-article edges without a CF node.
- Approximate candidate generation.
