# Specification: Direct Source-to-Post CF for Rel-Stack 2-Layer ID-GNN

## 0. Scope

- **Dataset:** `rel-stack`
- **Recommendation task:** `user-post-comment`
- **Source entity:** `users`
- **Destination entity:** `posts`
- **Existing item-CF package:** `examples/stack_cf`
- **New target package:** `examples/stack_direct_cf`
- **Goal:** materialize post-CF candidates derived from each source user's past
  comments as direct two-hop graph routes:

```text
users(src) <- direct_post_cf <- posts(cf candidate)
```

This is a separate experiment. Do not replace `examples/idgnn_recommendation_stack_cf.py`
or mutate the existing `post_cf` graph semantics.

## 1. Current vs Direct Route

Current Stack item-CF route needs four ID-GNN layers:

```text
users
  <- comments
  <- posts(src historical post)
  <- post_cf
  <- posts(dst CF post)
```

Direct route should need two ID-GNN layers:

```text
users
  <- direct_post_cf
  <- posts(dst CF post)
```

The direct route is still based on historical `comments` interactions. The
source user's historical comment nodes are resolved during direct snapshot
materialization, then replaced by direct source-to-CF-node topology.

## 2. Direct Snapshot Semantics

For each seed time `t`:

1. Load the existing seed-time post-CF snapshot from `examples/stack_cf`.
2. Collect source users that appear in task rows at exactly `t`.
3. Collect each source user's historical commented posts from `comments` with:

```text
comments.CreationDate <= t
```

4. Join historical posts to the post-CF snapshot:

```text
source user -> historical post -> post-CF dst post
```

5. Collapse duplicate `(source user, dst post)` candidates into one direct CF row.

The source-history boundary intentionally uses all history up to `seed_time`.
This matches the current four-hop graph route, where base interaction expansion
is controlled by temporal sampling rather than the post-CF rolling window.

Rows with null `UserId` or `PostId` are excluded by reusing
`filter_comment_interactions`.

## 3. Candidate Aggregation

Intermediate joined row:

```text
src_UserId, src_PostId, dst_PostId, cf_score, support, item_cf_rank
```

Collapse by:

```text
src_UserId, dst_PostId
```

Aggregate diagnostics:

| Column | Meaning |
|---|---|
| `direct_score` | sum of contributing `cf_score` values |
| `max_cf_score` | maximum contributing post-CF score |
| `support_sum` | sum of contributing supports |
| `support_max` | maximum contributing support |
| `num_source_posts` | number of distinct historical source posts contributing |
| `best_src_PostId` | deterministic best historical post provenance |
| `best_item_cf_rank` | post-CF rank for `best_src_PostId -> dst_PostId` |

Pick `best_src_PostId` by:

1. `cf_score` descending
2. `support` descending
3. `item_cf_rank` ascending
4. `src_PostId` ascending

Rank candidates per source user by:

1. `direct_score` descending
2. `num_source_posts` descending
3. `max_cf_score` descending
4. `support_sum` descending
5. `dst_PostId` ascending

Keep at most:

```text
direct_top_k = 128
```

per source user by default. This default is deliberately larger than the current
per-post `top_l=32`, but bounded enough for training. It also covers Stack's
`eval_k=100` better than a smaller default.

Do not filter already-commented destination posts by default. Add
`--filter-seen-dst` only as an explicit ablation.

## 4. Direct Snapshot Format

Directory:

```text
<output-root>/
  rel-stack/
    user-post-comment/
      direct_from_window_91d_alpha_0.5_support_3_itemtop32_srctop128/
        manifest.json
        direct_cf_snapshot_2020-10-01.parquet
```

Parquet schema:

| Column | Required dtype | Meaning |
|---|---:|---|
| `seed_time` | timestamp | exact task seed time |
| `src_UserId` | int64 | source/root Stack user |
| `dst_PostId` | int64 | direct CF candidate post |
| `direct_score` | float32 | source-level aggregate score used for ranking |
| `max_cf_score` | float32 | best single post-CF score |
| `support_sum` | int64 | sum of post-CF support values |
| `support_max` | int64 | max post-CF support value |
| `num_source_posts` | int64 | distinct source posts contributing |
| `best_src_PostId` | int64 | provenance post for diagnostics |
| `best_item_cf_rank` | int32 | post-CF rank of best provenance edge |
| `rank` | int32 | one-based direct rank within source user |

The model-facing TensorFrame for `direct_post_cf` exposes only:

```text
__const__ = 1.0
```

Diagnostic score/rank/support columns are not model inputs in the first
implementation.

## 5. Manifest

`manifest.json` must include:

```json
{
  "dataset": "rel-stack",
  "task": "user-post-comment",
  "cf_kind": "direct_source_item_from_item_cf",
  "source_entity_table": "users",
  "source_entity_col": "UserId",
  "dst_entity_table": "posts",
  "dst_entity_col": "PostId",
  "interaction_table": "comments",
  "interaction_time_col": "CreationDate",
  "source_history_boundary": "interaction_time <= seed_time",
  "parent_item_cf_node_type": "post_cf",
  "direct_cf_node_type": "direct_post_cf",
  "direct_top_k": 128,
  "aggregate_score": "sum_cf_score",
  "filter_seen_dst": false,
  "score_used_as_model_input": false
}
```

Also copy or reference the parent post-CF manifest fields that affect candidates:

```text
history_days
all_history
min_support
top_l
alpha
parent_item_cf_snapshot_dir
```

Manifest compatibility must reject mismatched dataset, task, parent post-CF
configuration, `direct_top_k`, aggregation mode, and `filter_seen_dst`.

## 6. Graph Representation

Node type:

```python
DIRECT_POST_CF = "direct_post_cf"
```

Edge types:

```python
DIRECT_CF_SRC_F2P = ("direct_post_cf", "f2p_src_UserId", "users")
DIRECT_CF_SRC_REV = ("users", "rev_f2p_src_UserId", "direct_post_cf")
DIRECT_CF_DST_F2P = ("direct_post_cf", "f2p_dst_PostId", "posts")
DIRECT_CF_DST_REV = ("posts", "rev_f2p_dst_PostId", "direct_post_cf")
```

For every snapshot row `r`:

```text
direct_post_cf[r] -> src user
direct_post_cf[r] -> dst post
```

Construct reverse edge stores as well because `subgraph_type="bidirectional"` is
used after sampling. Edge indices must be `torch.long` and sorted with
`sort_edge_index`.

Required graph helpers:

```text
examples/stack_direct_cf/graph.py
  make_direct_post_cf_tensor_frame
  direct_post_cf_col_stats
  attach_direct_cf_snapshot
  build_direct_cf_schema_template
  build_direct_cf_num_neighbors
```

The graph attachment function must shallow-copy `base_data`, replace only the
direct CF node/edge stores, and leave `base_data` immutable.

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
users(src) -> direct_post_cf rows -> dst posts
```

It disables accidental reverse direct-CF expansion:

```text
posts -> direct_post_cf -> users
```

## 8. CLIs

Build direct snapshots from existing post-CF snapshots:

```bash
python -m examples.build_stack_direct_cf_snapshots \
  --dataset rel-stack \
  --task user-post-comment \
  --item-cf-snapshot-dir /data/seonghun/cf_snapshots/rel-stack/user-post-comment/window_91d_alpha_0.5_support_3_top32 \
  --direct-top-k 128 \
  --output-root /data/seonghun/cf_snapshots
```

Coverage:

```bash
python -m examples.evaluate_stack_direct_cf_coverage \
  --dataset rel-stack \
  --task user-post-comment \
  --direct-cf-snapshot-dir /data/seonghun/cf_snapshots/rel-stack/user-post-comment/direct_from_window_91d_alpha_0.5_support_3_itemtop32_srctop128 \
  --splits val,test \
  --num-layers 2
```

Train 2-layer direct-CF ID-GNN:

```bash
python -m examples.idgnn_recommendation_stack_direct_cf \
  --dataset rel-stack \
  --task user-post-comment \
  --direct-cf-snapshot-dir /data/seonghun/cf_snapshots/rel-stack/user-post-comment/direct_from_window_91d_alpha_0.5_support_3_itemtop32_srctop128 \
  --num_layers 2 \
  --num_neighbors 128
```

The model and BCE/top-k evaluation loop stay unchanged from the current Stack
ID-GNN recommendation script.

## 9. Coverage Diagnostics

Direct coverage:

```text
candidate posts = direct snapshot dst_PostId values for src_UserId
```

For each task row:

1. load the exact seed-time direct snapshot,
2. read candidates for `src_UserId`,
3. compare candidates against the row's ground-truth `PostId` list,
4. report `coverage_rate`, `hit_rate`, `achievable_MAP`, and `coverage_gap`.

Coverage must require `num_layers >= 2`.

## 10. Tests

Use synthetic data only. Add:

```text
test/examples/test_stack_direct_cf_snapshot.py
test/examples/test_stack_direct_cf_io.py
test/examples/test_stack_direct_cf_graph.py
test/examples/test_stack_direct_cf_coverage.py
```

Required cases:

- source history uses `CreationDate <= seed_time`
- null `UserId` or `PostId` comments are excluded
- no future comment leakage
- source scope is only task source rows at seed time
- join from historical post to parent post-CF snapshot is exact
- duplicate `(src_UserId, dst_PostId)` rows collapse deterministically
- `direct_top_k` is enforced per source user
- default does not filter seen destination posts
- empty snapshots preserve schema
- manifest rejects incompatible parent post-CF config
- graph edge roles match the two-hop route
- base graph is not mutated by snapshot attachment
- fanout requires `num_layers >= 2`
- direct-CF reverse expansion is disabled

## 11. Acceptance Criteria

- [ ] Existing Stack item-CF snapshots and training remain unchanged.
- [ ] Direct snapshots are exact per seed time.
- [ ] Direct snapshots are generated only from historical comments and parent
      post-CF rows available at that seed time.
- [ ] `direct_post_cf` exposes only `__const__` to the model.
- [ ] `num_layers=2` can sample `users -> direct_post_cf -> posts`.
- [ ] Predictions preserve original task row order.
- [ ] Targeted direct-CF tests pass.

## 12. Out of Scope

- Recomputing post-CF scores.
- Feeding `direct_score` or `cf_score` as model features.
- Combining old four-hop `post_cf` and new direct-CF in one training script.
- Direct source-to-post edges without a CF node.
- Approximate candidate generation.
