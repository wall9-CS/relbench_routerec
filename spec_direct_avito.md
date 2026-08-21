# Specification: Direct Source-to-Ad CF for Rel-Avito 2-Layer ID-GNN

## 0. Scope

- **Dataset:** `rel-avito`
- **Recommendation task:** `user-ad-visit`
- **Source entity:** `UserInfo`
- **Destination entity:** `AdsInfo`
- **Existing item-CF package:** `examples/avito_cf`
- **New target package:** `examples/avito_direct_cf`
- **Goal:** materialize ad-CF candidates derived from each source user's past
  visits as direct two-hop graph routes:

```text
UserInfo(src) <- direct_ad_cf <- AdsInfo(cf candidate)
```

This is a separate experiment. Do not replace `examples/idgnn_recommendation_avito_cf.py`
or mutate the existing `ad_cf` graph semantics.

## 1. Current vs Direct Route

Current Avito item-CF route needs four ID-GNN layers:

```text
UserInfo
  <- VisitStream
  <- AdsInfo(src historical ad)
  <- ad_cf
  <- AdsInfo(dst CF ad)
```

Direct route should need two ID-GNN layers:

```text
UserInfo
  <- direct_ad_cf
  <- AdsInfo(dst CF ad)
```

The direct route is still based on historical `VisitStream` interactions. The
source user's historical visit nodes are resolved during direct snapshot
materialization, then replaced by direct source-to-CF-node topology.

## 2. Direct Snapshot Semantics

For each seed time `t`:

1. Load the existing seed-time ad-CF snapshot from `examples/avito_cf`.
2. Collect source users that appear in task rows at exactly `t`.
3. Collect each source user's historical visited ads from `VisitStream` with:

```text
VisitStream.ViewDate <= t
```

4. Join historical ads to the ad-CF snapshot:

```text
source user -> historical ad -> ad-CF dst ad
```

5. Collapse duplicate `(source user, dst ad)` candidates into one direct CF row.

The source-history boundary intentionally uses all history up to `seed_time`.
This matches the current four-hop graph route, where base interaction expansion
is controlled by temporal sampling rather than the ad-CF rolling window.

Rows with null `UserID` or `AdID` are excluded by reusing
`filter_visit_interactions`.

## 3. Candidate Aggregation

Intermediate joined row:

```text
src_UserID, src_AdID, dst_AdID, cf_score, support, item_cf_rank
```

Collapse by:

```text
src_UserID, dst_AdID
```

Aggregate diagnostics:

| Column | Meaning |
|---|---|
| `direct_score` | sum of contributing `cf_score` values |
| `max_cf_score` | maximum contributing ad-CF score |
| `support_sum` | sum of contributing supports |
| `support_max` | maximum contributing support |
| `num_source_ads` | number of distinct historical source ads contributing |
| `best_src_AdID` | deterministic best historical ad provenance |
| `best_item_cf_rank` | ad-CF rank for `best_src_AdID -> dst_AdID` |

Pick `best_src_AdID` by:

1. `cf_score` descending
2. `support` descending
3. `item_cf_rank` ascending
4. `src_AdID` ascending

Rank candidates per source user by:

1. `direct_score` descending
2. `num_source_ads` descending
3. `max_cf_score` descending
4. `support_sum` descending
5. `dst_AdID` ascending

Keep at most:

```text
direct_top_k = 128
```

per source user by default. `direct_top_k=None` may be supported for coverage
debugging, but it should not be the training default.

Do not filter already-visited destination ads by default. Add `--filter-seen-dst`
only as an explicit ablation.

## 4. Direct Snapshot Format

Directory:

```text
<output-root>/
  rel-avito/
    user-ad-visit/
      direct_from_window_4d_alpha_0.5_support_3_itemtop32_srctop128/
        manifest.json
        direct_cf_snapshot_2015-05-08.parquet
```

Parquet schema:

| Column | Required dtype | Meaning |
|---|---:|---|
| `seed_time` | timestamp | exact task seed time |
| `src_UserID` | int64 | source/root Avito user |
| `dst_AdID` | int64 | direct CF candidate ad |
| `direct_score` | float32 | source-level aggregate score used for ranking |
| `max_cf_score` | float32 | best single ad-CF score |
| `support_sum` | int64 | sum of ad-CF support values |
| `support_max` | int64 | max ad-CF support value |
| `num_source_ads` | int64 | distinct source ads contributing |
| `best_src_AdID` | int64 | provenance ad for diagnostics |
| `best_item_cf_rank` | int32 | ad-CF rank of best provenance edge |
| `rank` | int32 | one-based direct rank within source user |

The model-facing TensorFrame for `direct_ad_cf` exposes only:

```text
__const__ = 1.0
```

Diagnostic score/rank/support columns are not model inputs in the first
implementation.

## 5. Manifest

`manifest.json` must include:

```json
{
  "dataset": "rel-avito",
  "task": "user-ad-visit",
  "cf_kind": "direct_source_item_from_item_cf",
  "source_entity_table": "UserInfo",
  "source_entity_col": "UserID",
  "dst_entity_table": "AdsInfo",
  "dst_entity_col": "AdID",
  "interaction_table": "VisitStream",
  "interaction_time_col": "ViewDate",
  "source_history_boundary": "interaction_time <= seed_time",
  "parent_item_cf_node_type": "ad_cf",
  "direct_cf_node_type": "direct_ad_cf",
  "direct_top_k": 128,
  "aggregate_score": "sum_cf_score",
  "filter_seen_dst": false,
  "score_used_as_model_input": false
}
```

Also copy or reference the parent ad-CF manifest fields that affect candidates:

```text
history_days
all_history
min_support
top_l
alpha
parent_item_cf_snapshot_dir
```

Manifest compatibility must reject mismatched dataset, task, parent ad-CF
configuration, `direct_top_k`, aggregation mode, and `filter_seen_dst`.

## 6. Graph Representation

Node type:

```python
DIRECT_AD_CF = "direct_ad_cf"
```

Edge types:

```python
DIRECT_CF_SRC_F2P = ("direct_ad_cf", "f2p_src_UserID", "UserInfo")
DIRECT_CF_SRC_REV = ("UserInfo", "rev_f2p_src_UserID", "direct_ad_cf")
DIRECT_CF_DST_F2P = ("direct_ad_cf", "f2p_dst_AdID", "AdsInfo")
DIRECT_CF_DST_REV = ("AdsInfo", "rev_f2p_dst_AdID", "direct_ad_cf")
```

For every snapshot row `r`:

```text
direct_ad_cf[r] -> src user
direct_ad_cf[r] -> dst ad
```

Construct reverse edge stores as well because `subgraph_type="bidirectional"` is
used after sampling. Edge indices must be `torch.long` and sorted with
`sort_edge_index`.

Required graph helpers:

```text
examples/avito_direct_cf/graph.py
  make_direct_ad_cf_tensor_frame
  direct_ad_cf_col_stats
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
UserInfo(src) -> direct_ad_cf rows -> dst AdsInfo
```

It disables accidental reverse direct-CF expansion:

```text
AdsInfo -> direct_ad_cf -> UserInfo
```

## 8. CLIs

Build direct snapshots from existing ad-CF snapshots:

```bash
python -m examples.build_avito_direct_cf_snapshots \
  --dataset rel-avito \
  --task user-ad-visit \
  --item-cf-snapshot-dir /data/seonghun/cf_snapshots/rel-avito/user-ad-visit/window_4d_alpha_0.5_support_3_top32 \
  --direct-top-k 128 \
  --output-root /data/seonghun/cf_snapshots
```

Coverage:

```bash
python -m examples.evaluate_avito_direct_cf_coverage \
  --dataset rel-avito \
  --task user-ad-visit \
  --direct-cf-snapshot-dir /data/seonghun/cf_snapshots/rel-avito/user-ad-visit/direct_from_window_4d_alpha_0.5_support_3_itemtop32_srctop128 \
  --splits val,test \
  --num-layers 2
```

Train 2-layer direct-CF ID-GNN:

```bash
python -m examples.idgnn_recommendation_avito_direct_cf \
  --dataset rel-avito \
  --task user-ad-visit \
  --direct-cf-snapshot-dir /data/seonghun/cf_snapshots/rel-avito/user-ad-visit/direct_from_window_4d_alpha_0.5_support_3_itemtop32_srctop128 \
  --num_layers 2 \
  --num_neighbors 128
```

The model and BCE/top-k evaluation loop stay unchanged from the current Avito
ID-GNN recommendation script.

## 9. Coverage Diagnostics

Direct coverage:

```text
candidate ads = direct snapshot dst_AdID values for src_UserID
```

For each task row:

1. load the exact seed-time direct snapshot,
2. read candidates for `src_UserID`,
3. compare candidates against the row's ground-truth `AdID` list,
4. report `coverage_rate`, `hit_rate`, `achievable_MAP`, and `coverage_gap`.

Coverage must require `num_layers >= 2`.

## 10. Tests

Use synthetic data only. Add:

```text
test/examples/test_avito_direct_cf_snapshot.py
test/examples/test_avito_direct_cf_io.py
test/examples/test_avito_direct_cf_graph.py
test/examples/test_avito_direct_cf_coverage.py
```

Required cases:

- source history uses `ViewDate <= seed_time`
- null `UserID` or `AdID` visits are excluded
- no future visit leakage
- source scope is only task source rows at seed time
- join from historical ad to parent ad-CF snapshot is exact
- duplicate `(src_UserID, dst_AdID)` rows collapse deterministically
- `direct_top_k` is enforced per source user
- default does not filter seen destination ads
- empty snapshots preserve schema
- manifest rejects incompatible parent ad-CF config
- graph edge roles match the two-hop route
- base graph is not mutated by snapshot attachment
- fanout requires `num_layers >= 2`
- direct-CF reverse expansion is disabled

## 11. Acceptance Criteria

- [ ] Existing Avito item-CF snapshots and training remain unchanged.
- [ ] Direct snapshots are exact per seed time.
- [ ] Direct snapshots are generated only from historical visits and parent
      ad-CF rows available at that seed time.
- [ ] `direct_ad_cf` exposes only `__const__` to the model.
- [ ] `num_layers=2` can sample `UserInfo -> direct_ad_cf -> AdsInfo`.
- [ ] Predictions preserve original task row order.
- [ ] Targeted direct-CF tests pass.

## 12. Out of Scope

- Recomputing ad-CF scores.
- Feeding `direct_score` or `cf_score` as model features.
- Combining old four-hop `ad_cf` and new direct-CF in one training script.
- Direct source-to-ad edges without a CF node.
- Approximate candidate generation.
