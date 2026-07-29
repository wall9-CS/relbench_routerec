# Specification: Seed-Time-Specific CF Table for Rel-Avito Recommendation

## 0. Scope

- **Dataset:** `rel-avito`
- **Recommendation task:** `user-ad-visit`
- **Source entity:** `UserInfo`
- **Destination entity:** `AdsInfo`
- **Interaction table:** `VisitStream(UserID, AdID, ViewDate, ...)`
- **Goal:** augment each seed-time-specific ID-GNN graph with direct
  `AdsInfo`-to-`AdsInfo` CF edges so candidate ads can be reached without an
  intermediate CF node.

Rel-HM and Rel-Amazon code should remain compatible. Avito-specific code lives
under `examples/avito_cf` and uses Avito names (`src_AdID`, `dst_AdID`)
instead of article/product names.

## 1. Candidate Route

For each seed time, attach exactly one direct ad-CF snapshot:

```text
UserInfo
  <- VisitStream
  <- AdsInfo (historically visited source ad)
  <- AdsInfo (CF destination candidate)
```

Under RelBench foreign-key edge naming:

```text
UserInfo
  <-[VisitStream.f2p_UserID]- VisitStream
  <-[AdsInfo.rev_f2p_AdID]- AdsInfo(src)
  <-[AdsInfo.rev_cf_src_to_dst_AdID]- AdsInfo(dst)
```

Use `num_layers >= 3`. The default CF experiment uses `num_layers=3`.

## 2. Interaction Semantics

Use historical visit interactions from `VisitStream`:

```text
UserID, AdID, ViewDate
```

Rows with null `UserID` or `AdID` are excluded. For seed time `t`, the default
rolling history window is:

```text
t - 4 days < ViewDate <= t
```

This matches the `user-ad-visit` prediction horizon. Also support
`--all-history`, meaning `ViewDate <= seed_time`.

## 3. Sparse CF Construction

For every seed time:

1. Sort interactions by `ViewDate` once.
2. Slice the rolling window with `numpy.searchsorted`.
3. Deduplicate `(UserID, AdID)`.
4. Factorize active users in the slice.
5. Build sparse CSR matrix `M` with shape `(num_active_users, num_ads)`.
6. Compute sparse ad co-occurrence:

```python
C = (M.T @ M).tocsr()
```

7. Remove the diagonal.
8. Keep pairs with `support >= min_support`.
9. Score each ordered pair:

```text
score(i -> j) = C[i, j] / (count_i ** alpha * count_j ** (1 - alpha))
```

10. Keep top `top_l` per source ad, ordered by:
    1. `cf_score` descending
    2. `support` descending
    3. `dst_AdID` ascending

Defaults:

```text
history_days = 4
min_support = 3
top_l = 32
alpha = 0.5
```

## 4. Snapshot Format

Directory:

```text
<output-root>/
  rel-avito/
    user-ad-visit/
      window_4d_alpha_0.5_support_3_top32/
        manifest.json
        cf_snapshot_2015-05-08.parquet
        cf_snapshot_2015-05-14.parquet
```

Parquet schema:

| Column | Meaning |
|---|---|
| `seed_time` | exact task seed time |
| `src_AdID` | source ad index |
| `dst_AdID` | CF destination ad index |
| `support` | co-occurrence count |
| `cf_score` | normalized CF score |
| `rank` | one-based rank within source ad |

The graph-facing CF rows are attached as direct edge indices only. `support`,
`cf_score`, and `rank` are not model inputs.

## 5. CLIs

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

Coverage diagnostics:

```bash
python -m examples.evaluate_avito_cf_coverage \
  --dataset rel-avito \
  --task user-ad-visit \
  --cf-snapshot-dir /data/seonghun/cf_snapshots/rel-avito/user-ad-visit/window_4d_alpha_0.5_support_3_top32 \
  --splits val,test \
  --num-layers 3
```

Train CF-augmented ID-GNN:

```bash
python -m examples.idgnn_recommendation_avito_cf \
  --dataset rel-avito \
  --task user-ad-visit \
  --cf-snapshot-dir /data/seonghun/cf_snapshots/rel-avito/user-ad-visit/window_4d_alpha_0.5_support_3_top32 \
  --num_layers 3 \
  --num_neighbors 128
```

## 6. Tests

Use synthetic data only. Cover:

- `VisitStream` interaction filtering
- rolling-window boundaries over `ViewDate`
- binary user-ad interactions
- ad snapshot validation
- exact seed-time loading
- direct ad-CF graph edge roles
- three-hop fanout schedule
- coverage metrics for partial coverage

## 7. Acceptance Criteria

- Rel-HM and Rel-Amazon tests remain compatible.
- `rel-avito/user-ad-visit` is accepted by config validation.
- Snapshots use `src_AdID` / `dst_AdID`.
- No `ad_cf` graph node type is attached.
- The model receives no CF node features or CF score features.
- `num_layers < 3` fails for CF coverage/graph fanouts.
- Targeted tests pass.
