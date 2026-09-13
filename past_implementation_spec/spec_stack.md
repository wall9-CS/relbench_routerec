# Specification: Seed-Time-Specific CF Table for Rel-Stack Recommendation

## 0. Scope

- **Dataset:** `rel-stack`
- **Recommendation task:** `user-post-comment`
- **Source entity:** `users`
- **Destination entity:** `posts`
- **Interaction table:** `comments(UserId, PostId, CreationDate, ...)`
- **Goal:** augment each seed-time-specific ID-GNN graph with a `post_cf`
  fact table so candidate posts can be reached through a four-hop route.

Stack-specific code lives under `examples/stack_cf` and uses Stack names
(`post_cf`, `src_PostId`, `dst_PostId`).

## 1. Candidate Route

For each seed time, attach exactly one `post_cf` snapshot:

```text
users
  <- comments
  <- posts (historically commented source post)
  <- post_cf
  <- posts (CF destination candidate)
```

Under RelBench foreign-key edge naming:

```text
users
  <-[comments.f2p_UserId]- comments
  <-[posts.rev_f2p_PostId]- posts(src)
  <-[post_cf.f2p_src_PostId]- post_cf
  <-[posts.rev_f2p_dst_PostId]- posts(dst)
```

Use `num_layers >= 4`. The default CF experiment uses `num_layers=4`.

## 2. Interaction Semantics

Use historical comment interactions from `comments`:

```text
UserId, PostId, CreationDate
```

Rows with null `UserId` or `PostId` are excluded. For seed time `t`, the default
rolling history window is:

```text
t - 91 days < CreationDate <= t
```

This matches `UserPostCommentTask.timedelta = 365 // 4`. Also support
`--all-history`, meaning `CreationDate <= seed_time`.

## 3. Sparse CF Construction

For every seed time:

1. Sort interactions by `CreationDate` once.
2. Slice the rolling window with `numpy.searchsorted`.
3. Deduplicate `(UserId, PostId)`.
4. Factorize active users in the slice.
5. Build sparse CSR matrix `M` with shape `(num_active_users, num_posts)`.
6. Compute sparse post co-occurrence: `C = (M.T @ M).tocsr()`.
7. Remove the diagonal.
8. Keep pairs with `support >= min_support`.
9. Score each ordered pair:

```text
score(i -> j) = C[i, j] / (count_i ** alpha * count_j ** (1 - alpha))
```

10. Keep top `top_l` per source post, ordered by `cf_score` descending,
    `support` descending, then `dst_PostId` ascending.

Defaults:

```text
history_days = 91
min_support = 3
top_l = 32
alpha = 0.5
```

## 4. Snapshot Format

Directory:

```text
<output-root>/
  rel-stack/
    user-post-comment/
      window_91d_alpha_0.5_support_3_top32/
        manifest.json
        cf_snapshot_2020-10-01.parquet
        cf_snapshot_2021-01-01.parquet
```

Parquet schema:

| Column | Meaning |
|---|---|
| `seed_time` | exact task seed time |
| `src_PostId` | source post index |
| `dst_PostId` | CF destination post index |
| `support` | co-occurrence count |
| `cf_score` | normalized CF score |
| `rank` | one-based rank within source post |

The graph-facing `post_cf` TensorFrame exposes only `__const__ = 1.0`.

## 5. CLIs

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

Coverage diagnostics:

```bash
python -m examples.evaluate_stack_cf_coverage \
  --dataset rel-stack \
  --task user-post-comment \
  --cf-snapshot-dir /data/seonghun/cf_snapshots/rel-stack/user-post-comment/window_91d_alpha_0.5_support_3_top32 \
  --splits val,test \
  --num-layers 4
```

Train CF-augmented ID-GNN:

```bash
python -m examples.idgnn_recommendation_stack_cf \
  --dataset rel-stack \
  --task user-post-comment \
  --cf-snapshot-dir /data/seonghun/cf_snapshots/rel-stack/user-post-comment/window_91d_alpha_0.5_support_3_top32 \
  --num_layers 4 \
  --num_neighbors 128
```

## 6. Tests

Use synthetic data only. Cover:

- comment interaction filtering
- rolling-window boundaries over `CreationDate`
- binary user-post interactions
- post snapshot validation
- exact seed-time loading
- `post_cf` graph edge roles
- four-hop fanout schedule
- coverage metrics for partial coverage

## 7. Acceptance Criteria

- Existing Rel-HM, Rel-Amazon, and Rel-Avito tests remain compatible.
- `rel-stack/user-post-comment` is accepted by config validation.
- Snapshots use `src_PostId` / `dst_PostId`.
- The graph node type is `post_cf`.
- The model receives only `__const__` for `post_cf`.
- `num_layers < 4` fails for CF coverage/graph fanouts.
- Targeted tests pass.

