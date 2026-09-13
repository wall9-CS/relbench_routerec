# Specification: Seed-Time-Specific CF Table for Rel-Amazon Recommendation

## 0. Scope

- **Dataset:** `rel-amazon`
- **Recommendation tasks:**
  - `user-item-purchase`
  - `user-item-rate`
  - `user-item-review`
- **Source entity:** `customer`
- **Destination entity:** `product`
- **Interaction table:** `review(customer_id, product_id, review_time, rating, review_text, ...)`
- **Goal:** augment each seed-time-specific ID-GNN graph with a temporal
  `product_cf` fact table so product candidates can be reached through a
  four-hop route.

The Rel-HM implementation remains unchanged. Amazon code should live under
`examples/amazon_cf` and use Amazon names (`product_cf`, `src_product_id`,
`dst_product_id`) rather than reusing H&M `article_cf` names.

## 1. Candidate Route

For each seed time, attach exactly one `product_cf` snapshot:

```text
customer
  <- review
  <- product (historically reviewed source product)
  <- product_cf
  <- product (CF destination candidate)
```

Under the existing foreign-key edge convention:

```text
customer
  <-[review.f2p_customer_id]- review
  <-[product.rev_f2p_product_id]- product(src)
  <-[product_cf.f2p_src_product_id]- product_cf
  <-[product.rev_f2p_dst_product_id]- product(dst)
```

Use `num_layers >= 4`. Default `num_layers` for CF experiments is `4`.

## 2. Task-Specific Interaction Semantics

Build the binary customer-product matrix from the same historical event
definition as the task label:

| Task | Historical interaction rows used for CF |
|---|---|
| `user-item-purchase` | all reviews with non-null `customer_id` and `product_id` |
| `user-item-rate` | reviews where `rating == 5.0` |
| `user-item-review` | reviews where `review_text` is non-null and `len(review_text) > 300` |

For seed time `t`, the default history window is:

```text
t - 365//4 days < review_time <= t
```

This matches the Amazon recommendation task horizon length. Also support
`--all-history`, meaning `review_time <= seed_time`.

The source graph still uses the base RelBench `review` table. The snapshot
itself is task-filtered so the CF table reflects the selected recommendation
objective.

## 3. Sparse CF Construction

For every seed time:

1. Sort filtered interactions by `review_time` once.
2. Slice the rolling window with `numpy.searchsorted`.
3. Deduplicate `(customer_id, product_id)`.
4. Factorize active customers in the slice.
5. Build sparse CSR matrix `M` with shape
   `(num_active_customers, num_products)`.
6. Compute sparse item co-occurrence:

```python
C = (M.T @ M).tocsr()
```

7. Remove the diagonal.
8. Keep rows with `support >= min_support`.
9. Score each ordered pair:

```text
score(i -> j) = C[i, j] / (count_i ** alpha * count_j ** (1 - alpha))
```

10. Keep top `top_l` per source product, ordered by:
    1. `cf_score` descending
    2. `support` descending
    3. `dst_product_id` ascending

Defaults:

```text
history_days = 365//4
min_support = 3
top_l = 32
alpha = 0.5
```

## 4. Snapshot Format

Directory:

```text
<output-root>/
  rel-amazon/
    <task>/
      window_91d_alpha_0.5_support_3_top32/
        manifest.json
        cf_snapshot_2015-10-01.parquet
        cf_snapshot_2016-01-01.parquet
```

Parquet schema:

| Column | Meaning |
|---|---|
| `seed_time` | exact task seed time |
| `src_product_id` | source product index |
| `dst_product_id` | CF destination product index |
| `support` | co-occurrence count |
| `cf_score` | normalized CF score |
| `rank` | one-based rank within source product |

The graph-facing `product_cf` TensorFrame must expose only:

```text
__const__ = 1.0
```

`support`, `cf_score`, and `rank` are stored only for validation and analysis.

## 5. Manifest

`manifest.json` must include:

```json
{
  "dataset": "rel-amazon",
  "task": "user-item-purchase",
  "history_days": 91,
  "all_history": false,
  "window_boundary": "(seed_time - history_days, seed_time]",
  "binary_interactions": true,
  "customer_scope": "all_database_customers",
  "interaction_table": "review",
  "src_entity_table": "customer",
  "dst_entity_table": "product",
  "cf_node_type": "product_cf",
  "min_support": 3,
  "top_l": 32,
  "alpha": 0.5,
  "score_used_as_model_input": false
}
```

Manifest compatibility must include dataset, task, history mode, support,
top-L, and alpha.

## 6. CLIs

Build snapshots:

```bash
python -m examples.build_amazon_cf_snapshots \
  --dataset rel-amazon \
  --task user-item-purchase \
  --history-days 91 \
  --min-support 3 \
  --top-l 32 \
  --alpha 0.5 \
  --output-root /data/seonghun/cf_snapshots
```

Coverage diagnostics:

```bash
python -m examples.evaluate_amazon_cf_coverage \
  --dataset rel-amazon \
  --task user-item-purchase \
  --cf-snapshot-dir /data/seonghun/cf_snapshots/rel-amazon/user-item-purchase/window_91d_alpha_0.5_support_3_top32 \
  --splits val,test \
  --num-layers 4
```

The same commands must work for:

```text
user-item-rate
user-item-review
```

## 7. Tests

Use synthetic data only. Cover:

- task-specific Amazon review filters
- rolling-window boundaries over `review_time`
- binary interactions
- known sparse co-occurrence support
- product snapshot validation
- exact seed-time loading
- product CF graph edge roles
- four-hop fanout schedule
- coverage metrics for partial and zero coverage

## 8. Acceptance Criteria

- Rel-HM code and behavior remain compatible.
- All three Rel-Amazon recommendation tasks are accepted by config validation.
- Snapshots use `src_product_id` / `dst_product_id`.
- The graph node type is `product_cf`.
- The model receives only a constant feature for `product_cf`.
- `num_layers < 4` fails for CF coverage/graph fanouts.
- Targeted tests pass.
