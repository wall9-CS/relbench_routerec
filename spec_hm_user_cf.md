# Specification: Seed-Time-Specific User-CF Table for Rel-HM ID-GNN Recommendation

## 0. Scope

- **Dataset:** `rel-hm`
- **Task:** `user-item-purchase`
- **Current reference implementation:** `examples/idgnn_recommendation_cf.py`
- **Current CF pattern:** seed-time-specific item-item CF through `article_cf`
- **New target pattern:** seed-time-specific user-user CF through `user_cf`
- **Goal:** explicitly connect each source `customer` to top-k similar customers
  computed from historical purchase overlap, so ID-GNN can use similar users'
  representations and historical articles when scoring `article` candidates.

This should be implemented as a separate experimental pipeline first, not as an
in-place replacement of the existing item-CF script. Keep
`examples/idgnn_recommendation_cf.py` behavior unchanged for clean ablations.

Recommended new entry point:

```bash
python -m examples.idgnn_recommendation_user_cf
```

Recommended helper package:

```text
examples/hm_user_cf/
```

## 1. Target graph route

For each task seed time `t`, attach exactly one `user_cf` snapshot.

The intended expansion route from an input source customer is:

```text
customer(src)
  <- user_cf row
  <- customer(similar user)
  <- transactions
  <- article(candidate/history item)
```

Under the existing RelBench foreign-key edge convention and incoming-neighbor
sampling, use:

```text
customer(src)
  <-[user_cf.f2p_src_customer_id]- user_cf
  <-[customer.rev_f2p_dst_customer_id]- customer(similar)
  <-[transactions.f2p_customer_id]- transactions
  <-[article.rev_f2p_article_id]- article
```

With `num_layers=4`, this allows a source customer to sample:

1. its `user_cf` pair rows,
2. the top-k similar customers attached to those rows,
3. those customers' historical transactions,
4. articles from those transactions.

Keep `subgraph_type="bidirectional"` for message passing after sampling, as in
the current ID-GNN scripts. Directionality should be controlled by the
edge-type-specific fanout schedule.

## 2. Methodological decisions

### 2.1 History window

Use the same temporal boundary convention as the current H&M item-CF pipeline.

For seed time `t` and default rolling window `W=4` weeks:

```text
t - W < transactions.t_dat <= t
```

Also support:

```text
--all-history
```

meaning:

```text
transactions.t_dat <= t
```

`--window-weeks` and `--all-history` must be mutually exclusive.

### 2.2 Source user scope

Do not build a dense all-user-to-all-user graph. For each seed time, compute
neighbors only for source customers that appear in the task rows for that seed
time and requested splits.

Manifest wording:

```text
source_customer_scope = task_source_customers_at_seed_time
candidate_customer_scope = all_database_customers_with_history
```

Rationale: Rel-HM has many customers; full `M @ M.T` user-user co-occurrence can
explode around popular articles. The model only needs explicit `user_cf` rows
for customers that can be input roots.

### 2.3 Binary user-item history

Within each seed-time window, deduplicate repeated purchases:

```text
M_t[u, i] = 1 if customer u purchased article i at least once in the window
```

Repeated purchases of the same article by the same user contribute one overlap.

### 2.4 Similarity definition

Let:

```text
overlap(u, v) = |H_t(u) intersection H_t(v)|
count_u = |H_t(u)|
count_v = |H_t(v)|
```

Default score:

```text
score(u -> v) = overlap(u, v) / (count_u ** alpha * count_v ** (1 - alpha))
```

Defaults:

```text
alpha = 0.5
min_overlap = 2
top_k = 32
```

`alpha=0.5` is cosine normalization over binary histories. Keeping `alpha`
configurable mirrors the existing item-CF pipeline and supports directional
scores when `alpha != 0.5`.

### 2.5 Pair semantics

Store ordered pairs:

```text
src_customer_id
dst_customer_id
```

Rules:

- no self-pairs
- no duplicate ordered pairs
- keep at most `top_k` destination customers per source customer
- do not require reciprocal pairs
- do not infer `dst -> src` from `src -> dst`

Deterministic ordering per `src_customer_id`:

1. `user_cf_score` descending
2. `overlap` descending
3. `dst_customer_id` ascending

Assign one-based `rank`.

### 2.6 Model input

Persist overlap diagnostics for validation and analysis, but do not feed them as
model features initially.

The graph-facing `user_cf` TensorFrame should expose only:

```text
__const__ = 1.0
```

Do not expose these columns to `Model`:

```text
overlap
user_cf_score
rank
src_history_size
dst_history_size
```

This keeps the experiment aligned with the existing `article_cf` design: topology
changes, feature surface stays minimal.

## 3. Snapshot construction algorithm

### 3.1 Inputs

Use the reindexed RelBench database:

```python
dataset = get_dataset("rel-hm", download=True)
task = get_task("rel-hm", "user-item-purchase", download=True)
db = dataset.get_db()
transactions = db.table_dict["transactions"].df
```

Required transaction columns:

```text
customer_id
article_id
t_dat
```

Validate:

- `customer_id` is integer-like and in range `[0, num_customers)`
- `article_id` is integer-like and in range `[0, num_articles)`
- `t_dat` is datetime
- requested source customers are valid customer IDs

### 3.2 Seed-time discovery

Reuse the pattern in `examples/hm_cf/io.py`:

```text
discover_seed_times(task, splits)
parse_seed_times(...)
```

For each seed time, also collect source customers from task rows at that exact
time. Snapshot creation should be able to run for:

```text
train,val,test
```

by default, while a debugging run can target one seed time:

```bash
--seed-time 2020-03-02
```

### 3.3 Sparse exact top-k computation

Sort transactions by `t_dat` once. For each seed time:

```python
left = searchsorted(times, seed_time - window, side="right")
right = searchsorted(times, seed_time, side="right")
window = transactions.iloc[left:right][["customer_id", "article_id"]]
pairs = window.drop_duplicates()
```

Build a sparse binary matrix:

```text
M shape = (num_active_customers, num_articles)
```

Use compact active-customer row IDs, but preserve mapping to global
`customer_id`.

Do not compute full:

```python
M @ M.T
```

unless the active source set is tiny. Instead compute source rows in chunks:

```python
overlap = M[src_rows_chunk] @ M.T
```

For each source row in the chunk:

1. read sparse nonzero candidate users,
2. remove self,
3. apply `overlap >= min_overlap`,
4. compute normalized score,
5. keep top `top_k` by deterministic ordering,
6. append compact arrays.

This is exact for the requested source customers and avoids materializing global
all-user similarities.

Recommended CLI argument:

```text
--source-chunk-size 4096
```

### 3.4 Empty and cold-start cases

If a source customer has no history in the snapshot window, write no rows for
that source customer.

If no source customer has valid neighbors, write an empty parquet with the normal
schema.

The training graph must still contain the `user_cf` node type and four edge types
through a schema template, even for empty snapshots.

## 4. Snapshot format

Directory:

```text
<output-root>/
  rel-hm/
    user-item-purchase/
      user_cf_window_8w_alpha_0.5_overlap_2_top32/
        manifest.json
        user_cf_snapshot_2020-03-02.parquet
```

All-history directory:

```text
user_cf_all_history_alpha_0.5_overlap_2_top32/
```

Parquet schema:

| Column | Required dtype | Meaning |
|---|---:|---|
| `seed_time` | timestamp | exact task seed time |
| `src_customer_id` | int64 | source customer/root user |
| `dst_customer_id` | int64 | similar customer |
| `overlap` | int64 | number of shared historical articles |
| `user_cf_score` | float32 | normalized overlap score |
| `src_history_size` | int64 | distinct historical article count for source |
| `dst_history_size` | int64 | distinct historical article count for neighbor |
| `rank` | int32 | one-based rank within source customer |

Do not persist:

```text
user_cf_node_id
__const__
```

At graph attachment time:

```text
user_cf node IDs = [0, num_snapshot_rows)
```

### 4.1 Manifest

`manifest.json` must include at least:

```json
{
  "dataset": "rel-hm",
  "task": "user-item-purchase",
  "cf_kind": "user_user_history_overlap",
  "window_weeks": 8,
  "all_history": false,
  "window_boundary": "(seed_time - window, seed_time]",
  "binary_interactions": true,
  "source_customer_scope": "task_source_customers_at_seed_time",
  "candidate_customer_scope": "all_database_customers_with_history",
  "min_overlap": 2,
  "top_k": 32,
  "alpha": 0.5,
  "score_used_as_model_input": false,
  "snapshot_files": {}
}
```

Manifest compatibility must reject mismatched dataset, task, history mode,
`min_overlap`, `top_k`, `alpha`, and `cf_kind`.

Writes must be atomic, following `examples/hm_cf/io.py`.

## 5. Graph representation

### 5.1 Node type

Add:

```python
USER_CF = "user_cf"
```

Each snapshot row becomes one `user_cf` node with:

```text
time = seed_time
TensorFrame feature = __const__
```

### 5.2 Edge types

Use names matching `make_pkey_fkey_graph` conventions:

```python
USER_CF_SRC_F2P = ("user_cf", "f2p_src_customer_id", "customer")
USER_CF_SRC_REV = ("customer", "rev_f2p_src_customer_id", "user_cf")
USER_CF_DST_F2P = ("user_cf", "f2p_dst_customer_id", "customer")
USER_CF_DST_REV = ("customer", "rev_f2p_dst_customer_id", "user_cf")
```

Construct all four edge stores:

```text
user_cf -> src customer
src customer -> user_cf
user_cf -> dst customer
dst customer -> user_cf
```

Use `torch.long` indices and `sort_edge_index`.

Validate that all customer IDs are in range.

### 5.3 Snapshot isolation

For samples at seed time `t`, the graph must contain only:

```text
user_cf_snapshot_<t>.parquet
```

Do not concatenate snapshots. Do not rely on temporal filtering over older
`user_cf` rows, because stale neighbor rows would change source-user topology.

### 5.4 Base graph reuse

Follow the existing `examples/hm_cf/graph.py` ownership pattern:

1. build the base RelBench graph once with `make_pkey_fkey_graph`,
2. shallow-copy the `HeteroData` container per seed time,
3. attach/replace only `user_cf` node and edge stores,
4. do not mutate `base_data`,
5. delete loader and attached graph after each group.

Provide:

```python
attach_user_cf_snapshot(...)
build_user_cf_schema_template(...)
user_cf_col_stats()
```

Do not add `user_cf` to `shallow_list`.

## 6. Fanout schedule

Build an edge-type-specific `num_neighbors` dictionary.

Keep normal fanouts for base graph edges:

```python
hop_fanouts = [num_neighbors // 2**i for i in range(num_layers)]
```

Require:

```text
num_layers >= 4
```

Set all `user_cf` edge fanouts to zero, then enable only:

```text
USER_CF_SRC_F2P at hop index 0
USER_CF_DST_REV at hop index 1
```

For `num_layers=4` and `num_neighbors=128`:

```python
USER_CF_SRC_F2P: [128, 0, 0, 0]
USER_CF_DST_REV: [0, 64, 0, 0]
USER_CF_SRC_REV: [0, 0, 0, 0]
USER_CF_DST_F2P: [0, 0, 0, 0]
```

This preserves the directed expansion:

```text
src customer -> its user_cf rows -> selected dst customers
```

and avoids accidental reverse expansion:

```text
customer who appears as somebody else's dst -> that somebody's src
```

Base edge fanouts at hops 2 and 3 then allow:

```text
dst customer -> transactions -> article
```

## 7. Training script design

Create:

```text
examples/idgnn_recommendation_user_cf.py
```

Use `examples/idgnn_recommendation_cf.py` as the closest template, replacing
item-CF imports and graph attachment with user-CF helpers.

CLI additions:

```text
--user-cf-snapshot-dir
--report-user-cf-coverage
```

Initial defaults should mirror the current script:

```text
num_layers = 4
num_neighbors = 128
temporal_strategy = last
num_workers = 0
```

Training loop:

1. group task rows by seed time with existing `SeedTimeGroup` logic,
2. load exactly one `user_cf` snapshot for the group,
3. attach it to the base graph,
4. build `NeighborLoader` with user-CF fanouts,
5. keep the existing BCE loss and `forward_dst_readout`,
6. scatter group predictions back to original row order.

The model remains unchanged.

## 8. Preprocessing CLI

Create:

```text
examples/build_hm_user_cf_snapshots.py
```

Example:

```bash
python -m examples.build_hm_user_cf_snapshots \
  --dataset rel-hm \
  --task user-item-purchase \
  --window-weeks 8 \
  --min-overlap 2 \
  --top-k 32 \
  --alpha 0.5 \
  --source-chunk-size 4096 \
  --output-root /data/seonghun/cf_snapshots
```

Arguments:

```text
--dataset
--task
--window-weeks
--all-history
--min-overlap
--top-k
--alpha
--source-chunk-size
--output-root
--splits
--seed-time
--overwrite
--download
```

Behavior:

- skip valid existing snapshots unless `--overwrite`
- fail clearly for incompatible manifest
- update manifest after each snapshot so interrupted runs resume
- include per-snapshot stats in the manifest
- print final snapshot directory

## 9. Coverage diagnostics

Add an optional coverage module only after the graph path is implemented.

For user-CF, deterministic coverage means:

```text
candidate articles = historical articles of top-k similar users up to seed_time
```

For each task row:

1. load the seed-time `user_cf` snapshot,
2. get `dst_customer_id` for the row's `src_customer_id`,
3. collect those users' historical articles from transactions with
   `t_dat <= seed_time`,
4. compare against ground-truth future `article_id` labels.

Report the same style of metrics as `examples/hm_cf/coverage.py`:

```text
coverage_rate
hit_rate
achievable_MAP
coverage_gap
```

This diagnostic is an oracle upper bound over the deterministic user-CF candidate
set, not a model metric.

## 10. Files to add

Recommended additions:

```text
examples/hm_user_cf/__init__.py
examples/hm_user_cf/config.py
examples/hm_user_cf/snapshot.py
examples/hm_user_cf/io.py
examples/hm_user_cf/graph.py
examples/hm_user_cf/coverage.py
examples/build_hm_user_cf_snapshots.py
examples/evaluate_hm_user_cf_coverage.py
examples/idgnn_recommendation_user_cf.py
test/examples/test_hm_user_cf_snapshot.py
test/examples/test_hm_user_cf_io.py
test/examples/test_hm_user_cf_graph.py
test/examples/test_hm_user_cf_coverage.py
```

Reuse `examples/hm_cf/seed_time_loader.py` directly unless the user-CF pipeline
needs source-customer collection helpers that do not belong there.

## 11. Tests

Tests must use synthetic data only.

Required coverage:

- window boundary: lower exclusive, upper inclusive
- binary user-item history: duplicate purchases count once
- overlap support: exact known overlaps
- `min_overlap` filtering
- score formula for `alpha=0.5` and `alpha=1.0`
- top-k tie-break and consecutive ranks
- source scope: only requested source customers get rows
- no self-pairs
- parquet and manifest round trip
- graph edge roles and edge indices
- base graph immutability
- schema template supports empty snapshots
- fanout route reaches similar user's historical article in four hops
- reverse route is not sampled by user-CF fanouts
- smoke model forward/backward with `user_cf`

Run at minimum:

```bash
python -m pytest \
  test/examples/test_hm_user_cf_snapshot.py \
  test/examples/test_hm_user_cf_io.py \
  test/examples/test_hm_user_cf_graph.py \
  test/examples/test_hm_user_cf_coverage.py
```

Also run:

```bash
python -m pytest test/examples/test_hm_cf_seed_time_loader.py
python -m pytest test/modeling/test_link_idgnn.py
```

## 12. Acceptance criteria

- [ ] Existing item-CF script still works unchanged.
- [ ] User-CF snapshots are exact per seed time.
- [ ] Snapshot construction uses only history up to `seed_time`.
- [ ] Source customers are limited to task source rows at the seed time.
- [ ] Candidate users may come from all active historical customers.
- [ ] No full dense user-user matrix is built.
- [ ] Similarity rows are ordered, top-k, deterministic, and self-pair free.
- [ ] `user_cf` exposes only `__const__` to the model.
- [ ] Fanouts allow `src customer -> user_cf -> similar customer`.
- [ ] Base graph fanouts then allow `similar customer -> transaction -> article`.
- [ ] Reverse user-CF expansion is disabled.
- [ ] Predictions preserve original task row order.
- [ ] Targeted unit tests pass.

## 13. Out of scope for the first implementation

Do not implement initially:

- combining `article_cf` and `user_cf` in one script
- score-aware node features
- score-weighted neighbor sampling
- direct customer-customer edges
- approximate nearest-neighbor search
- item popularity downsampling
- future-label-aware neighbor construction
- changes to `examples/model.py`
- changes to RelBench task or dataset definitions

Once user-CF alone is verified, a second spec can define a combined
`article_cf + user_cf` ablation by composing the two graph attachment helpers and
merging their fanout schedules.
