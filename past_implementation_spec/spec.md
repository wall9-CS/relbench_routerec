# Specification: Seed-Time-Specific Collaborative-Filtering Table for RelBench ID-GNN Recommendation

## 0. Document status

- **Target repository:** `stanford-star/relbench`
- **Initial supported dataset:** `rel-hm`
- **Initial supported task:** `user-item-purchase`
- **Primary baseline:** `examples/idgnn_recommendation.py`
- **Implementation style:** add a separate experimental pipeline; preserve the behavior of the existing baseline.
- **Main objective:** augment each seed-time-specific RDL graph with a temporal `article_cf` fact table so that ID-GNN can reach collaborative-filtering candidates through a four-hop path.

The implementation must follow the current checkout rather than relying blindly on line numbers in this specification.

---

## 1. Motivation and target behavior

The current RelBench ID-GNN recommendation example samples a temporal source-centered subgraph and scores destination `article` nodes that appear in that subgraph. Its performance is therefore upper-bounded by candidate coverage.

For `rel-hm/user-item-purchase`, add a seed-time-specific collaborative-filtering fact table:

```text
customer
  <- transactions
  <- article (historically purchased source item)
  <- article_cf
  <- article (CF destination candidate)
```

Equivalently, under the RelBench foreign-key edge convention and incoming-neighbor sampling, the intended four-hop candidate expansion is:

```text
customer
  <-[transactions.f2p_customer_id]- transactions
  <-[article.rev_f2p_article_id]- article(src)
  <-[article_cf.f2p_src_article_id]- article_cf
  <-[article.rev_f2p_dst_article_id]- article(dst)
```

The `article_cf` table must be computed independently for every task seed time. Only the exact snapshot for a seed time may be present in the graph used for samples at that seed time.

---

## 2. Existing repository integration points

Before coding, inspect the current versions of these files:

- `examples/idgnn_recommendation.py`
  - Baseline argument parsing, database/task loading, `make_pkey_fkey_graph`, temporal `NeighborLoader`, ID-GNN loss, validation, and test evaluation.
- `examples/model.py`
  - `Model`, `HeteroEncoder`, `HeteroTemporalEncoder`, `HeteroGraphSAGE`, and `forward_dst_readout`.
- `relbench/modeling/graph.py`
  - `make_pkey_fkey_graph`, `get_link_train_table_input`, foreign-key edge naming, reverse-edge naming, time tensors, and constant features for featureless tables.
- `relbench/modeling/loader.py`
  - `SparseTensor`, current temporal loading helpers, and timestamp grouping utilities.
- `relbench/datasets/hm.py`
  - Table names and foreign keys:
    - `customer`
    - `article`
    - `transactions(customer_id, article_id, t_dat, ...)`
- `relbench/tasks/hm.py`
  - `UserItemPurchaseTask`
  - prediction interval: `(seed_time, seed_time + 7 days]`
  - source table: `customer`
  - destination table: `article`
  - task time column: `timestamp`
- `test/modeling/test_link_idgnn.py`
  - Existing test conventions for ID-GNN, temporal `NeighborLoader`, and fake datasets.

Do not change the semantic definitions in `relbench/datasets/hm.py` or `relbench/tasks/hm.py`.

---

## 3. Fixed methodological decisions

### 3.1 Temporal interaction window

For seed time \(t\), with default rolling window \(W=8\) weeks, use transactions satisfying:

\[
t-W < \mathrm{t\_dat} \le t.
\]

The upper boundary is inclusive because the prediction interval is:

\[
(t,\;t+7\text{ days}].
\]

The lower boundary is exclusive.

Default:

```text
window_weeks = 8
```

Also support all-history mode as a future ablation:

```text
transaction_time <= seed_time
```

`--window-weeks` and `--all-history` must be mutually exclusive.

### 3.2 User scope

Build CF from all database customers represented in the eligible transaction window, not only customers appearing in the task table at that seed time.

### 3.3 Binary user-item matrix

For each seed time \(t\), deduplicate repeated customer-article purchases within the window:

\[
M_t[u,i]
=
\mathbb{1}\left[
u \text{ purchased } i \text{ at least once in } (t-W,t]
\right].
\]

Each `(customer_id, article_id)` pair contributes exactly one nonzero entry.

### 3.4 Sparse item-item co-occurrence

Compute:

\[
C_t = M_t^\top M_t.
\]

Interpretation:

- \(C_t[i,j]\): number of distinct customers that purchased both `i` and `j` in the window.
- \(c_i = C_t[i,i]\): number of distinct customers that purchased `i`.

Requirements:

- Use a sparse matrix implementation.
- Use `scipy.sparse`.
- Never instantiate dense `#items × #items` or `#users × #items` arrays.
- Use integer support counts with a dtype that cannot overflow at Rel-HM scale; use `int32` or wider for multiplication and support.

### 3.5 CF score

For every non-diagonal pair that passes support filtering:

\[
s_{i\rightarrow j}^{(t)}
=
\frac{C_t[i,j]}
{c_i^\alpha c_j^{1-\alpha}}.
\]

Default:

```text
alpha = 0.5
```

Expose `alpha` as an argument and validate:

```text
0.0 <= alpha <= 1.0
```

### 3.6 Filtering and top-L

Defaults:

```text
min_support = 3
top_l = 32
```

Keep only pairs satisfying:

\[
C_t[i,j] \ge \texttt{min\_support}.
\]

Then retain at most `top_l` destination articles for each source article.

Deterministic ordering for each source article:

1. `cf_score` descending
2. `support` descending
3. `dst_article_id` ascending

Assign rank starting from 1.

Both `min_support` and `top_l` must be configurable CLI arguments.

### 3.7 Ordered pairs

Store:

```text
src_article_id
dst_article_id
```

Do not impose:

```text
src_article_id < dst_article_id
```

Reasons:

- `alpha != 0.5` makes the score directional.
- Row-wise top-L pruning can be directional even when `alpha == 0.5`.
- `A -> B` and `B -> A` may both exist as separate rows.

### 3.8 Model input

Store `support`, `cf_score`, and `rank` for preprocessing, validation, and analysis, but do not feed them to the model.

The graph-facing `article_cf` node input must contain only a constant feature:

```text
__const__ = 1.0
```

The model must not receive:

- `cf_score`
- `support`
- `rank`

The `article_cf` node timestamp is the snapshot seed time.

### 3.9 Sampling

Keep random/uniform temporal neighbor sampling.

Do not implement:

- score-weighted sampling
- score-biased sampling
- deterministic CF-rank sampling
- additional negative sampling logic
- loss reweighting for additional negatives
- explicit total candidate-budget balancing

Candidate count is controlled by the existing neighbor-sampling fanout and `top_l`.

---

## 4. Output snapshot format

### 4.1 Directory layout

Use a configuration-specific directory:

```text
<output-root>/
└── rel-hm/
    └── user-item-purchase/
        └── window_8w_alpha_0.5_support_3_top32/
            ├── manifest.json
            ├── cf_snapshot_2019-09-09.parquet
            ├── cf_snapshot_2019-09-16.parquet
            └── ...
```

For all-history:

```text
all_history_alpha_0.5_support_3_top32/
```

### 4.2 File naming

For midnight timestamps use:

```text
cf_snapshot_YYYY-MM-DD.parquet
```

Provide one canonical timestamp-to-filename function and use it in both preprocessing and training.

If a future task contains non-midnight seed times, the function must avoid collisions by including the time component.

### 4.3 Parquet schema

Each snapshot parquet must contain:

| Column | Required dtype | Meaning |
|---|---:|---|
| `seed_time` | timestamp | exact snapshot/task seed time |
| `src_article_id` | int32 or int64 | source article index in the reindexed RelBench database |
| `dst_article_id` | int32 or int64 | destination article index |
| `support` | int32 or int64 | \(C_t[i,j]\) |
| `cf_score` | float32 | normalized CF score |
| `rank` | int16 or int32 | one-based rank within source article |

Do not persist:

- a dummy feature column
- a global `cf_node_id`

At runtime, define:

```text
cf_node_id = 0, 1, ..., number_of_rows - 1
__const__ = 1.0
```

Parquet writes must be atomic:

1. write to a temporary file in the same directory
2. validate the temporary file
3. rename to the final filename

### 4.4 Manifest

`manifest.json` must include at least:

```json
{
  "dataset": "rel-hm",
  "task": "user-item-purchase",
  "window_weeks": 8,
  "all_history": false,
  "window_boundary": "(seed_time - window, seed_time]",
  "binary_interactions": true,
  "customer_scope": "all_database_customers",
  "min_support": 3,
  "top_l": 32,
  "alpha": 0.5,
  "score_used_as_model_input": false,
  "snapshot_files": {
    "2019-09-09T00:00:00": {
      "file": "cf_snapshot_2019-09-09.parquet",
      "num_rows": 0,
      "num_active_customers": 0,
      "num_active_articles": 0,
      "num_unique_user_item_pairs": 0,
      "cooccurrence_nnz_before_support": 0,
      "cooccurrence_nnz_after_support": 0,
      "elapsed_seconds": 0.0
    }
  }
}
```

The exact additional diagnostics may be extended.

Training must validate that the manifest matches the selected dataset and task.

---

## 5. Seed-time discovery

Do not hardcode the weekly timestamp list.

For all requested splits, obtain task tables and take the sorted union of unique task timestamps:

```python
task.get_table("train")
task.get_table("val")
task.get_table("test")
```

Default preprocessing splits:

```text
train,val,test
```

Expose a repeatable or comma-separated seed-time filter for debugging, for example:

```bash
--seed-time 2020-03-02
```

This enables a one-snapshot pilot without building the full collection.

---

## 6. Efficient snapshot-construction algorithm

### 6.1 Input preparation

Use the reindexed database returned by RelBench:

```python
dataset = get_dataset("rel-hm", download=True)
db = dataset.get_db()
transactions = db.table_dict["transactions"].df
```

Validate:

- `customer_id` is integer-like and within `[0, num_customers)`
- `article_id` is integer-like and within `[0, num_articles)`
- `t_dat` is datetime
- transactions are available through the maximum required seed time

Sort transactions by `t_dat` once.

Use `numpy.searchsorted` over the sorted timestamp array to obtain each rolling slice. Do not scan all 15M rows with a full boolean mask for every snapshot.

Boundary implementation:

```python
left = np.searchsorted(times, seed_time - window, side="right")
right = np.searchsorted(times, seed_time, side="right")
window_df = transactions.iloc[left:right]
```

For all-history:

```python
left = 0
right = np.searchsorted(times, seed_time, side="right")
```

### 6.2 Deduplication

Select only:

```text
customer_id
article_id
```

and drop duplicate pairs.

Avoid carrying unused transaction columns through sparse-matrix construction.

### 6.3 User indexing

Factorize only active customer IDs within the current window to obtain compact row indices.

Use the global reindexed article IDs directly as sparse-matrix column indices.

Recommended shape:

```text
(num_active_customers, num_total_articles)
```

### 6.4 Sparse multiplication

Construct `M_t` as CSR and compute:

```python
C_t = (M_t.T @ M_t).tocsr()
```

Then:

```python
item_counts = C_t.diagonal()
C_t.setdiag(0)
C_t.eliminate_zeros()
```

### 6.5 Memory-safe row-wise filtering

Do not convert the entire `C_t` to a dense matrix.

Avoid constructing one giant pandas DataFrame from all unfiltered nonzeros when it causes unnecessary peak memory.

Iterate CSR rows:

1. read row destinations and support values
2. apply `support >= min_support`
3. compute scores vectorially for the row
4. select at most top-L with `argpartition` when needed
5. apply deterministic final sorting
6. append compact NumPy arrays
7. concatenate once at the end

A Python loop over item rows is allowed; a Python nested loop over all customers and all purchased-item pairs is not allowed.

### 6.6 Logging

For every snapshot log:

- seed time
- window boundaries
- raw transaction count
- unique user-item pair count
- active customer count
- active article count
- mean/max distinct items per active customer
- estimated pair contributions:
  \[
  \sum_u k_u(k_u-1)
  \]
- `C_t.nnz` before support filtering
- nonzeros after support filtering
- final CF row count
- elapsed time by major stage
- output file size
- process peak resident memory when available through the standard library

Do not add `psutil` solely for logging.

---

## 7. Graph representation

### 7.1 Node type

Add node type:

```text
article_cf
```

Each parquet row becomes one `article_cf` node.

For a snapshot with \(N\) rows:

```text
article_cf node IDs = [0, N)
```

Each node has:

```text
time = seed_time
TensorFrame feature: __const__ = 1.0
```

### 7.2 Edge types

Use names consistent with `make_pkey_fkey_graph`:

```python
CF_SRC_F2P = ("article_cf", "f2p_src_article_id", "article")
CF_SRC_REV = ("article", "rev_f2p_src_article_id", "article_cf")
CF_DST_F2P = ("article_cf", "f2p_dst_article_id", "article")
CF_DST_REV = ("article", "rev_f2p_dst_article_id", "article_cf")
```

Construct:

```text
article_cf -> src article
src article -> article_cf
article_cf -> dst article
dst article -> article_cf
```

Use `torch.long` edge indices and `sort_edge_index`.

Validate that source and destination article IDs are within the article-node range.

### 7.3 Exact snapshot isolation

For samples at seed time \(t\), the graph must contain only the CF rows from:

```text
cf_snapshot_<t>.parquet
```

Do not concatenate previous snapshots.

Do not rely on `cf_node.time <= seed_time` to select the correct version. Temporal loading would otherwise accumulate stale rows from older snapshots.

### 7.4 Base graph reuse

Build the original RelBench graph once with `make_pkey_fkey_graph`.

For each seed time:

1. shallow-copy the `HeteroData` container without cloning all base tensors
2. attach/replace only the `article_cf` node store and four CF edge stores
3. do not mutate the original base graph
4. delete references to the snapshot graph and loader after the group is processed

Provide a helper with clear ownership semantics, for example:

```python
attach_cf_snapshot(
    base_data: HeteroData,
    snapshot: pd.DataFrame,
    seed_time: pd.Timestamp,
) -> HeteroData
```

A unit test must prove that attaching a snapshot does not mutate the base graph.

### 7.5 Model schema template

`Model` is initialized once, but snapshot node counts vary.

Build a schema template that contains:

- all base node/edge types
- node type `article_cf`
- the four CF edge types
- a constant-feature TensorFrame for `article_cf`
- empty CF edge indices

The model schema must be identical to every actual snapshot graph.

Generate `col_stats_dict["article_cf"]` once from the constant feature and reuse it.

Do not add `article_cf` to `shallow_list`.

### 7.6 Empty snapshot

If a snapshot produces zero valid CF rows:

- preserve the `article_cf` node type and all four edge types
- use empty edge indices
- use a safe featureless-node representation supported by the installed PyTorch Frame version
- if a zero-row TensorFrame is not supported, use one isolated dummy node with no incident edges
- the dummy node must never become a candidate or affect reachability
- log that the snapshot has zero real CF rows

---

## 8. Direction-preserving sampling route

Although every CF row has two article foreign keys, candidate expansion must follow:

```text
historical src article -> article_cf -> dst article
```

It must not automatically treat `dst -> src` as the same candidate-generation route.

Build an edge-type-specific `num_neighbors` dictionary.

Let the standard hop fanouts be:

```python
hop_fanouts = [
    int(args.num_neighbors // 2**i)
    for i in range(args.num_layers)
]
```

For the initial four-layer setup, expected default fanouts are:

```text
[128, 64, 32, 16]
```

Requirements:

- require `num_layers >= 4` when CF augmentation is enabled
- retain the standard fanout list for original base edge types
- set all CF edge-type fanouts to zero by default
- enable only:
  - `CF_SRC_F2P` at hop index 2
  - `CF_DST_REV` at hop index 3
- leave `CF_DST_F2P` and `CF_SRC_REV` disabled for expansion
- keep `subgraph_type="bidirectional"` as in the baseline for message passing after sampling

Conceptually:

```python
num_neighbors[CF_SRC_F2P] = [0, 0, hop_fanouts[2], 0]
num_neighbors[CF_DST_REV] = [0, 0, 0, hop_fanouts[3]]
num_neighbors[CF_DST_F2P] = [0, 0, 0, 0]
num_neighbors[CF_SRC_REV] = [0, 0, 0, 0]
```

Generalize carefully if more than four layers are allowed; the intended CF expansion remains on hops 3 and 4.

A toy integration test must verify that:

- a customer seed reaches a historical source article
- the source article reaches its CF row
- the CF row reaches the destination article
- a row where the historical article appears only as `dst_article_id` does not create the reverse candidate route

---

## 9. Seed-time-specific loader design

### 9.1 Why separate loaders are required

One `NeighborLoader` uses one fixed graph topology. Since the CF topology changes by seed time, use one exact graph/loader at a time.

Do not load all parquet snapshots simultaneously.

### 9.2 Group task rows

For each split:

1. preserve original task-table row positions
2. group rows by exact `task.time_col`
3. construct a group-local `Table`
4. call `get_link_train_table_input` on the group-local table
5. build a `NeighborLoader` using the exact snapshot graph

Represent each group with a small dataclass, for example:

```python
@dataclass(frozen=True)
class SeedTimeGroup:
    seed_time: pd.Timestamp
    original_row_indices: np.ndarray
    table: Table
```

### 9.3 Loader settings

Use the same temporal semantics as the baseline:

```python
NeighborLoader(
    snapshot_graph,
    num_neighbors=edge_type_num_neighbors,
    time_attr="time",
    input_nodes=table_input.src_nodes,
    input_time=table_input.src_time,
    subgraph_type="bidirectional",
    batch_size=args.batch_size,
    temporal_strategy=args.temporal_strategy,
    shuffle=is_train,
    num_workers=args.num_workers,
    persistent_workers=args.num_workers > 0,
)
```

All rows in one loader have the same `input_time`.

### 9.4 Train loop

Within each epoch:

1. randomly permute train seed-time groups
2. for each seed-time group:
   - load exactly one parquet
   - attach it to the base graph
   - construct the loader
   - construct the group-local target `SparseTensor`
   - train over batches
   - release loader and snapshot graph
3. enforce `max_steps_per_epoch` globally across all groups, not separately for every group

Keep the baseline loss unchanged:

```python
binary_cross_entropy_with_logits(out, target)
```

Do not add negative handling.

### 9.5 Validation and test order

Predictions must be returned in the original task-table row order.

For each split:

```python
pred = np.empty(
    (len(task_table.df), task.eval_k),
    dtype=np.int64,
)
```

After evaluating a group:

```python
pred[group.original_row_indices] = group_predictions
```

Do not assume the task table is already sorted by timestamp.

Use:

```python
task.evaluate(pred, task.get_table("val"))
```

for validation and preserve the baseline test evaluation behavior.

### 9.6 Loader lifecycle and memory

Default behavior:

- load one snapshot at a time
- do not keep 52 `NeighborLoader` objects alive
- do not keep all snapshot graphs in RAM
- use `num_workers=0` by default
- allow the user to increase workers explicitly

An optional disk cache for graph-ready snapshot components may be added only after correctness is complete. Parquet remains the authoritative artifact.

---

## 10. Files to add or modify

### 10.1 Add

```text
examples/__init__.py
examples/hm_cf/__init__.py
examples/hm_cf/config.py
examples/hm_cf/snapshot.py
examples/hm_cf/io.py
examples/hm_cf/graph.py
examples/hm_cf/seed_time_loader.py
examples/build_hm_cf_snapshots.py
examples/idgnn_recommendation_cf.py
examples/hm_cf/README.md
test/examples/test_hm_cf_snapshot.py
test/examples/test_hm_cf_io.py
test/examples/test_hm_cf_graph.py
test/examples/test_hm_cf_seed_time_loader.py
```

The implementation may merge small helper modules when that materially improves clarity, but keep snapshot computation, graph attachment, and seed-time loader logic separately testable.

### 10.2 Modify

```text
pyproject.toml
```

Add an explicit SciPy dependency to the `example` optional dependency group because the new example imports `scipy.sparse`.

Do not modify the baseline behavior in:

```text
examples/idgnn_recommendation.py
```

Do not modify task or dataset semantics.

---

## 11. Public APIs and configuration

### 11.1 Configuration dataclass

Create an immutable dataclass:

```python
@dataclass(frozen=True)
class CFSnapshotConfig:
    dataset: str = "rel-hm"
    task: str = "user-item-purchase"
    window_weeks: int | None = 8
    all_history: bool = False
    min_support: int = 3
    top_l: int = 32
    alpha: float = 0.5
```

Validation:

- only `rel-hm/user-item-purchase` is initially supported
- exactly one history mode is selected
- `window_weeks > 0` when set
- `min_support >= 1`
- `top_l >= 1`
- `0 <= alpha <= 1`

Provide deterministic:

- config directory name
- manifest representation
- snapshot filename

### 11.2 Snapshot computation function

Provide a testable pure-ish function:

```python
build_snapshot(
    transactions: pd.DataFrame,
    seed_time: pd.Timestamp,
    num_articles: int,
    config: CFSnapshotConfig,
) -> tuple[pd.DataFrame, SnapshotStats]
```

The function must not access task labels.

### 11.3 Snapshot loading

Provide:

```python
load_snapshot(
    snapshot_dir: Path,
    seed_time: pd.Timestamp,
    *,
    validate: bool = True,
) -> pd.DataFrame
```

Validate:

- required columns
- exact seed time
- no self-pairs
- no duplicates in `(src_article_id, dst_article_id)`
- support threshold
- rank range
- at most top-L rows per source
- finite, nonnegative scores
- article ID bounds when bounds are supplied

### 11.4 Graph attachment

Provide:

```python
attach_cf_snapshot(...)
build_cf_schema_template(...)
build_cf_num_neighbors(...)
```

Use constants for node and edge type names rather than repeated string literals.

### 11.5 Seed-time grouping

Provide:

```python
group_recommendation_table_by_seed_time(
    table: Table,
    task: RecommendationTask,
) -> list[SeedTimeGroup]
```

Preserve original row indices.

---

## 12. CLI: snapshot preprocessing

Entry point:

```bash
python -m examples.build_hm_cf_snapshots \
  --dataset rel-hm \
  --task user-item-purchase \
  --window-weeks 8 \
  --min-support 3 \
  --top-l 32 \
  --alpha 0.5 \
  --output-root /data/seonghun/cf_snapshots
```

Required/expected arguments:

```text
--dataset
--task
--window-weeks
--all-history
--min-support
--top-l
--alpha
--output-root
--splits
--seed-time
--overwrite
--download
```

Defaults:

```text
dataset=rel-hm
task=user-item-purchase
window_weeks=8
min_support=3
top_l=32
alpha=0.5
splits=train,val,test
download=True
overwrite=False
```

Behavior:

- fail if an existing manifest conflicts with requested configuration
- skip a valid existing snapshot unless `--overwrite`
- regenerate an invalid/incomplete file only with `--overwrite`, otherwise fail clearly
- update manifest atomically after each successful snapshot so interrupted runs can resume
- print the final output directory

---

## 13. CLI: CF-augmented ID-GNN

Entry point:

```bash
python -m examples.idgnn_recommendation_cf \
  --dataset rel-hm \
  --task user-item-purchase \
  --cf-snapshot-dir /data/seonghun/cf_snapshots/rel-hm/user-item-purchase/window_8w_alpha_0.5_support_3_top32 \
  --num_layers 4 \
  --num_neighbors 128
```

Copy the relevant baseline arguments from `examples/idgnn_recommendation.py`, including:

```text
--lr
--epochs
--eval_epochs_interval
--batch_size
--channels
--aggr
--num_layers
--num_neighbors
--temporal_strategy
--max_steps_per_epoch
--num_workers
--seed
--cache_dir
```

Add:

```text
--cf-snapshot-dir
```

Defaults:

```text
num_layers=4
temporal_strategy=last
num_workers=0
```

Training must fail fast when:

- the manifest is missing or incompatible
- any required snapshot file is missing
- `num_layers < 4`
- graph schema differs across snapshots
- snapshot IDs exceed article bounds

The CF score must not be used by `Model`.

---

## 14. Tests

Tests must use small synthetic data and must not download Rel-HM.

### 14.1 Window-boundary test

Verify:

- transaction exactly at `seed_time` is included
- transaction after `seed_time` is excluded
- transaction exactly at `seed_time - 8 weeks` is excluded
- transaction after the lower boundary is included

### 14.2 Binary interaction test

Multiple purchases of the same article by one customer count once.

### 14.3 Known co-occurrence test

Construct a toy matrix with manually known `C = M.T @ M` and verify exact support.

### 14.4 Minimum-support test

Pairs with support 2 are removed when `min_support=3`; support 3 remains.

### 14.5 Alpha test

Verify the score formula for:

- `alpha=0.5`
- one non-0.5 value, such as `alpha=1.0`

Confirm directional scores when expected.

### 14.6 Top-L and tie-break test

Verify:

- no source has more than top-L rows
- order is score desc, support desc, destination ID asc
- ranks start at 1 and are consecutive

### 14.7 Parquet/manifest round trip

Verify dtypes, seed time, atomic path behavior, manifest compatibility, and resume behavior.

### 14.8 No-model-feature test

After graph attachment, the `article_cf` TensorFrame exposes only `__const__`; it must not expose score, support, or rank.

### 14.9 Base graph immutability test

Attaching one or more snapshots must not mutate base graph node stores or edge stores.

### 14.10 Graph edge-role test

For a small snapshot, verify the exact four edge indices and their direction.

### 14.11 Four-hop reachability integration test

Using a small temporal heterogeneous graph and `NeighborLoader`, verify:

```text
customer -> past transaction -> src article -> CF node -> dst article
```

and verify that reverse top-L semantics are not introduced by the expansion configuration.

Skip with a clear reason if the installed PyG backend lacks required temporal sampling support.

### 14.12 Seed-time isolation test

Create two snapshots with conflicting CF rows and verify that the loader for time `t1` sees only `t1`, and the loader for `t2` sees only `t2`.

### 14.13 Prediction-order test

Create a task table whose rows are interleaved across timestamps; group, predict with identifiable dummy outputs, scatter, and verify original row order.

### 14.14 Smoke model test

On a tiny fake graph:

- initialize the schema-template model
- run one training forward/backward pass
- run one evaluation pass
- confirm finite loss and correctly shaped top-k output

---

## 15. Acceptance criteria

Implementation is complete only when all conditions below hold.

### Correctness

- [ ] Snapshot window is exactly `(seed_time - W, seed_time]`.
- [ ] User-item interactions are binary.
- [ ] CF uses all eligible database customers.
- [ ] `C = M.T @ M` is sparse.
- [ ] `min_support=3`, `top_l=32`, and `alpha=0.5` are defaults and configurable.
- [ ] No self-pairs are stored.
- [ ] Ordered pairs are retained.
- [ ] Exact seed-time snapshots are isolated.
- [ ] CF score/support/rank are not model input.
- [ ] The intended direction-preserving four-hop route works.
- [ ] Validation/test predictions preserve original row order.
- [ ] Baseline `examples/idgnn_recommendation.py` behavior remains unchanged.

### Scalability

- [ ] No dense 106K-by-106K matrix is created.
- [ ] No Python nested loop enumerates all item pairs per customer.
- [ ] Transactions are sorted once and rolling windows use `searchsorted`.
- [ ] Only one snapshot graph/loader is retained at a time.
- [ ] The full base graph is not tensor-cloned for every seed time.
- [ ] Snapshot creation emits resource diagnostics.

### Reproducibility

- [ ] Manifest records all methodological settings.
- [ ] Tie-breaking is deterministic.
- [ ] Writes are atomic and resumable.
- [ ] Randomness follows the script seed.
- [ ] Commands are documented in `examples/hm_cf/README.md`.

### Quality

- [ ] New functions have type hints and docstrings.
- [ ] Error messages identify the seed time and file involved.
- [ ] Unit tests pass.
- [ ] Existing relevant tests pass.
- [ ] Code is formatted consistently with the repository.

---

## 16. Required commands to run during implementation

At minimum:

```bash
python -m pytest \
  test/examples/test_hm_cf_snapshot.py \
  test/examples/test_hm_cf_io.py \
  test/examples/test_hm_cf_graph.py \
  test/examples/test_hm_cf_seed_time_loader.py
```

Also run existing relevant tests:

```bash
python -m pytest test/modeling/test_link_idgnn.py
```

Run CLI help:

```bash
python -m examples.build_hm_cf_snapshots --help
python -m examples.idgnn_recommendation_cf --help
```

When Rel-HM data is available, run a one-snapshot pilot only:

```bash
python -m examples.build_hm_cf_snapshots \
  --dataset rel-hm \
  --task user-item-purchase \
  --window-weeks 8 \
  --min-support 3 \
  --top-l 32 \
  --alpha 0.5 \
  --seed-time 2020-03-02 \
  --output-root /tmp/relbench_cf_pilot
```

Do not automatically launch the full 52-snapshot preprocessing or full GPU training as part of implementation verification.

---

## 17. Out of scope

Do not implement in this change:

- direct article-article CF edges
- recency-weighted user-item matrices
- time-decayed CF
- score-aware GNN input
- weighted neighbor sampling
- hard-negative mining
- negative-count balancing
- candidate-budget reallocation
- direct optimization of locality/MAP
- learned CF relation weights
- modifications to official task labels or split definitions
- checkpoint save/load extensions
- generalization to other datasets or tasks beyond clean interfaces

---

## 18. Implementation order

1. Inspect current repository and confirm integration points.
2. Add config, filename, manifest, and validation utilities.
3. Implement and test sparse snapshot computation.
4. Implement atomic parquet I/O and resume behavior.
5. Implement graph schema template and snapshot attachment.
6. Implement direction-preserving edge-type fanouts.
7. Implement seed-time grouping and prediction-order scatter.
8. Add the preprocessing CLI.
9. Add the CF-augmented ID-GNN script.
10. Add integration and smoke tests.
11. Run targeted and existing tests.
12. Document exact commands and report any environment-dependent limitations.

---

## 19. Final implementation report

After implementation, report:

1. files added/modified
2. algorithmic choices actually implemented
3. tests and commands run
4. test results
5. one-snapshot pilot statistics, only if data was available
6. any deviation from this specification and the reason
7. remaining performance risks, without changing the methodology
