# Test-Only Product-Code Virtual Transaction Expansion for rel-hm ID-GNN

## 1. Repository and baseline

- Repository: `stanford-star/relbench`
- Expected layout: identical to the upstream repository.
- Baseline branch: `main`
- Baseline commit used to prepare this specification: `10d9cfe41413ae2548018ab219cbcd6ec177b364`
- Primary entry point: `examples/idgnn_recommendation.py`

The implementation must work with the code actually present in the checkout. Do not rely on line numbers in this document.

## 2. Objective

Add an optional **test-only graph augmentation** for:

- dataset: `rel-hm`
- task: `user-item-purchase`
- model: the existing ID-GNN example

For each test source customer, use historical transactions available at the test seed time:

```text
customer_a -- transaction_i -- article_a
```

If another existing article `article_b` has the same `product_code` as `article_a`, and `customer_a` has never transacted with `article_b` at or before the test seed time, add a synthetic transaction node:

```text
customer_a -- synthetic_transaction_i_b -- article_b
```

The synthetic transaction node must reuse the complete materialized feature row and timestamp of `transaction_i`. In rel-hm this includes the transaction information represented by columns such as `price`, `t_dat`, and `sales_channel_id`, but the implementation must copy the **entire TensorFrame row**, not manually reconstruct only selected columns.

Training and validation must remain unchanged. The augmented graph is used only by the test `NeighborLoader`.

## 3. Core design decision

### Required approach

Augment the already materialized `HeteroData` by:

1. selecting source transaction row positions from the raw rel-hm database;
2. cloning those rows from `data["transactions"].tf`;
3. cloning their values from `data["transactions"].time`;
4. appending the four existing primary-key/foreign-key edge relations for the new transaction nodes.

Use the existing node and edge types only.

### Forbidden approaches

Do not:

- add a new `same_product_code` edge type;
- connect one existing transaction node to multiple articles;
- retrain or fine-tune the model on augmented data;
- augment train or validation graphs;
- use future transactions after the test seed time;
- use test destination labels to construct candidates;
- rematerialize the full database with duplicated rows;
- recompute categorical mappings or column statistics for the augmented graph;
- mutate the base `HeteroData` object in place;
- change recommendation labels, loss, model head, ID-awareness logic, or evaluation semantics.

Cloning materialized rows is required because it guarantees that categorical IDs, numerical values, timestamp encodings, text/embedding representations, and all other transaction features are exactly identical to the source transaction representation seen by the trained encoder.

## 4. Existing repository behavior to preserve

The upstream code currently:

- constructs one graph with `make_pkey_fkey_graph`;
- uses that graph for train, validation, and test loaders;
- initializes `Model` from the base graph;
- scores sampled article nodes in `test()` without requiring changes to the prediction head;
- represents the rel-hm transaction table as a node type with:
  - `customer_id -> customer`,
  - `article_id -> article`,
  - `t_dat` as the time column.

The graph builder creates these existing edge types:

```python
("transactions", "f2p_customer_id", "customer")
("customer", "rev_f2p_customer_id", "transactions")
("transactions", "f2p_article_id", "article")
("article", "rev_f2p_article_id", "transactions")
```

The augmentation must append edges to exactly these relations.

## 5. Files to change

### 5.1 New file

Create:

```text
examples/hm_product_code_expansion.py
```

This module must contain the pure candidate-generation logic, the graph-augmentation logic, validation helpers, and a small stats/result data structure.

### 5.2 Modify

Modify:

```text
examples/idgnn_recommendation.py
```

Integrate the optional test-only augmentation and test-specific neighbor budget.

### 5.3 New tests

Create:

```text
test/examples/test_hm_product_code_expansion.py
```

Create `test/examples/` if it does not exist.

### 5.4 Files that should normally remain unchanged

Do not change these unless the existing checkout makes it strictly necessary:

```text
relbench/modeling/graph.py
relbench/modeling/loader.py
relbench/modeling/nn.py
examples/model.py
relbench/tasks/hm.py
relbench/datasets/hm.py
```

If a change outside the requested files is unavoidable, keep it minimal and explain it in the final report.

## 6. Public CLI

Add these arguments to `examples/idgnn_recommendation.py`:

```python
--test_product_code_expansion
```

- `store_true`
- default: disabled
- when disabled, preserve baseline behavior

```python
--product_code_max_virtual_per_customer
```

- integer
- default: `0`
- `0` means unlimited
- a positive value caps synthetic transaction nodes per test customer after deterministic ranking

```python
--test_num_neighbors
```

- optional integer
- default: `None`
- when `None`, use `args.num_neighbors`
- otherwise derive the test hop schedule from this value using the same formula as the existing schedule:

```python
[int(base // 2**i) for i in range(args.num_layers)]
```

Reject invalid negative caps and non-positive `test_num_neighbors`.

When expansion is requested for a dataset/task other than `rel-hm/user-item-purchase`, fail with a clear `ValueError`.

## 7. Candidate-generation API and contract

Implement a typed function with behavior equivalent to:

```python
def build_product_code_virtual_candidates(
    db: Database,
    test_table: Table,
    *,
    src_col: str = "customer_id",
    article_col: str = "article_id",
    product_code_col: str = "product_code",
    transaction_time_col: str = "t_dat",
    seed_time_col: str = "timestamp",
    max_virtual_per_customer: int = 0,
) -> tuple[pd.DataFrame, ProductCodeExpansionStats]:
    ...
```

Exact naming may vary slightly, but keep the API small, typed, documented, and independently testable.

### 7.1 Required inputs

Use:

```python
db.table_dict["transactions"].df
db.table_dict["article"].df
test_table.df[src_col]
test_table.df[seed_time_col]
```

Do not read `test_table.df[article_col]` for candidate construction.

### 7.2 Seed-time rule

The requested experiment assumes one fixed test timestamp.

- Extract unique non-null seed times from `test_table.df[seed_time_col]`.
- Require exactly one unique value.
- Raise a clear error otherwise.
- Normalize it to `pd.Timestamp`.

### 7.3 Test-customer rule

Only generate candidates for customers present in the test source rows.

Deduplicate customer IDs before filtering transactions.

### 7.4 Historical transaction rule

A source transaction is eligible only when:

```text
transaction.t_dat <= test_seed_time
```

Never include a future transaction.

Use the transaction row's **positional row number** in the DataFrame as `source_tx_pos`:

```python
np.arange(len(transactions_df))
```

Do not use the pandas index as a graph node ID because a filtered DataFrame may retain a non-consecutive index.

### 7.5 Product-code lookup

Join each historical source transaction to its source article's `product_code`.

Exclude rows with null `product_code`.

### 7.6 Source-transaction selection

To avoid multiple synthetic nodes for repeated purchases of the same product, select exactly one source transaction per:

```text
(customer_id, product_code)
```

Selection must be deterministic:

1. greatest `t_dat`;
2. on a timestamp tie, greatest `source_tx_pos`.

This is the `latest-per-customer-product` policy and is the only required policy.

### 7.7 Variant expansion

For each selected source transaction, enumerate all articles with the same `product_code`.

Exclude:

- the source article itself;
- every target article that the same customer already transacted with at or before the seed time;
- duplicate `(customer_id, target_article_id)` candidates.

If duplicate candidates are encountered, retain the one backed by the latest source transaction using:

1. greatest source `t_dat`;
2. greatest `source_tx_pos`;
3. smallest `target_article_id` only as a final deterministic ordering key.

### 7.8 Optional per-customer cap

Before applying the cap, sort candidates deterministically by:

1. `customer_id` ascending;
2. source `t_dat` descending;
3. `source_tx_pos` descending;
4. `target_article_id` ascending.

For positive `max_virtual_per_customer`, retain the first N rows per customer.

For `0`, retain all candidates.

### 7.9 Candidate output schema

Return a DataFrame containing at least:

```text
customer_id
source_article_id
target_article_id
product_code
source_tx_pos
source_t_dat
```

The DataFrame row order determines synthetic transaction node ID assignment and must be deterministic.

Return an empty DataFrame with the same columns when there are no candidates.

## 8. Expansion statistics

Define a lightweight typed result, preferably a frozen dataclass, containing at least:

```text
seed_time
num_test_customers
num_historical_transactions
num_selected_source_transactions
num_candidates_before_seen_filter
num_candidates_after_seen_filter
num_synthetic_transactions
num_customers_with_synthetic_transactions
mean_synthetic_per_expanded_customer
max_synthetic_per_customer
```

Zero-candidate cases must not produce division-by-zero or NaN unless explicitly documented.

Add a compact logging function or `__str__` representation suitable for printing from the example script.

## 9. HeteroData augmentation API and contract

Implement a typed function with behavior equivalent to:

```python
def augment_hm_graph_with_virtual_transactions(
    data: HeteroData,
    candidates: pd.DataFrame,
) -> HeteroData:
    ...
```

It may also return assigned synthetic IDs or updated stats if useful.

### 9.1 Base graph immutability

The returned graph must be a separate `HeteroData` object.

Avoid `deepcopy(data)` because rel-hm is large.

Create a new `HeteroData` and copy existing store attributes by reference, or use another proven shallow structural-copy strategy. Then replace only the modified transaction attributes and four modified edge tensors.

The original graph must retain:

- its original transaction TensorFrame length;
- its original transaction time length;
- its original edge tensors and edge counts.

### 9.2 Synthetic transaction IDs

Let:

```python
base_num_transactions = len(data["transactions"].tf)
num_synthetic = len(candidates)
synthetic_tx_ids = (
    base_num_transactions
    + torch.arange(num_synthetic, dtype=torch.long)
)
```

The i-th candidate row must map to the i-th synthetic transaction ID.

### 9.3 Feature cloning

Use the public PyTorch Frame row indexing and concatenation API where available:

```python
source_pos = torch.as_tensor(
    candidates["source_tx_pos"].to_numpy(),
    dtype=torch.long,
)
synthetic_tf = data["transactions"].tf[source_pos]
augmented_tf = torch_frame.cat(
    [data["transactions"].tf, synthetic_tf],
    dim=0,
)
```

Do not reconstruct transaction feature tensors column by column unless compatibility with the installed minimum version makes this unavoidable.

### 9.4 Time cloning

Append:

```python
data["transactions"].time[source_pos]
```

to the existing transaction time tensor.

The cloned time must be bitwise equal to its source value.

### 9.5 Edge construction

For each candidate row, append:

```text
synthetic transaction -> candidate customer
candidate customer -> synthetic transaction
synthetic transaction -> target article
target article -> synthetic transaction
```

to the four existing edge types.

Use `torch.long`.

Preserve edge-index shape `[2, num_edges]`.

Sort each modified edge index with `torch_geometric.utils.sort_edge_index` after concatenation, unless the implementation provides an equivalently correct deterministic merge of sorted edge lists.

Do not alter unrelated edge types.

### 9.6 Node count and validation

Ensure PyG sees the augmented transaction-node count. Set the transaction store's explicit `num_nodes` only if needed by the installed PyG behavior.

Run:

```python
augmented_data.validate()
```

and fail clearly on schema mismatch.

The following sets must be identical between base and augmented graphs:

```python
set(data.node_types)
set(augmented_data.node_types)
set(data.edge_types)
set(augmented_data.edge_types)
```

### 9.7 Zero-candidate behavior

When `candidates` is empty:

- return a valid separate graph or safely reuse the base graph only if no mutation can occur later;
- do not fail;
- report zero expansion;
- preserve all baseline predictions.

Prefer returning a separate shallow structural copy for consistent semantics.

## 10. Integration into `examples/idgnn_recommendation.py`

Refactor the script so that graph ownership is explicit.

### 10.1 Base graph

Build the existing graph exactly once:

```python
base_data, col_stats_dict = make_pkey_fkey_graph(...)
```

Initialize the model from `base_data`.

### 10.2 Train and validation loaders

Before training, create loaders only for:

```text
train
val
```

Both must use `base_data`.

The existing training loss, sparse target handling, validation evaluation, checkpoint selection, and model loading must remain semantically unchanged.

### 10.3 Test graph creation timing

Create the test graph and test loader only after:

```python
model.load_state_dict(state_dict)
```

Behavior:

- expansion disabled:
  - test loader uses `base_data`;
- expansion enabled:
  1. read `task.get_table("test")`;
  2. generate candidates from the raw database and test source/time columns;
  3. augment `base_data` without mutating it;
  4. print expansion statistics;
  5. use the augmented graph for the test loader.

This ordering makes it structurally impossible for synthetic nodes to affect training or validation.

### 10.4 Test input rows

Preserve the original order of test source rows and test seed times.

Do not alter the task table.

The destination labels may continue to be used by the existing RelBench evaluation machinery, but they must not influence candidate generation, source-transaction selection, graph construction, or sampling configuration.

### 10.5 Test neighbor schedule

Use:

```python
test_neighbor_base = (
    args.num_neighbors
    if args.test_num_neighbors is None
    else args.test_num_neighbors
)
test_num_neighbors = [
    int(test_neighbor_base // 2**i)
    for i in range(args.num_layers)
]
```

Do not change train/validation neighbor schedules.

### 10.6 Test scoring

Do not change the existing `test()` score construction or top-k behavior unless required to fix an unrelated pre-existing bug.

Once `target_article_id` is sampled through the synthetic two-hop path, the existing ID-GNN destination readout must score it normally.

## 11. Runtime diagnostics

When expansion is enabled, print one compact block containing:

- seed time;
- test customer count;
- historical transaction count used;
- selected latest customer-product source transaction count;
- synthetic transaction count;
- expanded customer count;
- average and maximum synthetic transactions per expanded customer;
- configured per-customer cap;
- base and augmented transaction node counts;
- train/validation neighbor schedule;
- test neighbor schedule.

Do not print candidate rows or customer IDs by default.

## 12. Required invariants and assertions

Add inexpensive runtime checks for:

1. every `source_tx_pos` is in `[0, base_num_transactions)`;
2. every candidate customer ID is in the customer node range;
3. every target article ID is in the article node range;
4. candidate `source_t_dat <= seed_time`;
5. no `source_article_id == target_article_id`;
6. no duplicate `(customer_id, target_article_id)`;
7. no target candidate exists in the customer's historical seen set;
8. feature and time append lengths equal the synthetic count;
9. every modified edge relation gains exactly `num_synthetic` edges;
10. the base graph is not mutated;
11. augmented node and edge type sets are unchanged.

Use clear error messages rather than bare assertions for user-data/schema errors. Internal consistency checks may use assertions.

## 13. Unit tests

Tests must be small and must not download rel-hm.

### 13.1 Candidate-generation test

Construct an in-memory `Database` with:

- a small `article` table containing multiple articles per `product_code`;
- a small `customer` table;
- a `transactions` table with `customer_id`, `article_id`, `price`, `t_dat`, and `sales_channel_id`;
- a test `Table` with one seed timestamp.

Cover:

- only test customers are expanded;
- transactions after seed time are ignored;
- null product codes are ignored;
- source article is excluded;
- already-seen sibling articles are excluded;
- latest transaction per `(customer, product_code)` is selected;
- timestamp ties use greatest positional row;
- output is deterministic;
- positive per-customer cap is deterministic;
- `0` means unlimited;
- output has no duplicate `(customer, target_article)` pairs;
- candidate generation does not read or depend on test destination labels.

### 13.2 Graph-augmentation test

Construct a tiny `HeteroData` with:

- customer, article, and transaction node stores;
- a transaction `TensorFrame` with at least numerical, categorical, and timestamp-like feature tensors;
- transaction time;
- the four relevant edge types.

Verify:

- original graph is unchanged;
- augmented transaction TensorFrame length increases correctly;
- each synthetic TensorFrame row equals its source row;
- each synthetic time equals its source time;
- each of the four edge types gains exactly one edge per synthetic node;
- synthetic nodes connect to the expected customer and target article;
- unrelated edge types, if added to the fixture, remain unchanged;
- node/edge type sets remain identical;
- `augmented_data.validate()` passes.

### 13.3 Empty-candidate test

Verify a valid graph is returned and all counts remain unchanged.

### 13.4 Error tests

Cover at least:

- multiple test seed times;
- negative cap;
- missing required rel-hm table/column;
- out-of-range source transaction position passed to graph augmentation.

## 14. Compatibility and style

- Python: `>=3.10`
- Follow existing repository formatting and typing style.
- Prefer `from __future__ import annotations` in the new module.
- Add docstrings to public functions and the stats dataclass.
- Avoid introducing a new dependency.
- Use vectorized pandas operations; do not loop over customers or transactions in Python.
- Keep all tensors on CPU during graph construction.
- Preserve tensor dtypes and devices.
- Avoid large intermediate Cartesian products beyond the required product-code join. Filter to test customers and select latest customer-product transactions before expanding variants.
- Do not cache augmented TensorFrames to the existing materialization cache.

## 15. Performance requirements

The candidate-generation order must be:

1. filter to test customers and historical time;
2. attach product code;
3. reduce to latest `(customer, product_code)` source transaction;
4. expand same-product articles;
5. anti-join historical seen pairs;
6. deduplicate;
7. apply optional cap.

This ordering is required to limit memory.

Do not perform a full transactions-by-articles merge before filtering and source reduction.

## 16. Acceptance criteria

The implementation is complete when all of the following hold:

1. `pytest -q test/examples/test_hm_product_code_expansion.py` passes.
2. Existing relevant graph/model tests still pass.
3. With no new flag, the original train/validation/test graph semantics are preserved.
4. With `--test_product_code_expansion`, training and validation use only the base graph.
5. The test graph contains a two-hop path:

   ```text
   customer -> synthetic transaction -> same-product unseen article
   ```

6. Every synthetic transaction feature row and time are exact clones of its selected source transaction.
7. No new node type or edge type is introduced.
8. No test-future transaction or test destination label is used in augmentation.
9. The base graph is unchanged after test graph construction.
10. Test scoring and MAP@K evaluation run through the existing ID-GNN code path.
11. The implementation is deterministic for a fixed database and arguments.
12. The final report lists changed files, tests run, and any command that could not be run because rel-hm data was unavailable.

## 17. Suggested verification commands

Run at minimum:

```bash
python -m compileall \
  examples/hm_product_code_expansion.py \
  examples/idgnn_recommendation.py

pytest -q test/examples/test_hm_product_code_expansion.py
pytest -q test/modeling/test_graph.py
pytest -q test/modeling/test_link_idgnn.py
```

When rel-hm data and the example dependencies are available, run a small smoke test:

```bash
python examples/idgnn_recommendation.py \
  --dataset rel-hm \
  --task user-item-purchase \
  --epochs 1 \
  --max_steps_per_epoch 2 \
  --batch_size 64 \
  --num_workers 0 \
  --test_product_code_expansion \
  --product_code_max_virtual_per_customer 32 \
  --test_num_neighbors 256
```

The smoke test is not a substitute for unit tests.

## 18. Final implementation report

After coding, report:

- changed and added files;
- implementation summary;
- candidate policy and cap semantics;
- confirmation that base data is not mutated;
- test commands and results;
- smoke-test result or reason it was not run;
- any deviation from this specification and its justification.
