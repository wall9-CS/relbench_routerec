# Implementation Summary

This document explains the implementation from commit `906d15d` (`1st implementation`).
The change adds a test-only product-code expansion path for the rel-hm ID-GNN
recommendation example.

## Goal

The implementation expands the test graph for `rel-hm/user-item-purchase` so that
articles sharing a `product_code` with a customer's historical purchases can be
reached by the existing ID-GNN message-passing path:

```text
customer -> synthetic transaction -> sibling article
```

This is done without changing training or validation behavior. The model is still
trained and validated on the original graph.

## Files Added or Changed

- `examples/hm_product_code_expansion.py`
  - Adds candidate generation for same-product-code virtual transactions.
  - Adds graph augmentation that clones transaction TensorFrame rows and times.
  - Adds validation checks and diagnostic formatting.
- `examples/idgnn_recommendation.py`
  - Adds CLI flags for enabling test-only product-code expansion.
  - Builds train and validation loaders from the base graph only.
  - Builds the augmented graph only after the best checkpoint is loaded, and only
    for the test loader.
- `test/examples/test_hm_product_code_expansion.py`
  - Adds focused in-memory tests for candidate generation and graph augmentation.
- `description_spec.md`
  - Tracks the written implementation specification.

## Candidate Generation

`build_product_code_virtual_candidates()` constructs a deterministic candidate
DataFrame using:

- test source customers;
- exactly one test seed timestamp;
- transactions at or before that seed timestamp;
- article `product_code` values.

It intentionally does not use test destination labels. For each test customer and
product code, it keeps the latest historical source transaction, expands to other
articles with the same `product_code`, removes articles the customer already
bought before the seed time, deduplicates `(customer_id, target_article_id)`, and
optionally applies a per-customer cap.

The resulting candidate rows contain:

```text
customer_id
source_article_id
target_article_id
product_code
source_tx_pos
source_t_dat
```

`source_tx_pos` is a positional transaction node index, not a pandas index. This
matters because graph transaction node IDs match materialized row positions.

## Graph Augmentation

`augment_hm_graph_with_virtual_transactions()` returns a separate augmented
`HeteroData` object. It does not mutate the base graph.

For each candidate, it:

- creates one synthetic transaction node after the existing transaction nodes;
- clones the complete source transaction TensorFrame row;
- clones the source transaction `time`;
- appends one edge to each existing rel-hm transaction relation:
  - `transactions -> customer`
  - `customer -> transactions`
  - `transactions -> article`
  - `article -> transactions`

No new edge type is introduced. This keeps the existing ID-GNN model schema and
learned relation parameters usable at test time.

## ID-GNN Integration

The feature is disabled by default. Enable it with:

```bash
python examples/idgnn_recommendation.py \
  --dataset rel-hm \
  --task user-item-purchase \
  --test_product_code_expansion
```

Optional controls:

```bash
--product_code_max_virtual_per_customer 32
--test_num_neighbors 256
```

`--product_code_max_virtual_per_customer` limits how many synthetic candidates
are added per test customer. A value of `0` means unlimited.

`--test_num_neighbors` changes the neighbor sampling budget for the test loader
only. Training and validation continue to use `--num_neighbors`.

The expansion flag is restricted to `rel-hm/user-item-purchase`. Other datasets
or tasks raise an error if the flag is used.

## Safety Properties

The implementation enforces these invariants:

- only test-time graph augmentation is performed;
- test labels are not used for candidate generation;
- future transactions after the test seed time are excluded;
- source and target articles must differ;
- duplicate `(customer_id, target_article_id)` candidates are rejected;
- source transaction positions must be in range;
- customer and article node IDs must be in range;
- the base graph's transaction TensorFrame, transaction time tensor, and edge
  tensors must remain unchanged;
- augmented graphs must preserve the original node and edge type sets.

## Verification

Focused verification run:

```bash
python -m pytest test\examples\test_hm_product_code_expansion.py
```

Result:

```text
6 passed, 2 warnings
```

The warnings come from `torch.jit.script` deprecation messages in installed
dependencies.

No rel-hm smoke test was run during this documentation update.
