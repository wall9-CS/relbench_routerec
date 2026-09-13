# Preprocessing Specification: Snapshot Graphs, Katz Anchors, and Feasibility Statistics

## 1. Purpose

This preprocessing stage prepares all data required for the GT-only candidate-injection experiments on:

- Dataset: `rel-hm`
- Task: `user-item-purchase`

The preprocessing has four goals:

1. identify GT items missing from the raw ID-GNN candidate support;
2. build historical snapshot-specific item-item and user-user graphs;
3. compute Katz-based Item-Anchor and Source-Anchor candidates;
4. report feasibility statistics before expensive model training.

The preprocessing must not modify the base ID-GNN graph permanently.

---

## 2. Temporal Snapshot Semantics

Reuse the existing snapshot interval / boundaries used by the previous CF-based experiments.

For a query:

\[
q = (u, t_q)
\]

resolve the historical snapshot:

\[
s(q) = \max\{s \mid s \le t_q\}
\]

or the exact equivalent boundary rule already used by the existing temporal pipeline.

Do not invent a new temporal convention.

All historical interactions used for projection graphs and anchor selection must satisfy the existing no-future-information rule.

The snapshot state must not contain any oracle relation created from GT labels.

---

## 3. Historical User-Item Interaction Matrix

For each snapshot \(s\), build a sparse binary interaction matrix:

\[
B_s \in \{0,1\}^{|U| \times |I|}
\]

where:

\[
B_s[u,i] = 1
\]

iff user \(u\) interacted with item \(i\) in historical data available at snapshot \(s\).

For the first implementation:

- use binary interaction presence;
- repeated purchases do not increase \(B_s[u,i]\);
- exclude future interactions;
- preserve the exact source/item ID mapping used by the existing rel-hm pipeline.

Use sparse representations. Do not construct a dense matrix at rel-hm scale.

---

## 4. Item-Item Projection Graph

Construct an item-item projection from:

\[
A_s^{item,raw} = B_s^\top B_s
\]

Remove the diagonal.

For the first Katz experiment, convert the projection to a binary adjacency graph:

\[
A_s^{item}[i,j] =
\mathbf{1}(A_s^{item,raw}[i,j] > 0)
\]

Interpretation:

> two items are adjacent if at least one historical user interacted with both before the snapshot cutoff.

The Katz measure is then used to capture multi-hop structural relatedness on this homogeneous item graph.

If full projection is too large, use the existing sparse CF infrastructure and safe pruning/indexing mechanisms, but do not silently change the semantic definition above. Any pruning rule must be logged in the preprocessing metadata.

---

## 5. User-User Projection Graph

Construct a user-user projection from:

\[
A_s^{user,raw} = B_s B_s^\top
\]

Remove the diagonal.

For the first Katz experiment, convert it to binary adjacency:

\[
A_s^{user}[u,v] =
\mathbf{1}(A_s^{user,raw}[u,v] > 0)
\]

Interpretation:

> two users are adjacent if they share at least one historically interacted item before the snapshot cutoff.

Again, use sparse graph representations and explicit pruning metadata if required for scalability.

---

## 6. Katz Measure

Use truncated Katz:

\[
Katz_L(a,b) =
\sum_{\ell=1}^{L}
\beta^\ell
(A^\ell)_{a,b}
\]

Evaluate two path horizons:

- `L = 3`
- `L = 5`

The purpose is to test whether L=3 is too local for anchor selection and whether L=5 discovers meaningfully different higher-order structure.

Use the same beta policy for both L values.

Do not tune beta separately for L=3 and L=5.

Beta must satisfy a stable, documented policy. Prefer one deterministic rule such as a fixed safe fraction of the inverse spectral radius, or reuse an already-established project convention if one exists.

Store beta and all Katz configuration in preprocessing metadata.

---

## 7. Avoid Full All-Pairs Katz

Do not compute dense all-pairs Katz matrices unless clearly feasible.

Only the following pair types are required.

### 7.1 Item-Anchor candidate pairs

For query source \(u\), missing GT \(y\), and historical item set:

\[
H_u(s) =
\{i \mid u \text{ interacted with } i \text{ before } s\}
\]

compute:

\[
Katz_{item}(i,y)
\]

only for:

\[
i \in H_u(s)
\]

### 7.2 Source-Anchor candidate pairs

For missing GT \(y\), define historical supporting users:

\[
S_y(s) =
\{v \neq u \mid v \text{ interacted with } y \text{ before } s\}
\]

compute:

\[
Katz_{user}(u,v)
\]

only for:

\[
v \in S_y(s)
\]

Use sparse truncated-path computation, cached powers, or another scalable equivalent implementation.

The implementation should prioritize exact semantics for L=3 and L=5 over unnecessary all-pairs materialization.

---

## 8. Missing-GT Identification

For every train/validation/test query, derive:

- \(Y_q\): GT items
- \(C_q^{raw}\): raw ID-GNN candidate support under the baseline setting

Then compute:

\[
Y_q^{miss} =
Y_q \setminus C_q^{raw}
\]

Only \(Y_q^{miss}\) is processed for oracle anchor lookup.

Already-covered GTs must be marked but must not receive synthetic injection.

The raw candidate-support extraction must match the actual baseline ID-GNN evaluation pipeline.

---

## 9. Item-Anchor Lookup

For each:

\[
y \in Y_q^{miss}
\]

evaluate historical items:

\[
i \in H_u(s)
\]

For each Katz horizon \(L \in \{3,5\}\), find:

\[
i_L^* =
\arg\max_{i \in H_u(s)}
Katz^{(L)}_{item}(i,y)
\]

A valid anchor requires:

\[
Katz^{(L)}_{item}(i_L^*,y) > 0
\]

If no positive-Katz historical item exists:

- mark Item-Anchor as unavailable for this GT;
- do not apply fallback;
- do not substitute latest item;
- do not use random anchor.

Tie-breaking must be deterministic.

Recommended tie-break order:

1. larger Katz score;
2. more recent interaction between query source and anchor item;
3. stable item ID ordering.

Store both Top-1 and Top-3 candidates if inexpensive, even though the first model experiment uses Top-1.

---

## 10. Source-Anchor Lookup

For each:

\[
y \in Y_q^{miss}
\]

first find historical users that interacted with the GT:

\[
S_y(s) =
\{v \neq u \mid v \text{ interacted with } y \text{ before } s\}
\]

For each Katz horizon \(L \in \{3,5\}\), find:

\[
v_L^* =
\arg\max_{v \in S_y(s)}
Katz^{(L)}_{user}(u,v)
\]

A valid anchor requires:

- \(S_y(s)\neq\emptyset\)
- \(Katz^{(L)}_{user}(u,v_L^*) > 0\)

If no valid supporting user exists:

- mark Source-Anchor as unavailable for this GT;
- do not apply fallback;
- do not connect to a random user.

Tie-breaking must be deterministic.

Recommended tie-break order:

1. larger Katz score;
2. stronger historical direct overlap if already available as metadata;
3. stable source ID ordering.

Store both Top-1 and Top-3 candidates if inexpensive.

---

## 11. L=3 vs L=5 Diagnostic Statistics

Before model training, compare Katz horizon choices.

Report separately for Item-Anchor and Source-Anchor.

### 11.1 Anchor availability

For each L:

\[
Availability_L =
\frac{
\#\{\text{missing GTs with valid positive-Katz anchor}\}
}{
\#\{\text{missing GTs}\}
}
\]

### 11.2 Top-1 anchor agreement

Among GTs where both L values have valid anchors:

\[
Agreement@1 =
P(a_{L=3}^* = a_{L=5}^*)
\]

### 11.3 Top-3 overlap

For GTs with sufficient valid anchors, compute Jaccard overlap between Top-3 sets for L=3 and L=5.

### 11.4 Score separation

Record:

\[
Katz(top1) - Katz(top2)
\]

for both L values.

### 11.5 Common support

Count missing GTs where:

- Item-Anchor is available for both L=3 and L=5;
- Source-Anchor is available for both L=3 and L=5;
- both Item-Anchor and Source-Anchor are simultaneously available.

Do not choose the final Katz horizon using test recommendation performance.

Use preprocessing / validation diagnostics only.

---

## 12. Required Preprocessing Statistics

Produce a machine-readable summary and a human-readable table for each split.

At minimum report:

### Query / GT statistics

- number of queries
- total GT count
- raw-covered GT count
- missing GT count
- missing GT rate
- mean / median / max missing GT per query

### Item history statistics

- mean / median / max historical items per query
- item-anchor availability for L=3
- item-anchor availability for L=5
- Top-1 agreement between L=3 and L=5
- Top-3 overlap
- distribution of Top-1 Katz scores
- distribution of number of positive-Katz historical items

### Source-support statistics

- percentage of missing GTs with at least one historical supporting source
- source-anchor availability for L=3
- source-anchor availability for L=5
- Top-1 agreement between L=3 and L=5
- Top-3 overlap
- distribution of Top-1 Katz scores
- distribution of number of valid supporting sources

### Joint statistics

- both Item and Source anchors available
- Item only
- Source only
- neither available
- common-support count for future controlled comparison

---

## 13. Preprocessing Output Format

Create reusable artifacts rather than recomputing Katz inside each training iteration.

Suggested output hierarchy:

```text
preprocessed/
  rel-hm/
    snapshots/
      <snapshot_id>/
        metadata.json
        item_graph.*
        user_graph.*
    anchors/
      train.parquet
      val.parquet
      test.parquet
    stats/
      preprocessing_summary.json
      preprocessing_summary.csv
```

Each anchor row should contain enough data for query-local graph materialization.

Suggested columns:

```text
query_id
split
seed_time
snapshot_id
src_id
gt_id
raw_covered

item_anchor_available_l3
item_anchor_id_l3
item_katz_l3
item_top3_ids_l3
item_top3_scores_l3

item_anchor_available_l5
item_anchor_id_l5
item_katz_l5
item_top3_ids_l5
item_top3_scores_l5

src_anchor_available_l3
supporting_src_id_l3
src_katz_l3
src_top3_ids_l3
src_top3_scores_l3

src_anchor_available_l5
supporting_src_id_l5
src_katz_l5
src_top3_ids_l5
src_top3_scores_l5
```

Add implementation-specific stable identifiers as needed.

---

## 14. Validation / Sanity Checks

The preprocessing pipeline must fail loudly if any of the following occurs:

1. anchor selection uses an interaction after the allowed snapshot cutoff;
2. a Source-Anchor supporting source does not actually have a historical interaction with the GT;
3. an Item-Anchor item was not historically interacted with by the query source;
4. an already raw-covered GT is marked for oracle injection;
5. an unavailable anchor receives a fallback implicitly;
6. L=3 and L=5 use different beta policies without explicit configuration;
7. ID mappings differ from the baseline rel-hm pipeline.

Add small-unit tests on toy graphs where Katz values and anchor choices can be verified manually.

---

## 15. Completion Criteria for Preprocessing

Preprocessing is complete when:

- all train/val/test queries have missing-GT records;
- item/user snapshot projection graphs are reproducible;
- Katz L=3 and L=5 anchor lookup is available;
- no-fallback behavior is enforced;
- required statistics are generated;
- temporal leakage checks pass;
- output anchor tables can be consumed without recomputing Katz during ID-GNN training.
