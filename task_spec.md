# Task Specification: Oracle GT Injection for ID-GNN on RelBench

## 1. Objective

This task studies **how to inject additional recommendation candidates into the relational graph**, not how to retrieve those candidates.

The broader research problem is decomposed into:

1. **Candidate retrieval**: which additional destination nodes should be selected?
2. **Candidate injection / materialization**: given selected destination nodes, how should they be placed into the relational graph so that ID-GNN can rank them effectively?

This task intentionally isolates (2).

For the first-stage diagnostic experiment, the additional candidate identity is given by an **oracle ground-truth (GT)**. The experiment therefore does **not** evaluate a practical candidate retrieval method. Its purpose is to measure the capability of different relational placements to convert candidate availability into ranking performance.

Primary research question:

> Given the same oracle GT candidate, which relational injection structure allows ID-GNN to use the candidate most effectively?

The initial target is the RelBench `rel-hm` dataset and the `user-item-purchase` recommendation task.

---

## 2. Background and Motivation

Previous experiments showed two important observations.

### 2.1 Increasing candidate coverage is not sufficient

A subgraph can contain more GT items without improving final recommendation MAP. Therefore, candidate coverage and ranking quality must be treated separately.

### 2.2 Different graph structures can produce different ID-GNN performance

Recent experiments compared deeper ID-GNN, item-side CF expansion, source-side CF expansion, and direct CF-style connections. Their results differed even when all were intended to expose non-local recommendation candidates.

This motivates a controlled oracle experiment where candidate identity is fixed and only the **graph materialization structure** changes.

---

## 3. Core Experimental Principle

For query

\[
q = (u, t_q)
\]

let:

- \(u\): source node
- \(t_q\): query / seed time
- \(Y_q\): GT destination nodes
- \(C_q^{raw}\): destination candidates reachable in the baseline raw ID-GNN candidate support
- \(Y_q^{miss}\): GT nodes not already available to the raw model

\[
Y_q^{miss} = Y_q \setminus C_q^{raw}
\]

Only \(Y_q^{miss}\) is eligible for oracle injection.

Already-covered GT nodes must not receive additional oracle edges or nodes.

The first experiment uses **GT-only injection**:

- only missing GT nodes are synthetically added;
- no synthetic negative candidates are added in this stage.

Important: GT-only means only the **new synthetic candidates** are GT. The existing raw recommendation candidates remain unchanged.

---

## 4. Baseline Model and Fixed Conditions

Unless explicitly changed by an experiment, preserve the existing rel-hm ID-GNN pipeline.

Use:

- Dataset: `rel-hm`
- Task: `user-item-purchase`
- Model: 4-layer ID-GNN
- Existing node encoder
- Existing temporal sampling logic
- Existing seed-time semantics
- Existing neighbor sampling budget
- Existing loss function
- Existing optimization and training schedule
- Existing evaluation metrics
- Existing random-seed convention

Do not change baseline graph construction except for the experiment-specific synthetic table / relation.

The experiment must support training, validation, and test under the same injection definition.

Because GT identity is used to build oracle relations, this is a **diagnostic upper-bound experiment**, not a deployable recommendation method.

---

## 5. RDB-to-Graph Assumption

RelBench converts the relational database into a heterogeneous graph where:

- a table corresponds to a node type,
- a row corresponds to a node,
- PK/FK references induce graph edges.

Therefore a synthetic many-to-many relation should be implemented as a synthetic relation / junction table rather than as an arbitrary graph-only edge whenever practical.

This means the direct candidate relation is materialized as:

```text
src -> candidate_anchor_row -> GT
```

not as a single graph edge:

```text
src -> GT
```

The same RDB-first principle applies to item-anchor and source-anchor relations.

---

## 6. Injection Methods

The first-stage comparison contains three injection methods.

### 6.1 Direct Candidate Injection

Create a synthetic junction relation between the query source and the missing GT candidate.

RDB abstraction:

```text
candidate_anchor
---------------
anchor_id
src_id        -> source table
candidate_id  -> destination/item table
```

Graph path:

```text
src -> candidate_anchor -> GT
```

Purpose:

> Measure how well ID-GNN can use a GT candidate when it is exposed through the shortest RDB-compatible synthetic relation.

This method has no anchor-availability constraint. Every missing GT can be injected.

---

### 6.2 Item-Anchor Injection

Use one historical item of the query source as relational evidence for the missing GT.

Graph path:

```text
src -> fact -> historical item -> item_anchor -> GT
```

Only the final relation is synthetic:

```text
historical item -> item_anchor -> GT
```

The historical item and the path:

```text
src -> fact -> historical item
```

must already exist in the historical RDB snapshot.

The anchor item is selected by Katz similarity on a snapshot-specific **item-item projected graph**.

For query source \(u\) and missing GT \(y\):

\[
i^* =
\arg\max_{i \in H_u(t)}
Katz_{item}(i, y)
\]

where \(H_u(t)\) is the set of items historically interacted with by \(u\) before the applicable snapshot cutoff.

If no valid historical item has positive Katz score, do not inject that GT under Item-Anchor.

Initial main experiment uses Top-1 anchor.

Later ablation may use Top-3 anchors.

---

### 6.3 Source-Anchor Injection

Use another source node that interacted with the GT in historical data.

Graph path:

```text
query src -> src_anchor -> supporting src -> fact -> GT
```

Only the source-to-source anchor relation is synthetic:

```text
query src -> src_anchor -> supporting src
```

The path:

```text
supporting src -> fact -> GT
```

must correspond to a real historical interaction available before the applicable snapshot cutoff.

For missing GT \(y\), define historical supporting sources:

\[
S_y(t) =
\{v \neq u \mid v \text{ interacted with } y \text{ before the cutoff}\}
\]

Among them, select:

\[
v^* =
\arg\max_{v \in S_y(t)}
Katz_{user}(u, v)
\]

where Katz is computed on a snapshot-specific **user-user projected graph**.

If there is no supporting source with positive Katz score, do not inject that GT under Source-Anchor.

Initial main experiment uses Top-1 anchor.

Later ablation may use Top-3 anchors.

---

## 7. Methods Explicitly Excluded from the First Experiment

### 7.1 Schema-Preserving Injection

Excluded.

Reason: using an existing fact-table type for a synthetic fact row requires assigning meaningful fact features. Those feature values are undefined for a non-existent interaction and would introduce a major confounder.

### 7.2 Natural-Path Grafting

Excluded.

Reason: if the GT is already within the raw receptive field, it is already available. If it lies outside the fixed GNN hop range, preserving the natural path does not make it reachable unless the path is shortened, in which case the method becomes another form of shortcut injection.

### 7.3 Controlled-Negative Injection

Deferred to a later phase.

The first study is GT-only and is intended as a capability / mechanism diagnostic.

---

## 8. Candidate-Set Isolation

Injection methods may introduce additional contextual nodes through message passing.

For example, Source-Anchor may expose the neighborhood of a supporting source. These contextual destination nodes must not automatically become new recommendation candidates in the controlled experiment.

The scoring candidate set should be isolated as:

\[
C_q^{score} =
C_q^{raw} \cup Y_q^{successfully\ injected}
\]

Additional nodes exposed only as graph context may participate in message passing but should be excluded from recommendation scoring unless they were already part of \(C_q^{raw}\).

This isolation is important because the experiment is intended to compare **candidate placement**, not accidental candidate-set expansion.

---

## 9. Snapshot Principle

All anchor construction is time-dependent.

For each query time \(t_q\), use the most recent valid historical snapshot according to the existing CF snapshot convention.

The snapshot must contain only information available before its cutoff.

Synthetic oracle relations are **query-local augmentation**:

- the schema / relation type is global;
- the oracle synthetic rows for one query must not become historical observations for later queries.

Do not persist GT-derived oracle relations as ordinary history.

---

## 10. Primary Outputs

The main comparison is:

1. Raw 4-layer ID-GNN
2. Direct GT injection
3. Item-Anchor@1
4. Source-Anchor@1

Report at minimum:

- injection coverage
- achievable MAP
- realized MAP
- MAP@K / existing recommendation metrics
- injected-GT sampled rate
- injected-GT rank statistics
- anchor availability statistics

Because Item-Anchor and Source-Anchor may fail to find an anchor, distinguish:

### Overall oracle utility

Performance over the normal evaluation population using each method's actual injection availability.

### Conditional placement quality

Performance / GT rank conditional on successful injection.

Also retain the common-support subset where all compared methods can inject the same GTs for later controlled analysis.

---

## 11. Expected Implementation Outputs

The implementation should produce reusable components for:

- historical snapshot loading / generation
- projected item-item graph generation
- projected user-user graph generation
- truncated Katz calculation
- per-query missing-GT identification
- item anchor lookup
- source anchor lookup
- synthetic relation materialization
- candidate scoring masks
- per-GT diagnostic logging

The code should avoid coupling the oracle GT selection logic with the ID-GNN model internals. Injection should be implemented as a graph/data construction layer so later candidate-retrieval methods can replace the oracle GT source without redesigning the model.
