# rel-hm UPR Experiment Specification: GT-Only Candidate Injection

## 1. Experiment Goal

Run a controlled oracle study on:

- Dataset: `rel-hm`
- Task: `user-item-purchase`

The experiment evaluates how different relational materializations of the same missing GT candidate affect ID-GNN ranking performance.

This experiment is not a candidate-retrieval benchmark.

GT identity is intentionally given by an oracle so that retrieval quality is removed from the comparison.

Main question:

> When a missing GT item is known, does ID-GNN use it better when the GT is directly exposed, connected through a historical item anchor, or connected through a historical supporting-source anchor?

---

## 2. Experiment Variants

The first experiment suite contains:

1. `raw`
2. `direct`
3. `item_anchor`
4. `source_anchor`

Initial anchor cardinality:

- Item-Anchor: Top-1
- Source-Anchor: Top-1

Katz horizon should be selected from preprocessing diagnostics between:

- `L = 3`
- `L = 5`

Do not select L using test MAP.

If preprocessing results do not clearly justify one horizon, support running both as explicit experiment variants:

```text
item_anchor_l3
item_anchor_l5
source_anchor_l3
source_anchor_l5
```

---

## 3. Fixed Baseline Configuration

Use the existing 4-layer rel-hm ID-GNN baseline.

Keep fixed:

- node encoder
- hidden dimensions
- 4 GNN layers
- loss
- optimizer
- learning rate schedule
- epochs / steps per epoch
- temporal sampler
- neighbor schedule / budget
- training negative-sampling policy
- random-seed convention
- evaluation K
- data split
- seed-time semantics

Any unavoidable deviation must be recorded in experiment metadata.

Before running injection experiments, reproduce the raw baseline and verify it is consistent with the existing project result within normal seed variation.

---

## 4. Query-Level Injection Input

For each query, preprocessing provides records for missing GTs:

\[
Y_q^{miss}
\]

Each record contains:

- query ID
- source ID
- seed time
- snapshot ID
- GT item ID
- whether Item-Anchor exists
- Item-Anchor ID and Katz score
- whether Source-Anchor exists
- Source-Anchor ID and Katz score

Only missing GTs are injection targets.

Already-covered GTs stay untouched.

---

## 5. Variant A: Raw Baseline

No synthetic relation is added.

Use the standard raw graph and standard scoring candidate set.

Purpose:

- regression check;
- reference performance;
- reference achievable MAP / coverage.

No preprocessing anchor is used.

---

## 6. Variant B: Direct Injection

For every missing GT \(y\), create a query-local synthetic relation row:

```text
candidate_anchor
---------------
anchor_id
src_id
candidate_id
```

Graph path:

```text
src -> candidate_anchor -> GT
```

All missing GTs can be injected under this method.

The synthetic row must not contain descriptive features that differ by label or GT identity.

If a feature tensor is mandatory:

- use an identical constant representation for all rows, or
- use only table/type embedding already supported by the model.

Do not use GT interaction time as a feature.

Do not persist the oracle row as historical data for other queries.

---

## 7. Variant C: Item-Anchor Injection

For each missing GT \(y\), read the selected historical item anchor \(i^*\) from preprocessing.

If Item-Anchor is unavailable:

- skip injection for that GT;
- do not apply fallback.

Create query-local synthetic relation row:

```text
item_anchor
-----------
anchor_id
historical_item_id
candidate_item_id
```

Graph path:

```text
src
 -> historical fact
 -> historical item i*
 -> item_anchor
 -> GT y
```

The prefix:

```text
src -> historical fact -> historical item i*
```

must already exist in historical data.

Only:

```text
historical item i* -> item_anchor -> GT
```

is synthetic.

For the first experiment, use one anchor.

If Top-3 is later enabled, create one anchor row per selected historical item.

---

## 8. Variant D: Source-Anchor Injection

For each missing GT \(y\), read the selected supporting source \(v^*\) from preprocessing.

If Source-Anchor is unavailable:

- skip injection for that GT;
- do not apply fallback.

Create query-local synthetic relation row:

```text
src_anchor
----------
anchor_id
query_src_id
supporting_src_id
```

Graph path:

```text
query src
 -> src_anchor
 -> supporting src v*
 -> historical fact
 -> GT y
```

The suffix:

```text
supporting src v* -> historical fact -> GT y
```

must already exist before the applicable snapshot cutoff.

Only:

```text
query src -> src_anchor -> supporting src
```

is synthetic.

For the first experiment, use one supporting source.

---

## 9. Candidate Scoring Mask

The experiment must prevent anchor context from accidentally changing the recommendation candidate set.

For each query, define the scoring set as:

\[
C_q^{score} =
C_q^{raw} \cup Y_q^{injected}
\]

where \(Y_q^{injected}\) contains only successfully injected missing GTs for the current method.

Nodes exposed only because of Source-Anchor or Item-Anchor context may participate in message passing but must not be newly scored as recommendation candidates unless they were already in \(C_q^{raw}\).

Implement an explicit candidate mask if required by the current ID-GNN code.

Log candidate-set size before and after injection.

---

## 10. Training Semantics

Train a separate model per graph-injection variant.

Required models:

```text
raw
direct
item_anchor
source_anchor
```

If both Katz horizons are run:

```text
raw
direct
item_anchor_l3
item_anchor_l5
source_anchor_l3
source_anchor_l5
```

Training split uses training GTs for oracle injection.

Validation split uses validation GTs for oracle injection.

Test split uses test GTs for oracle injection.

This is intentional because the experiment is an oracle diagnostic.

Do not describe these results as a deployable recommendation model.

---

## 11. Sampling and Injection Reachability

Use the existing temporal neighbor sampler and existing neighbor budget for the first controlled experiment.

However, injection paths may be dropped by sampling.

For every injected GT, log whether the complete intended path was present in the sampled computation graph.

Examples:

### Direct

```text
src -> candidate_anchor -> GT
```

### Item-Anchor

```text
src -> fact -> historical item -> item_anchor -> GT
```

### Source-Anchor

```text
src -> src_anchor -> supporting src -> fact -> GT
```

Compute:

\[
InjectionSampleRate =
\frac{
\#\text{injected GTs whose injection path is sampled}
}{
\#\text{injected GTs}
}
\]

Do not initially force a reserved sampling quota.

If sampling rates differ strongly between methods, add a later sampling-control ablation rather than silently modifying the base sampler.

---

## 12. Primary Metrics

Keep the existing recommendation metrics, including the project's main MAP metric.

Additionally log:

### 12.1 Injection coverage

\[
InjectionCoverage =
\frac{
\#\text{successfully injected missing GTs}
}{
\#\text{missing GTs}
}
\]

Direct should normally have maximal coverage.

Item / Source may have lower coverage due to no-anchor cases.

### 12.2 Achievable MAP

Compute under the actual scoring candidate set.

This indicates the upper bound caused by candidate availability.

### 12.3 Realized MAP

Standard model ranking result.

### 12.4 Ranking efficiency

Optionally report:

\[
RankingEfficiency =
\frac{RealizedMAP}{AchievableMAP}
\]

Use only when denominator handling is well-defined.

### 12.5 Injected-GT rank statistics

For successfully injected GTs:

- mean rank
- median rank
- MRR-like reciprocal rank if useful
- Hit@K / Recall@K conditional on injection

### 12.6 Sampled injection rate

Measure whether the injected relational path actually survives sampling.

---

## 13. Overall vs Conditional Evaluation

Because Direct can inject all missing GTs while Item/Source may not, report two views.

### 13.1 Overall performance

Evaluate all queries with each method's natural availability.

This answers:

> How useful is the entire injection strategy as defined?

### 13.2 Conditional placement performance

Evaluate GT-level ranking statistics only on successfully injected GTs.

This answers:

> When the method can inject a GT, how well does ID-GNN use that placement?

Also store a common-support subset:

```text
missing GT
AND item-anchor available
AND source-anchor available
```

for later pure placement comparison.

Do not replace the main evaluation with only common-support evaluation.

---

## 14. Required Per-GT Diagnostics

Write a row-level diagnostic artifact for every missing GT.

Recommended fields:

```text
query_id
split
seed_time
snapshot_id
src_id
gt_id

method
raw_covered
anchor_available
anchor_id
katz_L
katz_score
injected
injection_path_sampled

raw_candidate_count
score_candidate_count
achievable_positive

gt_score
gt_rank
gt_hit_at_k
```

For Source-Anchor also store:

```text
supporting_src_id
supporting_src_gt_interaction_verified
```

For Item-Anchor also store:

```text
historical_item_id
src_historical_item_interaction_verified
```

---

## 15. Experiment Execution Order

### Step 1. Preprocessing completion check

Do not start full model training until:

- missing GT statistics exist;
- item-anchor availability exists for L=3 and L=5;
- source-anchor availability exists for L=3 and L=5;
- temporal leakage checks pass.

### Step 2. Raw baseline reproduction

Run `raw`.

If baseline is not consistent with the existing 4-layer ID-GNN result, debug before continuing.

### Step 3. Direct

Run `direct`.

This establishes the GT-visibility / shortest-bridge baseline.

### Step 4. Item-Anchor Top-1

Run selected Katz horizon.

If L=3 vs L=5 preprocessing differences are meaningful and unresolved, run both.

### Step 5. Source-Anchor Top-1

Same L policy as Item-Anchor unless preprocessing clearly motivates different horizons.

### Step 6. Compare mechanism statistics

Before adding more variants, compare:

- injection coverage
- achievable MAP
- realized MAP
- GT rank
- sampled injection rate

### Step 7. Multi-seed confirmation

If a meaningful difference appears, rerun the core methods with multiple random seeds and report mean/std.

### Step 8. Top-3 follow-up

Only after Top-1 results:

```text
item_anchor_top3
source_anchor_top3
```

This is an ablation on amount of relational evidence, not part of the first core run.

---

## 16. Interpretation Guide

Possible outcomes:

### Direct ≈ Item ≈ Source

Interpretation:

> Once the GT is visible, relational placement contributes little. Candidate availability may dominate.

### Item / Source > Direct

Interpretation:

> Candidate availability alone is insufficient. Relational evidence helps ID-GNN convert GT coverage into ranking performance.

### Item > Source

Interpretation:

> Historical item-side context may provide stronger pair-wise evidence than collaborative source-side context for rel-hm.

### Source > Item

Interpretation:

> Supporting-user relational evidence may be more effective than historical item anchoring.

### High achievable MAP but low realized MAP for every method

Interpretation:

> Even with oracle candidate injection, a substantial ranking bottleneck remains in ID-GNN.

### Low injection sampling rate

Do not interpret low realized MAP as a topology failure until sampling is controlled.

---

## 17. Non-Goals for This Experiment

Do not include:

- learned candidate retrieval
- CF candidate quality comparison
- CBF candidate quality comparison
- negative synthetic candidate injection
- schema-preserving fake fact rows
- natural-path grafting
- learned anchor selector
- joint route-selection model

These belong to later stages after the GT-only materialization study is understood.

---

## 18. Completion Criteria

The first GT-only experiment is complete when:

1. raw baseline is reproduced;
2. Direct, Item-Anchor@1, Source-Anchor@1 are trainable and evaluable;
3. candidate scoring masks prevent accidental candidate expansion;
4. temporal leakage checks pass;
5. missing anchors use no fallback;
6. injection coverage is reported;
7. sampled injection rate is reported;
8. realized and achievable ranking metrics are reported;
9. per-GT diagnostics are saved;
10. L=3 vs L=5 preprocessing comparison is documented.
