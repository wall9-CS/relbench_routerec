
Implement the task defined in `spec.md`.

Required workflow:

1. Read `spec.md` completely before editing.
2. Inspect the current checkout, especially:
   - `examples/idgnn_recommendation.py`
   - `examples/model.py`
   - `relbench/modeling/graph.py`
   - `relbench/datasets/hm.py`
   - existing tests under `test/`
   - the installed PyTorch Frame API for TensorFrame row indexing and concatenation
3. Implement the smallest maintainable change that satisfies the specification.
4. Keep the feature test-only. Training and validation must use the unmodified base graph.
5. Do not add a new edge type and do not reuse one transaction node for multiple articles.
6. Clone complete materialized transaction TensorFrame rows and transaction times for synthetic nodes.
7. Do not rematerialize the database or recompute feature/category statistics.
8. Do not use test destination labels or transactions after the test seed time.
9. Do not mutate the base `HeteroData`.
10. Add focused unit tests that use only tiny in-memory fixtures and do not download rel-hm.
11. Run the verification commands from `spec.md`.
12. Fix failures caused by your changes. Do not weaken tests to make them pass.
13. Preserve all unrelated local modifications in the checkout.

Implementation targets:

- Add `examples/hm_product_code_expansion.py`.
- Modify `examples/idgnn_recommendation.py`.
- Add `test/examples/test_hm_product_code_expansion.py`.
- Avoid changing core RelBench files unless strictly necessary.

Before finishing, review the diff against every acceptance criterion in `spec.md`.

Your final response must contain:

- a concise summary of the implementation;
- the exact files changed;
- tests and commands run with pass/fail results;
- whether a rel-hm smoke test was run;
- any deviations from `spec.md`;
- remaining risks or performance concerns.

Do not stop after proposing code. Make the edits and run the tests.
