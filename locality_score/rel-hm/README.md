# rel-hm user-item-purchase locality score

This folder measures locality for `rel-hm/user-item-purchase`.

Run:

```bash
python locality_score/rel-hm/locality_score_user_item_purchase.py --download
```

Useful options:

```bash
python locality_score/rel-hm/locality_score_user_item_purchase.py \
  --num_neighbors 128 \
  --max_hops 3 \
  --no-save_rows \
  --out_dir locality_score/rel-hm/results
```

Definitions:

- `hop_1`: articles the customer purchased before the prediction timestamp.
- `hop_3`: articles purchased before the prediction timestamp by customers who also purchased the customer's hop-1 articles.
- Conceptual hop `i` samples at most `num_neighbors // 2**(i-1)` recent
  neighbors per frontier node, matching IDGNN's layer-indexed fanout schedule.
- `locality_score`: `reachable groundtruth articles / all groundtruth articles`.
- `neighbor_groundtruth_ratio`: `reachable groundtruth articles / all reachable articles`.

Speed notes:

- The script precomputes recent temporal adjacency per evaluation timestamp.
- Repeated `(customer_id, timestamp)` reachable sets are memoized.
- Use `--no-save_rows` to skip row-level CSV output and write only `summary.json`.

Outputs:

- `summary.json`: micro/macro scores for validation and test.
- `val_locality_rows.csv`: row-level validation scores.
- `test_locality_rows.csv`: row-level test scores.
