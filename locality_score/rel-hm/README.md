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
  --out_dir locality_score/rel-hm/results
```

Definitions:

- `hop_1`: articles the customer purchased before the prediction timestamp.
- `hop_3`: articles purchased before the prediction timestamp by customers who also purchased the customer's hop-1 articles.
- Conceptual hop `i` adds at most `num_neighbors // 2**(i-1)` globally recent
  destination articles, matching IDGNN's layer-indexed fanout schedule.
- Reachable counts are cumulative over odd destination hops, so with
  `num_neighbors=128` and `max_hops=3`, `hop_3_reachable_articles` is at most
  `128 + 32 = 160`.
- `locality_score`: `reachable groundtruth articles / all groundtruth articles`.
- `neighbor_groundtruth_ratio`: `reachable groundtruth articles / all reachable articles`.

Outputs:

- `summary.json`: micro/macro scores for validation and test.
- `val_locality_rows.csv`: row-level validation scores.
- `test_locality_rows.csv`: row-level test scores.
