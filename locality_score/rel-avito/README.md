# rel-avito user-ad-visit locality score

This folder measures locality for `rel-avito/user-ad-visit`.

Run:

```bash
python localty_score/rel-avito/locality_score_user_ad_visit.py --download
```

Useful options:

```bash
python localty_score/rel-avito/locality_score_user_ad_visit.py \
  --num_neighbors 128 \
  --max_hops 3 \
  --out_dir localty_score/rel-avito/results
```

Definitions:

- `hop_1`: ads the user visited before the prediction timestamp.
- `hop_3`: ads visited before the prediction timestamp by users who also visited the user's hop-1 ads.
- Avito's physical schema path is `UserInfo - VisitStream - AdsInfo`, but this script treats a historical user-ad visit as one conceptual hop for locality measurement.
- Conceptual hop `i` adds at most `num_neighbors // 2**(i-1)` globally recent
  destination ads, matching IDGNN's layer-indexed fanout schedule.
- Reachable counts are cumulative over odd destination hops, so with
  `num_neighbors=128` and `max_hops=3`, `hop_3_reachable_ads` is at most
  `128 + 32 = 160`.
- `locality_score`: `reachable groundtruth ads / all groundtruth ads`.
- `neighbor_groundtruth_ratio`: `reachable groundtruth ads / all reachable ads`.

Outputs:

- `summary.json`: micro/macro scores for validation and test.
- `val_locality_rows.csv`: row-level validation scores.
- `test_locality_rows.csv`: row-level test scores.
