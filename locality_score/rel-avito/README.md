# rel-avito user-ad-visit locality score

This folder measures locality for `rel-avito/user-ad-visit`.

Run:

```bash
python locality_score/rel-avito/locality_score_user_ad_visit.py --download
```

Useful options:

```bash
python locality_score/rel-avito/locality_score_user_ad_visit.py \
  --num_neighbors 128 \
  --max_hops 3 \
  --out_dir locality_score/rel-avito/results
```

Definitions:

- `hop_1`: ads the user visited before the prediction timestamp.
- `hop_3`: ads visited before the prediction timestamp by users who also visited the user's hop-1 ads.
- Avito's physical schema path is `UserInfo - VisitStream - AdsInfo`, but this script treats a historical user-ad visit as one conceptual hop for locality measurement.
- Conceptual hop `i` samples at most `num_neighbors // 2**(i-1)` recent
  neighbors per frontier node, matching IDGNN's layer-indexed fanout schedule.
- `locality_score`: `reachable groundtruth ads / all groundtruth ads`.
- `neighbor_groundtruth_ratio`: `reachable groundtruth ads / all reachable ads`.

Outputs:

- `summary.json`: micro/macro scores for validation and test.
- `val_locality_rows.csv`: row-level validation scores.
- `test_locality_rows.csv`: row-level test scores.
