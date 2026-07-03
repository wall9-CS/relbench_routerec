# rel-avito Feature Edge Locality

`analyze_feature_edges.py` measures how rel-avito `user-ad-visit` locality changes
when analysis-only virtual user-ad edges are added from shared features.

The baseline locality logic follows `locality_score/rel-avito/locality_score_avito_fast.py`:

- `VisitStream` is processed once in increasing seed time.
- `u2a` and `a2u` histories only contain events with `ViewDate <= seed_time`.
- Conceptual fanout is `num_neighbors // 4**i`.
- Only odd conceptual hops are scored, so with `--max_hops 3` the output has
  hop 1 and hop 3.

## Run

```powershell
python locality_score/rel-avito/feature_edge/analyze_feature_edges.py
```

Outputs are written to:

```text
locality_score/rel-avito/feature_edge/results/
```

The script writes:

- `summary.json`
- `val_feature_edge_rows.csv`
- `test_feature_edge_rows.csv`

Use `--no-save_rows` if only `summary.json` is needed.

## Default Experiments

Default feature sets:

```text
loc=ad:LocationID
cat=ad:CategoryID
loc_cat_pair=ad_tuple:LocationID,CategoryID
```

Default expressions:

```text
loc=loc
cat=cat
loc_or_cat=loc|cat
loc_and_cat=loc&cat
loc_cat_pair=loc_cat_pair
```

`loc&cat` intersects the independently sampled `loc` and `cat` candidate sets.
`loc_cat_pair` instead means exact tuple matching on `(LocationID, CategoryID)`.

## Feature Set Syntax

```text
name=source:Column
name=source_tuple:ColumnA,ColumnB
```

Supported sources:

- `ad`: static feature from `AdsInfo`.
- `ad_tuple`: exact tuple feature from `AdsInfo`.
- `user`: static feature from `UserInfo`.
- `visit`: temporal feature from `VisitStream`.

Examples:

```powershell
python locality_score/rel-avito/feature_edge/analyze_feature_edges.py `
  --feature-set loc=ad:LocationID `
  --feature-set cat=ad:CategoryID `
  --expr loc_only=loc `
  --expr cat_only=cat `
  --expr either=loc|cat `
  --expr both=loc&cat `
  --expr loc_without_cat=loc-cat
```

Expression operators:

- `|`: union
- `&`: intersection
- `-`: difference
- `^`: symmetric difference
- parentheses are allowed

When `--feature-set` or `--expr` is provided, the provided list replaces the
default list.

## Temporal Handling

Feature candidates never use future visit history. Static `AdsInfo` buckets are
filtered by `ad_last_seen`, so an ad can only be added through a feature edge if
it has appeared in historical `VisitStream` by the current seed time. Candidate
ads are sorted by most recent historical appearance and capped by
`--feature_fanout`.

