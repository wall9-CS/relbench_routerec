# Specification: Rel-HM Same Product Code Graph Table

## Goal

Add a new relational table named `same_product_code` for the `rel-hm` dataset and use it when training the `user-item-purchase` recommendation task with `examples/idgnn_recommendation.py` at `--num_layers 4`.

The table should connect H&M articles that share the same `product_code`, allowing ID-GNN message passing between variants of the same product.

## Current State

- The `rel-hm` database is built in `relbench/datasets/hm.py`.
- It currently contains:
  - `article`, primary key `article_id`
  - `customer`, primary key `customer_id`
  - `transactions`, foreign keys `customer_id -> customer` and `article_id -> article`, time column `t_dat`
- The `user-item-purchase` task in `relbench/tasks/hm.py` predicts future purchased `article_id` values for each `customer_id`.
- `examples/idgnn_recommendation.py` builds its PyG graph with `make_pkey_fkey_graph()`, which automatically creates graph edges for every foreign key declared on every table.

## Required Table

Create a table named `same_product_code`.

Schema:

| Column | Type | Meaning |
| --- | --- | --- |
| `article_id_left` | foreign key to `article.article_id` | One article in a same-product pair |
| `article_id_right` | foreign key to `article.article_id` | Another article with the same `product_code` |

Do not use duplicate column names like two separate `article_id` columns. Pandas and `Table.fkey_col_to_pkey_table` require unique column names.

Table metadata:

```python
Table(
    df=same_product_code_df,
    fkey_col_to_pkey_table={
        "article_id_left": "article",
        "article_id_right": "article",
    },
    pkey_col=None,
    time_col=None,
)
```

## Pair Construction

Build `same_product_code_df` from the `article` table before database reindexing.

For each `product_code` group:

1. Collect all `article_id` values with that `product_code`.
2. Drop groups with fewer than two articles.
3. Emit unordered article pairs only:
   - include `(a, b)` where `a < b`
   - do not include `(b, a)`
   - do not include self-pairs `(a, a)`

This creates one relationship row per article pair. `make_pkey_fkey_graph()` already adds reverse graph edges for each foreign key, so storing both directions in the table is unnecessary.

Expected graph effect:

- `same_product_code` becomes a relation-node table.
- Each row connects to two `article` nodes.
- With bidirectional sampling, an article can receive messages from another article with the same `product_code` through:

```text
article -> same_product_code -> article
```

## Implementation Location

Modify `relbench/datasets/hm.py`.

Recommended helper:

```python
def _make_same_product_code_df(articles_df: pd.DataFrame) -> pd.DataFrame:
    ...
```

Then add `same_product_code` to the `Database(table_dict={...})` returned by `HMDataset.make_db()`.

Important: create the table using the original raw `article_id` values. `Dataset.get_db()` calls `db.reindex_pkeys_and_fkeys()` after `make_db()`, and that method will reindex both `article_id_left` and `article_id_right` into consecutive article node IDs.

## Cache Handling

The default example uses:

```python
get_dataset(args.dataset, download=True)
```

If a prepared RelBench cache already exists, `HMDataset.make_db()` may not run, and the cached database may not contain `same_product_code`.

The implementation must make the new table visible in one of these ways:

1. Preferred: override or extend `HMDataset.get_db()` so it augments both freshly built and cached `rel-hm` databases with `same_product_code` when the table is missing.
2. Acceptable for one-off experiments: delete the cached `rel-hm/db` directory and run with `download=False` after placing the raw Kaggle H&M files under `data/hm-recommendation`.

The preferred approach is more robust because it works with the existing `idgnn_recommendation.py` default behavior.

Also handle the example graph cache:

- `examples/idgnn_recommendation.py` caches stype proposals in:

```text
{cache_dir}/{dataset}/stypes.json
```

- If that file was created before `same_product_code` existed, it will not contain a `same_product_code` entry and graph construction can fail.
- Update the example to regenerate or repair `stypes.json` when current database tables are missing from the cache.
- Alternatively, run with a fresh `--cache_dir`.

## Run Command

After implementation, run:

```powershell
venv\Scripts\python.exe examples\idgnn_recommendation.py --dataset rel-hm --task user-item-purchase --num_layers 4
```

The argparse option is named `--num_layers`, not `--num_layer`.

## Validation

Minimum checks:

1. `rel-hm` database contains `same_product_code`.
2. `same_product_code` has exactly two columns: `article_id_left`, `article_id_right`.
3. Both columns are registered as foreign keys to `article`.
4. No self-pairs exist.
5. No duplicate reversed pairs exist.
6. `make_pkey_fkey_graph()` produces edge types for both same-product foreign keys:

```text
("same_product_code", "f2p_article_id_left", "article")
("article", "rev_f2p_article_id_left", "same_product_code")
("same_product_code", "f2p_article_id_right", "article")
("article", "rev_f2p_article_id_right", "same_product_code")
```

7. The ID-GNN recommendation script starts successfully with `--num_layers 4`.

## Notes

- Do not change the `user-item-purchase` task target definition.
- Do not remove `product_code` from the `article` table.
- Do not add timestamps to `same_product_code`; it is a static product relationship table.
- Avoid constructing a fully directed pair table because it doubles rows without adding graph connectivity under the current bidirectional graph construction.
