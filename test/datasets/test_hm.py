import pandas as pd
from torch_frame import stype

from relbench.base import Database, Table
from relbench.datasets.hm import _same_product_code_table
from relbench.modeling.graph import make_pkey_fkey_graph


def test_same_product_code_table_reindexes_article_pairs():
    articles_df = pd.DataFrame(
        {
            "article_id": [100, 300, 200, 400],
            "product_code": [10, 10, 10, 20],
        }
    )
    db = Database(
        table_dict={
            "article": Table(
                df=articles_df,
                fkey_col_to_pkey_table={},
                pkey_col="article_id",
            ),
            "same_product_code": _same_product_code_table(articles_df),
        }
    )

    db.reindex_pkeys_and_fkeys()

    same_product_code = db.table_dict["same_product_code"]
    assert list(same_product_code.df.columns) == [
        "article_id_left",
        "article_id_right",
    ]
    assert same_product_code.fkey_col_to_pkey_table == {
        "article_id_left": "article",
        "article_id_right": "article",
    }
    assert same_product_code.df.values.tolist() == [
        [0, 2],
        [0, 1],
        [2, 1],
    ]
    assert (
        same_product_code.df["article_id_left"]
        != same_product_code.df["article_id_right"]
    ).all()

    data, _ = make_pkey_fkey_graph(
        db,
        col_to_stype_dict={
            "article": {
                "article_id": stype.numerical,
                "product_code": stype.numerical,
            },
            "same_product_code": {
                "article_id_left": stype.numerical,
                "article_id_right": stype.numerical,
            },
        },
    )
    assert (
        "same_product_code",
        "f2p_article_id_left",
        "article",
    ) in data.edge_types
    assert (
        "article",
        "rev_f2p_article_id_left",
        "same_product_code",
    ) in data.edge_types
    assert (
        "same_product_code",
        "f2p_article_id_right",
        "article",
    ) in data.edge_types
    assert (
        "article",
        "rev_f2p_article_id_right",
        "same_product_code",
    ) in data.edge_types
