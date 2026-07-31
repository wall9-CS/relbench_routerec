import numpy as np
import pandas as pd

from examples.hm_cf.seed_time_loader import (
    group_recommendation_table_by_seed_time,
    scatter_group_predictions,
)
from relbench.base import Table


class DummyTask:
    time_col = "timestamp"
    src_entity_col = "customer_id"
    src_entity_table = "customer"
    dst_entity_col = "article_id"
    dst_entity_table = "article"
    num_dst_nodes = 5
    eval_k = 2


def test_grouping_preserves_original_row_indices_and_tables():
    table = Table(
        df=pd.DataFrame(
            {
                "timestamp": pd.to_datetime(
                    ["2020-03-09", "2020-03-02", "2020-03-09", "2020-03-02"]
                ),
                "customer_id": [10, 11, 12, 13],
                "article_id": [[1], [2], [3], [4]],
            }
        ),
        fkey_col_to_pkey_table={"customer_id": "customer", "article_id": "article"},
        time_col="timestamp",
    )
    groups = group_recommendation_table_by_seed_time(table, DummyTask())
    assert [g.seed_time for g in groups] == [
        pd.Timestamp("2020-03-02"),
        pd.Timestamp("2020-03-09"),
    ]
    assert groups[0].original_row_indices.tolist() == [1, 3]
    assert groups[0].table.df["customer_id"].tolist() == [11, 13]
    assert groups[1].original_row_indices.tolist() == [0, 2]


def test_prediction_order_scatter_with_interleaved_timestamps():
    output = np.empty((4, 2), dtype=np.int64)
    table = Table(
        df=pd.DataFrame(
            {
                "timestamp": pd.to_datetime(
                    ["2020-03-09", "2020-03-02", "2020-03-09", "2020-03-02"]
                ),
                "customer_id": [0, 1, 2, 3],
                "article_id": [[0], [1], [2], [3]],
            }
        ),
        fkey_col_to_pkey_table={"customer_id": "customer", "article_id": "article"},
        time_col="timestamp",
    )
    groups = group_recommendation_table_by_seed_time(table, DummyTask())
    scatter_group_predictions(output, groups[0], np.array([[20, 21], [30, 31]]))
    scatter_group_predictions(output, groups[1], np.array([[0, 1], [2, 3]]))
    assert output.tolist() == [[0, 1], [20, 21], [2, 3], [30, 31]]
