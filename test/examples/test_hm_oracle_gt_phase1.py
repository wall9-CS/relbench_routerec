from __future__ import annotations

import pandas as pd
import pytest
from scipy import sparse

from examples.hm_oracle_gt.config import OracleGTPhase1Config
from examples.hm_oracle_gt.history import build_historical_interaction_matrix
from examples.hm_oracle_gt.preprocess import build_query_gt_table, summarize_split
from examples.hm_oracle_gt.raw_support import QUERY_ID_COL
from relbench.base import Table


class _Task:
    src_entity_col = "customer_id"
    src_entity_table = "customer"
    dst_entity_col = "article_id"
    dst_entity_table = "article"
    time_col = "timestamp"
    eval_k = 12


def test_config_rejects_non_four_layer_baseline():
    with pytest.raises(ValueError, match="4-layer"):
        OracleGTPhase1Config(num_layers=2)


def test_history_matrix_uses_existing_cf_boundary_and_binary_pairs():
    transactions = pd.DataFrame(
        {
            "customer_id": pd.Series([0, 0, 0, 1, 2], dtype="int64"),
            "article_id": pd.Series([1, 1, 2, 3, 4], dtype="int64"),
            "t_dat": pd.to_datetime(
                [
                    "2020-01-01",
                    "2020-01-02",
                    "2020-01-03",
                    "2020-01-04",
                    "2020-01-10",
                ]
            ),
        }
    )
    config = OracleGTPhase1Config(window_weeks=1)

    matrix, active_users, stats = build_historical_interaction_matrix(
        transactions,
        pd.Timestamp("2020-01-08"),
        num_users=3,
        num_items=5,
        config=config,
    )

    assert sparse.isspmatrix_csr(matrix)
    assert matrix.shape == (2, 5)
    assert active_users.tolist() == [0, 1]
    assert matrix.nnz == 3
    assert matrix[0, 1] == 1
    assert matrix[0, 2] == 1
    assert matrix[1, 3] == 1
    assert stats.num_raw_transactions == 3
    assert stats.num_historical_interactions == 3
    assert stats.temporal_boundary_ok
    assert stats.max_interaction_time == "2020-01-04T00:00:00"


def test_query_gt_table_marks_only_raw_covered_gt():
    table = Table(
        df=pd.DataFrame(
            {
                QUERY_ID_COL: pd.Series([0, 1], dtype="int64"),
                "timestamp": pd.to_datetime(["2020-01-08", "2020-01-08"]),
                "customer_id": pd.Series([10, 11], dtype="int64"),
                "article_id": [[1, 2, 3], [4]],
            }
        ),
        fkey_col_to_pkey_table={"customer_id": "customer", "article_id": "article"},
        pkey_col=None,
        time_col="timestamp",
    )
    raw_support = pd.DataFrame(
        {
            QUERY_ID_COL: [0, 1],
            "split": ["val", "val"],
            "seed_time": pd.to_datetime(["2020-01-08", "2020-01-08"]),
            "src_id": [10, 11],
            "raw_candidate_ids": [[2, 9], [5]],
            "raw_candidate_count": [2, 1],
        }
    )

    gt = build_query_gt_table(table, raw_support, _Task(), "val")

    assert gt[["query_id", "gt_id", "raw_covered"]].to_dict("records") == [
        {"query_id": 0, "gt_id": 1, "raw_covered": False},
        {"query_id": 0, "gt_id": 2, "raw_covered": True},
        {"query_id": 0, "gt_id": 3, "raw_covered": False},
        {"query_id": 1, "gt_id": 4, "raw_covered": False},
    ]
    summary = summarize_split("val", table, gt, [], _Task())
    assert summary["num_queries"] == 2
    assert summary["total_gt_count"] == 4
    assert summary["raw_covered_gt_count"] == 1
    assert summary["missing_gt_count"] == 3
    assert summary["max_missing_gt_per_query"] == 2
