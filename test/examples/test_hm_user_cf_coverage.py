import numpy as np
import pandas as pd

from relbench.base import Table
from examples.hm_user_cf.config import UserCFSnapshotConfig
from examples.hm_user_cf.coverage import (
    combine_user_cf_coverage_metrics,
    compute_user_cf_coverage_for_table,
)
from examples.hm_user_cf.io import initialize_or_load_manifest, write_snapshot_atomic


class DummyTask:
    src_entity_col = "customer_id"
    src_entity_table = "customer"
    dst_entity_col = "article_id"
    dst_entity_table = "article"
    time_col = "timestamp"
    eval_k = 2


def _table(rows):
    return Table(
        df=pd.DataFrame(rows, columns=["timestamp", "customer_id", "article_id"]).assign(
            timestamp=lambda df: pd.to_datetime(df["timestamp"])
        ),
        fkey_col_to_pkey_table={"customer_id": "customer", "article_id": "article"},
        pkey_col=None,
        time_col="timestamp",
    )


def _tx(rows):
    return pd.DataFrame(rows, columns=["customer_id", "article_id", "t_dat"]).astype(
        {"customer_id": "int64", "article_id": "int64"}
    ).assign(t_dat=lambda df: pd.to_datetime(df["t_dat"]))


def _snapshot(seed="2020-03-02"):
    return pd.DataFrame(
        {
            "seed_time": pd.to_datetime([seed, seed]),
            "src_customer_id": pd.Series([0, 0], dtype="int64"),
            "dst_customer_id": pd.Series([1, 2], dtype="int64"),
            "overlap": pd.Series([2, 2], dtype="int64"),
            "user_cf_score": pd.Series([1.0, 0.8], dtype="float32"),
            "src_history_size": pd.Series([2, 2], dtype="int64"),
            "dst_history_size": pd.Series([2, 2], dtype="int64"),
            "rank": pd.Series([1, 2], dtype="int32"),
        }
    )


def test_user_cf_coverage_metrics(tmp_path):
    cfg = UserCFSnapshotConfig(min_overlap=1, top_k=2)
    snapshot_dir = cfg.snapshot_dir(tmp_path)
    initialize_or_load_manifest(snapshot_dir, cfg)
    write_snapshot_atomic(
        _snapshot(),
        snapshot_dir,
        pd.Timestamp("2020-03-02"),
        config=cfg,
        num_customers=3,
    )
    table = _table(
        [
            ("2020-03-02", 0, [3, 4]),
            ("2020-03-02", 1, [4]),
        ]
    )
    transactions = _tx(
        [
            (1, 3, "2020-02-01"),
            (2, 4, "2020-02-02"),
            (2, 5, "2020-03-03"),
        ]
    )
    metrics = compute_user_cf_coverage_for_table(
        table,
        DummyTask(),
        transactions,
        snapshot_dir,
        config=cfg,
        num_customers=3,
    )
    assert metrics.num_rows == 2
    assert metrics.num_positive_rows == 2
    assert metrics.num_rows_with_neighbors == 1
    assert metrics.num_rows_with_candidates == 1
    assert metrics.num_rows_with_hit == 1
    assert metrics.num_groundtruth_labels == 3
    assert metrics.num_covered_groundtruth_labels == 2
    assert metrics.coverage_rate == 2 / 3
    assert metrics.hit_rate == 1 / 2
    assert metrics.achievable_map == 0.5


def test_combine_user_cf_coverage_metrics(tmp_path):
    cfg = UserCFSnapshotConfig(min_overlap=1, top_k=2)
    snapshot_dir = cfg.snapshot_dir(tmp_path)
    initialize_or_load_manifest(snapshot_dir, cfg)
    write_snapshot_atomic(
        _snapshot(),
        snapshot_dir,
        pd.Timestamp("2020-03-02"),
        config=cfg,
        num_customers=3,
    )
    metrics = compute_user_cf_coverage_for_table(
        _table([("2020-03-02", 0, [3])]),
        DummyTask(),
        _tx([(1, 3, "2020-02-01")]),
        snapshot_dir,
        config=cfg,
        num_customers=3,
    )
    doubled = combine_user_cf_coverage_metrics([metrics, metrics])
    assert doubled.num_rows == metrics.num_rows * 2
    assert np.isclose(doubled.coverage_rate, metrics.coverage_rate)

