import pandas as pd
import pytest

from examples.hm_cf.config import CFSnapshotConfig
from examples.hm_cf.coverage import (
    combine_cf_coverage_metrics,
    compute_cf_coverage_for_table,
)
from examples.hm_cf.io import write_snapshot_atomic
from relbench.base import Table


class DummyTask:
    time_col = "timestamp"
    src_entity_col = "customer_id"
    src_entity_table = "customer"
    dst_entity_col = "article_id"
    dst_entity_table = "article"
    eval_k = 2


def _transactions(rows):
    return pd.DataFrame(rows, columns=["customer_id", "article_id", "t_dat"]).astype(
        {"customer_id": "int64", "article_id": "int64"}
    ).assign(t_dat=lambda df: pd.to_datetime(df["t_dat"]))


def _table(rows):
    return Table(
        df=pd.DataFrame(rows, columns=["timestamp", "customer_id", "article_id"]).assign(
            timestamp=lambda df: pd.to_datetime(df["timestamp"])
        ),
        fkey_col_to_pkey_table={"customer_id": "customer", "article_id": "article"},
        time_col="timestamp",
    )


def _snapshot(rows, seed_time):
    if not rows:
        return pd.DataFrame(
            {
                "seed_time": pd.Series([], dtype="datetime64[ns]"),
                "src_article_id": pd.Series([], dtype="int64"),
                "dst_article_id": pd.Series([], dtype="int64"),
                "support": pd.Series([], dtype="int64"),
                "cf_score": pd.Series([], dtype="float32"),
                "rank": pd.Series([], dtype="int32"),
            }
        )
    return pd.DataFrame(
        rows,
        columns=["src_article_id", "dst_article_id", "support", "cf_score", "rank"],
    ).assign(seed_time=pd.Timestamp(seed_time))[
        ["seed_time", "src_article_id", "dst_article_id", "support", "cf_score", "rank"]
    ].astype(
        {
            "src_article_id": "int64",
            "dst_article_id": "int64",
            "support": "int64",
            "cf_score": "float32",
            "rank": "int32",
        }
    )


def test_partial_cf_coverage_metrics(tmp_path):
    cfg = CFSnapshotConfig(min_support=1, top_l=10)
    seed = pd.Timestamp("2020-03-02")
    write_snapshot_atomic(
        _snapshot(
            [
                (0, 1, 2, 0.9, 1),
                (0, 4, 1, 0.7, 2),
                (2, 3, 2, 0.8, 1),
            ],
            seed,
        ),
        tmp_path,
        seed,
        config=cfg,
        num_articles=5,
    )

    metrics = compute_cf_coverage_for_table(
        _table(
            [
                ("2020-03-02", 0, [1, 2]),
                ("2020-03-02", 1, [3]),
                ("2020-03-02", 2, [4]),
            ]
        ),
        DummyTask(),
        _transactions(
            [
                (0, 0, "2020-03-01"),
                (1, 2, "2020-03-01"),
            ]
        ),
        tmp_path,
        config=cfg,
        num_articles=5,
    )

    assert metrics.num_rows == 3
    assert metrics.num_groundtruth_labels == 4
    assert metrics.num_covered_groundtruth_labels == 2
    assert metrics.coverage_rate == 0.5
    assert metrics.hit_rate == pytest.approx(2 / 3)
    assert metrics.achievable_map == pytest.approx(0.5)
    assert metrics.coverage_gap == pytest.approx(0.5)

    doubled = combine_cf_coverage_metrics([metrics, metrics])
    assert doubled.num_rows == 6
    assert doubled.coverage_rate == 0.5
    assert doubled.achievable_map == pytest.approx(0.5)


def test_uses_exact_seed_time_snapshot(tmp_path):
    cfg = CFSnapshotConfig(min_support=1, top_l=10)
    seed1 = pd.Timestamp("2020-03-02")
    seed2 = pd.Timestamp("2020-03-09")
    write_snapshot_atomic(
        _snapshot([(0, 1, 2, 0.9, 1)], seed1),
        tmp_path,
        seed1,
        config=cfg,
        num_articles=3,
    )
    write_snapshot_atomic(
        _snapshot([(0, 2, 2, 0.9, 1)], seed2),
        tmp_path,
        seed2,
        config=cfg,
        num_articles=3,
    )

    metrics = compute_cf_coverage_for_table(
        _table(
            [
                ("2020-03-09", 0, [2]),
                ("2020-03-02", 0, [1]),
            ]
        ),
        DummyTask(),
        _transactions([(0, 0, "2020-03-01")]),
        tmp_path,
        config=cfg,
        num_articles=3,
    )

    assert metrics.coverage_rate == 1.0
    assert metrics.achievable_map == 1.0


def test_source_articles_are_optional_candidates(tmp_path):
    cfg = CFSnapshotConfig(min_support=1, top_l=10)
    seed = pd.Timestamp("2020-03-02")
    write_snapshot_atomic(
        _snapshot([], seed),
        tmp_path,
        seed,
        config=cfg,
        num_articles=3,
    )

    table = _table([("2020-03-02", 0, [1])])
    transactions = _transactions([(0, 1, "2020-03-01")])
    default_metrics = compute_cf_coverage_for_table(
        table,
        DummyTask(),
        transactions,
        tmp_path,
        config=cfg,
        num_articles=3,
    )
    source_metrics = compute_cf_coverage_for_table(
        table,
        DummyTask(),
        transactions,
        tmp_path,
        config=cfg,
        num_articles=3,
        include_source_articles=True,
    )

    assert default_metrics.coverage_rate == 0.0
    assert source_metrics.coverage_rate == 1.0


def test_num_layers_guard(tmp_path):
    cfg = CFSnapshotConfig(min_support=1, top_l=10)
    seed = pd.Timestamp("2020-03-02")
    write_snapshot_atomic(
        _snapshot([], seed),
        tmp_path,
        seed,
        config=cfg,
        num_articles=3,
    )
    with pytest.raises(ValueError, match="num_layers"):
        compute_cf_coverage_for_table(
            _table([("2020-03-02", 0, [1])]),
            DummyTask(),
            _transactions([(0, 1, "2020-03-01")]),
            tmp_path,
            config=cfg,
            num_articles=3,
            num_layers=3,
        )
