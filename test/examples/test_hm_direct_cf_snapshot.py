import pandas as pd

from examples.hm_direct_cf.config import HMDirectCFSnapshotConfig
from examples.hm_direct_cf.snapshot import build_snapshot


def _interactions():
    return pd.DataFrame(
        {
            "customer_id": pd.Series([0, 0, 0, 0, 1], dtype="int64"),
            "article_id": pd.Series([0, 2, 0, 3, 0], dtype="int64"),
            "t_dat": pd.to_datetime(
                [
                    "2020-02-28",
                    "2020-03-02",
                    "2020-02-28",
                    "2020-03-03",
                    "2020-02-28",
                ]
            ),
        }
    )


def _item_cf(seed="2020-03-02"):
    return pd.DataFrame(
        {
            "seed_time": pd.to_datetime([seed, seed, seed, seed]),
            "src_article_id": pd.Series([0, 2, 2, 3], dtype="int64"),
            "dst_article_id": pd.Series([5, 5, 4, 1], dtype="int64"),
            "support": pd.Series([3, 2, 4, 10], dtype="int64"),
            "cf_score": pd.Series([0.5, 0.25, 0.7, 1.0], dtype="float32"),
            "rank": pd.Series([1, 1, 2, 1], dtype="int32"),
        }
    )


def test_build_direct_snapshot_collapses_historical_item_cf_candidates():
    cfg = HMDirectCFSnapshotConfig(direct_top_k=2)
    df, stats = build_snapshot(
        _interactions(),
        _item_cf(),
        pd.Timestamp("2020-03-02"),
        [0],
        cfg,
    )

    assert df["dst_article_id"].tolist() == [5, 4]
    assert df["rank"].tolist() == [1, 2]
    top = df.iloc[0]
    assert top["direct_score"] == 0.75
    assert top["num_source_articles"] == 2
    assert top["support_sum"] == 5
    assert top["best_src_article_id"] == 0
    assert 1 not in df["dst_article_id"].tolist()
    assert stats.num_joined_rows == 3
    assert stats.num_collapsed_pairs_before_topk == 2


def test_direct_top_k_and_seen_filter_are_applied():
    cf = _item_cf()
    cf.loc[len(cf)] = [pd.Timestamp("2020-03-02"), 0, 2, 9, 0.9, 2]
    cfg = HMDirectCFSnapshotConfig(direct_top_k=1, filter_seen_dst=True)
    df, _ = build_snapshot(
        _interactions(),
        cf,
        pd.Timestamp("2020-03-02"),
        [0],
        cfg,
    )

    assert len(df) == 1
    assert df["dst_article_id"].tolist() == [5]


def test_source_scope_without_history_yields_empty_schema():
    cfg = HMDirectCFSnapshotConfig()
    df, stats = build_snapshot(
        _interactions(),
        _item_cf(),
        pd.Timestamp("2020-03-02"),
        [2],
        cfg,
    )

    assert len(df) == 0
    assert list(df.columns) == [
        "seed_time",
        "src_customer_id",
        "dst_article_id",
        "direct_score",
        "max_cf_score",
        "support_sum",
        "support_max",
        "num_source_articles",
        "best_src_article_id",
        "best_item_cf_rank",
        "rank",
    ]
    assert stats.num_source_nodes == 1
