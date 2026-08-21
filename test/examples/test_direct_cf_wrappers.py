import pandas as pd

from examples.avito_direct_cf.config import AvitoDirectCFSnapshotConfig
from examples.avito_direct_cf.snapshot import build_snapshot as build_avito_snapshot
from examples.stack_direct_cf.config import StackDirectCFSnapshotConfig
from examples.stack_direct_cf.snapshot import build_snapshot as build_stack_snapshot


def test_stack_direct_snapshot_uses_stack_column_names():
    interactions = pd.DataFrame(
        {
            "UserId": pd.Series([0, 0], dtype="int64"),
            "PostId": pd.Series([1, 2], dtype="int64"),
            "CreationDate": pd.to_datetime(["2020-01-01", "2020-01-02"]),
        }
    )
    item_cf = pd.DataFrame(
        {
            "seed_time": pd.to_datetime(["2020-01-03", "2020-01-03"]),
            "src_PostId": pd.Series([1, 2], dtype="int64"),
            "dst_PostId": pd.Series([3, 3], dtype="int64"),
            "support": pd.Series([3, 5], dtype="int64"),
            "cf_score": pd.Series([0.4, 0.6], dtype="float32"),
            "rank": pd.Series([1, 1], dtype="int32"),
        }
    )

    df, _ = build_stack_snapshot(
        interactions,
        item_cf,
        pd.Timestamp("2020-01-03"),
        [0],
        StackDirectCFSnapshotConfig(),
    )

    assert list(df[["src_UserId", "dst_PostId", "num_source_posts"]].iloc[0]) == [
        0,
        3,
        2,
    ]
    assert df["direct_score"].iloc[0] == 1.0


def test_avito_direct_snapshot_uses_avito_column_names():
    interactions = pd.DataFrame(
        {
            "UserID": pd.Series([1], dtype="int64"),
            "AdID": pd.Series([2], dtype="int64"),
            "ViewDate": pd.to_datetime(["2015-05-08"]),
        }
    )
    item_cf = pd.DataFrame(
        {
            "seed_time": pd.to_datetime(["2015-05-09"]),
            "src_AdID": pd.Series([2], dtype="int64"),
            "dst_AdID": pd.Series([4], dtype="int64"),
            "support": pd.Series([3], dtype="int64"),
            "cf_score": pd.Series([0.8], dtype="float32"),
            "rank": pd.Series([1], dtype="int32"),
        }
    )

    df, _ = build_avito_snapshot(
        interactions,
        item_cf,
        pd.Timestamp("2015-05-09"),
        [1],
        AvitoDirectCFSnapshotConfig(),
    )

    assert list(df[["src_UserID", "dst_AdID", "best_src_AdID"]].iloc[0]) == [1, 4, 2]
