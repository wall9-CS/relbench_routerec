import pandas as pd
import pytest

from examples.hm_user_cf.config import UserCFSnapshotConfig, snapshot_filename
from examples.hm_user_cf.io import (
    initialize_or_load_manifest,
    load_snapshot,
    validate_manifest_for_training,
    validate_snapshot_frame,
    write_snapshot_atomic,
)


def _snapshot(seed="2020-03-02"):
    return pd.DataFrame(
        {
            "seed_time": pd.to_datetime([seed, seed]),
            "src_customer_id": pd.Series([0, 0], dtype="int64"),
            "dst_customer_id": pd.Series([1, 2], dtype="int64"),
            "overlap": pd.Series([2, 2], dtype="int64"),
            "user_cf_score": pd.Series([1.0, 0.5], dtype="float32"),
            "src_history_size": pd.Series([2, 2], dtype="int64"),
            "dst_history_size": pd.Series([2, 1], dtype="int64"),
            "rank": pd.Series([1, 2], dtype="int32"),
        }
    )


def test_snapshot_filename():
    assert snapshot_filename(pd.Timestamp("2020-03-02")) == "user_cf_snapshot_2020-03-02.parquet"
    assert (
        snapshot_filename(pd.Timestamp("2020-03-02 01:02:03"))
        == "user_cf_snapshot_2020-03-02T01-02-03.parquet"
    )


def test_write_load_and_manifest_round_trip(tmp_path):
    cfg = UserCFSnapshotConfig(min_overlap=1, top_k=2)
    snapshot_dir = cfg.snapshot_dir(tmp_path)
    manifest = initialize_or_load_manifest(snapshot_dir, cfg)
    assert manifest["cf_kind"] == "user_user_history_overlap"

    path = write_snapshot_atomic(
        _snapshot(),
        snapshot_dir,
        pd.Timestamp("2020-03-02"),
        config=cfg,
        num_customers=3,
    )
    assert path.exists()
    loaded = load_snapshot(
        snapshot_dir,
        pd.Timestamp("2020-03-02"),
        config=cfg,
        num_customers=3,
    )
    assert loaded["user_cf_score"].dtype == "float32"
    assert validate_manifest_for_training(snapshot_dir) == cfg


def test_manifest_incompatibility(tmp_path):
    cfg = UserCFSnapshotConfig()
    snapshot_dir = cfg.snapshot_dir(tmp_path)
    initialize_or_load_manifest(snapshot_dir, cfg)
    with pytest.raises(ValueError, match="min_overlap"):
        initialize_or_load_manifest(snapshot_dir, UserCFSnapshotConfig(min_overlap=3))


def test_snapshot_validation_rejects_bad_rows():
    cfg = UserCFSnapshotConfig(min_overlap=2, top_k=2)
    bad = _snapshot()
    bad.loc[1, "rank"] = 3
    with pytest.raises(ValueError, match="consecutive"):
        validate_snapshot_frame(bad, pd.Timestamp("2020-03-02"), config=cfg)

    bad = _snapshot()
    bad.loc[1, "dst_customer_id"] = 0
    with pytest.raises(ValueError, match="self-pairs"):
        validate_snapshot_frame(bad, pd.Timestamp("2020-03-02"), config=cfg)

    bad = _snapshot()
    bad.loc[1, "overlap"] = 1
    with pytest.raises(ValueError, match="min_overlap"):
        validate_snapshot_frame(bad, pd.Timestamp("2020-03-02"), config=cfg)
