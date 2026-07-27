import json

import pandas as pd
import pytest

from examples.hm_cf.config import CFSnapshotConfig, snapshot_filename, timestamp_key
from examples.hm_cf.io import (
    initialize_or_load_manifest,
    load_manifest,
    load_snapshot,
    manifest_key,
    validate_manifest_for_training,
    write_manifest_atomic,
    write_snapshot_atomic,
)


def _snapshot(seed_time):
    return pd.DataFrame(
        {
            "seed_time": pd.to_datetime([seed_time, seed_time]),
            "src_article_id": pd.Series([0, 0], dtype="int64"),
            "dst_article_id": pd.Series([1, 2], dtype="int64"),
            "support": pd.Series([3, 3], dtype="int64"),
            "cf_score": pd.Series([0.5, 0.4], dtype="float32"),
            "rank": pd.Series([1, 2], dtype="int32"),
        }
    )


def test_snapshot_filename_midnight_and_non_midnight():
    assert snapshot_filename(pd.Timestamp("2020-03-02")) == "cf_snapshot_2020-03-02.parquet"
    assert (
        snapshot_filename(pd.Timestamp("2020-03-02 01:02:03"))
        == "cf_snapshot_2020-03-02T01-02-03.parquet"
    )
    assert timestamp_key(pd.Timestamp("2020-03-02")) == "2020-03-02T00:00:00"


def test_parquet_manifest_roundtrip_and_atomic_cleanup(tmp_path):
    cfg = CFSnapshotConfig(min_support=3, top_l=2)
    snapshot_dir = cfg.snapshot_dir(tmp_path)
    manifest = initialize_or_load_manifest(snapshot_dir, cfg)
    assert manifest["dataset"] == "rel-hm"

    seed = pd.Timestamp("2020-03-02")
    path = write_snapshot_atomic(
        _snapshot(seed),
        snapshot_dir,
        seed,
        config=cfg,
        num_articles=3,
    )
    assert path.exists()
    assert not list(snapshot_dir.glob("*.tmp.parquet"))
    loaded = load_snapshot(snapshot_dir, seed, config=cfg, num_articles=3)
    assert loaded["cf_score"].dtype == "float32"
    assert pd.Timestamp(loaded["seed_time"].iloc[0]) == seed

    manifest["snapshot_files"][manifest_key(seed)] = {"file": path.name, "num_rows": 2}
    write_manifest_atomic(snapshot_dir, manifest)
    assert load_manifest(snapshot_dir)["snapshot_files"][manifest_key(seed)]["num_rows"] == 2
    assert validate_manifest_for_training(snapshot_dir) == cfg


def test_manifest_conflict_fails(tmp_path):
    cfg = CFSnapshotConfig()
    snapshot_dir = cfg.snapshot_dir(tmp_path)
    initialize_or_load_manifest(snapshot_dir, cfg)
    manifest = load_manifest(snapshot_dir)
    manifest["top_l"] = 99
    with open(snapshot_dir / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f)
    with pytest.raises(ValueError, match="top_l"):
        initialize_or_load_manifest(snapshot_dir, cfg)


def test_invalid_snapshot_requires_overwrite_policy_hook(tmp_path):
    cfg = CFSnapshotConfig()
    snapshot_dir = cfg.snapshot_dir(tmp_path)
    snapshot_dir.mkdir(parents=True)
    seed = pd.Timestamp("2020-03-02")
    bad = _snapshot(seed)
    bad.loc[0, "dst_article_id"] = 0
    bad.to_parquet(snapshot_dir / snapshot_filename(seed), index=False)
    with pytest.raises(ValueError, match="self-pairs"):
        load_snapshot(snapshot_dir, seed, config=cfg, num_articles=3)
