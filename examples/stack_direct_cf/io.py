from __future__ import annotations

from pathlib import Path
import os
import tempfile

import pandas as pd

from examples.hm_direct_cf.common import DirectCFColumns, required_direct_columns
from examples.hm_direct_cf.io import (
    _validate_direct_snapshot_frame,
    discover_seed_times,
    load_manifest,
    parse_seed_times,
    source_nodes_by_seed_time,
    write_manifest_atomic,
)

from .config import (
    StackDirectCFSnapshotConfig,
    ensure_manifest_compatible,
    snapshot_filename,
    timestamp_key,
)


STACK_COLUMNS = DirectCFColumns(
    interaction_src_col="UserId",
    interaction_item_col="PostId",
    interaction_time_col="CreationDate",
    item_cf_src_col="src_PostId",
    item_cf_dst_col="dst_PostId",
    out_src_col="src_UserId",
    out_dst_col="dst_PostId",
    num_source_items_col="num_source_posts",
    best_source_item_col="best_src_PostId",
)
REQUIRED_COLUMNS = required_direct_columns(STACK_COLUMNS)


def validate_snapshot_frame(
    df: pd.DataFrame,
    seed_time: pd.Timestamp,
    *,
    config: StackDirectCFSnapshotConfig | None = None,
    num_users: int | None = None,
    num_posts: int | None = None,
    path: Path | None = None,
) -> None:
    _validate_direct_snapshot_frame(
        df,
        seed_time,
        columns=STACK_COLUMNS,
        config=config,
        source_upper=num_users,
        dst_upper=num_posts,
        path=path,
        label="Stack direct-CF snapshot",
    )


def load_snapshot(
    snapshot_dir: Path,
    seed_time: pd.Timestamp,
    *,
    validate: bool = True,
    config: StackDirectCFSnapshotConfig | None = None,
    num_users: int | None = None,
    num_posts: int | None = None,
) -> pd.DataFrame:
    path = Path(snapshot_dir) / snapshot_filename(seed_time)
    if not path.exists():
        raise FileNotFoundError(
            f"Missing Stack direct-CF snapshot for {pd.Timestamp(seed_time)}: {path}"
        )
    df = pd.read_parquet(path)
    if validate:
        validate_snapshot_frame(
            df,
            seed_time,
            config=config,
            num_users=num_users,
            num_posts=num_posts,
            path=path,
        )
    return df


def write_snapshot_atomic(
    df: pd.DataFrame,
    snapshot_dir: Path,
    seed_time: pd.Timestamp,
    *,
    config: StackDirectCFSnapshotConfig,
    num_users: int | None = None,
    num_posts: int | None = None,
) -> Path:
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    final_path = snapshot_dir / snapshot_filename(seed_time)
    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{final_path.stem}.", suffix=".tmp.parquet", dir=snapshot_dir
    )
    os.close(fd)
    tmp_path = Path(tmp_name)
    try:
        df.to_parquet(tmp_path, index=False)
        tmp_df = pd.read_parquet(tmp_path)
        validate_snapshot_frame(
            tmp_df,
            seed_time,
            config=config,
            num_users=num_users,
            num_posts=num_posts,
            path=tmp_path,
        )
        os.replace(tmp_path, final_path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()
    return final_path


def initialize_or_load_manifest(
    snapshot_dir: Path,
    config: StackDirectCFSnapshotConfig,
) -> dict:
    path = Path(snapshot_dir) / "manifest.json"
    if path.exists():
        manifest = load_manifest(snapshot_dir)
        ensure_manifest_compatible(manifest, config)
        manifest.setdefault("snapshot_files", {})
        return manifest
    manifest = config.manifest_dict()
    manifest["snapshot_files"] = {}
    write_manifest_atomic(snapshot_dir, manifest)
    return manifest


def validate_manifest_for_training(
    snapshot_dir: Path,
    config: StackDirectCFSnapshotConfig | None = None,
) -> StackDirectCFSnapshotConfig:
    manifest = load_manifest(snapshot_dir)
    loaded = StackDirectCFSnapshotConfig.from_manifest(manifest)
    if config is not None:
        ensure_manifest_compatible(manifest, config)
    return loaded


def manifest_key(seed_time: pd.Timestamp) -> str:
    return timestamp_key(seed_time)
