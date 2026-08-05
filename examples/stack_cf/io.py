from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from .config import (
    StackCFSnapshotConfig,
    ensure_manifest_compatible,
    snapshot_filename,
    timestamp_key,
)


REQUIRED_COLUMNS = [
    "seed_time",
    "src_PostId",
    "dst_PostId",
    "support",
    "cf_score",
    "rank",
]


def validate_snapshot_frame(
    df: pd.DataFrame,
    seed_time: pd.Timestamp,
    *,
    config: StackCFSnapshotConfig | None = None,
    num_posts: int | None = None,
    path: Path | None = None,
) -> None:
    location = f" in {path}" if path is not None else ""
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"Snapshot{location} is missing columns: {missing}")
    if not pd.api.types.is_datetime64_any_dtype(df["seed_time"]):
        raise ValueError(f"Snapshot{location} seed_time must be datetime.")
    if len(df) == 0:
        return

    times = pd.to_datetime(df["seed_time"]).unique()
    if len(times) != 1 or pd.Timestamp(times[0]) != pd.Timestamp(seed_time):
        raise ValueError(
            f"Snapshot{location} does not match seed time {pd.Timestamp(seed_time)}."
        )
    if (df["src_PostId"] == df["dst_PostId"]).any():
        raise ValueError(f"Snapshot{location} contains self-pairs.")
    if df.duplicated(["src_PostId", "dst_PostId"]).any():
        raise ValueError(f"Snapshot{location} contains duplicate ordered pairs.")
    if config is not None:
        if (df["support"] < config.min_support).any():
            raise ValueError(f"Snapshot{location} violates min_support.")
        if df.groupby("src_PostId").size().max() > config.top_l:
            raise ValueError(f"Snapshot{location} violates top_l.")
    if (df["rank"] < 1).any():
        raise ValueError(f"Snapshot{location} contains invalid ranks.")
    if not np.isfinite(df["cf_score"].to_numpy()).all():
        raise ValueError(f"Snapshot{location} contains non-finite scores.")
    if (df["cf_score"] < 0).any():
        raise ValueError(f"Snapshot{location} contains negative scores.")
    if num_posts is not None:
        ids = df[["src_PostId", "dst_PostId"]]
        if ids.lt(0).any().any() or ids.ge(num_posts).any().any():
            raise ValueError(f"Snapshot{location} contains PostIds out of range.")
    for _, ranks in df.groupby("src_PostId")["rank"]:
        expected = np.arange(1, len(ranks) + 1)
        if not np.array_equal(ranks.to_numpy(), expected):
            raise ValueError(f"Snapshot{location} ranks must be consecutive.")


def load_snapshot(
    snapshot_dir: Path,
    seed_time: pd.Timestamp,
    *,
    validate: bool = True,
    config: StackCFSnapshotConfig | None = None,
    num_posts: int | None = None,
) -> pd.DataFrame:
    path = Path(snapshot_dir) / snapshot_filename(seed_time)
    if not path.exists():
        raise FileNotFoundError(
            f"Missing Stack CF snapshot for {pd.Timestamp(seed_time)}: {path}"
        )
    df = pd.read_parquet(path)
    if validate:
        validate_snapshot_frame(
            df,
            seed_time,
            config=config,
            num_posts=num_posts,
            path=path,
        )
    return df


def write_snapshot_atomic(
    df: pd.DataFrame,
    snapshot_dir: Path,
    seed_time: pd.Timestamp,
    *,
    config: StackCFSnapshotConfig,
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
            num_posts=num_posts,
            path=tmp_path,
        )
        os.replace(tmp_path, final_path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()
    return final_path


def load_manifest(snapshot_dir: Path) -> dict:
    path = Path(snapshot_dir) / "manifest.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing manifest: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_manifest_atomic(snapshot_dir: Path, manifest: dict) -> None:
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    path = snapshot_dir / "manifest.json"
    fd, tmp_name = tempfile.mkstemp(
        prefix=".manifest.", suffix=".tmp", dir=snapshot_dir
    )
    os.close(fd)
    tmp_path = Path(tmp_name)
    try:
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2, sort_keys=True)
            f.write("\n")
        os.replace(tmp_path, path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


def initialize_or_load_manifest(
    snapshot_dir: Path,
    config: StackCFSnapshotConfig,
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
    config: StackCFSnapshotConfig | None = None,
) -> StackCFSnapshotConfig:
    manifest = load_manifest(snapshot_dir)
    loaded = StackCFSnapshotConfig.from_manifest(manifest)
    if config is not None:
        ensure_manifest_compatible(manifest, config)
    return loaded


def discover_seed_times(task, splits: Iterable[str]) -> list[pd.Timestamp]:
    values = []
    for split in splits:
        table = task.get_table(split)
        values.extend(pd.to_datetime(table.df[task.time_col]).tolist())
    return sorted({pd.Timestamp(value) for value in values})


def parse_seed_times(values: list[str] | None) -> list[pd.Timestamp] | None:
    if not values:
        return None
    out: list[pd.Timestamp] = []
    for value in values:
        for part in value.split(","):
            part = part.strip()
            if part:
                out.append(pd.Timestamp(part))
    return sorted(set(out))


def manifest_key(seed_time: pd.Timestamp) -> str:
    return timestamp_key(seed_time)
