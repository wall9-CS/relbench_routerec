from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from relbench.base import RecommendationTask

from .config import (
    AvitoUserCFSnapshotConfig,
    ensure_manifest_compatible,
    snapshot_filename,
    timestamp_key,
)


REQUIRED_COLUMNS = [
    "seed_time",
    "src_UserID",
    "dst_UserID",
    "overlap",
    "user_cf_score",
    "src_history_size",
    "dst_history_size",
    "rank",
]


def validate_snapshot_frame(
    df: pd.DataFrame,
    seed_time: pd.Timestamp,
    *,
    config: AvitoUserCFSnapshotConfig | None = None,
    num_users: int | None = None,
    path: Path | None = None,
) -> None:
    location = f" in {path}" if path is not None else ""
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"Avito user-CF snapshot{location} is missing columns: {missing}")
    if not pd.api.types.is_datetime64_any_dtype(df["seed_time"]):
        raise ValueError(f"Avito user-CF snapshot{location} seed_time must be datetime.")
    if len(df) == 0:
        return
    times = pd.to_datetime(df["seed_time"]).unique()
    if len(times) != 1 or pd.Timestamp(times[0]) != pd.Timestamp(seed_time):
        raise ValueError(
            f"Avito user-CF snapshot{location} does not match seed time "
            f"{pd.Timestamp(seed_time)}."
        )
    if (df["src_UserID"] == df["dst_UserID"]).any():
        raise ValueError(f"Avito user-CF snapshot{location} contains self-pairs.")
    if df.duplicated(["src_UserID", "dst_UserID"]).any():
        raise ValueError(
            f"Avito user-CF snapshot{location} contains duplicate ordered pairs."
        )
    if config is not None:
        if (df["overlap"] < config.min_overlap).any():
            raise ValueError(f"Avito user-CF snapshot{location} violates min_overlap.")
        if df.groupby("src_UserID").size().max() > config.top_k:
            raise ValueError(f"Avito user-CF snapshot{location} violates top_k.")
    if (df["rank"] < 1).any():
        raise ValueError(f"Avito user-CF snapshot{location} contains invalid ranks.")
    if not np.isfinite(df["user_cf_score"].to_numpy()).all():
        raise ValueError(f"Avito user-CF snapshot{location} contains non-finite scores.")
    if (df["user_cf_score"] < 0).any():
        raise ValueError(f"Avito user-CF snapshot{location} contains negative scores.")
    if (df[["overlap", "src_history_size", "dst_history_size"]] < 1).any().any():
        raise ValueError(
            f"Avito user-CF snapshot{location} contains invalid history counts."
        )
    if num_users is not None:
        ids = df[["src_UserID", "dst_UserID"]]
        if ids.lt(0).any().any() or ids.ge(num_users).any().any():
            raise ValueError(
                f"Avito user-CF snapshot{location} contains UserIDs out of range."
            )
    for _, ranks in df.groupby("src_UserID")["rank"]:
        expected = np.arange(1, len(ranks) + 1)
        if not np.array_equal(ranks.to_numpy(), expected):
            raise ValueError(
                f"Avito user-CF snapshot{location} ranks must be consecutive."
            )


def load_snapshot(
    snapshot_dir: Path,
    seed_time: pd.Timestamp,
    *,
    validate: bool = True,
    config: AvitoUserCFSnapshotConfig | None = None,
    num_users: int | None = None,
) -> pd.DataFrame:
    path = Path(snapshot_dir) / snapshot_filename(seed_time)
    if not path.exists():
        raise FileNotFoundError(
            f"Missing Avito user-CF snapshot for {pd.Timestamp(seed_time)}: {path}"
        )
    df = pd.read_parquet(path)
    if validate:
        validate_snapshot_frame(
            df,
            seed_time,
            config=config,
            num_users=num_users,
            path=path,
        )
    return df


def write_snapshot_atomic(
    df: pd.DataFrame,
    snapshot_dir: Path,
    seed_time: pd.Timestamp,
    *,
    config: AvitoUserCFSnapshotConfig,
    num_users: int | None = None,
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
    fd, tmp_name = tempfile.mkstemp(prefix=".manifest.", suffix=".tmp", dir=snapshot_dir)
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
    config: AvitoUserCFSnapshotConfig,
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
    config: AvitoUserCFSnapshotConfig | None = None,
) -> AvitoUserCFSnapshotConfig:
    manifest = load_manifest(snapshot_dir)
    loaded = AvitoUserCFSnapshotConfig.from_manifest(manifest)
    if config is not None:
        ensure_manifest_compatible(manifest, config)
    return loaded


def discover_seed_times(task: RecommendationTask, splits: Iterable[str]) -> list[pd.Timestamp]:
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


def source_users_by_seed_time(
    task: RecommendationTask,
    splits: Iterable[str],
) -> dict[pd.Timestamp, np.ndarray]:
    out: dict[pd.Timestamp, set[int]] = {}
    for split in splits:
        table = task.get_table(split)
        times = pd.to_datetime(table.df[task.time_col])
        for seed_time, group in table.df.groupby(times, sort=False):
            key = pd.Timestamp(seed_time)
            values = group[task.src_entity_col].dropna().astype(int).to_numpy()
            out.setdefault(key, set()).update(int(value) for value in values)
    return {
        seed_time: np.array(sorted(values), dtype=np.int64)
        for seed_time, values in out.items()
    }


def manifest_key(seed_time: pd.Timestamp) -> str:
    return timestamp_key(seed_time)

