from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from relbench.base import RecommendationTask

from .common import DirectCFColumns, required_direct_columns
from .config import (
    HMDirectCFSnapshotConfig,
    ensure_manifest_compatible,
    snapshot_filename,
    timestamp_key,
)


HM_COLUMNS = DirectCFColumns(
    interaction_src_col="customer_id",
    interaction_item_col="article_id",
    interaction_time_col="t_dat",
    item_cf_src_col="src_article_id",
    item_cf_dst_col="dst_article_id",
    out_src_col="src_customer_id",
    out_dst_col="dst_article_id",
    num_source_items_col="num_source_articles",
    best_source_item_col="best_src_article_id",
)
REQUIRED_COLUMNS = required_direct_columns(HM_COLUMNS)


def validate_snapshot_frame(
    df: pd.DataFrame,
    seed_time: pd.Timestamp,
    *,
    config: HMDirectCFSnapshotConfig | None = None,
    num_customers: int | None = None,
    num_articles: int | None = None,
    path: Path | None = None,
) -> None:
    _validate_direct_snapshot_frame(
        df,
        seed_time,
        columns=HM_COLUMNS,
        config=config,
        source_upper=num_customers,
        dst_upper=num_articles,
        path=path,
        label="HM direct-CF snapshot",
    )


def _validate_direct_snapshot_frame(
    df: pd.DataFrame,
    seed_time: pd.Timestamp,
    *,
    columns: DirectCFColumns,
    config,
    source_upper: int | None,
    dst_upper: int | None,
    path: Path | None,
    label: str,
) -> None:
    location = f" in {path}" if path is not None else ""
    missing = [col for col in required_direct_columns(columns) if col not in df.columns]
    if missing:
        raise ValueError(f"{label}{location} is missing columns: {missing}")
    if not pd.api.types.is_datetime64_any_dtype(df["seed_time"]):
        raise ValueError(f"{label}{location} seed_time must be datetime.")
    if len(df) == 0:
        return
    times = pd.to_datetime(df["seed_time"]).unique()
    if len(times) != 1 or pd.Timestamp(times[0]) != pd.Timestamp(seed_time):
        raise ValueError(f"{label}{location} does not match seed time {seed_time}.")
    if df.duplicated([columns.out_src_col, columns.out_dst_col]).any():
        raise ValueError(f"{label}{location} contains duplicate source-destination pairs.")
    if config is not None and df.groupby(columns.out_src_col).size().max() > config.direct_top_k:
        raise ValueError(f"{label}{location} violates direct_top_k.")
    if (df["rank"] < 1).any():
        raise ValueError(f"{label}{location} contains invalid ranks.")
    for score_col in ["direct_score", "max_cf_score"]:
        values = df[score_col].to_numpy()
        if not np.isfinite(values).all():
            raise ValueError(f"{label}{location} contains non-finite scores.")
        if (values < 0).any():
            raise ValueError(f"{label}{location} contains negative scores.")
    count_cols = [
        "support_sum",
        "support_max",
        columns.num_source_items_col,
        "best_item_cf_rank",
    ]
    if (df[count_cols] < 1).any().any():
        raise ValueError(f"{label}{location} contains invalid positive counts.")
    if source_upper is not None:
        src = df[columns.out_src_col]
        if src.lt(0).any() or src.ge(source_upper).any():
            raise ValueError(f"{label}{location} contains source IDs out of range.")
    if dst_upper is not None:
        dst = df[columns.out_dst_col]
        if dst.lt(0).any() or dst.ge(dst_upper).any():
            raise ValueError(f"{label}{location} contains destination IDs out of range.")
        best_src_item = df[columns.best_source_item_col]
        if best_src_item.lt(0).any() or best_src_item.ge(dst_upper).any():
            raise ValueError(
                f"{label}{location} contains provenance item IDs out of range."
            )
    for _, ranks in df.groupby(columns.out_src_col)["rank"]:
        expected = np.arange(1, len(ranks) + 1)
        if not np.array_equal(ranks.to_numpy(), expected):
            raise ValueError(f"{label}{location} ranks must be consecutive.")


def load_snapshot(
    snapshot_dir: Path,
    seed_time: pd.Timestamp,
    *,
    validate: bool = True,
    config: HMDirectCFSnapshotConfig | None = None,
    num_customers: int | None = None,
    num_articles: int | None = None,
) -> pd.DataFrame:
    path = Path(snapshot_dir) / snapshot_filename(seed_time)
    if not path.exists():
        raise FileNotFoundError(
            f"Missing HM direct-CF snapshot for {pd.Timestamp(seed_time)}: {path}"
        )
    df = pd.read_parquet(path)
    if validate:
        validate_snapshot_frame(
            df,
            seed_time,
            config=config,
            num_customers=num_customers,
            num_articles=num_articles,
            path=path,
        )
    return df


def write_snapshot_atomic(
    df: pd.DataFrame,
    snapshot_dir: Path,
    seed_time: pd.Timestamp,
    *,
    config: HMDirectCFSnapshotConfig,
    num_customers: int | None = None,
    num_articles: int | None = None,
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
            num_customers=num_customers,
            num_articles=num_articles,
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
    config: HMDirectCFSnapshotConfig,
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
    config: HMDirectCFSnapshotConfig | None = None,
) -> HMDirectCFSnapshotConfig:
    manifest = load_manifest(snapshot_dir)
    loaded = HMDirectCFSnapshotConfig.from_manifest(manifest)
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


def source_nodes_by_seed_time(
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
