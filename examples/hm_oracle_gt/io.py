from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import sparse

from .config import (
    OracleGTPhase1Config,
    ensure_manifest_compatible,
    snapshot_directory_name,
)


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return [_jsonable(item) for item in value.tolist()]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    return value


def write_json_atomic(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    os.close(fd)
    tmp_path = Path(tmp_name)
    try:
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(_jsonable(payload), f, indent=2, sort_keys=True)
            f.write("\n")
        os.replace(tmp_path, path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


def load_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_parquet_atomic(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{path.stem}.", suffix=".tmp.parquet", dir=path.parent
    )
    os.close(fd)
    tmp_path = Path(tmp_name)
    try:
        df.to_parquet(tmp_path, index=False)
        os.replace(tmp_path, path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


def write_csv_atomic(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.stem}.", suffix=".tmp.csv", dir=path.parent)
    os.close(fd)
    tmp_path = Path(tmp_name)
    try:
        df.to_csv(tmp_path, index=False)
        os.replace(tmp_path, path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


def initialize_or_load_manifest(
    output_dir: Path,
    config: OracleGTPhase1Config,
    *,
    overwrite: bool = False,
) -> dict:
    path = output_dir / "manifest.json"
    if path.exists() and not overwrite:
        manifest = load_json(path)
        ensure_manifest_compatible(manifest, config)
        manifest.setdefault("files", {})
        return manifest
    manifest = config.manifest_dict()
    manifest["files"] = {}
    write_json_atomic(path, manifest)
    return manifest


def snapshot_dir(output_dir: Path, seed_time: pd.Timestamp) -> Path:
    return output_dir / "snapshots" / snapshot_directory_name(seed_time)


def write_history_snapshot(
    output_dir: Path,
    seed_time: pd.Timestamp,
    matrix: sparse.csr_matrix,
    active_user_ids: np.ndarray,
    metadata: dict,
) -> dict:
    directory = snapshot_dir(output_dir, seed_time)
    directory.mkdir(parents=True, exist_ok=True)
    matrix_path = directory / "history_matrix.npz"
    users_path = directory / "active_user_ids.npy"
    metadata_path = directory / "metadata.json"
    sparse.save_npz(matrix_path, matrix)
    np.save(users_path, active_user_ids)
    write_json_atomic(metadata_path, metadata)
    return {
        "snapshot_id": metadata["snapshot_id"],
        "directory": str(directory.relative_to(output_dir)),
        "matrix_file": str(matrix_path.relative_to(output_dir)),
        "active_user_ids_file": str(users_path.relative_to(output_dir)),
        "metadata_file": str(metadata_path.relative_to(output_dir)),
    }

