from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd


SUPPORTED_DATASET = "rel-hm"
SUPPORTED_TASK = "user-item-purchase"


def _format_float(value: float) -> str:
    return f"{value:g}"


def timestamp_key(seed_time: pd.Timestamp) -> str:
    """Return the canonical manifest key for an exact seed time."""
    ts = pd.Timestamp(seed_time)
    if ts.tzinfo is not None:
        ts = ts.tz_convert(None)
    return ts.isoformat()


def snapshot_filename(seed_time: pd.Timestamp) -> str:
    """Return a collision-resistant parquet filename for a seed time."""
    ts = pd.Timestamp(seed_time)
    if ts.tzinfo is not None:
        ts = ts.tz_convert(None)
    if (
        ts.hour == 0
        and ts.minute == 0
        and ts.second == 0
        and ts.microsecond == 0
        and ts.nanosecond == 0
    ):
        stamp = ts.strftime("%Y-%m-%d")
    else:
        stamp = ts.strftime("%Y-%m-%dT%H-%M-%S")
        if ts.microsecond or ts.nanosecond:
            stamp = f"{stamp}-{ts.microsecond:06d}{ts.nanosecond:03d}"
    return f"cf_snapshot_{stamp}.parquet"


@dataclass(frozen=True)
class CFSnapshotConfig:
    """Configuration for seed-time-specific Rel-HM CF snapshots."""

    dataset: str = SUPPORTED_DATASET
    task: str = SUPPORTED_TASK
    window_weeks: int | None = 8
    all_history: bool = False
    min_support: int = 3
    top_l: int = 32
    alpha: float = 0.5

    def __post_init__(self) -> None:
        if self.dataset != SUPPORTED_DATASET or self.task != SUPPORTED_TASK:
            raise ValueError(
                "Only rel-hm/user-item-purchase is currently supported "
                f"(got {self.dataset}/{self.task})."
            )
        if self.all_history and self.window_weeks is not None:
            raise ValueError("--window-weeks and --all-history are mutually exclusive.")
        if not self.all_history and self.window_weeks is None:
            raise ValueError("window_weeks must be set unless all_history=True.")
        if self.window_weeks is not None and self.window_weeks <= 0:
            raise ValueError("window_weeks must be positive.")
        if self.min_support < 1:
            raise ValueError("min_support must be at least 1.")
        if self.top_l < 1:
            raise ValueError("top_l must be at least 1.")
        if not 0.0 <= self.alpha <= 1.0:
            raise ValueError("alpha must satisfy 0.0 <= alpha <= 1.0.")

    @property
    def directory_name(self) -> str:
        alpha = _format_float(self.alpha)
        if self.all_history:
            return (
                f"all_history_alpha_{alpha}_support_{self.min_support}"
                f"_top{self.top_l}"
            )
        return (
            f"window_{self.window_weeks}w_alpha_{alpha}_support_{self.min_support}"
            f"_top{self.top_l}"
        )

    def snapshot_dir(self, output_root: str | Path) -> Path:
        return Path(output_root) / self.dataset / self.task / self.directory_name

    def manifest_dict(self) -> dict:
        return {
            "dataset": self.dataset,
            "task": self.task,
            "window_weeks": self.window_weeks,
            "all_history": self.all_history,
            "window_boundary": "transaction_time <= seed_time"
            if self.all_history
            else "(seed_time - window, seed_time]",
            "binary_interactions": True,
            "customer_scope": "all_database_customers",
            "min_support": self.min_support,
            "top_l": self.top_l,
            "alpha": self.alpha,
            "score_used_as_model_input": False,
        }

    @classmethod
    def from_manifest(cls, manifest: dict) -> "CFSnapshotConfig":
        return cls(
            dataset=manifest["dataset"],
            task=manifest["task"],
            window_weeks=manifest.get("window_weeks"),
            all_history=bool(manifest.get("all_history", False)),
            min_support=int(manifest["min_support"]),
            top_l=int(manifest["top_l"]),
            alpha=float(manifest["alpha"]),
        )


def ensure_manifest_compatible(manifest: dict, config: CFSnapshotConfig) -> None:
    """Raise when a manifest was produced with a different CF configuration."""
    expected = config.manifest_dict()
    for key, value in expected.items():
        if manifest.get(key) != value:
            raise ValueError(
                f"Manifest is incompatible for {key}: expected {value!r}, "
                f"found {manifest.get(key)!r}."
            )
