from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd


SUPPORTED_DATASET = "rel-avito"
SUPPORTED_TASK = "user-ad-visit"
DEFAULT_HISTORY_DAYS = 4


def _format_float(value: float) -> str:
    return f"{value:g}"


def timestamp_key(seed_time: pd.Timestamp) -> str:
    ts = pd.Timestamp(seed_time)
    if ts.tzinfo is not None:
        ts = ts.tz_convert(None)
    return ts.isoformat()


def snapshot_filename(seed_time: pd.Timestamp) -> str:
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
class AvitoCFSnapshotConfig:
    """Configuration for seed-time-specific Rel-Avito ad CF snapshots."""

    dataset: str = SUPPORTED_DATASET
    task: str = SUPPORTED_TASK
    history_days: int | None = DEFAULT_HISTORY_DAYS
    all_history: bool = False
    min_support: int = 3
    top_l: int = 32
    alpha: float = 0.5

    def __post_init__(self) -> None:
        if self.dataset != SUPPORTED_DATASET or self.task != SUPPORTED_TASK:
            raise ValueError(
                "Only rel-avito/user-ad-visit is currently supported "
                f"(got {self.dataset}/{self.task})."
            )
        if self.all_history and self.history_days is not None:
            raise ValueError("--history-days and --all-history are mutually exclusive.")
        if not self.all_history and self.history_days is None:
            raise ValueError("history_days must be set unless all_history=True.")
        if self.history_days is not None and self.history_days <= 0:
            raise ValueError("history_days must be positive.")
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
            f"window_{self.history_days}d_alpha_{alpha}_support_{self.min_support}"
            f"_top{self.top_l}"
        )

    @property
    def window_weeks(self) -> float | None:
        if self.history_days is None:
            return None
        return self.history_days / 7

    def snapshot_dir(self, output_root: str | Path) -> Path:
        return Path(output_root) / self.dataset / self.task / self.directory_name

    def manifest_dict(self) -> dict:
        return {
            "dataset": self.dataset,
            "task": self.task,
            "history_days": self.history_days,
            "all_history": self.all_history,
            "window_boundary": "ViewDate <= seed_time"
            if self.all_history
            else "(seed_time - history_days, seed_time]",
            "binary_interactions": True,
            "user_scope": "all_database_users",
            "interaction_table": "VisitStream",
            "src_entity_table": "UserInfo",
            "dst_entity_table": "AdsInfo",
            "cf_node_type": "ad_cf",
            "min_support": self.min_support,
            "top_l": self.top_l,
            "alpha": self.alpha,
            "score_used_as_model_input": False,
        }

    @classmethod
    def from_manifest(cls, manifest: dict) -> "AvitoCFSnapshotConfig":
        return cls(
            dataset=manifest["dataset"],
            task=manifest["task"],
            history_days=manifest.get("history_days"),
            all_history=bool(manifest.get("all_history", False)),
            min_support=int(manifest["min_support"]),
            top_l=int(manifest["top_l"]),
            alpha=float(manifest["alpha"]),
        )


def ensure_manifest_compatible(manifest: dict, config: AvitoCFSnapshotConfig) -> None:
    expected = config.manifest_dict()
    for key, value in expected.items():
        if manifest.get(key) != value:
            raise ValueError(
                f"Manifest is incompatible for {key}: expected {value!r}, "
                f"found {manifest.get(key)!r}."
            )
