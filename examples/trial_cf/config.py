from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd


SUPPORTED_DATASET = "rel-trial"
SUPPORTED_TASKS = {
    "condition-sponsor-run": {
        "src_entity_col": "condition_id",
        "src_entity_table": "conditions",
        "source_study_table": "conditions_studies",
    },
    "site-sponsor-run": {
        "src_entity_col": "facility_id",
        "src_entity_table": "facilities",
        "source_study_table": "facilities_studies",
    },
}
DEFAULT_HISTORY_DAYS = 365 * 5
CF_NODE_TYPE = "trial_cf"


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
class TrialCFSnapshotConfig:
    """Configuration for Rel-Trial source-sponsor route-collapsed CF snapshots."""

    dataset: str = SUPPORTED_DATASET
    task: str = "condition-sponsor-run"
    history_days: int | None = DEFAULT_HISTORY_DAYS
    all_history: bool = False
    min_support: int = 1
    top_l: int = 64
    alpha: float = 0.5

    def __post_init__(self) -> None:
        if self.dataset != SUPPORTED_DATASET:
            raise ValueError(f"Only {SUPPORTED_DATASET} is supported.")
        if self.task not in SUPPORTED_TASKS:
            raise ValueError(f"Unsupported Rel-Trial recommendation task: {self.task!r}.")
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
    def task_spec(self) -> dict[str, str]:
        return SUPPORTED_TASKS[self.task]

    @property
    def src_entity_col(self) -> str:
        return self.task_spec["src_entity_col"]

    @property
    def src_entity_table(self) -> str:
        return self.task_spec["src_entity_table"]

    @property
    def source_study_table(self) -> str:
        return self.task_spec["source_study_table"]

    @property
    def directory_name(self) -> str:
        alpha = _format_float(self.alpha)
        if self.all_history:
            return (
                f"source_sponsor_all_history_alpha_{alpha}"
                f"_support_{self.min_support}_top{self.top_l}"
            )
        return (
            f"source_sponsor_window_{self.history_days}d_alpha_{alpha}"
            f"_support_{self.min_support}_top{self.top_l}"
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
            "window_boundary": "source_study.date <= seed_time"
            if self.all_history
            else "(seed_time - history_days, seed_time]",
            "cf_route": "source_sponsor_route_collapsed",
            "interaction_table": self.source_study_table,
            "source_study_table": self.source_study_table,
            "sponsor_study_table": "sponsors_studies",
            "src_entity_col": self.src_entity_col,
            "src_entity_table": self.src_entity_table,
            "dst_entity_col": "sponsor_id",
            "dst_entity_table": "sponsors",
            "cf_node_type": CF_NODE_TYPE,
            "min_support": self.min_support,
            "top_l": self.top_l,
            "alpha": self.alpha,
            "score_used_as_model_input": False,
        }

    @classmethod
    def from_manifest(cls, manifest: dict) -> "TrialCFSnapshotConfig":
        return cls(
            dataset=manifest["dataset"],
            task=manifest["task"],
            history_days=manifest.get("history_days"),
            all_history=bool(manifest.get("all_history", False)),
            min_support=int(manifest["min_support"]),
            top_l=int(manifest["top_l"]),
            alpha=float(manifest["alpha"]),
        )


def ensure_manifest_compatible(
    manifest: dict,
    config: TrialCFSnapshotConfig,
) -> None:
    expected = config.manifest_dict()
    for key, value in expected.items():
        if manifest.get(key) != value:
            raise ValueError(
                f"Manifest is incompatible for {key}: expected {value!r}, "
                f"found {manifest.get(key)!r}."
            )

