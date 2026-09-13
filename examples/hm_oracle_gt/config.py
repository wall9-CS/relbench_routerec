from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd


SUPPORTED_DATASET = "rel-hm"
SUPPORTED_TASK = "user-item-purchase"
DEFAULT_NUM_LAYERS = 4
DEFAULT_NUM_NEIGHBORS = 128
DEFAULT_TEMPORAL_STRATEGY = "last"
DEFAULT_SAMPLER_SEED = 42


def _format_float(value: float) -> str:
    return f"{value:g}"


def timestamp_key(seed_time: pd.Timestamp) -> str:
    ts = pd.Timestamp(seed_time)
    if ts.tzinfo is not None:
        ts = ts.tz_convert(None)
    return ts.isoformat()


def snapshot_id(seed_time: pd.Timestamp) -> str:
    return timestamp_key(seed_time)


def snapshot_directory_name(seed_time: pd.Timestamp) -> str:
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
        return ts.strftime("%Y-%m-%d")
    stamp = ts.strftime("%Y-%m-%dT%H-%M-%S")
    if ts.microsecond or ts.nanosecond:
        stamp = f"{stamp}-{ts.microsecond:06d}{ts.nanosecond:03d}"
    return stamp


@dataclass(frozen=True)
class OracleGTPhase1Config:
    dataset: str = SUPPORTED_DATASET
    task: str = SUPPORTED_TASK
    window_weeks: int | None = 8
    all_history: bool = False
    num_layers: int = DEFAULT_NUM_LAYERS
    num_neighbors: int = DEFAULT_NUM_NEIGHBORS
    temporal_strategy: str = DEFAULT_TEMPORAL_STRATEGY
    sampler_seed: int = DEFAULT_SAMPLER_SEED

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
        if self.num_layers != DEFAULT_NUM_LAYERS:
            raise ValueError(
                "Oracle-GT Phase 1 preprocessing is fixed to the 4-layer baseline; "
                f"got num_layers={self.num_layers}."
            )
        if self.num_neighbors < 1:
            raise ValueError("num_neighbors must be positive.")
        if self.sampler_seed < 0:
            raise ValueError("sampler_seed must be nonnegative.")

    @property
    def fanout(self) -> tuple[int, ...]:
        return tuple(int(self.num_neighbors // 2**i) for i in range(self.num_layers))

    @property
    def directory_name(self) -> str:
        if self.all_history:
            history = "all_history"
        else:
            history = f"window_{self.window_weeks}w"
        return (
            f"phase1_{history}_layers_{self.num_layers}"
            f"_neighbors_{self.num_neighbors}_samplerseed_{self.sampler_seed}"
            f"_{self.temporal_strategy}"
        )

    def output_dir(self, output_root: str | Path) -> Path:
        return Path(output_root) / self.dataset / self.task / self.directory_name

    def manifest_dict(self) -> dict:
        return {
            "dataset": self.dataset,
            "task": self.task,
            "phase": "oracle_gt_phase1_preprocessing",
            "contains_oracle_relations": False,
            "contains_katz": False,
            "contains_graph_augmentation": False,
            "window_weeks": self.window_weeks,
            "all_history": self.all_history,
            "window_boundary": "transaction_time <= seed_time"
            if self.all_history
            else "(seed_time - window, seed_time]",
            "binary_interactions": True,
            "history_matrix_layout": "active_user_rows_by_global_article_columns",
            "num_layers": self.num_layers,
            "num_neighbors": self.num_neighbors,
            "fanout": list(self.fanout),
            "temporal_strategy": self.temporal_strategy,
            "sampler_seed": self.sampler_seed,
        }


def ensure_manifest_compatible(manifest: dict, config: OracleGTPhase1Config) -> None:
    expected = config.manifest_dict()
    for key, value in expected.items():
        if manifest.get(key) != value:
            raise ValueError(
                f"Manifest is incompatible for {key}: expected {value!r}, "
                f"found {manifest.get(key)!r}."
            )

