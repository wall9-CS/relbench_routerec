from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd


SUPPORTED_DATASET = "rel-hm"
SUPPORTED_TASK = "user-item-purchase"
CF_KIND = "direct_source_item_from_item_cf"
DIRECT_CF_NODE_TYPE = "direct_article_cf"


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
    return f"direct_cf_snapshot_{stamp}.parquet"


@dataclass(frozen=True)
class HMDirectCFSnapshotConfig:
    dataset: str = SUPPORTED_DATASET
    task: str = SUPPORTED_TASK
    parent_item_cf_snapshot_dir: str = ""
    window_weeks: int | None = 8
    all_history: bool = False
    min_support: int = 3
    top_l: int = 32
    alpha: float = 0.5
    direct_top_k: int = 128
    filter_seen_dst: bool = False

    def __post_init__(self) -> None:
        if self.dataset != SUPPORTED_DATASET or self.task != SUPPORTED_TASK:
            raise ValueError(
                "Only rel-hm/user-item-purchase is currently supported "
                f"(got {self.dataset}/{self.task})."
            )
        if self.all_history and self.window_weeks is not None:
            raise ValueError("window_weeks and all_history are mutually exclusive.")
        if not self.all_history and self.window_weeks is None:
            raise ValueError("window_weeks must be set unless all_history=True.")
        if self.min_support < 1:
            raise ValueError("min_support must be at least 1.")
        if self.top_l < 1:
            raise ValueError("top_l must be at least 1.")
        if self.direct_top_k < 1:
            raise ValueError("direct_top_k must be at least 1.")
        if not 0.0 <= self.alpha <= 1.0:
            raise ValueError("alpha must satisfy 0.0 <= alpha <= 1.0.")

    @property
    def directory_name(self) -> str:
        alpha = _format_float(self.alpha)
        if self.all_history:
            parent = (
                f"all_history_alpha_{alpha}_support_{self.min_support}"
                f"_itemtop{self.top_l}"
            )
        else:
            parent = (
                f"window_{self.window_weeks}w_alpha_{alpha}_support_{self.min_support}"
                f"_itemtop{self.top_l}"
            )
        return f"direct_from_{parent}_srctop{self.direct_top_k}"

    def snapshot_dir(self, output_root: str | Path) -> Path:
        return Path(output_root) / self.dataset / self.task / self.directory_name

    def manifest_dict(self) -> dict:
        return {
            "dataset": self.dataset,
            "task": self.task,
            "cf_kind": CF_KIND,
            "source_entity_table": "customer",
            "source_entity_col": "customer_id",
            "dst_entity_table": "article",
            "dst_entity_col": "article_id",
            "interaction_table": "transactions",
            "interaction_time_col": "t_dat",
            "source_history_boundary": "interaction_time <= seed_time",
            "parent_item_cf_node_type": "article_cf",
            "direct_cf_node_type": DIRECT_CF_NODE_TYPE,
            "parent_item_cf_snapshot_dir": self.parent_item_cf_snapshot_dir,
            "window_weeks": self.window_weeks,
            "all_history": self.all_history,
            "min_support": self.min_support,
            "top_l": self.top_l,
            "alpha": self.alpha,
            "direct_top_k": self.direct_top_k,
            "aggregate_score": "sum_cf_score",
            "filter_seen_dst": self.filter_seen_dst,
            "score_used_as_model_input": False,
        }

    @classmethod
    def from_manifest(cls, manifest: dict) -> "HMDirectCFSnapshotConfig":
        if manifest.get("cf_kind") != CF_KIND:
            raise ValueError(
                f"Manifest cf_kind must be {CF_KIND!r}; found {manifest.get('cf_kind')!r}."
            )
        return cls(
            dataset=manifest["dataset"],
            task=manifest["task"],
            parent_item_cf_snapshot_dir=manifest.get("parent_item_cf_snapshot_dir", ""),
            window_weeks=manifest.get("window_weeks"),
            all_history=bool(manifest.get("all_history", False)),
            min_support=int(manifest["min_support"]),
            top_l=int(manifest["top_l"]),
            alpha=float(manifest["alpha"]),
            direct_top_k=int(manifest["direct_top_k"]),
            filter_seen_dst=bool(manifest.get("filter_seen_dst", False)),
        )


def ensure_manifest_compatible(
    manifest: dict,
    config: HMDirectCFSnapshotConfig,
) -> None:
    expected = config.manifest_dict()
    for key, value in expected.items():
        if manifest.get(key) != value:
            raise ValueError(
                f"Manifest is incompatible for {key}: expected {value!r}, "
                f"found {manifest.get(key)!r}."
            )
