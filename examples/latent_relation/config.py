from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd


SOFTWARE_VERSION = 1


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
    return f"latent_relation_snapshot_{stamp}.parquet"


@dataclass(frozen=True)
class LatentRelationConfig:
    dataset: str
    task: str
    history_limit: int = 64
    relation_dim: int = 128
    temperature: float = 0.07
    num_negatives: int = 256
    top_l: int = 32
    retrieval_backend: str = "torch"
    encoder_channels: int = 128
    checkpoint_hash: str | None = None
    checkpoint_path: str | None = None
    training_cutoff: str | None = None

    def __post_init__(self) -> None:
        if self.history_limit < 1:
            raise ValueError("history_limit must be positive.")
        if self.relation_dim < 1:
            raise ValueError("relation_dim must be positive.")
        if self.temperature <= 0:
            raise ValueError("temperature must be positive.")
        if self.num_negatives < 1:
            raise ValueError("num_negatives must be positive.")
        if self.top_l < 1:
            raise ValueError("top_l must be positive.")
        if self.encoder_channels < 1:
            raise ValueError("encoder_channels must be positive.")
        if self.retrieval_backend not in {"torch", "faiss"}:
            raise ValueError("retrieval_backend must be 'torch' or 'faiss'.")

    @property
    def directory_name(self) -> str:
        return (
            f"latent_dim{self.relation_dim}_temp{self.temperature:g}"
            f"_hist{self.history_limit}_top{self.top_l}_{self.retrieval_backend}"
        )

    def snapshot_dir(self, output_root: str | Path) -> Path:
        return Path(output_root) / self.dataset / self.task / self.directory_name

    def manifest_dict(self) -> dict:
        out = asdict(self)
        out["software_version"] = SOFTWARE_VERSION
        out["edge_semantics"] = "direct_destination_to_destination"
        out["history_boundary"] = "interaction_time <= seed_time"
        out["supervision"] = "training_split_future_recommendation_labels"
        return out

    @classmethod
    def from_manifest(cls, manifest: dict) -> "LatentRelationConfig":
        fields = {
            "dataset",
            "task",
            "history_limit",
            "relation_dim",
            "temperature",
            "num_negatives",
            "top_l",
            "retrieval_backend",
            "encoder_channels",
            "checkpoint_hash",
            "checkpoint_path",
            "training_cutoff",
        }
        return cls(**{key: manifest[key] for key in fields if key in manifest})


def ensure_manifest_compatible(manifest: dict, config: LatentRelationConfig) -> None:
    expected = config.manifest_dict()
    compared_keys = [
        "dataset",
        "task",
        "history_limit",
        "relation_dim",
        "temperature",
        "top_l",
        "retrieval_backend",
        "encoder_channels",
        "checkpoint_hash",
    ]
    for key in compared_keys:
        if manifest.get(key) != expected.get(key):
            raise ValueError(
                f"Manifest is incompatible for {key}: expected {expected.get(key)!r}, "
                f"found {manifest.get(key)!r}."
            )

