from __future__ import annotations

import json
import os
from pathlib import Path

import torch
from torch_frame import stype
from torch_frame.config.text_embedder import TextEmbedderConfig

from examples.text_embedder import GloveTextEmbedding
from relbench.datasets import get_dataset
from relbench.modeling.graph import make_pkey_fkey_graph
from relbench.modeling.utils import get_stype_proposal
from relbench.tasks import get_task

from .adapters import make_task_adapter
from .config import LatentRelationConfig
from .model import LatentRelationModel, TableDestinationEncoder, TwoTowerRelationScorer


def default_cache_dir() -> str:
    return os.path.expanduser("~/.cache/relbench_examples")


def load_stypes(dataset, dataset_name: str, cache_dir: str | Path) -> dict:
    stypes_cache_path = Path(cache_dir) / dataset_name / "stypes.json"
    try:
        with open(stypes_cache_path, "r", encoding="utf-8") as f:
            col_to_stype_dict = json.load(f)
        for _, col_to_stype in col_to_stype_dict.items():
            for col, stype_str in col_to_stype.items():
                col_to_stype[col] = stype(stype_str)
    except FileNotFoundError:
        col_to_stype_dict = get_stype_proposal(dataset.get_db())
        stypes_cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(stypes_cache_path, "w", encoding="utf-8") as f:
            json.dump(col_to_stype_dict, f, indent=2, default=str)
    return col_to_stype_dict


def load_relbench_context(
    *,
    dataset_name: str,
    task_name: str,
    cache_dir: str | Path,
    device: torch.device,
    download: bool = True,
):
    dataset = get_dataset(dataset_name, download=download)
    task = get_task(dataset_name, task_name, download=download)
    db = dataset.get_db()
    col_to_stype_dict = load_stypes(dataset, dataset_name, cache_dir)
    data, col_stats_dict = make_pkey_fkey_graph(
        db,
        col_to_stype_dict=col_to_stype_dict,
        text_embedder_cfg=TextEmbedderConfig(
            text_embedder=GloveTextEmbedding(device=device),
            batch_size=256,
        ),
        cache_dir=str(Path(cache_dir) / dataset_name / "materialized"),
    )
    adapter = make_task_adapter(dataset_name, task_name, task, db)
    return dataset, task, db, data, col_stats_dict, adapter


def build_table_relation_model(
    data,
    col_stats_dict,
    adapter,
    *,
    config: LatentRelationConfig,
) -> LatentRelationModel:
    encoder = TableDestinationEncoder(
        data,
        col_stats_dict,
        adapter.destination_type,
        channels=config.encoder_channels,
    )
    scorer = TwoTowerRelationScorer(
        config.encoder_channels,
        relation_dim=config.relation_dim,
        temperature=config.temperature,
    )
    return LatentRelationModel(encoder, scorer)


def load_relation_checkpoint(
    checkpoint_path: Path,
    data,
    col_stats_dict,
    adapter,
    *,
    device: torch.device,
) -> tuple[LatentRelationModel, LatentRelationConfig, dict]:
    payload = torch.load(checkpoint_path, map_location=device)
    config = LatentRelationConfig.from_manifest(payload["config"])
    model = build_table_relation_model(
        data,
        col_stats_dict,
        adapter,
        config=config,
    )
    model.load_state_dict(payload["model_state"])
    model.to(device)
    model.eval()
    return model, config, payload.get("extra", {})

