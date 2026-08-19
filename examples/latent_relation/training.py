from __future__ import annotations

import json
import random
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

from .adapters import QueryExample, RecommendationTaskAdapter
from .config import LatentRelationConfig
from .model import LatentRelationModel


def set_deterministic_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@dataclass
class RelationTrainingStats:
    epoch: int
    train_loss: float
    positive_score_mean: float
    negative_score_mean: float
    history_size_mean: float


class RelationTrainingDataset(Dataset):
    def __init__(
        self,
        examples: list[QueryExample],
        adapter: RecommendationTaskAdapter,
        *,
        history_limit: int,
        num_negatives: int,
        seed: int = 42,
    ) -> None:
        self.rows: list[dict] = []
        self.adapter = adapter
        self.history_limit = int(history_limit)
        self.num_negatives = int(num_negatives)
        self.rng = np.random.default_rng(seed)

        for example in examples:
            positives = tuple(
                dst for dst in example.positives if 0 <= dst < adapter.num_destinations
            )
            if not positives:
                continue
            history = adapter.history_for_source(
                example.source_id,
                example.seed_time,
                history_limit=history_limit,
            )
            if not history:
                continue
            self.rows.append(
                {
                    "source_id": example.source_id,
                    "seed_time": example.seed_time,
                    "history": history,
                    "positives": positives,
                    "row_index": example.row_index,
                }
            )

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict:
        row = self.rows[index]
        positives = set(row["positives"])
        candidates = self.adapter.destination_ids_available_at(row["seed_time"])
        if len(candidates) <= len(positives):
            raise ValueError("No valid negative destinations are available.")
        negatives: list[int] = []
        while len(negatives) < self.num_negatives:
            sample = int(candidates[self.rng.integers(0, len(candidates))])
            if sample not in positives:
                negatives.append(sample)
        return {**row, "negatives": tuple(negatives)}


def collate_relation_batch(rows: list[dict]) -> dict[str, object]:
    max_history = max(len(row["history"]) for row in rows)
    max_pos = max(len(row["positives"]) for row in rows)
    num_neg = len(rows[0]["negatives"])
    history = torch.zeros((len(rows), max_history), dtype=torch.long)
    history_mask = torch.zeros((len(rows), max_history), dtype=torch.bool)
    positives = torch.zeros((len(rows), max_pos), dtype=torch.long)
    positive_mask = torch.zeros((len(rows), max_pos), dtype=torch.bool)
    negatives = torch.zeros((len(rows), num_neg), dtype=torch.long)
    history_sizes = []
    for i, row in enumerate(rows):
        h = torch.tensor(row["history"], dtype=torch.long)
        p = torch.tensor(row["positives"], dtype=torch.long)
        n = torch.tensor(row["negatives"], dtype=torch.long)
        history[i, : h.numel()] = h
        history_mask[i, : h.numel()] = True
        positives[i, : p.numel()] = p
        positive_mask[i, : p.numel()] = True
        negatives[i] = n
        history_sizes.append(int(h.numel()))
    return {
        "history": history,
        "history_mask": history_mask,
        "positives": positives,
        "positive_mask": positive_mask,
        "negatives": negatives,
        "history_sizes": torch.tensor(history_sizes, dtype=torch.float),
    }


def relation_loss(
    model: LatentRelationModel,
    batch: dict[str, object],
    *,
    device: torch.device,
    normalize_logsumexp: bool = False,
) -> tuple[Tensor, Tensor, Tensor]:
    history = batch["history"].to(device)
    history_mask = batch["history_mask"].to(device)
    positives = batch["positives"].to(device)
    positive_mask = batch["positive_mask"].to(device)
    negatives = batch["negatives"].to(device)
    candidates = torch.cat([positives, negatives], dim=1)
    scores = model.score_history_candidates(
        history,
        history_mask,
        candidates,
        normalize_logsumexp=normalize_logsumexp,
    )
    pos_scores = scores[:, : positives.size(1)].masked_fill(~positive_mask, -torch.inf)
    neg_scores = scores[:, positives.size(1) :]
    numerator = torch.logsumexp(pos_scores, dim=1)
    denominator = torch.logsumexp(torch.cat([pos_scores, neg_scores], dim=1), dim=1)
    loss = -(numerator - denominator).mean()
    finite_pos = pos_scores[positive_mask]
    return loss, finite_pos.detach(), neg_scores.detach().reshape(-1)


def train_relation_model(
    model: LatentRelationModel,
    dataset: RelationTrainingDataset,
    *,
    device: torch.device,
    epochs: int,
    batch_size: int,
    lr: float,
    seed: int,
    max_steps_per_epoch: int | None = None,
) -> list[RelationTrainingStats]:
    set_deterministic_seed(seed)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_relation_batch,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model.to(device)
    logs: list[RelationTrainingStats] = []
    for epoch in range(1, epochs + 1):
        model.train()
        loss_sum = count = 0
        pos_scores: list[Tensor] = []
        neg_scores: list[Tensor] = []
        history_sizes: list[Tensor] = []
        for step, batch in enumerate(loader, start=1):
            loss, pos, neg = relation_loss(model, batch, device=device)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            batch_size_actual = int(batch["history"].size(0))
            loss_sum += float(loss.detach()) * batch_size_actual
            count += batch_size_actual
            pos_scores.append(pos.cpu())
            neg_scores.append(neg.cpu())
            history_sizes.append(batch["history_sizes"])
            if max_steps_per_epoch is not None and step >= max_steps_per_epoch:
                break
        stats = RelationTrainingStats(
            epoch=epoch,
            train_loss=loss_sum / max(count, 1),
            positive_score_mean=float(torch.cat(pos_scores).mean())
            if pos_scores
            else float("nan"),
            negative_score_mean=float(torch.cat(neg_scores).mean())
            if neg_scores
            else float("nan"),
            history_size_mean=float(torch.cat(history_sizes).mean())
            if history_sizes
            else float("nan"),
        )
        logs.append(stats)
        print(json.dumps(asdict(stats), sort_keys=True))
    return logs


def save_checkpoint_atomic(
    model: LatentRelationModel,
    path: Path,
    *,
    config: LatentRelationConfig,
    extra: dict | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_state": model.state_dict(),
        "config": config.manifest_dict(),
        "extra": extra or {},
    }
    with tempfile.NamedTemporaryFile(
        prefix=f".{path.stem}.",
        suffix=".tmp",
        dir=path.parent,
        delete=False,
    ) as tmp:
        tmp_path = Path(tmp.name)
    try:
        torch.save(payload, tmp_path)
        tmp_path.replace(path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()

