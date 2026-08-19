from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor
from torch_frame.data.stats import StatType
from torch_geometric.data import HeteroData

from relbench.modeling.nn import HeteroEncoder


class DestinationEncoder(torch.nn.Module):
    def forward(self, destination_ids: Tensor) -> Tensor:
        raise NotImplementedError


class TableDestinationEncoder(DestinationEncoder):
    """Feature-based destination encoder backed by the existing TensorFrame table."""

    def __init__(
        self,
        data: HeteroData,
        col_stats_dict: dict[str, dict[str, dict[StatType, Any]]],
        destination_type: str,
        *,
        channels: int,
    ) -> None:
        super().__init__()
        self.destination_type = destination_type
        self.channels = int(channels)
        self.encoder = HeteroEncoder(
            channels=channels,
            node_to_col_names_dict={
                destination_type: data[destination_type].tf.col_names_dict
            },
            node_to_col_stats={
                destination_type: col_stats_dict[destination_type],
            },
        )
        self.tensor_frame = data[destination_type].tf

    def forward(self, destination_ids: Tensor) -> Tensor:
        tf = self.tensor_frame[destination_ids.detach().cpu()]
        tf = tf.to(destination_ids.device)
        return self.encoder({self.destination_type: tf})[self.destination_type]


class EmbeddingDestinationEncoder(DestinationEncoder):
    """Small test/backend encoder when table features are unavailable."""

    def __init__(self, num_destinations: int, channels: int) -> None:
        super().__init__()
        self.embedding = torch.nn.Embedding(num_destinations, channels)
        self.channels = int(channels)

    def forward(self, destination_ids: Tensor) -> Tensor:
        return self.embedding(destination_ids)


class TwoTowerRelationScorer(torch.nn.Module):
    def __init__(self, input_dim: int, relation_dim: int = 128, temperature: float = 0.07):
        super().__init__()
        self.query = torch.nn.Linear(input_dim, relation_dim, bias=False)
        self.key = torch.nn.Linear(input_dim, relation_dim, bias=False)
        self.temperature = float(temperature)

    def query_embeddings(self, h: Tensor) -> Tensor:
        return F.normalize(self.query(h), p=2, dim=-1)

    def key_embeddings(self, h: Tensor) -> Tensor:
        return F.normalize(self.key(h), p=2, dim=-1)

    def forward(self, left: Tensor, right: Tensor) -> Tensor:
        q = self.query_embeddings(left)
        k = self.key_embeddings(right)
        return (q * k).sum(dim=-1) / self.temperature


class LatentRelationModel(torch.nn.Module):
    def __init__(self, encoder: DestinationEncoder, scorer: TwoTowerRelationScorer):
        super().__init__()
        self.encoder = encoder
        self.scorer = scorer

    def encode(self, destination_ids: Tensor) -> Tensor:
        return self.encoder(destination_ids)

    def score_pairs(self, src_ids: Tensor, dst_ids: Tensor) -> Tensor:
        return self.scorer(self.encoder(src_ids), self.encoder(dst_ids))

    def score_history_candidates(
        self,
        history_ids: Tensor,
        history_mask: Tensor,
        candidate_ids: Tensor,
        *,
        normalize_logsumexp: bool = False,
    ) -> Tensor:
        """Return g_theta for each query and each candidate.

        history_ids: [B, H], candidate_ids: [B, C].
        """
        batch_size, history_width = history_ids.shape
        candidate_width = candidate_ids.size(1)
        flat_ids = torch.cat([history_ids.reshape(-1), candidate_ids.reshape(-1)])
        unique_ids, inverse = torch.unique(flat_ids, sorted=False, return_inverse=True)
        embeddings = self.encoder(unique_ids)
        hist = embeddings[inverse[: batch_size * history_width]].view(
            batch_size, history_width, -1
        )
        cand = embeddings[inverse[batch_size * history_width :]].view(
            batch_size, candidate_width, -1
        )
        q = self.scorer.query_embeddings(hist)
        k = self.scorer.key_embeddings(cand)
        scores = torch.einsum("bhd,bcd->bhc", q, k) / self.scorer.temperature
        scores = scores.masked_fill(~history_mask.unsqueeze(-1), -torch.inf)
        out = torch.logsumexp(scores, dim=1)
        if normalize_logsumexp:
            denom = history_mask.sum(dim=1).clamp_min(1).float().log().unsqueeze(-1)
            out = out - denom
        return out

