from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(frozen=True)
class RetrievalResult:
    indices: Tensor
    scores: Tensor


class RelationRetriever:
    def build_index(self, candidate_ids: Tensor, key_embeddings: Tensor) -> None:
        raise NotImplementedError

    def search(
        self,
        query_embeddings: Tensor,
        *,
        top_k: int,
        exclude_ids: Tensor | None = None,
    ) -> RetrievalResult:
        raise NotImplementedError


class TorchExactRetriever(RelationRetriever):
    """Chunked exact inner-product retrieval without an all-pairs matrix."""

    def __init__(self, *, candidate_chunk_size: int = 65536) -> None:
        self.candidate_chunk_size = int(candidate_chunk_size)
        self.candidate_ids: Tensor | None = None
        self.key_embeddings: Tensor | None = None

    def build_index(self, candidate_ids: Tensor, key_embeddings: Tensor) -> None:
        if candidate_ids.ndim != 1:
            raise ValueError("candidate_ids must be one-dimensional.")
        if key_embeddings.size(0) != candidate_ids.numel():
            raise ValueError("candidate_ids and key_embeddings disagree on rows.")
        self.candidate_ids = candidate_ids.detach().cpu().long()
        self.key_embeddings = key_embeddings.detach().cpu().float()

    def search(
        self,
        query_embeddings: Tensor,
        *,
        top_k: int,
        exclude_ids: Tensor | None = None,
    ) -> RetrievalResult:
        if self.candidate_ids is None or self.key_embeddings is None:
            raise RuntimeError("build_index must be called before search.")
        if top_k < 1:
            raise ValueError("top_k must be positive.")

        device = query_embeddings.device
        candidate_ids = self.candidate_ids.to(device)
        key_embeddings = self.key_embeddings.to(device)
        k = min(int(top_k), candidate_ids.numel())
        best_scores = torch.full(
            (query_embeddings.size(0), 0), -torch.inf, device=device
        )
        best_ids = torch.empty((query_embeddings.size(0), 0), dtype=torch.long, device=device)
        exclude_ids = exclude_ids.to(device) if exclude_ids is not None else None

        for start in range(0, candidate_ids.numel(), self.candidate_chunk_size):
            end = min(start + self.candidate_chunk_size, candidate_ids.numel())
            ids = candidate_ids[start:end]
            scores = query_embeddings @ key_embeddings[start:end].T
            if exclude_ids is not None:
                scores = scores.masked_fill(ids.unsqueeze(0) == exclude_ids.unsqueeze(1), -torch.inf)
            merged_scores = torch.cat([best_scores, scores], dim=1)
            merged_ids = torch.cat([best_ids, ids.expand(query_embeddings.size(0), -1)], dim=1)
            take = min(k, merged_scores.size(1))
            best_scores, order = torch.topk(merged_scores, k=take, dim=1)
            best_ids = torch.gather(merged_ids, 1, order)

        return RetrievalResult(indices=best_ids.detach().cpu(), scores=best_scores.detach().cpu())


class FaissRetriever(RelationRetriever):
    def __init__(self) -> None:
        try:
            import faiss  # type: ignore
        except ImportError as exc:
            raise RuntimeError(
                "FAISS retrieval was requested, but faiss is not installed. "
                "Use --retrieval-backend torch or install faiss separately."
            ) from exc
        self.faiss = faiss
        self.index = None
        self.candidate_ids: Tensor | None = None

    def build_index(self, candidate_ids: Tensor, key_embeddings: Tensor) -> None:
        keys = key_embeddings.detach().cpu().float().numpy()
        self.index = self.faiss.IndexFlatIP(keys.shape[1])
        self.index.add(keys)
        self.candidate_ids = candidate_ids.detach().cpu().long()

    def search(
        self,
        query_embeddings: Tensor,
        *,
        top_k: int,
        exclude_ids: Tensor | None = None,
    ) -> RetrievalResult:
        if self.index is None or self.candidate_ids is None:
            raise RuntimeError("build_index must be called before search.")
        extra = 1 if exclude_ids is not None else 0
        scores, positions = self.index.search(
            query_embeddings.detach().cpu().float().numpy(),
            int(top_k) + extra,
        )
        ids = self.candidate_ids[torch.from_numpy(positions).long()]
        score_tensor = torch.from_numpy(scores).float()
        if exclude_ids is not None:
            rows = []
            row_scores = []
            for row_ids, row_score, exclude in zip(ids, score_tensor, exclude_ids.cpu()):
                keep = row_ids != int(exclude)
                rows.append(row_ids[keep][:top_k])
                row_scores.append(row_score[keep][:top_k])
            ids = torch.stack(rows, dim=0)
            score_tensor = torch.stack(row_scores, dim=0)
        return RetrievalResult(indices=ids[:, :top_k], scores=score_tensor[:, :top_k])


def make_retriever(backend: str, *, candidate_chunk_size: int) -> RelationRetriever:
    if backend == "torch":
        return TorchExactRetriever(candidate_chunk_size=candidate_chunk_size)
    if backend == "faiss":
        return FaissRetriever()
    raise ValueError(f"Unknown retrieval backend: {backend}")

