from __future__ import annotations

import time
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import Tensor

from .adapters import RecommendationTaskAdapter
from .config import LatentRelationConfig
from .model import LatentRelationModel
from .retrieval import make_retriever


@dataclass(frozen=True)
class MaterializationStats:
    seed_time: str
    num_destination_candidates: int
    num_anchors: int
    num_edges: int
    avg_edges_per_anchor: float
    retrieval_seconds: float
    peak_resident_memory_bytes: int | None = None

    def to_manifest_entry(self, file: str, file_size_bytes: int | None = None) -> dict:
        out = asdict(self)
        out["file"] = file
        if file_size_bytes is not None:
            out["file_size_bytes"] = file_size_bytes
        return out


def _peak_resident_memory_bytes() -> int | None:
    try:
        import os
        import resource

        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return int(rss if os.name == "nt" else rss * 1024)
    except Exception:
        return None


@torch.inference_mode()
def encode_projected_destinations(
    model: LatentRelationModel,
    ids: Tensor,
    *,
    device: torch.device,
    chunk_size: int,
    projection: str,
) -> Tensor:
    model.eval()
    chunks: list[Tensor] = []
    for chunk in ids.split(chunk_size):
        chunk = chunk.to(device)
        h = model.encode(chunk)
        if projection == "query":
            projected = model.scorer.query_embeddings(h)
        elif projection == "key":
            projected = model.scorer.key_embeddings(h)
        else:
            raise ValueError("projection must be 'query' or 'key'.")
        chunks.append(projected.detach().cpu())
    return torch.cat(chunks, dim=0) if chunks else torch.empty((0, 0))


def build_latent_snapshot(
    model: LatentRelationModel,
    adapter: RecommendationTaskAdapter,
    seed_time: pd.Timestamp,
    anchor_ids: np.ndarray,
    config: LatentRelationConfig,
    *,
    device: torch.device,
    query_chunk_size: int = 4096,
    candidate_chunk_size: int = 65536,
    embedding_chunk_size: int = 4096,
    allow_self_loop: bool = False,
) -> tuple[pd.DataFrame, MaterializationStats]:
    tic = time.perf_counter()
    seed_time = pd.Timestamp(seed_time)
    anchors = torch.as_tensor(np.asarray(sorted(set(map(int, anchor_ids))), dtype=np.int64))
    candidates = torch.as_tensor(adapter.destination_ids_available_at(seed_time)).long()
    if candidates.numel() == 0 or anchors.numel() == 0:
        empty = empty_snapshot(seed_time)
        elapsed = time.perf_counter() - tic
        return empty, MaterializationStats(
            seed_time=seed_time.isoformat(),
            num_destination_candidates=int(candidates.numel()),
            num_anchors=int(anchors.numel()),
            num_edges=0,
            avg_edges_per_anchor=0.0,
            retrieval_seconds=elapsed,
            peak_resident_memory_bytes=_peak_resident_memory_bytes(),
        )

    key_embeddings = encode_projected_destinations(
        model,
        candidates,
        device=device,
        chunk_size=embedding_chunk_size,
        projection="key",
    )
    retriever = make_retriever(
        config.retrieval_backend,
        candidate_chunk_size=candidate_chunk_size,
    )
    retriever.build_index(candidates, key_embeddings)

    src_chunks: list[np.ndarray] = []
    dst_chunks: list[np.ndarray] = []
    score_chunks: list[np.ndarray] = []
    rank_chunks: list[np.ndarray] = []
    search_k = config.top_l if allow_self_loop else min(config.top_l + 1, candidates.numel())
    for anchor_chunk in anchors.split(query_chunk_size):
        query_embeddings = encode_projected_destinations(
            model,
            anchor_chunk,
            device=device,
            chunk_size=embedding_chunk_size,
            projection="query",
        ).to(device)
        result = retriever.search(
            query_embeddings,
            top_k=search_k,
            exclude_ids=None if allow_self_loop else anchor_chunk,
        )
        for src, ids, scores in zip(anchor_chunk.tolist(), result.indices, result.scores):
            seen: set[int] = set()
            dst_values: list[int] = []
            score_values: list[float] = []
            for dst, score in zip(ids.tolist(), scores.tolist()):
                if not np.isfinite(score):
                    continue
                if not allow_self_loop and int(dst) == int(src):
                    continue
                if int(dst) in seen:
                    continue
                seen.add(int(dst))
                dst_values.append(int(dst))
                score_values.append(float(score) / model.scorer.temperature)
                if len(dst_values) >= config.top_l:
                    break
            if not dst_values:
                continue
            src_chunks.append(np.full(len(dst_values), int(src), dtype=np.int64))
            dst_chunks.append(np.asarray(dst_values, dtype=np.int64))
            score_chunks.append(np.asarray(score_values, dtype=np.float32))
            rank_chunks.append(np.arange(1, len(dst_values) + 1, dtype=np.int32))

    if src_chunks:
        out = pd.DataFrame(
            {
                "src_id": np.concatenate(src_chunks),
                "dst_id": np.concatenate(dst_chunks),
                "score": np.concatenate(score_chunks),
                "rank": np.concatenate(rank_chunks),
                "seed_time": pd.Series(
                    np.repeat(seed_time.to_datetime64(), sum(map(len, src_chunks))),
                    dtype="datetime64[ns]",
                ),
            }
        )[["src_id", "dst_id", "score", "rank", "seed_time"]]
    else:
        out = empty_snapshot(seed_time)
    elapsed = time.perf_counter() - tic
    stats = MaterializationStats(
        seed_time=seed_time.isoformat(),
        num_destination_candidates=int(candidates.numel()),
        num_anchors=int(anchors.numel()),
        num_edges=int(len(out)),
        avg_edges_per_anchor=float(len(out) / max(int(anchors.numel()), 1)),
        retrieval_seconds=elapsed,
        peak_resident_memory_bytes=_peak_resident_memory_bytes(),
    )
    return out, stats


def empty_snapshot(seed_time: pd.Timestamp) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "src_id": pd.Series([], dtype="int64"),
            "dst_id": pd.Series([], dtype="int64"),
            "score": pd.Series([], dtype="float32"),
            "rank": pd.Series([], dtype="int32"),
            "seed_time": pd.Series([], dtype="datetime64[ns]"),
        }
    )
