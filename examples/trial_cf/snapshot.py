from __future__ import annotations

import math
import os
import time
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
from scipy import sparse

from .config import TrialCFSnapshotConfig


@dataclass(frozen=True)
class TrialCFSnapshotStats:
    seed_time: str
    window_start: str | None
    window_end: str
    num_raw_interactions: int
    num_active_sources: int
    num_active_sponsors: int
    num_unique_source_sponsor_pairs: int
    mean_distinct_sponsors_per_source: float
    max_distinct_sponsors_per_source: int
    cooccurrence_nnz_after_support: int
    num_rows: int
    elapsed_seconds: float
    peak_resident_memory_bytes: int | None = None

    def to_manifest_entry(self, file: str, file_size_bytes: int | None = None) -> dict:
        out = asdict(self)
        out["file"] = file
        if file_size_bytes is not None:
            out["file_size_bytes"] = file_size_bytes
        return out


def _peak_resident_memory_bytes() -> int | None:
    try:
        import resource

        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return int(rss if os.name == "nt" else rss * 1024)
    except Exception:
        return None


def _validate_interactions(
    interactions: pd.DataFrame,
    num_sources: int,
    num_sponsors: int,
) -> None:
    required = {"src_id", "sponsor_id", "date"}
    missing = required.difference(interactions.columns)
    if missing:
        raise ValueError(f"interactions is missing columns: {sorted(missing)}")
    if not pd.api.types.is_datetime64_any_dtype(interactions["date"]):
        raise ValueError("interactions.date must be datetime dtype.")
    for col, upper in [("src_id", num_sources), ("sponsor_id", num_sponsors)]:
        if not pd.api.types.is_integer_dtype(interactions[col]):
            raise ValueError(f"interactions.{col} must be integer-like.")
        if (interactions[col] < 0).any():
            raise ValueError(f"interactions.{col} must be nonnegative.")
        if len(interactions) > 0 and interactions[col].max() >= upper:
            raise ValueError(f"interactions.{col} contains out-of-range IDs.")


def _sort_interactions_once(interactions: pd.DataFrame) -> pd.DataFrame:
    if interactions["date"].is_monotonic_increasing:
        return interactions
    return interactions.sort_values("date", kind="mergesort").reset_index(drop=True)


def build_snapshot(
    interactions: pd.DataFrame,
    seed_time: pd.Timestamp,
    num_sources: int,
    num_sponsors: int,
    config: TrialCFSnapshotConfig,
) -> tuple[pd.DataFrame, TrialCFSnapshotStats]:
    """Build one source-sponsor route-collapsed CF snapshot."""
    tic = time.perf_counter()
    seed_time = pd.Timestamp(seed_time)
    _validate_interactions(interactions, num_sources, num_sponsors)
    interactions = _sort_interactions_once(interactions)
    times = interactions["date"].to_numpy(dtype="datetime64[ns]", copy=False)
    seed_np = np.datetime64(seed_time.to_datetime64(), "ns")

    if config.all_history:
        left = 0
        window_start = None
    else:
        lower = seed_time - pd.Timedelta(days=config.history_days)
        left = int(
            np.searchsorted(
                times,
                np.datetime64(lower.to_datetime64(), "ns"),
                side="right",
            )
        )
        window_start = lower.isoformat()
    right = int(np.searchsorted(times, seed_np, side="right"))
    window = interactions.iloc[left:right]
    raw_count = int(right - left)

    if "nct_id" in window.columns:
        pairs = window[["src_id", "sponsor_id", "nct_id"]].drop_duplicates(
            ignore_index=True
        )
    else:
        pairs = window[["src_id", "sponsor_id"]].drop_duplicates(ignore_index=True)

    if len(pairs) == 0:
        elapsed = time.perf_counter() - tic
        return _empty_snapshot(seed_time), TrialCFSnapshotStats(
            seed_time=seed_time.isoformat(),
            window_start=window_start,
            window_end=seed_time.isoformat(),
            num_raw_interactions=raw_count,
            num_active_sources=0,
            num_active_sponsors=0,
            num_unique_source_sponsor_pairs=0,
            mean_distinct_sponsors_per_source=0.0,
            max_distinct_sponsors_per_source=0,
            cooccurrence_nnz_after_support=0,
            num_rows=0,
            elapsed_seconds=elapsed,
            peak_resident_memory_bytes=_peak_resident_memory_bytes(),
        )

    grouped = (
        pairs.groupby(["src_id", "sponsor_id"], sort=False)
        .size()
        .rename("support")
        .reset_index()
    )
    src_ids = grouped["src_id"].to_numpy(dtype=np.int64, copy=False)
    sponsor_ids = grouped["sponsor_id"].to_numpy(dtype=np.int64, copy=False)
    support_values = grouped["support"].to_numpy(dtype=np.int32, copy=False)
    matrix = sparse.csr_matrix(
        (support_values, (src_ids, sponsor_ids)),
        shape=(num_sources, num_sponsors),
        dtype=np.int32,
    )
    matrix.sort_indices()
    source_counts = np.asarray(matrix.sum(axis=1)).ravel().astype(np.float64)
    sponsor_counts = np.asarray(matrix.sum(axis=0)).ravel().astype(np.float64)
    distinct_sponsors = np.diff(matrix.indptr)

    src_chunks: list[np.ndarray] = []
    dst_chunks: list[np.ndarray] = []
    support_chunks: list[np.ndarray] = []
    score_chunks: list[np.ndarray] = []
    rank_chunks: list[np.ndarray] = []
    nnz_after_support = 0

    for src in np.flatnonzero(distinct_sponsors):
        start, end = int(matrix.indptr[src]), int(matrix.indptr[src + 1])
        dst = matrix.indices[start:end].astype(np.int64, copy=False)
        support = matrix.data[start:end].astype(np.int64, copy=False)
        keep = support >= config.min_support
        if not np.any(keep):
            continue
        dst = dst[keep]
        support = support[keep]
        nnz_after_support += int(len(dst))
        denom = np.power(source_counts[src], config.alpha) * np.power(
            sponsor_counts[dst], 1.0 - config.alpha
        )
        scores = (support / denom).astype(np.float32, copy=False)
        order = np.lexsort((dst, -support, -scores))[: config.top_l]
        dst = dst[order]
        support = support[order]
        scores = scores[order]
        ranks = np.arange(1, len(dst) + 1, dtype=np.int32)
        src_chunks.append(np.full(len(dst), src, dtype=np.int64))
        dst_chunks.append(dst.astype(np.int64, copy=False))
        support_chunks.append(support.astype(np.int64, copy=False))
        score_chunks.append(scores)
        rank_chunks.append(ranks)

    if src_chunks:
        num_rows = sum(map(len, src_chunks))
        out = pd.DataFrame(
            {
                "seed_time": pd.Series(
                    np.repeat(seed_time.to_datetime64(), num_rows),
                    dtype="datetime64[ns]",
                ),
                "src_id": np.concatenate(src_chunks).astype(np.int64),
                "dst_sponsor_id": np.concatenate(dst_chunks).astype(np.int64),
                "support": np.concatenate(support_chunks).astype(np.int64),
                "cf_score": np.concatenate(score_chunks).astype(np.float32),
                "rank": np.concatenate(rank_chunks).astype(np.int32),
            }
        )
    else:
        out = _empty_snapshot(seed_time)

    elapsed = time.perf_counter() - tic
    nonzero_sources = distinct_sponsors[distinct_sponsors > 0]
    stats = TrialCFSnapshotStats(
        seed_time=seed_time.isoformat(),
        window_start=window_start,
        window_end=seed_time.isoformat(),
        num_raw_interactions=raw_count,
        num_active_sources=int(np.count_nonzero(distinct_sponsors)),
        num_active_sponsors=int(np.count_nonzero(sponsor_counts)),
        num_unique_source_sponsor_pairs=int(len(grouped)),
        mean_distinct_sponsors_per_source=float(np.mean(nonzero_sources))
        if len(nonzero_sources)
        else math.nan,
        max_distinct_sponsors_per_source=int(np.max(nonzero_sources))
        if len(nonzero_sources)
        else 0,
        cooccurrence_nnz_after_support=nnz_after_support,
        num_rows=int(len(out)),
        elapsed_seconds=elapsed,
        peak_resident_memory_bytes=_peak_resident_memory_bytes(),
    )
    return out, stats


def _empty_snapshot(seed_time: pd.Timestamp) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "seed_time": pd.Series([], dtype="datetime64[ns]"),
            "src_id": pd.Series([], dtype="int64"),
            "dst_sponsor_id": pd.Series([], dtype="int64"),
            "support": pd.Series([], dtype="int64"),
            "cf_score": pd.Series([], dtype="float32"),
            "rank": pd.Series([], dtype="int32"),
        }
    )

