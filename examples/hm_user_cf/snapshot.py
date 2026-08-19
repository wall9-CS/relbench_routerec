from __future__ import annotations

import math
import os
import time
from dataclasses import asdict, dataclass
from typing import Iterable

import numpy as np
import pandas as pd
from scipy import sparse

from .config import UserCFSnapshotConfig


@dataclass(frozen=True)
class UserCFSnapshotStats:
    seed_time: str
    window_start: str | None
    window_end: str
    num_raw_transactions: int
    num_source_customers: int
    num_source_customers_with_history: int
    num_active_customers: int
    num_active_articles: int
    num_unique_user_item_pairs: int
    mean_distinct_items_per_customer: float
    max_distinct_items_per_customer: int
    overlap_nnz_before_filter: int
    overlap_nnz_after_filter: int
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


def _validate_transactions(
    transactions: pd.DataFrame,
    num_customers: int,
    num_articles: int,
) -> None:
    required = {"customer_id", "article_id", "t_dat"}
    missing = required.difference(transactions.columns)
    if missing:
        raise ValueError(f"transactions is missing columns: {sorted(missing)}")
    if not pd.api.types.is_datetime64_any_dtype(transactions["t_dat"]):
        raise ValueError("transactions.t_dat must be datetime dtype.")
    for col, upper in [("customer_id", num_customers), ("article_id", num_articles)]:
        if not pd.api.types.is_integer_dtype(transactions[col]):
            raise ValueError(f"transactions.{col} must be integer-like.")
        if (transactions[col] < 0).any():
            raise ValueError(f"transactions.{col} must be nonnegative.")
        if len(transactions) > 0 and transactions[col].max() >= upper:
            raise ValueError(f"transactions.{col} contains out-of-range IDs.")


def _validate_source_customers(
    source_customer_ids: Iterable[int],
    num_customers: int,
) -> np.ndarray:
    values = np.fromiter((int(value) for value in source_customer_ids), dtype=np.int64)
    if len(values) == 0:
        return values
    if values.min() < 0 or values.max() >= num_customers:
        raise ValueError("source_customer_ids contains out-of-range customer IDs.")
    return np.array(sorted(set(values.tolist())), dtype=np.int64)


def _sort_transactions_once(transactions: pd.DataFrame) -> pd.DataFrame:
    if transactions["t_dat"].is_monotonic_increasing:
        return transactions
    return transactions.sort_values("t_dat", kind="mergesort").reset_index(drop=True)


def build_snapshot(
    transactions: pd.DataFrame,
    seed_time: pd.Timestamp,
    num_customers: int,
    num_articles: int,
    source_customer_ids: Iterable[int],
    config: UserCFSnapshotConfig,
    *,
    source_chunk_size: int = 4096,
) -> tuple[pd.DataFrame, UserCFSnapshotStats]:
    """Build one source-scoped user-CF snapshot for an exact seed time."""
    if source_chunk_size < 1:
        raise ValueError("source_chunk_size must be positive.")
    tic = time.perf_counter()
    seed_time = pd.Timestamp(seed_time)
    _validate_transactions(transactions, num_customers, num_articles)
    sources = _validate_source_customers(source_customer_ids, num_customers)
    transactions = _sort_transactions_once(transactions)
    times = transactions["t_dat"].to_numpy(dtype="datetime64[ns]", copy=False)
    seed_np = np.datetime64(seed_time.to_datetime64(), "ns")

    if config.all_history:
        left = 0
        window_start = None
    else:
        lower = seed_time - pd.Timedelta(weeks=config.window_weeks)
        left = int(
            np.searchsorted(
                times,
                np.datetime64(lower.to_datetime64(), "ns"),
                side="right",
            )
        )
        window_start = lower.isoformat()
    right = int(np.searchsorted(times, seed_np, side="right"))
    window = transactions.iloc[left:right][["customer_id", "article_id"]]
    raw_count = int(right - left)

    pairs = window.drop_duplicates(ignore_index=True)
    if len(pairs) == 0 or len(sources) == 0:
        elapsed = time.perf_counter() - tic
        return _empty_snapshot(seed_time), UserCFSnapshotStats(
            seed_time=seed_time.isoformat(),
            window_start=window_start,
            window_end=seed_time.isoformat(),
            num_raw_transactions=raw_count,
            num_source_customers=int(len(sources)),
            num_source_customers_with_history=0,
            num_active_customers=0,
            num_active_articles=0,
            num_unique_user_item_pairs=int(len(pairs)),
            mean_distinct_items_per_customer=0.0,
            max_distinct_items_per_customer=0,
            overlap_nnz_before_filter=0,
            overlap_nnz_after_filter=0,
            num_rows=0,
            elapsed_seconds=elapsed,
            peak_resident_memory_bytes=_peak_resident_memory_bytes(),
        )

    customer_codes, active_customers_index = pd.factorize(
        pairs["customer_id"], sort=False
    )
    active_customer_ids = active_customers_index.to_numpy(dtype=np.int64, copy=False)
    article_ids = pairs["article_id"].to_numpy(dtype=np.int64, copy=False)
    values = np.ones(len(pairs), dtype=np.int32)
    matrix = sparse.csr_matrix(
        (values, (customer_codes.astype(np.int64, copy=False), article_ids)),
        shape=(len(active_customer_ids), num_articles),
        dtype=np.int32,
    )
    matrix.sort_indices()
    history_sizes = np.diff(matrix.indptr).astype(np.float64, copy=False)
    active_row_by_customer = {
        int(customer_id): row for row, customer_id in enumerate(active_customer_ids)
    }
    active_source_globals = [
        int(customer_id)
        for customer_id in sources
        if int(customer_id) in active_row_by_customer
    ]
    source_rows = np.array(
        [active_row_by_customer[customer_id] for customer_id in active_source_globals],
        dtype=np.int64,
    )
    source_global_ids = np.array(active_source_globals, dtype=np.int64)
    num_sources_with_history = int(len(source_rows))

    if len(source_rows) == 0:
        elapsed = time.perf_counter() - tic
        distinct_items = np.diff(matrix.indptr)
        return _empty_snapshot(seed_time), UserCFSnapshotStats(
            seed_time=seed_time.isoformat(),
            window_start=window_start,
            window_end=seed_time.isoformat(),
            num_raw_transactions=raw_count,
            num_source_customers=int(len(sources)),
            num_source_customers_with_history=0,
            num_active_customers=int(len(active_customer_ids)),
            num_active_articles=int(np.unique(article_ids).size),
            num_unique_user_item_pairs=int(len(pairs)),
            mean_distinct_items_per_customer=float(np.mean(distinct_items))
            if len(distinct_items)
            else math.nan,
            max_distinct_items_per_customer=int(np.max(distinct_items))
            if len(distinct_items)
            else 0,
            overlap_nnz_before_filter=0,
            overlap_nnz_after_filter=0,
            num_rows=0,
            elapsed_seconds=elapsed,
            peak_resident_memory_bytes=_peak_resident_memory_bytes(),
        )

    src_chunks: list[np.ndarray] = []
    dst_chunks: list[np.ndarray] = []
    overlap_chunks: list[np.ndarray] = []
    score_chunks: list[np.ndarray] = []
    src_size_chunks: list[np.ndarray] = []
    dst_size_chunks: list[np.ndarray] = []
    rank_chunks: list[np.ndarray] = []
    nnz_before_filter = 0
    nnz_after_filter = 0

    matrix_t = matrix.T.tocsr()
    for chunk_start in range(0, len(source_rows), source_chunk_size):
        chunk_rows = source_rows[chunk_start : chunk_start + source_chunk_size]
        chunk_global = source_global_ids[chunk_start : chunk_start + source_chunk_size]
        overlap_matrix = (matrix[chunk_rows] @ matrix_t).tocsr()
        for local_idx, src_row in enumerate(chunk_rows):
            start = int(overlap_matrix.indptr[local_idx])
            end = int(overlap_matrix.indptr[local_idx + 1])
            if start == end:
                continue
            dst_rows = overlap_matrix.indices[start:end].astype(np.int64, copy=False)
            overlap = overlap_matrix.data[start:end].astype(np.int64, copy=False)
            keep = dst_rows != src_row
            if not np.any(keep):
                continue
            dst_rows = dst_rows[keep]
            overlap = overlap[keep]
            nnz_before_filter += int(len(dst_rows))
            keep = overlap >= config.min_overlap
            if not np.any(keep):
                continue
            dst_rows = dst_rows[keep]
            overlap = overlap[keep]
            nnz_after_filter += int(len(dst_rows))

            dst_global = active_customer_ids[dst_rows]
            src_size = history_sizes[src_row]
            dst_sizes = history_sizes[dst_rows]
            denom = np.power(src_size, config.alpha) * np.power(
                dst_sizes, 1.0 - config.alpha
            )
            scores = (overlap / denom).astype(np.float32, copy=False)
            order = np.lexsort((dst_global, -overlap, -scores))[: config.top_k]
            dst_global = dst_global[order]
            overlap = overlap[order]
            scores = scores[order]
            dst_sizes = dst_sizes[order].astype(np.int64, copy=False)
            ranks = np.arange(1, len(dst_global) + 1, dtype=np.int32)
            src_chunks.append(np.full(len(dst_global), chunk_global[local_idx], dtype=np.int64))
            dst_chunks.append(dst_global.astype(np.int64, copy=False))
            overlap_chunks.append(overlap.astype(np.int64, copy=False))
            score_chunks.append(scores)
            src_size_chunks.append(np.full(len(dst_global), int(src_size), dtype=np.int64))
            dst_size_chunks.append(dst_sizes)
            rank_chunks.append(ranks)

    if src_chunks:
        num_rows = sum(map(len, src_chunks))
        out = pd.DataFrame(
            {
                "seed_time": pd.Series(
                    np.repeat(seed_time.to_datetime64(), num_rows),
                    dtype="datetime64[ns]",
                ),
                "src_customer_id": np.concatenate(src_chunks).astype(np.int64),
                "dst_customer_id": np.concatenate(dst_chunks).astype(np.int64),
                "overlap": np.concatenate(overlap_chunks).astype(np.int64),
                "user_cf_score": np.concatenate(score_chunks).astype(np.float32),
                "src_history_size": np.concatenate(src_size_chunks).astype(np.int64),
                "dst_history_size": np.concatenate(dst_size_chunks).astype(np.int64),
                "rank": np.concatenate(rank_chunks).astype(np.int32),
            }
        )
    else:
        out = _empty_snapshot(seed_time)

    distinct_items = np.diff(matrix.indptr)
    elapsed = time.perf_counter() - tic
    stats = UserCFSnapshotStats(
        seed_time=seed_time.isoformat(),
        window_start=window_start,
        window_end=seed_time.isoformat(),
        num_raw_transactions=raw_count,
        num_source_customers=int(len(sources)),
        num_source_customers_with_history=num_sources_with_history,
        num_active_customers=int(len(active_customer_ids)),
        num_active_articles=int(np.unique(article_ids).size),
        num_unique_user_item_pairs=int(len(pairs)),
        mean_distinct_items_per_customer=float(np.mean(distinct_items))
        if len(distinct_items)
        else math.nan,
        max_distinct_items_per_customer=int(np.max(distinct_items))
        if len(distinct_items)
        else 0,
        overlap_nnz_before_filter=nnz_before_filter,
        overlap_nnz_after_filter=nnz_after_filter,
        num_rows=int(len(out)),
        elapsed_seconds=elapsed,
        peak_resident_memory_bytes=_peak_resident_memory_bytes(),
    )
    return out, stats


def _empty_snapshot(seed_time: pd.Timestamp) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "seed_time": pd.Series([], dtype="datetime64[ns]"),
            "src_customer_id": pd.Series([], dtype="int64"),
            "dst_customer_id": pd.Series([], dtype="int64"),
            "overlap": pd.Series([], dtype="int64"),
            "user_cf_score": pd.Series([], dtype="float32"),
            "src_history_size": pd.Series([], dtype="int64"),
            "dst_history_size": pd.Series([], dtype="int64"),
            "rank": pd.Series([], dtype="int32"),
        }
    )
