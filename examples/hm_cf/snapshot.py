from __future__ import annotations

import math
import os
import time
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
from scipy import sparse

from .config import CFSnapshotConfig


@dataclass(frozen=True)
class SnapshotStats:
    seed_time: str
    window_start: str | None
    window_end: str
    num_raw_transactions: int
    num_active_customers: int
    num_active_articles: int
    num_unique_user_item_pairs: int
    mean_distinct_items_per_customer: float
    max_distinct_items_per_customer: int
    estimated_pair_contributions: int
    cooccurrence_nnz_before_support: int
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


def _validate_transactions(transactions: pd.DataFrame, num_articles: int) -> None:
    required = {"customer_id", "article_id", "t_dat"}
    missing = required.difference(transactions.columns)
    if missing:
        raise ValueError(f"transactions is missing columns: {sorted(missing)}")
    if not pd.api.types.is_datetime64_any_dtype(transactions["t_dat"]):
        raise ValueError("transactions.t_dat must be datetime dtype.")
    for col in ["customer_id", "article_id"]:
        if not pd.api.types.is_integer_dtype(transactions[col]):
            raise ValueError(f"transactions.{col} must be integer-like.")
        if (transactions[col] < 0).any():
            raise ValueError(f"transactions.{col} must be nonnegative.")
    if len(transactions) > 0 and transactions["article_id"].max() >= num_articles:
        raise ValueError("transactions.article_id contains an out-of-range article id.")


def _sort_transactions_once(transactions: pd.DataFrame) -> pd.DataFrame:
    if transactions["t_dat"].is_monotonic_increasing:
        return transactions
    return transactions.sort_values("t_dat", kind="mergesort").reset_index(drop=True)


def build_snapshot(
    transactions: pd.DataFrame,
    seed_time: pd.Timestamp,
    num_articles: int,
    config: CFSnapshotConfig,
) -> tuple[pd.DataFrame, SnapshotStats]:
    """Build one sparse CF table for the exact recommendation seed time.

    The transaction window is `(seed_time - window, seed_time]` unless
    `config.all_history` is enabled, in which case all transactions up to and
    including `seed_time` are used.
    """
    tic = time.perf_counter()
    seed_time = pd.Timestamp(seed_time)
    _validate_transactions(transactions, num_articles)
    transactions = _sort_transactions_once(transactions)
    times = transactions["t_dat"].to_numpy(dtype="datetime64[ns]", copy=False)
    seed_np = np.datetime64(seed_time.to_datetime64(), "ns")

    if config.all_history:
        left = 0
        window_start = None
    else:
        window_delta = pd.Timedelta(weeks=config.window_weeks)
        lower = seed_time - window_delta
        left = int(np.searchsorted(times, np.datetime64(lower.to_datetime64(), "ns"), side="right"))
        window_start = lower.isoformat()
    right = int(np.searchsorted(times, seed_np, side="right"))
    window = transactions.iloc[left:right][["customer_id", "article_id"]]
    raw_count = int(right - left)

    pairs = window.drop_duplicates(ignore_index=True)
    if len(pairs) == 0:
        elapsed = time.perf_counter() - tic
        empty = _empty_snapshot(seed_time)
        return empty, SnapshotStats(
            seed_time=seed_time.isoformat(),
            window_start=window_start,
            window_end=seed_time.isoformat(),
            num_raw_transactions=raw_count,
            num_active_customers=0,
            num_active_articles=0,
            num_unique_user_item_pairs=0,
            mean_distinct_items_per_customer=0.0,
            max_distinct_items_per_customer=0,
            estimated_pair_contributions=0,
            cooccurrence_nnz_before_support=0,
            cooccurrence_nnz_after_support=0,
            num_rows=0,
            elapsed_seconds=elapsed,
            peak_resident_memory_bytes=_peak_resident_memory_bytes(),
        )

    customer_codes, _ = pd.factorize(pairs["customer_id"], sort=False)
    article_ids = pairs["article_id"].to_numpy(dtype=np.int64, copy=False)
    active_customers = int(customer_codes.max()) + 1
    values = np.ones(len(pairs), dtype=np.int32)
    matrix = sparse.csr_matrix(
        (values, (customer_codes.astype(np.int64, copy=False), article_ids)),
        shape=(active_customers, num_articles),
        dtype=np.int32,
    )
    cooc = (matrix.T @ matrix).tocsr()
    item_counts = np.asarray(cooc.diagonal(), dtype=np.float64)
    cooc.setdiag(0)
    cooc.eliminate_zeros()
    nnz_before_support = int(cooc.nnz)

    distinct_items = np.diff(matrix.indptr)
    pair_contrib = int(np.sum(distinct_items.astype(np.int64) * (distinct_items - 1)))

    src_chunks: list[np.ndarray] = []
    dst_chunks: list[np.ndarray] = []
    support_chunks: list[np.ndarray] = []
    score_chunks: list[np.ndarray] = []
    rank_chunks: list[np.ndarray] = []
    nnz_after_support = 0

    for src in range(num_articles):
        start, end = int(cooc.indptr[src]), int(cooc.indptr[src + 1])
        if start == end:
            continue
        dst = cooc.indices[start:end].astype(np.int64, copy=False)
        support = cooc.data[start:end].astype(np.int64, copy=False)
        keep = support >= config.min_support
        if not np.any(keep):
            continue
        dst = dst[keep]
        support = support[keep]
        nnz_after_support += int(len(dst))
        denom = np.power(item_counts[src], config.alpha) * np.power(
            item_counts[dst], 1.0 - config.alpha
        )
        scores = (support / denom).astype(np.float32, copy=False)
        order = np.lexsort((dst, -support, -scores))
        order = order[: config.top_l]
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
        out = pd.DataFrame(
            {
                "seed_time": pd.Series(
                    np.repeat(seed_time.to_datetime64(), sum(map(len, src_chunks))),
                    dtype="datetime64[ns]",
                ),
                "src_article_id": np.concatenate(src_chunks).astype(np.int64),
                "dst_article_id": np.concatenate(dst_chunks).astype(np.int64),
                "support": np.concatenate(support_chunks).astype(np.int64),
                "cf_score": np.concatenate(score_chunks).astype(np.float32),
                "rank": np.concatenate(rank_chunks).astype(np.int32),
            }
        )
    else:
        out = _empty_snapshot(seed_time)

    elapsed = time.perf_counter() - tic
    stats = SnapshotStats(
        seed_time=seed_time.isoformat(),
        window_start=window_start,
        window_end=seed_time.isoformat(),
        num_raw_transactions=raw_count,
        num_active_customers=active_customers,
        num_active_articles=int(np.unique(article_ids).size),
        num_unique_user_item_pairs=int(len(pairs)),
        mean_distinct_items_per_customer=float(np.mean(distinct_items))
        if len(distinct_items)
        else math.nan,
        max_distinct_items_per_customer=int(np.max(distinct_items))
        if len(distinct_items)
        else 0,
        estimated_pair_contributions=pair_contrib,
        cooccurrence_nnz_before_support=nnz_before_support,
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
            "src_article_id": pd.Series([], dtype="int64"),
            "dst_article_id": pd.Series([], dtype="int64"),
            "support": pd.Series([], dtype="int64"),
            "cf_score": pd.Series([], dtype="float32"),
            "rank": pd.Series([], dtype="int32"),
        }
    )
