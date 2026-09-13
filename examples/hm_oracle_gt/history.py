from __future__ import annotations

import time
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
from scipy import sparse

from .config import OracleGTPhase1Config, snapshot_id


@dataclass(frozen=True)
class HistoricalSnapshotStats:
    snapshot_id: str
    seed_time: str
    window_start: str | None
    window_end: str
    num_raw_transactions: int
    num_historical_interactions: int
    num_active_users: int
    num_active_items: int
    num_global_users: int
    num_global_items: int
    max_interaction_time: str | None
    min_interaction_time: str | None
    temporal_boundary_ok: bool
    elapsed_seconds: float

    def to_dict(self) -> dict:
        return asdict(self)


def validate_transactions(
    transactions: pd.DataFrame,
    *,
    num_users: int,
    num_items: int,
) -> None:
    required = {"customer_id", "article_id", "t_dat"}
    missing = required.difference(transactions.columns)
    if missing:
        raise ValueError(f"transactions is missing columns: {sorted(missing)}")
    if not pd.api.types.is_datetime64_any_dtype(transactions["t_dat"]):
        raise ValueError("transactions.t_dat must be datetime dtype.")
    for col, upper in [("customer_id", num_users), ("article_id", num_items)]:
        if not pd.api.types.is_integer_dtype(transactions[col]):
            raise ValueError(f"transactions.{col} must be integer-like.")
        if (transactions[col] < 0).any():
            raise ValueError(f"transactions.{col} must be nonnegative.")
        if len(transactions) > 0 and transactions[col].max() >= upper:
            raise ValueError(f"transactions.{col} contains out-of-range IDs.")


def sort_transactions_once(transactions: pd.DataFrame) -> pd.DataFrame:
    if transactions["t_dat"].is_monotonic_increasing:
        return transactions
    return transactions.sort_values("t_dat", kind="mergesort").reset_index(drop=True)


def slice_transactions_for_snapshot(
    transactions: pd.DataFrame,
    seed_time: pd.Timestamp,
    config: OracleGTPhase1Config,
) -> tuple[pd.DataFrame, pd.Timestamp | None, pd.Timestamp]:
    transactions = sort_transactions_once(transactions)
    times = transactions["t_dat"].to_numpy(dtype="datetime64[ns]", copy=False)
    seed_time = pd.Timestamp(seed_time)
    seed_np = np.datetime64(seed_time.to_datetime64(), "ns")

    if config.all_history:
        left = 0
        window_start = None
    else:
        window_start = seed_time - pd.Timedelta(weeks=config.window_weeks)
        left = int(
            np.searchsorted(
                times,
                np.datetime64(window_start.to_datetime64(), "ns"),
                side="right",
            )
        )
    right = int(np.searchsorted(times, seed_np, side="right"))
    return transactions.iloc[left:right], window_start, seed_time


def build_historical_interaction_matrix(
    transactions: pd.DataFrame,
    seed_time: pd.Timestamp,
    *,
    num_users: int,
    num_items: int,
    config: OracleGTPhase1Config,
) -> tuple[sparse.csr_matrix, np.ndarray, HistoricalSnapshotStats]:
    """Build one sparse binary active-user x global-item history matrix.

    This intentionally mirrors the existing H&M CF window slicing:
    `(seed_time - window, seed_time]` or all history up to `seed_time`.
    Duplicate `(customer_id, article_id)` rows collapse to one binary entry.
    """
    tic = time.perf_counter()
    seed_time = pd.Timestamp(seed_time)
    validate_transactions(transactions, num_users=num_users, num_items=num_items)
    window, window_start, window_end = slice_transactions_for_snapshot(
        transactions,
        seed_time,
        config,
    )

    pairs = window[["customer_id", "article_id"]].drop_duplicates(ignore_index=True)
    if len(pairs) == 0:
        matrix = sparse.csr_matrix((0, num_items), dtype=np.int8)
        active_user_ids = np.array([], dtype=np.int64)
        min_time = max_time = None
        temporal_ok = True
    else:
        user_codes, active_users = pd.factorize(pairs["customer_id"], sort=False)
        active_user_ids = active_users.to_numpy(dtype=np.int64, copy=False)
        item_ids = pairs["article_id"].to_numpy(dtype=np.int64, copy=False)
        values = np.ones(len(pairs), dtype=np.int8)
        matrix = sparse.csr_matrix(
            (values, (user_codes.astype(np.int64, copy=False), item_ids)),
            shape=(len(active_user_ids), num_items),
            dtype=np.int8,
        )
        matrix.sort_indices()
        min_time = pd.Timestamp(window["t_dat"].min())
        max_time = pd.Timestamp(window["t_dat"].max())
        temporal_ok = max_time <= seed_time
        if window_start is not None:
            temporal_ok = temporal_ok and bool((window["t_dat"] > window_start).all())

    elapsed = time.perf_counter() - tic
    stats = HistoricalSnapshotStats(
        snapshot_id=snapshot_id(seed_time),
        seed_time=seed_time.isoformat(),
        window_start=None if window_start is None else window_start.isoformat(),
        window_end=window_end.isoformat(),
        num_raw_transactions=int(len(window)),
        num_historical_interactions=int(len(pairs)),
        num_active_users=int(len(active_user_ids)),
        num_active_items=int(pairs["article_id"].nunique()) if len(pairs) else 0,
        num_global_users=int(num_users),
        num_global_items=int(num_items),
        max_interaction_time=None if max_time is None else max_time.isoformat(),
        min_interaction_time=None if min_time is None else min_time.isoformat(),
        temporal_boundary_ok=bool(temporal_ok),
        elapsed_seconds=float(elapsed),
    )
    if not stats.temporal_boundary_ok:
        raise ValueError(
            f"Snapshot {stats.snapshot_id} contains interactions outside the "
            f"allowed historical window."
        )
    return matrix, active_user_ids, stats

