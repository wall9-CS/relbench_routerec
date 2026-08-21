from __future__ import annotations

import math
import os
import time
from dataclasses import asdict, dataclass
from typing import Iterable

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class DirectCFColumns:
    interaction_src_col: str
    interaction_item_col: str
    interaction_time_col: str
    item_cf_src_col: str
    item_cf_dst_col: str
    out_src_col: str
    out_dst_col: str
    num_source_items_col: str
    best_source_item_col: str


@dataclass(frozen=True)
class DirectCFSnapshotStats:
    seed_time: str
    window_end: str
    num_raw_interactions: int
    num_source_nodes: int
    num_source_nodes_with_history: int
    num_unique_source_item_pairs: int
    num_parent_cf_rows: int
    num_joined_rows: int
    num_collapsed_pairs_before_topk: int
    num_rows: int
    elapsed_seconds: float
    peak_resident_memory_bytes: int | None = None

    def to_manifest_entry(self, file: str, file_size_bytes: int | None = None) -> dict:
        out = asdict(self)
        out["file"] = file
        if file_size_bytes is not None:
            out["file_size_bytes"] = file_size_bytes
        return out


def peak_resident_memory_bytes() -> int | None:
    try:
        import resource

        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return int(rss if os.name == "nt" else rss * 1024)
    except Exception:
        return None


def empty_direct_snapshot(seed_time: pd.Timestamp, columns: DirectCFColumns) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "seed_time": pd.Series([], dtype="datetime64[ns]"),
            columns.out_src_col: pd.Series([], dtype="int64"),
            columns.out_dst_col: pd.Series([], dtype="int64"),
            "direct_score": pd.Series([], dtype="float32"),
            "max_cf_score": pd.Series([], dtype="float32"),
            "support_sum": pd.Series([], dtype="int64"),
            "support_max": pd.Series([], dtype="int64"),
            columns.num_source_items_col: pd.Series([], dtype="int64"),
            columns.best_source_item_col: pd.Series([], dtype="int64"),
            "best_item_cf_rank": pd.Series([], dtype="int32"),
            "rank": pd.Series([], dtype="int32"),
        }
    )


def required_direct_columns(columns: DirectCFColumns) -> list[str]:
    return [
        "seed_time",
        columns.out_src_col,
        columns.out_dst_col,
        "direct_score",
        "max_cf_score",
        "support_sum",
        "support_max",
        columns.num_source_items_col,
        columns.best_source_item_col,
        "best_item_cf_rank",
        "rank",
    ]


def build_direct_snapshot(
    interactions: pd.DataFrame,
    item_cf_snapshot: pd.DataFrame,
    seed_time: pd.Timestamp,
    source_ids: Iterable[int],
    *,
    columns: DirectCFColumns,
    direct_top_k: int,
    filter_seen_dst: bool = False,
) -> tuple[pd.DataFrame, DirectCFSnapshotStats]:
    """Build one direct source-to-item CF snapshot from a parent item-CF snapshot."""
    if direct_top_k < 1:
        raise ValueError("direct_top_k must be at least 1.")
    tic = time.perf_counter()
    seed_time = pd.Timestamp(seed_time)
    _validate_inputs(interactions, item_cf_snapshot, columns)

    sources = _normalize_source_ids(source_ids)
    interactions = _sort_interactions_once(interactions, columns.interaction_time_col)
    times = interactions[columns.interaction_time_col].to_numpy(
        dtype="datetime64[ns]", copy=False
    )
    seed_np = np.datetime64(seed_time.to_datetime64(), "ns")
    right = int(np.searchsorted(times, seed_np, side="right"))
    history = interactions.iloc[:right][
        [columns.interaction_src_col, columns.interaction_item_col]
    ]
    raw_count = int(right)
    if len(sources) == 0:
        history = history.iloc[0:0]
    else:
        history = history[history[columns.interaction_src_col].isin(sources)]
    history = history.drop_duplicates(ignore_index=True)
    source_nodes_with_history = int(history[columns.interaction_src_col].nunique())

    if len(history) == 0 or len(item_cf_snapshot) == 0 or len(sources) == 0:
        elapsed = time.perf_counter() - tic
        return empty_direct_snapshot(seed_time, columns), DirectCFSnapshotStats(
            seed_time=seed_time.isoformat(),
            window_end=seed_time.isoformat(),
            num_raw_interactions=raw_count,
            num_source_nodes=int(len(sources)),
            num_source_nodes_with_history=source_nodes_with_history,
            num_unique_source_item_pairs=int(len(history)),
            num_parent_cf_rows=int(len(item_cf_snapshot)),
            num_joined_rows=0,
            num_collapsed_pairs_before_topk=0,
            num_rows=0,
            elapsed_seconds=elapsed,
            peak_resident_memory_bytes=peak_resident_memory_bytes(),
        )

    history_frame = history.rename(
        columns={
            columns.interaction_src_col: columns.out_src_col,
            columns.interaction_item_col: "_src_item",
        }
    )
    cf_frame = item_cf_snapshot[
        [
            columns.item_cf_src_col,
            columns.item_cf_dst_col,
            "support",
            "cf_score",
            "rank",
        ]
    ].rename(
        columns={
            columns.item_cf_src_col: "_src_item",
            columns.item_cf_dst_col: columns.out_dst_col,
            "rank": "item_cf_rank",
        }
    )
    joined = history_frame.merge(cf_frame, on="_src_item", how="inner", sort=False)

    if filter_seen_dst and len(joined) > 0:
        seen = pd.MultiIndex.from_frame(
            history_frame[[columns.out_src_col, "_src_item"]].rename(
                columns={"_src_item": columns.out_dst_col}
            )
        )
        joined_index = pd.MultiIndex.from_frame(
            joined[[columns.out_src_col, columns.out_dst_col]]
        )
        joined = joined.loc[~joined_index.isin(seen)].reset_index(drop=True)

    if len(joined) == 0:
        elapsed = time.perf_counter() - tic
        return empty_direct_snapshot(seed_time, columns), DirectCFSnapshotStats(
            seed_time=seed_time.isoformat(),
            window_end=seed_time.isoformat(),
            num_raw_interactions=raw_count,
            num_source_nodes=int(len(sources)),
            num_source_nodes_with_history=source_nodes_with_history,
            num_unique_source_item_pairs=int(len(history)),
            num_parent_cf_rows=int(len(item_cf_snapshot)),
            num_joined_rows=0,
            num_collapsed_pairs_before_topk=0,
            num_rows=0,
            elapsed_seconds=elapsed,
            peak_resident_memory_bytes=peak_resident_memory_bytes(),
        )

    group_cols = [columns.out_src_col, columns.out_dst_col]
    aggregate = (
        joined.groupby(group_cols, sort=False)
        .agg(
            direct_score=("cf_score", "sum"),
            max_cf_score=("cf_score", "max"),
            support_sum=("support", "sum"),
            support_max=("support", "max"),
            **{columns.num_source_items_col: ("_src_item", "nunique")},
        )
        .reset_index()
    )
    collapsed_pairs = int(len(aggregate))

    best = (
        joined.sort_values(
            group_cols + ["cf_score", "support", "item_cf_rank", "_src_item"],
            ascending=[True, True, False, False, True, True],
            kind="mergesort",
        )
        .drop_duplicates(group_cols, keep="first")
        [group_cols + ["_src_item", "item_cf_rank"]]
        .rename(
            columns={
                "_src_item": columns.best_source_item_col,
                "item_cf_rank": "best_item_cf_rank",
            }
        )
    )
    out = aggregate.merge(best, on=group_cols, how="left", sort=False)
    out = out.sort_values(
        [
            columns.out_src_col,
            "direct_score",
            columns.num_source_items_col,
            "max_cf_score",
            "support_sum",
            columns.out_dst_col,
        ],
        ascending=[True, False, False, False, False, True],
        kind="mergesort",
    )
    out["rank"] = out.groupby(columns.out_src_col, sort=False).cumcount() + 1
    out = out[out["rank"] <= direct_top_k].reset_index(drop=True)
    out.insert(
        0,
        "seed_time",
        pd.Series(
            np.repeat(seed_time.to_datetime64(), len(out)),
            dtype="datetime64[ns]",
        ),
    )
    out = out[required_direct_columns(columns)]
    out = out.astype(
        {
            columns.out_src_col: "int64",
            columns.out_dst_col: "int64",
            "direct_score": "float32",
            "max_cf_score": "float32",
            "support_sum": "int64",
            "support_max": "int64",
            columns.num_source_items_col: "int64",
            columns.best_source_item_col: "int64",
            "best_item_cf_rank": "int32",
            "rank": "int32",
        }
    )

    elapsed = time.perf_counter() - tic
    stats = DirectCFSnapshotStats(
        seed_time=seed_time.isoformat(),
        window_end=seed_time.isoformat(),
        num_raw_interactions=raw_count,
        num_source_nodes=int(len(sources)),
        num_source_nodes_with_history=source_nodes_with_history,
        num_unique_source_item_pairs=int(len(history)),
        num_parent_cf_rows=int(len(item_cf_snapshot)),
        num_joined_rows=int(len(joined)),
        num_collapsed_pairs_before_topk=collapsed_pairs,
        num_rows=int(len(out)),
        elapsed_seconds=elapsed,
        peak_resident_memory_bytes=peak_resident_memory_bytes(),
    )
    return out, stats


def _validate_inputs(
    interactions: pd.DataFrame,
    item_cf_snapshot: pd.DataFrame,
    columns: DirectCFColumns,
) -> None:
    required_interaction = {
        columns.interaction_src_col,
        columns.interaction_item_col,
        columns.interaction_time_col,
    }
    missing = required_interaction.difference(interactions.columns)
    if missing:
        raise ValueError(f"interactions is missing columns: {sorted(missing)}")
    if not pd.api.types.is_datetime64_any_dtype(
        interactions[columns.interaction_time_col]
    ):
        raise ValueError(
            f"interactions.{columns.interaction_time_col} must be datetime dtype."
        )
    for col in [columns.interaction_src_col, columns.interaction_item_col]:
        if not pd.api.types.is_integer_dtype(interactions[col]):
            raise ValueError(f"interactions.{col} must be integer-like.")
        if (interactions[col] < 0).any():
            raise ValueError(f"interactions.{col} must be nonnegative.")

    required_cf = {
        columns.item_cf_src_col,
        columns.item_cf_dst_col,
        "support",
        "cf_score",
        "rank",
    }
    missing_cf = required_cf.difference(item_cf_snapshot.columns)
    if missing_cf:
        raise ValueError(f"item_cf_snapshot is missing columns: {sorted(missing_cf)}")


def _normalize_source_ids(source_ids: Iterable[int]) -> np.ndarray:
    values = np.fromiter((int(value) for value in source_ids), dtype=np.int64)
    if len(values) == 0:
        return values
    if values.min() < 0:
        raise ValueError("source_ids must be nonnegative.")
    return np.array(sorted(set(values.tolist())), dtype=np.int64)


def _sort_interactions_once(interactions: pd.DataFrame, time_col: str) -> pd.DataFrame:
    if interactions[time_col].is_monotonic_increasing:
        return interactions
    return interactions.sort_values(time_col, kind="mergesort").reset_index(drop=True)


def oracle_average_precision_at_k(
    num_covered: int,
    num_groundtruth: int,
    eval_k: int,
) -> float:
    if num_groundtruth == 0:
        return 0.0
    denominator = min(num_groundtruth, eval_k)
    if denominator == 0:
        return 0.0
    return min(num_covered, eval_k) / denominator
