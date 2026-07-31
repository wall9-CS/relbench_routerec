from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from relbench.base import RecommendationTask, Table

from .config import CFSnapshotConfig
from .io import load_snapshot
from .seed_time_loader import group_recommendation_table_by_seed_time


@dataclass(frozen=True)
class CFCoverageMetrics:
    """Coverage upper-bound diagnostics for a CF-augmented source subgraph."""

    num_rows: int
    num_positive_rows: int
    num_rows_with_candidates: int
    num_rows_with_hit: int
    num_groundtruth_labels: int
    num_covered_groundtruth_labels: int
    achievable_ap_sum: float

    @property
    def coverage_rate(self) -> float:
        if self.num_groundtruth_labels == 0:
            return float("nan")
        return self.num_covered_groundtruth_labels / self.num_groundtruth_labels

    @property
    def hit_rate(self) -> float:
        if self.num_positive_rows == 0:
            return float("nan")
        return self.num_rows_with_hit / self.num_positive_rows

    @property
    def achievable_map(self) -> float:
        if self.num_positive_rows == 0:
            return float("nan")
        return self.achievable_ap_sum / self.num_positive_rows

    @property
    def coverage_gap(self) -> float:
        achievable_map = self.achievable_map
        if np.isnan(achievable_map):
            return float("nan")
        return 1.0 - achievable_map

    def to_dict(self) -> dict[str, float | int]:
        out = asdict(self)
        out["coverage_rate"] = self.coverage_rate
        out["hit_rate"] = self.hit_rate
        out["achievable_MAP"] = self.achievable_map
        out["coverage_gap"] = self.coverage_gap
        return out


def combine_cf_coverage_metrics(
    metrics: Iterable[CFCoverageMetrics],
) -> CFCoverageMetrics:
    """Combine already-aggregated CF coverage metrics by summing counters."""
    totals = _MutableTotals()
    for metric in metrics:
        totals.num_rows += metric.num_rows
        totals.num_positive_rows += metric.num_positive_rows
        totals.num_rows_with_candidates += metric.num_rows_with_candidates
        totals.num_rows_with_hit += metric.num_rows_with_hit
        totals.num_groundtruth_labels += metric.num_groundtruth_labels
        totals.num_covered_groundtruth_labels += metric.num_covered_groundtruth_labels
        totals.achievable_ap_sum += metric.achievable_ap_sum
    return totals.freeze()


def compute_cf_coverage_for_split(
    task: RecommendationTask,
    split: str,
    transactions: pd.DataFrame,
    snapshot_dir: Path,
    *,
    config: CFSnapshotConfig,
    num_articles: int,
    num_layers: int = 4,
    include_source_articles: bool = False,
) -> CFCoverageMetrics:
    """Compute CF subgraph label coverage for one task split."""
    table = task.get_table(split, mask_input_cols=False)
    return compute_cf_coverage_for_table(
        table,
        task,
        transactions,
        snapshot_dir,
        config=config,
        num_articles=num_articles,
        num_layers=num_layers,
        include_source_articles=include_source_articles,
    )


def compute_cf_coverage_for_table(
    table: Table,
    task: RecommendationTask,
    transactions: pd.DataFrame,
    snapshot_dir: Path,
    *,
    config: CFSnapshotConfig,
    num_articles: int,
    num_layers: int = 4,
    include_source_articles: bool = False,
) -> CFCoverageMetrics:
    """Compare ground-truth labels with deterministic CF-reachable candidates.

    With the existing CF neighbor schedule, `num_layers=4` enables the route:
    source customer -> historical transaction -> src article -> article_cf row
    -> dst article. By default the candidate set contains the final CF dst
    articles on that route. Set `include_source_articles=True` to also count the
    intermediate historical src articles as candidates in the source subgraph.
    """
    if num_layers < 4:
        raise ValueError("CF coverage requires num_layers >= 4.")
    _validate_inputs(table, task, transactions)

    transactions = _sort_transactions_once(transactions)
    times = transactions["t_dat"].to_numpy(dtype="datetime64[ns]", copy=False)

    totals = _MutableTotals()
    groups = group_recommendation_table_by_seed_time(table, task)
    for group in groups:
        snapshot = load_snapshot(
            snapshot_dir,
            group.seed_time,
            validate=True,
            config=config,
            num_articles=num_articles,
        )
        cf_candidates = _snapshot_candidate_map(snapshot)
        customer_sources = _customer_source_articles(
            transactions,
            times,
            group.seed_time,
            group.table.df[task.src_entity_col].unique(),
        )
        for _, row in group.table.df.iterrows():
            true_dst = set(row[task.dst_entity_col])
            totals.num_rows += 1
            if not true_dst:
                continue

            src_articles = customer_sources.get(int(row[task.src_entity_col]), set())
            candidates: set[int] = set(src_articles) if include_source_articles else set()
            for src in src_articles:
                candidates.update(cf_candidates.get(src, ()))

            covered = len(true_dst.intersection(candidates))
            totals.num_positive_rows += 1
            totals.num_groundtruth_labels += len(true_dst)
            totals.num_covered_groundtruth_labels += covered
            totals.num_rows_with_candidates += int(bool(candidates))
            totals.num_rows_with_hit += int(covered > 0)
            totals.achievable_ap_sum += _oracle_average_precision_at_k(
                covered,
                len(true_dst),
                task.eval_k,
            )

    return totals.freeze()


@dataclass
class _MutableTotals:
    num_rows: int = 0
    num_positive_rows: int = 0
    num_rows_with_candidates: int = 0
    num_rows_with_hit: int = 0
    num_groundtruth_labels: int = 0
    num_covered_groundtruth_labels: int = 0
    achievable_ap_sum: float = 0.0

    def freeze(self) -> CFCoverageMetrics:
        return CFCoverageMetrics(**asdict(self))


def _validate_inputs(
    table: Table,
    task: RecommendationTask,
    transactions: pd.DataFrame,
) -> None:
    for col in [task.time_col, task.src_entity_col, task.dst_entity_col]:
        if col not in table.df.columns:
            raise ValueError(f"Task table is missing column: {col}")
    required_transaction_cols = {"customer_id", "article_id", "t_dat"}
    missing = required_transaction_cols.difference(transactions.columns)
    if missing:
        raise ValueError(f"transactions is missing columns: {sorted(missing)}")
    if not pd.api.types.is_datetime64_any_dtype(transactions["t_dat"]):
        raise ValueError("transactions.t_dat must be datetime dtype.")


def _sort_transactions_once(transactions: pd.DataFrame) -> pd.DataFrame:
    if transactions["t_dat"].is_monotonic_increasing:
        return transactions
    return transactions.sort_values("t_dat", kind="mergesort").reset_index(drop=True)


def _snapshot_candidate_map(snapshot: pd.DataFrame) -> dict[int, tuple[int, ...]]:
    if len(snapshot) == 0:
        return {}
    return {
        int(src): tuple(int(dst) for dst in group["dst_article_id"].to_numpy())
        for src, group in snapshot.groupby("src_article_id", sort=False)
    }


def _customer_source_articles(
    transactions: pd.DataFrame,
    times: np.ndarray,
    seed_time: pd.Timestamp,
    customer_ids,
) -> dict[int, set[int]]:
    seed_np = np.datetime64(pd.Timestamp(seed_time).to_datetime64(), "ns")
    right = int(np.searchsorted(times, seed_np, side="right"))
    if right == 0:
        return {}

    customer_ids = {int(value) for value in customer_ids}
    history = transactions.iloc[:right][["customer_id", "article_id"]]
    history = history[history["customer_id"].isin(customer_ids)].drop_duplicates()
    out: dict[int, set[int]] = {}
    for customer_id, group in history.groupby("customer_id", sort=False):
        out[int(customer_id)] = {int(article_id) for article_id in group["article_id"]}
    return out


def _oracle_average_precision_at_k(
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
