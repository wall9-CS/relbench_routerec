from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from relbench.base import RecommendationTask, Table

from examples.hm_cf.seed_time_loader import group_recommendation_table_by_seed_time

from .common import oracle_average_precision_at_k
from .config import HMDirectCFSnapshotConfig
from .io import load_snapshot


@dataclass(frozen=True)
class HMDirectCFCoverageMetrics:
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
        value = self.achievable_map
        if np.isnan(value):
            return float("nan")
        return 1.0 - value

    def to_dict(self) -> dict[str, float | int]:
        out = asdict(self)
        out["coverage_rate"] = self.coverage_rate
        out["hit_rate"] = self.hit_rate
        out["achievable_MAP"] = self.achievable_map
        out["coverage_gap"] = self.coverage_gap
        return out


def combine_direct_cf_coverage_metrics(
    metrics: Iterable[HMDirectCFCoverageMetrics],
) -> HMDirectCFCoverageMetrics:
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


def compute_direct_cf_coverage_for_split(
    task: RecommendationTask,
    split: str,
    snapshot_dir: Path,
    *,
    config: HMDirectCFSnapshotConfig,
    num_customers: int,
    num_articles: int,
    num_layers: int = 2,
) -> HMDirectCFCoverageMetrics:
    table = task.get_table(split, mask_input_cols=False)
    return compute_direct_cf_coverage_for_table(
        table,
        task,
        snapshot_dir,
        config=config,
        num_customers=num_customers,
        num_articles=num_articles,
        num_layers=num_layers,
    )


def compute_direct_cf_coverage_for_table(
    table: Table,
    task: RecommendationTask,
    snapshot_dir: Path,
    *,
    config: HMDirectCFSnapshotConfig,
    num_customers: int,
    num_articles: int,
    num_layers: int = 2,
) -> HMDirectCFCoverageMetrics:
    if num_layers < 2:
        raise ValueError("HM direct-CF coverage requires num_layers >= 2.")
    totals = _MutableTotals()
    for group in group_recommendation_table_by_seed_time(table, task):
        snapshot = load_snapshot(
            snapshot_dir,
            group.seed_time,
            validate=True,
            config=config,
            num_customers=num_customers,
            num_articles=num_articles,
        )
        candidates = _snapshot_candidate_map(snapshot)
        for _, row in group.table.df.iterrows():
            true_dst = set(row[task.dst_entity_col])
            totals.num_rows += 1
            if not true_dst:
                continue
            src_candidates = set(candidates.get(int(row[task.src_entity_col]), ()))
            covered = len(true_dst.intersection(src_candidates))
            totals.num_positive_rows += 1
            totals.num_groundtruth_labels += len(true_dst)
            totals.num_covered_groundtruth_labels += covered
            totals.num_rows_with_candidates += int(bool(src_candidates))
            totals.num_rows_with_hit += int(covered > 0)
            totals.achievable_ap_sum += oracle_average_precision_at_k(
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

    def freeze(self) -> HMDirectCFCoverageMetrics:
        return HMDirectCFCoverageMetrics(**asdict(self))


def _snapshot_candidate_map(snapshot: pd.DataFrame) -> dict[int, tuple[int, ...]]:
    if len(snapshot) == 0:
        return {}
    return {
        int(src): tuple(int(dst) for dst in group["dst_article_id"].to_numpy())
        for src, group in snapshot.groupby("src_customer_id", sort=False)
    }
