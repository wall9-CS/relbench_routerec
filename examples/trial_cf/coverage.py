from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from relbench.base import RecommendationTask, Table

from .config import TrialCFSnapshotConfig
from .io import load_snapshot

try:
    from examples.hm_cf.seed_time_loader import group_recommendation_table_by_seed_time
except ImportError:
    from ..hm_cf.seed_time_loader import group_recommendation_table_by_seed_time


@dataclass(frozen=True)
class TrialCFCoverageMetrics:
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
    metrics: Iterable[TrialCFCoverageMetrics],
) -> TrialCFCoverageMetrics:
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
    snapshot_dir: Path,
    *,
    config: TrialCFSnapshotConfig,
    num_sources: int,
    num_sponsors: int,
    num_layers: int = 2,
) -> TrialCFCoverageMetrics:
    table = task.get_table(split, mask_input_cols=False)
    return compute_cf_coverage_for_table(
        table,
        task,
        snapshot_dir,
        config=config,
        num_sources=num_sources,
        num_sponsors=num_sponsors,
        num_layers=num_layers,
    )


def compute_cf_coverage_for_table(
    table: Table,
    task: RecommendationTask,
    snapshot_dir: Path,
    *,
    config: TrialCFSnapshotConfig,
    num_sources: int,
    num_sponsors: int,
    num_layers: int = 2,
) -> TrialCFCoverageMetrics:
    """Compare labels with candidates reachable by source->trial_cf->sponsor."""
    if num_layers < 2:
        raise ValueError("Trial CF coverage requires num_layers >= 2.")
    _validate_inputs(table, task, config)

    totals = _MutableTotals()
    groups = group_recommendation_table_by_seed_time(table, task)
    for group in groups:
        snapshot = load_snapshot(
            snapshot_dir,
            group.seed_time,
            validate=True,
            config=config,
            num_sources=num_sources,
            num_sponsors=num_sponsors,
        )
        cf_candidates = _snapshot_candidate_map(snapshot)
        for _, row in group.table.df.iterrows():
            true_dst = set(row[task.dst_entity_col])
            totals.num_rows += 1
            if not true_dst:
                continue

            candidates = set(cf_candidates.get(int(row[task.src_entity_col]), ()))
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

    def freeze(self) -> TrialCFCoverageMetrics:
        return TrialCFCoverageMetrics(**asdict(self))


def _validate_inputs(
    table: Table,
    task: RecommendationTask,
    config: TrialCFSnapshotConfig,
) -> None:
    if task.src_entity_col != config.src_entity_col:
        raise ValueError(
            f"Task source column {task.src_entity_col!r} does not match "
            f"config source column {config.src_entity_col!r}."
        )
    if task.dst_entity_col != "sponsor_id":
        raise ValueError("Trial CF coverage only supports sponsor recommendation.")
    for col in [task.time_col, task.src_entity_col, task.dst_entity_col]:
        if col not in table.df.columns:
            raise ValueError(f"Task table is missing column: {col}")


def _snapshot_candidate_map(snapshot: pd.DataFrame) -> dict[int, tuple[int, ...]]:
    if len(snapshot) == 0:
        return {}
    return {
        int(src): tuple(int(dst) for dst in group["dst_sponsor_id"].to_numpy())
        for src, group in snapshot.groupby("src_id", sort=False)
    }


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

