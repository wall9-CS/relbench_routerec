from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd

from relbench.base import RecommendationTask, Table

from .adapters import RecommendationTaskAdapter
from .io import load_snapshot


@dataclass
class QueryCoverage:
    split: str
    row_index: int
    seed_time: str
    source_id: int
    num_ground_truth: int
    num_ground_truth_in_candidate_subgraph: int
    candidate_count: int
    already_local: int
    newly_reached_by_latent_relation: int
    still_unreachable: int

    @property
    def coverage(self) -> float:
        if self.num_ground_truth == 0:
            return float("nan")
        return self.num_ground_truth_in_candidate_subgraph / self.num_ground_truth


@dataclass
class CoverageAggregate:
    num_queries: int
    num_ground_truth: int
    num_covered_ground_truth: int
    baseline_ground_truth: int
    newly_reached_ground_truth: int
    still_unreachable_ground_truth: int
    avg_candidate_count: float
    mean_coverage: float

    @property
    def baseline_locality(self) -> float:
        if self.num_ground_truth == 0:
            return float("nan")
        return self.baseline_ground_truth / self.num_ground_truth

    @property
    def latent_locality(self) -> float:
        if self.num_ground_truth == 0:
            return float("nan")
        return self.num_covered_ground_truth / self.num_ground_truth

    @property
    def absolute_locality_gain(self) -> float:
        return self.latent_locality - self.baseline_locality

    @property
    def relative_locality_gain(self) -> float:
        base = self.baseline_locality
        if base == 0 or pd.isna(base):
            return float("nan")
        return self.absolute_locality_gain / base

    def to_dict(self) -> dict:
        out = asdict(self)
        out["baseline_locality"] = self.baseline_locality
        out["latent_locality"] = self.latent_locality
        out["absolute_locality_gain"] = self.absolute_locality_gain
        out["relative_locality_gain"] = self.relative_locality_gain
        return out


def snapshot_candidate_map(snapshot: pd.DataFrame) -> dict[int, tuple[int, ...]]:
    if len(snapshot) == 0:
        return {}
    return {
        int(src): tuple(int(dst) for dst in group["dst_id"].to_numpy())
        for src, group in snapshot.groupby("src_id", sort=False)
    }


def compute_latent_coverage_for_table(
    table: Table,
    task: RecommendationTask,
    adapter: RecommendationTaskAdapter,
    snapshot_dir: Path,
    *,
    split: str,
    history_limit: int,
    include_history_candidates: bool = True,
) -> list[QueryCoverage]:
    rows = adapter.examples_from_table(table)
    out: list[QueryCoverage] = []
    snapshot_cache: dict[pd.Timestamp, dict[int, tuple[int, ...]]] = {}
    for example in rows:
        positives = set(example.positives)
        history = set(
            adapter.history_for_source(
                example.source_id,
                example.seed_time,
                history_limit=history_limit,
            )
        )
        if example.seed_time not in snapshot_cache:
            snapshot = load_snapshot(
                snapshot_dir,
                example.seed_time,
                validate=True,
                num_destinations=adapter.num_destinations,
            )
            snapshot_cache[example.seed_time] = snapshot_candidate_map(snapshot)
        latent_map = snapshot_cache[example.seed_time]
        latent: set[int] = set()
        for anchor in history:
            latent.update(latent_map.get(anchor, ()))
        candidates = set(latent)
        if include_history_candidates:
            candidates.update(history)
        already = len(positives.intersection(history))
        newly = len(positives.difference(history).intersection(latent))
        covered = len(positives.intersection(candidates))
        still = len(positives) - covered
        out.append(
            QueryCoverage(
                split=split,
                row_index=example.row_index,
                seed_time=example.seed_time.isoformat(),
                source_id=example.source_id,
                num_ground_truth=len(positives),
                num_ground_truth_in_candidate_subgraph=covered,
                candidate_count=len(candidates),
                already_local=already,
                newly_reached_by_latent_relation=newly,
                still_unreachable=still,
            )
        )
    return out


def aggregate_query_coverage(rows: Iterable[QueryCoverage]) -> CoverageAggregate:
    items = list(rows)
    num_gt = sum(row.num_ground_truth for row in items)
    num_covered = sum(row.num_ground_truth_in_candidate_subgraph for row in items)
    baseline = sum(row.already_local for row in items)
    newly = sum(row.newly_reached_by_latent_relation for row in items)
    still = sum(row.still_unreachable for row in items)
    candidate_total = sum(row.candidate_count for row in items)
    coverage_values = [
        row.coverage for row in items if row.num_ground_truth > 0 and not pd.isna(row.coverage)
    ]
    return CoverageAggregate(
        num_queries=len(items),
        num_ground_truth=num_gt,
        num_covered_ground_truth=num_covered,
        baseline_ground_truth=baseline,
        newly_reached_ground_truth=newly,
        still_unreachable_ground_truth=still,
        avg_candidate_count=candidate_total / max(len(items), 1),
        mean_coverage=sum(coverage_values) / max(len(coverage_values), 1),
    )


def write_coverage_outputs(
    rows: list[QueryCoverage],
    output_json: Path | None = None,
    output_csv: Path | None = None,
) -> dict:
    aggregate = aggregate_query_coverage(rows)
    report = {
        "aggregate": aggregate.to_dict(),
        "per_query": [asdict(row) | {"coverage": row.coverage} for row in rows],
    }
    if output_json is not None:
        output_json.parent.mkdir(parents=True, exist_ok=True)
        with open(output_json, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, sort_keys=True)
            f.write("\n")
    if output_csv is not None:
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        with open(output_csv, "w", encoding="utf-8", newline="") as f:
            fieldnames = (
                list(report["per_query"][0].keys())
                if report["per_query"]
                else [
                    "split",
                    "row_index",
                    "seed_time",
                    "source_id",
                    "num_ground_truth",
                    "num_ground_truth_in_candidate_subgraph",
                    "candidate_count",
                    "already_local",
                    "newly_reached_by_latent_relation",
                    "still_unreachable",
                    "coverage",
                ]
            )
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(report["per_query"])
    return report
