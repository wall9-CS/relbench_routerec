from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import HeteroData
from torch_geometric.utils import sort_edge_index

from relbench.base import RecommendationTask, Table
from relbench.datasets import get_dataset
from relbench.modeling.utils import to_unix_time
from relbench.tasks import get_task

from .avito_cf.graph import (
    attach_cf_snapshot as attach_avito_item_cf_snapshot,
    build_cf_num_neighbors as build_avito_item_cf_num_neighbors,
)
from .avito_cf.interactions import filter_visit_interactions
from .avito_cf.io import load_snapshot as load_avito_item_cf_snapshot
from .avito_cf.io import validate_manifest_for_training as validate_avito_item_cf
from .avito_user_cf.graph import (
    attach_user_cf_snapshot as attach_avito_user_cf_snapshot,
    build_user_cf_num_neighbors as build_avito_user_cf_num_neighbors,
)
from .avito_user_cf.io import load_snapshot as load_avito_user_cf_snapshot
from .avito_user_cf.io import validate_manifest_for_training as validate_avito_user_cf
from .hm_cf.graph import (
    attach_cf_snapshot as attach_hm_item_cf_snapshot,
    build_cf_num_neighbors as build_hm_item_cf_num_neighbors,
)
from .hm_cf.io import load_snapshot as load_hm_item_cf_snapshot
from .hm_cf.io import validate_manifest_for_training as validate_hm_item_cf
from .hm_cf.seed_time_loader import (
    group_recommendation_table_by_seed_time,
    make_seed_time_loader,
)
from .hm_user_cf.graph import (
    attach_user_cf_snapshot as attach_hm_user_cf_snapshot,
    build_user_cf_num_neighbors as build_hm_user_cf_num_neighbors,
)
from .hm_user_cf.io import load_snapshot as load_hm_user_cf_snapshot
from .hm_user_cf.io import validate_manifest_for_training as validate_hm_user_cf
from .stack_cf.graph import (
    attach_cf_snapshot as attach_stack_item_cf_snapshot,
    build_cf_num_neighbors as build_stack_item_cf_num_neighbors,
)
from .stack_cf.interactions import filter_comment_interactions
from .stack_cf.io import load_snapshot as load_stack_item_cf_snapshot
from .stack_cf.io import validate_manifest_for_training as validate_stack_item_cf
from .stack_user_cf.graph import (
    attach_user_cf_snapshot as attach_stack_user_cf_snapshot,
    build_user_cf_num_neighbors as build_stack_user_cf_num_neighbors,
)
from .stack_user_cf.io import load_snapshot as load_stack_user_cf_snapshot
from .stack_user_cf.io import validate_manifest_for_training as validate_stack_user_cf
from .trial_cf.config import TrialCFSnapshotConfig
from .trial_cf.graph import (
    attach_cf_snapshot as attach_trial_cf_snapshot,
    build_cf_num_neighbors as build_trial_cf_num_neighbors,
)
from .trial_cf.interactions import build_source_sponsor_interactions
from .trial_cf.io import load_snapshot as load_trial_cf_snapshot
from .trial_cf.io import validate_manifest_for_training as validate_trial_route_cf


@dataclass(frozen=True)
class CoverageMetrics:
    num_rows: int
    num_positive_rows: int
    num_rows_with_candidates: int
    num_rows_with_hit: int
    num_sampled_candidates: int
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

    @property
    def sampled_candidate_precision(self) -> float:
        if self.num_sampled_candidates == 0:
            return float("nan")
        return self.num_covered_groundtruth_labels / self.num_sampled_candidates

    @property
    def mean_sampled_candidates_per_positive_row(self) -> float:
        if self.num_positive_rows == 0:
            return float("nan")
        return self.num_sampled_candidates / self.num_positive_rows

    def to_dict(self) -> dict[str, float | int]:
        out = asdict(self)
        out["coverage_rate"] = self.coverage_rate
        out["sampled_subgraph_coverage_rate"] = self.coverage_rate
        out["hit_rate"] = self.hit_rate
        out["achievable_MAP"] = self.achievable_map
        out["coverage_gap"] = self.coverage_gap
        out["sampled_candidate_precision"] = self.sampled_candidate_precision
        out["mean_sampled_candidates_per_positive_row"] = (
            self.mean_sampled_candidates_per_positive_row
        )
        return out


@dataclass(frozen=True)
class TargetSpec:
    dataset: str
    task: str
    item_cf: bool
    user_cf: bool
    route_cf: bool = False


TARGETS = [
    TargetSpec("rel-hm", "user-item-purchase", item_cf=True, user_cf=True),
    TargetSpec("rel-avito", "user-ad-visit", item_cf=True, user_cf=True),
    TargetSpec("rel-stack", "user-post-comment", item_cf=True, user_cf=True),
    TargetSpec(
        "rel-trial",
        "condition-sponsor-run",
        item_cf=False,
        user_cf=False,
        route_cf=True,
    ),
    TargetSpec(
        "rel-trial",
        "site-sponsor-run",
        item_cf=False,
        user_cf=False,
        route_cf=True,
    ),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate base, item-CF, user-CF, and Rel-Trial route-CF coverage "
            "for the supported recommendation tasks."
        )
    )
    parser.add_argument(
        "--snapshot-root",
        type=Path,
        action="append",
        default=[],
        help="Root(s) containing CF snapshot manifest.json files.",
    )
    parser.add_argument(
        "--item-cf-root",
        type=Path,
        action="append",
        default=[],
        help="Additional root(s) to search only for item-CF snapshots.",
    )
    parser.add_argument(
        "--user-cf-root",
        type=Path,
        action="append",
        default=[],
        help="Additional root(s) to search only for user-CF snapshots.",
    )
    parser.add_argument(
        "--trial-cf-root",
        type=Path,
        action="append",
        default=[],
        help="Additional root(s) to search only for Rel-Trial route-CF snapshots.",
    )
    parser.add_argument(
        "--dataset",
        action="append",
        default=[],
        help="Dataset name(s) to evaluate. Defaults to all supported datasets.",
    )
    parser.add_argument(
        "--task",
        action="append",
        default=[],
        help="Task name(s) to evaluate. Defaults to all supported tasks.",
    )
    parser.add_argument(
        "--coverage-kinds",
        default="base,item_cf,user_cf,trial_cf",
        help="Comma-separated coverage kinds to evaluate.",
    )
    parser.add_argument("--splits", default="val,test")
    parser.add_argument("--output-csv", type=Path)
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--download", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--num-neighbors", type=int, default=128)
    parser.add_argument("--temporal-strategy", default="last")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--base-num-layers", type=int, default=4)
    parser.add_argument("--item-cf-num-layers", type=int, default=4)
    parser.add_argument("--user-cf-num-layers", type=int, default=4)
    parser.add_argument("--trial-cf-num-layers", type=int, default=2)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    splits = [split.strip() for split in args.splits.split(",") if split.strip()]
    coverage_kinds = {
        value.strip() for value in args.coverage_kinds.split(",") if value.strip()
    }
    unknown_kinds = coverage_kinds.difference({"base", "item_cf", "user_cf", "trial_cf"})
    if unknown_kinds:
        raise ValueError(f"Unknown coverage kinds: {sorted(unknown_kinds)}")
    targets = _selected_targets(args.dataset, args.task)
    manifest_index = ManifestIndex(
        item_roots=[*args.snapshot_root, *args.item_cf_root],
        user_roots=[*args.snapshot_root, *args.user_cf_root],
        trial_roots=[*args.snapshot_root, *args.trial_cf_root],
    )

    rows: list[dict] = []
    for spec in targets:
        dataset = get_dataset(spec.dataset, download=args.download)
        task = get_task(spec.dataset, spec.task, download=args.download)
        if not isinstance(task, RecommendationTask):
            raise TypeError(f"{spec.dataset}/{spec.task} is not a RecommendationTask.")
        db = dataset.get_db()
        base_graph = make_topology_graph(db)
        base_num_neighbors = build_uniform_num_neighbors(
            base_graph.edge_types,
            num_layers=args.base_num_layers,
            num_neighbors=args.num_neighbors,
        )

        for split in splits:
            if "base" in coverage_kinds:
                try:
                    rows.append(
                        _result_row(
                            spec,
                            split,
                            "base",
                            None,
                            "ok",
                            compute_sampled_subgraph_coverage_for_split(
                                base_graph,
                                task,
                                split,
                                num_neighbors=base_num_neighbors,
                                batch_size=args.batch_size,
                                temporal_strategy=args.temporal_strategy,
                                num_workers=args.num_workers,
                            ),
                        )
                    )
                except Exception as exc:
                    row = _unavailable_row(spec, split, "base", "error")
                    row["error"] = str(exc)
                    rows.append(row)

            if "item_cf" in coverage_kinds and spec.item_cf:
                item_dirs = manifest_index.find(spec.dataset, spec.task, "item_cf")
                rows.extend(
                    _cf_rows(
                        spec,
                        split,
                        "item_cf",
                        item_dirs,
                        lambda snapshot_dir: _compute_item_cf(
                            spec,
                            task,
                            split,
                            db,
                            base_graph,
                            snapshot_dir,
                            args.item_cf_num_layers,
                            args.num_neighbors,
                            args.batch_size,
                            args.temporal_strategy,
                            args.num_workers,
                        ),
                    )
                )
            elif "item_cf" in coverage_kinds:
                rows.append(_unavailable_row(spec, split, "item_cf", "not_supported"))

            if "user_cf" in coverage_kinds and spec.user_cf:
                user_dirs = manifest_index.find(spec.dataset, spec.task, "user_cf")
                rows.extend(
                    _cf_rows(
                        spec,
                        split,
                        "user_cf",
                        user_dirs,
                        lambda snapshot_dir: _compute_user_cf(
                            spec,
                            task,
                            split,
                            db,
                            base_graph,
                            snapshot_dir,
                            args.user_cf_num_layers,
                            args.num_neighbors,
                            args.batch_size,
                            args.temporal_strategy,
                            args.num_workers,
                        ),
                    )
                )
            elif "user_cf" in coverage_kinds:
                rows.append(_unavailable_row(spec, split, "user_cf", "not_supported"))

            if "trial_cf" in coverage_kinds and spec.route_cf:
                trial_dirs = manifest_index.find(spec.dataset, spec.task, "trial_cf")
                rows.extend(
                    _cf_rows(
                        spec,
                        split,
                        "trial_cf",
                        trial_dirs,
                        lambda snapshot_dir: _compute_trial_cf(
                            spec,
                            task,
                            split,
                            base_graph,
                            snapshot_dir,
                            args.trial_cf_num_layers,
                            args.num_neighbors,
                            args.batch_size,
                            args.temporal_strategy,
                            args.num_workers,
                        ),
                    )
                )
            elif "trial_cf" in coverage_kinds and spec.dataset == "rel-trial":
                rows.append(_unavailable_row(spec, split, "trial_cf", "not_supported"))

    df = pd.DataFrame(rows)
    _write_outputs(df, args.output_csv, args.output_json)
    print(df.to_string(index=False))


class ManifestIndex:
    def __init__(
        self,
        *,
        item_roots: Iterable[Path],
        user_roots: Iterable[Path],
        trial_roots: Iterable[Path],
    ) -> None:
        self._manifests = {
            "item_cf": self._load_manifests(item_roots),
            "user_cf": self._load_manifests(user_roots),
            "trial_cf": self._load_manifests(trial_roots),
        }

    def find(self, dataset: str, task: str, coverage_kind: str) -> list[Path]:
        matches: list[Path] = []
        for path, manifest in self._manifests[coverage_kind]:
            if manifest.get("dataset") != dataset or manifest.get("task") != task:
                continue
            if coverage_kind == "item_cf" and not _is_item_cf_manifest(manifest):
                continue
            if coverage_kind == "user_cf" and not _is_user_cf_manifest(manifest):
                continue
            if coverage_kind == "trial_cf" and not _is_trial_cf_manifest(manifest):
                continue
            matches.append(path.parent)
        return sorted(set(matches), key=str)

    @staticmethod
    def _load_manifests(roots: Iterable[Path]) -> list[tuple[Path, dict]]:
        out: list[tuple[Path, dict]] = []
        seen: set[Path] = set()
        for root in roots:
            root = Path(root)
            if not root.exists():
                continue
            paths = [root / "manifest.json"] if root.is_dir() else []
            if root.is_dir():
                paths.extend(root.rglob("manifest.json"))
            elif root.name == "manifest.json":
                paths.append(root)
            for path in paths:
                path = path.resolve()
                if path in seen or not path.exists():
                    continue
                seen.add(path)
                with open(path, "r", encoding="utf-8") as f:
                    out.append((path, json.load(f)))
        return out


def _selected_targets(dataset_names: list[str], task_names: list[str]) -> list[TargetSpec]:
    dataset_filter = set(dataset_names)
    task_filter = set(task_names)
    targets = [
        spec
        for spec in TARGETS
        if (not dataset_filter or spec.dataset in dataset_filter)
        and (not task_filter or spec.task in task_filter)
    ]
    if not targets:
        raise ValueError(
            "No supported targets matched "
            f"dataset={sorted(dataset_filter)} task={sorted(task_filter)}."
        )
    return targets


def make_topology_graph(db) -> HeteroData:
    """Build a feature-free graph with the same pkey/fkey topology as RelBench."""
    data = HeteroData()
    for table_name, table in db.table_dict.items():
        data[table_name].num_nodes = len(table.df)
        if table.time_col is not None:
            data[table_name].time = torch.from_numpy(to_unix_time(table.df[table.time_col]))

        df = table.df
        for fkey_name, pkey_table_name in table.fkey_col_to_pkey_table.items():
            pkey_index = df[fkey_name]
            mask = ~pkey_index.isna()
            fkey_index = torch.arange(len(pkey_index))
            pkey_tensor = torch.from_numpy(pkey_index[mask].astype(int).values)
            fkey_tensor = fkey_index[torch.from_numpy(mask.values)]

            edge_index = torch.stack([fkey_tensor, pkey_tensor], dim=0)
            edge_type = (table_name, f"f2p_{fkey_name}", pkey_table_name)
            data[edge_type].edge_index = sort_edge_index(edge_index)

            edge_index = torch.stack([pkey_tensor, fkey_tensor], dim=0)
            edge_type = (pkey_table_name, f"rev_f2p_{fkey_name}", table_name)
            data[edge_type].edge_index = sort_edge_index(edge_index)
    data.validate()
    return data


def build_uniform_num_neighbors(
    edge_types,
    *,
    num_layers: int,
    num_neighbors: int,
) -> dict:
    if num_layers < 1:
        raise ValueError("num_layers must be positive.")
    hop_fanouts = [int(num_neighbors // 2**i) for i in range(num_layers)]
    return {edge_type: list(hop_fanouts) for edge_type in edge_types}


def compute_sampled_subgraph_coverage_for_split(
    graph: HeteroData,
    task: RecommendationTask,
    split: str,
    *,
    num_neighbors,
    batch_size: int,
    temporal_strategy: str,
    num_workers: int,
) -> CoverageMetrics:
    table = task.get_table(split, mask_input_cols=False)
    return compute_sampled_subgraph_coverage_for_table(
        graph,
        table,
        task,
        num_neighbors=num_neighbors,
        batch_size=batch_size,
        temporal_strategy=temporal_strategy,
        num_workers=num_workers,
    )


def compute_sampled_subgraph_coverage_for_table(
    graph: HeteroData,
    table: Table,
    task: RecommendationTask,
    *,
    num_neighbors,
    batch_size: int,
    temporal_strategy: str,
    num_workers: int,
) -> CoverageMetrics:
    totals = _MutableTotals()
    for group in group_recommendation_table_by_seed_time(table, task):
        loader, _ = make_seed_time_loader(
            graph,
            group,
            task,
            num_neighbors=num_neighbors,
            batch_size=batch_size,
            temporal_strategy=temporal_strategy,
            shuffle=False,
            num_workers=num_workers,
        )
        for batch in loader:
            _accumulate_sampled_batch_coverage(totals, batch, group.table, task)
    return totals.freeze()


def compute_dynamic_sampled_subgraph_coverage_for_split(
    task: RecommendationTask,
    split: str,
    make_graph_and_neighbors: Callable[[object], tuple[HeteroData, object]],
    *,
    batch_size: int,
    temporal_strategy: str,
    num_workers: int,
) -> CoverageMetrics:
    table = task.get_table(split, mask_input_cols=False)
    totals = _MutableTotals()
    for group in group_recommendation_table_by_seed_time(table, task):
        graph, num_neighbors = make_graph_and_neighbors(group)
        loader, _ = make_seed_time_loader(
            graph,
            group,
            task,
            num_neighbors=num_neighbors,
            batch_size=batch_size,
            temporal_strategy=temporal_strategy,
            shuffle=False,
            num_workers=num_workers,
        )
        for batch in loader:
            _accumulate_sampled_batch_coverage(totals, batch, group.table, task)
    return totals.freeze()


def _accumulate_sampled_batch_coverage(
    totals: "_MutableTotals",
    batch: HeteroData,
    table: Table,
    task: RecommendationTask,
) -> None:
    src_store = batch[task.src_entity_table]
    batch_size = int(src_store.batch_size)
    input_ids = src_store.input_id.cpu().numpy()
    sampled_by_row = _sampled_dst_by_batch_row(batch, task.dst_entity_table, batch_size)

    for row_pos, input_id in enumerate(input_ids):
        row = table.df.iloc[int(input_id)]
        true_dst = set(int(value) for value in row[task.dst_entity_col])
        totals.num_rows += 1
        if not true_dst:
            continue

        candidates = sampled_by_row.get(row_pos, set())
        covered = len(true_dst.intersection(candidates))
        totals.num_positive_rows += 1
        totals.num_rows_with_candidates += int(bool(candidates))
        totals.num_rows_with_hit += int(covered > 0)
        totals.num_sampled_candidates += len(candidates)
        totals.num_groundtruth_labels += len(true_dst)
        totals.num_covered_groundtruth_labels += covered
        totals.achievable_ap_sum += _oracle_average_precision_at_k(
            covered,
            len(true_dst),
            task.eval_k,
        )


def _sampled_dst_by_batch_row(
    batch: HeteroData,
    dst_entity_table: str,
    batch_size: int,
) -> dict[int, set[int]]:
    if dst_entity_table not in batch.node_types:
        return {}
    dst_store = batch[dst_entity_table]
    if "batch" not in dst_store or "n_id" not in dst_store:
        raise RuntimeError(
            f"Sampled batch for {dst_entity_table!r} does not expose per-root "
            "batch assignments. Install a temporal sampling backend compatible "
            "with PyG disjoint NeighborLoader."
        )
    out = {idx: set() for idx in range(batch_size)}
    row_ids = dst_store.batch.cpu().numpy()
    node_ids = dst_store.n_id.cpu().numpy()
    for row_id, node_id in zip(row_ids, node_ids):
        row_id = int(row_id)
        if 0 <= row_id < batch_size:
            out[row_id].add(int(node_id))
    return out


def compute_base_coverage_for_split(
    task: RecommendationTask,
    split: str,
    interactions: pd.DataFrame,
) -> CoverageMetrics:
    table = task.get_table(split, mask_input_cols=False)
    return compute_base_coverage_for_table(table, task, interactions)


def compute_base_coverage_for_table(
    table: Table,
    task: RecommendationTask,
    interactions: pd.DataFrame,
) -> CoverageMetrics:
    required = {"src_id", "dst_id", "time"}
    missing = required.difference(interactions.columns)
    if missing:
        raise ValueError(f"interactions is missing columns: {sorted(missing)}")
    if not pd.api.types.is_datetime64_any_dtype(interactions["time"]):
        raise ValueError("interactions.time must be datetime dtype.")
    interactions = _sort_interactions_once(interactions)
    times = interactions["time"].to_numpy(dtype="datetime64[ns]", copy=False)

    totals = _MutableTotals()
    for group in group_recommendation_table_by_seed_time(table, task):
        source_candidates = _source_candidates_until(
            interactions,
            times,
            group.seed_time,
            group.table.df[task.src_entity_col].unique(),
        )
        for _, row in group.table.df.iterrows():
            true_dst = set(row[task.dst_entity_col])
            totals.num_rows += 1
            if not true_dst:
                continue

            candidates = source_candidates.get(int(row[task.src_entity_col]), set())
            covered = len(true_dst.intersection(candidates))
            totals.num_positive_rows += 1
            totals.num_groundtruth_labels += len(true_dst)
            totals.num_covered_groundtruth_labels += covered
            totals.num_rows_with_candidates += int(bool(candidates))
            totals.num_rows_with_hit += int(covered > 0)
            totals.num_sampled_candidates += len(candidates)
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
    num_sampled_candidates: int = 0
    num_groundtruth_labels: int = 0
    num_covered_groundtruth_labels: int = 0
    achievable_ap_sum: float = 0.0

    def freeze(self) -> CoverageMetrics:
        return CoverageMetrics(**asdict(self))


def _base_interactions(spec: TargetSpec, db) -> pd.DataFrame:
    if spec.dataset == "rel-hm":
        df = db.table_dict["transactions"].df
        return _standardize_interactions(df, "customer_id", "article_id", "t_dat")
    if spec.dataset == "rel-avito":
        df = filter_visit_interactions(db.table_dict["VisitStream"].df)
        return _standardize_interactions(df, "UserID", "AdID", "ViewDate")
    if spec.dataset == "rel-stack":
        df = filter_comment_interactions(db.table_dict["comments"].df)
        return _standardize_interactions(df, "UserId", "PostId", "CreationDate")
    if spec.dataset == "rel-trial":
        config = TrialCFSnapshotConfig(
            dataset=spec.dataset,
            task=spec.task,
            history_days=None,
            all_history=True,
        )
        df = build_source_sponsor_interactions(db, config)
        return _standardize_interactions(df, "src_id", "sponsor_id", "date")
    raise ValueError(f"Unsupported dataset: {spec.dataset}")


def _standardize_interactions(
    df: pd.DataFrame,
    src_col: str,
    dst_col: str,
    time_col: str,
) -> pd.DataFrame:
    out = df[[src_col, dst_col, time_col]].dropna().rename(
        columns={src_col: "src_id", dst_col: "dst_id", time_col: "time"}
    )
    out["src_id"] = out["src_id"].astype("int64")
    out["dst_id"] = out["dst_id"].astype("int64")
    out["time"] = pd.to_datetime(out["time"])
    return out.drop_duplicates(ignore_index=True)


def _compute_item_cf(
    spec: TargetSpec,
    task: RecommendationTask,
    split: str,
    db,
    base_graph: HeteroData,
    snapshot_dir: Path,
    num_layers: int,
    num_neighbors: int,
    batch_size: int,
    temporal_strategy: str,
    num_workers: int,
):
    if spec.dataset == "rel-hm":
        config = validate_hm_item_cf(snapshot_dir)

        def make_graph_and_neighbors(group):
            snapshot = load_hm_item_cf_snapshot(
                snapshot_dir,
                group.seed_time,
                validate=True,
                config=config,
                num_articles=task.num_dst_nodes,
            )
            graph = attach_hm_item_cf_snapshot(
                base_graph,
                snapshot,
                group.seed_time,
                num_articles=task.num_dst_nodes,
            )
            return graph, build_hm_item_cf_num_neighbors(
                graph.edge_types,
                num_layers=num_layers,
                num_neighbors=num_neighbors,
            )

        return compute_dynamic_sampled_subgraph_coverage_for_split(
            task,
            split,
            make_graph_and_neighbors,
            batch_size=batch_size,
            temporal_strategy=temporal_strategy,
            num_workers=num_workers,
        )
    if spec.dataset == "rel-avito":
        config = validate_avito_item_cf(snapshot_dir)

        def make_graph_and_neighbors(group):
            snapshot = load_avito_item_cf_snapshot(
                snapshot_dir,
                group.seed_time,
                validate=True,
                config=config,
                num_ads=task.num_dst_nodes,
            )
            graph = attach_avito_item_cf_snapshot(
                base_graph,
                snapshot,
                group.seed_time,
                num_ads=task.num_dst_nodes,
            )
            return graph, build_avito_item_cf_num_neighbors(
                graph.edge_types,
                num_layers=num_layers,
                num_neighbors=num_neighbors,
            )

        return compute_dynamic_sampled_subgraph_coverage_for_split(
            task,
            split,
            make_graph_and_neighbors,
            batch_size=batch_size,
            temporal_strategy=temporal_strategy,
            num_workers=num_workers,
        )
    if spec.dataset == "rel-stack":
        config = validate_stack_item_cf(snapshot_dir)

        def make_graph_and_neighbors(group):
            snapshot = load_stack_item_cf_snapshot(
                snapshot_dir,
                group.seed_time,
                validate=True,
                config=config,
                num_posts=task.num_dst_nodes,
            )
            graph = attach_stack_item_cf_snapshot(
                base_graph,
                snapshot,
                group.seed_time,
                num_posts=task.num_dst_nodes,
            )
            return graph, build_stack_item_cf_num_neighbors(
                graph.edge_types,
                num_layers=num_layers,
                num_neighbors=num_neighbors,
            )

        return compute_dynamic_sampled_subgraph_coverage_for_split(
            task,
            split,
            make_graph_and_neighbors,
            batch_size=batch_size,
            temporal_strategy=temporal_strategy,
            num_workers=num_workers,
        )
    raise ValueError(f"Item-CF is not supported for {spec.dataset}/{spec.task}.")


def _compute_user_cf(
    spec: TargetSpec,
    task: RecommendationTask,
    split: str,
    db,
    base_graph: HeteroData,
    snapshot_dir: Path,
    num_layers: int,
    num_neighbors: int,
    batch_size: int,
    temporal_strategy: str,
    num_workers: int,
):
    if spec.dataset == "rel-hm":
        config = validate_hm_user_cf(snapshot_dir)

        def make_graph_and_neighbors(group):
            snapshot = load_hm_user_cf_snapshot(
                snapshot_dir,
                group.seed_time,
                validate=True,
                config=config,
                num_customers=task.num_src_nodes,
            )
            graph = attach_hm_user_cf_snapshot(
                base_graph,
                snapshot,
                group.seed_time,
                num_customers=task.num_src_nodes,
            )
            return graph, build_hm_user_cf_num_neighbors(
                graph.edge_types,
                num_layers=num_layers,
                num_neighbors=num_neighbors,
            )

        return compute_dynamic_sampled_subgraph_coverage_for_split(
            task,
            split,
            make_graph_and_neighbors,
            batch_size=batch_size,
            temporal_strategy=temporal_strategy,
            num_workers=num_workers,
        )
    if spec.dataset == "rel-avito":
        config = validate_avito_user_cf(snapshot_dir)

        def make_graph_and_neighbors(group):
            snapshot = load_avito_user_cf_snapshot(
                snapshot_dir,
                group.seed_time,
                validate=True,
                config=config,
                num_users=task.num_src_nodes,
            )
            graph = attach_avito_user_cf_snapshot(
                base_graph,
                snapshot,
                group.seed_time,
                num_users=task.num_src_nodes,
            )
            return graph, build_avito_user_cf_num_neighbors(
                graph.edge_types,
                num_layers=num_layers,
                num_neighbors=num_neighbors,
            )

        return compute_dynamic_sampled_subgraph_coverage_for_split(
            task,
            split,
            make_graph_and_neighbors,
            batch_size=batch_size,
            temporal_strategy=temporal_strategy,
            num_workers=num_workers,
        )
    if spec.dataset == "rel-stack":
        config = validate_stack_user_cf(snapshot_dir)

        def make_graph_and_neighbors(group):
            snapshot = load_stack_user_cf_snapshot(
                snapshot_dir,
                group.seed_time,
                validate=True,
                config=config,
                num_users=task.num_src_nodes,
            )
            graph = attach_stack_user_cf_snapshot(
                base_graph,
                snapshot,
                group.seed_time,
                num_users=task.num_src_nodes,
            )
            return graph, build_stack_user_cf_num_neighbors(
                graph.edge_types,
                num_layers=num_layers,
                num_neighbors=num_neighbors,
            )

        return compute_dynamic_sampled_subgraph_coverage_for_split(
            task,
            split,
            make_graph_and_neighbors,
            batch_size=batch_size,
            temporal_strategy=temporal_strategy,
            num_workers=num_workers,
        )
    raise ValueError(f"User-CF is not supported for {spec.dataset}/{spec.task}.")


def _compute_trial_cf(
    spec: TargetSpec,
    task: RecommendationTask,
    split: str,
    base_graph: HeteroData,
    snapshot_dir: Path,
    num_layers: int,
    num_neighbors: int,
    batch_size: int,
    temporal_strategy: str,
    num_workers: int,
):
    config = validate_trial_route_cf(snapshot_dir)
    if config.task != spec.task:
        raise ValueError(f"Snapshot task {config.task!r} does not match {spec.task!r}.")

    def make_graph_and_neighbors(group):
        snapshot = load_trial_cf_snapshot(
            snapshot_dir,
            group.seed_time,
            validate=True,
            config=config,
            num_sources=task.num_src_nodes,
            num_sponsors=task.num_dst_nodes,
        )
        graph = attach_trial_cf_snapshot(
            base_graph,
            snapshot,
            group.seed_time,
            config=config,
            num_sources=task.num_src_nodes,
            num_sponsors=task.num_dst_nodes,
        )
        return graph, build_trial_cf_num_neighbors(
            graph.edge_types,
            config=config,
            num_layers=num_layers,
            num_neighbors=num_neighbors,
        )

    return compute_dynamic_sampled_subgraph_coverage_for_split(
        task,
        split,
        make_graph_and_neighbors,
        batch_size=batch_size,
        temporal_strategy=temporal_strategy,
        num_workers=num_workers,
    )


def _cf_rows(
    spec: TargetSpec,
    split: str,
    coverage_kind: str,
    snapshot_dirs: list[Path],
    compute: Callable[[Path], object],
) -> list[dict]:
    if not snapshot_dirs:
        return [_unavailable_row(spec, split, coverage_kind, "missing_snapshot")]

    rows: list[dict] = []
    for snapshot_dir in snapshot_dirs:
        try:
            rows.append(
                _result_row(
                    spec,
                    split,
                    coverage_kind,
                    snapshot_dir,
                    "ok",
                    compute(snapshot_dir),
                )
            )
        except Exception as exc:
            rows.append(
                {
                    **_unavailable_row(spec, split, coverage_kind, "error"),
                    "snapshot_dir": str(snapshot_dir),
                    "error": str(exc),
                }
            )
    return rows


def _result_row(
    spec: TargetSpec,
    split: str,
    coverage_kind: str,
    snapshot_dir: Path | None,
    status: str,
    metrics,
) -> dict:
    row = {
        "dataset": spec.dataset,
        "task": spec.task,
        "split": split,
        "coverage_kind": coverage_kind,
        "snapshot_dir": "" if snapshot_dir is None else str(snapshot_dir),
        "status": status,
        "error": "",
    }
    row.update(metrics.to_dict())
    return row


def _unavailable_row(
    spec: TargetSpec,
    split: str,
    coverage_kind: str,
    status: str,
) -> dict:
    return {
        "dataset": spec.dataset,
        "task": spec.task,
        "split": split,
        "coverage_kind": coverage_kind,
        "snapshot_dir": "",
        "status": status,
        "error": "",
        "num_rows": np.nan,
        "num_positive_rows": np.nan,
        "num_rows_with_candidates": np.nan,
        "num_rows_with_hit": np.nan,
        "num_sampled_candidates": np.nan,
        "num_groundtruth_labels": np.nan,
        "num_covered_groundtruth_labels": np.nan,
        "achievable_ap_sum": np.nan,
        "coverage_rate": np.nan,
        "sampled_subgraph_coverage_rate": np.nan,
        "hit_rate": np.nan,
        "achievable_MAP": np.nan,
        "coverage_gap": np.nan,
        "sampled_candidate_precision": np.nan,
        "mean_sampled_candidates_per_positive_row": np.nan,
    }


def _sort_interactions_once(interactions: pd.DataFrame) -> pd.DataFrame:
    if interactions["time"].is_monotonic_increasing:
        return interactions
    return interactions.sort_values("time", kind="mergesort").reset_index(drop=True)


def _source_candidates_until(
    interactions: pd.DataFrame,
    times: np.ndarray,
    seed_time: pd.Timestamp,
    source_ids,
) -> dict[int, set[int]]:
    seed_np = np.datetime64(pd.Timestamp(seed_time).to_datetime64(), "ns")
    right = int(np.searchsorted(times, seed_np, side="right"))
    if right == 0:
        return {}

    source_ids = {int(value) for value in source_ids}
    history = interactions.iloc[:right][["src_id", "dst_id"]]
    history = history[history["src_id"].isin(source_ids)].drop_duplicates()
    out: dict[int, set[int]] = {}
    for source_id, group in history.groupby("src_id", sort=False):
        out[int(source_id)] = {int(dst_id) for dst_id in group["dst_id"]}
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


def _is_item_cf_manifest(manifest: dict) -> bool:
    if manifest.get("cf_kind") == "user_user_history_overlap":
        return False
    if manifest.get("cf_route") == "source_sponsor_route_collapsed":
        return False
    if manifest.get("cf_node_type") in {"article_cf", "ad_cf", "post_cf"}:
        return True
    return (
        manifest.get("dataset") == "rel-hm"
        and manifest.get("task") == "user-item-purchase"
        and "window_weeks" in manifest
        and "min_support" in manifest
        and "top_l" in manifest
        and "alpha" in manifest
    )


def _is_user_cf_manifest(manifest: dict) -> bool:
    return (
        manifest.get("cf_kind") == "user_user_history_overlap"
        or manifest.get("cf_node_type") == "user_cf"
    )


def _is_trial_cf_manifest(manifest: dict) -> bool:
    return manifest.get("cf_route") == "source_sponsor_route_collapsed"


def _write_outputs(
    df: pd.DataFrame,
    output_csv: Path | None,
    output_json: Path | None,
) -> None:
    if output_csv is not None:
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_csv, index=False)
    if output_json is not None:
        output_json.parent.mkdir(parents=True, exist_ok=True)
        df.to_json(output_json, orient="records", indent=2)


if __name__ == "__main__":
    main()
