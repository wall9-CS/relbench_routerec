from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from relbench.base import RecommendationTask, Table

from .config import OracleGTPhase1Config, snapshot_id
from .history import HistoricalSnapshotStats, build_historical_interaction_matrix
from .io import (
    initialize_or_load_manifest,
    write_csv_atomic,
    write_history_snapshot,
    write_json_atomic,
    write_parquet_atomic,
)
from .raw_support import (
    QUERY_ID_COL,
    collect_raw_candidate_support,
    ensure_temporal_neighbor_sampling_backend,
    make_raw_topology_graph,
    task_table_with_query_ids,
)


@dataclass(frozen=True)
class Phase1Outputs:
    output_dir: Path
    raw_support_files: dict[str, str]
    gt_files: dict[str, str]
    split_summary_file: str
    snapshot_summary_file: str


def build_query_gt_table(
    table: Table,
    raw_support: pd.DataFrame,
    task: RecommendationTask,
    split: str,
) -> pd.DataFrame:
    support_by_query = {
        int(row[QUERY_ID_COL]): set(int(value) for value in row["raw_candidate_ids"])
        for _, row in raw_support.iterrows()
    }
    rows: list[dict] = []
    for _, row in table.df.iterrows():
        query_id = int(row[QUERY_ID_COL])
        raw_candidates = support_by_query.get(query_id, set())
        gt_values = sorted(set(int(value) for value in row[task.dst_entity_col]))
        for gt_id in gt_values:
            rows.append(
                {
                    QUERY_ID_COL: query_id,
                    "split": split,
                    "seed_time": pd.Timestamp(row[task.time_col]),
                    "src_id": int(row[task.src_entity_col]),
                    "gt_id": int(gt_id),
                    "raw_covered": bool(gt_id in raw_candidates),
                    "snapshot_id": snapshot_id(pd.Timestamp(row[task.time_col])),
                }
            )
    if not rows:
        return pd.DataFrame(
            {
                QUERY_ID_COL: pd.Series([], dtype="int64"),
                "split": pd.Series([], dtype="object"),
                "seed_time": pd.Series([], dtype="datetime64[ns]"),
                "src_id": pd.Series([], dtype="int64"),
                "gt_id": pd.Series([], dtype="int64"),
                "raw_covered": pd.Series([], dtype="bool"),
                "snapshot_id": pd.Series([], dtype="object"),
            }
        )
    return pd.DataFrame(rows)


def summarize_split(
    split: str,
    table: Table,
    gt_table: pd.DataFrame,
    snapshot_stats: list[HistoricalSnapshotStats],
    task: RecommendationTask,
) -> dict:
    gt_counts = table.df[task.dst_entity_col].map(len).to_numpy(dtype=np.int64)
    missing_counts = (
        gt_table.loc[~gt_table["raw_covered"]]
        .groupby(QUERY_ID_COL)
        .size()
        .reindex(table.df[QUERY_ID_COL], fill_value=0)
        .to_numpy(dtype=np.int64)
    )
    total_gt = int(len(gt_table))
    raw_covered = int(gt_table["raw_covered"].sum()) if total_gt else 0
    missing = int(total_gt - raw_covered)
    return {
        "split": split,
        "num_queries": int(len(table.df)),
        "total_gt_count": total_gt,
        "raw_covered_gt_count": raw_covered,
        "missing_gt_count": missing,
        "missing_gt_rate": float(missing / total_gt) if total_gt else float("nan"),
        "mean_gt_per_query": float(np.mean(gt_counts)) if len(gt_counts) else float("nan"),
        "median_gt_per_query": float(np.median(gt_counts)) if len(gt_counts) else float("nan"),
        "max_gt_per_query": int(np.max(gt_counts)) if len(gt_counts) else 0,
        "mean_missing_gt_per_query": float(np.mean(missing_counts))
        if len(missing_counts)
        else float("nan"),
        "median_missing_gt_per_query": float(np.median(missing_counts))
        if len(missing_counts)
        else float("nan"),
        "max_missing_gt_per_query": int(np.max(missing_counts)) if len(missing_counts) else 0,
        "num_exact_seed_time_snapshots": int(len(snapshot_stats)),
        "historical_interactions_per_snapshot": [
            stats.num_historical_interactions for stats in snapshot_stats
        ],
        "active_users_per_snapshot": [stats.num_active_users for stats in snapshot_stats],
        "active_items_per_snapshot": [stats.num_active_items for stats in snapshot_stats],
        "mean_historical_interactions_per_snapshot": float(
            np.mean([stats.num_historical_interactions for stats in snapshot_stats])
        )
        if snapshot_stats
        else float("nan"),
        "median_historical_interactions_per_snapshot": float(
            np.median([stats.num_historical_interactions for stats in snapshot_stats])
        )
        if snapshot_stats
        else float("nan"),
        "max_historical_interactions_per_snapshot": int(
            np.max([stats.num_historical_interactions for stats in snapshot_stats])
        )
        if snapshot_stats
        else 0,
        "mean_active_users_per_snapshot": float(
            np.mean([stats.num_active_users for stats in snapshot_stats])
        )
        if snapshot_stats
        else float("nan"),
        "median_active_users_per_snapshot": float(
            np.median([stats.num_active_users for stats in snapshot_stats])
        )
        if snapshot_stats
        else float("nan"),
        "max_active_users_per_snapshot": int(
            np.max([stats.num_active_users for stats in snapshot_stats])
        )
        if snapshot_stats
        else 0,
        "mean_active_items_per_snapshot": float(
            np.mean([stats.num_active_items for stats in snapshot_stats])
        )
        if snapshot_stats
        else float("nan"),
        "median_active_items_per_snapshot": float(
            np.median([stats.num_active_items for stats in snapshot_stats])
        )
        if snapshot_stats
        else float("nan"),
        "max_active_items_per_snapshot": int(
            np.max([stats.num_active_items for stats in snapshot_stats])
        )
        if snapshot_stats
        else 0,
    }


def run_phase1_preprocessing(
    *,
    dataset,
    task: RecommendationTask,
    output_root: Path,
    config: OracleGTPhase1Config,
    splits: list[str],
    batch_size: int,
    num_workers: int,
    overwrite: bool = False,
    max_seed_times_per_split: int | None = None,
    max_queries_per_split: int | None = None,
    validation_print_limit: int = 5,
) -> Phase1Outputs:
    ensure_temporal_neighbor_sampling_backend()
    output_dir = config.output_dir(output_root)
    manifest = initialize_or_load_manifest(output_dir, config, overwrite=overwrite)

    db = dataset.get_db()
    raw_graph = make_raw_topology_graph(db)
    transactions = db.table_dict["transactions"].df.sort_values(
        "t_dat", kind="mergesort"
    ).reset_index(drop=True)
    num_users = len(db.table_dict["customer"])
    num_items = len(db.table_dict["article"])

    print(
        "Oracle-GT Phase 1 raw support uses "
        f"num_layers={config.num_layers}, fanout={list(config.fanout)}, "
        f"temporal_strategy={config.temporal_strategy!r}, "
        f"sampler_seed={config.sampler_seed}."
    )

    tables: dict[str, Table] = {}
    raw_support_files: dict[str, str] = {}
    gt_files: dict[str, str] = {}
    split_summaries: list[dict] = []
    snapshot_rows: list[dict] = []
    snapshot_file_entries: dict[str, dict] = {}
    sample_rows: list[dict] = []

    for split in splits:
        table = task_table_with_query_ids(
            task,
            split,
            max_seed_times=max_seed_times_per_split,
            max_queries=max_queries_per_split,
        )
        tables[split] = table
        seed_times = sorted(pd.Timestamp(value) for value in pd.to_datetime(table.df[task.time_col]).unique())

        split_snapshot_stats: list[HistoricalSnapshotStats] = []
        for seed_time in seed_times:
            matrix, active_user_ids, stats = build_historical_interaction_matrix(
                transactions,
                seed_time,
                num_users=num_users,
                num_items=num_items,
                config=config,
            )
            metadata = {
                **stats.to_dict(),
                "matrix_shape": list(matrix.shape),
                "matrix_nnz": int(matrix.nnz),
                "active_user_ids_dtype": str(active_user_ids.dtype),
            }
            file_entry = write_history_snapshot(
                output_dir,
                seed_time,
                matrix,
                active_user_ids,
                metadata,
            )
            snapshot_file_entries[stats.snapshot_id] = file_entry
            split_snapshot_stats.append(stats)
            snapshot_rows.append({"split": split, **stats.to_dict(), **file_entry})

        raw_support = collect_raw_candidate_support(
            raw_graph,
            table,
            task,
            split,
            config=config,
            batch_size=batch_size,
            num_workers=num_workers,
        )
        raw_path = output_dir / "raw_support" / f"{split}.parquet"
        write_parquet_atomic(raw_support, raw_path)
        raw_support_files[split] = str(raw_path.relative_to(output_dir))

        gt_table = build_query_gt_table(table, raw_support, task, split)
        gt_path = output_dir / "gt" / f"{split}.parquet"
        write_parquet_atomic(gt_table, gt_path)
        gt_files[split] = str(gt_path.relative_to(output_dir))

        split_summaries.append(
            summarize_split(split, table, gt_table, split_snapshot_stats, task)
        )

        if len(sample_rows) < validation_print_limit:
            sample_rows.extend(
                _validation_sample_rows(
                    table,
                    raw_support,
                    task,
                    split,
                    limit=validation_print_limit - len(sample_rows),
                )
            )

    split_summary_df = pd.DataFrame(split_summaries)
    snapshot_summary_df = pd.DataFrame(snapshot_rows)
    split_summary_path = output_dir / "stats" / "split_summary.csv"
    snapshot_summary_path = output_dir / "stats" / "snapshot_summary.csv"
    write_csv_atomic(split_summary_df, split_summary_path)
    write_csv_atomic(snapshot_summary_df, snapshot_summary_path)
    write_json_atomic(output_dir / "stats" / "split_summary.json", {"rows": split_summaries})
    write_json_atomic(
        output_dir / "stats" / "snapshot_summary.json",
        {"rows": [row for row in snapshot_rows]},
    )

    manifest["files"] = {
        "raw_support": raw_support_files,
        "gt": gt_files,
        "snapshots": snapshot_file_entries,
        "split_summary_csv": str(split_summary_path.relative_to(output_dir)),
        "snapshot_summary_csv": str(snapshot_summary_path.relative_to(output_dir)),
        "split_summary_json": "stats/split_summary.json",
        "snapshot_summary_json": "stats/snapshot_summary.json",
    }
    write_json_atomic(output_dir / "manifest.json", manifest)

    _print_validation(sample_rows, snapshot_rows, config)

    return Phase1Outputs(
        output_dir=output_dir,
        raw_support_files=raw_support_files,
        gt_files=gt_files,
        split_summary_file=str(split_summary_path.relative_to(output_dir)),
        snapshot_summary_file=str(snapshot_summary_path.relative_to(output_dir)),
    )


def _validation_sample_rows(
    table: Table,
    raw_support: pd.DataFrame,
    task: RecommendationTask,
    split: str,
    *,
    limit: int,
) -> list[dict]:
    support_by_query = {
        int(row[QUERY_ID_COL]): set(int(value) for value in row["raw_candidate_ids"])
        for _, row in raw_support.iterrows()
    }
    rows: list[dict] = []
    for _, row in table.df.head(limit).iterrows():
        query_id = int(row[QUERY_ID_COL])
        gt_set = sorted(set(int(value) for value in row[task.dst_entity_col]))
        raw_support_set = support_by_query.get(query_id, set())
        intersection = sorted(set(gt_set).intersection(raw_support_set))
        missing = sorted(set(gt_set).difference(raw_support_set))
        rows.append(
            {
                "query_id": query_id,
                "split": split,
                "src_id": int(row[task.src_entity_col]),
                "seed_time": pd.Timestamp(row[task.time_col]).isoformat(),
                "gt_set": gt_set,
                "raw_gt_intersection": intersection,
                "missing_gt": missing,
                "raw_candidate_count": len(raw_support_set),
            }
        )
    return rows


def _print_validation(
    sample_rows: list[dict],
    snapshot_rows: list[dict],
    config: OracleGTPhase1Config,
) -> None:
    print("\nSmall-subset validation samples:")
    for row in sample_rows[:5]:
        print(
            "  "
            f"query_id={row['query_id']} split={row['split']} "
            f"src_id={row['src_id']} seed_time={row['seed_time']} "
            f"GT={row['gt_set']} raw_intersection_GT={row['raw_gt_intersection']} "
            f"missing={row['missing_gt']} raw_candidate_count={row['raw_candidate_count']}"
        )
    boundary_ok = all(bool(row["temporal_boundary_ok"]) for row in snapshot_rows)
    print(
        "\nHistorical snapshot boundary check: "
        f"{'passed' if boundary_ok else 'failed'} for {len(snapshot_rows)} snapshot rows."
    )
    for row in snapshot_rows[:5]:
        print(
            "  "
            f"snapshot_id={row['snapshot_id']} split={row['split']} "
            f"window=({row['window_start']}, {row['window_end']}] "
            f"max_interaction_time={row['max_interaction_time']} "
            f"historical_interactions={row['num_historical_interactions']} "
            f"active_users={row['num_active_users']} "
            f"active_items={row['num_active_items']}"
        )
    print(
        "\nRaw-support extraction check: "
        f"num_layers={config.num_layers}, fanout={list(config.fanout)}."
    )
