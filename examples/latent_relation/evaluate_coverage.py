from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from relbench.base import Table, TaskType
from relbench.datasets import get_dataset
from relbench.tasks import get_task

from .adapters import make_task_adapter
from .cli_utils import default_cache_dir
from .coverage import (
    aggregate_query_coverage,
    compute_latent_coverage_for_table,
    write_coverage_outputs,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate latent-relation locality.")
    parser.add_argument("--dataset", default="rel-hm")
    parser.add_argument("--task", default="user-item-purchase")
    parser.add_argument("--snapshot-dir", type=Path, required=True)
    parser.add_argument("--history-limit", type=int, default=64)
    parser.add_argument("--splits", default="val,test")
    parser.add_argument("--seed-time", action="append", default=None)
    parser.add_argument("--row-limit", type=int, default=None)
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--output-csv", type=Path, default=None)
    parser.add_argument("--cache-dir", default=default_cache_dir())
    parser.add_argument("--download", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset = get_dataset(args.dataset, download=args.download)
    task = get_task(args.dataset, args.task, download=args.download)
    db = dataset.get_db()
    adapter = make_task_adapter(args.dataset, args.task, task, db)
    if task.task_type != TaskType.LINK_PREDICTION:
        raise ValueError("Coverage diagnostics require a recommendation task.")
    rows = []
    by_split = {}
    seed_times = None
    if args.seed_time:
        seed_times = {pd.Timestamp(value) for value in args.seed_time}
    for split in [value.strip() for value in args.splits.split(",") if value.strip()]:
        table = task.get_table(split)
        if seed_times is not None:
            mask = pd.to_datetime(table.df[task.time_col]).isin(seed_times)
            table = Table(
                df=table.df.loc[mask].reset_index(drop=True),
                fkey_col_to_pkey_table=table.fkey_col_to_pkey_table,
                pkey_col=table.pkey_col,
                time_col=table.time_col,
            )
        if args.row_limit is not None:
            table = Table(
                df=table.df.head(args.row_limit).reset_index(drop=True),
                fkey_col_to_pkey_table=table.fkey_col_to_pkey_table,
                pkey_col=table.pkey_col,
                time_col=table.time_col,
            )
        split_rows = compute_latent_coverage_for_table(
            table,
            task,
            adapter,
            args.snapshot_dir,
            split=split,
            history_limit=args.history_limit,
        )
        rows.extend(split_rows)
        by_split[split] = aggregate_query_coverage(split_rows).to_dict()
    report = write_coverage_outputs(rows, args.output_json, args.output_csv)
    report["splits"] = by_split
    aggregate = report["aggregate"]
    print(
        json.dumps(
            {
                "aggregate": aggregate,
                "splits": by_split,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
