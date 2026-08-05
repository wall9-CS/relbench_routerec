from __future__ import annotations

import argparse
import json
from pathlib import Path

from relbench.base import RecommendationTask, TaskType
from relbench.datasets import get_dataset
from relbench.tasks import get_task

from .stack_cf.coverage import (
    combine_cf_coverage_metrics,
    compute_cf_coverage_for_split,
)
from .stack_cf.interactions import filter_comment_interactions
from .stack_cf.io import validate_manifest_for_training


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Report Rel-Stack post-CF snapshot coverage diagnostics."
    )
    parser.add_argument("--dataset", type=str, default="rel-stack")
    parser.add_argument("--task", type=str, default="user-post-comment")
    parser.add_argument("--cf-snapshot-dir", type=Path, required=True)
    parser.add_argument("--splits", default="val,test")
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument(
        "--include-source-posts",
        action="store_true",
        help="Count historical source posts as candidates in addition to CF dst posts.",
    )
    parser.add_argument(
        "--download", action=argparse.BooleanOptionalAction, default=True
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = validate_manifest_for_training(args.cf_snapshot_dir)
    if config.dataset != args.dataset or config.task != args.task:
        raise ValueError(
            f"Snapshot manifest is for {config.dataset}/{config.task}, "
            f"but script requested {args.dataset}/{args.task}."
        )

    dataset = get_dataset(args.dataset, download=args.download)
    task: RecommendationTask = get_task(args.dataset, args.task, download=args.download)
    if task.task_type != TaskType.LINK_PREDICTION:
        raise ValueError("Stack CF coverage diagnostics require a recommendation task.")

    db = dataset.get_db()
    interactions = filter_comment_interactions(db.table_dict["comments"].df)
    num_posts = len(db.table_dict["posts"])

    report = {}
    split_metrics = []
    for split in [value.strip() for value in args.splits.split(",") if value.strip()]:
        metrics = compute_cf_coverage_for_split(
            task,
            split,
            interactions,
            args.cf_snapshot_dir,
            config=config,
            num_posts=num_posts,
            num_layers=args.num_layers,
            include_source_posts=args.include_source_posts,
        )
        split_metrics.append(metrics)
        report[split] = metrics.to_dict()
    if len(split_metrics) > 1:
        report["overall"] = combine_cf_coverage_metrics(split_metrics).to_dict()

    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
