from __future__ import annotations

import argparse
import os
from pathlib import Path

import pandas as pd

from relbench.datasets import get_dataset
from relbench.tasks import get_task

from .stack_cf.interactions import filter_comment_interactions
from .stack_cf.io import load_snapshot as load_item_cf_snapshot
from .stack_cf.io import validate_manifest_for_training as validate_item_cf_manifest
from .stack_direct_cf.config import StackDirectCFSnapshotConfig, snapshot_filename
from .stack_direct_cf.io import (
    discover_seed_times,
    initialize_or_load_manifest,
    load_snapshot,
    manifest_key,
    parse_seed_times,
    source_nodes_by_seed_time,
    write_manifest_atomic,
    write_snapshot_atomic,
)
from .stack_direct_cf.snapshot import build_snapshot


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build direct source-to-post CF snapshots for Rel-Stack."
    )
    parser.add_argument("--dataset", default="rel-stack")
    parser.add_argument("--task", default="user-post-comment")
    parser.add_argument("--item-cf-snapshot-dir", type=Path, required=True)
    parser.add_argument("--direct-top-k", type=int, default=128)
    parser.add_argument("--filter-seen-dst", action="store_true")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--splits", default="train,val,test")
    parser.add_argument("--seed-time", action="append", default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--download", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    parent_config = validate_item_cf_manifest(args.item_cf_snapshot_dir)
    if parent_config.dataset != args.dataset or parent_config.task != args.task:
        raise ValueError(
            f"Parent item-CF manifest is for {parent_config.dataset}/{parent_config.task}, "
            f"but requested {args.dataset}/{args.task}."
        )
    config = StackDirectCFSnapshotConfig(
        dataset=args.dataset,
        task=args.task,
        parent_item_cf_snapshot_dir=str(args.item_cf_snapshot_dir),
        history_days=parent_config.history_days,
        all_history=parent_config.all_history,
        min_support=parent_config.min_support,
        top_l=parent_config.top_l,
        alpha=parent_config.alpha,
        direct_top_k=args.direct_top_k,
        filter_seen_dst=args.filter_seen_dst,
    )
    snapshot_dir = config.snapshot_dir(args.output_root)
    manifest = initialize_or_load_manifest(snapshot_dir, config)

    dataset = get_dataset(args.dataset, download=args.download)
    task = get_task(args.dataset, args.task, download=args.download)
    splits = [split.strip() for split in args.splits.split(",") if split.strip()]
    seed_times = parse_seed_times(args.seed_time)
    if seed_times is None:
        seed_times = discover_seed_times(task, splits)
    source_by_time = source_nodes_by_seed_time(task, splits)

    db = dataset.get_db()
    interactions = filter_comment_interactions(db.table_dict["comments"].df).sort_values(
        "CreationDate", kind="mergesort"
    ).reset_index(drop=True)
    num_users = len(db.table_dict["users"])
    num_posts = task.num_dst_nodes

    print(f"Writing direct snapshots to {snapshot_dir}")
    for seed_time in seed_times:
        key = manifest_key(seed_time)
        final_path = snapshot_dir / snapshot_filename(seed_time)
        if final_path.exists() and not args.overwrite:
            try:
                load_snapshot(
                    snapshot_dir,
                    seed_time,
                    validate=True,
                    config=config,
                    num_users=num_users,
                    num_posts=num_posts,
                )
            except Exception as exc:
                raise RuntimeError(
                    f"Existing direct snapshot for {seed_time} is invalid. "
                    "Rerun with --overwrite to regenerate it."
                ) from exc
            print(f"Skipping valid existing snapshot for {seed_time}: {final_path}")
            continue

        item_cf = load_item_cf_snapshot(
            args.item_cf_snapshot_dir,
            seed_time,
            validate=True,
            config=parent_config,
            num_posts=num_posts,
        )
        df, stats = build_snapshot(
            interactions,
            item_cf,
            seed_time,
            source_by_time.get(pd.Timestamp(seed_time), []),
            config,
        )
        path = write_snapshot_atomic(
            df,
            snapshot_dir,
            seed_time,
            config=config,
            num_users=num_users,
            num_posts=num_posts,
        )
        file_size = os.path.getsize(path)
        manifest["snapshot_files"][key] = stats.to_manifest_entry(
            file=path.name,
            file_size_bytes=file_size,
        )
        write_manifest_atomic(snapshot_dir, manifest)
        print(
            f"{pd.Timestamp(seed_time)}: rows={len(df)} "
            f"sources={stats.num_source_nodes} "
            f"joined={stats.num_joined_rows} "
            f"collapsed={stats.num_collapsed_pairs_before_topk} "
            f"elapsed={stats.elapsed_seconds:.2f}s file={path}"
        )

    print(snapshot_dir)


if __name__ == "__main__":
    main()
