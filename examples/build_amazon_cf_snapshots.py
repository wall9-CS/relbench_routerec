from __future__ import annotations

import argparse
import os
from pathlib import Path

import pandas as pd

from relbench.datasets import get_dataset
from relbench.tasks import get_task

from .amazon_cf.config import AmazonCFSnapshotConfig, DEFAULT_HISTORY_DAYS, snapshot_filename
from .amazon_cf.interactions import filter_review_interactions
from .amazon_cf.io import (
    discover_seed_times,
    initialize_or_load_manifest,
    load_snapshot,
    manifest_key,
    parse_seed_times,
    write_manifest_atomic,
    write_snapshot_atomic,
)
from .amazon_cf.snapshot import build_snapshot


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build seed-time-specific Rel-Amazon product-CF parquet snapshots."
    )
    parser.add_argument("--dataset", default="rel-amazon")
    parser.add_argument(
        "--task",
        default="user-item-purchase",
        choices=["user-item-purchase", "user-item-rate", "user-item-review"],
    )
    parser.add_argument("--history-days", type=int, default=None)
    parser.add_argument("--all-history", action="store_true")
    parser.add_argument("--min-support", type=int, default=3)
    parser.add_argument("--top-l", type=int, default=32)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--splits", default="train,val,test")
    parser.add_argument("--seed-time", action="append", default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--download", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.all_history and args.history_days is not None:
        raise SystemExit("--history-days and --all-history are mutually exclusive.")
    history_days = None if args.all_history else (
        args.history_days or DEFAULT_HISTORY_DAYS
    )
    config = AmazonCFSnapshotConfig(
        dataset=args.dataset,
        task=args.task,
        history_days=history_days,
        all_history=args.all_history,
        min_support=args.min_support,
        top_l=args.top_l,
        alpha=args.alpha,
    )
    snapshot_dir = config.snapshot_dir(args.output_root)
    manifest = initialize_or_load_manifest(snapshot_dir, config)

    dataset = get_dataset(args.dataset, download=args.download)
    task = get_task(args.dataset, args.task, download=args.download)
    splits = [split.strip() for split in args.splits.split(",") if split.strip()]
    seed_times = parse_seed_times(args.seed_time)
    if seed_times is None:
        seed_times = discover_seed_times(task, splits)

    db = dataset.get_db()
    interactions = filter_review_interactions(db.table_dict["review"].df, args.task)
    interactions = interactions.sort_values("review_time", kind="mergesort").reset_index(
        drop=True
    )
    num_products = len(db.table_dict["product"])

    print(f"Writing snapshots to {snapshot_dir}")
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
                    num_products=num_products,
                )
            except Exception as exc:
                raise RuntimeError(
                    f"Existing Amazon CF snapshot for {seed_time} is invalid. "
                    "Rerun with --overwrite to regenerate it."
                ) from exc
            print(f"Skipping valid existing snapshot for {seed_time}: {final_path}")
            continue

        df, stats = build_snapshot(interactions, seed_time, num_products, config)
        path = write_snapshot_atomic(
            df,
            snapshot_dir,
            seed_time,
            config=config,
            num_products=num_products,
        )
        file_size = os.path.getsize(path)
        manifest["snapshot_files"][key] = stats.to_manifest_entry(
            file=path.name,
            file_size_bytes=file_size,
        )
        write_manifest_atomic(snapshot_dir, manifest)
        print(
            f"{pd.Timestamp(seed_time)}: rows={len(df)} "
            f"active_customers={stats.num_active_customers} "
            f"unique_pairs={stats.num_unique_user_item_pairs} "
            f"C_nnz={stats.cooccurrence_nnz_before_support} "
            f"elapsed={stats.elapsed_seconds:.2f}s file={path}"
        )

    print(snapshot_dir)


if __name__ == "__main__":
    main()
