from __future__ import annotations

import argparse
from pathlib import Path

from relbench.datasets import get_dataset
from relbench.tasks import get_task

from .hm_oracle_gt.config import OracleGTPhase1Config
from .hm_oracle_gt.preprocess import run_phase1_preprocessing


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build Phase-1 preprocessing artifacts for the Rel-HM oracle-GT "
            "candidate injection experiment. This creates raw 4-hop sampled "
            "support, GT/missing-GT tables, historical binary matrices, and "
            "summary statistics only."
        )
    )
    parser.add_argument("--dataset", default="rel-hm")
    parser.add_argument("--task", default="user-item-purchase")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--splits", default="train,val,test")
    parser.add_argument("--window-weeks", type=int, default=None)
    parser.add_argument("--all-history", action="store_true")
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--num-neighbors", type=int, default=128)
    parser.add_argument("--temporal-strategy", default="last")
    parser.add_argument("--sampler-seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-seed-times-per-split", type=int)
    parser.add_argument("--max-queries-per-split", type=int)
    parser.add_argument("--validation-print-limit", type=int, default=5)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--download", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.all_history and args.window_weeks is not None:
        raise SystemExit("--window-weeks and --all-history are mutually exclusive.")
    window_weeks = None if args.all_history else (args.window_weeks or 8)
    config = OracleGTPhase1Config(
        dataset=args.dataset,
        task=args.task,
        window_weeks=window_weeks,
        all_history=args.all_history,
        num_layers=args.num_layers,
        num_neighbors=args.num_neighbors,
        temporal_strategy=args.temporal_strategy,
        sampler_seed=args.sampler_seed,
    )
    splits = [split.strip() for split in args.splits.split(",") if split.strip()]
    dataset = get_dataset(args.dataset, download=args.download)
    task = get_task(args.dataset, args.task, download=args.download)
    outputs = run_phase1_preprocessing(
        dataset=dataset,
        task=task,
        output_root=args.output_root,
        config=config,
        splits=splits,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        overwrite=args.overwrite,
        max_seed_times_per_split=args.max_seed_times_per_split,
        max_queries_per_split=args.max_queries_per_split,
        validation_print_limit=args.validation_print_limit,
    )
    print(f"\nWrote Oracle-GT Phase 1 artifacts to {outputs.output_dir}")
    print(f"Raw support files: {outputs.raw_support_files}")
    print(f"GT files: {outputs.gt_files}")
    print(f"Split summary: {outputs.split_summary_file}")
    print(f"Snapshot summary: {outputs.snapshot_summary_file}")


if __name__ == "__main__":
    main()

