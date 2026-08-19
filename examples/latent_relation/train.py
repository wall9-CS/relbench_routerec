from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch

from relbench.base import Table, TaskType

from .cli_utils import (
    build_table_relation_model,
    default_cache_dir,
    load_relbench_context,
)
from .config import LatentRelationConfig
from .io import file_sha256
from .training import RelationTrainingDataset, save_checkpoint_atomic, train_relation_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a supervised latent relation model.")
    parser.add_argument("--dataset", default="rel-hm")
    parser.add_argument("--task", default="user-item-purchase")
    parser.add_argument("--history-limit", type=int, default=64)
    parser.add_argument("--relation-dim", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--num-negatives", type=int, default=256)
    parser.add_argument("--top-l", type=int, default=32)
    parser.add_argument("--retrieval-backend", choices=["torch", "faiss"], default="torch")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--encoder-channels", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--cache-dir", default=default_cache_dir())
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--max-steps-per-epoch", type=int, default=None)
    parser.add_argument(
        "--train-row-limit",
        type=int,
        default=None,
        help="Optional first-N train rows for smoke tests and debugging.",
    )
    parser.add_argument("--download", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    _, task, _, data, col_stats, adapter = load_relbench_context(
        dataset_name=args.dataset,
        task_name=args.task,
        cache_dir=args.cache_dir,
        device=device,
        download=args.download,
    )
    if task.task_type != TaskType.LINK_PREDICTION:
        raise ValueError("Latent relation training requires a recommendation task.")
    train_table = task.get_table("train")
    if args.train_row_limit is not None:
        train_table = Table(
            df=train_table.df.head(args.train_row_limit).reset_index(drop=True),
            fkey_col_to_pkey_table=train_table.fkey_col_to_pkey_table,
            pkey_col=train_table.pkey_col,
            time_col=train_table.time_col,
        )
    cutoff = pd.to_datetime(train_table.df[task.time_col]).max().isoformat()
    config = LatentRelationConfig(
        dataset=args.dataset,
        task=args.task,
        history_limit=args.history_limit,
        relation_dim=args.relation_dim,
        temperature=args.temperature,
        num_negatives=args.num_negatives,
        top_l=args.top_l,
        retrieval_backend=args.retrieval_backend,
        encoder_channels=args.encoder_channels,
        training_cutoff=cutoff,
    )
    model = build_table_relation_model(data, col_stats, adapter, config=config)
    examples = adapter.examples_from_table(train_table)
    dataset = RelationTrainingDataset(
        examples,
        adapter,
        history_limit=args.history_limit,
        num_negatives=args.num_negatives,
        seed=args.seed,
    )
    if len(dataset) == 0:
        raise RuntimeError("No train examples have both history and positive labels.")
    train_relation_model(
        model,
        dataset,
        device=device,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        seed=args.seed,
        max_steps_per_epoch=args.max_steps_per_epoch,
    )
    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    path = args.checkpoint_dir / f"{args.dataset}_{args.task}_latent_relation.pt"
    save_checkpoint_atomic(
        model,
        path,
        config=config,
        extra={
            "num_train_rows": len(train_table.df),
            "num_train_rows_with_history": len(dataset),
            "seed": args.seed,
        },
    )
    print(f"checkpoint={path}")
    print(f"checkpoint_sha256={file_sha256(path)}")


if __name__ == "__main__":
    main()
