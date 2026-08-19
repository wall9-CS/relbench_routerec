from __future__ import annotations

import argparse
import os
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import torch

from .cli_utils import default_cache_dir, load_relation_checkpoint, load_relbench_context
from .config import LatentRelationConfig, snapshot_filename
from .io import (
    discover_seed_times,
    file_sha256,
    initialize_or_load_manifest,
    load_snapshot,
    manifest_key,
    parse_seed_times,
    should_reuse_snapshot,
    write_manifest_atomic,
    write_snapshot_atomic,
)
from .materialize import build_latent_snapshot


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build latent relation snapshots.")
    parser.add_argument("--dataset", default="rel-hm")
    parser.add_argument("--task", default="user-item-purchase")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--snapshot-dir", type=Path, required=True)
    parser.add_argument("--history-limit", type=int, default=64)
    parser.add_argument("--top-l", type=int, default=32)
    parser.add_argument("--retrieval-backend", choices=["torch", "faiss"], default="torch")
    parser.add_argument("--retrieval-query-chunk-size", type=int, default=4096)
    parser.add_argument("--retrieval-candidate-chunk-size", type=int, default=65536)
    parser.add_argument("--embedding-chunk-size", type=int, default=4096)
    parser.add_argument("--splits", default="train,val,test")
    parser.add_argument("--seed-time", action="append", default=None)
    parser.add_argument("--allow-self-loop", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--cache-dir", default=default_cache_dir())
    parser.add_argument("--download", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    _, task, _, data, col_stats, adapter = load_relbench_context(
        dataset_name=args.dataset,
        task_name=args.task,
        cache_dir=args.cache_dir,
        device=device,
        download=args.download,
    )
    model, checkpoint_config, _ = load_relation_checkpoint(
        args.checkpoint,
        data,
        col_stats,
        adapter,
        device=device,
    )
    checkpoint_hash = file_sha256(args.checkpoint)
    config = LatentRelationConfig(
        dataset=args.dataset,
        task=args.task,
        history_limit=args.history_limit,
        relation_dim=checkpoint_config.relation_dim,
        temperature=checkpoint_config.temperature,
        num_negatives=checkpoint_config.num_negatives,
        top_l=args.top_l,
        retrieval_backend=args.retrieval_backend,
        encoder_channels=checkpoint_config.encoder_channels,
        checkpoint_hash=checkpoint_hash,
        checkpoint_path=str(args.checkpoint),
        training_cutoff=checkpoint_config.training_cutoff,
    )
    manifest = initialize_or_load_manifest(args.snapshot_dir, config)
    manifest["created_or_updated_at"] = datetime.now(timezone.utc).isoformat()
    splits = [split.strip() for split in args.splits.split(",") if split.strip()]
    seed_times = parse_seed_times(args.seed_time) or discover_seed_times(task, splits)
    print(f"Writing latent relation snapshots to {args.snapshot_dir}")
    for seed_time in seed_times:
        final_path = args.snapshot_dir / snapshot_filename(seed_time)
        if should_reuse_snapshot(
            args.snapshot_dir,
            seed_time,
            config=config,
            num_destinations=adapter.num_destinations,
            allow_self_loop=args.allow_self_loop,
            force=args.force,
        ):
            load_snapshot(
                args.snapshot_dir,
                seed_time,
                validate=True,
                config=config,
                num_destinations=adapter.num_destinations,
                allow_self_loop=args.allow_self_loop,
            )
            print(f"Skipping valid existing snapshot for {seed_time}: {final_path}")
            continue
        anchors: set[int] = set()
        for split in splits:
            anchors.update(
                adapter.anchor_destinations_for_table(
                    task.get_table(split),
                    seed_time=pd.Timestamp(seed_time),
                    history_limit=args.history_limit,
                ).tolist()
            )
        snapshot, stats = build_latent_snapshot(
            model,
            adapter,
            pd.Timestamp(seed_time),
            pd.Series(sorted(anchors), dtype="int64").to_numpy(),
            config,
            device=device,
            query_chunk_size=args.retrieval_query_chunk_size,
            candidate_chunk_size=args.retrieval_candidate_chunk_size,
            embedding_chunk_size=args.embedding_chunk_size,
            allow_self_loop=args.allow_self_loop,
        )
        path = write_snapshot_atomic(
            snapshot,
            args.snapshot_dir,
            pd.Timestamp(seed_time),
            config=config,
            num_destinations=adapter.num_destinations,
            allow_self_loop=args.allow_self_loop,
        )
        manifest["snapshot_files"][manifest_key(seed_time)] = stats.to_manifest_entry(
            file=path.name,
            file_size_bytes=os.path.getsize(path),
        )
        write_manifest_atomic(args.snapshot_dir, manifest)
        print(
            f"{pd.Timestamp(seed_time)}: candidates={stats.num_destination_candidates} "
            f"anchors={stats.num_anchors} edges={stats.num_edges} "
            f"avg_edges_per_anchor={stats.avg_edges_per_anchor:.2f} "
            f"retrieval={stats.retrieval_seconds:.2f}s"
        )


if __name__ == "__main__":
    main()
