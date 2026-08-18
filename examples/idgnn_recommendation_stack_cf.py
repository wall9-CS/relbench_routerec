from __future__ import annotations

import argparse
import copy
import gc
import json
import os
import random
import warnings
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch_frame import stype
from torch_frame.config.text_embedder import TextEmbedderConfig
from torch_geometric.seed import seed_everything
from tqdm import tqdm

try:
    from .model import Model
    from .text_embedder import GloveTextEmbedding
except ImportError:
    from model import Model
    from text_embedder import GloveTextEmbedding

from relbench.base import Dataset, RecommendationTask, TaskType
from relbench.datasets import get_dataset
from relbench.modeling.graph import make_pkey_fkey_graph
from relbench.modeling.loader import SparseTensor
from relbench.modeling.utils import get_stype_proposal
from relbench.tasks import get_task

from .hm_cf.seed_time_loader import (
    SeedTimeGroup,
    group_recommendation_table_by_seed_time,
    make_seed_time_loader,
    scatter_group_predictions,
)
from .stack_cf.coverage import compute_cf_coverage_for_split
from .stack_cf.graph import (
    attach_cf_snapshot,
    build_cf_num_neighbors,
    build_cf_schema_template,
    post_cf_col_stats,
)
from .stack_cf.interactions import filter_comment_interactions
from .stack_cf.io import load_snapshot, validate_manifest_for_training


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train ID-GNN with seed-time-specific Rel-Stack CF snapshots."
    )
    parser.add_argument("--dataset", type=str, default="rel-stack")
    parser.add_argument("--task", type=str, default="user-post-comment")
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--eval_epochs_interval", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--channels", type=int, default=128)
    parser.add_argument("--aggr", type=str, default="sum")
    parser.add_argument("--num_layers", type=int, default=4)
    parser.add_argument("--num_neighbors", type=int, default=128)
    parser.add_argument("--temporal_strategy", type=str, default="last")
    parser.add_argument("--max_steps_per_epoch", type=int, default=2000)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=os.path.expanduser("/data/seonghun/.cache/relbench_examples"),
    )
    parser.add_argument("--cf-snapshot-dir", type=Path, required=True)
    parser.add_argument("--report-cf-coverage", action="store_true")
    return parser.parse_args()


def _load_stypes(dataset: Dataset, args: argparse.Namespace) -> dict:
    stypes_cache_path = Path(f"{args.cache_dir}/{args.dataset}/stypes.json")
    try:
        with open(stypes_cache_path, "r", encoding="utf-8") as f:
            col_to_stype_dict = json.load(f)
        for _, col_to_stype in col_to_stype_dict.items():
            for col, stype_str in col_to_stype.items():
                col_to_stype[col] = stype(stype_str)
    except FileNotFoundError:
        col_to_stype_dict = get_stype_proposal(dataset.get_db())
        stypes_cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(stypes_cache_path, "w", encoding="utf-8") as f:
            json.dump(col_to_stype_dict, f, indent=2, default=str)
    return col_to_stype_dict


def _load_group_graph(base_data, snapshot_dir, group, config, num_posts):
    snapshot = load_snapshot(
        snapshot_dir,
        group.seed_time,
        validate=True,
        config=config,
        num_posts=num_posts,
    )
    return attach_cf_snapshot(
        base_data,
        snapshot,
        group.seed_time,
        num_posts=num_posts,
    )


def main() -> None:
    args = parse_args()
    if args.num_layers < 4:
        raise ValueError("Stack CF augmentation requires --num_layers >= 4.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.set_num_threads(1)
    seed_everything(args.seed)
    random.seed(args.seed)

    config = validate_manifest_for_training(args.cf_snapshot_dir)
    if config.dataset != args.dataset or config.task != args.task:
        raise ValueError(
            f"Snapshot manifest is for {config.dataset}/{config.task}, "
            f"but script requested {args.dataset}/{args.task}."
        )

    dataset: Dataset = get_dataset(args.dataset, download=True)
    task: RecommendationTask = get_task(args.dataset, args.task, download=True)
    tune_metric = "link_prediction_map"
    assert task.task_type == TaskType.LINK_PREDICTION
    num_posts = task.num_dst_nodes
    db = dataset.get_db()

    if args.report_cf_coverage:
        interactions = filter_comment_interactions(db.table_dict["comments"].df)
        for split in ["val", "test"]:
            metrics = compute_cf_coverage_for_split(
                task,
                split,
                interactions,
                args.cf_snapshot_dir,
                config=config,
                num_posts=num_posts,
                num_layers=args.num_layers,
            )
            print(f"Stack CF coverage {split}: {metrics.to_dict()}")

    col_to_stype_dict = _load_stypes(dataset, args)
    base_data, col_stats_dict = make_pkey_fkey_graph(
        db,
        col_to_stype_dict=col_to_stype_dict,
        text_embedder_cfg=TextEmbedderConfig(
            text_embedder=GloveTextEmbedding(device=device), batch_size=256
        ),
        cache_dir=f"{args.cache_dir}/{args.dataset}/materialized",
    )
    schema_data = build_cf_schema_template(base_data)
    col_stats_dict = {**col_stats_dict, "post_cf": post_cf_col_stats()}
    num_neighbors = build_cf_num_neighbors(
        schema_data.edge_types,
        num_layers=args.num_layers,
        num_neighbors=args.num_neighbors,
    )

    groups = {
        split: group_recommendation_table_by_seed_time(task.get_table(split), task)
        for split in ["train", "val", "test"]
    }
    for split_groups in groups.values():
        for group in split_groups:
            load_snapshot(
                args.cf_snapshot_dir,
                group.seed_time,
                validate=True,
                config=config,
                num_posts=num_posts,
            )

    model = Model(
        data=schema_data,
        col_stats_dict=col_stats_dict,
        num_layers=args.num_layers,
        channels=args.channels,
        out_channels=1,
        aggr=args.aggr,
        norm="layer_norm",
        id_awareness=True,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    def train() -> float:
        model.train()
        loss_accum = count_accum = 0
        steps = 0
        train_groups = list(groups["train"])
        random.shuffle(train_groups)
        pbar = tqdm(total=args.max_steps_per_epoch)
        for group in train_groups:
            graph = _load_group_graph(
                base_data, args.cf_snapshot_dir, group, config, num_posts
            )
            loader, table_input = make_seed_time_loader(
                graph,
                group,
                task,
                num_neighbors=num_neighbors,
                batch_size=args.batch_size,
                temporal_strategy=args.temporal_strategy,
                shuffle=True,
                num_workers=args.num_workers,
            )
            train_sparse_tensor = SparseTensor(table_input.dst_nodes[1], device=device)
            for batch in loader:
                batch = batch.to(device)
                out = model.forward_dst_readout(
                    batch, task.src_entity_table, task.dst_entity_table
                ).flatten()
                batch_size = batch[task.src_entity_table].batch_size
                input_id = batch[task.src_entity_table].input_id
                src_batch, dst_index = train_sparse_tensor[input_id]
                target = torch.isin(
                    batch[task.dst_entity_table].batch
                    + batch_size * batch[task.dst_entity_table].n_id,
                    src_batch + batch_size * dst_index,
                ).float()
                optimizer.zero_grad()
                loss = F.binary_cross_entropy_with_logits(out, target)
                loss.backward()
                optimizer.step()
                loss_accum += float(loss) * out.numel()
                count_accum += out.numel()
                steps += 1
                pbar.update(1)
                if steps >= args.max_steps_per_epoch:
                    break
            del loader, graph
            gc.collect()
            if steps >= args.max_steps_per_epoch:
                break
        pbar.close()
        if count_accum == 0:
            warnings.warn(
                f"Did not sample a single '{task.dst_entity_table}' node. "
                "Try increasing layers/hops or decreasing batch size."
            )
        return loss_accum / count_accum if count_accum > 0 else float("nan")

    @torch.no_grad()
    def test_group(group: SeedTimeGroup) -> np.ndarray:
        model.eval()
        graph = _load_group_graph(
            base_data, args.cf_snapshot_dir, group, config, num_posts
        )
        loader, _ = make_seed_time_loader(
            graph,
            group,
            task,
            num_neighbors=num_neighbors,
            batch_size=args.batch_size,
            temporal_strategy=args.temporal_strategy,
            shuffle=False,
            num_workers=args.num_workers,
        )
        pred = np.empty((len(group.table.df), task.eval_k), dtype=np.int64)
        for batch in tqdm(loader):
            batch = batch.to(device)
            out = model.forward_dst_readout(
                batch, task.src_entity_table, task.dst_entity_table
            ).detach().flatten()
            batch_size = batch[task.src_entity_table].batch_size
            scores = torch.zeros(batch_size, task.num_dst_nodes, device=out.device)
            scores[
                batch[task.dst_entity_table].batch,
                batch[task.dst_entity_table].n_id,
            ] = torch.sigmoid(out)
            _, pred_mini = torch.topk(scores, k=task.eval_k, dim=1)
            input_id = batch[task.src_entity_table].input_id.cpu().numpy()
            pred[input_id] = pred_mini.cpu().numpy()
        del loader, graph
        gc.collect()
        return pred

    def test_split(split: str) -> np.ndarray:
        table = task.get_table(split)
        pred = np.empty((len(table.df), task.eval_k), dtype=np.int64)
        for group in groups[split]:
            scatter_group_predictions(pred, group, test_group(group))
        return pred

    state_dict = None
    best_val_metric = 0
    for epoch in range(1, args.epochs + 1):
        train_loss = train()
        if epoch % args.eval_epochs_interval == 0:
            val_pred = test_split("val")
            val_metrics = task.evaluate(val_pred, task.get_table("val"))
            print(
                f"Epoch: {epoch:02d}, Train loss: {train_loss}, "
                f"Val metrics: {val_metrics}"
            )
            if val_metrics[tune_metric] > best_val_metric:
                best_val_metric = val_metrics[tune_metric]
                state_dict = copy.deepcopy(model.state_dict())

    if state_dict is not None:
        model.load_state_dict(state_dict)
    val_pred = test_split("val")
    val_metrics = task.evaluate(val_pred, task.get_table("val"))
    print(f"Best Val metrics: {val_metrics}")

    test_pred = test_split("test")
    test_metrics = task.evaluate(test_pred)
    print(f"Best test metrics: {test_metrics}")


if __name__ == "__main__":
    main()
