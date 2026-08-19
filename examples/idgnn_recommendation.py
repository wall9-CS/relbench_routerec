import argparse
import copy
import gc
import json
import os
import random
import warnings
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from model import Model
from text_embedder import GloveTextEmbedding
from torch import Tensor
from torch_frame import stype
from torch_frame.config.text_embedder import TextEmbedderConfig
from torch_geometric.loader import NeighborLoader
from torch_geometric.seed import seed_everything
from torch_geometric.typing import NodeType
from tqdm import tqdm

from relbench.base import Dataset, RecommendationTask, TaskType
from relbench.datasets import get_dataset
from relbench.modeling.graph import get_link_train_table_input, make_pkey_fkey_graph
from relbench.modeling.loader import SparseTensor
from relbench.modeling.utils import get_stype_proposal
from relbench.tasks import get_task

try:
    from latent_relation.graph import (
        attach_latent_snapshot,
        build_latent_num_neighbors,
        build_latent_schema_template,
    )
    from latent_relation.io import load_snapshot as load_latent_snapshot
    from latent_relation.io import validate_manifest_for_training as validate_latent_manifest
    from hm_cf.seed_time_loader import (
        SeedTimeGroup,
        group_recommendation_table_by_seed_time,
        make_seed_time_loader,
        scatter_group_predictions,
    )
except ImportError:
    from .latent_relation.graph import (
        attach_latent_snapshot,
        build_latent_num_neighbors,
        build_latent_schema_template,
    )
    from .latent_relation.io import load_snapshot as load_latent_snapshot
    from .latent_relation.io import (
        validate_manifest_for_training as validate_latent_manifest,
    )
    from .hm_cf.seed_time_loader import (
        SeedTimeGroup,
        group_recommendation_table_by_seed_time,
        make_seed_time_loader,
        scatter_group_predictions,
    )

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, default="rel-hm")
parser.add_argument("--task", type=str, default="user-item-purchase")
parser.add_argument("--lr", type=float, default=0.001)
parser.add_argument("--epochs", type=int, default=20)
parser.add_argument("--eval_epochs_interval", type=int, default=1)
parser.add_argument("--batch_size", type=int, default=512)
parser.add_argument("--channels", type=int, default=128)
parser.add_argument("--aggr", type=str, default="sum")
parser.add_argument("--num_layers", type=int, default=2)
parser.add_argument("--num_neighbors", type=int, default=128)
parser.add_argument("--temporal_strategy", type=str, default="last")
parser.add_argument("--max_steps_per_epoch", type=int, default=2000)
parser.add_argument("--num_workers", type=int, default=0)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--augmentation", choices=["none", "latent"], default="none")
parser.add_argument("--latent-snapshot-dir", type=Path, default=None)
parser.add_argument("--budget-mode", choices=["additive", "fixed"], default="additive")
parser.add_argument("--top-l", type=int, default=32)
parser.add_argument(
    "--cache_dir", type=str, default=os.path.expanduser("/data/seonghun/.cache/relbench_examples")
)
args = parser.parse_args()


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    torch.set_num_threads(1)
seed_everything(args.seed)
random.seed(args.seed)

dataset: Dataset = get_dataset(args.dataset, download=True)
task: RecommendationTask = get_task(args.dataset, args.task, download=True)
tune_metric = "link_prediction_map"
assert task.task_type == TaskType.LINK_PREDICTION

stypes_cache_path = Path(f"{args.cache_dir}/{args.dataset}/stypes.json")
try:
    with open(stypes_cache_path, "r") as f:
        col_to_stype_dict = json.load(f)
    for table, col_to_stype in col_to_stype_dict.items():
        for col, stype_str in col_to_stype.items():
            col_to_stype[col] = stype(stype_str)
except FileNotFoundError:
    col_to_stype_dict = get_stype_proposal(dataset.get_db())
    Path(stypes_cache_path).parent.mkdir(parents=True, exist_ok=True)
    with open(stypes_cache_path, "w") as f:
        json.dump(col_to_stype_dict, f, indent=2, default=str)

data, col_stats_dict = make_pkey_fkey_graph(
    dataset.get_db(),
    col_to_stype_dict=col_to_stype_dict,
    text_embedder_cfg=TextEmbedderConfig(
        text_embedder=GloveTextEmbedding(device=device), batch_size=256
    ),
    cache_dir=f"{args.cache_dir}/{args.dataset}/materialized",
)

num_neighbors = [int(args.num_neighbors // 2**i) for i in range(args.num_layers)]

loader_dict: Dict[str, NeighborLoader] = {}
dst_nodes_dict: Dict[str, Tuple[NodeType, Tensor]] = {}
groups = None
latent_config = None
model_data = data

if args.augmentation == "none":
    for split in ["train", "val", "test"]:
        table = task.get_table(split)
        table_input = get_link_train_table_input(table, task)
        dst_nodes_dict[split] = table_input.dst_nodes
        loader_dict[split] = NeighborLoader(
            data,
            num_neighbors=num_neighbors,
            time_attr="time",
            input_nodes=table_input.src_nodes,
            input_time=table_input.src_time,
            subgraph_type="bidirectional",
            batch_size=args.batch_size,
            temporal_strategy=args.temporal_strategy,
            shuffle=split == "train",
            num_workers=args.num_workers,
            persistent_workers=args.num_workers > 0,
        )
elif args.augmentation == "latent":
    if args.latent_snapshot_dir is None:
        raise ValueError("--latent-snapshot-dir is required with --augmentation latent.")
    if args.num_layers < 3:
        raise ValueError("Latent augmentation requires --num_layers >= 3.")
    latent_config = validate_latent_manifest(args.latent_snapshot_dir)
    if latent_config.dataset != args.dataset or latent_config.task != args.task:
        raise ValueError(
            f"Snapshot manifest is for {latent_config.dataset}/{latent_config.task}, "
            f"but script requested {args.dataset}/{args.task}."
        )
    model_data = build_latent_schema_template(
        data,
        destination_type=task.dst_entity_table,
    )
    num_neighbors = build_latent_num_neighbors(
        model_data.edge_types,
        destination_type=task.dst_entity_table,
        num_layers=args.num_layers,
        num_neighbors=args.num_neighbors,
        top_l=args.top_l,
        budget_mode=args.budget_mode,
    )
    groups = {
        split: group_recommendation_table_by_seed_time(task.get_table(split), task)
        for split in ["train", "val", "test"]
    }
    for split_groups in groups.values():
        for group in split_groups:
            load_latent_snapshot(
                args.latent_snapshot_dir,
                group.seed_time,
                validate=True,
                config=latent_config,
                num_destinations=task.num_dst_nodes,
            )
    print(
        "Latent augmentation budget: "
        f"mode={args.budget_mode}, num_layers={args.num_layers}, "
        f"base_num_neighbors={args.num_neighbors}, top_l={args.top_l}"
    )

model = Model(
    data=model_data,
    col_stats_dict=col_stats_dict,
    num_layers=args.num_layers,
    channels=args.channels,
    out_channels=1,
    aggr=args.aggr,
    norm="layer_norm",
    id_awareness=True,
).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
train_sparse_tensor = (
    SparseTensor(dst_nodes_dict["train"][1], device=device)
    if args.augmentation == "none"
    else None
)


def _load_latent_group_graph(group: SeedTimeGroup):
    snapshot = load_latent_snapshot(
        args.latent_snapshot_dir,
        group.seed_time,
        validate=True,
        config=latent_config,
        num_destinations=task.num_dst_nodes,
    )
    return attach_latent_snapshot(
        data,
        snapshot,
        group.seed_time,
        destination_type=task.dst_entity_table,
        num_destinations=task.num_dst_nodes,
    )


def train() -> float:
    model.train()

    loss_accum = count_accum = 0
    steps = 0
    if args.augmentation == "none":
        total_steps = min(len(loader_dict["train"]), args.max_steps_per_epoch)
        iterator = tqdm(loader_dict["train"], total=total_steps)
        for batch in iterator:
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
            if steps > args.max_steps_per_epoch:
                break
    else:
        train_groups = list(groups["train"])
        random.shuffle(train_groups)
        pbar = tqdm(total=args.max_steps_per_epoch)
        for group in train_groups:
            graph = _load_latent_group_graph(group)
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
            sparse_tensor = SparseTensor(table_input.dst_nodes[1], device=device)
            for batch in loader:
                batch = batch.to(device)
                out = model.forward_dst_readout(
                    batch, task.src_entity_table, task.dst_entity_table
                ).flatten()
                batch_size = batch[task.src_entity_table].batch_size
                input_id = batch[task.src_entity_table].input_id
                src_batch, dst_index = sparse_tensor[input_id]
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
            f"Did not sample a single '{task.dst_entity_table}' "
            f"node in any mini-batch. Try to increase the number "
            f"of layers/hops and re-try. If you run into memory "
            f"issues with deeper nets, decrease the batch size."
        )

    return loss_accum / count_accum if count_accum > 0 else float("nan")


@torch.no_grad()
def test(loader: NeighborLoader) -> np.ndarray:
    model.eval()

    pred_list: list[Tensor] = []
    for batch in tqdm(loader):
        batch = batch.to(device)
        out = (
            model.forward_dst_readout(
                batch, task.src_entity_table, task.dst_entity_table
            )
            .detach()
            .flatten()
        )
        batch_size = batch[task.src_entity_table].batch_size
        scores = torch.zeros(batch_size, task.num_dst_nodes, device=out.device)
        scores[
            batch[task.dst_entity_table].batch, batch[task.dst_entity_table].n_id
        ] = torch.sigmoid(out)
        _, pred_mini = torch.topk(scores, k=task.eval_k, dim=1)
        pred_list.append(pred_mini)
    pred = torch.cat(pred_list, dim=0).cpu().numpy()
    return pred


@torch.no_grad()
def test_latent_group(group: SeedTimeGroup) -> np.ndarray:
    model.eval()
    graph = _load_latent_group_graph(group)
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
    sampled_nodes = []
    sampled_edges = []
    candidate_counts = []
    for batch in tqdm(loader):
        batch = batch.to(device)
        out = (
            model.forward_dst_readout(
                batch, task.src_entity_table, task.dst_entity_table
            )
            .detach()
            .flatten()
        )
        batch_size = batch[task.src_entity_table].batch_size
        scores = torch.zeros(batch_size, task.num_dst_nodes, device=out.device)
        scores[
            batch[task.dst_entity_table].batch,
            batch[task.dst_entity_table].n_id,
        ] = torch.sigmoid(out)
        _, pred_mini = torch.topk(scores, k=task.eval_k, dim=1)
        input_id = batch[task.src_entity_table].input_id.cpu().numpy()
        pred[input_id] = pred_mini.cpu().numpy()
        sampled_nodes.append(sum(batch.num_nodes_dict.values()) / max(batch_size, 1))
        sampled_edges.append(
            sum(store.edge_index.size(1) for store in batch.edge_stores)
            / max(batch_size, 1)
        )
        candidate_counts.append(batch[task.dst_entity_table].batch.numel() / max(batch_size, 1))
    if sampled_nodes:
        print(
            "Sampled graph stats: "
            f"avg_nodes_per_query={float(np.mean(sampled_nodes)):.2f}, "
            f"avg_edges_per_query={float(np.mean(sampled_edges)):.2f}, "
            f"avg_destination_candidates_per_query={float(np.mean(candidate_counts)):.2f}"
        )
    del loader, graph
    gc.collect()
    return pred


def test_latent_split(split: str) -> np.ndarray:
    table = task.get_table(split)
    pred = np.empty((len(table.df), task.eval_k), dtype=np.int64)
    for group in groups[split]:
        scatter_group_predictions(pred, group, test_latent_group(group))
    return pred


state_dict = None
best_val_metric = 0
for epoch in range(1, args.epochs + 1):
    train_loss = train()
    if epoch % args.eval_epochs_interval == 0:
        val_pred = (
            test(loader_dict["val"])
            if args.augmentation == "none"
            else test_latent_split("val")
        )
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
val_pred = (
    test(loader_dict["val"])
    if args.augmentation == "none"
    else test_latent_split("val")
)
val_metrics = task.evaluate(val_pred, task.get_table("val"))
print(f"Best Val metrics: {val_metrics}")

test_pred = (
    test(loader_dict["test"])
    if args.augmentation == "none"
    else test_latent_split("test")
)
test_metrics = task.evaluate(test_pred)
print(f"Best test metrics: {test_metrics}")
