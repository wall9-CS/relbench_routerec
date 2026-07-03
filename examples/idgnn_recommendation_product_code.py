import argparse
import copy
import json
import os
import sys
import warnings
from pathlib import Path
from typing import Dict, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import pandas as pd
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

from relbench.base import Database, Dataset, RecommendationTask, Table, TaskType
from relbench.datasets import get_dataset
from relbench.modeling.graph import get_link_train_table_input, make_pkey_fkey_graph
from relbench.modeling.loader import SparseTensor
from relbench.modeling.utils import get_stype_proposal
from relbench.tasks import get_task


def _safe_cache_part(value: object) -> str:
    raw = str(value).lower()
    return "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in raw)


def augment_hm_product_code_virtual_transactions(
    db: Database,
    feature_col: str,
    virtual_table_name: str,
    product_code_fanout: int,
    candidate_pool_size: int,
    exclude_existing_transactions: bool,
    user_product_time_strategy: str,
    article_time_strategy: str,
    candidate_sampling: str,
    virtual_build_chunk_size: int,
    max_virtual_rows: int,
    seed: int,
) -> tuple[Database, pd.DataFrame]:
    if "transactions" not in db.table_dict:
        raise ValueError("Expected rel-hm table 'transactions'.")
    if "article" not in db.table_dict:
        raise ValueError("Expected rel-hm table 'article'.")

    transactions = (
        db.table_dict["transactions"]
        .df[["t_dat", "customer_id", "article_id"]]
        .dropna()
        .copy()
    )
    article_df = db.table_dict["article"].df
    if feature_col not in article_df.columns:
        raise ValueError(
            f"article is missing feature column {feature_col!r}. "
            f"Available columns: {list(article_df.columns)}"
        )

    transactions["customer_id"] = transactions["customer_id"].astype(np.int64)
    transactions["article_id"] = transactions["article_id"].astype(np.int64)
    transactions["t_unix"] = transactions["t_dat"].astype(np.int64) // 10**9

    article_info = article_df[["article_id", feature_col]].dropna().copy()
    article_info["article_id"] = article_info["article_id"].astype(np.int64)

    tx_with_feature = transactions.merge(
        article_info,
        on="article_id",
        how="inner",
        copy=False,
    )

    article_time_agg = "max" if article_time_strategy == "last" else "min"
    user_time_agg = "max" if user_product_time_strategy == "last" else "min"

    article_time = (
        tx_with_feature.groupby("article_id", sort=False)["t_unix"]
        .agg(article_time_agg)
        .rename("article_t_unix")
        .reset_index()
    )
    candidate_df = article_info.merge(article_time, on="article_id", how="inner")
    candidate_pool_size = max(candidate_pool_size, product_code_fanout)

    if candidate_sampling == "recent":
        candidate_df = candidate_df.sort_values(
            [feature_col, "article_t_unix", "article_id"],
            ascending=[True, False, False],
        )
        candidate_df = (
            candidate_df.groupby(feature_col, sort=False)
            .head(candidate_pool_size)
            .reset_index(drop=True)
        )
    elif candidate_sampling == "random":
        rng = np.random.default_rng(seed)
        sampled = []
        for _, group in candidate_df.groupby(feature_col, sort=False):
            if len(group) > candidate_pool_size:
                sampled_idx = rng.choice(
                    group.index.to_numpy(),
                    candidate_pool_size,
                    replace=False,
                )
                group = group.loc[sampled_idx]
            sampled.append(group)
        candidate_df = pd.concat(sampled, ignore_index=True) if sampled else candidate_df
        candidate_df = candidate_df.sort_values(
            [feature_col, "article_t_unix", "article_id"],
            ascending=[True, False, False],
        )
    else:
        raise ValueError(
            f"Unknown candidate_sampling={candidate_sampling!r}. "
            "Expected 'recent' or 'random'."
        )

    user_feature_time = (
        tx_with_feature.groupby(["customer_id", feature_col], sort=False)["t_unix"]
        .agg(user_time_agg)
        .rename("user_t_unix")
        .reset_index()
    )

    num_articles = len(db.table_dict["article"])
    existing_pair_keys = None
    if exclude_existing_transactions:
        existing_pair_keys = np.unique(
            transactions["customer_id"].to_numpy(np.int64) * np.int64(num_articles)
            + transactions["article_id"].to_numpy(np.int64)
        )

    chunks: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    total_rows = 0
    candidate_groups = {
        feature: (
            group["article_id"].to_numpy(np.int64),
            group["article_t_unix"].to_numpy(np.int64),
        )
        for feature, group in candidate_df.groupby(feature_col, sort=False)
    }

    if virtual_build_chunk_size < 1:
        raise ValueError("--virtual_build_chunk_size must be positive.")

    for feature, users in tqdm(
        user_feature_time.groupby(feature_col, sort=False),
        total=user_feature_time[feature_col].nunique(),
        desc="Building virtual product_code transactions (numpy)",
    ):
        candidates = candidate_groups.get(feature)
        if candidates is None:
            continue

        cand_articles, cand_times = candidates
        if cand_articles.size == 0:
            continue

        user_ids = users["customer_id"].to_numpy(np.int64)
        user_times = users["user_t_unix"].to_numpy(np.int64)

        for start in range(0, user_ids.size, virtual_build_chunk_size):
            stop = min(start + virtual_build_chunk_size, user_ids.size)
            user_chunk = user_ids[start:stop]
            user_time_chunk = user_times[start:stop]

            if exclude_existing_transactions and existing_pair_keys is not None:
                pair_keys = (
                    user_chunk[:, None] * np.int64(num_articles)
                    + cand_articles[None, :]
                )
                valid = ~np.isin(pair_keys, existing_pair_keys, assume_unique=True)
            else:
                valid = np.ones(
                    (user_chunk.size, cand_articles.size),
                    dtype=bool,
                )

            keep = valid & (np.cumsum(valid, axis=1) <= product_code_fanout)
            row_idx, cand_idx = np.nonzero(keep)
            if row_idx.size == 0:
                continue

            out_users = user_chunk[row_idx]
            out_articles = cand_articles[cand_idx]
            out_t = np.maximum(user_time_chunk[row_idx], cand_times[cand_idx])

            if max_virtual_rows > 0 and total_rows + out_users.size > max_virtual_rows:
                keep_n = max_virtual_rows - total_rows
                out_users = out_users[:keep_n]
                out_articles = out_articles[:keep_n]
                out_t = out_t[:keep_n]

            chunks.append((out_users, out_articles, out_t))
            total_rows += out_users.size

            if max_virtual_rows > 0 and total_rows >= max_virtual_rows:
                warnings.warn(
                    f"Reached --max_virtual_rows={max_virtual_rows:,}; "
                    "virtual table was truncated."
                )
                break

        if max_virtual_rows > 0 and total_rows >= max_virtual_rows:
            break

    if chunks:
        virtual_df = pd.DataFrame(
            {
                "customer_id": np.concatenate([chunk[0] for chunk in chunks]),
                "article_id": np.concatenate([chunk[1] for chunk in chunks]),
                "t_unix": np.concatenate([chunk[2] for chunk in chunks]),
            }
        )
        virtual_df = virtual_df.drop_duplicates(
            subset=["customer_id", "article_id"], keep="first"
        )
        virtual_df["t_dat"] = pd.to_datetime(virtual_df["t_unix"], unit="s")
        virtual_df = virtual_df[["t_dat", "customer_id", "article_id"]]
    else:
        virtual_df = pd.DataFrame(
            {
                "t_dat": pd.Series(dtype="datetime64[ns]"),
                "customer_id": pd.Series(dtype=np.int64),
                "article_id": pd.Series(dtype=np.int64),
            }
        )

    table_dict = dict(db.table_dict)
    table_dict[virtual_table_name] = Table(
        df=virtual_df,
        fkey_col_to_pkey_table={
            "customer_id": "customer",
            "article_id": "article",
        },
        pkey_col=None,
        time_col="t_dat",
    )
    return Database(table_dict), virtual_df


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
parser.add_argument("--product_code_feature_col", type=str, default="product_code")
parser.add_argument("--product_code_fanout", type=int, default=16)
parser.add_argument(
    "--candidate_pool_size",
    type=int,
    default=128,
    help=(
        "Number of same-product-code articles to keep per product_code "
        "before excluding existing user-article transactions."
    ),
)
parser.add_argument(
    "--candidate_sampling",
    choices=["recent", "random"],
    default="recent",
)
parser.add_argument(
    "--exclude_existing_transactions",
    action=argparse.BooleanOptionalAction,
    default=True,
)
parser.add_argument(
    "--user_product_time_strategy",
    choices=["last", "first"],
    default="last",
)
parser.add_argument(
    "--article_time_strategy",
    choices=["last", "first"],
    default="last",
)
parser.add_argument("--virtual_build_chunk_size", type=int, default=50_000)
parser.add_argument("--max_virtual_rows", type=int, default=0)
parser.add_argument(
    "--dry_run_virtual_build",
    action=argparse.BooleanOptionalAction,
    default=False,
)
parser.add_argument(
    "--virtual_table_name",
    type=str,
    default="virtual_product_code_transactions",
)
parser.add_argument(
    "--cache_dir", type=str, default=os.path.expanduser("~/.cache/relbench_examples")
)
args = parser.parse_args()

if args.dataset != "rel-hm" or args.task != "user-item-purchase":
    raise ValueError(
        "This product-code virtual transaction example is currently rel-hm/"
        "user-item-purchase specific."
    )
if args.product_code_fanout < 1:
    raise ValueError("--product_code_fanout must be positive.")
if args.candidate_pool_size < args.product_code_fanout:
    raise ValueError("--candidate_pool_size must be >= --product_code_fanout.")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    torch.set_num_threads(1)
seed_everything(args.seed)

dataset: Dataset = get_dataset(args.dataset, download=True)
task: RecommendationTask = get_task(args.dataset, args.task, download=True)
tune_metric = "link_prediction_map"
assert task.task_type == TaskType.LINK_PREDICTION

db, virtual_df = augment_hm_product_code_virtual_transactions(
    dataset.get_db(),
    feature_col=args.product_code_feature_col,
    virtual_table_name=args.virtual_table_name,
    product_code_fanout=args.product_code_fanout,
    candidate_pool_size=args.candidate_pool_size,
    exclude_existing_transactions=args.exclude_existing_transactions,
    user_product_time_strategy=args.user_product_time_strategy,
    article_time_strategy=args.article_time_strategy,
    candidate_sampling=args.candidate_sampling,
    virtual_build_chunk_size=args.virtual_build_chunk_size,
    max_virtual_rows=args.max_virtual_rows,
    seed=args.seed,
)
print(
    f"Added {args.virtual_table_name}: {len(virtual_df):,} rows "
    f"(fanout={args.product_code_fanout}, pool={args.candidate_pool_size}, "
    f"sampling={args.candidate_sampling}, "
    f"user_time={args.user_product_time_strategy}, "
    f"article_time={args.article_time_strategy})"
)
if args.dry_run_virtual_build:
    print(virtual_df.head())
    raise SystemExit(0)

cache_suffix = (
    f"product_code_virtual_"
    f"{args.product_code_feature_col}_"
    f"f{args.product_code_fanout}_"
    f"p{args.candidate_pool_size}_"
    f"u{args.user_product_time_strategy}_"
    f"a{args.article_time_strategy}_"
    f"sampling{args.candidate_sampling}_"
    f"seed{args.seed}_"
    f"exclude{int(args.exclude_existing_transactions)}_"
    f"max{args.max_virtual_rows}"
)
cache_suffix = _safe_cache_part(cache_suffix)

stypes_cache_path = Path(
    f"{args.cache_dir}/{args.dataset}/{cache_suffix}/stypes.json"
)
try:
    with open(stypes_cache_path, "r") as f:
        col_to_stype_dict = json.load(f)
    for table, col_to_stype in col_to_stype_dict.items():
        for col, stype_str in col_to_stype.items():
            col_to_stype[col] = stype(stype_str)
except FileNotFoundError:
    col_to_stype_dict = get_stype_proposal(db)
    Path(stypes_cache_path).parent.mkdir(parents=True, exist_ok=True)
    with open(stypes_cache_path, "w") as f:
        json.dump(col_to_stype_dict, f, indent=2, default=str)

data, col_stats_dict = make_pkey_fkey_graph(
    db,
    col_to_stype_dict=col_to_stype_dict,
    text_embedder_cfg=TextEmbedderConfig(
        text_embedder=GloveTextEmbedding(device=device), batch_size=256
    ),
    cache_dir=f"{args.cache_dir}/{args.dataset}/{cache_suffix}/materialized",
)
print(f"Node types: {data.node_types}")
print(f"Edge types: {data.edge_types}")

num_neighbors = [int(args.num_neighbors // 2**i) for i in range(args.num_layers)]

loader_dict: Dict[str, NeighborLoader] = {}
dst_nodes_dict: Dict[str, Tuple[NodeType, Tensor]] = {}
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

model = Model(
    data=data,
    col_stats_dict=col_stats_dict,
    num_layers=args.num_layers,
    channels=args.channels,
    out_channels=1,
    aggr=args.aggr,
    norm="layer_norm",
    id_awareness=True,
).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
train_sparse_tensor = SparseTensor(dst_nodes_dict["train"][1], device=device)


def train() -> float:
    model.train()

    loss_accum = count_accum = 0
    steps = 0
    total_steps = min(len(loader_dict["train"]), args.max_steps_per_epoch)
    for batch in tqdm(loader_dict["train"], total=total_steps):
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


state_dict = None
best_val_metric = 0
for epoch in range(1, args.epochs + 1):
    train_loss = train()
    if epoch % args.eval_epochs_interval == 0:
        val_pred = test(loader_dict["val"])
        val_metrics = task.evaluate(val_pred, task.get_table("val"))
        print(
            f"Epoch: {epoch:02d}, Train loss: {train_loss}, "
            f"Val metrics: {val_metrics}"
        )

        if val_metrics[tune_metric] > best_val_metric:
            best_val_metric = val_metrics[tune_metric]
            state_dict = copy.deepcopy(model.state_dict())


if state_dict is None:
    state_dict = copy.deepcopy(model.state_dict())

model.load_state_dict(state_dict)
val_pred = test(loader_dict["val"])
val_metrics = task.evaluate(val_pred, task.get_table("val"))
print(f"Best Val metrics: {val_metrics}")

test_pred = test(loader_dict["test"])
test_metrics = task.evaluate(test_pred)
print(f"Best test metrics: {test_metrics}")
