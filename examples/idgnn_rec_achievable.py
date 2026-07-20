import argparse
import copy
import json
import os
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
parser.add_argument(
    "--cache_dir", type=str, default=os.path.expanduser("~/.cache/relbench_examples")
)
args = parser.parse_args()


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    torch.set_num_threads(1)
seed_everything(args.seed)

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

# === [수정 1] 진단을 위해 val/test 스플릿의 SparseTensor(=정답 매핑)도 함께 구성 ===
# 기존에는 train만 만들었으나, 커버리지 측정에는 각 스플릿의 정답이 필요하다.
sparse_tensor_dict: Dict[str, SparseTensor] = {
    split: SparseTensor(dst_nodes_dict[split][1], device=device)
    for split in ["train", "val", "test"]
}
train_sparse_tensor = sparse_tensor_dict["train"]  # 기존 이름 유지 (train()에서 사용)


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

        # Get ground-truth
        input_id = batch[task.src_entity_table].input_id
        src_batch, dst_index = train_sparse_tensor[input_id]

        # Get target label
        target = torch.isin(
            batch[task.dst_entity_table].batch
            + batch_size * batch[task.dst_entity_table].n_id,
            src_batch + batch_size * dst_index,
        ).float()

        # Optimization
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


# === [수정 2] test()와 동일한 forward pass 안에서 커버리지까지 측정하는 함수 ===
# 반환: (pred, covered, total_gt)
#   - pred       : [num_users, eval_k]  실제 예측 (realized-MAP 계산용)
#   - covered[u] : 유저 u의 정답 중 subgraph(=후보)에 실제로 포함된 개수
#   - total_gt[u]: 유저 u의 전체 정답 개수 R
# 핵심: covered는 train()의 target 생성과 '똑같은' key 인코딩(isin)으로 계산하므로,
#       "이 아이템이 이 유저의 subgraph에 샘플되었는가"를 정확히 판정한다.
@torch.no_grad()
def test_with_coverage(
    loader: NeighborLoader, sparse_tensor: SparseTensor
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    model.eval()

    pred_list: list[Tensor] = []
    covered_list: list[Tensor] = []
    total_gt_list: list[Tensor] = []

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

        # --- (원본과 동일) realized 예측 ---
        scores = torch.zeros(batch_size, task.num_dst_nodes, device=out.device)
        scores[
            batch[task.dst_entity_table].batch, batch[task.dst_entity_table].n_id
        ] = torch.sigmoid(out)
        _, pred_mini = torch.topk(scores, k=task.eval_k, dim=1)
        pred_list.append(pred_mini)

        # --- 커버리지 진단 ---
        input_id = batch[task.src_entity_table].input_id
        src_batch, dst_index = sparse_tensor[input_id]  # 정답 (유저-local, 아이템-global) 쌍

        # train()의 target 생성과 동일한 key 인코딩
        gt_key = src_batch + batch_size * dst_index
        sampled_key = (
            batch[task.dst_entity_table].batch
            + batch_size * batch[task.dst_entity_table].n_id
        )
        gt_in_sub = torch.isin(gt_key, sampled_key)  # 각 정답 쌍이 subgraph에 있는가 (bool)

        # 유저별 전체 정답 수(R)와 커버된 정답 수
        total_gt = torch.bincount(src_batch, minlength=batch_size)
        covered = torch.bincount(src_batch[gt_in_sub], minlength=batch_size)

        covered_list.append(covered.cpu())
        total_gt_list.append(total_gt.cpu())

    pred = torch.cat(pred_list, dim=0).cpu().numpy()
    covered = torch.cat(covered_list, dim=0).numpy()
    total_gt = torch.cat(total_gt_list, dim=0).numpy()
    return pred, covered, total_gt


# === [수정 3] 커버리지 배열로부터 achievable-MAP 등을 계산 ===
# achievable-AP@K[u] = min(covered[u], K) / min(R[u], K)
#   -> "현재 subgraph에 들어온 정답을 top에 완벽히 몰아넣었을 때"의 AP.
#      이는 link_prediction_map 공식에 [1]*c + [0]*(K-c) 오라클을 넣은 것과 정확히 동일하다
#      (앞자리 c개가 모두 hit이면 precision 합 = c, 분모 = min(R,K) 이므로 c/min(R,K)).
# R==0 유저는 공식의 _filter와 동일하게 제외.
def coverage_report(
    covered: np.ndarray, total_gt: np.ndarray, eval_k: int
) -> Dict[str, float]:
    mask = total_gt > 0
    if mask.sum() == 0:
        warnings.warn("정답이 있는 유저가 없습니다 (해당 스플릿 라벨이 없을 수 있음).")
        return {
            "coverage_rate": float("nan"),
            "user_hit_rate": float("nan"),
            "achievable_map": float("nan"),
            "num_users": 0,
        }

    covered = covered[mask].astype(np.float64)
    R = total_gt[mask].astype(np.float64)

    c = np.minimum(covered, eval_k)          # top-k에 넣을 수 있는 커버된 정답 수
    denom = np.minimum(R, eval_k)            # min(K, R)
    achievable_ap = c / denom                # 유저별 achievable AP@K
    achievable_map = float(achievable_ap.mean())

    coverage_rate = float((covered / R).mean())   # 후보 집합의 정답 recall (raw)
    user_hit_rate = float((covered > 0).mean())   # 정답이 1개 이상 후보에 든 유저 비율

    return {
        "coverage_rate": coverage_rate,
        "user_hit_rate": user_hit_rate,
        "achievable_map": achievable_map,
        "num_users": int(mask.sum()),
    }


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


model.load_state_dict(state_dict)


# === [수정 4] 최종 평가를 '진단 포함' 버전으로 교체 ===
def run_diagnosis(split: str, has_label_table: bool) -> None:
    pred, covered, total_gt = test_with_coverage(
        loader_dict[split], sparse_tensor_dict[split]
    )
    if has_label_table:
        metrics = task.evaluate(pred, task.get_table(split))
    else:
        metrics = task.evaluate(pred)  # test는 내부 라벨 사용

    cov = coverage_report(covered, total_gt, task.eval_k)

    realized_map = float(metrics.get("link_prediction_map", float("nan")))
    achievable_map = cov["achievable_map"]
    ceiling_gap = 1.0 - achievable_map            # 후보 부재로 인한 손실 (coverage ceiling)
    ranking_gap = achievable_map - realized_map    # 랭킹 불완전으로 인한 손실

    print(f"\n===== [{split}] 진단 (K = {task.eval_k}, 유저 {cov['num_users']}명) =====")
    print(f"Full metrics           : {metrics}")
    print(f"Candidate coverage     : {cov['coverage_rate']:.4f}  (정답이 후보에 든 비율, raw recall)")
    print(f"User hit-rate          : {cov['user_hit_rate']:.4f}  (정답 1개+ 든 유저 비율)")
    print(f"Realized-MAP@{task.eval_k:<3d}      : {realized_map:.4f}")
    print(f"Achievable-MAP@{task.eval_k:<3d}    : {achievable_map:.4f}  (현재 커버리지 상한)")
    print(f"Ceiling gap            : {ceiling_gap:.4f}  (= 1 - Achievable, 커버리지 손실)")
    print(f"Ranking  gap           : {ranking_gap:.4f}  (= Achievable - Realized, 랭킹 손실)")
    print(
        f"분해: Realized {realized_map:.4f} "
        f"= 1 - {ceiling_gap:.4f}(coverage) - {ranking_gap:.4f}(ranking)"
    )


run_diagnosis("val", has_label_table=True)
try:
    run_diagnosis("test", has_label_table=False)
except Exception as e:  # test 라벨이 없거나 접근 불가한 경우 대비
    warnings.warn(f"test 스플릿 진단을 건너뜁니다: {e}")