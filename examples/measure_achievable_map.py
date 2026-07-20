"""
Achievable-MAP 진단 (모델/학습 불필요).

achievable-MAP@K[u] = min(covered[u], K) / min(R[u], K)
  covered[u] = 유저 u의 정답 중 subgraph(=후보)에 샘플된 개수
  R[u]       = 유저 u의 전체 정답 수

이 값은 "현재 커버리지에서 랭킹을 완벽히 했을 때의 MAP 상한"이며,
오직 샘플러(NeighborLoader) 구조로만 결정된다 -> model.forward / 학습이 전혀 필요 없다.
realized-MAP은 별도(학습된 모델)로 이미 확보했다고 가정한다.

사용 예:
  python measure_achievable_map.py --dataset rel-hm --task user-item-purchase \
      --num_neighbors 16,32,64,128,256 --splits val

여러 num_neighbors 값을 주면 예산별 achievable 곡선을 한 번에 뽑는다.
"""

import argparse
import json
import os
import warnings
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from text_embedder import GloveTextEmbedding
from torch_frame import stype
from torch_frame.config.text_embedder import TextEmbedderConfig
from torch_geometric.loader import NeighborLoader
from torch_geometric.seed import seed_everything
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
parser.add_argument("--batch_size", type=int, default=512)
parser.add_argument("--num_layers", type=int, default=2)
# 콤마로 여러 값을 주면 예산별로 스윕한다 (예: "16,32,64,128,256")
parser.add_argument("--num_neighbors", type=str, default="128")
parser.add_argument("--temporal_strategy", type=str, default="last")
# 콤마로 여러 스플릿 (val은 라벨 확실, test는 라벨 없으면 자동 스킵)
parser.add_argument("--splits", type=str, default="val")
parser.add_argument("--num_workers", type=int, default=0)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument(
    "--cache_dir", type=str, default=os.path.expanduser("~/.cache/relbench_examples")
)
args = parser.parse_args()

seed_everything(args.seed)

# 텍스트 임베딩(그래프 materialize)은 1회만 필요하고 캐시된다.
# 커버리지는 '구조'만으로 결정되므로 이 device는 성능에만 영향, 결과엔 무관.
embed_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# 커버리지 계산 자체는 CPU에서 (작은 인덱스 텐서들이라 충분히 빠름)
cov_device = torch.device("cpu")

dataset: Dataset = get_dataset(args.dataset, download=True)
task: RecommendationTask = get_task(args.dataset, args.task, download=True)
assert task.task_type == TaskType.LINK_PREDICTION
eval_k = task.eval_k

# ---- stype 캐시 (원본 스크립트와 동일) ----
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

# ---- 그래프 materialize (1회, 캐시됨) ----
data, col_stats_dict = make_pkey_fkey_graph(
    dataset.get_db(),
    col_to_stype_dict=col_to_stype_dict,
    text_embedder_cfg=TextEmbedderConfig(
        text_embedder=GloveTextEmbedding(device=embed_device), batch_size=256
    ),
    cache_dir=f"{args.cache_dir}/{args.dataset}/materialized",
)

# ---- 스플릿별 정답 매핑(SparseTensor)과 seed 노드 입력을 1회 구성 ----
# (num_neighbors에 독립적이므로 스윕 중 재사용)
split_list = [s.strip() for s in args.splits.split(",") if s.strip()]
table_input_dict = {}
sparse_tensor_dict: Dict[str, SparseTensor] = {}
for split in split_list:
    table = task.get_table(split)
    table_input = get_link_train_table_input(table, task)
    table_input_dict[split] = table_input
    sparse_tensor_dict[split] = SparseTensor(
        table_input.dst_nodes[1], device=cov_device
    )


@torch.no_grad()
def measure_coverage(
    loader: NeighborLoader, sparse_tensor: SparseTensor
) -> Tuple[np.ndarray, np.ndarray]:
    """모델 없이 유저별 (covered, total_gt)를 수집한다."""
    covered_list: List[torch.Tensor] = []
    total_list: List[torch.Tensor] = []
    for batch in tqdm(loader, desc="coverage"):
        # 주의: batch를 device로 옮기지 않는다 (feature 불필요). 전부 CPU에서 처리.
        batch_size = batch[task.src_entity_table].batch_size
        input_id = batch[task.src_entity_table].input_id
        src_batch, dst_index = sparse_tensor[input_id]  # 정답 (유저-local, 아이템-global)

        # train()의 target 생성과 동일한 key 인코딩으로 "정답이 subgraph에 있나" 판정
        gt_key = src_batch + batch_size * dst_index
        sampled_key = (
            batch[task.dst_entity_table].batch
            + batch_size * batch[task.dst_entity_table].n_id
        )
        gt_in_sub = torch.isin(gt_key, sampled_key)

        total_gt = torch.bincount(src_batch, minlength=batch_size)
        covered = torch.bincount(src_batch[gt_in_sub], minlength=batch_size)

        covered_list.append(covered)
        total_list.append(total_gt)

    covered = torch.cat(covered_list).numpy()
    total_gt = torch.cat(total_list).numpy()
    return covered, total_gt


def achievable_from_coverage(
    covered: np.ndarray, total_gt: np.ndarray, k: int
) -> Dict[str, float]:
    """(covered, total_gt) -> achievable-MAP 및 커버리지 통계. R==0 유저 제외."""
    mask = total_gt > 0
    if mask.sum() == 0:
        warnings.warn("정답이 있는 유저가 없습니다 (해당 스플릿 라벨이 없을 수 있음).")
        return {"num_users": 0, "achievable_map": float("nan")}

    covered = covered[mask].astype(np.float64)
    R = total_gt[mask].astype(np.float64)

    c = np.minimum(covered, k)          # top-k에 넣을 수 있는 커버된 정답 수
    achievable_ap = c / np.minimum(R, k)  # 유저별 achievable AP@K

    return {
        "num_users": int(mask.sum()),
        "mean_gt_per_user": float(R.mean()),
        "coverage_rate": float((covered / R).mean()),   # 후보의 정답 recall (raw)
        "user_hit_rate": float((covered > 0).mean()),   # 정답 1개+ 든 유저 비율
        "achievable_map": float(achievable_ap.mean()),  # <-- 핵심 지표
    }


# ---- num_neighbors 스윕 ----
nn_values = [int(x) for x in args.num_neighbors.split(",") if x.strip()]
results = []  # (split, nn, stats)

for nn in nn_values:
    num_neighbors = [int(nn // 2**i) for i in range(args.num_layers)]
    for split in split_list:
        table_input = table_input_dict[split]
        loader = NeighborLoader(
            data,
            num_neighbors=num_neighbors,
            time_attr="time",
            input_nodes=table_input.src_nodes,
            input_time=table_input.src_time,
            subgraph_type="bidirectional",
            batch_size=args.batch_size,
            temporal_strategy=args.temporal_strategy,
            shuffle=False,
            num_workers=args.num_workers,
            persistent_workers=args.num_workers > 0,
        )
        print(f"\n[measuring] split={split}  num_neighbors={num_neighbors}")
        try:
            covered, total_gt = measure_coverage(loader, sparse_tensor_dict[split])
            stats = achievable_from_coverage(covered, total_gt, eval_k)
        except Exception as e:
            warnings.warn(f"split={split} 진단 스킵: {e}")
            continue
        results.append((split, nn, stats))

        if stats.get("num_users", 0) > 0:
            print(
                f"  users={stats['num_users']}  "
                f"mean_GT/user={stats['mean_gt_per_user']:.2f}"
            )
            print(
                f"  coverage_rate={stats['coverage_rate']:.4f}  "
                f"user_hit_rate={stats['user_hit_rate']:.4f}"
            )
            print(
                f"  Achievable-MAP@{eval_k} = {stats['achievable_map']:.4f}  "
                f"(Ceiling gap = {1.0 - stats['achievable_map']:.4f})"
            )

# ---- 요약 표 (예산별 achievable 곡선용) ----
print("\n================ Achievable-MAP summary ================")
print(f"dataset={args.dataset}  task={args.task}  K={eval_k}")
print(f"{'split':<6} {'num_nbr':>8} {'coverage':>10} {'hit_rate':>10} "
      f"{'ach_MAP':>10} {'ceiling_gap':>12}")
for split, nn, stats in results:
    if stats.get("num_users", 0) == 0:
        continue
    print(
        f"{split:<6} {nn:>8} {stats['coverage_rate']:>10.4f} "
        f"{stats['user_hit_rate']:>10.4f} {stats['achievable_map']:>10.4f} "
        f"{1.0 - stats['achievable_map']:>12.4f}"
    )
print("========================================================")
print("해석: realized-MAP(이미 확보) 대비")
print("  realized ≈ achievable ≪ 1  -> 병목은 '커버리지'. Fill로 상한을 올려야 함.")
print("  achievable ≫ realized      -> 병목은 '랭킹'. ranker/학습을 손봐야 함.")