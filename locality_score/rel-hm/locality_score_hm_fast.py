"""
rel-hm Locality / Precision Score (fast, single-pass incremental)
=====================================================================
기존 코드는 timestamp마다 transactions.loc[t_dat <= timestamp] 필터링 +
정렬을 다시 수행(build_recent_adjacency_for_timestamp)해서, 고유
timestamp 개수가 많아지면 매우 느려짐.

이 버전은 문서1 스타일을 적용:
  - transactions를 t_dat으로 1회만 정렬.
  - val+test 전체 seed_time을 오름차순으로 "합쳐서" 순회하며,
    포인터로 이벤트를 그 시점까지만 증분(incremental) 추가.
  - customer_to_articles[cid], article_to_customers[aid] 리스트는 항상
    "현재 seed_time 이하" 이벤트만 담고 있으므로, per-row 타임스탬프
    필터링/재정렬 없이 [-N:] 슬라이싱만으로 "가장 최근 N개"를 얻을 수 있음
    (temporal_strategy="last").
  - 결과적으로 전체 복잡도가 O(n_transactions + n_rows * hop_budget)
    수준으로 떨어져서, timestamp마다 필터링을 반복하던 기존 방식보다
    훨씬 빠름.

기능(hop 개수 가변, precision/recall micro/macro, CLI, CSV/JSON 저장)은
기존 코드와 동일하게 유지.
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from relbench.datasets import get_dataset
from relbench.tasks import get_task


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="rel-hm")
    parser.add_argument("--task", type=str, default="user-item-purchase")
    parser.add_argument("--num_neighbors", type=int, default=128)
    parser.add_argument("--max_hops", type=int, default=3)
    parser.add_argument("--download", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--save_rows", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--out_dir",
        type=Path,
        default=Path("locality_score_fast/rel-hm/results"),
    )
    args = parser.parse_args()

    if args.max_hops < 1:
        raise ValueError("--max_hops must be positive.")

    odd_hops = [hop for hop in range(1, args.max_hops + 1) if hop % 2 == 1]
    # NN[hop-1] = 해당 conceptual hop에서의 이웃 수 제한.
    # 주의: customer<->article은 항상 transaction(tx) 노드를 매개로 연결되고
    # tx->article, tx->customer는 1:1이라 실질적으로 무효한 hetero layer이므로,
    # conceptual hop 1개는 hetero layer 2개를 소모함 -> 2배가 아니라 4배씩 감소.
    # (128, 32, 8, 2, ...) 가 정답이며 (128, 64, 32, 16, ...)이 아님.
    NN = [max(1, args.num_neighbors // (4**i)) for i in range(args.max_hops)]

    # ── 1. Data Loading ──────────────────────────────────────────────────
    print("Loading data...")
    dataset = get_dataset(args.dataset, download=args.download)
    task = get_task(args.dataset, args.task, download=args.download)
    db = dataset.get_db()

    src_col = task.src_entity_col  # customer_id
    dst_col = task.dst_entity_col  # article_id

    # transactions를 시간순 정렬 (temporal "last" 구현을 위해, 문서1과 동일)
    tx_df = db.table_dict["transactions"].df[["t_dat", "customer_id", "article_id"]].dropna().copy()
    tx_df["t_unix"] = tx_df["t_dat"].astype(np.int64) // 10**9
    tx_df = tx_df.sort_values("t_unix").reset_index(drop=True)
    print(f"  transactions: {len(tx_df):,} rows")

    # ── 2. GT 로드 (val/test를 합쳐서 seed_time 순으로 함께 처리) ──────────
    def load_gt(split: str) -> pd.DataFrame:
        df = task.get_table(split).df.copy()
        df["seed_time_unix"] = (
            pd.to_datetime(df[task.time_col]).astype(np.int64) // 10**9
        )
        df[dst_col] = df[dst_col].apply(
            lambda x: set(map(int, x))
            if hasattr(x, "__iter__") and not isinstance(x, str)
            else set()
        )
        df = df[df[dst_col].map(len) > 0].reset_index(drop=True)
        df["split"] = split
        return df

    gt_df = pd.concat([load_gt("val"), load_gt("test")], ignore_index=True)
    gt_df = gt_df.sort_values("seed_time_unix").reset_index(drop=True)
    for split in ["val", "test"]:
        n = len(gt_df[gt_df["split"] == split])
        print(f"  {split} rows: {n:,}")

    # ── 3. 증분 인접 리스트 구축 + 스코어링 (단일 패스) ─────────────────────
    c2a: dict = defaultdict(list)  # customer -> [article_id, ...] 시간순, incremental
    a2c: dict = defaultdict(list)  # article -> [customer_id, ...] 시간순, incremental

    tx_ptr = 0
    tx_cust = tx_df["customer_id"].to_numpy()
    tx_art = tx_df["article_id"].to_numpy()
    tx_t = tx_df["t_unix"].to_numpy()
    n_tx = len(tx_df)

    totals = {
        split: {
            hop: {
                "num_rows": 0,
                "num_gt": 0,
                "num_reachable": 0,
                "num_hits": 0,
                "sum_row_recall": 0.0,
                "sum_row_precision": 0.0,
            }
            for hop in odd_hops
        }
        for split in ["val", "test"]
    }
    row_records: list = []

    unique_seed_times = gt_df["seed_time_unix"].unique()  # gt_df가 이미 정렬되어 있음
    grouped = gt_df.groupby("seed_time_unix")

    for seed_t in tqdm(unique_seed_times, desc="Scoring"):
        # seed_t 이하 이벤트만 incremental 추가 (시간순 포인터 전진)
        while tx_ptr < n_tx and tx_t[tx_ptr] <= seed_t:
            cid = int(tx_cust[tx_ptr])
            aid = int(tx_art[tx_ptr])
            c2a[cid].append(aid)
            a2c[aid].append(cid)
            tx_ptr += 1

        rows_at_t = grouped.get_group(seed_t)

        for _, row in rows_at_t.iterrows():
            customer_id = int(row[src_col])
            gt_articles = row[dst_col]
            split = row["split"]

            # hop별 "신규" 도달 집합 계산 (문서1처럼 [-N:] 슬라이싱만 사용,
            # timestamp 비교 없음 — c2a/a2c가 이미 seed_t 이하로만 채워져 있음)
            current_customers = {customer_id}
            current_articles: set = set()
            seen_articles: set = set()
            articles_by_hop: dict = {}

            for hop in range(1, args.max_hops + 1):
                limit = NN[hop - 1]
                if hop % 2 == 1:
                    cand: set = set()
                    for c in current_customers:
                        cand.update(c2a.get(c, [])[-limit:])
                    new_articles = cand - seen_articles
                    current_articles = new_articles
                    seen_articles.update(current_articles)
                    articles_by_hop[hop] = set(current_articles)
                else:
                    next_customers: list = []
                    for aid in current_articles:
                        next_customers.extend(a2c.get(aid, [])[-limit:])
                    current_customers = set(next_customers)

                if not current_customers and not current_articles:
                    break

            cumulative_articles: set = set()
            row_result = {
                "split": split,
                "customer_id": customer_id,
                "seed_time_unix": seed_t,
                "num_groundtruth": len(gt_articles),
            }

            for hop in odd_hops:
                cumulative_articles.update(articles_by_hop.get(hop, set()))
                hits = len(cumulative_articles & gt_articles)
                recall = hits / len(gt_articles)
                precision = hits / len(cumulative_articles) if cumulative_articles else 0.0

                stats = totals[split][hop]
                stats["num_rows"] += 1
                stats["num_gt"] += len(gt_articles)
                stats["num_reachable"] += len(cumulative_articles)
                stats["num_hits"] += hits
                stats["sum_row_recall"] += recall
                stats["sum_row_precision"] += precision

                if args.save_rows:
                    row_result[f"hop_{hop}_reachable_articles"] = len(cumulative_articles)
                    row_result[f"hop_{hop}_hits"] = hits
                    row_result[f"hop_{hop}_locality_score"] = recall
                    row_result[f"hop_{hop}_precision"] = precision

            if args.save_rows:
                row_records.append(row_result)

    # ── 4. 결과 집계 및 저장 ─────────────────────────────────────────────
    args.out_dir.mkdir(parents=True, exist_ok=True)

    all_summary = {
        "dataset": args.dataset,
        "task": args.task,
        "num_neighbors": args.num_neighbors,
        "max_hops": args.max_hops,
        "hop_neighbor_budget": {str(h): NN[h - 1] for h in range(1, args.max_hops + 1)},
        "definition": {
            "locality_score": "groundtruth articles reachable from the IDGNN-style k-hop local subgraph divided by all groundtruth articles",
            "neighbor_groundtruth_ratio": "reachable articles that are groundtruth divided by all reachable articles",
            "hop_convention": "Customer-article historical purchases are treated as conceptual 1-hop local neighbors.",
        },
        "splits": {},
    }

    if args.save_rows and row_records:
        all_rows_df = pd.DataFrame(row_records)
    else:
        all_rows_df = pd.DataFrame()

    for split in ["val", "test"]:
        summary: dict = {}
        for hop, stats in totals[split].items():
            num_rows = stats["num_rows"]
            summary[f"hop_{hop}"] = {
                "locality_score_micro": (
                    stats["num_hits"] / stats["num_gt"] if stats["num_gt"] else float("nan")
                ),
                "locality_score_macro": (
                    stats["sum_row_recall"] / num_rows if num_rows else float("nan")
                ),
                "neighbor_groundtruth_ratio_micro": (
                    stats["num_hits"] / stats["num_reachable"]
                    if stats["num_reachable"]
                    else float("nan")
                ),
                "neighbor_groundtruth_ratio_macro": (
                    stats["sum_row_precision"] / num_rows if num_rows else float("nan")
                ),
                "num_rows": num_rows,
                "num_groundtruth_articles": stats["num_gt"],
                "num_reachable_articles": stats["num_reachable"],
                "num_groundtruth_reachable_articles": stats["num_hits"],
            }
        all_summary["splits"][split] = summary

        if args.save_rows and not all_rows_df.empty:
            split_df = all_rows_df[all_rows_df["split"] == split]
            split_df.to_csv(args.out_dir / f"{split}_locality_rows.csv", index=False)

    with open(args.out_dir / "summary.json", "w") as f:
        json.dump(all_summary, f, indent=2)

    print(json.dumps(all_summary, indent=2))


if __name__ == "__main__":
    main()