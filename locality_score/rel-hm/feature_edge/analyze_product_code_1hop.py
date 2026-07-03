"""
rel-hm product_code-sharing 1-hop locality analysis (static, single-cutoff).

Baseline locality follows locality_score_hm_fast.py with max_hops=1:
- A customer's recent historical transactions are treated as conceptual 1-hop
  reachable articles.
- This script assumes each split has exactly one seed_time, builds one
  transactions snapshot per split, and scores every row in that split against
  that snapshot.

The product_code case additionally treats articles sharing article.product_code
with the baseline reachable articles as analysis-only 1-hop reachable articles.
"""

import argparse
import heapq
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from tqdm import tqdm

from relbench.datasets import get_dataset
from relbench.tasks import get_task


def _json_safe(value: Any) -> Any:
    if isinstance(value, float) and pd.isna(value):
        return None
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    return value


def safe_name(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch == "_" else "_" for ch in value.lower())


def _empty_stats() -> dict[str, float]:
    return {
        "num_rows": 0,
        "num_gt": 0,
        "num_reachable": 0,
        "num_hits": 0,
        "sum_row_recall": 0.0,
        "sum_row_precision": 0.0,
    }


def _update_stats(
    stats: dict[str, float],
    gt_articles: set[int],
    reachable_articles: set[int],
) -> tuple[int, float, float]:
    hits = len(gt_articles & reachable_articles)
    recall = hits / len(gt_articles)
    precision = hits / len(reachable_articles) if reachable_articles else 0.0

    stats["num_rows"] += 1
    stats["num_gt"] += len(gt_articles)
    stats["num_reachable"] += len(reachable_articles)
    stats["num_hits"] += hits
    stats["sum_row_recall"] += recall
    stats["sum_row_precision"] += precision
    return hits, recall, precision


def _summarize(stats: dict[str, float]) -> dict[str, float]:
    num_rows = stats["num_rows"]
    return {
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


def _add_delta(summary: dict[str, dict[str, float]], case_name: str) -> None:
    base = summary["baseline"]
    product = summary[case_name]
    product["delta_vs_baseline"] = {
        "locality_score_micro": (
            product["locality_score_micro"] - base["locality_score_micro"]
        ),
        "locality_score_macro": (
            product["locality_score_macro"] - base["locality_score_macro"]
        ),
        "neighbor_groundtruth_ratio_micro": (
            product["neighbor_groundtruth_ratio_micro"]
            - base["neighbor_groundtruth_ratio_micro"]
        ),
        "neighbor_groundtruth_ratio_macro": (
            product["neighbor_groundtruth_ratio_macro"]
            - base["neighbor_groundtruth_ratio_macro"]
        ),
        "num_reachable_articles": (
            product["num_reachable_articles"] - base["num_reachable_articles"]
        ),
        "num_groundtruth_reachable_articles": (
            product["num_groundtruth_reachable_articles"]
            - base["num_groundtruth_reachable_articles"]
        ),
    }


def build_snapshot(
    tx_df: pd.DataFrame,
    cutoff_t: int,
    num_neighbors: int,
    need_historical: bool,
) -> tuple[dict[int, set[int]], dict[int, int], set[int]]:
    """Build transactions history up to and including cutoff_t."""
    scoped = tx_df[tx_df["t_unix"] <= cutoff_t]

    last_seen_df = scoped.drop_duplicates(subset="article_id", keep="last")
    article_last_seen = dict(
        zip(
            last_seen_df["article_id"].to_numpy(),
            last_seen_df["t_unix"].to_numpy(),
        )
    )

    c2a_series = scoped.groupby("customer_id", sort=False)["article_id"].apply(
        lambda s: set(s.to_numpy()[-num_neighbors:].tolist())
    )
    c2a: dict[int, set[int]] = c2a_series.to_dict()

    seen_articles: set[int] = set(article_last_seen) if need_historical else set()
    return c2a, article_last_seen, seen_articles


def build_feature_candidate_lists(
    feature_to_articles: dict[Any, set[int]],
    article_last_seen: dict[int, int],
    seen_articles: set[int],
    candidate_scope: str,
    feature_fanout: int,
    num_neighbors: int,
) -> dict[Any, list[int]]:
    """Precompute recent article candidates per feature value for one split."""
    per_feature_limit = feature_fanout + num_neighbors
    candidate_lists: dict[Any, list[int]] = {}

    for feature_value, article_ids in feature_to_articles.items():
        if candidate_scope == "historical":
            candidates = article_ids & seen_articles
        else:
            candidates = article_ids

        if not candidates:
            continue

        top_articles = heapq.nlargest(
            per_feature_limit,
            candidates,
            key=lambda article_id: (article_last_seen.get(article_id, -1), article_id),
        )
        candidate_lists[feature_value] = top_articles

    return candidate_lists


def score_split(
    group: pd.DataFrame,
    src_col: str,
    dst_col: str,
    c2a: dict[int, set[int]],
    article_last_seen: dict[int, int],
    article_to_feature: dict[int, Any],
    feature_candidate_lists: dict[Any, list[int]],
    feature_fanout: int,
    case_name: str,
    save_rows: bool,
    split_name: str,
) -> tuple[dict[str, dict[str, float]], list[dict[str, Any]]]:
    stats = {"baseline": _empty_stats(), case_name: _empty_stats()}
    row_records: list[dict[str, Any]] = []

    src_idx = group.columns.get_loc(src_col)
    dst_idx = group.columns.get_loc(dst_col)

    for row in tqdm(
        group.itertuples(index=False, name=None),
        total=len(group),
        desc=f"Scoring {split_name}",
    ):
        customer_id = int(row[src_idx])
        gt_articles = row[dst_idx]

        baseline_articles = c2a.get(customer_id, set())
        feature_values = {
            article_to_feature[article_id]
            for article_id in baseline_articles
            if article_id in article_to_feature
        }

        feature_candidates: set[int] = set()
        for feature_value in feature_values:
            feature_candidates.update(feature_candidate_lists.get(feature_value, ()))

        feature_candidates -= baseline_articles

        if len(feature_candidates) <= feature_fanout:
            added_articles = set(feature_candidates)
        else:
            top = heapq.nlargest(
                feature_fanout,
                feature_candidates,
                key=lambda article_id: (
                    article_last_seen.get(article_id, -1),
                    article_id,
                ),
            )
            added_articles = set(top)

        product_code_articles = baseline_articles | added_articles

        base_hits, base_recall, base_precision = _update_stats(
            stats["baseline"], gt_articles, baseline_articles
        )
        product_hits, product_recall, product_precision = _update_stats(
            stats[case_name], gt_articles, product_code_articles
        )

        if save_rows:
            row_records.append(
                {
                    "split": split_name,
                    "customer_id": customer_id,
                    "num_groundtruth": len(gt_articles),
                    "baseline_reachable_articles": len(baseline_articles),
                    f"{case_name}_reachable_articles": len(product_code_articles),
                    f"{case_name}_added_articles": len(added_articles),
                    "baseline_hits": base_hits,
                    f"{case_name}_hits": product_hits,
                    "hit_delta": product_hits - base_hits,
                    "baseline_locality_score": base_recall,
                    f"{case_name}_locality_score": product_recall,
                    "locality_score_delta": product_recall - base_recall,
                    "baseline_precision": base_precision,
                    f"{case_name}_precision": product_precision,
                    "precision_delta": product_precision - base_precision,
                }
            )

    return stats, row_records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="rel-hm")
    parser.add_argument("--task", type=str, default="user-item-purchase")
    parser.add_argument("--num_neighbors", type=int, default=128)
    parser.add_argument("--feature_fanout", type=int, default=128)
    parser.add_argument("--feature_col", type=str, default="product_code")
    parser.add_argument("--case_name", type=str, default=None)
    parser.add_argument(
        "--candidate_scope",
        choices=["historical", "all_articles"],
        default="historical",
        help=(
            "historical only adds same-product_code articles that appeared in "
            "transactions by seed_time; all_articles uses every article row."
        ),
    )
    parser.add_argument(
        "--max_feature_group_size",
        type=int,
        default=0,
        help=(
            "If > 0, cap each feature -> articles group to at most this many "
            "articles at build time (uniform random subsample). 0 disables "
            "capping."
        ),
    )
    parser.add_argument("--download", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--save_rows", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--out_dir",
        type=Path,
        default=Path("locality_score/rel-hm/feature_edge/product_code_1hop_results"),
    )
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if args.num_neighbors < 1:
        raise ValueError("--num_neighbors must be positive.")
    if args.feature_fanout < 1:
        raise ValueError("--feature_fanout must be positive.")

    rng = np.random.default_rng(args.seed)
    case_name = safe_name(args.case_name or args.feature_col)

    print("Loading data...")
    dataset = get_dataset(args.dataset, download=args.download)
    task = get_task(args.dataset, args.task, download=args.download)
    db = dataset.get_db()

    src_col = task.src_entity_col
    dst_col = task.dst_entity_col

    tx_df = (
        db.table_dict["transactions"]
        .df[["t_dat", "customer_id", "article_id"]]
        .dropna()
        .copy()
    )
    print("converting t_dat to unix timestamp...")
    tx_df["t_unix"] = tx_df["t_dat"].astype(np.int64) // 10**9
    tx_df = tx_df.sort_values("t_unix").reset_index(drop=True)

    article_df = db.table_dict["article"].df
    if args.feature_col not in article_df.columns:
        raise ValueError(
            f"article is missing feature column {args.feature_col!r}. "
            f"Available columns: {list(article_df.columns)}"
        )

    print(f"Loading article {args.feature_col} mapping (vectorized)...")
    article_info = article_df[["article_id", args.feature_col]].dropna().copy()
    article_info["article_id"] = article_info["article_id"].astype(np.int64)

    article_to_feature: dict[int, Any] = dict(
        zip(article_info["article_id"].to_numpy(), article_info[args.feature_col].to_numpy())
    )

    print(f"Building {args.feature_col} -> articles groups (vectorized groupby)...")
    feature_to_articles: dict[Any, set[int]] = {
        feature: set(group["article_id"].to_numpy().tolist())
        for feature, group in article_info.groupby(args.feature_col, sort=False)
    }

    if args.max_feature_group_size > 0:
        print(
            f"Capping {args.feature_col} groups to at most "
            f"{args.max_feature_group_size} articles..."
        )
        capped = 0
        for feature, article_set in feature_to_articles.items():
            if len(article_set) > args.max_feature_group_size:
                arr = np.fromiter(article_set, dtype=np.int64)
                chosen = rng.choice(
                    arr, size=args.max_feature_group_size, replace=False
                )
                feature_to_articles[feature] = set(chosen.tolist())
                capped += 1
        print(f"  capped {capped:,} / {len(feature_to_articles):,} feature groups")

    print(f"  transactions: {len(tx_df):,} rows")
    print(f"  articles with {args.feature_col}: {len(article_to_feature):,}")
    print(f"  unique {args.feature_col}: {len(feature_to_articles):,}")

    print("Loading ground truth data...")

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

    gt_by_split = {split: load_gt(split) for split in ["val", "test"]}

    for split, df in gt_by_split.items():
        n_unique = df["seed_time_unix"].nunique()
        if n_unique != 1:
            raise ValueError(
                f"Expected exactly one seed_time for split={split!r}, found "
                f"{n_unique}. This static-snapshot script assumes a single "
                f"cutoff per split; use locality_score_hm_fast.py or an "
                f"incremental feature version instead."
            )
        print(f"  {split} rows: {len(df):,}, seed_time={df['seed_time_unix'].iloc[0]}")

    need_historical = args.candidate_scope == "historical"

    totals: dict[str, dict[str, dict[str, float]]] = {}
    row_records: list[dict[str, Any]] = []

    for split in ["val", "test"]:
        df = gt_by_split[split]
        cutoff_t = int(df["seed_time_unix"].iloc[0])
        print(f"Building static snapshot for split={split} (cutoff={cutoff_t})...")
        c2a, article_last_seen, seen_articles = build_snapshot(
            tx_df, cutoff_t, args.num_neighbors, need_historical
        )
        print(
            f"  customers with history: {len(c2a):,}, "
            f"articles seen: {len(article_last_seen):,}"
        )
        print(f"  precomputing {args.feature_col} candidate lists...")
        feature_candidate_lists = build_feature_candidate_lists(
            feature_to_articles,
            article_last_seen,
            seen_articles,
            args.candidate_scope,
            args.feature_fanout,
            args.num_neighbors,
        )
        print(f"  feature groups with candidates: {len(feature_candidate_lists):,}")

        stats, records = score_split(
            df,
            src_col,
            dst_col,
            c2a,
            article_last_seen,
            article_to_feature,
            feature_candidate_lists,
            args.feature_fanout,
            case_name,
            args.save_rows,
            split,
        )
        totals[split] = stats
        row_records.extend(records)

    args.out_dir.mkdir(parents=True, exist_ok=True)

    all_summary = {
        "dataset": args.dataset,
        "task": args.task,
        "num_neighbors": args.num_neighbors,
        "max_hops": 1,
        "feature": f"article.{args.feature_col}",
        "feature_fanout": args.feature_fanout,
        "candidate_scope": args.candidate_scope,
        "max_feature_group_size": args.max_feature_group_size or None,
        "snapshot_mode": "static_single_cutoff_per_split",
        "definition": {
            "baseline": (
                "Customer's recent historical transactions (as of the split's "
                "single seed_time) are treated as conceptual 1-hop reachable "
                "articles."
            ),
            case_name: (
                f"Articles sharing {args.feature_col} with baseline reachable "
                "articles are also treated as analysis-only 1-hop reachable "
                "articles."
            ),
            "locality_score": (
                "groundtruth articles reachable from the 1-hop candidate set "
                "divided by all groundtruth articles"
            ),
        },
        "splits": {},
    }

    for split in ["val", "test"]:
        split_summary = {
            case_name: _summarize(case_stats)
            for case_name, case_stats in totals[split].items()
        }
        _add_delta(split_summary, case_name)
        all_summary["splits"][split] = split_summary

        if args.save_rows and row_records:
            split_df = pd.DataFrame(
                row for row in row_records if row["split"] == split
            )
            split_df.to_csv(
                args.out_dir / f"{split}_{case_name}_1hop_rows.csv",
                index=False,
            )

    with open(args.out_dir / "summary.json", "w") as f:
        json.dump(all_summary, f, indent=2, default=_json_safe)

    print(json.dumps(all_summary, indent=2, default=_json_safe))


if __name__ == "__main__":
    main()
