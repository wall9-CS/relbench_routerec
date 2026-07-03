"""
Batch rel-hm article-feature 1-hop locality analysis.

Runs the same static single-cutoff feature-sharing analysis for multiple
article table columns. The transactions snapshots are built once per split and
reused across all requested features.
"""

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from analyze_product_code_1hop import (
    _add_delta,
    _json_safe,
    _summarize,
    build_feature_candidate_lists,
    build_snapshot,
    safe_name,
    score_split,
)
from relbench.datasets import get_dataset
from relbench.tasks import get_task


DEFAULT_FEATURE_COLS = [
    "product_code",
    "product_type_no",
    "product_group_name",
    "graphical_appearance_no",
    "colour_group_code",
    "perceived_colour_value_id",
    "perceived_colour_master_id",
    "department_no",
    "index_code",
    "section_no",
    "garment_group_no",
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="rel-hm")
    parser.add_argument("--task", type=str, default="user-item-purchase")
    parser.add_argument("--num_neighbors", type=int, default=128)
    parser.add_argument("--feature_fanout", type=int, default=128)
    parser.add_argument(
        "--feature_col",
        action="append",
        default=None,
        help="Article feature column to analyze. Repeatable. Defaults to HM feature list.",
    )
    parser.add_argument(
        "--candidate_scope",
        choices=["historical", "all_articles"],
        default="historical",
        help=(
            "historical only adds same-feature articles that appeared in "
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
    parser.add_argument(
        "--skip_missing",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Skip missing feature columns instead of failing.",
    )
    parser.add_argument("--download", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--save_rows", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--out_dir",
        type=Path,
        default=Path("locality_score/rel-hm/feature_edge/article_feature_1hop_results"),
    )
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if args.num_neighbors < 1:
        raise ValueError("--num_neighbors must be positive.")
    if args.feature_fanout < 1:
        raise ValueError("--feature_fanout must be positive.")

    feature_cols = args.feature_col or DEFAULT_FEATURE_COLS
    rng = np.random.default_rng(args.seed)

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
    missing = [col for col in feature_cols if col not in article_df.columns]
    if missing and not args.skip_missing:
        raise ValueError(
            f"article is missing feature columns {missing}. "
            f"Available columns: {list(article_df.columns)}"
        )
    if missing:
        print(f"Skipping missing feature columns: {missing}")
        feature_cols = [col for col in feature_cols if col in article_df.columns]

    print(f"  transactions: {len(tx_df):,} rows")
    print(f"  requested feature columns: {feature_cols}")

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
                f"{n_unique}. This static-snapshot script assumes a single cutoff per split."
            )
        print(f"  {split} rows: {len(df):,}, seed_time={df['seed_time_unix'].iloc[0]}")

    need_historical = args.candidate_scope == "historical"
    snapshots = {}
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
        snapshots[split] = (c2a, article_last_seen, seen_articles)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    all_feature_summaries: dict[str, Any] = {
        "dataset": args.dataset,
        "task": args.task,
        "num_neighbors": args.num_neighbors,
        "max_hops": 1,
        "feature_fanout": args.feature_fanout,
        "candidate_scope": args.candidate_scope,
        "max_feature_group_size": args.max_feature_group_size or None,
        "features": {},
    }

    for feature_col in feature_cols:
        case_name = safe_name(feature_col)
        feature_out_dir = args.out_dir / case_name
        feature_out_dir.mkdir(parents=True, exist_ok=True)

        print(f"\nAnalyzing article.{feature_col}...")
        article_info = article_df[["article_id", feature_col]].dropna().copy()
        article_info["article_id"] = article_info["article_id"].astype(np.int64)

        article_to_feature: dict[int, Any] = dict(
            zip(
                article_info["article_id"].to_numpy(),
                article_info[feature_col].to_numpy(),
            )
        )
        feature_to_articles: dict[Any, set[int]] = {
            feature: set(group["article_id"].to_numpy().tolist())
            for feature, group in article_info.groupby(feature_col, sort=False)
        }

        if args.max_feature_group_size > 0:
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

        print(f"  articles with feature: {len(article_to_feature):,}")
        print(f"  unique feature values: {len(feature_to_articles):,}")

        totals: dict[str, dict[str, dict[str, float]]] = {}
        row_records: list[dict[str, Any]] = []

        for split in ["val", "test"]:
            c2a, article_last_seen, seen_articles = snapshots[split]
            print(f"  precomputing {feature_col} candidate lists for {split}...")
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
                gt_by_split[split],
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

        feature_summary = {
            "feature": f"article.{feature_col}",
            "case_name": case_name,
            "snapshot_mode": "static_single_cutoff_per_split",
            "splits": {},
        }

        for split in ["val", "test"]:
            split_summary = {
                case: _summarize(case_stats)
                for case, case_stats in totals[split].items()
            }
            _add_delta(split_summary, case_name)
            feature_summary["splits"][split] = split_summary

            if args.save_rows and row_records:
                split_df = pd.DataFrame(
                    row for row in row_records if row["split"] == split
                )
                split_df.to_csv(
                    feature_out_dir / f"{split}_{case_name}_1hop_rows.csv",
                    index=False,
                )

        with open(feature_out_dir / "summary.json", "w") as f:
            json.dump(feature_summary, f, indent=2, default=_json_safe)

        all_feature_summaries["features"][feature_col] = feature_summary

    with open(args.out_dir / "summary_all.json", "w") as f:
        json.dump(all_feature_summaries, f, indent=2, default=_json_safe)

    print(json.dumps(all_feature_summaries, indent=2, default=_json_safe))


if __name__ == "__main__":
    main()
