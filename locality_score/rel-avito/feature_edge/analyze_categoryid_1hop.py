"""
rel-avito CategoryID-sharing 1-hop locality analysis (static, single-cutoff).

This is the CategoryID counterpart of analyze_locationid_1hop.py. It assumes
each split has exactly one seed_time, builds one VisitStream snapshot per split,
and treats ads sharing AdsInfo.CategoryID with a user's baseline 1-hop ads as
analysis-only 1-hop reachable ads.
"""

import argparse
import heapq
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from tqdm import tqdm

from analyze_locationid_1hop import (
    _empty_stats,
    _json_safe,
    _summarize,
    _update_stats,
    build_location_candidate_lists,
    build_snapshot,
)
from relbench.datasets import get_dataset
from relbench.tasks import get_task


def _add_delta(summary: dict[str, dict[str, float]]) -> None:
    base = summary["baseline"]
    cat = summary["categoryid"]
    cat["delta_vs_baseline"] = {
        "locality_score_micro": (
            cat["locality_score_micro"] - base["locality_score_micro"]
        ),
        "locality_score_macro": (
            cat["locality_score_macro"] - base["locality_score_macro"]
        ),
        "neighbor_groundtruth_ratio_micro": (
            cat["neighbor_groundtruth_ratio_micro"]
            - base["neighbor_groundtruth_ratio_micro"]
        ),
        "neighbor_groundtruth_ratio_macro": (
            cat["neighbor_groundtruth_ratio_macro"]
            - base["neighbor_groundtruth_ratio_macro"]
        ),
        "num_reachable_ads": cat["num_reachable_ads"] - base["num_reachable_ads"],
        "num_groundtruth_reachable_ads": (
            cat["num_groundtruth_reachable_ads"]
            - base["num_groundtruth_reachable_ads"]
        ),
    }


def score_split(
    group: pd.DataFrame,
    src_col: str,
    dst_col: str,
    u2a: dict[int, set[int]],
    ad_last_seen: dict[int, int],
    ad_to_category: dict[int, Any],
    category_candidate_lists: dict[Any, list[int]],
    feature_fanout: int,
    save_rows: bool,
    split_name: str,
) -> tuple[dict[str, dict[str, float]], list[dict[str, Any]]]:
    stats = {"baseline": _empty_stats(), "categoryid": _empty_stats()}
    row_records: list[dict[str, Any]] = []

    src_idx = group.columns.get_loc(src_col)
    dst_idx = group.columns.get_loc(dst_col)

    for row in tqdm(
        group.itertuples(index=False, name=None),
        total=len(group),
        desc=f"Scoring {split_name}",
    ):
        user_id = int(row[src_idx])
        gt_ads = row[dst_idx]

        baseline_ads = u2a.get(user_id, set())
        category_ids = {
            ad_to_category[ad_id]
            for ad_id in baseline_ads
            if ad_id in ad_to_category
        }

        category_candidates: set[int] = set()
        for category_id in category_ids:
            category_candidates.update(category_candidate_lists.get(category_id, ()))

        category_candidates -= baseline_ads

        if len(category_candidates) <= feature_fanout:
            added_ads = set(category_candidates)
        else:
            top = heapq.nlargest(
                feature_fanout,
                category_candidates,
                key=lambda ad_id: (ad_last_seen.get(ad_id, -1), ad_id),
            )
            added_ads = set(top)

        categoryid_ads = baseline_ads | added_ads

        base_hits, base_recall, base_precision = _update_stats(
            stats["baseline"], gt_ads, baseline_ads
        )
        cat_hits, cat_recall, cat_precision = _update_stats(
            stats["categoryid"], gt_ads, categoryid_ads
        )

        if save_rows:
            row_records.append(
                {
                    "split": split_name,
                    "UserID": user_id,
                    "num_groundtruth": len(gt_ads),
                    "baseline_reachable_ads": len(baseline_ads),
                    "categoryid_reachable_ads": len(categoryid_ads),
                    "categoryid_added_ads": len(added_ads),
                    "baseline_hits": base_hits,
                    "categoryid_hits": cat_hits,
                    "hit_delta": cat_hits - base_hits,
                    "baseline_locality_score": base_recall,
                    "categoryid_locality_score": cat_recall,
                    "locality_score_delta": cat_recall - base_recall,
                    "baseline_precision": base_precision,
                    "categoryid_precision": cat_precision,
                    "precision_delta": cat_precision - base_precision,
                }
            )

    return stats, row_records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="rel-avito")
    parser.add_argument("--task", type=str, default="user-ad-visit")
    parser.add_argument("--num_neighbors", type=int, default=128)
    parser.add_argument("--feature_fanout", type=int, default=128)
    parser.add_argument(
        "--candidate_scope",
        choices=["historical", "all_adsinfo"],
        default="historical",
        help=(
            "historical only adds same-CategoryID ads that appeared in "
            "VisitStream by seed_time; all_adsinfo uses every AdsInfo ad."
        ),
    )
    parser.add_argument(
        "--max_category_group_size",
        type=int,
        default=0,
        help=(
            "If > 0, cap each CategoryID -> ads group to at most this many "
            "ads at build time (uniform random subsample). 0 disables "
            "capping."
        ),
    )
    parser.add_argument("--download", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--save_rows", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--out_dir",
        type=Path,
        default=Path("locality_score/rel-avito/feature_edge/categoryid_1hop_results"),
    )
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if args.num_neighbors < 1:
        raise ValueError("--num_neighbors must be positive.")
    if args.feature_fanout < 1:
        raise ValueError("--feature_fanout must be positive.")

    rng = np.random.default_rng(args.seed)

    print("Loading data...")
    dataset = get_dataset(args.dataset, download=args.download)
    task = get_task(args.dataset, args.task, download=args.download)
    db = dataset.get_db()

    src_col = task.src_entity_col
    dst_col = task.dst_entity_col

    visit_df = (
        db.table_dict["VisitStream"]
        .df[["ViewDate", "UserID", "AdID"]]
        .dropna()
        .copy()
    )
    print("converting ViewDate to unix timestamp...")
    visit_df["t_unix"] = visit_df["ViewDate"].astype(np.int64) // 10**9
    visit_df = visit_df.sort_values("t_unix").reset_index(drop=True)

    print("Loading AdsInfo CategoryID mapping (vectorized)...")
    ads_info = db.table_dict["AdsInfo"].df[["AdID", "CategoryID"]].dropna().copy()
    ads_info["AdID"] = ads_info["AdID"].astype(np.int64)

    ad_to_category: dict[int, Any] = dict(
        zip(ads_info["AdID"].to_numpy(), ads_info["CategoryID"].to_numpy())
    )

    print("Building CategoryID -> ads groups (vectorized groupby)...")
    category_to_ads: dict[Any, set[int]] = {
        category: set(group["AdID"].to_numpy().tolist())
        for category, group in ads_info.groupby("CategoryID", sort=False)
    }

    if args.max_category_group_size > 0:
        print(
            f"Capping CategoryID groups to at most "
            f"{args.max_category_group_size} ads..."
        )
        capped = 0
        for category, ad_set in category_to_ads.items():
            if len(ad_set) > args.max_category_group_size:
                arr = np.fromiter(ad_set, dtype=np.int64)
                chosen = rng.choice(
                    arr, size=args.max_category_group_size, replace=False
                )
                category_to_ads[category] = set(chosen.tolist())
                capped += 1
        print(f"  capped {capped:,} / {len(category_to_ads):,} CategoryID groups")

    print(f"  visits: {len(visit_df):,} rows")
    print(f"  ads with CategoryID: {len(ad_to_category):,}")
    print(f"  unique CategoryID: {len(category_to_ads):,}")

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
                f"cutoff per split; use the incremental version instead."
            )
        print(f"  {split} rows: {len(df):,}, seed_time={df['seed_time_unix'].iloc[0]}")

    need_historical = args.candidate_scope == "historical"

    totals: dict[str, dict[str, dict[str, float]]] = {}
    row_records: list[dict[str, Any]] = []

    for split in ["val", "test"]:
        df = gt_by_split[split]
        cutoff_t = int(df["seed_time_unix"].iloc[0])
        print(f"Building static snapshot for split={split} (cutoff={cutoff_t})...")
        u2a, ad_last_seen, seen_ads = build_snapshot(
            visit_df, cutoff_t, args.num_neighbors, need_historical
        )
        print(f"  users with history: {len(u2a):,}, ads seen: {len(ad_last_seen):,}")
        print("  precomputing CategoryID candidate lists...")
        category_candidate_lists = build_location_candidate_lists(
            category_to_ads,
            ad_last_seen,
            seen_ads,
            args.candidate_scope,
            args.feature_fanout,
            args.num_neighbors,
        )
        print(f"  CategoryID groups with candidates: {len(category_candidate_lists):,}")

        stats, records = score_split(
            df,
            src_col,
            dst_col,
            u2a,
            ad_last_seen,
            ad_to_category,
            category_candidate_lists,
            args.feature_fanout,
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
        "feature": "AdsInfo.CategoryID",
        "feature_fanout": args.feature_fanout,
        "candidate_scope": args.candidate_scope,
        "max_category_group_size": args.max_category_group_size or None,
        "snapshot_mode": "static_single_cutoff_per_split",
        "definition": {
            "baseline": (
                "User's recent historical VisitStream ads (as of the split's "
                "single seed_time) are treated as conceptual 1-hop reachable "
                "ads."
            ),
            "categoryid": (
                "Ads sharing CategoryID with baseline reachable ads are also "
                "treated as analysis-only 1-hop reachable ads."
            ),
            "locality_score": (
                "groundtruth ads reachable from the 1-hop candidate set divided "
                "by all groundtruth ads"
            ),
        },
        "splits": {},
    }

    for split in ["val", "test"]:
        split_summary = {
            case_name: _summarize(case_stats)
            for case_name, case_stats in totals[split].items()
        }
        _add_delta(split_summary)
        all_summary["splits"][split] = split_summary

        if args.save_rows and row_records:
            split_df = pd.DataFrame(
                row for row in row_records if row["split"] == split
            )
            split_df.to_csv(args.out_dir / f"{split}_categoryid_1hop_rows.csv", index=False)

    with open(args.out_dir / "summary.json", "w") as f:
        json.dump(all_summary, f, indent=2, default=_json_safe)

    print(json.dumps(all_summary, indent=2, default=_json_safe))


if __name__ == "__main__":
    main()
