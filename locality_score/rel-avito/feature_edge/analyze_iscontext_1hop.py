"""
rel-avito IsContext-sharing 1-hop locality analysis (static, single-cutoff).

This is the IsContext counterpart of analyze_locationid_1hop.py. It assumes
each split has exactly one seed_time, builds one VisitStream snapshot per split,
and treats ads sharing AdsInfo.IsContext with a user's baseline 1-hop ads as
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
    ctx = summary["iscontext"]
    ctx["delta_vs_baseline"] = {
        "locality_score_micro": (
            ctx["locality_score_micro"] - base["locality_score_micro"]
        ),
        "locality_score_macro": (
            ctx["locality_score_macro"] - base["locality_score_macro"]
        ),
        "neighbor_groundtruth_ratio_micro": (
            ctx["neighbor_groundtruth_ratio_micro"]
            - base["neighbor_groundtruth_ratio_micro"]
        ),
        "neighbor_groundtruth_ratio_macro": (
            ctx["neighbor_groundtruth_ratio_macro"]
            - base["neighbor_groundtruth_ratio_macro"]
        ),
        "num_reachable_ads": ctx["num_reachable_ads"] - base["num_reachable_ads"],
        "num_groundtruth_reachable_ads": (
            ctx["num_groundtruth_reachable_ads"]
            - base["num_groundtruth_reachable_ads"]
        ),
    }


def score_split(
    group: pd.DataFrame,
    src_col: str,
    dst_col: str,
    u2a: dict[int, set[int]],
    ad_last_seen: dict[int, int],
    ad_to_context: dict[int, Any],
    context_candidate_lists: dict[Any, list[int]],
    feature_fanout: int,
    save_rows: bool,
    split_name: str,
) -> tuple[dict[str, dict[str, float]], list[dict[str, Any]]]:
    stats = {"baseline": _empty_stats(), "iscontext": _empty_stats()}
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
        context_values = {
            ad_to_context[ad_id]
            for ad_id in baseline_ads
            if ad_id in ad_to_context
        }

        context_candidates: set[int] = set()
        for context_value in context_values:
            context_candidates.update(context_candidate_lists.get(context_value, ()))

        context_candidates -= baseline_ads

        if len(context_candidates) <= feature_fanout:
            added_ads = set(context_candidates)
        else:
            top = heapq.nlargest(
                feature_fanout,
                context_candidates,
                key=lambda ad_id: (ad_last_seen.get(ad_id, -1), ad_id),
            )
            added_ads = set(top)

        iscontext_ads = baseline_ads | added_ads

        base_hits, base_recall, base_precision = _update_stats(
            stats["baseline"], gt_ads, baseline_ads
        )
        ctx_hits, ctx_recall, ctx_precision = _update_stats(
            stats["iscontext"], gt_ads, iscontext_ads
        )

        if save_rows:
            row_records.append(
                {
                    "split": split_name,
                    "UserID": user_id,
                    "num_groundtruth": len(gt_ads),
                    "baseline_reachable_ads": len(baseline_ads),
                    "iscontext_reachable_ads": len(iscontext_ads),
                    "iscontext_added_ads": len(added_ads),
                    "baseline_hits": base_hits,
                    "iscontext_hits": ctx_hits,
                    "hit_delta": ctx_hits - base_hits,
                    "baseline_locality_score": base_recall,
                    "iscontext_locality_score": ctx_recall,
                    "locality_score_delta": ctx_recall - base_recall,
                    "baseline_precision": base_precision,
                    "iscontext_precision": ctx_precision,
                    "precision_delta": ctx_precision - base_precision,
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
        "--feature_col",
        type=str,
        default="IsContext",
        help="AdsInfo column to use for this feature. Use iscontext if needed.",
    )
    parser.add_argument(
        "--candidate_scope",
        choices=["historical", "all_adsinfo"],
        default="historical",
        help=(
            "historical only adds same-IsContext ads that appeared in "
            "VisitStream by seed_time; all_adsinfo uses every AdsInfo ad."
        ),
    )
    parser.add_argument(
        "--max_iscontext_group_size",
        type=int,
        default=0,
        help=(
            "If > 0, cap each IsContext -> ads group to at most this many "
            "ads at build time (uniform random subsample). 0 disables "
            "capping."
        ),
    )
    parser.add_argument("--download", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--save_rows", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--out_dir",
        type=Path,
        default=Path("locality_score/rel-avito/feature_edge/iscontext_1hop_results"),
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

    ads_df = db.table_dict["AdsInfo"].df
    if args.feature_col not in ads_df.columns:
        raise ValueError(
            f"AdsInfo is missing feature column {args.feature_col!r}. "
            f"Available columns: {list(ads_df.columns)}"
        )

    print(f"Loading AdsInfo {args.feature_col} mapping (vectorized)...")
    ads_info = ads_df[["AdID", args.feature_col]].dropna().copy()
    ads_info["AdID"] = ads_info["AdID"].astype(np.int64)

    ad_to_context: dict[int, Any] = dict(
        zip(ads_info["AdID"].to_numpy(), ads_info[args.feature_col].to_numpy())
    )

    print(f"Building {args.feature_col} -> ads groups (vectorized groupby)...")
    context_to_ads: dict[Any, set[int]] = {
        context: set(group["AdID"].to_numpy().tolist())
        for context, group in ads_info.groupby(args.feature_col, sort=False)
    }

    if args.max_iscontext_group_size > 0:
        print(
            f"Capping {args.feature_col} groups to at most "
            f"{args.max_iscontext_group_size} ads..."
        )
        capped = 0
        for context, ad_set in context_to_ads.items():
            if len(ad_set) > args.max_iscontext_group_size:
                arr = np.fromiter(ad_set, dtype=np.int64)
                chosen = rng.choice(
                    arr, size=args.max_iscontext_group_size, replace=False
                )
                context_to_ads[context] = set(chosen.tolist())
                capped += 1
        print(f"  capped {capped:,} / {len(context_to_ads):,} {args.feature_col} groups")

    print(f"  visits: {len(visit_df):,} rows")
    print(f"  ads with {args.feature_col}: {len(ad_to_context):,}")
    print(f"  unique {args.feature_col}: {len(context_to_ads):,}")

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
        print(f"  precomputing {args.feature_col} candidate lists...")
        context_candidate_lists = build_location_candidate_lists(
            context_to_ads,
            ad_last_seen,
            seen_ads,
            args.candidate_scope,
            args.feature_fanout,
            args.num_neighbors,
        )
        print(f"  {args.feature_col} groups with candidates: {len(context_candidate_lists):,}")

        stats, records = score_split(
            df,
            src_col,
            dst_col,
            u2a,
            ad_last_seen,
            ad_to_context,
            context_candidate_lists,
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
        "feature": f"AdsInfo.{args.feature_col}",
        "feature_fanout": args.feature_fanout,
        "candidate_scope": args.candidate_scope,
        "max_iscontext_group_size": args.max_iscontext_group_size or None,
        "snapshot_mode": "static_single_cutoff_per_split",
        "definition": {
            "baseline": (
                "User's recent historical VisitStream ads (as of the split's "
                "single seed_time) are treated as conceptual 1-hop reachable "
                "ads."
            ),
            "iscontext": (
                f"Ads sharing {args.feature_col} with baseline reachable ads "
                "are also treated as analysis-only 1-hop reachable ads."
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
            split_df.to_csv(args.out_dir / f"{split}_iscontext_1hop_rows.csv", index=False)

    with open(args.out_dir / "summary.json", "w") as f:
        json.dump(all_summary, f, indent=2, default=_json_safe)

    print(json.dumps(all_summary, indent=2, default=_json_safe))


if __name__ == "__main__":
    main()
