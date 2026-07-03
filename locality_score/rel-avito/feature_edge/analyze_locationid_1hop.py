"""
rel-avito LocationID-sharing 1-hop locality analysis (static, single-cutoff).

This assumes (verified: gt_df.groupby("split")["seed_time_unix"].nunique()
== 1 for both val and test) that each split has exactly ONE seed_time. That
means there is no need for a per-seed_time incremental scan: we just build
two static snapshots of VisitStream history (one truncated at the val
cutoff, one truncated at the test cutoff) and score every row in that split
against its single snapshot.

This removes the entire seed_time outer loop, the pointer-advancing scan,
and the per-seed_time historical_ads recomputation from the original
script -- none of that machinery is needed when there's only one cutoff per
split. The remaining cost is the per-user candidate-set computation, which
is unavoidable (each user has a different baseline neighborhood) but is now
vectorized wherever possible and only touches each split's rows once.
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
    gt_ads: set[int],
    reachable_ads: set[int],
) -> tuple[int, float, float]:
    hits = len(gt_ads & reachable_ads)
    recall = hits / len(gt_ads)
    precision = hits / len(reachable_ads) if reachable_ads else 0.0

    stats["num_rows"] += 1
    stats["num_gt"] += len(gt_ads)
    stats["num_reachable"] += len(reachable_ads)
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
        "num_groundtruth_ads": stats["num_gt"],
        "num_reachable_ads": stats["num_reachable"],
        "num_groundtruth_reachable_ads": stats["num_hits"],
    }


def _add_delta(summary: dict[str, dict[str, float]]) -> None:
    base = summary["baseline"]
    loc = summary["locationid"]
    loc["delta_vs_baseline"] = {
        "locality_score_micro": (
            loc["locality_score_micro"] - base["locality_score_micro"]
        ),
        "locality_score_macro": (
            loc["locality_score_macro"] - base["locality_score_macro"]
        ),
        "neighbor_groundtruth_ratio_micro": (
            loc["neighbor_groundtruth_ratio_micro"]
            - base["neighbor_groundtruth_ratio_micro"]
        ),
        "neighbor_groundtruth_ratio_macro": (
            loc["neighbor_groundtruth_ratio_macro"]
            - base["neighbor_groundtruth_ratio_macro"]
        ),
        "num_reachable_ads": loc["num_reachable_ads"] - base["num_reachable_ads"],
        "num_groundtruth_reachable_ads": (
            loc["num_groundtruth_reachable_ads"]
            - base["num_groundtruth_reachable_ads"]
        ),
    }


def build_snapshot(
    visit_df: pd.DataFrame,
    cutoff_t: int,
    num_neighbors: int,
    need_historical: bool,
) -> tuple[dict[int, set[int]], dict[int, int], set[int]]:
    """Build a static snapshot of VisitStream history up to (and including)
    cutoff_t.

    Returns:
        u2a: user_id -> set of that user's most-recent `num_neighbors` ads
             (as of cutoff_t)
        ad_last_seen: ad_id -> most recent t_unix at or before cutoff_t
        seen_ads: set of all ad_ids seen at or before cutoff_t (empty set
             if need_historical is False, to skip the work)
    """
    scoped = visit_df[visit_df["t_unix"] <= cutoff_t]

    # ad_last_seen: last occurrence per AdID within scope.
    # scoped is already sorted by t_unix (visit_df is pre-sorted), so
    # drop_duplicates(keep="last") gives the most recent visit per ad.
    last_seen_df = scoped.drop_duplicates(subset="AdID", keep="last")
    ad_last_seen = dict(
        zip(
            last_seen_df["AdID"].to_numpy(),
            last_seen_df["t_unix"].to_numpy(),
        )
    )

    # u2a: last `num_neighbors` ads per user, within scope.
    # groupby + tail-like slicing via list aggregation; scoped is already
    # time-sorted so we just need the last num_neighbors per group.
    u2a_series = scoped.groupby("UserID", sort=False)["AdID"].apply(
        lambda s: set(s.to_numpy()[-num_neighbors:].tolist())
    )
    u2a: dict[int, set[int]] = u2a_series.to_dict()

    seen_ads: set[int] = set(ad_last_seen) if need_historical else set()

    return u2a, ad_last_seen, seen_ads


def build_location_candidate_lists(
    location_to_ads: dict[Any, set[int]],
    ad_last_seen: dict[int, int],
    seen_ads: set[int],
    candidate_scope: str,
    feature_fanout: int,
    num_neighbors: int,
) -> dict[Any, list[int]]:
    """Precompute per-LocationID candidates for one split snapshot.

    Only the first feature_fanout + num_neighbors ads per LocationID are needed:
    a row can exclude at most num_neighbors baseline ads, so any final top-k
    candidate must be inside that prefix for its LocationID.
    """
    per_location_limit = feature_fanout + num_neighbors
    candidate_lists: dict[Any, list[int]] = {}

    for location_id, ad_ids in location_to_ads.items():
        if candidate_scope == "historical":
            candidates = ad_ids & seen_ads
        else:
            candidates = ad_ids

        if not candidates:
            continue

        top_ads = heapq.nlargest(
            per_location_limit,
            candidates,
            key=lambda ad_id: (ad_last_seen.get(ad_id, -1), ad_id),
        )
        candidate_lists[location_id] = top_ads

    return candidate_lists


def score_split(
    group: pd.DataFrame,
    src_col: str,
    dst_col: str,
    u2a: dict[int, set[int]],
    ad_last_seen: dict[int, int],
    ad_to_location: dict[int, Any],
    location_candidate_lists: dict[Any, list[int]],
    feature_fanout: int,
    save_rows: bool,
    split_name: str,
) -> tuple[dict[str, dict[str, float]], list[dict[str, Any]]]:
    stats = {"baseline": _empty_stats(), "locationid": _empty_stats()}
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
        location_ids = {
            ad_to_location[ad_id]
            for ad_id in baseline_ads
            if ad_id in ad_to_location
        }

        location_candidates: set[int] = set()
        for location_id in location_ids:
            location_candidates.update(location_candidate_lists.get(location_id, ()))

        location_candidates -= baseline_ads

        if len(location_candidates) <= feature_fanout:
            added_ads = set(location_candidates)
        else:
            top = heapq.nlargest(
                feature_fanout,
                location_candidates,
                key=lambda ad_id: (ad_last_seen.get(ad_id, -1), ad_id),
            )
            added_ads = set(top)

        locationid_ads = baseline_ads | added_ads

        base_hits, base_recall, base_precision = _update_stats(
            stats["baseline"], gt_ads, baseline_ads
        )
        loc_hits, loc_recall, loc_precision = _update_stats(
            stats["locationid"], gt_ads, locationid_ads
        )

        if save_rows:
            row_records.append(
                {
                    "split": split_name,
                    "UserID": user_id,
                    "num_groundtruth": len(gt_ads),
                    "baseline_reachable_ads": len(baseline_ads),
                    "locationid_reachable_ads": len(locationid_ads),
                    "locationid_added_ads": len(added_ads),
                    "baseline_hits": base_hits,
                    "locationid_hits": loc_hits,
                    "hit_delta": loc_hits - base_hits,
                    "baseline_locality_score": base_recall,
                    "locationid_locality_score": loc_recall,
                    "locality_score_delta": loc_recall - base_recall,
                    "baseline_precision": base_precision,
                    "locationid_precision": loc_precision,
                    "precision_delta": loc_precision - base_precision,
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
            "historical only adds same-LocationID ads that appeared in "
            "VisitStream by seed_time; all_adsinfo uses every AdsInfo ad."
        ),
    )
    parser.add_argument(
        "--max_location_group_size",
        type=int,
        default=0,
        help=(
            "If > 0, cap each LocationID -> ads group to at most this many "
            "ads at build time (uniform random subsample). 0 disables "
            "capping."
        ),
    )
    parser.add_argument("--download", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--save_rows", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--out_dir",
        type=Path,
        default=Path("locality_score/rel-avito/feature_edge/locationid_1hop_results"),
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

    print("Loading AdsInfo LocationID mapping (vectorized)...")
    ads_info = db.table_dict["AdsInfo"].df[["AdID", "LocationID"]].dropna().copy()
    ads_info["AdID"] = ads_info["AdID"].astype(np.int64)

    ad_to_location: dict[int, Any] = dict(
        zip(ads_info["AdID"].to_numpy(), ads_info["LocationID"].to_numpy())
    )

    print("Building LocationID -> ads groups (vectorized groupby)...")
    location_to_ads: dict[Any, set[int]] = {
        loc: set(group["AdID"].to_numpy().tolist())
        for loc, group in ads_info.groupby("LocationID", sort=False)
    }

    if args.max_location_group_size > 0:
        print(
            f"Capping LocationID groups to at most "
            f"{args.max_location_group_size} ads..."
        )
        capped = 0
        for loc, ad_set in location_to_ads.items():
            if len(ad_set) > args.max_location_group_size:
                arr = np.fromiter(ad_set, dtype=np.int64)
                chosen = rng.choice(
                    arr, size=args.max_location_group_size, replace=False
                )
                location_to_ads[loc] = set(chosen.tolist())
                capped += 1
        print(f"  capped {capped:,} / {len(location_to_ads):,} LocationID groups")

    print(f"  visits: {len(visit_df):,} rows")
    print(f"  ads with LocationID: {len(ad_to_location):,}")
    print(f"  unique LocationID: {len(location_to_ads):,}")

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

    # Sanity check: this whole static-snapshot approach is only valid if
    # each split has exactly one seed_time.
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
        print("  precomputing LocationID candidate lists...")
        location_candidate_lists = build_location_candidate_lists(
            location_to_ads,
            ad_last_seen,
            seen_ads,
            args.candidate_scope,
            args.feature_fanout,
            args.num_neighbors,
        )
        print(f"  LocationID groups with candidates: {len(location_candidate_lists):,}")

        stats, records = score_split(
            df,
            src_col,
            dst_col,
            u2a,
            ad_last_seen,
            ad_to_location,
            location_candidate_lists,
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
        "feature": "AdsInfo.LocationID",
        "feature_fanout": args.feature_fanout,
        "candidate_scope": args.candidate_scope,
        "max_location_group_size": args.max_location_group_size or None,
        "snapshot_mode": "static_single_cutoff_per_split",
        "definition": {
            "baseline": (
                "User's recent historical VisitStream ads (as of the split's "
                "single seed_time) are treated as conceptual 1-hop reachable "
                "ads."
            ),
            "locationid": (
                "Ads sharing LocationID with baseline reachable ads are also "
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
            split_df.to_csv(args.out_dir / f"{split}_locationid_1hop_rows.csv", index=False)

    with open(args.out_dir / "summary.json", "w") as f:
        json.dump(all_summary, f, indent=2, default=_json_safe)

    print(json.dumps(all_summary, indent=2, default=_json_safe))


if __name__ == "__main__":
    main()
