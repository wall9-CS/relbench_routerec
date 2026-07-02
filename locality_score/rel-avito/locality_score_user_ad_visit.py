import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import pandas as pd
from tqdm import tqdm

from relbench.datasets import get_dataset
from relbench.tasks import get_task


def _limit_recent(
    entries: list[tuple[pd.Timestamp, int]],
    timestamp: pd.Timestamp,
    limit: int,
) -> list[int]:
    """Return recent neighbor ids observed no later than timestamp."""
    if limit <= 0:
        return []

    out: list[int] = []
    for event_time, node_id in reversed(entries):
        if event_time <= timestamp:
            out.append(node_id)
            if len(out) >= limit:
                break
    return out


def _dedupe_keep_order(values: Iterable[int]) -> list[int]:
    seen: set[int] = set()
    out: list[int] = []
    for value in values:
        if value not in seen:
            seen.add(value)
            out.append(value)
    return out


def _limit_unique_recent(
    candidates: list[tuple[pd.Timestamp, int]],
    timestamp: pd.Timestamp,
    limit: int,
    excluded: set[int] | None = None,
) -> list[int]:
    """Return globally recent unique ids no later than timestamp."""
    if limit <= 0:
        return []

    excluded = excluded or set()
    out: list[int] = []
    seen: set[int] = set()
    for event_time, node_id in sorted(candidates, reverse=True):
        if event_time > timestamp or node_id in excluded or node_id in seen:
            continue
        seen.add(node_id)
        out.append(node_id)
        if len(out) >= limit:
            break
    return out


def build_temporal_indices(visit_stream: pd.DataFrame):
    user_to_ads: dict[int, list[tuple[pd.Timestamp, int]]] = defaultdict(list)
    ad_to_users: dict[int, list[tuple[pd.Timestamp, int]]] = defaultdict(list)

    cols = ["ViewDate", "UserID", "AdID"]
    events = visit_stream[cols].dropna().sort_values("ViewDate")
    for event_time, user_id, ad_id in events.itertuples(index=False, name=None):
        user = int(user_id)
        ad = int(ad_id)
        user_to_ads[user].append((event_time, ad))
        ad_to_users[ad].append((event_time, user))

    return user_to_ads, ad_to_users


def reachable_ads_for_user(
    user_id: int,
    timestamp: pd.Timestamp,
    user_to_ads: dict[int, list[tuple[pd.Timestamp, int]]],
    ad_to_users: dict[int, list[tuple[pd.Timestamp, int]]],
    max_hops: int,
    num_neighbors: int,
) -> dict[int, set[int]]:
    """Compute collapsed user-ad k-hop reachable ads.

    In the original Avito schema, UserInfo and AdsInfo are connected through
    VisitStream rows. For locality measurement we collapse a past user-ad visit
    into one conceptual hop:
      1-hop: ads this user visited before timestamp.
      3-hop: ads visited by other users who visited the 1-hop ads.
    """
    ads_by_hop: dict[int, set[int]] = {}

    current_users = {int(user_id)}
    current_ads: set[int] = set()
    seen_ads: set[int] = set()

    for hop in range(1, max_hops + 1):
        limit = max(1, num_neighbors // (2 ** (hop - 1)))

        if hop % 2 == 1:
            next_ads_with_time: list[tuple[pd.Timestamp, int]] = []
            for user in current_users:
                next_ads_with_time.extend(user_to_ads.get(user, []))
            next_ads = _limit_unique_recent(
                next_ads_with_time,
                timestamp,
                limit,
                excluded=seen_ads,
            )
            current_ads = set(next_ads)
            seen_ads.update(current_ads)
            ads_by_hop[hop] = set(current_ads)
        else:
            next_users: list[int] = []
            for ad in current_ads:
                next_users.extend(_limit_recent(ad_to_users.get(ad, []), timestamp, limit))
            current_users = set(_dedupe_keep_order(next_users))

        if not current_users and not current_ads:
            break

    return ads_by_hop


def score_split(
    split: str,
    task,
    user_to_ads: dict[int, list[tuple[pd.Timestamp, int]]],
    ad_to_users: dict[int, list[tuple[pd.Timestamp, int]]],
    max_hops: int,
    num_neighbors: int,
) -> tuple[pd.DataFrame, dict[str, dict[str, float]]]:
    table = task.get_table(split, mask_input_cols=False)
    rows = []

    odd_hops = [hop for hop in range(1, max_hops + 1) if hop % 2 == 1]
    totals = {
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

    iterator = tqdm(
        table.df.itertuples(index=False),
        total=len(table.df),
        desc=f"Scoring {split}",
    )
    for row in iterator:
        user_id = int(getattr(row, task.src_entity_col))
        timestamp = getattr(row, task.time_col)
        gt_ads = set(int(ad) for ad in getattr(row, task.dst_entity_col))
        if not gt_ads:
            continue

        reachable_by_hop = reachable_ads_for_user(
            user_id=user_id,
            timestamp=timestamp,
            user_to_ads=user_to_ads,
            ad_to_users=ad_to_users,
            max_hops=max_hops,
            num_neighbors=num_neighbors,
        )

        cumulative_ads: set[int] = set()
        row_result = {
            "split": split,
            "UserID": user_id,
            "timestamp": timestamp,
            "num_groundtruth": len(gt_ads),
        }

        for hop in odd_hops:
            cumulative_ads.update(reachable_by_hop.get(hop, set()))
            max_reachable = sum(
                max(1, num_neighbors // (2 ** (prev_hop - 1)))
                for prev_hop in odd_hops
                if prev_hop <= hop
            )
            if len(cumulative_ads) > max_reachable:
                raise RuntimeError(
                    f"hop_{hop} reachable ads exceeded the budget "
                    f"({len(cumulative_ads)} > {max_reachable})."
                )
            hits = len(cumulative_ads & gt_ads)
            recall = hits / len(gt_ads)
            precision = hits / len(cumulative_ads) if cumulative_ads else 0.0

            stats = totals[hop]
            stats["num_rows"] += 1
            stats["num_gt"] += len(gt_ads)
            stats["num_reachable"] += len(cumulative_ads)
            stats["num_hits"] += hits
            stats["sum_row_recall"] += recall
            stats["sum_row_precision"] += precision

            row_result[f"hop_{hop}_reachable_ads"] = len(cumulative_ads)
            row_result[f"hop_{hop}_hits"] = hits
            row_result[f"hop_{hop}_locality_score"] = recall
            row_result[f"hop_{hop}_precision"] = precision

        rows.append(row_result)

    summary: dict[str, dict[str, float]] = {}
    for hop, stats in totals.items():
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
            "num_groundtruth_ads": stats["num_gt"],
            "num_reachable_ads": stats["num_reachable"],
            "num_groundtruth_reachable_ads": stats["num_hits"],
        }

    return pd.DataFrame(rows), summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="rel-avito")
    parser.add_argument("--task", type=str, default="user-ad-visit")
    parser.add_argument("--num_neighbors", type=int, default=128)
    parser.add_argument("--max_hops", type=int, default=3)
    parser.add_argument("--download", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument(
        "--out_dir",
        type=Path,
        default=Path("localty_score/rel-avito/results"),
    )
    args = parser.parse_args()

    if args.max_hops < 1:
        raise ValueError("--max_hops must be positive.")

    dataset = get_dataset(args.dataset, download=args.download)
    task = get_task(args.dataset, args.task, download=args.download)
    db = dataset.get_db(upto_test_timestamp=False)

    visit_stream = db.table_dict["VisitStream"].df
    user_to_ads, ad_to_users = build_temporal_indices(visit_stream)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    all_summary = {
        "dataset": args.dataset,
        "task": args.task,
        "num_neighbors": args.num_neighbors,
        "max_hops": args.max_hops,
        "hop_neighbor_budget": {
            str(hop): max(1, args.num_neighbors // (2 ** (hop - 1)))
            for hop in range(1, args.max_hops + 1)
        },
        "definition": {
            "locality_score": "groundtruth ads reachable from the IDGNN-style k-hop local subgraph divided by all groundtruth ads",
            "neighbor_groundtruth_ratio": "reachable ads that are groundtruth divided by all reachable ads",
            "hop_convention": "User-Ad historical visits are treated as conceptual 1-hop even though the schema path is UserInfo-VisitStream-AdsInfo.",
        },
        "splits": {},
    }

    for split in ["val", "test"]:
        detail_df, summary = score_split(
            split=split,
            task=task,
            user_to_ads=user_to_ads,
            ad_to_users=ad_to_users,
            max_hops=args.max_hops,
            num_neighbors=args.num_neighbors,
        )
        detail_df.to_csv(args.out_dir / f"{split}_locality_rows.csv", index=False)
        all_summary["splits"][split] = summary

    with open(args.out_dir / "summary.json", "w") as f:
        json.dump(all_summary, f, indent=2)

    print(json.dumps(all_summary, indent=2))


if __name__ == "__main__":
    main()
