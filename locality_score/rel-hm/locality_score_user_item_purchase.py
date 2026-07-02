import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Iterable, NamedTuple

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


class TemporalAdjacency(NamedTuple):
    customer_to_articles: dict[int, list[tuple[pd.Timestamp, int]]]
    article_to_customers: dict[int, list[tuple[pd.Timestamp, int]]]


def build_temporal_indices(transactions: pd.DataFrame):
    customer_to_articles: dict[int, list[tuple[pd.Timestamp, int]]] = defaultdict(list)
    article_to_customers: dict[int, list[tuple[pd.Timestamp, int]]] = defaultdict(list)

    cols = ["t_dat", "customer_id", "article_id"]
    events = transactions[cols].dropna().sort_values("t_dat")
    for event_time, customer_id, article_id in events.itertuples(index=False, name=None):
        customer = int(customer_id)
        article = int(article_id)
        customer_to_articles[customer].append((event_time, article))
        article_to_customers[article].append((event_time, customer))

    return customer_to_articles, article_to_customers


def build_recent_adjacency_for_timestamp(
    transactions: pd.DataFrame,
    timestamp: pd.Timestamp,
    max_article_neighbors: int,
    max_customer_neighbors: int,
) -> TemporalAdjacency:
    """Precompute recent temporal neighbors needed for one timestamp.

    Lists are globally time-descending within each node and already deduplicated.
    """
    customer_to_articles: dict[int, list[tuple[pd.Timestamp, int]]] = defaultdict(list)
    article_to_customers: dict[int, list[tuple[pd.Timestamp, int]]] = defaultdict(list)
    seen_customer_article: set[tuple[int, int]] = set()
    seen_article_customer: set[tuple[int, int]] = set()

    cols = ["t_dat", "customer_id", "article_id"]
    events = (
        transactions.loc[transactions["t_dat"] <= timestamp, cols]
        .dropna()
        .sort_values("t_dat", ascending=False)
    )
    for event_time, customer_id, article_id in events.itertuples(index=False, name=None):
        customer = int(customer_id)
        article = int(article_id)

        if (
            len(customer_to_articles[customer]) < max_article_neighbors
            and (customer, article) not in seen_customer_article
        ):
            customer_to_articles[customer].append((event_time, article))
            seen_customer_article.add((customer, article))

        if (
            len(article_to_customers[article]) < max_customer_neighbors
            and (article, customer) not in seen_article_customer
        ):
            article_to_customers[article].append((event_time, customer))
            seen_article_customer.add((article, customer))

    return TemporalAdjacency(
        customer_to_articles=dict(customer_to_articles),
        article_to_customers=dict(article_to_customers),
    )


def build_recent_adjacencies(
    transactions: pd.DataFrame,
    timestamps: Iterable[pd.Timestamp],
    max_hops: int,
    num_neighbors: int,
) -> dict[pd.Timestamp, TemporalAdjacency]:
    odd_limits = [
        max(1, num_neighbors // (2 ** (hop - 1)))
        for hop in range(1, max_hops + 1)
        if hop % 2 == 1
    ]
    even_limits = [
        max(1, num_neighbors // (2 ** (hop - 1)))
        for hop in range(1, max_hops + 1)
        if hop % 2 == 0
    ]
    max_article_neighbors = max(odd_limits, default=1)
    max_customer_neighbors = max(even_limits, default=1)

    return {
        timestamp: build_recent_adjacency_for_timestamp(
            transactions,
            timestamp,
            max_article_neighbors=max_article_neighbors,
            max_customer_neighbors=max_customer_neighbors,
        )
        for timestamp in tqdm(
            sorted(set(pd.Timestamp(ts) for ts in timestamps)),
            desc="Precomputing timestamp adjacencies",
        )
    }


def reachable_articles_for_customer(
    customer_id: int,
    timestamp: pd.Timestamp,
    customer_to_articles: dict[int, list[tuple[pd.Timestamp, int]]],
    article_to_customers: dict[int, list[tuple[pd.Timestamp, int]]],
    max_hops: int,
    num_neighbors: int,
) -> dict[int, set[int]]:
    """Compute collapsed customer-article k-hop reachable articles.

    A historical purchase is treated as one conceptual hop:
      1-hop: articles this customer purchased before timestamp.
      3-hop: articles purchased before timestamp by customers who also
             purchased the customer's hop-1 articles.
    """
    articles_by_hop: dict[int, set[int]] = {}

    current_customers = {int(customer_id)}
    current_articles: set[int] = set()
    seen_articles: set[int] = set()

    for hop in range(1, max_hops + 1):
        limit = max(1, num_neighbors // (2**(hop - 1)))

        if hop % 2 == 1:
            next_articles_with_time: list[tuple[pd.Timestamp, int]] = []
            for customer in current_customers:
                next_articles_with_time.extend(customer_to_articles.get(customer, []))
            next_articles = _limit_unique_recent(
                next_articles_with_time,
                timestamp,
                limit,
                excluded=seen_articles,
            )
            current_articles = set(next_articles)
            seen_articles.update(current_articles)
            articles_by_hop[hop] = set(current_articles)
        else:
            next_customers: list[int] = []
            for article in current_articles:
                next_customers.extend(
                    _limit_recent(
                        article_to_customers.get(article, []),
                        timestamp,
                        limit,
                    )
                )
            current_customers = set(_dedupe_keep_order(next_customers))

        if not current_customers and not current_articles:
            break

    return articles_by_hop


def reachable_articles_for_customer_fast(
    customer_id: int,
    adjacency: TemporalAdjacency,
    max_hops: int,
    num_neighbors: int,
) -> dict[int, set[int]]:
    """Compute reachable articles with frontier-node fanout sampling."""
    articles_by_hop: dict[int, set[int]] = {}
    current_customers = {int(customer_id)}
    current_articles: set[int] = set()
    seen_articles: set[int] = set()

    for hop in range(1, max_hops + 1):
        limit = max(1, num_neighbors // (2 ** (hop - 1)))

        if hop % 2 == 1:
            next_articles_with_time: list[tuple[pd.Timestamp, int]] = []
            for customer in current_customers:
                next_articles_with_time.extend(
                    adjacency.customer_to_articles.get(customer, [])[:limit]
                )
            current_articles = set(
                _limit_unique_recent(
                    next_articles_with_time,
                    pd.Timestamp.max,
                    len(next_articles_with_time),
                    excluded=seen_articles,
                )
            )
            seen_articles.update(current_articles)
            articles_by_hop[hop] = set(current_articles)
        else:
            next_customers_with_time: list[tuple[pd.Timestamp, int]] = []
            for article in current_articles:
                next_customers_with_time.extend(
                    adjacency.article_to_customers.get(article, [])[:limit]
                )
            current_customers = set(
                _limit_unique_recent(
                    next_customers_with_time,
                    pd.Timestamp.max,
                    len(next_customers_with_time),
                )
            )

        if not current_customers and not current_articles:
            break

    return articles_by_hop


def score_split(
    split: str,
    task,
    timestamp_adjacencies: dict[pd.Timestamp, TemporalAdjacency],
    max_hops: int,
    num_neighbors: int,
    save_rows: bool,
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
    reachable_cache: dict[tuple[int, pd.Timestamp], dict[int, set[int]]] = {}

    iterator = tqdm(
        table.df.itertuples(index=False),
        total=len(table.df),
        desc=f"Scoring {split}",
    )
    for row in iterator:
        customer_id = int(getattr(row, task.src_entity_col))
        timestamp = pd.Timestamp(getattr(row, task.time_col))
        gt_articles = set(int(article) for article in getattr(row, task.dst_entity_col))
        if not gt_articles:
            continue

        cache_key = (customer_id, timestamp)
        if cache_key not in reachable_cache:
            reachable_cache[cache_key] = reachable_articles_for_customer_fast(
                customer_id=customer_id,
                adjacency=timestamp_adjacencies[timestamp],
                max_hops=max_hops,
                num_neighbors=num_neighbors,
            )
        reachable_by_hop = reachable_cache[cache_key]

        cumulative_articles: set[int] = set()
        row_result = None
        if save_rows:
            row_result = {
                "split": split,
                "customer_id": customer_id,
                "timestamp": timestamp,
                "num_groundtruth": len(gt_articles),
            }

        for hop in odd_hops:
            cumulative_articles.update(reachable_by_hop.get(hop, set()))
            hits = len(cumulative_articles & gt_articles)
            recall = hits / len(gt_articles)
            precision = hits / len(cumulative_articles) if cumulative_articles else 0.0

            stats = totals[hop]
            stats["num_rows"] += 1
            stats["num_gt"] += len(gt_articles)
            stats["num_reachable"] += len(cumulative_articles)
            stats["num_hits"] += hits
            stats["sum_row_recall"] += recall
            stats["sum_row_precision"] += precision

            if row_result is not None:
                row_result[f"hop_{hop}_reachable_articles"] = len(cumulative_articles)
                row_result[f"hop_{hop}_hits"] = hits
                row_result[f"hop_{hop}_locality_score"] = recall
                row_result[f"hop_{hop}_precision"] = precision

        if row_result is not None:
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
            "num_groundtruth_articles": stats["num_gt"],
            "num_reachable_articles": stats["num_reachable"],
            "num_groundtruth_reachable_articles": stats["num_hits"],
        }

    return pd.DataFrame(rows), summary


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
        default=Path("locality_score/rel-hm/results"),
    )
    args = parser.parse_args()

    if args.max_hops < 1:
        raise ValueError("--max_hops must be positive.")

    dataset = get_dataset(args.dataset, download=args.download)
    task = get_task(args.dataset, args.task, download=args.download)
    db = dataset.get_db(upto_test_timestamp=False)

    transactions = db.table_dict["transactions"].df
    split_tables = {
        split: task.get_table(split, mask_input_cols=False)
        for split in ["val", "test"]
    }
    timestamps = []
    for table in split_tables.values():
        timestamps.extend(pd.Timestamp(ts) for ts in table.df[task.time_col].unique())
    timestamp_adjacencies = build_recent_adjacencies(
        transactions,
        timestamps=timestamps,
        max_hops=args.max_hops,
        num_neighbors=args.num_neighbors,
    )

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
            "locality_score": "groundtruth articles reachable from the IDGNN-style k-hop local subgraph divided by all groundtruth articles",
            "neighbor_groundtruth_ratio": "reachable articles that are groundtruth divided by all reachable articles",
            "hop_convention": "Customer-article historical purchases are treated as conceptual 1-hop local neighbors.",
        },
        "splits": {},
    }

    for split in ["val", "test"]:
        detail_df, summary = score_split(
            split=split,
            task=task,
            timestamp_adjacencies=timestamp_adjacencies,
            max_hops=args.max_hops,
            num_neighbors=args.num_neighbors,
            save_rows=args.save_rows,
        )
        if args.save_rows:
            detail_df.to_csv(args.out_dir / f"{split}_locality_rows.csv", index=False)
        all_summary["splits"][split] = summary

    with open(args.out_dir / "summary.json", "w") as f:
        json.dump(all_summary, f, indent=2)

    print(json.dumps(all_summary, indent=2))


if __name__ == "__main__":
    main()
