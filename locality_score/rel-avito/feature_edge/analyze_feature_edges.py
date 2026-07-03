"""
rel-avito feature-sharing virtual edge locality analysis.

This script follows the fast locality implementation:
- VisitStream is processed once in seed-time order.
- u2a/a2u histories contain only events with ViewDate <= current seed time.
- Conceptual hop fanout is NN[h-1] = num_neighbors // 4**(h-1).

Feature edges are analysis-only virtual user-ad edges. They are added to the
reachable candidate set for scoring, but RelBench tables are not modified.
"""

import argparse
import ast
import json
import operator
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from tqdm import tqdm

from relbench.datasets import get_dataset
from relbench.tasks import get_task


@dataclass(frozen=True)
class FeatureSetSpec:
    name: str
    source: str
    columns: tuple[str, ...]
    tuple_match: bool = False


def parse_feature_set(raw: str) -> FeatureSetSpec:
    """Parse name=source:col[,col...] or name=source_tuple:col1,col2."""
    if "=" not in raw or ":" not in raw:
        raise ValueError(
            f"Invalid --feature-set {raw!r}. Expected e.g. loc=ad:LocationID "
            "or loc_cat=ad_tuple:LocationID,CategoryID."
        )
    name, rest = raw.split("=", 1)
    source, cols = rest.split(":", 1)
    columns = tuple(col.strip() for col in cols.split(",") if col.strip())
    if not name.strip() or not columns:
        raise ValueError(f"Invalid --feature-set {raw!r}.")

    tuple_match = source.endswith("_tuple")
    base_source = source.removesuffix("_tuple")
    if base_source not in {"ad", "user", "visit"}:
        raise ValueError(f"Unknown feature source {source!r}.")
    if base_source != "ad" and len(columns) != 1:
        raise ValueError(f"{base_source} feature sets support one column for now.")
    return FeatureSetSpec(
        name=name.strip(),
        source=base_source,
        columns=columns,
        tuple_match=tuple_match,
    )


def parse_experiment(raw: str) -> tuple[str, str]:
    if "=" not in raw:
        raise ValueError(f"Invalid --expr {raw!r}. Expected name=expression.")
    name, expr = raw.split("=", 1)
    if not name.strip() or not expr.strip():
        raise ValueError(f"Invalid --expr {raw!r}.")
    return name.strip(), expr.strip()


def _json_safe(value):
    if isinstance(value, float) and pd.isna(value):
        return None
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    return value


def _feature_value(row: dict[str, Any], columns: tuple[str, ...]):
    values = []
    for col in columns:
        value = row.get(col)
        if value is None or pd.isna(value):
            return None
        values.append(value)
    return tuple(values) if len(values) > 1 else values[0]


def _sort_by_recent_ads(
    ads: set[int],
    ad_last_seen: dict[int, int],
    feature_fanout: int,
    exclude: set[int],
) -> set[int]:
    candidates = [
        (ad_last_seen.get(int(ad), -1), int(ad))
        for ad in ads
        if int(ad) not in exclude and int(ad) in ad_last_seen
    ]
    candidates.sort(reverse=True)
    return {ad for _, ad in candidates[:feature_fanout]}


class SetExpressionEvaluator(ast.NodeVisitor):
    OPS = {
        ast.BitAnd: operator.and_,
        ast.BitOr: operator.or_,
        ast.Sub: operator.sub,
        ast.BitXor: operator.xor,
    }

    def __init__(self, sets: dict[str, set[int]]):
        self.sets = sets

    def visit_Expression(self, node):
        return self.visit(node.body)

    def visit_Name(self, node):
        if node.id not in self.sets:
            raise ValueError(f"Unknown feature set {node.id!r} in expression.")
        return self.sets[node.id]

    def visit_BinOp(self, node):
        op_type = type(node.op)
        if op_type not in self.OPS:
            raise ValueError(f"Unsupported operator {op_type.__name__}.")
        return self.OPS[op_type](self.visit(node.left), self.visit(node.right))

    def visit_UnaryOp(self, node):
        raise ValueError("Unary operators are not supported.")

    def generic_visit(self, node):
        raise ValueError(f"Unsupported expression node {type(node).__name__}.")


def eval_set_expr(expr: str, sets: dict[str, set[int]]) -> set[int]:
    tree = ast.parse(expr, mode="eval")
    return set(SetExpressionEvaluator(sets).visit(tree))


def summarize_totals(totals: dict[int, dict[str, float]]) -> dict[str, dict[str, float]]:
    summary = {}
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
    return summary


def update_totals(
    totals: dict[int, dict[str, float]],
    hop: int,
    gt_ads: set[int],
    reachable: set[int],
) -> tuple[int, float, float]:
    hits = len(gt_ads & reachable)
    recall = hits / len(gt_ads)
    precision = hits / len(reachable) if reachable else 0.0
    stats = totals[hop]
    stats["num_rows"] += 1
    stats["num_gt"] += len(gt_ads)
    stats["num_reachable"] += len(reachable)
    stats["num_hits"] += hits
    stats["sum_row_recall"] += recall
    stats["sum_row_precision"] += precision
    return hits, recall, precision


def add_delta(summary: dict[str, dict[str, dict[str, float]]]) -> None:
    baseline = summary["baseline"]
    for case_name, case_summary in summary.items():
        if case_name == "baseline":
            continue
        for hop_name, metrics in case_summary.items():
            base_metrics = baseline[hop_name]
            metrics["delta_vs_baseline"] = {
                "locality_score_micro": (
                    metrics["locality_score_micro"]
                    - base_metrics["locality_score_micro"]
                ),
                "locality_score_macro": (
                    metrics["locality_score_macro"]
                    - base_metrics["locality_score_macro"]
                ),
                "neighbor_groundtruth_ratio_micro": (
                    metrics["neighbor_groundtruth_ratio_micro"]
                    - base_metrics["neighbor_groundtruth_ratio_micro"]
                ),
                "neighbor_groundtruth_ratio_macro": (
                    metrics["neighbor_groundtruth_ratio_macro"]
                    - base_metrics["neighbor_groundtruth_ratio_macro"]
                ),
                "num_groundtruth_reachable_ads": (
                    metrics["num_groundtruth_reachable_ads"]
                    - base_metrics["num_groundtruth_reachable_ads"]
                ),
                "num_reachable_ads": (
                    metrics["num_reachable_ads"] - base_metrics["num_reachable_ads"]
                ),
            }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="rel-avito")
    parser.add_argument("--task", type=str, default="user-ad-visit")
    parser.add_argument("--num_neighbors", type=int, default=128)
    parser.add_argument("--max_hops", type=int, default=3)
    parser.add_argument("--feature_fanout", type=int, default=128)
    parser.add_argument("--download", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--save_rows", action=argparse.BooleanOptionalAction, default=True)
    default_feature_sets = [
        "loc=ad:LocationID",
        "cat=ad:CategoryID",
        "loc_cat_pair=ad_tuple:LocationID,CategoryID",
    ]
    default_expressions = [
        "loc=loc",
        "cat=cat",
        "loc_or_cat=loc|cat",
        "loc_and_cat=loc&cat",
        "loc_cat_pair=loc_cat_pair",
    ]

    parser.add_argument(
        "--feature-set",
        action="append",
        default=None,
        help=(
            "Feature set definition, repeatable. Examples: loc=ad:LocationID, "
            "loc_cat=ad_tuple:LocationID,CategoryID, user_loc=user:LocationID, "
            "visit_col=visit:SomeColumn."
        ),
    )
    parser.add_argument(
        "--expr",
        action="append",
        default=None,
        help=(
            "Experiment expression, repeatable. Supports &, |, -, ^ and "
            "parentheses. Example: strict=loc&cat."
        ),
    )
    parser.add_argument(
        "--out_dir",
        type=Path,
        default=Path("locality_score/rel-avito/feature_edge/results"),
    )
    args = parser.parse_args()

    if args.max_hops < 1:
        raise ValueError("--max_hops must be positive.")
    if args.feature_fanout < 1:
        raise ValueError("--feature_fanout must be positive.")

    raw_feature_sets = args.feature_set or default_feature_sets
    raw_expressions = args.expr or default_expressions

    feature_specs = [parse_feature_set(raw) for raw in raw_feature_sets]
    feature_spec_by_name = {spec.name: spec for spec in feature_specs}
    if len(feature_spec_by_name) != len(feature_specs):
        raise ValueError("--feature-set names must be unique.")
    experiments = dict(parse_experiment(raw) for raw in raw_expressions)
    if len(experiments) != len(raw_expressions):
        raise ValueError("--expr names must be unique.")
    if "baseline" in experiments:
        raise ValueError("'baseline' is reserved and cannot be used as expr name.")

    odd_hops = [hop for hop in range(1, args.max_hops + 1) if hop % 2 == 1]
    nn = [max(1, args.num_neighbors // (4**i)) for i in range(args.max_hops)]

    print("Loading data...")
    dataset = get_dataset(args.dataset, download=args.download)
    task = get_task(args.dataset, args.task, download=args.download)
    db = dataset.get_db()

    src_col = task.src_entity_col
    dst_col = task.dst_entity_col

    ads_info = db.table_dict["AdsInfo"].df.copy()
    user_info = db.table_dict["UserInfo"].df.copy()
    visit_df = (
        db.table_dict["VisitStream"]
        .df.dropna(subset=["ViewDate", "UserID", "AdID"])
        .copy()
    )
    visit_df["t_unix"] = visit_df["ViewDate"].astype(np.int64) // 10**9
    visit_df = visit_df.sort_values("t_unix").reset_index(drop=True)

    ads_info_by_id = {
        int(row["AdID"]): row.to_dict()
        for _, row in ads_info.dropna(subset=["AdID"]).iterrows()
    }
    user_info_by_id = {
        int(row["UserID"]): row.to_dict()
        for _, row in user_info.dropna(subset=["UserID"]).iterrows()
    }

    ad_feature_index: dict[tuple[str, tuple[str, ...], Any], set[int]] = defaultdict(set)
    user_feature_index: dict[tuple[str, Any], set[int]] = defaultdict(set)

    for spec in feature_specs:
        if spec.source == "ad":
            missing = [col for col in spec.columns if col not in ads_info.columns]
            if missing:
                raise ValueError(f"AdsInfo is missing columns {missing} for {spec.name}.")
            for ad_id, row in ads_info_by_id.items():
                key = _feature_value(row, spec.columns)
                if key is not None:
                    ad_feature_index[(spec.name, spec.columns, key)].add(ad_id)
        elif spec.source == "user":
            col = spec.columns[0]
            if col not in user_info.columns:
                raise ValueError(f"UserInfo is missing column {col!r}.")
            for user_id, row in user_info_by_id.items():
                key = _feature_value(row, spec.columns)
                if key is not None:
                    user_feature_index[(spec.name, key)].add(user_id)
        elif spec.source == "visit":
            col = spec.columns[0]
            if col not in visit_df.columns:
                raise ValueError(f"VisitStream is missing column {col!r}.")

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

    u2a: dict[int, list[int]] = defaultdict(list)
    a2u: dict[int, list[int]] = defaultdict(list)
    ad_last_seen: dict[int, int] = {}
    visit_feature_to_ads: dict[tuple[str, Any], list[int]] = defaultdict(list)
    user_visit_values: dict[tuple[int, str], set[Any]] = defaultdict(set)

    v_ptr = 0
    v_user = visit_df["UserID"].to_numpy()
    v_ad = visit_df["AdID"].to_numpy()
    v_t = visit_df["t_unix"].to_numpy()
    visit_cols = {spec.columns[0] for spec in feature_specs if spec.source == "visit"}
    visit_values = {
        col: visit_df[col].to_numpy()
        for col in visit_cols
    }
    n_v = len(visit_df)

    case_names = ["baseline", *experiments.keys()]
    totals = {
        split: {
            case: {
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
            for case in case_names
        }
        for split in ["val", "test"]
    }
    row_records: list[dict[str, Any]] = []

    grouped = gt_df.groupby("seed_time_unix")
    for seed_t in tqdm(gt_df["seed_time_unix"].unique(), desc="Scoring"):
        while v_ptr < n_v and v_t[v_ptr] <= seed_t:
            uid = int(v_user[v_ptr])
            aid = int(v_ad[v_ptr])
            u2a[uid].append(aid)
            a2u[aid].append(uid)
            ad_last_seen[aid] = int(v_t[v_ptr])

            for col, values in visit_values.items():
                value = values[v_ptr]
                if not pd.isna(value):
                    visit_feature_to_ads[(col, value)].append(aid)
                    user_visit_values[(uid, col)].add(value)
            v_ptr += 1

        for _, row in grouped.get_group(seed_t).iterrows():
            user_id = int(row[src_col])
            gt_ads = row[dst_col]
            split = row["split"]

            current_users = {user_id}
            current_ads: set[int] = set()
            seen_ads: set[int] = set()
            ads_by_hop: dict[int, set[int]] = {}

            for hop in range(1, args.max_hops + 1):
                limit = nn[hop - 1]
                if hop % 2 == 1:
                    cand: set[int] = set()
                    for uid in current_users:
                        cand.update(u2a.get(uid, [])[-limit:])
                    current_ads = cand - seen_ads
                    seen_ads.update(current_ads)
                    ads_by_hop[hop] = set(current_ads)
                else:
                    next_users: list[int] = []
                    for aid in current_ads:
                        next_users.extend(a2u.get(aid, [])[-limit:])
                    current_users = set(next_users)

                if not current_users and not current_ads:
                    break

            cumulative_ads: set[int] = set()
            for hop in odd_hops:
                cumulative_ads.update(ads_by_hop.get(hop, set()))

                feature_sets: dict[str, set[int]] = {}
                for name, spec in feature_spec_by_name.items():
                    raw_candidates: set[int] = set()
                    if spec.source == "ad":
                        values = {
                            _feature_value(ads_info_by_id.get(aid, {}), spec.columns)
                            for aid in cumulative_ads
                        }
                        values.discard(None)
                        for value in values:
                            raw_candidates.update(
                                ad_feature_index.get((name, spec.columns, value), set())
                            )
                    elif spec.source == "user":
                        value = _feature_value(
                            user_info_by_id.get(user_id, {}),
                            spec.columns,
                        )
                        for other_user in user_feature_index.get((name, value), set()):
                            raw_candidates.update(u2a.get(int(other_user), []))
                    elif spec.source == "visit":
                        col = spec.columns[0]
                        for value in user_visit_values.get((user_id, col), set()):
                            raw_candidates.update(visit_feature_to_ads.get((col, value), []))

                    feature_sets[name] = _sort_by_recent_ads(
                        raw_candidates,
                        ad_last_seen=ad_last_seen,
                        feature_fanout=args.feature_fanout,
                        exclude=cumulative_ads,
                    )

                cases: dict[str, set[int]] = {"baseline": set(cumulative_ads)}
                for case_name, expr in experiments.items():
                    cases[case_name] = set(cumulative_ads) | eval_set_expr(
                        expr,
                        feature_sets,
                    )

                for case_name, reachable in cases.items():
                    hits, recall, precision = update_totals(
                        totals[split][case_name],
                        hop,
                        gt_ads,
                        reachable,
                    )
                    if args.save_rows:
                        row_records.append(
                            {
                                "split": split,
                                "case": case_name,
                                "UserID": user_id,
                                "seed_time_unix": int(seed_t),
                                "hop": hop,
                                "num_groundtruth": len(gt_ads),
                                "baseline_reachable_ads": len(cumulative_ads),
                                "reachable_ads": len(reachable),
                                "feature_added_ads": max(
                                    0, len(reachable) - len(cumulative_ads)
                                ),
                                "hits": hits,
                                "locality_score": recall,
                                "precision": precision,
                            }
                        )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    all_summary = {
        "dataset": args.dataset,
        "task": args.task,
        "num_neighbors": args.num_neighbors,
        "max_hops": args.max_hops,
        "feature_fanout": args.feature_fanout,
        "hop_neighbor_budget": {str(h): nn[h - 1] for h in range(1, args.max_hops + 1)},
        "feature_sets": {
            spec.name: {
                "source": spec.source,
                "columns": list(spec.columns),
                "tuple_match": spec.tuple_match,
            }
            for spec in feature_specs
        },
        "experiments": experiments,
        "definition": {
            "baseline": "Fast incremental IDGNN-style locality over historical VisitStream.",
            "feature_edge": "Analysis-only virtual user-ad edges from feature-set expressions.",
            "expression_ops": "& intersection, | union, - difference, ^ symmetric difference",
        },
        "splits": {},
    }

    for split in ["val", "test"]:
        split_summary = {
            case_name: summarize_totals(case_totals)
            for case_name, case_totals in totals[split].items()
        }
        add_delta(split_summary)
        all_summary["splits"][split] = split_summary

    if args.save_rows and row_records:
        all_rows_df = pd.DataFrame(row_records)
        for split in ["val", "test"]:
            split_df = all_rows_df[all_rows_df["split"] == split]
            split_df.to_csv(args.out_dir / f"{split}_feature_edge_rows.csv", index=False)

    with open(args.out_dir / "summary.json", "w") as f:
        json.dump(all_summary, f, indent=2, default=_json_safe)

    print(json.dumps(all_summary, indent=2, default=_json_safe))


if __name__ == "__main__":
    main()
