from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd
import torch
import torch_frame
from torch_geometric.data import HeteroData
from torch_geometric.utils import sort_edge_index

from relbench.base import Database, Table


CANDIDATE_COLUMNS = [
    "customer_id",
    "source_article_id",
    "target_article_id",
    "product_code",
    "source_tx_pos",
    "source_t_dat",
]

CUSTOMER_EDGE = ("transactions", "f2p_customer_id", "customer")
REV_CUSTOMER_EDGE = ("customer", "rev_f2p_customer_id", "transactions")
ARTICLE_EDGE = ("transactions", "f2p_article_id", "article")
REV_ARTICLE_EDGE = ("article", "rev_f2p_article_id", "transactions")
HM_TRANSACTION_EDGE_TYPES = (
    CUSTOMER_EDGE,
    REV_CUSTOMER_EDGE,
    ARTICLE_EDGE,
    REV_ARTICLE_EDGE,
)


@dataclass(frozen=True)
class ProductCodeExpansionStats:
    """Summary of rel-hm product-code virtual transaction expansion."""

    seed_time: pd.Timestamp
    num_test_customers: int
    num_historical_transactions: int
    num_selected_source_transactions: int
    num_candidates_before_seen_filter: int
    num_candidates_after_seen_filter: int
    num_synthetic_transactions: int
    num_customers_with_synthetic_transactions: int
    mean_synthetic_per_expanded_customer: float
    max_synthetic_per_customer: int

    def __str__(self) -> str:
        return (
            "Product-code expansion stats:\n"
            f"  seed_time: {self.seed_time}\n"
            f"  test_customers: {self.num_test_customers}\n"
            f"  historical_transactions: {self.num_historical_transactions}\n"
            "  selected_latest_customer_product_transactions: "
            f"{self.num_selected_source_transactions}\n"
            f"  candidates_before_seen_filter: "
            f"{self.num_candidates_before_seen_filter}\n"
            f"  candidates_after_seen_filter: "
            f"{self.num_candidates_after_seen_filter}\n"
            f"  synthetic_transactions: {self.num_synthetic_transactions}\n"
            "  customers_with_synthetic_transactions: "
            f"{self.num_customers_with_synthetic_transactions}\n"
            "  mean_synthetic_per_expanded_customer: "
            f"{self.mean_synthetic_per_expanded_customer:.2f}\n"
            "  max_synthetic_per_customer: "
            f"{self.max_synthetic_per_customer}"
        )


def build_product_code_virtual_candidates(
    db: Database,
    test_table: Table,
    *,
    src_col: str = "customer_id",
    article_col: str = "article_id",
    product_code_col: str = "product_code",
    transaction_time_col: str = "t_dat",
    seed_time_col: str = "timestamp",
    max_virtual_per_customer: int = 0,
) -> tuple[pd.DataFrame, ProductCodeExpansionStats]:
    """Build deterministic same-product virtual transaction candidates.

    Candidate construction uses only test source customers, the single test seed
    timestamp, historical transactions at or before that seed time, and article
    product codes. Destination labels in the test table are intentionally unused.
    """

    if max_virtual_per_customer < 0:
        raise ValueError("max_virtual_per_customer must be non-negative")

    transactions_df = _required_table(db, "transactions").df
    article_df = _required_table(db, "article").df
    test_df = test_table.df

    _require_columns(
        transactions_df,
        "transactions",
        [src_col, article_col, transaction_time_col],
    )
    _require_columns(article_df, "article", [article_col, product_code_col])
    _require_columns(test_df, "test table", [src_col, seed_time_col])

    seed_time = _get_single_seed_time(test_df[seed_time_col], seed_time_col)
    empty = _empty_candidates()

    test_customers = pd.Series(test_df[src_col].dropna().unique(), name=src_col)
    num_test_customers = int(len(test_customers))
    if num_test_customers == 0:
        return empty, _make_stats(seed_time, 0, 0, 0, 0, 0, empty)

    tx_time = pd.to_datetime(transactions_df[transaction_time_col])
    source_tx_pos = np.arange(len(transactions_df), dtype=np.int64)
    historical_mask = (
        transactions_df[src_col].isin(test_customers)
        & tx_time.notna()
        & (tx_time <= seed_time)
    )

    historical = transactions_df.loc[
        historical_mask, [src_col, article_col, transaction_time_col]
    ].copy()
    historical["source_tx_pos"] = source_tx_pos[historical_mask.to_numpy()]
    historical["source_t_dat"] = pd.to_datetime(historical[transaction_time_col])
    historical = historical.drop(columns=[transaction_time_col])
    num_historical_transactions = int(len(historical))

    if historical.empty:
        return empty, _make_stats(
            seed_time,
            num_test_customers,
            num_historical_transactions,
            0,
            0,
            0,
            empty,
        )

    source = historical.merge(
        article_df[[article_col, product_code_col]],
        how="left",
        on=article_col,
    )
    source = source.dropna(subset=[product_code_col])
    if source.empty:
        return empty, _make_stats(
            seed_time,
            num_test_customers,
            num_historical_transactions,
            0,
            0,
            0,
            empty,
        )

    source = source.rename(columns={article_col: "source_article_id"})
    source = source.sort_values(
        [src_col, product_code_col, "source_t_dat", "source_tx_pos"],
        ascending=[True, True, True, True],
        kind="mergesort",
    )
    source = source.drop_duplicates(
        subset=[src_col, product_code_col], keep="last"
    ).reset_index(drop=True)
    num_selected_source_transactions = int(len(source))

    variants = article_df[[article_col, product_code_col]].dropna(
        subset=[product_code_col]
    )
    variants = variants.rename(columns={article_col: "target_article_id"})

    candidates = source.merge(variants, how="inner", on=product_code_col)
    candidates = candidates[
        candidates["source_article_id"] != candidates["target_article_id"]
    ].copy()
    num_candidates_before_seen_filter = int(len(candidates))

    if candidates.empty:
        return empty, _make_stats(
            seed_time,
            num_test_customers,
            num_historical_transactions,
            num_selected_source_transactions,
            num_candidates_before_seen_filter,
            0,
            empty,
        )

    seen = historical[[src_col, article_col]].drop_duplicates()
    seen = seen.rename(columns={article_col: "target_article_id"})
    candidates = candidates.merge(
        seen.assign(_seen=True),
        how="left",
        on=[src_col, "target_article_id"],
    )
    candidates = candidates[candidates["_seen"].isna()].drop(columns=["_seen"])
    num_candidates_after_seen_filter = int(len(candidates))

    if candidates.empty:
        return empty, _make_stats(
            seed_time,
            num_test_customers,
            num_historical_transactions,
            num_selected_source_transactions,
            num_candidates_before_seen_filter,
            num_candidates_after_seen_filter,
            empty,
        )

    candidates = candidates.sort_values(
        [src_col, "target_article_id", "source_t_dat", "source_tx_pos"],
        ascending=[True, True, False, False],
        kind="mergesort",
    )
    candidates = candidates.drop_duplicates(
        subset=[src_col, "target_article_id"], keep="first"
    )
    candidates = candidates.sort_values(
        [src_col, "source_t_dat", "source_tx_pos", "target_article_id"],
        ascending=[True, False, False, True],
        kind="mergesort",
    )

    if max_virtual_per_customer > 0:
        rank = candidates.groupby(src_col, sort=False).cumcount()
        candidates = candidates[rank < max_virtual_per_customer]

    candidates = candidates.rename(
        columns={src_col: "customer_id", product_code_col: "product_code"}
    )
    candidates = candidates[CANDIDATE_COLUMNS].reset_index(drop=True)

    _validate_candidates_against_history(
        candidates,
        historical=historical,
        seed_time=seed_time,
        src_col=src_col,
        article_col=article_col,
    )

    return candidates, _make_stats(
        seed_time,
        num_test_customers,
        num_historical_transactions,
        num_selected_source_transactions,
        num_candidates_before_seen_filter,
        num_candidates_after_seen_filter,
        candidates,
    )


def augment_hm_graph_with_virtual_transactions(
    data: HeteroData,
    candidates: pd.DataFrame,
) -> HeteroData:
    """Return a graph with cloned transaction nodes for product-code candidates."""

    _require_candidate_columns(candidates)
    _require_graph_schema(data)

    base_node_types = set(data.node_types)
    base_edge_types = set(data.edge_types)
    base_tx_tf_len = len(data["transactions"].tf)
    base_tx_time_len = int(data["transactions"].time.numel())
    base_edge_counts = {
        edge_type: int(data[edge_type].edge_index.size(1))
        for edge_type in HM_TRANSACTION_EDGE_TYPES
    }

    augmented = _shallow_copy_heterodata(data)
    num_synthetic = int(len(candidates))
    if num_synthetic == 0:
        augmented.validate()
        if set(augmented.node_types) != base_node_types:
            raise RuntimeError("Augmented graph changed node types")
        if set(augmented.edge_types) != base_edge_types:
            raise RuntimeError("Augmented graph changed edge types")
        return augmented

    source_pos = torch.as_tensor(
        candidates["source_tx_pos"].to_numpy(), dtype=torch.long
    )
    if bool((source_pos < 0).any()) or bool((source_pos >= base_tx_tf_len).any()):
        raise ValueError(
            "Every source_tx_pos must be in "
            f"[0, {base_tx_tf_len}); got min={int(source_pos.min())}, "
            f"max={int(source_pos.max())}"
        )

    customer_ids = torch.as_tensor(
        candidates["customer_id"].to_numpy(), dtype=torch.long
    )
    target_article_ids = torch.as_tensor(
        candidates["target_article_id"].to_numpy(), dtype=torch.long
    )
    _validate_node_ids(data, "customer", customer_ids, "candidate customer_id")
    _validate_node_ids(data, "article", target_article_ids, "candidate target_article_id")

    if (candidates["source_article_id"] == candidates["target_article_id"]).any():
        raise ValueError("Candidates must not have source_article_id == target_article_id")
    if candidates.duplicated(["customer_id", "target_article_id"]).any():
        raise ValueError("Candidates contain duplicate (customer_id, target_article_id)")

    synthetic_tf = data["transactions"].tf[source_pos]
    augmented_tf = torch_frame.cat([data["transactions"].tf, synthetic_tf], dim=0)

    source_time = data["transactions"].time[source_pos.to(data["transactions"].time.device)]
    augmented_time = torch.cat([data["transactions"].time, source_time], dim=0)
    if len(augmented_tf) != base_tx_tf_len + num_synthetic:
        raise RuntimeError("Augmented TensorFrame length does not match synthetic count")
    if int(augmented_time.numel()) != base_tx_time_len + num_synthetic:
        raise RuntimeError("Augmented transaction time length does not match synthetic count")

    augmented["transactions"].tf = augmented_tf
    augmented["transactions"].time = augmented_time
    augmented["transactions"].num_nodes = len(augmented_tf)

    synthetic_tx_ids = base_tx_tf_len + torch.arange(num_synthetic, dtype=torch.long)
    _append_edge_index(
        augmented,
        CUSTOMER_EDGE,
        torch.stack([synthetic_tx_ids, customer_ids], dim=0),
    )
    _append_edge_index(
        augmented,
        REV_CUSTOMER_EDGE,
        torch.stack([customer_ids, synthetic_tx_ids], dim=0),
    )
    _append_edge_index(
        augmented,
        ARTICLE_EDGE,
        torch.stack([synthetic_tx_ids, target_article_ids], dim=0),
    )
    _append_edge_index(
        augmented,
        REV_ARTICLE_EDGE,
        torch.stack([target_article_ids, synthetic_tx_ids], dim=0),
    )

    for edge_type in HM_TRANSACTION_EDGE_TYPES:
        gained = int(augmented[edge_type].edge_index.size(1)) - base_edge_counts[edge_type]
        if gained != num_synthetic:
            raise RuntimeError(
                f"Edge type {edge_type} gained {gained} edges; "
                f"expected {num_synthetic}"
            )

    if len(data["transactions"].tf) != base_tx_tf_len:
        raise RuntimeError("Base graph transaction TensorFrame was mutated")
    if int(data["transactions"].time.numel()) != base_tx_time_len:
        raise RuntimeError("Base graph transaction time was mutated")
    for edge_type in HM_TRANSACTION_EDGE_TYPES:
        if int(data[edge_type].edge_index.size(1)) != base_edge_counts[edge_type]:
            raise RuntimeError(f"Base graph edge {edge_type} was mutated")

    if set(augmented.node_types) != base_node_types:
        raise RuntimeError("Augmented graph changed node types")
    if set(augmented.edge_types) != base_edge_types:
        raise RuntimeError("Augmented graph changed edge types")

    augmented.validate()
    return augmented


def format_expansion_diagnostics(
    stats: ProductCodeExpansionStats,
    *,
    max_virtual_per_customer: int,
    base_num_transactions: int,
    augmented_num_transactions: int,
    train_val_num_neighbors: list[int],
    test_num_neighbors: list[int],
) -> str:
    """Format the compact diagnostics printed by the ID-GNN example."""

    return (
        f"{stats}\n"
        f"  configured_per_customer_cap: {max_virtual_per_customer}\n"
        f"  base_transaction_nodes: {base_num_transactions}\n"
        f"  augmented_transaction_nodes: {augmented_num_transactions}\n"
        f"  train_validation_neighbor_schedule: {train_val_num_neighbors}\n"
        f"  test_neighbor_schedule: {test_num_neighbors}"
    )


def _required_table(db: Database, table_name: str) -> Table:
    try:
        return db.table_dict[table_name]
    except KeyError as exc:
        raise ValueError(f"Required table '{table_name}' is missing") from exc


def _require_columns(df: pd.DataFrame, table_name: str, columns: Iterable[str]) -> None:
    missing = [col for col in columns if col not in df.columns]
    if missing:
        raise ValueError(f"{table_name} is missing required columns: {missing}")


def _require_candidate_columns(candidates: pd.DataFrame) -> None:
    _require_columns(candidates, "candidates", CANDIDATE_COLUMNS)


def _get_single_seed_time(seed_times: pd.Series, seed_time_col: str) -> pd.Timestamp:
    unique_seed_times = pd.Series(seed_times.dropna().unique())
    if len(unique_seed_times) != 1:
        raise ValueError(
            f"Expected exactly one unique non-null '{seed_time_col}', "
            f"found {len(unique_seed_times)}"
        )
    return pd.Timestamp(unique_seed_times.iloc[0])


def _empty_candidates() -> pd.DataFrame:
    return pd.DataFrame(columns=CANDIDATE_COLUMNS)


def _make_stats(
    seed_time: pd.Timestamp,
    num_test_customers: int,
    num_historical_transactions: int,
    num_selected_source_transactions: int,
    num_candidates_before_seen_filter: int,
    num_candidates_after_seen_filter: int,
    candidates: pd.DataFrame,
) -> ProductCodeExpansionStats:
    if candidates.empty:
        num_customers = 0
        mean_per_customer = 0.0
        max_per_customer = 0
    else:
        counts = candidates.groupby("customer_id").size()
        num_customers = int(len(counts))
        mean_per_customer = float(counts.mean())
        max_per_customer = int(counts.max())

    return ProductCodeExpansionStats(
        seed_time=seed_time,
        num_test_customers=num_test_customers,
        num_historical_transactions=num_historical_transactions,
        num_selected_source_transactions=num_selected_source_transactions,
        num_candidates_before_seen_filter=num_candidates_before_seen_filter,
        num_candidates_after_seen_filter=num_candidates_after_seen_filter,
        num_synthetic_transactions=int(len(candidates)),
        num_customers_with_synthetic_transactions=num_customers,
        mean_synthetic_per_expanded_customer=mean_per_customer,
        max_synthetic_per_customer=max_per_customer,
    )


def _validate_candidates_against_history(
    candidates: pd.DataFrame,
    *,
    historical: pd.DataFrame,
    seed_time: pd.Timestamp,
    src_col: str,
    article_col: str,
) -> None:
    if candidates.empty:
        return
    if (pd.to_datetime(candidates["source_t_dat"]) > seed_time).any():
        raise RuntimeError("Candidate source_t_dat contains a future transaction")
    if (candidates["source_article_id"] == candidates["target_article_id"]).any():
        raise RuntimeError("Candidate source and target article IDs must differ")
    if candidates.duplicated(["customer_id", "target_article_id"]).any():
        raise RuntimeError("Duplicate (customer_id, target_article_id) candidates found")

    seen = historical[[src_col, article_col]].rename(
        columns={src_col: "customer_id", article_col: "target_article_id"}
    )
    seen["_seen"] = True
    leaked = candidates[["customer_id", "target_article_id"]].merge(
        seen,
        how="inner",
        on=["customer_id", "target_article_id"],
    )
    if not leaked.empty:
        raise RuntimeError("Candidate target article exists in historical seen set")


def _require_graph_schema(data: HeteroData) -> None:
    for node_type in ("transactions", "customer", "article"):
        if node_type not in data.node_types:
            raise ValueError(f"Graph is missing node type '{node_type}'")
    if "tf" not in data["transactions"]:
        raise ValueError("Graph transactions store is missing TensorFrame 'tf'")
    if "time" not in data["transactions"]:
        raise ValueError("Graph transactions store is missing 'time'")
    for edge_type in HM_TRANSACTION_EDGE_TYPES:
        if edge_type not in data.edge_types:
            raise ValueError(f"Graph is missing edge type {edge_type}")
        if "edge_index" not in data[edge_type]:
            raise ValueError(f"Graph edge type {edge_type} is missing edge_index")


def _shallow_copy_heterodata(data: HeteroData) -> HeteroData:
    out = HeteroData()
    for key, value in data._global_store.items():
        out[key] = value
    for node_type in data.node_types:
        for key, value in data[node_type].items():
            out[node_type][key] = value
    for edge_type in data.edge_types:
        for key, value in data[edge_type].items():
            out[edge_type][key] = value
    return out


def _validate_node_ids(
    data: HeteroData,
    node_type: str,
    node_ids: torch.Tensor,
    label: str,
) -> None:
    num_nodes = data[node_type].num_nodes
    if num_nodes is None:
        raise ValueError(f"Cannot infer num_nodes for node type '{node_type}'")
    if bool((node_ids < 0).any()) or bool((node_ids >= int(num_nodes)).any()):
        raise ValueError(
            f"Every {label} must be in [0, {int(num_nodes)}); "
            f"got min={int(node_ids.min())}, max={int(node_ids.max())}"
        )


def _append_edge_index(
    data: HeteroData,
    edge_type: tuple[str, str, str],
    new_edges: torch.Tensor,
) -> None:
    edge_index = data[edge_type].edge_index
    new_edges = new_edges.to(device=edge_index.device, dtype=torch.long)
    data[edge_type].edge_index = sort_edge_index(
        torch.cat([edge_index, new_edges], dim=1)
    )
