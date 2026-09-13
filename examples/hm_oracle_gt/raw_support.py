from __future__ import annotations

import random

import numpy as np
import pandas as pd
import torch
from torch_geometric.seed import seed_everything
from torch_geometric.typing import WITH_PYG_LIB, WITH_TORCH_SPARSE

from relbench.base import RecommendationTask, Table

from examples.evaluate_recommendation_coverage import (
    _sampled_dst_by_batch_row,
    build_uniform_num_neighbors,
    make_topology_graph,
)
from examples.hm_cf.seed_time_loader import (
    group_recommendation_table_by_seed_time,
    make_seed_time_loader,
)

from .config import OracleGTPhase1Config


QUERY_ID_COL = "query_id"


def set_sampler_seed(seed: int) -> None:
    seed_everything(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def ensure_temporal_neighbor_sampling_backend() -> None:
    if not (WITH_PYG_LIB or WITH_TORCH_SPARSE):
        raise ImportError(
            "Oracle-GT Phase 1 raw-support extraction requires the same PyG "
            "NeighborLoader path as the 4-layer baseline, but neither 'pyg-lib' "
            "nor 'torch-sparse' is installed in this Python environment."
        )


def four_layer_num_neighbors(edge_types, config: OracleGTPhase1Config) -> dict:
    if config.num_layers != 4:
        raise ValueError(f"Expected 4 layers, found {config.num_layers}.")
    return build_uniform_num_neighbors(
        edge_types,
        num_layers=config.num_layers,
        num_neighbors=config.num_neighbors,
    )


def task_table_with_query_ids(
    task: RecommendationTask,
    split: str,
    *,
    max_seed_times: int | None = None,
    max_queries: int | None = None,
) -> Table:
    table = task.get_table(split, mask_input_cols=False)
    df = table.df.copy()
    df.insert(0, QUERY_ID_COL, np.arange(len(df), dtype=np.int64))
    if max_seed_times is not None:
        seed_times = sorted(pd.Timestamp(value) for value in pd.to_datetime(df[task.time_col]).unique())
        keep = set(seed_times[:max_seed_times])
        df = df[pd.to_datetime(df[task.time_col]).map(pd.Timestamp).isin(keep)]
    if max_queries is not None:
        df = df.head(max_queries)
    df = df.reset_index(drop=True)
    return Table(
        df=df,
        fkey_col_to_pkey_table=table.fkey_col_to_pkey_table,
        pkey_col=table.pkey_col,
        time_col=table.time_col,
    )


def collect_raw_candidate_support(
    graph,
    table: Table,
    task: RecommendationTask,
    split: str,
    *,
    config: OracleGTPhase1Config,
    batch_size: int,
    num_workers: int,
) -> pd.DataFrame:
    """Collect per-query raw sampled destination support for the 4-layer graph."""
    if config.num_layers != 4:
        raise ValueError(f"Raw support must use the 4-layer baseline, got {config.num_layers}.")
    set_sampler_seed(config.sampler_seed)
    num_neighbors = build_uniform_num_neighbors(
        graph.edge_types,
        num_layers=config.num_layers,
        num_neighbors=config.num_neighbors,
    )
    rows: list[dict] = []
    for group in group_recommendation_table_by_seed_time(table, task):
        loader, _ = make_seed_time_loader(
            graph,
            group,
            task,
            num_neighbors=num_neighbors,
            batch_size=batch_size,
            temporal_strategy=config.temporal_strategy,
            shuffle=False,
            num_workers=num_workers,
        )
        for batch in loader:
            src_store = batch[task.src_entity_table]
            root_batch_size = int(src_store.batch_size)
            input_ids = src_store.input_id.cpu().numpy()
            sampled_by_row = _sampled_dst_by_batch_row(
                batch,
                task.dst_entity_table,
                root_batch_size,
            )
            for row_pos, input_id in enumerate(input_ids):
                row = group.table.df.iloc[int(input_id)]
                candidates = sorted(sampled_by_row.get(row_pos, set()))
                rows.append(
                    {
                        QUERY_ID_COL: int(row[QUERY_ID_COL]),
                        "split": split,
                        "seed_time": pd.Timestamp(row[task.time_col]),
                        "src_id": int(row[task.src_entity_col]),
                        "raw_candidate_ids": candidates,
                        "raw_candidate_count": int(len(candidates)),
                    }
                )
    if not rows:
        return pd.DataFrame(
            {
                QUERY_ID_COL: pd.Series([], dtype="int64"),
                "split": pd.Series([], dtype="object"),
                "seed_time": pd.Series([], dtype="datetime64[ns]"),
                "src_id": pd.Series([], dtype="int64"),
                "raw_candidate_ids": pd.Series([], dtype="object"),
                "raw_candidate_count": pd.Series([], dtype="int64"),
            }
        )
    return pd.DataFrame(rows).sort_values(QUERY_ID_COL).reset_index(drop=True)


def make_raw_topology_graph(db):
    return make_topology_graph(db)
