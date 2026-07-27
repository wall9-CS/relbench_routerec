from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from torch_geometric.loader import NeighborLoader

from relbench.base import RecommendationTask, Table
from relbench.modeling.graph import get_link_train_table_input


@dataclass(frozen=True)
class SeedTimeGroup:
    seed_time: pd.Timestamp
    original_row_indices: np.ndarray
    table: Table


def group_recommendation_table_by_seed_time(
    table: Table,
    task: RecommendationTask,
) -> list[SeedTimeGroup]:
    """Group a recommendation table by exact seed time, preserving row indices."""
    times = pd.to_datetime(table.df[task.time_col])
    groups: list[SeedTimeGroup] = []
    for seed_time in sorted(pd.Timestamp(value) for value in times.unique()):
        mask = times == seed_time
        original = np.flatnonzero(mask.to_numpy())
        df = table.df.iloc[original].reset_index(drop=True)
        groups.append(
            SeedTimeGroup(
                seed_time=seed_time,
                original_row_indices=original.astype(np.int64),
                table=Table(
                    df=df,
                    fkey_col_to_pkey_table=table.fkey_col_to_pkey_table,
                    pkey_col=table.pkey_col,
                    time_col=table.time_col,
                ),
            )
        )
    return groups


def make_seed_time_loader(
    snapshot_graph,
    group: SeedTimeGroup,
    task: RecommendationTask,
    *,
    num_neighbors,
    batch_size: int,
    temporal_strategy: str,
    shuffle: bool,
    num_workers: int,
) -> tuple[NeighborLoader, object]:
    """Create a NeighborLoader and table input for a single seed time group."""
    table_input = get_link_train_table_input(group.table, task)
    loader = NeighborLoader(
        snapshot_graph,
        num_neighbors=num_neighbors,
        time_attr="time",
        input_nodes=table_input.src_nodes,
        input_time=table_input.src_time,
        subgraph_type="bidirectional",
        batch_size=batch_size,
        temporal_strategy=temporal_strategy,
        shuffle=shuffle,
        num_workers=num_workers,
        persistent_workers=num_workers > 0,
    )
    return loader, table_input


def scatter_group_predictions(
    output: np.ndarray,
    group: SeedTimeGroup,
    group_predictions: np.ndarray,
) -> None:
    """Scatter group-local predictions back to original task-table row order."""
    output[group.original_row_indices] = group_predictions
