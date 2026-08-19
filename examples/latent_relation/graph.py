from __future__ import annotations

import copy
from typing import Dict

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import HeteroData
from torch_geometric.typing import EdgeType
from torch_geometric.utils import sort_edge_index

from .io import validate_snapshot_frame


LATENT_RELATION = "latent_relation"
REV_LATENT_RELATION = "rev_latent_relation"


def latent_edge_types(destination_type: str) -> tuple[EdgeType, EdgeType]:
    return (
        (destination_type, LATENT_RELATION, destination_type),
        (destination_type, REV_LATENT_RELATION, destination_type),
    )


def attach_latent_snapshot(
    base_data: HeteroData,
    snapshot: pd.DataFrame,
    seed_time: pd.Timestamp,
    *,
    destination_type: str,
    num_destinations: int | None = None,
    allow_self_loop: bool = False,
) -> HeteroData:
    """Attach direct destination-to-destination latent edges for one seed time."""
    if num_destinations is None:
        num_destinations = int(base_data[destination_type].num_nodes)
    validate_snapshot_frame(
        snapshot,
        seed_time,
        num_destinations=num_destinations,
        allow_self_loop=allow_self_loop,
    )
    data = copy.copy(base_data)
    src = torch.from_numpy(snapshot["src_id"].to_numpy(dtype=np.int64)).long()
    dst = torch.from_numpy(snapshot["dst_id"].to_numpy(dtype=np.int64)).long()
    score = torch.from_numpy(snapshot["score"].to_numpy(dtype=np.float32))
    rank = torch.from_numpy(snapshot["rank"].to_numpy(dtype=np.int32)).int()

    forward, reverse = latent_edge_types(destination_type)
    data[forward].edge_index = sort_edge_index(torch.stack([src, dst], dim=0))
    data[forward].edge_score = score
    data[forward].edge_rank = rank
    data[reverse].edge_index = sort_edge_index(torch.stack([dst, src], dim=0))
    data[reverse].edge_score = score
    data[reverse].edge_rank = rank
    data.validate()
    return data


def build_latent_schema_template(
    base_data: HeteroData,
    *,
    destination_type: str,
) -> HeteroData:
    empty = pd.DataFrame(
        {
            "src_id": pd.Series([], dtype="int64"),
            "dst_id": pd.Series([], dtype="int64"),
            "score": pd.Series([], dtype="float32"),
            "rank": pd.Series([], dtype="int32"),
            "seed_time": pd.Series([], dtype="datetime64[ns]"),
        }
    )
    return attach_latent_snapshot(
        base_data,
        empty,
        pd.Timestamp("1970-01-01"),
        destination_type=destination_type,
    )


def build_latent_num_neighbors(
    edge_types: list[EdgeType] | tuple[EdgeType, ...],
    *,
    destination_type: str,
    num_layers: int,
    num_neighbors: int,
    top_l: int,
    budget_mode: str = "additive",
) -> Dict[EdgeType, list[int]]:
    if num_layers < 3:
        raise ValueError("Latent destination relations require num_layers >= 3.")
    if budget_mode not in {"additive", "fixed"}:
        raise ValueError("budget_mode must be 'additive' or 'fixed'.")
    hop_fanouts = [int(num_neighbors // 2**i) for i in range(num_layers)]
    forward, reverse = latent_edge_types(destination_type)
    out: Dict[EdgeType, list[int]] = {
        edge_type: list(hop_fanouts) for edge_type in edge_types
    }
    zeros = [0 for _ in range(num_layers)]
    latent_fanout = int(top_l)
    if budget_mode == "fixed":
        latent_fanout = min(latent_fanout, hop_fanouts[2])
    for edge_type in (forward, reverse):
        out[edge_type] = list(zeros)
    out[reverse][2] = latent_fanout

    if budget_mode == "fixed":
        for edge_type in out:
            if edge_type in (forward, reverse):
                continue
            out[edge_type][2] = max(0, out[edge_type][2] - latent_fanout)
    return out

