from __future__ import annotations

import copy
from typing import Dict

import numpy as np
import pandas as pd
import torch
from torch_frame import stype
from torch_frame.data import Dataset
from torch_frame.data.stats import StatType
from torch_geometric.data import HeteroData
from torch_geometric.typing import EdgeType
from torch_geometric.utils import sort_edge_index

from relbench.modeling.utils import to_unix_time

from .config import CF_NODE_TYPE, TrialCFSnapshotConfig
from .io import validate_snapshot_frame


def cf_src_edge_type(config: TrialCFSnapshotConfig) -> EdgeType:
    return (CF_NODE_TYPE, f"f2p_{config.src_entity_col}", config.src_entity_table)


def cf_src_rev_edge_type(config: TrialCFSnapshotConfig) -> EdgeType:
    return (config.src_entity_table, f"rev_f2p_{config.src_entity_col}", CF_NODE_TYPE)


def cf_dst_edge_type(config: TrialCFSnapshotConfig) -> EdgeType:
    return (CF_NODE_TYPE, "f2p_dst_sponsor_id", "sponsors")


def cf_dst_rev_edge_type(config: TrialCFSnapshotConfig) -> EdgeType:
    return ("sponsors", "rev_f2p_dst_sponsor_id", CF_NODE_TYPE)


def cf_edge_types(config: TrialCFSnapshotConfig) -> tuple[EdgeType, ...]:
    return (
        cf_src_edge_type(config),
        cf_src_rev_edge_type(config),
        cf_dst_edge_type(config),
        cf_dst_rev_edge_type(config),
    )


def make_trial_cf_tensor_frame(num_rows: int):
    df = pd.DataFrame({"__const__": np.ones(num_rows, dtype=np.float32)})
    return Dataset(df=df, col_to_stype={"__const__": stype.numerical}).materialize()


def trial_cf_col_stats() -> dict[str, dict[StatType, object]]:
    dataset = make_trial_cf_tensor_frame(1)
    return dataset.col_stats


def attach_cf_snapshot(
    base_data: HeteroData,
    snapshot: pd.DataFrame,
    seed_time: pd.Timestamp,
    *,
    config: TrialCFSnapshotConfig,
    num_sources: int | None = None,
    num_sponsors: int | None = None,
) -> HeteroData:
    """Return a shallow graph copy with one Rel-Trial route-collapsed CF snapshot."""
    if num_sources is None:
        num_sources = int(base_data[config.src_entity_table].num_nodes)
    if num_sponsors is None:
        num_sponsors = int(base_data["sponsors"].num_nodes)
    validate_snapshot_frame(
        snapshot,
        seed_time,
        config=config,
        num_sources=num_sources,
        num_sponsors=num_sponsors,
    )
    data = copy.copy(base_data)
    num_rows = len(snapshot)
    cf_ids = torch.arange(num_rows, dtype=torch.long)
    seed_time = pd.Timestamp(seed_time)

    dataset = make_trial_cf_tensor_frame(num_rows)
    data[CF_NODE_TYPE].tf = dataset.tensor_frame
    data[CF_NODE_TYPE].time = torch.from_numpy(
        to_unix_time(pd.Series([seed_time] * num_rows, dtype="datetime64[ns]"))
    )

    src = torch.from_numpy(snapshot["src_id"].to_numpy(dtype=np.int64)).long()
    dst = torch.from_numpy(snapshot["dst_sponsor_id"].to_numpy(dtype=np.int64)).long()
    data[cf_src_edge_type(config)].edge_index = sort_edge_index(
        torch.stack([cf_ids, src], dim=0)
    )
    data[cf_src_rev_edge_type(config)].edge_index = sort_edge_index(
        torch.stack([src, cf_ids], dim=0)
    )
    data[cf_dst_edge_type(config)].edge_index = sort_edge_index(
        torch.stack([cf_ids, dst], dim=0)
    )
    data[cf_dst_rev_edge_type(config)].edge_index = sort_edge_index(
        torch.stack([dst, cf_ids], dim=0)
    )
    data.validate()
    return data


def build_cf_schema_template(
    base_data: HeteroData,
    *,
    config: TrialCFSnapshotConfig,
) -> HeteroData:
    empty = pd.DataFrame(
        {
            "seed_time": pd.Series([], dtype="datetime64[ns]"),
            "src_id": pd.Series([], dtype="int64"),
            "dst_sponsor_id": pd.Series([], dtype="int64"),
            "support": pd.Series([], dtype="int64"),
            "cf_score": pd.Series([], dtype="float32"),
            "rank": pd.Series([], dtype="int32"),
        }
    )
    return attach_cf_snapshot(
        base_data,
        empty,
        pd.Timestamp("1970-01-01"),
        config=config,
    )


def build_cf_num_neighbors(
    edge_types: list[EdgeType] | tuple[EdgeType, ...],
    *,
    config: TrialCFSnapshotConfig,
    num_layers: int,
    num_neighbors: int,
) -> Dict[EdgeType, list[int]]:
    if num_layers < 2:
        raise ValueError("Trial CF augmentation requires num_layers >= 2.")
    hop_fanouts = [int(num_neighbors // 2**i) for i in range(num_layers)]
    out: Dict[EdgeType, list[int]] = {
        edge_type: list(hop_fanouts) for edge_type in edge_types
    }
    zeros = [0 for _ in range(num_layers)]
    for edge_type in cf_edge_types(config):
        out[edge_type] = list(zeros)
    out[cf_src_edge_type(config)][0] = hop_fanouts[0]
    out[cf_dst_rev_edge_type(config)][1] = hop_fanouts[1]
    return out

