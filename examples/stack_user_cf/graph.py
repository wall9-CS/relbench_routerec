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

from .io import validate_snapshot_frame


USER_CF = "user_cf"
USER_CF_SRC_F2P: EdgeType = (USER_CF, "f2p_src_UserId", "users")
USER_CF_SRC_REV: EdgeType = ("users", "rev_f2p_src_UserId", USER_CF)
USER_CF_DST_F2P: EdgeType = (USER_CF, "f2p_dst_UserId", "users")
USER_CF_DST_REV: EdgeType = ("users", "rev_f2p_dst_UserId", USER_CF)
USER_CF_EDGE_TYPES = (
    USER_CF_SRC_F2P,
    USER_CF_SRC_REV,
    USER_CF_DST_F2P,
    USER_CF_DST_REV,
)


def make_user_cf_tensor_frame(num_rows: int):
    df = pd.DataFrame({"__const__": np.ones(num_rows, dtype=np.float32)})
    return Dataset(df=df, col_to_stype={"__const__": stype.numerical}).materialize()


def user_cf_col_stats() -> dict[str, dict[StatType, object]]:
    dataset = make_user_cf_tensor_frame(1)
    return dataset.col_stats


def attach_user_cf_snapshot(
    base_data: HeteroData,
    snapshot: pd.DataFrame,
    seed_time: pd.Timestamp,
    *,
    num_users: int | None = None,
) -> HeteroData:
    """Return a shallow graph copy with exactly one Stack user-CF snapshot."""
    if num_users is None:
        num_users = int(base_data["users"].num_nodes)
    validate_snapshot_frame(snapshot, seed_time, config=None, num_users=num_users)
    data = copy.copy(base_data)
    num_rows = len(snapshot)
    cf_ids = torch.arange(num_rows, dtype=torch.long)
    seed_time = pd.Timestamp(seed_time)

    dataset = make_user_cf_tensor_frame(num_rows)
    data[USER_CF].tf = dataset.tensor_frame
    data[USER_CF].time = torch.from_numpy(
        to_unix_time(pd.Series([seed_time] * num_rows, dtype="datetime64[ns]"))
    )

    src = torch.from_numpy(snapshot["src_UserId"].to_numpy(dtype=np.int64)).long()
    dst = torch.from_numpy(snapshot["dst_UserId"].to_numpy(dtype=np.int64)).long()
    data[USER_CF_SRC_F2P].edge_index = sort_edge_index(torch.stack([cf_ids, src], dim=0))
    data[USER_CF_SRC_REV].edge_index = sort_edge_index(torch.stack([src, cf_ids], dim=0))
    data[USER_CF_DST_F2P].edge_index = sort_edge_index(torch.stack([cf_ids, dst], dim=0))
    data[USER_CF_DST_REV].edge_index = sort_edge_index(torch.stack([dst, cf_ids], dim=0))
    data.validate()
    return data


def build_user_cf_schema_template(base_data: HeteroData) -> HeteroData:
    empty = pd.DataFrame(
        {
            "seed_time": pd.Series([], dtype="datetime64[ns]"),
            "src_UserId": pd.Series([], dtype="int64"),
            "dst_UserId": pd.Series([], dtype="int64"),
            "overlap": pd.Series([], dtype="int64"),
            "user_cf_score": pd.Series([], dtype="float32"),
            "src_history_size": pd.Series([], dtype="int64"),
            "dst_history_size": pd.Series([], dtype="int64"),
            "rank": pd.Series([], dtype="int32"),
        }
    )
    return attach_user_cf_snapshot(base_data, empty, pd.Timestamp("1970-01-01"))


def build_user_cf_num_neighbors(
    edge_types: list[EdgeType] | tuple[EdgeType, ...],
    *,
    num_layers: int,
    num_neighbors: int,
) -> Dict[EdgeType, list[int]]:
    if num_layers < 4:
        raise ValueError("Stack user-CF augmentation requires num_layers >= 4.")
    hop_fanouts = [int(num_neighbors // 2**i) for i in range(num_layers)]
    out: Dict[EdgeType, list[int]] = {
        edge_type: list(hop_fanouts) for edge_type in edge_types
    }
    zeros = [0 for _ in range(num_layers)]
    for edge_type in USER_CF_EDGE_TYPES:
        out[edge_type] = list(zeros)
    out[USER_CF_SRC_F2P][0] = hop_fanouts[0]
    out[USER_CF_DST_REV][1] = hop_fanouts[1]
    return out

