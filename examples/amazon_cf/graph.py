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

PRODUCT_CF = "product_cf"
CF_SRC_F2P: EdgeType = (PRODUCT_CF, "f2p_src_product_id", "product")
CF_SRC_REV: EdgeType = ("product", "rev_f2p_src_product_id", PRODUCT_CF)
CF_DST_F2P: EdgeType = (PRODUCT_CF, "f2p_dst_product_id", "product")
CF_DST_REV: EdgeType = ("product", "rev_f2p_dst_product_id", PRODUCT_CF)
CF_EDGE_TYPES = (CF_SRC_F2P, CF_SRC_REV, CF_DST_F2P, CF_DST_REV)


def make_product_cf_tensor_frame(num_rows: int):
    df = pd.DataFrame({"__const__": np.ones(num_rows, dtype=np.float32)})
    return Dataset(df=df, col_to_stype={"__const__": stype.numerical}).materialize()


def product_cf_col_stats() -> dict[str, dict[StatType, object]]:
    dataset = make_product_cf_tensor_frame(1)
    return dataset.col_stats


def attach_cf_snapshot(
    base_data: HeteroData,
    snapshot: pd.DataFrame,
    seed_time: pd.Timestamp,
    *,
    num_products: int | None = None,
) -> HeteroData:
    """Return a shallow graph copy with exactly one Amazon product-CF snapshot."""
    if num_products is None:
        num_products = int(base_data["product"].num_nodes)
    validate_snapshot_frame(
        snapshot,
        seed_time,
        config=None,
        num_products=num_products,
    )
    data = copy.copy(base_data)
    num_rows = len(snapshot)
    cf_ids = torch.arange(num_rows, dtype=torch.long)
    seed_time = pd.Timestamp(seed_time)

    dataset = make_product_cf_tensor_frame(num_rows)
    data[PRODUCT_CF].tf = dataset.tensor_frame
    data[PRODUCT_CF].time = torch.from_numpy(
        to_unix_time(pd.Series([seed_time] * num_rows, dtype="datetime64[ns]"))
    )

    src = torch.from_numpy(snapshot["src_product_id"].to_numpy(dtype=np.int64)).long()
    dst = torch.from_numpy(snapshot["dst_product_id"].to_numpy(dtype=np.int64)).long()
    data[CF_SRC_F2P].edge_index = sort_edge_index(torch.stack([cf_ids, src], dim=0))
    data[CF_SRC_REV].edge_index = sort_edge_index(torch.stack([src, cf_ids], dim=0))
    data[CF_DST_F2P].edge_index = sort_edge_index(torch.stack([cf_ids, dst], dim=0))
    data[CF_DST_REV].edge_index = sort_edge_index(torch.stack([dst, cf_ids], dim=0))
    data.validate()
    return data


def build_cf_schema_template(base_data: HeteroData) -> HeteroData:
    empty = pd.DataFrame(
        {
            "seed_time": pd.Series([], dtype="datetime64[ns]"),
            "src_product_id": pd.Series([], dtype="int64"),
            "dst_product_id": pd.Series([], dtype="int64"),
            "support": pd.Series([], dtype="int64"),
            "cf_score": pd.Series([], dtype="float32"),
            "rank": pd.Series([], dtype="int32"),
        }
    )
    return attach_cf_snapshot(base_data, empty, pd.Timestamp("1970-01-01"))


def build_cf_num_neighbors(
    edge_types: list[EdgeType] | tuple[EdgeType, ...],
    *,
    num_layers: int,
    num_neighbors: int,
) -> Dict[EdgeType, list[int]]:
    if num_layers < 4:
        raise ValueError("Amazon CF augmentation requires num_layers >= 4.")
    hop_fanouts = [int(num_neighbors // 2**i) for i in range(num_layers)]
    out: Dict[EdgeType, list[int]] = {
        edge_type: list(hop_fanouts) for edge_type in edge_types
    }
    zeros = [0 for _ in range(num_layers)]
    for edge_type in CF_EDGE_TYPES:
        out[edge_type] = list(zeros)
    out[CF_SRC_F2P][2] = hop_fanouts[2]
    out[CF_DST_REV][3] = hop_fanouts[3]
    return out
