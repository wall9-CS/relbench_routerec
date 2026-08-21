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

DIRECT_ARTICLE_CF = "direct_article_cf"
DIRECT_CF_SRC_F2P: EdgeType = (DIRECT_ARTICLE_CF, "f2p_src_customer_id", "customer")
DIRECT_CF_SRC_REV: EdgeType = ("customer", "rev_f2p_src_customer_id", DIRECT_ARTICLE_CF)
DIRECT_CF_DST_F2P: EdgeType = (DIRECT_ARTICLE_CF, "f2p_dst_article_id", "article")
DIRECT_CF_DST_REV: EdgeType = ("article", "rev_f2p_dst_article_id", DIRECT_ARTICLE_CF)
DIRECT_CF_EDGE_TYPES = (
    DIRECT_CF_SRC_F2P,
    DIRECT_CF_SRC_REV,
    DIRECT_CF_DST_F2P,
    DIRECT_CF_DST_REV,
)


def make_direct_article_cf_tensor_frame(num_rows: int):
    df = pd.DataFrame({"__const__": np.ones(num_rows, dtype=np.float32)})
    return Dataset(df=df, col_to_stype={"__const__": stype.numerical}).materialize()


def direct_article_cf_col_stats() -> dict[str, dict[StatType, object]]:
    dataset = make_direct_article_cf_tensor_frame(1)
    return dataset.col_stats


def attach_direct_cf_snapshot(
    base_data: HeteroData,
    snapshot: pd.DataFrame,
    seed_time: pd.Timestamp,
    *,
    num_customers: int | None = None,
    num_articles: int | None = None,
) -> HeteroData:
    if num_customers is None:
        num_customers = int(base_data["customer"].num_nodes)
    if num_articles is None:
        num_articles = int(base_data["article"].num_nodes)
    validate_snapshot_frame(
        snapshot,
        seed_time,
        config=None,
        num_customers=num_customers,
        num_articles=num_articles,
    )
    data = copy.copy(base_data)
    num_rows = len(snapshot)
    cf_ids = torch.arange(num_rows, dtype=torch.long)
    seed_time = pd.Timestamp(seed_time)

    dataset = make_direct_article_cf_tensor_frame(num_rows)
    data[DIRECT_ARTICLE_CF].tf = dataset.tensor_frame
    data[DIRECT_ARTICLE_CF].time = torch.from_numpy(
        to_unix_time(pd.Series([seed_time] * num_rows, dtype="datetime64[ns]"))
    )
    src = torch.from_numpy(snapshot["src_customer_id"].to_numpy(dtype=np.int64)).long()
    dst = torch.from_numpy(snapshot["dst_article_id"].to_numpy(dtype=np.int64)).long()
    data[DIRECT_CF_SRC_F2P].edge_index = sort_edge_index(torch.stack([cf_ids, src], dim=0))
    data[DIRECT_CF_SRC_REV].edge_index = sort_edge_index(torch.stack([src, cf_ids], dim=0))
    data[DIRECT_CF_DST_F2P].edge_index = sort_edge_index(torch.stack([cf_ids, dst], dim=0))
    data[DIRECT_CF_DST_REV].edge_index = sort_edge_index(torch.stack([dst, cf_ids], dim=0))
    data.validate()
    return data


def build_direct_cf_schema_template(base_data: HeteroData) -> HeteroData:
    empty = pd.DataFrame(
        {
            "seed_time": pd.Series([], dtype="datetime64[ns]"),
            "src_customer_id": pd.Series([], dtype="int64"),
            "dst_article_id": pd.Series([], dtype="int64"),
            "direct_score": pd.Series([], dtype="float32"),
            "max_cf_score": pd.Series([], dtype="float32"),
            "support_sum": pd.Series([], dtype="int64"),
            "support_max": pd.Series([], dtype="int64"),
            "num_source_articles": pd.Series([], dtype="int64"),
            "best_src_article_id": pd.Series([], dtype="int64"),
            "best_item_cf_rank": pd.Series([], dtype="int32"),
            "rank": pd.Series([], dtype="int32"),
        }
    )
    return attach_direct_cf_snapshot(base_data, empty, pd.Timestamp("1970-01-01"))


def build_direct_cf_num_neighbors(
    edge_types: list[EdgeType] | tuple[EdgeType, ...],
    *,
    num_layers: int,
    num_neighbors: int,
) -> Dict[EdgeType, list[int]]:
    if num_layers < 2:
        raise ValueError("Direct CF augmentation requires num_layers >= 2.")
    hop_fanouts = [int(num_neighbors // 2**i) for i in range(num_layers)]
    out: Dict[EdgeType, list[int]] = {
        edge_type: list(hop_fanouts) for edge_type in edge_types
    }
    zeros = [0 for _ in range(num_layers)]
    for edge_type in DIRECT_CF_EDGE_TYPES:
        out[edge_type] = list(zeros)
    out[DIRECT_CF_SRC_F2P][0] = hop_fanouts[0]
    out[DIRECT_CF_DST_REV][1] = hop_fanouts[1]
    return out
