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

CF_SRC_TO_DST: EdgeType = ("article", "cf_src_to_dst_article_id", "article")
CF_DST_TO_SRC: EdgeType = ("article", "rev_cf_src_to_dst_article_id", "article")
CF_EDGE_TYPES = (CF_SRC_TO_DST, CF_DST_TO_SRC)


def attach_cf_snapshot(
    base_data: HeteroData,
    snapshot: pd.DataFrame,
    seed_time: pd.Timestamp,
    *,
    num_articles: int | None = None,
) -> HeteroData:
    """Return a shallow graph copy with exactly one direct article-CF snapshot.

    Existing base graph tensors are shared with `base_data`; only direct
    article-to-article CF edge stores are added or replaced on the returned
    object. The reverse edge direction is the sampling expansion route:
    destination candidate article -> historical source article.
    """
    if num_articles is None:
        num_articles = int(base_data["article"].num_nodes)
    validate_snapshot_frame(
        snapshot,
        seed_time,
        config=None,
        num_articles=num_articles,
    )
    data = copy.copy(base_data)

    src = torch.from_numpy(snapshot["src_article_id"].to_numpy(dtype=np.int64)).long()
    dst = torch.from_numpy(snapshot["dst_article_id"].to_numpy(dtype=np.int64)).long()
    data[CF_SRC_TO_DST].edge_index = sort_edge_index(torch.stack([src, dst], dim=0))
    data[CF_DST_TO_SRC].edge_index = sort_edge_index(torch.stack([dst, src], dim=0))
    data.validate()
    return data


def build_cf_schema_template(base_data: HeteroData) -> HeteroData:
    """Build a reusable model schema containing direct CF edge types."""
    empty = pd.DataFrame(
        {
            "seed_time": pd.Series([], dtype="datetime64[ns]"),
            "src_article_id": pd.Series([], dtype="int64"),
            "dst_article_id": pd.Series([], dtype="int64"),
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
    """Build direction-preserving fanouts for direct article-CF expansion."""
    if num_layers < 3:
        raise ValueError("CF augmentation requires num_layers >= 3.")
    hop_fanouts = [int(num_neighbors // 2**i) for i in range(num_layers)]
    out: Dict[EdgeType, list[int]] = {
        edge_type: list(hop_fanouts) for edge_type in edge_types
    }
    zeros = [0 for _ in range(num_layers)]
    for edge_type in CF_EDGE_TYPES:
        out[edge_type] = list(zeros)
    out[CF_DST_TO_SRC][2] = hop_fanouts[2]
    return out
