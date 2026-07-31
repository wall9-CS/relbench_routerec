import numpy as np
import pandas as pd
import pytest
import torch
import torch.nn.functional as F
from torch_geometric.data import HeteroData
from torch_geometric.loader import NeighborLoader
from torch_geometric.typing import WITH_PYG_LIB

from examples.model import Model
from examples.hm_cf.graph import (
    ARTICLE_CF,
    CF_DST_F2P,
    CF_DST_REV,
    CF_EDGE_TYPES,
    CF_SRC_F2P,
    CF_SRC_REV,
    article_cf_col_stats,
    attach_cf_snapshot,
    build_cf_num_neighbors,
    build_cf_schema_template,
    make_article_cf_tensor_frame,
)
from relbench.modeling.utils import to_unix_time


def _base_graph(num_articles=4):
    data = HeteroData()
    data["article"].tf = make_article_cf_tensor_frame(num_articles).tensor_frame
    data["article"].time = torch.zeros(num_articles, dtype=torch.long)
    data["customer"].tf = make_article_cf_tensor_frame(1).tensor_frame
    data["customer"].time = torch.tensor([to_unix_time(pd.Series([pd.Timestamp("2020-03-02")]))[0]])
    data["transactions"].tf = make_article_cf_tensor_frame(1).tensor_frame
    data["transactions"].time = torch.tensor(
        [to_unix_time(pd.Series([pd.Timestamp("2020-02-28")]))[0]]
    )
    data[("transactions", "f2p_customer_id", "customer")].edge_index = torch.tensor(
        [[0], [0]], dtype=torch.long
    )
    data[("customer", "rev_f2p_customer_id", "transactions")].edge_index = torch.tensor(
        [[0], [0]], dtype=torch.long
    )
    data[("transactions", "f2p_article_id", "article")].edge_index = torch.tensor(
        [[0], [0]], dtype=torch.long
    )
    data[("article", "rev_f2p_article_id", "transactions")].edge_index = torch.tensor(
        [[0], [0]], dtype=torch.long
    )
    return data


def _snapshot(seed="2020-03-02"):
    return pd.DataFrame(
        {
            "seed_time": pd.to_datetime([seed, seed]),
            "src_article_id": pd.Series([0, 2], dtype="int64"),
            "dst_article_id": pd.Series([1, 0], dtype="int64"),
            "support": pd.Series([3, 3], dtype="int64"),
            "cf_score": pd.Series([0.7, 0.6], dtype="float32"),
            "rank": pd.Series([1, 1], dtype="int32"),
        }
    )


def test_no_model_feature_and_col_stats_are_constant_only():
    data = attach_cf_snapshot(_base_graph(), _snapshot(), pd.Timestamp("2020-03-02"))
    assert data[ARTICLE_CF].tf.col_names_dict.keys()
    assert data[ARTICLE_CF].tf.col_names_dict[next(iter(data[ARTICLE_CF].tf.col_names_dict))] == [
        "__const__"
    ]
    assert set(article_cf_col_stats()) == {"__const__"}


def test_base_graph_immutability():
    base = _base_graph()
    original_node_types = list(base.node_types)
    original_edge_types = list(base.edge_types)
    attached = attach_cf_snapshot(base, _snapshot(), pd.Timestamp("2020-03-02"))
    assert base.node_types == original_node_types
    assert base.edge_types == original_edge_types
    assert ARTICLE_CF in attached.node_types
    assert ARTICLE_CF not in base.node_types


def test_graph_edge_roles_exact():
    data = attach_cf_snapshot(_base_graph(), _snapshot(), pd.Timestamp("2020-03-02"))
    assert torch.equal(data[CF_SRC_F2P].edge_index, torch.tensor([[0, 1], [0, 2]]))
    assert torch.equal(data[CF_SRC_REV].edge_index, torch.tensor([[0, 2], [0, 1]]))
    assert torch.equal(data[CF_DST_F2P].edge_index, torch.tensor([[0, 1], [1, 0]]))
    assert torch.equal(data[CF_DST_REV].edge_index, torch.tensor([[0, 1], [1, 0]]))


def test_schema_template_and_fanouts():
    template = build_cf_schema_template(_base_graph())
    for edge_type in CF_EDGE_TYPES:
        assert edge_type in template.edge_types
        assert template[edge_type].edge_index.numel() == 0
    fanouts = build_cf_num_neighbors(template.edge_types, num_layers=4, num_neighbors=128)
    assert fanouts[CF_SRC_F2P] == [0, 0, 32, 0]
    assert fanouts[CF_DST_REV] == [0, 0, 0, 16]
    assert fanouts[CF_DST_F2P] == [0, 0, 0, 0]
    assert fanouts[CF_SRC_REV] == [0, 0, 0, 0]
    with pytest.raises(ValueError, match="num_layers"):
        build_cf_num_neighbors(template.edge_types, num_layers=3, num_neighbors=128)


def test_seed_time_isolation_by_exact_attachment():
    base = _base_graph()
    t1 = pd.Timestamp("2020-03-02")
    t2 = pd.Timestamp("2020-03-09")
    g1 = attach_cf_snapshot(base, _snapshot(t1), t1)
    snap2 = _snapshot(t2)
    snap2["dst_article_id"] = [3, 1]
    g2 = attach_cf_snapshot(base, snap2, t2)
    assert set(g1[CF_DST_F2P].edge_index[1].tolist()) == {0, 1}
    assert set(g2[CF_DST_F2P].edge_index[1].tolist()) == {1, 3}
    assert g1[ARTICLE_CF].time.unique().item() != g2[ARTICLE_CF].time.unique().item()


def test_four_hop_reachability_route_with_static_graph():
    data = attach_cf_snapshot(_base_graph(), _snapshot(), pd.Timestamp("2020-03-02"))
    tx_to_customer = data[("transactions", "f2p_customer_id", "customer")].edge_index
    tx = tx_to_customer[0, tx_to_customer[1] == 0]
    article_src = data[("transactions", "f2p_article_id", "article")].edge_index[1]
    assert article_src[tx].tolist() == [0]
    cf_from_src = data[CF_SRC_F2P].edge_index[0][data[CF_SRC_F2P].edge_index[1] == 0]
    dst = data[CF_DST_F2P].edge_index[1][torch.isin(data[CF_DST_F2P].edge_index[0], cf_from_src)]
    assert dst.tolist() == [1]
    reverse_only = data[CF_DST_F2P].edge_index[0][data[CF_DST_F2P].edge_index[1] == 0]
    assert set(reverse_only.tolist()) == {1}
    assert not torch.isin(reverse_only, cf_from_src).any()


def test_neighbor_loader_route_skips_without_temporal_backend():
    if not WITH_PYG_LIB:
        pytest.skip("PyG temporal NeighborLoader requires pyg-lib in this environment.")
    data = attach_cf_snapshot(_base_graph(), _snapshot(), pd.Timestamp("2020-03-02"))
    fanouts = build_cf_num_neighbors(data.edge_types, num_layers=4, num_neighbors=8)
    loader = NeighborLoader(
        data,
        num_neighbors=fanouts,
        time_attr="time",
        input_nodes=("customer", torch.tensor([0])),
        input_time=torch.tensor([to_unix_time(pd.Series([pd.Timestamp("2020-03-02")]))[0]]),
        subgraph_type="bidirectional",
        batch_size=1,
        temporal_strategy="uniform",
    )
    batch = next(iter(loader))
    assert 1 in batch["article"].n_id.tolist()


def test_smoke_model_forward_backward_and_topk():
    data = attach_cf_snapshot(_base_graph(), _snapshot(), pd.Timestamp("2020-03-02"))
    for node_type in data.node_types:
        num_nodes = data[node_type].tf.num_rows
        data[node_type].n_id = torch.arange(num_nodes)
        data[node_type].batch = torch.zeros(num_nodes, dtype=torch.long)
    data["customer"].batch_size = 1
    data["customer"].input_id = torch.tensor([0])
    data["customer"].seed_time = torch.tensor(
        [to_unix_time(pd.Series([pd.Timestamp("2020-03-02")]))[0]]
    )
    col_stats = {node_type: article_cf_col_stats() for node_type in data.node_types}
    model = Model(
        data=data,
        col_stats_dict=col_stats,
        num_layers=4,
        channels=8,
        out_channels=1,
        aggr="sum",
        norm="layer_norm",
        id_awareness=True,
    )
    out = model.forward_dst_readout(data, "customer", "article").flatten()
    assert out.shape == (data["article"].tf.num_rows,)
    loss = F.binary_cross_entropy_with_logits(out, torch.zeros_like(out))
    assert torch.isfinite(loss)
    loss.backward()
    scores = torch.sigmoid(out.detach()).view(1, -1)
    _, pred = torch.topk(scores, k=2, dim=1)
    assert pred.shape == (1, 2)
