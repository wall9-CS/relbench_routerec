import numpy as np
import pandas as pd
import pytest
import torch
import torch.nn.functional as F
from torch_frame import stype
from torch_frame.data import Dataset
from torch_geometric.data import HeteroData
from torch_geometric.loader import NeighborLoader
from torch_geometric.typing import WITH_PYG_LIB

from examples.model import Model
from examples.hm_cf.graph import (
    CF_DST_TO_SRC,
    CF_EDGE_TYPES,
    CF_SRC_TO_DST,
    attach_cf_snapshot,
    build_cf_num_neighbors,
    build_cf_schema_template,
)
from relbench.modeling.utils import to_unix_time


def _const_tensor_frame(num_rows: int):
    df = pd.DataFrame({"__const__": np.ones(num_rows, dtype=np.float32)})
    return Dataset(df=df, col_to_stype={"__const__": stype.numerical}).materialize()


def _base_graph(num_articles=4):
    data = HeteroData()
    data["article"].tf = _const_tensor_frame(num_articles).tensor_frame
    data["article"].time = torch.zeros(num_articles, dtype=torch.long)
    data["customer"].tf = _const_tensor_frame(1).tensor_frame
    data["customer"].time = torch.tensor(
        [to_unix_time(pd.Series([pd.Timestamp("2020-03-02")]))[0]]
    )
    data["transactions"].tf = _const_tensor_frame(1).tensor_frame
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


def test_snapshot_columns_are_not_model_features():
    data = attach_cf_snapshot(_base_graph(), _snapshot(), pd.Timestamp("2020-03-02"))
    assert data.node_types == ["article", "customer", "transactions"]
    for edge_type in CF_EDGE_TYPES:
        store = data[edge_type]
        assert "edge_attr" not in store
        assert "support" not in store
        assert "cf_score" not in store
        assert "rank" not in store


def test_base_graph_immutability():
    base = _base_graph()
    original_node_types = list(base.node_types)
    original_edge_types = list(base.edge_types)
    attached = attach_cf_snapshot(base, _snapshot(), pd.Timestamp("2020-03-02"))
    assert base.node_types == original_node_types
    assert base.edge_types == original_edge_types
    assert attached.node_types == original_node_types
    assert CF_SRC_TO_DST in attached.edge_types
    assert CF_DST_TO_SRC in attached.edge_types
    assert CF_SRC_TO_DST not in base.edge_types
    assert CF_DST_TO_SRC not in base.edge_types


def test_graph_edge_roles_exact():
    data = attach_cf_snapshot(_base_graph(), _snapshot(), pd.Timestamp("2020-03-02"))
    assert torch.equal(data[CF_SRC_TO_DST].edge_index, torch.tensor([[0, 2], [1, 0]]))
    assert torch.equal(data[CF_DST_TO_SRC].edge_index, torch.tensor([[0, 1], [2, 0]]))


def test_schema_template_and_fanouts():
    template = build_cf_schema_template(_base_graph())
    for edge_type in CF_EDGE_TYPES:
        assert edge_type in template.edge_types
        assert template[edge_type].edge_index.numel() == 0
    fanouts = build_cf_num_neighbors(
        template.edge_types, num_layers=4, num_neighbors=128
    )
    assert fanouts[CF_SRC_TO_DST] == [0, 0, 0, 0]
    assert fanouts[CF_DST_TO_SRC] == [0, 0, 32, 0]
    with pytest.raises(ValueError, match="num_layers"):
        build_cf_num_neighbors(template.edge_types, num_layers=2, num_neighbors=128)


def test_seed_time_isolation_by_exact_attachment():
    base = _base_graph()
    t1 = pd.Timestamp("2020-03-02")
    t2 = pd.Timestamp("2020-03-09")
    g1 = attach_cf_snapshot(base, _snapshot(t1), t1)
    snap2 = _snapshot(t2)
    snap2["dst_article_id"] = [3, 1]
    g2 = attach_cf_snapshot(base, snap2, t2)
    assert set(g1[CF_SRC_TO_DST].edge_index[1].tolist()) == {0, 1}
    assert set(g2[CF_SRC_TO_DST].edge_index[1].tolist()) == {1, 3}


def test_direct_three_hop_reachability_route_with_static_graph():
    data = attach_cf_snapshot(_base_graph(), _snapshot(), pd.Timestamp("2020-03-02"))
    tx_to_customer = data[("transactions", "f2p_customer_id", "customer")].edge_index
    tx = tx_to_customer[0, tx_to_customer[1] == 0]
    article_src = data[("transactions", "f2p_article_id", "article")].edge_index[1]
    assert article_src[tx].tolist() == [0]
    candidates = data[CF_DST_TO_SRC].edge_index[0][
        data[CF_DST_TO_SRC].edge_index[1] == article_src[tx].item()
    ]
    assert candidates.tolist() == [1]
    assert not torch.isin(torch.tensor([2]), candidates).any()


def test_neighbor_loader_route_skips_without_temporal_backend():
    if not WITH_PYG_LIB:
        pytest.skip("PyG temporal NeighborLoader requires pyg-lib in this environment.")
    data = attach_cf_snapshot(_base_graph(), _snapshot(), pd.Timestamp("2020-03-02"))
    fanouts = build_cf_num_neighbors(data.edge_types, num_layers=3, num_neighbors=8)
    loader = NeighborLoader(
        data,
        num_neighbors=fanouts,
        time_attr="time",
        input_nodes=("customer", torch.tensor([0])),
        input_time=torch.tensor(
            [to_unix_time(pd.Series([pd.Timestamp("2020-03-02")]))[0]]
        ),
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
    col_stats = {
        node_type: _const_tensor_frame(1).col_stats for node_type in data.node_types
    }
    model = Model(
        data=data,
        col_stats_dict=col_stats,
        num_layers=3,
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
