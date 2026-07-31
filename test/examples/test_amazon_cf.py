import pandas as pd
import pytest
import torch
from torch_geometric.data import HeteroData

from examples.amazon_cf.config import AmazonCFSnapshotConfig
from examples.amazon_cf.coverage import compute_cf_coverage_for_table
from examples.amazon_cf.graph import (
    CF_DST_F2P,
    CF_DST_REV,
    CF_EDGE_TYPES,
    CF_SRC_F2P,
    CF_SRC_REV,
    PRODUCT_CF,
    attach_cf_snapshot,
    build_cf_num_neighbors,
    build_cf_schema_template,
    make_product_cf_tensor_frame,
)
from examples.amazon_cf.interactions import filter_review_interactions
from examples.amazon_cf.io import load_snapshot, write_snapshot_atomic
from examples.amazon_cf.snapshot import build_snapshot
from relbench.base import Table
from relbench.modeling.utils import to_unix_time


class DummyTask:
    time_col = "timestamp"
    src_entity_col = "customer_id"
    src_entity_table = "customer"
    dst_entity_col = "product_id"
    dst_entity_table = "product"
    eval_k = 2


def _review(rows):
    return pd.DataFrame(
        rows,
        columns=[
            "customer_id",
            "product_id",
            "review_time",
            "rating",
            "review_text",
        ],
    ).astype({"customer_id": "int64", "product_id": "int64"}).assign(
        review_time=lambda df: pd.to_datetime(df["review_time"])
    )


def _table(rows):
    return Table(
        df=pd.DataFrame(rows, columns=["timestamp", "customer_id", "product_id"]).assign(
            timestamp=lambda df: pd.to_datetime(df["timestamp"])
        ),
        fkey_col_to_pkey_table={"customer_id": "customer", "product_id": "product"},
        time_col="timestamp",
    )


def _snapshot(seed="2020-03-02"):
    return pd.DataFrame(
        {
            "seed_time": pd.to_datetime([seed, seed]),
            "src_product_id": pd.Series([0, 2], dtype="int64"),
            "dst_product_id": pd.Series([1, 0], dtype="int64"),
            "support": pd.Series([3, 3], dtype="int64"),
            "cf_score": pd.Series([0.7, 0.6], dtype="float32"),
            "rank": pd.Series([1, 1], dtype="int32"),
        }
    )


def _base_graph(num_products=4):
    data = HeteroData()
    data["product"].tf = make_product_cf_tensor_frame(num_products).tensor_frame
    data["product"].time = torch.zeros(num_products, dtype=torch.long)
    data["customer"].tf = make_product_cf_tensor_frame(1).tensor_frame
    data["customer"].time = torch.tensor(
        [to_unix_time(pd.Series([pd.Timestamp("2020-03-02")]))[0]]
    )
    return data


def test_amazon_task_specific_review_filters():
    reviews = _review(
        [
            (0, 0, "2020-01-01", 5.0, "short"),
            (0, 1, "2020-01-02", 4.0, "x" * 301),
            (1, 2, "2020-01-03", 5.0, "x" * 301),
        ]
    )
    assert filter_review_interactions(reviews, "user-item-purchase")[
        "product_id"
    ].tolist() == [0, 1, 2]
    assert filter_review_interactions(reviews, "user-item-rate")[
        "product_id"
    ].tolist() == [0, 2]
    assert filter_review_interactions(reviews, "user-item-review")[
        "product_id"
    ].tolist() == [1, 2]


def test_amazon_snapshot_window_and_product_columns(tmp_path):
    cfg = AmazonCFSnapshotConfig(min_support=1, top_l=10)
    interactions = filter_review_interactions(
        _review(
            [
                (0, 0, "2019-12-02", 5.0, "a"),
                (0, 1, "2019-12-03", 5.0, "a"),
                (0, 0, "2020-03-02", 5.0, "a"),
                (1, 1, "2020-03-03", 5.0, "a"),
            ]
        ),
        "user-item-purchase",
    )
    out, stats = build_snapshot(interactions, pd.Timestamp("2020-03-02"), 3, cfg)
    assert stats.num_raw_transactions == 2
    assert list(out.columns) == [
        "seed_time",
        "src_product_id",
        "dst_product_id",
        "support",
        "cf_score",
        "rank",
    ]
    assert set(map(tuple, out[["src_product_id", "dst_product_id"]].to_numpy())) == {
        (0, 1),
        (1, 0),
    }

    path = write_snapshot_atomic(out, tmp_path, pd.Timestamp("2020-03-02"), config=cfg)
    loaded = load_snapshot(tmp_path, pd.Timestamp("2020-03-02"), config=cfg)
    assert path.exists()
    assert loaded["cf_score"].dtype == "float32"


def test_product_cf_graph_edges_and_fanouts():
    data = attach_cf_snapshot(_base_graph(), _snapshot(), pd.Timestamp("2020-03-02"))
    assert PRODUCT_CF in data.node_types
    assert torch.equal(data[CF_SRC_F2P].edge_index, torch.tensor([[0, 1], [0, 2]]))
    assert torch.equal(data[CF_SRC_REV].edge_index, torch.tensor([[0, 2], [0, 1]]))
    assert torch.equal(data[CF_DST_F2P].edge_index, torch.tensor([[0, 1], [1, 0]]))
    assert torch.equal(data[CF_DST_REV].edge_index, torch.tensor([[0, 1], [1, 0]]))

    template = build_cf_schema_template(_base_graph())
    for edge_type in CF_EDGE_TYPES:
        assert edge_type in template.edge_types
    fanouts = build_cf_num_neighbors(template.edge_types, num_layers=4, num_neighbors=128)
    assert fanouts[CF_SRC_F2P] == [0, 0, 32, 0]
    assert fanouts[CF_DST_REV] == [0, 0, 0, 16]
    with pytest.raises(ValueError, match="num_layers"):
        build_cf_num_neighbors(template.edge_types, num_layers=3, num_neighbors=128)


def test_amazon_cf_coverage_metrics(tmp_path):
    cfg = AmazonCFSnapshotConfig(min_support=1, top_l=10)
    seed = pd.Timestamp("2020-03-02")
    write_snapshot_atomic(
        pd.DataFrame(
            {
                "seed_time": pd.to_datetime([seed, seed]),
                "src_product_id": pd.Series([0, 2], dtype="int64"),
                "dst_product_id": pd.Series([1, 3], dtype="int64"),
                "support": pd.Series([1, 1], dtype="int64"),
                "cf_score": pd.Series([0.9, 0.8], dtype="float32"),
                "rank": pd.Series([1, 1], dtype="int32"),
            }
        ),
        tmp_path,
        seed,
        config=cfg,
        num_products=4,
    )
    metrics = compute_cf_coverage_for_table(
        _table(
            [
                ("2020-03-02", 0, [1, 2]),
                ("2020-03-02", 1, [3]),
            ]
        ),
        DummyTask(),
        filter_review_interactions(
            _review(
                [
                    (0, 0, "2020-03-01", 5.0, "a"),
                    (1, 2, "2020-03-01", 5.0, "a"),
                ]
            ),
            "user-item-purchase",
        ),
        tmp_path,
        config=cfg,
        num_products=4,
    )
    assert metrics.coverage_rate == pytest.approx(2 / 3)
    assert metrics.hit_rate == 1.0
    assert metrics.achievable_map == pytest.approx(0.75)
