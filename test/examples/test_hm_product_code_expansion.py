import pandas as pd
import pytest
import torch
from torch_frame import stype
from torch_frame.data import Dataset
from torch_geometric.data import HeteroData

from examples.hm_product_code_expansion import (
    ARTICLE_EDGE,
    CUSTOMER_EDGE,
    REV_ARTICLE_EDGE,
    REV_CUSTOMER_EDGE,
    augment_hm_graph_with_virtual_transactions,
    build_product_code_virtual_candidates,
)
from relbench.base import Database, Table


def _make_db() -> Database:
    article_df = pd.DataFrame(
        {
            "article_id": [0, 1, 2, 3, 4, 5],
            "product_code": ["pc10", "pc10", "pc10", "pc20", "pc20", None],
        }
    )
    customer_df = pd.DataFrame({"customer_id": [0, 1, 2]})
    transactions_df = pd.DataFrame(
        {
            "customer_id": [0, 0, 0, 0, 0, 0, 1, 1, 2],
            "article_id": [0, 1, 0, 3, 4, 5, 0, 3, 0],
            "price": [10.0, 11.0, 12.0, 20.0, 21.0, 99.0, 30.0, 40.0, 50.0],
            "t_dat": pd.to_datetime(
                [
                    "2020-01-01",
                    "2020-01-05",
                    "2020-01-05",
                    "2020-01-03",
                    "2020-01-11",
                    "2020-01-08",
                    "2020-01-02",
                    "2020-01-02",
                    "2020-01-01",
                ]
            ),
            "sales_channel_id": [1, 1, 2, 1, 2, 1, 1, 2, 1],
        }
    )
    return Database(
        table_dict={
            "article": Table(article_df, {}, pkey_col="article_id"),
            "customer": Table(customer_df, {}, pkey_col="customer_id"),
            "transactions": Table(
                transactions_df,
                {"customer_id": "customer", "article_id": "article"},
                time_col="t_dat",
            ),
        }
    )


def _make_test_table(article_labels=None) -> Table:
    if article_labels is None:
        article_labels = [[999], [998]]
    return Table(
        pd.DataFrame(
            {
                "timestamp": pd.to_datetime(["2020-01-10", "2020-01-10"]),
                "customer_id": [0, 1],
                "article_id": article_labels,
            }
        ),
        {"customer_id": "customer", "article_id": "article"},
        time_col="timestamp",
    )


def test_build_product_code_virtual_candidates_is_deterministic_and_label_free():
    db = _make_db()
    candidates, stats = build_product_code_virtual_candidates(db, _make_test_table())

    expected = pd.DataFrame(
        {
            "customer_id": [0, 0, 1, 1, 1],
            "source_article_id": [0, 3, 3, 0, 0],
            "target_article_id": [2, 4, 4, 1, 2],
            "product_code": ["pc10", "pc20", "pc20", "pc10", "pc10"],
            "source_tx_pos": [2, 3, 7, 6, 6],
            "source_t_dat": pd.to_datetime(
                [
                    "2020-01-05",
                    "2020-01-03",
                    "2020-01-02",
                    "2020-01-02",
                    "2020-01-02",
                ]
            ),
        }
    )
    pd.testing.assert_frame_equal(candidates, expected)

    relabeled, _ = build_product_code_virtual_candidates(
        db, _make_test_table(article_labels=[[0, 1, 2], [3, 4]])
    )
    pd.testing.assert_frame_equal(relabeled, candidates)

    assert set(candidates["customer_id"]) == {0, 1}
    assert (candidates["source_t_dat"] <= pd.Timestamp("2020-01-10")).all()
    assert not (
        candidates["source_article_id"] == candidates["target_article_id"]
    ).any()
    assert not candidates.duplicated(["customer_id", "target_article_id"]).any()
    assert stats.num_test_customers == 2
    assert stats.num_historical_transactions == 7
    assert stats.num_selected_source_transactions == 4
    assert stats.num_candidates_before_seen_filter == 6
    assert stats.num_candidates_after_seen_filter == 5
    assert stats.num_synthetic_transactions == 5
    assert stats.num_customers_with_synthetic_transactions == 2
    assert stats.max_synthetic_per_customer == 3


def test_build_product_code_virtual_candidates_cap_zero_unlimited_and_positive_cap():
    db = _make_db()
    unlimited, _ = build_product_code_virtual_candidates(
        db, _make_test_table(), max_virtual_per_customer=0
    )
    capped, _ = build_product_code_virtual_candidates(
        db, _make_test_table(), max_virtual_per_customer=1
    )

    assert len(unlimited) == 5
    pd.testing.assert_frame_equal(
        capped,
        unlimited.groupby("customer_id", sort=False).head(1).reset_index(drop=True),
    )
    assert capped["target_article_id"].tolist() == [2, 4]


def _make_graph() -> HeteroData:
    tx_df = pd.DataFrame(
        {
            "price": [1.0, 2.0, 3.0],
            "sales_channel_id": ["store", "online", "store"],
            "t_dat": pd.to_datetime(["2020-01-01", "2020-01-02", "2020-01-03"]),
        }
    )
    tx_tf = Dataset(
        tx_df,
        {
            "price": stype.numerical,
            "sales_channel_id": stype.categorical,
            "t_dat": stype.timestamp,
        },
    ).materialize().tensor_frame

    data = HeteroData()
    data["customer"].num_nodes = 2
    data["article"].num_nodes = 5
    data["transactions"].tf = tx_tf
    data["transactions"].time = torch.tensor([10, 20, 30])
    data["transactions"].num_nodes = len(tx_tf)

    customer = torch.tensor([0, 0, 1])
    article = torch.tensor([0, 1, 3])
    tx = torch.arange(3)
    data[CUSTOMER_EDGE].edge_index = torch.stack([tx, customer])
    data[REV_CUSTOMER_EDGE].edge_index = torch.stack([customer, tx])
    data[ARTICLE_EDGE].edge_index = torch.stack([tx, article])
    data[REV_ARTICLE_EDGE].edge_index = torch.stack([article, tx])
    data[("customer", "self", "customer")].edge_index = torch.tensor(
        [[0, 1], [0, 1]]
    )
    data.validate()
    return data


def _assert_tf_rows_equal(tf, left: torch.Tensor, right: torch.Tensor) -> None:
    for feat in tf.feat_dict.values():
        assert torch.equal(feat[left], feat[right])


def test_augment_hm_graph_with_virtual_transactions_clones_rows_and_edges():
    data = _make_graph()
    base_edge_index = {
        edge_type: data[edge_type].edge_index.clone() for edge_type in data.edge_types
    }
    candidates = pd.DataFrame(
        {
            "customer_id": [0, 1],
            "source_article_id": [0, 3],
            "target_article_id": [2, 4],
            "product_code": ["pc10", "pc20"],
            "source_tx_pos": [0, 2],
            "source_t_dat": pd.to_datetime(["2020-01-01", "2020-01-03"]),
        }
    )

    augmented = augment_hm_graph_with_virtual_transactions(data, candidates)

    assert augmented is not data
    assert set(augmented.node_types) == set(data.node_types)
    assert set(augmented.edge_types) == set(data.edge_types)
    assert len(data["transactions"].tf) == 3
    assert data["transactions"].time.tolist() == [10, 20, 30]
    for edge_type, edge_index in base_edge_index.items():
        assert torch.equal(data[edge_type].edge_index, edge_index)

    assert len(augmented["transactions"].tf) == 5
    assert augmented["transactions"].time.tolist() == [10, 20, 30, 10, 30]
    _assert_tf_rows_equal(
        augmented["transactions"].tf,
        torch.tensor([0, 2]),
        torch.tensor([3, 4]),
    )

    for edge_type in [CUSTOMER_EDGE, REV_CUSTOMER_EDGE, ARTICLE_EDGE, REV_ARTICLE_EDGE]:
        assert augmented[edge_type].edge_index.size(1) == (
            data[edge_type].edge_index.size(1) + 2
        )

    assert _has_edge(augmented[CUSTOMER_EDGE].edge_index, 3, 0)
    assert _has_edge(augmented[CUSTOMER_EDGE].edge_index, 4, 1)
    assert _has_edge(augmented[ARTICLE_EDGE].edge_index, 3, 2)
    assert _has_edge(augmented[ARTICLE_EDGE].edge_index, 4, 4)
    assert torch.equal(
        augmented[("customer", "self", "customer")].edge_index,
        data[("customer", "self", "customer")].edge_index,
    )
    augmented.validate()


def test_augment_hm_graph_with_empty_candidates_returns_valid_separate_graph():
    data = _make_graph()
    augmented = augment_hm_graph_with_virtual_transactions(
        data,
        pd.DataFrame(columns=[
            "customer_id",
            "source_article_id",
            "target_article_id",
            "product_code",
            "source_tx_pos",
            "source_t_dat",
        ]),
    )

    assert augmented is not data
    assert len(augmented["transactions"].tf) == len(data["transactions"].tf)
    assert augmented["transactions"].time.numel() == data["transactions"].time.numel()
    for edge_type in data.edge_types:
        assert augmented[edge_type].edge_index.size(1) == data[edge_type].edge_index.size(1)
    augmented.validate()


def test_candidate_generation_errors():
    db = _make_db()
    multi_seed_table = Table(
        pd.DataFrame(
            {
                "timestamp": pd.to_datetime(["2020-01-10", "2020-01-11"]),
                "customer_id": [0, 1],
                "article_id": [[1], [2]],
            }
        ),
        {"customer_id": "customer", "article_id": "article"},
        time_col="timestamp",
    )
    with pytest.raises(ValueError, match="exactly one unique"):
        build_product_code_virtual_candidates(db, multi_seed_table)

    with pytest.raises(ValueError, match="non-negative"):
        build_product_code_virtual_candidates(
            db, _make_test_table(), max_virtual_per_customer=-1
        )

    missing_table = Database({"transactions": db.table_dict["transactions"]})
    with pytest.raises(ValueError, match="Required table 'article'"):
        build_product_code_virtual_candidates(missing_table, _make_test_table())

    missing_col = _make_db()
    missing_col.table_dict["article"].df = missing_col.table_dict["article"].df.drop(
        columns=["product_code"]
    )
    with pytest.raises(ValueError, match="missing required columns"):
        build_product_code_virtual_candidates(missing_col, _make_test_table())


def test_graph_augmentation_rejects_out_of_range_source_position():
    data = _make_graph()
    candidates = pd.DataFrame(
        {
            "customer_id": [0],
            "source_article_id": [0],
            "target_article_id": [2],
            "product_code": ["pc10"],
            "source_tx_pos": [99],
            "source_t_dat": pd.to_datetime(["2020-01-01"]),
        }
    )
    with pytest.raises(ValueError, match="source_tx_pos"):
        augment_hm_graph_with_virtual_transactions(data, candidates)


def _has_edge(edge_index: torch.Tensor, src: int, dst: int) -> bool:
    return bool(((edge_index[0] == src) & (edge_index[1] == dst)).any())
