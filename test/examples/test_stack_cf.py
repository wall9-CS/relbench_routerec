import pandas as pd
import pytest
import torch
from torch_geometric.data import HeteroData

from examples.stack_cf.config import StackCFSnapshotConfig
from examples.stack_cf.coverage import compute_cf_coverage_for_table
from examples.stack_cf.graph import (
    CF_DST_F2P,
    CF_DST_REV,
    CF_EDGE_TYPES,
    CF_SRC_F2P,
    CF_SRC_REV,
    POST_CF,
    attach_cf_snapshot,
    build_cf_num_neighbors,
    build_cf_schema_template,
    make_post_cf_tensor_frame,
)
from examples.stack_cf.interactions import filter_comment_interactions
from examples.stack_cf.io import load_snapshot, write_snapshot_atomic
from examples.stack_cf.snapshot import build_snapshot
from relbench.base import Table


class DummyTask:
    time_col = "timestamp"
    src_entity_col = "UserId"
    src_entity_table = "users"
    dst_entity_col = "PostId"
    dst_entity_table = "posts"
    eval_k = 2


def _comments(rows):
    return pd.DataFrame(rows, columns=["UserId", "PostId", "CreationDate"]).astype(
        {"UserId": "int64", "PostId": "int64"}
    ).assign(CreationDate=lambda df: pd.to_datetime(df["CreationDate"]))


def _table(rows):
    return Table(
        df=pd.DataFrame(rows, columns=["timestamp", "UserId", "PostId"]).assign(
            timestamp=lambda df: pd.to_datetime(df["timestamp"])
        ),
        fkey_col_to_pkey_table={"UserId": "users", "PostId": "posts"},
        time_col="timestamp",
    )


def _snapshot(seed="2020-03-05"):
    return pd.DataFrame(
        {
            "seed_time": pd.to_datetime([seed, seed]),
            "src_PostId": pd.Series([0, 2], dtype="int64"),
            "dst_PostId": pd.Series([1, 0], dtype="int64"),
            "support": pd.Series([3, 3], dtype="int64"),
            "cf_score": pd.Series([0.7, 0.6], dtype="float32"),
            "rank": pd.Series([1, 1], dtype="int32"),
        }
    )


def _base_graph(num_posts=4):
    data = HeteroData()
    data["posts"].tf = make_post_cf_tensor_frame(num_posts).tensor_frame
    data["posts"].time = torch.zeros(num_posts, dtype=torch.long)
    data["users"].tf = make_post_cf_tensor_frame(1).tensor_frame
    data["users"].time = torch.zeros(1, dtype=torch.long)
    return data


def test_comment_interaction_filtering():
    comments = pd.DataFrame(
        {
            "UserId": [0, 1, None],
            "PostId": [0, None, 2],
            "CreationDate": pd.to_datetime(
                ["2020-03-01", "2020-03-02", "2020-03-03"]
            ),
        }
    )
    out = filter_comment_interactions(comments)
    assert out[["UserId", "PostId"]].to_numpy().tolist() == [[0.0, 0.0]]


def test_stack_snapshot_window_and_columns(tmp_path):
    cfg = StackCFSnapshotConfig(min_support=1, top_l=10)
    interactions = filter_comment_interactions(
        _comments(
            [
                (0, 0, "2019-12-05"),
                (0, 1, "2019-12-06"),
                (0, 0, "2020-03-05"),
                (1, 1, "2020-03-06"),
            ]
        )
    )
    out, stats = build_snapshot(interactions, pd.Timestamp("2020-03-05"), 3, cfg)
    assert stats.num_raw_transactions == 2
    assert list(out.columns) == [
        "seed_time",
        "src_PostId",
        "dst_PostId",
        "support",
        "cf_score",
        "rank",
    ]
    assert set(map(tuple, out[["src_PostId", "dst_PostId"]].to_numpy())) == {
        (0, 1),
        (1, 0),
    }

    path = write_snapshot_atomic(out, tmp_path, pd.Timestamp("2020-03-05"), config=cfg)
    loaded = load_snapshot(tmp_path, pd.Timestamp("2020-03-05"), config=cfg)
    assert path.exists()
    assert loaded["cf_score"].dtype == "float32"


def test_post_cf_graph_edges_and_fanouts():
    data = attach_cf_snapshot(_base_graph(), _snapshot(), pd.Timestamp("2020-03-05"))
    assert POST_CF in data.node_types
    assert torch.equal(data[CF_SRC_F2P].edge_index, torch.tensor([[0, 1], [0, 2]]))
    assert torch.equal(data[CF_SRC_REV].edge_index, torch.tensor([[0, 2], [0, 1]]))
    assert torch.equal(data[CF_DST_F2P].edge_index, torch.tensor([[0, 1], [1, 0]]))
    assert torch.equal(data[CF_DST_REV].edge_index, torch.tensor([[0, 1], [1, 0]]))

    template = build_cf_schema_template(_base_graph())
    for edge_type in CF_EDGE_TYPES:
        assert edge_type in template.edge_types
    fanouts = build_cf_num_neighbors(
        template.edge_types, num_layers=4, num_neighbors=128
    )
    assert fanouts[CF_SRC_F2P] == [0, 0, 32, 0]
    assert fanouts[CF_DST_REV] == [0, 0, 0, 16]
    with pytest.raises(ValueError, match="num_layers"):
        build_cf_num_neighbors(template.edge_types, num_layers=3, num_neighbors=128)


def test_stack_cf_coverage_metrics(tmp_path):
    cfg = StackCFSnapshotConfig(min_support=1, top_l=10)
    seed = pd.Timestamp("2020-03-05")
    write_snapshot_atomic(
        pd.DataFrame(
            {
                "seed_time": pd.to_datetime([seed, seed]),
                "src_PostId": pd.Series([0, 2], dtype="int64"),
                "dst_PostId": pd.Series([1, 3], dtype="int64"),
                "support": pd.Series([1, 1], dtype="int64"),
                "cf_score": pd.Series([0.9, 0.8], dtype="float32"),
                "rank": pd.Series([1, 1], dtype="int32"),
            }
        ),
        tmp_path,
        seed,
        config=cfg,
        num_posts=4,
    )
    metrics = compute_cf_coverage_for_table(
        _table(
            [
                ("2020-03-05", 0, [1, 2]),
                ("2020-03-05", 1, [3]),
            ]
        ),
        DummyTask(),
        filter_comment_interactions(
            _comments(
                [
                    (0, 0, "2020-03-04"),
                    (1, 2, "2020-03-04"),
                ]
            )
        ),
        tmp_path,
        config=cfg,
        num_posts=4,
    )
    assert metrics.coverage_rate == pytest.approx(2 / 3)
    assert metrics.hit_rate == 1.0
    assert metrics.achievable_map == pytest.approx(0.75)
