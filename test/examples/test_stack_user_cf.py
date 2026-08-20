from __future__ import annotations

import math

import pandas as pd
import pytest
import torch
from torch_geometric.data import HeteroData

from examples.stack_cf.interactions import filter_comment_interactions
from examples.stack_user_cf.config import StackUserCFSnapshotConfig, snapshot_filename
from examples.stack_user_cf.coverage import compute_user_cf_coverage_for_table
from examples.stack_user_cf.graph import (
    USER_CF,
    USER_CF_DST_F2P,
    USER_CF_DST_REV,
    USER_CF_EDGE_TYPES,
    USER_CF_SRC_F2P,
    USER_CF_SRC_REV,
    attach_user_cf_snapshot,
    build_user_cf_num_neighbors,
    build_user_cf_schema_template,
    make_user_cf_tensor_frame,
)
from examples.stack_user_cf.io import (
    initialize_or_load_manifest,
    load_snapshot,
    source_users_by_seed_time,
    write_snapshot_atomic,
)
from examples.stack_user_cf.snapshot import build_snapshot
from relbench.base import Table


class DummyTask:
    time_col = "timestamp"
    src_entity_col = "UserId"
    src_entity_table = "users"
    dst_entity_col = "PostId"
    dst_entity_table = "posts"
    eval_k = 2

    def __init__(self, tables=None):
        self.tables = tables or {}

    def get_table(self, split, *args, **kwargs):
        return self.tables[split]


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
        pkey_col=None,
        time_col="timestamp",
    )


def _snapshot(seed="2020-03-05"):
    return pd.DataFrame(
        {
            "seed_time": pd.to_datetime([seed, seed]),
            "src_UserId": pd.Series([0, 2], dtype="int64"),
            "dst_UserId": pd.Series([1, 0], dtype="int64"),
            "overlap": pd.Series([2, 2], dtype="int64"),
            "user_cf_score": pd.Series([1.0, 0.8], dtype="float32"),
            "src_history_size": pd.Series([2, 2], dtype="int64"),
            "dst_history_size": pd.Series([2, 2], dtype="int64"),
            "rank": pd.Series([1, 1], dtype="int32"),
        }
    )


def _base_graph(num_users=3, num_posts=3):
    data = HeteroData()
    data["users"].tf = make_user_cf_tensor_frame(num_users).tensor_frame
    data["users"].time = torch.zeros(num_users, dtype=torch.long)
    data["posts"].tf = make_user_cf_tensor_frame(num_posts).tensor_frame
    data["posts"].time = torch.zeros(num_posts, dtype=torch.long)
    data["comments"].tf = make_user_cf_tensor_frame(2).tensor_frame
    data["comments"].time = torch.zeros(2, dtype=torch.long)
    data[("comments", "f2p_UserId", "users")].edge_index = torch.tensor(
        [[0, 1], [1, 2]], dtype=torch.long
    )
    data[("users", "rev_f2p_UserId", "comments")].edge_index = torch.tensor(
        [[1, 2], [0, 1]], dtype=torch.long
    )
    data[("comments", "f2p_PostId", "posts")].edge_index = torch.tensor(
        [[0, 1], [0, 2]], dtype=torch.long
    )
    data[("posts", "rev_f2p_PostId", "comments")].edge_index = torch.tensor(
        [[0, 2], [0, 1]], dtype=torch.long
    )
    return data


def test_stack_user_cf_snapshot_window_columns_and_score():
    cfg = StackUserCFSnapshotConfig(min_overlap=1, top_k=10)
    interactions = filter_comment_interactions(
        _comments(
            [
                (0, 0, "2020-03-01"),
                (0, 1, "2020-03-02"),
                (0, 0, "2020-03-05"),
                (1, 0, "2020-03-03"),
                (1, 1, "2020-03-04"),
                (2, 1, "2020-03-04"),
                (1, 2, "2020-03-06"),
            ]
        )
    )
    out, stats = build_snapshot(
        interactions,
        pd.Timestamp("2020-03-05"),
        3,
        3,
        [0],
        cfg,
    )
    assert stats.num_raw_transactions == 6
    assert list(out.columns) == [
        "seed_time",
        "src_UserId",
        "dst_UserId",
        "overlap",
        "user_cf_score",
        "src_history_size",
        "dst_history_size",
        "rank",
    ]
    assert out["dst_UserId"].tolist() == [1, 2]
    score = out.set_index("dst_UserId")["user_cf_score"]
    assert math.isclose(score[1], 2 / math.sqrt(2 * 2), rel_tol=1e-6)
    assert math.isclose(score[2], 1 / math.sqrt(2 * 1), rel_tol=1e-6)


def test_stack_user_cf_io_manifest_and_source_users(tmp_path):
    cfg = StackUserCFSnapshotConfig(min_overlap=1, top_k=2)
    snapshot_dir = cfg.snapshot_dir(tmp_path)
    manifest = initialize_or_load_manifest(snapshot_dir, cfg)
    assert manifest["cf_node_type"] == "user_cf"
    assert snapshot_filename(pd.Timestamp("2020-03-05")) == "user_cf_snapshot_2020-03-05.parquet"

    path = write_snapshot_atomic(
        _snapshot(),
        snapshot_dir,
        pd.Timestamp("2020-03-05"),
        config=cfg,
        num_users=3,
    )
    loaded = load_snapshot(
        snapshot_dir,
        pd.Timestamp("2020-03-05"),
        config=cfg,
        num_users=3,
    )
    assert path.exists()
    assert loaded["user_cf_score"].dtype == "float32"

    task = DummyTask(
        {
            "train": _table(
                [
                    ("2020-03-05", 2, [0]),
                    ("2020-03-05", 1, [1]),
                    ("2020-03-06", 2, [2]),
                ]
            )
        }
    )
    source_map = source_users_by_seed_time(task, ["train"])
    assert source_map[pd.Timestamp("2020-03-05")].tolist() == [1, 2]


def test_stack_user_cf_graph_edges_and_fanouts():
    data = attach_user_cf_snapshot(_base_graph(), _snapshot(), pd.Timestamp("2020-03-05"))
    assert USER_CF in data.node_types
    assert torch.equal(data[USER_CF_SRC_F2P].edge_index, torch.tensor([[0, 1], [0, 2]]))
    assert torch.equal(data[USER_CF_SRC_REV].edge_index, torch.tensor([[0, 2], [0, 1]]))
    assert torch.equal(data[USER_CF_DST_F2P].edge_index, torch.tensor([[0, 1], [1, 0]]))
    assert torch.equal(data[USER_CF_DST_REV].edge_index, torch.tensor([[0, 1], [1, 0]]))

    template = build_user_cf_schema_template(_base_graph())
    for edge_type in USER_CF_EDGE_TYPES:
        assert edge_type in template.edge_types
    fanouts = build_user_cf_num_neighbors(template.edge_types, num_layers=4, num_neighbors=128)
    assert fanouts[USER_CF_SRC_F2P] == [128, 0, 0, 0]
    assert fanouts[USER_CF_DST_REV] == [0, 64, 0, 0]
    assert fanouts[USER_CF_SRC_REV] == [0, 0, 0, 0]
    assert fanouts[USER_CF_DST_F2P] == [0, 0, 0, 0]
    with pytest.raises(ValueError, match="num_layers"):
        build_user_cf_num_neighbors(template.edge_types, num_layers=3, num_neighbors=128)


def test_stack_user_cf_coverage_metrics(tmp_path):
    cfg = StackUserCFSnapshotConfig(min_overlap=1, top_k=2)
    seed = pd.Timestamp("2020-03-05")
    write_snapshot_atomic(
        pd.DataFrame(
            {
                "seed_time": pd.to_datetime([seed, seed]),
                "src_UserId": pd.Series([0, 0], dtype="int64"),
                "dst_UserId": pd.Series([1, 2], dtype="int64"),
                "overlap": pd.Series([1, 1], dtype="int64"),
                "user_cf_score": pd.Series([0.9, 0.8], dtype="float32"),
                "src_history_size": pd.Series([1, 1], dtype="int64"),
                "dst_history_size": pd.Series([1, 1], dtype="int64"),
                "rank": pd.Series([1, 2], dtype="int32"),
            }
        ),
        tmp_path,
        seed,
        config=cfg,
        num_users=3,
    )
    metrics = compute_user_cf_coverage_for_table(
        _table(
            [
                ("2020-03-05", 0, [3, 4]),
                ("2020-03-05", 1, [4]),
            ]
        ),
        DummyTask(),
        filter_comment_interactions(
            _comments(
                [
                    (1, 3, "2020-03-04"),
                    (2, 4, "2020-03-04"),
                    (2, 5, "2020-03-06"),
                ]
            )
        ),
        tmp_path,
        config=cfg,
        num_users=3,
    )
    assert metrics.num_rows == 2
    assert metrics.num_rows_with_neighbors == 1
    assert metrics.num_rows_with_candidates == 1
    assert metrics.num_rows_with_hit == 1
    assert metrics.coverage_rate == pytest.approx(2 / 3)
    assert metrics.hit_rate == pytest.approx(1 / 2)
    assert metrics.achievable_map == pytest.approx(0.5)
