import numpy as np
import pandas as pd
import pytest
import torch
from torch_frame import stype
from torch_frame.data import Dataset
from torch_geometric.data import HeteroData

from examples.avito_cf.config import AvitoCFSnapshotConfig
from examples.avito_cf.coverage import compute_cf_coverage_for_table
from examples.avito_cf.graph import (
    CF_DST_TO_SRC,
    CF_EDGE_TYPES,
    CF_SRC_TO_DST,
    attach_cf_snapshot,
    build_cf_num_neighbors,
    build_cf_schema_template,
)
from examples.avito_cf.interactions import filter_visit_interactions
from examples.avito_cf.io import load_snapshot, write_snapshot_atomic
from examples.avito_cf.snapshot import build_snapshot
from relbench.base import Table


class DummyTask:
    time_col = "timestamp"
    src_entity_col = "UserID"
    src_entity_table = "UserInfo"
    dst_entity_col = "AdID"
    dst_entity_table = "AdsInfo"
    eval_k = 2


def _visits(rows):
    return pd.DataFrame(rows, columns=["UserID", "AdID", "ViewDate"]).astype(
        {"UserID": "int64", "AdID": "int64"}
    ).assign(ViewDate=lambda df: pd.to_datetime(df["ViewDate"]))


def _table(rows):
    return Table(
        df=pd.DataFrame(rows, columns=["timestamp", "UserID", "AdID"]).assign(
            timestamp=lambda df: pd.to_datetime(df["timestamp"])
        ),
        fkey_col_to_pkey_table={"UserID": "UserInfo", "AdID": "AdsInfo"},
        time_col="timestamp",
    )


def _snapshot(seed="2020-03-05"):
    return pd.DataFrame(
        {
            "seed_time": pd.to_datetime([seed, seed]),
            "src_AdID": pd.Series([0, 2], dtype="int64"),
            "dst_AdID": pd.Series([1, 0], dtype="int64"),
            "support": pd.Series([3, 3], dtype="int64"),
            "cf_score": pd.Series([0.7, 0.6], dtype="float32"),
            "rank": pd.Series([1, 1], dtype="int32"),
        }
    )


def _const_tensor_frame(num_rows: int):
    df = pd.DataFrame({"__const__": np.ones(num_rows, dtype=np.float32)})
    return Dataset(df=df, col_to_stype={"__const__": stype.numerical}).materialize()


def _base_graph(num_ads=4):
    data = HeteroData()
    data["AdsInfo"].tf = _const_tensor_frame(num_ads).tensor_frame
    data["AdsInfo"].time = torch.zeros(num_ads, dtype=torch.long)
    data["UserInfo"].tf = _const_tensor_frame(1).tensor_frame
    data["UserInfo"].time = torch.zeros(1, dtype=torch.long)
    return data


def test_visit_interaction_filtering():
    visits = pd.DataFrame(
        {
            "UserID": [0, 1, None],
            "AdID": [0, None, 2],
            "ViewDate": pd.to_datetime(["2020-03-01", "2020-03-02", "2020-03-03"]),
        }
    )
    out = filter_visit_interactions(visits)
    assert out[["UserID", "AdID"]].to_numpy().tolist() == [[0.0, 0.0]]


def test_avito_snapshot_window_and_columns(tmp_path):
    cfg = AvitoCFSnapshotConfig(min_support=1, top_l=10)
    interactions = filter_visit_interactions(
        _visits(
            [
                (0, 0, "2020-03-01"),
                (0, 1, "2020-03-02"),
                (0, 0, "2020-03-05"),
                (1, 1, "2020-03-06"),
            ]
        )
    )
    out, stats = build_snapshot(interactions, pd.Timestamp("2020-03-05"), 3, cfg)
    assert stats.num_raw_transactions == 2
    assert list(out.columns) == [
        "seed_time",
        "src_AdID",
        "dst_AdID",
        "support",
        "cf_score",
        "rank",
    ]
    assert set(map(tuple, out[["src_AdID", "dst_AdID"]].to_numpy())) == {
        (0, 1),
        (1, 0),
    }

    path = write_snapshot_atomic(out, tmp_path, pd.Timestamp("2020-03-05"), config=cfg)
    loaded = load_snapshot(tmp_path, pd.Timestamp("2020-03-05"), config=cfg)
    assert path.exists()
    assert loaded["cf_score"].dtype == "float32"


def test_direct_ad_cf_graph_edges_and_fanouts():
    data = attach_cf_snapshot(_base_graph(), _snapshot(), pd.Timestamp("2020-03-05"))
    assert data.node_types == ["AdsInfo", "UserInfo"]
    assert torch.equal(data[CF_SRC_TO_DST].edge_index, torch.tensor([[0, 2], [1, 0]]))
    assert torch.equal(data[CF_DST_TO_SRC].edge_index, torch.tensor([[0, 1], [2, 0]]))

    template = build_cf_schema_template(_base_graph())
    for edge_type in CF_EDGE_TYPES:
        assert edge_type in template.edge_types
    fanouts = build_cf_num_neighbors(template.edge_types, num_layers=4, num_neighbors=128)
    assert fanouts[CF_SRC_TO_DST] == [0, 0, 0, 0]
    assert fanouts[CF_DST_TO_SRC] == [0, 0, 32, 0]
    with pytest.raises(ValueError, match="num_layers"):
        build_cf_num_neighbors(template.edge_types, num_layers=2, num_neighbors=128)


def test_avito_cf_coverage_metrics(tmp_path):
    cfg = AvitoCFSnapshotConfig(min_support=1, top_l=10)
    seed = pd.Timestamp("2020-03-05")
    write_snapshot_atomic(
        pd.DataFrame(
            {
                "seed_time": pd.to_datetime([seed, seed]),
                "src_AdID": pd.Series([0, 2], dtype="int64"),
                "dst_AdID": pd.Series([1, 3], dtype="int64"),
                "support": pd.Series([1, 1], dtype="int64"),
                "cf_score": pd.Series([0.9, 0.8], dtype="float32"),
                "rank": pd.Series([1, 1], dtype="int32"),
            }
        ),
        tmp_path,
        seed,
        config=cfg,
        num_ads=4,
    )
    metrics = compute_cf_coverage_for_table(
        _table(
            [
                ("2020-03-05", 0, [1, 2]),
                ("2020-03-05", 1, [3]),
            ]
        ),
        DummyTask(),
        filter_visit_interactions(
            _visits(
                [
                    (0, 0, "2020-03-04"),
                    (1, 2, "2020-03-04"),
                ]
            )
        ),
        tmp_path,
        config=cfg,
        num_ads=4,
    )
    assert metrics.coverage_rate == pytest.approx(2 / 3)
    assert metrics.hit_rate == 1.0
    assert metrics.achievable_map == pytest.approx(0.75)


def test_avito_cf_coverage_num_layers_guard(tmp_path):
    cfg = AvitoCFSnapshotConfig(min_support=1, top_l=10)
    seed = pd.Timestamp("2020-03-05")
    write_snapshot_atomic(
        pd.DataFrame(
            {
                "seed_time": pd.Series([], dtype="datetime64[ns]"),
                "src_AdID": pd.Series([], dtype="int64"),
                "dst_AdID": pd.Series([], dtype="int64"),
                "support": pd.Series([], dtype="int64"),
                "cf_score": pd.Series([], dtype="float32"),
                "rank": pd.Series([], dtype="int32"),
            }
        ),
        tmp_path,
        seed,
        config=cfg,
        num_ads=3,
    )
    with pytest.raises(ValueError, match="num_layers"):
        compute_cf_coverage_for_table(
            _table([("2020-03-05", 0, [1])]),
            DummyTask(),
            filter_visit_interactions(_visits([(0, 1, "2020-03-04")])),
            tmp_path,
            config=cfg,
            num_ads=3,
            num_layers=2,
        )
