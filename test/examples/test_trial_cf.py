from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch
from torch_geometric.data import HeteroData

from examples.trial_cf.config import TrialCFSnapshotConfig
from examples.trial_cf.coverage import compute_cf_coverage_for_table
from examples.trial_cf.graph import (
    attach_cf_snapshot,
    build_cf_num_neighbors,
    build_cf_schema_template,
    cf_dst_rev_edge_type,
    cf_src_edge_type,
    make_trial_cf_tensor_frame,
)
from examples.trial_cf.io import (
    initialize_or_load_manifest,
    load_snapshot,
    write_snapshot_atomic,
)
from examples.trial_cf.snapshot import build_snapshot
from relbench.base import Table


def _config(**kwargs) -> TrialCFSnapshotConfig:
    return TrialCFSnapshotConfig(
        history_days=None,
        all_history=True,
        min_support=1,
        top_l=2,
        **kwargs,
    )


def _interactions() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "src_id": pd.Series([0, 0, 0, 1], dtype="int64"),
            "sponsor_id": pd.Series([2, 1, 2, 3], dtype="int64"),
            "nct_id": ["n0", "n1", "n2", "n3"],
            "date": pd.to_datetime(
                ["2019-01-01", "2019-02-01", "2019-03-01", "2019-04-01"]
            ),
        }
    )


def _base_graph() -> HeteroData:
    data = HeteroData()
    data["conditions"].tf = make_trial_cf_tensor_frame(2).tensor_frame
    data["sponsors"].tf = make_trial_cf_tensor_frame(4).tensor_frame
    return data


def _snapshot() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "seed_time": pd.Series(
                pd.to_datetime(["2020-01-01", "2020-01-01"]),
                dtype="datetime64[ns]",
            ),
            "src_id": pd.Series([0, 0], dtype="int64"),
            "dst_sponsor_id": pd.Series([2, 1], dtype="int64"),
            "support": pd.Series([2, 1], dtype="int64"),
            "cf_score": pd.Series([0.8, 0.5], dtype="float32"),
            "rank": pd.Series([1, 2], dtype="int32"),
        }
    )


def test_trial_cf_snapshot_builds_ranked_source_sponsor_candidates():
    cfg = _config()
    out, stats = build_snapshot(
        _interactions(),
        pd.Timestamp("2020-01-01"),
        num_sources=2,
        num_sponsors=4,
        config=cfg,
    )

    src0 = out[out["src_id"] == 0]
    assert src0["dst_sponsor_id"].tolist() == [2, 1]
    assert src0["support"].tolist() == [2, 1]
    assert src0["rank"].tolist() == [1, 2]
    assert stats.num_active_sources == 2
    assert stats.num_unique_source_sponsor_pairs == 3


def test_trial_cf_io_manifest_and_snapshot_roundtrip(tmp_path):
    cfg = _config()
    snapshot_dir = cfg.snapshot_dir(tmp_path)
    manifest = initialize_or_load_manifest(snapshot_dir, cfg)

    assert manifest["cf_route"] == "source_sponsor_route_collapsed"
    assert manifest["src_entity_col"] == "condition_id"
    path = write_snapshot_atomic(
        _snapshot(),
        snapshot_dir,
        pd.Timestamp("2020-01-01"),
        config=cfg,
        num_sources=2,
        num_sponsors=4,
    )
    loaded = load_snapshot(
        snapshot_dir,
        pd.Timestamp("2020-01-01"),
        config=cfg,
        num_sources=2,
        num_sponsors=4,
    )

    assert path.name == "cf_snapshot_2020-01-01.parquet"
    assert loaded["cf_score"].dtype == "float32"


def test_trial_cf_graph_edges_and_two_hop_fanouts():
    cfg = _config()
    data = attach_cf_snapshot(
        _base_graph(),
        _snapshot(),
        pd.Timestamp("2020-01-01"),
        config=cfg,
    )

    assert data["trial_cf"].num_nodes == 2
    assert torch.equal(data[cf_src_edge_type(cfg)].edge_index[1], torch.tensor([0, 0]))
    assert torch.equal(
        data[cf_dst_rev_edge_type(cfg)].edge_index[0],
        torch.tensor([1, 2]),
    )

    template = build_cf_schema_template(_base_graph(), config=cfg)
    fanouts = build_cf_num_neighbors(
        template.edge_types,
        config=cfg,
        num_layers=2,
        num_neighbors=128,
    )
    assert fanouts[cf_src_edge_type(cfg)] == [128, 0]
    assert fanouts[cf_dst_rev_edge_type(cfg)] == [0, 64]
    with pytest.raises(ValueError, match="num_layers >= 2"):
        build_cf_num_neighbors(template.edge_types, config=cfg, num_layers=1, num_neighbors=8)


def test_trial_cf_coverage_metrics(tmp_path):
    cfg = _config()
    snapshot_dir = cfg.snapshot_dir(tmp_path)
    initialize_or_load_manifest(snapshot_dir, cfg)
    write_snapshot_atomic(
        _snapshot(),
        snapshot_dir,
        pd.Timestamp("2020-01-01"),
        config=cfg,
        num_sources=2,
        num_sponsors=4,
    )
    table = Table(
        df=pd.DataFrame(
            {
                "timestamp": pd.to_datetime(["2020-01-01", "2020-01-01"]),
                "condition_id": pd.Series([0, 1], dtype="int64"),
                "sponsor_id": [[2, 3], [3]],
            }
        ),
        fkey_col_to_pkey_table={"condition_id": "conditions", "sponsor_id": "sponsors"},
        pkey_col=None,
        time_col="timestamp",
    )

    class Task:
        src_entity_col = "condition_id"
        src_entity_table = "conditions"
        dst_entity_col = "sponsor_id"
        dst_entity_table = "sponsors"
        time_col = "timestamp"
        eval_k = 10

    metrics = compute_cf_coverage_for_table(
        table,
        Task(),
        snapshot_dir,
        config=cfg,
        num_sources=2,
        num_sponsors=4,
    )

    assert metrics.num_rows == 2
    assert metrics.num_rows_with_candidates == 1
    assert metrics.num_rows_with_hit == 1
    assert metrics.num_groundtruth_labels == 3
    assert metrics.num_covered_groundtruth_labels == 1
    assert np.isclose(metrics.coverage_rate, 1 / 3)

