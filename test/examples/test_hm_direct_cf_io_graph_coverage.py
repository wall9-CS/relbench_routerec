import json

import pandas as pd
import pytest
import torch
from torch_geometric.data import HeteroData

from examples.hm_direct_cf.config import HMDirectCFSnapshotConfig, snapshot_filename
from examples.hm_direct_cf.coverage import compute_direct_cf_coverage_for_table
from examples.hm_direct_cf.graph import (
    DIRECT_ARTICLE_CF,
    DIRECT_CF_DST_F2P,
    DIRECT_CF_DST_REV,
    DIRECT_CF_EDGE_TYPES,
    DIRECT_CF_SRC_F2P,
    DIRECT_CF_SRC_REV,
    attach_direct_cf_snapshot,
    build_direct_cf_num_neighbors,
    build_direct_cf_schema_template,
    direct_article_cf_col_stats,
    make_direct_article_cf_tensor_frame,
)
from examples.hm_direct_cf.io import (
    initialize_or_load_manifest,
    load_manifest,
    load_snapshot,
    validate_manifest_for_training,
    write_snapshot_atomic,
)
from relbench.base import Table
from relbench.modeling.utils import to_unix_time


def _snapshot(seed="2020-03-02"):
    return pd.DataFrame(
        {
            "seed_time": pd.to_datetime([seed, seed]),
            "src_customer_id": pd.Series([0, 0], dtype="int64"),
            "dst_article_id": pd.Series([1, 2], dtype="int64"),
            "direct_score": pd.Series([0.7, 0.6], dtype="float32"),
            "max_cf_score": pd.Series([0.7, 0.6], dtype="float32"),
            "support_sum": pd.Series([3, 3], dtype="int64"),
            "support_max": pd.Series([3, 3], dtype="int64"),
            "num_source_articles": pd.Series([1, 1], dtype="int64"),
            "best_src_article_id": pd.Series([3, 4], dtype="int64"),
            "best_item_cf_rank": pd.Series([1, 1], dtype="int32"),
            "rank": pd.Series([1, 2], dtype="int32"),
        }
    )


def _base_graph():
    data = HeteroData()
    data["article"].tf = make_direct_article_cf_tensor_frame(5).tensor_frame
    data["article"].time = torch.zeros(5, dtype=torch.long)
    data["customer"].tf = make_direct_article_cf_tensor_frame(1).tensor_frame
    data["customer"].time = torch.tensor(
        [to_unix_time(pd.Series([pd.Timestamp("2020-03-02")]))[0]]
    )
    return data


def test_io_roundtrip_and_manifest_conflict(tmp_path):
    cfg = HMDirectCFSnapshotConfig(direct_top_k=2)
    snapshot_dir = cfg.snapshot_dir(tmp_path)
    manifest = initialize_or_load_manifest(snapshot_dir, cfg)
    assert manifest["direct_cf_node_type"] == "direct_article_cf"

    seed = pd.Timestamp("2020-03-02")
    path = write_snapshot_atomic(
        _snapshot(seed),
        snapshot_dir,
        seed,
        config=cfg,
        num_customers=2,
        num_articles=5,
    )
    assert path.name == snapshot_filename(seed)
    loaded = load_snapshot(snapshot_dir, seed, config=cfg, num_customers=2, num_articles=5)
    assert loaded["direct_score"].dtype == "float32"
    assert validate_manifest_for_training(snapshot_dir) == cfg

    manifest = load_manifest(snapshot_dir)
    manifest["direct_top_k"] = 99
    with open(snapshot_dir / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f)
    with pytest.raises(ValueError, match="direct_top_k"):
        initialize_or_load_manifest(snapshot_dir, cfg)


def test_graph_edge_roles_schema_and_fanouts():
    data = attach_direct_cf_snapshot(_base_graph(), _snapshot(), pd.Timestamp("2020-03-02"))
    assert DIRECT_ARTICLE_CF in data.node_types
    assert set(direct_article_cf_col_stats()) == {"__const__"}
    assert torch.equal(data[DIRECT_CF_SRC_F2P].edge_index, torch.tensor([[0, 1], [0, 0]]))
    assert torch.equal(data[DIRECT_CF_SRC_REV].edge_index, torch.tensor([[0, 0], [0, 1]]))
    assert torch.equal(data[DIRECT_CF_DST_F2P].edge_index, torch.tensor([[0, 1], [1, 2]]))
    assert torch.equal(data[DIRECT_CF_DST_REV].edge_index, torch.tensor([[1, 2], [0, 1]]))

    template = build_direct_cf_schema_template(_base_graph())
    for edge_type in DIRECT_CF_EDGE_TYPES:
        assert edge_type in template.edge_types
        assert template[edge_type].edge_index.numel() == 0
    fanouts = build_direct_cf_num_neighbors(
        template.edge_types,
        num_layers=2,
        num_neighbors=128,
    )
    assert fanouts[DIRECT_CF_SRC_F2P] == [128, 0]
    assert fanouts[DIRECT_CF_DST_REV] == [0, 64]
    assert fanouts[DIRECT_CF_SRC_REV] == [0, 0]
    assert fanouts[DIRECT_CF_DST_F2P] == [0, 0]
    with pytest.raises(ValueError, match="num_layers"):
        build_direct_cf_num_neighbors(template.edge_types, num_layers=1, num_neighbors=8)


def test_direct_coverage_for_table(tmp_path):
    cfg = HMDirectCFSnapshotConfig(direct_top_k=2)
    snapshot_dir = cfg.snapshot_dir(tmp_path)
    initialize_or_load_manifest(snapshot_dir, cfg)
    seed = pd.Timestamp("2020-03-02")
    write_snapshot_atomic(
        _snapshot(seed),
        snapshot_dir,
        seed,
        config=cfg,
        num_customers=2,
        num_articles=5,
    )

    class Task:
        src_entity_col = "customer_id"
        dst_entity_col = "article_id"
        time_col = "timestamp"
        eval_k = 2

    table = Table(
        df=pd.DataFrame(
            {
                "timestamp": pd.to_datetime([seed]),
                "customer_id": [0],
                "article_id": [[2, 3]],
            }
        ),
        fkey_col_to_pkey_table={"customer_id": "customer", "article_id": "article"},
        pkey_col=None,
        time_col="timestamp",
    )

    metrics = compute_direct_cf_coverage_for_table(
        table,
        Task(),
        snapshot_dir,
        config=cfg,
        num_customers=2,
        num_articles=5,
        num_layers=2,
    )
    assert metrics.num_covered_groundtruth_labels == 1
    assert metrics.hit_rate == 1.0
    with pytest.raises(ValueError, match="num_layers"):
        compute_direct_cf_coverage_for_table(
            table,
            Task(),
            snapshot_dir,
            config=cfg,
            num_customers=2,
            num_articles=5,
            num_layers=1,
        )
