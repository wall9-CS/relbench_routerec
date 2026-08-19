import copy

import pandas as pd
import pytest
import torch
from torch_geometric.data import HeteroData

from examples.hm_cf.graph import make_article_cf_tensor_frame
from examples.latent_relation.adapters import RecommendationTaskAdapter
from examples.latent_relation.config import LatentRelationConfig
from examples.latent_relation.coverage import (
    aggregate_query_coverage,
    compute_latent_coverage_for_table,
)
from examples.latent_relation.graph import (
    LATENT_RELATION,
    REV_LATENT_RELATION,
    attach_latent_snapshot,
    build_latent_num_neighbors,
    build_latent_schema_template,
    latent_edge_types,
)
from examples.latent_relation.io import (
    load_snapshot,
    should_reuse_snapshot,
    write_snapshot_atomic,
)
from examples.latent_relation.materialize import build_latent_snapshot
from examples.latent_relation.model import (
    EmbeddingDestinationEncoder,
    LatentRelationModel,
    TwoTowerRelationScorer,
)
from examples.latent_relation.training import (
    RelationTrainingDataset,
    collate_relation_batch,
    relation_loss,
    set_deterministic_seed,
)
from relbench.base import Database, Table


class DummyTask:
    src_entity_table = "user"
    src_entity_col = "user_id"
    dst_entity_table = "item"
    dst_entity_col = "item_id"
    time_col = "timestamp"
    num_dst_nodes = 5
    eval_k = 2


def _db(interactions):
    return Database(
        {
            "user": Table(pd.DataFrame({"user_id": [0, 1]}), {}, pkey_col="user_id"),
            "item": Table(
                pd.DataFrame({"item_id": [0, 1, 2, 3, 4]}),
                {},
                pkey_col="item_id",
            ),
            "events": Table(
                pd.DataFrame(
                    interactions,
                    columns=["user_id", "item_id", "event_time"],
                )
                .astype({"user_id": "int64", "item_id": "int64"})
                .assign(event_time=lambda df: pd.to_datetime(df["event_time"])),
                {"user_id": "user", "item_id": "item"},
                time_col="event_time",
            ),
        }
    )


def _adapter(interactions):
    return RecommendationTaskAdapter(
        DummyTask(),
        _db(interactions),
        interaction_table="events",
        interaction_source_col="user_id",
        interaction_destination_col="item_id",
        interaction_time_col="event_time",
    )


def _task_table(rows):
    return Table(
        pd.DataFrame(rows, columns=["timestamp", "user_id", "item_id"]).assign(
            timestamp=lambda df: pd.to_datetime(df["timestamp"])
        ),
        {"user_id": "user", "item_id": "item"},
        time_col="timestamp",
    )


def _snapshot(rows, seed="2020-01-03"):
    frame = pd.DataFrame(
        rows,
        columns=["src_id", "dst_id", "score", "rank"],
    ).astype({"src_id": "int64", "dst_id": "int64", "score": "float32", "rank": "int32"})
    frame["seed_time"] = pd.Series(
        [pd.Timestamp(seed).to_datetime64()] * len(frame), dtype="datetime64[ns]"
    )
    return frame[["src_id", "dst_id", "score", "rank", "seed_time"]]


def test_no_future_leakage_and_history_limit():
    adapter = _adapter(
        [
            (0, 1, "2020-01-01"),
            (0, 2, "2020-01-02"),
            (0, 1, "2020-01-03"),
            (0, 3, "2020-01-04"),
        ]
    )
    assert adapter.history_for_source(0, pd.Timestamp("2020-01-03"), history_limit=4) == (
        1,
        2,
    )
    assert adapter.history_for_source(0, pd.Timestamp("2020-01-03"), history_limit=1) == (
        1,
    )


def test_relation_loss_training_decreases_and_separates_scores():
    set_deterministic_seed(7)
    adapter = _adapter(
        [
            (0, 0, "2020-01-01"),
            (1, 2, "2020-01-01"),
        ]
    )
    examples = adapter.examples_from_table(
        _task_table(
            [
                ("2020-01-02", 0, [1]),
                ("2020-01-02", 1, [3]),
            ]
        )
    )
    dataset = RelationTrainingDataset(
        examples,
        adapter,
        history_limit=4,
        num_negatives=2,
        seed=11,
    )
    model = LatentRelationModel(
        EmbeddingDestinationEncoder(5, 8),
        TwoTowerRelationScorer(8, relation_dim=8, temperature=0.2),
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=0.05)
    batch = collate_relation_batch([dataset[0], dataset[1]])
    loss0, pos0, neg0 = relation_loss(model, batch, device=torch.device("cpu"))
    for _ in range(80):
        loss, _, _ = relation_loss(model, batch, device=torch.device("cpu"))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    loss1, pos1, neg1 = relation_loss(model, batch, device=torch.device("cpu"))
    assert float(loss1.detach()) < float(loss0.detach())
    assert float(pos1.mean()) > float(neg1.mean())
    assert float(pos1.mean()) > float(pos0.mean()) or float(neg1.mean()) < float(neg0.mean())


def test_top_l_unique_and_self_loop_removal():
    model = LatentRelationModel(
        EmbeddingDestinationEncoder(5, 5),
        TwoTowerRelationScorer(5, relation_dim=5, temperature=1.0),
    )
    with torch.no_grad():
        weight = torch.eye(5)
        model.encoder.embedding.weight.copy_(weight)
        model.scorer.query.weight.copy_(weight)
        model.scorer.key.weight.copy_(weight)
    adapter = _adapter([(0, 0, "2020-01-01")])
    cfg = LatentRelationConfig("rel-toy", "toy", top_l=2, relation_dim=5, temperature=1.0)
    snapshot, _ = build_latent_snapshot(
        model,
        adapter,
        pd.Timestamp("2020-01-02"),
        anchor_ids=pd.Series([0, 1], dtype="int64").to_numpy(),
        config=cfg,
        device=torch.device("cpu"),
        query_chunk_size=1,
        candidate_chunk_size=2,
        embedding_chunk_size=2,
    )
    assert snapshot.groupby("src_id").size().max() <= 2
    assert not (snapshot["src_id"] == snapshot["dst_id"]).any()
    assert not snapshot.duplicated(["src_id", "dst_id"]).any()


def _base_graph():
    data = HeteroData()
    data["item"].tf = make_article_cf_tensor_frame(4).tensor_frame
    return data


def test_direct_graph_edge_without_new_node_type():
    base = _base_graph()
    before_nodes = set(base.node_types)
    data = attach_latent_snapshot(
        base,
        _snapshot([(0, 2, 0.9, 1), (1, 3, 0.7, 1)]),
        pd.Timestamp("2020-01-03"),
        destination_type="item",
        num_destinations=4,
    )
    assert set(data.node_types) == before_nodes
    assert ("item", LATENT_RELATION, "item") in data.edge_types
    assert ("item", REV_LATENT_RELATION, "item") in data.edge_types
    assert torch.equal(
        data[("item", LATENT_RELATION, "item")].edge_index,
        torch.tensor([[0, 1], [2, 3]]),
    )


def test_new_reachability_outside_baseline(tmp_path):
    seed = pd.Timestamp("2020-01-03")
    adapter = _adapter([(0, 0, "2020-01-01")])
    cfg = LatentRelationConfig("rel-toy", "toy", top_l=2)
    write_snapshot_atomic(
        _snapshot([(0, 2, 0.9, 1)], seed),
        tmp_path,
        seed,
        config=cfg,
        num_destinations=5,
    )
    rows = compute_latent_coverage_for_table(
        _task_table([("2020-01-03", 0, [2])]),
        DummyTask(),
        adapter,
        tmp_path,
        split="val",
        history_limit=4,
    )
    report = aggregate_query_coverage(rows)
    assert rows[0].already_local == 0
    assert rows[0].newly_reached_by_latent_relation == 1
    assert report.baseline_locality == 0.0
    assert report.latent_locality == 1.0


def test_snapshot_consistency_and_resume(tmp_path):
    seed = pd.Timestamp("2020-01-03")
    cfg = LatentRelationConfig("rel-toy", "toy", top_l=2)
    frame = _snapshot([(0, 2, 0.9, 1), (0, 3, 0.8, 2)], seed)
    write_snapshot_atomic(frame, tmp_path, seed, config=cfg, num_destinations=5)
    loaded = load_snapshot(tmp_path, seed, config=cfg, num_destinations=5)
    pd.testing.assert_frame_equal(loaded.reset_index(drop=True), frame.reset_index(drop=True))
    assert should_reuse_snapshot(tmp_path, seed, config=cfg, num_destinations=5)
    assert not should_reuse_snapshot(tmp_path, seed, config=cfg, num_destinations=5, force=True)


def test_latent_schema_template_does_not_mutate_base_and_budget_modes():
    base = _base_graph()
    original = copy.copy(base)
    template = build_latent_schema_template(base, destination_type="item")
    assert base.edge_types == original.edge_types
    forward, reverse = latent_edge_types("item")
    assert forward in template.edge_types
    additive = build_latent_num_neighbors(
        template.edge_types,
        destination_type="item",
        num_layers=3,
        num_neighbors=8,
        top_l=3,
        budget_mode="additive",
    )
    fixed = build_latent_num_neighbors(
        template.edge_types,
        destination_type="item",
        num_layers=3,
        num_neighbors=8,
        top_l=3,
        budget_mode="fixed",
    )
    assert additive[reverse] == [0, 0, 3]
    assert fixed[reverse] == [0, 0, 2]
    assert fixed[forward] == [0, 0, 0]
    with pytest.raises(ValueError, match="num_layers"):
        build_latent_num_neighbors(
            template.edge_types,
            destination_type="item",
            num_layers=2,
            num_neighbors=8,
            top_l=3,
        )
