from __future__ import annotations

import json

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import HeteroData

from examples.evaluate_recommendation_coverage import (
    ManifestIndex,
    _MutableTotals,
    _accumulate_sampled_batch_coverage,
    _selected_targets,
    compute_base_coverage_for_table,
)
from relbench.base import Table


class _Task:
    src_entity_col = "user_id"
    src_entity_table = "users"
    dst_entity_col = "item_id"
    dst_entity_table = "items"
    time_col = "timestamp"
    eval_k = 10


def test_base_coverage_uses_history_until_seed_time():
    table = Table(
        df=pd.DataFrame(
            {
                "timestamp": pd.to_datetime(["2020-01-03", "2020-01-03"]),
                "user_id": pd.Series([0, 1], dtype="int64"),
                "item_id": [[2, 3], [4]],
            }
        ),
        fkey_col_to_pkey_table={"user_id": "users", "item_id": "items"},
        pkey_col=None,
        time_col="timestamp",
    )
    interactions = pd.DataFrame(
        {
            "src_id": pd.Series([0, 0, 0, 1], dtype="int64"),
            "dst_id": pd.Series([2, 5, 3, 4], dtype="int64"),
            "time": pd.to_datetime(
                ["2020-01-01", "2020-01-02", "2020-01-04", "2020-01-03"]
            ),
        }
    )

    metrics = compute_base_coverage_for_table(table, _Task(), interactions)

    assert metrics.num_rows == 2
    assert metrics.num_rows_with_candidates == 2
    assert metrics.num_rows_with_hit == 2
    assert metrics.num_groundtruth_labels == 3
    assert metrics.num_covered_groundtruth_labels == 2
    assert np.isclose(metrics.coverage_rate, 2 / 3)


def test_manifest_index_distinguishes_cf_kinds(tmp_path):
    item_dir = tmp_path / "item"
    legacy_hm_item_dir = tmp_path / "legacy_hm_item"
    user_dir = tmp_path / "user"
    stack_user_dir = tmp_path / "stack_user"
    trial_dir = tmp_path / "trial"
    item_dir.mkdir()
    legacy_hm_item_dir.mkdir()
    user_dir.mkdir()
    stack_user_dir.mkdir()
    trial_dir.mkdir()
    (item_dir / "manifest.json").write_text(
        json.dumps(
            {
                "dataset": "rel-hm",
                "task": "user-item-purchase",
                "cf_node_type": "article_cf",
            }
        ),
        encoding="utf-8",
    )
    (legacy_hm_item_dir / "manifest.json").write_text(
        json.dumps(
            {
                "dataset": "rel-hm",
                "task": "user-item-purchase",
                "window_weeks": 8,
                "all_history": False,
                "min_support": 3,
                "top_l": 32,
                "alpha": 0.5,
            }
        ),
        encoding="utf-8",
    )
    (user_dir / "manifest.json").write_text(
        json.dumps(
            {
                "dataset": "rel-hm",
                "task": "user-item-purchase",
                "cf_kind": "user_user_history_overlap",
                "cf_node_type": "user_cf",
            }
        ),
        encoding="utf-8",
    )
    (stack_user_dir / "manifest.json").write_text(
        json.dumps(
            {
                "dataset": "rel-stack",
                "task": "user-post-comment",
                "cf_kind": "user_user_history_overlap",
                "cf_node_type": "user_cf",
            }
        ),
        encoding="utf-8",
    )
    (trial_dir / "manifest.json").write_text(
        json.dumps(
            {
                "dataset": "rel-trial",
                "task": "condition-sponsor-run",
                "cf_route": "source_sponsor_route_collapsed",
            }
        ),
        encoding="utf-8",
    )

    index = ManifestIndex(
        item_roots=[tmp_path],
        user_roots=[tmp_path],
        trial_roots=[tmp_path],
    )

    assert index.find("rel-hm", "user-item-purchase", "item_cf") == [
        item_dir,
        legacy_hm_item_dir,
    ]
    assert index.find("rel-hm", "user-item-purchase", "user_cf") == [user_dir]
    assert index.find("rel-stack", "user-post-comment", "user_cf") == [
        stack_user_dir
    ]
    assert index.find("rel-trial", "condition-sponsor-run", "trial_cf") == [
        trial_dir
    ]


def test_selected_targets_filters_dataset_and_task():
    targets = _selected_targets(["rel-hm"], ["user-item-purchase"])

    assert len(targets) == 1
    assert targets[0].dataset == "rel-hm"
    assert targets[0].task == "user-item-purchase"


def test_sampled_batch_coverage_is_source_specific():
    batch = HeteroData()
    batch["users"].batch_size = 2
    batch["users"].input_id = torch.tensor([0, 1])
    batch["items"].n_id = torch.tensor([2, 3, 4, 5])
    batch["items"].batch = torch.tensor([0, 0, 1, 1])
    table = Table(
        df=pd.DataFrame(
            {
                "timestamp": pd.to_datetime(["2020-01-01", "2020-01-01"]),
                "user_id": pd.Series([0, 1], dtype="int64"),
                "item_id": [[2, 9], [4]],
            }
        ),
        fkey_col_to_pkey_table={"user_id": "users", "item_id": "items"},
        pkey_col=None,
        time_col="timestamp",
    )
    totals = _MutableTotals()

    _accumulate_sampled_batch_coverage(totals, batch, table, _Task())
    metrics = totals.freeze()

    assert metrics.num_rows == 2
    assert metrics.num_sampled_candidates == 4
    assert metrics.num_groundtruth_labels == 3
    assert metrics.num_covered_groundtruth_labels == 2
    assert np.isclose(metrics.coverage_rate, 2 / 3)
    assert np.isclose(metrics.sampled_candidate_precision, 2 / 4)
