from __future__ import annotations

import json

import numpy as np
import pandas as pd

from examples.evaluate_recommendation_coverage import (
    ManifestIndex,
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
    user_dir = tmp_path / "user"
    trial_dir = tmp_path / "trial"
    item_dir.mkdir()
    user_dir.mkdir()
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

    assert index.find("rel-hm", "user-item-purchase", "item_cf") == [item_dir]
    assert index.find("rel-hm", "user-item-purchase", "user_cf") == [user_dir]
    assert index.find("rel-trial", "condition-sponsor-run", "trial_cf") == [
        trial_dir
    ]

