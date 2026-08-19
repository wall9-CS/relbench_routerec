import math

import numpy as np
import pandas as pd
import pytest

from examples.hm_user_cf.config import UserCFSnapshotConfig
from examples.hm_user_cf.snapshot import build_snapshot


def _tx(rows):
    return pd.DataFrame(rows, columns=["customer_id", "article_id", "t_dat"]).astype(
        {"customer_id": "int64", "article_id": "int64"}
    ).assign(t_dat=lambda df: pd.to_datetime(df["t_dat"]))


def test_window_boundary_is_lower_exclusive_upper_inclusive():
    seed = pd.Timestamp("2020-03-02")
    tx = _tx(
        [
            (0, 0, "2020-01-06"),
            (1, 0, "2020-01-07"),
            (0, 1, "2020-03-02"),
            (1, 1, "2020-03-03"),
        ]
    )
    out, stats = build_snapshot(
        tx,
        seed,
        3,
        3,
        [0],
        UserCFSnapshotConfig(min_overlap=1, top_k=10),
    )
    assert stats.num_raw_transactions == 2
    assert out.empty


def test_binary_history_and_known_overlap():
    tx = _tx(
        [
            (0, 0, "2020-02-01"),
            (0, 0, "2020-02-02"),
            (0, 1, "2020-02-03"),
            (1, 0, "2020-02-04"),
            (1, 1, "2020-02-05"),
            (1, 2, "2020-02-06"),
            (2, 1, "2020-02-07"),
        ]
    )
    out, stats = build_snapshot(
        tx,
        pd.Timestamp("2020-03-02"),
        3,
        3,
        [0],
        UserCFSnapshotConfig(min_overlap=1, top_k=10),
    )
    overlaps = {
        int(row[0]): int(row[1])
        for row in out[["dst_customer_id", "overlap"]].to_numpy()
    }
    assert stats.num_unique_user_item_pairs == 6
    assert overlaps[1] == 2
    assert overlaps[2] == 1


def test_min_overlap_filtering():
    tx = _tx(
        [
            (0, 0, "2020-02-01"),
            (0, 1, "2020-02-01"),
            (1, 0, "2020-02-02"),
            (2, 0, "2020-02-03"),
            (2, 1, "2020-02-03"),
        ]
    )
    out, _ = build_snapshot(
        tx,
        pd.Timestamp("2020-03-02"),
        3,
        2,
        [0],
        UserCFSnapshotConfig(min_overlap=2, top_k=10),
    )
    assert out["dst_customer_id"].tolist() == [2]
    assert out["overlap"].tolist() == [2]


def test_alpha_score_formula_and_directionality():
    tx = _tx(
        [
            (0, 0, "2020-02-01"),
            (0, 1, "2020-02-01"),
            (0, 2, "2020-02-01"),
            (1, 0, "2020-02-02"),
            (1, 1, "2020-02-02"),
        ]
    )
    out, _ = build_snapshot(
        tx,
        pd.Timestamp("2020-03-02"),
        2,
        3,
        [0, 1],
        UserCFSnapshotConfig(min_overlap=1, top_k=10, alpha=0.5),
    )
    score = out.set_index(["src_customer_id", "dst_customer_id"])["user_cf_score"]
    assert math.isclose(score[(0, 1)], 2 / math.sqrt(3 * 2), rel_tol=1e-6)

    out, _ = build_snapshot(
        tx,
        pd.Timestamp("2020-03-02"),
        2,
        3,
        [0, 1],
        UserCFSnapshotConfig(min_overlap=1, top_k=10, alpha=1.0),
    )
    score = out.set_index(["src_customer_id", "dst_customer_id"])["user_cf_score"]
    assert math.isclose(score[(0, 1)], 2 / 3, rel_tol=1e-6)
    assert math.isclose(score[(1, 0)], 2 / 2, rel_tol=1e-6)
    assert score[(0, 1)] != score[(1, 0)]


def test_top_k_tie_break_ranks_and_source_scope():
    rows = []
    rows.extend([(0, 0, "2020-02-01"), (0, 1, "2020-02-01")])
    rows.extend([(1, 0, "2020-02-01"), (1, 1, "2020-02-01")])
    rows.extend([(2, 0, "2020-02-01"), (2, 1, "2020-02-01")])
    rows.extend([(3, 0, "2020-02-01"), (3, 1, "2020-02-01")])
    out, _ = build_snapshot(
        _tx(rows),
        pd.Timestamp("2020-03-02"),
        4,
        2,
        [0],
        UserCFSnapshotConfig(min_overlap=1, top_k=2, alpha=0.5),
    )
    assert out["src_customer_id"].unique().tolist() == [0]
    assert out["dst_customer_id"].tolist() == [1, 2]
    assert out["rank"].tolist() == [1, 2]
    assert not (out["src_customer_id"] == out["dst_customer_id"]).any()


def test_config_validation_defaults_and_bounds():
    cfg = UserCFSnapshotConfig()
    assert cfg.window_weeks == 8
    assert cfg.min_overlap == 2
    assert cfg.top_k == 32
    assert cfg.alpha == 0.5
    with pytest.raises(ValueError, match="mutually exclusive"):
        UserCFSnapshotConfig(all_history=True, window_weeks=8)
    with pytest.raises(ValueError, match="alpha"):
        UserCFSnapshotConfig(alpha=1.1)
    with pytest.raises(ValueError, match="source_chunk_size"):
        build_snapshot(_tx([]), pd.Timestamp("2020-03-02"), 1, 1, [], cfg, source_chunk_size=0)

