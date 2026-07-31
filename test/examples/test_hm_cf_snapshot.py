import math

import numpy as np
import pandas as pd
import pytest

from examples.hm_cf.config import CFSnapshotConfig
from examples.hm_cf.snapshot import build_snapshot


def _tx(rows):
    return pd.DataFrame(rows, columns=["customer_id", "article_id", "t_dat"]).astype(
        {"customer_id": "int64", "article_id": "int64"}
    ).assign(t_dat=lambda df: pd.to_datetime(df["t_dat"]))


def test_window_boundary_is_lower_exclusive_upper_inclusive():
    seed = pd.Timestamp("2020-03-02")
    tx = _tx(
        [
            (0, 0, "2020-01-06"),
            (0, 1, "2020-01-07"),
            (0, 0, "2020-03-02"),
            (1, 1, "2020-03-03"),
        ]
    )
    out, stats = build_snapshot(
        tx,
        seed,
        3,
        CFSnapshotConfig(min_support=1, top_l=10),
    )
    assert stats.num_raw_transactions == 2
    assert set(map(tuple, out[["src_article_id", "dst_article_id"]].to_numpy())) == {
        (0, 1),
        (1, 0),
    }


def test_binary_interactions_and_known_cooccurrence():
    tx = _tx(
        [
            (0, 0, "2020-02-01"),
            (0, 0, "2020-02-02"),
            (0, 1, "2020-02-03"),
            (1, 0, "2020-02-04"),
            (1, 1, "2020-02-05"),
            (1, 2, "2020-02-06"),
        ]
    )
    out, stats = build_snapshot(
        tx,
        pd.Timestamp("2020-03-02"),
        3,
        CFSnapshotConfig(min_support=1, top_l=10),
    )
    support = {
        tuple(row[:2]): row[2]
        for row in out[["src_article_id", "dst_article_id", "support"]].to_numpy()
    }
    assert stats.num_unique_user_item_pairs == 5
    assert support[(0, 1)] == 2
    assert support[(1, 0)] == 2
    assert support[(0, 2)] == 1
    assert support[(2, 0)] == 1
    assert support[(1, 2)] == 1
    assert support[(2, 1)] == 1


def test_minimum_support_filtering():
    tx = _tx(
        [
            (0, 0, "2020-02-01"),
            (0, 1, "2020-02-01"),
            (1, 0, "2020-02-02"),
            (1, 1, "2020-02-02"),
            (2, 0, "2020-02-03"),
            (2, 2, "2020-02-03"),
            (3, 0, "2020-02-04"),
            (3, 2, "2020-02-04"),
            (4, 0, "2020-02-05"),
            (4, 2, "2020-02-05"),
        ]
    )
    out, _ = build_snapshot(
        tx,
        pd.Timestamp("2020-03-02"),
        3,
        CFSnapshotConfig(min_support=3, top_l=10),
    )
    assert set(map(tuple, out[["src_article_id", "dst_article_id"]].to_numpy())) == {
        (0, 2),
        (2, 0),
    }
    assert set(out["support"]) == {3}


def test_alpha_score_formula_and_directionality():
    tx = _tx(
        [
            (0, 0, "2020-02-01"),
            (0, 1, "2020-02-01"),
            (1, 0, "2020-02-02"),
            (1, 1, "2020-02-02"),
            (2, 0, "2020-02-03"),
        ]
    )
    out, _ = build_snapshot(
        tx,
        pd.Timestamp("2020-03-02"),
        2,
        CFSnapshotConfig(min_support=1, top_l=10, alpha=0.5),
    )
    score = out.set_index(["src_article_id", "dst_article_id"])["cf_score"]
    assert math.isclose(score[(0, 1)], 2 / math.sqrt(3 * 2), rel_tol=1e-6)

    out, _ = build_snapshot(
        tx,
        pd.Timestamp("2020-03-02"),
        2,
        CFSnapshotConfig(min_support=1, top_l=10, alpha=1.0),
    )
    score = out.set_index(["src_article_id", "dst_article_id"])["cf_score"]
    assert math.isclose(score[(0, 1)], 2 / 3, rel_tol=1e-6)
    assert math.isclose(score[(1, 0)], 2 / 2, rel_tol=1e-6)
    assert score[(0, 1)] != score[(1, 0)]


def test_top_l_tie_break_and_ranks():
    rows = []
    for customer in range(4):
        rows.extend(
            [
                (customer, 0, "2020-02-01"),
                (customer, 2, "2020-02-01"),
                (customer, 1, "2020-02-01"),
                (customer, 3, "2020-02-01"),
            ]
        )
    out, _ = build_snapshot(
        _tx(rows),
        pd.Timestamp("2020-03-02"),
        4,
        CFSnapshotConfig(min_support=1, top_l=2, alpha=0.5),
    )
    src0 = out[out["src_article_id"] == 0]
    assert src0["dst_article_id"].tolist() == [1, 2]
    assert src0["rank"].tolist() == [1, 2]
    assert out.groupby("src_article_id").size().max() <= 2


def test_config_validation_defaults_and_bounds():
    cfg = CFSnapshotConfig()
    assert cfg.window_weeks == 8
    assert cfg.min_support == 3
    assert cfg.top_l == 32
    assert cfg.alpha == 0.5
    with pytest.raises(ValueError, match="mutually exclusive"):
        CFSnapshotConfig(all_history=True, window_weeks=8)
    with pytest.raises(ValueError, match="alpha"):
        CFSnapshotConfig(alpha=1.1)
