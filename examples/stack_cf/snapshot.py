from __future__ import annotations

import pandas as pd

from examples.hm_cf.snapshot import (
    SnapshotStats,
    build_snapshot as _build_item_snapshot,
)

from .config import StackCFSnapshotConfig


def build_snapshot(
    interactions: pd.DataFrame,
    seed_time: pd.Timestamp,
    num_posts: int,
    config: StackCFSnapshotConfig,
) -> tuple[pd.DataFrame, SnapshotStats]:
    """Build one sparse post-CF snapshot for Rel-Stack."""
    required = {"UserId", "PostId", "CreationDate"}
    missing = required.difference(interactions.columns)
    if missing:
        raise ValueError(f"interactions is missing columns: {sorted(missing)}")

    item_frame = interactions.rename(
        columns={
            "UserId": "customer_id",
            "PostId": "article_id",
            "CreationDate": "t_dat",
        }
    )[["customer_id", "article_id", "t_dat"]]
    snapshot, stats = _build_item_snapshot(
        item_frame,
        seed_time,
        num_posts,
        config,
    )
    snapshot = snapshot.rename(
        columns={
            "src_article_id": "src_PostId",
            "dst_article_id": "dst_PostId",
        }
    )
    return snapshot, stats
