from __future__ import annotations

from typing import Iterable

import pandas as pd

from examples.hm_user_cf.config import UserCFSnapshotConfig
from examples.hm_user_cf.snapshot import (
    UserCFSnapshotStats,
    build_snapshot as _build_user_snapshot,
)

from .config import StackUserCFSnapshotConfig


def build_snapshot(
    interactions: pd.DataFrame,
    seed_time: pd.Timestamp,
    num_users: int,
    num_posts: int,
    source_user_ids: Iterable[int],
    config: StackUserCFSnapshotConfig,
    *,
    source_chunk_size: int = 4096,
) -> tuple[pd.DataFrame, UserCFSnapshotStats]:
    """Build one source-scoped user-CF snapshot for Rel-Stack."""
    required = {"UserId", "PostId", "CreationDate"}
    missing = required.difference(interactions.columns)
    if missing:
        raise ValueError(f"interactions is missing columns: {sorted(missing)}")

    user_frame = interactions.rename(
        columns={"UserId": "customer_id", "PostId": "article_id", "CreationDate": "t_dat"}
    )[["customer_id", "article_id", "t_dat"]]
    hm_config = UserCFSnapshotConfig(
        dataset="rel-hm",
        task="user-item-purchase",
        window_weeks=config.window_weeks,
        all_history=config.all_history,
        min_overlap=config.min_overlap,
        top_k=config.top_k,
        alpha=config.alpha,
    )
    snapshot, stats = _build_user_snapshot(
        user_frame,
        seed_time,
        num_users,
        num_posts,
        source_user_ids,
        hm_config,
        source_chunk_size=source_chunk_size,
    )
    snapshot = snapshot.rename(
        columns={
            "src_customer_id": "src_UserId",
            "dst_customer_id": "dst_UserId",
        }
    )
    return snapshot, stats

