from __future__ import annotations

import pandas as pd

from examples.hm_cf.snapshot import SnapshotStats, build_snapshot as _build_item_snapshot

from .config import AvitoCFSnapshotConfig


def build_snapshot(
    interactions: pd.DataFrame,
    seed_time: pd.Timestamp,
    num_ads: int,
    config: AvitoCFSnapshotConfig,
) -> tuple[pd.DataFrame, SnapshotStats]:
    """Build one sparse ad-CF snapshot for Rel-Avito."""
    required = {"UserID", "AdID", "ViewDate"}
    missing = required.difference(interactions.columns)
    if missing:
        raise ValueError(f"interactions is missing columns: {sorted(missing)}")

    item_frame = interactions.rename(
        columns={"UserID": "customer_id", "AdID": "article_id", "ViewDate": "t_dat"}
    )[["customer_id", "article_id", "t_dat"]]
    snapshot, stats = _build_item_snapshot(
        item_frame,
        seed_time,
        num_ads,
        config,
    )
    snapshot = snapshot.rename(
        columns={
            "src_article_id": "src_AdID",
            "dst_article_id": "dst_AdID",
        }
    )
    return snapshot, stats
