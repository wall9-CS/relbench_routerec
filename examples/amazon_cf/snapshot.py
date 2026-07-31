from __future__ import annotations

import pandas as pd

from examples.hm_cf.snapshot import SnapshotStats, build_snapshot as _build_item_snapshot

from .config import AmazonCFSnapshotConfig


def build_snapshot(
    interactions: pd.DataFrame,
    seed_time: pd.Timestamp,
    num_products: int,
    config: AmazonCFSnapshotConfig,
) -> tuple[pd.DataFrame, SnapshotStats]:
    """Build one task-specific Rel-Amazon sparse product-CF snapshot."""
    required = {"customer_id", "product_id", "review_time"}
    missing = required.difference(interactions.columns)
    if missing:
        raise ValueError(f"interactions is missing columns: {sorted(missing)}")

    item_frame = interactions.rename(
        columns={"product_id": "article_id", "review_time": "t_dat"}
    )[["customer_id", "article_id", "t_dat"]]
    snapshot, stats = _build_item_snapshot(
        item_frame,
        seed_time,
        num_products,
        config,
    )
    snapshot = snapshot.rename(
        columns={
            "src_article_id": "src_product_id",
            "dst_article_id": "dst_product_id",
        }
    )
    return snapshot, stats
