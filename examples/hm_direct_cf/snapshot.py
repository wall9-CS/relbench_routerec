from __future__ import annotations

from typing import Iterable

import pandas as pd

from .common import DirectCFSnapshotStats, build_direct_snapshot
from .config import HMDirectCFSnapshotConfig
from .io import HM_COLUMNS


def build_snapshot(
    transactions: pd.DataFrame,
    item_cf_snapshot: pd.DataFrame,
    seed_time: pd.Timestamp,
    source_customer_ids: Iterable[int],
    config: HMDirectCFSnapshotConfig,
) -> tuple[pd.DataFrame, DirectCFSnapshotStats]:
    return build_direct_snapshot(
        transactions,
        item_cf_snapshot,
        seed_time,
        source_customer_ids,
        columns=HM_COLUMNS,
        direct_top_k=config.direct_top_k,
        filter_seen_dst=config.filter_seen_dst,
    )
