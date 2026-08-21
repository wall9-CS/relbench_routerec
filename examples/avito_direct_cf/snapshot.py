from __future__ import annotations

from typing import Iterable

import pandas as pd

from examples.hm_direct_cf.common import DirectCFSnapshotStats, build_direct_snapshot

from .config import AvitoDirectCFSnapshotConfig
from .io import AVITO_COLUMNS


def build_snapshot(
    interactions: pd.DataFrame,
    item_cf_snapshot: pd.DataFrame,
    seed_time: pd.Timestamp,
    source_user_ids: Iterable[int],
    config: AvitoDirectCFSnapshotConfig,
) -> tuple[pd.DataFrame, DirectCFSnapshotStats]:
    return build_direct_snapshot(
        interactions,
        item_cf_snapshot,
        seed_time,
        source_user_ids,
        columns=AVITO_COLUMNS,
        direct_top_k=config.direct_top_k,
        filter_seen_dst=config.filter_seen_dst,
    )
