from __future__ import annotations

import pandas as pd


INTERACTION_COLUMNS = ["UserID", "AdID", "ViewDate"]


def filter_visit_interactions(visit_stream: pd.DataFrame) -> pd.DataFrame:
    """Return historical user-ad visit interactions for Rel-Avito CF."""
    missing = set(INTERACTION_COLUMNS).difference(visit_stream.columns)
    if missing:
        raise ValueError(f"VisitStream is missing columns: {sorted(missing)}")
    mask = visit_stream["UserID"].notna() & visit_stream["AdID"].notna()
    return visit_stream.loc[mask, INTERACTION_COLUMNS].reset_index(drop=True)
