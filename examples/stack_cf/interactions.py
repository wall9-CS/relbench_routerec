from __future__ import annotations

import pandas as pd


INTERACTION_COLUMNS = ["UserId", "PostId", "CreationDate"]


def filter_comment_interactions(comments: pd.DataFrame) -> pd.DataFrame:
    """Return historical user-post comment interactions for Rel-Stack CF."""
    missing = set(INTERACTION_COLUMNS).difference(comments.columns)
    if missing:
        raise ValueError(f"comments is missing columns: {sorted(missing)}")
    mask = comments["UserId"].notna() & comments["PostId"].notna()
    out = comments.loc[mask, INTERACTION_COLUMNS].reset_index(drop=True)
    out = out.astype({"UserId": "int64", "PostId": "int64"})
    return out
