from __future__ import annotations

import pandas as pd

from .config import SUPPORTED_TASKS


REVIEW_LENGTH_THRESHOLD = 300
INTERACTION_COLUMNS = ["customer_id", "product_id", "review_time"]


def filter_review_interactions(
    review: pd.DataFrame,
    task: str,
) -> pd.DataFrame:
    """Return task-specific historical customer-product interactions."""
    if task not in SUPPORTED_TASKS:
        raise ValueError(f"Unsupported Rel-Amazon recommendation task: {task!r}.")
    missing = set(INTERACTION_COLUMNS).difference(review.columns)
    if missing:
        raise ValueError(f"review is missing columns: {sorted(missing)}")

    mask = review["customer_id"].notna() & review["product_id"].notna()
    if task == "user-item-rate":
        if "rating" not in review.columns:
            raise ValueError("review is missing column: 'rating'")
        mask &= review["rating"] == 5.0
    elif task == "user-item-review":
        if "review_text" not in review.columns:
            raise ValueError("review is missing column: 'review_text'")
        mask &= review["review_text"].notna()
        mask &= review["review_text"].str.len() > REVIEW_LENGTH_THRESHOLD

    return review.loc[mask, INTERACTION_COLUMNS].reset_index(drop=True)
