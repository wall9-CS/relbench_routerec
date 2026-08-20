from __future__ import annotations

import pandas as pd

from relbench.base import Database

from .config import TrialCFSnapshotConfig


def build_source_sponsor_interactions(
    db: Database,
    config: TrialCFSnapshotConfig,
) -> pd.DataFrame:
    """Collapse the Rel-Trial source->study->sponsor route into interactions."""
    source_studies = db.table_dict[config.source_study_table].df
    sponsor_studies = db.table_dict["sponsors_studies"].df

    required_source = {"nct_id", config.src_entity_col, "date"}
    missing_source = required_source.difference(source_studies.columns)
    if missing_source:
        raise ValueError(
            f"{config.source_study_table} is missing columns: {sorted(missing_source)}"
        )
    required_sponsor = {"nct_id", "sponsor_id"}
    missing_sponsor = required_sponsor.difference(sponsor_studies.columns)
    if missing_sponsor:
        raise ValueError(f"sponsors_studies is missing columns: {sorted(missing_sponsor)}")

    interactions = source_studies[[config.src_entity_col, "nct_id", "date"]].merge(
        sponsor_studies[["nct_id", "sponsor_id"]],
        on="nct_id",
        how="inner",
    )
    interactions = interactions.rename(columns={config.src_entity_col: "src_id"})
    interactions = interactions[["src_id", "sponsor_id", "nct_id", "date"]]
    interactions = interactions.dropna(subset=["src_id", "sponsor_id", "date"])
    interactions["date"] = pd.to_datetime(interactions["date"])
    interactions["src_id"] = interactions["src_id"].astype("int64")
    interactions["sponsor_id"] = interactions["sponsor_id"].astype("int64")
    return interactions.drop_duplicates(ignore_index=True)

