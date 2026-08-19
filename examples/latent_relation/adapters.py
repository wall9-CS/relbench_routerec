from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

from relbench.base import Database, RecommendationTask, Table


@dataclass(frozen=True)
class QueryExample:
    source_id: int
    seed_time: pd.Timestamp
    positives: tuple[int, ...]
    row_index: int


class RecommendationTaskAdapter:
    """Dataset/task-specific access to temporal source-destination histories."""

    def __init__(
        self,
        task: RecommendationTask,
        db: Database,
        *,
        interaction_table: str,
        interaction_source_col: str,
        interaction_destination_col: str,
        interaction_time_col: str,
    ) -> None:
        self.task = task
        self.db = db
        self.interaction_table = interaction_table
        self.interaction_source_col = interaction_source_col
        self.interaction_destination_col = interaction_destination_col
        self.interaction_time_col = interaction_time_col

        frame = db.table_dict[interaction_table].df
        required = {
            interaction_source_col,
            interaction_destination_col,
            interaction_time_col,
        }
        missing = required.difference(frame.columns)
        if missing:
            raise ValueError(
                f"{interaction_table} is missing interaction columns: {sorted(missing)}"
            )
        frame = frame.loc[
            frame[interaction_source_col].notna()
            & frame[interaction_destination_col].notna(),
            [interaction_source_col, interaction_destination_col, interaction_time_col],
        ].copy()
        frame[interaction_source_col] = frame[interaction_source_col].astype("int64")
        frame[interaction_destination_col] = frame[interaction_destination_col].astype(
            "int64"
        )
        frame[interaction_time_col] = pd.to_datetime(frame[interaction_time_col])
        self.interactions = frame.sort_values(
            interaction_time_col, kind="mergesort"
        ).reset_index(drop=True)
        self._times = self.interactions[interaction_time_col].to_numpy(
            dtype="datetime64[ns]", copy=False
        )

    @property
    def source_type(self) -> str:
        return self.task.src_entity_table

    @property
    def destination_type(self) -> str:
        return self.task.dst_entity_table

    @property
    def num_destinations(self) -> int:
        return int(self.task.num_dst_nodes)

    def destination_ids_available_at(self, seed_time: pd.Timestamp) -> np.ndarray:
        table = self.db.table_dict[self.destination_type]
        if table.time_col is None:
            return np.arange(self.num_destinations, dtype=np.int64)
        times = pd.to_datetime(table.df[table.time_col])
        mask = times <= pd.Timestamp(seed_time)
        if table.pkey_col is None:
            return np.flatnonzero(mask.to_numpy()).astype(np.int64)
        return table.df.loc[mask, table.pkey_col].to_numpy(dtype=np.int64)

    def history_for_source(
        self,
        source_id: int,
        seed_time: pd.Timestamp,
        *,
        history_limit: int,
    ) -> tuple[int, ...]:
        histories = self.histories_for_sources(
            [source_id],
            seed_time,
            history_limit=history_limit,
        )
        return histories.get(int(source_id), tuple())

    def histories_for_sources(
        self,
        source_ids: Iterable[int],
        seed_time: pd.Timestamp,
        *,
        history_limit: int,
    ) -> dict[int, tuple[int, ...]]:
        source_set = {int(value) for value in source_ids if not pd.isna(value)}
        if not source_set:
            return {}
        seed_np = np.datetime64(pd.Timestamp(seed_time).to_datetime64(), "ns")
        right = int(np.searchsorted(self._times, seed_np, side="right"))
        if right == 0:
            return {source_id: tuple() for source_id in source_set}

        frame = self.interactions.iloc[:right]
        frame = frame[frame[self.interaction_source_col].isin(source_set)]
        out: dict[int, tuple[int, ...]] = {}
        for source_id, group in frame.groupby(self.interaction_source_col, sort=False):
            seen: set[int] = set()
            values: list[int] = []
            dst_values = group[self.interaction_destination_col].to_numpy(dtype=np.int64)
            for dst_id in dst_values[::-1]:
                dst = int(dst_id)
                if dst in seen:
                    continue
                seen.add(dst)
                values.append(dst)
                if len(values) >= history_limit:
                    break
            out[int(source_id)] = tuple(values)
        for source_id in source_set:
            out.setdefault(source_id, tuple())
        return out

    def examples_from_table(self, table: Table) -> list[QueryExample]:
        out: list[QueryExample] = []
        frame = table.df
        for row_index, row in frame.iterrows():
            source = row[self.task.src_entity_col]
            if pd.isna(source):
                continue
            positives = normalize_destination_list(row[self.task.dst_entity_col])
            out.append(
                QueryExample(
                    source_id=int(source),
                    seed_time=pd.Timestamp(row[self.task.time_col]),
                    positives=positives,
                    row_index=int(row_index),
                )
            )
        return out

    def anchor_destinations_for_table(
        self,
        table: Table,
        *,
        seed_time: pd.Timestamp,
        history_limit: int,
    ) -> np.ndarray:
        frame = table.df[pd.to_datetime(table.df[self.task.time_col]) == seed_time]
        histories = self.histories_for_sources(
            frame[self.task.src_entity_col].dropna().astype("int64").tolist(),
            seed_time,
            history_limit=history_limit,
        )
        anchors = sorted({dst for history in histories.values() for dst in history})
        return np.asarray(anchors, dtype=np.int64)


def normalize_destination_list(value) -> tuple[int, ...]:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return tuple()
    if isinstance(value, np.ndarray):
        values: Sequence = value.tolist()
    elif isinstance(value, (list, tuple, set)):
        values = list(value)
    else:
        values = [value]
    out: list[int] = []
    seen: set[int] = set()
    for item in values:
        if pd.isna(item):
            continue
        dst = int(item)
        if dst not in seen:
            seen.add(dst)
            out.append(dst)
    return tuple(out)


def make_task_adapter(dataset_name: str, task_name: str, task, db) -> RecommendationTaskAdapter:
    if dataset_name == "rel-hm" and task_name == "user-item-purchase":
        return RecommendationTaskAdapter(
            task,
            db,
            interaction_table="transactions",
            interaction_source_col="customer_id",
            interaction_destination_col="article_id",
            interaction_time_col="t_dat",
        )
    if dataset_name == "rel-avito" and task_name == "user-ad-visit":
        return RecommendationTaskAdapter(
            task,
            db,
            interaction_table="VisitStream",
            interaction_source_col="UserID",
            interaction_destination_col="AdID",
            interaction_time_col="ViewDate",
        )
    raise ValueError(
        "Latent relation adapters currently support rel-hm/user-item-purchase "
        "and rel-avito/user-ad-visit. "
        f"Got {dataset_name}/{task_name}."
    )
