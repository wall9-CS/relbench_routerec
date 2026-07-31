import os

import numpy as np
import pandas as pd
import pooch

from relbench.base import Database, Dataset, Table
from relbench.utils import clean_datetime, unzip_processor


def _make_similar_category_location_price_ad_df(
    ads_info_df: pd.DataFrame,
    num_price_neighbors: int = 1,
) -> pd.DataFrame:
    if num_price_neighbors < 1:
        raise ValueError("num_price_neighbors must be positive.")

    required_cols = ["AdID", "CategoryID", "LocationID", "Price"]
    ads_df = ads_info_df[required_cols].copy()
    ads_df["Price"] = pd.to_numeric(ads_df["Price"], errors="coerce")
    ads_df = ads_df.dropna(subset=required_cols)
    if ads_df.empty:
        return pd.DataFrame(columns=["AdID_left", "AdID_right"])

    price = ads_df["Price"].clip(lower=0)
    ads_df["price_bin"] = np.floor(np.log1p(price)).astype("int16")
    ads_df = ads_df.sort_values(
        ["CategoryID", "LocationID", "price_bin", "Price", "AdID"],
        kind="mergesort",
    )

    left_chunks = []
    right_chunks = []
    for _, group in ads_df.groupby(
        ["CategoryID", "LocationID", "price_bin"],
        sort=False,
        observed=True,
    ):
        ad_ids = group["AdID"].to_numpy()
        max_offset = min(num_price_neighbors, len(ad_ids) - 1)
        for offset in range(1, max_offset + 1):
            left_chunks.append(ad_ids[:-offset])
            right_chunks.append(ad_ids[offset:])

    if not left_chunks:
        return pd.DataFrame(columns=["AdID_left", "AdID_right"])

    return pd.DataFrame(
        {
            "AdID_left": np.concatenate(left_chunks),
            "AdID_right": np.concatenate(right_chunks),
        }
    )


def _similar_category_location_price_ad_table(ads_info_df: pd.DataFrame) -> Table:
    return Table(
        df=_make_similar_category_location_price_ad_df(ads_info_df),
        fkey_col_to_pkey_table={
            "AdID_left": "AdsInfo",
            "AdID_right": "AdsInfo",
        },
        pkey_col=None,
        time_col=None,
    )


class AvitoDataset(Dataset):
    """Original data source:
    https://www.kaggle.com/competitions/avito-context-ad-clicks"""

    # search stream ranges from 2015-04-25 to 2015-05-20
    val_timestamp = pd.Timestamp("2015-05-08")
    test_timestamp = pd.Timestamp("2015-05-14")

    def make_db(self) -> Database:
        # subsampled version of the original dataset
        # Customize path as necessary
        r"""Process the raw files into a database."""
        url = "https://relbench.stanford.edu/data/rel-avito-raw-100k.zip"
        path = pooch.retrieve(
            url,
            known_hash="ad4fc1789d8a5073ea449049888c671899525c9a8a42359ca75d1f17d04d7929",
            progressbar=True,
            processor=unzip_processor,
        )
        path = os.path.join(path, "avito_100k_integ_test")

        # Define table names
        ads_info = os.path.join(path, "AdsInfo")
        category = os.path.join(path, "Category")
        location = os.path.join(path, "Location")
        phone_requests_stream = os.path.join(path, "PhoneRequestsStream")
        search_info = os.path.join(path, "SearchInfo")
        search_stream = os.path.join(path, "SearchStream")
        user_info = os.path.join(path, "UserInfo")
        visit_stream = os.path.join(path, "VisitStream")
        if not os.path.exists(ads_info):
            raise RuntimeError(
                self.err_msg.format(data="Dataset", url=self.url, path=path)
            )

        # Load table as pandas dataframes
        ads_info_df = pd.read_parquet(ads_info)
        ads_info_df.dropna(subset=["AdID"], inplace=True)
        # Params column contains a dictionary of type Dict[int, str].
        # Drop it for now since we can not handle this column type yet.
        ads_info_df.drop(columns=["Params"], inplace=True)
        ads_info_df["Title"].fillna("", inplace=True)
        category_df = pd.read_parquet(category)
        location_df = pd.read_parquet(location)
        location_df.dropna(subset=["LocationID"], inplace=True)
        phone_requests_stream_df = pd.read_parquet(phone_requests_stream)
        search_info_df = pd.read_parquet(search_info)
        # SearchParams column contains a dictionary of type Dict[int, str].
        # Drop it for now since we can not handle this column type yet.
        search_info_df.drop(columns=["SearchParams"], inplace=True)
        search_stream_df = pd.read_parquet(search_stream)
        user_info_df = pd.read_parquet(user_info)
        visit_stream_df = pd.read_parquet(visit_stream)
        search_info_df = clean_datetime(search_info_df, "SearchDate")
        search_stream_df = clean_datetime(search_stream_df, "SearchDate")
        phone_requests_stream_df = clean_datetime(
            phone_requests_stream_df, "PhoneRequestDate"
        )
        visit_stream_df = clean_datetime(visit_stream_df, "ViewDate")

        category_df.drop(columns=["__index_level_0__"], inplace=True)

        tables = {}
        tables["AdsInfo"] = Table(
            df=ads_info_df,
            fkey_col_to_pkey_table={
                "LocationID": "Location",
                "CategoryID": "Category",
            },
            pkey_col="AdID",
        )
        tables["Category"] = Table(
            df=category_df,
            fkey_col_to_pkey_table={},
            pkey_col="CategoryID",
        )
        tables["Location"] = Table(
            df=location_df,
            fkey_col_to_pkey_table={},
            pkey_col="LocationID",
        )
        tables["PhoneRequestsStream"] = Table(
            df=phone_requests_stream_df,
            fkey_col_to_pkey_table={
                "UserID": "UserInfo",
                "AdID": "AdsInfo",
            },
            time_col="PhoneRequestDate",
        )
        tables["SearchInfo"] = Table(
            df=search_info_df,
            fkey_col_to_pkey_table={
                "UserID": "UserInfo",
                "LocationID": "Location",
                "CategoryID": "Category",
            },
            pkey_col="SearchID",
            time_col="SearchDate",
        )
        tables["SearchStream"] = Table(
            df=search_stream_df,
            fkey_col_to_pkey_table={
                "SearchID": "SearchInfo",
                "AdID": "AdsInfo",
            },
            time_col="SearchDate",
        )
        tables["UserInfo"] = Table(
            df=user_info_df,
            fkey_col_to_pkey_table={},
            pkey_col="UserID",
        )
        tables["VisitStream"] = Table(
            df=visit_stream_df,
            fkey_col_to_pkey_table={
                "UserID": "UserInfo",
                "AdID": "AdsInfo",
            },
            time_col="ViewDate",
        )
        tables["similar_category_location_price_ad"] = (
            _similar_category_location_price_ad_table(ads_info_df)
        )
        db = Database(tables)

        db = db.from_(pd.Timestamp("2015-04-25"))

        return db

    def get_db(self, upto_test_timestamp=True) -> Database:
        db = super().get_db(upto_test_timestamp=upto_test_timestamp)
        if "similar_category_location_price_ad" not in db.table_dict:
            db.table_dict["similar_category_location_price_ad"] = (
                _similar_category_location_price_ad_table(
                    db.table_dict["AdsInfo"].df
                )
            )
        return db
