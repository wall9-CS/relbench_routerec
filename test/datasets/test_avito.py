import pandas as pd
from torch_frame import stype

from relbench.base import Database, Table
from relbench.datasets.avito import _similar_category_location_price_ad_table
from relbench.modeling.graph import make_pkey_fkey_graph


def test_similar_category_location_price_ad_table_reindexes_ad_pairs():
    ads_info_df = pd.DataFrame(
        {
            "AdID": [100, 300, 200, 700, 400, 500],
            "CategoryID": [1, 1, 1, 1, 1, 2],
            "LocationID": [10, 10, 10, 10, 11, 10],
            "Price": [100.0, 105.0, 1000.0, 106.0, 102.0, 103.0],
        }
    )
    db = Database(
        table_dict={
            "AdsInfo": Table(
                df=ads_info_df,
                fkey_col_to_pkey_table={},
                pkey_col="AdID",
            ),
            "similar_category_location_price_ad": (
                _similar_category_location_price_ad_table(ads_info_df)
            ),
        }
    )

    db.reindex_pkeys_and_fkeys()

    similar_ads = db.table_dict["similar_category_location_price_ad"]
    assert list(similar_ads.df.columns) == [
        "AdID_left",
        "AdID_right",
    ]
    assert similar_ads.fkey_col_to_pkey_table == {
        "AdID_left": "AdsInfo",
        "AdID_right": "AdsInfo",
    }
    assert similar_ads.df.values.tolist() == [
        [0, 1],
        [1, 3],
    ]
    assert (similar_ads.df["AdID_left"] != similar_ads.df["AdID_right"]).all()

    data, _ = make_pkey_fkey_graph(
        db,
        col_to_stype_dict={
            "AdsInfo": {
                "AdID": stype.numerical,
                "CategoryID": stype.numerical,
                "LocationID": stype.numerical,
                "Price": stype.numerical,
            },
            "similar_category_location_price_ad": {
                "AdID_left": stype.numerical,
                "AdID_right": stype.numerical,
            },
        },
    )
    assert (
        "similar_category_location_price_ad",
        "f2p_AdID_left",
        "AdsInfo",
    ) in data.edge_types
    assert (
        "AdsInfo",
        "rev_f2p_AdID_left",
        "similar_category_location_price_ad",
    ) in data.edge_types
    assert (
        "similar_category_location_price_ad",
        "f2p_AdID_right",
        "AdsInfo",
    ) in data.edge_types
    assert (
        "AdsInfo",
        "rev_f2p_AdID_right",
        "similar_category_location_price_ad",
    ) in data.edge_types
