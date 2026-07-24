import os
import shutil
from pathlib import Path

import pandas as pd

from relbench.base import Database, Dataset, Table


def _make_same_product_code_df(articles_df: pd.DataFrame) -> pd.DataFrame:
    pair_list = []
    for _, group in articles_df.groupby("product_code", sort=False):
        article_ids = sorted(group["article_id"].dropna().unique())
        if len(article_ids) < 2:
            continue
        for left_idx, article_id_left in enumerate(article_ids[:-1]):
            for article_id_right in article_ids[left_idx + 1 :]:
                pair_list.append((article_id_left, article_id_right))

    return pd.DataFrame(
        pair_list,
        columns=["article_id_left", "article_id_right"],
    )


def _same_product_code_table(articles_df: pd.DataFrame) -> Table:
    return Table(
        df=_make_same_product_code_df(articles_df),
        fkey_col_to_pkey_table={
            "article_id_left": "article",
            "article_id_right": "article",
        },
        pkey_col=None,
        time_col=None,
    )


class HMDataset(Dataset):
    url = (
        "https://www.kaggle.com/competitions/"
        "h-and-m-personalized-fashion-recommendations"
    )

    val_timestamp = pd.Timestamp("2020-09-07")
    test_timestamp = pd.Timestamp("2020-09-14")

    def make_db(self) -> Database:
        path = os.path.join("data", "hm-recommendation")
        zip = os.path.join(path, "h-and-m-personalized-fashion-recommendations.zip")
        customers = os.path.join(path, "customers.csv")
        articles = os.path.join(path, "articles.csv")
        transactions = os.path.join(path, "transactions_train.csv")
        if not os.path.exists(customers):
            if not os.path.exists(zip):
                raise RuntimeError(
                    f"Dataset not found. Please download "
                    f"h-and-m-personalized-fashion-recommendations.zip from "
                    f"'{self.url}' and move it to '{path}'. Once you have your"
                    f"Kaggle API key, you can use the following command: "
                    f"kaggle competitions download -c h-and-m-personalized-fashion-recommendations"
                )
            else:
                print("Unpacking")
                shutil.unpack_archive(zip, Path(zip).parent)

        articles_df = pd.read_csv(articles)
        customers_df = pd.read_csv(customers)
        transactions_df = pd.read_csv(transactions)
        transactions_df["t_dat"] = pd.to_datetime(
            transactions_df["t_dat"], format="%Y-%m-%d"
        )

        db = Database(
            table_dict={
                "article": Table(
                    df=articles_df,
                    fkey_col_to_pkey_table={},
                    pkey_col="article_id",
                ),
                "customer": Table(
                    df=customers_df,
                    fkey_col_to_pkey_table={},
                    pkey_col="customer_id",
                ),
                "transactions": Table(
                    df=transactions_df,
                    fkey_col_to_pkey_table={
                        "customer_id": "customer",
                        "article_id": "article",
                    },
                    time_col="t_dat",
                ),
                "same_product_code": _same_product_code_table(articles_df),
            }
        )

        db = db.from_(pd.Timestamp("2019-09-07"))

        return db

    def get_db(self, upto_test_timestamp=True) -> Database:
        db = super().get_db(upto_test_timestamp=upto_test_timestamp)
        if "same_product_code" not in db.table_dict:
            db.table_dict["same_product_code"] = _same_product_code_table(
                db.table_dict["article"].df
            )
        return db
