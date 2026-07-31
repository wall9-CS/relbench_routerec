"""rel-hm 의 product_code 그룹을 들여다보는 진단 스크립트.

modes
-----
groups   : 그룹 크기 분포 + 실제 그룹 샘플 출력 (같은 product_code 인 article 들이 뭔지)
columns  : 그룹 내에서 어떤 컬럼이 변하고 어떤 컬럼이 고정인지 (엣지의 의미 파악)
signal   : val ground-truth 중 "과거 구매 article 과 product_code 를 공유하는" 비율.
           = 4-layer 에서 same_product_code 엣지가 새로 열어주는 reachability 상한.
all      : 위 셋 다

usage
-----
    python inspect_same_product_code.py --mode all
    python inspect_same_product_code.py --mode groups --num_samples 15 --min_size 4
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from relbench.datasets import get_dataset
from relbench.tasks import get_task

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 50)


# --------------------------------------------------------------------------- #
def mode_groups(article: pd.DataFrame, num_samples: int, min_size: int, seed: int):
    sizes = article.groupby("product_code").size()

    print("\n=== product_code 그룹 크기 분포 ===")
    print(f"#articles          : {len(article):,}")
    print(f"#product_codes     : {len(sizes):,}")
    print(f"mean group size    : {sizes.mean():.2f}")
    print(f"median group size  : {sizes.median():.0f}")
    print(f"max group size     : {sizes.max()}")
    print(f"singleton groups   : {(sizes == 1).sum():,} ({(sizes == 1).mean()*100:.1f}%)")
    print(
        f"articles in singleton groups : "
        f"{sizes[sizes == 1].sum():,} ({sizes[sizes == 1].sum()/len(article)*100:.1f}%)"
    )
    print("\nquantiles:")
    print(sizes.quantile([0.5, 0.75, 0.9, 0.95, 0.99, 1.0]).to_string())
    print("\nsize histogram (상위 15):")
    print(sizes.value_counts().sort_index().head(15).to_string())

    # complete graph 로 연결했을 때 생길 엣지 수
    n = sizes.values.astype(np.int64)
    print(f"\ncomplete-graph pair 수 (sum n(n-1)/2): {(n * (n - 1) // 2).sum():,}")

    show_cols = [
        c
        for c in [
            "article_id",
            "prod_name",
            "colour_group_name",
            "perceived_colour_master_name",
            "graphical_appearance_name",
            "product_type_name",
            "department_name",
            "index_name",
        ]
        if c in article.columns
    ]

    print("\n=== 가장 큰 그룹 3개 ===")
    for pc in sizes.sort_values(ascending=False).head(3).index:
        g = article[article["product_code"] == pc]
        print(f"\n--- product_code={pc}  (n={len(g)}) ---")
        print(g[show_cols].head(20).to_string(index=False))
        if len(g) > 20:
            print(f"... ({len(g) - 20} more)")

    rng = np.random.default_rng(seed)
    cand = sizes[sizes >= min_size].index.values
    if len(cand) == 0:
        print(f"\n(min_size={min_size} 이상인 그룹 없음)")
        return
    picks = rng.choice(cand, size=min(num_samples, len(cand)), replace=False)
    print(f"\n=== 랜덤 그룹 샘플 (size >= {min_size}) ===")
    for pc in picks:
        g = article[article["product_code"] == pc]
        print(f"\n--- product_code={pc}  (n={len(g)}) ---")
        print(g[show_cols].to_string(index=False))


# --------------------------------------------------------------------------- #
def mode_columns(article: pd.DataFrame):
    """그룹 내에서 무엇이 변하는가 = 이 엣지가 무슨 관계를 의미하는가."""
    multi = article[article.groupby("product_code")["product_code"].transform("size") > 1]
    print(f"\n=== 그룹 내 컬럼 variability (n>1 그룹만, {len(multi):,} articles) ===")
    rows = []
    for col in article.columns:
        if col in ("article_id", "product_code"):
            continue
        try:
            nun = multi.groupby("product_code")[col].nunique(dropna=False)
        except TypeError:
            continue
        rows.append(
            {
                "column": col,
                "mean_nunique_in_group": round(nun.mean(), 3),
                "pct_groups_constant": round((nun <= 1).mean() * 100, 1),
            }
        )
    df = pd.DataFrame(rows).sort_values("mean_nunique_in_group", ascending=False)
    print(df.to_string(index=False))
    print(
        "\n해석: pct_groups_constant 가 100 에 가까운 컬럼 = product_code 가 이미 결정하는 속성.\n"
        "      mean_nunique 가 큰 컬럼 = 그룹 내 article 을 구분짓는 축 (보통 colour 계열)."
    )


# --------------------------------------------------------------------------- #
def mode_signal(dataset, task, db, article: pd.DataFrame, split: str = "val"):
    """same_product_code 엣지가 실제로 정답을 데려오는가?

    4-layer 에서 customer -> transactions -> article -> spc -> article' 로 도달 가능한
    article' 집합 = {과거 구매 article 과 product_code 를 공유하는 모든 article}.
    따라서 아래 'product_code 공유' 비율이 이 엣지가 기여하는 reachability 상한이다.
    """
    ts = dataset.val_timestamp if split == "val" else dataset.test_timestamp
    tx = db.table_dict["transactions"].df
    tx = tx[tx["t_dat"] <= ts]

    # article_id 는 reindex 되어 0..N-1 -> 배열 인덱싱으로 product_code 조회
    pcode_codes, _ = pd.factorize(article["product_code"])
    pcode_of_article = np.asarray(pcode_codes, dtype=np.int64)  # article_id -> pcode idx
    n_pcode = pcode_of_article.max() + 1
    n_article = len(article)

    cust = tx["customer_id"].astype("int64").to_numpy()
    art = tx["article_id"].astype("int64").to_numpy()
    del tx

    # 과거 구매 (customer, article) / (customer, product_code) 집합
    hist_art = np.unique(cust * np.int64(n_article) + art)
    hist_pc = np.unique(cust * np.int64(n_pcode) + pcode_of_article[art])
    del cust, art

    table = task.get_table(split)
    df = table.df
    exploded = df[[task.src_entity_col, task.dst_entity_col]].explode(task.dst_entity_col)
    exploded = exploded.dropna()
    gt_cust = exploded[task.src_entity_col].astype("int64").to_numpy()
    gt_art = exploded[task.dst_entity_col].astype("int64").to_numpy()

    def isin_sorted(keys, sorted_pool):
        pos = np.searchsorted(sorted_pool, keys)
        pos = np.clip(pos, 0, len(sorted_pool) - 1)
        return sorted_pool[pos] == keys

    hit_art = isin_sorted(gt_cust * np.int64(n_article) + gt_art, hist_art)
    hit_pc = isin_sorted(
        gt_cust * np.int64(n_pcode) + pcode_of_article[gt_art], hist_pc
    )

    total = len(gt_art)
    print(f"\n=== [{split}] ground-truth {total:,} 건에 대한 도달성 ===")
    print(
        f"이미 구매한 적 있는 article (article-level repeat) : "
        f"{hit_art.sum():,} ({hit_art.mean()*100:.2f}%)"
    )
    print(
        f"과거 구매와 product_code 공유 (spc 로 도달 가능)    : "
        f"{hit_pc.sum():,} ({hit_pc.mean()*100:.2f}%)"
    )
    new = hit_pc & ~hit_art
    print(
        f"  └ 그 중 article-level repeat 이 아닌 신규 도달    : "
        f"{new.sum():,} ({new.mean()*100:.2f}%)   <-- spc 테이블의 순수 기여분"
    )
    print(
        f"둘 다 아님 (spc 로도 도달 불가)                     : "
        f"{(~hit_pc).sum():,} ({(~hit_pc).mean()*100:.2f}%)"
    )

    # customer 당 후보 팽창률
    per_cust_pc = np.bincount(
        (hist_pc // n_pcode).astype(np.int64), minlength=0
    )
    sizes = article.groupby("product_code").size()
    print(
        f"\n참고: 평균 그룹 크기 {sizes.mean():.2f} 이므로, "
        f"과거 구매 article 1개당 평균 {sizes.mean()-1:.2f} 개의 형제 article 이 "
        f"후보로 추가된다 (customer 당 과거 구매 product_code 수 중앙값 "
        f"{np.median(per_cust_pc[per_cust_pc > 0]):.0f})."
    )


# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--mode", type=str, default="all",
                   choices=["groups", "columns", "signal", "all"])
    p.add_argument("--split", type=str, default="val", choices=["val", "test"])
    p.add_argument("--num_samples", type=int, default=10)
    p.add_argument("--min_size", type=int, default=3)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    dataset = get_dataset("rel-hm", download=True)
    db = dataset.get_db()
    article = db.table_dict["article"].df

    if args.mode in ("groups", "all"):
        mode_groups(article, args.num_samples, args.min_size, args.seed)
    if args.mode in ("columns", "all"):
        mode_columns(article)
    if args.mode in ("signal", "all"):
        task = get_task("rel-hm", "user-item-purchase", download=True)
        mode_signal(dataset, task, db, article, split=args.split)