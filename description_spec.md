# `spec.md`의 핵심 설계

이 spec의 목적은 **학습된 ID-GNN은 전혀 변경하지 않고**, 테스트 시점에만 ID-GNN이 점수를 매길 수 있는 article 후보를 늘리는 것이다.

기존 그래프에서 고객이 구매한 article이 다음과 같이 연결되어 있다고 하자.

```text
customer_a
    └── transaction_10
            └── article_a
```

`article_a.product_code == article_b.product_code`이고, 고객이 `article_b`를 과거에 구매한 적이 없다면 테스트 그래프에 다음 node를 추가한다.

```text
customer_a
    ├── transaction_10
    │       └── article_a
    │
    └── synthetic_transaction_100
            └── article_b
```

`synthetic_transaction_100`의 feature와 time은 `transaction_10`을 그대로 복제한다.

```text
price            = transaction_10.price
sales_channel_id = transaction_10.sales_channel_id
t_dat            = transaction_10.t_dat
기타 feature      = 모두 동일
```

이렇게 하면 `article_b`가 고객으로부터 2-hop 거리에 들어오므로, 기존 ID-GNN의 다음 로직이 별도 변경 없이 `article_b`에 대한 pair-wise score를 생성할 수 있다.

```text
customer_a → synthetic transaction → article_b
```

---

## 1. 왜 raw DataFrame을 복제하지 않고 TensorFrame을 복제하는가

이전 설명에서는 transaction DataFrame을 복제한 후 전체 test graph를 다시 materialize하는 방법도 가능하다고 했다. 하지만 최종 spec에서는 더 안전한 방식으로 바꿨다.

### 최종 spec의 방식

```text
원본 DB
  ↓ make_pkey_fkey_graph
base HeteroData
  ↓ transaction TensorFrame row 직접 복제
augmented test HeteroData
```

즉, 이미 변환이 끝난 다음을 직접 복제한다.

```python
data["transactions"].tf[source_tx_pos]
data["transactions"].time[source_tx_pos]
```

이 방식을 택한 이유는 transaction의 categorical encoding이 정확히 유지되기 때문이다.

예를 들어 `sales_channel_id`가 categorical feature라면 PyTorch Frame이 다음과 같은 내부 ID로 변환했을 수 있다.

```text
sales_channel_id 1 → category index 0
sales_channel_id 2 → category index 1
```

test graph를 별도로 materialize하면 통계나 categorical mapping이 달라질 위험이 있다. 반면 materialized TensorFrame row 자체를 복제하면 학습 때 사용한 representation과 완전히 같다.

```python
synthetic_tf = base_transaction_tf[source_pos]
```

이 방식은 다음 모든 값을 자동으로 보존한다.

* numerical feature
* categorical feature의 내부 index
* timestamp feature encoding
* text feature 또는 embedding
* 누락값 표현
* 이후 transaction table에 feature가 추가되더라도 전체 row

따라서 spec에서는 `price`, `t_dat`, `sales_channel_id`만 수동으로 복제하지 말고 **전체 TensorFrame row를 복제하라**고 명시했다.

---

# 2. 구현을 두 단계로 분리한 이유

구현은 크게 두 함수로 나뉜다.

```text
1. 어떤 virtual transaction을 만들지 결정
2. 실제 HeteroData에 node와 edge를 추가
```

권장 파일은 다음이다.

```text
examples/hm_product_code_expansion.py
```

이 파일에 두 로직을 모두 넣되 서로 독립적으로 테스트할 수 있도록 한다.

---

## 2.1 Candidate 생성 함수

예상 API는 다음과 같다.

```python
def build_product_code_virtual_candidates(
    db: Database,
    test_table: Table,
    *,
    src_col: str = "customer_id",
    article_col: str = "article_id",
    product_code_col: str = "product_code",
    transaction_time_col: str = "t_dat",
    seed_time_col: str = "timestamp",
    max_virtual_per_customer: int = 0,
) -> tuple[pd.DataFrame, ProductCodeExpansionStats]:
    ...
```

이 함수는 아직 graph를 수정하지 않는다. 다음과 같은 명세 DataFrame만 만든다.

| customer_id | source_article_id | target_article_id | product_code | source_tx_pos | source_t_dat |
| ----------: | ----------------: | ----------------: | -----------: | ------------: | ------------ |
|          12 |               100 |               101 |           50 |          3021 | 2020-09-10   |
|          12 |               100 |               102 |           50 |          3021 | 2020-09-10   |
|          37 |               500 |               505 |          200 |          9182 | 2020-09-13   |

각 row는 다음을 뜻한다.

```text
source_tx_pos=3021인 transaction node의 feature를 복제해서
customer 12와 target article 101을 연결한다.
```

---

# 3. Candidate 생성 과정

spec은 메모리 사용과 leakage 방지를 위해 처리 순서를 엄격하게 지정한다.

```text
1. test customer만 선택
2. seed time 이전 transaction만 선택
3. transaction에 product_code 결합
4. customer-product별 최신 transaction 하나 선택
5. 동일 product_code article 확장
6. 이미 본 article 제거
7. 중복 제거
8. customer별 cap 적용
```

순서가 중요한 이유를 단계별로 설명하면 다음과 같다.

---

## 3.1 Test seed time 확인

rel-hm test task는 고정된 test timestamp를 전제로 한다.

```python
seed_times = test_table.df["timestamp"].dropna().unique()
```

spec에서는 반드시 unique timestamp가 하나인지 확인하도록 했다.

```python
if len(seed_times) != 1:
    raise ValueError(...)
```

이유는 augmentation이 다음 의미를 가져야 하기 때문이다.

```text
test seed time T에서 알 수 있는 과거 transaction만 사용한다.
```

여러 timestamp가 섞인 경우 고객마다 허용되는 과거 transaction 범위가 달라지므로 단일 augmented graph를 그대로 사용할 수 없다. 현재 실험의 범위를 명확하게 제한하기 위해 고정 timestamp 하나만 허용한다.

---

## 3.2 Test customer만 처리

전체 rel-hm 고객을 확장할 필요가 없다. 실제 테스트 source row에 등장하는 customer만 처리한다.

```python
test_customers = (
    test_table.df["customer_id"]
    .dropna()
    .drop_duplicates()
)
```

그 후 transaction을 먼저 줄인다.

```python
historical_tx = transactions[
    transactions["customer_id"].isin(test_customers)
]
```

이 필터가 중요한 이유는 rel-hm transaction table이 크기 때문이다. 전체 고객에 대해 product variant를 생성하면 실제 test loader가 사용하지 않는 synthetic node가 대량 생성된다.

---

## 3.3 미래 transaction 제거

각 source transaction은 다음 조건을 만족해야 한다.

```python
transaction.t_dat <= test_seed_time
```

즉, 예측 시점 이후의 구매는 절대 사용하지 않는다.

```text
허용:
2020-09-10 transaction
test seed = 2020-09-14

금지:
2020-09-16 transaction
test seed = 2020-09-14
```

이 조건을 지키지 않으면 test future information이 graph에 들어가므로 명백한 temporal leakage가 된다.

---

## 3.4 `source_tx_pos`가 중요한 이유

그래프의 transaction node ID는 transaction DataFrame의 **행 위치**에 대응한다.

따라서 source transaction을 식별할 때 pandas index가 아니라 positional index를 사용해야 한다.

```python
transactions_df = transactions_df.copy()
transactions_df["source_tx_pos"] = np.arange(len(transactions_df))
```

예를 들어 filtering 후 pandas index가 다음과 같을 수 있다.

```text
index:          100, 205, 999
graph node ID:    0,   1,   2
```

pandas index `205`를 TensorFrame row index로 사용하면 완전히 잘못된 node를 복제할 수 있다.

따라서 spec은 다음을 금지한다.

```python
source_tx_pos = filtered_df.index
```

그리고 다음을 요구한다.

```python
source_tx_pos = np.arange(len(original_transactions_df))
```

이 positional ID가 이후 다음 코드에 그대로 사용된다.

```python
data["transactions"].tf[source_pos]
data["transactions"].time[source_pos]
```

---

## 3.5 Transaction에 product code 연결

transaction table에는 `product_code`가 없으므로 source article을 통해 붙인다.

```python
historical_tx = historical_tx.merge(
    articles[["article_id", "product_code"]],
    on="article_id",
    how="inner",
)
```

`product_code`가 null이면 확장 기준이 없으므로 제외한다.

```python
historical_tx = historical_tx.dropna(subset=["product_code"])
```

---

## 3.6 고객-product별 최신 transaction 하나 선택

한 고객이 동일 product의 여러 variant를 여러 번 구매했을 수 있다.

예:

| customer | article | product_code | t_dat      |
| -------- | ------: | -----------: | ---------- |
| A        |     101 |           50 | 2020-08-01 |
| A        |     102 |           50 | 2020-09-01 |
| A        |     103 |           50 | 2020-09-10 |

이 세 transaction을 모두 source로 사용하면 같은 target article에 대해 synthetic transaction이 여러 개 생길 수 있다.

spec에서는 다음 group마다 하나만 사용한다.

```text
(customer_id, product_code)
```

선택 기준은 다음이다.

1. 가장 최신 `t_dat`
2. 시간이 같으면 가장 큰 `source_tx_pos`

```python
source_tx = (
    historical_tx
    .sort_values(
        ["customer_id", "product_code", "t_dat", "source_tx_pos"]
    )
    .drop_duplicates(
        ["customer_id", "product_code"],
        keep="last",
    )
)
```

위 예에서는 2020-09-10 transaction만 source가 된다.

### 왜 최신 transaction인가

재사용할 transaction feature에 다음과 같은 시점 관련 정보가 포함되기 때문이다.

* 최근 가격
* 최근 sales channel
* 최근 transaction timestamp
* temporal encoder가 계산할 seed time과의 거리

가장 최근 구매 context가 recommendation 시점과 가장 관련성이 높다고 보는 정책이다.

---

## 3.7 동일 product code article 확장

선택한 source transaction을 article table과 `product_code`로 조인한다.

```python
candidates = source_tx.merge(
    article_variants,
    on="product_code",
    how="inner",
)
```

예를 들어 article table이 다음과 같다면:

| article_id | product_code |
| ---------: | -----------: |
|        101 |           50 |
|        102 |           50 |
|        103 |           50 |
|        104 |           50 |

source article이 103이면 초기 candidate는 다음이다.

```text
101, 102, 103, 104
```

여기서 source article 자체인 103은 제거한다.

```python
candidates = candidates[
    candidates["target_article_id"]
    != candidates["source_article_id"]
]
```

---

## 3.8 고객이 이미 구매한 sibling 제거

고객이 `article_101`을 이미 구매했다면 해당 article은 실제 transaction 경로로 이미 graph에 존재한다.

```text
customer → real transaction → article_101
```

이 article에 virtual transaction까지 추가하면 동일한 historical item에 불필요한 경로가 하나 더 생긴다.

따라서 test seed time 이전 historical seen pair를 만든다.

```python
seen_pairs = historical_tx[
    ["customer_id", "article_id"]
].drop_duplicates()
```

그리고 candidate와 anti-join한다.

```text
candidate - historical seen pairs
```

결과적으로 virtual transaction은 오직 다음 article에만 추가된다.

```text
동일 product_code이지만 그 고객이 아직 구매하지 않은 article
```

---

## 3.9 Test target label을 읽지 않는 이유

test task table에는 evaluation용 destination article 목록이 존재할 수 있다.

```python
test_table.df["article_id"]
```

하지만 candidate generation에서는 이 column을 절대 읽으면 안 된다.

사용 가능한 test table 정보는 다음뿐이다.

```python
test_table.df["customer_id"]
test_table.df["timestamp"]
```

금지되는 정보는 다음이다.

```python
test_table.df["article_id"]
```

왜냐하면 test destination은 실제 정답이므로 이를 보고 candidate를 만들면 target leakage가 된다.

spec에서 이 부분을 명시적으로 테스트하도록 했다. 예를 들어 test label을 완전히 다른 값으로 바꿔도 생성되는 candidates가 같아야 한다.

---

## 3.10 Candidate 중복 제거

한 customer-target article pair가 여러 source에서 생성될 가능성이 있다.

최종적으로 다음 key는 unique해야 한다.

```text
(customer_id, target_article_id)
```

중복이 있다면 다음 순서로 source를 고른다.

1. 최신 source `t_dat`
2. 가장 큰 `source_tx_pos`
3. `target_article_id` 오름차순은 결과 정렬 안정성을 위한 tie-break

이렇게 해야 동일 입력에서 항상 동일 synthetic node가 생성된다.

---

## 3.11 Customer별 virtual node cap

동일 product code에 article이 매우 많으면 한 고객에게 수백 또는 수천 개 후보가 생길 수 있다.

이를 제어하는 인자가 다음이다.

```bash
--product_code_max_virtual_per_customer
```

의미는 다음과 같다.

```text
0  → 제한 없음
32 → 고객당 최대 32개
64 → 고객당 최대 64개
```

cap 전에 정렬 순서를 고정했다.

```text
customer_id          ascending
source_t_dat         descending
source_tx_pos        descending
target_article_id    ascending
```

따라서 cap을 적용해도 매 실행마다 같은 후보가 선택된다.

---

# 4. 실제 그래프 확장

Candidate DataFrame이 만들어지면 두 번째 함수가 `HeteroData`를 확장한다.

예상 API:

```python
def augment_hm_graph_with_virtual_transactions(
    data: HeteroData,
    candidates: pd.DataFrame,
) -> HeteroData:
    ...
```

---

## 4.1 Synthetic transaction ID 할당

기존 transaction node 수가 `N`이라면 새 node ID는 다음과 같다.

```text
기존 transaction IDs:    0 ... N-1
synthetic transaction:   N ... N+M-1
```

구현은 다음과 같다.

```python
base_num_transactions = len(data["transactions"].tf)

synthetic_tx_ids = (
    base_num_transactions
    + torch.arange(len(candidates), dtype=torch.long)
)
```

Candidate DataFrame의 row 순서가 synthetic ID를 결정한다.

```text
candidate row 0 → transaction ID N
candidate row 1 → transaction ID N+1
candidate row 2 → transaction ID N+2
```

그래서 candidate 정렬의 deterministic 조건이 중요하다.

---

## 4.2 Transaction feature 복제

각 candidate가 가리키는 `source_tx_pos`를 tensor로 만든다.

```python
source_pos = torch.as_tensor(
    candidates["source_tx_pos"].to_numpy(),
    dtype=torch.long,
)
```

그다음 기존 transaction TensorFrame에서 해당 row를 조회한다.

```python
synthetic_tf = data["transactions"].tf[source_pos]
```

그리고 기존 TensorFrame 뒤에 붙인다.

```python
augmented_tf = torch_frame.cat(
    [
        data["transactions"].tf,
        synthetic_tf,
    ],
    dim=0,
)
```

예를 들어 source positions가 `[5, 5, 10]`이면:

```text
synthetic node N     = transaction row 5 복제
synthetic node N+1   = transaction row 5 복제
synthetic node N+2   = transaction row 10 복제
```

동일 source transaction을 여러 sibling article에 사용할 수 있으므로 동일 row가 여러 번 복제되는 것은 정상이다.

---

## 4.3 Transaction time 복제

PyG의 temporal neighbor sampling은 transaction store의 `time`을 사용한다.

따라서 TensorFrame뿐 아니라 time도 복제해야 한다.

```python
synthetic_time = data["transactions"].time[source_pos]

augmented_time = torch.cat(
    [
        data["transactions"].time,
        synthetic_time,
    ],
    dim=0,
)
```

중요한 점은 synthetic transaction 시간을 test seed time으로 설정하지 않는다는 것이다.

잘못된 예:

```python
synthetic_time = test_seed_time
```

올바른 예:

```python
synthetic_time = source_transaction_time
```

즉, virtual transaction은 “새로 일어난 구매”가 아니라 기존 transaction의 context를 다른 variant candidate로 확장하는 구조다.

---

# 5. 네 종류 edge를 모두 추가하는 이유

기존 graph에는 transaction과 customer/article 사이에 정방향과 역방향 edge가 모두 있다.

```python
("transactions", "f2p_customer_id", "customer")
("customer", "rev_f2p_customer_id", "transactions")
("transactions", "f2p_article_id", "article")
("article", "rev_f2p_article_id", "transactions")
```

Candidate 하나에 대해 네 edge를 추가한다.

예를 들어:

```text
synthetic_tx_id = 1000
customer_id = 12
target_article_id = 101
```

추가 edge는 다음과 같다.

### Transaction → Customer

```text
1000 → 12
```

```python
[synthetic_tx_id, customer_id]
```

### Customer → Transaction

```text
12 → 1000
```

```python
[customer_id, synthetic_tx_id]
```

### Transaction → Article

```text
1000 → 101
```

```python
[synthetic_tx_id, target_article_id]
```

### Article → Transaction

```text
101 → 1000
```

```python
[target_article_id, synthetic_tx_id]
```

ID-GNN loader가 `subgraph_type="bidirectional"`을 사용하지만, 원본 heterogeneous graph schema 자체도 양방향 FK edge type을 갖고 있으므로 동일한 구조를 유지해야 한다.

각 edge tensor는 기존 tensor 뒤에 붙인 뒤 정렬한다.

```python
new_edge_index = torch.cat(
    [old_edge_index, synthetic_edge_index],
    dim=1,
)

new_edge_index = sort_edge_index(new_edge_index)
```

---

# 6. 기존 transaction node를 그대로 공유하지 않는 이유

다음 구조는 만들지 않는다.

```text
transaction_10
    ├── article_a
    └── article_b
```

원본 rel-hm graph에서 transaction node는 article 하나를 가리킨다. 기존 node 하나에 article을 여러 개 연결하면 test 시점에 transaction node의 degree와 의미가 바뀐다.

특히 GNN message passing에서 transaction node가 두 article의 embedding을 모두 받게 된다.

```text
article_a → transaction_10
article_b → transaction_10
```

이렇게 되면 `transaction_10` representation 자체가 기존 학습 분포와 달라진다.

반면 transaction node를 복제하면 각 node는 계속 다음 구조를 유지한다.

```text
customer 1개 ↔ transaction 1개 ↔ article 1개
```

즉, 모델이 학습 때 본 schema motif를 유지하면서 candidate만 추가한다.

---

# 7. 새로운 `same_product_code` edge를 만들지 않는 이유

다음 edge를 test 시점에 추가하는 방법도 생각할 수 있다.

```text
article_a --same_product_code--> article_b
```

하지만 모델의 GNN layer는 초기화할 때 기존 `data.edge_types`를 기준으로 edge-type별 convolution parameter를 만든다.

test에만 새로운 relation을 추가하면:

```text
("article", "same_product_code", "article")
```

이 edge를 위한 학습된 parameter가 없다.

반면 synthetic transaction 방식은 기존 relation만 사용한다.

```text
customer ↔ transactions ↔ article
```

따라서 새로운 GNN parameter가 필요하지 않고 기존 checkpoint를 그대로 적용할 수 있다.

---

# 8. Base graph를 수정하지 않는 이유

학습, validation, test loader가 같은 graph 객체를 공유하는 구조에서는 test augmentation을 in-place로 수행하면 예상하지 못한 부작용이 생길 수 있다.

예:

```python
data["transactions"].tf = augmented_tf
```

이렇게 원본 객체를 바꾸면 이미 만들어진 val loader나 다른 reference가 augmented graph를 볼 가능성이 있다.

spec은 다음을 요구한다.

```text
base_data      : train/validation용, 절대 변경하지 않음
augmented_data : test 전용 별도 HeteroData
```

단, rel-hm graph는 크기 때문에 다음은 피한다.

```python
copy.deepcopy(base_data)
```

deep copy는 모든 TensorFrame과 edge tensor를 복제하므로 메모리가 크게 증가할 수 있다.

권장 방식은 새 `HeteroData` 컨테이너를 만들고 기존 store의 변경되지 않는 tensor는 reference로 공유하는 것이다.

개념적으로:

```python
augmented_data = shallow_structural_copy(base_data)
```

이후 변경 대상만 새 객체로 교체한다.

```python
augmented_data["transactions"].tf = augmented_tf
augmented_data["transactions"].time = augmented_time

augmented_data[customer_tx_edge].edge_index = new_customer_tx_edges
augmented_data[tx_customer_edge].edge_index = new_tx_customer_edges
augmented_data[article_tx_edge].edge_index = new_article_tx_edges
augmented_data[tx_article_edge].edge_index = new_tx_article_edges
```

변하지 않는 article/customer feature와 다른 edge는 기존 tensor를 공유해도 된다.

---

# 9. `idgnn_recommendation.py`의 실행 흐름 변경

기존 코드는 대략 다음 순서다.

```text
graph 생성
train/val/test loader 모두 생성
model 생성
training
validation
test
```

spec에 따른 새 흐름은 다음이다.

```text
base graph 생성
train loader 생성
val loader 생성
model 생성
training
best checkpoint 선택
best checkpoint load
test graph 생성
test loader 생성
test
```

중요한 차이는 **test augmentation이 학습이 완전히 끝난 뒤 실행된다는 것**이다.

---

## 9.1 Base graph 생성

```python
base_data, col_stats_dict = make_pkey_fkey_graph(...)
```

모델도 base graph schema를 기준으로 생성한다.

```python
model = Model(
    data=base_data,
    col_stats_dict=col_stats_dict,
    ...
)
```

---

## 9.2 Train/validation loader

초기 loop에서는 test를 제외한다.

```python
for split in ["train", "val"]:
    ...
```

두 loader 모두 `base_data`를 사용한다.

```python
NeighborLoader(
    base_data,
    ...
)
```

따라서 synthetic transaction은 training과 validation에 존재할 수 없다.

---

## 9.3 Best model 선택 이후 test graph 생성

다음 코드 뒤에서만 augmentation을 실행한다.

```python
model.load_state_dict(state_dict)
```

flag가 꺼져 있으면:

```python
test_data = base_data
```

flag가 켜져 있으면:

```python
candidates, stats = build_product_code_virtual_candidates(
    db,
    task.get_table("test"),
    ...
)

test_data = augment_hm_graph_with_virtual_transactions(
    base_data,
    candidates,
)
```

그리고 test loader만 `test_data`를 사용한다.

```python
test_loader = NeighborLoader(
    test_data,
    ...
)
```

---

# 10. CLI 옵션의 의미

## `--test_product_code_expansion`

기능을 활성화한다.

```bash
--test_product_code_expansion
```

기본값은 꺼짐이다. 이 flag 없이 실행하면 기존 baseline과 동일해야 한다.

```bash
python examples/idgnn_recommendation.py \
  --dataset rel-hm \
  --task user-item-purchase
```

---

## `--product_code_max_virtual_per_customer`

고객당 synthetic transaction 수를 제한한다.

```bash
--product_code_max_virtual_per_customer 32
```

의미:

```text
각 test customer에 최대 32개 variant article만 추가
```

`0`은 제한 없음이다.

```bash
--product_code_max_virtual_per_customer 0
```

---

## `--test_num_neighbors`

test loader의 sampling budget만 별도로 설정한다.

기존 2-layer 설정에서:

```bash
--num_neighbors 128
```

이면 schedule은 다음이다.

```python
[128, 64]
```

test에만 더 큰 budget을 주려면:

```bash
--test_num_neighbors 256
```

test schedule은 다음이다.

```python
[256, 128]
```

train/validation은 여전히:

```python
[128, 64]
```

이다.

---

# 11. Candidate cap과 neighbor budget은 서로 다른 역할

이 둘을 구분해야 한다.

## Candidate cap

```bash
--product_code_max_virtual_per_customer 64
```

graph에 실제로 추가되는 synthetic node 수를 제한한다.

```text
customer당 graph에 최대 64개 virtual transaction 추가
```

## Neighbor budget

```bash
--test_num_neighbors 256
```

추가된 node 중 NeighborLoader가 얼마나 sampling할 수 있는지를 제어한다.

graph에 100개의 virtual transaction이 있어도 첫 hop budget이 작으면 일부만 sampling된다.

```text
graph 후보 수 ≠ 실제 sampled 후보 수
```

실험은 보통 다음 두 방향으로 나눌 수 있다.

### Fixed-budget

```bash
--num_neighbors 128
--test_num_neighbors 128
```

기존 graph와 virtual transaction이 동일 sampling budget 안에서 경쟁한다.

### Expanded-budget

```bash
--num_neighbors 128
--test_num_neighbors 256
```

기존 후보를 밀어내는 효과를 줄이고 virtual candidate를 추가로 포함시킨다.

---

# 12. `temporal_strategy="last"`와의 관계

기본 설정은 다음이다.

```python
temporal_strategy="last"
```

synthetic transaction은 source transaction의 `t_dat`을 그대로 사용한다.

따라서 최근 transaction을 복제한 virtual node가 temporal sampling에서 우선될 가능성이 높다.

Candidate 생성 단계에서도 `(customer, product_code)`별 최신 transaction을 source로 선택하므로 두 정책이 일치한다.

```text
candidate source 선택: 가장 최근 transaction
neighbor sampling: 최근 neighbor 우선
```

다만 graph에 synthetic node가 추가되었다고 해서 모두 sampling되는 것은 아니다. 이 때문에 `test_num_neighbors`가 별도 옵션으로 포함되어 있다.

---

# 13. Expansion statistics가 필요한 이유

실험 결과만 보면 성능 상승이나 하락의 원인을 알기 어렵다. 따라서 다음 통계를 출력하도록 spec에 넣었다.

```text
seed_time
num_test_customers
num_historical_transactions
num_selected_source_transactions
num_candidates_before_seen_filter
num_candidates_after_seen_filter
num_synthetic_transactions
num_customers_with_synthetic_transactions
mean_synthetic_per_expanded_customer
max_synthetic_per_customer
```

예:

```text
Product-code expansion
  seed_time:                         2020-09-14
  test customers:                    67,144
  historical transactions:           215,320
  selected customer-product sources: 112,405
  candidates before seen filtering:  381,220
  candidates after seen filtering:   294,101
  synthetic transactions:            181,752
  expanded customers:                51,002
  mean per expanded customer:        3.56
  max per customer:                  32
```

이 통계로 다음을 확인할 수 있다.

* 실제로 몇 명의 고객이 확장되었는가
* variant 후보가 얼마나 생성되었는가
* seen-item 제외가 얼마나 영향을 미쳤는가
* cap이 얼마나 강하게 작동했는가
* graph size 증가가 합리적인가

---

# 14. Runtime invariant의 의미

spec의 assertion은 단순한 방어 코드가 아니라 실험의 정당성을 보장한다.

## Source transaction 범위

```python
0 <= source_tx_pos < base_num_transactions
```

잘못된 TensorFrame row 접근 방지.

## Customer/article 범위

```python
0 <= customer_id < num_customer_nodes
0 <= target_article_id < num_article_nodes
```

dangling edge 방지.

## 시간 조건

```python
source_t_dat <= seed_time
```

future leakage 방지.

## Source와 target이 같지 않음

```python
source_article_id != target_article_id
```

기존 article 자체를 virtual candidate로 추가하지 않도록 보장.

## Candidate pair 중복 없음

```python
(customer_id, target_article_id)
```

가 unique여야 한다.

## Seen pair 제외

과거 구매 article이 virtual candidate에 다시 들어오지 않아야 한다.

## Edge 증가량

synthetic node가 `M`개면 네 edge relation 모두 정확히 `M`개씩 증가해야 한다.

```text
transaction → customer: +M
customer → transaction: +M
transaction → article: +M
article → transaction: +M
```

## Base graph 불변

augmentation 이후에도:

```python
len(base_data["transactions"].tf)
```

가 이전과 같아야 한다.

---

# 15. Unit test 설계

실제 rel-hm은 크고 다운로드가 필요하므로 테스트에서는 작은 in-memory fixture를 사용한다.

---

## 15.1 Candidate 생성 테스트 예시

Article:

| article_id | product_code |
| ---------: | -----------: |
|          0 |           10 |
|          1 |           10 |
|          2 |           10 |
|          3 |           20 |
|          4 |         null |

Transaction:

| row pos | customer | article | t_dat      |
| ------: | -------: | ------: | ---------- |
|       0 |        0 |       0 | 2020-01-01 |
|       1 |        0 |       1 | 2020-01-05 |
|       2 |        0 |       2 | 2020-02-01 |
|       3 |        1 |       3 | 2020-01-10 |
|       4 |        0 |       0 | 2020-03-01 |

Test seed:

```text
2020-02-15
```

row 4는 미래이므로 제외한다.

customer 0은 product 10의 article 0, 1, 2를 이미 모두 구매했기 때문에 variant candidate가 없다.

다른 fixture에서는 일부 variant를 unseen 상태로 두어 candidate 생성 여부를 확인한다.

---

## 15.2 Tie-break 테스트

동일한 customer-product에 같은 timestamp transaction 두 개를 둔다.

| source_tx_pos | customer | product | t_dat      |
| ------------: | -------: | ------: | ---------- |
|             4 |        0 |      10 | 2020-01-10 |
|             7 |        0 |      10 | 2020-01-10 |

선택 결과는 반드시:

```text
source_tx_pos = 7
```

이어야 한다.

---

## 15.3 Graph augmentation 테스트

작은 TensorFrame을 직접 만든다.

예:

```python
TensorFrame(
    feat_dict={
        numerical: ...,
        categorical: ...,
        timestamp: ...,
    },
    col_names_dict=...,
)
```

그 후 source row 1을 복제했을 때 모든 stype tensor가 동일한지 확인한다.

```python
assert cloned_numerical == source_numerical
assert cloned_categorical == source_categorical
assert cloned_timestamp == source_timestamp
```

단순히 `price`만 같다고 검사하는 것이 아니라 **전체 TensorFrame row**를 비교한다.

---

# 16. 왜 core RelBench 파일을 수정하지 않도록 했는가

이 기능은 일반적인 RelBench graph construction 기능이 아니라 다음 조건에 특화된 실험이다.

```text
dataset = rel-hm
task = user-item-purchase
test-only product variant expansion
```

이를 `relbench/modeling/graph.py` 같은 core API에 넣으면 범용 코드에 task-specific 정책이 섞인다.

그래서 기본 변경 범위는 다음으로 제한한다.

```text
추가:
examples/hm_product_code_expansion.py

수정:
examples/idgnn_recommendation.py

추가:
test/examples/test_hm_product_code_expansion.py
```

이렇게 하면:

* upstream RelBench API에 영향을 주지 않음
* 기존 다른 dataset/task가 영향받지 않음
* 실험 기능을 쉽게 제거하거나 변경 가능
* baseline과 augmentation 비교가 명확함

---

# 17. 최종 실행 예시

```bash
python examples/idgnn_recommendation.py \
  --dataset rel-hm \
  --task user-item-purchase \
  --epochs 20 \
  --num_layers 2 \
  --num_neighbors 128 \
  --test_product_code_expansion \
  --product_code_max_virtual_per_customer 32 \
  --test_num_neighbors 256
```

이 설정의 의미는 다음과 같다.

```text
training:
  원본 graph
  neighbor schedule = [128, 64]

validation:
  원본 graph
  neighbor schedule = [128, 64]

test:
  product_code virtual transaction graph
  고객당 최대 32 synthetic nodes
  neighbor schedule = [256, 128]
```

---

# 18. Codex가 최종적으로 만들어야 하는 구조

```text
relbench/
├── examples/
│   ├── idgnn_recommendation.py          # 수정
│   ├── hm_product_code_expansion.py     # 신규
│   └── model.py                         # 변경하지 않음
│
├── relbench/
│   ├── modeling/
│   │   └── graph.py                     # 원칙적으로 변경하지 않음
│   └── datasets/
│       └── hm.py                        # 변경하지 않음
│
└── test/
    └── examples/
        └── test_hm_product_code_expansion.py
```

---

# 19. spec의 가장 중요한 보장

이 spec이 지키려는 핵심은 다음 다섯 가지다.

1. **학습은 완전히 기존 baseline과 동일하다.**
2. **test future transaction과 test label을 사용하지 않는다.**
3. **새로운 edge type 없이 기존 ID-GNN을 그대로 사용한다.**
4. **synthetic transaction representation은 실제 source transaction과 정확히 같다.**
5. **base graph를 수정하지 않고 test graph만 별도로 확장한다.**

원본 문서는 여기에서 확인할 수 있다.

[spec.md](sandbox:/mnt/data/spec.md)

