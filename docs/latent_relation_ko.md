# Learned Latent Relation 구현 및 실행 가이드

이 문서는 RelBench 추천 실험에서 사용할 수 있는 learned latent destination-to-destination relation 증강 기능의 구현 내용과 실행 방법을 정리한다.

## 목표

기존 ID-GNN 추천 파이프라인은 source 중심으로 샘플링된 local subgraph 안에 포함된 destination만 점수화할 수 있다. 이 구현은 추천 supervision으로 전역 destination 간 관계를 먼저 학습한 뒤, seed time별로 sparse virtual edge를 materialize하고, 기존 ID-GNN이 증강된 local graph에서 reasoning하도록 한다.

핵심 구조:

```text
learn globally -> retrieve distant candidates -> reason locally with ID-GNN
```

## 구현 파일

주요 신규 파일:

- `examples/latent_relation/adapters.py`: dataset/task별 source, destination, historical interaction resolver
- `examples/latent_relation/model.py`: destination encoder, two-tower relation scorer, LogSumExp aggregation
- `examples/latent_relation/training.py`: relation training dataset, sampled-softmax loss, checkpoint 저장
- `examples/latent_relation/retrieval.py`: chunked torch exact retrieval, optional FAISS backend
- `examples/latent_relation/materialize.py`: sparse top-L latent relation snapshot 생성
- `examples/latent_relation/io.py`: parquet snapshot, manifest, atomic write, resume validation
- `examples/latent_relation/graph.py`: direct destination-to-destination edge injection
- `examples/latent_relation/coverage.py`: GT locality/coverage 진단
- `examples/latent_relation/train.py`: relation generator 학습 CLI
- `examples/latent_relation/build_snapshots.py`: latent snapshot 생성 CLI
- `examples/latent_relation/evaluate_coverage.py`: coverage 진단 CLI
- `test/examples/test_latent_relation.py`: toy leakage, loss, top-L, direct edge, reachability, snapshot tests

수정된 기존 파일:

- `examples/idgnn_recommendation.py`: `--augmentation latent` 옵션 추가. 기본값은 `none`이라 기존 baseline 동작은 유지된다.

## 지원 데이터셋

MVP에서 지원하는 task:

- `rel-hm / user-item-purchase`
- `rel-avito / user-ad-visit`

각 task의 historical interaction:

| Dataset | Source | Destination | Interaction table | Time column |
|---|---|---|---|---|
| `rel-hm` | `customer` | `article` | `transactions` | `t_dat` |
| `rel-avito` | `UserInfo` | `AdsInfo` | `VisitStream` | `ViewDate` |

## 방법 요약

각 query `(u, t, Y)`에 대해 source `u`의 과거 destination history를 만든다.

```text
H_{u,t} = destinations interacted by u with interaction_time <= t
```

중복 destination은 최근 interaction을 우선으로 deduplicate하고, `--history-limit`로 최대 길이를 제한한다. 기본값은 `64`.

Destination encoder는 기존 RelBench/PyTorch Frame table encoder를 사용한다. 개별 컬럼을 수동 선택하지 않고 destination table feature 전체를 표준 encoder에 넣는다.

Relation scorer:

```text
q_i = W_Q h_i
k_j = W_K h_j
s(i, j) = normalize(q_i)^T normalize(k_j) / tau
```

`W_Q`, `W_K`는 서로 다른 projection이므로 relation은 대칭이라고 가정하지 않는다.

History-to-candidate score:

```text
g(u, j, t) = LogSumExp_{i in H_{u,t}} s(i, j)
```

즉, 미래 target `j`가 과거 모든 item과 평균적으로 비슷해야 하는 것이 아니라, 과거 history 중 하나 이상의 anchor와 강하게 연결되면 높은 점수를 받는다.

학습 objective는 training split의 실제 미래 추천 label만 positive로 사용한다. 4-hop, 6-hop neighborhood, CF edge, manual heuristic은 teacher로 사용하지 않는다.

## Leakage 방지

중요한 timestamp 필터링은 `examples/latent_relation/adapters.py`에서 수행한다.

- history interaction은 항상 `interaction_time <= seed_time`
- relation training은 `task.get_table("train")`만 사용
- validation/test snapshot은 학습된 checkpoint를 freeze한 뒤 생성
- validation/test label은 snapshot 생성에 사용하지 않음
- ID-GNN은 seed time group별로 정확히 해당 seed time snapshot만 load

## Graph Injection

Latent relation은 direct destination-to-destination edge로 들어간다.

예:

```text
("article", "latent_relation", "article")
("article", "rev_latent_relation", "article")
```

Avito의 경우:

```text
("AdsInfo", "latent_relation", "AdsInfo")
("AdsInfo", "rev_latent_relation", "AdsInfo")
```

`article -> latent_node -> article` 같은 intermediate virtual node는 만들지 않는다.

## Snapshot 구조

각 snapshot은 parquet로 저장되며 최소 컬럼은 다음과 같다.

```text
src_id
dst_id
score
rank
seed_time
```

같은 디렉터리에 `manifest.json`이 저장된다. manifest에는 dataset, task, checkpoint hash, top-L, history limit, relation dim, temperature, retrieval backend 등이 기록된다.

기존 valid snapshot이 있으면 기본적으로 재사용한다. 다시 만들려면 `--force`를 사용한다.

## 실행 방법

아래 명령은 예시 경로를 사용한다. 실제 실험에서는 output 경로를 원하는 위치로 바꿔도 된다.

### 1. H&M relation model 학습

```bash
python -m examples.latent_relation.train \
  --dataset rel-hm \
  --task user-item-purchase \
  --checkpoint-dir outputs/latent_checkpoints \
  --history-limit 64 \
  --relation-dim 128 \
  --temperature 0.07 \
  --num-negatives 256 \
  --batch-size 256 \
  --epochs 5 \
  --lr 1e-3 \
  --device cuda
```

작은 smoke test:

```bash
python -m examples.latent_relation.train \
  --dataset rel-hm \
  --task user-item-purchase \
  --checkpoint-dir outputs/latent_checkpoints_smoke \
  --history-limit 4 \
  --relation-dim 8 \
  --encoder-channels 8 \
  --num-negatives 2 \
  --batch-size 2 \
  --epochs 1 \
  --max-steps-per-epoch 1 \
  --train-row-limit 2000 \
  --device cpu
```

### 2. H&M latent snapshot 생성

```bash
python -m examples.latent_relation.build_snapshots \
  --dataset rel-hm \
  --task user-item-purchase \
  --checkpoint outputs/latent_checkpoints/rel-hm_user-item-purchase_latent_relation.pt \
  --snapshot-dir outputs/latent_snapshots/rel-hm/user-item-purchase \
  --history-limit 64 \
  --top-l 32 \
  --retrieval-backend torch \
  --device cuda
```

특정 seed time만 생성:

```bash
python -m examples.latent_relation.build_snapshots \
  --dataset rel-hm \
  --task user-item-purchase \
  --checkpoint outputs/latent_checkpoints/rel-hm_user-item-purchase_latent_relation.pt \
  --snapshot-dir outputs/latent_snapshots/rel-hm/user-item-purchase \
  --seed-time 2019-09-09 \
  --history-limit 64 \
  --top-l 32 \
  --retrieval-backend torch
```

### 3. H&M coverage/locality 평가

```bash
python -m examples.latent_relation.evaluate_coverage \
  --dataset rel-hm \
  --task user-item-purchase \
  --snapshot-dir outputs/latent_snapshots/rel-hm/user-item-purchase \
  --history-limit 64 \
  --splits val,test \
  --output-json outputs/latent_coverage_hm.json \
  --output-csv outputs/latent_coverage_hm.csv
```

특정 seed time과 일부 row만 확인:

```bash
python -m examples.latent_relation.evaluate_coverage \
  --dataset rel-hm \
  --task user-item-purchase \
  --snapshot-dir outputs/latent_snapshots/rel-hm/user-item-purchase \
  --history-limit 64 \
  --splits train \
  --seed-time 2019-09-09 \
  --row-limit 2000
```

### 4. H&M ID-GNN 학습 with latent augmentation

```bash
python examples/idgnn_recommendation.py \
  --dataset rel-hm \
  --task user-item-purchase \
  --augmentation latent \
  --latent-snapshot-dir outputs/latent_snapshots/rel-hm/user-item-purchase \
  --budget-mode additive \
  --num_layers 3 \
  --num_neighbors 128 \
  --top-l 32 \
  --epochs 20
```

Fixed-budget mode:

```bash
python examples/idgnn_recommendation.py \
  --dataset rel-hm \
  --task user-item-purchase \
  --augmentation latent \
  --latent-snapshot-dir outputs/latent_snapshots/rel-hm/user-item-purchase \
  --budget-mode fixed \
  --num_layers 3 \
  --num_neighbors 128 \
  --top-l 32
```

기존 baseline은 그대로 실행한다.

```bash
python examples/idgnn_recommendation.py \
  --dataset rel-hm \
  --task user-item-purchase \
  --augmentation none
```

## Avito 실행 예시

### Avito relation model 학습

```bash
python -m examples.latent_relation.train \
  --dataset rel-avito \
  --task user-ad-visit \
  --checkpoint-dir outputs/latent_checkpoints \
  --history-limit 64 \
  --relation-dim 128 \
  --temperature 0.07 \
  --num-negatives 256 \
  --batch-size 256 \
  --epochs 5 \
  --lr 1e-3 \
  --device cuda
```

### Avito snapshot 생성

```bash
python -m examples.latent_relation.build_snapshots \
  --dataset rel-avito \
  --task user-ad-visit \
  --checkpoint outputs/latent_checkpoints/rel-avito_user-ad-visit_latent_relation.pt \
  --snapshot-dir outputs/latent_snapshots/rel-avito/user-ad-visit \
  --history-limit 64 \
  --top-l 32 \
  --retrieval-backend torch \
  --device cuda
```

### Avito coverage 평가

```bash
python -m examples.latent_relation.evaluate_coverage \
  --dataset rel-avito \
  --task user-ad-visit \
  --snapshot-dir outputs/latent_snapshots/rel-avito/user-ad-visit \
  --history-limit 64 \
  --splits val,test \
  --output-json outputs/latent_coverage_avito.json \
  --output-csv outputs/latent_coverage_avito.csv
```

### Avito ID-GNN 학습

```bash
python examples/idgnn_recommendation.py \
  --dataset rel-avito \
  --task user-ad-visit \
  --augmentation latent \
  --latent-snapshot-dir outputs/latent_snapshots/rel-avito/user-ad-visit \
  --budget-mode additive \
  --num_layers 3 \
  --num_neighbors 128 \
  --top-l 32
```

## 주요 옵션

| Option | 의미 | 기본값 |
|---|---|---|
| `--history-limit` | source별 최근 historical destination 개수 | `64` |
| `--relation-dim` | two-tower projection dimension | `128` |
| `--temperature` | cosine/IP score temperature | `0.07` |
| `--num-negatives` | query당 sampled negative 수 | `256` |
| `--top-l` | anchor destination당 latent neighbor 수 | `32` |
| `--retrieval-backend` | retrieval backend | `torch` |
| `--budget-mode` | ID-GNN sampling budget mode | `additive` |
| `--force` | snapshot 강제 재생성 | off |

## Budget Mode

`additive`:

- 기존 sampling fanout은 유지
- latent relation fanout을 추가로 사용
- candidate coverage 개선 가능성을 크게 보는 설정

`fixed`:

- latent edge fanout만큼 기존 hop-3 fanout 일부를 줄임
- PyG fanout은 edge type별 설정이라 완전한 global budget equality는 아니지만 deterministic하게 근사

## 테스트

추가된 테스트:

```bash
python -m pytest test/examples/test_latent_relation.py
```

관련 CF graph regression 포함:

```bash
python -m pytest test/examples/test_latent_relation.py test/examples/test_hm_cf_graph.py
```

확인된 결과:

```text
test/examples/test_latent_relation.py: 7 passed
latent + hm_cf_graph targeted suite: 14 passed, 1 skipped
```

전체 테스트 실행 시 현재 환경에서는 `pyg-lib` 또는 `torch-sparse` 미설치로 PyG `NeighborSampler` 관련 기존 테스트 일부가 실패할 수 있다.

## 현재 한계

- single-head latent relation만 구현
- relation generator와 ID-GNN은 분리 학습
- uniform random negative만 기본 지원
- destination encoder는 table-feature encoder 기반
- FAISS는 optional import이며 자동 설치하지 않음
- coverage diagnostic은 cheap diagnostic 목적이며 full sampled PyG subgraph replay는 아님

