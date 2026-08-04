# E02 · 어떤 문제를 풀려는가

## 이 권이 답할 질문

Sleep-time compute가 등장한 이유를 “AI도 사람처럼 자야 해서”라고 설명하면 문제를 놓친다. 실제 출발점은 배포된 agent가 장기간 경험을 쌓을 때 생기는 다섯 병목이다: **context overflow, state staleness, catastrophic forgetting, finite capacity, governance debt**. 이 중 일부는 retrieval이나 cache로 해결되고, 일부는 background consolidation이 유리하며, 일부는 아직 어떤 방식으로도 해결되지 않았다.

이 권은 {{STUDY:STC-S04|문제 분해}}를 바탕으로 {{CLAIM:STC-C032}}, {{CLAIM:STC-C033}}, {{CLAIM:STC-C035}}, {{CLAIM:STC-C037}}을 설명한다.

## 먼저 세 문장

1. Long context는 “많이 넣는 능력”이지 “무엇을 오래 보존하고 갱신할지 결정하는 능력”이 아니다. context window가 커져도 오래된 episode의 충돌·삭제·중요도 문제는 남는다.
2. 지속적으로 weight나 memory를 갱신하면 유한한 {{BG:capacity-budget|용량 예산}} 안에서 새 지식이 옛 지식의 **접근 가능성**을 손상시킬 수 있다. 정보가 parameter 안에 남아 있어도 행동으로 꺼내지 못하면 실용적으로는 잊은 것이다.
3. Sleep은 여러 episode를 batch로 비교하고 정리할 기회를 주지만, synthetic replay의 오류·memory poisoning·삭제 전파라는 새 문제도 만든다.

## 직관

### 문제 1 — Context는 작업대이지 창고가 아니다

Context window는 현재 reasoning에 필요한 자료를 펼쳐 놓는 작업대다. 작업대가 커지면 더 많은 문서를 동시에 볼 수 있지만, 매번 모든 과거를 다시 펼치는 비용은 sequence 길이에 따라 증가한다. 또한 중요하지 않은 기록이 signal을 가리고, 서로 충돌하는 사실의 시간 순서를 모델이 잘못 읽을 수 있다. 그래서 long context와 long-term memory는 보완 관계이지 대체 관계가 아니다.

### 문제 2 — 기억은 저장보다 선택이 어렵다

새 episode를 모두 저장하면 memory는 단조 증가한다. 검색 후보가 늘면서 index 비용, false retrieval, compaction 비용, privacy exposure가 함께 증가한다. 반대로 aggressive summary는 중요한 tail detail을 잃는다. 어떤 record를 admit·merge·supersede·evict할지 결정하는 정책이 필요하다.

### 문제 3 — Weight에는 주소가 없다

외부 DB에서는 “이 문장은 어느 대화에서 왔는가”를 pointer로 남길 수 있다. Weight update는 많은 example을 gradient로 섞기 때문에 특정 사실이 어느 parameter에 얼마나 들어갔는지 역추적하기 어렵다. 삭제 요청이 오면 adapter 전체를 버리거나 원 dataset에서 재학습해야 할 수도 있다. 따라서 parametric memory는 빠른 access를 얻는 대신 provenance와 surgical delete를 잃기 쉽다.

### 문제 4 — Continual learning은 retention과 plasticity의 줄다리기다

{{BG:continual-learning|지속학습}}은 새 경험에 적응해야 하지만 옛 능력을 지켜야 한다. update를 크게 하면 plastic하지만 망각이 커지고, 작게 하면 안정적이지만 새 정보가 안 들어간다. 이를 {{BG:catastrophic-forgetting|파국적 망각}}과 stability–plasticity dilemma라고 부른다. 단순히 parameter를 늘리면 시간을 벌 수 있지만 capacity와 serving cost가 계속 자란다.

> **직관.** 도서관의 진짜 병목은 책장을 더 사는 일이 아니라, 중복판을 합치고 낡은 정보를 표시하고 출처를 남기며 필요한 책을 제때 찾는 일이다.

## 예와 반례

### 예 — “내 주소”가 세 번 바뀐 사용자

2025년, 2026년 봄, 2026년 여름 주소가 서로 다른 세 episode에 있다. Naive vector search는 query와 가장 비슷한 과거 주소를 가져올 수 있다. Temporal graph는 valid-time과 transaction-time을 붙여 최신 주소를 고를 수 있다. Summary는 “현재 주소”만 남겨 간단하지만, 왜 바뀌었는지와 과거 배송 분쟁을 잃을 수 있다. Weight update는 답을 빠르게 만들 수 있지만 어떤 주소가 학습되었는지 검증·삭제가 가장 어렵다.

### 예 — 새로운 error code의 반복 학습

공장 장비의 새 error code가 매일 수백 번 등장한다. 단기에는 external memory에 원 log와 복구 절차를 저장하는 편이 안전하다. 사용 빈도와 안정성이 확인되면 domain adapter로 승격할 수 있다. 이후 제조사 문서가 수정되면 adapter를 rollback하고 다시 build해야 한다. 이 흐름은 sleep을 “기억 저장”이 아니라 **promotion pipeline**으로 본다.

### 반례 — Long context만 늘리기

과거 1년의 모든 log를 million-token context에 넣을 수 있다고 해도 매 요청마다 읽을 이유는 없다. 비용·latency·irrelevant evidence가 늘고, 삭제 요청이 반영됐는지 확인하기도 어렵다. Long context가 유용한 것은 선택된 evidence를 깊게 읽을 때다. 선택·정리는 여전히 memory lifecycle의 일이다.

### 반례 — 모든 것을 nightly replay

Generated replay가 과거 distribution을 완벽히 대표한다고 가정하면 현실을 과도하게 단순화한다. 생성 모델은 rare tail을 빼먹거나 자기 오류를 반복할 수 있다. {{CLAIM:STC-C035}}는 real-data support 없이 synthetic data를 재귀적으로 학습할 때 distribution tail이 붕괴할 수 있음을 경고한다.

## 그림 읽기

{{FIG:STC-F004}}

행은 해결해야 할 문제, 열은 대안이다. Long context는 immediate access에는 강하지만 durable update와 delete에는 약하다. External memory는 provenance·rollback이 강하지만 retrieval과 context cost가 있다. Adapter/weight는 access latency를 줄일 수 있지만 interference와 governance가 어렵다. Sleep은 독립 열이 아니라 이 방법들을 **응답 후 batch로 조합하는 lifecycle**이다.

{{FIG:STC-F009}}

가로축은 memory load, 세로축은 retained utility로 읽는다. 초반에는 memory를 늘릴수록 utility가 빠르게 증가하지만 어느 지점부터 중복·간섭·검색 noise가 커져 marginal utility가 꺾인다. 이 “capacity knee”는 이미 보편 law로 입증된 값이 아니라 연구 가설이다. 중요한 것은 device capacity만 늘리는 것이 아니라 knee의 위치를 admission·compaction·tiering으로 옮기는 것이다.

## 대안과 비교

| 근본 문제 | sleep이 제공하는 것 | sleep 없이 가능한 강한 대안 | 판단 기준 |
|---|---|---|---|
| context overflow | 응답 후 요약·index rebuild | retrieval, recurrent state, compression | answer quality / token·latency |
| stale/conflicting facts | batch conflict resolution | temporal KG, system-of-record | temporal accuracy / delete |
| forgetting | replay·distillation·isolation | EWC, GEM, adapters, expansion | retained utility / interference |
| finite memory | compaction·promotion·eviction | rate-distortion policy | utility per byte / tail loss |
| governance | versioned sleep output | immutable event log | provenance / rollback time |

Sleep이 특히 유리한 곳은 **한 episode만 봐서는 판단할 수 없는 일**이다. 여러 기록에서 stable preference를 찾아내거나, conflict를 시간 순으로 해결하거나, reuse가 충분한 지식을 adapter로 승격하는 일이다. 즉시 반응이 중요하거나 source-of-truth가 명확한 factual lookup에는 external retrieval이 더 단순하고 안전하다.

## 아직 모르는 것

- Static memorization capacity는 lifelong safe capacity와 다르다. sequential interference·retrieval failure·delete overhead를 함께 재는 방법이 필요하다.
- Weight 안에 정보가 “encoded”되었다는 probe 결과와 실제 answer에서 “accessible”하다는 결과 사이의 gap을 어떻게 측정할지 합의가 없다.
- Synthetic replay에서 real-data anchor를 얼마나 유지해야 tail collapse를 막는지 task별 scaling law가 없다.
- External memory poisoning과 parametric poisoning을 같은 benchmark에서 비교한 공개 증거가 부족하다.
- Memory capacity knee가 workload, user 수, record entropy, device tier에 따라 어떻게 이동하는지 측정되지 않았다.

> **주의.** 문제의 원인은 memory byte 부족 하나가 아니다. 잘못된 기억을 오래·빠르게 제공하는 시스템은 큰 memory를 가진 실패한 시스템이다.

## 더 깊이 읽기

- 원 분석: {{STUDY:STC-S04|다섯 문제의 분해와 solution boundary}}
- Training Background: {{BG:continual-learning|지속학습}}, {{BG:catastrophic-forgetting|파국적 망각}}, {{BG:capacity-budget|용량 예산}}, {{BG:provenance|출처 추적성}}, {{BG:generalization|일반화}}
- Claim route: {{CLAIM:STC-C032}} · {{CLAIM:STC-C033}} · {{CLAIM:STC-C035}} · {{CLAIM:STC-C037}}
