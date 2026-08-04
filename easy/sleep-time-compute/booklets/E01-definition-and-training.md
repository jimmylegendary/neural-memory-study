# E01 · Sleep-time compute란 무엇이며 왜 training인가

## 이 권이 답할 질문

“모델이 잔다”는 표현을 잠시 버리고도 sleep-time compute를 정확히 정의할 수 있을까? 답은 **사용자 응답이 끝난 뒤 계산한다**는 시간 조건과, 그 계산이 **다음 wake에서 재사용될 상태를 바꾼다**는 상태 조건을 함께 보는 것이다. 단순한 야간 batch job, 로그 압축, model checkpoint 복사는 첫 조건만 만족할 뿐이다. 반대로 응답 도중의 chain-of-thought나 긴 추론은 계산량이 커도 다음 episode에 남지 않으면 두 번째 조건을 만족하지 못한다.

이 권은 {{STUDY:STC-S02|Study의 operational definition}}과 {{STUDY:STC-S03|training map}}을 쉬운 말로 연결한다. 핵심 근거는 {{CLAIM:STC-C005}}, {{CLAIM:STC-C006}}, {{CLAIM:STC-C015}}이다.

## 먼저 세 문장

1. **Sleep-time compute는 latency-critical response 뒤로 미룬 계산이 미래의 reusable state를 바꾸는 lifecycle이다.** “sleep”이라는 이름보다 이 두 조건이 중요하다.
2. 바뀌는 state는 foundation weight만이 아니다. adapter, recurrent state, text/vector memory, temporal graph, summary, index도 가능하므로 **언제 갱신하는가**와 **어디에 기억하는가**는 서로 독립된 축이다.
3. state를 경험에 맞춰 개선하려면 데이터→{{BG:forward-pass|순전파}}→{{BG:loss|손실}}→{{BG:backpropagation|역전파}}→{{BG:optimizer|옵티마이저 갱신}} 중 일부 또는 전부가 필요하다. 그래서 sleep은 inference의 부속 기능이 아니라 작은 training system에 가깝다.

> **한 줄 요약.** Wake가 “지금 답하기”라면 sleep은 “다음번에 더 잘 답하도록 재사용 상태를 고치기”다.

## 직관

### 두 개의 경계선을 그리면 된다

첫 번째 경계는 **응답 SLA**다. 사용자가 기다리는 동안 반드시 끝나야 하는 계산이 wake path이고, 사용자가 답을 받은 뒤 실행해도 되는 계산이 deferred path다. 두 번째 경계는 **episode**다. 계산 결과가 현재 응답에서 사라지는지, 다음 대화·다음 task·다음 날에도 재사용되는지를 나눈다.

| 계산 | 응답 뒤 실행 | 다음 episode에 재사용 | STC인가 |
|---|---:|---:|---:|
| 응답 중 더 오래 생각하기 | 아니오 | 대개 아니오 | 아니오 |
| 로그를 cold storage로 복사 | 예 | 행동을 바꾸지 않음 | 보통 아니오 |
| 대화에서 stable preference를 추출해 memory store에 기록 | 예 | 예 | 예, external-memory STC |
| replay로 adapter를 갱신하고 새 version을 배포 | 예 | 예 | 예, parametric STC |
| search index를 rebuild해 이후 retrieval을 개선 | 예 | 예 | 예, state/index STC |

> **직관.** “밤에 돌았다”가 아니라 “다음번 행동을 바꾸는 상태가 남았다”가 판정 기준이다. 이 정의를 쓰면 마케팅 용어가 달라도 dreaming, reflection, synthesis, consolidation을 같은 좌표계에 놓을 수 있다.

### Training은 weight update보다 넓다

Training을 “거대한 pretraining cluster에서 모든 weight를 바꾸는 일”로만 보면 external-memory sleep은 training이 아닌 것처럼 보인다. 그러나 시스템 관점의 training은 **경험으로부터 미래 행동을 개선할 state transformation을 선택하는 일**이다. text summary를 새로 만들 때도 어떤 episode를 넣고 뺄지, 충돌을 어떻게 합칠지, 품질을 어떤 evaluator로 판단할지에 objective가 숨어 있다.

Parametric sleep은 그 objective를 gradient로 직접 최적화한다. External sleep은 LLM extraction, deduplication, clustering, graph update처럼 더 이산적인 procedure를 쓴다. 둘 다 오류가 누적될 수 있고 validation·rollback·provenance가 필요하다. 차이는 “학습인가 아닌가”보다 **학습 결과가 어느 medium에 저장되고 얼마나 되돌리기 쉬운가**다.

## 예와 반례

### 예 1 — 외부 기억 공고화

한 사용자가 여러 대화에서 “회의 요약은 결론부터, 표는 싫다”고 반복했다고 하자. sleep job은 episode들을 모아 중복을 제거하고, 일시적 요청과 stable preference를 구분하고, “결론 우선·표 지양”이라는 durable memory를 만든다. foundation weight는 고정이지만 다음 대화에서 retrieval되어 행동을 바꾸므로 STC다. 장점은 원 episode와 연결해 수정·삭제하기 쉽다는 점이다.

### 예 2 — adapter로 선택적 승격

한 공장 line에서 반복되는 error code와 복구 절차가 수천 번 재사용된다면, 매번 긴 문서를 retrieve하는 비용보다 domain adapter로 compile하는 편이 유리할 수 있다. sleep job은 검증된 record만 골라 {{BG:distillation|증류}} 또는 fine-tuning으로 작은 delta를 만들고, canary 평가를 통과한 뒤 활성화한다. 이것은 더 빠를 수 있지만 삭제·오염·rollback은 어려워진다.

### 반례 1 — 긴 test-time reasoning

한 질문에 100배 많은 token을 써서 답을 찾았지만 그 reasoning이 버려졌다면 test-time compute이지 STC는 아니다. 같은 reasoning에서 reusable strategy를 추출해 다음 task의 strategy memory로 남겼을 때 비로소 lifecycle의 sleep branch가 된다.

### 반례 2 — 무조건적인 nightly fine-tuning

매일 밤 모든 user log로 LoRA를 학습한다고 해서 좋은 STC가 되는 것은 아니다. 잘못된 label, transient intent, privacy-sensitive record까지 섞이면 다음날 성능이 악화된다. “뒤에서 학습했다”는 사실은 mechanism이고, **무엇을 admit하고 무엇을 보존하며 어떻게 되돌릴지**가 solution이다.

> **주의.** Sleep-time compute는 자동으로 안전하거나 효율적이라는 뜻이 아니다. latency budget을 sleep budget으로 옮겼을 뿐, 잘못된 학습의 위험과 비용은 사라지지 않는다.

## 그림 읽기

{{FIG:STC-F001}}

이 그림은 pretraining/post-training 이후의 배포 lifecycle을 wake와 sleep으로 다시 자른다. 왼쪽에서 wake는 요청 처리, retrieval, fast-state update를 담당한다. 오른쪽 sleep은 replay·reflection·distillation·compaction을 실행한다. 가운데 reusable-state boundary가 가장 중요하다. 그 경계를 넘는 object가 text인지 graph인지 adapter인지에 따라 storage와 governance가 달라진다.

{{FIG:STC-F002}}

두 번째 그림은 “시간”과 “medium”을 직교축으로 놓는다. wake-time weight update도 가능하고 sleep-time external-memory update도 가능하다. 따라서 “parametric vs external”과 “wake vs sleep”을 한 축으로 섞으면 안 된다. 실제 system은 네 사분면을 동시에 쓸 수 있다: wake 중 recurrent state, 응답 후 summary, 주기적 graph compaction, 드물게 adapter promotion.

## 대안과 비교

### Pretraining·post-training·test-time·sleep-time의 역할

| phase | 대표 데이터 | 바뀌는 것 | 강점 | 약점 |
|---|---|---|---|---|
| pretraining | 대규모 일반 corpus | foundation weights | 넓은 prior | 매우 비싸고 최신성 낮음 |
| post-training | instruction·preference·domain data | weights/adapter | 행동 정렬 | 배포 전 분포에 묶임 |
| wake/test-time learning | 현재 context·feedback | activation/fast state/weight | 즉시 적응 | 응답 latency와 interference |
| sleep-time learning | 누적 episode·replay·derived data | external/parametric reusable state | batch·검증·공고화 | staleness, pipeline 복잡도 |

Sleep의 고유 장점은 **현재 응답의 latency와 학습 계산을 분리**하고, 여러 episode를 한꺼번에 비교해 dedup·conflict resolution·evaluation할 수 있다는 점이다. 반면 즉시 반영이 필요한 정보에는 늦다. 그래서 최선의 framing은 네 phase 중 하나를 승자로 고르는 것이 아니라, 각 state를 어느 phase에서 갱신할지 결정하는 것이다.

### Fixed와 session-specific weights

Model artifact를 foundation weights, shared domain adapter, user adapter, external memory, ephemeral state로 분리하면 위험이 보인다. foundation은 느리고 검증이 엄격해야 한다. user adapter는 빠르게 personalized될 수 있지만 tenant isolation과 version explosion이 생긴다. external memory는 설명 가능하지만 retrieval latency와 context budget을 쓴다. “모든 기억을 weight로” 또는 “모든 기억을 vector DB로”는 둘 다 극단이다.

## 아직 모르는 것

- 어떤 deferred job이 실제 미래 utility를 만들었는지 측정하는 공통 benchmark가 없다.
- Google의 *Language Models Need Sleep*은 parametric consolidation의 직접 증거지만, 공개 실험이 곧 장기 fleet 운영·무제한 user별 weight 성장의 증거는 아니다.
- External-memory sleep은 이미 product 형태가 보이지만, derived summary가 원 evidence를 왜곡하지 않는다는 보장은 없다.
- “얼마나 자주 자야 하나”는 traffic, staleness, batching, interference의 함수이며 고정 주기로 해결되지 않는다.
- 같은 sleep budget을 retrieval 개선, synthetic data, adapter training 중 어디에 배분할지에 대한 universal policy가 없다.

> **핵심.** 이 분야의 첫 연구 질문은 “sleep이 가능한가?”가 아니라 “어떤 reusable state를 어떤 evidence와 objective로, 어느 cadence에, 얼마나 되돌릴 수 있게 바꿀 것인가?”다.

## 더 깊이 읽기

- 원 분석: {{STUDY:STC-S02|운영적 정의}}와 {{STUDY:STC-S03|배포 전·wake·sleep training map}}
- Training Background: {{BG:wake-state|Wake 상태}}, {{BG:sleep-state|Sleep 상태}}, {{BG:loss|손실}}, {{BG:backpropagation|역전파}}, {{BG:optimizer|옵티마이저}}
- Claim route: {{CLAIM:STC-C005}} · {{CLAIM:STC-C006}} · {{CLAIM:STC-C015}}
