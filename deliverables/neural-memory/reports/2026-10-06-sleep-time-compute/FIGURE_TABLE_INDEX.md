# Figure / Table index

## Figures

| 번호 | PDF 쪽 | 형식 | 설명 | 원본 |
|---|---:|---|---|---|
| 01 | 5 | 본 보고서 종합 도식 | 기억의 내용·표현·실행 형태와 제어 메타데이터를 분리한 분석 지도. | [SVG](source/assets/fig-01-taxonomy.svg) |
| 02 | 6 | 발표 시점 기반 재구성 | 대표 자료의 공개 순서. 분야가 단일 선형 단계로 발전했다는 증명은 아니며, 병렬 연구 흐름을 묶은 지도다. | [SVG](source/assets/fig-02-timeline.svg) |
| 03 | 7 | 개념 재구성 | 질문 이전 계산과 질문 이후 계산의 분리. 재사용 대상과 미래 query의 예측 가능성이 비용 상각을 결정한다. | [SVG](source/assets/fig-03-sleep-amortization.svg) |
| 04 | 8 | 원문 Eq.70–71 기반 독립 재구성 | CMS의 read/forward와 update schedule은 다르다. 아래 1/4/16 주기는 설명용이며 원문 실험 설정을 재현한 값이 아니다. | [SVG](source/assets/fig-04-cms-forward-update.svg) |
| 05 | 9 | 본 보고서 설계 공간 | 물리 배치의 최소 두 축: read demand와 write rate. 객체의 의미 범주만으로 placement를 정하지 않는다. | [SVG](source/assets/fig-05-read-write-space.svg) |
| 06 | 10 | 개념 재구성 | 고정된 recurrent state만 유지하는 방식과 checkpoint를 선택적으로 보존하는 방식을 구별한다. | [SVG](source/assets/fig-06-recurrent-checkpoints.svg) |
| 07 | 11 | 개념 재구성 | 두 parametric-memory 경로. canonical latent/belief와 실행용 weight는 구별되며, 서로 다른 논문의 구성요소를 한 구현으로 합친 것이 아니다. | [SVG](source/assets/fig-07-parametric-compiler.svg) |
| 08 | 12 | 본 보고서 종합 도식 | write-time preprocessing과 read-time curation의 hybrid. 원본을 폐기하지 않는 한 서로 대체 관계일 필요가 없다. | [SVG](source/assets/fig-08-jit-branch.svg) |
| 09 | 12 | 저자 보고값 재시각화 | 동일 논문 안에서 보고된 과제별 success improvement. 세 과제의 난이도·분모가 같다는 뜻은 아니다. | [SVG](source/assets/fig-09-jit-gains.svg) |
| 10 | 13 | 개념 재구성 | 현재 답이 같아도 미래 update에 필요한 구별은 다를 수 있다. 설명을 위한 재구성 예시이며 논문의 자연어 사례를 복제한 것이 아니다. | [SVG](source/assets/fig-10-update-sufficiency.svg) |
| 11 | 14 | 본 보고서 분석 도식 | 효용 근거의 층위. 오른쪽이 단순히 “진짜 인과”라는 순위가 아니라, 서로 다른 질문을 측정하는 개입 범위다. | [SVG](source/assets/fig-11-credit-ladder.svg) |
| 12 | 15 | 독립 메커니즘의 종합 도식 | MemoryAthena의 path 선택과 MemCalib의 influence 관점을 나란히 놓은 합성 설계. 하나의 실험으로 결합 검증된 구조는 아니다. | [SVG](source/assets/fig-12-routing-influence.svg) |
| 13 | 16 | 본 보고서 분류 | Dreaming을 세 가지 계산 작업으로 나누는 보고서 분류. 제품의 /dream 명칭과 neural policy optimization은 같지 않다. | [SVG](source/assets/fig-13-dream-types.svg) |
| 14 | 17 | 개념 재구성 | Dream-RSI의 핵심 아이디어를 재구성한 offline/online loop. replay support 밖의 정책 효과는 추가 환경 실행으로 확인해야 한다. | [SVG](source/assets/fig-14-dream-rsi-loop.svg) |
| 15 | 18 | 본 보고서 종합 도식 | WikiSkill의 semantic/procedural 분리와 Schema의 executable world model을 연결한 표현 선택 지도. | [SVG](source/assets/fig-15-knowledge-program.svg) |
| 16 | 19 | 본 보고서 종합 도식 | harness 자체의 수정과 개선 과제 선택을 분리한 loop. 고정 평가 경계를 agent의 수정 권한 밖에 둔다. | [SVG](source/assets/fig-16-self-improvement.svg) |
| 17 | 20 | 본 보고서 종합 도식 | 메모리를 관리하는 algorithm과 현재 task state는 별도 갱신 대상이다. | [SVG](source/assets/fig-17-program-state.svg) |
| 18 | 21 | 개념 재구성 | 최신 fact를 읽는 것과 plan의 derivation이 유효한지는 다르다. dependency-scoped 실행 검사의 필요성을 보여준다. | [SVG](source/assets/fig-18-planfence.svg) |
| 19 | 22 | 본 보고서 프로토콜 제안 | 파생 artifact의 versioned publish 제안. DB의 atomic pointer 교체와 외부 서비스의 실제 side effect는 구분해야 한다. | [SVG](source/assets/fig-19-transaction.svg) |
| 20 | 22 | 저자 보고값 재시각화 | 같은 논문에서 보고한 fault 조건별 중복 비율. 전체 agent 신뢰성이나 모든 도구의 확률로 일반화하지 않는다. | [SVG](source/assets/fig-20-limbo-duplicates.svg) |
| 21 | 23 | 본 보고서 방어 아키텍처 | 메모리 ingestion, reasoning, 실행 권한, 감사 로그를 분리한 방어 설계. 공격 재현 절차가 아닌 신뢰 경계 명세다. | [SVG](source/assets/fig-21-trust-boundary.svg) |
| 22 | 24 | 설명용 workload schedule | tool wait·LLM 실행·background 작업의 겹침. 대기 시간은 무조건 공짜 GPU 시간이 아니며 다른 session과 경합한다. | [SVG](source/assets/fig-22-phase-schedule.svg) |
| 23 | 25 | 개념 종합 | schema를 재사용 가능한 KV artifact로 분리하고 reader를 조정하는 구조. runtime compatibility와 조합 의존성을 먼저 확인해야 한다. | [SVG](source/assets/fig-23-compiled-kv.svg) |
| 24 | 26 | 정성적 설계 지도 | 물리 hierarchy의 역할을 보여주는 개념도. 막대 길이는 정성적 capacity 표현이며 실제 제품 용량·bandwidth 비율이 아니다. | [SVG](source/assets/fig-24-memory-hierarchy.svg) |
| 25 | 27 | 개념 재구성 | HBF architecture의 두 설계점. 어느 쪽이 우월한지는 workload, memory budget, controller 가정에 따라 바뀐다. | [SVG](source/assets/fig-25-hbf-branches.svg) |
| 26 | 28 | 원문 Figure 1 / §4 기반 재구성 | HBFSim의 GPU execution·shared ABI·host model 분리를 독립 도식으로 재구성. 원문 Figure 1의 메시지만 요약했다. | [SVG](source/assets/fig-26-hbfsim.svg) |
| 27 | 29 | 본 보고서 설계 가설 | 객체별 후보 placement. 점선은 특히 workload 조건에 민감한 경로다. slow-CMS도 read-hot이면 HBM에 남을 수 있다. | [SVG](source/assets/fig-27-logical-physical.svg) |
| 28 | 30 | 분석식 기반 예시 | 가상의 비용 단위로 설명한 손익 분기점. query당 2를 절감하고 build/refresh에 100을 쓴다면 50회에서 비용이 같다. 실험값이 아니다. | [SVG](source/assets/fig-28-break-even.svg) |
| 29 | 31 | 본 보고서 설계 가설 | 통합 reference architecture 제안. 모든 경로에 공통 state contract를 두되 foreground와 background의 자원·권한을 분리한다. | [SVG](source/assets/fig-29-reference-architecture.svg) |
| 30 | 32 | 본 보고서 계측 설계 | semantic event와 실제 resource operation을 연결하는 trace contract. 로그만 추가했다고 학습 효용이 증명되는 것은 아니다. | [SVG](source/assets/fig-30-event-contract.svg) |
| 31 | 34 | 본 보고서 실행 계획 | 구현 순서 제안. 날짜 약속이 아닌 검증 gate 중심의 실행 로드맵이다. | [SVG](source/assets/fig-31-roadmap.svg) |

## Tables

| 번호 | PDF 쪽 | 제목 |
|---|---:|---|
| 01 | 3 | 목적에 따른 읽기 경로 |
| 02 | 4 | 근거 유형과 사용 범위 |
| 03 | 10 | 지속 상태를 구현하는 세 경로 |
| 04 | 13 | 압축 평가에 추가할 시험 |
| 05 | 15 | 메모리 상태의 네 가지 질문 |
| 06 | 17 | Replay world에 필요한 메타데이터 |
| 07 | 19 | Self-improvement 연구의 서로 다른 제어점 |
| 08 | 25 | 재사용 가능한 state의 세 설계 |
| 09 | 26 | HBF를 평가할 때 구분할 항목 |
| 10 | 27 | 논문 결과를 읽는 비교 틀 |
| 11 | 30 | Lifetime 비용표에 반드시 들어갈 항목 |
| 12 | 32 | HAT / runtime event 계약 제안 |
| 13 | 33 | 동일 조건 비교를 위한 benchmark scorecard |
| 14 | 34 | 우선순위 실험과 중단 조건 |
| 15 | 35 | DSE parameter와 response 변수 |
| 16 | 36 | 확인한 제품 신호 |
| 17 | 37 | 이번 판에서 바로잡은 식별자 |
| 18 | 38 | 이전 조사 보존 목록 A |
| 19 | 39 | 이전 조사 보존 목록 B |
| 20 | 40 | 이 보고서의 용어 계약 |
| 21 | 49 | 미디어 제작·해석 계약 |
