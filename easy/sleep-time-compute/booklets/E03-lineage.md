# E03 · 뇌의 수면에서 agent memory까지의 계보

## 이 권이 답할 질문

Sleep-time compute는 2026년에 갑자기 생긴 이름인가, 아니면 오래된 연구 흐름들이 한 lifecycle에서 만난 것인가? 계보를 보면 적어도 네 갈래가 합쳐진다. 생물학적 memory consolidation, continual-learning replay, periodic knowledge compression, agent external memory다. 이 계보를 알아야 “뇌를 모방했다”는 약한 비유와 실제 algorithmic contribution을 구분할 수 있다.

이 권은 {{STUDY:STC-S05|1995–2026 계보}}를 풀고 {{CLAIM:STC-C016}}, {{CLAIM:STC-C017}}, {{CLAIM:STC-C018}}, {{CLAIM:STC-C019}}, {{CLAIM:STC-C022}}을 연결한다.

## 먼저 세 문장

1. 생물학 연구가 준 핵심은 “수면”이라는 단어가 아니라 **online acquisition과 offline reorganization을 분리**하고 서로 다른 memory system이 상호작용한다는 가설이다.
2. Machine continual learning은 sleep이라는 이름 없이도 replay, regularization, parameter isolation, periodic compression으로 망각을 줄여 왔다.
3. 최근 agent system은 background synthesis와 temporal memory maintenance를 실제 product lifecycle에 붙였고, neural-memory 연구는 weight/state 자체를 multi-timescale memory로 해석하기 시작했다.

## 직관

### 계보 A — Complementary learning systems

빠르게 배우는 hippocampus와 천천히 일반화하는 neocortex를 분리한다는 관점은 한 저장소가 동시에 “즉시 기록”과 “안정된 일반화”를 잘하기 어렵다는 문제에서 출발한다. 낮에는 episode를 빠르게 포착하고, offline에는 재활성화해 느린 system으로 통합한다. AI로 옮기면 fast episodic store와 slow parametric/generalized store의 역할 분리가 된다.

### 계보 B — Replay

과거 example을 그대로 저장하는 {{BG:replay-buffer|리플레이 버퍼}}와, generator가 pseudo-example을 만드는 {{BG:generated-replay|생성 리플레이}}가 있다. 둘 다 새 data만 학습할 때 old task가 사라지는 문제를 완화한다. 원 data replay는 fidelity가 좋지만 privacy·storage cost가 있고, generated replay는 저장을 줄이지만 generator bias와 model collapse risk가 있다.

### 계보 C — 보호와 격리

EWC 같은 {{BG:regularization|정규화}}는 옛 task에 중요한 parameter를 덜 움직이게 한다. GEM은 old example loss가 악화되지 않도록 gradient 방향을 제약한다. Adapter나 dynamic expansion은 새 parameter 공간을 열어 옛 parameter와 충돌을 줄인다. 이들은 모두 stability–plasticity의 다른 해법이다.

### 계보 D — Periodic consolidation

Progress & Compress는 active column에서 새 task를 배우고, 별도 phase에서 knowledge base로 distill한다. 이는 명시적 “sleep” 이름 전부터 존재한 wake/periodic-consolidation 구조다. 오늘날의 STC는 이를 long-lived agent, external memory, deployment scheduler, rollback까지 확장한다.

> **직관.** 새 아이디어는 보통 재료가 아니라 조립법에 있다. Replay·distillation·adapter·graph는 오래된 부품이고, STC의 새로움은 이들을 response-critical path 밖의 반복 lifecycle로 묶는 데 있다.

## 예와 반례

### 2022 PLOS sleep-like replay

두 task를 순서대로 배운 spiking network가 sleep-like replay 동안 joint synaptic representation을 만들고 forgetting을 줄였다는 결과는 biological analogy와 computational mechanism 사이의 중요한 다리다. 다만 작은 two-task setting의 결과를 그대로 billion-parameter LLM fleet의 경제성으로 확대할 수는 없다. {{CLAIM:STC-C016}}은 “망각 완화의 직접 증거”이지 “LLM sleep scaling law”의 증거가 아니다.

### NREM/REM alternation과 dreaming

Hippocampus–neocortex model은 autonomous offline dynamics와 NREM/REM-like regime을 번갈아 새 정보 통합과 old knowledge 보호를 연구했다. Perturbed and Adversarial Dreaming은 wake, perturbed replay, adversarial dreaming을 분리했다. 여기서 얻을 수 있는 design hint는 phase별 objective가 다를 수 있다는 점이다. 하지만 인간 수면 stage의 이름을 붙였다고 algorithm이 생물학적으로 동일해지는 것은 아니다.

### Deep Generative Replay

Generator가 old data와 비슷한 sample을 만들고 solver가 new+generated data를 함께 학습한다. 이는 명시적 sleep 없이도 offline replay의 핵심 기능을 수행한다. 현대 sleep system이 이 baseline보다 우월하려면 단순 retention 외에 cadence, provenance, rollback, utility per cost에서 이겨야 한다.

### 반례 — “NREM과 REM을 구현했으니 더 인간답다”

Stage label은 evidence가 아니다. NREM-like perturbation과 REM-like adversarial objective가 실제 어떤 ablation에서 도움이 되었는지, 동일 compute의 replay·regularization baseline보다 나은지 확인해야 한다. 생물학적 plausibility와 engineering optimality는 독립된 질문이다.

## 그림 읽기

{{FIG:STC-F003}}

Timeline은 1995년 complementary learning systems에서 2026년 explicit sleep/product synthesis까지 이어진다. 한 줄로 읽지 말고 네 층으로 본다. 아래에는 biological theory, 그 위에는 continual-learning algorithm, 그 위에는 neural/test-time memory, 맨 위에는 agent product가 있다. 최근으로 올수록 “망각 accuracy”뿐 아니라 serving, user state, governance가 연구 문제에 들어온다.

## 대안과 비교

| 계보 | 보존 대상 | 대표 mechanism | 현대 STC에 남긴 것 | 해결하지 못한 것 |
|---|---|---|---|---|
| biological consolidation | episode·representation | offline replay, phase alternation | fast/slow memory separation | fleet cost, delete |
| continual learning | task performance | replay, EWC, GEM, expansion | retention baselines | agent state semantics |
| periodic compression | active knowledge | distillation | wake/consolidate lifecycle | long-term governance |
| agent external memory | user/entity facts | synthesis, graph maintenance | production background job | parametric generalization |
| neural memory | learned state | test-time update, multi-rate weights | state as memory | prefix reuse, serving maturity |

이 표가 보여 주는 것은 sleep이 단일 method가 아니라 **research program**이라는 점이다. 같은 lifecycle 안에서도 어떤 workload에는 raw replay, 어떤 workload에는 graph merge, 어떤 workload에는 adapter distillation이 맞다.

## 아직 모르는 것

- Biological stage를 어느 수준까지 모방해야 engineering gain이 있는지 모른다. phase separation 자체가 충분할 수도 있다.
- Replay가 retention을 높여도 privacy, energy, rare-tail fidelity를 포함하면 최적이 아닐 수 있다.
- Multi-timescale weight memory와 external temporal graph를 함께 쓸 때 어느 layer가 system-of-record인지 정립되지 않았다.
- 과거 연구는 짧은 task sequence가 많아 수백·수천 sleep cycle의 cumulative error를 보여 주지 못한다.
- Consolidation이 old knowledge를 “보존”하는지, 단지 benchmark answer를 재현하는지 분리할 evaluation이 필요하다.

> **핵심.** Sleep-time compute의 정당성은 인간이 잔다는 사실에서 나오지 않는다. Online path와 offline path를 분리했을 때 retention·utility·cost·governance가 더 좋아진다는 비교 실험에서 나와야 한다.

## 더 깊이 읽기

- 원 분석: {{STUDY:STC-S05|수면·continual learning·agent memory의 시간순 계보}}
- Training Background: {{BG:generated-replay|생성 리플레이}}, {{BG:replay-buffer|리플레이 버퍼}}, {{BG:consolidation|공고화}}, {{BG:regularization|정규화}}, {{BG:dynamic-expansion|동적 확장}}
- Claim route: {{CLAIM:STC-C016}} · {{CLAIM:STC-C017}} · {{CLAIM:STC-C018}} · {{CLAIM:STC-C019}} · {{CLAIM:STC-C022}}
