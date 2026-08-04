# E08 · 정말 promising하며 주류가 될까

## 이 권이 답할 질문

Sleep-time compute이라는 새 scaling 축은 실제로 promising한가? 학계·업계가 이 방향으로 발전할까? 이 질문에는 단일 yes/no보다 **어느 variant가 어떤 문제에서 주류가 되는가**가 더 정확하다. Background synthesis와 external consolidation은 이미 practical value가 보이지만, user별 foundation-weight sleep은 증거·안전·economics가 부족하다.

이 권은 {{STUDY:STC-S10|promisingness 판정}}과 {{STUDY:STC-S11|mainstream scenario}}를 연결한다. 근거는 {{CLAIM:STC-C038}}, {{CLAIM:STC-C039}}, {{CLAIM:STC-C040}}, {{CLAIM:STC-C042}}다.

## 먼저 세 문장

1. **Lifecycle은 promising하다.** 응답 밖에서 memory를 정리·검증·압축하는 요구는 long-lived agent가 늘수록 커진다.
2. **Medium별 전망은 다르다.** External-memory sleep은 provenance와 update가 쉬워 먼저 확산될 가능성이 높고, broad parametric sleep은 selective reversible adapter 형태가 더 현실적이다.
3. “Sleep-time compute”라는 용어가 주류가 되지 않아도 dreaming, reflection, memory synthesis, self-evolution, background learning이라는 기능은 주류가 될 수 있다.

## 직관

### Promisingness를 네 조건으로 본다

1. **Problem pressure** — 현재 방법으로 남는 큰 문제가 있는가?
2. **Unique leverage** — 응답 후 batch가 그 문제에 특별한 이점을 주는가?
3. **Evidence maturity** — modern model과 product에서 재현되는가?
4. **Operational viability** — cost, rollback, delete, security를 감당할 수 있는가?

External-memory sleep은 1·2·4에서 비교적 강하고 3도 product signal이 있다. Parametric sleep은 1·2의 잠재력은 크지만 3·4가 약하다. 그래서 전체 verdict는 “조건부 promising”이다.

### 왜 수요가 생기는가

Agent가 하루가 아니라 수개월 일하면 context를 계속 늘릴 수 없다. Preference, project state, entity relation, reusable strategy를 durable하게 만들고 stale record를 정리해야 한다. 이 일은 single request 중 처리하기 어렵고 여러 episode를 비교해야 하므로 background phase가 자연스럽다.

### 왜 full-weight personalization이 바로 오지 않는가

{{BG:parametric-memory|모수적 기억}}은 inference access가 빠르고 behavior를 직접 바꿀 수 있다. 그러나 source attribution, user delete, poisoning containment, multi-tenant batching, version explosion이 어렵다. Base model을 user마다 바꾸면 shared serving economics가 깨진다. 작은 {{BG:serving-artifact|서빙 산출물}}인 adapter나 compiled cache가 먼저 현실적이다.

### 왜 external-only도 최종 답은 아닐 수 있는가

{{BG:external-memory|외부 기억}}는 audit에 강하지만 every-query retrieval, context token, miss, prompt injection 비용이 있다. 매우 자주 재사용되는 stable pattern은 parametric/latent cache로 compile하면 이득일 수 있다. 따라서 medium-term mainstream은 hybrid다.

## 예와 반례

### Scenario A — External synthesis가 기본 기능이 됨

Agent platform은 episode log, user memory, temporal graph를 유지하고 idle time에 dedup·summary·conflict resolution을 수행한다. Output은 원 record와 lineage로 연결되고 사용자가 inspect·correct·delete할 수 있다. 이 scenario는 기술·규제 양쪽에서 가장 가까운 경로다.

### Scenario B — Domain adapter promotion

기업 내부에서 반복되는 stable procedure를 shared domain adapter로 nightly/weekly compile한다. Adapter는 {{BG:versioning|버전 관리}}되고 canary 평가를 통과해야 하며, external knowledge base가 authoritative source로 남는다. User별보다 domain별 reuse가 높아 batching economics도 좋다.

### Scenario C — On-device private sleep

개인 data가 cloud로 나가기 어렵고 device가 idle/charging일 때 local summary나 small adapter를 갱신한다. Privacy는 좋아지지만 thermal, battery, write endurance, rollback UI가 제약이다. 특정 workload에는 유망하지만 universal default는 아니다.

### 반례 — “Hyperscaler가 paper를 냈으니 곧 전 모델에 적용”

Research paper는 option value를 보여 준다. Production adoption에는 inference architecture, abuse control, SRE, compliance가 추가된다. 특히 user-specific weights는 base-model sharing과 cache locality를 해칠 수 있어 경제성 검증이 필요하다.

### 반례 — “External product가 있으니 scaling law도 증명됨”

Feature가 존재하는 것과 sleep FLOP을 늘리면 utility가 예측 가능하게 증가한다는 scaling law는 다르다. 현재는 accounting identity와 break-even hypothesis가 있을 뿐 universal empirical law는 없다.

## 그림 읽기

{{FIG:STC-F005}}

Conditional verdict 그림은 green/yellow/red 조건을 나눈다. Cross-episode synthesis, index maintenance, selective adapter는 green/yellow에 가깝다. Broad autonomous weight rewrite는 evidence·governance가 약해 red에 가깝다. 그림의 목적은 기술을 찬반으로 나누는 것이 아니라 **promotion 범위를 단계적으로 제한**하는 것이다.

{{FIG:STC-F014}}

Academia는 method frontier를, industry는 external-memory lifecycle을 밀고 있다. 두 방향이 만나는 곳은 benchmark, hybrid architecture, reversible artifact, memory infra다. Memory device 회사의 기회도 이 교차점에 있다: 모든 weight training을 가속하는 것보다 derived state의 저장·이동·version을 최적화하는 것.

## 대안과 비교

| 전망 | 2–3년 가능성 | 필요한 조건 | 주류화 장벽 |
|---|---|---|---|
| external synthesis | 높음 | provenance, correction UX | quality drift, poisoning |
| temporal graph maintenance | 중상 | schema/temporal query | extraction error, write cost |
| shared domain adapter sleep | 중간 | reuse, canary, rollback | update pipeline, fragmentation |
| user adapter sleep | 중하 | consent, isolation, cache | millions of versions |
| foundation-weight sleep | 낮음/연구 | robust continual learning | delete, safety, fleet cost |
| learned cadence | 연구 frontier | utility telemetry | delayed credit assignment |

여기서 “주류”는 모든 model이 같은 implementation을 쓴다는 뜻이 아니다. Database에 backup·compaction이 기본 기능이듯, agent platform에 background memory lifecycle이 기본 layer가 되는 것을 뜻한다.

## 아직 모르는 것

- User가 background-derived memory를 얼마나 신뢰하고 correction하는지 product telemetry가 부족하다.
- Sleep job이 만든 utility를 해당 job에 credit assignment하는 방법이 없다.
- Shared adapter와 external retrieval의 break-even이 hardware·batching에 따라 크게 달라진다.
- Regulation이 weight-based personalization을 external memory보다 더 엄격히 다룰 가능성이 있다.
- “Sleep” label이 과도한 anthropomorphism을 낳아 system boundary를 흐릴 수 있다.

> **핵심.** 주류가 될 가능성이 가장 높은 것은 “모델이 인간처럼 잔다”는 서사가 아니라, response path와 background memory maintenance를 분리하는 operational architecture다.

## 더 깊이 읽기

- 원 분석: {{STUDY:STC-S10|조건부 promisingness}}와 {{STUDY:STC-S11|주류화 scenario}}
- Training Background: {{BG:serving-artifact|서빙 산출물}}, {{BG:external-memory|외부 기억}}, {{BG:parametric-memory|모수적 기억}}, {{BG:versioning|버전 관리}}, {{BG:rollback|롤백}}
- Claim route: {{CLAIM:STC-C038}} · {{CLAIM:STC-C039}} · {{CLAIM:STC-C040}} · {{CLAIM:STC-C042}}
