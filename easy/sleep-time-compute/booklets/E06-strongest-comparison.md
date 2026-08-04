# E06 · 강한 대안과 정면 비교하면 무엇이 남는가

## 이 권이 답할 질문

Sleep-time compute가 promising하다는 판단은 약한 baseline과의 accuracy 비교로는 설득력이 없다. “그 문제를 지금 가장 잘 푸는 대안”과 같은 compute·memory·governance budget에서 비교해야 한다. 이 권은 replay, EWC, GEM, parameter isolation, RAG, model editing, rate-distortion compaction을 sleep과 같은 표에 놓는다.

기준은 {{STUDY:STC-S08|strongest-alternative comparison}}이며, 근거는 {{CLAIM:STC-C020}}, {{CLAIM:STC-C021}}, {{CLAIM:STC-C026}}, {{CLAIM:STC-C034}}, {{CLAIM:STC-C040}}이다.

## 먼저 세 문장

1. 망각만 줄이는 문제라면 replay·EWC·GEM·parameter isolation이 이미 강한 baseline이다. Sleep은 이들보다 retention뿐 아니라 lifecycle cost와 governance에서 더 나아야 한다.
2. 사실 기억은 RAG/temporal graph가 출처·수정·삭제 면에서 우세하다. Parametric sleep은 repeated access나 latent skill처럼 weight가 실제로 유리한 subset에 한정해야 한다.
3. 현재 evidence를 종합하면 가장 강한 system hypothesis는 **external system-of-record + selective reversible promotion**의 hybrid다.

## 직관

### EWC — 중요한 parameter를 덜 움직인다

{{BG:ewc|EWC}}는 old task에 중요하다고 본 parameter의 이동에 큰 penalty를 준다. 저장해야 할 것은 과거 parameter와 importance estimate다. 장점은 raw replay가 없어도 된다는 점이고, 약점은 task가 많아질수록 penalty가 새 학습을 막고 importance approximation이 부정확할 수 있다는 점이다.

### GEM — 과거 loss를 악화시키지 않는 방향으로 움직인다

{{BG:gem|GEM}}은 replay buffer의 old example에 대해 loss가 증가하지 않도록 new gradient를 projection한다. Local constraint가 명확하지만 buffer·gradient comparison cost가 있다. 모든 future behavior의 불변을 보장하는 것은 아니다.

### Parameter isolation — 충돌할 공간을 분리한다

Adapter, expert, mask, progressive network는 task나 user별 parameter를 분리한다. {{BG:parameter-isolation|파라미터 격리}}는 interference를 줄이지만 capacity가 단조 증가하고 routing·batch fragmentation이 생긴다. Dynamic expansion은 “잊지 않기”를 “계속 늘리기”로 바꾼다.

### Rate–distortion compaction — 무엇을 버릴지 수치화한다

{{BG:rate-distortion|율–왜곡}} 관점은 memory byte라는 rate와 answer utility 손실이라는 distortion의 tradeoff를 명시한다. 모든 memory를 보존할 수 없으므로 어떤 detail을 압축할지 결정한다. 이론적 framing은 강하지만 agent의 실제 utility와 distortion을 어떻게 정의할지는 열린 문제다.

### RAG — Mutable truth를 밖에 둔다

External store는 audit·delete·freshness의 기준선이다. Weakness는 retrieval miss와 token cost다. Sleep synthesis로 index와 summary를 개선할 수 있지만 source-of-record를 weight로 대체할 이유는 없다.

## 예와 반례

### 공정한 comparison protocol

동일한 1년짜리 user stream을 가정한다. 각 method에 같은 total compute, 같은 retained byte, 같은 evaluation query, 같은 delete request를 준다. 다음을 함께 잰다.

- 새 정보 적응: current utility
- 옛 정보 보존: backward retention
- unrelated behavior 변화: locality
- 원 evidence 연결: provenance coverage
- 삭제 후 잔존: delete leakage
- update/serve cost: joule, byte, latency
- rollback: 이전 generation 복구 시간

이 protocol에서 accuracy만 높은 method는 승자가 아니다. 예를 들어 adapter가 빠르게 답해도 delete leakage가 크면 regulated domain에서 탈락할 수 있다.

### 예 — Stable skill promotion

도메인별 수백 개 procedure를 external store에서 제공하다가, query frequency가 높고 내용이 3개월 이상 안정된 subset만 adapter로 distill한다. 원 record와 adapter version lineage를 유지하고, 변경 시 adapter를 rebuild한다. 이 hybrid는 RAG의 governance와 adapter의 low-latency access를 결합한다.

### 반례 — 모든 user에게 별도 full model

Parameter isolation을 극단적으로 쓰면 interference는 줄지만 model count와 memory가 user 수에 비례한다. Hardware memory를 늘려도 scheduler·batching·cold-start가 병목이 된다. 중요한 것은 parameter 수가 아니라 active working set과 reuse distribution이다.

### 반례 — 무제한 raw replay

Raw replay는 강한 retention baseline이지만 privacy와 storage를 무시할 수 없다. Consent 철회가 일어나면 training example과 그것에서 파생된 state를 찾아 제거해야 한다. Replay buffer가 system-of-record가 되면 data governance 비용이 model cost보다 커질 수 있다.

## 그림 읽기

{{FIG:STC-F004}}

이 matrix는 method를 단일 ranking으로 만들지 않는다. 각 problem row마다 가장 강한 baseline이 다르다. Sleep은 timing column을 더해 batch evaluation과 cross-episode consolidation을 제공하지만, medium의 약점을 없애지 않는다. Weight sleep은 여전히 delete가 어렵고, external sleep은 여전히 retrieval이 필요하다.

{{FIG:STC-F006}}

Hybrid에서 admission gate는 “이 기억이 중요해 보이는가”만 보지 않는다. Reuse, stability, confidence, privacy, reversibility를 함께 본다. Promotion은 one-way migration이 아니라 source-of-record에서 다시 생성 가능한 cache/compiled artifact여야 한다. 이 원칙이 weight를 authoritative memory로 두는 설계와 가장 큰 차이다.

## 대안과 비교

| 방법 | retention | plasticity | capacity growth | provenance/delete | serving cost |
|---|---|---|---|---|---|
| replay | 높음, data quality 의존 | 높음 | buffer 증가 | raw data는 추적 가능 | sleep train 큼 |
| EWC | 중간 | task 증가 시 낮아짐 | importance state 증가 | weight source 불명확 | update 계산 추가 |
| GEM | local constraint 강함 | projection 범위 내 | buffer 증가 | example lineage 가능 | gradient projection |
| adapter isolation | 충돌 적음 | 새 slot에 높음 | 단조 증가 | version 단위 rollback | routing fragmentation |
| RAG/graph | 원문 보존 | 즉시 update | record/index 증가 | 가장 강함 | retrieval+tokens |
| hybrid sleep | 선택에 따라 | 선택에 따라 | tier별 관리 | external anchor 유지 | orchestration 복잡 |

Strongest alternative 관점에서 sleep이 우월해질 조건은 세 가지다. 첫째, 여러 episode를 batch로 봐야만 얻는 pattern이 있어야 한다. 둘째, 그 pattern이 미래에 충분히 재사용되어 sleep cost를 상각해야 한다. 셋째, derived state를 audit·rollback·delete할 수 있어야 한다.

## 아직 모르는 것

- Hybrid promotion gate를 어떤 signal로 학습해야 false promotion을 줄이는지 모른다.
- Rate–distortion의 distortion을 answer accuracy, preference fidelity, safety 중 어떻게 합칠지 합의가 없다.
- Adapter isolation의 long-tail user 분포에서 active set, cache hit, batching loss 측정이 부족하다.
- EWC/GEM/replay와 modern external-memory product를 같은 long-lived agent benchmark에서 비교한 연구가 없다.
- Parametric memory를 external record에서 완전히 rebuild할 수 있는 reproducibility standard가 없다.

> **핵심.** Sleep의 strongest case는 “모든 기억을 weight로 옮긴다”가 아니라 “외부 truth를 유지하면서 반복 가치가 검증된 일부만 되돌릴 수 있게 compile한다”다.

## 더 깊이 읽기

- 원 분석: {{STUDY:STC-S08|strongest alternatives와 hybrid verdict}}
- Training Background: {{BG:ewc|EWC}}, {{BG:gem|GEM}}, {{BG:parameter-isolation|파라미터 격리}}, {{BG:rate-distortion|율–왜곡}}, {{BG:memory-admission|기억 입장}}, {{BG:memory-eviction|기억 퇴거}}
- Claim route: {{CLAIM:STC-C020}} · {{CLAIM:STC-C021}} · {{CLAIM:STC-C026}} · {{CLAIM:STC-C034}} · {{CLAIM:STC-C040}}
