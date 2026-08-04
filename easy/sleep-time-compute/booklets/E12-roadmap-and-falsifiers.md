# E12 · 무엇을 측정하면 이 가설을 믿거나 버릴 수 있는가

## 이 권이 답할 질문

Sleep-time compute를 research program으로 밀기 전에 어떤 실험이 성공해야 하고, 어떤 결과가 나오면 방향을 버리거나 축소해야 하는가? Promising한 가설은 반증 조건을 가져야 한다. 이 권은 benchmark, staged roadmap, failure control, go/no-go rule을 하나로 묶는다.

원문 경로는 {{STUDY:STC-S15|benchmark roadmap}}, {{STUDY:STC-S16|limitations and falsifiers}}, {{STUDY:STC-S17|conclusion}}이다. 근거는 {{CLAIM:STC-C035}}, {{CLAIM:STC-C036}}, {{CLAIM:STC-C037}}, {{CLAIM:STC-C039}}, {{CLAIM:STC-C041}}, {{CLAIM:STC-C057}}이다.

## 먼저 세 문장

1. Benchmark는 current accuracy가 아니라 retention, access, provenance, deletion, poisoning, cost, wake-SLA를 함께 재야 한다.
2. 연구는 external synthesis baseline→selective adapter promotion→learned cadence→long-horizon fleet simulation 순으로 단계화한다. 앞 단계가 실패하면 뒤 단계로 가지 않는다.
3. Sleep이 동일 budget의 retrieval/replay/adapter baseline보다 utility per cost에서 이기지 못하거나 rollback·delete를 만족하지 못하면 broad deployment 가설은 반증된다.

## 직관

### Benchmark unit은 “기억 하나”가 아니다

하나의 memory record가 여러 future query에 다른 방식으로 쓰인다. 따라서 benchmark episode에는 source, valid time, sensitivity, expected reuse, contradiction, delete event가 함께 있어야 한다. Evaluation은 write 직후가 아니라 여러 sleep cycle 뒤에도 반복한다.

### Metric family

- **Utility**: future task success, answer accuracy, strategy reuse
- **Retention**: old episode/task performance, backward transfer
- **Access**: encoded information이 실제 answer에서 회수되는 비율
- **Freshness**: change 후 stale answer 지속 시간
- **Provenance**: answer/derived memory가 source로 연결되는 비율
- **Deletion**: source delete 후 external·parametric leakage
- **Robustness**: poisoned memory의 spread와 containment
- **Systems**: sleep joule, bytes moved, write amplification, TTFT/ITL impact
- **Recovery**: rollback time, rebuild time, lost generation 수

{{BG:observability|관측 가능성}}은 이 metric을 state generation과 연결하는 telemetry다. “평균 score가 내려갔다”가 아니라 어느 sleep job과 evidence가 원인인지 찾아야 한다.

### Four-stage roadmap

**Stage 1 — External baseline.** Immutable episode, temporal graph, vector index, background synthesis를 구현한다. Provenance·delete·correction이 되는 기준선을 만든다.

**Stage 2 — Selective promotion.** High-reuse stable subset만 adapter로 compile하고 RAG와 break-even을 비교한다. Adapter는 external record에서 재현 가능해야 한다.

**Stage 3 — Adaptive cadence.** Fixed schedule sweep으로 utility/staleness/cost curve를 얻은 뒤 event-triggered 또는 learned scheduler를 시험한다.

**Stage 4 — Long-horizon simulation.** 수백 generation, user churn, concept drift, poisoning, deletion을 포함해 capacity knee와 lifecycle economics를 측정한다.

## 예와 반례

### 최소 viable benchmark stream

1. 10만 episode를 시간순으로 흘린다.
2. 일부는 stable fact, 일부는 preference, 일부는 transient request, 일부는 poison이다.
3. Fact change와 delete request를 중간에 넣는다.
4. 매 interval에 future query와 unrelated control task를 평가한다.
5. External-only, replay, EWC/GEM, adapter, hybrid sleep을 같은 byte/compute budget으로 비교한다.
6. 100회 이상 generation에서 cumulative error와 recovery를 기록한다.

숫자는 illustrative benchmark scale이며 universal requirement가 아니다. 핵심은 short static test가 아니라 time-ordered lifecycle을 만드는 것이다.

### Go condition

Hybrid sleep이 external-only보다 같은 quality에서 later-query cost를 줄이고, adapter-only보다 provenance·delete를 유지하며, wake SLA 영향 없이 positive net utility를 보이면 다음 단계로 간다.

### No-go condition

- Derived memory error가 cycle마다 누적되고 correction으로 회복되지 않는다.
- 동일 budget의 RAG/replay baseline보다 retained utility가 낮다.
- Delete request 후 parametric leakage를 허용 기준 아래로 내릴 수 없다.
- Adapter/version fragmentation이 serving gain을 상쇄한다.
- Sleep job이 wake peak와 경쟁해 SLO를 지속적으로 위반한다.

### 반례 — Synthetic collapse는 항상 일어난다

{{CLAIM:STC-C035}}는 real-data support가 사라진 recursive training의 위험을 보여 주지만, {{CLAIM:STC-C036}}은 적절한 real/synthetic accumulation에서 collapse가 필연은 아님을 한정한다. 따라서 benchmark는 “synthetic인가 아닌가”보다 recursive depth, anchor ratio, tail retention을 sweep해야 한다.

### 반례 — 한 번의 delete test 통과

External record에서 삭제됐어도 summary cache, graph edge, adapter, backup generation에 남아 있을 수 있다. {{BG:unlearning|머신 언러닝}}과 lineage-driven rebuild를 함께 검증해야 한다. Delete completeness는 artifact 전체의 property다.

## 그림 읽기

{{FIG:STC-F013}}

Failure-mode map은 benchmark injection point를 준다. Ingest poisoning, summary hallucination, replay collapse, adapter interference, version mismatch, stale serve, incomplete delete를 각각 의도적으로 넣고 detection·containment·recovery를 측정한다.

{{FIG:STC-F015}}

Research agenda는 질문을 measurement와 falsifier로 연결한다. “주류가 될까?” 같은 큰 질문을 capacity knee, break-even reuse, cadence frontier, delete leakage, state migration으로 분해한다. 각 연구 package는 성공 score와 중단 threshold를 가져야 한다.

## 대안과 비교

| 연구 package | baseline | 핵심 metric | go | stop/falsifier |
|---|---|---|---|---|
| external synthesis | raw RAG/KG | utility, correction | conflict 감소 | hallucinated merge 증가 |
| adapter promotion | RAG, LoRA-only | break-even reuse | net cost 절감 | delete/locality 실패 |
| cadence | fixed intervals | staleness+cost | interior gain | boundary가 항상 우세 |
| capacity | larger store | utility/byte | knee 이동 | noise가 gain 상쇄 |
| device tiering | flat HBM/SSD | bytes, WA, SLA | migration 감소 | software overhead 우세 |
| long horizon | short benchmark | cycle drift | stable 100+ gen | error 단조 누적 |

{{BG:off-policy-evaluation|오프폴리시 평가}}는 cadence 후보를 싸게 거를 수 있지만 final go/no-go에는 live or controlled intervention이 필요하다. {{BG:publication-gate|출판 게이트}}는 claim을 direct fact, synthesis, hypothesis로 표시하고 falsifier가 없는 scaling-law 표현을 막는다.

## 아직 모르는 것

- Human correction cost를 automated metric과 어떻게 합칠지 모른다.
- User privacy와 personalization utility를 동일 benchmark에서 비교할 consent framework가 없다.
- Parametric delete completeness를 black-box behavior만으로 증명할 수 있는지 불명확하다.
- Long-horizon simulation이 실제 user behavior drift를 얼마나 대표하는지 모른다.
- Device trace가 공개되지 않으면 infra scaling hypothesis를 independent하게 검증하기 어렵다.

> **핵심.** Sleep-time compute를 믿는 가장 좋은 방법은 멋진 sleep metaphor가 아니라, strongest baseline과의 시간순 benchmark에서 이길 조건과 질 조건을 미리 적어 두는 것이다.

## 더 깊이 읽기

- 원 분석: {{STUDY:STC-S15|benchmark roadmap}}, {{STUDY:STC-S16|limitations·falsifiers}}, {{STUDY:STC-S17|조건부 결론}}
- Training Background: {{BG:validation-set|검증 세트}}, {{BG:off-policy-evaluation|오프폴리시 평가}}, {{BG:unlearning|머신 언러닝}}, {{BG:publication-gate|출판 게이트}}, {{BG:observability|관측 가능성}}, {{BG:provenance|출처 추적성}}
- Claim route: {{CLAIM:STC-C035}} · {{CLAIM:STC-C036}} · {{CLAIM:STC-C037}} · {{CLAIM:STC-C039}} · {{CLAIM:STC-C041}} · {{CLAIM:STC-C057}}
