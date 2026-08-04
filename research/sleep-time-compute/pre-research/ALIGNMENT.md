# Sleep-Time Compute Pre-Research Alignment

- 기준일: 2026-08-05
- 대상 독자: training 배경이 제한적인 Samsung SAIT memory·system 연구자, ML 연구자, 학술 심사자
- 목적: 결론을 미리 고정하지 않고 STC의 투자·연구 가치가 성립하는 조건과 반증 조건을 결정 질문으로 고정한다.
- canonical 후속 산출물: `deep-research/SOURCE-REGISTRY.json`, `claims/stc-study/claim-map.json`, Study Paper, Training Background Paper

## 1. 연구 대상의 작동 정의

이 프로그램에서 **Sleep-Time Compute(STC)** 는 다음 세 조건을 모두 만족하는 계산으로 정의한다.

1. wake 중 축적된 experience, trace, memory, feedback 또는 state snapshot을 입력으로 삼는다.
2. 현재 사용자 응답의 latency-critical path 밖에서 transform, selection, replay, distillation, optimization, compaction 또는 verification을 수행한다.
3. 결과가 이후 wake에서 소비되는 durable state로 publish된다.

durable destination은 model weight만 뜻하지 않는다. 다음을 모두 포함한다.

```text
text / event log / summary
vector index / graph / database state
latent cartridge / recurrent state / neural memory
adapter / sparse expert / model weight delta
policy / skill / tool program / router
verification metadata / provenance / rollback state
```

단순 prefetch, 동일 request 안의 chain-of-thought, context window 안의 recurrent update, 다음 wake에서 쓰이지 않는 batch job은 STC로 세지 않는다. 다만 STC와 경쟁하거나 STC의 primitive를 제공하는 adjacent approach로는 포함한다.

## 2. 사용자의 framing과 본 연구의 정렬

### 2.1 Learning lifecycle

| 단계 | 시점 | 주된 constraint | 가능한 trainable state | 본 연구에서 묻는 것 |
|---|---|---|---|---|
| Pre-training / post-training | 배포 전 또는 global refresh | throughput, total FLOPs, corpus governance | shared model weight, tokenizer, router, reward model | global model이 무엇을 미리 흡수해야 하는가 |
| Wake test-time inference / training | active session·request | TTFT, ITL, P99, isolation | context, KV/recurrent state, fast weight, session adapter, external memory | 지금 당장 배워야 하는 것은 무엇인가 |
| Sleep-time learning / consolidation | request critical path 밖 | staleness, publish safety, amortization, interference | external/latent/parametric memory, policy, skill | 무엇을 압축·승격·삭제·검증할 것인가 |

이 셋은 서로 배타적인 model category가 아니라 하나의 memory lifecycle이다. canonical experience가 exact external tier에 먼저 기록되고, reuse·value·confidence가 충분한 일부만 semantic/latent/parametric tier로 승격되며, source lineage와 rollback state는 유지되는 hybrid를 기본 비교축으로 둔다.

### 2.2 Infrastructure framing

| plane | 최소 책임 | 핵심 비용 |
|---|---|---|
| Wake inference/training plane | low-latency retrieval, state update, session isolation | HBM residency, KV/state bandwidth, tail latency |
| Sleep compute plane | replay, curation, distillation, fine-tuning, evaluation, compaction | accelerator-hours, source read, optimizer/checkpoint traffic |
| Durable memory/control plane | canonical source, index, version, provenance, delete fan-out, rollback | capacity, consistency, random I/O, metadata amplification |
| Promotion/verification plane | candidate build, safety gate, canary, atomic publish | duplicate residency, validation latency, failure recovery |

별도 sleep cluster가 항상 정답이라고 가정하지 않는다. shared time-partitioned fleet, separate fleet, tenant-local accelerator, storage-near compute, device-side consolidation을 workload별로 비교한다.

## 3. 여덟 개 결정 질문

| ID | 결정 질문 | 답으로 인정할 기준 | 필요한 근거 | 답이 불가능할 때 허용되는 출력 |
|---|---|---|---|---|
| DQ1 | STC는 정확히 어떤 미해결 문제를 해결하려는가? | latency·quality·capacity·governance 중 target을 분리하고 operator와 measurable outcome을 연결한다. | 논문 problem statement, open discussion, production lifecycle, failure case | problem–solution matrix와 범위 제한 |
| DQ2 | 같은 문제를 RAG, external memory, long context/state, TTT, continual learning, model editing, periodic refresh로 해결할 수 없는가? | 각 문제마다 strongest non-STC baseline을 지정하고 matched budget 비교를 제안한다. | 최신 primary paper, official implementation, negative result | 비교 불가능한 변수와 필요한 experiment |
| DQ3 | STC가 대안보다 우월하거나 열등해지는 조건은 무엇인가? | query reuse, staleness, privacy, interference, delete, data movement에 대한 crossover 조건을 제시한다. | empirical curves 또는 반증 가능한 hypothesis | 조건부 scenario; 보편 우위 주장 금지 |
| DQ4 | STC는 정말 promising한가? | evidence maturity와 expected value를 분리해 score하고, 실패 조건을 같은 비중으로 적는다. | replicated results, independent benchmark, code, duration, scale | `promising but unproven`, `niche`, `not supported` 중 조건부 판정 |
| DQ5 | 학계·산업계가 STC를 주류 축으로 채택하는가? | 이름이 아니라 기능적 lifecycle로 계보를 분류하고 research prototype과 deployment를 구분한다. | 2025–2026 papers, official product docs, release/tag evidence | 2026-08-05 공개 근거 범위의 scenario probability band |
| DQ6 | 독립 scaling axis라면 어떤 law와 saturation을 가져야 하는가? | sleep FLOPs 하나가 아니라 source volume, reuse, interference, state capacity, validation, data movement를 포함한 candidate family를 제시한다. | scaling measurements, capacity theory, matched-budget benchmark | law가 아니라 hypothesis family와 preregistration |
| DQ7 | wake–sleep–memory system은 어떻게 구성해야 하는가? | state ownership, snapshot, isolation, scheduling, publish/rollback, delete, offload, placement를 연결한다. | systems primary sources, workload traces, roofline/data movement model | reference architecture와 unresolved trade-off |
| DQ8 | memory-device 기업에는 어떤 기회가 생기는가? | algorithm 명칭이 아니라 persistent-state primitive와 measurable device requirement로 환원한다. | bytes moved, update granularity, endurance, latency, locality, consistency | opportunity matrix와 공동 benchmark 요구사항 |

## 4. 경쟁 가설

| 가설 | 핵심 명제 | 강해지는 관측 | 약해지는 관측 | 최소 반증 실험 |
|---|---|---|---|---|
| H-STC | 비동기 consolidation은 pre-training·wake compute와 구별되는 유효한 scaling axis가 된다. | sleep budget 증가가 matched wake/test compute보다 lifetime utility를 안정적으로 개선 | reuse가 낮거나 publish overhead가 gain을 상쇄 | 동일 model·data·total FLOPs에서 wake-only 대 sleep arm 비교 |
| H-EXT | 장기기억은 대부분 external memory와 retrieval로 해결되고 parametric sleep은 제한적이다. | exact recall·delete·provenance·freshness에서 external tier가 지속 우세 | retrieval latency·context pollution·reasoning composition이 지배 병목 | external-only와 promotion arm을 matched storage/token budget으로 비교 |
| H-HYBRID | canonical external memory와 selective parametric/latent promotion의 hybrid가 지배적이다. | high-reuse item만 승격할 때 latency와 quality가 개선되고 rollback 가능 | routing·duplication·consistency 비용이 이득보다 큼 | value/reuse threshold sweep과 promotion/eviction ablation |
| H-REFRESH | 별도 STC보다 periodic global post-training/model refresh가 경제적으로 우세하다. | cross-user commonality가 높고 personalization 가치가 낮음 | local/private/nonstationary experience가 중요 | tenant STC와 global refresh의 amortized cost·staleness 비교 |
| H-NICHE | STC는 personal agent, on-device adaptation, embodied system 같은 일부 workload에서만 주류다. | repeated user/device loop와 offline idle window가 명확 | workload가 one-shot·stateless·highly regulated | workload strata별 adoption utility와 governance cost 비교 |

## 5. 성공 기준

연구 패키지는 다음 조건을 동시에 만족해야 한다.

1. 여덟 질문 각각에 `현재 답`, `근거 수준`, `반대 근거`, `남은 실험`, `판정 범위`가 있다.
2. 모든 load-bearing claim은 fixed-version primary source와 page/line/section locator를 가진다.
3. promisingness, mainstream likelihood, technical feasibility, device value를 하나의 점수로 합치지 않는다.
4. STC arm은 각 문제의 strongest alternative와 matched compute/data/storage/governance budget으로 비교된다.
5. parametric, latent, text/vector/graph external memory의 capacity와 delete/rollback 차이를 명시한다.
6. scaling 관계는 established law와 candidate hypothesis를 시각적으로 구분한다.
7. Study Paper의 training 용어는 Training Background Paper의 정확한 section으로 cross-link된다.
8. figure는 license ledger를 통과하거나 원자료에서 독립적으로 redraw된다.
9. 12–15편 번역은 원문의 equation/table/figure/section 구조를 보존하며 해설을 삽입하지 않는다.
10. seminar slide는 claim/figure/source ID를 speaker notes에서 추적할 수 있다.

## 6. 명시적 제외와 비주장

- 생물학적 NREM/REM과 특정 ML objective의 일대일 동등성을 주장하지 않는다.
- “sleep”이라는 이름을 썼다는 이유만으로 STC deployment로 분류하지 않는다.
- company 내부 비공개 roadmap을 공개 논문에서 추정하지 않는다.
- parameter count를 semantic memory capacity의 충분한 척도로 사용하지 않는다.
- vendor 자체 benchmark 하나로 mainstream 또는 superiority를 판정하지 않는다.
- 별도 sleep cluster, CXL, near-memory compute를 선험적 정답으로 두지 않는다.
- 논문 figure의 공개 접근 가능성과 재배포 허가를 혼동하지 않는다.
- 2026-08-05 이후의 결과를 동결 corpus에 소급 포함하지 않는다.

## 7. Evidence maturity rubric

| level | 의미 | 공개 문장 형태 |
|---|---|---|
| E0 | 아이디어·명칭만 존재 | “제안됐다” |
| E1 | 단일 synthetic/offline experiment | “제한된 setting에서 관측됐다” |
| E2 | 다중 task/model, code 또는 독립 재현 일부 | “조건부로 지지된다” |
| E3 | matched strong baseline, ablation, long-horizon evaluation | “해당 범위에서 경쟁력이 입증됐다” |
| E4 | production lifecycle, reliability, governance, cost evidence | “배포 근거가 있다” |

expected value는 별도 `low / medium / high`로 기록한다. E1/high-value 아이디어와 E4/low-value utility를 구분하는 것이 이 연구의 핵심이다.

## 8. 기존 corpus의 사용 규칙

기존 audit와 monograph는 discovery와 discrepancy detection에 사용한다. 최종 claim은 다음 경로로 다시 승격한다.

```text
기존 audit 문장
→ 원 논문/공식 artifact fixed version 확인
→ exact locator와 quote/paraphrase 기록
→ supportive·qualifying·contradicting edge 부여
→ claim gate
→ Study / Background / Easy / Slide에서 재사용
```

주요 입력은 `SLEEP-TIME-COMPUTE-MONOGRAPH-KR.md`, `OPEN-QUESTIONS-ANSWERABILITY-MATRIX.md`, `SCALING-LAWS-THEORY-AGENDA.md`, `TRAINING-DATA-METHODS-ATLAS.md`, `SYSTEM-INFRA-BLUEPRINT.md`, primary-source audit 묶음, `research/google-meeting/` 근거표다.
