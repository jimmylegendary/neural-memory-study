# Unknown-Unknown and Disconfirmation Register

- 기준일: 2026-08-05
- 목적: `sleep`, `dream`, `consolidation`이라는 표면 용어를 쓰지 않는 인접 연구와, 중심 thesis를 약화하는 evidence를 의도적으로 찾는다.
- closure status: `open`, `bounded`, `closed`, `not-answerable-at-freeze`

## 1. Discovery register

| unknown_id | Ordinary search가 놓치는 이유 | Discovery query / family | 필요한 evidence | Owner artifact | Closure condition | status |
|---|---|---|---|---|---|---|
| UU-001 | database는 learning 대신 compaction/LSM merge라는 말을 쓴다. | `LSM compaction learned policy write amplification tiering` | foreground/background 분리, rewrite bytes, QoS law | Systems audit | STC operator와 직접 대응되는 primitive·non-equivalence 기록 | open |
| UU-002 | cache lifecycle은 memory 대신 admission/eviction/refresh를 쓴다. | `learned cache admission eviction refresh offline trace` | reuse-distance, hit/cost curves, stale invalidation | Scaling section | promotion threshold와 cache crossover를 수식화 | open |
| UU-003 | on-device adaptation은 sleep 대신 charging/idle/federated round를 쓴다. | `on-device personalization idle charging adapter training` | energy, privacy, thermal, update cadence, rollback | Device matrix | 최소 3개 primary system과 STC mapping | open |
| UU-004 | robotics는 memory consolidation보다 experience replay/world-model update라 부른다. | `continual robotics offline replay lifelong embodied agent memory` | nonstationary control, safety, physical consequence | Workload scenarios | embodied niche의 benefit/failure curve 확보 | open |
| UU-005 | federated personalization은 cross-device aggregation에 숨는다. | `federated continual personalization local adapter drift deletion` | local/global separation, privacy, version and delete | Mainstream scenarios | H-NICHE와 H-REFRESH를 가르는 근거 확보 | open |
| UU-006 | autonomous agent는 memory 대신 reflection, skill library, self-improvement를 쓴다. | `agent reflection skill library post-response daemon experience learning` | exact lifecycle, durable destination, evaluation leakage | Agent audit | post-response transform→later wake consumption 확인 | open |
| UU-007 | knowledge editing은 sleep이 아니라 batched edit로 같은 목적을 달성한다. | `sequential batch model editing accumulation locality rollback` | 100+ edits, paraphrase/application/composition, rollback | Alternatives paper | parametric STC의 strongest edit baseline 지정 | open |
| UU-008 | unlearning은 기억 추가 연구와 별도 community다. | `machine unlearning derived memory delete adapter checkpoint index` | causal delete across external/latent/weight copies | Governance section | end-to-end delete fan-out capability map | open |
| UU-009 | storage-near compute는 STC가 아니라 computational storage로 분류된다. | `computational storage near-data training embedding compaction CXL` | bytes moved, placement, programmability, isolation | Device opportunity | candidate device primitive와 workload intensity 도출 | open |
| UU-010 | recommendation systems는 continual user memory를 embedding refresh로 처리한다. | `recommender user embedding periodic refresh lifelong personalization` | update cadence, cold/warm state, serving consistency | Industry audit | personal-agent STC와 functional equivalence 판정 | open |
| UU-011 | database materialized view는 future query를 미리 계산한다. | `incremental materialized view maintenance query reuse background` | amortization and invalidation equations | Scaling section | anticipatory compute prior와 비교표 완성 | open |
| UU-012 | compiler PGO/autotuning도 wake trace→offline transform→later use lifecycle다. | `profile guided optimization offline trace later deployment` | reuse, staleness, publish/canary mechanics | System analogy | analogy와 ML-specific difference를 명시 | open |
| UU-013 | security literature는 durable memory poisoning을 model supply-chain 문제로 다룬다. | `persistent agent memory poisoning retrieval adapter backdoor` | attack persistence, quarantine, provenance, recovery | Threat model | external/parametric 각 attack surface와 gate 확보 | open |
| UU-014 | checkpoint/version pressure가 semantic capacity보다 먼저 포화될 수 있다. | `continual adapters checkpoint retention model version storage serving` | version count, recovery SLA, compatibility debt | Capacity section | governance capacity curve와 measurement plan | open |
| UU-015 | generated replay의 recursive teacher corruption은 generic synthetic-data collapse로 연구된다. | `recursive synthetic data model collapse replay canonical anchor` | generation depth, anchor mixture, error curves | Negative evidence | sleep dream arm의 corruption baseline 확보 | open |
| UU-016 | rare-event value는 average accuracy에서 사라진다. | `continual learning rare event tail utility retention safety critical` | value-weighted retention and tail metric | Benchmark | frequency와 value를 분리한 metric 지정 | open |
| UU-017 | exact source memory와 behavior memory가 같은 benchmark로 평가된다. | `agent memory benchmark exact recall skill transfer composition` | task taxonomy and metric leakage | Benchmark | fact/preference/skill/policy별 metric 분리 | open |
| UU-018 | inactive model time이 실제로 존재하지 않을 수 있다. | `LLM serving utilization diurnal idle fleet training scheduling` | utilization trace, model residency, burstiness | Infra paper | separate fleet vs shared fleet crossover 데이터 | open |
| UU-019 | sleep output가 작지 않으면 compute-to-data 이점이 사라진다. | `distributed training data movement delta checkpoint publish bandwidth` | source read, weight load, optimizer, delta size | Roofline model | bytes/FLOP and placement sensitivity sweep | open |
| UU-020 | base-model upgrade가 tenant memory를 무효화한다. | `adapter migration model upgrade latent memory compatibility` | compatibility matrix, recompile cost, failure modes | Lifecycle section | model churn 항을 scaling objective에 포함 | open |
| UU-021 | memory scheduler가 reward hacking을 할 수 있다. | `learned memory admission reinforcement learning reward hacking` | off-policy evaluation, long-term regret, safety constraints | Training section | RL scheduler의 safe baseline과 holdout protocol | open |
| UU-022 | privacy consent가 지나간 experience의 training 사용을 금지할 수 있다. | `personal data continual learning consent retention training deletion` | legal/technical data lineage requirements | Governance section | permitted data classes와 delete semantics 명시 | open |
| UU-023 | multimodal state가 text-centric memory hierarchy와 다르다. | `video embodied memory compression long-term agent multimodal` | video feature retention, temporal grounding, storage | Workload section | multimodal tier와 device bytes accounting | open |
| UU-024 | evaluation query generation이 training data와 누출될 수 있다. | `memory benchmark generated queries contamination evaluator leakage` | independent queries, temporal split, human labels | Claim gate | evaluation leakage test가 release gate에 포함 | open |

## 2. Disconfirmation register

| disconfirm_id | 중심 가설을 약화하는 관측 | 우선 탐색 문구 | 허용되는 판정 변화 | 필요한 control |
|---|---|---|---|---|
| DC-001 | external retrieval가 동일 latency·cost에서 parametric promotion과 동등하거나 우세 | `retrieval versus fine tuning knowledge injection matched` | H-EXT 강화, H-STC 범위 축소 | same model, source, queries, token/storage budget |
| DC-002 | sleep gain이 extra compute/data control로 완전히 설명 | `sleep replay ablation matched compute augmentation` | biological/stage framing 제거 | replay, dream, extra-data, no-sleep factorial |
| DC-003 | sequential weight writes가 수십~수백 item에서 composition과 old recall을 붕괴 | `continually learn facts weights interference` | weights=cache only 판정 | prompt ceiling, external source, multi-seed |
| DC-004 | learned summaries/graphs가 rare facts와 deletion provenance를 체계적으로 손실 | `memory compaction loss rare facts deletion` | external exact archive 의무화 | exact raw fallback and value-weighted metric |
| DC-005 | wake P99 간섭 때문에 shared fleet consolidation이 비경제적 | `background training inference interference tail latency` | separate/local fleet scenario 강화 | utilization and residency trace |
| DC-006 | 별도 fleet의 model/state movement가 compute 절감보다 큼 | `training checkpoint network bottleneck inference model residency` | storage-near/shared placement 강화 | bytes moved and FLOPs jointly measured |
| DC-007 | adapter/module growth의 router error와 load latency가 dense refresh보다 큼 | `mixture adapter routing many tasks serving` | H-REFRESH 강화 | same task mix and residency budget |
| DC-008 | repeated synthetic replay가 model collapse/confirmation bias를 증폭 | `recursive generation collapse continual replay` | canonical anchor와 verifier 의무화 | generation-depth sweep |
| DC-009 | delete/rollback requirement가 parametric memory를 실용 범위 밖으로 밀어냄 | `weight unlearning audit rollback continual` | parametric tier를 non-canonical cache로 제한 | end-to-end derived-state inventory |
| DC-010 | query reuse가 낮아 compile cost를 회수하지 못함 | `offline document compilation amortization query frequency` | H-NICHE/H-EXT 강화 | real workload reuse distribution |
| DC-011 | product “sleep” 기능이 사실 ingest-time summary 또는 manual job | `official docs async memory reflection schedule` | mainstream likelihood 하향 | tagged code/release lifecycle |
| DC-012 | STC 관련 논문 증가가 독립 연구팀이 아닌 한 lineage의 반복 출판 | affiliation/citation genealogy audit | evidence independence 하향 | unique team/dataset/code accounting |

## 3. Search hygiene

- positive와 negative query 결과를 같은 log에 섞지 않는다.
- abstract-only, withdrawn, unaccepted submission, vendor benchmark를 status 그대로 보존한다.
- 논문 수가 많아져도 independent dataset·team·workload가 늘지 않으면 evidence maturity를 올리지 않는다.
- source freeze 이후 변경된 product page는 retrieved hash와 release/tag가 없으면 historical claim에 쓰지 않는다.
- 검색 실패는 absence proof가 아니다. targeted public-corpus audit 범위로 문장을 제한한다.
