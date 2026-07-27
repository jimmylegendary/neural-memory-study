# Google 미팅용 6축 Problem / Open-Question Map

> 대상 논문: **Nested Learning / HOPE**, **Language Models Need Sleep**, **Memory Caching**, **NSTM**<br>
> 목적: 개별 논문을 요약하는 데 그치지 않고, Google과 Samsung SAIT가 공유할 수 있는 문제의식을 여섯 개의 연구·시스템 축으로 재정의한다.<br>
> 근거 기준일: 2026-07-27. 아래 행 번호는 저장소의 논문 원문 추출본 기준이다.

> **패키지 역할:** ① evidence pre-read / problem map · [② canonical facilitator playbook](02-GOOGLE-KICK-QUESTION-PLAYBOOK.md) · [③ device/system appendix](03-DEVICE-SOLUTION-BRIDGE.md)

---

## 0. 읽는 법과 증거 등급

이 문서는 논문의 주장과 우리의 확장을 의도적으로 분리한다.

| 표기 | 의미 | 미팅에서의 사용법 |
|---|---|---|
| **D-P** | 논문이 직접 제시한 problem statement | “논문이 푸는 문제”로 그대로 말해도 됨 |
| **D-O** | discussion·limitation·future work에서 저자가 직접 연 open issue | Google에 저자 관점의 설명을 요청하기 좋은 지점 |
| **D-R** | 논문이 실험·분석으로 직접 보인 result/constraint | 질문의 전제 또는 예상 답변의 근거 |
| **I-Q** | 여러 논문을 연결해 우리가 도출한 research question | “우리의 해석/가설”이라고 명시해야 함 |
| **I-S** | serving·memory tier·device 관점으로 옮긴 시스템 추론 | 공동연구 또는 device solution으로 연결할 부분 |

핵심적으로 다음 세 문장을 혼동하지 않는다.

1. **Memory Caching이 prefix cache를 해결했다** — 사실이 아니다. 논문은 한 시퀀스 내부의 recurrent-state checkpoint와 query-time aggregation을 다룬다.
2. **NSTM이 LLM serving을 해결했다** — 사실이 아니다. NSTM은 온라인 dynamic NVS에서 read/write frequency asymmetry를 실증했다. LLM owner/version scheduling은 우리의 전이 가설이다.
3. **Sleep이 무한 continual learning을 해결했다** — 사실이 아니다. 새 low-rank expert를 활성화하는 실험적 방법을 보였지만, lifetime slot exhaustion·merge·prune·eviction 정책은 검증하지 않았다.

---

## 1. 네 논문이 각각 푸는 문제와 스스로 남긴 문제

### 1.1 Nested Learning / HOPE — online self-modification과 multi-frequency memory

#### 논문이 직접 정의한 문제와 제안한 해법

- **D-P · static deployment gap.** 배포 뒤 LLM은 pre/post-training에서 얻은 오래된 지식과 현재 context만 사용할 뿐, context 밖으로 새로운 능력을 지속적으로 축적하지 못한다. 기존 대안은 계산비용, 외부 구성요소, 일반화, catastrophic forgetting 가운데 하나 이상에서 문제를 가진다고 규정한다. [P-HOPE:78–94](../../papers/2512.24695.txt#L78-L94)
- **D-R · learning architecture의 재정의.** 모델·optimizer·backpropagation을 분리된 부품으로 보지 않고, 서로 다른 context flow와 update frequency를 가진 nested associative memories로 통합해 continual/in-context/pre-training을 같은 압축 문제로 재정식화한다. [P-HOPE:6–25](../../papers/2512.24695.txt#L6-L25), [P-HOPE:3537–3552](../../papers/2512.24695.txt#L3537-L3552)
- **D-P · 단일 fast state의 제한.** Transformer는 context를 정확히 보존하지만 계산 깊이와 in-context self-modification이 제한되고, 같은 objective·parameter search space의 parametric RNN은 scale에서 perfect non-parametric attention을 단순 모방해서 이기기 어렵다고 본다. [P-HOPE:1996–2020](../../papers/2512.24695.txt#L1996-L2020)
- **D-R · long/short 이분법의 확장.** CMS는 MLP blocks를 서로 다른 update period로 배치해 fast adaptation과 persistent knowledge를 연속적인 frequency spectrum으로 만든다. [P-HOPE:213–230](../../papers/2512.24695.txt#L213-L230), [P-HOPE:1768–1798](../../papers/2512.24695.txt#L1768-L1798)
- **D-R · 표현력과 용량의 결합.** Self-modifying Titans는 작은 state에 복잡한 update rule을, CMS는 큰 persistent capacity에 단순 update를 맡기고 HOPE가 둘을 직렬 결합한다. [P-HOPE:2239–2249](../../papers/2512.24695.txt#L2239-L2249)

#### 논문이 직접 드러낸 제약과 open issue

- **D-O · catastrophic forgetting은 미해결.** 저자는 “Is Catastrophic Forgetting Solved?”에 명시적으로 아니라고 답한다. 유한 용량 아래 compression이 필연적으로 forgetting을 낳으며, NL은 solution이 아니라 levels라는 추가 설계축을 제시하는 roadmap이다. [P-HOPE:3553–3559](../../papers/2512.24695.txt#L3553-L3559)
- **D-R · optimizer 자체의 장기문맥 한계.** vanilla momentum은 최근 43개 gradient가 누적 기여의 99%를 차지하는 low-pass filter라서 장기 loss landscape와 과거 task subspace를 잃는다. 저자는 이것을 model capacity가 아니라 optimizer memory-management failure로 분리한다. [P-HOPE:1113–1153](../../papers/2512.24695.txt#L1113-L1153)
- **D-O · 효율·병렬화.** self-modifying method의 실용적 핵심 난제를 training efficiency와 parallelizability라고 직접 지목하고, chunk-start state를 고정한 parallel gradient/dual form으로 대응한다. [P-HOPE:2162–2212](../../papers/2512.24695.txt#L2162-L2212)
- **D-O · 공정한 systems 비교 부재.** Cartridges와는 memory usage와 self-studying compute가 달라 통제된 비교를 하지 않았으며, 향후 controlled study가 필요하다고 적는다. [P-HOPE:2625–2629](../../papers/2512.24695.txt#L2625-L2629)
- **D-R · long-context capacity 의존.** 10M-token 과제를 풀려면 답에 필요한 token을 압축할 충분한 capacity와 해당 memory management를 학습시키는 fine-tuning이 필요하다고 관찰한다. [P-HOPE:3091–3100](../../papers/2512.24695.txt#L3091-L3100)

#### 논문 사이에서 드러나는 미해결 이음새

- **I-Q.** HOPE의 CMS는 **온라인 consolidation**이며 원문 자체가 offline systems consolidation을 연구 범위 밖에 둔다. 따라서 wake에서 형성된 state 가운데 무엇을 offline consolidation 대상으로 인정하고, Sleep의 pre-update full-model teacher를 어떻게 qualify할지는 미정이다. [P-HOPE:102–120](../../papers/2512.24695.txt#L102-L120)
- **I-S.** §8.2는 prefill/training의 chunk 병렬화이지, batch별로 다른 mutable state를 가진 decode kernel, state paging, prefix reuse, update isolation을 푼 것이 아니다.
- **I-Q.** CMS frequency가 “얼마나 자주 업데이트할지”는 표현하지만, **왜 지금 업데이트해야 하는지**를 판단하는 learned trigger나 “무엇을 지워야 하는지”의 capacity-control policy는 제공하지 않는다.

---

### 1.2 Language Models Need Sleep — offline parametric consolidation과 dreaming

#### 논문이 직접 정의한 문제와 제안한 해법

- **D-P · short-to-long transfer.** ICL이 얻은 지식은 context/session 끝에 사라진다. 논문은 “fragile short-term memory를 stable long-term parameter로 어떻게 옮기는가”를 중심 질문으로 둔다. [P-SLEEP:13–18](../../papers/2606.03979v2.txt#L13-L18), [P-SLEEP:45–76](../../papers/2606.03979v2.txt#L45-L76)
- **D-P · online consolidation의 한계.** HOPE식 online transfer는 같은 model capacity와 같은 abstraction level을 사용하고, active retrieval에 편향되며, 현재 context만 보아 새 지식과 기존 지식의 상위 abstraction을 놓친다고 지적한다. [P-SLEEP:92–107](../../papers/2606.03979v2.txt#L92-L107)
- **D-R · wake/sleep lifecycle.** continual learner의 lifecycle을 train/test가 아니라 external input을 처리하는 wake와, 새 입력 없이 기존 지식을 통합·자기개선하는 sleep으로 나눈다. [P-SLEEP:146–160](../../papers/2606.03979v2.txt#L146-L160), [P-SLEEP:312–327](../../papers/2606.03979v2.txt#L312-L327)
- **D-R · consolidation operator.** sender의 base-weight update를 계산만 한 pre-update state를 teacher로, prospective sender update·fast-expert reset·새 slow low-rank expert를 포함한 expanded state를 student로 둔다. GKD+RL imitation으로 새 expanded parameter만 학습한 뒤 sender update, fast-expert reset, slow-expert activation을 적용한다. [P-SLEEP:373–450](../../papers/2606.03979v2.txt#L373-L450), [P-SLEEP:456–489](../../papers/2606.03979v2.txt#L456-L489)
- **D-R · dream-data operator.** model이 synthetic dreams를 생성하고, gradient-based importance로 top-k+random을 선택하며, 개별 LoRA SFT의 downstream improvement를 reward로 ReSTEM을 수행한다. [P-SLEEP:499–554](../../papers/2606.03979v2.txt#L499-L554)

#### 논문이 직접 드러낸 제약과 open issue

- **D-O · repeated consolidation bottleneck.** 1K→10K frequency chain에서는 slow block 한 번의 update 전에 fast→slow consolidation이 10회 필요하며, 저자는 이를 continual learning의 critical bottleneck으로 규정한다. [P-SLEEP:349–356](../../papers/2606.03979v2.txt#L349-L356)
- **D-O · offline-data constraint.** sleep에서는 외부 dataset 접근이 제한되어 conventional distillation이 바로 적용되지 않으며, larger student가 teacher-generated data만으로 capacity를 제대로 쓰지 못할 수 있다. [P-SLEEP:425–432](../../papers/2606.03979v2.txt#L425-L432)
- **D-O · iterative dream failure.** inner SFT가 비싸 dream 수를 제한하고, 반복 self-improvement는 catastrophic forgetting을 낳을 수 있으며, 자기 지식 공간만 sampling하면 novel synthesis가 부족하다. [P-SLEEP:506–515](../../papers/2606.03979v2.txt#L506-L515)
- **D-R · fixed-capacity implementation tension.** 동적 tensor growth가 어렵기 때문에 실제 구현은 처음부터 parameter를 준비해 masking/activation할 수 있다고 한다. 즉 논문 속 “성장”은 physical allocation growth가 아니라 finite preallocation의 logical activation일 수도 있다. [P-SLEEP:490–495](../../papers/2606.03979v2.txt#L490-L495)
- **D-O · self-distillation collapse risk.** 부록은 privileged teacher가 OOD reasoning을 크게 악화시키고, naive iterative self-distillation이 information leakage/training collapse를 낳을 수 있다고 정리하며, consolidate-then-dream 분리가 이 위험을 줄인다고 주장한다. [P-SLEEP:1448–1459](../../papers/2606.03979v2.txt#L1448-L1459) **I-Q:** 이 결과는 무한 수명 반복에서의 안정성을 증명하지 않는다.

#### 논문이 말하지 않은 핵심 공백

- **I-Q · lifetime exhaustion.** 새 slow expert를 계속 활성화하지만 총 slot 수, slot 소진 뒤 merge/prune/recycle/offload, router entropy와 expert fragmentation, 오래된 expert의 검증·rollback 정책은 없다.
- **I-Q · sleep trigger.** paper의 sleep은 block frequency boundary에 고정된다. surprise, novelty, conflict, memory-only probe failure, idle-resource price를 함께 보는 event-driven trigger는 검증되지 않았다.
- **I-S · cluster economics.** sleep FLOPs, state movement bytes, wake SLA 손실, expert placement, replica synchronization을 포함한 cost model 및 sleep budget scaling law가 없다.
- **I-Q · teacher qualification.** Knowledge Seeding의 pre-update full-model teacher가 recent noise나 corrupted fast memory를 포함할 때 이를 통째로 신뢰해도 되는지 검증 규칙이 없다. 이 공백에 NSTM의 stable-read shadow state가 후보가 될 수 있지만 이는 결합 가설이다.

---

### 1.3 Memory Caching — fixed-state RNN의 용량을 history checkpoints로 확장

#### 논문이 직접 정의한 문제와 제안한 해법

- **D-P · attention–RNN dilemma.** attention은 context와 함께 addressable memory capacity가 자라고 dense compute는 O(L²)인 반면, RNN은 O(L)이지만 fixed state overflow 때문에 recall-intensive long context에서 진다. [P-MC:14–33](../../papers/2602.24281.txt#L14-L33), [P-MC:36–54](../../papers/2602.24281.txt#L36-L54) **I-S:** 물리 비용을 분리하면 KV-cache capacity와 token당 decode read는 L에 선형이다. FlashAttention 계열은 S² score matrix의 HBM materialization을 피하지만 dense-attention FLOPs와 누적 KV read를 없애지는 않는다.
- **D-R · growing recurrent memory.** sequence를 segment로 나누고 각 segment 끝의 memory state를 cache하여, 현재 online memory와 과거 cached memories에 동일 query를 적용한다. update rule은 유지하고 retrieval 대상만 늘린다. [P-MC:151–180](../../papers/2602.24281.txt#L151-L180)
- **D-R · controlled interpolation.** segment 수 `N`으로 O(L) recurrence와 O(L²) attention 사이 O(L + pNL)를 조절하며, SSC는 query-dependent router로 일부 checkpoint만 선택한다. [P-MC:511–544](../../papers/2602.24281.txt#L511-L544), [P-MC:327–358](../../papers/2602.24281.txt#L327-L358)
- **D-R · 네 aggregation variants.** Residual, gated residual, Memory Soup, SSC를 통해 all-cache read, gated read, parameter interpolation, sparse selection을 비교한다. [P-MC:69–80](../../papers/2602.24281.txt#L69-L80)

#### 논문이 직접 드러낸 제약과 open issue

- **D-R · recall gap remains.** in-context recall에서는 Transformer가 여전히 최고이며 MC는 recurrent baseline과의 격차를 줄이는 수준이다. [P-MC:783–798](../../papers/2602.24281.txt#L783-L798)
- **D-R · segmentation Pareto.** 짧은 segment는 해상도와 recall을 보존하지만 캐시 수와 compute를 늘린다. logarithmic segmentation은 O(L log L)로 싸지만 먼 과거를 거칠게 압축해 recall을 손상시킨다. [P-MC:511–544](../../papers/2602.24281.txt#L511-L544)
- **D-O · checkpoint vs independent compressor.** 하나의 optimizer trajectory snapshot을 잇는 방식과 segment별 독립 compressor는 서로 다른 interference/continuity 특성을 가지며, 논문은 둘 다 장단이 있다고 남긴다. [P-MC:367–385](../../papers/2602.24281.txt#L367-L385)
- **D-O · routing quality.** 결론은 더 expressive한 pooling/routing을 future work로 명시한다. [P-MC:871–876](../../papers/2602.24281.txt#L871-L876)
- **D-R · query-dependent cache는 재사용 불가.** gated variant의 token-dependent weights는 미리 합치거나 다음 token에 재사용할 수 없어 매 token recompute와 state caching이 필요하다. [P-MC:206–240](../../papers/2602.24281.txt#L206-L240)

#### 논문이 말하지 않은 핵심 공백

- **I-S · serving state accounting.** cached memory 하나가 fast-weight module 하나의 parameter set일 때 `N` checkpoints는 per-session state를 `N×` 키운다. paper는 accelerator residency, HBM/host/SSD tiering, cache miss, prefetch, eviction을 측정하지 않는다.
- **I-S · prefix-cache semantics.** cached recurrent state가 동일 prefix를 공유한 request 사이에서 안전하게 재사용되는 조건—model version, initial state, update rule, stochasticity, tenant/privacy, branch 이후 copy-on-write—을 정의하지 않는다.
- **I-Q · state equality.** parameter-space 평균이 function-space 의미를 보존하는 조건을 이론화하지 않는다. NSTM은 특정 domain에서 실패/성공 조건의 강한 실험 단서를 주지만 일반 정리는 아니다.
- **I-Q · bounded MC.** SSC가 active read 수를 `top-k`로 제한해도 저장된 checkpoint 총수는 계속 자랄 수 있다. total capacity를 제한하는 retention/merge/eviction objective는 없다.

---

### 1.4 NSTM — live stream에서 heavy write와 frequent read를 분리

#### 논문이 직접 정의한 문제와 제안한 해법

- **D-P · real-time vs persistent memory.** online dynamic NVS는 가려진 영역을 복원할 minute-scale memory와 strict real-time latency를 동시에 요구한다. per-frame TTT update는 비싸고 long-horizon drift를 낳는다. [P-NSTM:51–70](../../papers/2607.15271.txt#L51-L70)
- **D-R · update/apply asymmetry.** H100에서 update+apply는 58.14 ms, apply-only는 27.01 ms다. NSTM은 memory write를 1 FPS로, read/apply를 30 FPS로 분리한다. [P-NSTM:92–110](../../papers/2607.15271.txt#L92-L110), [P-NSTM:721–742](../../papers/2607.15271.txt#L721-L742)
- **D-R · memory-bypass 방지.** 현재 입력을 보는 cross-view attention이 쉬운 우회로가 되어 parametric memory를 무시할 수 있으므로, current input을 차단한 memory-only reconstruction loss를 교대로 학습한다. [P-NSTM:112–136](../../papers/2607.15271.txt#L112-L136), [P-NSTM:301–315](../../papers/2607.15271.txt#L301-L315)
- **D-R · drift stabilization.** dot-product inner loss의 magnitude inflation을 L2 objective, Muon, weight normalization으로 완화하고, active state와 historical checkpoint running mean을 결합해 읽는다. [P-NSTM:232–248](../../papers/2607.15271.txt#L232-L248), [P-NSTM:364–383](../../papers/2607.15271.txt#L364-L383)

#### 논문이 직접 드러낸 제약과 open issue

- **D-O · update frequency vs missed events.** periodic update 사이의 사건을 놓칠 수 있고, time embedding을 가진 large-chunk update를 후보로 제시하며 frequency–chunk-size trade-off를 열어 둔다. [P-NSTM:483–493](../../papers/2607.15271.txt#L483-L493)
- **D-O · finite capacity.** fixed memory는 결국 capacity limit에 도달하고 learned caching weights가 필요하다고 직접 적는다. [P-NSTM:490–493](../../papers/2607.15271.txt#L490-L493)
- **D-O · primacy bias.** 초기 observation을 더 잘 기억하며, longer-sequence training과 mid-sequence sampling이 필요하다. [P-NSTM:493–497](../../papers/2607.15271.txt#L493-L497)
- **D-O · complex motion.** compressed memory 안에서 long-range camera/subject motion을 푸는 일은 여전히 어렵다. [P-NSTM:497–503](../../papers/2607.15271.txt#L497-L503)
- **D-O · live streaming과 chunk buffering.** prior chunk-parallel TTT는 chunk buffering latency 때문에 strict online setting에 맞지 않으며, per-frame sequential update는 58 ms라 비현실적이다. [P-NSTM:256–274](../../papers/2607.15271.txt#L256-L274)

#### 직접 결과에서 도출되는 이음새

- **D-R · aggregation은 구조 의존.** coupled LaCT fast weights에 cache 평균을 적용하면 temporal pose가 parameter에 섞여 27.77→20.25 PSNR로 붕괴하고, deformation을 별도 attention으로 분리한 NSTM에서는 caching이 29.24→30.09로 이득이다. 즉 “모든 fast-weight snapshot은 평균 가능”하지 않다. [P-NSTM:391–400](../../papers/2607.15271.txt#L391-L400), [P-NSTM:423–442](../../papers/2607.15271.txt#L423-L442), [P-NSTM:458–481](../../papers/2607.15271.txt#L458-L481)
- **D-R · repeated read independence.** 두 memorization 사이 synthesis들은 동일 frozen memory state를 읽고 서로 독립이며, training은 1:1이지만 inference는 29:1도 가능하다. [P-NSTM:301–310](../../papers/2607.15271.txt#L301-L310)
- **I-Q · 일반화 가설.** LLM에서도 consolidation 대상에는 비교적 invariant한 semantic memory를, turn/session-specific alignment에는 별도 context adapter를 두어야 parameter merge가 안전할 가능성이 있다. 이는 NVS 결과의 domain transfer이며 아직 검증되지 않았다.
- **I-S · serving 가설.** read-only requests를 동일 state version으로 batch하고 update requests를 별도 queue에서 처리한 뒤 atomic version switch하는 스케줄이 가능하다. NSTM은 이 구조의 계산 비대칭을 보여주지만 multi-tenant LLM batching을 구현하지 않았다.

---

### 1.5 관련 primary source가 채워 주는 systems 경계: TTT

- **D-P.** TTT는 fixed-size RNN state의 표현력 한계를, hidden state 자체를 test sequence에서 학습되는 model weight로 바꿔 해결하려 한다. 동시에 TTT-MLP의 memory I/O를 미해결 과제로 남긴다. [P-TTT:14–26](../../papers/external/2407.04620.txt#L14-L26)
- **D-R.** mini-batch TTT와 dual form은 TPU에서 naive primal보다 5× 이상 빠르지만, arbitrary neural state는 여전히 wall-clock overhead가 크다. [P-TTT:188–202](../../papers/external/2407.04620.txt#L188-L202)
- **D-R.** paper는 prefill/forward와 backward에는 dual form, inherently sequential decode에는 primal form을 사용한다. 이 구분이 HOPE 분석에서도 유지되어야 한다. [P-TTT:1217–1230](../../papers/external/2407.04620.txt#L1217-L1230)
- **D-O.** systems optimization은 preliminary이며, million-token sequence를 위한 time-pipeline parallelism과 larger hidden model이 future work다. [P-TTT:1399–1418](../../papers/external/2407.04620.txt#L1399-L1418)

---

## 2. 여섯 축의 한눈 지도

범례: **●** 직접 주목적/해결, **◐** 일부 메커니즘·실험, **○** 직접 open issue, **△** 우리 추론으로만 연결.

| 논문 | ① Long context / test-time scaling | ② State reuse / serving | ③ Train·prefill·decode 효율 | ④ Long-term / continual | ⑤ Bounded capacity | ⑥ Sleep-time scaling |
|---|---:|---:|---:|---:|---:|---:|
| HOPE / NL | ● | △ | ◐·○ | ●·○ | ○ | △ |
| Language Models Need Sleep | ◐ | △ | ○ | ● | ◐·○ | ●·○ |
| Memory Caching | ●·○ | ◐·○ | ◐ | △ | ◐·○ | △ |
| NSTM | ◐ | ◐·○ | ●·○ | ◐ | ○ | △ |
| TTT supporting source | ●·○ | △ | ●·○ | ◐ | ○ | △ |

여섯 축은 독립적인 checklist가 아니라 다음 인과 사슬이다.

> **더 긴 context를 실제로 이해하려면(①)** 압축 state가 필요하고 → 그 state를 효율적으로 재사용·격리해야 하며(②) → training/prefill/decode마다 다른 kernel과 scheduling을 설계해야 하고(③) → wake update를 durable knowledge로 바꾸는 continual-learning 원리가 필요하며(④) → 유한 용량에서 select/merge/forget을 해야 하고(⑤) → 이 비동기 변환에 별도 compute budget을 투입하는 sleep scaling이 등장한다(⑥).

---

## 3. 축 ① — Full attention이 못 푸는 long-context 이해와 test-time scaling

### 3.1 문제 재정의

문제는 단순히 “context window가 몇 token인가”가 아니다.

1. **Retention:** 필요한 사실이 state에 남아 있는가.
2. **Retrieval:** query가 그 사실을 선택할 수 있는가.
3. **Computation:** 단순 lookup을 넘어 긴 context 위의 state tracking·algorithmic computation을 수행하는가.
4. **Generalization:** training length 밖에서도 위 세 기능이 유지되는가.
5. **Economics:** 추가 context 또는 추가 test-time compute가 latency·HBM·energy 대비 유의미한 quality gain을 내는가.

### 3.2 직접 근거

- **D-P · Attention의 장점과 비용:** 모든 token에 직접 접근하는 growing memory로 recall은 강하지만 dense attention은 O(L²) complexity를 지불한다. MC가 이 trade-off를 논문의 출발점으로 둔다. [P-MC:14–33](../../papers/2602.24281.txt#L14-L33) **I-S:** 구현상 KV-cache capacity와 token당 decode read는 O(L)이며, FlashAttention은 S² score를 HBM에 저장하지 않지만 compute와 누적 KV read는 남는다.
- **Fixed-state RNN의 실패:** context를 고정 state로 압축하면 overflow/forgetting이 발생한다. TTT와 MC가 독립적으로 같은 diagnosis를 제시한다. [P-TTT:108–124](../../papers/external/2407.04620.txt#L108-L124), [P-MC:45–54](../../papers/2602.24281.txt#L45-L54)
- **Capacity만으로 충분하지 않음:** HOPE는 Transformer의 limited computational depth와 fixed contextualization을 별도 한계로 지적한다. [P-HOPE:1996–2020](../../papers/2512.24695.txt#L1996-L2020)
- **MC의 partial answer:** checkpoint 수를 늘려 compression resolution을 키우지만 recall 왕좌는 Transformer가 유지한다. [P-MC:783–798](../../papers/2602.24281.txt#L783-L798)
- **HOPE의 partial answer:** deep memory + self-modification + CMS는 long-context 성능을 높이지만, 10M 성능은 task-specific fine-tuning과 충분한 capacity에 의존한다. [P-HOPE:2679–2686](../../papers/2512.24695.txt#L2679-L2686), [P-HOPE:3091–3100](../../papers/2512.24695.txt#L3091-L3100)

### 3.3 아직 답하지 못한 질문

- **I-Q1.1 · understanding decomposition.** attention, fast weight, cached snapshots, slow expert가 각각 retention/retrieval/computation 가운데 무엇을 담당할 때 가장 효율적인가?
- **I-Q1.2 · test-time scaling law.** quality를 `Q(C_ctx, C_wake-update, C_retrieval, C_sleep)`로 볼 때 네 compute 축의 대체율과 포화점은 무엇인가?
- **I-Q1.3 · compression-aware benchmark.** NIAH 성공이 “이해”를 과대평가하지 않도록, state tracking·multi-hop·interference·counterfactual update·long-horizon calibration을 어떻게 함께 측정할 것인가?
- **I-Q1.4 · exact vs lossy memory.** 어떤 token/episode는 exact KV로 남기고 어떤 정보는 parametric state로 lossy compress하며 어떤 것은 external index로 offload할 것인가?

### 3.4 미팅에서 확인해야 할 Google의 암묵적 선택

- HOPE의 long-context gain을 **capacity 증가**, **update-rule 표현력**, **CMS frequency**, **추가 training compute**로 분해한 ablation이 있는가?
- Transformer와 HOPE/MC를 동일 total HBM bytes, total FLOPs, wall-clock, parameter/state budget으로 맞췄을 때 Pareto가 유지되는가?
- “더 긴 context를 잘 쓴다”의 내부 operational metric을 retrieval accuracy 외에 무엇으로 보고 있는가?

---

## 4. 축 ② — Recurrent-state serving, prefix cache, state reuse

### 4.1 문제 재정의

Transformer serving은 shared immutable weights + request-specific KV라는 비교적 명확한 contract를 가진다. self-modifying RNN은 request/session마다 **weights 일부 자체가 달라지는 mutable model**이므로 contract가 바뀐다.

필요한 상태 identity는 최소 다음 tuple이다.

`StateID = (base_model_version, memory_init, prefix_hash, update_rule_version, update_count, owner, privacy_domain, RNG/data-order provenance)`

이 tuple이 일치하지 않으면 “같은 prefix”여도 fast state를 그대로 공유하기 어렵다.

### 4.2 직접 근거와 경계

- **MC가 제공하는 것:** segment-end state checkpoint와 query-dependent selection. [P-MC:151–180](../../papers/2602.24281.txt#L151-L180)
- **MC가 제공하지 않는 것:** multi-request prefix deduplication, branch copy-on-write, tenant isolation, state-version coherency.
- **NSTM이 제공하는 것:** 동일 frozen memory state를 여러 synthesis step이 독립적으로 읽을 수 있고, write 1회 사이 read를 29회 수행할 수 있다는 실증. [P-NSTM:301–310](../../papers/2607.15271.txt#L301-L310)
- **NSTM이 제공하지 않는 것:** request마다 다른 fast weight를 가진 LLM batch의 grouped GEMM, scheduler, state migration.
- **HOPE가 제공하는 것:** chunk 내부 parallel update 계산. [P-HOPE:2162–2212](../../papers/2512.24695.txt#L2162-L2212)
- **HOPE가 제공하지 않는 것:** prefix-cache hit semantics와 serving rollback.

### 4.3 핵심 open questions

- **I-Q2.1 · shareability.** fast state가 deterministic function of prefix일 때 어느 layer·update step까지 prefix state를 공유할 수 있는가? branch 이후에는 full copy, delta log, low-rank patch 중 무엇이 유리한가?
- **I-Q2.2 · reuse unit.** state 재사용 단위를 token prefix, segment checkpoint, CMS level, expert, stable-read snapshot 가운데 무엇으로 정할 것인가?
- **I-Q2.3 · validity.** state cache hit를 byte-identical state로 정의할지, functionally equivalent output tolerance로 정의할지?
- **I-Q2.4 · mutable batching.** `(StateID, active expert set, shape)`가 같은 request를 묶고, write 요청은 별도 update queue로 분리한 뒤 atomic version switch할 수 있는가?
- **I-Q2.5 · state placement.** active mutable state는 HBM, stable read-only checkpoints는 capacity tier, cold snapshots는 host/SSD에 두는 계층에서 prefetch miss가 decode SLA를 얼마나 해치는가?
- **I-Q2.6 · security.** learned state가 raw prefix보다 더 압축됐다는 이유로 tenant 간 공유할 수 있는가, 아니면 membership/privacy leakage 때문에 같은 protection domain이 필요한가?

### 4.4 시스템 가설

- **I-S2.A · read/write split scheduler.** `read_epoch`에서는 같은 state version의 requests를 크게 batch하고, `write_epoch`는 별도 accelerator/stream에서 update한 뒤 version pointer를 바꾼다.
- **I-S2.B · copy-on-write fast state.** shared prefix snapshot은 immutable base로 두고 request-specific updates를 low-rank/delta log로 누적한다. merge threshold에서 새 checkpoint를 만든다.
- **I-S2.C · content-addressed state.** raw token hash뿐 아니라 update semantics까지 포함한 StateID로 cache key를 구성한다.
- **I-S2.D · bounded reuse.** reuse hit-rate, bytes moved, recompute FLOPs, staleness quality loss를 함께 최소화하는 admission policy가 필요하다.

이 네 항목은 논문 결과가 아니라 검증할 systems hypotheses다.

---

## 5. 축 ③ — Recurrent model의 효율적 pre-training, prefill, decode

### 5.1 phase마다 다른 계산 그래프

| Phase | 사용 가능한 병렬성 | 핵심 병목 | 논문 근거 |
|---|---|---|---|
| Outer-loop pre-training | sequence·batch·device 병렬, activation backward 필요 | nested gradient graph, checkpointing, state materialization | TTT dual form 및 5× TPU speedup [P-TTT:188–202](../../papers/external/2407.04620.txt#L188-L202) |
| Prefill / forward | 전체 prompt가 알려져 chunk dual form 가능, backward activation 불필요 | chunk boundary state, GEMM화, HBM state traffic | TTT가 prefill=forward, dual form이라고 명시 [P-TTT:1217–1221](../../papers/external/2407.04620.txt#L1217-L1221) |
| Decode | 다음 token이 생성돼야 다음 update 가능 | sequential RMW, per-session state read/write, small GEMM utilization | TTT가 primal form 사용 [P-TTT:1217–1230](../../papers/external/2407.04620.txt#L1217-L1230) |
| Live-stream read-mostly | future chunk buffering 불가, 동일 frozen state read는 병렬 | update cadence, missed event, version switch | NSTM [P-NSTM:256–310](../../papers/2607.15271.txt#L256-L310) |
| Sleep | **D-R:** teacher generation과 student learning. **I-S:** wake latency path와 분리한 batch/throughput scheduling 후보 | **D-R:** GKD/LTI compute. **I-S:** state movement, validation, publication | Sleep GKD/LTI [P-SLEEP:425–489](../../papers/2606.03979v2.txt#L425-L489) |

### 5.2 직접적으로 해결된 부분

- **HOPE/TTT:** chunk-start state에 대해 gradients를 병렬 계산하고 dual form으로 outer products를 materialize하지 않는 방향. [P-HOPE:2162–2212](../../papers/2512.24695.txt#L2162-L2212), [P-TTT:584–602](../../papers/external/2407.04620.txt#L584-L602)
- **MC:** context 증가 시 Transformer보다 training throughput이 유리하고 SSC overhead가 작다는 proof-of-concept. 그러나 end-to-end serving latency나 memory-tier bytes는 아니다. [P-MC:862–869](../../papers/2602.24281.txt#L862-L869)
- **NSTM:** update+apply와 apply-only의 실제 latency 차이를 측정하고 update frequency를 낮춰 amortize한다. [P-NSTM:721–742](../../papers/2607.15271.txt#L721-L742)

### 5.3 직접 또는 간접 open issues

- **D-O / I-Q3.1 · chunk size.** TTT/HOPE chunk size는 parallelism과 online fidelity의 trade-off다. NSTM은 buffering latency와 missed event라는 별도 축을 보여준다. 한 개의 static chunk size로 training/prefill/live decode를 모두 최적화할 수 없다.
- **I-Q3.2 · kernel boundary.** update에 필요한 `(Mk−v)kᵀ`, optimizer state, weight RMW를 얼마나 fuse하고 intermediate tensor를 SRAM/register에 유지할 수 있는가?
- **I-Q3.3 · heterogeneous state.** session마다 다른 MLP weights를 읽는 decode를 dense batched GEMM처럼 활용할 것인가, grouped GEMM, block-sparse expert batching, PIM update 중 무엇이 필요한가?
- **I-Q3.4 · state checkpoint cost.** MC snapshot과 CMS levels를 HBM에 모두 둘 수 없을 때 prefetch overlap과 recompute 경계는 어디인가?
- **I-Q3.5 · training–inference mismatch.** NSTM처럼 training cadence 1:1, inference 29:1인 모델이 frequency shift에도 안정적인 조건은 무엇인가? HOPE도 train chunk와 online decode cadence가 달라질 때 검증이 필요하다.
- **I-Q3.6 · full-budget fairness.** attention과 recurrent memory를 FLOPs만이 아니라 HBM bytes, mutable-state bytes, kernel launch, synchronization, prefix hit-rate까지 포함해 비교해야 한다.

### 5.4 device/system 관찰량

향후 실험에서는 최소 다음을 phase별로 분리 측정해야 한다.

- physical DRAM read/write bytes와 L2 hit rate
- mutable parameter + optimizer state RMW bytes
- chunk size별 tensor-core utilization
- state checkpoint 생성·restore·migration bytes
- same-state batch size와 request fragmentation
- update queue wait, version-switch stall, p50/p99 latency
- wake inference energy와 sleep training energy

---

## 6. 축 ④ — 장기기억과 continual learning의 근본 문제

### 6.1 네 가지 서로 다른 “기억”

| 기억 매체 | 쓰기 방식 | 장점 | 근본 한계 | 대표 논문 |
|---|---|---|---|---|
| Token/KV | append | exact episodic recall | context와 함께 용량·read cost 증가 | full attention |
| Fast parametric state | online gradient/update | 고정 크기, semantic compression 가능 | interference, drift, per-session mutable state | TTT, Titans, HOPE, NSTM |
| Historical snapshots | checkpoint + selection/average | overwrite 이전 상태 보존 | 총 저장량 증가, selection/merge 문제 | MC, NSTM |
| Slow parametric expert | offline distillation | durable abstraction·shared reuse 가능 | distillation error, routing, growth/exhaustion | Sleep |

이 구분으로 보면 continual learning은 단일 algorithm이 아니라 다음 lifecycle이다.

`encode → qualify → stabilize → consolidate → validate → retain/merge/forget → retrieve`

현재 논문들은 encode(HOPE/TTT), stabilize(MC/NSTM), consolidate(Sleep)를 각각 강하게 다루지만 전체 closed loop를 구현하지 않았다.

### 6.2 직접 근거

- **HOPE:** learning=useful memory acquisition, 여러 frequency levels가 각 context를 압축·전달한다. 그러나 forgetting은 limited capacity의 자연 결과다. [P-HOPE:3537–3559](../../papers/2512.24695.txt#L3537-L3559)
- **Sleep:** online recall-dependent consolidation만으로는 abstraction과 capacity가 부족해 offline replay/distillation이 필요하다고 본다. [P-SLEEP:92–107](../../papers/2606.03979v2.txt#L92-L107)
- **NSTM:** context를 볼 수 있는 bypass가 있으면 parametric memory가 실제로 정보를 담지 않을 수 있으므로 memory-only loss가 필요하다. [P-NSTM:112–136](../../papers/2607.15271.txt#L112-L136)
- **NSTM:** 같은 snapshot average도 representation이 invariant/contextual factor를 분리했는지에 따라 성공과 붕괴가 갈린다. [P-NSTM:423–442](../../papers/2607.15271.txt#L423-L442)

### 6.3 미해결 원리 질문

- **I-Q4.1 · write-worthiness.** surprise가 큰 정보와 장기적으로 유용한 정보는 동일하지 않다. 어떤 신호가 durable-memory admission을 결정해야 하는가?
- **I-Q4.2 · consolidation target.** token, output distribution, gradient subspace, fast-weight function, low-rank adapter, concept graph 가운데 무엇을 slow memory로 옮겨야 abstraction을 보존하는가?
- **I-Q4.3 · teacher trust.** Knowledge Seeding의 pre-update full-model teacher가 오염·편향·최근성 drift를 포함할 때 stable-read ensemble, held-out replay, disagreement test 중 어떤 검증이 필요한가?
- **I-Q4.4 · memory-only certification.** external context/KV/retrieval을 차단해 parametric recall을 측정하는 probe를 LLM consolidation admission gate로 사용할 수 있는가?
- **I-Q4.5 · stability–plasticity control.** 기존 capability retention과 new knowledge acquisition을 단일 scalar loss가 아니라 per-memory-level constraint로 어떻게 관리할 것인가?
- **I-Q4.6 · provenance/rollback.** 어떤 wake episode가 어느 expert/state delta를 만들었는지 추적하여 오류 발견 시 선택적으로 되돌릴 수 있는가?

### 6.4 통합 가설: 3-stage parametric lifecycle

**I-Q / I-S 가설**

1. `W_active`: wake에서 빠르게 적응하되 drift 가능.
2. `W_stable-read`: 여러 checkpoint의 기능적 일치와 memory-only probe를 통과한 pre-consolidation state.
3. `W_slow-expert`: Sleep의 new low-rank expert에 distill된 durable shared state.

NSTM은 1→2의 특정 domain 사례, Sleep은 1 또는 2→3의 후보 operator다. 이 둘을 연결한 실험은 아직 없다.

---

## 7. 축 ⑤ — 유한 memory budget에서 단조 증가를 막는 방법

### 7.1 각 논문의 capacity 전략과 남은 구멍

| 방법 | capacity 대응 | bounded인가? | 남은 문제 |
|---|---|---:|---|
| HOPE/CMS | levels·memory depth·better update rule | 물리 parameter는 고정 | 저자가 forgetting 미해결을 명시; overwrite/level collision |
| Sleep | 새 slow low-rank expert 활성화, fast expert reset | 실험 구현은 preallocated finite pool일 수 있음 | slow slot exhaustion, merge/prune/offload 부재 |
| MC | snapshot 수 `N` 증가, SSC top-k read | active read는 bounded 가능; total storage는 아님 | admission/eviction/merge가 없음 |
| NSTM | active + running-average shadow state | state shape는 고정 | finite capacity를 직접 한계로 명시; primacy/missed events |

### 7.2 직접 근거

- **HOPE:** forgetting은 network의 limited capacity가 새 정보를 위해 과거를 압축·삭제하게 만드는 자연 결과다. [P-HOPE:3553–3559](../../papers/2512.24695.txt#L3553-L3559)
- **Sleep:** new low-rank expert로 interference를 피하고, fast-side old low-rank parameters는 consolidation 후 reset해 재사용한다. [P-SLEEP:362–369](../../papers/2606.03979v2.txt#L362-L369), [P-SLEEP:484–489](../../papers/2606.03979v2.txt#L484-L489)
- **MC:** total compute는 checkpoint 수에 따라 증가하고 segmentation이 compression–cost trade-off를 만든다. [P-MC:511–544](../../papers/2602.24281.txt#L511-L544)
- **NSTM:** finite capacity와 learned caching-weight management를 future work로 직접 제시한다. [P-NSTM:490–493](../../papers/2607.15271.txt#L490-L493)

### 7.3 필요한 비단조 memory operator

단순 append/grow가 아니라 다음 operator가 lifecycle에 포함돼야 한다.

1. **Admission:** novelty가 아니라 expected future utility / interference-adjusted utility가 높은 memory만 받는다.
2. **Deduplication:** functionally equivalent memories를 찾아 중복 제거한다.
3. **Merge:** parameter average가 아니라 held-out behavior를 보존하는 functional distillation 또는 subspace merge를 사용한다.
4. **Compression:** exact episode → prototype/semantic rule → low-rank expert 순으로 representation을 낮춘다.
5. **Eviction:** age가 아니라 utility, confidence, provenance, reconstructability, regulatory retention을 함께 본다.
6. **Offload:** HBM→capacity memory→host→SSD/external graph로 이동하되 recall SLA를 보장한다.
7. **Reactivation:** cold memory가 다시 중요해질 때 faster tier 또는 active expert로 승격한다.
8. **Garbage collection:** consolidation 후 fast state와 obsolete checkpoints를 회수한다.

### 7.4 open questions

- **I-Q5.1 · universal budget.** token KV, fast weights, snapshots, slow experts, external memory를 하나의 byte/energy budget 아래 어떻게 공동 최적화할 것인가?
- **I-Q5.2 · mergeability criterion.** parameter distance가 아니라 function-space agreement, Fisher geometry, gradient subspace, representation disentanglement 중 무엇이 safe merge를 가장 잘 예측하는가?
- **I-Q5.3 · eviction loss.** memory를 지웠을 때 미래 query distribution에서 생기는 regret를 online estimate할 수 있는가?
- **I-Q5.4 · expert exhaustion.** slow pool이 찼을 때 oldest, least-used, redundant expert 중 무엇을 merge/evict해야 기존 capability를 최소 손실로 보존하는가?
- **I-Q5.5 · capacity scaling law.** 필요한 durable capacity가 unique facts, task entropy, environmental non-stationarity, desired retention horizon에 대해 어떻게 증가하는가?
- **I-Q5.6 · reversible forgetting.** 삭제된 parametric memory를 external provenance/log에서 재구성할 수 있도록 “forget but reconstruct” contract를 만들 수 있는가?

---

## 8. 축 ⑥ — Sleep-time compute라는 새 scaling axis

### 8.1 논문이 직접 제시한 것

- continual learner를 wake와 sleep의 periodic lifecycle로 분리한다. [P-SLEEP:146–160](../../papers/2606.03979v2.txt#L146-L160)
- sleep의 NREM analogue는 knowledge seeding/upward distillation, REM analogue는 synthetic dream curriculum과 self-improvement다. [P-SLEEP:10–23](../../papers/2606.03979v2.txt#L10-L23), [P-SLEEP:499–554](../../papers/2606.03979v2.txt#L499-L554)
- consolidation은 fast block의 update 직전 frequency-aligned boundary에서 실행된다. [P-SLEEP:349–351](../../papers/2606.03979v2.txt#L349-L351)
- KS optimization 중에는 기존 student parameter를 freeze하고 새 expanded expert만 학습해 interference를 줄인다. 최적화 후에는 미리 계산한 sender base-weight update와 fast-expert reset·slow-expert activation을 적용한다. [P-SLEEP:397–407](../../papers/2606.03979v2.txt#L397-L407), [P-SLEEP:447–450](../../papers/2606.03979v2.txt#L447-L450)

### 8.2 sleep과 혼동하면 안 되는 것

| 방법 | deferred/offline? | durable cross-timescale transfer? | strict sleep인가? |
|---|---:|---:|---:|
| HOPE CMS wake update | 아니오 | 부분적 online transfer | 아니오 |
| MC checkpoint/cache | 아니오 | snapshot 보존, slow knowledge transfer 없음 | 아니오 |
| NSTM periodic TTT | online 저주파 | active→slow backbone transfer 없음 | 아니오 |
| Google Sleep consolidation | 예 | fast→new slow expert | 예 |
| Dreaming | 예 | capability update | sleep의 두 번째 stage |

### 8.3 scaling 변수

기존 `N`(parameter), `D_pre`(pre-training data), `C_train`, `C_test` 외에 다음 축을 별도로 둔다.

- `C_wake`: online/test-time update FLOPs
- `C_sleep-consolidate`: teacher generation + GKD/LTI + validation FLOPs
- `C_sleep-dream`: dream generation + selection + inner LoRA/SFT + RL FLOPs
- `B_move`: wake state를 sleep cluster와 memory tiers 사이에 옮기는 bytes
- `M_durable`: 활성 slow-expert bytes
- `T_ret`: 목표 retention horizon
- `U_sleep`: sleep accelerator utilization / idle-resource capture

연구할 response surface의 최소 형태는 다음과 같다.

`Quality_retained = F(N, D_pre, C_test, C_wake, C_sleep-consolidate, C_sleep-dream, M_durable, T_ret)`

여기서 중요한 것은 `C_sleep`가 무조건 quality를 높이는 단조축이 아니라는 점이다. **D-O:** Sleep paper 자체가 iterative self-distillation의 OOD/collapse 위험을 정리한다. [P-SLEEP:1448–1459](../../papers/2606.03979v2.txt#L1448-L1459) **I-H:** 따라서 verification과 capacity management가 없으면 추가 sleep compute의 한계수익이 음수가 될 수 있다.

### 8.4 open questions

- **I-Q6.1 · trigger law.** 고정 주기, episode boundary, surprise, conflict, memory-only failure, resource price 중 무엇이 sleep을 시작하는 최적 신호인가?
- **I-Q6.2 · budget allocation.** 동일 추가 FLOP를 long-context inference, wake update, consolidation, dreaming에 배분할 때 marginal quality gain은 어떻게 달라지는가?
- **I-Q6.3 · consolidation scaling.** teacher-state 수, replay diversity, student expert rank, validation size가 retention/transfer에 어떤 power law 또는 saturation law를 가지는가?
- **I-Q6.4 · negative scaling.** dream 수나 sleep cycle이 늘 때 model collapse, expert fragmentation, OOD degradation이 시작되는 임계점은 무엇인가?
- **I-Q6.5 · asynchronous consistency.** sleep 중에도 wake serving을 계속할 때 old/new expert version의 replica consistency와 rollback을 어떻게 보장하는가?
- **I-Q6.6 · placement.** wake cluster, sleep cluster, durable-memory tier를 물리적으로 분리할지, 같은 accelerator의 temporal partition으로 둘지? interconnect bytes가 compute 절감보다 커지는 경계는 어디인가?
- **I-Q6.7 · amortization.** 한 번의 sleep cost가 이후 몇 query/session에서 재사용돼야 pretraining refresh, RAG, external memory보다 경제적인가?

### 8.5 예상 infra 분해

**I-S 가설**

1. **Wake inference/update plane:** latency-critical inference, small frequent fast-state update, per-session ownership.
2. **Sleep training plane:** large-batch replay/distillation/RL, throughput-oriented, checkpoint rollback.
3. **Memory control plane:** provenance, state/expert versioning, tier placement, admission/eviction/merge, fast retrieval.
4. **Data plane:** raw episodes, stable snapshots, teacher logits, dreams, validation probes의 이동을 최소화하는 locality-aware pipeline.

핵심 co-design metric은 단순 FLOPs가 아니라 `quality gain / (compute energy + data-movement energy + wake-SLA opportunity cost)`다.

---

## 9. 여섯 축을 관통하는 research-question DAG

### RQ-A · 무엇이 장기기억 후보인가?

- 직접 단서: NSTM memory-only supervision, MC query relevance, HOPE surprise/update.
- 미해결: 반복되는 것, 놀라운 것, 미래 utility가 높은 것, 기존 지식과 conflict하는 것을 어떻게 구분할지.
- 다음 질문: **RQ-B**로 연결.

### RQ-B · 후보가 merge/consolidate 가능한 표현인가?

- 직접 단서: NSTM에서 coupled pose-containing weights의 평균은 붕괴, disentangled memory에서는 성공.
- 미해결: LLM에서 invariant semantic component와 context-dependent alignment를 구조적으로 분해하는 방법.
- 다음 질문: 가능하면 **RQ-C**, 불가능하면 external/exact memory 유지.

### RQ-C · 언제 어느 level로 옮길 것인가?

- 직접 단서: CMS frequency levels, Sleep frequency-aligned boundary, NSTM update-frequency trade-off.
- 미해결: fixed cadence가 아닌 learned trigger와 target-level selection.
- 다음 질문: **RQ-D**.

### RQ-D · 유한 용량에서 무엇을 회수할 것인가?

- 직접 단서: HOPE의 forgetting=limited compression, Sleep fast reset, NSTM finite capacity, MC total checkpoint growth.
- 미해결: cross-media utility와 safe merge/eviction.
- 다음 질문: **RQ-E**.

### RQ-E · hardware에서 감당 가능한가?

- 직접 단서: NSTM 58.14/27.01 ms asymmetry, TTT memory I/O, MC O(NL), Sleep repeated consolidation bottleneck.
- 미해결: phase별 FLOPs/bytes/latency/energy와 multi-tenant reuse.
- 결과: trigger, chunk, rank, top-k, placement, sleep budget을 다시 RQ-A~D에 feedback.

이 DAG의 독창적 핵심은 **기억 정책을 accuracy-only 문제가 아니라 representation qualification × lifecycle × systems cost의 closed loop로 보는 것**이다.

---

## 10. 논문 근거로 답할 수 있는 것과 아직 답할 수 없는 것

| 질문 | 현재 근거로 답변 가능? | 가장 강한 근거 | 남은 검증 |
|---|---:|---|---|
| 왜 fixed-state RNN이 long recall에서 약한가? | 높음 | TTT, MC의 fixed-capacity diagnosis | task별 정보이론적 capacity |
| checkpoint를 늘리면 성능이 오르는가? | 중간 | MC experiments | total serving memory budget 공정 비교 |
| 모든 fast-weight snapshot을 평균해도 되는가? | “아니오”는 높음 | NSTM failure/success ablation | 일반 mergeability theorem |
| update보다 read를 더 자주 해도 되는가? | domain 제한 중간 | NSTM 29:1 inference | language model frequency shift |
| chunk dual form이 prefill을 병렬화하는가? | 높음 | TTT/HOPE | HOPE production kernel |
| chunk dual form이 decode도 병렬화하는가? | 아니오 | TTT는 decode primal | speculative/multi-token update 가능성 |
| Sleep이 short→long parametric transfer를 하는가? | proof-of-concept 수준 예 | Sleep experiments/mechanism | independent replication, frontier scale |
| Sleep이 무한 lifetime capacity를 해결하는가? | 아니오 | finite/preallocated expert design | exhaustion/merge/prune policy |
| MC가 prefix cache reuse를 해결하는가? | 아니오 | paper scope가 intra-sequence cache | state identity/COW serving design |
| NSTM scheduling을 LLM serving에 바로 쓸 수 있는가? | 아니오 | CV live-stream evidence only | per-session mutable weight batching |
| sleep-time scaling law가 존재하는가? | 미답 | mechanism과 변수는 있음 | controlled budget sweep |

---

## 11. Google 미팅 전에 반드시 유지할 정합성 가드레일

1. **HOPE의 self-modifying forward와 update-side loss를 섞지 않는다.** direct retrieval은 learned query를 `M_memory`에 적용하는 경로이고, auxiliary memories/inner loss는 update를 위한 계산이다.
2. **query-side mutable memory를 별도 구현 구성요소로 두지 않는다.** 실제 architecture 설명처럼 `q=xW_q`인 fixed projection을 사용하며, 원문의 상충하는 update-set 표기는 오류 가능성이 있는 것으로 취급한다.
3. **CMS와 Sleep의 sparse experts를 구분한다.** original HOPE CMS는 multi-frequency MLP chain이고, sleep paper가 consolidation을 위해 sparse low-rank expert activation을 도입한다.
4. **MC top-k는 active read를 제한할 뿐 total stored checkpoints를 자동으로 제한하지 않는다.**
5. **NSTM의 running average는 slow-weight consolidation이 아니라 stable read shadow state다.**
6. **NSTM의 invariant/contextual 분해 원리는 paper의 NVS 결과에서 강하게 지지되지만 LLM 일반화는 우리의 가설이다.**
7. **latency 수치는 phase와 hardware를 붙여 말한다.** NSTM 58.14/27.01 ms는 256×256, single H100의 memorization(update+apply)/synthesis(apply-only)다.
8. **long-context benchmark result를 lifetime continual learning 증거로 확대하지 않는다.** 10M token success와 cross-session durable memory는 다른 주장이다.
9. **Sleep의 parameter growth를 무한 physical growth로 표현하지 않는다.** 원문은 preallocate-and-mask implementation도 제시한다.
10. **모든 systems/device bridge는 “paper result”가 아니라 “joint hypothesis”로 말한다.**

---

## 12. 핵심 gap — 질문 playbook이 겨냥해야 할 여섯 문장

1. **Attention은 보존을 잘하지만 계산·비용이 자라고, recurrent memory는 싸지만 무엇을 잃었는지 모른다.** 동일 budget에서 retention/retrieval/computation을 분해하는 benchmark와 scaling law가 없다.
2. **Mutable recurrent state는 KV cache와 다른 serving object다.** prefix shareability, state identity, copy-on-write, versioned batching의 contract가 없다.
3. **Prefill의 chunk 병렬화는 진전됐지만 decode의 per-session state RMW와 memory I/O는 남아 있다.** end-to-end production kernel/roofline evidence가 부족하다.
4. **Wake memory를 얻는 법과 Sleep에서 옮기는 법은 각각 제시됐지만, pre-update full-model teacher를 언제 신뢰할지가 비어 있다.** memory-only certification과 stable-read state가 연결 후보다.
5. **HOPE·Sleep·MC·NSTM 모두 결국 finite capacity에 부딪힌다.** append/growth가 아닌 admission–merge–evict–offload–reactivate의 비단조 memory controller가 없다.
6. **Sleep은 새 compute phase를 제안했지만 아직 scaling law가 아니다.** quality gain을 sleep FLOPs, moved bytes, durable capacity, retention horizon, wake-SLA opportunity cost에 연결하는 law와 infra가 필요하다.

이 여섯 gap이 이후 [`Top-6 kick questions`와 `Google 예상 답변`](02-GOOGLE-KICK-QUESTION-PLAYBOOK.md), [`Samsung device-solution bridge`](03-DEVICE-SOLUTION-BRIDGE.md)의 근거 골격이다.

---

## 13. Primary-source index

| ID | 논문 | 원문 URL | 저장소 근거 |
|---|---|---|---|
| P-HOPE | *Nested Learning: The Illusion of Deep Learning Architectures* / HOPE | <https://arxiv.org/abs/2512.24695> | [`papers/2512.24695.txt`](../../papers/2512.24695.txt) |
| P-SLEEP | *Language Models Need Sleep: Learning to Self-Modify and Consolidate Memories* (arXiv v2, 2026-07-10) | <https://arxiv.org/abs/2606.03979v2> | [`papers/2606.03979v2.txt`](../../papers/2606.03979v2.txt) (v2 line evidence) |
| P-MC | *Memory Caching: RNNs with Growing Memory* | <https://arxiv.org/abs/2602.24281> | [`papers/2602.24281.txt`](../../papers/2602.24281.txt) |
| P-NSTM | *Online Neural Space Time Memory for Dynamic Novel View Synthesis* | <https://arxiv.org/abs/2607.15271> | [`papers/2607.15271.txt`](../../papers/2607.15271.txt) |
| P-TTT | *Learning to (Learn at Test Time): RNNs with Expressive Hidden States* | <https://arxiv.org/abs/2407.04620> | [`papers/external/2407.04620.txt`](../../papers/external/2407.04620.txt) |
