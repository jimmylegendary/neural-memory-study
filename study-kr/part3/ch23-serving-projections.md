# ch23. 대규모 학습·서빙 projection

> **이 장의 목표** — 독자가 이 장을 마치면 (1) [TNT]가 남긴 유일한 wall-clock 증거(학습 처리율)를 7–70B로 외삽할 때 무엇이 근거이고 무엇이 directional한지 구분할 수 있고, (2) 하나의 서빙 요청 안에서 prefill과 decode가 pair thesis의 반대편 두 영역(compute-bound accelerator / memory-bound memory-centric)을 각각 부른다는 것을 설명할 수 있으며, (3) per-session weight state를 sizing·checkpoint/restore·multi-tenant batching·frequency-tier 배치·sleep-as-background-job의 다섯 축을 가진 **새로운 cache class**로 설계하고, 이를 오늘의 KV-cache 서빙과 비용 모델로 나란히 놓을 수 있어야 한다.
> **왜 필요한가** — 18장이 세운 D4 pair thesis는 배포 형태의 명제다. 그 배포를 실제로 돌리면 얼마가 드는가는 여섯 편 어디에도 없다 — decode wall-clock 수치는 6편 전체에 부재하고([TNT]는 학습 wall-clock만), serving 경제학은 다섯 공백 중 하나로 명시적으로 남았다(→ 18장 §18.2). 이 장은 그 빈 자리를 로컬 실측 프로그램(claim 1–6)의 비율·crossover·tier 순서로 채우고, 채울 수 없는 절대치는 사내 A100 runbook으로 이월한다.

## 23.1 Bridge-in: 완성형을 배포하면 청구서가 두 장 나온다

22장은 이 라인의 자연스러운 저자가 왜 Google인지, 각 player가 다음에 무엇을 합리적으로 두는지를 분석했다. 이 장은 그 다음 질문을 받는다 — **그 배포를 실제로 서빙하면 얼마가 드는가.** 18장에서 완성형은 "세션 상태가 per-session/per-tenant mutable weights인 continually-learning LLM에, TNT 경제학으로 학습되고, sleep 서비스가 붙어 있는 시스템"으로 정식화됐다(§18.2의 [평가]). 이 형태를 청구서로 바꾸면 두 장이 나온다: **학습·prefill 청구서**(accelerator 영역)와 **decode·serving-state 청구서**(memory-centric 영역). 두 청구서는 같은 알고리즘에서 나오지만 지불 통화가 다르다 — 하나는 FLOP, 하나는 byte.

이 장의 정직성 바닥을 먼저 못 박는다. 이 라인의 실증 상한은 from-scratch 1.3B params / 100B tokens이고([TNT]는 150M), decode wall-clock은 6편 전체에 부재하다(→ 18장 §18.2 유보 1, §6.4 라인 공통 caveat). 따라서 이 장의 **load-bearing 주장은 비율·crossover 위치·tier 순서·bound class뿐**이다. 절대 µs/token·mJ/token·B_max 절대값은 roofline 하한이거나 upper bound이며, 사내 A100 runbook(`runbook/hope-reproduction-a100.md`)이 실 kernel로 닫을 이월 항목이다. novel scratchpad/PIM twin의 수치는 `simulation_ready=False`인 directional DSE로만 등장한다. 이 태그를 절마다 그대로 붙인다.

## 23.2 TNT economics를 7–70B로 외삽 (accelerator 절반)

[TNT]는 라인 전체에서 wall-clock을 보고한 유일한 편이다(→ 15장). 세 수치가 앵커다: 정확한 Titans baseline 대비 **target loss까지 최대 17.4× 빠르고**[TNT], plain-JAX 구현이 **32K 문맥에서 step당 FlashAttention을 이기며**[TNT], 550M Titans를 $C{=}64$로 학습하면 ppl 13.78인데 $C{=}8$로 서빙하면 36.45로 무너진다[TNT Fig. 2]. 앞의 둘은 accelerator 영역이 실재함을 보인다 — nonlinear deep-memory recurrence를 reset으로 끊어 context parallelism을 열고, chunk를 키워 arithmetic intensity를 제조하면 이 부하는 tensor core로 간다.

이 처리율을 7–70B로 외삽할 때 무엇이 자라는가는 chunk 크기 $C$가 결정한다. claim7(E2.1, → 9장·15장)은 $C$가 roofline의 x축임을 보인다: $C{=}1$(per-token rank-1 RMW = decode 영역)은 AI≈1 FLOP/byte로 memory-bound 평원(host roofline의 약 12%)에 있고, $C$를 키우면 crossover $C^\star$를 넘어 compute-bound로 오른다. prefill/training은 큰 $C$에서 돌므로 accelerator 영역에 산다. 다만 $C^\star$의 **절대값은 ridge 의존**이다 — host CPU에서 측정된 $C^\star\approx32$는 곡선의 **모양**만 옮기고, H100 twin ridge(295 FLOP/byte)에서 닫힌 형식 $C^\star$는 $d$에 따라 306–430이다[E3]. 즉 "$C$를 키우면 compute-bound가 된다"는 load-bearing이고, "얼마나 커야 하나"는 하드웨어별로 다시 재야 한다. [CPU-SHAPE]

> **[평가]** TNT의 17.4×를 7–70B로 곧이곧대로 옮겨서는 안 된다. [TNT App. D]는 momentum·gating·Muon을 "명료성을 위해" 제거한 단순화 Titans로 경제학을 검증했다 — 라인 전체(self-modifying Titans + CMS의 HOPE)와의 합성은 미검증이다(→ 15장의 의무 caveat). HOPE의 실제 학습 비용은 outer gradient가 inner-loop의 unrolled per-token/per-chunk update를 통째로 관통하는 데서 나오며(runbook §0), 이 그래프 폭발은 TNT의 단순화 Titans에는 없다. 따라서 이 절의 외삽은 **accelerator 영역이 존재하고 chunk가 그 x축이라는 구조적 주장**까지만 단정하고, 학습 벽시계의 절대 예측은 A100 runbook의 carry-forward #1(760M/30B 학습 벽시계는 논문 부재로 사전 예측 불가, 실측 대기)로 넘긴다.

decode 쪽 학습 후 비용의 스케일링은 claim1(E1.1/E3)이 준다. 전체 state 크기는 폭에 따라 $m\,d^2\,L_{\mathrm{layer}}$로 자라고, 그에 비례해 per-token RMW 트래픽이 커진다:

표 23-1 — 모델 폭에 따른 decode-step RMW 트래픽 스케일링(anchor $m{=}16$; 문맥 $S$에 **무관**, KV와의 결정적 대비). 절대 GB/token은 canonical transformer shape 가정 위의 roofline 값이다. [REL, M-ASSUMED]

| 규모 | RMW GB/token | S\* read-crossover (token) | B_max @10ms (seq) |
|---|---|---|---|
| Titans-170M | 0.453 | — | — |
| Titans-340M | 1.611 | 16,384 | 20.8 |
| Titans-760M | 3.624 | 36,864 | 9.24 |
| neural-mem-1.3B | 6.442 | 65,536 | 5.2 |
| hypo-7B | 34.36 | 262,144 | 0.97 |
| hypo-70B | 343.6 | 1,048,576 | 0.1 |

한 줄로 읽으면: **decode 트래픽은 폭에 따라 0.45→6.44→343 GB/token으로 자라고**[E3], 이것이 서빙 청구서의 지배항이다. 70B에서 token당 343 GB를 움직인다는 것은, HBM3(3.35 TB/s)에서 state 이동만으로 token당 하한이 100 ms 규모라는 뜻이다 — 절대치는 A100 실측 대기지만, 폭에 대한 세제곱 근처 성장(∝ $d^2$)은 load-bearing이다.

## 23.3 Hierarchical memory의 prefill/decode 분리

오늘의 KV-cache 서빙에서 prefill과 decode는 **같은 연산**(attention)의 두 arithmetic intensity 점이다 — prefill은 긴 문맥의 GEMM(compute-bound), decode는 KV의 read-many(점점 memory-bound). 이 라인에서는 그 구조가 질적으로 달라진다.

> **[해설]** hierarchical memory(→ 15장의 global $W^{\mathrm{g}}$ + local $W^{\mathrm{l}(i)}$)를 서빙에 얹으면, **하나의 요청 안에서 prefill과 decode가 pair thesis의 반대편 두 영역을 각각 부른다.**
> - **prefill = 큰 chunk의 병렬 write(compression).** 입력 문맥을 큰 $C_{\mathrm{g}}$로 쪼개 global memory에 한 번에 압축한다 — claim7의 큰-$C$ 영역, compute-bound, tensor core. 이것은 독자의 prefill 감각(큰 GEMM)과 정확히 유비된다(→ 1장 Rosetta의 "prefill = 큰 chunk의 병렬 write" 행).
> - **decode = $C{=}1$의 per-token online RMW + read.** claim1의 memory-bound 영역, state 트래픽이 GEMV 연산을 약 394× 압도한다[E1.1] — step 비용이 곧 state 트래픽이다.

이 분리가 서빙 스케줄러에 주는 함의는 직접적이다. 오늘의 엔진은 prefill과 decode를 같은 kernel family로 배치(batch)하고 continuous batching으로 섞는다. 이 라인에서는 **두 phase가 다른 병목 자원을 문다** — prefill은 FLOP, decode는 HBM 대역폭. 따라서 prefill-decode disaggregation(두 phase를 다른 하드웨어 풀로 분리)이 KV-cache 서빙에서보다 **더 강하게** 정당화된다: prefill 풀은 compute-dense accelerator를, decode 풀은 high-RMW-bandwidth memory 시스템을 원한다. 이것이 이 장이 pair thesis를 서빙 토폴로지로 옮기는 첫 결론이다.

## 23.4 Per-session weight state: 새로운 cache class의 sizing과 batching

per-request로 변하는 weights는 오늘의 서빙 불변식 — "추론 중 weights는 shared·불변" — 을 깬다(→ 18장 §18.1). 그 결과 필요한 것이 **per-session weight state라는 새 cache class**다. KV cache와 나란히 놓되 성질이 반대인 이 class를 다섯 축으로 설계한다. 먼저 sizing과 batching.

**Sizing.** state 크기는 문맥 $S$에 **무관하게** $m\,d^2\,L_{\mathrm{layer}}$로 고정된다(anchor에서 layer당 134 MB, [E1.1/E3]). 이것은 KV cache와의 근본 대비다: KV는 문맥에 선형으로 자라 capacity-bound가 되지만, session-weight는 고정 크기라 **on-chip 상주 여부가 설계 가능한 knob**이 된다(claim4, → 20장·24장). 단 그 knob에는 상한이 있다 — on-die L2(50 MB)는 **어느 규모에서도 한 sequence의 whole-model state조차 담지 못한다**[claim3]. 따라서 상주는 whole-model pin이 아니라 **per-layer/streamed**여야 하고, 268 MB scratchpad에 whole-model이 들어가는 것은 Titans-170M(226 MB)뿐이다[E1.2]. 340M 이상은 전부 spill한다.

**Batching.** 여기서 KV와 가장 극적으로 갈라진다. 집계 state RMW는 **HBM 용량보다 대역폭이 먼저** 막힌다 — claim3(E1.3/E3)의 핵심이다. 10 ms/token 목표에서 대역폭 상한 배치 $B_{\max}$는 340M/1.3B/7B에서 **20.8 / 5.2 / 0.97 sequence**(표 23-1), 50 ms 목표에서 104 / 26 / 4.87이다[E3]. KV cache가 capacity-bound인 것과 **정반대로 bandwidth wall**이다.

> **[평가]** 이 $B_{\max}$ 절대값은 두 겹의 유보를 진다. 첫째, roofline 하한 latency에서 나온 값이라 절대치는 A100 실측 대기다[NO-WALLCLOCK]. 둘째, E3의 $B_{\max}$는 weights+activations+KV가 co-resident인 full serving stack을 빼고 계산한 **upper bound**다(carry-forward #6). 그러므로 load-bearing은 "7B에서 10 ms 목표 시 배치가 한 자리수, 사실상 batch-1로 몰린다"는 **class와 방향**이지 "0.97"이라는 숫자가 아니다. 함의는 무겁다 — per-request unshared weights는 shared-weight batching을 깨고, decode는 grouped-GEMM으로 가야 하며(→ 24장), 대역폭 벽이 배치를 조기에 닫으므로 memory-centric 논증의 decode 절반이 바로 여기서 실측 근거를 얻는다.

**Checkpoint/restore.** session-weight는 KV cache가 지지 않는 두 종류의 상태 관리 부채를 진다. (1) **snapshot/rollback** — speculative decoding의 rollback은 KV에서는 pointer 되감기지만, 여기서는 fast weights $W_t$뿐 아니라 **optimizer-trajectory 상태(momentum $S_t$)까지** snapshot해야 한다(→ 18장의 systems 공백; runbook X-계열). (2) **dirty eviction의 writeback** — session-weight는 항상 dirty(매 token 갱신)라 evict할 때 recompute가 아니라 **full writeback**을 진다. 이 부채가 오늘의 KV manager에서 0원으로 잘못 계상되는 것이 다음 절이다.

## 23.5 KV-manager의 범주 오류: 왜 새 매니저가 필요한가

pair thesis의 소프트웨어 절반이다. `hat-kv-manager`(오늘의 KV-cache 매니저 축소판)에 session-weight를 얹어 보면 세 지점이 **범주 오류**로 드러난다(claim6, E1.5).

- **G1 — reuse = 0.** content-addressed prefix 재사용은 KV 매니저의 1번 가치(anchor에서 hit-rate 0.5)지만, RMW state에는 **구조적으로 0**이다 — 내용이 매 token 바뀌어 재현되는 block hash가 없다.
- **G2 — stale accretion 256×.** in-place update가 없는 append-only `place()`는 256 token에 걸쳐 $T{=}256$개의 stale 버전을 쌓아 참 1-state working set의 **256×로 부풀린다**.
- **G3 — 값 매겨지지 않은 dirty writeback.** `_evict_from()`은 dirty state를 공짜로 `del`한다(clean/recomputable 가정). 실제 TTT eviction은 full writeback을 지며, 그 값은 anchor에서 약 **3.05 mJ / 130 µs**인데 매니저는 **0으로 계상**한다[NO-WALLCLOCK for 절대치; 범주 오류 자체는 load-bearing].

결론: session-weight 매니저는 KVCacheManager에 없는 event type — `update_in_place`, `mark_dirty/writeback`, `checkpoint/rollback`, `bind_to_sequence`, `free_on_update` — 을 1급으로 가져야 한다. KV 매니저를 확장하는 것이 아니라 **다른 class의 매니저**다. 이것이 §23.4의 checkpoint/restore 부채를 소프트웨어로 갚는 자리이며, `hat-kv-manager`의 로드맵 메모로 이월된다.

## 23.6 Frequency-tiered placement: 서빙 메모리 계층 그 자체

새 cache class의 네 번째 축이자 이 장에서 가장 신선한 memory-architecture 주장이다. [NL]/[Sleep]이 각 memory level에 부여한 **update cadence**(→ 16장·17장)는 그대로 **memory 계층의 tier 배치 명세**로 번역된다(claim5, E1.4). pair thesis가 memory-centric 논증을 펴는 두 정당한 지점 중 두 번째다(첫째는 §23.4의 decode RMW 트래픽).

<!-- FIG: exp-d -->

![그림 23-1 — update cadence에서 memory-tier admissibility로의 번역: 상주 사본은 read/write 중 빠른 쪽 cadence로 tier가 정해진다](../../figures/exp-d-frequency-tiers.png)

그림 23-1 — 각 memory level의 update cadence가 admissible한 memory tier를 정한다. 상주 사본은 read/write 중 **빠른 쪽** cadence에 의해 pin되며(gating rule), sub-per-token access는 트래픽을 임계 경로 아래로 amortise해 write-heavy RMW state를 legally CXL로 내려보낸다. anchor(neural-mem-1.3B, per-token 임계 경로 예산 $t_{\mathrm{tok}}\approx962\ \mu\mathrm{s}$)에서의 tier 배정은 표 23-2와 같다(실험 E1.4 재구성). [REL, NOVEL-SIM-FALSE for scratchpad]

**gating rule**: 상주 사본은 read cadence와 write cadence 중 **빠른 쪽**으로 tier가 정해진다. 이 한 규칙이 latency tolerance를 update period에 선형($P_{\mathrm{read}}\cdot t_{\mathrm{tok}}$)으로 넓혀 준다 — 자주 접근하는 것은 뜨거운 tier에 pin되고, 드물게 접근하는 것은 느린 tier가 허용된다.

표 23-2 — update cadence → memory-tier 배정(E1.4, anchor). "read cadence가 pin한다"가 핵심 — chunk마다 쓰더라도 매 token 읽으면 HBM에 남는다. [REL, NOVEL-SIM-FALSE, M-ASSUMED]

| memory level | cadence (read / write) | tier | 근거 |
|---|---|---|---|
| fast-weights | token / token | HBM3 | read-gated, 가장 뜨거운 admissible tier |
| CMS-chunk | token / 64 tokens | HBM3 | read cadence가 pin(chunked write에도 불구) |
| CMS-mid | 4096 / 4096 tokens | CXL | down-sampled cadence → 여전히 admissible한 최저 tier |
| sleep-experts | 262144 / 262144 tokens | CXL | offline/idle-window cadence |
| frozen-weights | token / never | HBM3 | read-gated, read-heavy(accelerator 친화) 쪽 |

> **[해설]** 이 표는 오늘의 서빙 엔지니어에게 익숙한 것의 재서술이다 — paged KV cache가 hot page를 HBM에, cold page를 host/CXL로 내리는 것과 같은 문법이다. 다른 점은 **무엇이 hot인지를 문맥 길이가 아니라 update frequency가 정한다**는 것이다. per-token fast-weight는 HBM에 pin되고, 진짜로 down-sample된 CMS-mid level(4096 token마다 read+write)은 CXL-class latency를 감내하며, sleep으로 consolidate된 expert는 offline cadence라 CXL로 내려간다. 이 매핑은 거의 문자 그대로이고 선행 연구가 진술한 바 없다(→ 18장 §18.3의 두 정당한 memory-centric 지점 중 둘째).

per-layer로 쪼갠 fast-weight block(67 MB)은 268 MB scratchpad에 fit해 HBM3 대비 **7.4× 에너지 이득**을 낸다[E1.4] — 단 all-layer는 fit하지 않는다. **이 scratchpad 수치는 novel twin이라 directional DSE로만 인용한다**[NOVEL-SIM-FALSE]: 배치의 방향(per-layer streamed면 on-chip 이득이 실재)만 load-bearing이고, 7.4×는 silicon 전까지 device 주장이 아니다(→ 20장·24장의 상세 DSE, 그림 (b)/(e)는 → 20장).

## 23.7 Sleep을 fleet-level background job으로

새 cache class의 다섯 번째 축이자 서빙의 세 번째 regime이다. [Sleep](→ 17장)의 wake/sleep lifecycle을 배포에 얹으면, decode 요청(wake) 밖에 **주기적 offline job**(sleep)이 fleet에 붙는다.

> **[해설]** sleep phase = **weights의 background compaction job**이다(→ 1장 Rosetta의 "서빙 fleet의 background job" 행). CMS의 update 경계에서 발화해([Sleep]은 sleep을 chunk 경계에 hard-wire), (i) 빠른 block의 지식을 느린 block으로 상향 distill(Knowledge Seeding)하고, (ii) self-generated data로 Dreaming을 돌린다. 서빙 관점에서 이것은 GPU를 점유하는, 중단 가능한, 주기적 fine-tuning job이다 — 배포가 "훈련이 끝난" 상태로 존재하지 않는다는 §18.2 완성형의 직접 귀결이다.

fleet 관점의 비용은 세 가지다. (1) **cadence → tier**: sleep-consolidated expert는 read+write가 262144 token마다(offline)라 표 23-2대로 CXL로 내려가고, 임계 경로 밖 데이터 migration으로 스케줄된다. (2) **versioning**: 매 sleep이 새 weight 버전을 만든다 — per-tenant divergence·rollback·provenance가 서빙 스택의 1급 관심사가 된다(→ 18장 systems 공백). (3) **dream fan-out**: [Sleep]의 Dreaming은 격리된 instance에서 randomized routing으로 5개 롤아웃을 낸다 — fleet 용량 계획에 background 훈련 fan-out을 넣어야 한다.

> **[평가]** 이 절의 모든 서술은 [Sleep]의 기제가 pre-trained Llama/Qwen 위의 graft라는 의무 caveat 위에 선다(→ 17장) — 라인 최초로 end-to-end meta-learn되지 않은 편이다. 따라서 sleep-as-fleet-job은 **배포 아키텍처의 설계 주장**이지 성능 약속이 아니다. 공개 구현도 0이라(runbook §6.5), 이 job의 실 비용은 HOPE 재현이 성공한 뒤에야 wake-sleep cycle을 얹어 측정할 수 있다(runbook C6/C7, 범위 밖으로 이월).

## 23.8 오늘의 KV-cache 서빙 대비 비용 모델

다섯 축을 한 표로 접어 오늘의 KV-cache 서빙과 나란히 놓는다. 이 표가 이 장의 결산이다.

표 23-3 — KV-cache 서빙 vs per-session-weight 서빙의 비용 모델 대비. load-bearing 열은 bound class·shareable·문맥 의존성·crossover이고, 절대치는 §23.9로 이월. [REL]

| 축 | KV-cache 서빙(오늘) | per-session-weight 서빙(이 라인) |
|---|---|---|
| 상태 성격 | append-once / read-many | per-token RMW(read-modify-write) |
| write / read 비 | 0.003%[E1.3] | 100%(대칭)[E1.3] |
| 공유 가능? | yes(prefix 재사용, hit 0.5) | **no**(per-seq, 항상 dirty, reuse 0) |
| 문맥 $S$ 의존 | read ∝ $S$로 증가 | **일정**($m\,d^2\,L_{\mathrm{layer}}$로 고정) |
| 배치 병목 | capacity-bound | **bandwidth-bound**($B_{\max}$ 벽)[claim3] |
| prefill | attention GEMM(compute) | 큰-$C$ chunk compression(compute) |
| decode | KV read-many(점점 mem-bound) | whole-state RMW(항상 mem-bound, 394×)[E1.1] |
| eviction 비용 | 공짜 drop(recompute) | **full writeback 부채**(≈3.05 mJ/130 µs 계상 필요)[E1.5] |
| tier 배치 | 문맥 길이(hot/cold page) | **update frequency**(cadence → tier)[E1.4] |

비용 모델의 한 문장 요약은 **crossover 문맥 길이 $S^\star$**다(claim2, → 18장 그림 18-1). KV read는 문맥에 비례해 자라고 TTT RMW는 일정하므로, $S^\star$ 아래에서는 KV가 싼 메모리 시스템이고 위에서는 TTT가 싸다. anchor에서 read-crossover $S^\star\approx65\mathrm{k}$ token, 전체 RMW-crossover $\approx131\mathrm{k}$ token이며[E1.3/E3], $S^\star$는 폭에 따라 $m\,d^2$로 **16k → 1.05M token**을 이동한다(표 23-1). fp8 KV는 $S^\star$를 두 배로 밀고, MHA(GQA 대신)는 선형으로 collapse시킨다[E3].

> **[해설]** 이 crossover가 서빙 배치 결정 규칙을 준다. 짧은-문맥·다중-테넌트(챗봇류, 문맥 ≪ 65k)에서는 오늘의 KV cache가 여전히 싼 메모리 시스템이다 — dense-attention KV의 per-token latency가 문맥에 따라 오르지만(2k에서 0.785, 32k에서 2.581 ms[E1.3]) 65k 아래에서는 고정 RMW보다 싸다. 긴-문맥·소수-세션(문서·에이전트류, 문맥 ≫ 131k)에서는 TTT state가 싸다 — latency가 문맥에 **무관하게** 일정하기 때문이다. 완성형이 hybrid일 가능성(→ 18장 유보 2)이 여기서 경제적으로도 지지된다: 두 메모리 시스템은 문맥 축에서 서로의 영역을 나눠 가진다.

## 23.9 정직성 결산과 A100 runbook 이월

이 장이 단정한 것과 유보한 것을 명시적으로 가른다. **단정(load-bearing)**: (a) 하나의 요청 안에서 prefill=compute-bound / decode=memory-bound로 갈라진다(§23.3), (b) 배치는 capacity가 아니라 대역폭이 먼저 막힌다(class·방향, §23.4), (c) KV-manager는 RMW state에 범주 오류다(§23.5), (d) update cadence가 memory-tier를 정한다는 gating rule과 tier 순서(§23.6), (e) crossover $S^\star$가 KV/TTT 영역을 문맥 축에서 나눈다(§23.8). 이들은 세 방법(E3 닫힌 형식, hatir E1.x, plan hand-calc)이 anchor에서 1% 이내로 합치한 비율·crossover·순서다.

**유보(사내 A100 runbook으로 이월)**: 아래 여섯 항목은 `experiments/results.json`의 `carry_forward_to_A100_runbook`이며, 이 장의 절대치는 전부 여기에 속한다.

1. **절대 per-token decode wall-clock**(ms/token, tok/s) — 논문·로컬 모두 하한만. anchor의 1.92 ms/token·46.5 mJ/token은 HBM3 하한이며 A100 실측이 대체한다.
2. **achieved-vs-roofline 효율**(kernel-launch·scheduling·decode tail) — E1.1은 per-kernel only, per-token latency는 하한.
3. **실 energy/token**(r/w-asymmetric DRAM) — twin은 symmetric 7 pJ/B 가정.
4. **$S^\star$ 검증** — 실 KV read kernel vs 실 TTT-state kernel head-to-head(로컬은 roofline half). 앵커 $S^\star_{\mathrm{read}}\approx65\mathrm{k}$, $S^\star_{\mathrm{rmw}}\approx131\mathrm{k}$는 directional.
5. **novel-twin(scratchpad/PIM)** — §23.6의 7.4×, §23.4의 residency 이득은 silicon 전까지 directional DSE. A100 실측은 HBM 실수치만 닫는다.
6. **$B_{\max}$ under full serving stack**(weights+activations+KV co-resident) — §23.4의 $B_{\max}$는 upper bound.

> **[평가]** 이 장은 pair thesis의 **양쪽 청구서를 서빙 토폴로지로 옮겼다**. decode·serving-state 절반은 새 cache class(sizing·checkpoint·batching·frequency-tier·sleep-job)로 구체화되어 memory-centric 논증의 실측 근거를 얻었고(claim 1–6), training·prefill 절반은 chunk가 x축인 accelerator 영역으로 남아 fused kernel·grouped-GEMM의 본진이 됐다(→ 24장). 어느 한쪽만 미는 것은 부하의 절반을 무시하는 것이다(→ 18장 §18.3). 이 장이 낸 절대치는 하나도 device 주장이 아니다 — 전부 A100 runbook이 실 kernel로 닫을 하한·upper bound이며, 그 이월이 정직성의 조건이다.

**다음 장으로.** 이 장은 배포를 두 청구서로 갈랐다. 24장은 그 두 청구서에 각각 제안을 붙인다 — 알고리즘 수준(§5의 소규모 실험)과 하드웨어 수준(frequency-tiered placement·RMW-bandwidth decode device의 memory-centric 반쪽 + fused chunk kernel·grouped-GEMM decode engine의 accelerator 반쪽). pair thesis의 흥미로운 하드웨어 제안은 둘 중 하나가 아니라 **쌍**이라는 §18.3의 명제를, 24장이 구체적 device 제안으로 닫는다.
