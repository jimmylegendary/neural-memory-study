# ch18. 여섯 편의 수렴과 Part III의 기여 프로그램

> **이 장의 목표** — 독자가 이 장을 마치면 (1) 여섯 편이 합성해 낸 "완성형"을 텍스트 근거와 두 개의 정직한 유보와 함께 재구성할 수 있고, (2) 그 완성형을 엔지니어의 질문으로 옮긴 이 책의 척추 명제 — **D4 workload-split pair thesis** — 를 정식으로 진술할 수 있으며, (3) Part III가 그 명제를 검증하려고 돌린 8개 실측 실험이 어느 장에서 무엇을 뒷받침하는지 지도로 읽을 수 있어야 한다.
> **왜 필요한가** — Part II는 여섯 논문을 각각 해부했고, 17장은 그것들을 겹쳐 하나의 완성형을 그렸다(§17.8의 [평가]). 그 완성형은 layer가 아니라 **배포 형태**의 명제다. Part III는 그 배포 형태를 systems 엔지니어의 도구 — roofline, GEMM shape, memory 계층, batching — 로 측정하는 원저 기여다. 이 장은 Part II의 결산을 이어받아 Part III의 프로그램 전체를 세운다.

## 18.1 Bridge-in: 초대장에서 시작한다

17장은 이렇게 닫혔다. "[Titans]가 'test time에 외우는 법을 배우자'로 열었던 질문은, 여섯 편째에 이르러 '모델은 언제 깨어 있고 언제 자야 하는가'로 바뀌었다. 그 질문의 답이 논문 한 편이 아니라 serving 시스템의 설계 문서처럼 생겼다는 것 — 그것이 이 라인이 inference 엔지니어에게 남긴 초대장이다." 이 장은 그 초대에 응하는 첫 걸음이다.

초대의 성격을 분명히 해 둔다. 여섯 편은 layer를 여섯 번 바꾸었지만, 그 누적 효과는 layer 교체가 아니라 **불변식 하나의 폐기**다: "추론 중 weights는 변하지 않는다"는 transformer serving의 정의적 전제(→ 1장 Rosetta의 "weight update 없음" 행)가 이 라인에서 사라진다. decode가 weights를 고쳐 쓰고, 배포에 훈련 job이 붙는다. 이 폐기가 무엇을 요구하는지는 여섯 논문 어디에도 측정되어 있지 않다 — decode wall-clock 수치는 6편 전체에 부재하다(§6.4의 라인 공통 caveat). Part III는 그 빈 자리를 채운다.

## 18.2 완성형, 여섯 편의 텍스트만으로

17장이 정식화한 완성형을 여기서는 systems 함의의 순서로 다시 읽는다. 논문들이 **스스로 세우고 검증한 것만** 합성하면 다음이 도출된다(dossier의 convergence 분석을 본문 서사로 옮긴 것이며, 각 성분의 근거는 Part II의 해당 장에 있다).

1. **state는 모든 시간 규모에서 weights다.** KV cache는 update frequency로 색인된 파라미터 블록의 스펙트럼으로 대체(hybrid에서는 sliding window로 최소화)된다 — per-token fast memory([Titans]·[Atlas], → 12장·14장), chunk 주기의 CMS level([NL], → 16장), sleep 주기의 grown expert([Sleep], → 17장), 그리고 frequency 0의 persistent weights. Titans→Sleep은 $\{\infty, 0\}$ 두 점뿐이던 frequency 축을 연속체로 넓혀 온 하나의 긴 운동이다.
2. **모든 블록은 같은 객체다** — (architecture, attentional bias, retention, inner optimizer, frequency)로 정의되는 associative memory이고, attention 자체가 그 안의 non-parametric·무한 capacity·frequency-$\infty$ 꼭짓점이다(→ 14장의 $\phi^*$).
3. **inner optimizer는 architecture와 대등한 설계 표면이다** — GD(→ 8장)→momentum([Titans])→objective 동물원([Miras])→window+Muon([Atlas])→자기 생성 target([NL]).
4. **훈련은 어디서나 chunk-anchored이고 필요한 곳에서 계층적이다** — stale-snapshot 근사가 여섯 편 전부의 병렬화 기반이며(→ 9장), 그 경제학과 reset을 [TNT]가 정리했다(→ 15장).
5. **lifecycle은 wake/sleep이다** — 모델은 결코 훈련이 끝나지 않고, 서빙 fleet은 설계상 주기적 fine-tuning job을 돌린다(→ 17장).

> **[평가]** 제품으로 읽으면 이것은 **세션 상태가 per-session/per-tenant mutable weights인 continually-learning LLM에, TNT 경제학으로 학습되고, sleep 서비스가 배포에 붙어 있는 시스템**이다. 저자의 논지 — "여섯 편을 다 읽으면 Google의 다음 수가 완성형으로 보인다" — 는 실질적으로 지지된다. 여섯 편의 open-questions 절이 전부 같은 다섯 공백(스케일, retrieval 격차, serving 경제학·kernel, 학습되는 스케줄, task-free·안전한 self-modification)으로 수렴하기 때문이다. 남은 일이 텍스트에 의해 과잉 결정되어 있다는 것이 이 논지의 힘이다.
>
> 단, 두 유보 없이는 정직하지 않다. **유보 1 — 완성은 개념적 완성이다.** from-scratch 실증은 1.3B params / 100B tokens에서 멈춰 있고([TNT]는 150M), 이 상한이 7B+·SFT/RLHF·production 데이터에서 유지되는지는 미지수다. **유보 2 — in-context retrieval 격차가 측정된 채 닫히지 않았다**: attention 53.55 vs 43.70 [Atlas Table 5], FDA 67.3 vs 41.9 [NL] (→ 14장·16장). capacity 이론이 그 이유($\phi^*$의 무한 capacity)까지 말해 주므로, 완성형은 attention과의 hybrid일 가능성이 열려 있다. 이 두 유보는 Part III의 모든 정량 주장이 딛고 서는 바닥이다.

## 18.3 완성형을 엔지니어의 질문으로: D4 pair thesis

완성형이 옳다면 그것은 하나의 **workload**를 만든다. 그 workload를 systems 관점에서 반으로 가르는 것이 이 책 Part III의 척추 명제다.

> **[평가] D4 workload-split pair thesis.** 이 라인의 배포는 질적으로 다른 두 부하로 갈라지며, 각각의 최적 하드웨어 전략이 다르다.
> - **decode/serving-state 관리 = memory-centric 기회.** decode step은 token마다 fast-weight state 전체의 **read-modify-write(RMW)** — 읽고, 갱신하고, 되쓴다 — 를 수행한다. 이 트래픽은 sequence마다 **unshared**이고, **write-heavy**이며, content로 주소 지정되지 않는다. 이것은 append-once/read-many이고 prefix로 공유 가능한 KV cache와 **질적으로 다른** 메모리 부하다.
> - **training/prefill = accelerator 영역.** 같은 알고리즘도 chunk 크기 $C$를 키우면 compute-bound로 옮겨 간다. chunk $C$는 문자 그대로 roofline의 x축이며, 이 영역의 승부는 fused chunk kernel과 grouped-GEMM에서 난다(→ 15장의 TNT가 이미 plain JAX로 FlashAttention을 이긴 지점).
>
> 흥미로운 하드웨어 제안은 둘 중 하나가 아니라 **쌍(pair)** 이다. 어느 한쪽만 옹호하는 것은 부하의 절반을 무시하는 것이다.

이 명제의 두 절반이 이 책이 memory-centric 논증을 펴는 **유일한** 두 지점이다. 그 밖 — "이 라인의 훈련이 새 메모리 소자를 요구한다", "일반 PIM이 답이다" — 은 논문들의 자체 증거가 반대 방향을 가리키므로(여섯 편이 알고리즘을 dense matmul로 다시 빚어 기존 accelerator에 맞췄다) 이 책은 주장하지 않는다. PIM은 오직 write-heavy elementwise epilogue(decay·renorm·AXPY)라는 소수 FLOP 지점에서만, 그것도 directional하게 등장한다(§18.6, → 20장·24장).

decode 절반이 "새 기회"인 이유는 그림 18-1의 한 교차점으로 압축된다. KV cache의 읽기 트래픽은 문맥 길이에 비례해 자라지만(read ∝ 문맥), TTT state의 RMW 트래픽은 문맥에 **무관하게 일정**하다(state 크기 = $m\,d^2\,L_{\mathrm{layer}}$로 고정; $m$은 **state multiplier** — state가 $d^2$의 몇 배인가로, deep memory $m\approx8$·momentum 포함 $m\approx16$ — 이고 $L_{\mathrm{layer}}$는 layer 수다). 따라서 그 아래에서는 KV가 싼 메모리 시스템이고 그 위에서는 TTT가 싼 **crossover 문맥 길이**가 존재한다.

![그림 18-1 — KV cache와 TTT state의 트래픽 교차: 문맥 길이에 따른 crossover와 그 스케일링](../../figures/exp-a-kv-ttt-crossover.png)

그림 18-1 — KV cache 읽기 트래픽은 문맥 길이에 비례해 증가하고 TTT state RMW 트래픽은 문맥에 무관하게 일정하므로, 두 곡선이 만나는 crossover 문맥 길이가 존재한다. anchor(neural-mem-1.3B)에서 read-crossover는 약 65k token, 전체 RMW-crossover는 약 131k token이며, crossover는 모델 폭에 따라 $m\,d^2$로 이동한다(실험 E1.3/E3 재구성).

## 18.4 이 책이 돌린 실측 프로그램: 8개 claim

Part III는 위 pair thesis의 두 절반을 각각 정량화하는 8개 실험을 돌렸다. 방법은 세 갈래다: 닫힌 형태 cost model(E3), analytic twin 도구(hatir + hat-schema, E1.x), CPU micro-benchmark(E2.x). 세 방법이 anchor 설정 `neural-mem-1.3B (d=2048, m=16, L=24, GQA-8, bf16)`에서 **1% 이내로 일치**한다(RMW 6.44 GB/token, 상태 134 MB/layer, read-crossover 65k token — E3·E1.1·E1.3·plan 4자 합치). 이 교차검증이 비율·crossover·순서를 본문으로 승격할 근거이며, 절대값은 승격하지 않는 이유는 §18.6에서 명문화한다.

**decode 절반(claim 1–6).** (1) decode step은 anchor에서 6.44 GB/token을 움직이고 arithmetic intensity가 0.59 FLOP/byte로, H100 twin ridge 295 FLOP/byte보다 두 자릿수 아래다 — **결정적으로 memory-bound**이며, state 트래픽이 GEMV 연산을 약 394× 압도한다: step 비용이 곧 state 트래픽이다. (2) 그 트래픽의 성격이 KV와 갈라진다 — KV의 write는 read의 0.003%(append-once)인데 TTT는 100%(대칭 RMW). (3) 그 결과 batch를 키우면 HBM 용량보다 **대역폭이 먼저** 막힌다 — KV cache가 capacity-bound인 것과 정반대다. (4) TTT state 크기가 문맥에 무관하므로 on-chip 상주 여부가 **설계 가능한 knob**이 되며(KV와 달리), state가 fit하는 폭에서는 scratchpad가 HBM 대비 이득이지만 폭이 커지면 spill한다(residency crossover). (5) NL/Sleep의 update-frequency 연속체가 memory 계층 배치로 **직접** 번역된다 — 상주 사본은 read/write 중 **빠른 쪽** cadence로 tier가 정해진다. (6) 기존 KV-manager의 semantics(prefix 재사용, append-only placement)는 RMW state에 대해 범주 오류다(재사용률이 구조적으로 0).

**training/prefill 절반(claim 7–8).** (7) chunk $C$가 roofline의 x축이다 — $C{=}1$(per-token RMW = decode 영역)은 AI≈1로 memory-bound 평원에 있고, $C$를 키우면 crossover를 넘어 compute-bound로 오른다: **한 알고리즘, 두 영역**. (8) on-die/off-die 경계에서 RMW 대역폭이 불연속으로 꺾이며(cliff), RMW는 read의 절반 throughput만 낸다 — append-once KV가 피하는 **write-back 세(稅)** 를 직접 측정한 값이다.

표 18-1 — 8개 claim과 뒷받침 실험·figure·장 매핑.

| claim | 한 줄 결론 | 실험 | figure | 주 담당 장 |
|---|---|---|---|---|
| 1 | decode = 전체 state의 memory-bound RMW | E1.1/E3 | — | 19, 20 |
| 2 | KV(append-once) vs TTT(RMW) 트래픽 crossover | E1.3/E3 | 18-1 | 18, 19 |
| 3 | batch는 대역폭이 먼저 막힘(≠ capacity-bound KV) | E1.3/E3 | — | 23 |
| 4 | state 상주는 설계 knob; scratchpad fit→spill | E1.2 | (b) | 20, 24 |
| 5 | update cadence → memory-tier 배치 규칙 | E1.4 | (d) | 20, 23, 24 |
| 6 | KV-manager는 RMW state에 범주 오류 | E1.5 | — | 10, 23 |
| 7 | chunk $C$ = roofline x축(memory→compute) | E2.1 | (c) | 9, 15 |
| 8 | on/off-die RMW 대역폭 cliff + write-back 세 | E2.2 | (e) | 20 |

## 18.5 Part III 로드맵

Part III는 위 프로그램을 일곱 장으로 편다. pair thesis의 어느 절반을 미는지 괄호로 표시한다.

- **19장 — Scaling analysis와 memory-centric scaling law 후보(decode).** 여섯 편의 ppl-vs-FLOPs/params/context 점을 digitize하고, state-bytes와 capacity($O(d_k^p)$, → 14장)를 params·tokens와 나란한 일급 scaling 축으로 제안한다. RMW 트래픽이 폭에 따라 0.45→6.44→343 GB/token으로, crossover가 16k→1.05M token으로 자라는 스케일링이 여기 들어간다.
- **20장 — Hardware bottleneck analysis(양쪽).** 아키텍처 class별 decode roofline(state read+**write** vs KV append-read), decode에 들어온 backward-pass라는 새 serving primitive, NS-5·deep-memory의 FLOP density, state placement(그림 b·e). memory-centric 논증의 본진이자 accelerator 반쪽(fused kernel)의 본진.
- **21장 — Transformer/NVIDIA scaling era의 교훈(accelerator).** hardware-lottery 독법: attention은 GEMM density로 이겼고, 이 라인은 chunkwise·NS·reset으로 **matmul 안에 설계되어** 있다 — 이 가족이 아직 기다리는 FlashAttention-moment이 무엇을 예고하는지.
- **22장 — Player-strategy analysis.** Google Research(TPU pod·JAX·long-context 제품)가 왜 자연스러운 저자인지, NVIDIA의 위치(Gated DeltaNet 계보), open-source kernel 생태계(flash-linear-attention)가 각각 합리적으로 두는 다음 수.
- **23장 — 대규모 training·serving 투영(양쪽).** TNT 경제학을 7–70B로 외삽; 계층적 memory의 prefill/decode 분할; **per-session weight state를 새 cache class로**(sizing, checkpoint/restore, multi-tenant batching — claim 3의 $B_{\max}$가 여기 근거 — 와 **frequency-tiered placement** — claim 5의 cadence→tier가 여기 본진, §23.6); sleep을 fleet 수준 background job으로. 오늘의 KV-cache 서빙과의 비용 모델 대조.
- **24장 — Proposals(양쪽, pair로).** 알고리즘 수준의 소규모 실험 제안; HW 수준의 정직한 memory-centric 평가(frequency-tiered placement, RMW-bandwidth decode 소자)와 accelerator 반쪽(fused chunk kernel, grouped-GEMM decode engine)을 **쌍으로**. 그림 b·d가 근거.
- **25장 — Conclusion과 research agenda.**

## 18.6 정직성 계약: exploration-grade

Part III의 모든 수치는 아래 계약 아래에서만 읽어야 한다. 이 계약은 §6.4 의무 caveat의 Part III 확장이다.

- **grade = exploration-grade.** 근거는 pre-silicon analytic cost model(hatir + hat-schema twin)과 CPU micro-benchmark다. 이들은 **비율·crossover 위치·tier 순서·bound class**를 산출하도록 자기 선언되어 있으며, silicon-accurate 절대값이 아니다. 따라서 load-bearing으로 본문에 승격되는 것은 이 네 종류의 양뿐이다.
- **절대값은 하한이고, 이월된다.** 6편의 원 논문이 H100 decode wall-clock을 **하나도** 공개하지 않으므로, 이 책이 내는 절대 µs/token·mJ/token(예: anchor의 1.92 ms/token, 46.5 mJ/token)은 roofline **하한**이며 외부 검증 불가다. 이 값들은 본문 주장의 근거가 아니라, 사내 A100 runbook(Part III-a)으로 이월되는 검증 대상이다. 본문이 딛는 것은 그 하한이 드러내는 **구조**(memory-bound, RMW-지배)뿐이다.
- **novel twin은 directional이다.** scratchpad·PIM twin은 `simulation_ready=False`로, verify committee가 PPA를 abstain한다. scratchpad의 6.8× energy / 14.9× time 이득, PIM의 1.6× energy 이득은 shipping-device 주장이 아니라 **directional DSE**로만 인용한다.
- **CPU 실측은 shape만 이전한다.** E2.x의 host roofline(ridge ≈ 34 FLOP/byte)은 H100 twin ridge(295 FLOP/byte)의 약 1/9이므로, 측정된 $C^*{\approx}32$나 cliff의 GB/s 절대값을 H100으로 옮기지 않는다. 이전되는 것은 곡선의 **모양**과 memory↔compute 교차의 **존재**뿐이다(H100 closed-form $C^*$는 306–430).
- **state multiplier는 가정값이다.** $m$(2-layer MLP $8d^2$, +momentum $16d^2$)과 $(d, L)$은 공개된 아키텍처 서술에서 온 canonical shape이지 공개된 decode trace가 아니다.

> **[해설]** 이 계약의 실천적 의미는 단순하다. Part III의 문장이 "TTT decode는 KV보다 폭이 클수록, 문맥이 65k token을 넘을수록 상대적으로 유리해진다"고 말하면 그것은 load-bearing 주장이다(비율·crossover). 반면 "그 decode가 1.92 ms 걸린다"고 말하면 그것은 하한의 보고이지 주장이 아니다 — 그 자리에는 항상 "roofline 하한, A100 runbook으로 이월"이라는 꼬리표가 붙는다. 이 구분을 흐리는 것이 이 책이 피하려는 유일한 과장이다.

## 요약

- 17장의 "초대장"을 이어받아, 이 장은 여섯 편의 완성형(state = 모든 시간 규모의 weights; 모든 블록이 같은 associative-memory 객체; inner optimizer가 설계 표면; chunk-anchored·계층적 훈련; wake/sleep lifecycle)을 systems 함의 순으로 재구성했다.
- 저자 논지("Google의 next가 완성형으로 보인다")는 open-questions의 수렴을 근거로 실질 지지되나, 두 유보(1.3B/100B 실증 상한, 미해소 retrieval 격차 53.55 vs 43.70)를 명시해야 정직하다.
- Part III의 척추는 **D4 pair thesis**다: decode/serving-state 관리 = memory-centric 기회(unshared·write-heavy·비-content-addressable한 whole-state RMW), training/prefill = accelerator 영역(chunk $C$ = roofline x축). 흥미로운 HW 제안은 쌍이다.
- 이 책은 pair thesis를 8개 실험으로 측정했고, 세 방법이 anchor에서 1% 이내로 일치한다. claim은 표 18-1로 장에 매핑된다(19–25장).
- 모든 수치는 exploration-grade 계약 아래 읽는다: 비율·crossover·tier·bound만 load-bearing, 절대 µs/mJ은 roofline 하한(A100 runbook 이월), novel scratchpad/PIM은 directional.

## 자가 점검 체크리스트

- [ ] 완성형의 다섯 성분을 텍스트 근거와 함께 재구성하고, 두 유보를 짚을 수 있다.
- [ ] D4 pair thesis의 두 절반과 각각의 최적 하드웨어 전략을 진술할 수 있다.
- [ ] memory-centric 논증이 허용되는 두 지점(decode RMW 트래픽, update-frequency↔tier)과 금지되는 두 지점(훈련용 새 소자, 일반 PIM)을 구별할 수 있다.
- [ ] KV(append-once)와 TTT(RMW)의 트래픽 crossover가 왜 생기고 어디에 있는지 설명할 수 있다.
- [ ] 어느 claim이 어느 장을 뒷받침하는지(표 18-1) 읽을 수 있다.
- [ ] 어떤 수치가 load-bearing이고 어떤 수치가 이월되는 하한인지 판별할 수 있다.
- [ ] pair thesis를 inference 어휘(paged KV cache ↔ per-session weight state; append vs RMW)로 옮길 수 있다.

## 다음 장으로

이 장은 완성형을 workload로, workload를 pair thesis로, pair thesis를 8개 측정으로 내렸다. 남은 것은 각 측정을 그 자체의 깊이로 펴는 일이다. 19장은 그 첫 축 — state-bytes와 capacity를 params·tokens 옆의 일급 scaling 축으로 세우고, RMW 트래픽과 crossover가 스케일에 따라 어떻게 자라는지를 여섯 편의 공개 점 위에 얹는다 — 로 시작한다.
