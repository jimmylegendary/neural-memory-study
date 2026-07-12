# ch24. 제안: 알고리즘 레벨과 하드웨어 레벨

> **이 장의 목표** — 독자가 이 장을 마치면 (1) 이 라인을 검증하고 밀어붙이기 위한 **알고리즘 레벨 제안** — 이미 돌린 소규모 실험이 무엇을 확정했고, 그것이 어느 falsifiable 대형 실험(train/serve chunk-consistency, state-capacity scaling)과 HOPE/Sleep 재현 runbook으로 이어지는지 — 을 진술할 수 있고, (2) **하드웨어 레벨 제안**을 D4 pair thesis의 두 절반으로 나눠, memory-centric 쪽(RMW-bandwidth decode 소자, frequency-tiered placement)과 accelerator 쪽(fused chunk kernel, grouped-GEMM decode engine, backward-capable serving kernel)을 **쌍으로** 제시할 수 있으며, (3) 그 사이에 놓이는 소프트웨어 제안(per-session-weights serving 스택)과, 이 책이 **명시적으로 배제**하는 두 FORCED 방향(훈련용 새 메모리 소자, 일반 PIM 옹호)을 구별할 수 있어야 한다.
> **왜 필요한가** — 19–23장은 pair thesis의 두 절반을 각각 측정하고 투영했다. 이 장은 그 측정을 **설계 제안**으로 바꾼다. 제안은 pair thesis의 규율을 그대로 물려받는다: memory-centric 논증은 정직 판정이 인정한 두 지점(decode RMW 트래픽, update-frequency↔tier)에서만 펴고, 나머지는 accelerator 쪽 제안으로 배치한다. 흥미로운 하드웨어 제안은 어느 한쪽이 아니라 **쌍**이라는 18장의 명제(§18.3)가 이 장의 조직 원리다.

## 24.1 Bridge-in: 측정에서 제안으로

23장은 TNT 경제학을 7–70B로 외삽하고 per-session weight state를 새 cache class로 세우며 pair thesis의 serving 절반을 투영으로 닫았다. 남은 것은 그 투영이 가리키는 **구체적 다음 수** — 무엇을 실험하고, 무엇을 만들 것인가 — 를 명제로 내리는 일이다. 이 장은 그것을 세 층으로 편다: 알고리즘(§24.2), 하드웨어(§24.3–§24.5), 그리고 배제(§24.6).

제안 전체를 관통하는 한 문장을 먼저 고정한다. **decode/serving-state 관리는 memory-centric 기회이고, training/prefill은 accelerator 영역이다**(D4 pair thesis, → 18장 §18.3). 이 장의 모든 하드웨어 제안은 이 분할의 어느 쪽에 속하는지로 정당성을 얻는다. memory-centric 제안이 정당한 이유는 그것이 논문들 자신의 cost accounting — token마다 fast-weight state 전체를 read-modify-write(RMW)하는 트래픽 — 에 근거하기 때문이고(§18.2의 두 진짜 지점), accelerator 제안이 정당한 이유는 같은 알고리즘이 chunk 크기 $C$를 키우면 compute-bound로 옮겨 가기 때문이다. 두 제안이 하나의 workload를 나눠 맡는다는 점이 이 장의 유일한 새 주장이다.

정직성 계약(→ 18장 §18.6)은 이 장에도 그대로 적용된다. 아래 인용하는 실측 수치 중 **load-bearing으로 승격되는 것은 비율·crossover 위치·tier 순서·bound class뿐**이다. 절대 µs/token·mJ/token은 roofline 하한이고 사내 A100 runbook(Part III-a)으로 이월된다. novel scratchpad·PIM twin의 이득 배율은 `simulation_ready=False`인 directional DSE로만 인용한다. 이 계약이 특히 이 장에서 중요한 이유는, 제안은 미래형이라 과장의 유혹이 가장 크기 때문이다.

## 24.2 알고리즘 레벨 제안

### 24.2.1 이미 확정한 것: 두 microbench가 고정한 곡선

Part III는 pair thesis를 8개 실험으로 측정했고(→ 18장 표 18-1), 그중 두 CPU microbench가 **알고리즘 레벨의 곡선 모양**을 하드웨어 불문으로 고정했다. 이 둘은 제안의 출발점이자 이후 대형 실험의 기대값이다.

첫째, **chunk $C$가 roofline의 x축이다**(claim7, E2.1). host roofline에서 $C{=}1$의 per-token rank-1 RMW — 이것이 곧 TTT decode 영역이다 — 은 arithmetic intensity 약 1 FLOP/byte로 memory-bound 평원(host roofline의 약 12%)에 붙어 있고, $C$를 키우면 crossover를 넘어 compute-bound로 오른다: **한 알고리즘, 두 영역**. crossover 위치의 절대값($C^*{\approx}32$ host)은 ridge 의존적이라 H100으로 옮기지 않는다(H100 closed-form $C^*\approx306$–$430$, → 9장·15장). 이전되는 것은 곡선의 **모양**과 memory↔compute 교차의 **존재**뿐이다. 둘째, **on-die/off-die 경계에서 RMW 대역폭이 계단식으로 꺾인다**(claim8, E2.2): in-cache RMW peak 대비 DRAM으로 spill하면 유효 대역폭이 약 3.9× 붕괴하고(in-cache peak÷DRAM), RMW는 element당 **2× 바이트**(read+write)를 움직인다 — append-once KV가 피하는 **write-back 세(稅)** 를 직접 측정한 값이고, 동일 byte 대역폭 환산 시 유효 element throughput은 read의 약 0.5×다(GB/s 절대값은 host 좌표, 이전 금지).

> **[해설]** 이 두 곡선은 대칭이다. 같은 rank-1 RMW가 decode에서는 memory-bound의 근원($C{=}1$)이고 prefill에서는 $C$를 키워 벗어나는 대상이다. 제안이 pair로 갈리는 알고리즘적 뿌리가 여기 있다 — 하드웨어 두 제안(§24.3, §24.4)은 이 한 곡선의 두 끝을 각각 겨냥한다.

### 24.2.2 대형 실험 제안 A: train/serve chunk-consistency

TNT의 chunk-size mismatch(→ 15장)는 라인 전체에서 **단일 설정의 한 그림**에만 근거한다: 550M Titans를 $C{=}64$로 훈련하면 $C{=}64$에서 ppl 13.78이던 것이 $C{=}8$ 서빙에서 36.45로 무너진다는 [TNT Fig. 2]. 이 그림에는 gating도 momentum도 없고, mismatch가 **왜** 생기는지에 대한 이론도 없다. TNT App. D는 gated/momentum 변형과의 합성을 명시적으로 유보했다.

> **[평가] 제안 A — chunk-consistency grid.** 100–150M 규모에서 (train chunk × inference chunk) 격자를 {plain GD, +momentum $S_t$, +retention gate $\alpha_t$} 세 update rule에 대해 돌린다. 목적은 두 가지다: (i) mismatch가 세 rule에서 모두 재현되는가, 아니면 momentum/gating이 그것을 완화하는가; (ii) 왜 over-specialization이 일어나는지의 loss-landscape probe. 이 실험은 어느 방향으로 떨어져도 publishable하다 — mismatch가 재현되면 TNT의 발견을 라인 전체로 일반화하는 것이고, gating/momentum에서 사라지면 TNT의 일반성 주장에 대한 **정직한 negative result**다. 비용은 TNT의 10B-token 예산을 축소 스케일에서 크게 밑돈다.

이 실험이 서 있는 알고리즘적 근거는 §24.2.1의 claim7 곡선이다: chunk 크기는 throughput knob이면서 **계산되는 함수 자체를 바꾸는 semantic hyperparameter**다(stale-snapshot 근사, → 9장). train/serve chunk가 다르면 함수가 다르다는 것이 mismatch의 **유력한 후보 기제**이며(TNT Fig. 2는 550M·$C_{\mathrm{train}}{=}64$ 단일 설정이고 원전도 인과를 이론으로 규명하지 않았다 — 확정된 원인이 아니라 제안 A가 loss-landscape probe로 검증할 가설이다), 이 실험은 그 기제가 gate·momentum에 얼마나 민감한지를 측정한다.

### 24.2.3 대형 실험 제안 B: state-capacity를 scaling 축으로

19장은 state-bytes와 capacity($O(d_k^p)$, → 14장의 $\phi^*$)를 params·tokens와 나란한 일급 scaling 축으로 제안했다. 그 제안을 실험으로 닫는 것이 제안 B다.

> **[평가] 제안 B — capacity scaling fit.** 3–4개 크기 × 3개 architecture class(matrix memory, deep-MLP memory, deep+featurized memory)를 **동일 토큰**으로 훈련하고, ppl을 (params, state-bytes, capacity-proxy)에 대해 fit한다. 핵심 질문: capacity-proxy가 raw state-bytes보다 long-context 벤치마크를 더 잘 예측하는가. falsifier는 정직하게 둘이다 — (i) 공개 점들이 tokenizer·데이터가 이질적(라인 전반 Llama-2 vs T5 vocab)이라 cross-architecture fit이 불안정하고, (ii) 자체 run이 exponent를 안정화하기엔 너무 작을 수 있다. 이 실험은 안정적 exponent가 아니라 **capacity 축의 예측력 유무**를 판정하는 것이 목표다.

> **[결과] E4가 이미 좁힌 것.** 제안 B의 대형 run은 미실행이지만, 그 전초로 여섯 편의 공개 점(params·tokens·context·ppl/score)을 digitize해 후보 scaling fit을 돌린 소규모 실험(E4)이 이미 A3를 **날카롭게 좁혔다**. 결과는 정직하게 둘로 갈린다. (i) **perplexity 축에서 capacity는 아무것도 더하지 않는다**: parametric family(matrix→deep-MLP→deep+poly) 안에서 state-bytes와 capacity-ordinal은 rank-동치이고 둘 다 ppl과 완전 반상관($\rho{=}-0.95$)이라, capacity-proxy가 raw state-bytes 위에 얹는 예측력이 0이다. 게다가 attention corner(capacity 무한이되 ppl은 최악, Transformer++ 18.53)를 넣으면 capacity↔ppl 상관이 뒤집힌다($\rho{=}0.09$; raw state-bytes는 attention=0으로 $\rho{=}-0.69$ 유지) — ppl은 압축을 보상하지 raw retrieval capacity를 보상하지 않으므로, capacity를 ppl 축으로 fit하면 **틀린 것을 재는** 셈이다. (ii) 반대로 **long-context/retention 축에서는 capacity가 살아난다**: BABILong 지속 정확도 길이가 parametric family 안에서 capacity-ordinal과 단조($\rho{=}1.0$)이고, 특히 Titans→Atlas의 약 6.7× retention 도약을 약 1.5×에 불과한 state-byte 증가가 underpredict하는데 capacity-**class** 변화(deep-MLP→deep+poly, ordinal 2→3)가 그 도약을 설명한다. 그러므로 제안 B의 남은 대형 run은 ppl 예측이 아니라 **retention 예측**으로 재조준되고, 19장은 state-bytes·capacity를 **long-context 타깃에 한해** 일급 scaling 축으로 제안하되 cross-architecture ppl↔capacity fit은 명시적으로 경고한다(→ 19장). (모든 점은 tokenizer·데이터가 이질적이라 ordinal·directional; 자체 3–8점은 Spearman $\rho$만 load-bearing, 절대 exponent는 directional 하한.)

### 24.2.4 이 실험들의 vehicle: HOPE/Sleep 재현 runbook

제안 A·B와, 라인 전체의 부재한 decode wall-clock(6편 전부 미공개, → 18장 §18.2)을 실측하려면 실행 가능한 재현 경로가 있어야 한다. 정찰 결과는 냉정하다: **6편 전부 공식 구현이 없고**, 사실상의 reference는 lucidrains/titans-pytorch 하나이며 그마저 논문 밖 확장이 섞여 있다. HOPE는 비공식 2종(84★/76★)이 소규모·단일 GPU 수준으로만 존재하고, **Sleep의 wake/sleep 파이프라인은 지구상에 구현이 0개**다.

> **[평가] 제안 C — 하이브리드 4층 runbook.** (1) 베이스라인·백본 = flash-linear-attention + flame(공식급, multi-GPU 검증됨); (2) Titans neural memory = lucidrains 참조를 **논문 모드로 flag 고정**하고 `fla/ops/titans` naive를 수치 oracle로 병용; (3) HOPE = 자체 구현(비공식 2종은 참조·감사 대상이지 신뢰 기반 아님 — 둘 다 surrogate loss를 섞으므로 fork하면 "무엇을 측정했는지"가 모호해진다); (4) Sleep/Dreaming = 완전 자체 구현. 4층은 제안 A·B의 model-training 실험과 A100 runbook의 wall-clock 실측을 동시에 실어 나른다. Sleep 구현은 이 스터디의 잠재적 오픈소스 기여 지점이다.

이 runbook이 memory-centric 서사에 종속되지 않는다는 점을 분명히 한다. runbook의 1차 산출물은 알고리즘 검증(chunk semantics, capacity 예측력)과 절대 wall-clock이며, 그 wall-clock이 나와야 §24.3의 하드웨어 제안의 절대값 하한이 검증 가능한 실측으로 바뀐다. 즉 알고리즘 제안이 하드웨어 제안의 warrant를 공급한다.

> **[결과] runbook이 실측할 것과, 이미 digitize한 것.** 4층 runbook의 1차 산출물은 셋이다: (a) chunk semantics — 제안 A의 (train chunk × infer chunk) 격자; (b) capacity 예측력 — 제안 B의 retention fit; (c) 절대 decode wall-clock — 라인 전체가 미공개한, §24.3 하드웨어 제안의 절대값 하한을 검증으로 바꾸는 수치. (c)는 A100 runbook이 실측하고, (a)·(b)의 **부호**는 E4가 공개 점 digitize로 이미 예열했다. 두 TNT 점이 그것이다: (1) TNT 안에서 local memory 수를 0→4로 늘리면 avg ppl이 23.53→20.15로 **단조** 감소하되($\rho{=}-1.0$) +1 이후 saturate(21.04→20.15)해 state 수의 diminishing return을 보이고 — 이는 제안 B가 "state가 많을수록 좋다"를 순진하게 fit하지 못하게 하는 경계다; (2) 550M Titans를 $C{=}64$로 훈련한 뒤 infer chunk를 쓸면 ppl이 **훈련 chunk에서 최소**인 U자($C{=}8{:}36.45$, $C{=}64{:}13.78$, $C{=}512{:}22.4$; argmin $C{=}64$)로, scaling law가 아니라 train/serve resolution mismatch(→ 15장)의 digitize된 재확인이다. 두 점 모두 제안 A·B의 대형 run이 어느 쪽으로 떨어질지의 부호를 미리 고정한다 — 대형 run이 이 부호를 뒤집으면 그 자체가 라인의 일반성 주장에 대한 정직한 negative result다. (모든 수치는 published-data digitize 또는 exploration-grade fit; 절대 ppl은 원문 좌표.)

### 24.2.5 알고리즘 제안 J: learned chunk/cadence schedule

제안 A·B·C가 **고정된** schedule 위에서 라인을 측정한다면, 마지막 알고리즘 제안은 그 schedule 자체를 학습 대상으로 올린다. 라인 전체에서 chunk 크기 $C$, window $c$, CMS 갱신 주기 $C^{(\ell)}$, level 수, sleep 발화 시점은 전부 hand-set이다(→ dossier §3.2의 다섯 공백 중 네 번째). NL은 잘못 배치된 level이 오히려 해가 됨을 자기 ablation(inner-$q$)으로 보였고, Sleep은 sleep을 CMS chunk 경계에 hard-wire했다 — **무엇을 언제 갱신할지**를 배우는 기제는 아직 없다.

> **[평가] 제안 J — learned update schedule.** claim7이 고정한 사실 — chunk 크기는 throughput knob이자 계산되는 함수를 바꾸는 semantic hyperparameter(stale-snapshot 근사, → 9장) — 이 이 제안의 근거다. $C$가 semantic이면 그것을 상수로 두는 것은 손실이다: token·context에 따라 update cadence를 내는 작은 controller(예: momentary surprise 크기 $\|g_t^{\mathrm{in}}\|$에 조건한 chunk-boundary gate)를 outer loop으로 meta-learn한다. 이것은 §24.3.2의 frequency-tiered placement와 직접 맞물린다 — cadence가 학습되면 tier admissibility도 정적 표가 아니라 **동적 배치 신호**가 되기 때문이다. falsifier는 둘이다: (i) 학습된 schedule이 잘 튜닝된 상수 $C$를 유의하게 이기지 못하면(제안 B의 diminishing-return 경계가 시사하듯 라인 성능이 $C$·state 수에 완만하면) 제안은 복잡도만 늘린다; (ii) chunk 경계의 이산성이 gradient를 끊으므로 straight-through류 완화가 필요하고, 그 완화가 원래의 stale-snapshot 근사와 겹쳐 "무엇을 측정했는가"를 흐릴 위험이 있다. 이 제안은 그래서 제안 A(chunk-consistency)가 mismatch의 뿌리를 gate·momentum에 대해 고정한 **뒤에야** 정직하게 돌릴 수 있다.

## 24.3 하드웨어 레벨 — memory-centric 절반

이 절의 두 제안은 pair thesis가 인정한 **진짜 두 지점**에서만 나온다. 그 밖의 memory-centric 주장은 §24.6에서 명시적으로 배제한다.

### 24.3.1 RMW-bandwidth decode 소자

decode step은 anchor(neural-mem-1.3B)에서 token마다 6.44 GB를 움직이고 arithmetic intensity가 0.59 FLOP/byte로, H100 twin ridge 295 FLOP/byte보다 두 자릿수 아래다 — **결정적으로 memory-bound**이고, state RMW 트래픽이 decode step의 GEMV 연산을 약 394× 압도한다(claim1, E1.1/E3). **step 비용이 곧 state 트래픽이다.** 이 트래픽은 모델 폭에 따라 $m\,d^2\,L_{\mathrm{layer}}$로 자라 0.45→6.44→343 GB/token(Titans-170M→가상 70B)으로 증가하고, KV와 달리 문맥에 무관하게 일정하다. (절대 1.92 ms/token·46.5 mJ/token은 roofline 하한이며 A100 runbook으로 이월한다 — 본문이 딛는 것은 memory-bound·RMW-지배라는 **구조**뿐이다.)

이 구조가 요구하는 소자의 성격은 분명하다: **높은 RMW 대역폭**, **per-tenant state residency**, 그리고 write-heavy elementwise epilogue를 위한 **near-memory update engine**. 근거는 두 실측이다. 첫째, TTT state 크기는 문맥에 무관하게 $(d, m, L)$로 고정되므로 on-chip 상주가 **설계 가능한 knob**이다(context에 따라 자라는 KV와 결정적으로 다른 지점, claim4/E1.2). state가 fit하는 폭에서는 268 MB scratchpad가 HBM3 대비 6.8× energy / 14.9× time 이득을 내지만(directional DSE), 폭이 커지면 spill한다 — residency crossover는 hidden dim $d^*{\approx}2896$이고, 268 MB 버퍼는 한 layer state를 $d{=}2048$(1.3B, 134 MB/layer)까지 담고 $d{=}4096$(7B, 537 MB/layer)에서 넘친다. whole-model 상주는 Titans-170M(226 MB) 하나만 fit하므로, **스케일에서 placement는 whole-model pin이 아니라 per-layer/streamed**다.

![그림 24-1 — state placement의 device별 energy/latency 트레이드오프와 residency crossover](../../figures/exp-b-state-placement.png)

그림 24-1 — 134 MB/layer RMW state를 device별로 배치했을 때의 energy·time과, 268 MB scratchpad가 한 layer state를 담을 수 있는 폭의 상한(residency crossover $d^*{\approx}2896$). state가 fit하는 구간에서만 scratchpad가 HBM3를 이기며, 이득 배율은 `simulation_ready=False`인 directional DSE다(실험 E1.2 재구성).

둘째, 이 상주/spill 불연속이 로컬 실측으로도 재현된다: on-die/off-die 경계에서 RMW 대역폭이 약 3.9× 꺾이는 cliff가 그것이다(claim8/E2.2, 그림 24-2). 소자 제안의 요지는 이 cliff의 **on-die 쪽에 layer state를 붙잡아 두는** 것이며, per-layer 단위로는 그것이 가능하다는 것이 E1.2/E1.4가 함께 보이는 결론이다.

![그림 24-2 — on-die/off-die RMW 대역폭 cliff (E1.2 residency crossover의 로컬 아날로그)](../../figures/exp-e-rmw-cliff.png)

그림 24-2 — state가 on-chip cache에 상주하는 동안 RMW는 빠르고, 용량 경계를 넘겨 DRAM으로 spill하면 유효 대역폭이 약 3.9× 붕괴한다(in-cache peak÷DRAM). RMW가 element당 2× 바이트(동일 대역폭 환산 시 element throughput 0.5×)를 움직이는 것이 append-once KV가 피하는 write-back 세다. GB/s 절대값은 host 좌표이며 H100으로 이전하지 않는다 — 이전되는 것은 cliff의 **존재**뿐이다(실험 E2.2).

near-memory update engine에 대해서는 PIM이 **오직 write-heavy elementwise epilogue**(decay·renorm·AXPY)라는 소수 FLOP 지점에서만, 그것도 directional하게 등장한다. TTT epilogue의 약 3 MAC/elem에서 PIM은 1.6× energy 이득이나 2.1× latency 비용을 내고, rank-1에 가까운 1 MAC/elem에서는 energy와 latency 둘 다 이득이다(claim4/E1.2). latency 손해는 1과 3 MAC/elem 사이에서 뒤집혀 측정 격자의 3 MAC/elem에서 이미 물린다(1 MAC/elem에서는 0.7× time으로 이득).

이 세 요구(높은 RMW 대역폭·per-tenant residency·write-heavy epilogue)를 하나의 소자 스케치로 모으면 **near-memory update engine**의 마이크로아키텍처가 된다(directional, silicon 미검증). 데이터패스는 세 스테이지다. (1) **state-tile 상주 버퍼** — 한 layer의 fast-weight $W$(+momentum $S_t$)를 tile 단위로 on-die/scratchpad에 붙잡는 SRAM. per-layer 상주가 가능한 폭(residency crossover $d^*{\approx}2896$ 아래)에서만 tile이 fit하고, 넘으면 streamed다 — whole-model pin은 Titans-170M만 가능하므로(claim4) 상주 단위는 layer다. (2) **epilogue ALU lane** — read된 state tile에 master update $W_t=\alpha_t W_{t-1}+S_t$(→ STYLE (M2))를 적용하는 write-heavy 경로: decay($\alpha_t\odot$), AXPY($S_t{=}\beta_t S_{t-1}-\eta_t g_t^{\mathrm{in}}$ 누적), renorm을 한 pass에 fuse한다. 이 lane이 §24.3.1이 PIM-적합으로 한정한 minority-FLOP epilogue(약 1–3 MAC/elem)를 흡수하고, GEMV/GEMM 형태의 gradient 계산 본체는 tensor-core로 되돌린다 — §24.6 배제 2와 정합이다. (3) **per-tenant residency slot table** — state가 sequence별 unshared 사본이므로(claim4), engine은 slot↔sequence 바인딩과 slot별 dirty 비트를 유지하는 작은 directory를 갖는다. 이 directory가 곧 §24.5 소프트웨어 층의 `bind_to_sequence`·`mark_dirty`의 하드웨어 짝이다: update는 append가 아니라 in-place로 slot을 덮어 T개 stale 버전 누적을 막고(claim6 G2, 256×), eviction 시 dirty slot은 반드시 writeback 경로로 흐른다(무료 drop 금지, G3). 요컨대 소자·매니저·kernel이 같은 세 measured 사실(memory-bound whole-state RMW, per-seq residency, 항상-dirty)을 세 층에서 각각 구현하며, 소자 D의 근거는 이 세 사실 어느 것도 위에 얹은 것이 아니라 논문들의 cost accounting에서 읽어 낸 것이다.

> **[평가] 제안 D — RMW-bandwidth decode 소자.** high-RMW-bandwidth state residency(per-layer 상주를 겨냥한 on-die/scratchpad 계층) + per-tenant state placement + elementwise epilogue를 위한 near-memory update path. 이 제안은 논문들의 cost accounting에 **근거한 것이지 그 위에 얹은 것이 아니다** — decode가 memory-bound whole-state RMW라는 사실은 measured structure이고, epilogue의 PIM 적합성은 minority-FLOP·directional로 한정된다. 절대 이득 배율은 silicon 검증 전까지 directional로만 인용한다. 한 요구가 더 있다 — **multi-tenant isolation과 privacy**다(dossier §3.2의 serving 공백). KV page는 append-once·공유 가능이라 tenant 간 격리가 논리적 partition으로 족하지만, RMW state는 tenant의 문맥이 weight에 **흡수**된 사본이라 slot이 물리적으로 격리돼야 하고(한 slot의 잔여가 다음 tenant에 새면 문맥 누출이다), free는 반드시 잔여 zeroization을 동반해야 한다. 이는 slot table의 bind/free semantics(claim6)를 소자 수준 격리 요구로 승격한다. 이 요구는 정직 판정의 두 지점에 얹는 새 주장이 아니라 per-tenant residency의 직접 귀결이다 — falsifier: state가 사실상 문맥을 복원 불가능하게 압축한다면(privacy가 압축으로 자동 확보되면) 물리 격리의 필요성이 약해진다. 그 복원 가능성 자체가 미해결 질문이므로 이 요구는 보수적으로 유지한다.

### 24.3.2 frequency-tiered memory placement

두 번째 memory-centric 지점은 NL/Sleep의 update-frequency 정의에서 **직역**된다. Nested Learning(HOPE)과 Sleep은 각 memory level에 update cadence를 부여한다(per-token fast weights → chunk 주기 CMS level → sleep 주기 grown expert → frozen weights, → 16장·17장). 이 cadence는 memory 계층의 tier 배치 사양으로 거의 문자 그대로 번역된다(claim5/E1.4).

번역의 규칙은 하나다: **상주 사본은 read/write 중 빠른 쪽 cadence로 tier가 정해진다**(gating rule). 그 귀결이 placement 표다 — per-token fast weights는 HBM3에 pin되고; per-token read이되 chunk 단위 write인 CMS-chunk level은 read cadence가 뜨겁게 붙잡아 HBM3에 남으며; read·write가 모두 4096 token 주기로 down-sample된 CMS-mid level은 CXL로 legally 내려가고; offline cadence의 Sleep-consolidated expert도 CXL로 간다; frozen weights는 per-token read라 HBM3의 read-heavy(accelerator-friendly) 쪽에 놓인다. 핵심 통찰은 **sub-per-token access cadence가 트래픽을 critical path 아래로 amortise한다**는 것이다 — write-heavy RMW state라도 갱신 주기가 길면 느린 tier가 허용된다.

**정직한 경계 하나 — read cadence의 출처.** CMS-mid를 CXL로 내리는 근거는 그 level의 **read**가 4096 주기로 down-sample된다는 가정이다. 그러나 Nested Learning Eq. 70과 Sleep Eq. 1의 일반 forward는 **모든 CMS block을 각 token의 forward chain에서 호출**하며, 주기가 다른 것은 parameter **update**(Eq. 71/Eq. 2)뿐이다. 즉 write cadence가 4096이라고 read cadence까지 4096인 것은 아니다 — block이 매 token forward에 남으면 read는 per-token이고 gating rule은 그것을 HBM3에 pin한다. 따라서 CMS-mid의 CXL 강등은 그 block이 **sparse routing·cached activation·비활성 expert offload로 forward에서 genuinely 건너뛰어질 때에 한해** admissible하며(E1.4의 read cadence는 공개 decode trace가 아닌 그 가정 위의 값이다), 그렇지 않으면 활성 block은 per-token read 때문에 hot tier에 남는다. Sleep-consolidated expert도 consolidation cadence(262144)는 offline write이되 wake 중 read는 routing에 의존한다. 이 tier 배치가 조건부 directional 가설인 이유가 여기 있다.

![그림 24-3 — update cadence → memory-tier 허용 표](../../figures/exp-d-frequency-tiers.png)

그림 24-3 — CMS level별 update 주기(행)와 memory tier(열)의 허용 관계. latency tolerance가 update 주기에 선형으로 넓어지므로($P_{\text{read}}\cdot t_{\text{tok}}$), down-sampled level일수록 싼 tier가 legally 허용된다. per-layer fast-weight block(67 MB)은 268 MB scratchpad에 fit해 HBM3 대비 7.4× energy 이득을 내지만 all-layer는 fit하지 않는다(실험 E1.4, scratchpad 이득은 directional).

> **[평가] 제안 E — frequency-tiered placement engine.** update-frequency 연속체를 tier admissibility 규칙으로 소비하는 배치 계층: hot fast-weights를 HBM3/on-die에, down-sampled CMS-mid를 DDR/CXL-class latency로, sleep-consolidated expert를 명시적 data-migration 스케줄로. 이것은 이 스터디에서 가장 신선한 memory-architecture 주장이다 — NL/Sleep의 정의가 **이미** memory-hierarchy 사양이라는 것을 어떤 선행 연구도 진술하지 않았고, 매핑이 거의 문자 그대로이기 때문이다. 다만 이 규칙은 cadence를 tier의 **한 입력 변수**로 삼는 directional 가설이지 단독 admissibility 법칙이 아니다: 드문 접근이 평균 대역폭·energy를 amortise하더라도, 그 접근이 실제로 일어나는 token은 prefetch·비동기 실행이 없으면 여전히 느린 tier의 **동기 latency**를 문다. 따라서 CXL 배치의 실 admissibility에는 cadence 외에 transfer size, link latency, prefetch window, 동시 상주 sequence 수가 함께 들어가며, 이 tier 판정은 `simulation_ready=False`인 directional DSE로만 인용한다.

placement 표를 **정적** 배치로만 읽으면 절반이다. update-frequency가 tier를 정한다는 규칙은 곧 **cadence가 바뀌는 순간이 마이그레이션 트리거**라는 뜻이고, wake/sleep lifecycle(→ 17장)이 정확히 그 순간을 만든다. 그래서 표(그림 24-3)는 다음 마이그레이션 스케줄로 펴진다. (a) **promotion(CXL→HBM3)** — sleep이 consolidated expert를 grow해 그것이 wake에서 per-token read 대상이 되면, read cadence가 그 사본을 hot으로 재분류하므로 다음 wake 진입 전에 HBM3로 올린다. (b) **demotion(HBM3→CXL)** — sleep의 synaptic-pruning reset이 fast block의 낡은 expert를 무효화하면 그 사본은 더 이상 per-token 접근되지 않으므로 CXL/idle로 내린다. (c) **정적 상주** — per-token fast weights와 frozen weights는 cadence가 바뀌지 않으므로 이동 없이 HBM3에 남는다(각각 write-gated·read-gated). 마이그레이션 비용 자체가 tier 판정의 입력이라는 점이 중요하다 — sub-per-token cadence가 트래픽을 amortise하더라도, 승격/강등 transfer가 임계 window 안에 끝나지 못하면 그 접근 token은 여전히 느린 tier의 동기 latency를 문다. 따라서 이 스케줄의 admissibility에는 cadence 외에 transfer size, link latency, prefetch window가 함께 들어가며, 스케줄 전체는 `simulation_ready=False`인 directional 가설이지 완결된 배치 법칙이 아니다.

## 24.4 하드웨어 레벨 — accelerator 절반 (pair의 다른 쪽)

memory-centric 제안(D·E)은 pair thesis의 절반일 뿐이다. training/prefill과 decode의 dense-matmul 부분은 accelerator 영역이며, 여기서의 승부는 kernel에서 난다. 세 제안이 §24.2.1의 claim7 곡선의 compute-bound 끝을 겨냥한다. E2.1 host 측정에서 같은 chunkwise scan이 $C{=}1$의 4.56 GFLOP/s에서 $C{=}512$의 689.5 GFLOP/s로 약 151× 오르는 것이(측정 wall-clock, shape-only) 이 끝의 존재를 로컬로 확인한다 — 절대 GFLOP/s는 host 좌표라 이전 금지, 이전되는 것은 상승의 **모양**뿐이다.

**제안 F — fused chunk kernel.** claim7이 보이듯 $C$를 키우면 chunkwise scan이 compute-bound로 오른다. 그런데 정찰 증거는 이 kernel이 **아직 없다**는 것을 구조적으로 드러낸다: flash-linear-attention의 `fla/ops/ttt`는 진짜 Triton chunkwise 커널을 가졌지만, `fla/ops/titans`는 naive PyTorch 참조 구현에만 머문다. TTT만 Triton화되고 Titans가 naive에 남은 이 사실 자체가 **momentum·retention을 포함한 deep-memory chunk의 fused 커널이 아직 없다**는 생태계 측 물증이다 — 그것이 단순한 구현 공백인지 momentum의 chunkwise 병렬화 자체의 난점인지는 제안 F가 가를 열린 질문이다(→ 9장·15장; 아래 [리스크] F). 제안은 momentum·retention을 포함한 deep-memory chunk를 fuse하는 커널이며, 이것이 라인이 아직 기다리는 "FlashAttention-moment"의 후보다(→ 21장). TNT가 plain JAX만으로 32K에서 step당 FlashAttention을 이미 이겼다는 점(→ 15장)이 이 방향의 상한이 존재함을 시사한다. (단, TNT는 momentum·gating·Muon을 "명료성을 위해" 제거한 단순화 Titans로 검증했으므로 — App. D — 라인 전체와의 합성은 미검증이다.)

> **[리스크] F.** falsifier: momentum·retention을 포함한 chunk의 chunkwise closed form이 존재하지 않아 fuse해도 sequential 의존이 남으면 — 즉 fla가 Titans를 naive PyTorch에 둔 것이 편의가 아니라 필연이면 — 커널은 부분적 fusion에 그친다. 또 TNT의 FlashAttention 우위는 momentum·gating을 제거한 단순화 Titans에서만 측정됐으므로(App. D), 상한의 존재는 시사일 뿐 full-line 합성 커널로 이전된다는 보장이 아니다.

**제안 G — grouped-GEMM decode engine.** per-request mutable fast weights는 shared-weight batching을 **깨뜨린다**(→ 23장, 1장 Rosetta의 batching 행). token마다 request별로 다른 $W$를 RMW하므로, 오늘의 decode kernel이 전제하는 "여러 request가 같은 weight를 공유한다"가 성립하지 않는다. 제안은 request별 state를 batch 축으로 묶어 처리하는 grouped-GEMM decode path다. 이것이 accelerator 제안인 이유는, request별 unshared weight의 grouped-GEMV가 **근본 arithmetic intensity를 올리지는 못하지만**(각 $W_b$가 한 번 읽혀 한 번 쓰이므로 FLOP·traffic이 함께 $B$에 비례해 AI는 $B$와 무관), kernel-launch·occupancy·scheduling 파편화를 걷어 memory-bound 평원의 **포획 손실**을 줄이기 때문이다. AI 자체를 올리려면 같은 session $W$에 여러 token/vector를 태우는 chunk·speculative batch나 구조화·state 공유가 따로 필요하다(→ §24.5 `bind_to_sequence`).

> **[리스크] G.** falsifier: decode가 결정적으로 memory-bound(AI 0.59 ≪ ridge 295, claim1)이므로 grouped-GEMM이 arithmetic intensity를 끌어올려도 state RMW 트래픽이 여전히 지배하면 이득이 상한에 막힌다. G는 batching **가능성**을 복원하는 것이지 트래픽 하한을 낮추지 않는다 — 그 하한은 소자 D의 몫이며, 그래서 G와 D는 같은 decode step을 나눠 맡는다.

**제안 H — backward-capable serving kernel.** 이 라인의 decode는 inference인데 **backward pass를 품는다** — inner loop이 $\nabla_W \ell$를 계산해 state를 갱신하기 때문이다(→ 1장 Rosetta의 "backward pass가 decode에 들어온다", 8장). 오늘의 serving kernel은 forward-only로 설계되어 있어 이 primitive를 지원하지 않는다. 제안은 decode 경로에 inner-loop backward를 일급 연산으로 포함하는 serving kernel이다. NS-5(Newton–Schulz, → 14장)와 deep-memory의 FLOP density가 여기서 accelerator 자원을 요구하는 지점이며(→ 20장), 이것은 memory-centric 소자가 아니라 tensor-core 영역의 문제다.

> **[리스크] H.** falsifier: inner-loop backward가 forward 대비 지배적 FLOP이 아니면(deep memory가 얕고 NS-$\kappa$ 반복이 적으면) 전용 primitive의 이득이 작아 기존 forward-only 커널을 재사용하는 편이 낫다. 즉 H의 정당성은 deep-memory·NS-5의 FLOP density가 실제로 높은 config에 한정되며(→ 20장), 그 density를 재는 것이 A100 runbook의 과제 중 하나다.

> **[평가]** 제안 F·G·H는 D·E와 **경쟁하지 않고 짝을 이룬다**. decode의 state 트래픽은 memory-centric 소자(D)가 흡수하고, decode의 backward 연산과 batching 재조직은 accelerator kernel(G·H)이 맡으며, prefill/training의 compute-bound chunk는 fused kernel(F)이 처리한다. 어느 한쪽만 옹호하는 제안은 workload의 절반을 무시한다 — 이것이 18장 pair thesis의 실천적 귀결이다.

## 24.5 소프트웨어 제안: per-session-weights serving 스택

하드웨어 제안 사이에 소프트웨어 층이 놓인다. 기존 KV-cache manager의 semantics는 RMW state에 대해 **범주 오류**임이 코드 레벨에서 확인된다(claim6/E1.5, `hat-kv-manager/manager.py` 근거). 세 gap이 결정적이다: (G1) content-addressed prefix reuse — manager의 1번 가치, KV에서 0.5 hit-rate — 가 RMW state에서는 **구조적으로 0**이다(content가 token마다 변이해 block hash가 재현되지 않음). 단 이 0은 **content-addressed(block-hash) 재사용에 한정**된다 — 같은 prefix를 결정론적으로 처리한 서로 다른 요청은 분기 전까지 동일한 state snapshot에 도달하므로 prefix-snapshot 공유 + copy-on-write는 원리상 가능하며, append-only KVCacheManager가 그 mutable-state 재사용을 표현할 API를 갖지 못한 것이 바로 새 manager가 필요한 이유다; (G2) in-place update 없는 append-only placement가 T개의 stale 버전을 쌓아 256 token에서 참 working set의 **256×**로 부푼다; (G3) `_evict_from()`이 dirty state를 clean/recomputable로 가정해 **무료로 drop**하는데, 실제 TTT eviction은 writeback(또는 결정론적 prefix-replay 재계산)을 진다(E1.5의 416 MiB eviction set에서 약 3.05 mJ / 130 µs, whole-session은 footprint 비례 — NO-WALLCLOCK, A100 runbook 이월).

그 귀결로 TTT-state manager는 KVCacheManager에 없는 다섯 event type을 요구한다(claim6). 이것을 그대로 per-session-weights serving 설계로 옮긴다.

> **[평가] 제안 I — per-session-weights serving stack.** claim6의 다섯 누락 API를 설계 표면으로 삼는다.
> - **`update_in_place(slot, Δ)`** — append가 아니라 slot의 state를 제자리 RMW($W \mathrel{+}= \Delta$). 불변식: 한 sequence당 살아 있는 state 버전은 항상 1개. G2의 256× stale accretion을 원천 차단하고, 소자 D의 slot table 위에서 직접 실행된다. 비용: read+write $= 2\times$state_bytes(claim1의 대칭 RMW).
> - **`mark_dirty(slot)` / `writeback(slot)`** — 모든 TTT state는 write 직후 항상 dirty이므로 `mark_dirty`는 update의 부수효과로 자동이고, eviction은 `writeback` **또는** 결정론적 prefix-replay 재계산 중 싼 쪽을 **가격**한다(dirty이므로 unrecomputable인 것은 아니다 — 상태는 초기값·update rule·prefix의 결정적 함수다, → §23.4). G3의 unpriced drop을 명시적 비용으로 승격: writeback 비용은 상태 footprint에 비례한다(E1.5의 416 MiB eviction set에서 약 3.05 mJ / 130 µs; NO-WALLCLOCK, A100 runbook 이월). 불변식: dirty slot은 clean(writeback 또는 replay-checkpoint) 경유 없이 free될 수 없다.
> - **`checkpoint(seq) → tok` / `rollback(seq, tok)`** — speculative decoding의 rollback이 optimizer-trajectory state($W_t$**와** momentum $S_t$)까지 되돌려야 한다(→ 23장). KV의 단순 truncate와 질적으로 다르다: `checkpoint`는 $(W,S)$ 쌍의 snapshot을 뜨고 `rollback`은 그 쌍을 복원한다. 비용: snapshot당 state_bytes 사본 — checkpoint 빈도가 새 메모리 예산 축이다.
> - **`bind_to_sequence(seq) → slot` / `unbind(slot)`** — state가 sequence별 unshared 사본이므로(content-addressed prefix 공유 불가, G1) 배치는 slot↔seq bind로 표현된다. 단 결정론적 동일 prefix는 분기 전까지 같은 snapshot에 도달하므로, bind는 prefix-snapshot 공유 + 분기 시 copy-on-write를 옵션으로 허용한다. §24.4 제안 G의 grouped-GEMM이 이 bind를 batch 축으로 묶는다.
> - **`free_on_update(slot)`** — retention gate $\alpha_t$(→ 13장)는 상태를 제자리에서 감쇠시키는 **학습된 decay**이지 slot 해제 신호가 아니다(→ 1장 Rosetta): $\alpha_t\to0$이어도 활성 sequence는 다음 token의 update·retrieval을 위해 같은 고정 크기 tensor를 계속 쓴다. 따라서 이 API가 하는 일은 즉시 dealloc이 아니라, $\alpha_t\to0$으로 정보를 비운 slot을 **저비용 재구성 가능(cold)** 으로 표시해 서빙 층 eviction의 **비용 신호**로 삼는 것이다(→ §23.4의 retention-gate 협력). 물리적 free는 sequence 종료·offload·checkpoint 정책이 정하고, gate-triggered zero-page 회수는 별도 검증 대상이다. 불변식: 물리 free는 sequence lifecycle이 승인한 slot에서만 일어난다.
>
> 이 스택은 §24.3의 memory-centric 소자(state가 어디 사는가)와 §24.4의 accelerator kernel(state를 어떻게 계산하는가) 사이의 접착층이며, claim6의 semantic gap을 설계 요구사항으로 직접 번역한 것이다.

## 24.6 배제: FORCED 방향 두 가지

정직 판정(→ 18장 §18.3, dossier §5 verdict)이 **금지**한 두 memory-centric 방향을 이 장은 명시적으로 배제한다. 배제 자체가 pair thesis의 규율을 지키는 행위다.

**배제 1 — "이 라인의 훈련이 새 메모리 소자를 요구한다."** 증거는 반대 방향을 가리킨다. 여섯 편이 연속으로 알고리즘을 **dense matmul로 다시 빚어** 기존 accelerator에 맞췄다: chunk anchoring, banded mask, batched NS-5, parallelism을 위한 reset. TNT는 plain JAX로 stock TPU 위에서 FlashAttention을 이겼다(→ 15장). training/prefill은 tensor-core 영역이며 앞으로도 그렇다 — 여기서 device-level 필요성을 주장하는 것은 이 라인의 중심 엔지니어링 패턴과 모순된다. 훈련 쪽 제안은 그래서 §24.4의 accelerator kernel(F)이지 새 소자가 아니다.

**배제 2 — 일반 PIM 옹호.** per-token update는 MLP를 통한 gradient 계산으로 GEMV/GEMM 형태이지, 대부분의 PIM 제안을 정당화하는 sparse·irregular access 패턴이 아니다. PIM 형태인 것은 elementwise state-update epilogue(AXPY·decay·renorm)뿐이며, 그것은 FLOP의 소수다(§24.3.1). 따라서 PIM은 일반 해법이 아니라 epilogue 국소의 directional 옵션으로만 등장한다.

> **[평가]** 이 두 배제는 memory-centric 논증을 약화하는 것이 아니라 **신뢰 가능하게** 만든다. decode RMW 트래픽(제안 D)과 update-frequency↔tier(제안 E)라는 두 지점은 논문들 자신의 cost accounting에 뿌리내렸기에 살아남고, 훈련용 새 소자와 일반 PIM은 논문들 자신의 증거에 반하기에 배제된다. 종합 view는 이 라인이 memory-centric **이면서** accelerator-centric이라는 것이다 — 한 workload의 두 절반이기 때문이다.

## 요약

- 알고리즘 제안은 세 층이다. 두 CPU microbench가 chunk $C$-roofline 곡선(claim7)과 on/off-die RMW cliff(claim8)의 **모양**을 하드웨어 불문으로 고정했고, 이것이 제안 A(train/serve chunk-consistency grid)와 제안 B(state-capacity scaling fit)의 기대값이며, 두 실험의 vehicle은 하이브리드 4층 HOPE/Sleep 재현 runbook(제안 C)이다. E4가 공개 점 digitize로 두 부호를 이미 예열했다 — capacity는 ppl이 아니라 **long-context/retention 축**에서만 예측력을 얻고(A3를 좁힘), local-memory 수와 chunk mismatch U자가 제안 A·B의 낙하 방향을 고정한다. 여기에 제안 J(learned chunk/cadence schedule)가 고정 schedule을 학습 대상으로 올려 §24.3.2의 tier 배치와 맞물린다.
- 하드웨어 memory-centric 절반은 정직 판정의 두 지점에서만 나온다: 제안 D(RMW-bandwidth decode 소자 — per-layer residency, per-tenant placement, near-memory epilogue engine; claim1/4/8)와 제안 E(frequency-tiered placement — NL/Sleep cadence의 tier 직역; claim5).
- accelerator 절반이 그 짝이다: 제안 F(fused chunk kernel — momentum이 깨뜨린 chunkwise closed form의 미해결 커널), 제안 G(grouped-GEMM decode engine — per-request weights가 깬 batching 복원), 제안 H(backward-capable serving kernel — decode에 들어온 backward primitive).
- 소프트웨어 접착층은 제안 I(per-session-weights serving stack)로, claim6의 다섯 누락 API(`update_in_place`·`mark_dirty/writeback`·`checkpoint/rollback`·`bind_to_sequence`·`free_on_update`)를 설계 요구사항으로 번역한다.
- 두 FORCED 방향 — 훈련용 새 소자, 일반 PIM — 은 논문들 자신의 증거(알고리즘을 dense matmul로 다시 빚음)에 반하므로 명시적으로 배제한다. 흥미로운 하드웨어 제안은 어느 한쪽이 아니라 **쌍**이다.
- 모든 수치는 exploration-grade 계약 아래 읽는다: 비율·crossover·tier·bound만 load-bearing, 절대 µs/mJ은 roofline 하한(A100 runbook 이월), scratchpad/PIM 이득 배율은 directional.

## 자가 점검 체크리스트

- [ ] 알고리즘 제안 A·B가 각각 무엇을 falsify하려 하고, 어느 방향으로 떨어져도 왜 publishable인지 설명할 수 있다.
- [ ] HOPE/Sleep 재현이 왜 하이브리드 4층으로 강제되는지(공식 구현 0, Sleep 구현 0)를 진술할 수 있다.
- [ ] memory-centric 제안 D·E가 정직 판정의 어느 두 지점에 근거하는지, 그리고 각 근거 claim을 짚을 수 있다.
- [ ] accelerator 제안 F·G·H가 memory-centric 제안과 경쟁이 아니라 pair를 이루는 이유를 설명할 수 있다.
- [ ] claim6의 다섯 누락 API가 per-session-weights serving 설계로 어떻게 번역되는지 대응시킬 수 있다.
- [ ] 배제되는 두 FORCED 방향과, 그 배제가 memory-centric 논증을 왜 오히려 신뢰 가능하게 만드는지 말할 수 있다.
- [ ] E4가 capacity 축을 ppl이 아니라 long-context/retention 축으로 좁힌 근거(family 안 rank-동치, attention corner에서의 상관 반전, Titans→Atlas 도약을 설명하는 capacity-class 변화)를 설명할 수 있다.
- [ ] near-memory update engine의 세 스테이지(state-tile 상주 버퍼, epilogue ALU lane, per-tenant slot table)가 claim1/4/8과 claim6의 API에 각각 어떻게 대응하는지 말할 수 있다.
- [ ] 제안 J(learned schedule)가 왜 제안 A의 뒤에야 정직하게 돌릴 수 있는지, 그 두 falsifier가 무엇인지 진술할 수 있다.
- [ ] 어떤 제안 수치가 load-bearing 비율·crossover이고 어떤 것이 directional·이월 하한인지 판별할 수 있다.

## 다음 장으로

이 장은 pair thesis의 두 절반을 각각의 제안 — 알고리즘 실험, memory-centric 소자, accelerator kernel, serving 소프트웨어 — 으로 내렸다. 25장은 이 제안들을 하나의 research agenda로 묶고, 이 라인이 열어 둔 다섯 공백(스케일, retrieval 격차, serving 경제학·kernel, 학습되는 스케줄, task-free·안전한 self-modification, → 18장 §18.2)에 이 책의 기여가 어디에 놓이는지를 정리하며 스터디를 닫는다.
