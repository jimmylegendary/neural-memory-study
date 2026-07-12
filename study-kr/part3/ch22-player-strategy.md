# ch22. Player-strategy 분석: 누가 TTT scaling에 어떻게 참여하는가

> **이 장의 목표** — 독자가 이 장을 마치면 (1) 이 라인의 세 player — Google Research, NVIDIA, open-source kernel 생태계 — 가 각각 어떤 자산을 쥐고 있고 그 자산이 어떤 다음 수를 합리적으로 만드는지 설명할 수 있고, (2) 왜 이 연구 프로그램이 Google Research에서 나오는 것이 자산 구조상 거의 필연이었는지 논증할 수 있으며, (3) [TNT]가 custom kernel 없이 stock TPU 위 plain JAX로 FlashAttention을 이긴 사실이 player 경쟁에 대해 무엇을 예고하는지 — kernel 성숙도가 아직 아키텍처 승부를 결정하지 못했다는 것 — 을 진술할 수 있어야 한다.
> **왜 필요한가** — 21장은 hardware-lottery 독법으로 "이 라인이 왜 matmul 안에 설계되어 있는가"를 물었다(chunkwise·NS·reset이 전부 GEMM density를 사는 수다). 그 독법의 자연스러운 다음 질문은 전략적이다: 그 게임을 실제로 두는 주체는 누구이며, 각자의 자산이 서로 다른 수를 강제하는가. 이 장은 기술 분석이 아니라 그 전략 분석이다. Part III의 척추인 D4 pair thesis(→ 18장 §18.3)는 여기서 player 지도로 번역된다 — decode 반쪽(memory-centric)과 training/prefill 반쪽(accelerator)에 각 player의 자산이 비대칭적으로 걸려 있기 때문이다.

## 22.1 Bridge-in: hardware lottery에서 player 지도로

21장의 결론은 서술적이었다: attention은 GEMM density로 hardware lottery를 이겼고, 이 라인은 그 교훈을 내면화해 자기 알고리즘을 dense matmul로 다시 빚었다(chunkwise anchoring으로 exactness를 GEMM으로 바꾸고, Newton–Schulz로 orthogonalization을 matmul로 바꾸고, reset으로 non-linear recurrence를 병렬 shard로 바꾼다, → 9장·14장·15장). 이 장은 그 서술을 행위자에게 귀속시킨다. 누가 이 수들을 두었고, 누가 다음 수를 둘 자산을 쥐고 있는가.

이 장의 대부분은 논문이 보고한 사실이 아니라 이 책의 전략적 판단이다. 따라서 §6.2 규칙에 따라 판단은 **[평가]** 블록으로 명시하고, 사실 — 어느 스택에서 무엇이 측정되었는가 — 만 산문으로 단정한다. 정직하게 밝혀 둔다: player-strategy 분석은 falsifiable한 실험 주장이 아니라 자산 구조에서 다음 수를 읽는 framing이며, 그 자체로는 검증 대상이 아니다. 이 장의 무게는 근거가 되는 사실(누가 어떤 하드웨어에서 무엇을 냈는가, crossover가 어디인가)의 단단함에서 나오지, 예측의 확실성에서 나오지 않는다.

## 22.2 Google Research가 자연스러운 저자인 이유

여섯 편은 모두 한 조직에서 나왔다(Behrouz et al., Google Research, 일부 USC 공저). 이것은 우연이 아니라 자산 구조의 귀결이다. 이 라인을 검증하려면 세 자산이 한 지붕 아래 있어야 하고, 2024–2026 시점에 그 셋을 동시에 쥔 곳은 사실상 Google Research뿐이었다.

**자산 1 — TPU pod + JAX/XLA 스택.** 이 라인의 핵심 도박은 "non-linear deep-memory recurrence를 dense matmul로 다시 빚으면 기존 accelerator에서 경쟁력이 난다"이다. 그 도박이 참인지는 large-GEMM에 최적화된 하드웨어와, custom kernel 없이도 그 GEMM을 잘 뽑는 컴파일러가 있어야 검증된다. TPU의 systolic array는 큰 dense GEMM에 특화된 소자이고, XLA는 그 위로 JAX 프로그램을 융합·스케줄한다. chunk 크기 $C$를 키우면 같은 알고리즘이 memory-bound에서 compute-bound로 오른다는 것(claim7, → 15장·9장)이 이 스택에서 곧바로 throughput으로 환원된다 — $C$가 roofline의 x축이고, TPU+XLA가 그 x축의 오른쪽(큰 $C$)을 값싸게 만든다.

<!-- FIG: exp-c -->
그림 22-1 — chunk 크기 $C$가 roofline의 x축이다: $C{=}1$(per-token RMW = decode 영역)은 memory-bound 평원에 있고 $C$를 키우면 crossover를 넘어 compute-bound로 오른다. TPU+XLA 스택의 자산은 이 곡선의 오른쪽 구간(대형 dense GEMM)을 custom kernel 없이 값싸게 만든다는 데 있다(실험 E2.1 재구성; host ridge는 H100 twin ridge의 약 1/9이므로 곡선의 **모양**만 이전, 측정 $C^*{\approx}32$는 host 값 — H100 closed-form은 306–430).

이 자산의 결정적 증거가 [TNT]다. 150M Titans를 10B tokens로 TPUv5 pod 위 plain JAX(**custom kernel 없이**)로 훈련한 결과, 가장 정확한 Titans baseline(C=64) 대비 target loss까지 **17.4× 빠르면서** 평균 perplexity를 오히려 개선했고(23.09 vs 25.07), 32K 문맥에서는 $C_L{=}128$의 순수 JAX TNT가 FlashAttention(Pallas kernel)보다 step당 **1.3× 빠르다** [TNT §experiments]. 이 사실의 함의는 §22.4·§22.6에서 되짚지만, Google 자산 논증에 국한하면 명료하다: **CUDA/cuDNN moat를 우회한 채로도 이 라인이 성립함을 Google 스택이 자기 하드웨어 위에서 이미 보였다.** 아키텍처를 matmul로 빚어 두면 컴파일러가 kernel 성숙도의 공백을 상당 부분 메운다.

**자산 2 — long-context 제품.** decode 절반의 경제학은 crossover 문맥 길이 $S^*$에 걸려 있다. KV cache 읽기 트래픽은 문맥에 비례해 자라지만 TTT state의 RMW 트래픽은 문맥에 무관하게 일정하므로, 그 아래에서는 KV가 싸고 그 위에서는 TTT가 싼 crossover가 존재한다(claim2, → 18장 §18.3). anchor에서 read-crossover는 약 65k token, 전체 RMW-crossover는 약 131k token이며, 폭에 따라 16k→1.05M token으로 이동한다.

<!-- FIG: exp-a -->
그림 22-2 — KV(문맥에 비례해 자라는 read)와 TTT(문맥에 무관한 RMW)의 트래픽 crossover $S^*$. TTT의 경제적 이점이 나타나는 문맥 길이대(약 65k–131k token 이상)가, 정확히 long-context 제품이 판매하는 영역이다(실험 E1.3/E3 재구성; crossover **위치**만 load-bearing, 절대 µs/token은 roofline 하한으로 이월).

> **[평가]** crossover가 놓인 자리가 전략적으로 결정적이다. TTT가 KV보다 유리해지는 문맥 길이는 대략 수만~수십만 token 이상인데, 이것은 정확히 Google이 제품으로 파는 regime(장문맥 Gemini 계열)이다. 즉 이 라인의 경제적 이점이 발현되는 지점과 Google 제품 라인의 판매 지점이 겹친다. long-context를 파는 조직만이 이 라인을 자기 제품 곡선 위에서 정당화할 수 있다 — 짧은 문맥만 서빙하는 곳에는 crossover 아래라 KV cache가 여전히 싸고, 이 라인을 도입할 사업적 이유가 약하다.

**자산 3 — 연구팀과 저작권.** 여섯 편의 개념 소유권(Titans의 deep memory, Miras의 taxonomy, Atlas의 capacity 이론, TNT의 training 경제학, NL의 nested ontology, Sleep의 lifecycle)이 한 팀에 누적되어 있다. 후속 수의 설계 지식이 조직 내부에 있다는 것 자체가 재현 비용이 큰 자산이다.

> **[평가]** 세 자산은 곱셈적이다. TPU+XLA만으로는 이 라인을 팔 제품이 없고, long-context 제품만으로는 도박을 검증할 스택이 없으며, 연구팀만으로는 검증도 배포도 못 한다. 셋이 한 지붕 아래 있어야 "chunkwise-into-matmul 도박 → 자기 하드웨어에서 검증 → 자기 장문맥 제품에 배포"의 폐루프가 돈다. 그래서 이 프로그램이 Google Research에서 나온 것은 취향이 아니라 자산 구조의 귀결이며, 저자이자 자연스러운 첫 배포자라는 지위가 여기서 나온다.

## 22.3 Google이 합리적으로 두는 다음 수

자산이 다음 수를 강제한다. Google의 합리적 세 수는 모두 자기 자산에 정렬된다.

**수 1 — deep-memory fused kernel.** [TNT]는 custom kernel 없이도 이겼지만, 동시에 kernel-optimized Gated Transformer에는 time-to-loss에서 아직 진다(0.96h vs 1.12h)고 스스로 보고하고, fused kernel을 명시적으로 미래 과제로 남긴다 [TNT §systems]. 그 kernel의 본진이 decode다: claim1대로 decode step은 fast-weight state 전체의 memory-bound RMW(anchor에서 6.44 GB/token, arithmetic intensity 0.59 FLOP/byte로 결정적 memory-bound, state 트래픽이 GEMV 연산을 약 394× 압도)이므로, per-chunk의 batched forward+backward와 cumulative-sum state update를 하나로 융합하는 kernel이 남은 격차를 닫을 자리다. 이 수는 TPU 자산에 정확히 얹힌다.

**수 2 — scale up.** 라인 전체의 from-scratch 실증 상한은 1.3B params / 100B tokens이고([TNT]는 150M), chunk-size mismatch 현상조차 550M 단일 그림에서만 관측되었다(§6.4 의무 caveat, → 15장). [Titans]가 2024-12에 약속한 "larger models"는 끝내 배달되지 않았다. momentum·deep-memory·self-modification의 이점이 7B+·SFT/RLHF·production 데이터에서 유지되는지는 이 라인 최대의 미지수다(→ 18장 §18.2 유보 1). 이 규모의 from-scratch 검증을 값싸게 돌릴 수 있는 자산은 소수의 조직에만 있고, Google은 그중 하나다.

**수 3 — hybrid로 제품에 삽입.** in-context retrieval 격차가 측정된 채 닫히지 않았다(attention 53.55 vs 43.70 [Atlas Table 5], FDA 67.3 vs 41.9 [NL], → 14장·16장). 따라서 합리적 배포는 순수 TTT가 아니라 attention과의 MAG/MAC hybrid이며(→ 12장), crossover 위 문맥에서만 TTT 경로를 켜는 adaptive 배치다 — crossover 아래에서는 KV가 싸므로 TTT를 켤 이유가 없다(claim2). 이 수는 자산 2(long-context 제품)에 얹힌다.

> **[평가]** 세 수가 모두 Google 자산에 정렬된다는 사실이 §22.2의 "자연스러운 저자" 논지를 완성한다. 저자이자 첫 배포자일 뿐 아니라, **다음 수를 값싸게 둘 수 있는 유일한 위치**에 가깝다. 다만 이 정렬은 Google이 반드시 둔다는 예측이 아니라, 두면 다른 누구보다 저비용이라는 자산 판정이다.

## 22.4 NVIDIA의 위치: Gated DeltaNet 계보와 kernel moat

NVIDIA는 이 라인(deep memory)의 저자가 아니지만, 인접 계보인 linear memory의 주요 player다. 그 접점이 Gated DeltaNet(이하 GDN, Yang et al. 2024)이다. GDN은 delta rule에 retention을 더한 **linear-state** 모델로(§1.6 카탈로그: $W_t=\alpha_tW_{t-1}(I-\eta_tk_tk_t^\top)+\eta_tv_tk_t^\top$), 이 라인의 deep memory가 특수 경우로 흡수한다고 [Titans]가 자기 Appendix C에서 보인 바로 그 계보에 있다.
<!-- TODO-VERIFY: Gated DeltaNet의 정확한 arXiv ID와 저자 소속(NVIDIA) 확정 필요. 확인 방법: GDN 원문 arXiv 페이지 저자 affiliation 확인 후 §2.5 형식(저자-연도+arXiv ID)으로 병기. dossier §4 ch22가 "NVIDIA's position (Gated DeltaNet lineage)"로 지정한 framing 근거. -->

NVIDIA의 자산은 두 가지다: GPU의 CUDA/cuDNN moat, 그리고 linear-attention kernel의 성숙도. 그러나 여기 전략적 함정이 있다. **linear-attention SRAM chunk kernel(GLA/DeltaNet/GDN 계보)은 deep memory에 port되지 않는다** — inter-chunk state propagation이 non-linearity(MLP forward/backward, LayerNorm)를 통과하기 때문이며, parallel scan이 적용되지 않는 바로 그 이유다 [TNT §systems]. 즉 NVIDIA가 성숙시킨 kernel 자산은 linear 쪽에서만 통하고, deep-memory 쪽에서는 TNT의 reset trick이 현재 유일한 지렛대다.

> **[평가]** NVIDIA의 합리적 수는 자산 방어에서 나온다. **수 A — "linear로 충분하다"를 논증.** crossover 아래(대략 <65k token)에서는 KV/linear가 이기므로(claim2), 대부분의 제품 문맥이 그 아래에 있는 한 GDN 계보로 충분하다는 방어가 성립한다. NVIDIA의 이익은 아키텍처가 linear에 머무를 때 커진다 — 그쪽에 성숙한 kernel이 이미 있기 때문이다. **수 B — deep memory가 이기면 kernel로 재진입.** 만약 deep memory가 crossover 위 제품에서 승기를 잡으면, NVIDIA는 아키텍처 승부가 아니라 deep-memory fused kernel을 CUDA로 내는 kernel 승부에서 이기려 할 것이다. 어느 쪽이든 NVIDIA의 승부처는 아키텍처가 아니라 kernel/하드웨어다.

그런데 이 방어를 정확히 겨냥해 흔드는 것이 §22.2의 TNT 사실이다. TNT가 stock TPU 위 plain JAX로, 즉 **custom kernel 없이** 32K에서 FlashAttention을 이겼다는 것은, 이 라인에서 CUDA moat가 예상보다 약하다는 신호다. 아키텍처가 dense matmul로 빚어져 있으면 XLA 같은 컴파일러가 kernel 우위의 상당 부분을 상쇄한다. NVIDIA에게 이것은 "성숙한 CUDA kernel = 지속 우위"라는 등식이 이 라인에서는 자동으로 성립하지 않는다는 위협이다.

## 22.5 Open-source kernel 생태계: flash-linear-attention

세 번째 player는 조직이 아니라 생태계다. flash-linear-attention(fla)은 GLA·DeltaNet·GDN·RWKV 등 linear-memory 모델의 Triton kernel을 모은, 이 계보의 사실상 kernel 인프라다. 그러나 fla 역시 NVIDIA 자산과 같은 함정 위에 있다: **linear state를 전제**하므로, deep memory(TTT/Titans/Atlas)의 non-linear recurrence에는 그대로 통하지 않는다. deep-memory fused kernel은 open-source 쪽에서도 아직 공백이며, 이 공백이 21장이 말한 "이 가족이 아직 기다리는 FlashAttention-moment"의 kernel 측면이다.

공백은 kernel에 그치지 않는다. serving 스택도 RMW state를 관리하지 못한다. 기존 KV-cache manager의 semantics — content-addressed prefix 재사용, append-only placement — 는 RMW state에 대해 범주 오류다(claim6, → 23장): 재사용률이 구조적으로 0이고(내용이 매 token 바뀌어 block hash가 재발하지 않는다), append-only가 stale 사본을 누적하며, dirty writeback이 값매김되지 않는다. TTT-state manager는 `update_in_place`·`mark_dirty/writeback`·`checkpoint/rollback`·`bind_to_sequence` 같은, KVCacheManager에 없는 event type을 필요로 한다.

> **[평가]** open-source의 합리적 수는 이 두 공백을 먼저 메우는 것이다. **수 A — fla를 deep memory로 확장.** deep-memory용 Triton fused forward+backward chunk kernel(per-chunk batched fwd/bwd와 cumulative-sum state update의 융합)을 먼저 내는 팀이, 이 라인에서 linear-attention 시대의 FlashAttention이 그랬던 역할 — 사실상의 kernel 표준 — 을 차지한다. **수 B — serving 인프라에 RMW API를 추가.** vLLM-class 스택에 claim6의 missing API를 얹는 팀이 decode 반쪽의 소프트웨어 열쇠를 쥔다. 두 수 모두 하드웨어 자산 없이 둘 수 있고, 그래서 open-source가 진입 가능한 지점이다.

open-source가 왜 구조적으로 중요한가. Google이 논문으로 아키텍처를 공개해도 TPU kernel과 내부 serving 스택은 열지 않는다. GPU 세계에서 이 라인을 돌리려면 그 공백을 누군가 메워야 하고, 역사적으로 그 역할은 open-source kernel 생태계가 맡아 왔다(linear-attention에서 fla가 그랬듯). 즉 Google이 아키텍처를 열수록 open-source의 기회는 커진다.

## 22.6 세 player의 균형과 pair thesis

player 지도를 Part III의 척추인 D4 pair thesis 위에 겹치면 비대칭이 선명해진다. decode 반쪽(memory-centric: unshared·write-heavy·비-content-addressable한 whole-state RMW)의 승부는 serving 인프라와 RMW-aware 소프트웨어에서 나고(claim1·claim2·claim6), training/prefill 반쪽(accelerator: chunk $C$ = roofline x축)의 승부는 kernel과 대형 GEMM 스택에서 난다(claim7).

이 비대칭 위에 세 player를 놓으면, **어느 한 player도 pair의 양쪽을 다 쥐지 못한다.**

| player | 강한 자산 | pair 상 위치 | 합리적 다음 수 |
|---|---|---|---|
| Google Research | TPU+XLA, long-context 제품, 저작권 | training 반쪽(강)·아키텍처 저자 | fused deep-memory kernel; scale up; hybrid 제품 배포 |
| NVIDIA | CUDA moat, linear kernel 성숙도 | accelerator 하드웨어(강)·아키텍처 비저자 | "linear로 충분" 방어; deep-memory 이기면 CUDA kernel 재진입 |
| open-source(fla 등) | Triton kernel·serving 커뮤니티 | decode 반쪽 소프트웨어(진입 가능) | fla를 deep memory로 확장; serving에 RMW API 추가 |

표 22-1 — 세 player의 자산·pair 상 위치·합리적 다음 수.

Google은 training 반쪽과 아키텍처에 강하지만 GPU serving 생태계 밖에 있다. NVIDIA는 하드웨어·kernel에 강하지만 아키텍처의 저자가 아니고 그 자산이 linear 계보에 묶여 있다. open-source는 decode 반쪽의 소프트웨어 공백을 메울 수 있지만 하드웨어가 없다. pair thesis가 예측하는 것은 단일 승자가 아니라 **분업**이다 — 각 player가 자기 자산이 걸린 절반에서 다음 수를 두고, 완성형(→ 18장)이 배포되려면 세 절반의 수가 모두 놓여야 한다.

마지막으로 §22.2·§22.4에서 미룬 TNT-beats-FlashAttention 사실의 가장 넓은 함의를 짚는다. 성숙한 kernel(수년간 최적화된 FlashAttention)이 미성숙 경쟁자(custom kernel 없는 plain-JAX TNT)에게 32K에서 진다는 것은, 이 라인의 승부가 **아직 kernel이 성숙하기 전 국면**에 있다는 뜻이다.

> **[평가]** hardware lottery의 통상 독법은 "성숙한 kernel을 가진 연산이 이긴다"이다(21장). 그러나 TNT의 사실은 그 등식이 이 라인에서 아직 잠겨 있지 않음을 보인다 — kernel 우위가 아직 아키텍처 승부를 결정하지 못했다. 이것이 세 player 모두에게 뜻하는 바는 하나다: **FlashAttention-moment가 오기 전, 지금이 진입 기회의 창이다.** deep-memory fused kernel을 먼저 내는 player(Google·NVIDIA·open-source 누구든)가 이 라인의 kernel 표준을 정의하고, 그 순간 창은 닫히기 시작한다. 이 라인의 player 경쟁이 "누가 그 kernel을 먼저 성숙시키는가"의 경주인 이유가 여기 있다.

## 요약

- 이 라인이 Google Research에서 나온 것은 자산 구조의 귀결이다: TPU+XLA 스택, long-context 제품, 여섯 편의 저작권이 곱셈적으로 결합해 "chunkwise-into-matmul 도박 → 자기 하드웨어 검증 → 자기 장문맥 제품 배포"의 폐루프를 돌릴 수 있는 유일한 위치에 가깝다.
- Google의 합리적 다음 수 셋(deep-memory fused kernel, scale up, hybrid 제품 배포)은 모두 자기 자산에 정렬된다 — 저자이자 다음 수를 값싸게 둘 위치.
- NVIDIA는 deep memory의 저자가 아니라 인접 linear 계보(Gated DeltaNet, 이하 GDN)의 player이며, 그 kernel 자산은 linear에만 통한다(deep memory의 non-linear recurrence에 SRAM chunk kernel이 port되지 않음). 합리적 수는 "linear로 충분" 방어이거나, 지면 CUDA deep-memory kernel로 재진입.
- open-source kernel 생태계(flash-linear-attention)도 같은 함정 위에 있어 deep-memory fused kernel과 RMW-aware serving API가 공백이다 — 먼저 메우는 팀이 이 라인의 kernel/serving 표준을 쥔다.
- pair thesis 위에 겹치면 어느 player도 양쪽(decode 소프트웨어 + accelerator kernel)을 다 쥐지 못하는 분업 구조가 드러나며, 완성형 배포는 세 절반의 수가 모두 놓여야 성립한다.
- [TNT]가 custom kernel 없이 stock TPU에서 FlashAttention을 이긴 사실의 함의: kernel 우위가 아직 아키텍처 승부를 결정하지 못했다 — FlashAttention-moment 이전인 지금이 진입 기회의 창이다.

## 자가 점검 체크리스트

- [ ] Google Research를 자연스러운 저자로 만드는 세 자산과 그것이 곱셈적으로 결합하는 이유를 설명할 수 있다.
- [ ] crossover $S^*$의 위치가 왜 long-context 제품 자산과 겹치는지 설명할 수 있다.
- [ ] NVIDIA의 kernel 자산이 linear 계보에만 통하고 deep memory에 port되지 않는 이유(non-linear recurrence)를 말할 수 있다.
- [ ] open-source 생태계의 두 공백(deep-memory fused kernel, RMW-aware serving API)을 짚을 수 있다.
- [ ] 세 player를 pair thesis 위에 겹쳤을 때 왜 단일 승자가 아니라 분업이 예측되는지 논증할 수 있다.
- [ ] TNT-beats-FlashAttention이 왜 "진입 기회의 창"을 뜻하는지 진술할 수 있다.
- [ ] player 지도를 inference 어휘(누가 kernel을 쥐는가, 누가 serving 스택을 쥐는가)로 옮길 수 있다.

## 다음 장으로

이 장은 각 player의 합리적 다음 수를 자산에서 읽었다. 그 수들 — 특히 Google의 scale up과 decode 반쪽의 serving 인프라 — 이 실제로 놓이면 무엇이 필요한가. 23장은 그 요구를 정량으로 내린다: TNT 경제학을 7–70B로 외삽하고, 계층적 memory의 prefill/decode 분할을 그리며, per-session weight state를 새 cache class로 설계하고(sizing·checkpoint/restore·multi-tenant batching — claim 3의 $B_{\max}$가 근거), sleep을 fleet 수준 background job으로 놓아, 오늘의 KV-cache 서빙과 비용 모델로 대조한다.
