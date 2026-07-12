# ch19. Scaling 분석과 memory-centric 모델의 후보 scaling law

> **이 장의 목표** — 독자가 이 장을 마치면 (1) 여섯 편이 공개한 ppl-vs-{params, tokens, context} 점을 하나의 표로 digitize하고 그 비교가 어디까지 정당한지(tokenizer 이질성) 판별할 수 있고, (2) 왜 params·tokens 두 축만으로는 이 라인의 scaling을 기술할 수 없는지 — **state-bytes**와 **capacity**($O(d_k^p)$)가 빠진 두 개의 독립 축임을 — 논증할 수 있으며, (3) decode 비용·crossover·quality 각각에 대한 후보 scaling law의 형태를 진술하고, capacity 축이 raw state size 대비 예측력을 더하는지에 대한 **A3 falsification 조건**을 명시할 수 있어야 한다.
> **왜 필요한가** — 18장은 완성형을 workload로, workload를 D4 pair thesis로 내렸고, Part III를 일곱 장으로 폈다. 이 장은 그 첫 축이다. scaling law는 inference 엔지니어가 fleet을 sizing할 때 쓰는 도구인데, 이 라인은 여섯 편에 걸쳐 quality scaling을 단 한 번도 law로 fit하지 않았다(dossier §3.2 공백 1: "스케일"). 이 장은 그 빈자리에 두 개의 새 축을 세우고, 그 축들이 pair thesis의 decode 절반과 어떻게 맞물리는지를 실측 claim으로 뒷받침한다.

## 19.1 Bridge-in: scaling law는 왜 두 축이 모자란가

18장의 [평가]는 이 라인의 배포를 "세션 상태가 per-session mutable weights인 continually-learning LLM"으로 읽었다. inference 엔지니어가 그런 시스템의 fleet을 계획하려면 익숙한 두 질문에 답해야 한다: **주어진 예산에서 어떤 품질이 나오는가**(quality scaling), 그리고 **주어진 품질을 서빙하는 데 자원이 얼마나 드는가**(cost scaling). transformer 세계에서 두 질문의 도구는 각각 Chinchilla식 법칙 $\mathrm{L}(N,D)$(Hoffmann et al. 2022, arXiv:2203.15556)과 KV cache sizing이다 — 둘 다 params $N$과 tokens $D$라는 두 축 위에 산다.

이 라인은 그 두 축을 깨뜨린다. 정확히는, 두 축을 **불충분하게** 만든다. 이유는 18장에서 이미 나왔다: state가 KV cache가 아니라 weights이고(완성형 성분 1), 그 weights의 크기가 params와 **분리된** 독립 knob이며(§18.3의 decode 절반), 같은 $(N, D)$ 아래서도 architecture class가 바뀌면 memory **capacity**가 $O(d_k)$에서 무한대까지 움직인다(→ 14장의 $\phi^*$). 즉 params·tokens는 quality의 일부만, cost의 일부만 설명한다. 이 장의 명제는 하나다: **memory-centric 모델의 scaling law는 params·tokens 위에 state-bytes와 capacity라는 두 개의 일급 축을 얹어야 완결된다.** 그리고 그 두 축이 정확히 pair thesis의 decode 절반이 사는 곳이다.

## 19.2 여섯 편의 공개 scaling 점을 digitize한다

먼저 재료를 모은다. 여섯 편이 본문·표에 공개한, 서로 다른 스케일에서의 품질 점을 하나의 표로 옮긴다(표 19-1). 원문 수치이므로 출처를 병기하고, 반올림하지 않는다.

표 19-1 — 여섯 편의 공개 품질 점(from-scratch LM). ppl은 Wiki / LAMBADA, acc는 common-sense 평균. tokenizer 열이 비교 가능성의 관문이다.

| 모델 | 스케일 | tokens | tokenizer | Wiki ppl | avg acc | 출처 |
|---|---|---|---|---|---|---|
| Titans (LMM) | 760M | 30B | Llama-2 | 20.04 | 51.56 | [Titans Table 1] |
| Titans (LMM) | 340M | 15B | Llama-2 | — | 46.17 | [Titans Table 1] |
| Atlas | 1.3B | 100B | T5-32K | 14.97 | 57.62 | [Atlas] |
| Atlas++ | 1.3B | 100B | T5-32K | 14.40 | 58.03 | [Atlas] |
| Titans (재실행) | 1.3B | 100B | T5-32K | 15.60 | 56.82 | [Atlas] |
| Gated DeltaNet | 1.3B | 100B | T5-32K | 16.42 | 55.32 | [Atlas] |
| Transformer++ | 1.3B | 100B | T5-32K | 18.53 | 52.25 | [Atlas] |
| Atlas | 760M | 30B | T5-32K | 19.97 | 52.77 | [Atlas Table 6] |
| Hope | 1.3B | 100B | 32K | 14.39 | 58.04 | [NL] |
| Titans (재실행) | 1.3B | 100B | 32K | — | 56.82 | [NL] |
| Hope | 760M | 30B | 32K | 18.68 | 52.28 | [NL] |
| TNT (best) | 150M | 10B | T5-32K | — | ~41.0 | [TNT Table 2] |

이 표를 세로로 읽으면 스케일에 따른 단조 개선이 드러난다: Atlas의 avg acc는 760M 52.77 → 1.3B 57.62로, Hope는 760M 52.28 → 1.3B 58.04로 오른다. [NL]은 "gains grow with scale"을 명시하고, [Atlas Fig. 8]은 params와 training context 양쪽에서 baseline 대비 우호적 scaling을 보고한다 — **이 라인이 공개한 유일한 scaling-law 모양의 증거다.**

> **[평가]** 표 19-1은 세로로는 읽히지만 가로로는 함부로 읽히지 않는다. tokenizer 열이 이유다. Titans 원 논문의 점은 **Llama-2** tokenizer로, Atlas·TNT·NL의 **T5/32K** 점과 같은 척도의 perplexity가 아니다 — vocab이 다르면 per-token cross-entropy의 분모가 달라 ppl이 직접 비교 불가다. 따라서 "Titans 760M 20.04가 Atlas 1.3B 14.97보다 나쁘다"는 스케일 차이와 tokenizer 차이가 뒤섞인 무의미한 비교다. **비교가 정당한 곳은 두 군데뿐이다.** (a) 한 논문 안에서 자기가 같은 protocol로 재실행한 baseline과의 대조(Atlas가 재실행한 Titans 1.3B 15.60 vs Atlas 14.97). (b) tokenizer·데이터가 정렬된 논문 간 대조 — 그리고 여기 한 개의 단단한 앵커가 있다: Atlas와 NL이 각각 재실행한 **Titans 1.3B의 avg acc가 56.82로 정확히 일치**한다(FineWeb 계열 + 32K vocab). 이 일치는 두 논문의 1.3B 점을 같은 축 위에 놓을 수 있게 해 주며, 그 위에서 Atlas++ 58.03과 Hope 58.04가 사실상 동률로 Titans 56.82를 넘는다는 **cross-architecture ordering**은 load-bearing하다. 넘어서는 안 되는 선은 이 ordering을 fitted exponent로 승격하는 것이다(§19.6).

## 19.3 빠진 축 하나: state-bytes

params·tokens가 놓치는 첫 번째 축은 cost 쪽에 있다. transformer에서 서빙 상태(KV cache)는 $2\,L\,d$로 **문맥 길이** $L$에 비례해 자라고 params와는 다른 축이다 — 엔지니어는 이 사실에 익숙하다. memory-centric 모델에서 서빙 상태는 **fast-weight state**이고, 그 크기는 문맥에 무관하며 다음으로 고정된다:

$$
B_{\mathrm{state}} \;=\; m\,d^2\,L_{\mathrm{layer}}\cdot s
\tag{19-1}
$$

여기서 **state multiplier** $m$은 표준 deep memory의 상태 배수(§1.6; 2-layer MLP이면 $8d^2$, momentum 포함이면 $16d^2$), $d$는 model 폭, $L_{\mathrm{layer}}$는 layer 수(sequence 길이 $L$이 **아님**), $s$는 원소당 바이트다. anchor `neural-mem-1.3B`($d{=}2048$, $m{=}16$, $L_{\mathrm{layer}}{=}24$, bf16)에서 layer당 $16\cdot2048^2\cdot2 = 134\ \mathrm{MB}$, 전체 $B_{\mathrm{state}} \approx 3.2\ \mathrm{GB}$다(실험 E1.1/E3, 세 방법이 layer당 134 MB로 1% 이내 일치).

이 $B_{\mathrm{state}}$가 왜 일급 scaling 축인가. 18장 claim 1이 답했다: decode step은 token마다 이 상태 전체를 읽고 되쓰므로(RMW), 이동 트래픽이 $2B_{\mathrm{state}}$이고, 그것이 decode step의 GEMV 연산을 약 394× 압도한다 — **step 비용이 곧 state 트래픽이다.** 그러므로 (19-1)은 단순한 memory 회계가 아니라 decode **cost의 scaling 축**이다. 그 축을 따라 트래픽이 어떻게 자라는지가 표 19-2다.

표 19-2 — RMW 트래픽(read + write, per token)의 폭 scaling. 문맥 길이에 **무관**하고 모델 폭에 $\propto m\,d^2 L_{\mathrm{layer}}$로 자란다(실험 E3, tag: REL·M-ASSUMED).

| 스케일 | Titans-170M | Titans-340M | Titans-760M | neural-mem-1.3B | hypo-7B | hypo-70B |
|---|---|---|---|---|---|---|
| RMW GB/token | 0.45 | 1.61 | 3.62 | 6.44 | 34.36 | 343.6 |

<!-- FIG: exp-a -->

![그림 19-1 — KV vs TTT 트래픽 crossover와 그 scaling: 문맥에 따라 자라는 KV 읽기 vs 폭에 따라 자라는 TTT RMW, 그리고 crossover 문맥 $S^*$의 스케일 이동](../../figures/exp-a-kv-ttt-crossover.png)

그림 19-1 — KV cache 읽기 트래픽은 문맥 길이에 비례해 자라고 TTT state RMW는 문맥에 무관하게 고정이므로 crossover 문맥 $S^*$가 존재한다. 오른쪽 panel의 $S^*$ 스케일링(모델 폭에 따른 이동)이 이 장의 load-bearing 결과다(실험 E1.3/E3 재구성).

> **[해설]** state-bytes가 memory-centric 논증이 정직하게 서는 두 지점 중 첫째(§18.3의 decode RMW 트래픽)와 정확히 겹친다는 점이 중요하다. (19-1)은 "이 모델이 새 메모리 소자를 요구한다"는 주장이 **아니다** — 그것은 dossier가 FORCED로 금지한 지점이다. (19-1)이 말하는 것은 좁고 방어 가능하다: 서빙 상태의 크기가 문맥이 아니라 폭$^2$에 스케일하고, 그 상태가 매 token RMW되므로, decode cost의 scaling 변수는 tokens $D$가 아니라 $B_{\mathrm{state}} \propto d^2 L_{\mathrm{layer}}$라는 것이다. 이것이 params·tokens 축계에 없던 세 번째 sizing 숫자다.

## 19.4 빠진 축 둘: capacity를 일급 축으로

두 번째 빠진 축은 quality 쪽에 있고, 이론은 이미 14장에 있다. [Atlas]의 capacity 정리는 memory가 정확히 저장할 수 있는 linearly-independent (key, value) 쌍의 최대 수를 architecture class별로 준다: matrix memory는 $O(d_k)$(Prop. 1), deep MLP는 subquadratic(Thm. 1), degree-$p$ polynomial feature는 $O(d_k^p)$(Prop. 2), 그리고 exponential map $\phi^*$의 softmax attention은 **무한**([Atlas §3]). 이 사다리는 architecture를 quality 잠재력 순으로 **정렬**한다 — params나 state-bytes로는 보이지 않는 순서다.

**capacity를 params·tokens·state-bytes와 나란한 네 번째 축으로 제안한다.** 근거는 이것이 두 관측을 동시에 설명하기 때문이다. 첫째, 미해소 retrieval 격차(§18.2 유보 2): attention이 in-context recall에서 앞선다(53.55 vs 43.70 [Atlas Table 5]; FDA 67.3 vs 41.9 [NL]). capacity 사다리가 그 이유를 준다 — $\phi^*$의 무한 capacity 대 fixed-state의 유한 capacity. 둘째, feature map을 켜면 recall이 오른다는 ablation([Atlas]의 "w/o Polynomial Mapping" 22.14 → full 19.97 ppl)은 capacity 축을 따라 움직인 것이다.

핵심 질문은 A3의 질문이다: **capacity는 raw state-bytes 대비 예측력을 더하는가?** 둘은 상관은 있지만 동일하지 않다. degree-$p$ feature는 capacity를 $O(d_k^p)$로 사지만 그 대가를 memory MLP 첫 layer의 폭 — 즉 state-bytes — 으로 치른다([Atlas], key 차원이 $\sim d_k^p$로 부풀어 $B_{\mathrm{state}}$를 키운다). 따라서 **capacity-per-byte**는 architecture class마다 다르다. 만약 두 memory가 같은 $B_{\mathrm{state}}$인데 하나는 poly feature로 capacity가 높다면, capacity는 state-bytes가 놓치는 무언가를 잡는 것이고 — 그때만 — 일급 축 자격이 있다.

> **[평가]** 이 제안에는 정직하게 붙일 별표가 하나 있다. [Atlas]의 capacity는 **exact interpolation**(linearly independent key의 zero inner loss, piecewise-affine 가정)의 이상화된 proxy이지, 측정된 retrieval 품질이 아니다([Atlas assumptions_and_scope caveat 1]). 즉 capacity 축은 **이론적으로** 근거가 있고(정리), state-bytes 축은 **경험적으로** 근거가 있다(claim 1의 측정된 decode cost). 이 비대칭이 §19.6의 falsification을 필요하게 만든다.

## 19.5 후보 scaling law의 형태

이제 세 축(quality, decode cost, crossover)에 대한 후보 법칙을 **형태로** 진술한다. exponent를 fit하지 않는 이유는 §19.6에서 밝힌다 — 여기서 제안하는 것은 함수 형태와, 그 안에서 실측이 승격한 scaling **모양**이다.

**(a) Quality law(제안, 미fit).** Chinchilla 형태에 class 항을 더한다:

$$
\mathrm{ppl}(N, D;\ \mathrm{class}) \;\approx\; E_{\mathrm{class}} + \frac{A}{N^{\alpha}} + \frac{B}{D^{\beta}}
\tag{19-2}
$$

여기서 architecture class(capacity $O(d_k)$ / $O(d_k^p)$ / 무한)가 무한-capacity 극한 손실 $E_{\mathrm{class}}$를, 그리고 어쩌면 exponent를 설정한다. 표 19-1의 세로 개선(52.28→58.04 등)은 (19-2)의 $N$ 의존성과 정합하지만, 이 정합은 **ordering 수준**이지 fit이 아니다.

**(b) Decode cost law(하한).** state-bytes 축의 직접 귀결이다. decode는 memory-bound RMW이므로(claim 1):

$$
t_{\mathrm{dec}} \;\gtrsim\; \frac{2\,B_{\mathrm{state}}}{\mathrm{BW}} \;=\; \frac{2\,m\,d^2 L_{\mathrm{layer}}\,s}{\mathrm{BW}}
\tag{19-3}
$$

즉 per-token decode 비용은 tokens $D$에 **무관**하고 폭$^2$에 스케일한다 — quality law (19-2)와 **다른 축**을 따른다. anchor에서 이 하한은 1.92 ms/token(HBM3 3.35 TB/s)이지만, 이 절대값은 roofline **하한**이고 A100 runbook으로 이월된다(§18.6). 본문이 딛는 것은 우변의 **구조**($\propto m\,d^2 L_{\mathrm{layer}}$, memory-bound)뿐이다.

**(c) Crossover law(load-bearing scaling).** KV와 TTT가 갈리는 문맥 $S^*$의 폭 scaling이다(claim 2):

$$
S^* \;\propto\; \frac{m\,d^2}{d_{kv}}\cdot\frac{s_{\mathrm{ttt}}}{s_{\mathrm{kv}}}
\tag{19-4}
$$

이 scaling의 **위치**가 이 장의 가장 단단한 정량 결과다(표 19-3). $S^*$ 아래에서는 KV cache가 더 싼 메모리 시스템이고, 위에서는 TTT state가 더 싸다.

표 19-3 — read-crossover $S^*$의 폭 scaling(GQA-8, bf16 TTT vs fp16 KV; 실험 E3/E1.3, tag: REL·M-ASSUMED). fp8 KV는 $S^*$를 2배로, MHA는 선형으로 축소한다.

| 스케일 | Titans-340M | Titans-760M | neural-mem-1.3B | hypo-7B | hypo-70B |
|---|---|---|---|---|---|
| $S^*$ (tokens) | 16,384 | 36,864 | 65,536 | 262,144 | 1,048,576 |

> **[해설]** (19-2)–(19-4)를 나란히 놓으면 pair thesis가 **scaling law 안에서** 드러난다. quality (19-2)는 params·tokens·capacity 위에 살고, decode cost (19-3)는 state-bytes 위에 살며, 둘은 서로 다른 변수를 따른다. 그리고 training/prefill의 cost는 세 번째 법칙 — chunk $C$가 x축인 roofline(claim 7) — 을 따른다: 같은 알고리즘이 $C{=}1$(decode 영역)에서 AI≈1로 memory-bound이다가 $C$를 키우면 crossover를 넘어 compute-bound로 오른다. **하나의 모델, 두 cost 영역, 각자의 scaling law** — 이것이 D4 pair thesis의 정량적 얼굴이다.

<!-- FIG: exp-c -->

![그림 19-2 — chunk $C$가 roofline의 x축: $C{=}1$의 memory-bound 평원에서 $C$를 키우면 compute-bound로 넘어가는 곡선(host 측정, 모양만 이전)](../../figures/exp-c-chunk-roofline.png)

그림 19-2 — chunk $C$를 키우면 AI가 오르며 memory→compute 영역을 넘는다. host 측정의 crossover $C^*{\approx}32$는 **모양**만 이전되고, H100 twin의 closed-form $C^*$는 306–430이다(실험 E2.1, tag: CPU-SHAPE).

## 19.6 정직 평가: A3 falsification 조건

state-bytes와 capacity를 일급 축으로 **제안**했다. 정직하려면 이 제안이 무엇에 의해 반증되는지를 못박아야 한다. dossier §5의 A3 falsifier를 이 장의 구체로 옮긴다.

1. **공개 점의 이질성 → cross-architecture quality fit 불가능.** §19.2가 확인했듯 Titans(Llama-2)와 나머지(T5/32K)는 다른 tokenizer다. 따라서 (19-2)의 exponent $\alpha, \beta$를 **여섯 편의 공개 점만으로** 식별하는 것은 불가능하다 — 지지되는 것은 tokenizer가 정렬된 곳의 **ordering**뿐(Titans-1.3B 56.82 앵커 위의 Hope/Atlas++ 58.0대). 이 falsifier는 부분적으로 **이미 발효**되어 있으며, 그래서 이 장은 quality exponent를 내지 않는다.
2. **자체 run이 exponent를 주기엔 너무 작다.** Part III는 cost model과 micro-benchmark이지 from-scratch training이 아니다 — 이 장은 3–4 스케일 × 3 architecture class의 controlled scaling run을 돌리지 않았다. 따라서 (19-2)의 fit은 그런 run이 나올 때까지 **이월**된다(wall-clock을 A100 runbook으로 이월하는 것과 같은 성격의 유보).
3. **capacity가 state-bytes 대비 예측력을 못 더하면, capacity 축은 재명명일 뿐이다.** 이것이 결정적 falsifier다. 판정에는 **같은 $B_{\mathrm{state}}$**에서 matrix vs deep-MLP vs poly-feature memory를 동일 tokens로 훈련해, capacity-proxy가 long-context 벤치마크(BABILong·retrieval) 예측을 raw state size 위로 **개선하는지**를 fit하는 실험이 필요하다. 개선하지 못하면 capacity는 일급 축이 아니라 state-bytes의 그림자다.

> **[평가]** 세 falsifier의 현재 판정은 비대칭이다. **state-bytes 축은 살아 있다** — (19-1)은 정의이고, 그 위의 decode cost scaling(표 19-2)과 crossover scaling(표 19-3)은 세 방법이 anchor에서 1% 이내로 합치한 측정된 **모양**이다(REL·M-ASSUMED tag 아래 load-bearing). **capacity 축은 미결이다** — 이론(Atlas 정리)은 있으나 state-bytes 대비 증분 예측력은 이 장에서 검증되지 않았고, falsifier 3이 그것을 결정할 유일한 실험을 지목한다. 정직한 최종 판정은 이렇다: **state-bytes는 params·tokens 옆의 일급 cost 축으로 승격되고, capacity는 일급 quality 축 후보로 제안되며 그 자격은 falsifier 3의 controlled run에 걸려 있다.** 이 유보를 지우고 capacity를 확정 축으로 파는 것이 이 장이 피하는 과장이다.

## 19.7 systems 함의: sizing 숫자가 둘에서 셋으로

이 장의 실천적 결론은 fleet sizing의 산수를 바꾼다. transformer를 서빙하는 엔지니어는 두 숫자로 계획한다: params(weights 상주분, 고정)와 문맥 길이(KV cache, 세션마다 자람). memory-centric 모델은 **세 번째 숫자**를 강제한다 — $B_{\mathrm{state}} \propto m\,d^2 L_{\mathrm{layer}}$(per-session, 문맥 무관, 매 token RMW). 이 숫자는 KV cache의 자리를 대신하되 성격이 정반대다: 문맥이 아니라 폭으로 자라고, capacity-bound가 아니라 **bandwidth-bound**이며(claim 3: 10 ms/token에서 $B_{\max}$가 1.3B에 5.2 sequence로 대역폭이 먼저 막힘 — 23장), on-chip 상주 여부가 **설계 knob**이다(claim 4의 residency crossover $d^*{\approx}2896$ — 20장).

세 번째 숫자가 폭$^2$에 스케일한다는 (19-1)의 귀결은 배포에 직접적이다: 7B로 가면 RMW가 34 GB/token, $S^*$가 262k token으로, 70B에서는 344 GB/token, $S^*$가 1.05M token으로 이동한다(표 19-2·19-3). 즉 모델이 커질수록 (a) decode가 더 깊이 memory-bound로 밀리고, (b) TTT state가 KV cache를 이기는 문맥 문턱이 더 높아진다 — 큰 모델일수록 **더 긴 문맥에서만** state 방식이 유리해진다. 이 두 scaling이 23장의 대규모 serving 투영과 24장의 HW proposal이 딛는 정량 바닥이다.

## 요약

- 이 라인의 scaling은 params·tokens 두 축으로 기술되지 않는다. cost 쪽에 **state-bytes**($B_{\mathrm{state}}=m\,d^2 L_{\mathrm{layer}}s$, 식 19-1), quality 쪽에 **capacity**($O(d_k)$/$O(d_k^p)$/무한, → 14장)라는 두 독립 축이 빠져 있다.
- 여섯 편의 공개 품질 점은 세로(스케일)로는 단조 개선을 보이나, 가로(모델 간)로는 tokenizer 이질성(Llama-2 vs T5/32K) 때문에 직접 비교 불가다. 정당한 대조는 논문 내 재실행 baseline과, Titans-1.3B avg 56.82라는 cross-paper 앵커뿐이며, 그 위에서 Atlas++ 58.03·Hope 58.04가 동률로 Titans를 넘는다.
- 후보 법칙은 셋이다: quality (19-2, 제안·미fit), decode cost 하한 (19-3, 구조만 load-bearing·절대값 A100 이월), crossover (19-4, 위치가 load-bearing). RMW는 폭에 따라 0.45→6.44→343 GB/token, $S^*$는 16k→1.05M token으로 스케일한다.
- pair thesis가 scaling law 안에서 드러난다: quality는 params·tokens·capacity 위에, decode cost는 state-bytes 위에, training/prefill cost는 chunk $C$의 roofline 위에 산다 — 하나의 모델, 두(세) cost 영역, 각자의 법칙.
- A3 판정은 비대칭이다: **state-bytes 축은 측정된 모양으로 살아 있고**, **capacity 축은 제안이며** falsifier 3(같은 state-bytes에서 capacity-proxy가 long-context 예측을 개선하는가)의 controlled run에 자격이 걸려 있다. quality exponent와 그 run은 이월된다.

## 자가 점검 체크리스트

- [ ] 표 19-1의 세로 비교와 가로 비교의 정당성을 tokenizer 근거로 구별할 수 있다.
- [ ] state-bytes가 왜 params·tokens와 독립인 일급 cost 축인지, decode RMW와 어떻게 연결되는지(claim 1) 설명할 수 있다.
- [ ] Atlas의 capacity 사다리($O(d_k)$→$O(d_k^p)$→무한)를 quality 축으로 옮기고, 그것이 왜 state-bytes와 동일하지 않은지(capacity-per-byte) 말할 수 있다.
- [ ] 세 후보 법칙 (19-2)–(19-4)에서 무엇이 load-bearing이고 무엇이 이월되는 하한·미fit인지 판별할 수 있다.
- [ ] $S^*$ scaling(16k→1.05M)과 RMW scaling(0.45→343 GB/token)이 폭에 따라 어떻게 자라는지, 그것이 큰 모델 서빙에 무엇을 뜻하는지 설명할 수 있다.
- [ ] A3의 세 falsification 조건을 진술하고, capacity 축의 미결 자격이 어느 실험에 걸려 있는지 지목할 수 있다.
- [ ] 이 장의 결론을 inference 어휘로 옮길 수 있다: sizing 숫자가 {params, 문맥-KV} 둘에서 {params, state-bytes} 둘 + 문맥-무관으로 바뀌며, 세 번째 숫자는 폭$^2$·bandwidth-bound다.

## 다음 장으로

이 장은 state-bytes를 일급 cost 축으로 세우고 decode cost가 그 위에서 memory-bound로 스케일함을 (19-3)으로 보였다 — 그러나 "memory-bound"의 절대값은 하한으로 남겼다. 20장은 그 하한 아래로 내려가, 아키텍처 class별 decode roofline을 실제로 그린다: state read+**write**가 KV append-read와 어떻게 갈리는지, decode에 들어온 backward-pass가 왜 새 serving primitive인지, NS-5·deep-memory의 FLOP density가 어디에 쌓이는지, 그리고 state placement(그림 b·e)가 왜 설계 가능한 knob인지. scaling 축이 가리킨 병목을 hardware의 언어로 해부하는 것이 다음 장이다.
