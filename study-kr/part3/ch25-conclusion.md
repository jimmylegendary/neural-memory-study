# ch25. 결론과 연구 어젠다

> **이 장의 목표** — 독자가 이 장을 마치면 (1) Part III가 D4 pair thesis를 측정으로 어디까지 정착시켰고 어디를 정직하게 열어 두었는지 한 문단으로 진술할 수 있고, (2) 이 책이 이 라인에 새로 연 것(배포 형태로서의 완성형 + exploration-grade 실측 warrant + 쌍(pair)으로서의 하드웨어 제안)과 남긴 것을 구별할 수 있으며, (3) 여섯 편의 open-questions가 수렴한 다섯 공백을 systems 엔지니어가 이어받을 연구 어젠다로 옮길 수 있어야 한다.
> **왜 필요한가** — 18장이 완성형을 workload로, workload를 pair thesis로, pair thesis를 8개 측정으로 내렸고, 19–24장이 그 측정을 각 축의 깊이로 폈다. 이 장은 그 프로그램의 결산이다. 결산의 값은 무엇을 증명했는가만큼 무엇을 증명하지 못한 채 남겼는가에 있다 — Part III의 정직성 계약(§18.6)은 결론에서 가장 엄격하게 지켜져야 한다.

## 25.1 Bridge-in: Part III가 한 일

이 책의 Part I은 inference 엔지니어에게 훈련의 어휘를 (state, update, cost) 객체로 건네주었고, Part II는 여섯 편을 하나의 연속된 프로그램으로 해부했으며, 그 프로그램의 완성형이 layer가 아니라 **배포 형태** — 세션 상태가 mutable weights인 continually-learning LLM에 sleep 서비스가 붙은 시스템(→ 18장) — 임을 확인했다. Part III는 그 배포 형태를 systems 엔지니어의 도구로 측정한 원저 기여였다.

측정의 척추는 **D4 workload-split pair thesis**였다: 이 라인의 배포는 질적으로 다른 두 부하로 갈라지고, 각각의 최적 하드웨어 전략이 다르다. decode/serving-state 관리는 token마다 fast-weight state 전체를 read-modify-write(RMW)하는 memory-centric 부하이고, training/prefill은 chunk 크기 $C$가 roofline의 x축인 accelerator 부하다. Part III는 이 두 절반을 각각 정량화한 8개 실험을 돌렸고, 세 방법(닫힌 형태 cost model E3, analytic twin 도구 E1.x, CPU micro-benchmark E2.x)이 anchor 설정 `neural-mem-1.3B (d=2048, m=16, L=24, GQA-8, bf16)`에서 1% 이내로 일치했다. 이 장은 그 8개 측정이 무엇을 정착시켰고 무엇을 열어 두었는지부터 결산한다.

## 25.2 pair thesis 재확인: 측정이 정착시킨 것과 열어 둔 것

pair thesis의 **decode 절반**은 측정으로 견고해졌다. anchor에서 decode step은 6.44 GB/token을 움직이고 arithmetic intensity가 0.59 FLOP/byte로 H100 twin ridge(295 FLOP/byte)보다 두 자릿수 아래에 있어 결정적으로 memory-bound이며, state 트래픽이 GEMV 연산을 약 394× 압도한다 — step 비용이 곧 state 트래픽이다. 이 트래픽의 성격이 KV cache와 갈라지는 지점이 명제의 핵심이다: KV의 write는 read의 0.003%(append-once)인데 TTT state는 100%(대칭 RMW)이고, KV read가 문맥 길이에 비례해 자라는 반면 TTT RMW는 문맥에 무관하게 일정($m\,d^2\,L_{\mathrm{layer}}$로 고정)하다. 그래서 그 아래에서는 KV가 싸고 그 위에서는 TTT가 싼 **crossover 문맥 길이**가 존재한다 — anchor에서 read-crossover 약 65k token, 전체 RMW-crossover 약 131k token, 그리고 crossover는 모델 폭에 따라 $m\,d^2$로 이동한다(16k→1.05M token). 이 명제의 세 따름 결과 — 배치를 키우면 용량보다 대역폭이 먼저 막힌다(≠ capacity-bound KV), state 상주 여부가 문맥과 무관하므로 설계 가능한 knob이 된다(scratchpad fit→spill, 폭 crossover $d^*{\approx}2896$), update-frequency 연속체가 memory 계층 배치로 직접 번역된다(빠른 쪽 cadence로 tier 결정) — 도 같은 강도로 측정되었다. 그리고 기존 KV-cache manager의 semantics는 RMW state에 대해 범주 오류다: content-addressed 재사용률이 구조적으로 0이고, append-only placement가 stale 상태를 T배(256×)로 누적하며, dirty writeback을 0으로 값매긴다.

pair thesis의 **training/prefill 절반**도 측정으로 확인되었다. chunk $C{=}1$(per-token RMW = decode 영역)은 AI≈1로 memory-bound 평원에 있고, $C$를 키우면 crossover를 넘어 compute-bound로 오른다 — 한 알고리즘이 knob 하나로 두 영역을 오간다. on-die/off-die 경계에서 RMW 대역폭이 불연속으로 꺾이고 RMW가 read의 절반 throughput만 낸다는 것 — append-once KV가 피하는 **write-back 세(稅)** — 도 직접 측정되었다.

> **[평가]** 이 결산에서 정착된 것은 **비율·crossover 위치·tier 순서·bound class**뿐이다. 이것이 정직성 계약(§18.6)의 실천적 의미다. "TTT decode는 폭이 클수록·문맥이 65k token을 넘을수록 KV 대비 상대적으로 유리해진다"는 load-bearing 주장이고(비율·crossover), "그 decode가 1.92 ms 걸린다"는 하한의 보고이지 주장이 아니다. 6편의 원 논문이 H100 decode wall-clock을 하나도 공개하지 않았으므로, 이 책이 낸 절대 µs/token·mJ/token은 roofline **하한**이며 외부 검증 불가다 — 그 값들은 본문 주장의 근거가 아니라 사내 A100 runbook(Part III-a)으로 이월되는 검증 대상이다. scratchpad의 6.8× energy 이득이나 PIM의 1.6× energy 이득은 `simulation_ready=False`인 novel twin의 **directional DSE**이지 shipping-device 주장이 아니다. Part III가 정착시킨 것은 workload의 **구조**(memory-bound·RMW-지배·context-무관)이지 그 구조가 실리콘에서 내는 절대 성능이 아니다.

이 유보는 pair thesis를 약화시키지 않는다. 오히려 명제의 형태를 정확히 한다: 이 책은 "TTT decode 소자를 지금 만들어라"고 말하지 않는다. 이 책은 **decode가 append-once KV와 질적으로 다른 memory 부하라는 사실**과, 그 다름이 만드는 crossover·tier·bound의 **구조**를 측정으로 확립하고, 그 구조가 실리콘에서 어느 절대값으로 나타나는지를 A100 runbook의 과제로 넘긴다. memory-centric 논증은 이 두 지점 — decode state RMW 트래픽(§25.2 앞부분)과 update-frequency↔memory 계층 배치(claim 5) — 에서만 편다. "이 라인의 훈련이 새 메모리 소자를 요구한다", "일반 PIM이 답이다"는 논문들의 자체 증거가 반대 방향(알고리즘을 dense matmul로 다시 빚어 기존 accelerator에 맞춤)을 가리키므로 이 책은 주장하지 않았고, 결론에서도 주장하지 않는다.

## 25.3 이 책이 연 것

Part III의 기여는 세 겹이다.

**첫째, 재구성(reframing).** 이 책은 여섯 편의 종착점을 layer의 진화가 아니라 하나의 serving workload로 읽었고, 그 workload가 반으로 갈라진다는 것을 명제로 세웠다. 이 재구성이 있기 전까지 이 라인은 inference 엔지니어에게 "또 하나의 sub-quadratic sequence model 후보"였다. 재구성 이후 이것은 **추론 중 weights는 변하지 않는다는 정의적 불변식의 폐기**(→ 1장 Rosetta의 "weight update 없음" 행)이고, 그 폐기가 만드는 새 memory 부하와 새 cache class의 문제다. pair thesis는 그 문제를 systems 관점에서 최초로 분할한 진술이다.

**둘째, warrant.** 6편이 decode 비용을 하나도 측정하지 않은 자리에, 이 책은 exploration-grade의 실측 warrant를 놓았다. 세 독립 방법이 anchor에서 1% 이내로 일치하는 교차검증(RMW 6.44 GB/token, 상태 134 MB/layer, read-crossover 65k token, $B_{\max}$ 등)이 비율·crossover·tier·bound를 본문으로 승격할 근거를 주었다. 이것은 "누구나 할 수 있는 산수"가 아니다 — 어느 crossover가 어디에 있는지, 어느 자원이 먼저 막히는지, 어느 KV-manager 가정이 RMW에서 무너지는지는 계산되기 전까지 명제가 아니었다. 이 warrant는 그것들을 명제로 바꾸었다.

**셋째, 쌍(pair)으로서의 제안.** 이 책의 하드웨어 제안은 어느 한쪽이 아니라 쌍이다(→ 24장). decode 쪽에는 RMW-bandwidth 상주 소자·frequency-tiered placement·per-tenant state 배치를, training/prefill 쪽에는 fused chunk kernel·grouped-GEMM decode engine·backward-capable serving kernel을 나란히 놓는다. 어느 한쪽만 옹호하는 것은 부하의 절반을 무시하는 것이다. 이 대칭이 이 책이 "종합 view"를 자처하는 근거이며, memory-centric thread가 정직하게 load-bearing인 두 지점과 accelerator thread를 소홀히 하지 않는 균형을 동시에 지키는 방식이다.

> **[해설]** 세 기여의 관계는 위계적이다. 재구성이 없으면 warrant는 측정할 대상이 없고, warrant가 없으면 제안은 의견이며, 제안이 한쪽뿐이면 재구성이 세운 pair thesis를 배반한다. 이 책은 세 겹을 한 척추 위에 세웠다 — 그 척추가 pair thesis다.

## 25.4 남긴 것: 연구 어젠다

여섯 편의 open-questions 절이 전부 같은 다섯 공백으로 수렴했다는 것이 저자 논지("Google의 next가 완성형으로 보인다")의 힘이었다. 그 다섯 공백은 이제 Part III의 측정이 어디까지 갔는지가 확정된 만큼, systems 엔지니어가 이어받을 구체적 어젠다가 된다. 각 항목에 이 책이 준 것과 A100 runbook·후속 연구로 넘기는 것을 함께 표시한다.

**어젠다 1 — 스케일.** 모든 품질 주장이 1.3B params / 100B tokens에서 멈춰 있고([TNT]는 150M), decode wall-clock은 6편 전체에 부재하다. 이 책은 RMW 트래픽이 폭에 따라 0.45→6.44→343 GB/token으로, crossover가 16k→1.05M token으로 자라는 스케일링을 twin의 analytic 외삽으로 제시했다(→ 19장). 그러나 이것은 측정이 아니라 $m\,d^2\,L_{\mathrm{layer}}$ 공식의 확장이다. 열린 문제: memory-bound RMW라는 **구조**가 7B+·SFT/RLHF·production 데이터에서 유지되는가, state-bytes와 capacity($O(d_k^p)$)가 params·tokens와 나란한 일급 scaling 축이 되는가. runbook 과제: 각 스케일의 절대 wall-clock — 논문이 하나도 주지 않은 값.

**어젠다 2 — retrieval 격차와 hybrid workload.** attention이 in-context recall에서 여전히 이긴다: 53.55 vs 43.70 [Atlas Table 5], FDA 67.3 vs 41.9 [NL]. capacity 이론이 그 이유($\phi^*$의 무한 capacity, → 14장)까지 말해 준다. 이 격차가 parametric하게 닫히지 않으면 완성형은 attention과의 **hybrid**다. 그리고 hybrid는 pair thesis를 한 소자 위로 접는다: sliding-window KV(append-once·read-many)와 RMW state(write-heavy·unshared)가 **공존**하는 decode다. 이 책의 $B_{\max}$·crossover는 단일 workload 가정 위에서 측정되었다(carry-forward 6: weights+activations+KV 공존 시 $B_{\max}$는 상한). 열린 문제: 두 부하가 한 배포에 공존할 때 crossover가 여전히 의미가 있는가, hybrid decode의 비용 모델은 무엇인가.

**어젠다 3 — serving 경제학과 kernel.** 이것이 이 라인이 아직 기다리는 FlashAttention-moment이다(→ 21장). per-request mutable weights는 shared-weight batching을 깨뜨리고(grouped-GEMM decode), speculative decoding의 rollback에 optimizer-trajectory state의 snapshot을 요구하며, multi-tenant 격리와 weight에 흡수된 context의 privacy를 새 문제로 만든다. 이 책은 batch가 대역폭에 먼저 막힌다는 것(claim 3)과, 기존 KV-manager가 RMW state에 대해 범주 오류라는 것(claim 6: `update_in_place`, `mark_dirty/writeback`, `checkpoint/rollback`, `bind_to_sequence`, `free_on_update`가 KVCacheManager에 없음)을 보였다. 열린 문제: TTT-state manager와 backward-capable fused decode kernel의 실제 구현. runbook 과제: 진짜 KV read kernel과 진짜 TTT-state kernel의 head-to-head로 crossover를 검증(현재 $S^*$는 roofline 반쪽 대조).

**어젠다 4 — 학습되는 스케줄.** chunk 크기, window $c$, CMS frequency, level 수, sleep timing이 전부 손으로 설정되어 있다. [NL]은 잘못 놓인 level이 품질을 해칠 수 있음을 보였고(inner-q ablation), [Sleep]은 sleep을 chunk 경계에 하드와이어한다. 이 책의 claim 5는 cadence가 **주어지면** tier가 결정적 규칙으로 따라옴을 보였다 — 그러나 cadence 자체를 학습하는 것은 아무것도 없다. 여기서 두 정직한 memory-centric 지점이 교차한다: 잘못 놓인 level은 이제 품질 비용만이 아니라 **잘못 놓인 tier**라는 systems 비용이기도 하다(HBM에 있어야 할 것이 CXL에 내려가면 critical path를 막는다). 열린 문제: memory 계층과 함께 co-design되는 스케줄 — *무엇을 언제 갱신할지*가 *무엇을 어디에 둘지*와 함께 학습되는 것.

**어젠다 5 — task-free·안전한 self-modification, 그리고 미룬 이론.** [Sleep]의 Dreaming은 명시적 (context, metric) 쌍을 요구하고, reward-model 의존은 검토되지 않았으며, 재귀적 weight self-editing에는 안전성 분석이 없다. 배포 관점에서 이것은 **자기 weights를 세션마다 고쳐 쓰는 serving fleet의 안전**이다: weight에 흡수된 context의 privacy, per-tenant 발산, rollback과 provenance, 매 sleep의 versioning(→ 23장). 그리고 미룬 이론이 있다 — linear 특수 경우 너머의 regret·capacity·expressivity 결과가 없고, chunkwise staleness의 오차 한계가 6편 어디에도 없다(→ 9장·15장). 이 책도 그 공백을 메우지 못했다: $C^*$의 **모양**은 측정했으나 staleness가 만드는 근사 오차의 **한계**는 측정하지 못했다. 열린 문제: staleness 오차 한계, 그리고 self-modifying serving의 안전 프레임.

> **[평가]** 이 다섯 어젠다는 임의로 고른 미래 과제가 아니다. 여섯 편이 스스로의 open-questions에서 같은 곳을 가리켰기 때문에, 남은 일은 텍스트에 의해 과잉 결정되어 있다. Part III가 한 것은 그 다섯 중 serving 경제학의 절반(어젠다 3의 측정 가능한 부분)을 exploration-grade로 먼저 밟은 것이다. 나머지 — 스케일의 실증, 격차의 해소, 스케줄의 학습, 안전의 정식화 — 는 이 책이 연 재구성 위에서 다음 연구가 밟을 자리다.

## 25.5 닫는 말

여섯 편은 "test time에 외우는 법을 배우자"([Titans])로 열어 "모델은 언제 깨어 있고 언제 자야 하는가"([Sleep])로 닫혔고, 그 사이에 추론 중 weights 불변이라는 transformer serving의 정의적 전제를 지웠다. 그 지움이 만든 것이 이 책이 측정한 새 workload다. 완성형은 개념적으로는 여섯 편의 텍스트만으로 도출되지만, 실증적으로는 1.3B/100B에서 멈춰 있고 retrieval 격차가 열린 채이며 decode wall-clock은 부재하다 — 이 세 유보가 이 책의 모든 정량 주장이 딛는 바닥이다.

그래서 이 책은 완결이 아니라 **on-ramp**이다. pair thesis는 명제이고, 8개 측정은 그 명제를 비율·crossover·tier·bound의 수준에서 정착시킨 warrant이며, 다섯 어젠다는 그 명제가 실리콘과 스케일에서 검증될 자리를 표시한 지도다. 이 라인이 attention의 GEMM density에 맞선 자기만의 FlashAttention-moment을 맞을지, 아니면 sliding-window KV와 RMW state가 한 소자 위에 공존하는 hybrid로 안착할지는 아직 열려 있다. 어느 쪽이든, 그 결정은 layer의 우아함이 아니라 **decode가 움직이는 바이트와 그것이 앉는 memory 계층**에서 난다 — 이 책이 그 자리를 inference 엔지니어의 도구로 처음 측정한 이유가 거기에 있다.

## 요약

- Part III는 D4 pair thesis를 8개 실험으로 측정했고, decode 절반(memory-bound whole-state RMW, KV와의 crossover, BW-먼저-막힘, 상주 knob, cadence→tier, KV-manager 범주 오류)과 training/prefill 절반(chunk $C$ = roofline x축, RMW cliff·write-back 세)을 모두 정착시켰다.
- 정착된 것은 비율·crossover·tier·bound뿐이다. 절대 µs/token·mJ/token은 roofline 하한이며 A100 runbook(Part III-a)으로 이월되고, scratchpad·PIM은 `simulation_ready=False`의 directional DSE다. memory-centric 논증은 decode RMW 트래픽과 update-frequency↔tier 두 지점에서만 편다.
- 이 책이 연 것은 세 겹이다: 여섯 편의 종착점을 serving workload로 읽는 **재구성**, 6편이 비운 자리에 놓은 exploration-grade **warrant**(세 방법 1% 이내 교차검증), 어느 한쪽이 아니라 **쌍**으로서의 하드웨어 제안.
- 남긴 것은 여섯 편의 open-questions가 수렴한 다섯 어젠다다: 스케일(구조가 7B+에서 유지되는가), retrieval 격차와 hybrid workload, serving 경제학·kernel(이 가족의 FlashAttention-moment), 학습되는 스케줄(계층과 co-design), task-free·안전한 self-modification과 미룬 이론(staleness 오차 한계).
- 완성형은 개념적으로 도출되나 실증적으로 세 유보(1.3B/100B 상한, 미해소 retrieval 격차 53.55 vs 43.70, 부재한 decode wall-clock) 위에 있다. 이 책은 완결이 아니라 on-ramp이며, 승부는 layer의 우아함이 아니라 decode가 움직이는 바이트와 그것이 앉는 memory 계층에서 난다.

## 자가 점검 체크리스트

- [ ] Part III가 pair thesis의 두 절반을 각각 어떤 측정으로 정착시켰는지 나열할 수 있다.
- [ ] 어떤 양이 load-bearing(비율·crossover·tier·bound)이고 어떤 양이 이월되는 하한(절대 µs/mJ)·directional(novel twin)인지 판별할 수 있다.
- [ ] 이 책이 연 세 겹(재구성·warrant·pair 제안)과 그 위계 관계를 설명할 수 있다.
- [ ] 다섯 연구 어젠다 각각에 대해 이 책이 준 것과 남긴 것을 구별할 수 있다.
- [ ] retrieval 격차가 닫히지 않을 때 완성형이 왜 hybrid가 되고, 그것이 pair thesis를 어떻게 한 소자 위로 접는지 설명할 수 있다.
- [ ] 완성형의 세 실증 유보를 짚고, 이 책을 완결이 아닌 on-ramp으로 위치시킬 수 있다.
