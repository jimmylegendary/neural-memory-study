# ch10. 통합 systems bridge: cost model cheat sheet, roofline vs chunk size, glossary

> **이 장의 목표** — 독자가 이 장을 마치면 다음을 할 수 있어야 한다. (1) 식 (M1)–(M4) 형태의 임의 layer에 대해 decode FLOPs/token, state 트래픽 bytes/token, state 크기를 표 없이 유도한다. (2) chunk 크기 $C$를 roofline 위의 독립변수로 놓고 arithmetic intensity를 계산해, 품질-최적 $C$와 MFU-최적 $C$가 왜 갈라지는지 숫자로 보인다. (3) KV cache와 fast-weight state의 memory-budget 승부를 자기 서빙 config의 숫자로 판정한다. (4) Part II의 어떤 efficiency 주장이든 이 장의 cheat sheet로 환원해 검산(audit)한다.
>
> **왜 필요한가** — [TNT] (*TNT: Improving Chunkwise Training for Test-Time Memorization*, arXiv:2511.07343)는 사실상 이 장의 언어로 쓰인 논문이다: Challenge 1(작은 chunk 훈련의 peak FLOPs 5-10% 미만 utilization)과 Challenge 3(chunk-size mismatch)은 roofline 분석 없이는 주장 자체가 판독되지 않는다. [Titans §3.2]의 "병렬화 가능하다", [Atlas §3.4]의 "Omega rule도 병렬화된다"는 practicality 주장도 이 장의 도구로 audit해야 한다. 그리고 Part II 모든 장(ch12–ch17)의 §"systems/serving 함의" 절은 이 장의 표 10-2를 암묵적 참조 대상으로 삼는다. 마지막 절의 glossary는 Part II를 읽는 동안 옆에 두는 참조표다.

이 장이 인용하는 나머지 논문의 축약은 다음과 같다: [Titans] (*Titans: Learning to Memorize at Test Time*, arXiv:2501.00663), [Atlas] (*Atlas: Learning to Optimally Memorize the Context at Test Time*, arXiv:2505.23735), [NL] (*Nested Learning*, arXiv:2512.24695), [Sleep] (*Language Models Need Sleep*, arXiv:2606.03979).

Part I의 각 장은 말미마다 자기 몫의 systems bridge를 깔았다. 이 장은 그 조각들을 하나의 독립된 분석 도구 상자로 통합한다. 목적은 단순한 복습이 아니다. 이 라인의 논문 6편은 모두 "우리 방식이 hardware에 더 잘 맞는다"는 종류의 주장을 하는데, 그 주장을 뒷받침하는 측정은 대부분 훈련 쪽에 몰려 있고 decode 쪽은 비어 있다. 독자가 논문의 숫자를 받아 적는 대신 자기 roofline 위에서 재계산할 수 있게 만드는 것 — 그것이 이 장의 존재 이유다.

> **[평가]** 6편 전체를 통틀어 decode wall-clock 수치는 보고되지 않는다. serving 관점의 주장(상수 크기 state, chunk-1 decode의 우월성 등)은 전부 구조적 논증이지 측정이 아니다. 따라서 이 장의 cost model은 편의 도구가 아니라, 현재로서는 이 라인의 serving 비용을 알 수 있는 유일한 수단이다.

## 10.1 완성된 Rosetta-Stone 사전

1장에서 시작한 inference 어휘 ↔ learning 어휘의 사전을, Part I 아홉 개 장을 통과한 지금 시점의 완성형으로 다시 싣는다. 이 표를 쓸 때 결정적으로 중요한 것은 세 번째 열이다. 대응이 **동일**(같은 대상의 재서술)인지, **유비**(정확하지만 다른 대상)인지, **차이가 논점**(직관을 그대로 이식하면 틀리는 지점)인지 구분하지 않으면, 독자는 자기 세계의 직관을 잘못된 곳에 이식하게 된다. Part II를 읽다가 길을 잃으면 언제든 이 표로 돌아오면 된다.

표 10-1 — 완성된 Rosetta-Stone 사전 (확립 장 병기)

| inference 세계 (독자의 어휘) | 이 라인의 어휘 | 대응의 성격 | 확립 |
|---|---|---|---|
| KV cache | non-parametric memory state; softmax attention = capacity 무한($\phi^*$)의 associative memory | **동일 대상의 재서술** — attention은 압축하지 않는 memory | 1·5장 |
| KV cache append | memory **write** (outer-product Hebbian write가 최근접 대응; delta rule은 append가 아니라 **overwrite**) | 유비 + 차이: cache는 append-once, fast weights는 read-modify-write | 5장 |
| attention lookup ($q\cdot K$) | memory **read** $y_t=\mathcal{M}(q_t;W_t)$ | 동일 추상화 ($k\to v$ 연상) | 5장 |
| linear-RNN state ($d\times d$) | matrix memory = 고정 크기 lossy 압축 memory | 동일 | 6·7장 |
| prefill | 큰 chunk의 병렬 write (compression) — TNT에선 global memory가 담당 | 유비(정확) | 9장 |
| decode | $C=1$의 per-token online write + read | 유비(정확) — 단 **backward pass가 decode에 들어온다**는 점이 신세계 | 8장 |
| FlashAttention tiling | chunkwise-parallel training의 chunk | ⚠ **차이가 논점**: tiling은 exact, chunk는 함수 자체를 바꾸는 semantic 근사 (M4) | 9장 |
| GEMM shape 감각 | $\nabla_W\ell$의 outer-product 구조: $dW=(\text{오차})\,k^\top$는 rank-$C$ GEMM | 동일 — 훈련 수식을 GEMM shape로 읽는 것이 이 책의 교수법 | 2장 |
| scan/prefix-sum kernel | momentum의 associative scan (S5식), $\Pi_t$의 누적 합 | 동일 | 7·9장 |
| roofline / arithmetic intensity | chunk 크기 $C$가 arithmetic intensity를 결정; 품질 최적 $C$(작음) vs MFU 최적 $C$(큼)의 긴장 | 동일 도구, 새 독립변수 | 이 장 |
| batching (shared weights 전제) | per-request fast-weight state는 shared-weight batching을 **깨뜨린다** → grouped-GEMM decode | ⚠ 차이가 논점 | 이 장 |
| paged KV cache / session cache | per-session weight state — 새로운 cache class (sizing, checkpoint/restore, eviction) | 유비 → Part III의 주제 | 이 장 |
| cache eviction policy | retention gate = **학습된** eviction | 유비(정확) | 3·13장 |
| speculative decoding의 rollback | memory state snapshot/rollback (optimizer-trajectory 상태 포함) | 유비 + 미해결 문제 | 이 장 |
| optimizer state | 훈련의 "register file / accumulator": $m_t,h_t,S_t$ | 도입용 유비 | 2장 |
| weight update 없음(추론 불변식) | **폐기되는 불변식** — 이 라인의 정의적 특징 | 차이 그 자체 | 1·8장 |
| sequence 축 | 훈련의 batch 축 역할을 **sequence 축이 대신한다** (inner loop의 mini-batch = chunk) | ⚠ 최대 혼동 지점 | 2·9장 |
| distributed training | context parallelism: reset이 sequential chain을 끊어 shard 병렬화 (TNT) | 도입용 유비 (data parallelism과의 대비) | 9·15장 |
| 서빙 fleet의 백그라운드 job | sleep phase = 주기적 offline consolidation/dreaming job ("weights의 background compaction") | 유비 → 11·17장, Part III | 11장 |

⚠ 표시 세 행은 Part II에서 반복적으로 사고 사고(事故)가 나는 지점이므로 한 번 더 짚는다. 첫째, FlashAttention tiling의 tile 크기는 출력을 바꾸지 않는 구현 knob이지만, chunkwise training의 $C$는 계산되는 함수를 바꾼다(§10.5). 둘째, decode batching의 경제학은 "모든 request가 같은 weights를 읽는다"는 전제 위에 서 있는데, fast-weight 모델은 그 전제를 정의상 깨뜨린다(§10.4). 셋째, 이 라인에서 "mini-batch"라는 단어가 나오면 그것은 batch 축이 아니라 sequence 축의 token 묶음, 즉 chunk를 가리킨다.

## 10.2 Cost-model cheat sheet

아키텍처 클래스별 비용을 한 표로 모은다. 회계 규약을 먼저 고정한다. 이 규약은 표 10-2와 이후 모든 계산에 공통이다.

- **범위**: layer 1개, token 1개(decode 기준). head 병렬·batch 축·projection($W_K,W_V,W_Q$) 비용은 생략 — 모든 행에 공통이므로 비교에 영향이 없다.
- **FLOPs**: MAC 1회 = 2 FLOPs. backward pass ≈ forward의 2× (→ 2장). 지배항만 남기고 상수 차수의 벡터 연산은 버린다.
- **bytes**: bf16 기준 원소당 2 bytes. **state RMW 트래픽**은 state를 읽고(read) 갱신해(modify) 다시 쓰는(write) 왕복 트래픽이다: state 원소 수 $s$에 대해 $2\times s\times 2\,\mathrm{B} = 4s$ bytes.
- **기호**: $L$ = 문맥 길이, $w$ = sliding window 길이, $d$ = 차원(간결히 $d_k=d_v=d$), $c$ = Omega rule window 길이, $P_{\mathcal{M}}$ = deep memory $\mathcal{M}$의 parameter 수(장-국소 기호; 표준 deep memory인 expansion 4의 2-layer residual MLP라면 $P_{\mathcal{M}}=8d^2$), $N$ = TNT local memory 개수.

표 10-2 — 아키텍처 클래스별 cost-model cheat sheet (layer당, token당, 지배항)

| 아키텍처 | state (원소 수) | decode FLOPs/token | state 트래픽 (bytes/token) | prefill·train 병렬화 형태와 MFU 결정 요인 |
|---|---|---|---|---|
| softmax attention + KV cache | $2Ld$ (**증가**) | $\approx 4Ld$ | $4Ld$ 읽기 + $4d$ append | attention-like: $O(L^2 d)$ GEMM; FlashAttention tiling은 exact. decode는 cache 전체 재읽기가 지배 |
| sliding-window attention (SWA) | $2wd$ | $\approx 4wd$ | $4wd$ | 동일하되 window-local; $L$과 무관 |
| linear attention (un-gated) | $\approx d^2$ | $\approx 4d^2$ | $4d^2$ RMW | chunkwise-GEMM(**exact** 항등, → 9장) 또는 scan; $C$는 tile성 knob |
| GLA / Mamba-2 | $d^2$ (+gate 소량) | $\approx 5d^2$ | $4d^2$ RMW | SSD block 분해·2-level tiling(exact); scan 형태는 bandwidth-bound (→ 7장) |
| DeltaNet / Gated DeltaNet | $d^2$ | $\approx 6d^2$ | $4d^2$ RMW | WY/UT transform으로 chunkwise-GEMM(**exact**, → 9장); Householder 사슬이 kernel 복잡도 결정 |
| TTT-Linear | $d^2$ | $\approx 6d^2$ | $4d^2$ RMW | dual form chunkwise-GEMM — 단 **semantic**: $C$가 함수를 바꾼다 (M4) |
| TTT-MLP | $P_{\mathcal{M}}$ | $\approx 8$–$10\,P_{\mathcal{M}}$ | $4P_{\mathcal{M}}$ RMW | dual form(semantic) + 비선형 inter-chunk 의존(LN 등)이 chunk 간 직렬화를 강제 [TNT §1] |
| Titans-LMM | $2P_{\mathcal{M}}$ ($W_t,S_t$) | $\approx 12\,P_{\mathcal{M}}$ | $8P_{\mathcal{M}}$ RMW | chunkwise(semantic) + momentum의 associative scan + chunk-상수 gate 단순화 [Titans §3.2] |
| Atlas (OmegaNet 계열) | $2P_{\mathcal{M}}+2cd$ | 식 (M3) 그대로면 $\approx 6c\,P_{\mathcal{M}}$ + $\mathrm{NS}_\kappa$ 상각분 | $8P_{\mathcal{M}}+4cd$ | Omega rule 병렬화 [Atlas §3.4]; $\mathrm{NS}_\kappa$ = chunk당 $\kappa$회의 소형 GEMM |
| TNT (서빙형: global + $N$ local) | $(1{+}N)P_{\mathcal{M}}+Nd^2$ | $\approx 10\,P_{\mathcal{M}} + 4d^2$ (global write는 $C_{\mathrm{g}}$로 상각) | $4(1{+}N)P_{\mathcal{M}}+4Nd^2$ RMW | global = 순차지만 dense한 큰 GEMM, local = reset로 shard 병렬 [TNT §4.1] |

표를 읽는 법을 클래스별로 짚는다.

**KV cache 클래스** (1–2행). state가 자라는 유일한 클래스이고, decode 비용이 $L$에 비례하는 유일한 클래스다. 독자가 이미 아는 그대로다 — 이 표에서 이 두 행의 역할은 나머지 여덟 행의 대조군이다.

**matrix-state 클래스** (3–6행). state가 $d^2$로 고정이고, decode의 본질이 **state RMW**라는 점이 공통이다. token 하나가 유발하는 연산은 GEMV 몇 개와 rank-1 갱신뿐이므로, FLOPs/byte 비율은 한 자릿수에 머문다(§10.6에서 손으로 계산한다). 즉 이 클래스의 decode는 예외 없이 bandwidth-bound이고, 승부처는 FLOPs가 아니라 "state $4d^2$ bytes를 매 token 왕복시킬 수 있는가"이다. 같은 클래스 안에서 linear attention → GLA → DeltaNet으로 갈수록 FLOPs 계수가 4→5→6으로 오르지만 트래픽은 동일하다는 점에 주목하라. bandwidth-bound 영역에서는 **계수가 다른 FLOPs는 공짜다** — retention gate와 delta rule의 품질 이득이 decode 비용 관점에서 거의 무료인 이유다.

**deep-memory 클래스** (7–9행). state가 행렬 하나에서 작은 신경망의 weights 전체($P_{\mathcal{M}}$)로 커지고, write가 "outer product 하나"에서 "forward + backward + update"로 바뀐다. backward ≈ 2× forward 규약을 적용하면 compression에 $\approx 6P_{\mathcal{M}}$, retrieval에 $2P_{\mathcal{M}}$ — 그래서 계수가 8–10으로 뛴다. Titans-LMM은 여기에 momentum buffer $S_t$의 갱신과 retention 곱이 elementwise로 얹혀 $\approx 12P_{\mathcal{M}}$, 그리고 **state가 두 배가 된다**: $W_t$뿐 아니라 $S_t$도 session state다. Atlas 행은 경고 표지판이다. 식 (M3)은 매 step 최근 $c$쌍 전체에 대한 gradient를 요구하므로, 문자 그대로 실행하면 decode FLOPs가 $c$배로 뛰고 최근 $c$개의 $(k,v)$ buffer($2cd$ 원소)가 state에 추가된다. 이것을 어떻게 상각하는지가 [Atlas §3.4]의 주제이고, 14장에서 자세히 다룬다.

<!-- TODO-VERIFY: Atlas 실험에서 실제 사용한 window 길이 c의 값(들)과, decode 시 Omega rule을 incremental하게 유지하는지 여부. 확인 방법: papers/2505.23735.txt에서 "window length", "c =" 검색 후 ch14와 정합화 -->

**TNT 행** (10행). 위 회계가 그대로 합성됨을 보여 주는 행이다. global memory는 $C_{\mathrm{g}}$ token마다 한 번 큰 batch로 write하므로 per-token으로 상각하면 싸고, 매 token 비용은 local memory의 fwd+bwd+read, global의 read, 그리고 Q-K projection $\Pi_t$의 rank-1 갱신($2d^2$)과 mat-vec($2d^2$)이다. state는 $(1{+}N)P_{\mathcal{M}}+Nd^2$ — 문맥 길이와 무관한 상수라는 것이 [TNT]의 serving 쪽 논지다.

## 10.3 KV cache vs fast weights: memory-budget 산수

"고정 크기 state니까 이긴다"는 문장은 절반만 참이다. 언제부터 이기는지는 나눗셈 한 번으로 판정된다. layer당 state 원소 수를 $s$라 하면, KV cache는 layer당 token당 $2d$ 원소를 쌓으므로 crossover 문맥 길이는

$$
L^{\ast} \;=\; \frac{s}{2d}
\tag{10-1}
$$

이다. layer 수와 정밀도는 양변에서 소거된다. 세 가지 대표 state에 대입하면: matrix memory($s=d^2$)는 $L^\ast=d/2$, 표준 deep memory($s=P_{\mathcal{M}}=8d^2$)는 $L^\ast=4d$, deep memory + momentum($s=2P_{\mathcal{M}}$)은 $L^\ast=8d$. 구체적 config로 채우면 다음과 같다: $d=2048$, layer 24개, bf16, GQA 없는 MHA, 모든 layer가 memory layer라고 가정한다(이 가정들은 전부 보수적 단순화이며, 자기 스택의 값으로 갈아 끼우는 것이 이 절의 목적이다).

표 10-3 — memory budget: KV cache vs fast-weight state ($d=2048$, 24 layers, bf16)

| 항목 | 4K 문맥 | 64K 문맥 | 1M 문맥 | 비고 |
|---|---|---|---|---|
| KV cache (전체) | 768 MiB | 12 GiB | 192 GiB | token당 192 KiB씩 선형 증가 |
| matrix memory ($d^2$/layer) | 192 MiB | 192 MiB | 192 MiB | $L^\ast=d/2=1024$ tokens |
| deep memory ($8d^2$/layer) | 1.5 GiB | 1.5 GiB | 1.5 GiB | $L^\ast=4d=8192$ tokens |
| deep memory + momentum | 3 GiB | 3 GiB | 3 GiB | $L^\ast=8d=16384$ tokens |

이 표에서 읽어야 할 것은 두 가지다. 첫째, **fast weights는 짧은 문맥에서는 오히려 진다.** deep memory + momentum의 state 3 GiB는 16K token 어치의 KV cache와 맞먹는다. 4K짜리 챗 세션만 서빙하는 fleet라면 Titans류 layer는 memory budget 관점에서 손해다. 이 라인의 논문들이 하나같이 long-context 벤치마크를 앞세우는 것은 취향이 아니라 산수다. 둘째, **1M 문맥에서는 승부가 두 자릿수 배율로 갈린다**: 192 GiB(HBM 여러 장) vs 3 GiB(상수). KV cache 쪽은 paged allocation, quantization, eviction으로 싸울 수 있지만 $O(L)$ 성장 자체는 못 없앤다. 이 라인의 제안은 그 성장 곡선을 상수로 접는 대신, 그 대가를 §10.4의 새로운 serving 문제들로 지불하는 거래다.

트래픽 관점의 따름정리 하나. decode 시 softmax attention은 KV cache 전체를 매 token 다시 읽는다 — 위 config의 64K 문맥이면 token당 12 GiB. HBM 3 TB/s급 장비에서 그것만으로 4 ms/token이다. 반면 deep memory + momentum의 state RMW는 token당 6 GiB($3\,\mathrm{GiB}\times 2$ 왕복)로 보이지만, 이는 매 layer가 아니라 **전체 합**이고 문맥과 무관하다. 긴 문맥에서 이 라인이 파는 것은 FLOPs 절감이 아니라 bandwidth 절감이다.

## 10.4 Prefill/decode 비대칭과 serving engine: backward pass가 들어온다

memory-as-weights 모델의 prefill과 decode는 독자가 아는 비대칭을 그대로 물려받되, 내용물이 바뀐다. **prefill = 큰 chunk의 병렬 write**다: 프롬프트 $L$ token을 chunk 단위로 묶어 (M4)식의 batched gradient를 GEMM으로 흘리는, attention prefill과 같은 compute-bound 구간이다. **decode = $C=1$의 online write + read**다: token 하나마다 작은 net의 forward, backward, update, 그리고 read forward가 돈다. [TNT §4.2]는 이 대응을 아예 설계 목표로 삼는다 — global memory($C_{\mathrm{g}}=2048$)가 prefill을 담당하고, Stage 2에서 $C_{\mathrm{l}}'=1$로 적응시킨 local memory가 decode를 담당하는 구도이며, prefill 시 local memory는 reset 덕분에 $L/L_{\mathrm{s}}$개 shard로 나뉘어 병렬로 채워진다.

decode에 backward pass가 들어온다는 사실이 serving engine에 요구하는 변화를 네 가지로 정리한다. 이 목록은 Part III에서 설계 문제로 다시 다뤄지므로, 여기서는 비용 구조만 못박는다.

**1. autograd 없이 gradient를 계산해야 한다.** serving engine은 autograd graph를 싣지 않는다. 다행히 이 라인의 inner gradient는 전부 닫힌 형태다. linear memory라면 $\nabla_W\ell = 2(Wk_t-v_t)k_t^\top$ — GEMV 하나, 벡터 뺄셈 하나, outer product 하나다. 표준 deep memory(2-layer residual MLP)라도 손으로 유도한 GEMM 서너 개다. 즉 "backward"는 개념적으로는 신세계지만 kernel 관점에서는 **shape가 알려진 GEMM 몇 개를 fused RMW kernel로 굳히는 일**이다. 훈련 프레임워크가 아니라 FlashAttention을 짜던 방식의 일이라는 뜻이다.

**2. per-session state가 새로운 cache class가 된다.** 크기 산정은 표 10-3이 해결한다. paged KV cache와 달리 크기가 고정이라 allocator는 오히려 단순해지지만, 새 문제 세 개가 생긴다. (a) **checkpoint/restore**: 세션 선점 후 복원하려면 $W_t$(그리고 $S_t$, $\Pi_t$)를 통째로 저장하거나 프롬프트를 재생(replay)해야 한다. TNT의 periodic reset은 여기서 뜻밖의 선물을 준다 — local state는 어차피 $L_{\mathrm{s}}$ token마다 $W_{\mathrm{init}}$로 되돌아가므로, 재생해야 할 분량이 최대 $L_{\mathrm{s}}$ token으로 유계다. (b) **prefix 재사용**: state는 전체 history의 함수이므로 임의 중간 편집은 불가능하고, snapshot을 뜬 지점에서의 분기만 가능하다. 대신 state가 고정 크기라 snapshot 복사는 KV cache 복사보다 싸다. (c) **eviction**: cache 계층에서 손으로 짜던 eviction policy가 모델 안으로 들어와 retention gate $\alpha_t$라는 학습된 정책이 된다(→ 3장, 13장). fleet 수준 eviction(세션 자체를 내리는 결정)은 여전히 엔진 몫이다.

**3. shared-weight batching이 깨진다.** decode batching의 경제학은 batch $B$개의 request가 같은 weights를 읽으므로 weight 트래픽이 상각된다는 데 있다. fast-weight layer에서는 request $r$마다 자기 $W_t^{(r)}$가 있으므로 read는 $B$개의 서로 다른 작은 GEMV, write는 $B$개의 서로 다른 RMW가 된다 — multi-LoRA serving에서 이미 만난 grouped-GEMM 패턴이다. 결정적 차이: state 트래픽이 $B$에 비례해 자라므로 **batching이 arithmetic intensity를 올려 주지 않는다.** batch를 키워 compute-bound로 넘어가는 표준 탈출로가 막혀 있다는 것 — 이것이 fast-weight decode의 가장 불편한 systems 사실이다. (단 head 축과 layer 축의 병렬성은 남아 있고, TNT처럼 shard를 batch 축에 쌓는 훈련 쪽 트릭도 있다 [TNT §4.1].)

**4. rollback이 optimizer-trajectory 문제가 된다.** speculative decoding에서 draft가 기각되면 KV cache는 꼬리를 자르면 끝이다. fast-weight 모델은 $W_t$만이 아니라 $S_t$, $\Pi_t$까지 궤적 전체를 되돌려야 한다. 고정 크기라 snapshot 자체는 싸지만, "몇 token마다, 어느 buffer까지" 찍을지는 6편 어디에도 답이 없는 열린 설계 문제다(표 10-1의 마지막 ⚠ 유비).

## 10.5 FlashAttention tiling vs chunkwise training: roofline 위의 chunk 크기

9장의 핵심 명제를 systems 언어로 다시 세운다. FlashAttention의 tiling은 **같은 합을 다른 순서로 계산하는 재배열**이다 — tile 크기는 SRAM에 맞추는 구현 knob이고, 출력은 수학적으로 동일하다(부동소수 재배열 오차 수준의 차이만 남는다). 반면 chunkwise training의 $C$는 식 (M4)가 보여 주듯 **gradient의 평가 지점 자체를 바꾼다**: chunk 안의 모든 gradient가 chunk 시작 상태 $W_{\xi(t,C)}$에서 평가되는 stale-snapshot 근사이므로, $C$가 다르면 다른 함수가 계산된다. 그래서 $C$는 tile 크기가 아니라 **semantic hyperparameter**다(→ 9장). 단, 이 구분은 클래스를 가른다는 점을 다시 강조한다: linear-state 모델(linear attention, GLA, DeltaNet의 WY/UT)의 chunkwise 형태는 exact 항등 재배열이고, semantic해지는 것은 gradient가 비선형 $\mathcal{M}$을 통과하는 TTT/Titans 계열부터다.

semanticity의 실증은 [TNT Fig. 2]가 제공한다. $C=64$로 pre-train한 550M Titans를 inference chunk 크기만 바꿔 평가하면 perplexity가 8/16/32/64/128/256/512에서 36.45/34.15/24.23/13.78/15.5/17.88/22.4로 움직인다 — 훈련 chunk에서 최적이고 양쪽으로 급격히 무너진다. tile 크기를 바꿨는데 품질이 2.6× 나빠지는 kernel은 존재하지 않는다. 이 한 장의 그림이 "$C$는 tile이 아니다"의 증명이다(단일 설정의 단일 그림이라는 한계는 15장이 다룬다).

이제 $C$를 roofline 위에 놓는다. matrix-state 클래스(계수 6짜리 delta 계열)의 chunk 하나를 fused kernel이 처리한다고 하면, chunk당 FLOPs는 $\approx 6Cd^2$, 트래픽은 state RMW $4d^2$ bytes + token IO($k,v,q,y$ 네 벡터) $8Cd$ bytes이므로

$$
\mathrm{AI}(C) \;=\; \frac{6\,C\,d^{2}}{4d^{2} + 8Cd}\ \left[\mathrm{FLOP/byte}\right]
\tag{10-2}
$$

이다. 이 식은 두 극한에서 이 절 전체를 요약한다. $C$가 작으면 $\mathrm{AI}\approx 1.5C$ — intensity가 $C$에 **선형**으로 자란다. chunk를 두 배로 키우면 roofline 위에서 두 배 오른쪽으로 간다. $C$가 크면 $\mathrm{AI}\to 0.75d$ — state 대비 token IO가 지배해 **$d$가 정하는 천장**(이하 **intensity ceiling**)에서 포화한다. deep memory로 바꾸면 분자의 $d^2$이 $P_{\mathcal{M}}$으로 커져 ceiling이 $\sim P_{\mathcal{M}}/d$로 올라가는 대신, 비선형 inter-chunk 의존이 chunk 간 병렬화를 막는다 [TNT §1]. 세 실행 영역의 roofline 배치는 이렇게 정리된다: $C=1$(순수 recurrent)은 GEMV+elementwise로 좌측 끝의 bandwidth-bound 영역, $1<C<L$(chunkwise)은 식 (10-2)를 따라 ridge를 향해 이동, $C=L$(순수 병렬, attention-like)은 compute-bound이되 총 FLOPs가 $O(L^2 d)$로 폭발한다. chunkwise는 이 두 극단 사이의 연속 보간이고, $C$가 그 보간 위치다.

여기서 이 라인의 근본 긴장이 나온다. **품질-최적 $C$는 작고, MFU-최적 $C$는 크다.** 품질 쪽: 훈련·추론 chunk를 일치시킨 조건에서 Titans는 $C=8$이 $C=256$보다 좋다(avg ppl 25.07 vs 27.13 [TNT Table 2]) — 신선한 gradient가 이긴다. throughput 쪽: 같은 $C=8$은 목표 loss 도달에 19.48시간이 걸려 가장 느린 baseline이다 [TNT Table 1]. [TNT §1]은 작은 chunk의 deep-memory 훈련이 peak FLOPs의 5-10% 미만 utilization에 머문다고 인용하는데(LaCT, Zhang et al. 2025 인용), 식 (10-2)는 그 이유를 그대로 보여 준다 — $C$가 작으면 AI가 낮아 bandwidth-bound이고, FLOPs가 모자란 게 아니라 intensity가 모자란 것이다. TNT의 답이 두 knob의 분리다: Stage 1은 계층 memory(큰 $C_{\mathrm{g}}$의 global + reset되는 local)로 훈련을 throughput-최적 지점에서 돌리고, Stage 2(추가 ~5% compute [TNT Table 4])가 chunk-1 decode를 품질-최적 지점으로 만든다. 결과가 headline 수치인 17.37× 빠른 목표-loss 도달과 그러면서도 개선된 perplexity(23.09 vs 25.07)다 [TNT Table 1, Table 2]. 32K 문맥에서는 custom kernel 없는 순수 JAX 구현이 FlashAttention 대비 step당 1.3× 빠르다는 보고도 있다 [TNT Fig. 4].

<!-- TODO-VERIFY: 32K에서 1.3× vs FlashAttention 수치의 정확한 출처가 Fig. 4인지 Fig. 5인지. 확인 방법: papers/2511.07343.txt에서 "FlashAttention"·"1.3" 검색 -->
<!-- TODO-VERIFY: "peak FLOPs 5-10% 미만" 인용이 TNT 본문 §1(Challenge 1)에 위치하는지, LaCT 인용 표기가 정확한지. 확인 방법: papers/2511.07343.txt에서 "utilization" 검색 -->

> **[평가]** stale-snapshot 근사가 품질에 주는 오차의 정량적 한계는 6편 어디에도 없다. 품질-최적 $C$가 작다는 것도, chunk-size mismatch가 위험하다는 것도 전부 경험적 관찰이다. 독자의 세계로 옮기면: 이 knob에는 아직 "numerics 문서"가 없다. tile 크기를 바꾸듯 $C$를 바꾸면 안 되는 이유는 알려져 있지만, 얼마나 바꾸면 얼마나 나빠지는지의 이론은 공백이다.

## 10.6 Worked micro-example: chunk 하나의 arithmetic intensity 손계산

식 (10-2)를 실제 숫자로 확인한다. 모든 값을 2의 거듭제곱으로 잡아 암산 가능하게 한다: TTT-Linear형 matrix memory, $d=64$, bf16(원소당 2 bytes), hardware는 H100급 1장 — bf16 dense peak ≈ 1000 TFLOP/s, HBM ≈ 3.35 TB/s, 따라서 ridge point ≈ 300 FLOP/byte로 놓는다(근사 스펙, 자기 장비 값으로 교체 가능).

<!-- TODO-VERIFY: H100 SXM bf16 dense peak(≈989 TFLOP/s)와 HBM3 bandwidth(3.35 TB/s) 수치 재확인. 확인 방법: NVIDIA H100 datasheet -->

**Step 1 — token 1개의 FLOPs.** delta rule 한 step은 forward $W k_t$($2d^2$), update $W \mathrel{-}= \eta_t(Wk_t-v_t)k_t^\top$($2d^2$), read $Wq_t$($2d^2$)로 $6d^2 = 6\times 4096 = 24{,}576$ FLOPs다.

**Step 2 — $C=1$ (decode)의 트래픽과 AI.** state RMW = $4d^2$ bytes = $16{,}384$ bytes. token IO = $8d = 512$ bytes. 합계 ≈ $16{,}896$ bytes. 따라서 $\mathrm{AI}(1) = 24{,}576 / 16{,}896 \approx 1.5$ FLOP/byte — 식 (10-2)의 $1.5C$와 일치한다. bandwidth-bound 달성 성능은 $3.35\,\mathrm{TB/s} \times 1.5 \approx 5$ TFLOP/s, 즉 **peak의 약 0.5%**다. [TNT §1]이 인용하는 "5-10% 미만"이 어디서 오는지 손끝에서 재현된다(실제 구현은 head·batch 축으로 kernel을 살찌우므로 이보다는 낫다).

**Step 3 — $C=64$의 AI.** FLOPs = $6Cd^2 = 64 \times 24{,}576 = 1{,}572{,}864$. 트래픽 = $4d^2 + 8Cd = 16{,}384 + 8\times64\times64 = 16{,}384+32{,}768 = 49{,}152$ bytes. $\mathrm{AI}(64) = 1{,}572{,}864/49{,}152 = 32$ FLOP/byte — $C=1$ 대비 정확히 $22\times$ 개선이고, 달성 성능 상한은 $3.35 \times 32 \approx 107$ TFLOP/s ≈ peak의 11%다.

**Step 4 — ceiling 확인.** $C\to\infty$ 극한은 $0.75d = 48$ FLOP/byte. 즉 $d=64$짜리 per-head memory는 **chunk를 아무리 키워도 ridge(≈300)에 못 미친다** — peak의 ~16%가 이 kernel 단독의 상한이다. 나머지 intensity는 $C$가 아니라 다른 축(head 병합, request 병합, TNT처럼 shard를 batch 축에 쌓기)에서 와야 한다. dual form의 intra-chunk 보정항($O(C^2d)$의 attention-like GEMM)은 FLOPs를 더해 AI를 다소 올리지만 결론을 바꾸지 않는다.

**Step 5 — 교훈.** $C$를 1→64로 키워 얻은 22×의 intensity는 공짜가 아니다 — (M4)의 gradient가 그만큼 stale해지고, §10.5의 [TNT Fig. 2]가 보여 준 대로 품질은 훈련 $C$에 과적합된다. 한 장의 계산으로 이 라인의 훈련 경제학 전체 — "왜 chunkwise인가, 왜 그런데도 아픈가, 왜 TNT가 필요한가" — 가 요약된다. 이 절의 숫자는 이 책의 설명용 계산이며 어떤 논문의 측정치도 아니다.

## 10.7 Glossary: Part II에 등장하는 training 용어의 systems 한 줄 사전

Part II를 읽다가 낯선 용어를 만나면 이 표에서 찾는다. 각 항목의 정의 소유 장을 병기했다 — 여기의 gloss는 환기용 한 줄이지 정의가 아니다.

표 10-4 — training 용어 → systems 한 줄 gloss

| 용어 | 장 | systems 한 줄 gloss |
|---|---|---|
| backward pass | 2 | forward의 역방향 VJP 사슬; FLOPs ≈ forward의 2×; decode에 들어오면 read 한 번이 fwd+bwd+update 세 번이 된다 |
| gradient ($\nabla_W\ell$) | 2 | 오차 감소 방향; 이 라인에선 $(\text{오차})\,k^\top$ 꼴의 rank-$C$ GEMM으로 구현되는 write 연산 |
| inner loss $\ell$ / outer loss $\mathcal{L}$ | 2 | memory write의 목적함수 / pretraining의 목적함수; 소문자·대문자로 loop를 구분 |
| learning rate ($\eta$, $\eta_t$) | 2 | update 크기; inner에선 data-dependent gate = write 강도 |
| momentum ($S_t$) | 2 | gradient의 EMA buffer; linear recurrence라서 scan kernel 대상; session state에 포함됨 |
| weight decay / retention gate ($\alpha_t$) | 2, 13 | state를 매 step $\alpha_t$배 하는 곱; **학습된 eviction policy** |
| optimizer state ($m_t,h_t$) | 2 | step 사이에 살아남는 buffer; 훈련의 register file |
| AdamW | 2 | 좌표별 learning rate + decoupled decay; [NL]이 "optimal elementwise $\ell_2$ memory"로 재해석 |
| Muon / $\mathrm{NS}_\kappa$ | 2, 14 | update 행렬의 semi-orthogonalization; 비용 = step당 $\kappa$회의 소형 GEMM |
| mini-batch | 2 | GEMM을 살찌우는 묶음 단위; **이 라인에선 sequence 축의 chunk가 그 역할** |
| online learning / OGD | 3 | 도착 즉시 predict-loss-update; autoregressive decode와 동형의 루프 |
| regret | 3 | 최적 고정 상태 대비 누적 손실 격차; "고정 크기 state가 oracle compressor에 얼마나 뒤지나"의 상한 |
| FTRL | 3 | 과거 loss 합 + regularizer의 argmin; retention gate의 이론적 모체 |
| inner / outer loop | 1, 4 | token마다 도는 write 루프 / pretraining 루프; 기호로는 $W$ vs $\Theta$ |
| bilevel optimization | 4 | inner 최적화를 관통해 outer를 최적화; fast weights는 parameter가 아니라 activation처럼 미분된다 |
| meta-learning / MAML / $W_{\mathrm{init}}$ | 4, 15 | 초기 상태를 outer가 학습; TNT shard의 시작점이자 reset의 착지점 |
| unrolling / hypergradient | 4 | inner 궤적을 펼쳐 미분; 펼친 그래프 = 컴파일·fuse 가능한 static dataflow |
| associative memory | 5 | $k\to v$ 매핑 저장 장치의 총칭; KV cache의 압축판 일반화 |
| Hebbian write / outer product | 5 | $W \mathrel{+}= v_tk_t^\top$; append-only 누적, crosstalk 발생 |
| delta rule | 5 | 예측 오차만큼만 overwrite; read-modify-write의 원형 |
| crosstalk / capacity | 5, 14 | 비직교 key 간 간섭 / state 크기 대비 저장 가능한 쌍의 수 |
| fast weights / slow weights | 6 | request마다 움직이는 $W$ / 고정 $\Theta$; batching을 가르는 경계 |
| linear attention | 6 | KV cache를 $d\times d$ state로 압축한 극한 |
| DeltaNet / Gated DeltaNet | 6 | delta rule을 sequence layer로; Householder형 transition |
| SSD (state-space duality) | 7 | SSM ≡ masked linear attention; scan 실행형과 GEMM 실행형의 등가 |
| TTT layer / dual form | 8 | state = 작은 net의 weights, update = GD step / chunk의 GD를 GEMM으로 tensorize한 실행 형태 |
| chunkwise-parallel training | 9 | chunk 안 병렬(GEMM) + chunk 간 순차(state 전달); 이 라인의 훈련을 성립시키는 scheme |
| stale-snapshot 근사 | 9 | gradient anchor를 chunk 시작 상태 $W_{\xi(t,C)}$에 고정; $C$를 semantic하게 만드는 원인 |
| semantic hyperparameter $C$ | 9 | $C$가 함수 자체를 바꿈; tile 크기(exact)와의 결정적 차이 |
| WY / UT transform | 9 | rank-1 곱 사슬을 chunk당 GEMM 두 개로; exact 재배열 |
| associative scan | 7, 9 | log-depth prefix 연산; momentum과 $\Pi_t$ 누적에 사용 |
| surprise (momentary / past) | 12 | inner gradient / momentum buffer의 [Titans] 명명 |
| deep memory | 12 | state가 2-layer residual MLP의 weights인 memory |
| persistent memory | 12 | 입력 앞에 붙는 학습된 token들; state가 아니라 $\Theta$의 일부 |
| MAC / MAG / MAL | 12 | memory와 attention의 결합 topology 세 종 |
| attentional bias | 13 | inner objective의 family; "무엇을 loss로 쓸 것인가" 축 |
| Omega rule / window gate $\gamma_{t,i}$ | 14 | 최근 $c$쌍의 window objective / 쌍별 가중 gate(in-context pruning) |
| test-time memorization | 14 | [Atlas]의 재명명: inner loop가 하는 일은 learning이 아니라 문맥의 기억 |
| chunk-size mismatch | 15 | train $C$ ≠ serve $C$면 품질 붕괴 [TNT Fig. 2]; Stage 2의 존재 이유 |
| hierarchical memory (global/local) | 15 | 큰 $C_{\mathrm{g}}$의 순차 global + reset되는 병렬 local의 합성 |
| periodic state reset / context parallelism | 15 | $L_{\mathrm{s}}$마다 $W_{\mathrm{init}}$로 복귀 → sequential chain 절단 → shard 병렬화 |
| Q-K projection ($\Pi_t$) | 15 | query를 관측된 key subspace로 사영하는 $d\times d$ 보조 state |
| two-stage training | 15 | 큰 chunk로 pretrain(throughput), 작은 chunk로 finetune(품질·decode 정렬) |
| update frequency / level | 16 | 컴포넌트별 갱신 주기; cache 계층의 시간축 유비 |
| Continuum Memory System (CMS) | 16 | 갱신 주기 스펙트럼 위에 배치된 memory 사슬; short/long 이분법의 일반화 |
| Local Surprise Signal (LSS) | 16 | layer 출력에 걸리는 국소 오차 신호 $u_t$; backprop의 재해석 단위 |
| self-modifying Titans | 16 | 자기 update rule의 구성 요소를 스스로 생성하는 memory |
| catastrophic forgetting | 11 | 새 데이터의 SGD가 옛 매핑을 파괴; shared weights의 간섭(= crosstalk의 훈련판) |
| complementary learning systems (CLS) | 11 | 빠른 hippocampus / 느린 cortex 이중계; 이 라인의 neuro 유비의 해독기 |
| consolidation | 11, 16, 17 | fast state의 내용을 slow weights로 옮기는 작업; serving 관점에선 background compaction job |
| knowledge distillation / GKD | 11 | teacher 분포로 student를 훈련; GKD는 student 자신의 출력 위에서 하는 on-policy 변형 |
| replay | 11 | 과거 데이터 재주입으로 forgetting 완화; data-plane artifact |
| wake/sleep lifecycle | 17 | 온라인 서빙 phase / 주기적 offline 통합 phase의 교대 |
| Knowledge Seeding (KS/SKS) / Dreaming | 17 | 작은 fast memory → 큰 slow net으로의 상향 distillation / RL이 만드는 합성 커리큘럼 |

## 요약

- 이 라인의 decode 비용은 세 숫자로 환원된다: state 원소 수 $s$, token당 FLOPs, token당 state RMW bytes($4s$). 고정-state 모델의 decode는 예외 없이 bandwidth-bound이며, bandwidth-bound 영역에서 FLOPs 계수 차이(linear attention 4 vs Titans 12)는 사실상 공짜다 (표 10-2).
- KV cache vs fast weights의 memory 승부는 식 (10-1) $L^\ast = s/2d$ 한 줄로 판정된다: matrix memory는 $d/2$, 표준 deep memory는 $4d$, momentum 포함 시 $8d$ token부터 이긴다. 짧은 문맥에서는 fast weights가 진다.
- chunk 크기 $C$는 roofline 위의 독립변수다: 식 (10-2)에 따라 arithmetic intensity는 $C$에 선형으로 오르다 state 크기가 정하는 ceiling($0.75d$ 또는 $\sim P_{\mathcal{M}}/d$)에서 포화한다.
- FlashAttention tiling은 같은 함수의 exact 재배열이고, (M4)의 $C$는 계산되는 함수를 바꾸는 semantic knob이다 — [TNT Fig. 2]의 mismatch 곡선(13.78 vs 36.45)이 그 실증이다. 단 linear-state 클래스의 chunkwise는 exact다.
- 품질-최적 $C$(작음)와 MFU-최적 $C$(큼)는 반대 방향이며, TNT는 계층 memory + 두 단계 훈련으로 두 knob을 분리해 17.37× 빠른 훈련과 개선된 perplexity를 동시에 얻는다 [TNT Table 1, Table 2].
- backward-in-decode가 serving engine에 요구하는 것은 네 가지다: 손으로 유도한 closed-form gradient kernel, per-session state라는 새 cache class(checkpoint/restore·prefix 분기·학습된 eviction), grouped-GEMM decode(batching이 intensity를 못 올림), optimizer-trajectory까지 포함하는 rollback.
- 6편 모두 decode wall-clock을 보고하지 않으므로, Part II의 serving 관련 주장은 이 장의 도구로 각자 재계산하는 것이 기본자세다.

## 자가 점검 체크리스트

- [ ] (M1)–(M4) 형태의 임의 layer에 대해 decode FLOPs/token과 state RMW bytes/token을 지배항 수준에서 유도할 수 있다.
- [ ] 식 (10-1)로 자기 서빙 config의 KV-cache-vs-state crossover 길이를 계산할 수 있다.
- [ ] 식 (10-2)로 $\mathrm{AI}(C)$를 계산하고, 자기 hardware의 ridge point와 비교해 어느 $C$부터 compute-bound가 되는지(혹은 영영 안 되는지) 판정할 수 있다.
- [ ] FlashAttention tiling과 chunkwise training의 차이를 "exact 재배열 vs semantic 근사" 두 문장으로 설명하고, 이 구분이 어느 아키텍처 클래스부터 적용되는지 말할 수 있다.
- [ ] backward pass가 decode에 들어올 때 serving engine에 생기는 네 가지 요구 사항을 나열할 수 있다.
- [ ] Part II 임의의 장의 §"systems 함의" 절을 읽고 그 주장을 표 10-2의 행과 식 (10-1)·(10-2)로 환원해 검산할 수 있다 — 즉 이 라인의 어휘를 inference 어휘로 옮길 수 있다.

## 다음 장으로

이 장까지의 cost model은 전부 한 세션, 한 inner loop의 이야기였다: request가 시작되면 $W_t$가 움직이고, request가 끝나면 버려진다. 그런데 [NL]과 [Sleep]은 그 그림을 확장한다 — update가 일어나는 주기를 token 단위부터 pretraining 단위까지 연속 스펙트럼으로 펼치고(Continuum Memory System), 세션이 끝난 뒤 fast state의 내용을 slow weights로 옮기는 offline 단계(sleep, consolidation)를 lifecycle에 넣는다. 이것을 읽으려면 지금까지 다루지 않은 마지막 배경이 필요하다: 왜 새 데이터로 update하면 옛 지식이 부서지는가(catastrophic forgetting), 뇌는 왜 빠른 기억계와 느린 기억계를 분리했는가(complementary learning systems), 그리고 한 모델의 지식을 다른 모델로 옮기는 표준 도구(distillation, replay, RL-lite)는 무엇인가. 11장이 이 마지막 조각을 채우고, 그로써 Part II로 들어갈 준비가 끝난다.
