# ch10. 통합 systems bridge: cost model cheat sheet, roofline vs chunk size, glossary

> **이 장의 목표** — (1) (M1)–(M4) 형태의 임의 layer에 대해 decode FLOPs/token·state 트래픽·state 크기를 유도한다. (2) chunk 크기 $C$를 roofline 독립변수로 놓고 품질-최적 $C$와 MFU-최적 $C$가 갈라지는 이유를 숫자로 보인다. (3) KV cache vs fast-weight state의 memory-budget 승부를 자기 config로 판정한다. (4) Part II의 efficiency 주장을 이 cheat sheet로 환원해 검산한다.
>
> **왜 필요한가** — [TNT] (*Improving Chunkwise Training for Test-Time Memorization*, arXiv:2511.07343)는 사실상 이 장의 언어로 쓰인 논문이다: Challenge 1(peak FLOPs 5-10% 미만)·Challenge 3(chunk-size mismatch)은 roofline 없이는 판독되지 않는다. [Titans §3.2]·[Atlas §3.4]의 병렬화 주장도 이 도구로 audit하며, Part II 전 장의 §"systems 함의"는 표 10-2·10-4를 참조 대상으로 삼는다.

나머지 논문의 축약: [Titans] (*Titans: Learning to Memorize at Test Time*, arXiv:2501.00663), [Atlas] (*Atlas: Learning to Optimally Memorize the Context at Test Time*, arXiv:2505.23735), [NL] (*Nested Learning*, arXiv:2512.24695), [Sleep] (*Language Models Need Sleep*, arXiv:2606.03979).

Part I 각 장이 말미에 깐 systems bridge 조각을 하나의 분석 도구 상자로 통합한다. 6편 모두 "hardware에 더 잘 맞는다"고 주장하지만 측정은 훈련 쪽에 몰리고 decode 쪽은 비어 있다(6편 전체에 decode wall-clock 부재; serving 주장은 전부 구조적 논증이지 측정이 아니다) — 논문 숫자를 받아 적는 대신 이 장의 cost model 위에서 재계산하는 것이 현재 그 비용을 알 유일한 길이다.

## 10.1 완성된 Rosetta-Stone 사전

1장의 inference↔learning 사전에서 상세 해설은 표 1-1(→ 1장)이 소유하므로, 여기서는 이 장이 새로 확립하는 행과 직관을 이식하면 틀리는 ⚠ 지점만 **차분표**로 싣는다. 나머지 대응은 표 1-1을 보라.

표 10-1 — 이 장이 새로 확립하는 대응 + ⚠ 오이식 지점 (나머지 전체 대응·해설은 표 1-1(→ 1장))

| inference 세계 (독자의 어휘) | 이 라인의 어휘 | 확립한 장 |
|---|---|---|
| roofline / arithmetic intensity | chunk 크기 $C$ = 새 독립변수 | 이 장 |
| batching (shared weights 전제) | ⚠ per-request fast-weight state → grouped-GEMM decode | 이 장 |
| paged KV cache / session cache | per-session weight state — 새로운 cache class | 이 장 |
| speculative decoding의 rollback | memory state snapshot/rollback | 이 장 |

⚠ 세 지점은 inference 직관을 그대로 이식하면 틀린다: FlashAttention의 tile과 달리 chunk $C$는 계산되는 함수를 바꾸고(semantic, §10.5), fast-weight state는 shared-weight batching의 전제를 깨며(§10.4), 이 라인의 "mini-batch"는 batch 축이 아니라 sequence 축의 chunk다(→ 2·9장).

## 10.2 Cost-model cheat sheet

아키텍처 클래스별 비용을 한 표로 모은다. 회계 규약(표 10-2와 이후 모든 계산에 공통):

- **범위**: layer 1개·token 1개(decode 기준). head 병렬·batch 축·projection 비용은 전 행 공통이라 생략.
- **FLOPs**: MAC 1회 = 2 FLOPs; backward pass ≈ forward의 2× (→ 2장); 지배항만 남긴다.
- **bytes**: bf16 원소당 2 bytes. **state RMW 트래픽** = state를 읽고(read) 갱신해(modify) 다시 쓰는(write) 왕복 — 원소 수 $s$당 $4s$ bytes.
- **기호**: $L$ = 문맥 길이, $w$ = sliding window, $d$ = 차원($d_k=d_v=d$), $c$ = Omega rule window, $P_{\mathcal{M}}$ = deep memory parameter 수(표준이면 $8d^2$), $N$ = TNT local memory 개수.

표 10-2 — 아키텍처 클래스별 cost-model cheat sheet (layer당, token당, 지배항)

| 아키텍처 | state (원소 수) | decode FLOPs/token | state 트래픽 (bytes/token) | 병렬화 형태와 MFU 결정 요인 |
|---|---|---|---|---|
| softmax attention + KV cache | $2Ld$ (**증가**) | $\approx 4Ld$ | $4Ld$ 읽기 + $4d$ append | $O(L^2 d)$ GEMM; tiling은 exact; decode는 cache 재읽기 지배 |
| sliding-window attention (SWA) | $2wd$ | $\approx 4wd$ | $4wd$ | 동일하되 window-local; $L$과 무관 |
| linear attention (un-gated) | $\approx d^2$ | $\approx 4d^2$ | $4d^2$ RMW | chunkwise-GEMM(**exact**, → 9장) 또는 scan; $C$는 tile성 knob |
| GLA / Mamba-2 | $d^2$ (+gate 소량) | $\approx 5d^2$ | $4d^2$ RMW | SSD block 분해·2-level tiling(exact); scan형은 bandwidth-bound (→ 7장) |
| DeltaNet / Gated DeltaNet | $d^2$ | $\approx 6d^2$ | $4d^2$ RMW | WY/UT로 chunkwise-GEMM(**exact**, → 9장); Householder 사슬이 kernel 복잡도 결정 |
| TTT-Linear | $d^2$ | $\approx 6d^2$ | $4d^2$ RMW | dual form chunkwise-GEMM — 단 **semantic**: $C$가 함수를 바꾼다 (M4) |
| TTT-MLP | $P_{\mathcal{M}}$ | $\approx 8$–$10\,P_{\mathcal{M}}$ | $4P_{\mathcal{M}}$ RMW | dual form(semantic) + 비선형 inter-chunk 의존이 직렬화 강제 [TNT §1] |
| Titans-LMM | $2P_{\mathcal{M}}$ ($W_t,S_t$) | $\approx 12\,P_{\mathcal{M}}$ | $8P_{\mathcal{M}}$ RMW | chunkwise(semantic) + momentum scan + token-dependent gate [Titans §3.2] (chunk-상수 gate는 원문이 제시한 미검증 속도 최적화) |
| Atlas (OmegaNet 계열) | $2P_{\mathcal{M}}+2cd$ | (M3) 그대로면 $\approx 6c\,P_{\mathcal{M}}$ + $\mathrm{NS}_\kappa$ 상각분 | $8P_{\mathcal{M}}+4cd$ | Omega rule 병렬화 [Atlas §3.4]; $\mathrm{NS}_\kappa$ = chunk당 $\kappa$회 소형 GEMM |
| TNT (global + $N$ local) | $(1{+}N)P_{\mathcal{M}}+Nd^2$ | $\approx 2P_{\mathcal{M}} + N(8P_{\mathcal{M}}{+}4d^2)$ (global write는 $C_{\mathrm{g}}$로 상각) | global read $2P_{\mathcal{M}}$ + local RMW $4NP_{\mathcal{M}}$ + $\Pi$ $4Nd^2$ (global write RMW는 $C_{\mathrm{g}}$로 상각) | global = 순차 큰 GEMM, local = reset로 shard 병렬 [TNT §4.1] |

**KV cache 클래스**(1–2행). state가 자라고 decode 비용이 $L$에 비례하는 유일한 클래스 — 나머지 여덟 행의 대조군이다.

**matrix-state 클래스**(3–6행). state가 $d^2$ 고정이고 decode의 본질은 state RMW다(Mamba-2의 SSM state 차원 $N$은 자유 파라미터라 실제 state $d_v\times N$은 $d^2$보다 작을 수 있으나 표는 통일 규약값). token당 GEMV 몇 개 + rank-1 갱신뿐이라 FLOPs/byte가 한 자릿수(§10.6)이므로 decode는 예외 없이 bandwidth-bound다 — 승부처는 "state $4d^2$ bytes를 매 token 왕복시키는가". linear→GLA→DeltaNet으로 계수가 4→5→6이 되어도 트래픽은 동일: bandwidth-bound에서 **계수 큰 FLOPs는 공짜**라, retention gate·delta rule의 품질 이득이 decode에서 거의 무료다.

**deep-memory 클래스**(7–9행). state가 작은 신경망 weights 전체($P_{\mathcal{M}}$)가 되고 write가 forward+backward+update가 된다: compression $\approx 6P_{\mathcal{M}}$ + retrieval $2P_{\mathcal{M}}$ = 계수 8–10. Titans-LMM은 momentum·retention이 얹혀 $\approx 12P_{\mathcal{M}}$이고 $S_t$도 session state라 **state가 두 배**다. Atlas 행은 경고 표지판이다: 식 (M3)을 문자대로 실행하면 decode FLOPs가 $c$배로 뛰고 최근 $c$쌍 buffer($2cd$ 원소)가 state에 붙는다 — 이 상각이 [Atlas §3.4]의 주제다(→ 14장). 게다가 Muon의 $\mathrm{NS}_\kappa$(반복마다 행렬-행렬 곱)는 $C=1$ decode에 상각할 chunk 축이 없어 이 행만은 compute-bound가 될 수 있다 [Atlas Eq. 40–41]. $c$ 스윕 $\{2,4,8,16\}$($c$ 클수록 ppl 낮음)의 headline 값·decode window 유지 절차는 원문에 명시가 없다 [Atlas Fig. 5, Table 6].

**TNT 행**(10행). 위 회계의 합성이다: global write는 $C_{\mathrm{g}}$ token마다 한 번으로 상각되고, 매 token 비용은 $N$개 local의 fwd+bwd+read + 읽기 전용 global read + $N$개 $\Pi_t$ 갱신이라 FLOPs·트래픽 모두 $N$에 비례한다. state가 문맥 길이와 무관한 상수라는 것이 [TNT]의 serving 논지다.

## 10.3 KV cache vs fast weights: memory-budget 산수

"고정 크기 state니까 이긴다"는 절반만 참이다 — 언제부터 이기는지는 나눗셈 하나로 판정된다. layer당 state 원소 수를 $s$, KV cache는 token당 $2d$ 원소를 쌓으므로 crossover 문맥 길이는

$$
L^{\ast} \;=\; \frac{s}{2d}
\tag{10-1}
$$

이다(layer 수·정밀도 소거). matrix memory($s=d^2$)는 $L^\ast=d/2$, 표준 deep memory($8d^2$)는 $4d$, momentum 포함($2P_{\mathcal{M}}$)은 $8d$. 아래는 $d=2048$·24 layer·bf16·GQA 없는 MHA·전 layer memory인 보수적 config다 — 자기 스택 값으로 갈아 끼우는 것이 목적이다.

표 10-3 — memory budget: KV cache vs fast-weight state ($d=2048$, 24 layers, bf16)

| 항목 | 4K 문맥 | 64K 문맥 | 1M 문맥 | 비고 |
|---|---|---|---|---|
| KV cache (전체) | 768 MiB | 12 GiB | 192 GiB | token당 192 KiB씩 선형 증가 |
| matrix memory ($d^2$/layer) | 192 MiB | 192 MiB | 192 MiB | $L^\ast=d/2=1024$ tokens |
| deep memory ($8d^2$/layer) | 1.5 GiB | 1.5 GiB | 1.5 GiB | $L^\ast=4d=8192$ tokens |
| deep memory + momentum | 3 GiB | 3 GiB | 3 GiB | $L^\ast=8d=16384$ tokens |

<!-- FIG: ch10/fig-01-state-budget -->

첫째, **fast weights는 짧은 문맥에선 진다**: deep memory + momentum 3 GiB ≈ 16K token KV cache라, 4K 세션만 서빙하는 fleet엔 Titans류가 budget 손해다 — 이 라인이 long-context를 앞세우는 것은 취향이 아니라 산수다. 둘째, **1M에선 두 자릿수 배율로 갈린다**(192 GiB vs 3 GiB): paged·quantization·eviction으로도 $O(L)$ 성장 자체는 못 없앤다 — 성장 곡선을 상수로 접는 대신 §10.4의 새 serving 문제로 대가를 내는 거래다.

따름정리: 64K softmax decode는 매 token cache 12 GiB를 재읽어 HBM 3 TB/s급에서 4 ms/token. deep memory + momentum의 state RMW 6 GiB는 **전체 합**이고 문맥과 무관하다. 긴 문맥에서 이 라인이 파는 것은 FLOPs가 아니라 bandwidth 절감이다.

## 10.4 Prefill/decode 비대칭과 serving engine: backward pass가 들어온다

prefill/decode 비대칭은 물려받되 내용물이 바뀐다. **prefill = 큰 chunk의 병렬 write**(프롬프트를 chunk로 묶어 (M4)의 batched gradient를 GEMM으로 흘리는 compute-bound 구간), **decode = $C=1$의 online write + read**(token마다 작은 net의 fwd·bwd·update·read). [TNT §4.2]는 이 대응이 설계 목표다: global memory($C_{\mathrm{g}}=2048$)가 prefill을, Stage 2에서 $C_{\mathrm{l}}'=1$로 적응시킨 local이 decode를 맡고, prefill 시 local은 reset 덕에 $L/L_{\mathrm{s}}$개 shard로 병렬 채워진다.

backward pass가 decode에 들어온다는 사실이 serving engine에 요구하는 변화는 네 가지다(Part III에서 재론; 여기서는 비용 구조만).

**1. autograd 없이 gradient를 계산해야 한다.** serving engine은 autograd graph를 안 싣지만 이 라인의 inner gradient는 전부 닫힌 형태다: linear memory는 $\nabla_W\ell = 2(Wk_t-v_t)k_t^\top$(GEMV+뺄셈+outer product), deep memory도 손유도 GEMM 서너 개. "backward"는 **shape가 알려진 GEMM들을 fused RMW kernel로 굳히는 일** — 훈련 프레임워크가 아니라 FlashAttention을 짜던 방식이다.

**2. per-session state가 새로운 cache class가 된다.** 크기는 표 10-3이 해결하고, paged KV cache(Kwon et al. 2023, arXiv:2309.06180)와 달리 고정 크기라 allocator는 단순하지만 새 문제 셋이 생긴다. (a) **checkpoint/restore**: $W_t$($S_t,\Pi_t$ 포함)를 통째로 저장하거나 프롬프트를 replay — TNT의 periodic reset은 **local** 재생 분량을 $L_{\mathrm{s}}$ token으로 유계화하나 reset 안 되는 global $W^{\mathrm{g}}$는 여전히 snapshot/replay가 필요하다. (b) **prefix 재사용**: state가 전체 history의 함수라 중간 편집 불가·snapshot 분기만 가능하되, 고정 크기라 복사는 KV cache보다 싸다. (c) **eviction**: 손으로 짜던 policy가 모델 안 retention gate $\alpha_t$라는 학습된 정책이 된다(→ 3·13장); fleet 수준 eviction은 엔진 몫이다.

**3. shared-weight batching이 깨진다.** $B$개 request가 같은 weights를 읽어 트래픽을 상각한다는 전제가 무너진다: request마다 자기 $W_t^{(r)}$가 있어 read/write가 $B$개의 서로 다른 GEMV/RMW — multi-LoRA의 grouped-GEMM 패턴이다. state 트래픽이 $B$에 비례해 자라 **batching이 intensity를 못 올린다** — compute-bound 탈출로가 막힌다. (head·layer 병렬성은 남고, shard를 batch 축에 쌓는 훈련 트릭도 있다 [TNT §4.1].)

**4. rollback이 optimizer-trajectory 문제가 된다.** speculative decoding에서 draft 기각 시 KV cache는 꼬리만 자르면 되지만 fast-weight 모델은 $W_t$·$S_t$·$\Pi_t$ 궤적 전체를 되돌려야 한다. snapshot은 싸지만 "몇 token마다 어느 buffer까지" 찍을지는 6편 어디에도 답이 없는 열린 문제다.

## 10.5 FlashAttention tiling vs chunkwise training: roofline 위의 chunk 크기

FlashAttention(Dao et al. 2022, arXiv:2205.14135; FA-2는 Dao 2023, arXiv:2307.08691)의 tiling은 **같은 합의 순서 재배열**이라 tile 크기는 SRAM용 구현 knob이고 출력은 동일하다. 반면 chunkwise의 $C$는 (M4)대로 **gradient 평가 지점을 바꾼다**: 모든 gradient가 chunk 시작 $W_{\xi(t,C)}$에서 평가되는 stale-snapshot 근사라 $C$가 다르면 다른 함수가 나온다 — tile이 아니라 **semantic hyperparameter**다(→ 9장). 경계는 $\mathcal{M}$의 선형성이 아니라 **정의 방식**이다: 명시적 linear recurrence를 대수적으로 재배열하는 linear attention/GLA/DeltaNet은 chunkwise가 exact지만, gradient 평가점을 $W_\xi$에 고정하는 anchored mini-batch GD로 **정의된** 모델은 linear여도(TTT-Linear) $C$가 semantic이다 — §8.8이 linear memory에서 $C{=}1$은 4, $C{=}2$는 6을 낸 것이 증거다(→ 표 9-2).

실증은 [TNT Fig. 2]: $C=64$로 pre-train한 550M Titans를 inference chunk만 바꾸면 perplexity가 8/16/32/64/128/256/512에서 36.45/34.15/24.23/13.78/15.5/17.88/22.4 — 훈련 chunk에서 최적이고 양쪽으로 무너진다. tile을 바꿔 품질이 2.6× 나빠지는 kernel은 없다 — 이 그림이 "$C$는 tile이 아니다"의 증명이다(단일 설정 한계는 15장).

이제 $C$를 roofline(Williams, Waterman & Patterson, CACM 2009) 위에 놓는다. delta 계열 chunk 하나를 fused kernel이 처리하면 chunk당 FLOPs = state 처리(계수 6) $\approx 6Cd^2$ + chunk 내부 $C\times C$ attention형 항(dual form (8-4)) $\approx 4C^2d$, 트래픽 = state RMW $4d^2$ + token IO($k,v,q,y$) $8Cd$ bytes:

$$
\mathrm{AI}(C) \;=\; \frac{4C^2d + 6\,C\,d^{2}}{4d^{2} + 8Cd}\ \left[\mathrm{FLOP/byte}\right]
\tag{10-2}
$$

두 극한: $C\ll d$면 intra-chunk 항이 무시돼 $\mathrm{AI}\approx 1.5C$(decode 영역), $C\gg d$면 $4C^2d$ 항이 지배해 $\mathrm{AI}\approx 0.5C$ — **둘 다 $C$에 따라 자라 bandwidth 천장이 없고**, $C=L$ 극한에서 총 FLOPs $O(L^2d)$의 순수 attention(compute-bound)으로 이어진다. 작은 $d$의 head도 $C$를 충분히 키우면 원리적으론 ridge를 넘지만 그 $C$는 gradient를 파괴적으로 stale하게 만든다(§10.6). deep memory는 state 항이 $\sim P_{\mathcal{M}}/d$로 커지는 데다 비선형 inter-chunk 의존이 병렬화를 막는다 [TNT §1]. 세 영역($C=1$ recurrent = bandwidth-bound, $1<C<L$ chunkwise = ridge로 이동, $C=L$ 병렬 = compute-bound·$O(L^2d)$)은 하나의 roofline 위 연속 보간이고 $C$가 보간 위치다.

<!-- FIG-REF: ch09/fig-02-three-regimes -->

여기서 근본 긴장이 나온다. **품질-최적 $C$는 작고 MFU-최적 $C$는 크다.** 품질: chunk를 일치시키면 $C=8$이 $C=256$보다 낫다(avg ppl 25.07 vs 27.13 [TNT Table 2]) — 신선한 gradient가 이긴다. throughput: 같은 $C=8$은 목표 loss까지 19.48시간으로 가장 느린 baseline이다 [TNT Table 1]. [TNT §3]의 "작은 chunk 훈련은 peak FLOPs 5-10% 미만"(Zhang, Bi et al. 2025 = LaCT)의 이유가 식 (10-2)다 — FLOPs가 아니라 intensity 부족이다. TNT의 답은 두 knob 분리다: Stage 1은 계층 memory(큰 $C_{\mathrm{g}}$ global + reset local)로 훈련을 throughput-최적점에서 돌리고, Stage 2(추가 약 5–8% compute, 4-local 약 8.3% [TNT Table 4])가 chunk-1 decode를 품질-최적점으로 만든다. 결과는 서로 다른 구성의 두 최고치 — 최대 17.37× 빠른 목표-loss 도달({64} [TNT Table 1])과 다중-local Stage 2({2,4,8,16})의 평균 ppl 23.09(best Titans 25.07 대비 [TNT Table 2]).

> **[평가]** stale-snapshot 근사 오차의 정량적 한계는 6편 어디에도 없다. 품질-최적 $C$가 작다는 것도 mismatch가 위험하다는 것도 전부 경험적 관찰이라 이 knob엔 아직 "numerics 문서"가 없다.

## 10.6 Worked micro-example: chunk 하나의 arithmetic intensity 손계산

식 (10-2)를 숫자로 확인한다(모든 값 2의 거듭제곱): TTT-Linear형 matrix memory, $d=64$, bf16, H100급 1장 — bf16 peak 989 TFLOP/s·HBM3 3.35 TB/s [NVIDIA H100 SXM datasheet]를 1000 TFLOP/s로 근사, ridge ≈ 300 FLOP/byte(장비 값으로 교체 가능).

**Step 1 — token 1개 FLOPs.** delta rule 한 step = forward $Wk_t$($2d^2$) + update $W\mathrel{-}=\eta_t(Wk_t-v_t)k_t^\top$($2d^2$) + read $Wq_t$($2d^2$) = $6d^2 = 6\times4096 = 24{,}576$ FLOPs.

**Step 2 — $C=1$(decode).** state RMW $4d^2=16{,}384$ + token IO $8d=512$ = 합계 ≈ $16{,}896$ bytes. $\mathrm{AI}(1)=24{,}576/16{,}896\approx 1.5$ FLOP/byte — 식 (10-2)의 $1.5C$와 일치. 달성 성능 $3.35\,\mathrm{TB/s}\times1.5\approx 5$ TFLOP/s = **peak 약 0.5%**. [TNT §3]의 "5-10% 미만"이 손끝에서 재현된다(실제 구현은 head·batch 축으로 살찌워 이보다 낫다).

**Step 3 — $C=64$의 AI.** FLOPs = intra-chunk $4C^2d$ + state $6Cd^2 = 4\cdot262{,}144+6\cdot262{,}144 = 2{,}621{,}440$($C=d=64$이라 두 항이 같은 크기 — intra-chunk를 빠뜨리면 안 되는 이유). 트래픽 $4d^2+8Cd = 16{,}384+32{,}768 = 49{,}152$ bytes. $\mathrm{AI}(64)\approx 53$ FLOP/byte — $C=1$ 대비 약 $36\times$, 성능 상한 $3.35\times53\approx 178$ TFLOP/s ≈ peak 18%다.

**Step 4 — 큰 $C$.** $\mathrm{AI}$는 큰 $C$에서 $\approx 0.5C$로 계속 자라(bandwidth 천장 없음) $d=64$에서 ridge($\approx300$)를 넘기려면 $C\approx600$이 필요한데, 그만한 $C$는 gradient를 파괴적으로 stale하게 만든다 [TNT Fig. 2]. MFU를 얻는 현실적 경로는 $C$가 아니라 **다른 축**(head·request 병합, shard를 batch 축에 쌓기)이다.

**Step 5 — 교훈.** 1→64의 intensity 이득은 공짜가 아니다: (M4) gradient가 그만큼 stale해져 품질이 훈련 $C$에 과적합된다 [TNT Fig. 2]. 이 계산이 "왜 chunkwise인가, 왜 아픈가, 왜 TNT인가"를 요약한다. 숫자는 설명용이며 논문 측정치가 아니다.

## 10.7 Glossary: Part II에 등장하는 training 용어의 systems 한 줄 사전

Part II에서 낯선 용어를 만나면 이 표에서 찾는다. 정의 소유 장을 병기했다 — gloss는 환기용 한 줄이지 정의가 아니다.

표 10-4 — training 용어 → systems 한 줄 gloss

| 용어 | 장 | systems 한 줄 gloss |
|---|---|---|
| backward pass | 2 | 역방향 VJP 사슬; FLOPs ≈ forward 2× |
| gradient ($\nabla_W\ell$) | 2 | 오차 감소 방향; $(\text{오차})k^\top$ write |
| inner $\ell$ / outer $\mathcal{L}$ loss | 2 | write / pretraining 목적함수 |
| learning rate ($\eta$, $\eta_t$) | 2 | update 크기; inner에선 write 강도 gate |
| momentum ($S_t$) | 2 | gradient EMA; session state |
| weight decay / retention gate ($\alpha_t$) | 2, 13 | 매 step state $\times\alpha_t$; **학습된 eviction** |
| optimizer state ($m_t,h_t$) | 2 | step 간 살아남는 buffer; 훈련 register file |
| AdamW | 2 | 좌표별 lr + decoupled decay |
| Muon / $\mathrm{NS}_\kappa$ | 2, 14 | update 행렬 semi-orthogonalization; $\kappa$회 소형 GEMM |
| mini-batch | 2 | GEMM 살찌우는 묶음; **여기선 sequence 축 chunk** |
| online learning / OGD | 3 | 도착 즉시 predict-loss-update; decode와 동형 |
| regret | 3 | 최적 고정 상태 대비 누적 손실 격차 |
| FTRL | 3 | 과거 loss 합 + regularizer argmin; retention의 모체 |
| inner / outer loop | 1, 4 | write 루프 / pretraining 루프 ($W$ vs $\Theta$) |
| bilevel optimization | 4 | inner를 관통해 outer 최적화 |
| meta-learning / MAML / $W_{\mathrm{init}}$ | 4, 15 | 초기 상태를 outer가 학습; TNT reset의 착지점 |
| unrolling / hypergradient | 4 | inner 궤적 펼쳐 미분; fuse 가능 static dataflow |
| associative memory | 5 | $k\to v$ 매핑 저장 장치 |
| Hebbian write / outer product | 5 | $W\mathrel{+}=v_tk_t^\top$; append→crosstalk |
| delta rule | 5 | 오차만큼만 overwrite; RMW의 원형 |
| crosstalk / capacity | 5, 14 | 비직교 key 간섭 / 저장 쌍 수 |
| fast / slow weights | 6 | 움직이는 $W$ / 고정 $\Theta$; batching 경계 |
| linear attention | 6 | KV cache를 $d\times d$로 압축한 극한 |
| DeltaNet / Gated DeltaNet | 6 | delta rule의 layer화; Householder transition |
| SSD (state-space duality) | 7 | SSM ≡ masked linear attention; scan·GEMM 등가 |
| TTT layer / dual form | 8 | state = 작은 net weights, update = GD step / 그 GEMM형 |
| chunkwise-parallel training | 9 | chunk 안 병렬 + chunk 간 순차(state 전달) |
| stale-snapshot 근사 | 9 | gradient anchor를 chunk 시작 $W_{\xi(t,C)}$에 고정 |
| semantic hyperparameter $C$ | 9 | $C$가 함수를 바꿈; tile(exact)과의 차이 |
| WY / UT transform | 9 | rank-1 사슬 → chunk당 GEMM 2개; exact |
| associative scan | 7, 9 | log-depth prefix; momentum·$\Pi_t$ 누적 |
| surprise (momentary / past) | 12 | inner gradient / momentum buffer [Titans] |
| deep memory | 12 | state가 2-layer residual MLP weights |
| persistent memory | 12 | 입력 앞 학습된 token; $\Theta$의 일부 |
| MAC / MAG / MAL | 12 | memory·attention 결합 topology 세 종 |
| attentional bias | 13 | inner objective의 family — "무엇을 loss로" |
| Omega rule / window gate $\gamma_{t,i}$ | 14 | 최근 $c$쌍 window objective / 쌍별 가중 gate |
| test-time memorization | 14 | [Atlas] 재명명: learning 아닌 문맥 기억 |
| chunk-size mismatch | 15 | train $C$ ≠ serve $C$면 품질 붕괴 [TNT Fig. 2] |
| hierarchical memory (global/local) | 15 | 순차 global(큰 $C_{\mathrm{g}}$) + reset 병렬 local |
| periodic reset / context parallelism | 15 | $L_{\mathrm{s}}$마다 $W_{\mathrm{init}}$ 복귀 → chain 절단 → shard 병렬 |
| Q-K projection ($\Pi_t$) | 15 | key subspace로 사영하는 $d\times d$ 보조 state |
| two-stage training | 15 | 큰 chunk pretrain(throughput) + 작은 chunk finetune(품질) |
| update frequency / level | 16 | 컴포넌트별 갱신 주기; cache 계층의 시간축 유비 |
| Continuum Memory System (CMS) | 16 | 갱신 주기 스펙트럼 위의 memory 사슬 |
| Local Surprise Signal (LSS) | 16 | layer 출력의 국소 오차 신호 $u_t$ |
| self-modifying Titans | 16 | 자기 update rule을 스스로 생성하는 memory |
| catastrophic forgetting | 11 | 새 SGD가 옛 매핑 파괴 (crosstalk의 훈련판) |
| complementary learning systems (CLS) | 11 | 빠른 hippocampus / 느린 cortex 이중계 |
| consolidation | 11, 16, 17 | fast state → slow weights; background compaction |
| knowledge distillation / GKD | 11 | teacher 분포로 student 훈련; GKD는 on-policy |
| replay | 11 | 과거 데이터 재주입으로 forgetting 완화 |
| wake/sleep lifecycle | 17 | 온라인 서빙 / 주기적 offline 통합 phase의 교대 |
| Knowledge Seeding (KS/SKS) / Dreaming | 17 | fast→slow distillation / RL 합성 커리큘럼 |
| MoE / router | 이 장 | expert 부분망 일부 실행 / 선택 gate; [Sleep] parameter expansion 재료 (→ 17장) |

## 요약

- decode 비용은 세 숫자다: state 원소 수 $s$, token당 FLOPs, state RMW bytes($4s$). 고정-state decode는 대체로 bandwidth-bound(예외: Atlas/Muon의 $\mathrm{NS}_\kappa$는 compute-bound 가능)이고, 그 영역에서 FLOPs 계수 차이(4 vs 12)는 공짜다 (표 10-2).
- memory 승부는 식 (10-1) $L^\ast=s/2d$: matrix $d/2$, 표준 deep memory $4d$, momentum 포함 $8d$ token부터 이긴다 — 짧은 문맥에선 fast weights가 진다.
- chunk $C$는 roofline의 독립변수다: AI가 작은 $C$에서 $\approx1.5C$, intra-chunk $O(C^2d)$ 때문에 큰 $C$에서도 $\approx0.5C$로 계속 자라 $C=L$의 compute-bound($O(L^2d)$)로 이어진다 — bandwidth 천장 없음 (식 10-2).
- FlashAttention tiling은 exact 재배열, (M4)의 $C$는 semantic knob — [TNT Fig. 2]의 mismatch(13.78 vs 36.45)가 실증. 경계는 선형성이 아니라 정의 방식이다: 명시적 linear recurrence의 chunkwise는 exact지만 anchored mini-batch GD로 정의된 TTT-Linear는 linear여도 semantic.
- 품질-최적 $C$(작음)와 MFU-최적 $C$(큼)는 반대 방향이며, TNT는 계층 memory + 두 단계 훈련으로 분리해 (서로 다른 구성에서) 최대 17.37× 빠른 훈련과 개선된 perplexity를 얻는다 [TNT Table 1, 2].
- backward-in-decode 요구 넷: closed-form gradient kernel, per-session state라는 새 cache class(checkpoint/restore·prefix 분기·학습된 eviction), grouped-GEMM decode(batching이 intensity를 못 올림), optimizer-trajectory rollback.
- 6편 모두 decode wall-clock을 보고하지 않는다 — Part II의 serving 주장은 이 장의 도구로 각자 재계산하는 것이 기본이다.

## 자가 점검 체크리스트

- [ ] (M1)–(M4) 형태의 임의 layer에 대해 decode FLOPs/token·state RMW bytes/token을 지배항 수준에서 유도한다.
- [ ] 식 (10-1)로 자기 서빙 config의 KV-cache-vs-state crossover 길이를 계산한다.
- [ ] 식 (10-2)로 $\mathrm{AI}(C)$를 계산해 자기 hardware ridge point와 비교, 어느 $C$부터 compute-bound인지 판정한다.
- [ ] tiling vs chunkwise를 "exact 재배열 vs semantic 근사"로 설명하고 어느 클래스부터 적용되는지 말한다.
- [ ] backward가 decode에 들어올 때 serving engine의 네 가지 요구를 나열한다.
- [ ] Part II 임의 장의 §"systems 함의"를 표 10-2 행과 식 (10-1)·(10-2)로 환원해 검산한다.

## 다음 장으로

여기까지의 cost model은 전부 한 세션·한 inner loop의 이야기였다: request가 시작되면 $W_t$가 움직이고 끝나면 버려진다. [NL]과 [Sleep]은 이를 확장한다 — update 주기를 token부터 pretraining까지 연속 스펙트럼으로 펼치고(Continuum Memory System), 세션 후 fast state를 slow weights로 옮기는 offline 단계(sleep, consolidation)를 lifecycle에 넣는다. 이를 읽으려면 마지막 배경이 필요하다: update가 왜 옛 지식을 부수는가(catastrophic forgetting), 뇌는 왜 빠른/느린 기억계를 분리했는가(CLS), 지식 이전의 표준 도구(distillation, replay, RL-lite). 11장이 이 조각을 채워 Part II 준비를 끝낸다.
