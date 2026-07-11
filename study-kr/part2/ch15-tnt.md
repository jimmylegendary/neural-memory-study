# ch15. TNT: Improving Chunkwise Training for Test-Time Memorization

## 15.1 Bridge-in: 전작에서 남은 문제

[Atlas] (*Atlas: Learning to Optimally Memorize the Context at Test Time*, arXiv:2505.23735, → 14장)까지 이 라인은 update rule의 "무엇"을 완성했다. inner objective는 [Miras] (*It's All Connected*, arXiv:2504.13173)가 attentional bias라는 family로 일반화했고, retention은 같은 논문이 재이론화했으며, Atlas는 objective의 범위를 window로 넓히고(Omega rule) inner optimizer를 Muon으로 올렸다. 식 (M2)와 (M3)의 성분표는 사실상 채워졌다. 그런데 14장을 덮으며 남는 위화감이 하나 있다. Titans부터 Atlas까지 세 편의 논문 어디에도 **wall-clock 수치가 없다**. perplexity, capacity 정리, ablation은 있는데, "이 모델을 훈련하는 데 몇 시간이 걸리는가"라는, systems 엔지니어라면 첫 페이지에서 찾는 숫자가 없다.

없는 데는 이유가 있다. 이 라인의 모든 모델은 chunkwise-parallel training(→ 9장)이라는 하나의 트릭 위에 서 있다. chunk 안의 모든 inner gradient를 chunk 시작 상태 $W_{\xi(t,C)}$에서 평가하는 stale-snapshot 근사, 즉 식 (M4)다. 이 트릭은 두 가지를 미해결로 남겼다. 첫째, 근사의 품질. chunk 크기 $C$를 키우면 gradient가 낡아지고(staleness) 품질이 떨어지며, 줄이면 kernel이 잘게 쪼개져 hardware가 논다. 그래서 실무는 $C$를 16-64 같은 어중간한 값에 고정해 왔는데, 이 타협이 얼마나 비싼지 아무도 정량화하지 않았다. 둘째, $C$의 이중 신분. $C$는 병렬화 knob이면서 동시에 — 식 (M4)가 계산하는 함수 자체를 바꾸므로 — semantic hyperparameter다(→ 9장). 이 이중성은 9장에서 명제로 세운 것이지만, 그 명제의 실증적 발견자가 바로 이 장의 논문이다. Atlas까지는 훈련 때 쓴 $C$와 다른 $C$로 serving하면 어떻게 되는지 물은 적조차 없다.

한 가지 수치가 사태의 심각성을 요약한다. deep memory(→ 12장) 계열의 훈련은 품질이 좋은 작은 chunk에서 peak FLOPs 대비 5-10% 미만의 utilization으로 돌아간다 — [TNT]가 LaCT(Zhang, Bi, et al. 2025, arXiv:2505.23884)를 인용해 보고하는 값이다 [TNT §3]. 독자의 어휘로 말하면, 이 라인의 모델들은 지금까지 MFU 한 자릿수의 workload였다. 표현력 논쟁 이전에, 이 훈련 경제학이 해결되지 않으면 어떤 Titans 후속도 대규모로 갈 수 없다.

[TNT] (*TNT: Improving Chunkwise Training for Test-Time Memorization*, arXiv:2511.07343)는 이 지점을 정면으로 겨냥한 라인의 systems 편이다. 새 아키텍처가 아니라 **훈련 paradigm**이고, 주장하는 것도 표현력이 아니라 throughput과 chunk 경제학이다. 논문 제목의 TNT는 "Titans iNside Titans" 또는 "TTT iNside TTT"의 약자다 [TNT §1 각주 1] — 이름부터가 memory 안에 memory를 중첩하는 계층 구조를 가리킨다.

## 15.2 문제의식: 세 개의 challenge

[TNT]는 deep memory module의 실용화를 막는 문제를 세 개의 challenge로 명명해 분해한다 [TNT §3]. 이 분류법 자체가 논문의 기여 중 하나다.

**Challenge 1 — 효율적 훈련 구현의 부재.** deep memory는 fine-grained learning signal을 위해 작은 chunk(16-64 tokens)를 요구하는데 [TNT §3, Sun et al. 2024 (arXiv:2407.04620) 인용], 작은 chunk는 작은 연산을 대량 생산해 훈련을 compute-bound가 아니라 memory-bound로 만든다. linear memory 계열(GLA, DeltaNet, Mamba-2, → 6·7장)은 SRAM 상주 chunkwise kernel로 이를 회피하지만, 그 kernel들은 **linear state transition과 chunk 간 closed-form 분해**에 근본적으로 의존한다. deep memory의 recurrence는 MLP forward/backward와 chunk 사이 LayerNorm 같은 비선형을 통과하므로 그 kernel이 이식되지 않는다 [TNT §1, §3]. 비선형 recurrence를 sequence 길이 방향으로 병렬화하는 것은 parallel scan이 적용되지 않는, 오래된 미해결 문제다 [TNT §1].

**Challenge 2 — compression과 retrieval의 domain 불일치.** inner loop는 $\mathcal{M}(\cdot;W)$를 key를 입력으로 value를 맞히도록 적합시킨다(write는 $k_t \mapsto v_t$). 그런데 읽기는 $q_t$로 한다. 학습된 함수의 입력 domain 밖에서 함수를 평가하는 셈이고, 이 train/use domain shift가 retrieval 품질을 깎는다는 것이 논문의 진단이다 [TNT §3 Challenge 2].

> **[해설]** 이것은 memory 내부에서 일어나는 미니어처 train/test mismatch다. attention에는 이 문제가 구조적으로 없다 — softmax attention의 read는 $q$와 모든 $k$의 내적을 명시적으로 계산하므로 "key domain에 적합된 파라미터 함수"라는 중간물이 아예 없다. 압축하는 memory로 넘어오는 순간에만 생기는 세금이다.

**Challenge 3 — 고정 pre-training chunk 크기에 대한 성능 민감성.** 논문의 새 실증 발견이다. 550M Titans를 $C=64$로 pre-train한 뒤 inference chunk 크기를 바꿔 가며 validation perplexity를 재면: $C=8$에서 36.45, 16에서 34.15, 32에서 24.23, **64에서 13.78(최적)**, 128에서 15.5, 256에서 17.88, 512에서 22.4 [TNT Fig. 2]. 훈련 때 쓴 chunk 크기에서만 최적이고, 양쪽으로 벗어나면 급격히 나빠진다. 특히 왼쪽이 인상적이다 — 더 작은 chunk는 더 신선한 gradient를 뜻하므로 직관적으로는 inference에서 더 좋아야 하는데, 실제로는 ppl이 2.6× 이상 폭발한다. 모델이 훈련 해상도에 **over-specialize**된 것이다 [TNT §3 Challenge 3]. 이것이 이 책이 **chunk-size mismatch**라 부르는 현상이다: 같은 checkpoint가 serving 때 memory update를 얼마나 자주 적용하느냐에 따라 전혀 다른 품질을 낸다. 이 발견은 이상적 serving 구성 — decode에서 chunk 크기 1, 즉 매 token online update — 을 원천 봉쇄한다. 큰 chunk로 싸게 훈련한 모델을 chunk 1로 돌리면 무너지기 때문이다.

<!-- FIG-REF: ch09/fig-02-three-regimes -->

세 challenge를 관통하는 논문의 핵심 주장은 이렇다: **훈련 효율과 inference 성능을 한 개의 chunk 크기가 동시에 결정하도록 놔두지 말고, 두 단계로 분리(decouple)하라.** Stage 1은 hierarchical memory로 최대 throughput의 pre-training을 하고, Stage 2는 전체 비용의 약 5%로 작은 chunk에 fine-tune해서 chunk-1 decode를 품질 최적점으로 만든다. 결과 요약: 150M Titans 기준, 가장 정확한 Titans baseline($C=8$) 대비 목표 loss 도달까지 최대 17.37× 빠르면서 평균 perplexity는 오히려 개선(23.09 vs 25.07)되고 vanilla Transformer(23.58)도 이긴다 [TNT Table 1, Table 2].

선행 완화책에 대한 논문의 비판도 기록해 둔다 [TNT §1]. LaCT는 큰 chunk를 window attention과 결합하지만, 이는 비효율을 우회할 뿐 해결이 아니고, memory와 attention을 섞어 분석을 흐리며, decode에 필요한 chunk ~1을 외면한다. log-linear attention(Guo et al. 2025, arXiv:2506.04761)은 계층적이지만 linear memory에 국한된다.

## 15.3 Core mechanism (통일 표기)

### 15.3.1 두 연산으로 정식화된 deep memory

[TNT §2.1]은 라인 전체가 암묵적으로 쓰던 구조를 두 개의 per-token 연산으로 깔끔하게 정식화한다. fast weights $W$가 sub-network $\mathcal{M}(\cdot;W):\mathbb{R}^d\to\mathbb{R}^d$를 파라미터화하고, 매 token마다:

$$
W_t \;=\; W_{t-1} \;-\; \eta_t\,\nabla_W\,\ell\big(W_{t-1};\,k_t,v_t\big),
\qquad
y_t \;=\; \mathcal{M}(q_t;\,W_t)
\tag{15-1}
$$

첫 식이 **Memory Compression**(write) [TNT Eq. 1], 둘째가 **Memory Retrieval**(read) [TNT Eq. 2]다. 기호를 전부 확인하면: $x_t\in\mathbb{R}^d$는 입력 token 표현이고 slow-weight projection이 $q_t,k_t,v_t\in\mathbb{R}^d$를 만든다. $\ell$은 self-supervised inner loss로 기본형은 associative-memory regression $\ell(W;k,v)=\|\mathcal{M}(k;W)-v\|_2^2$ (원문은 MSE를 예시로 든다 [TNT §2.1]). $\eta_t$는 학습된(learned) per-token inner learning rate다 [TNT §2.1] — 시간 첨자가 붙어 있으므로 통일 규약(§1.2)상 data-dependent 게이트이며, 라인의 관행(→ 12장)대로 slow weights가 token마다 산출하는 값으로 읽는다. 다만 [TNT]는 산출 head의 구조를 명시하지 않는다. 식 (15-1)은 정확히 표준형 (M1)이다. (M2)와 비교하면 momentum $\beta_t$도 retention gate $\alpha_t$도 없다 — 이것은 실수가 아니라 의도된 단순화로, §15.6과 §15.8에서 다시 다룬다 [TNT App. D].

recurrence $W_t = W_{t-1}-\cdots$는 $W$에 대해 비선형이다. $\nabla_W\ell$이 deep net $\mathcal{M}$의 forward와 backward를 통과하기 때문이다. 이 한 문장이 Challenge 1의 근원이다: 상태 전이가 비선형이면 linear attention 계열의 chunk 간 closed-form 전파가 성립하지 않는다.

### 15.3.2 chunkwise parallel training 복습

token-serial recurrence를 병렬화하기 위해 라인 전체가 쓰는 방법이 chunkwise compression, 즉 식 (M4)다(유도와 일반론은 9장 소유; 여기서는 TNT 표기와의 접속만 확인한다). chunk 크기 $C$, chunk 시작 offset $\xi(t,C)=C\lfloor(t-1)/C\rfloor$에 대해:

$$
W_t \;=\; W_{\xi(t,C)} \;-\; \sum_{\tau=\xi(t,C)+1}^{t} \eta_\tau\,\nabla_W\,\ell\big(W_{\xi(t,C)};\,k_\tau,v_\tau\big),
\qquad
y_t=\mathcal{M}(q_t;W_t)
\tag{15-2}
$$

[TNT Eq. 3–4]가 이 형태다.[^xi] chunk 안의 모든 gradient가 frozen된 chunk 시작 상태 $W_{\xi(t,C)}$에서 평가되므로, chunk-start 상태가 주어지면 per-token gradient들은 서로 독립이고 하나의 batched forward/backward로 계산된다. per-token 상태 $W_t$는 그 gradient들의 누적 합(cumulative sum)이다. 남는 직렬 의존성은 chunk 경계 하나뿐이다: $n$번째 chunk의 마지막 상태 $W_{nC}$가 $n{+}1$번째 chunk를 seed한다 [TNT §2.2]. staleness가 품질을, matmul 크기가 throughput을 결정하므로 $C$가 두 세계의 유일한 교환 변수가 된다. 이 근사의 오차 한계(bound)는 [TNT]를 포함해 6편 어디에도 없다 — 전부 실증이다.

[^xi]: 원문의 chunk 시작 함수는 $\xi(i,j):=i-(i \bmod j)$로 0-based 관행이다 [TNT §1]. 이 책은 1-based token 인덱스에 맞춘 $\xi(t,C)=C\lfloor(t-1)/C\rfloor$(§1.2)를 쓴다. 합의 하한이 원문 $\tau=\xi(t,C)$ 대신 $\tau=\xi(t,C)+1$이 되는 것은 이 인덱스 관행 차이일 뿐 알고리즘은 동일하다.

### 15.3.3 Stage 1 — hierarchical memory: global + N local

TNT Stage 1의 구조를 한 문장으로 요약하면: **큰 chunk로 도는 순차적 global memory 하나가 long-range context를 담당하고, 주기적으로 초기화되는 N개의 병렬 local memory가 fine-grained 정보를 담당한다** [TNT §4.1.1]. 이것이 이 책이 **hierarchical memory**라 부르는 구조다.

<!-- FIG: ch15/fig-01-tnt-hierarchy -->

**Global memory.** 상태 $W^{\mathrm{g}}$ (원문 기호 $V$; value 행렬과의 충돌 때문에 개명 — 표 15-1)는 매우 큰 chunk 크기 $C_{\mathrm{g}}$ (실험에서 2048)로 식 (15-2)의 표준 chunkwise recursion을 돈다:

$$
W^{\mathrm{g}}_{(n+1)C_{\mathrm{g}}} \;=\; W^{\mathrm{g}}_{nC_{\mathrm{g}}}
\;-\; \sum_{t=nC_{\mathrm{g}}+1}^{(n+1)C_{\mathrm{g}}} \eta_t\,\nabla_{W^{\mathrm{g}}}\,\ell\big(W^{\mathrm{g}}_{nC_{\mathrm{g}}};\,k_t,v_t\big),
\qquad n=0,\ldots,L/C_{\mathrm{g}}-1
\tag{15-3}
$$

[TNT Eq. 5]다.[^slip] 상태는 sequence 전체를 관통해 순차적으로 이월되므로 long-range 정보가 살아남고, $C_{\mathrm{g}}$가 크므로 update는 드물고 크고 dense한 matmul이 된다 — 설계상 compute-bound다. 16K context에 $C_{\mathrm{g}}=2048$이면 sequence당 직렬 handoff는 8번뿐이다.

**Local memory와 periodic state reset.** 핵심 혁신은 local 쪽에 있다. 기본형($N=1$)에서 local memory $W^{\mathrm{l}}$은 chunk 크기 $C_{\mathrm{l}}$, shard 길이 $L_{\mathrm{s}}$ (원문 $S_L$), 그리고 **학습 가능한 초기 상태 $W_{\mathrm{init}}$**을 가지고 다음과 같이 갱신된다:

$$
W^{\mathrm{l}}_t \;=\;
\begin{cases}
W_{\mathrm{init}} & t \equiv 0 \pmod{L_{\mathrm{s}}} \\[4pt]
\displaystyle W^{\mathrm{l}}_{\xi(t,C_{\mathrm{l}})} \;-\; \sum_{\tau=\xi(t,C_{\mathrm{l}})+1}^{t} \eta_\tau\,\nabla_W\,\ell\big(W^{\mathrm{l}}_{\xi(t,C_{\mathrm{l}})};\,k_\tau,v_\tau\big) & \text{otherwise}
\end{cases}
\tag{15-4}
$$

[TNT Eq. 6]이다.[^slip] 첫 줄이 **periodic state reset**이다: $L_{\mathrm{s}}$ token마다(실험에서 2048-4096) local 상태를 통째로 버리고 outer loop가 학습한 $W_{\mathrm{init}}$으로 되돌린다. 즉 shard 경계 상태 $W^{\mathrm{l}}_{mL_{\mathrm{s}}}$가 직전 shard의 마지막 상태 대신 $W_{\mathrm{init}}$으로 대체되고, shard $m$의 계산은 위치 $mL_{\mathrm{s}}$ 이전의 그 무엇에도 의존하지 않게 된다.

이 reset이 왜 결정적인가. 비선형 recurrence는 parallel scan으로 병렬화할 수 없다 — scan은 결합법칙을 요구하는데 MLP를 통과하는 상태 전이에는 그것이 없다. reset은 병렬화가 안 되는 사슬을 **아예 끊어버린다**. 그 결과 $L/L_{\mathrm{s}}$개의 shard가 완전히 독립인 계산이 되고, 장치들에 분산하거나(**context parallelism**) 한 accelerator의 batch 축에 쌓아 kernel을 fatten할 수 있다 [TNT §4.1.1]. 비선형 deep-memory recurrence에 대해 현재 알려진 유일한 sequence-방향 병렬화 수단이다. 대가는 명확하다: local memory는 shard 경계에서 모든 것을 잊는다. 그 손실을 보전하는 것이 global memory의 존재 이유다 — reset 없는 global이 long range를 들고, reset 있는 local이 병렬성을 든다. ablation에서 global을 제거하면 ppl이 21.04에서 25.60으로 붕괴하는 것이 이 역할 분담의 실증이다 [TNT Table 3].

$W_{\mathrm{init}}$이 **학습된다**는 점도 하중을 받는 설계다. 모든 shard가 0이 아니라 meta-learn된 prior에서 inner loop를 시작한다. Titans의 $W_{\mathrm{init}}$(원문 $M_0$)은 암묵적 존재였지만(→ 12장), TNT에서는 reset을 생존 가능하게 만드는 load-bearing 부품으로 승격된다 — 개념 자체는 4장의 MAML류 meta-learned initialization이다.

local 시스템은 그 자체로 다중 해상도일 수 있다. $N$개의 local module $W^{\mathrm{l}(i)}$가 각자의 chunk 크기 $C_{\mathrm{l}}^{(i)}$, shard 길이 $L_{\mathrm{s}}^{(i)}$, 초기 상태 $W^{(i)}_{\mathrm{init}}$을 갖는 일반형은 [TNT App. E Eq. 14]에 있고, 구성은 chunk 크기 집합으로 표기한다: $C_{\mathrm{l}}=\{4,8,16,32\}$는 서로 다른 시간 스케일을 잡는 4개의 local module을 뜻한다 [TNT §5.1]. 실험에서 module을 하나 더할 때마다 ppl이 단조 개선된다(§15.6).

[^slip]: 원문 수식 두 곳에 표기 슬립이 있다(원문 자체의 문제로, 이 책의 통일 표기가 그 정정을 겸한다). (a) [TNT Eq. 5]는 chunk 인덱스로 $n$과 $k$를 한 식 안에 혼용한다; 정돈된 형태는 [TNT App. E Eq. 13]이다. (b) [TNT Eq. 6]은 둘째 경우의 base 상태를 $W_{t-1}$로 인쇄했지만 gradient anchor는 $W_{\xi(t,C_L)}$이다 — 문자 그대로 매 $t$에 적용하면 chunk 내부 gradient가 이중 누적되어 [TNT Eq. 3]과 모순된다. Eq. 3과 동일한 chunkwise semantics(chunk 시작 상태를 base로 한 누적 합)로 읽는 것이 유일하게 정합적인 독해이며, 식 (15-4)는 그렇게 표기했다. (c) 추가로 [TNT Eq. 7]의 projection 합 하한은 chunk 시작 $\xi(t,C_L)$로 인쇄되어 있으나, [TNT App. B]는 "reset 상태부터의 합"을, [TNT App. C]는 shard 경계에서 reset되는 누적 recurrence를 명시한다. 이 책은 구현을 기술한 App. C를 기준으로 삼는다.

### 15.3.4 Stage 1 — Q-K Projection: read를 write의 domain으로 되돌리기

Challenge 2의 처방이다. query를 그대로 memory에 넣는 대신, **지금까지 관측된 key들이 스팬하는 부분공간으로 query를 사영한 뒤** 넣는다. 사영을 담당하는 것이 **Q-K projection** 행렬 $\Pi_t\in\mathbb{R}^{d\times d}$ (원문 기호 $\mathcal{M}_t$; memory 기호와의 충돌 때문에 개명)이다. key가 L2 정규화되어 있다는 가정(이 계열 모델의 관행 [TNT §4.1.2]) 하에서 recurrence와 chunkwise 분해는:

$$
\Pi_t=
\begin{cases}
k_t k_t^\top & t \equiv 1 \pmod{L_{\mathrm{s}}}\\
\Pi_{t-1}+k_t k_t^\top & \text{otherwise}
\end{cases}
\qquad\;
\Pi_t=\underbrace{\Pi_{\xi(t,C_{\mathrm{l}})}}_{\text{carry-over}}+\underbrace{\sum_{\tau=\xi(t,C_{\mathrm{l}})+1}^{t}k_\tau k_\tau^\top}_{\text{intra-chunk prefix sum}}
\tag{15-5}
$$

[TNT App. C]다. 정규화 가정이 없으면 각 항이 $k_\tau k_\tau^\top/\|k_\tau\|^2$이 된다 [TNT Eq. 7]. 왼쪽 recurrence는 local memory와 같은 reset 규율을 따르고(shard 시작에서 $k_tk_t^\top$로 재시작), 오른쪽 분해는 chunk 내부를 outer product들의 parallel prefix sum(scan)으로, chunk 사이를 $d\times d$ 행렬 하나의 carry로 처리한다. 과거 key를 저장할 필요가 전혀 없다 — 상태는 상수 크기다 [TNT §4.1.2].

retrieval은 global과 local의 출력 합이다:

$$
y_t \;=\; \mathcal{M}\big(q_t;\; W^{\mathrm{g}}_{\xi(t,C_{\mathrm{g}})}\big) \;+\; \mathcal{M}\big(\Pi_t\, q_t;\; W^{\mathrm{l}}_t\big)
\tag{15-6}
$$

[TNT Eq. 7]. $N$-local 일반형은 둘째 항을 $\sum_{i=1}^{N}\mathcal{M}\big(\Pi^{(i)}_t q_t;\,W^{\mathrm{l}(i)}_t\big)$로 바꾼 것이다 [TNT App. E Eq. 15]. 두 가지 비대칭에 주목하라. 첫째, projection은 **local에만** 적용된다. fine-grained한 local memory가 domain mismatch에 더 민감하다는 판단이고, global은 raw query를 받는다 [TNT §4.1.2]. 둘째, global 읽기는 현재 상태가 아니라 **chunk 시작에 frozen된 상태** $W^{\mathrm{g}}_{\xi(t,C_{\mathrm{g}})}$를 읽는다. global chunk 하나(최대 2048 token) 동안 읽기가 고정된 상태를 보므로 retrieval도 chunk-병렬이 되지만, 그만큼(최대 $C_{\mathrm{g}}-1$ token) 낡은 global 정보를 읽는다는 뜻이기도 하다 — §15.8에서 recall 관련 미검증 지점으로 되돌아온다.

> **[해설]** $\Pi_t=\sum_\tau k_\tau k_\tau^\top$는 rank-1 projector들의 합이므로 $\Pi_t q_t$는 항상 관측된 key들의 span 안에 떨어지고, 반복 관측된 방향일수록 증폭된다. 엄밀한 의미의 직교 사영(idempotent)은 key들이 정규직교일 때뿐이므로, "사영"이라기보다 관측 빈도로 가중된 soft projection이다. inference 어휘로는, $\Pi_t$의 갱신은 KV cache append의 rank-1 GEMM 대응물이고 $\Pi_t q_t$는 mat-vec 한 번이다. attention이 read 시점에 $q^\top K$로 하던 "query를 key 통계와 대면시키는 일"을, 상수 크기 running sum으로 미리 압축해 두는 셈이다.

[TNT App. B]는 이 장치의 족보를 밝힌다. $\Pi'_t=\Pi'_{t-1}+k_tk_t^\top$는 그 자체로 보조 linear memory이고, $\Pi_t q_t$는 그 memory에 대한 forward pass다. 따라서 Q-K projection이 붙은 retrieval은 ABC(Peng, Kasai, et al. 2022)·Gated Slot Attention(Zhang, Yang, et al. 2024)·Trellis(Karami, Behrouz, et al. 2025) 같은 memory-bounded Transformer의 two-pass read — $W_t=W_{t-1}+\varphi_t v_t^\top$, $y_t=W_t\,\mathrm{softmax}(\sum_\tau \varphi_\tau\varphi_\tau^\top q_t)$ — 와 같은 형태다. 차이는 세 가지다: TNT의 projection은 linear뿐 아니라 deep memory에도 적용되고, 별도의 feature $\varphi$를 학습하는 대신 $k_t$와 묶여(tied) 있으며, 합산이 $\tau=1$부터가 아니라 마지막 reset부터다 [TNT App. B].

### 15.3.5 Stage 2 — 더 fine한 해상도로의 fine-tuning

Stage 1이 훈련 효율을 해결했으니 Challenge 3이 남는다. 큰 chunk로 pre-train한 모델을 그냥 작은 chunk로 평가하면 Fig. 2의 절벽에서 떨어진다. [TNT §4.2]의 관찰: **짧은 fine-tuning으로 이 train-test 불일치가 교정되며, 원래 성능을 회복하는 정도가 아니라 넘어선다.** Stage 2는 효율적으로 pre-train된 모델을 더 작은 local chunk 크기 $C_{\mathrm{l}}' < C_{\mathrm{l}}$로 계속 훈련하는 것이다. global의 $C_{\mathrm{g}}$는 그대로 둔다. 비용은 pre-training의 약 5% — Table 4 기준 Stage 1이 3.06-5.55시간일 때 Stage 2는 0.15-0.46시간이다 [TNT Table 4].

이상적 목표는 $C_{\mathrm{l}}'=1$이다. 이 지점이 autoregressive serving의 prefill-and-decode 패턴과 정확히 맞물린다: **global memory가 큰 chunk의 dense 연산으로 prompt를 흡수하고(prefill), Stage 2로 적응된 local memory가 생성 중 token 단위로 갱신된다(decode)** [TNT §4.2]. 이로써 chunk 크기는 더 이상 하나의 타협값이 아니다 — Stage 1에서는 훈련 throughput knob, Stage 2에서는 inference 해상도 knob이라는 서로 독립인 두 개의 knob이 된다. 이것이 이 책이 **two-stage training**(train-big / serve-small)이라 부르는 레시피다.

한 가지 원문의 모호함을 기록한다. abstract는 "only the local memory modules are adapted"라 쓰고, [TNT §4.2] 본문은 "continue training the efficiently pre-trained model with a smaller local chunk size"라 쓴다. 확실한 것은 바뀌는 것이 local chunk-size hyperparameter뿐이라는 점이고, Stage 2 동안 slow weights의 일부가 동결되는지는 명시되지 않았다. §4.2의 자연스러운 독해는 새 해상도에서 end-to-end로 훈련을 계속한다는 것이다.

### 15.3.6 각 부품이 존재하는 이유 (한 줄 정리)

- 큰 $C_{\mathrm{g}}$의 global memory: hardware 포화 + long-range context 유지.
- reset-to-$W_{\mathrm{init}}$ local memory: 비선형 deep-memory recurrence를 context-병렬화하는 유일한 알려진 수단.
- 학습된 $W_{\mathrm{init}}$: 모든 shard에 주어지는 meta-learn된 prior — reset을 정보 전멸이 아니게 만드는 완충재.
- 다중 해상도 $\{C_{\mathrm{l}}^{(i)}\}$: 서로 다른 시간 스케일의 feature 포착.
- Q-K projection: key-write/query-read domain 간극을 $O(d^2)$ 상태로 봉합.
- global에는 raw query: 거친 granularity에서는 mismatch가 덜 아프고 compute를 아낀다.
- Stage 2: chunk-size over-specialization을 ~5% 비용으로 치료하고 chunk-1 decode를 해금.

### 15.3.7 표기 대응표

표 15-1 — [TNT] 원 표기와 이 책의 통일 표기 대응 (§1.7.4 기준; 하단 4행은 장-국소 추가).

| TNT 원 표기 | 통일 표기 | 의미 / 주의 |
|---|---|---|
| $W_t$ — fast weights | $W_t$ | 동일 |
| $f(W,\cdot)$ — memory net | $\mathcal{M}(\cdot;W)$ | |
| $\mathcal{L}(\cdot,\cdot)$ — **inner** loss | $\ell$ | ⚠ 통일 체계에서 $\mathcal{L}$은 outer 전용 |
| $\eta_t$ — learned inner lr | $\eta_t$ | 동일 |
| $\xi(i,j)$ — chunk 시작 함수 | $\xi(t,C)$ | 채택 (1-based 정의는 §1.2) |
| $C$, $C_G$, $C_{L_i}$, $C_L'$ | $C$, $C_{\mathrm{g}}$, $C_{\mathrm{l}}^{(i)}$, $C_{\mathrm{l}}'$ | |
| $S_{L_i}$ — shard 길이(reset 주기) | $L_{\mathrm{s}}^{(i)}$ | ⚠ momentum $S_t$와 충돌 방지 |
| $V$ — global memory 상태 | $W^{\mathrm{g}}$ | ⚠ value 행렬 $V$와 충돌 방지 |
| $W^{(i)}$ — local memories | $W^{\mathrm{l}(i)}$ | |
| $W_{init}$ — 학습된 초기 상태 | $W_{\mathrm{init}}$ | 동일 |
| $\mathcal{M}_t=\sum_\tau k_\tau k_\tau^\top/\|k_\tau\|^2$ — Q-K projection | $\Pi_t$ | ⚠ memory 기호와 충돌 방지 |
| $o_t$ — 출력 | $y_t$ | |
| Eq. 5–6의 인덱스 표기 슬립 (원문 자체) | (M4)의 semantics로 통일 | 각주로 명시 (이 장 [^slip]) |
| $S_L$ — $N{=}1$ 기본형의 shard 길이 | $L_{\mathrm{s}}$ | 장-국소 추가 |
| $\mathcal{M}'_t$ — App. B의 보조 linear memory | $\Pi'_t$ | 장-국소 추가 |
| $\varphi_t$ — App. B의 ABC feature | $\varphi_t$ 유지 | 장-국소 추가 (직접 인용 문맥) |
| $n=\{0,\ldots,L//C_G\}$ — global chunk 범위 | $n=0,\ldots,L/C_{\mathrm{g}}-1$ | 장-국소 추가; $C_{\mathrm{g}}\mid L$ 가정 |

## 15.4 Outer-loop training vs inner-loop test-time learning

이 절이 이 장의 심장이다. TNT는 "무엇이 학습되는가"라는 질문에 대해 라인에서 가장 계층이 많은 답을 갖고 있기 때문이다.

표 15-2 — TNT의 구성 요소별 소속 loop와 갱신 규칙.

| 구성 요소 | loop | 갱신 rule | 갱신 주기 | 근거 |
|---|---|---|---|---|
| projection $W_K,W_V,W_Q$ (module별) | outer ($\Theta$) | AdamW (wd 0.1, cosine, peak lr $10^{-3}$) | 훈련의 매 step; inference에서 동결 | [TNT §5.1] |
| $\eta_t$ 산출 map | outer ($\Theta$) | AdamW | 훈련의 매 step (값 자체는 매 token 산출) | [TNT §2.1] |
| $W^{(i)}_{\mathrm{init}}$ | outer ($\Theta$) | AdamW; gradient가 **모든 shard로부터** 유입 | 훈련의 매 step; inference에서 동결 | [TNT §4.1.1] |
| backbone (FeedForward, embedding, head) | outer ($\Theta$) | AdamW | 훈련의 매 step | [TNT Fig. 3, §5.1] |
| global fast weights $W^{\mathrm{g}}_t$ | inner | (M4)형 chunkwise GD, anchor $W^{\mathrm{g}}_{nC_{\mathrm{g}}}$ | $C_{\mathrm{g}}=2048$ token마다 1회 | [TNT Eq. 5] |
| local fast weights $W^{\mathrm{l}(i)}_t$ | inner | (M4)형 chunkwise GD + periodic reset | $C_{\mathrm{l}}^{(i)}$ token마다; $L_{\mathrm{s}}^{(i)}$마다 reset | [TNT Eq. 6/14] |
| Q-K projection $\Pi^{(i)}_t$ | inner | rank-1 누적 $+\,k_tk_t^\top$ | 매 token; $L_{\mathrm{s}}^{(i)}$마다 reset | [TNT App. C] |

inner 열에 **없는 것**도 정보다: momentum buffer $S_t$도, retention gate $\alpha_t$도 없다. TNT의 inner optimizer는 학습된 step size $\eta_t$만 가진 plain GD다 [TNT App. D].

**Outer loop에서 일어나는 일.** slow weights $\Theta$는 next-token cross-entropy $\mathcal{L}$에 대해 AdamW로 훈련된다. training 무경험 독자를 위한 결정적 사실 하나: fast weights $W^{\mathrm{g}}_t, W^{\mathrm{l}(i)}_t$는 파라미터가 **아니다**. 이들은 식 (15-3)·(15-4)라는 기호적 gradient-descent 식이 만들어내는 outer 계산 그래프의 **activation**이다. 따라서 backprop은 그 inner GD step들을 **관통해** 미분한다 — $\mathcal{L}$의 $\Theta$에 대한 gradient에는 $\partial(\nabla_W\ell)/\partial\Theta$ 꼴의 2차 항이 매 inner update마다 실려 들어온다. "$\eta_t$가 학습된다", "$W_{\mathrm{init}}$이 학습된다"는 말의 정확한 의미가 이것이다: 그 값들을 바꾸면 inner 궤적 전체가 바뀌고, 그 파급이 LM loss에 미치는 효과가 backprop으로 계량되어 AdamW가 그 방향으로 움직인다(형식화는 → 4장).

이 그래프를 감당 가능하게 만드는 것이 chunkwise anchoring이고, TNT가 바꾸는 것은 그래프의 **모양**이다. (a) local 경로: reset이 $L_{\mathrm{s}}$ token마다 그래프를 절단한다. shard 경계를 local 상태를 통해 넘어가는 gradient 경로는 존재하지 않으므로, shard당 직렬 사슬은 $L_{\mathrm{s}}/C_{\mathrm{l}}$번의 chunk handoff뿐이고 $L/L_{\mathrm{s}}$개 shard는 훈련 그래프에서도 완전히 독립이다 — forward만이 아니라 backward도 context-병렬이다. 한편 $W_{\mathrm{init}}$은 **모든** shard의 시작점이므로 모든 shard로부터 gradient를 받는다: 조밀하고 잘 평균된 학습 신호다. (b) global 경로: 직렬 사슬은 sequence당 $L/C_{\mathrm{g}}$ handoff다(16K에서 8번). 비선형 recurrence이지만 이 정도 깊이는 싸다. (c) $\Pi_t$: 식 (15-5)의 chunk 내부 prefix scan + chunk 간 $d\times d$ carry + shard 경계 reset이므로 직렬 병목을 추가하지 않는다.

> **[해설]** 독자의 감각으로 옮기면: Titans의 훈련 그래프는 "sequence 길이만큼 긴 단일 파이프라인"이었다. TNT는 그것을 (i) 깊이 8짜리 굵은 파이프(global), (ii) 서로 독립인 $L/L_{\mathrm{s}}$개의 짧은 파이프 묶음(local, batch 축에 stack 가능), (iii) scan kernel 하나($\Pi$)로 재배관했다. 훈련의 batch 축 역할을 sequence 축이 대신한다는 9장의 경고를 상기하면, reset은 sequence 축을 잘라 **진짜 batch 축으로 되돌리는** 연산이다.

**Inner loop에서 일어나는 일 (inference).** 동결된 $\Theta$ 아래에서, memory layer당 진화하는 상태는 $W^{\mathrm{g}}$ 하나, $W^{\mathrm{l}(i)}$ $N$개, $\Pi^{(i)}$ $N$개다 — 전부 sequence 길이와 무관한 상수 크기다. update rule은 훈련 때와 **형태가 동일하다**(그것이 test-time memorization의 정의다, → 8·12장). global은 $C_{\mathrm{g}}$ token마다 한 번, chunk 시작 상태에서 2048개 token의 gradient를 한 번의 batched forward+backward로 모아 적용하고, 그 사이 읽기는 frozen 상태를 본다. local은 Stage 2가 $C_{\mathrm{l}}'=1$로 끝났다면 매 token마다 정확한 online GD $W^{\mathrm{l}}_t=W^{\mathrm{l}}_{t-1}-\eta_t\nabla_W\ell(W^{\mathrm{l}}_{t-1};k_t,v_t)$를 수행하고 — 이때 식 (15-4)의 stale-snapshot 근사는 소멸한다, chunk가 1이므로 — $L_{\mathrm{s}}$ token마다 $W_{\mathrm{init}}$으로 reset된다($\Pi$는 $k_tk_t^\top$로). decode 한 step의 비용은, memory sub-network 하나의 파라미터 수를 $P_f$라 할 때 local당 compression의 forward+backward ~$6P_f$ FLOPs + retrieval forward ~$2P_f$ + $\Pi$ 갱신과 적용 각각 ~$2d^2$이다.

> **[해설]** 위 per-token 비용 분해는 [TNT]가 보고한 수치가 아니라 이 책의 산정이다(원문에는 decode 비용 표가 없다). 요점은 두 가지다. 첫째, **backward pass가 decode 안에 들어온다** — weight update 없음이라는 추론 불변식은 이 라인에서 폐기된다(→ 1장 Rosetta). 둘째, 그 모든 비용이 $L$과 무관한 상수라는 것이 KV cache 대비 이 계열의 존재 이유다(정량 비교는 §15.7).

## 15.5 Concept ledger delta

표 15-3 — TNT가 라인의 개념 원장에 가한 변경.

| 유형 | 개념 | 내용 |
|---|---|---|
| 추가 | chunk-size mismatch (train/serve) | pre-training chunk에 대한 inference 품질의 over-specialization [TNT Fig. 2]. 9장의 "chunk 크기 = semantic hyperparameter" 명제의 실증적 뿌리 |
| 추가 | hierarchical memory (global/local) | 큰 $C_{\mathrm{g}}$의 순차 global + reset되는 병렬 local의 역할 분담 [TNT §4.1.1] |
| 추가 | periodic state reset / context parallelism | reset-to-$W_{\mathrm{init}}$이 비선형 recurrence의 sequence-방향 병렬화를 최초로 해금 [TNT §4.1.1] |
| 추가 | Q-K projection | write(key) domain과 read(query) domain의 정렬; $O(d^2)$ running-sum 상태 [TNT §4.1.2, App. B–C] |
| 추가 | two-stage training (train-big / serve-small) | throughput knob과 해상도 knob의 분리; ~5% fine-tune으로 chunk-1 decode를 품질 최적점화 [TNT §4.2] |
| 추가 | Challenge 1/2/3 분류법; compression/retrieval 2-연산 정식화; 다중 해상도 local 집합 $C_{\mathrm{l}}=\{\cdots\}$ | 논문의 어휘 정비 [TNT §2–3] |
| 확장 | meta-learned $W_{\mathrm{init}}$ | Titans의 암묵적 $M_0$(→ 12장)가 reset의 생존 조건이라는 하중 부품으로 승격 |
| 확장 | chunkwise-parallel training | 병렬화 트릭(→ 9장)에서 그 자체가 연구 대상인 훈련 경제학으로; staleness-throughput 트레이드오프 최초 정량화 |
| 확장 | multi-timescale memory | Titans의 persistent/long/short 삼분(→ 12장)이 global/local chunk 계층으로 공학화 — 16장 CMS의 전조 |
| 확장 | retention | reset은 스케줄된 hard retention: 게이트 언어로 $\alpha_t\in\{0,1\}$, 값이 데이터가 아니라 시계에 의해 결정되는 극단형 ([해설]적 재서술) |
| 폐기(일시) | momentum $\beta_t$, retention gate $\alpha_t$, Muon/Omega rule | "명료성을 위해" 제거된 단순화 [TNT App. D]; 라인 본류와의 합성은 미검증 채로 유보 |

## 15.6 실험과 스케일

**설정.** 품질 실험은 전부 150M 파라미터, 10B tokens, T5 tokenizer(32k vocab), AdamW(weight decay 0.1, cosine schedule, peak lr $10^{-3}$), TPUv5 pod(2x2x2 topology, model parallelism 2), custom kernel 없는 순수 JAX 구현이다 [TNT §5.1]. 효율 벤치마크는 context 2K-32K, batch 0.5M tokens, $L_{\mathrm{s}}=2048$; 품질 평가는 context 16K, batch 1M tokens, $L_{\mathrm{s}}=4096$, $C_{\mathrm{g}}=2048$이다 [TNT §5.1]. TTT와 Titans baseline은 저자들의 재구현이다 [TNT §5 각주 2]. chunk 민감도 실험(Fig. 2)만 550M 모델을 쓴다.

**속도 — step당.** token/batch를 0.5M으로 고정하고 sequence 길이를 늘리면, TNT의 step 시간은 선형으로 증가하는 반면 Titans와 JAX attention은 wall-clock에서 2차적으로 증가한다고 논문은 보고한다 [TNT §5.2, Fig. 4]. 32K에서 같은 memory chunk($C_{\mathrm{l}}=C=16$)의 Titans보다 5.1× 빠르고, $C_{\mathrm{l}}=\{128\}$의 순수-JAX TNT는 Pallas FlashAttention kernel보다도 1.3× 빠르다 [TNT §5.2].

**속도 — 목표 품질까지.** 실전 지표는 같은 training loss(3.20)에 도달하는 시간이다 [TNT Table 1].

표 15-4 — 목표 loss 3.20 도달 시간 (150M, [TNT Table 1] 발췌).

| 모델 (구현) | $C$ 또는 $C_{\mathrm{l}}$ | 시간(hrs) | 배속 |
|---|---|---|---|
| Titans (JAX) | 8 | 19.48 | 1.00× |
| Titans (JAX) | 64 | 4.18 | 4.67× |
| Titans (JAX) | 128 | 3.71 | 5.25× |
| Transformer w/ gating (JAX) | - | 1.38 | 14.10× |
| Transformer w/ gating (FlashAttention) | - | 0.96 | 20.22× |
| TNT (JAX) | {8} | 2.54 | 7.68× |
| TNT (JAX) | {64} | **1.12** | **17.37×** |
| TNT (JAX) | {128} | 1.16 | 16.75× |

헤드라인 17.37×는 "가장 정확한 Titans"($C=8$, ppl 25.07) 대비다. 같은 chunk 크기 8끼리 비교해도 TNT가 7.7× 빠르므로 이득의 원천이 chunk 확대만이 아니라 구조임을 알 수 있다 [TNT §5.2]. $C_{\mathrm{l}}$을 키우면 {64}까지는 단조로 빨라지다가 {128}에서 미세하게 되돌아간다 — step은 더 빠르지만 step당 품질 이득이 깎이는 지점이다. 그리고 정직하게: kernel 최적화된 Gated Transformer(0.96h)는 아직 이기지 못하며, 논문도 custom kernel 부재를 이유로 들며 future work로 미룬다 [TNT §5.2].

**품질.** C4/FineWeb/PG19 perplexity와 4개 common-sense reasoning 과제(PIQA, HellaSwag, ARC-e, CSQA) 평균 [TNT Table 2]:

표 15-5 — 품질 결과 (150M / 10B tokens, [TNT Table 2] 발췌; ppl은 3개 corpus 평균, acc는 4개 과제 평균).

| 모델 | 구성 | 평균 ppl ↓ | 평균 acc ↑ |
|---|---|---|---|
| Transformer (w/o gating) | - | 23.58 | 38.3 |
| Gated Transformer | - | **22.39** | 39.7 |
| TTT | $C=256$ | 27.62 | 38.1 |
| Titans | $C=256$ | 27.13 | 38.8 |
| Titans | $C=8$ | 25.07 | 39.0 |
| TNT Stage 1 | {8} | 24.10 | 40.6 |
| TNT Stage 1 | {4,8,16,32} | 23.13 | 40.6 |
| TNT Stage 2 | {1} | 23.99 | 40.9 |
| TNT Stage 2 | {2,4,8,16} | **23.09** | 40.9 |

Stage 1만으로 모든 RNN baseline과 vanilla Transformer의 ppl을 이기고, Stage 2가 각 구성에서 ppl을 추가로 내린다(23.13→23.09 등). reasoning acc에서는 Gated Transformer까지 이긴다(41.0 vs 39.7 [TNT §5.3]; 표의 40.9는 {2,4,8,16} 행, 41.0은 Stage 1 {8,16} 행이다). 단 ppl에서는 Gated Transformer(22.39)에 진다는 것을 논문 스스로 명시한다 [TNT §5.3]. 훈련 비용 전액은 Table 4가 준다: Titans $C=8$은 8.44h, TNT Stage 1은 {8} 3.06h에서 {4,8,16,32} 5.55h, Stage 2는 0.15-0.46h — 논문이 "약 5%"라 부르는 근거다 [TNT Table 4, §5.3].

**Ablation** [TNT Table 3]: base Titans ppl 23.53/acc 38.8에서 local memory를 1→4개 추가하면 ppl 21.04→20.74→20.47→20.15로 단조 개선. global memory 제거는 25.60으로 붕괴(base보다도 나쁨 — reset만 있고 global 맥락이 없으면 치명적). Q-K projection 제거는 21.04→22.01(projection의 가치 ≈ 1 ppl, acc는 40.6→36.4). $N=1$에 Stage 2를 얹으면 20.86/40.9. Table 3의 ppl 열은 열 머리에 corpus 표기가 없으나, 여섯 값 전부가 Table 2의 C4 열과 일치한다(23.53/21.04/20.74/20.47/20.15/20.86; 다른 corpus 열과는 불일치) — C4 기준으로 읽는다 [TNT Table 2–3].

**스케일 정직성.** 이 논문의 실증은 150M/10B tokens가 전부이고, chunk-size mismatch 현상 자체도 550M 한 설정의 그림 하나가 근거다. 라인 전체의 실증 상한이 1.3B params / 100B tokens([NL] (*Nested Learning*, arXiv:2512.24695), → 16장)임을 감안해도 TNT는 그 안에서 가장 작은 축이다. "17× 가속이 1B+/100B+에서도 성립하는가"는 열린 질문이며(§15.8), 무엇보다 이 모든 검증이 **momentum·gating·Muon을 '명료성을 위해' 제거한 단순화 Titans** 위에서 이뤄졌다는 것 [TNT App. D] — 즉 라인의 본류 모델과의 합성은 측정된 적이 없다는 것 — 이 이 장의 의무 caveat다.

## 15.7 Systems/serving 함의

**병렬화 구조.** TNT가 한 일을 구조적으로 요약하면, 알려진 것 중 가장 병렬화하기 나쁜 sequence layer(deep-net 값의 비선형 recurrence)를 세 개의 얌전한 조각으로 재배열한 것이다. (a) sequence당 $L/C_{\mathrm{g}}$번의 직렬 handoff만 갖는 global recurrence — 16K/$C_{\mathrm{g}}{=}2048$이면 8번이고, 각 handoff는 2048 token에 대한 거대한 batched matmul이라 설계상 compute-bound다. (b) 완전히 독립인 $L/L_{\mathrm{s}}$개의 local shard — 장치 간에는 local 경로의 상태 교환이 0인 진짜 context parallelism이고, 단일 accelerator에서는 shard들을 batch 축에 쌓아 모든 kernel launch를 fatten한다. (c) $d\times d$ carry 하나로 잇는 Q-K projection prefix scan. Challenge 1의 <5-10% FLOPs utilization의 뿌리는 FLOPs 부족이 아니라 arithmetic intensity 부족인데, TNT는 병렬 작업 단위를 크게(global) 만들거나 많고-독립적으로(local) 만들어 intensity를 제조한다.

> **[해설]** FlashAttention tiling과의 구분을 다시 새길 지점이다(→ 1장 Rosetta, 9장). tiling은 bit-exact한 계산 재배열이다. TNT의 chunk와 reset은 **계산되는 함수 자체를 바꾼다** — chunk는 stale-snapshot 근사이고 reset은 정보를 실제로 버린다. TNT의 통찰은 그 의미론적 변경을 없애려 하지 않고, 변경분(잃어버린 long range)을 global memory라는 별도 부품으로 회수한 뒤 두 부품에 서로 다른 하드웨어 체질을 부여했다는 데 있다.

**Kernel 관점.** 전부 순수 JAX이고 fused kernel은 없으며 저자들이 명시적으로 future work로 남겼다 [TNT §5.2]. 그런데도 32K에서 TNT($C_{\mathrm{l}}=128$)가 FlashAttention을 step당 이긴다는 것은, kernel 이전에 **병렬 구조 자체**가 승부를 갈랐다는 뜻이다. GLA/DeltaNet 계열의 SRAM chunk kernel이 deep memory에 이식되지 않는 이유(chunk 간 상태 전파가 비선형을 통과)를 상기하면, TNT의 reset은 현재로서 유일한 지렛대다. 미래의 fused TNT kernel은 chunk당 batched forward+backward와 누적 합 상태 갱신의 융합이 될 것이다.

**State 크기 vs KV cache — 정량 비교.** decode 시점에 memory layer당 상태는 $(1{+}N)\,P_f + N d^2$개의 값이다(global + $N$개 local의 fast weights + $N$개 projection 행렬). Transformer는 layer당 $2Ld$를 들고 매 step 전량 재독한다.

> **[해설]** 수치를 넣어 보자(이 책의 산정; 원문에 decode 메모리 표는 없다). 이 라인의 표준 deep memory인 expansion 4의 2-layer MLP를 가정하면 $P_f\approx 8d^2$. $N=4$ local 구성이면 상태 ≈ $5\cdot 8d^2+4d^2=44d^2$. $d=1024$일 때 약 46M 값/layer로, bf16 기준 ~92MB다 — 결코 가볍지 않다. 그러나 KV cache $2Ld$가 이를 넘는 지점은 $L\gtrsim 22d$, 즉 $d=1024$면 **약 22K tokens**이고, 그 뒤로 KV cache는 무한히 자라는 반면 TNT 상태는 그대로다. 32K+ 문맥에서의 통상적 RNN 논증이지만, TNT는 여기에 serving 특화 이득 두 개를 얹는다. (1) Stage 2 덕분에 chunk-1 decode가 **품질 최적** 모드다 — Challenge 3에서 보았듯 baseline deep memory는 $C=64$ 훈련 후 $C=8$ decode만으로 ppl이 2.6× 이상 나빠졌는데(13.78→36.45 [TNT Fig. 2]), TNT는 자연스러운 autoregressive loop과 품질이 정렬된다. (2) prefill이 하드웨어에 깨끗하게 맵핑된다 — global은 prompt를 $C_{\mathrm{g}}$ 단위 dense batch로 삼키고 local shard들은 서로 병렬로 prefill되므로, time-to-first-token이 직렬 recurrence가 아니라 chunked linear pass처럼 스케일한다.

**Decode의 memory traffic.** $C_{\mathrm{l}}'=1$ decode는 token마다 local fast weights $P_f$개 값의 read-modify-write다: 상태를 읽고, forward+backward(~$6P_f$ FLOPs)를 돌리고, 다시 쓴다. 값당 FLOP이 한 자릿수이므로 이 갱신 자체는 memory-bound다 — KV cache 재독과 같은 체질이되, 트래픽이 $L$에 비례하지 않고 상수라는 점이 다르다. 한편 갱신 주기의 계층성은 상태의 저장 위치 계층과 자연스럽게 대응한다: 매 token 갱신되는 $W^{\mathrm{l}}$·$\Pi$는 가능한 한 on-chip에 상주시킬 대상이고, 2048 token에 한 번 큰 batch로 갱신되는 $W^{\mathrm{g}}$는 HBM에 두고 드물게 대량으로 만지는 대상이다. update frequency가 배치(placement)를 결정한다는 이 관찰은 16장의 CMS에서 아키텍처 원리로 승격된다.

**Batching과 운영.** per-request fast-weight 상태는 shared-weight batching을 깨뜨리므로(→ 1장 Rosetta, 10장), TNT decode의 local 경로 batching은 request별 weight를 갖는 grouped-GEMM 형태가 된다. 운영 측면에서 reset은 뜻밖의 선물을 준다: preemption/restore 때 재구성해야 할 local 상태 이력이 최대 $L_{\mathrm{s}}$ token으로 유계다 — shard 시작 상태는 언제나 상수 $W_{\mathrm{init}}$이기 때문이다. 다중 해상도 구성의 비용도 계산에 넣어야 한다: Stage 1 훈련 시간이 {8}의 3.06h에서 {4,8,16,32}의 5.55h로 늘고 [TNT Table 4], decode 비용도 $N$에 비례하며, 이 스케일에서 품질 대가는 module당 대략 -0.3 ppl이다 [TNT Table 3].

마지막으로 의무 caveat: 이 절의 prefill/decode 서사는 **아키텍처 논증이지 벤치마크가 아니다**. chunk-1 모드의 decode throughput/latency/메모리를 KV-cache Transformer와 맞대 잰 수치는 [TNT]에 없고, decode wall-clock 수치는 6편 전체에 부재하다.

## 15.8 한계와 bridge-out

**논문이 스스로 여는 문제와 이 책의 평가.**

첫째, **단순화 위의 검증.** TNT는 Titans의 momentum과 forget gating, Comba류 closed-loop objective, Adam/Muon inner optimizer를 전부 "명료성을 위해" 뺀 plain-GD memory로 검증되었고, 이들과의 결합은 명시적으로 future work다 [TNT App. D]. 17× 가속과 품질 이득이 완전한 Titans나 Atlas에 이식되는지는 측정된 바 없다. reset·Q-K projection이 gated/momentum update와 합성 가능한지, Stage 2가 그때도 전이되는지가 라인의 다음 실무 질문이다.

둘째, **Challenge 3의 증거 폭.** chunk-size mismatch의 증거는 단일 설정 — 550M, gating/momentum 없는 단순화 모델, $C=64$ 훈련 — 의 그림 하나다 [TNT Fig. 2]. 현상이 스케일·아키텍처 변형에 걸쳐 얼마나 보편적인지, 그리고 **왜** 생기는지에 대한 이론은 없다. chunkwise staleness의 오차 한계가 6편 어디에도 없다는 9장의 지적이 여기서도 유효하다: Stage 2가 mismatch를 고친다는 것은 알지만, 무엇이 고쳐졌는지는 모른다.

셋째, **reset 아래의 recall.** local memory는 설계상 $L_{\mathrm{s}}$ token마다 전부 잊고, global retrieval은 최대 $C_{\mathrm{g}}{-}1$ token 낡은 frozen 상태를 읽는다(식 (15-6)). shard 경계를 넘는 정밀 recall이 global 경로로 얼마나 생존하는지 — needle-in-a-haystack류 벤치마크 — 는 평가되지 않았다. Atlas와 [NL]이 측정한 in-context retrieval gap(→ 14·16장)을 생각하면 이 공백은 사소하지 않다.

> **[평가]** TNT의 교환은 명료하다: 병렬성을 사기 위해 local의 기억을 주기적으로 태우고, 그 보험을 저해상도 global에 든다. 이 보험의 실효성이 미측정이라는 것이 이 논문의 가장 큰 빈칸이다. 아울러 Stage 2의 기제 — 무엇이 몇 step에 걸쳐 재조정되는가, {8}→{1}은 23.99인데 {4,8,16,32}→{2,4,8,16}은 23.09인 이유는 무엇인가 [TNT Table 2] — 도 열려 있다. 하이퍼파라미터 기하($L_{\mathrm{s}}$ vs $C_{\mathrm{g}}$ vs $\{C_{\mathrm{l}}^{(i)}\}$의 동시 선택)는 실험이 $L_{\mathrm{s}}\in\{2048,4096\}$, $C_{\mathrm{g}}=2048$만 훑었으므로 사실상 미탐이다.

넷째, **스케일과 kernel.** 150M/10B tokens라는 실증 범위, Gated Transformer 대비 남은 ppl 격차(22.39 vs 23.09)와 time-to-loss 격차(0.96h vs 1.12h), 그리고 custom kernel의 부재. 논문은 이 셋 모두를 감추지 않고 명시한다 [TNT §5.2–5.3] — 이 라인에서 보기 드물게 systems 논문다운 정직성이다.

**Bridge-out: TNT → [NL].** TNT의 global/local 계층은 throughput으로 정당화된 트릭이다. 그러나 이 트릭이 심어 놓은 관념을 다시 보라: **서로 다른 주기로 갱신되는 부품들이 있고, 느린 시간 스케일에 주차된 지식은 빠른 스케일의 reset에서 살아남는다.** global은 2048 token마다, local은 매 token마다, $W_{\mathrm{init}}$과 slow weights는 훈련에서만 — TNT는 이미 3개 층위의 update frequency로 돌아가는 시스템이다. 다만 TNT에게 그것은 공학적 방편이었지, 왜 그것이 모델 전체의 조직 원리여야 하는지는 말하지 않는다.

[NL](→ 16장)이 바로 그 선언을 한다: 모델과 훈련 절차 전체가 각자의 update frequency로 자기 context flow를 압축하는 중첩된 optimization 문제들의 시스템이라는 존재론이다. TNT의 global/local 이분은 geometrically spaced 주파수의 연속체(Continuum Memory System)로 일반화되고, TNT의 reset은 Nested-CMS의 re-initialization으로, load-bearing해진 $W_{\mathrm{init}}$은 다섯 가지 knowledge-transfer 기제 중 하나(MAML류 initialization)로 분류된다. TNT가 유보했던 것들 — momentum, gating, 더 강한 inner optimizer — 도 NL에서 "optimizer 역시 memory다"라는 프레임 아래 재입장한다. 한편 Q-K projection은 후속 논문들이 재채택하지 않은, 이 논문 고유의 열린 실마리로 남는다. 훈련 경제학의 바닥을 깐 것이 TNT라면, 그 위에 세워질 층위의 존재론이 다음 장이다.
