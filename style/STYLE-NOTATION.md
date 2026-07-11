# STYLE-NOTATION — 통일 표기·용어·템플릿 기준

**버전**: v1.1 (2026-07-12, P2 보정 라운드 기준판 — 변경 이력은 §8. W1 판은 v1.0)
**지위**: 17개 장의 모든 집필자는 이 문서를 따른다. 이 문서와 충돌하는 표기·용어·구조는 audit(P2)에서 결함으로 처리된다. 변경 요청은 장 안에 `<!-- STYLE-ISSUE: ... -->` 주석으로 남기고, 본문은 일단 이 문서대로 쓴다.
**근거 문서**: `dossier/PRE-RESEARCH.md`(특히 §2 concept ledger), `notes/{2501.00663,2504.13173,2505.23735,2511.07343,2512.24695,2606.03979}.json`, `notes/prereq-curriculum.md`.

대상 독자(전 장 공통): transformer inference / efficient-transformer를 잘 아는 AI system infra·architecture exploration 엔지니어. **training 경험 전무.** KV cache, paged attention, FlashAttention tiling, GEMM shape, MFU, roofline, prefill/decode, batching, scan kernel은 안다. backward pass, optimizer state, loss surface는 모른다. 모든 장은 이 독자 한 명을 상정하고 쓴다.

---

## 0. 요약 카드 (집필 중 항상 옆에 둘 것)

**6대 규칙**

1. **$W$ = fast weights(inner-loop 상태), $\Theta$ = slow weights(outer-loop parameter). 예외 없음.** 이 구분이 책 전체에서 가장 중요한 표기 결정이다.
2. **상태와 함수를 분리한다**: memory 읽기는 $y_t=\mathcal{M}(q_t; W_t)$, 쓰기는 $W_{t-1}\to W_t$의 명시적 update 식. Titans의 "$\mathcal{M}(x)$ = write 동반 forward" 관행은 쓰지 않는다.
3. **게이트 3종은 항상**: inner learning rate $\eta_t$, momentum decay $\beta_t$, retention gate $\alpha_t\in[0,1]$ ($\alpha_t$는 **남기는 비율**: $\alpha_t W_{t-1}$, $\alpha_t=1$이면 전부 유지). Titans 원문과 방향·기호가 다르다 — §1.7.1 대응표 필수.
4. **inner loss는 $\ell$, outer(task) loss는 $\mathcal{L}$.** 소문자/대문자로 두 loop를 시각적으로 구분한다.
5. **chunk 크기 $C$ ≠ window 크기 $c$.** $C$는 병렬화 knob(TNT 이후로는 semantic knob이기도 함), $c$는 Omega rule의 objective 파라미터.
6. 기술 용어는 영어 원어(로마자) 유지, 조사는 한글 발음 기준으로 직접 결합("chunk가", "momentum이"). 번역·음차 금지(§2 허용 목록 제외).

**master update (책의 기준 수식 — 모든 Part II 장이 이 형태로 환원해 설명한다)**

$$
S_t \;=\; \beta_t\, S_{t-1} \;-\; \eta_t\, \nabla_W\, \ell\big(W_{t-1};\, k_t, v_t\big),
\qquad
W_t \;=\; \alpha_t\, W_{t-1} \;+\; S_t
\tag{M}
$$

읽기: $y_t=\mathcal{M}(q_t;W_t)$. $\ell$의 기본형은 associative-memory regression $\ell(W;k,v)=\|\mathcal{M}(k;W)-v\|_2^2$. 이 한 쌍이 "GD + momentum + weight decay가 sequence layer다"라는 라인 전체의 요약이며, 각 논문은 (M)의 성분을 바꾼 것으로 서술한다: Miras는 $\ell$과 retention을, Atlas는 $\ell$의 범위(window)와 optimizer(NS)를, TNT는 훈련 경제학을, NL은 층위를, Sleep은 lifecycle을 바꾼다.

---

## 1. 통일 표기법 (unified notation)

### 1.1 설계 원칙

6편은 같은 대상을 서로 다른(때로 상충하는) 기호로 쓴다. 통일 원칙:

| # | 원칙 | 이유 |
|---|---|---|
| P1 | 상태($W_t$)와 함수($\mathcal{M}$)의 분리 | Titans의 $\mathcal{M}$/$\mathcal{M}^*$(write 동반/read-only) 구분이 불필요해지고, Miras·NL의 $\mathcal{M}^\star$(argmin 최적해)와의 충돌이 해소된다 |
| P2 | fast/slow의 대문자 구분: $W$ vs $\Theta$ | training 무경험 독자가 "지금 어느 loop인가"를 기호만으로 판별하게 한다 |
| P3 | 다수결 + 표준 optimizer 관행 | $\eta$=learning rate(Miras·Atlas·TNT·NL·Sleep 다수), $\beta$=momentum decay(optimizer 문헌 표준; ch02와 일치) |
| P4 | retention 방향 통일: $\alpha_t$ = 남기는 비율 | Miras의 retention 재이론화(라인의 공식 입장)를 따른다. Titans만 반대 방향($\,(1-\alpha_t)W_{t-1}$)이므로 Titans 쪽을 변환한다 |
| P5 | 충돌 기호는 소수 사용처를 개명 | TNT의 global memory $V$(value 행렬과 충돌)→$W^{\mathrm{g}}$, TNT의 Q-K projection $\mathcal{M}_t$(memory와 충돌)→$\Pi_t$, Sleep의 RL 스칼라 $\gamma,\alpha$(gate와 충돌)→$\rho,\lambda_{\mathrm{KD}}$ |

### 1.2 예약 기호 전역표

아래 기호는 책 전체에서 이 의미로 예약된다. 장-국소 기호(그 장에서만 쓰는 보조 기호)는 이 표와 충돌하지 않는 한 자유롭게 도입하되, 첫 사용 시 정의한다.

**입력·차원·인덱스**

| 기호 | 의미 | 비고 |
|---|---|---|
| $x_t\in\mathbb{R}^{d}$ | 입력 token 표현 (열벡터) | 원문들의 $d_{in}$은 $d$로 통일 |
| $X\in\mathbb{R}^{L\times d}$ | 시퀀스 행렬 (행 = token) | |
| $L$ | sequence 길이 | Titans의 $N$, Sleep의 $T$ 혼용을 $L$로 통일. **layer 수·MLP 깊이 의미로 사용 금지** (v1.1) |
| $L_{\mathrm{layer}}$ | 네트워크·MLP의 layer 수(깊이) | v1.1 신설 — $L$(sequence 길이)과의 충돌 해소. memory MLP 깊이는 별도로 $L_{\mathcal{M}}$(Titans 문맥) 유지 |
| $d,\ d_k,\ d_v,\ d_h$ | model/key/value/memory-hidden 차원 | |
| $t,\tau$ | token 인덱스, **1-based** | |
| $n$ | chunk 인덱스, 0-based | chunk $n$은 token $nC{+}1 \ldots (n{+}1)C$ |
| $i,j$ | 보조 인덱스 (window 내 위치, 성분 등) | |
| $\ell$ (첨자) | level 인덱스 (NL/Sleep 문맥), layer 인덱스 | 함수 $\ell(\cdot;\cdot)$는 항상 괄호를 동반하므로 구분됨 |

**두 loop의 대상**

| 기호 | 의미 | 비고 |
|---|---|---|
| $W_t$ | fast weights = memory 상태 (inner loop) | deep memory이면 $W=\{W_1,W_2,\dots\}$ |
| $W_{\mathrm{init}}$ | meta-learn된 초기 memory 상태 ($W_0=W_{\mathrm{init}}$) | Titans의 $M_0$, TNT의 $W_{init}$ |
| $\Theta$ | slow weights 전체 (outer loop; projection, gate-producer, backbone) | |
| $\theta^{(\ell)}$ | level $\ell$의 parameter (NL/Sleep의 CMS 문맥) | 원문 $\theta^{(f_\ell)}$의 축약 |
| $\mathcal{M}(\cdot\,; W)$ | memory module (함수) | 축약형 $\mathcal{M}_t(\cdot):=\mathcal{M}(\cdot\,;W_t)$ 허용 |
| $\mathcal{M}^\star$ | inner 문제의 argmin 최적해 | Miras Def. 1 / NL Def. 1 전용. read-only pass 의미로 사용 금지 |
| $k_t,v_t,q_t$ | key/value/query (열벡터) | $k_t=W_K x_t$ 등 |
| $W_K,W_V,W_Q$ | 투영 행렬 (slow, $\Theta$의 일부) | inner loss의 "hyperparameter" |
| $K,V,Q\in\mathbb{R}^{L\times d_k \text{ or } d_v}$ | 행-쌓기 형태 | $K=XW_K^\top$ |
| $y_t$ | layer 출력 | TNT/NL의 $o_t$를 $y_t$로 통일 |
| $P=[p_1\ldots p_{N_p}]$ | persistent memory tokens ($N_p$개) | Titans |

**loss·gradient·surprise**

| 기호 | 의미 | 비고 |
|---|---|---|
| $\ell(W;k_t,v_t)$ | inner objective (per-token) | 기본형 $\|\mathcal{M}(k_t;W)-v_t\|_2^2$ |
| $L$ | attentional bias (inner objective의 family) | Miras Def. 3.1 |
| $\hat\ell_i,\ \tilde\ell_t$ | 선형화/surrogate inner loss | Miras FTRL 유도용, 그대로 유지 |
| $\mathcal{L}$ | outer(task) loss (next-token prediction 등) | |
| $g_t$ | 스코프상 gradient. 두 loop가 한 식에 공존하면 $g_t^{\mathrm{in}},g_t^{\mathrm{out}}$ | momentary surprise $= g_t^{\mathrm{in}} = \nabla_W\ell(W_{t-1};k_t,v_t)$ |
| $u_t$ | Local Surprise Signal, $u_t=\nabla_{y_t}\mathcal{L}$ | NL. 부호 관행은 ch16에서 국소 선언 |
| $\delta_\ell$ | backprop 오차 (layer $\ell$) | 층 첨자. Huber threshold $\delta_t$(시간 첨자)와 구분 |

**optimizer 상태·게이트**

| 기호 | 의미 | 비고 |
|---|---|---|
| $S_t$ | inner momentum buffer ("past surprise") | Titans/Atlas의 $S_t$ 유지 |
| $m_t,\ h_t$ | outer 1차/2차 moment (ch02의 optimizer-as-object) | Adam의 관행 $m_t,v_t$에서 $v_t$는 value와 충돌하므로 $h_t$ |
| $\eta_t$ | inner learning rate (data-dependent gate) | 무첨자 $\eta$ = outer learning rate (상수) |
| $\beta_t$ | inner momentum decay (data-dependent gate) | ch02의 상수 $\beta_1,\beta_2$(EMA 계수)와 같은 계열임을 명시 |
| $\alpha_t\in[0,1]$ | retention gate: $\alpha_t W_{t-1}$, 1=전부 유지, 0=완전 소거 | **방향 주의**: Titans 원문과 반대 (§1.7.1) |
| $\lambda$ | outer weight decay | |
| $\gamma_{t,i}\in[0,1]$ | Omega rule window gate (in-context pruning) | Atlas의 $\gamma_i^{(t)}$ |
| $\delta_t$ | Huber threshold (learned, input-dependent) | Miras/Yaad |
| $\mathrm{NS}_\kappa(\cdot)$ | Newton–Schulz $\kappa$회 반복 (semi-orthogonalization) | $\kappa=5$가 표준("NS-5"). 반복 횟수에 $k$ 금지(key와 충돌) |

**chunk·window·계층**

| 기호 | 의미 | 비고 |
|---|---|---|
| $C$ | chunk 크기 (병렬화 knob이자, TNT 이후, semantic knob) | Titans의 mini-batch $b$ 포함 |
| $c$ | Omega rule window 길이 (objective 파라미터) | $C$와 반드시 구분 |
| $\xi(t,C)$ | chunk 시작 offset: $\xi(t,C)=C\lfloor (t-1)/C\rfloor$ | gradient anchor는 상태 $W_{\xi(t,C)}$ |
| $C_{\mathrm{g}},\ C_{\mathrm{l}}^{(i)}$ | global / $i$번째 local chunk 크기 (TNT) | 원문 $C_G, C_{L_i}$ |
| $L_{\mathrm{s}}^{(i)}$ | shard 길이 = local memory reset 주기 (TNT) | 원문 $S_{L_i}$ — momentum $S_t$와 충돌하므로 개명 |
| $W^{\mathrm{g}},\ W^{\mathrm{l}(i)}$ | global / local memory 상태 (TNT) | 원문 $V, W^{(i)}$ — $V$는 value 행렬과 충돌하므로 개명 |
| $\Pi_t$ | Q-K projection 행렬 $\sum_\tau k_\tau k_\tau^\top/\|k_\tau\|^2$ (TNT) | 원문은 $\mathcal{M}_t$ — memory와 충돌하므로 개명 |
| $f_\ell$ (또는 $f_A$) | update frequency (NL Def. 2 / Sleep Def. 1) | |
| $C^{(\ell)}$ | level $\ell$의 chunk 크기 (CMS) | |
| $e_{i,\ell}$ | CMS level $\ell$의 누적 error 항 (Sleep Eq. 2) | |

**feature map·기타**

| 기호 | 의미 | 비고 |
|---|---|---|
| $\phi_p$ | 차수 $p$ polynomial feature map | Atlas |
| $\phi^*$ | exponential feature map ($\exp(q^\top k)=\phi^*(q)^\top\phi^*(k)$) | Atlas |
| $D$ | lifted 차원 $\Theta(d_k^p)$ | 장-국소 성격, Atlas 문맥 |
| $\mathrm{Ret}_t,\ R_t,\ D_t,\ G_t$ | retention 항 (Learning–Retaining / FTRL / local / global) | Miras 그대로 |
| $A_t$ | 보조 accumulator (Moneta, FTRL dual 등) | 장-국소 |
| $\sigma(\cdot)$ | activation (GELU/SiLU는 본문에서 명시) | |
| $\odot$ | elementwise 곱 | |
| $\mathcal{D},\ \mathcal{F}$ | dataset, divergence (Sleep/GKD) | |
| $\rho$ | Sleep의 reward 혼합 계수 (원문 $\gamma$) | window gate와 충돌하므로 개명 |
| $\lambda_{\mathrm{on}},\ \lambda_{\mathrm{KD}}$ | GKD on-policy 비율(원문 $\lambda$), distill-vs-reward 계수(원문 $\alpha$) | retention gate와 충돌하므로 개명 |
| $A^{(\ell),j}, B^{(\ell),j},\ d_{\mathrm{low}}$ | low-rank expert 행렬과 rank (Sleep) | |
| $L_{\mathcal{M}}$ | memory MLP 깊이 | Titans |
| $\mathrm{Reg}_L$ | regret (horizon $L$) | ch03; comparator는 $W^\star$ |

### 1.3 벡터·행렬·연산 규약

- **열벡터가 기본.** $k_t,v_t,q_t,x_t$는 열벡터. 투영은 $k_t=W_Kx_t$, $W_K\in\mathbb{R}^{d_k\times d}$. 원문들(특히 Titans)의 행벡터 관행 $x_tW_K$는 대응표에서만 언급.
- matrix memory는 $W\in\mathbb{R}^{d_v\times d_k}$, 읽기 $y=Wq$, 쓰기의 기본 단위는 outer product $v_tk_t^\top$.
- **$\otimes$ 사용 금지.** outer product는 $vk^\top$로, elementwise 곱은 $\odot$로, gating 결합은 명시적으로 ($y\odot g$ 또는 $\mathrm{gate}(y_1,y_2)$) 쓴다. 예외(v1.1 확장): **Kronecker 연산의 표준 선형대수 용법** — $\phi^*$ 정의의 거듭제곱 $x^{\otimes i}$, vec–Kronecker 항등식($(K^\top\otimes I)\,\mathrm{vec}(W)$ 류의 증명 스케치) — 은 허용한다. 금지 대상은 gating·outer-product 의미의 $\otimes$뿐이다.
- transpose는 $^\top$, Frobenius norm은 $\|\cdot\|_F$, $\ell_p$ norm은 $\|\cdot\|_p$.
- gradient는 $\nabla_W\ell$처럼 변수 명시. backprop 전개에서만 $\partial\mathcal{L}/\partial W_\ell$ 허용.
- softmax·layernorm은 $\mathrm{softmax}(\cdot)$, $\mathrm{LN}(\cdot)$.
- 표준 memory 구조(라인 공통): 2-layer residual MLP $\mathcal{M}(z;W)=z+W_1\,\sigma(W_2 z)$, expansion 4, GELU. 각 장은 이 형태를 "표준 deep memory"라고 부른다.
- 게이트의 data-dependence는 $\eta_t=\eta(x_t;\Theta)$처럼 "slow weights가 만드는 token의 함수"로 서술한다(low-rank head, 채널별 여부는 장에서 명시).

### 1.4 표준형 수식 (master equations)

Part II 각 장은 core mechanism을 **아래 표준형과의 차이**로 서술한다. 표준형 자체를 다시 유도하지 않는다(ch09, ch12가 담당).

**(M1) 기본 write/read (delta rule = 1-step GD)**

$$
W_t = W_{t-1} - \eta_t\,\nabla_W \ell(W_{t-1};k_t,v_t), \qquad y_t=\mathcal{M}(q_t;W_t)
\tag{M1}
$$

**(M2) master update — momentum + retention (Titans-LMM의 통일형)**

$$
S_t = \beta_t S_{t-1} - \eta_t\,\nabla_W\ell(W_{t-1};k_t,v_t),
\qquad
W_t = \alpha_t W_{t-1} + S_t
\tag{M2}
$$

**(M3) Omega rule — windowed inner objective (Atlas의 통일형)**

$$
W_t = \alpha_t W_{t-1} \;-\; \nabla_W \sum_{i=t-c+1}^{t} \gamma_{t,i}\,\big\|\mathcal{M}(\phi(k_i);W_{t-1})-v_i\big\|_2^2
\tag{M3}
$$

(관행: per-token step size는 $\gamma_{t,i}$에 흡수 가능. Muon 버전은 $S_t=\beta_tS_{t-1}-\nabla_W\sum_i\gamma_{t,i}\,\ell(\cdots)$, $W_t=\alpha_tW_{t-1}+\eta_t\,\mathrm{NS}_\kappa(S_t)$. 원문들의 부호·step-size 배치는 표기 관행 차이일 뿐 알고리즘 동일 — [Atlas App. D.4] 취지를 따라 이 책은 (M2)/(M3)의 부호로 고정한다.)

**(M4) chunkwise compression — 병렬화의 보편 형태 (TTT dual form / TNT Eq. 3의 통일형)**

$$
W_t = W_{\xi(t,C)} \;-\; \sum_{\tau=\xi(t,C)+1}^{t} \eta_\tau\,\nabla_W\,\ell\big(W_{\xi(t,C)};\,k_\tau,v_\tau\big)
\tag{M4}
$$

핵심 문장(모든 관련 장에서 반복 허용): **chunk 안의 모든 gradient는 chunk 시작 상태 $W_{\xi(t,C)}$에서 평가된다(stale-snapshot 근사). 따라서 $C$는 스케줄이 아니라 계산되는 함수 자체를 바꾸는 semantic hyperparameter다** — FlashAttention tiling(bit-exact)과의 결정적 차이.

**(M5) CMS update — level별 주기 갱신 (NL Eq. 71 / Sleep Eq. 2의 통일형)**

$$
\theta^{(\ell)}_{i+1} = \theta^{(\ell)}_{i} - e_{i,\ell},
\qquad
e_{i,\ell} = \begin{cases}
\displaystyle\sum_{t=i-C^{(\ell)}+1}^{i} \eta^{(\ell)}_t\, \varepsilon\big(\theta^{(\ell)}_t; x_t\big) & i \equiv 0 \pmod{C^{(\ell)}}\\[2pt]
0 & \text{otherwise}
\end{cases}
\tag{M5}
$$

$\varepsilon(\cdot)$은 임의 optimizer의 error 항(GD이면 $\nabla_\theta\mathcal{L}$).

### 1.5 기호 충돌·금지 사항 (전 장 공통)

| 항목 | 규칙 |
|---|---|
| $\mathcal{M}^*$ | **read-only pass 의미로 금지.** 읽기는 $\mathcal{M}(q;W)$. $\mathcal{M}^\star$는 argmin 전용 |
| $\otimes$ | 금지 (§1.3). 예외(v1.1): Kronecker 표준 용법 — $x^{\otimes i}$, vec–Kronecker 항등식 |
| $L$ (무첨자) | sequence 길이 전용 (v1.1). layer 수·MLP 깊이는 $L_{\mathrm{layer}}$, memory MLP 깊이는 $L_{\mathcal{M}}$ |
| $\theta_t$ (소문자, 시간 첨자) | inner lr 의미로 금지 (Titans 원문 관행). inner lr는 $\eta_t$ |
| $V$ | value 행렬 전용. TNT global memory는 $W^{\mathrm{g}}$ |
| $S$ | $S_t$=inner momentum 전용. shard 길이는 $L_{\mathrm{s}}$, Titans MAC의 segment는 "segment"라고 산문으로 |
| $k$ | key/보조 인덱스 전용. NS 반복 횟수는 $\kappa$, level 수는 $K_{\mathrm{lv}}$ 또는 산문("level 수") |
| $\gamma$ | window gate 전용(예외: Miras soft-threshold $\mathcal{S}_\gamma$는 ch13 장-국소 선언 후 사용). Sleep의 reward 혼합은 $\rho$ |
| $p,q$ (무첨자 스칼라) | norm 차수·polynomial 차수. $q_t$(query)와 혼동 금지 — query는 항상 첨자 동반 |
| $b$ | Titans의 inner mini-batch 크기 의미로 금지 → $C$ |
| 부호 관행 | momentum·update의 부호는 (M2)/(M3)로 고정. 원문과 다르면 대응표에 "부호 관행 차이(알고리즘 동일)"로 기록 |
| 첨자 스타일 | data-dependent 게이트는 반드시 시간 첨자($\eta_t$); 상수면 무첨자($\eta$). 이 구분 자체가 정보다 |

### 1.6 아키텍처 카탈로그 (통일 표기 한 줄 정의)

ch06/ch07/ch08/ch13 집필자는 아래를 기준형으로 사용한다(각 항목의 유도는 해당 장 담당).

| 모델 | 통일 표기 update ($y_t$는 별도 read) | (M)과의 차이 |
|---|---|---|
| linear attention | $W_t=W_{t-1}+v_tk_t^\top$ | Hebbian write, gate 없음 |
| RetNet | $W_t=\alpha W_{t-1}+v_tk_t^\top$ | 상수 decay |
| GLA / Mamba-2 | $W_t=\alpha_t W_{t-1}+v_tk_t^\top$ ($\alpha_t$: diag/스칼라, data-dep.) | 학습된 retention |
| DeltaNet | $W_t=W_{t-1}(I-\eta_tk_tk_t^\top)+\eta_tv_tk_t^\top$ | delta rule (= (M1)의 linear memory 전개) |
| Gated DeltaNet | $W_t=\alpha_tW_{t-1}(I-\eta_tk_tk_t^\top)+\eta_tv_tk_t^\top$ | delta + retention |
| Longhorn | delta 형태, $\eta_t \to \eta_t/(1+\eta_tk_t^\top k_t)$ (implicit GD), retention 없음 | inner optimizer 교체 |
| RWKV-7 | Gated DeltaNet의 채널별(벡터) gate 일반화 | gate 구조 |
| TTT-Linear / TTT-MLP | (M1), retention·momentum 없음, $\mathcal{M}$= linear / 2-layer MLP | 원형 |
| Titans-LMM | (M2), $\mathcal{M}$= 표준 deep memory | 라인의 기준점 |
| Moneta / Yaad / Memora | (M1)에서 $\ell\to\ell_p$/Huber/$\ell_2{+}$KL-retention 교체 | attentional bias·retention 축 |
| Atlas (OmegaNet 계열) | (M3) + $\phi_p$/$\phi^*$ + Muon($\mathrm{NS}_\kappa$) | window·optimizer 축 |
| Mesa-layer | $c=$ 전체 문맥의 (M3)를 Newton법으로 정확히 푼 극한 | 대조군 |
| softmax attention | $\ell_2$ regression의 **non-parametric** Nadaraya–Watson 해; 상태 = KV cache 그 자체 | 압축하지 않는 극한 ($f\to\infty$, capacity 무한) |
| TNT | (M4) + 계층: $W^{\mathrm{g}}$(큰 $C_{\mathrm{g}}$) + $W^{\mathrm{l}(i)}$(reset to $W_{\mathrm{init}}$, 주기 $L_{\mathrm{s}}^{(i)}$) + $\Pi_t$ | 훈련 경제학 |
| Hope | self-modifying Titans + CMS(M5) | NL의 종합 |

### 1.7 논문별 표기 대응표

**사용법**: Part II 각 장은 "core mechanism" 절 안에 자기 논문의 표를 **그대로 복사**해 "표기 대응표" 소절로 넣는다. 행 추가(장-국소 기호)는 허용, 행 삭제·의미 변경은 금지. 원 논문 표기는 이 표와 직접 인용문 안에서만 등장할 수 있다 — 본문 수식은 전부 통일 표기.

#### 1.7.1 [Titans] (2501.00663) — ch12

| Titans 원 표기 | 통일 표기 | 의미 / 주의 |
|---|---|---|
| $\mathcal{M}_t$ (상태 겸 함수) | $W_t$ (상태), $\mathcal{M}(\cdot;W_t)$ (함수) | 상태/함수 분리 (P1) |
| $\mathcal{M}(x)$ — write 동반 forward | update 식 + read로 분해해 표기 | |
| $\mathcal{M}^*(x)$ — read-only forward | $\mathcal{M}(x;W_t)$ | ★는 통일 체계에서 argmin 전용 |
| $\theta_t$ — inner learning rate | $\eta_t$ | ⚠ 기호 교차 주의 |
| $\eta_t$ — surprise(momentum) decay | $\beta_t$ | ⚠ Titans의 $\eta$는 lr가 **아니다** |
| $\alpha_t$ — forget gate, $(1-\alpha_t)\mathcal{M}_{t-1}$ | $\alpha_t^{\text{(통일)}} = 1-\alpha_t^{\text{(Titans)}}$ | **방향 반전**: 통일 $\alpha_t$는 남기는 비율 |
| $S_t$ — past surprise (momentum buffer) | $S_t$ | 동일 |
| $\nabla\ell(\mathcal{M}_{t-1};x_t)$ — momentary surprise | $g_t^{\mathrm{in}}=\nabla_W\ell(W_{t-1};k_t,v_t)$ | |
| $\mathbf{k}_t=x_tW_K$ (행벡터 관행) | $k_t=W_Kx_t$ (열벡터) | |
| $b$ — inner mini-batch(chunk) 크기 | $C$ | |
| segment $S^{(i)}$, 크기 $C$ (MAC) | "segment" (산문), 크기 $C$ | segment 크기 = chunk 크기로 동일시함을 명시 |
| $N_p,\ P,\ p_i$ — persistent memory | 동일 | |
| $d_{in}$ | $d$ | |
| $L_{\mathcal{M}}$ — memory 깊이 | 동일 | |
| $M_0$ — 초기 상태 | $W_{\mathrm{init}}$ | meta-learn됨 (암묵적) |
| $y_t=\mathcal{M}^*(q_t)$ | $y_t=\mathcal{M}(q_t;W_t)$ | |
| $\otimes$ — MAG의 gating 결합 | $y\odot g$ 또는 $\mathrm{gate}(\cdot,\cdot)$ | $\otimes$ 금지 |

#### 1.7.2 [Miras] (2504.13173) — ch13

| Miras 원 표기 | 통일 표기 | 의미 / 주의 |
|---|---|---|
| $\mathcal{M}$, $W$ (혼용) | $W_t$ / $\mathcal{M}(\cdot;W_t)$ | |
| $\mathcal{M}^*=\arg\min_\mathcal{M} L(\cdot)$ | $\mathcal{M}^\star$ | Titans의 read-star와 무관 |
| $L$ — attentional bias | $L$ | 동일 (공식 용어) |
| $\ell(W_{t-1};\mathbf{k}_t,\mathbf{v}_t)$ | $\ell(W_{t-1};k_t,v_t)$ | 동일 |
| $\hat\ell_i$ (선형화), $\tilde\ell_t$ (surrogate) | 동일 | |
| $\eta_t$ — inner lr ("meta in-context lr") | $\eta_t$ | 동일 |
| $\alpha_t$ — global retention 계수 ($\alpha\mathcal{M}_{t-1}$) | $\alpha_t$ | 동일 방향 (통일안의 기준) |
| $\beta_t\in[0,1]^d$ — decoupled retention gate (§5.3) | $\alpha_t$ 계열로 흡수; 병기 필요시 $\alpha^{\mathrm{g}}_t$ | ⚠ 통일 $\beta_t$(momentum decay)와 무관 |
| $R_t$ (FTRL), $\mathrm{Ret}_t$, $D_t$ (local), $G_t$ (global) | 동일 | |
| $A_t$ — dual accumulator (Moneta, FTRL형) | $A_t$ | 장-국소 |
| $\mathcal{S}_\gamma$ — soft-thresholding | $\mathcal{S}_\gamma$ (장-국소 선언 후 사용) | ⚠ window gate $\gamma_{t,i}$와 구분 |
| $p,q$ — $\ell_p$ bias / $\ell_q$ stability 차수 | 동일 | query $q_t$와 혼동 금지 |
| $\delta_t$ — Huber threshold (learned) | $\delta_t$ | backprop $\delta_\ell$과 첨자로 구분 |
| $\lambda_t,\eta_t',\zeta$ — KKT 유도 파생 계수 | 장-국소 유지 | |
| GD with momentum 행 (Titans 행) | (M2) | |

#### 1.7.3 [Atlas] (2505.23735) — ch14

| Atlas 원 표기 | 통일 표기 | 의미 / 주의 |
|---|---|---|
| $M_t$ — memory 상태 | $W_t$ | |
| $\eta_t$ — learning-rate gate | $\eta_t$ | 동일 |
| $\theta_t$ — momentum decay | $\beta_t$ | ⚠ Titans와 정반대 배치였음 — 통일로 해소 |
| $\alpha_t$ — weight-decay(forget) gate, $\alpha_tM_{t-1}$ | $\alpha_t$ | 동일 방향 |
| $S_t$ — momentum (windowed gradient 누적) | $S_t$ | 동일 |
| $\gamma_i^{(t)}$ — window gates | $\gamma_{t,i}$ | 동일 (첨자 표기만 정리) |
| $c$ — sliding window 길이 | $c$ | 동일. chunk $C$와 구분 |
| $\phi_p$, $\phi^*$ | 동일 | |
| NewtonSchulz-$k$ / NS5 | $\mathrm{NS}_\kappa$, $\kappa=5$ | 반복 횟수 기호 $k$ 금지 |
| Eq. 32–33 vs App. D.4의 부호·step-size 차이 | (M3)의 부호로 고정 | "부호 관행 차이(알고리즘 동일)"로 처리 |
| capacity의 $m$ (저장 쌍 수), $D=\binom{d_k+p}{p}$ | 동일 | 장-국소 |
| DLA/SWLA/OmegaNet/DeepTransformers/Dot/SWDT | 고유명사 유지 | §1.6 카탈로그 형태로 소개 |

#### 1.7.4 [TNT] (2511.07343) — ch15

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
| Eq. 5–6의 인덱스 표기 슬립 (원문 자체) | (M4)의 semantics로 통일 | 노트의 "faithfulness caveat" 유지 — ch15에서 각주로 명시 |

#### 1.7.5 [NL] (2512.24695) — ch16

| NL 원 표기 | 통일 표기 | 의미 / 주의 |
|---|---|---|
| Def. 1의 $\mathcal{M}^*$ | $\mathcal{M}^\star$ | |
| $W_t$ (inner), $\theta^{(f_\ell)}$ (level params) | $W_t$, $\theta^{(\ell)}$ | frequency 병기 필요시 $f_\ell$ 사용 |
| $u_{t}=\nabla_{y}L$ — Local Surprise Signal | $u_t$ | 동일 (공식 용어 LSS) |
| $\delta_\ell$ — backprop 오차 | $\delta_\ell$ | 동일 |
| $m_{\ell,t}$ — momentum (outer) | $m_t$ | |
| $\tilde M$, $H$ — Adam 1차/2차 moment | $m_t$, $h_t$ | |
| $f_A$, levels, $C^{(\ell)}$ | $f_\ell$ (또는 $f_A$), $C^{(\ell)}$ | 동일 |
| $\mathcal{M}_{\square,t}$, $\square\in\{k,v,q,\eta,\alpha,\mathrm{mem}\}$ — self-modifying memories | 동일 | |
| $\hat v_{\square,t}$ — self-generated target | 동일 | |
| DGD의 $\eta_t'=\eta_t/(1+\eta_t)$ | 장-국소 유지 | |
| M3의 $M^{(1)},M^{(2)},V_t$; $\beta_1,\beta_2,\beta_3$ | $m^{(1)},m^{(2)},h_t$; $\beta$들은 장-국소 상수 | ⚠ $V_t\to h_t$ (value와 충돌 방지) |
| $\mathrm{NewtonSchulz}_T$ | $\mathrm{NS}_\kappa$ | |
| $\otimes$ — outer product | $vk^\top$ 표기 | $\otimes$ 금지 |
| 고유명사: NSAM, Neural Learning Module, CMS, Hope, DGD/GGD/GM, Delta Momentum, DMGD, M3, AdaTransformer | 유지 | 첫 등장 시 정의 |

#### 1.7.6 [Sleep] (2606.03979) — ch17

| Sleep 원 표기 | 통일 표기 | 의미 / 주의 |
|---|---|---|
| $\boldsymbol\theta^{(f_\ell)}$, $C^{(\ell)}$, $f_W$ | $\theta^{(\ell)}$, $C^{(\ell)}$, $f_\ell$ | (M5) |
| $e_{i,\ell}$ — 누적 error 항 | $e_{i,\ell}$ | 동일 |
| $L$/$T$ — sequence 길이 혼용 | $L$ | |
| $\mathrm{LM}_{\boldsymbol\theta}$ (teacher), $\mathrm{LM}_{\boldsymbol\theta_{exp}}$ (student) | 동일 | teacher = 확장 전 자기 자신 |
| $\lambda$ — GKD on-policy 비율 | $\lambda_{\mathrm{on}}$ | ⚠ weight decay $\lambda$와 구분 |
| $\gamma$ — reward 혼합 계수 (Eq. 3) | $\rho$ | ⚠ window gate $\gamma$와 충돌 방지 |
| $\alpha$ — distill-vs-reward 계수 ($\mathcal{L}_{KS}$) | $\lambda_{\mathrm{KD}}$ | ⚠ retention gate $\alpha_t$와 충돌 방지 |
| $\mathcal{F}$, $\mathcal{D}$ — divergence | 동일 | |
| $r_{sem}, r_{abs}$; $z(\cdot,\cdot)$, $z_0$ — 보상·Levenshtein | 동일 | |
| $\mathbf{A},\mathbf{B}$, $d_{low}$ — low-rank expert | $A^{(\ell),j},B^{(\ell),j}$, $d_{\mathrm{low}}$ | |
| $s_\ell$ — expert 수 | 동일 | |
| $\mathcal{R}^{(f_\ell)}$ — MoE router | $\mathcal{R}^{(\ell)}$ | |
| $\boldsymbol\omega^{(i)}$, $g^{(i)}_{DR}$ — dream 중요도·gradient | 장-국소 유지 | |
| DREAM$^{(i)}$, KS/SKS/LTI, ReST^EM, SEAL | 고유명사 유지 | |

---

## 2. 용어 정책

### 2.1 영어 원어 유지 원칙

- **기술 용어(개념 명사)는 로마자 영어 그대로 쓴다. 번역도, 한글 음차도 금지.**
  - 예: momentum, weight decay, chunkwise training, gradient, backward pass, retention gate, surprise, fast weights, slow weights, inner loop, outer loop, associative memory, attentional bias, update frequency, consolidation, distillation, prefill, decode, KV cache, roofline, batching, kernel, state, chunk, window, capacity, regret, meta-learning, test-time memorization.
  - 금지 예: ~~운동량~~, ~~가중치 감쇠~~, ~~망각 게이트~~, ~~놀라움 신호~~, ~~연상 기억~~, ~~증류~~, ~~청크~~(한글 음차), ~~모멘텀~~(한글 음차).
- **동사·서술어는 자연스러운 한국어를 허용**한다: "memory에 쓴다/읽는다", "gradient를 계산한다", "state를 갱신한다". 단, 명사로 쓸 때는 영어("update rule", "write 연산").
- **한글 음차 허용 목록(관용어)**: 모델, 데이터, 시스템, 하드웨어, 소프트웨어, 알고리즘, 클러스터, 레이어(단, 수식 문맥에서는 layer). 이 목록 밖은 애매하면 로마자.
- 복수형 s는 쓰지 않는다("gradient들" 허용, "gradients" 지양). 고정 관용구는 예외: fast weights, slow weights, Titans.
- **"weights" 단독 사용 허용(v1.1 명문화)**: 신경망 가중치의 총칭은 영어 관용상 복수형이 표준이므로, "weights", "memory weights", "그 weights 자체가" 같은 단독 복수형 사용을 허용 예외로 확정한다(W1 전 장의 사실상 표준 관행 추인). 단 특정 행렬 하나를 지칭할 때는 기호($W$, $W_K$) 또는 "weight 행렬"을 쓴다. "weight decay"는 종전대로 단수형 고정 관용구.
- 대소문자: 문중 소문자 유지(momentum, chunk). 고유명사·모델명은 원문 표기(Titans, Miras, Moneta, Hope, Muon, DeltaNet, RWKV-7, Mamba-2, FlashAttention).

### 2.2 조사 결합 규칙

로마자 용어 뒤 조사는 **그 용어의 한글 발음을 기준**으로 선택하고, 붙여 쓴다.

| 용어(발음) | 옳음 | 틀림 |
|---|---|---|
| momentum(모멘텀) | momentum이, momentum을 | momentum가 |
| chunk(청크) | chunk가, chunk를 | chunk이 |
| gradient(그레이디언트) | gradient가, gradient를 | |
| kernel(커널) | kernel이, kernel을 | |
| inner loop(이너 루프) | inner loop가, inner loop를 | inner loop이 |
| GEMM(젬) | GEMM이, GEMM을 | |
| state(스테이트) | state가, state를 | |
| weight decay(웨이트 디케이) | weight decay가 | |
| prefill(프리필) | prefill이, prefill을 | |

수식 기호 뒤 조사도 같은 원칙: "$W_t$는", "$\eta_t$가", "$C$를".

### 2.3 처음 등장 시 정의 규칙

- 각 장에서 개념이 **처음 등장할 때 볼드체 + 한 줄 정의**. 정의 문구는 §2.4의 공식 정의와 모순되어서는 안 된다.
- 이전 장에서 이미 정의된 개념은 재정의하지 않고 참조한다: "retention gate(→ 13장)".
- Part II 장은 Part I의 정의를 전제한다. Part II에서 배경 개념을 다시 가르치기 시작하면 결함이다(한 문장 환기 + cross-ref까지만 허용).
- 정의 장 열(§2.4)이 "이 개념은 누가 정의하는가"의 단일 소유권이다. **자기 장이 소유하지 않은 개념의 정의를 바꾸지 말 것** — 병렬 집필 충돌의 최대 원인.

### 2.4 공식 명칭 확정표 (concept ledger 기반)

| 이 책의 공식 명칭 | 기원 | 정의 장(소유) | 비고 / 별칭·금지 표기 |
|---|---|---|---|
| surprise (momentary / past) | Titans | ch12 | momentary surprise $=g_t^{\mathrm{in}}$; past surprise $=S_t$. ~~놀라움~~ 금지 |
| momentum-as-memory | Titans→NL | ch02(객체로서), ch16(정리로서) | |
| retention gate | Miras | ch13 | **공식 용어.** "forget gate"는 역사적 별칭 — 첫 등장 시 1회 병기만 허용 |
| weight decay = per-token retention | Titans | ch12 | Titans 문맥에선 "forgetting mechanism" 표현 인용 가능 |
| attentional bias | Miras | ch13 | inner objective의 family |
| Learning–Retaining viewpoint / FTRL viewpoint | Miras | ch13 | ch03이 FTRL 자체를 소유 |
| Omega rule | Atlas | ch14 | "Ω-rule" 표기 금지, 항상 "Omega rule" |
| window gate $\gamma_{t,i}$ (in-context pruning) | Atlas | ch14 | |
| memory capacity (formal) | Atlas | ch14 | ch05가 고전(Hopfield) capacity 소유 |
| deep memory | Titans | ch12 | 표준형은 §1.3의 2-layer residual MLP |
| persistent memory | Titans | ch12 | |
| contextual memory | Titans | ch12 | |
| MAC / MAG / MAL | Titans | ch12 | 풀네임 1회 병기(Memory as Context/Gate/Layer) 후 약어 |
| fast weights / slow weights | (Schmidhuber; FWP) | ch06 | |
| fast weight programming (FWP) | Schlag et al. | ch06 | |
| delta rule | Widrow–Hoff | ch05 | DeltaNet(ch06)과 구분: rule vs 모델 |
| associative memory | 고전→Miras Def. | ch05(고전), ch13(Def. 정식화) | |
| inner loop / outer loop | 라인 공통 | ch01(비형식), ch04(형식화) | |
| test-time training (TTT) | Sun et al. | ch08 | 라인 전체의 총칭은 "TTT 계열" |
| test-time memorization | Atlas | ch14 | Atlas 이후 문맥에서 원문 구분(learning이 아니라 memorization) 존중; 두 용어의 구분 자체가 ch14의 논점 |
| dual form (TTT) | Sun et al. | ch08 (일반화는 ch09) | |
| chunkwise-parallel training | 라인 공통 | ch09 | 축약 "chunkwise training" 허용 |
| stale-snapshot 근사 (chunk-start anchor) | TTT/Titans | ch09 | "gradient anchor" 표현 허용 |
| chunk 크기 = semantic hyperparameter | TNT (발견), ch09 (명제화) | ch09 | 모든 장에서 이 명제 인용 가능 |
| chunk-size mismatch (train/serve) | TNT | ch15 | |
| hierarchical memory (global/local) | TNT | ch15 | |
| periodic state reset / context parallelism | TNT | ch15 | |
| meta-learned initial state $W_{\mathrm{init}}$ | Titans(암묵)→TNT(load-bearing) | ch04(개념: MAML류), ch15(역할) | |
| Q-K projection | TNT | ch15 | |
| two-stage training (train-big / serve-small) | TNT | ch15 | |
| update frequency / level | NL | ch16 | Sleep도 동일 정의 재사용 |
| Local Surprise Signal (LSS) | NL | ch16 | |
| Continuum Memory System (CMS) | NL | ch16 | |
| self-modifying Titans | NL | ch16 | "self-referential"은 Schmidhuber 계보 언급 시 |
| Hope / Hope-Attention | NL | ch16 | |
| DGD / GGD / GM / Delta Momentum / DMGD / M3 | NL | ch16 | |
| NSAM / Neural Learning Module | NL | ch16 | |
| knowledge-transfer taxonomy (5 mechanisms) | NL | ch16 | |
| wake/sleep lifecycle | Sleep | ch17 | |
| online vs offline consolidation | NL/Sleep | ch11(개념), ch16/ch17(기법) | |
| Knowledge Seeding (KS) / Self-Knowledge Seeding (SKS) | Sleep | ch17 | "upward distillation" 별칭 허용 |
| Learning to Imitate (LTI) | Sleep | ch17 | ⚠ 약어 LTI가 ch07의 LTI(linear time-invariant)와 충돌(v1.1 판정): 책 전역의 기본 의미는 linear time-invariant(→ 7장)이고, Learning to Imitate 의미는 **ch17 내부 전용**. ch17 첫 등장 시 "7장의 LTI(linear time-invariant)와 무관한 약어"라는 구분 문장 의무 |
| Dreaming | Sleep | ch17 | |
| synaptic-pruning reset | Sleep | ch17 | |
| parameter expansion (periodic (de)activation) | Sleep | ch17 | |
| catastrophic forgetting | 고전 | ch11 | |
| complementary learning systems (CLS) | 고전 | ch11 | |
| knowledge distillation / GKD | 고전 | ch11 | |
| regret / FTRL / OGD | 고전 | ch03 | |
| bilevel optimization / meta-learning / MAML | 고전 | ch04 | |
| Hopfield network / crosstalk | 고전 | ch05 | |
| linear attention / KV cache 압축 관점 | 고전 | ch06 | |
| SSD (state-space duality) | Mamba-2 | ch07 | |
| Muon / Newton–Schulz | Jordan et al. | ch02(객체로서), ch14(inner 사용) | |
| roofline / arithmetic intensity / MFU | 고전 | ch10 (Part I 내 사용은 자유) | 독자의 모국어 — 정의 불필요, 표기만 통일 |

### 2.5 논문 명칭·인용 표기

- 6편의 공식 축약: **[Titans], [Miras], [Atlas], [TNT], [NL], [Sleep]**. 첫 등장 시(장마다 1회) 풀네임+arXiv ID 병기: "[Titans] (*Titans: Learning to Memorize at Test Time*, arXiv:2501.00663)".
- Miras의 실제 제목은 *It's All Connected*지만 이 책은 framework 이름 **Miras**로 통칭한다(첫 등장 시 명시).
- 원문 위치 인용: [Titans §3.1], [Atlas Eq. 32], [NL Def. 2], [Sleep App. B]. 원문 수식 번호는 항상 "Eq."로, 이 책 수식 번호는 (12-3) 형식으로 — 혼동 불가.
- 외부 문헌: 저자-연도 + 첫 인용 시 arXiv ID. 예: "Sun et al. 2024 (arXiv:2407.04620)". 서지 목록은 후공정(P4, Veridraft 검증 풀)에서 일괄 구축하므로 본문에는 위 형식만.
- **모델명 약어 GDN(v1.1 신설)**: Gated DeltaNet의 약어 GDN은 **장마다 첫 등장 시 "Gated DeltaNet(이하 GDN)" 병기 후에만** 사용할 수 있다. 병기 없는 GDN 사용 금지. 한 장 안에서 병기 이후에는 GDN으로 통일한다(풀네임/약어 혼용 금지; 표 안 포함). 변형 명칭(GDN-H2 등)도 이 병기 이후에만 쓴다. 다른 모델명 약어를 새로 도입할 때도 같은 패턴을 따른다.

---

## 3. Rosetta-Stone 사전 (inference 어휘 ↔ 이 라인의 어휘)

**용도**: ch01이 이 사전을 본문으로 확장하고, 이후 모든 장은 새 개념 도입 시 이 대응을 재사용한다(재발명 금지). "대응의 성격" 열이 중요하다 — **동일**인지 **유비**인지 **차이가 논점**인지 구분하지 않으면 독자가 잘못된 직관을 이식받는다.

| inference 세계 (독자의 어휘) | 이 라인의 어휘 | 대응의 성격 |
|---|---|---|
| KV cache | non-parametric memory state; softmax attention = capacity 무한($\phi^*$)의 associative memory | **동일 대상의 재서술** — attention은 압축하지 않는 memory |
| KV cache append | memory **write** (outer-product Hebbian write가 최근접 대응; delta rule은 append가 아니라 **overwrite**) | 유비 + 차이: cache는 append-once, fast weights는 read-modify-**write** |
| attention lookup (q·K) | memory **read** $y_t=\mathcal{M}(q_t;W_t)$ | 동일 추상화 (k→v 연상) |
| linear-RNN state ($d\times d$) | matrix memory = 고정 크기 lossy 압축 memory | 동일 |
| prefill | 큰 chunk의 병렬 write (compression) — TNT에선 global memory가 담당 | 유비(정확) |
| decode | $C=1$의 per-token online write + read | 유비(정확) — 단 **backward pass가 decode에 들어온다**는 점이 신세계 |
| FlashAttention tiling | chunkwise-parallel training의 chunk | ⚠ **차이가 논점**: tiling은 bit-exact, chunk는 함수 자체를 바꾸는 semantic 근사 (M4) |
| GEMM shape 감각 | $\nabla_W\ell$의 outer-product 구조: $dW = (\text{오차})\,k^\top$는 rank-$C$ GEMM | 동일 — 훈련 수식을 GEMM shape로 읽는 것이 이 책의 교수법 |
| scan/prefix-sum kernel | momentum의 associative scan (S5식), $\Pi_t$의 누적 합 | 동일 |
| roofline / arithmetic intensity | chunk 크기 $C$가 arithmetic intensity를 결정; 품질 최적 $C$(작음) vs MFU 최적 $C$(큼)의 긴장 | 동일 도구, 새 독립변수 |
| batching (shared weights 전제) | per-request fast-weight state는 shared-weight batching을 **깨뜨린다** → grouped-GEMM decode | ⚠ 차이가 논점 |
| paged KV cache / session cache | per-session weight state — 새로운 cache class (sizing, checkpoint/restore, eviction) | 유비 → Part III의 주제 |
| cache eviction policy | retention gate = **학습된** eviction | 유비(정확) |
| speculative decoding의 rollback | memory state snapshot/rollback (optimizer-trajectory 상태 포함) | 유비 + 미해결 문제 |
| optimizer state (경험 없음 → ch02에서 도입) | 훈련의 "register file / accumulator": $m_t,h_t$ | 도입용 유비 |
| weight update 없음(추론 불변식) | **폐기되는 불변식** — 이 라인의 정의적 특징 | 차이 그 자체 |
| sequence 축 | 훈련의 batch 축 역할을 **sequence 축이 대신한다** (inner loop의 mini-batch = chunk) | ⚠ 최대 혼동 지점 — ch02·ch09에서 반복 강조 |
| distributed training (경험 없음) | context parallelism: reset이 sequential chain을 끊어 shard 병렬화 (TNT) | 도입용 유비 (data parallelism과의 대비) |
| 서빙 fleet의 백그라운드 job | sleep phase = 주기적 offline consolidation/dreaming job ("weights의 background compaction") | 유비 → ch17, Part III |

---

## 4. 장 템플릿

공통: 각 장은 하나의 `.md` 파일(§5.1의 슬러그). 제목은 `# chNN. <제목>` 한 줄. 수식·그림·표 번호는 §5. 분량 가이드는 `dossier/PRE-RESEARCH.md` §4의 페이지 배정을 따른다(1p ≈ 500–550 한국어 단어 + 수식).

### 4.1 Part I 장 템플릿 (ch01–ch11)

```markdown
# ch05. Associative memory: Hopfield에서 delta rule까지

> **이 장의 목표** — 독자가 이 장을 마치면 할 수 있어야 하는 것 2–4개 (동사로).
> **왜 필요한가** — 6편 중 어느 논문의 어떤 절/수식이 이 장을 요구하는지 명시.
>   예: "[Miras Def. 3.1]과 [Atlas §3.1]의 capacity 정리는 이 장의 crosstalk 논의를 전제한다."

## 5.1 <본문 절> ... (필요한 만큼; 절마다 systems 독자 관점의 접점 한 번 이상)
## 5.n Worked micro-example
   — 손으로/암산으로 따라갈 수 있는 최소 수치 예제 (d=2~4 수준, 실제 숫자).
   — Part I의 모든 장에 필수. "코드 없이 표와 수식만으로" 원칙.
## 요약
   — 5–8개 bullet. 각 bullet은 단정문.
## 자가 점검 체크리스트
   — [ ] 형식 4–7개. "…를 설명할 수 있다 / …를 계산할 수 있다" 형태.
   — 마지막 항목은 항상 Rosetta 항목: "…를 inference 어휘로 옮길 수 있다."
## 다음 장으로
   — 이 장이 연 질문 → 다음 장이 답할 질문. 2–4문장.
```

작성 지침:
- 모든 훈련 개념은 **(state, update, cost)를 갖는 객체**로 제시한다(커리큘럼의 설계 원칙). "왜 필요한가" 박스가 6편과의 연결을 놓치면 결함.
- 각 장 말미 요약 직전에 **systems bridge** 내용을 배치한다(별도 절로 빼도 좋음): FLOP/byte 계산, GEMM shape, 독자의 production 감각과의 접점. `notes/prereq-curriculum.md`의 해당 모듈(B0–B9)의 "Systems bridge" 항목이 최소 요구사항이다.
- 장별 대응: ch01=B0, ch02=B1, ch03=B2, ch04=B3, ch05=B4, ch06=B5, ch07=B5b, ch08=B6, ch09=B7, ch10=B8, ch11=B9. 커리큘럼의 canonical refs를 그 장의 인용 기반으로 사용한다.

### 4.2 Part II 장 템플릿 (ch12–ch17)

```markdown
# ch14. Atlas: Learning to Optimally Memorize the Context at Test Time

## 14.1 Bridge-in: 전작에서 남은 문제
   — 전작(또는 ch12는 ch08의 TTT)이 명시적으로 남긴 open question을 인용으로 복원.
   — dossier §1의 해당 Bridge 절(무엇을 계승/일반화/폐기했는가)을 근거로 쓰되, 본문은 자립적으로.
## 14.2 문제의식
   — 이 논문이 스스로 정의한 문제. 논문의 주장과 이 책의 재구성을 §6.2 규칙으로 구분.
## 14.3 Core mechanism (통일 표기)
   — 모든 수식은 통일 표기. (M1)–(M5) 표준형과의 차이로 서술.
   — 원문 수식 번호를 [Atlas Eq. 9]처럼 병기해 독자가 원문과 대조 가능하게.
### 14.3.x 표기 대응표
   — §1.7의 해당 표를 그대로 복사 + 장-국소 행 추가 (행 삭제·의미 변경 금지).
## 14.4 Outer-loop training vs inner-loop test-time learning
   — 필수 독립 절. 이 논문에서 무엇이 outer에서 학습되고(projection, gate-producer,
     W_init, ...) 무엇이 inner에서 움직이는지(W_t, S_t, ...) 표 하나 + 산문.
   — "이 gate는 누가 학습하는가?"에 전부 답할 것 (training 무경험 독자의 1번 질문).
## 14.5 Concept ledger delta
   — 이 논문이 새로 추가한 개념 / 전작 개념을 확장·개명한 것 / 폐기한 것. 표 형식.
   — dossier §2 ledger의 해당 행들과 정합해야 함 (P2 continuity audit 대상).
## 14.6 실험과 스케일
   — 모델 크기·토큰 수·벤치마크를 정확한 수치로. 논문의 수치 주장에는 반드시 출처 절 병기.
   — 스케일 한계(이 라인의 상한 1.3B/100B tokens)를 이 절에서 정직하게.
## 14.7 Systems/serving 함의
   — 독자의 세계로 번역: decode 비용, state 크기, batching·kernel·memory 함의.
   — D4 제약: memory-centric 논증은 (a) decode state RMW 트래픽, (b) update-frequency
     ↔ memory 계층 배치, 두 지점 외 금지.
## 14.8 한계와 bridge-out
   — 논문 자신의 open questions + 이 책의 평가(§6.2 구분 표시) + 차작이 무엇을 받아가는지.
```

작성 지침:
- 장별 전작 관계: ch12←ch08(TTT), ch13←ch12, ch14←ch13, ch15←ch14, ch16←ch15, ch17←ch16. bridge-in/bridge-out은 이 사슬을 따른다.
- core mechanism 절의 사실 기반은 `notes/<arxiv-id>.json`의 `core_mechanism`이다. 노트에 없는 주장을 추가하려면 원문 확인 후 원문 위치를 병기하거나 `TODO-VERIFY`.
- §6.4 정직성 규칙의 해당 항목(예: ch14의 Muon ablation)은 그 장의 의무 서술 사항이다.

### 4.3 분량 가이드 (전 장 공통)

| 장 | 목표 pp. | 장 | 목표 pp. |
|---|---|---|---|
| ch01 | 4 | ch10 | 5 |
| ch02 | 10 | ch11 | 4–7 |
| ch03 | 6 | ch12 | 10 |
| ch04 | 6 | ch13 | 8 |
| ch05 | 8 | ch14 | 10 |
| ch06 | 10 | ch15 | 7 |
| ch07 | 6 | ch16 | 13 |
| ch08 | 7 | ch17 | 9 |
| ch09 | 10 | | |

(v1.1 재배정: ch01 3→4, ch16 11→13 — 근거·감축 지시는 `audit/w1-fixplan.md` §0. 그 외 장은 목표 유지, 허용 오차 ±15%.)

---

## 5. Cross-reference 규약

### 5.1 파일명 슬러그 (고정 — 임의 변경 금지)

```
study-kr/part1/ch01-orientation-rosetta.md
study-kr/part1/ch02-training-as-a-system.md
study-kr/part1/ch03-online-learning-regret.md
study-kr/part1/ch04-meta-learning-bilevel.md
study-kr/part1/ch05-associative-memory.md
study-kr/part1/ch06-linear-attention-fwp.md
study-kr/part1/ch07-ssm-lineage.md
study-kr/part1/ch08-ttt-lineage.md
study-kr/part1/ch09-chunkwise-parallel-training.md
study-kr/part1/ch10-systems-bridge.md
study-kr/part1/ch11-continual-learning.md
study-kr/part2/ch12-titans.md
study-kr/part2/ch13-miras.md
study-kr/part2/ch14-atlas.md
study-kr/part2/ch15-tnt.md
study-kr/part2/ch16-nested-learning.md
study-kr/part2/ch17-sleep.md
```

(v1.1: W1 실파일명을 기준으로 확정 — ch01/ch03/ch04/ch09의 슬러그를 지시서 경로 쪽으로 개정. 파일 rename 없음. 관련 `<!-- STYLE-ISSUE -->` 자기신고 주석 3건은 제거 대상.)

그림 파일: `study-kr/figures/chNN/fig-NN-<slug>.svg` (예: `figures/ch09/fig-02-three-regimes.svg`). 그림 스펙의 단일 SoT는 `figures/SPEC.md`(v1.1 신설, P2 figure pass에서 생성).

### 5.2 장·절 참조

- 산문 내 참조는 **텍스트로 통일**: "12장", "12장 §12.3", "→ 9장". Markdown 링크는 1차 초안에서 쓰지 않는다(pandoc 병합 시 P2에서 일괄 처리; 앵커 규칙을 지금 고정하면 한국어 heading id 문제로 전원이 깨진 링크를 만든다).
- 아직 없는 장을 참조해도 된다(병렬 집필 전제). 단 참조 대상은 반드시 위 슬러그의 장 번호.

### 5.3 수식 번호

- 표시 수식 중 **재참조되는 것만** 번호를 붙인다: `$$ ... \tag{12-3} $$` — 장 번호-일련번호, 장 내 1부터.
- 참조는 "식 (12-3)". 다른 장 수식은 "식 (9-2)(→ 9장)".
- 이 문서의 표준형 (M), (M1)–(M5)는 전역 번호로 그대로 인용 가능: "식 (M2)".
- 원 논문 수식은 [Titans Eq. 13] 형식(§2.5) — 책 번호와 절대 혼용 금지.

### 5.4 그림·표

- 그림: "그림 12-1" + 캡션 필수. `![그림 12-1 — MAC의 데이터 흐름](../figures/ch12/fig-01-mac-dataflow.svg)` 후 다음 줄에 캡션 텍스트 반복(pandoc 빌드 안전용).
- 표: "표 12-1" + 캡션을 표 **위**에. 표기 대응표도 표 번호를 받는다. 캡션 텍스트는 플레인으로 쓴다("표 12-1 — …") — **볼드 캡션 금지**(v1.1 통일 결정).
- 원 논문 그림을 다시 그릴 때 캡션에 "([Atlas Fig. 3] 재구성)" 표기. 원본 이미지 복사 금지(재작도 원칙).
- 넓은 표는 축약열보다 행 분할을 선호(빌드 폭 제한).

### 5.5 노트·dossier 참조

- 본문에서 `notes/*.json`이나 dossier를 직접 인용하지 않는다(내부 작업 문서). 사실의 출처는 항상 원 논문 위치로 귀속시킨다.

---

## 6. 문체 규칙

### 6.1 기본 문체

- 문어체 평서형("-이다/-한다"). 경어체 금지.
- **단정적으로 쓴다.** "~인 것 같다", "~로 보인다", "아마도" 금지. 단정할 수 없으면 (a) 논문 주장으로 귀속시키거나(§6.2) (b) `TODO-VERIFY`(§6.3)로 처리한다. hedging은 둘 중 하나를 회피한 것이다.
- 한 문단 = 한 논지. 두괄식. 수식 앞뒤로 "이 식이 무엇을 말하는가" 산문 한 문장 이상.
- 독자에게 말 걸기("여러분", "~해 보자")는 worked example 내부에서만 최소한으로.

### 6.2 논문 주장 vs 이 책의 해설 구분 (필수)

세 층위를 표기로 구분한다:

1. **논문의 주장/보고**: 항상 논문에 귀속 + 위치 병기.
   - "…이다. [Atlas §5.2]" (검증된 수식·정의 등 사실 서술)
   - "[Titans]는 이 update가 $TC^0$ 너머의 표현력을 준다고 주장한다(Thm 4.1, 증명 없음)." (논쟁적 주장은 "주장한다/보고한다"로)
2. **이 책의 해설/번역** (논문 내용을 독자의 어휘로 재서술): 일반 산문으로 쓰되, 논문이 하지 않은 해석임이 문맥상 드러나야 한다. 경계가 흐려질 위험이 있으면 블록으로:
   > **[해설]** inference 관점에서 이 식은 KV cache append를 rank-1 GEMM write로 바꾼 것이다. …
3. **이 책의 평가/비판** (논문과 다른 판단): 반드시 블록으로.
   > **[평가]** 이 ablation은 Muon 축의 가치가 760M 스케일에서 미결임을 뜻한다. …

### 6.3 TODO-VERIFY

확신 없는 서술은 본문에 남기지 않는다. 대신:

```markdown
<!-- TODO-VERIFY: Atlas의 BABILong 셋업이 MAC 변형인지 원문 §5.4 확인 필요. 확인 방법: papers/2505.23735.txt 검색 "BABILong" -->
```

- 위치: 해당 문장/절 바로 아래. 주장 + 확인 방법(어느 파일/절을 보면 판정되는지)을 반드시 포함.
- 수치는 TODO-VERIFY 상태로 본문에 쓰지 않는다 — 수치는 노트/원문에서 확인된 것만.

**해소 규약 (v1.1 신설)** — TODO-VERIFY를 해소할 때:

1. 원문(papers/·arXiv·PDF)에서 판정을 확인한 뒤 **주석을 삭제**한다. "확인 완료" 류의 잔류 주석을 본문에 남기지 않는다 — 해소 이력은 `audit/w1-verify-answers.md`에 (TODO 위치, 판정, 출처) 형식으로 기록한다.
2. 주장이 **확인**되면: 본문 해당 문장에 원문 위치 병기([X §n] / [X Eq. n] / [X Fig. n] / 저자-연도+arXiv ID)를 추가한다. 흐름을 깨면 각주로 대체해도 된다.
3. 주장이 **틀렸으면**: 본문을 원문에 맞게 고치고 2를 적용한다. 고친 결과가 다른 장의 서술과 연동되면(fixplan에 연동 표기) 해당 장 fixer에게 `audit/w1-verify-answers.md`를 통해 전달한다 — 남의 장 파일을 직접 고치지 않는다.
4. **판정 불가**(원문 미공개·부재)이면: 본문 단정을 삭제하거나 §6.2의 논문 귀속("논문은 명시하지 않는다")/[평가] 블록으로 전환한다. TODO-VERIFY를 무기한 잔류시키는 것은 결함이다.
5. 수치 금지 규칙은 해소 전까지 유지된다(위 두 번째 불릿).

### 6.4 정직성 규칙

논문에 불리한 사실의 완곡화·누락은 결함이다. 아래는 dossier가 확정한 **의무 서술 caveat**로, 해당 장은 반드시 본문에 담는다:

| Caveat | 의무 장 |
|---|---|
| Atlas 자체 ablation에서 Muon 제거가 perplexity를 **개선** (optimizer 축의 가치는 760M에서 미결) | ch14 |
| in-context retrieval gap 미해소: attention 53.55 vs 43.70 [Atlas Table 5], FDA 67.3 vs 41.9 [NL] | ch14, ch16 |
| TNT는 momentum·gating·Muon을 "명료성을 위해" 제거한 단순화 Titans로 검증 (App. D) — 라인 전체와의 합성은 미검증 | ch15 |
| chunk-size mismatch의 증거는 단일 설정(550M, gating/momentum 없음)의 한 그림 | ch15 |
| Sleep의 기제는 pre-trained Llama/Qwen 위의 graft — 라인 최초로 end-to-end meta-learn되지 않음 | ch17 |
| 라인 전체의 실증 상한 = 1.3B params / 100B tokens (TNT는 150M); decode wall-clock 수치는 6편 전체에 부재 | ch12–ch17 각 §"실험과 스케일" 및 §"systems 함의" |
| Titans Thm 4.1($TC^0$ 초과)은 v1에 증명 없음; NL도 정리 아닌 실증 | ch12, ch16 |
| chunkwise staleness의 오차 한계는 6편 어디에도 없음 | ch09, ch15 |

### 6.5 숫자·단위·표기

- 모델 크기 "760M", "1.3B"; 토큰 수 "100B tokens"; 문맥 길이 "10M tokens", "32K".
- 배율은 "17.4×" (×는 숫자 뒤 붙임). 퍼센트 "5%".
- perplexity는 소수점 둘째 자리까지 원문 그대로 (13.78). 반올림 금지.
- 범위는 en-dash 없이 물결 지양, "4K–1M"처럼 하이픈 허용.
- 논문 수치는 항상 출처 병기: "ppl 36.45 [TNT Fig. 2]".

---

## 7. 집필자 pre-flight checklist

집필 시작 전:
- [ ] 이 문서 §0 요약 카드와 자기 장의 대응표(§1.7)/정의 소유권(§2.4)을 확인했다.
- [ ] 자기 장의 사실 기반(`notes/*.json` 해당 필드, `prereq-curriculum.md` 해당 모듈)을 읽었다.
- [ ] 자기 장이 소유하지 않는 개념의 정의 장을 확인했다(재정의 금지, 참조만).

제출 전 자가 감사:
- [ ] 본문 수식에 원 논문 고유 표기가 남아 있지 않다(대응표·직접 인용 제외).
- [ ] $\eta_t/\beta_t/\alpha_t$의 의미가 §1.2와 일치한다 (특히 Titans 장: 방향 반전 처리).
- [ ] $W$ vs $\Theta$, $\ell$ vs $\mathcal{L}$ 구분이 전 수식에서 지켜졌다.
- [ ] 모든 수치 주장에 원문 위치가 병기되어 있고, 미확인 주장은 `TODO-VERIFY`로 빠졌다.
- [ ] 논문 주장/해설/평가의 3층 구분(§6.2)이 적용되었다.
- [ ] (Part II) 8개 절 구조 완비, 특히 §"outer vs inner" 절과 "표기 대응표" 소절.
- [ ] (Part I) worked micro-example, 체크리스트, systems bridge 존재.
- [ ] §6.4의 자기 장 의무 caveat이 본문에 있다.
- [ ] 용어 첫 등장 볼드+정의, 조사 결합 규칙(§2.2) 준수.
- [ ] 수식/그림/표 번호가 장번호-일련번호 규칙(§5)을 따른다.

---

## 8. 버전 이력

### v1.1 (2026-07-12) — P2 보정 라운드 기준판

W1 3종 감사(`audit/w1-{notation,continuity,coverage}-report.md`)가 요구한 기준 문서 수정을 반영. 각 항목의 근거 리포트 ID를 병기한다.

1. **§5.1 파일 슬러그 개정** — ch01/ch03/ch04/ch09 슬러그를 W1 실파일명(`ch01-orientation-rosetta`, `ch03-online-learning-regret`, `ch04-meta-learning-bilevel`, `ch09-chunkwise-parallel-training`)으로 확정. 파일 rename 없음, 장 내 STYLE-ISSUE 자기신고 주석 3건 제거 대상. [notation M-1, coverage minor-7]
2. **§2.1 "weights" 단독 복수형 허용 예외 명문화** — 전 장의 사실상 표준 관행 추인. [notation m-9]
3. **예약 기호 $L$ 충돌 해소** — §1.2에 $L_{\mathrm{layer}}$(네트워크·MLP 깊이) 신설, §1.5에 금지행 추가. $L$은 sequence 길이 전용, $L_{\mathcal{M}}$(memory MLP 깊이)은 유지. [notation M-2]
4. **§6.3 TODO-VERIFY 해소 규약 신설** — 해소 시 주석 삭제 + 원문 위치 병기(또는 각주) 대체, 이력은 `audit/w1-verify-answers.md`, 판정 불가 시 §6.2 전환. [coverage §3, major-5]
5. **§2.5 GDN 약어 도입 규칙 신설** — "Gated DeltaNet(이하 GDN)" 장별 첫 등장 병기 의무 + 장 내 혼용 금지. [notation m-5]
6. **§1.3·§1.5 $\otimes$ 예외 확장** — Kronecker 표준 선형대수 용법(vec–Kronecker 항등식 포함) 허용. ch14:31은 소급 적합. [notation m-1]
7. **§6.4 retrieval gap 수치 정정** — 53.6/43.7 → 원문 정밀도 53.55/43.70 [Atlas Table 5]. 기준 문서 자신이 §6.5 반올림 금지 규칙을 위반하고 있었음. [continuity F-06]
8. **§2.4 LTI 약어 충돌 판정 병기** — 전역 기본 의미 = linear time-invariant(ch07), Learning to Imitate는 ch17 내부 전용 + 첫 등장 구분 문장 의무. [continuity F-07]
9. **§4.3 분량 재배정** — ch01 3→4pp(worked-example 일부 존치 인정), ch16 11→13pp(NL의 최고 밀도 인정; 상세 근거는 `audit/w1-fixplan.md` §0 D7·D8). 그 외 목표 유지, 허용 오차 ±15% 명문화. [coverage §4]
10. **§5.4 표 캡션 서식 확정** — 플레인 캡션으로 통일(볼드 금지). [notation m-6]
11. **§5.1 그림 스펙 SoT 지정** — `figures/SPEC.md` 신설 참조. [coverage major-3]

### v1.0 (2026-07-11) — W1 동결판

17개 장 1차 초안 집필의 기준. W1 기간 동결.

---

*v1.1은 P2 보정 라운드(W2)의 기준판이다. 라운드 중 새 표기 충돌을 발견하면 종전처럼 본문에 `<!-- STYLE-ISSUE: ... -->` 주석을 남기고 이 문서대로 계속 쓴다. 다음 개정은 P2 종료 시 일괄 결정한다.*
