# ch02. training 레짐 지도 — 무엇이 언제 움직이는가

> **이 장의 목표** — 이 장을 마치면 독자는 다음을 할 수 있어야 한다.
> 1. 하나의 배포 checkpoint가 태어나기까지 거치는 레짐(pre-training · SFT · RLHF/RLVR · continual)을 이름으로 구분하고, 각 레짐에서 $\Theta$가 무엇에 의해 얼마나 움직이는지 말할 수 있다.
> 2. 임의의 training run을 다섯 구성요소(데이터 · 목적함수 · optimizer · 스케줄 · 평가)로 분해하고, 각 구성요소가 어느 자원을 먹는지 지목할 수 있다.
> 3. token budget $D$, step 수 $n_{\mathrm{step}}$, batch 크기 $N_{\mathrm{b}}$, epoch 수 $n_{\mathrm{ep}}$의 관계식을 쓰고, $6ND$로 한 레짐의 FLOPs를 즉석에서 산정할 수 있다.
> 4. 임의의 논문에 대해 **"이 값은 누가 학습하는가"**를 네 질문으로 판정하고, 그 결과를 (U-W)/(U-$\Theta$)/(U-E) 중 하나에 배치할 수 있다.
>
> **왜 필요한가** — Part II의 세 장이 이 장의 어휘를 전제한다.
> - **ch20 [SEAL]** (*Self-Adapting Language Models*, arXiv:2506.10943): SEAL이 "inner loop"라고 부르는 것은 이 라인의 inner loop(→ NM ch04)가 **아니라 한 번의 SFT run 전체**다("an outer RL loop, which optimizes the self-edit generation, and an inner update loop, which uses the generated self-edit to update the model via gradient descent" [SEAL §3 opening, p.3]). 레짐 어휘가 없으면 이 문장이 통째로 오독된다.
> - **ch21 [LM Need Sleep]** (*Language Models Need Sleep*, arXiv:2606.03979): 한 sleep 라운드의 내부가 SFT와 RL과 distillation의 조합이고, 그 하이퍼파라미터가 표로 실려 있다(LR 5e-6, effective batch size 32, steps 500/100/100 [LM Need Sleep Table 5, p.26]). 이 표를 읽으려면 step·batch·epoch이 무엇의 이름인지 알아야 한다.
> - **ch14 [Letta STC]** (*Sleep-time Compute*, arXiv:2504.13171): 이 논문은 레짐을 **하나도 실행하지 않는다.** 가중치 공간의 상태를 쓰지 않는다고 논문이 명시한다 [Letta STC §7, p.13]. "이 라인의 이름을 만든 논문이 $\Theta$를 건드리지 않는다"는 사실은 §02.10의 판별 습관으로만 즉시 확인된다.
>
> **NM과의 관계** — NM ch02는 training을 **한 step 안쪽**에서 해부했다: backward pass가 계산하는 두 GEMM, $dW=\delta x^\top$라는 shape, optimizer가 `update(state, g) → (state′, ΔΘ)` 서명의 객체라는 것, optimizer state의 바이트 회계. 그 전부를 여기서 반복하지 않는다(→ NM ch02 §2.3, §2.5, §2.8). 이 장이 더하는 것은 **step 바깥의 지도**다: 그 step들이 몇 개씩 묶여 어떤 이름의 레짐을 이루고, 레짐마다 데이터와 목적함수가 어떻게 갈리며, 배포 아티팩트가 어느 경계에서 바뀌는가.

---

## 02.1 레짐이라는 단위 — 하나의 checkpoint가 태어나기까지

독자가 서빙 엔진에 올리는 checkpoint 파일 하나는 여러 공정을 순서대로 통과한 결과물이다. 각 공정은 목적도 데이터도 예산도 다르며, 공정 사이에서 checkpoint가 한 번씩 저장된다. 이 공정 하나를 이 책은 레짐이라고 부른다.

> **정의.** **training run**이란 고정된 데이터 원천과 고정된 목적함수 아래에서 식 (U-$\Theta$)를 정해진 횟수만큼 반복해 하나의 checkpoint를 만들어내는 작업 단위다. **레짐(regime)**이란 그런 run들의 부류로서, (i) 데이터가 어디서 오는가, (ii) 목적함수가 무엇인가, (iii) $\Theta$의 어느 부분집합이 움직이는가 — 이 셋의 조합으로 식별된다.

레짐이 필요한 이유는 단순하다. 식 (U-$\Theta$)는 모든 레짐에서 글자 그대로 같기 때문이다.

$$
\Theta^{(s+1)} \;=\; \Theta^{(s)} \;-\; \eta_\Theta\, \nabla_\Theta\, \mathcal{L}\big(\mathcal{B}_s;\ \Theta^{(s)}\big)
\tag{2-1}
$$

이 식이 말하는 것은 "batch $\mathcal{B}_s$ 위에서 잰 outer loss $\mathcal{L}$의 gradient 반대 방향으로 slow weights를 $\eta_\Theta$만큼 옮긴다"이며, 이는 pre-training이든 SFT든 sleep 라운드든 동일하다. 여기서 $s$는 step 첨자이고 $\eta_\Theta$는 outer/sleep learning rate다(NM의 무첨자 $\eta$가 v2에서 $\eta_\Theta$다. inner rate $\eta_t$와 섞어 쓰지 않는다).

식 (2-1)을 전역 표준형 (U-$\Theta$)와 나란히 놓으면 이 장의 논지가 한눈에 보인다. (U-$\Theta$)는 라운드 첨자 $k$와 학습 집합 $\mathcal{R}_k = \mathrm{gen}(\mathcal{H}_k; B_s)$를 쓰고, (2-1)은 step 첨자 $s$와 batch $\mathcal{B}_s$를 쓴다. 대수는 같고 다른 것은 **첨자의 시간척도**와 **$\mathcal{B}$가 어디서 오는가**뿐이다.

> **[해설]** 그러므로 $\Theta$-경로의 "sleep 라운드"는 새로운 종류의 연산이 아니다. **데이터가 주어지는 대신 모델이 만들어낸다는 점만 다른, 아주 짧은 training run**이다. 이 등식이 이 책 전체의 지렛대다 — ch21과 ch20에서 sleep 라운드의 비용을 계산할 때 우리는 training run의 비용 회계를 그대로 쓰게 된다.

레짐은 순서를 갖는다. 오늘 독자가 서빙하는 instruct 계열 checkpoint 하나는 최소한 아래 사슬을 통과했다. 각 화살표에서 checkpoint가 저장되고, 앞 레짐의 산출물이 뒤 레짐의 초기값이 된다.

$$
\underbrace{\Theta_{\text{rand}}}_{\text{무작위}}
\;\xrightarrow{\ \text{pre-training}\ }\;
\Theta_{\text{base}}
\;\xrightarrow{\ \text{SFT}\ }\;
\Theta_{\text{sft}}
\;\xrightarrow{\ \text{RLHF / RLVR}\ }\;
\Theta_{\text{deploy}}
\;\xrightarrow{\ \text{continual}\ }\;
\Theta_{\text{deploy}}^{(k)}
$$

이 사슬에서 처음 세 화살표는 배포 **전에** 한 번씩 지나가고, 마지막 화살표만 배포 **후에** 반복된다. 이 책이 다루는 $\Theta$-경로 sleep-time compute은 예외 없이 마지막 화살표에 산다 — ch20과 ch21이 손대는 지점이 정확히 여기다. 앞 세 화살표를 이 장이 설명하는 이유는 그 자체가 목적이어서가 아니라, **마지막 화살표가 앞의 것들을 도구로 재사용하기 때문**이다. sleep 라운드 하나를 열어 보면 안에 SFT와 RL이 들어 있다.

systems 관점에서 레짐 경계는 **아티팩트 경계**다. 레짐이 끝날 때마다 checkpoint가 하나 나오고, 그 checkpoint가 다음 레짐의 초기값이 된다. 서빙 fleet가 실제로 mmap하는 파일은 이 사슬의 마지막 산출물이며, Rosetta 사전의 "모델 재배포 = (U-$\Theta$) 1회"라는 대응이 여기서 성립한다 — $\Theta$-경로가 $E$·$W$-경로와 갈라지는 결정적 지점은 **배포 아티팩트가 바뀐다**는 것 하나다.

## 02.2 training run의 구성요소 다섯

임의의 training run은 다섯 개의 슬롯으로 완전히 기술된다. 논문을 읽을 때 이 다섯 칸을 채우지 못하면 그 논문의 학습 절차를 이해하지 못한 것이다.

> **정의.** training run의 다섯 구성요소는 다음과 같다.
> 1. **데이터** — 식 (2-1)의 $\mathcal{B}_s$가 어디서 오는가. 고정 코퍼스, 사람이 쓴 시연, 모델이 생성한 rollout, 이전 라운드의 경험 $\mathcal{H}_k$ 중 하나 이상.
> 2. **목적함수** — 스칼라 $\mathcal{L}$의 정의. next-token cross-entropy, 시연에 대한 조건부 cross-entropy, 보상의 기댓값, teacher 분포와의 divergence 등.
> 3. **optimizer** — `update(state, g) → (state′, ΔΘ)` 서명의 객체(→ NM ch02 §2.5). AdamW가 사실상의 기본값이다.
> 4. **스케줄** — $\eta_\Theta$의 시간 프로파일(warmup·decay), batch 크기, 총 step 수, 정밀도.
> 5. **평가** — 이 run을 멈출지, 되돌릴지, 배포할지를 결정하는 측정 절차.

다섯 칸의 자원 성격이 서로 완전히 다르다는 점이 systems 독자에게 중요하다. **데이터**는 storage와 network를 먹고 GPU를 굶기는 항이다(입력 파이프라인이 GPU를 못 먹이면 MFU가 그대로 떨어진다). **목적함수**는 logits에서 스칼라로의 reduction이므로 비용이 사실상 0이다 — 즉 목적함수를 바꾸는 것은 공짜이고, 그래서 이 분야의 논문 대부분이 목적함수를 바꾼다. **optimizer**는 HBM을 먹는다(param당 state 바이트, → NM ch02 §2.8). **스케줄**은 FLOPs를 결정한다. **평가**는 training이 아니라 inference 워크로드이며, 별도의 fleet 또는 별도의 시간 슬롯을 요구한다.

넷째 슬롯의 첫 값인 $\eta_\Theta$는 논문을 읽을 때 가장 먼저 보아야 할 숫자다. corpus의 두 논문을 나란히 놓으면 그 이유가 보인다. [LM Need Sleep]은 GRPO·SFT·Sleep 세 run을 모두 LR 5e-6으로 돌리고 [LM Need Sleep Table 5, p.26], [SEAL]은 outer 루프의 갱신을 lr 3e-4로 돌린다 [SEAL §B.2, p.20]. 60배 차이다. 이것이 표기 실수가 아니라는 것은 두 논문이 갱신 부분공간의 크기까지 똑같이 잡았다는 데서 드러난다 — 양쪽 다 LoRA rank 64, alpha 128이다 [LM Need Sleep Table 5, p.26; SEAL §B.2, p.20]. **갱신되는 부분공간은 같고 보폭만 60배 다르다.**

> **[해설]** $\eta_\Theta$는 그 run이 무엇을 하려는지에 대한 가장 압축된 진술이다. 작으면 기존 $\Theta$를 보존하면서 미세하게 옮기려는 것이고, 크면 이번 데이터에 빠르게 맞추려는 것이다. 독자의 감각으로 옮기면 cache 정책과 같다 — 작은 $\eta_\Theta$는 보수적 갱신, 큰 $\eta_\Theta$는 공격적 덮어쓰기이며, 후자는 이전에 들어 있던 것을 지울 위험을 함께 산다(→ ch07).

> **[평가]** 그러므로 이 책은 서로 다른 논문의 성능 수치를 "같은 $\Theta$-경로이므로 비교 가능하다"는 전제로 나란히 놓지 않는다. 위 60배는 두 결과가 같은 종류의 쓰기를 재고 있다는 보장이 없음을 뜻한다. 경로가 같아도 스케줄이 다르면 비교의 근거는 따로 세워야 한다.

다섯째 슬롯이 이 책에서 특별히 중요하다. 평가가 training run **안으로 들어오는 순간** run의 구조가 바뀌기 때문이다. 지도학습에서 평가는 run 바깥의 감시자다. 그러나 RL에서 보상 계산은 매 step의 데이터 생성 경로 위에 앉고(§02.6), [SEAL]에서는 "이 self-edit을 적용한 모델이 실제로 과제를 푸는가"라는 평가 결과가 outer gradient의 계수가 된다 [SEAL §3.1, p.4]. 평가가 목적함수의 일부가 되는 것이 최근 레짐들의 공통 형태이며, 그 대가는 training 클러스터 안에 inference 클러스터가 들어온다는 것이다.

> **[해설]** 독자의 감각으로 옮기면 이렇다. 지도학습 run의 시간 구조는 "두꺼운 GEMM(forward/backward) + bandwidth-bound elementwise pass(optimizer step)"의 반복이고 — 즉 prefill 성격이다. RL run과 sleep 라운드의 시간 구조는 거기에 **decode 성격의 생성 단계**가 앞에 붙는다. 같은 (U-$\Theta$)를 돌면서도 클러스터 점유 패턴이 전혀 달라진다.

## 02.3 예산의 어휘 — token budget · step · batch · epoch

레짐의 크기를 말하려면 단위가 필요하다. 네 단어를 여기서 못 박는다.

> **정의.**
> - **step**(= iteration): 식 (2-1)을 한 번 적용하는 것. $\Theta$가 움직이는 최소 사건이다. 개수를 $n_{\mathrm{step}}$으로 쓴다.
> - **batch**: 한 step에서 gradient를 평균하는 표본의 집합 $\mathcal{B}_s$. 시퀀스 개수를 $N_{\mathrm{b}}$로 쓴다.
> - **epoch**: 데이터셋 전체를 한 번 통과하는 것. 개수를 $n_{\mathrm{ep}}$로 쓴다. 정수일 필요는 없다.
> - **token budget** $D$: 그 run이 소비한 총 토큰 수. 레짐의 크기를 재는 이 책의 기본 단위다.

이 장에서 $N$은 모델 parameter 개수, $D$는 token budget을 뜻한다(장-국소 선언). $L$은 시퀀스 길이이며 NM에서 그대로 승계한다 — 층 수를 뜻하는 기호는 $L_{\text{layer}}$이고, 이 책은 맨 $L$을 층 수 의미로 쓰지 않는다. 비용 4종의 $L_w$(wake 지연)와는 첨자로 구분된다. 네 값의 관계는 곱셈 하나다.

$$
D \;=\; n_{\mathrm{step}} \cdot N_{\mathrm{b}} \cdot L,
\qquad
n_{\mathrm{ep}} \;=\; \frac{D}{|\mathcal{D}|}
\tag{2-2}
$$

이 식이 말하는 것은 "총 소비 토큰은 step 수 × step당 토큰이고, epoch은 그 총량을 데이터셋 크기로 나눈 비율일 뿐"이라는 것이다. $|\mathcal{D}|$는 데이터셋의 토큰 수다. epoch이 독립 단위가 아니라 **유도량**이라는 사실이 중요하다 — 데이터셋이 예산보다 크면 $n_{\mathrm{ep}}<1$이고, 그 경우 "epoch"이라는 단어는 아무것도 설명하지 못한다.

$N_{\mathrm{b}}$를 읽을 때 한 가지 함정이 있다. 논문 표에 적히는 값은 대개 한 가속기가 한 번에 forward한 개수가 아니라 **effective batch size** — 즉 "한 번의 (2-1) 적용에 gradient가 평균되어 들어간 시퀀스의 총수"다. 이 값은 세 인자의 곱이다: 가속기당 micro-batch × data-parallel 랭크 수 × gradient accumulation 횟수. 앞의 둘은 병렬화이고 셋째는 순차 반복이다 — gradient를 여러 micro-batch에 걸쳐 누적해 두었다가 한 번에 optimizer step을 밟는 것으로, HBM이 부족할 때 $N_{\mathrm{b}}$를 사는 표준 수단이다.

> **[해설]** 이 구분이 실무적으로 중요한 이유는 **셋 중 어느 조합이든 식 (2-2)와 (2-3)의 값이 같기 때문**이다. effective batch size 32는 32개를 한 번에 밀어 넣든 8개씩 4번 누적하든 같은 $D$, 같은 FLOPs, 같은 $\Theta$ 궤적을 낸다(수치 오차 제외). 바뀌는 것은 벽시계 시간과 HBM 점유뿐이다. 그러므로 [LM Need Sleep Table 5, p.26]처럼 "effective batch size 32"만 적힌 표에서도 이 장의 회계는 그대로 성립한다 — 논문이 하드웨어를 밝히지 않아도 $D$는 확정된다.

FLOPs로의 환산은 NM ch02가 이미 확정한 사실 하나에서 곧바로 나온다. forward는 parameter당 토큰당 곱셈-덧셈 한 쌍, 즉 2 FLOPs이고 backward는 forward의 약 2배다(→ NM ch02 §2.3). 따라서

$$
\mathrm{FLOPs}_{\text{run}} \;\approx\; 6\,N\,D
\qquad\text{(full fine-tuning),}
\qquad
\mathrm{FLOPs}_{\text{fwd only}} \;\approx\; 2\,N\,D
\tag{2-3}
$$

이 식이 이 장의 계산 도구 전부다. $6ND$의 6은 forward 2 + backward 4이고, backward의 4는 다시 "입력 방향 오차 전파 2 + weight gradient 생성 2"로 갈린다(→ NM ch02 §2.3의 두 GEMM). 이 분해가 실무적으로 쓸모 있는 이유는, $\Theta$의 일부만 갱신하는 레짐에서는 **weight gradient 생성 항이 통째로 빠져** $6ND$가 $4ND$로 내려가기 때문이다(그 레짐의 이름과 기제는 ch03이 소유한다). 반대로 $E$-경로와 $W$-경로의 오프라인 처리 $S(c;B_s)$는 생성, 즉 forward만 돌므로 $2ND$ 쪽이다 — 경로 간 비용 비교의 첫 자릿수가 여기서 이미 갈린다.

> **[해설]** 독자가 쓰던 회계와 완전히 같은 도구다. serving에서 decode 한 토큰의 비용을 $2N$으로 잡던 그 습관에, "training은 같은 토큰에 3배를 쓴다"는 인자 하나를 붙이면 끝이다. MFU도 그대로 적용된다: $\mathrm{FLOPs}_{\text{run}}$을 (가속기 peak × 대수 × 시간 × MFU)로 나누면 GPU-hour가 나온다.

## 02.4 pre-training — $\Theta$의 처음이자 가장 큰 이동

> **정의.** **pre-training**이란 무작위 초기화된 $\Theta$를 대규모 미분류 텍스트 코퍼스 위에서 next-token prediction의 cross-entropy를 목적함수로 삼아 갱신하는 레짐이다. $\Theta$ 전체가 움직이며, 한 checkpoint의 생애에서 $D$가 압도적으로 큰 유일한 레짐이다.

데이터는 주어진 것이고, 사람이 라벨을 달지 않았으며, 목적함수는 데이터 자신이다 — 다음 토큰이 곧 정답이다. 이 자기지도 성질이 $D$를 코퍼스 크기까지 밀어 올릴 수 있게 만든 유일한 이유다.

pre-training에서 $n_{\mathrm{ep}}$가 대체로 1 부근이라는 사실은 독자가 반드시 기억할 구조적 특징이다. 식 (2-2)에서 $|\mathcal{D}|$가 $D$와 같은 자릿수면 $n_{\mathrm{ep}}\approx 1$이 되고, 이는 "모든 토큰을 한 번씩만 본다"는 뜻이다. 즉 pre-training의 대부분 구간에서 모델은 **처음 보는 데이터에 대해 online에 가까운 학습**을 한다. 뒤에서 다룰 replay·forgetting의 문제 설정(→ ch07)이 pre-training 안에서는 잘 드러나지 않는 이유가 이것이다 — 같은 데이터를 두 번 보지 않으면 "잊었다"를 관측할 기회 자체가 적다.

다섯째 슬롯인 평가도 이 레짐에서는 단순하다. 목적함수가 곧 지표이므로 held-out 코퍼스의 cross-entropy를 그대로 읽으면 run이 정상인지 판정된다. 이 성질 — **훈련 지표와 평가 지표가 같은 양**이라는 것 — 은 pre-training에만 있는 사치다. 뒤의 모든 레짐에서 $\mathcal{L}$이 내려가는 것과 능력이 오르는 것은 별개의 사건이 되며, §02.5의 SFT 수치가 그 첫 사례다.

**continued pre-training(CPT)**은 pre-training과 목적함수·데이터 형식이 같고 초기값만 기존 checkpoint인 레짐이다. 이 책에서 CPT가 중요한 이유는 $\Theta$-경로 논문들이 자기 갱신을 CPT 형태로 실행하기 때문이다. [SEAL]의 지식 통합 실험에서 $n=200$·$n=2067$ 설정은 문자 그대로 continued pretraining이며 full fine-tuning으로 돌린다 [SEAL Table 2 열 머리글: "Continued Pretraining (n=200; full-FT)"].

systems 접점은 두 가지다. 첫째, pre-training은 이 책에 등장하는 어떤 레짐보다 여러 자릿수 비싸다(이 장의 Worked micro-example에서 수치로 확인한다). 둘째, 그 비용의 대부분이 **한 번만** 지불된다. 이후 모든 레짐은 그 위에 얹히는 작은 편차이며, 이 비대칭이 "sleep 라운드가 상각될 수 있는가"라는 질문을 애초에 성립시킨다 — 큰 $\Theta$는 이미 지불되었고 우리가 논쟁하는 것은 그 위의 작은 증분이다.

## 02.5 SFT — 형식을 고정하는 짧은 레짐

> **정의.** **supervised fine-tuning(SFT)**이란 (입력, 원하는 출력) 쌍의 집합 위에서 **출력 토큰에 대해서만** 조건부 cross-entropy를 최소화하는 레짐이다. 데이터는 사람이 쓴 시연이거나 다른 모델의 출력이며, $D$는 pre-training보다 여러 자릿수 작다.

수식 수준에서 SFT와 pre-training의 차이는 단 하나, **loss mask**다. 시퀀스의 모든 토큰에 loss를 거는 대신 입력(prompt) 구간을 마스킹하고 출력 구간만 $\mathcal{L}$에 넣는다. 목적함수를 바꾸는 비용이 0이라는 §02.2의 관찰이 여기서 실감된다 — 레짐 하나가 마스크 텐서 한 장으로 정의된다.

이 마스크가 얼마나 조작 가능한 대상인지는 [SEAL]이 보여준다. few-shot 영역에서 모델이 생성하는 self-edit에는 텍스트만이 아니라 **inner learning rate, epoch 수, loss mask 자체가 필드로 들어간다** [SEAL Fig. 3, §3.2, p.6]. 즉 §02.2의 다섯 슬롯 중 "스케줄"과 "목적함수"의 일부를 모델이 스스로 써 넣는다. 레짐을 구성요소로 분해해 두면 그 논문이 정확히 어느 칸을 자동화했는지 한 줄로 말할 수 있다.

SFT는 지식을 주입하는 레짐으로 설계되지 않았다. 설계 의도는 이미 $\Theta$ 안에 있는 능력을 원하는 출력 형식으로 꺼내는 것이다. 그럼에도 $\Theta$-경로 논문들은 SFT를 지식 주입 수단으로 쓴다 — [SEAL]의 단일 문단 설정이 그렇고 [LM Need Sleep]의 sleep 라운드 내부가 그렇다. 이 용도 전용이 무해하지 않다는 증거가 corpus 안에 이미 있다. Qwen3-1.7B에서 SFT는 base 대비 AIME-24에서 49.8 → 47.3으로, HMMT-25에서 25.7 → 22.9로 **내려간다** [LM Need Sleep Table 2, p.12].

> **[평가]** SFT를 돌렸다는 사실 자체는 능력이 올랐다는 증거가 아니다. 위 두 수치는 같은 표 안에서 GRPO(51.0 / 26.1)와 Sleep(53.2 / 29.3)이 오르는 동안 SFT만 내려간 사례다 [LM Need Sleep Table 2, p.12]. 이 책은 이후 어느 장에서도 "fine-tuning했으므로 지식이 들어갔다"는 추론을 받아들이지 않는다 — 목적함수가 무엇이었는지, 평가가 무엇을 쟀는지를 함께 보아야 한다.

systems 접점은 크기다. SFT run의 $D$는 서빙 노드 몇 대의 하루치 처리량과 같은 자릿수이며, 이것이 SFT를 "유휴 시간에 돌릴 수 있는 작업"의 후보로 만든다. sleep-time compute의 $\Theta$-경로가 SFT를 기본 도구로 삼는 이유는 성능이 아니라 **예산이 맞기 때문**이다.

## 02.6 RLHF와 RLVR — 데이터를 run 안에서 만드는 레짐

> **정의.** **RLHF(RL from human feedback)**란 사람의 선호 비교로 학습된 별도 모델 $\Theta_{\mathrm{rm}}$이 매기는 점수의 기댓값을 최대화하도록 $\Theta$를 옮기는 레짐이다. **RLVR(RL with verifiable rewards)**이란 그 점수를 학습된 모델이 아니라 **프로그램으로 검증 가능한 정답 여부**가 매기는 레짐이다.

두 레짐의 구조적 특징은 보상이 아니라 데이터다. 지도학습 레짐에서 $\mathcal{B}_s$는 run 바깥에서 주어진다. RL 레짐에서 $\mathcal{B}_s$는 **현재의 $\Theta^{(s)}$가 생성한 rollout**이다. 즉

$$
\mathcal{B}_s \;=\; \mathrm{gen}\big(\text{prompt 집합};\ \Theta^{(s)}\big),
\qquad
\mathcal{L}(\mathcal{B}_s;\Theta) \;=\; -\,\mathbb{E}_{y\sim\mathcal{B}_s}\big[\,R(y)\,\big] \;+\; (\text{정규화 항})
\tag{2-4}
$$

$R(y)$는 생성 $y$에 매겨진 보상이다. RL 문헌은 보상을 $r$로 쓰는 것이 관행이지만 이 책에서 $r$은 LoRA rank로 예약되어 있고(→ ch03), 그래서 보상은 이 책 전체에서 $R$로 옮겨 적는다 — 원문 기호와 대조할 때 이 한 글자만 바꿔 읽으면 된다.

이 식이 말하는 것은 "데이터 슬롯이 parameter의 함수가 되었다"는 것이다. 이것이 식 (U-$\Theta$)에 $\mathcal{R}_k=\mathrm{gen}(\mathcal{H}_k;B_s)$라는 생성 항이 들어 있는 이유이며, [SEAL]이 자기참조적이라고 불리는 이유이기도 하다 — 그 논문은 $\mathrm{gen}(\cdot)$을 $\Theta$ 자신으로 놓는다 [SEAL Abstract; §3.1]. 개별 RL 알고리즘(PPO·GRPO·DPO)의 내부와 reward hacking의 기제는 ch05가 소유한다. 이 장이 확정하는 것은 슬롯 배치뿐이다.

"얼마나 드나"의 답이 여기서 뒤집힌다. [LM Need Sleep]에서 RL 레짐의 step 수는 SFT의 5배다($\text{GRPO } 500$ vs $\text{SFT } 100$ steps [LM Need Sleep Table 5, p.26]). 그러나 진짜 비용은 step 수가 아니라 **step당 생성량**이다. 생성이 붙은 레짐은 step 하나의 값이 다르다 — 같은 논문은 자기 sleep step 하나가 SFT step 하나의 4배라고 보고한다("SFT is 4x more efficient than our method" [LM Need Sleep App. B.5, p.26]). 그 생성분은 식 (2-3)의 $2ND$ 항으로 회계된다.

데이터 슬롯이 $\Theta$의 함수라는 사실에는 따름정리가 하나 붙는다. $\Theta^{(s)}$가 움직이는 순간 이전 step에서 만든 rollout은 **더 이상 현재 정책의 표본이 아니다**. 그래서 RL 레짐은 "생성한 데이터를 몇 step까지 재사용할 것인가"라는, 지도학습 레짐에는 존재하지 않는 결정을 반드시 내려야 한다. 재사용하지 않으면 매 step 생성 비용을 다시 치르고, 재사용하면 데이터와 정책이 어긋난다. 이 긴장을 다루는 알고리즘 장치는 ch05가 소유한다. 이 장에서 확정할 것은 그것이 **다섯 슬롯 중 "데이터"의 성질**이지 optimizer나 목적함수의 문제가 아니라는 점이다. 같은 결정이 sleep 라운드에도 그대로 나타난다 — $\mathcal{R}_k$를 어느 시점의 $\Theta$로 생성했는지가 라운드의 유효성을 정한다.

> **[해설]** systems 접점은 이 장에서 가장 강한 지점이다. **RL run은 한 job 안에서 두 roofline 레짐을 교대시킨다.** rollout 생성 구간은 batch가 얇고 KV cache를 순회하는 bandwidth-bound decode이고, gradient 구간은 두꺼운 GEMM의 compute-bound prefill이다. 지도학습 run은 후자만 있었다. 독자가 서빙에서 익힌 "prefill/decode 혼합 스케줄링" 문제가 training 클러스터 안으로 그대로 들어온 것이다. 그 교대의 값은 이 절이 이미 인용한 수치에 들어 있다 — sleep step 하나가 SFT step 하나의 4배로 보고되고 [LM Need Sleep App. B.5, p.26], 그 초과분이 들어갈 회계 칸이 식 (2-3)의 $2ND$ 항이다.

## 02.7 continual — 배포 이후에도 움직이는 $\Theta$

> **정의.** **continual training**이란 배포 이후에 도착하는 경험에 대해 (U-$\Theta$)를 반복해 도는 레짐이다. 데이터 원천이 시간에 대해 열려 있고, $k$번째 라운드의 학습 집합 $\mathcal{R}_k$가 그때까지 누적된 경험 $\mathcal{H}_k$에 의존한다.

앞의 세 레짐과 다른 점이 세 가지다.

첫째, **첨자가 하나 늘어난다.** step 첨자 $s$ 위에 라운드 첨자 $k$가 얹히고, 식 (U-$\Theta$)의 $k$가 바로 이것이다. 한 라운드 안쪽은 여전히 §02.5·§02.6의 레짐 중 하나이며 — [LM Need Sleep]의 sleep 라운드는 distillation과 RL의 조합이다 [LM Need Sleep §3.3, p.8] — 새로운 것은 라운드들이 **서로의 산출물 위에 쌓인다**는 사실뿐이다.

둘째, **데이터가 i.i.d.가 아니다.** 라운드마다 분포가 옮겨 가므로 이전 라운드에서 얻은 것이 다음 라운드에서 지워질 수 있다. 이 현상의 이름과 대응책은 ch07이 소유한다. 여기서 확정할 것은 그것이 **레짐의 성질**이지 알고리즘의 결함이 아니라는 점이다 — pre-training은 $n_{\mathrm{ep}}\approx1$로 한 번 흐르고 끝나므로 같은 문제를 겪지 않는다.

셋째, **배포 아티팩트가 갈라진다.** 라운드가 사용자별·세션별로 돌면 checkpoint가 하나가 아니라 사용자 수만큼 생긴다. corpus의 $\Theta$-경로 논문들이 이 문제에 쓰는 대표적 대응은 갱신 대상을 잘라내는 것이다. [LM Need Sleep]은 backbone 전체를 얼리고 새로 확장한 parameter만 갱신한다("we freeze all the parameters in the student model and only updates the expanded parameters" [LM Need Sleep §3.3, p.8]). 그 기제의 평가는 ch21이 한다.

넷째, **평가 슬롯이 열려 있다.** 앞의 세 레짐에서 평가 집합은 run 시작 전에 고정되고 훈련 데이터와 분리된다 — [SEAL]이 RL 학습 집합과 평가 문단 사이에 겹침이 없음을 명시하는 것이 그 표준 관행이다("there is no overlap between these sets, so we can be sure that there is no data contamination of the test passages due to RL training" [SEAL §B.1, p.20]). continual 레짐에서는 그 분리가 원리적으로 어렵다. 라운드 $k$의 데이터는 배포 중에 도착한 것이고, 무엇이 이미 평가 집합에 들어 있는지 run 시점에 알 수 없다. 더 근본적으로, 라운드가 쌓이면 **무엇에 대해 퇴행했는지를 물을 대상 자체가 커진다** — 지난달에 넣은 사실이 이번 달 라운드에서 지워졌는지 확인하려면 지난달의 평가를 계속 보관하고 다시 돌려야 한다.

> **[평가]** 이 누적 평가 비용을 비용 4종의 어느 칸에도 계상한 논문을 이 corpus에서 찾지 못했다. $\Theta$-경로 논문들은 라운드당 학습 하이퍼파라미터는 표로 싣지만, "지난 라운드들을 계속 재평가하는 비용"은 $B_s$에도 $\rho$에도 들어가 있지 않다. ch25의 통일 회계는 이 칸을 비워 둔 채 시작한다.

systems 접점은 Rosetta 사전의 두 행이 동시에 걸린다는 것이다. "모델 재배포 = (U-$\Theta$) 1회"와 "checkpoint 크기 = $C_{\text{cap}}$"이 continual 레짐에서 합쳐지면, 라운드마다 사용자당 새로운 상태 바이트가 생기고 shared-weight batching이 깨진다. $E$-경로는 이 대가를 치르지 않는다 — 상태가 텍스트로 남아 prefill로만 들어가기 때문이다. **경로 사이의 가장 큰 비용 차이는 정확도가 아니라 이 배치 가능성에서 나온다**는 것이 Part III의 논점 중 하나다(→ ch26, ch29).

## 02.8 네 칸 표 — 레짐 비교

이 장의 핵심 산출물이다. 어떤 논문을 읽든 그 논문의 학습 절차를 이 표의 한 행으로 적을 수 있어야 한다.

표 2-1 — 레짐별 네 칸

| 레짐 | 무엇이 데이터인가 | 무엇이 목적함수인가 | 무엇이 갱신되는가 | 얼마나 드나 |
|---|---|---|---|---|
| pre-training | 주어진 대규모 미분류 코퍼스. run 바깥에서 고정 | 전 토큰 next-token cross-entropy | $\Theta$ 전체 | $D$가 코퍼스 크기와 같은 자릿수. $n_{\mathrm{ep}}\approx1$. $6ND$ |
| continued pre-training | 같은 형식의 추가 코퍼스(도메인·신규 문서) | 동일 | $\Theta$ 전체 또는 부분집합 | $D \ll$ pre-training. [SEAL]은 $n{=}200$·$n{=}2067$ 문단에서 full-FT [SEAL Table 2] |
| SFT | (입력, 원하는 출력) 쌍. 사람 시연 또는 모델 출력 | 출력 구간만의 조건부 cross-entropy (loss mask) | $\Theta$ 전체 또는 부분집합 | steps 100, effective batch 32, LR 5e-6 [LM Need Sleep Table 5, p.26]; 또는 2 epochs, batch 10, lr 3e-4 [SEAL §B.2, p.20] |
| RLHF | 현재 $\Theta$가 만든 rollout. **run 안에서 생성** | 학습된 $\Theta_{\mathrm{rm}}$의 점수 기댓값 − 정규화 | $\Theta$(policy). 별도로 $\Theta_{\mathrm{rm}}$이 선행 학습됨 | step당 생성 비용이 지배. 생성분은 $2ND$ |
| RLVR | 현재 $\Theta$가 만든 rollout | 프로그램 검증 결과의 기댓값 | $\Theta$(policy)만 | steps 500 [LM Need Sleep Table 5, p.26] |
| continual / sleep 라운드 | $\mathcal{R}_k=\mathrm{gen}(\mathcal{H}_k;B_s)$ — 누적 경험에서 **모델이 만든다** | 라운드마다 다름(SFT loss, divergence, 보상 혼합) | $\Theta$의 지정된 부분집합. 나머지는 동결 | 라운드당 $B_s$. corpus에서 이를 FLOPs로 보고한 논문은 없다 |

마지막 행의 마지막 칸이 이 책이 반복해서 부딪히는 공백이다. 레짐의 이름과 하이퍼파라미터는 표로 실리지만, 그것이 몇 FLOPs인지, 라운드가 몇 번 도는지는 대부분의 논문이 적지 않는다. ch25의 통일 비용 회계가 이 칸을 채우는 작업이다.

## 02.9 Part I의 나머지 장이 채우는 칸

표 2-1은 지도이지 설명이 아니다. 각 칸을 깊이 파는 일은 Part I의 나머지 장들이 나누어 맡으며, 그 분업이 §02.2의 다섯 슬롯을 따른다. 이 대응을 여기서 못 박아 두면 이후 장들을 "또 다른 배경 지식"이 아니라 **표 2-1의 특정 칸에 대한 확대**로 읽을 수 있다.

- **"무엇이 갱신되는가" 칸** — $\Theta$의 어느 부분집합을 어떻게 잘라내는가. adapter와 LoRA가 그 도구이며 **ch03이 소유한다**. 이 장은 "부분 갱신이면 식 (2-3)의 weight gradient 항이 빠진다"는 회계 사실까지만 확정했다.
- **"무엇이 목적함수인가" 칸(교사가 있는 경우)** — 목표가 정답 라벨이 아니라 다른 모델의 출력 분포일 때. distillation이며 **ch04가 소유한다**. sleep 라운드의 목적함수가 대개 이 형태다.
- **"무엇이 목적함수인가" 칸(보상이 있는 경우)** — 식 (2-4)의 $R(\cdot)$과 그 최적화. PPO·GRPO·DPO의 내부와 reward hacking은 **ch05가 소유한다**.
- **"무엇이 데이터인가" 칸(모델이 만든 경우)** — $\mathrm{gen}(\cdot;B_s)$의 산출물을 다시 학습에 넣을 때 생기는 붕괴 위험. **ch06이 소유한다.** 판별 습관의 Q4가 이 장으로 연결된다.
- **"얼마나 드나" 칸의 시간 축** — 라운드를 반복할 때의 열화 $\rho$와 그 대응. **ch07이 소유한다.**
- **"얼마나 드나" 칸의 용량 축** — $\Theta$에 얼마나 쓸 수 있는가, 즉 $C_{\text{cap}}$의 상한. **ch09가 소유한다.**

즉 Part I의 ch03–ch09는 표 2-1의 칸 하나씩을 확대한 것이고, ch10만이 다른 층($E$)으로 넘어간다. 이 장은 그 지도의 좌표계를 놓은 것이다.

## 02.10 "누가 이 값을 학습하는가" — 판별 습관

레짐 지도의 목적은 분류 자체가 아니라 **판정 습관**이다. 논문에 등장하는 임의의 값에 대해 네 질문을 순서대로 던진다.

> **정의.** **"누가 이 값을 학습하는가" 판별**은 다음 네 질문에 답하는 절차다.
> **Q1. 그 값은 배포 아티팩트에 남는가?** 남으면 $\Theta$ 또는 $\Theta$의 증분이다. 세션과 함께 사라지면 $W$이거나 $E$다.
> **Q2. 그 값을 움직이는 gradient는 어느 loss의 것인가?** inner loss $\ell$이면 (U-W), outer loss $\mathcal{L}$이면 (U-$\Theta$), gradient가 아예 없으면 (U-E)다.
> **Q3. 어느 시계에서 움직이는가?** 토큰 $t$ / step $s$ / 라운드 $k$ / 세션 $\tau$.
> **Q4. 그 값을 만든 데이터는 주어진 것인가, 모델이 만든 것인가?** 후자면 ch06의 붕괴 위험이 이 논문에 걸린다.

Q1–Q3만으로도 이 책의 세 경로가 갈린다. Q4는 그 위에 얹히는 정직성 점검이다. 네 질문을 [SEAL]의 세 객체에 적용하면 다음과 같다 — 논문 한 편 안에 서로 다른 답이 세 개 들어 있다는 것이 이 습관의 필요성을 그대로 보여준다.

표 2-2 — 판별 습관의 적용 예 ([SEAL]의 세 객체; 기제 해설은 ch20)

| 값 | Q1 배포에 남는가 | Q2 어느 loss | Q3 시계 | Q4 데이터 출처 | 배치 |
|---|---|---|---|---|---|
| backbone weights (Qwen2.5-7B / Llama-3.2-1B) | 남는다 | 단일 문단 설정에서는 움직이지 않음(갱신이 LoRA 부분공간에 갇힘). CPT 설정에서는 full-FT로 전체가 움직임 | — / 라운드 $k$ | 논문 밖에서 상속 | 상속된 $\Theta$ [SEAL §4.1, p.6; §B.1, p.20], CPT 열은 full-FT [SEAL Table 2, p.8] |
| self-edit 생성 정책 | 남는다(출하 모델) | outer $\mathcal{L}$ (보상 필터 후 behavior cloning) | 라운드 $k$ | 모델 생성 | (U-$\Theta$) [SEAL §3.1, pp.3–4] |
| self-edit이 만든 가중치 증분 | 배포 시 남는다 | outer $\mathcal{L}$ (생성 텍스트에 대한 causal-LM loss) | 문맥마다 1회, 질의 전 | 모델 생성 | (U-$\Theta$) [SEAL §3.1, p.4] |

같은 절차를 [Letta STC]에 적용하면 세 행이 전부 비고, (U-E)만 남는다 — 그 논문은 가중치 공간의 상태를 쓰지 않는다고 스스로 밝힌다 [Letta STC §7, p.13]. "sleep-time compute"이라는 이름을 만든 논문과 "sleep"을 제목에 단 $\Theta$-경로 논문이 **같은 이름 아래 완전히 다른 층을 건드린다**는 사실이, 이 네 질문만으로 즉시 드러난다.

> **[해설]** 이 습관은 ch11의 판별식과 한 몸이다. ch11은 "질의 도착 **전에**, 예산 $B_s>0$을 써서, 세 층 중 하나를 실제로 바꾸고, 그 변화가 이후 질의에 쓰이는가"를 묻는다. 그 네 조건 중 셋째("어느 층이 실제로 바뀌는가")를 판정하는 도구가 바로 이 절의 Q1–Q3다. 레짐 어휘를 먼저 배우는 이유가 여기 있다 — 층을 판정하려면 그 층을 움직이는 절차의 이름을 알아야 한다.

## (state, update, cost) 정리

이 장에서 도입한 개념을 세 층 프레임 위에 배치한다.

표 2-3 — 이 장의 개념과 세 층 프레임

| 이 장의 개념 | state (어느 층) | update (어느 식) | cost (비용 4종의 어느 칸) |
|---|---|---|---|
| pre-training / CPT | $\Theta$ 전체 | (U-$\Theta$), $\mathcal{B}_s$ 주어짐 | 배포 전 1회 지불. 비용 4종에 들어가지 않음(상각 논의의 바깥) |
| SFT | $\Theta$ 전체 또는 부분집합 | (U-$\Theta$), loss mask 적용 | sleep 라운드 안에서 실행되면 $B_s$의 항 |
| RLHF / RLVR | $\Theta$(policy), $\Theta_{\mathrm{rm}}$(별도) | (U-$\Theta$), $\mathcal{B}_s=\mathrm{gen}(\cdot;\Theta^{(s)})$ | $B_s$ — 생성분 $2ND$ + gradient분 $6ND$ |
| continual 라운드 | $\Theta$의 지정 부분집합 | (U-$\Theta$), $\mathcal{R}_k=\mathrm{gen}(\mathcal{H}_k;B_s)$ | $B_s$(라운드당), $\rho$(라운드 누적 열화), $C_{\text{cap}}$(사용자당 증분 바이트) |
| token budget $D$ | — (회계 단위) | 식 (2-2) | $B_s$·$B_t$를 tokens 단위로 적을 때의 공통 척도 |
| 판별 습관 Q1–Q3 | 세 층 전부 | 세 갱신식 전부 | — (판정 도구) |

$W$-경로와 $E$-경로는 이 표에 행이 없다. 그 두 경로는 (U-$\Theta$)를 돌지 않기 때문이며, 그것이 정확히 이 장이 $\Theta$-경로 독해의 전제인 이유다. $W$의 갱신은 (U-W)로 NM이 이미 다뤘고(→ NM ch12), $E$의 갱신은 gradient 없는 쓰기 연산 $\mathrm{wr}(\cdot)$이다(→ ch10).

## Worked micro-example — 한 sleep 라운드의 값을 손으로 매긴다

이 절은 §02.3의 세 식만으로 실제 논문의 설정 하나를 끝까지 계산한다. 목표는 두 개다: 레짐의 크기를 자릿수로 감각하는 것, 그리고 독자의 1번 질문("그래서 decode에서 뭘 바꾸는데?")에 숫자로 답하는 것.

**설정.** 모델은 Qwen3-1.7B, 즉 $N = 1.7\times10^9$ [LM Need Sleep Table 2, p.12의 backbone]. sleep 라운드 안의 SFT 설정은 원문 표 그대로 **steps 100, effective batch size 32, LR 5e-6** [LM Need Sleep Table 5, p.26]. 원문이 적지 않은 세 값은 가정으로 명시하고, 이 셋은 논문 수치가 아니다: 시퀀스 길이 $L=1024$(가정 A), 한 질의의 wake 생성량 500 토큰(가정 B), 비교용 pre-training 예산 $D_{\text{pre}}=10^{12}$ 토큰(가정 C). 가정값은 결과의 자릿수를 바꾸지 않으며, 바뀌는 지점은 (e)에서 따로 짚는다. (a)–(d)에서 나오는 산출값은 전부 — $D$, 라운드 FLOPs, 표 2-4의 상각 열, 손익분기 $N_q$ — 논문이 보고한 수치가 아니라 위 하이퍼파라미터와 가정 A–C에서 **이 책이 유도한 예시 계산**이다. 논문에서 온 수치는 $N$과 위 세 하이퍼파라미터뿐이다.

**(a) token budget.** 식 (2-2)에 그대로 넣는다.

$$
D \;=\; n_{\mathrm{step}} \cdot N_{\mathrm{b}} \cdot L \;=\; 100 \times 32 \times 1024 \;=\; 3{,}276{,}800 \;\approx\; 3.28\times10^{6}\ \text{tokens}
$$

한 라운드가 소비하는 토큰이 330만 개다. 독자의 감각으로는 중형 서빙 노드 한 대의 몇 시간치 처리량이다.

**(b) 라운드 FLOPs.** 식 (2-3)의 $6ND$다.

$$
6ND \;=\; 6 \times 1.7\times10^{9} \times 3.2768\times10^{6} \;=\; 3.34\times10^{16}\ \text{FLOPs}
$$

$\Theta$의 일부만 갱신해 weight gradient 항이 빠지면 $4ND = 2.23\times10^{16}$이다 — 정확히 $2/3$이고, **자릿수는 그대로다.** 부분 갱신의 실제 이득이 FLOPs가 아니라 optimizer state 바이트에 있다는 ch03의 논점이 이 한 줄에서 미리 보인다.

**(c) pre-training 대비.** 같은 $N$에 가정 C의 $D_{\text{pre}} = 10^{12}$ 토큰을 넣으면 $6ND_{\text{pre}} = 1.02\times10^{22}$ FLOPs다. 비를 잡으면

$$
\frac{6ND_{\text{pre}}}{6ND} \;=\; \frac{1.02\times10^{22}}{3.34\times10^{16}} \;\approx\; 3.1\times10^{5}
$$

즉 **가정 C 아래에서 한 sleep 라운드는 pre-training의 약 $3\times10^{-6}$이다.** 이 비대칭이 $\Theta$-경로가 논의 대상이 되는 이유 전부다. 라운드를 하루 한 번씩 3년을 돌려도 pre-training 한 번의 0.4%에 못 미친다($1000 \times 3\times10^{-6} = 3\times10^{-3}$).

**(d) 상각 — 식 (A)로 손익분기 $N_q$ 찾기.** 이제 질문을 서빙 쪽으로 돌린다. 가정 B에서 한 질의의 wake 비용은 생성 토큰당 $2N$이므로

$$
C_{\text{wake}} \;=\; 2N \times 500 \;=\; 1.7\times10^{12}\ \text{FLOPs/질의}
$$

이 값은 prefill을 세지 않은 **하한**이다. 식 (A) $C_{\text{avg}} = C_{\text{wake}} + C_{\text{sleep}}/N_q$에 $C_{\text{sleep}} = 3.34\times10^{16}$을 넣고 $N_q$를 훑는다.

표 2-4 — sleep 라운드 비용의 질의당 상각 ($C_{\text{sleep}}=3.34\times10^{16}$, $C_{\text{wake}}=1.7\times10^{12}$ FLOPs). 이 책의 예시 계산이며 논문이 보고한 값이 아니다

| $N_q$ (문맥을 공유하는 질의 수) | $C_{\text{sleep}}/N_q$ [FLOPs/질의] | $C_{\text{wake}}$ 대비 |
|---|---|---|
| 1 | $3.3\times10^{16}$ | $1.97\times10^{4}\times$ |
| $10^{3}$ | $3.3\times10^{13}$ | $19.7\times$ |
| $2.0\times10^{4}$ | $1.7\times10^{12}$ | $1.00\times$ (손익분기) |
| $2.0\times10^{5}$ | $1.7\times10^{11}$ | $0.10\times$ |

손익분기는 $N_q \approx 2\times10^{4}$, sleep 오버헤드를 10% 이하로 누르려면 $N_q \approx 2\times10^{5}$가 필요하다. 하나의 문맥에 대해 질의가 **2만 번에서 20만 번** 들어와야 이 라운드가 회수된다.

**(e) 읽기.** 세 가지가 이 계산에서 나온다.

첫째, 가정의 민감도는 선형이고 자릿수를 넘지 않는다. $L$을 4096으로 올리면 (a)와 (b)가 4배가 되고 손익분기도 4배가 된다. 생성량을 2000 토큰으로 올리면 $C_{\text{wake}}$가 4배가 되어 손익분기가 1/4로 내려간다. 어느 쪽도 $N_q$의 자릿수를 $10^4$–$10^5$에서 빼내지 못한다.

둘째, 이 계산에서 **decode 쪽 FLOPs 변화는 정확히 0이다.** sleep 라운드는 wake의 토큰 수도, 토큰당 비용도 줄이지 않는다. $\Theta$-경로가 파는 것은 decode 비용의 절감이 아니라 **같은 decode 예산에서의 정확도**다. wake 토큰 수를 실제로 줄이는 것은 $E$-경로의 주장이며, 그 주장의 검증은 ch14가 한다.

셋째, 표 2-1의 마지막 칸이 방금 값을 얻었다. 그 값을 논문이 적어 준 것이 아니라, 논문이 적은 하이퍼파라미터 표에서 이 장의 세 식으로 **복원**한 것이다. ch25의 통일 비용 회계는 이 복원을 corpus 전체에 대해 기계적으로 수행하는 작업이며, 그때 쓰는 도구가 정확히 식 (2-2)·(2-3)과 식 (A)다.

> **[평가]** 어디까지 복원할 수 있는가는 이 책의 판단이다. steps·batch·모델 크기가 표에 있으면 $B_s$는 논문이 침묵해도 되살아난다 — 위 (a)–(b)가 그 절차 전부다. 되살아나지 않는 것은 서빙 쪽 변수인 $N_q$이며, 그것은 논문이 게을러서가 아니라 논문이 가질 수 없는 값이기 때문이다. 표 2-4의 오른쪽 열이 유효하려면 그 $N_q$를 알아야 한다. 그런데 이 corpus에서 실서빙의 $N_q$ 분포를 보고한 논문은 없다(→ ch25). 그러므로 이 절이 정직하게 산출한 것은 "sleep 라운드가 이득이다/아니다"가 아니라 **"이 설정에서 이득이 나려면 문맥당 질의가 최소 $2\times10^{4}$번 와야 한다"는 조건문**이다. Part III가 답해야 할 질문의 형태가 이것이다.

## 요약

- 모든 레짐은 같은 식 (U-$\Theta$)를 돈다. 레짐을 가르는 것은 대수가 아니라 데이터 슬롯 $\mathcal{B}_s$의 출처, 목적함수의 정의, 갱신되는 $\Theta$의 부분집합 셋뿐이다.
- 임의의 training run은 데이터·목적함수·optimizer·스케줄·평가의 다섯 슬롯으로 완전히 기술되며, 다섯 슬롯이 먹는 자원은 각각 storage/0/HBM/FLOPs/inference로 서로 다르다.
- epoch은 독립 단위가 아니라 $n_{\mathrm{ep}} = D/|\mathcal{D}|$라는 유도량이고, 레짐의 크기를 재는 단위는 token budget $D$다. full fine-tuning의 비용은 $6ND$, forward만 도는 생성은 $2ND$, weight gradient 항이 빠진 부분 갱신은 $4ND$이며 — 자릿수를 바꾸는 것은 갱신 방식이 아니라 $D$다.
- pre-training은 $n_{\mathrm{ep}}\approx1$로 한 번 흐르고, 이후 모든 레짐은 그 위의 작은 증분이다. 이 비대칭이 sleep 라운드의 상각 논의를 성립시킨다.
- SFT는 loss mask 한 장으로 정의되는 레짐이고, 돌렸다는 사실이 능력 향상의 증거가 아니다 — Qwen3-1.7B에서 SFT는 AIME-24를 49.8에서 47.3으로 낮춘다 [LM Need Sleep Table 2, p.12].
- RL 레짐의 정의적 특징은 보상이 아니라 데이터가 $\Theta$의 함수라는 것이며, 그 결과 한 job 안에서 decode 성격과 prefill 성격의 두 roofline 레짐이 교대한다.
- continual 레짐만이 라운드 첨자 $k$를 갖고, 배포 아티팩트가 사용자별로 갈라져 shared-weight batching을 깨뜨릴 수 있는 유일한 레짐이다.
- "누가 이 값을 학습하는가"는 네 질문(배포에 남는가 / 어느 loss인가 / 어느 시계인가 / 데이터는 누가 만들었나)으로 판정되며, 이 판정이 ch11 판별식의 셋째 조건을 집행하는 도구다.

## 자가 점검 체크리스트

- [ ] pre-training · CPT · SFT · RLHF · RLVR · continual 각각에 대해 표 2-1의 네 칸을 보지 않고 채울 수 있다.
- [ ] 임의의 논문 부록에서 "steps 100, batch 32, LR 5e-6"을 보았을 때 식 (2-2)와 (2-3)으로 $D$와 FLOPs를 즉석에서 산정할 수 있다.
- [ ] $6ND$의 6이 어떻게 2+2+2로 갈리는지 말하고, 어떤 조건에서 $4ND$·$2ND$로 내려가는지 설명할 수 있다.
- [ ] Worked micro-example의 (a)–(d)를 재현해, 다른 $L$과 다른 생성량에서 손익분기 $N_q$가 어떻게 움직이는지 계산할 수 있다.
- [ ] 임의의 논문에서 학습되는 값 하나를 골라 Q1–Q4에 답하고, 그 값을 (U-W)/(U-$\Theta$)/(U-E) 중 하나에 배치할 수 있다.
- [ ] $\eta_\Theta$와 $\eta_t$, $C_{\text{cap}}$과 chunk 크기 $C$를 혼동하지 않고, 이 장의 어느 수식에도 후자가 등장하지 않는 이유를 말할 수 있다.
- [ ] 이 장의 레짐 어휘를 inference 어휘로 옮길 수 있다: 어느 레짐이 "모델 재배포"에 해당하고, 어느 레짐이 "warm-up/사전 컴파일"에 해당하며, 어느 레짐만이 checkpoint 크기($C_{\text{cap}}$) 회계를 사용자 수만큼 곱하게 만드는지 말할 수 있다.

## 다음 장으로

이 장은 표 2-1의 셋째 칸 — "무엇이 갱신되는가" — 을 "$\Theta$ 전체 또는 부분집합"이라고만 적고 넘어갔다. 그러나 Worked micro-example (b)가 보였듯 그 선택은 FLOPs를 $2/3$로 줄일 뿐이고, 진짜 차이는 다른 곳에 있다: optimizer state 바이트, checkpoint 크기, 그리고 사용자당 저장해야 하는 증분의 크기다. continual 레짐에서 배포 아티팩트가 갈라진다는 §02.7의 문제는 바로 이 증분의 크기가 정한다. ch03은 그 부분집합을 자르는 방법 — full fine-tuning, adapter, LoRA와 그 변형 — 을 (state, update, cost) 객체로 세우고, "가중치에 쓰기 위해 얼마를 저장해야 하는가"를 바이트로 답한다. 그 답이 나와야 ch19의 쓰기 primitive와 ch26의 상태 용량 회계를 읽을 수 있다.
