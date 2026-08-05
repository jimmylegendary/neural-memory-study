# ch04. distillation — 한 모델의 앎을 다른 모델로 옮기기

> **이 장의 목표** — 이 장을 마치면 독자는 다음을 할 수 있어야 한다.
> 1. teacher와 student를 파라미터의 이동 여부로 정의하고, distillation loss를 (대상, divergence, 표본 출처) 세 좌표로 분해해 쓸 수 있다.
> 2. temperature $T$가 soft target과 gradient에 각각 무엇을 하는지 계산으로 보이고, $T^2$ 보정이 왜 붙는지 진술할 수 있다.
> 3. logit distillation과 feature distillation을 저장·대역폭 비용으로 구분하고, self-distillation에서 on-policy 표본이 sleep job의 비용 형태를 training-bound에서 decode-bound로 바꾼다는 것을 설명할 수 있다.
> 4. **upward distillation**(작은 자기 → 큰 자기)이 통상의 방향과 반대라는 사실을 말하고, 그 절차에서 실제로 옮겨지는 것이 지식이 아니라 좌표라는 것을 [LM Need Sleep]의 식으로 논증할 수 있다.
>
> **왜 필요한가** — Part II의 세 지점이 이 장을 전제하고 쓰인다.
> - **ch21 [LM Need Sleep]** (*Language Models Need Sleep*, arXiv:2606.03979) §3.3: Knowledge Seeding의 목적함수는 GKD 항과 imitation reward 항의 볼록결합이다. 그 GKD 항의 형태·표본 출처·frozen 범위를 이 장이 소유한다. 같은 절의 boxed 문단 제목이 그대로 "Upward Distillation (Knowledge Seeding)"이다.
> - **ch20 [SEAL]** (*Self-Adapting LMs*, arXiv:2506.10943) §3.1 마지막 문단: teacher와 student를 서로 다른 모델로 분리하는 확장을 제안한다 — 제안만 하고 실행하지 않는다. 그 제안이 무엇을 바꾸는 제안인지 읽으려면 teacher/student가 역할이지 개체가 아니라는 이 장의 정의가 필요하다.
> - ch21이 최강 baseline으로 세운 OPSD 계열, 그리고 [LM Need Sleep] App. A.4가 두 페이지에 걸쳐 열거하는 2026년 논문군 전체가 **on-policy self-distillation**이다. 그 이름이 무엇을 뜻하는지가 이 장 §04.4다.
>
> **NM과의 관계** — Neural Memory 모노그래프에 대응 장이 없다. 이 장은 NM이 세운 기계 위에서 시작한다: backward pass와 optimizer state는 → NM ch02, inner/outer loop의 이중 구조는 → NM ch04, chunk 단위 갱신은 → NM ch09로 넘긴다. 이 장이 더하는 것은 하나다 — **gradient의 출처를 label에서 다른 모델의 출력 분포로 바꾸면 무엇이 달라지는가.** NM은 $W$가 무엇을 배우는지를 다뤘고, 이 장은 $\Theta$에 무엇을 어떻게 써넣을 수 있는지의 두 번째 도구를 다룬다(첫 번째 도구는 ch03의 PEFT다).

---

## 04.1 gradient의 출처를 바꾼다 — teacher와 student

**distillation은 loss의 정답 자리에 사람이 붙인 label 대신 다른 모델의 출력 분포를 놓는 절차다.** 이 치환 하나가 이 장 전체를 결정한다.

> **정의.** **teacher**는 학습 중 파라미터가 움직이지 않고 그 출력이 학습 신호로 소비되는 모델이다. **student**는 그 신호를 재현하도록 파라미터가 움직이는 모델이다. 두 이름은 **역할**이지 개체가 아니다 — 크기·구조·정체성 어느 것도 정의에 들어가지 않는다. 이 장의 표기로 teacher 파라미터는 $\Theta^{\mathrm{te}}$, student 파라미터는 $\Theta^{\mathrm{st}}$이며, 갱신 대상은 항상 $\Theta^{\mathrm{st}}$ 하나다.

이 정의가 역할로만 되어 있다는 사실이 뒤의 세 절을 가능하게 한다. teacher와 student가 같은 개체일 수 있고(§04.4), teacher가 student보다 작을 수도 있다(§04.5). 정의에 크기가 들어 있었다면 둘 다 형용모순이 된다.

왜 label 대신 분포인가. 정답 label은 어휘 $\mathcal{V}$ 위의 one-hot 벡터로, 정답 토큰 하나에 대해서만 정보를 준다. teacher의 출력 분포는 같은 자리에서 $|\mathcal{V}|$개 좌표 전부에 값을 준다 — "정답은 A이고, B는 A의 1/8만큼 그럴듯하며, C는 사실상 불가능하다"는 순위와 간격까지 함께 온다. Hinton, Vinyals & Dean(2015, arXiv:1503.02531)이 dark knowledge라고 부른 것이 이 여분의 좌표들이다. 같은 토큰 하나를 소비하고도 student가 받는 신호의 차원이 1에서 $|\mathcal{V}|$로 늘어난다는 것, 그것이 distillation의 자본이다.

> **[해설]** systems 관점에서 distillation 파이프라인은 새로운 커널을 하나도 도입하지 않는다. teacher 쪽은 순수 forward — 독자가 매일 돌리는 batch prefill과 완전히 같은 연산이고, weight는 read-only이며, roofline 위치도 그대로다. student 쪽만 backward가 붙는다(→ NM ch02 §2.3). 새로 생기는 것은 연산이 아니라 **두 모델이 동시에 HBM에 상주한다**는 배치 제약과, teacher 출력을 student backward까지 살려두는 텐서 수명이다.

독자에게 이미 익숙한 사례가 하나 있다. speculative decoding의 draft model은 target model의 분포를 근사하도록 학습된 student이고, 그 학습 절차의 표준형이 바로 distillation이다. acceptance rate는 두 분포가 얼마나 가까운지의 직접 측정치다. 즉 독자는 distillation의 산출물을 이미 serving하고 있다 — 이 장은 그 산출물이 어떻게 만들어지는지를 연다.

이 장이 이 책에서 차지하는 자리도 여기서 정해진다. $\Theta$-경로가 sleep 라운드에 (U-$\Theta$)를 돌리려면 두 가지가 필요하다: **어디에 쓸 것인가**(파라미터의 어느 부분공간을 움직일 것인가 — ch03의 PEFT)와 **무엇을 목표로 쓸 것인가**(gradient를 만들어 낼 신호 — 이 장). 이 corpus에서 sleep 라운드의 목표는 거의 예외 없이 사람이 붙인 label이 아니다. 모델 자신이거나, 자신의 과거이거나, 다른 모델의 분포다. 그래서 distillation은 이 경로에서 선택지가 아니라 **기본 문법**이다.

---

## 04.2 logit distillation과 temperature

> **정의.** **logit distillation**은 teacher와 student의 **출력 분포**(softmax 이전의 logit을 온도로 나눠 만든 분포)를 맞추는 distillation이다. 중간 표현은 보지 않고 최종 층의 분포만 본다.

한 토큰 위치에서 teacher logit을 $z^{\mathrm{te}}\in\mathbb{R}^{|\mathcal{V}|}$, student logit을 $z^{\mathrm{st}}\in\mathbb{R}^{|\mathcal{V}|}$라 하자. **temperature** $T>0$는 softmax의 지수를 나누는 스칼라다(이 장에서만 쓰는 장-국소 기호다. sequence 길이는 NM 표기 그대로 $L$이고, $T$를 길이 의미로 쓰지 않는다).

$$
p^{(T)}_i \;=\; \frac{\exp(z_i/T)}{\sum_{j} \exp(z_j/T)}
\tag{4-1}
$$

이 식이 말하는 것은 하나다. $T$를 키우면 분포가 평평해지고, 낮은 확률의 좌표들이 상대적으로 커진다. $T\to 0$이면 argmax의 one-hot으로 붕괴하고, $T\to\infty$면 균등분포로 간다. dark knowledge는 정확히 그 "낮은 확률의 좌표들"에 있으므로, $T$는 **teacher의 어느 부분을 학습 신호로 삼을지 고르는 다이얼**이다. $T=1$은 teacher가 실제로 serving할 때의 분포이고, $T>1$은 serving에서는 결코 보지 않을 꼬리를 확대한 분포다.

> **정의.** **distillation loss**의 표준형은 hard label 항과 soft target 항의 볼록결합이다.
>
> $$
> \mathcal{L}_{\mathrm{KD}} \;=\; (1-w_{\mathrm{KD}})\,\mathcal{L}_{\mathrm{CE}}\big(y,\ p^{\mathrm{st},(1)}\big)
> \;+\; w_{\mathrm{KD}}\; T^2 \cdot \mathcal{F}\big(p^{\mathrm{te},(T)} \,\big\|\, p^{\mathrm{st},(T)}\big)
> \tag{4-2}
> $$
>
> $w_{\mathrm{KD}}\in[0,1]$은 두 신호의 혼합 비율, $\mathcal{F}$는 divergence(기본형은 forward KL), $y$는 정답 label이다. hard 항은 항상 $T=1$에서 계산한다 — 정답 label에는 온도가 없기 때문이다. ($w_{\mathrm{KD}}$는 hard 대 soft의 비율이고, §04.5에 나오는 $\lambda_{\mathrm{KD}}$는 [LM Need Sleep]의 divergence 대 reward 비율이다. 이름이 닮았을 뿐 다른 값이며 이 책은 둘을 섞지 않는다.)

$T^2$이 왜 붙는가. soft 항의 gradient는 student logit에 대해

$$
\frac{\partial}{\partial z^{\mathrm{st}}_i}\,\mathcal{F}\big(p^{\mathrm{te},(T)}\,\big\|\,p^{\mathrm{st},(T)}\big)
\;=\; \frac{1}{T}\Big(p^{\mathrm{st},(T)}_i - p^{\mathrm{te},(T)}_i\Big)
\tag{4-3}
$$

이고, 괄호 안의 확률 차이 자체도 $T$가 커질수록 작아진다. 두 효과가 겹쳐 gradient의 크기가 대략 $1/T^2$로 줄어든다. 보정을 넣지 않으면 $T$를 바꿀 때마다 $w_{\mathrm{KD}}$의 실효 의미가 함께 변해 두 항의 균형을 다시 튜닝해야 한다. $T^2$은 그 결합을 끊는 정규화 상수이며, **$1/T^2$ 근사가 정확한 것은 고온 극한에서뿐이다** — 이 장의 Worked micro-example이 $T=4$에서 실제 어긋남을 숫자로 보인다.

> **[해설]** $\mathcal{F}$의 방향이 실무에서 갈린다. forward KL $\mathcal{F}(p^{\mathrm{te}}\|p^{\mathrm{st}})$는 teacher가 확률을 준 곳을 student가 비우는 것을 강하게 벌하므로 student를 mode-covering으로 만들고, reverse KL $\mathcal{F}(p^{\mathrm{st}}\|p^{\mathrm{te}})$는 반대로 mode-seeking이다. student의 용량이 teacher보다 작을 때 forward KL은 teacher의 모든 mode를 평균하려다 어느 쪽도 아닌 분포를 만든다. [LM Need Sleep §3.3]이 채택한 GKD(Agarwal et al. 2024)가 divergence를 고정하지 않고 $\mathcal{F}$를 인자로 남겨 둔 이유가 이것이다.

systems 접점은 저장이다. logit distillation을 하려면 teacher의 출력 분포를 student의 backward가 소비할 때까지 들고 있어야 하는데, 그 텐서의 크기가 $L\times|\mathcal{V}|$다. 어휘가 십만 단위인 현대 LM에서 이것은 activation 하나가 아니라 **모델 가중치와 같은 자릿수의 텐서**다(이 장의 Worked micro-example에서 센다). 그래서 실무의 선택지는 둘뿐이다: teacher를 student와 같은 step에 함께 돌려 logit을 즉시 소비하고 버리거나(HBM에 두 모델 상주), teacher logit을 top-$k$로 잘라 디스크에 캐시하거나(top-$k$의 $k$는 남길 좌표 수를 뜻하는 장-국소 용법이며, 이 책 전역의 sleep 라운드 첨자 $k$와 무관하다). 전자는 sleep job의 메모리 상한을, 후자는 정확히 dark knowledge의 꼬리를 잘라낸다. 공짜 선택지는 없다.

---

## 04.3 feature distillation — 출력이 아니라 내부를 맞춘다

> **정의.** **feature distillation**(hidden-state distillation)은 최종 분포가 아니라 **중간 층의 활성값**을 맞추는 distillation이다. teacher의 층 $\ell$ 출력 $h^{\mathrm{te}}_\ell\in\mathbb{R}^{d_{\mathrm{te}}}$와 student의 대응 층 출력 $h^{\mathrm{st}}_\ell\in\mathbb{R}^{d_{\mathrm{st}}}$ 사이의 거리를 loss에 더한다.

$$
\mathcal{L}_{\mathrm{feat}} \;=\; \sum_{\ell \in \mathcal{A}} w_\ell \,\big\| P_\ell\, h^{\mathrm{st}}_\ell - h^{\mathrm{te}}_\ell \big\|_2^2
\tag{4-4}
$$

이 식이 logit 쪽과 다른 점은 세 가지다. 첫째, 층 대응 $\mathcal{A}$를 사람이 정해야 한다 — teacher가 32층이고 student가 12층이면 어느 층을 어느 층에 붙일지가 설계 변수다. 둘째, 폭이 다르면 학습 가능한 사영 $P_\ell\in\mathbb{R}^{d_{\mathrm{te}}\times d_{\mathrm{st}}}$가 필요하고, 이 $P_\ell$은 학습이 끝나면 버려지는 **일회용 파라미터**다. 셋째, divergence가 확률 사이의 것이 아니라 벡터 사이의 것이므로 표현의 스케일과 회전에 민감하다 — 같은 함수를 계산하는 두 네트워크도 내부 좌표계가 다르면 $\mathcal{L}_{\mathrm{feat}}$가 크게 나온다.

같은 계열에 attention map distillation이 있다. teacher의 attention 확률 행렬을 student가 재현하게 하는 것으로, 대상이 $L\times L$ 행렬이라는 점만 다르고 구조는 (4-4)와 같다.

> **[평가]** feature distillation은 logit distillation보다 강한 가정 위에 서 있다 — **teacher의 내부 표현이 재현할 가치가 있다**는 가정이다. logit 쪽은 입출력 사상만 맞추므로 student가 다른 내부 경로로 같은 함수를 계산해도 무방하지만, feature 쪽은 경로까지 지정한다. student의 구조가 teacher와 다를수록 이 가정은 약해지고, 사영 $P_\ell$이 흡수해야 할 불일치가 커진다. 이 corpus에서 sleep-time 계열 논문이 전부 logit·분포 쪽만 쓰고 feature 쪽을 쓰지 않는 것은 우연이 아니다 — sleep-time distillation의 teacher와 student는 구조가 거의 같으면서도 **파라미터 집합이 다르므로**, 내부 좌표를 고정하면 새로 늘린 용량이 쓰일 자리가 없어진다(§04.5).

systems 접점은 다시 저장이지만 성질이 다르다. logit은 마지막 층 하나에서 $|\mathcal{V}|$ 폭으로 나오고, feature는 $|\mathcal{A}|$개 층에서 $d$ 폭으로 나온다. $d\ll|\mathcal{V}|$이므로 층당 텐서는 작지만, 이 텐서들은 **teacher forward의 중간 activation이므로 checkpointing으로 버릴 수가 없다** — 버리면 다시 계산해야 한다. 결과적으로 feature distillation은 teacher forward의 activation 수명을 student backward 끝까지 늘리고, 그만큼 batch 크기를 깎는다. logit 쪽이 "한 번 크게 쓰고 버리는" 트래픽이라면 feature 쪽은 "여러 곳에 오래 붙잡아 두는" 점유다.

---

## 04.4 self-distillation — teacher가 자기 자신일 때

> **정의.** **self-distillation**은 teacher와 student가 같은 모델일 때의 distillation이다. "같다"는 것은 세 가지 중 하나를 뜻한다: (a) 같은 파라미터의 서로 다른 시점(라운드 $k$의 스냅샷이 라운드 $k{+}1$의 teacher), (b) 같은 모델의 서로 다른 파라미터 부분집합(일부만 학습 대상), (c) 같은 파라미터·다른 표본 경로. 세 경우 모두 teacher 쪽은 stop-gradient로 얼려 두므로 $\Theta^{\mathrm{te}}$는 상수다.

여기서 독자가 즉시 던지는 질문이 이 절의 전부다. **teacher가 student보다 아는 것이 없는데 어떻게 이득이 나는가.** 답은 셋이고, 셋 다 "지식이 이동해서"가 아니다.

첫째, **표본 출처가 바뀐다.** teacher가 만든 문장 위에서만 배운 student는 자기가 만든 문장 위에서 처음 보는 상태에 놓인다 — 학습 분포와 추론 분포의 불일치다. student가 스스로 디코딩한 토큰 위에서 teacher가 점수를 매기면 이 불일치가 사라진다.

> **정의.** distillation의 표본 $y$를 고정된 데이터셋 $\mathcal{D}$에서 뽑으면 **off-policy**, student 자신의 분포 $\mathrm{LM}_{\Theta^{\mathrm{st}}}(\cdot|x)$에서 뽑으면 **on-policy**다. 혼합 비율을 $\lambda_{\mathrm{on}}\in[0,1]$로 둔 형태가 GKD(Agarwal et al. 2024)이며, [LM Need Sleep §3.3]이 그대로 채택한다.

$$
\mathcal{L}_{\mathrm{GKD}}\;=\;(1-\lambda_{\mathrm{on}})\;\mathbb{E}_{(x,y)\sim\mathcal{D}}\Big[\mathcal{F}\big(\mathrm{LM}_{\Theta^{\mathrm{te}}}\,\big\|\,\mathrm{LM}_{\Theta^{\mathrm{st}}}\big)(y|x)\Big]
\;+\;\lambda_{\mathrm{on}}\;\mathbb{E}_{x\sim\mathcal{D}}\ \mathbb{E}_{y\sim\mathrm{LM}_{\Theta^{\mathrm{st}}}(\cdot|x)}\Big[\mathcal{F}\big(\mathrm{LM}_{\Theta^{\mathrm{te}}}\,\big\|\,\mathrm{LM}_{\Theta^{\mathrm{st}}}\big)(y|x)\Big]
\tag{4-5}
$$

이 식이 말하는 것은 divergence를 어디서 재느냐가 유일한 차이라는 것이다. 두 항의 $\mathcal{F}$는 같고, 기대값을 취하는 $y$의 출처만 다르다. [LM Need Sleep §3.3]은 여기에 두 개의 설계 결정을 명시로 붙인다 — "we do not backpropagate through the sampling distribution of the student"(샘플링 경로로는 gradient를 흘리지 않는다), 그리고 "we freeze all the parameters in the student model and only updates the expanded parameters". 앞의 것은 이산 샘플링이 미분 불가능하기 때문의 표준 처리이고, 뒤의 것은 §04.5의 핵심이다.

둘째, **목표가 label보다 부드럽다.** one-hot label은 정답 이외 좌표를 전부 0으로 밀지만 자기 자신의 분포는 그렇지 않다. 자기 출력을 목표로 두는 것은 정칙화로 작동한다.

셋째, **teacher가 앵커가 된다.** 라운드 $k$의 자기를 목표로 두면 라운드 $k{+}1$의 갱신은 옛 행동에서 멀어지는 것을 loss로 벌받는다. 이것은 옛 데이터를 다시 넣는 것과 목적은 같고 기제는 다르다 — 데이터가 아니라 함수를 붙잡는다. 옛 데이터를 다시 넣는 쪽(replay)은 ch07이 소유한다.

> **[해설]** on-policy 항이 sleep job의 비용 **형태**를 바꾼다. $\lambda_{\mathrm{on}}=0$이면 sleep은 평범한 training job이고 지배 비용은 forward+backward GEMM이다. $\lambda_{\mathrm{on}}>0$이면 매 optimizer step $s$ 앞에 student의 autoregressive 생성이 붙는다(한 sleep 라운드 $k$ 안에 step $s$가 여러 번 돈다 — 두 첨자는 다른 시간척도다) — 즉 sleep job의 지배 커널이 훈련의 compute-bound GEMM이 아니라 decode의 memory-bandwidth-bound 구간이 된다. 결과적으로 sleep job은 training 클러스터가 아니라 **wake 트래픽과 같은 하드웨어·같은 병목을 놓고 경쟁한다.** [LM Need Sleep App. B.5]가 보고하는 "같은 step 수에서 SFT가 자기 방법보다 4× 효율적"이라는 비율은 이 형태 변화의 값이다(반대로 같은 성능에 도달하려면 SFT가 4.3× / 3.6× / 4.8× 벽시계 시간을 쓴다고 같은 절이 보고한다).

경계 하나를 못 박아 둔다. 모델이 **텍스트로** 자기 학습 데이터를 써낸 다음 그 텍스트로 자기를 SFT하는 절차 — [SEAL §3.1]의 self-edit이 그것이다 — 도 흔히 self-distillation이라 불리지만, 그것은 데이터 공간의 절차이고 이 장의 대상이 아니다. 데이터를 만드는 법과 그 위험(자기생성 분포로 반복 학습할 때의 붕괴)은 ch06이 소유한다. 이 장은 **분포 공간에서 divergence를 최소화하는 절차**만 다룬다. 두 절차는 이름을 공유할 뿐 loss의 형태도 실패 양식도 다르다.

---

## 04.5 upward distillation — 작은 자기에서 큰 자기로

> **정의.** **upward distillation**은 **teacher가 student보다 파라미터가 적은** distillation이다. [LM Need Sleep §3.3]은 이것을 boxed 문단 제목 "Upward Distillation (Knowledge Seeding)"으로 도입하고, teacher와 student가 파라미터 확장 전후의 같은 모델일 때를 **self-Knowledge Seeding**이라 부른다 [LM Need Sleep §1 Contributions 2, p.4].

방향이 반대다. 통상의 distillation은 큰 teacher에서 작은 student로 간다. 그 절차의 정당화는 한 줄로 요약된다 — teacher가 student보다 많이 알고, student는 그중 담을 수 있는 만큼을 담는다. upward distillation은 그 정당화를 그대로 뒤집는다. teacher가 **덜** 아는데 무엇을 옮기는가.

핵심은 **옮겨지는 것이 정보량이 아니라 좌표라는 것**이다. [LM Need Sleep §3.2, p.7]의 sleep 라운드는 먼저 receiver 블록에 low-rank expert 한 쌍을 새로 만들어 붙인다 — $A\in\mathbb{R}^{d\times d_{\mathrm{low}}}$, $B\in\mathbb{R}^{d_{\mathrm{low}}\times d}$, $d_{\mathrm{low}}\ll d$. 이 확장 직후의 모델은 확장 전의 모델과 **다른 함수**다. 새 expert의 값이 무엇이든 그것이 forward에 더해지는 순간 출력이 흔들린다. 즉 **용량을 늘리는 행위 자체가 손상**이다.

upward distillation은 그 손상을 되돌리는 절차다. teacher는 확장 전의 자기이고, student는 확장 후의 자기이며, 학습 대상은 새로 붙인 $\{A,B\}$뿐이다(나머지는 전부 frozen — §04.4에서 인용한 그 문장이다). 목적은 **큰 파라미터 공간 안에서 작은 파라미터가 계산하던 함수를 재현하는 것**이고, 성공하면 새 expert는 "무해하면서 동시에 옛 행동을 자기 좌표로 표현할 줄 아는" 상태가 된다. 새로 생긴 것은 지식이 아니라 **비어 있고 정렬된 용량**이다.

> **[해설]** 그래서 upward distillation은 초기화 절차로 읽는 것이 정확하다. 랜덤 초기화된 새 파라미터는 "아무것도 모르는" 상태가 아니라 "옛 함수를 망가뜨리는" 상태다. seeding은 새 파라미터를 옛 함수의 값에 정렬시켜, 이후 wake에서 들어올 새 지식이 **옛 지식을 덮어쓰지 않고 옆에 앉을 자리**를 만든다. [LM Need Sleep App. A.4 (i), p.24]가 catastrophic forgetting을 "분포의 문제가 아니라 용량 부족의 문제"로 재정의하는 것이 이 설계의 전제다.

역방향이 자명하지 않은 두 번째 이유는 **수렴 목표가 다르다**는 데 있다. 통상의 distillation에서 student의 상한은 teacher다 — divergence가 0이면 완벽한 성공이고, 그 이상은 정의상 목표가 아니다. upward distillation은 그럴 수 없다. divergence를 0으로 만든 student는 더 큰 파라미터 공간에서 더 작은 함수를 정확히 흉내 내는 모델일 뿐이고, 늘린 용량은 한 비트도 쓰이지 않은 채로 남는다. 즉 이 절차의 성공 기준은 "완전한 모방"이 아니라 **"옛 함수를 망가뜨리지 않으면서 새 자유도를 열어 둔 상태"**이며, 그 둘 사이의 지점이 어디인지를 정하는 값이 (4-6)의 $\lambda_{\mathrm{KD}}$다. 그 값이 논문 어디에도 없다는 것이 이 절 끝에서 다시 문제가 된다.

무엇이 버려지는지도 짚어야 한다. 새 expert가 표현할 수 있는 것은 rank $d_{\mathrm{low}}$의 부분공간뿐이고, 정식화에서 논문이 $d_{\mathrm{low}}$에 대해 말하는 것은 $d_{\mathrm{low}}\ll d$ 하나다 [LM Need Sleep §3.2, p.7]. 실험 부록이 밝히는 것도 "차원 64의 MLP 블록 5개를 추가 파라미터로 쓴다"까지이며, $d$와 $d_{\mathrm{low}}$의 값 자체는 논문 어디에도 없다 [LM Need Sleep App. B, p.25]. 옛 함수 전체가 그 rank 안에 들어간다는 보장은 없으므로 seeding은 원리적으로 **손실 압축**이며, 무엇이 남고 무엇이 버려지는지에 대한 측정도 rate–distortion 형태의 진술도 논문에 없다. 얼마의 rank가 얼마의 망각을 막는지 역시 정식화되어 있지 않다. 용량의 상한 자체는 ch09가, 이 공백의 판정은 ch21과 ch22가 받는다.

이 절차가 복사가 아니라 **전송**인 이유는 마지막 단계에 있다. [LM Need Sleep §3.3(c), p.8]은 seeding이 끝난 뒤 sender 블록에 그동안 쌓인 low-rank expert들을 리셋한다("synaptic pruning"). 지운다는 사실이 이것을 전송으로 만든다 — 같은 내용이 빠른 블록과 느린 블록에 동시에 남아 있지 않다.

> **[평가]** 이 절차는 이 책의 표준형 (U-$\Theta$)에 담기지 않는다. (U-$\Theta$)는 고정 차원에서 gradient를 빼는 식인데, 여기서는 차원이 늘고 그 뒤 일부가 지워진다. 확장과 리셋에 대응하는 연산자가 표준형에 없다는 것은 프레임의 결함이 아니라 관할 밖이라는 뜻이다 — ch01이 세운 세 층은 상태가 **어디에** 있는지를 정하지, 그 상태의 차원이 라운드마다 바뀌는 경우를 정하지 않는다. 이 장은 그 두 연산을 (U-$\Theta$)의 바깥에 붙는 전후 처리로 읽고, 차원 변동을 회계에 넣는 문제는 ch09와 ch21로 넘긴다.

systems 접점은 텐서 모양이다. 차원이 실제로 늘어나면 컴파일된 그래프도 kernel autotuning도 sleep마다 무효가 되므로, 논문은 확장분을 처음부터 모델에 넣어 두고 활성화 전까지 forward·backward에서 masking하는 구현을 제시한다 [LM Need Sleep §3.3 Note on the Implementation, p.9]. 즉 자라는 $\Theta$는 서빙 쪽에 고정 모양으로 위장되고, 그 대가로 용량 상한이 배포 시점에 못 박힌다 — 그 상한의 크기는 논문에 없다.

> **[평가]** upward distillation의 정당화는 논리적으로 깔끔하지만, **그 기여의 크기는 논문 자신의 표에서 작다.** [LM Need Sleep Table 1, p.11](Qwen3-8B, AIME-24 / AIME-25 / HMMT-25, avg@16)에서 전체 Sleep은 79.2 / 69.0 / 46.1이고, 확장을 뺀 "Sleep w/o Expansion"은 78.2 / 67.9 / 44.9다. 반대 방향으로, 확장만 baseline OPSD에 얹은 "OPSD + Expansion"은 77.9 / 68.2 / 45.9로 OPSD의 76.6 / 67.4 / 45.1보다 +1.3 / +0.8 / +0.8 오른다. Sleep이 OPSD를 이기는 총 마진이 +2.6 / +1.6 / +1.0이므로, **그 마진의 절반가량 — HMMT-25에서는 1.0 중 0.8 — 이 Knowledge Seeding 없이 구조적 확장만으로 얻어진다.** 이 표에는 시드도 신뢰구간도 없다. 판정은 ch21로 넘긴다.

> **[평가]** 같은 표의 다른 행은 더 직접적이다. Semantic Reward를 제거하면 AIME-25가 **오른다**(69.2 대 전체 Sleep의 69.0) [LM Need Sleep Table 1, p.11]. 그런데 §4.2 p.13은 "All the components contribute positively to the performance of our method"라고 쓴다. 자기 표가 반증하는 문장이다. Imitation Learning 제거는 76.8 / 67.9 / 45.0으로 세 열 모두 내려가므로, 이 논문에서 실제로 하중을 받는 항은 semantic 쪽이 아니라 imitation 쪽이다.

seeding의 목적함수 자체는 §04.6의 좌표로 그대로 읽힌다. [LM Need Sleep §3.3, Eq. 4 뒤의 무번호 식]은 divergence 항과 reward 항을 섞는다.

$$
\mathcal{L}_{\mathrm{KS}}\;=\;\mathbb{E}_{x\sim\mathcal{D}}\Big[(1-\lambda_{\mathrm{KD}})\;\mathbb{E}_{y\sim\mathrm{LM}_{\Theta^{\mathrm{st}}}(\cdot|x)}\big[R(y)\big]\;-\;\lambda_{\mathrm{KD}}\;\mathbb{E}_{y\sim\mathrm{LM}_{\Theta^{\mathrm{st}}}(\cdot|x)}\,\mathcal{F}\big(\mathrm{LM}_{\Theta^{\mathrm{te}}}\,\big\|\,\mathrm{LM}_{\Theta^{\mathrm{st}}}\big)(y|x)\Big]
\tag{4-6}
$$

$R(\cdot)$는 teacher의 생성 $d^{(i)}$의 임의 접두사를 student가 이어 쓰게 하고 그 결과를 채점하는 imitation reward이며, semantic 항과 Levenshtein 기반 절대 항의 볼록결합이다 [LM Need Sleep Eq. 3, Eq. 4]. (표기 대응: 원문은 이 보상을 $r(\cdot)$, 두 성분을 $r_{\mathrm{sem}}$·$r_{\mathrm{abs}}$로 쓴다 [LM Need Sleep Eq. 3]. 이 책에서 $r$은 LoRA rank로 예약되어 있으므로 보상은 $R$로 옮긴다.) reward로 $\Theta$를 움직이는 절차 일반은 ch05가 소유한다 — 여기서 볼 것은 그 항이 **distillation loss와 같은 식 안에 앉는다**는 사실뿐이다. 즉 (4-6)은 "teacher의 분포를 따라가라"(divergence 항)와 "teacher가 쓸 법한 문장을 실제로 써내라"(reward 항)를 하나의 목적으로 묶는다.

> **[평가]** 세 개의 caveat을 여기에 붙여 둔다. 첫째, **(4-6)의 최적화 방향이 원문에 없다.** reward에서 divergence를 뺀 형태이므로 최대화해야 하지만, 같은 페이지의 산문은 "minimize the distillation loss between teacher and student"라고 쓴다 [LM Need Sleep §3.3(b)]. 둘째, 원문은 같은 줄에서 $\mathcal{D}$를 데이터셋과 divergence 양쪽에 쓰고, 한 문단 앞 (4-5)에서는 divergence를 $\mathcal{F}$로 쓴다 — 이 책은 $\mathcal{F}$로 통일하고 $\mathcal{D}$는 데이터셋 전용으로 둔다. 셋째, $\lambda_{\mathrm{on}}$, $\lambda_{\mathrm{KD}}$, reward 혼합 계수, Levenshtein 임계값 어느 것도 **논문 어디에도 값이 없다** — Table 5는 learning rate·batch·steps·LoRA rank/alpha만 싣는다. 세 항을 합치면 판정은 하나다: **이 절차는 논문만으로 재현되지 않는다.**

<!-- TODO-VERIFY: [LM Need Sleep]이 Knowledge Seeding에 쓰는 frozen semantic reward model의 정체. 확인 방법: papers/2606.03979v2.txt 검색 "reward model"; 없으면 재현 불가로 ch21에 기록. -->

기제 전체가 pre-trained Llama/Qwen 위에 얹은 graft이며 end-to-end로 meta-learn된 것이 아니라는 사실은 NM ch17에서 이미 확정되었고, ch21이 그대로 승계한다.

한 가지 더. [SEAL §3.1, p.4] 마지막 문단은 teacher와 student를 **다른 모델**로 분리하는 확장을 제안한다 — teacher가 편집을 제안하고, student가 그것으로 갱신되며, teacher는 student의 개선을 최대화하도록 학습된다는 구도다. 제안만 되어 있고 실행된 실험은 없다. 이 장의 정의가 teacher/student를 역할로만 규정한 이유가 여기서 드러난다: 같은 프레임 안에서 self-distillation, upward distillation, 그리고 이 분리형 제안이 전부 **$\Theta^{\mathrm{te}}$와 $\Theta^{\mathrm{st}}$의 관계를 무엇으로 두느냐**의 선택지로 정리된다.

---

## 04.6 distillation loss의 형태 — 세 좌표로 읽는다

지금까지의 모든 변형은 하나의 뼈대에 붙는다.

$$
\mathcal{L}_{\mathrm{distill}}\;=\;\sum_{i} w_i\;\mathbb{E}_{y\sim \pi_i}\Big[\mathcal{F}_i\big(g_i(\Theta^{\mathrm{te}};x,y)\ \big\|\ g_i(\Theta^{\mathrm{st}};x,y)\big)\Big]
\tag{4-7}
$$

이 식이 말하는 것은 distillation을 정하는 자유도가 정확히 셋이라는 것이다 — **무엇을 맞출 것인가**($g_i$: 대상), **어떤 거리로 잴 것인가**($\mathcal{F}_i$: divergence), **어느 분포에서 표본을 뽑을 것인가**($\pi_i$: 표본 출처). 표 4-1이 이 장에서 나온 것들을 그 좌표에 배치한다.

표 4-1 — distillation의 세 좌표

| 변형 | 대상 $g$ | divergence $\mathcal{F}$ | 표본 출처 $\pi$ |
|---|---|---|---|
| logit distillation (§04.2) | $\lvert\mathcal{V}\rvert$차원 출력 분포, 온도 $T$ | forward/reverse KL | 고정 데이터셋 |
| feature distillation (§04.3) | 층 $\ell$의 hidden $h_\ell$ (사영 $P_\ell$ 경유) | 제곱 $\ell_2$ | 고정 데이터셋 |
| attention distillation (§04.3) | $L\times L$ attention 확률 | KL 또는 제곱 $\ell_2$ | 고정 데이터셋 |
| off-policy self-distill (§04.4) | 출력 분포 | KL | 고정 데이터셋 |
| on-policy self-distill / GKD (§04.4) | 출력 분포 | KL | student 자신, 비율 $\lambda_{\mathrm{on}}$ |
| Knowledge Seeding (§04.5) | 출력 분포 **+** 생성 접두사 완성 | $\mathcal{F}$ + imitation reward, 비율 $\lambda_{\mathrm{KD}}$ | student 자신 [LM Need Sleep Eq. 3] |

마지막 행이 이 corpus가 실제로 쓰는 좌표다. 대상이 둘이고(분포와 생성), 표본이 student 쪽이며, teacher는 자기 자신의 확장 이전 상태다.

> **[해설]** 이 세 좌표는 비용과 1:1로 대응한다. **대상**은 저장량을 정한다(logit이면 $|\mathcal{V}|$ 폭, feature면 층 수 × $d$). **divergence**는 사실상 공짜다 — elementwise 연산이다. **표본 출처**는 지배 커널을 정한다(고정 데이터셋이면 GEMM, student 자신이면 decode). 실무에서 distillation 파이프라인의 비용을 추정할 때 먼저 봐야 할 칸은 세 번째다.

---

## (state, update, cost) 정리

이 장의 개념을 세 층 프레임에 놓으면 표 4-2가 된다. 요점은 단순하다 — **distillation은 전부 $\Theta$ 층의 도구다.** $W$도 $E$도 건드리지 않는다.

표 4-2 — 이 장의 개념과 세 층 프레임

| 개념 | 어느 층 | 갱신식 | 비용 4종에서 어디에 뜨는가 |
|---|---|---|---|
| teacher forward | 어느 층도 갱신하지 않음 ($\Theta^{\mathrm{te}}$ read-only) | 없음 | $B_s$ — batch prefill과 같은 커널 |
| logit distillation | $\Theta^{\mathrm{st}}$ | (U-$\Theta$), $\mathcal{R}_k$ = (입력, teacher 분포) 쌍 | $B_s$ + teacher logit 저장 |
| feature distillation | $\Theta^{\mathrm{st}}$ + 일회용 사영 $P_\ell$ | (U-$\Theta$) | $B_s$ + teacher activation 점유 |
| self-distillation (off-policy) | $\Theta^{\mathrm{st}}$, teacher = $\Theta_k$ 스냅샷 | (U-$\Theta$) | $B_s$; 두 파라미터 사본이 동시 상주 |
| on-policy self-distill ($\lambda_{\mathrm{on}}>0$) | 위와 같음 | (U-$\Theta$), $\mathcal{R}_k$가 매 step 재샘플 | $B_s$가 decode에 지배됨 |
| upward distillation | $\Theta_{k+1}\supsetneq\Theta_k$ (차원 증가) | (U-$\Theta$) + 확장 + sender 리셋 | $C_{\mathrm{cap}}$ — consolidation당 $2\,d\,d_{\mathrm{low}}$ 파라미터 [LM Need Sleep §3.2] |

$L_w$(wake 지연)는 이 장의 어떤 절차에서도 직접 오르지 않는다. distillation은 전부 질의 도착 전에 끝나고, wake에 남는 것은 바뀐 배포 아티팩트뿐이다. $\rho$(망각률)는 반대로 **이 장의 절차들이 정면으로 겨냥하는 값**이지만, 그것을 측정한 수치는 이 장의 근거 논문 어디에도 없다.

---

## Worked micro-example — temperature 하나가 gradient와 바이트에 하는 일

이 절의 수치는 전부 **이 책의 예시 계산**이다. 논문에서 가져온 값이 하나도 없고, 손으로 검산할 수 있도록 로짓·어휘 크기·시퀀스 길이를 이 책이 골랐다.

어휘가 세 개인 장난감 모델을 쓴다. 한 토큰 위치에서 teacher logit이 $z^{\mathrm{te}}=(4,\,2,\,0)$, student logit이 $z^{\mathrm{st}}=(2,\,2,\,2)$라 하자. student는 아직 아무것도 모르는 균등분포다.

**(1) soft target.** (4-1)로 $T=1$과 $T=4$의 teacher 분포를 만든다. $T=4$면 지수의 인자가 $(1.0,\,0.5,\,0)$이 되어 $e^1=2.71828$, $e^{0.5}=1.64872$, $e^0=1$이므로 합이 $5.36700$이다.

| $T$ | $p^{\mathrm{te},(T)}$ |
|---|---|
| 1 | (0.86681, 0.11731, 0.01588) |
| 4 | (0.50648, 0.30720, 0.18632) |

세 번째 좌표가 0.0159에서 0.1863으로 **11.7배** 커진다. 이것이 dark knowledge를 학습 신호로 끌어올린다는 말의 실제 내용이다.

**(2) divergence.** student는 어느 $T$에서도 $(1/3,1/3,1/3)$이므로 $\mathcal{F}(p^{\mathrm{te}}\|p^{\mathrm{st}})=\ln 3-H(p^{\mathrm{te}})$다. $\ln 3=1.09861$이고 엔트로피는 각각 $0.44106$, $1.02019$ nat이므로

$$
\mathcal{F}\big|_{T=1}=0.65755\ \text{nat},\qquad
\mathcal{F}\big|_{T=4}=0.07842\ \text{nat}.
$$

같은 두 모델인데 loss 값이 **8.38배** 차이 난다. $T$를 바꾸면 loss의 눈금 자체가 바뀐다는 뜻이고, 이것이 (4-2)에 정규화 상수가 필요한 이유다.

**(3) gradient와 $T^2$.** (4-3)으로 student logit에 대한 gradient를 직접 계산한다.

| $T$ | $\partial\mathcal{F}/\partial z^{\mathrm{st}}$ |
|---|---|
| 1 | $(-0.53348,\ +0.21602,\ +0.31746)$ |
| 4 | $(-0.04329,\ +0.00653,\ +0.03675)$ |

첫 좌표의 크기 비는 $0.53348/0.04329=\mathbf{12.32}$다. 여기에 $T^2=16$을 곱하면 $0.04329\times16=0.69269$가 되어 $T=1$의 $0.53348$을 **30% 초과한다**. 즉 $T^2$ 보정은 이 온도·이 로짓 간격에서 정확하지 않다 — 정확한 것은 로짓 차이가 $T$에 비해 작은 고온 극한이다. 실무적 함의는 분명하다: $T$를 크게 흔들면 $w_{\mathrm{KD}}$의 실효값이 따라 움직이므로 두 값을 독립적으로 튜닝했다고 믿으면 안 된다.

**(4) 바이트.** 이제 같은 계산을 실제 어휘 크기로 옮긴다. 예제 값으로 $|\mathcal{V}|=128{,}000$, 시퀀스 길이 $L=2{,}048$, dtype bf16(2 bytes)을 쓴다. teacher logit 전체를 들고 있으려면

$$
2{,}048 \times 128{,}000 \times 2\ \text{B} \;=\; 524{,}288{,}000\ \text{B} \;=\; 500\ \text{MiB}
$$

가 **시퀀스 한 개당** 필요하다. 8B 모델의 bf16 가중치가 16 GB이므로, 시퀀스 하나의 logit 텐서가 이미 가중치의 약 3.3%다. distillation 집합이 200 시퀀스면 97.7 GiB — 캐시할 수 있는 크기가 아니다. top-$k$로 자르면($k=64$, 값 bf16 2 B + 인덱스 int32 4 B)

$$
2{,}048 \times 64 \times 6\ \text{B} \;=\; 786{,}432\ \text{B} \;=\; 768\ \text{KiB},
$$

즉 **666.7배** 작아진다. 잘려 나간 $128{,}000-64$개 좌표가 정확히 (1)에서 확대했던 꼬리라는 점이 이 절의 결론이다. **temperature로 꼬리를 키우는 일과 top-$k$로 꼬리를 버리는 일은 같은 파이프라인 안에서 서로를 상쇄한다.** 두 값을 따로 정하면 안 된다.

**(5) 표본 출처가 이 계산을 무의미하게 만드는 경우.** $\lambda_{\mathrm{on}}>0$이면 위의 캐시 논의 자체가 성립하지 않는다. student가 매 step 새로 생성한 $y$ 위에서 divergence를 재므로 teacher logit을 미리 계산해 둘 대상이 없고, teacher는 매번 student의 생성물을 다시 forward해야 한다. 저장 문제가 사라지는 대신 지배 커널이 decode로 바뀐다. 이 교환의 크기를 재려면 절대 비용이 필요한데 논문이 내놓는 것은 비율 하나뿐이다 — [LM Need Sleep App. B.5]는 같은 step 수에서 SFT가 자기 방법보다 4× 효율적이라고 보고하고, 그 비교 대상은 SFT 하나뿐이다.

---

## 요약

- distillation은 gradient의 출처를 label에서 다른 모델의 출력 분포로 바꾸는 절차이고, teacher와 student는 크기가 아니라 **파라미터의 이동 여부**로 정의된다.
- temperature $T$는 teacher 분포의 꼬리를 학습 신호로 끌어올리는 다이얼이며, $T^2$ 보정은 hard/soft 혼합 비율 $w_{\mathrm{KD}}$의 실효 의미를 $T$와 분리하기 위한 것이다. 그 보정이 정확한 것은 고온 극한뿐이고, 이 책의 예시 계산($T=4$)에서 실제 비는 12.32 대 16이다.
- logit distillation의 비용은 $|\mathcal{V}|$ 폭 텐서의 저장이고 feature distillation의 비용은 teacher activation의 수명 연장이다. 전자는 top-$k$로 줄일 수 있으나(이 책의 예시 계산에서 666.7배) 그 대가가 정확히 dark knowledge의 꼬리다.
- self-distillation이 이득을 내는 이유는 지식의 이동이 아니라 세 가지다: 표본 출처의 정합(on-policy), 목표의 부드러움, 그리고 옛 함수를 붙잡는 앵커 효과.
- on-policy 비율 $\lambda_{\mathrm{on}}$은 sleep job의 지배 커널을 GEMM에서 autoregressive decode로 바꾼다. 그 결과 sleep은 training 클러스터가 아니라 wake 트래픽과 같은 병목을 놓고 경쟁한다.
- upward distillation에서 옮겨지는 것은 정보량이 아니라 좌표다. 용량 확장 자체가 옛 함수에 대한 손상이고, Knowledge Seeding은 그 손상을 되돌려 **비어 있고 정렬된 용량**을 남긴다. sender 블록의 리셋이 이것을 복사가 아니라 전송으로 만든다.
- [LM Need Sleep Table 1, p.11]에서 Sleep이 OPSD를 이기는 마진 +2.6 / +1.6 / +1.0 중 +1.3 / +0.8 / +0.8은 Knowledge Seeding 없이 구조적 확장만으로 얻어지고, Semantic Reward를 빼면 AIME-25는 69.0에서 69.2로 오른다. 시드도 신뢰구간도 없다.
- distillation은 전부 $\Theta$ 층의 도구다. $W$도 $E$도 건드리지 않고 $L_w$를 직접 올리지 않으며, 정면으로 겨냥하는 $\rho$는 이 장의 근거 논문 어디에도 측정되어 있지 않다.

## 자가 점검 체크리스트

- [ ] teacher/student를 크기가 아니라 파라미터의 이동 여부로 정의했고, 그 정의 덕분에 "작은 teacher, 큰 student"가 형용모순이 아님을 설명할 수 있다.
- [ ] (4-2)에서 $T^2$이 붙는 이유와, 그 보정이 정확해지는 조건을 함께 말할 수 있다.
- [ ] logit distillation과 feature distillation 중 어느 쪽이 batch 크기를 더 깎는지, 그 이유를 저장의 성질(일회성 대 점유)로 설명할 수 있다.
- [ ] $\lambda_{\mathrm{on}}$을 0에서 1로 올렸을 때 sleep job의 지배 커널과 roofline 위치가 어떻게 바뀌는지 말할 수 있다.
- [ ] upward distillation에서 새 expert로 옮겨지는 것과, 반대로 그 절차로는 **생기지 않는** 것을 구분해 말할 수 있다.
- [ ] [LM Need Sleep Table 1]의 어느 행 비교가 "구조적 확장만의 기여"를 분리해 주는지 지목하고 그 값을 숫자로 답할 수 있다.
- [ ] (Rosetta) "모델 재배포 = (U-$\Theta$) 1회"라는 대응을 이 장의 절차에 적용해, distillation이 끝나는 순간 serving에서 정확히 무엇이 달라지는지 말할 수 있다. 그리고 자신이 이미 배포 중인 speculative decoding의 draft model이 바로 이 장의 student라는 것을 확인했다.

## 다음 장으로

이 장의 모든 절차는 하나의 전제 위에 서 있다 — **teacher의 출력이 따라갈 만하다**는 전제다. 그 전제가 성립할 때 divergence를 줄이는 것은 곧 개선이다. 그러나 sleep-time compute의 목표는 배포된 모델이 **자기보다 나아지는 것**이고, 자기 자신을 teacher로 삼는 순간 divergence를 0으로 만드는 최적해는 아무것도 바꾸지 않는 것이다. 실제로 (4-6)은 이미 그 벽을 넘고 있었다: divergence 항 옆에 붙은 imitation reward 항은 "teacher를 닮아라"가 아니라 "채점을 통과하라"는 신호이며, 그것은 distillation의 문법이 아니다.

따라서 다음 질문은 이것이다 — **teacher가 없거나 teacher를 넘어서야 할 때, 학습 신호를 어디서 얻는가.** ch05가 그 답을 다룬다: 정답 분포 대신 스칼라 보상으로 $\Theta$를 움직이는 절차, 그 절차가 요구하는 채점기, 그리고 채점기가 있으면 반드시 따라오는 reward hacking이다. [SEAL]과 [LM Need Sleep]이 공유하는 ReST$^{\mathrm{EM}}$ 루프도 거기서 정의된다.
