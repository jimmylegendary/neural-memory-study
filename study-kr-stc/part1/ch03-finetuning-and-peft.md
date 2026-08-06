# ch03. fine-tuning과 PEFT — 가중치에 쓰는 법

> **이 장의 목표** — 이 장을 마치면 다음 넷을 할 수 있어야 한다.
> 1. full fine-tuning · adapter · LoRA를 "$\Theta$의 어느 부분집합을 움직이는가"로 구분하고, 각각이 배포 아티팩트로 무엇을 남기는지 말한다.
> 2. $d$, $L_{\mathrm{layer}}$, $r$, 적용 행렬 집합만 주어졌을 때 LoRA delta의 파라미터 수와 바이트를 손으로 계산한다.
> 3. merge된 형태와 분리된 delta 형태가 decode에서 무엇을 바꾸는지, 특히 shared-weight batching이 정확히 어디서 깨지는지 설명한다.
> 4. delta의 dtype이 저장 바이트당 정보량을 어떻게 바꾸는지 계산하고, QLoRA와 delta 양자화가 서로 다른 축임을 구분한다.
>
> **왜 필요한가** — Part II의 Θ-경로 세 장이 이 장 없이는 읽히지 않는다. ch19는 LoRA를 논문 장으로 전개하고, ch20의 SEAL은 self-edit의 **산출물이 LoRA delta**이며, ch21의 `Language Models Need Sleep`은 CMS의 low-rank expert $A^{(\ell),j}, B^{(\ell),j}$와 폭 $d_{\mathrm{low}}$를 이 장의 $(A, B, r)$ 위에 세운다. ch09의 용량 논의는 `Understanding LoRA as Knowledge Memory`(arXiv:2603.01097)를 다루는데, 그 논문이 재는 대상이 정확히 이 장의 $r$이다. Part III의 상태 용량 회계(ch26)와 서빙 판정(ch29)은 이 장이 정의하는 **delta 아티팩트의 크기 계산**을 기준선으로 쓴다.
>
> **NM과의 관계** — Neural Memory에 대응 장이 없다. 새로 쓴다. 다만 backward pass, optimizer state, outer loop의 의미는 전제한다(→ NM ch02, → NM ch04). 이 장이 더하는 것은 하나다: (U-$\Theta$)를 **전부가 아니라 부분적으로** 도는 방법, 그리고 그 부분이 **파일로 떨어져 배포 아티팩트가 된다**는 사실. NM이 300쪽에 걸쳐 다룬 $W$(fast weights)는 이 장에 한 번도 등장하지 않는다 — 이 장의 모든 기제는 test-time에 아무것도 움직이지 않는다.

> **[표기 경고 — 이 장 전용]** LoRA 원문은 사전학습 weight 행렬을 $W$로 쓰고, 학습되는 소수의 파라미터 집합을 $\Theta$로 쓴다 [lora §4.1, lora Eq. 2]. 이 책의 예약 기호와 **둘 다 정반대**다. 이 책에서 $W$는 fast weights(test-time에 움직이는 상태)이고 $\Theta$는 slow weights 전체다. 이 장의 모든 수식은 통일 표기로 옮긴다: 원문 $\Phi_0 \to \Theta_0$, 원문 $W_0 \to \Theta_0^{(m)}$, 원문 $\Theta \to (A, B)$, 원문 $\alpha \to \alpha_{\mathrm{LoRA}}$, 원문 $L$(층 수) $\to L_{\mathrm{layer}}$. attention의 네 투영 행렬도 같은 규칙을 따른다: 원문 $W_q, W_k, W_v, W_o \to \Theta^{(q)}, \Theta^{(k)}, \Theta^{(v)}, \Theta^{(o)}$ — 위첨자는 식 (3-2)의 행렬 인덱스 $m$이 취하는 값이며, 시간 첨자가 아니다. 원문 기호는 이 문단과 원문 직접 인용 밖에 등장하지 않는다. 또 이 장은 LoRA를 적용한 행렬들의 집합을 $\mathcal{S}$로 쓴다 — 이 장이 도입해 이 책 전체의 예약 기호가 된 것이고(ch09·ch19가 그대로 쓴다), 정의는 이 장이 소유한다. 예약 기호 $S_t$(momentum)·$S(c; B_s)$(오프라인 처리 절차) 어느 쪽과도 무관하다.

---

## 03.1 full fine-tuning — 기준선, 그리고 그것이 남기는 것

**full fine-tuning**은 사전학습된 slow weights $\Theta_0$ 전체를 downstream 목적함수의 gradient로 움직여 새 $\Theta$를 얻는 절차다. 움직임에 구조적 제약이 없다는 것이 정의의 전부다.

$$
\max_{\Theta}\ \sum_{(x,y)\in\mathcal{D}}\ \sum_{t=1}^{|y|} \log P_{\Theta}\big(y_t \mid x,\ y_{<t}\big)
\tag{3-1}
$$

이 식이 말하는 것은 단순하다. 학습 집합 $\mathcal{D}$의 모든 (입력, 정답) 쌍에 대해 정답 토큰의 로그확률 합이 커지도록 $\Theta$를 옮기고, 초기값은 $\Theta_0$다 [lora Eq. 1]. 이것이 ch01이 세운 표준형 (U-$\Theta$)를 실제 목적함수로 채운 가장 평범한 형태이며, 이 장의 나머지 방법은 전부 "같은 목적함수를, 더 작은 부분공간에서" 푸는 변형이다.

표준형 (U-$\Theta$)와 대조하면 이 장 전체의 위치가 정해진다. 표준형은 $\Theta_{k+1} = \Theta_k - \eta_\Theta \nabla_\Theta \mathcal{L}(\mathcal{R}_k; \Theta_k)$이고 $\mathcal{R}_k = \mathrm{gen}(\mathcal{H}_k; B_s)$인데, 식 (3-1)에서는 $\mathcal{R}_k$가 사람이 미리 만들어 놓은 $\mathcal{D}$로 고정되어 $\mathrm{gen}(\cdot)$이 항등이 된다. 또 반복 첨자가 sleep 라운드 $k$가 아니라 한 번의 adaptation run 안의 SGD step $s$다 — 이 장은 step 첨자로 $s$만 쓰고, $k$는 ch04 이후의 sleep 라운드에 남겨 둔다. **즉 이 장의 방법은 (U-$\Theta$)의 왼쪽 절반(무엇을, 어떻게 움직이는가)만 채우고 오른쪽 절반(학습 집합을 어떻게 만드는가)은 비워 둔다.** 그 빈칸을 채우는 것이 ch04 이후의 일이다.

먼저 왜 이 절차가 필요한지부터 정한다. 문맥에 예시를 넣는 것으로는 대체되지 않기 때문이다. GPT-3 175B에서 few-shot과 fine-tuning의 격차는 MNLI-m 40.6 대 89.5, RTE 69.0 대 85.4다 [lora Table 8, lora App. A]. 40점 이상의 격차는 프롬프트 기법으로 메울 수 있는 종류가 아니다. **가중치에 쓰는 것과 문맥에 쓰는 것은 같은 일이 아니다** — 이 사실이 Θ-경로 전체의 존재 이유이며, Part III에서 E-경로와 비교할 때 다시 불려 나온다.

문제는 그 대가다. full fine-tuning의 산출물은 $\Theta_0$와 **크기가 같은** 새 checkpoint다. 논문의 표현으로 "the new model contains as many parameters as in the original model" [lora §1]이고, 더 정확하게는 "for each downstream task, we learn a different set of parameters $\Delta\Phi$ whose dimension $|\Delta\Phi|$ equals $|\Phi_0|$" [lora §2]다. GPT-3 175B의 checkpoint는 350GB이므로 [lora §4.2], task 하나마다 350GB가 새로 생긴다.

> **[해설]** 독자의 어휘로 옮기면 이렇다. full fine-tuning은 **모델 재배포**다 — Rosetta 사전의 "모델 재배포 ↔ (U-$\Theta$) 1회" 행이 바로 이 절차를 가리킨다. 배포 파이프라인·이미지 크기·롤백 전략이 전부 딸려 온다. task가 $n$개면 아티팩트도 $n$개이고, 그 $n$개는 서로 다른 GPU에 올라가야 한다. 100개 task를 독립적으로 fine-tune하면 약 35TB다 [lora §4.2 각주 4].

쓰기 시점의 비용도 배포 비용과 별개로 존재한다. GPT-3 175B의 full fine-tuning은 VRAM 1.2TB를 쓴다 [lora §4.2]. 이 숫자의 출처를 정확히 아는 것이 이 장의 나머지를 이해하는 열쇠다 — 파라미터 자체가 아니라 **optimizer state와 gradient**가 지배한다 [lora §1]. Adam 계열은 파라미터당 1차·2차 moment 두 벌을 들고 있어야 하고, 여기에 gradient 한 벌이 더 붙는다(→ NM ch02의 optimizer-as-object). 따라서 학습되는 파라미터 수를 줄이면 메모리는 파라미터 수보다 **가파르게** 줄어든다. 이것이 PEFT의 계산적 동기이며, 뒤에 나올 "1.2TB → 350GB"라는 수치의 기제다.

---

## 03.2 adapter — 삽입의 대가는 depth이고, depth의 대가는 decode다

**adapter**는 얼린 $\Theta_0$의 층 **사이에** 작은 학습 가능 모듈을 직렬로 삽입하고 그 모듈만 학습하는 방식이다(Houlsby et al. 2019). 저장 효율의 관점에서는 성공적이다 — GPT-3에서 Adapter$_H$는 7.1M 파라미터로 WikiSQL 71.9 / MNLI-m 89.8을, 40.1M으로 73.2 / 91.5를 낸다 [lora Table 4]. 175B 대비 $10^{-4}$ 수준의 아티팩트로 full fine-tuning(73.8 / 89.5)에 근접한다.

문제는 저장이 아니라 계산 그래프다. adapter를 **삽입**한다는 말은 forward의 depth가 늘어난다는 뜻이고, 늘어난 depth는 순차적으로만 처리된다. 논문은 이를 정면으로 지적한다: "large neural networks rely on hardware parallelism to keep the latency low, and adapter layers have to be processed sequentially. This makes a difference in the online inference setting where the batch size is typically as small as one." [lora §3]

이 주장은 측정으로 뒷받침된다. GPT-2 medium, NVIDIA Quadro RTX8000, single forward pass 100회 평균 [lora Table 1, lora App. B]:

표 3-1. adapter의 inference latency 오버헤드 (ms, 괄호는 Fine-Tune/LoRA 대비 증가율) [lora Table 1]

| 설정 | Fine-Tune / LoRA | Adapter$_L$ | Adapter$_H$ |
|---|---|---|---|
| batch 32, seq 512, 0.5M | 1449.4 ± 0.8 | 1482.0 ± 1.0 (+2.2%) | 1492.2 ± 1.0 (+3.0%) |
| batch 16, seq 256, 11M | 338.0 ± 0.6 | 354.8 ± 0.5 (+5.0%) | 366.3 ± 0.5 (+8.4%) |
| batch 1, seq 128, 11M | 19.8 ± 2.7 | 23.9 ± 2.1 (+20.7%) | 25.8 ± 2.2 (+30.3%) |

**순차 depth의 비용은 batch가 작을수록 커진다.** 이 표는 그것을 세 줄로 말한다. batch 32 / seq 512에서는 3% 이하이고, batch 1 / seq 128에서는 30%까지 간다. 독자에게 이 패턴은 낯설지 않다 — batch가 작은 decode는 compute-bound가 아니라 memory-bound이며, 층 하나를 추가할 때 실제로 지불하는 것은 FLOPs가 아니라 kernel launch와 그 사이의 동기화다. adapter의 bottleneck GEMM은 FLOPs 기준으로 무시할 수준이지만 launch 횟수 기준으로는 무시되지 않는다. 그리고 online serving의 decode는 정확히 표의 마지막 줄에 있다.

> **[평가]** 논문은 여기서 한 걸음 더 나가 "이 문제는 모델을 shard하면 더 나빠진다 — AllReduce와 Broadcast 같은 동기 연산이 늘기 때문"이라고 쓴다 [lora §3]. 방향은 옳지만 **측정 없이 단언된다.** Table 1과 Fig. 5는 모두 단일 GPU 실험이고, tensor-parallel 환경의 adapter 오버헤드를 잰 숫자는 이 논문에 없다. 이 책은 이 주장을 "그럴듯한 기제"로 인용하되 근거 있는 사실로 취급하지 않는다.

삽입 대신 **문맥을 쓰는** 계열도 있다. prefix/prompt tuning은 학습 가능한 special token을 입력 앞에 붙이는 방식인데, 이 계열은 depth가 아니라 다른 예산을 먹는다 — "reduce the model's usable sequence length" [lora §1]. 독자의 회계로 옮기면 학습 파라미터를 늘릴수록 KV cache 예산과 유효 문맥이 줄어든다는 뜻이고, 그래서 파라미터를 늘리면 성능이 단조롭게 오르지 않는다. GPT-3에서 PrefixEmbed는 0.4M → 6.4M 구간에서 WikiSQL 55.9 / 58.7 / 60.6 / 63.1 / 55.9로 마지막에 무너지고, PrefixLayer도 5.1M → 76.1M 구간에서 68.5 / 69.8 / 70.1 / 66.4 / 64.9로 꺾인다 [lora Table 15]. 논문 자신의 관찰도 같다: "prefix tuning is difficult to optimize and ... its performance changes non-monotonically in trainable parameters" [lora §3]. **PEFT 계열은 저마다 다른 예산을 소모하며, adapter는 depth를, prefix 계열은 sequence를 판다.** LoRA가 파는 것이 무엇인지는 §03.5에서 확정한다.

품질에서도 adapter가 일방적으로 지는 것은 아니다. E2E NLG에서 GPT-2 Large의 CIDEr는 Adapter$_L$(0.88M) 2.49가 LoRA(0.77M) 2.47보다 높고 [lora Table 3], WebNLG GPT-2 Large의 BLEU-All은 Adapter$_L$(23M) 57.7이 LoRA(0.77M) 57.0보다 높다 [lora Table 14]. 즉 adapter가 밀리는 축은 품질이 아니라 **merge 가능성**이다.

> **[평가] 다만 이 비교표들의 baseline 상당수는 재실행된 값이 아니다.** [lora Table 2, lora Table 3]의 캡션은 "* indicates numbers published in prior works"라고 밝히고, RoBERTa base/large의 full fine-tuning, DeBERTa XXL full fine-tuning, GPT-2 M/L full fine-tuning, FT$_{\mathrm{Top2}}$, PreLayer가 모두 그 표시가 붙은 행이다. 즉 이 장이 인용하는 "full fine-tuning 대비" 격차의 상당수는 학습 프로토콜이 LoRA 행과 같다는 보장이 없는 인용 수치와의 비교다. 이 책은 그 격차를 순서 정보로만 쓰고 소수점 차이를 논거로 쓰지 않는다.

mergeability가 1등 요구사항으로 올라온 것은 이 관찰 때문이며, 다음 절이 그 요구사항을 만족시키는 구성을 다룬다.

---

## 03.3 LoRA — 변화량에 rank를 걸다

LoRA가 상속한 질문은 명시적이다. 선행 연구(Li et al. 2018a, arXiv:1804.08838; Aghajanyan et al. 2020, arXiv:2012.13255)는 과매개화된 학습 모델이 낮은 intrinsic dimension 위에 놓인다는 것을 objective landscape 수준에서 보였다. LoRA가 받아 가는 열린 질문은 **변화량 자체**로 옮긴 형태다: "We hypothesize that the change in weights during model adaptation also has a low 'intrinsic rank'" [lora §1, lora §4.1].

> **정의.** **LoRA**(Low-Rank Adaptation)는 slow weights의 증분 $\Delta\Theta$를 두 저차원 행렬의 곱으로 제약하고 그 두 행렬만 학습하는 (U-$\Theta$)의 부분공간 버전이다. **rank $r$**은 그 곱의 내부 폭이며, 이 방법의 유일한 용량 knob이다.

$$
\Delta\Theta^{(m)} = B^{(m)} A^{(m)},
\qquad
B^{(m)} \in \mathbb{R}^{d_{\mathrm{out}} \times r},
\quad
A^{(m)} \in \mathbb{R}^{r \times d_{\mathrm{in}}},
\quad
r \ll \min(d_{\mathrm{out}}, d_{\mathrm{in}})
\tag{3-2}
$$

여기서 $m$은 LoRA를 적용한 개별 weight 행렬의 인덱스다 [lora §4.1]. 층 출력은 두 경로의 합이 된다.

$$
y \;=\; \Theta_0^{(m)} x \;+\; \frac{\alpha_{\mathrm{LoRA}}}{r}\, B^{(m)} A^{(m)} x
\tag{3-3}
$$

이 식이 말하는 것은 "얼린 경로 하나와 저랭크 경로 하나가 **나란히** 붙고 출력에서 더해진다"이다. 나란히 붙는다는 것이 §03.2와의 결정적 차이다 — depth가 늘지 않으므로 순차 처리가 강제되지 않고, 두 항이 같은 입력 $x$에 대한 선형사상이므로 배포 전에 하나로 합칠 수 있다.

> **[해설]** 식 (3-3)의 정확한 출처를 밝혀 둔다. [lora Eq. 3]은 scaling **없이** $h = W_0x + BAx$로 쓰였고, $\alpha_{\mathrm{LoRA}}/r$ 배율은 그 식 직후 산문에만 나온다("We then scale $\Delta Wx$ by $\frac{\alpha}{r}$", [lora §4.1]). 실제 구현은 둘을 합친 것이므로 이 책은 합친 형태를 표준으로 삼는다. $\alpha_{\mathrm{LoRA}}$는 8·16·32 같은 상수이며, 이 책의 예약 기호 $\alpha_t$(retention gate, 남기는 비율 $\in [0,1]$)와 **아무 관계가 없다**. 시간 첨자가 없다는 것으로 구분한다.

초기화는 방법의 성질을 결정한다. $A^{(m)}$은 Gaussian, $B^{(m)}$은 0이므로 학습 시작 시점에 $\Delta\Theta = 0$이다 [lora §4.1, lora Fig. 1]. 즉 **쓰기가 항등원에서 출발한다** — 쓰기 전 모델이 정확히 보존된다는 이 성질은 Θ-경로에서 반복 등장하며, ch20의 SEAL과 ch21의 CMS expert가 같은 구성을 승계한다.

학습 규칙은 (U-$\Theta$)를 $(A, B)$로 제한한 것이다.

$$
(A, B)_{s+1} \;=\; (A, B)_{s} \;-\; \eta_\Theta \nabla_{(A,B)}\, \mathcal{L}\!\left(\mathcal{D};\ \Theta_0 + \tfrac{\alpha_{\mathrm{LoRA}}}{r} B_s A_s\right),
\qquad \Theta_0\ \text{frozen}
\tag{3-4}
$$

표준형 (U-$\Theta$)와의 차이는 세 가지이고, 셋 다 이 책의 논지에 직접 관계한다.

1. **부분공간 제약.** gradient를 $\Theta$ 전체가 아니라 $(A, B)$에 대해 취하므로 $\Theta_s - \Theta_0$가 모든 step $s$에서 rank $\le r$ manifold에 갇힌다 [lora §4.1].
2. **$B_s$ knob이 없다.** 표준형의 $\mathcal{R}_k = \mathrm{gen}(\mathcal{H}_k; B_s)$에서 $\mathrm{gen}(\cdot)$이 항등이다. 학습 집합은 사람이 만든 고정 dataset(GLUE, WikiSQL, SAMSum, E2E, DART, WebNLG)이고 [lora §2, lora App. C], replay도 합성 데이터도 없다. **sleep 예산이라는 축 자체가 존재하지 않는다.**
3. **첨자가 라운드가 아니다.** 식 (3-4)의 $s$는 한 번의 adaptation run 안의 평범한 SGD step이며, sleep 라운드 첨자 $k$가 **아니다**(GPT-3은 2 epoch, GPT-2는 5 epoch [lora App. D]). 이 구분이 이 장의 논지다 — step을 아무리 많이 돌려도 그것은 라운드 하나 안의 일이고, 라운드가 반복되지 않으므로 누적 열화 문제가 제기되지 않는다. 라운드 첨자 $k$가 실제로 1보다 커지는 것은 ch20·ch21에 가서다.

품질은 GPT-3 175B에서 full fine-tuning과 사실상 같다. LoRA 4.7M이 WikiSQL 73.4 / MNLI-m 91.7 / SAMSum 53.8·29.8·45.9, full fine-tuning 175,255.8M이 73.8 / 89.5 / 52.0·28.0·44.5다 [lora Table 4]. 학습 파라미터가 $2.7\times10^{-5}$배(4.7M / 175,255.8M ≈ 1/37,000, 이 책의 산술)인데 MNLI-m과 SAMSum에서는 오히려 앞선다.

> **[평가] 그러나 완승은 아니며, 가장 인상적인 격차 하나는 공정한 비교가 아니다.** 다섯 가지를 병기한다. (i) full fine-tuning에 지는 지점이 남는다 — QQP에서 LoRA는 RoBERTa 두 크기 모두에서 낮다(base 90.8 대 91.9, large 91.6 대 92.2) [lora Table 2]. 저데이터 레짐 MNLI-1k에서도 LoRA 85.6이 Fine-Tune 85.8보다 낮다 [lora Table 16]. (ii) "다른 방법과 직교적으로 조합된다"는 주장의 절반이 음의 결과다 — LoRA+PrefixLayer(52.8M)는 LoRA 단독(37.7M)보다 WikiSQL 72.9 대 73.8, MNLI 90.2 대 91.6으로 **더 나쁘다** [lora Table 15]. (iii) 가장 널리 인용되는 격차는 초기화 이점을 포함한다 — GLUE의 MRPC·RTE·STS-B에서 LoRA 모듈은 "our best MNLI checkpoint"로 warm-start된다 [lora App. D.1]. 그 이점을 제거한 † setup에서 RoBERTa large의 RTE는 87.4에서 **85.2로 떨어져** full fine-tuning의 86.6보다 낮아진다 [lora Table 2]. (iv) GPT-3의 최고 성적도 순수 LoRA가 아니다 — WikiSQL 최고치는 LoRA+PrefixEmbed($r^{(q)}=r^{(v)}=64$, 302.1M)의 76.2이고, LoRA 단독 최고치 74.0보다 2.2점 높다 [lora Table 15]. (v) 그리고 이 절의 GPT-3 수치는 전부 validation accuracy다 — 별도 test split 평가가 없고 "We report the best validation performance from each run"이며 [lora App. D.4], 표준편차도 항목별이 아니라 task당 대표값 하나뿐이다 [lora §5.5]. 이 책은 LoRA를 "full fine-tuning과 대등한 write primitive"로 취급하되, "언제나 우월하다"로는 취급하지 않는다.

쓰기 시점 비용은 §03.1에서 예고한 대로 움직인다. GPT-3 175B에서 VRAM은 1.2TB → 350GB, throughput은 V100당 32.5 → 43.1 tokens/s(논문 표현으로 25% speedup)다 [lora §4.2, lora §4.2 각주 5]. 절감의 출처는 FLOPs가 아니라 optimizer state와 gradient를 대부분 들고 있지 않아도 된다는 것이다 [lora §1] — 즉 backward의 계산량은 거의 그대로이고 **메모리만** 줄어든다. 이 구분은 Θ-경로의 sleep 예산을 회계할 때 반복해서 필요하다.

> **[평가]** 논문의 절차 서술 하나가 부록과 어긋난다. [lora §4.1]은 "we simply set $\alpha$ to the first $r$ we try and do not tune it"이라고 선언하지만, 보고된 값은 RoBERTa base $r=8,\ \alpha=8$(일치), RoBERTa large $r=8,\ \alpha=16$(불일치), DeBERTa $r=8,\ \alpha=8$(일치), GPT-2 $r=4,\ \alpha=32$(불일치)다 [lora Table 9, lora Table 10, lora Table 11]. 네 설정 중 둘이 선언한 절차를 따르지 않는다. $\alpha_{\mathrm{LoRA}}$를 "튜닝하지 않아도 되는 상수"로 소개하는 2차 문헌이 많으나, 원문의 근거는 그만큼 약하다.

---

## 03.4 어디에 걸 것인가, $r$은 얼마인가 — 그리고 논문이 자기 표와 어긋나는 자리

식 (3-2)는 두 개의 자유도를 남긴다. **어느 weight 행렬에 걸 것인가**(집합 $\mathcal{S}$)와 **$r$을 얼마로 둘 것인가**다. 이 둘은 품질 knob이면서 동시에 §03.6의 크기 식에 그대로 들어가는 **바이트 knob**이므로, systems 독자에게는 하나를 고르면 다른 하나가 결정되는 관계다.

논문은 학습 파라미터를 18M으로 고정한 채 $\mathcal{S}$만 바꾼다. GPT-3 175B, 96 layers [lora Table 5, lora §7.1]:

표 3-2. 적용 행렬 집합에 따른 성능 (GPT-3 175B, 학습 파라미터 18M 고정) [lora Table 5]

| $\mathcal{S}$ | $r$ | WikiSQL (±0.5) | MultiNLI (±0.1) |
|---|---|---|---|
| $\{\Theta^{(q)}\}$ | 8 | 70.4 | 91.0 |
| $\{\Theta^{(k)}\}$ | 8 | 70.0 | 90.8 |
| $\{\Theta^{(v)}\}$ | 8 | 73.0 | 91.0 |
| $\{\Theta^{(o)}\}$ | 8 | 73.2 | 91.3 |
| $\{\Theta^{(q)}, \Theta^{(k)}\}$ | 4 | 71.4 | 91.3 |
| $\{\Theta^{(q)}, \Theta^{(v)}\}$ | 4 | 73.7 | 91.3 |
| $\{\Theta^{(q)}, \Theta^{(k)}, \Theta^{(v)}, \Theta^{(o)}\}$ | 2 | 73.7 | **91.7** |

> **[평가] 이 표의 결론은 논문 본문의 결론과 다르다.** [lora §7.1]은 "adapting both $W_q$ and $W_v$ yields the best result" / "Adapting both $W_q$ and $W_v$ gives the best performance overall"이라고 쓴다. 그러나 같은 표에서 $\{\Theta^{(q)}, \Theta^{(k)}, \Theta^{(v)}, \Theta^{(o)}\}$($r=2$)는 WikiSQL 73.7로 **동률**이고 MultiNLI 91.7로 $\{\Theta^{(q)},\Theta^{(v)}\}$의 91.3보다 **높다**. 표가 실제로 보여주는 것은 "$\{\Theta^{(q)},\Theta^{(v)}\}$가 최선"이 아니라 "**같은 파라미터 예산이면 rank를 낮추더라도 더 많은 행렬에 나누어 거는 편이 열등하지 않다**"이다. 이 책은 후자를 채택한다. 2차 문헌이 널리 인용하는 "q와 v에 걸어라"라는 실무 규칙은 이 표에서 유도되지 않는다.

$r$ 자체의 스윕은 더 조심해서 읽어야 한다. GPT-3 $\{\Theta^{(q)},\Theta^{(v)}\}$의 WikiSQL은 $r=1,2,4,8,64$에서 각각 73.4 / 73.3 / 73.7 / 73.8 / 73.5다 [lora Table 6]. 다섯 점의 전체 폭은 0.5이고, 같은 표의 캡션이 밝힌 변동 폭은 ±0.5%다 [lora Table 6].

> **[평가] rank 차이를 이 데이터로는 해상할 수 없다.** 스윕 전체의 폭이 논문 자신이 선언한 변동 폭과 같은 크기이므로, 이 표는 rank들 사이의 **순서조차** 결정하지 못한다. "$r=1$이면 충분하다"는 널리 인용되는 결론은 자기 오차막대 안에 근거를 두고 있다. 이 책은 이 스윕을 "작은 $r$이 큰 $r$에 크게 밀리지 않았다"는 부재 서술로만 쓰고, 특정 $r$의 우열 논거로 쓰지 않는다.

작은 모델에서는 결론이 뒤집힌다. GPT-2 Medium의 E2E 스윕에서 저자 스스로 "Unlike on GPT-3 where $r=1$ suffices for many tasks, here the performance peaks at $r=16$ for validation loss and $r=4$ for BLEU"라고 쓴다 [lora Table 18 캡션, lora App. H.2]. 게다가 같은 캡션이 "some of our hyperparameters are tuned on $r=4$ ... and thus might not be optimal for other choices of $r$"이라고 인정하므로, rank 축을 단독 변수로 읽을 수도 없다. 논문 자신의 정리가 정확하다: "the relationship between model size and the optimal rank for adaptation is still an open question" [lora App. H.2].

$\mathcal{S}$의 범위에도 선언된 한계가 있다. 이 논문의 모든 "어디에 걸 것인가" 결론은 self-attention 안으로 한정된다 — "We limit our study to only adapting the attention weights for downstream tasks and freeze the MLP modules ... both for simplicity and parameter-efficiency" [lora §4.2]. MLP는 검증되어 배제된 것이 아니라 편의로 배제됐다. 사실 지식이 MLP에 저장된다고 보는 모델 편집 계열(ROME·MEMIT)과 정면으로 만나는 지점이 여기인데, 이 논문에는 그 축의 증거가 없다. 두 계열의 충돌은 ch19가 다룬다.

그렇다면 그 delta는 실제로 무엇을 쓰는가. 논문의 답은 "새 방향을 넣는 것이 아니라 이미 있는 방향을 키운다"이다. GPT-3의 48번째 층에서 $\lVert \Theta_0^{(q)} \rVert_F = 61.95$이고, $r=4$일 때 $\lVert \Delta\Theta^{(q)} \rVert_F = 6.91$인 반면 $\Delta\Theta^{(q)}$의 top-$r$ 부분공간으로 사영한 $\Theta_0^{(q)}$의 norm은 0.32에 불과하다 — 논문은 이 비를 feature amplification factor 21.5로 부른다 [lora Table 7, lora §7.3]. 즉 delta는 base가 이미 갖고 있으나 강조하지 않던 방향을 골라 증폭한다는 것이 논문의 해석이다.

> **[평가] 이 서사는 rank에 따라 뒤집히고, 근거는 층 하나다.** [lora §7.3]은 "the amplification factor is rather huge: 21.5"를 강조하지만, [lora App. H.4]는 $r=64$에서 그 값이 "약 2"에 불과하다고 인정한다(3.57 / 1.90). 21.5는 $r=4$ 설정의 산물이다. 게다가 Table 7과 Fig. 3·4의 모든 수치는 96개 층 중 **48번째 층 하나**에서 나온다 — "We only look at the 48th layer (out of 96) due to space constraint" [lora §7.2]. 부록 H.1이 층 1/32/64/96으로 확장하지만 여전히 96개 중 4개이고, Table 7의 norm 수치는 층 48 외에는 어디에도 없다. 따라서 "Θ-경로의 쓰기는 추가가 아니라 증폭이다"라는 명제는 이 논문에서 **가설로만** 성립한다. 이 질문은 ch09(용량)와 ch19(모델 편집과의 대조)가 받는다.

> **[해설]** rank가 늘어날 때 실제로 무엇이 늘어나는지에 대한 **용량 상한** 논의는 이 장의 소관이 아니다(→ ch09). 이 장이 확정하는 것은 그보다 앞선 것 하나다: $r$과 $\mathcal{S}$는 품질 hyperparameter이기 이전에 **아티팩트 크기를 결정하는 두 변수**이며, 식 (3-6)에서 정확히 선형으로 들어간다.

---

## 03.5 merge와 서빙 형태 — $L_w = 0$은 무엇과 맞바꾼 것인가

> **정의.** **merge**는 학습이 끝난 뒤 delta를 base에 흡수해 하나의 weight 행렬로 만드는, 배포 시점의 산술 연산이다. gradient step이 아니다.

$$
\adjustbox{max width=\linewidth}{$\displaystyle
\Theta^{(m)} \leftarrow \Theta_0^{(m)} + \tfrac{\alpha_{\mathrm{LoRA}}}{r} B^{(m)} A^{(m)},
\qquad
\text{task switch: } \Theta_0^{(m)} = \Theta^{(m)} - \tfrac{\alpha_{\mathrm{LoRA}}}{r} B^{(m)} A^{(m)},\ \ \Theta'^{(m)} = \Theta_0^{(m)} + \tfrac{\alpha_{\mathrm{LoRA}}}{r} B'^{(m)} A'^{(m)}
$}
\tag{3-5}
$$

이 한 줄이 이 장에서 가장 systems적인 대상이다 [lora §4.1]. merge하면 GEMM shape가 base 모델과 글자 그대로 같아진다 — 층이 늘지도, sequence가 줄지도 않는다. 논문이 "we do not introduce any additional latency during inference compared to a fine-tuned model **by construction**"이라고 쓰는 근거가 이것이다 [lora §4.1]. 비용 4종의 언어로: $L_w = 0$이다.

> **정의.** 같은 delta를 서빙에 올리는 방식은 두 가지이며, 이 책은 이를 **서빙 형태**라 부른다. **merge된 형태**는 식 (3-5)를 배포 전에 실행해 $\Theta_0 + \Delta\Theta$ 하나만 GPU에 두는 것이고, **분리된 delta 형태**는 $\Theta_0$를 공유한 채 $(A, B)$를 요청마다 별도 경로로 적용하는 것이다.

두 형태의 차이는 latency가 아니라 **누가 무엇을 공유하는가**다.

merge된 형태에서 한 forward pass는 **정확히 하나의 task**에만 유효하다. 논문 자신이 인정한다: "it is not straightforward to batch inputs to different tasks with different $A$ and $B$ in a single forward pass, if one chooses to absorb $A$ and $B$ into $W$ to eliminate additional inference latency" [lora §4.2 Limitations]. Rosetta 사전의 "batch로 weight 공유 ↔ Θ-경로에서 깨짐" 행이 가리키는 지점이 바로 여기다 — **shared-weight batching은 서빙의 대전제이고, merge는 그것을 대가로 $L_w=0$을 산다.**

분리된 delta 형태는 그 대전제를 지킨다. $\Theta_0$는 배치 전체가 공유하고, 요청마다 다른 $(A,B)$만 별도 경로로 태우면 된다. 대신 층마다 GEMM 두 개가 추가된다.

> **[해설] 이 책의 산술.** 적용 행렬 하나당 base 경로는 토큰당 약 $d^2$ MAC이고, LoRA 경로는 $d \times r$과 $r \times d$ 두 번, 즉 $2dr$ MAC이다. 비율은 $2r/d$다. GPT-3($d = 12{,}288$, $r = 4$)이면 $8/12{,}288 \approx 0.065\%$, $d=4096$·$r=16$이면 $32/4096 = 0.78\%$다. **FLOPs 관점에서는 무시할 수준이다.** 그러나 batch가 작은 decode의 병목은 FLOPs가 아니라 kernel launch와 메모리 왕복이며, 표 3-1이 adapter에서 보인 +20.7~30.3%가 경고하는 바가 정확히 그것이다. LoRA의 두 GEMM은 **병렬**이라 depth를 늘리지 않으므로 adapter만큼 나쁠 이유는 없다. 다만 이 산술은 이 책의 것이며 논문의 것이 아니다.

> **[평가]** 그리고 논문에는 그 숫자가 하나도 없다. 논문은 "many customized models that can be swapped in and out on the fly"를 팔면서 [lora §4.2] 동시에 merge하면 배칭이 안 된다고 인정하고, 분리된 delta 형태의 latency는 "가능하다"고만 쓰고 **측정하지 않는다** [lora §4.2 Limitations]. 나아가 latency 근거 전체의 규모가 주장의 규모와 다르다 — 표 3-1의 모든 측정은 GPT-2 medium 단일 GPU이고 [lora Table 1, lora App. B], 정작 배포 논거의 대상인 GPT-3 175B에서는 어떤 latency도 측정되지 않았다. **품질의 상한과 latency 근거의 상한이 서로 다른 모델·다른 하드웨어에 있다.**

이 책의 실험 [X2]는 세 경로의 서빙 구조를 네 줄로 정리한다(Θ-경로는 delta 형태와 직접 편집으로 갈린다): E-경로는 shared-weight batching이 **유지된다**(상태가 프롬프트로 들어오므로 가중치는 전부 공유), W-경로는 **부분적으로 깨진다**(세션마다 별도 fast-weight 상태가 HBM에 상주), Θ-경로의 LoRA는 **delta 단위로 깨진다**(요청마다 사용자 delta를 로드·머지해야 하며 배치 안의 사용자가 다르면 가중치가 다르다), Θ-경로의 직접 편집은 **완전히 깨진다**(개인화하려면 모델 전체 사본) [X2 Q4]. 마지막 항목이 delta 형태의 존재 이유다 — 직접 편집(→ ch19)은 공유 가중치를 제자리에서 고치므로 애초에 아티팩트가 생기지 않고, 그래서 사용자별 개인화가 구조적으로 불가능하다.

트래픽의 성격도 경로마다 다르다. Θ-경로는 **요청당 delta 로드(수 MB) + 갱신은 sleep 라운드에만**이므로 읽기 편향이며 사용자별이다 [X2 Q4]. W-경로가 토큰마다 상태를 read-modify-write하는 쓰기 편향인 것과 대조된다(→ NM ch10).

> **[평가]** 서빙 fleet 관점에서 이 논문이 비워 둔 칸은 정확히 네 개다. (a) 여러 task를 동시에 배칭할 때의 비용 모델, (b) delta swap의 지연과 정합성, (c) 여러 delta를 동시에 얹었을 때의 간섭, (d) 반복 write의 열화 $\rho$. 네 항목이 이 논문에 없다는 것은 사실 서술이고, **이 넷이 Θ-경로를 sleep-time write로 쓰려는 순간 전부 필수 항목이 된다**는 것은 이 책의 판단이다. (c)는 ch09가 다루는 후속 연구가 처음으로 측정하고, (d)는 ch20·ch21까지 가서야 부분적으로 채워진다. (a)와 (b)는 이 corpus 안에서 끝내 채워지지 않으며, 그 사실 자체를 Part III(ch29)가 결론의 재료로 쓴다.

---

## 03.6 delta 아티팩트의 크기, 그리고 dtype이 정하는 정보 밀도

> **정의.** **delta 아티팩트**는 (U-$\Theta$)의 산출물 중 $\Theta_0$에 더해지는 부분만 따로 저장한 파일이다. LoRA의 경우 그 크기는 정확히 다음 식으로 주어진다.

$$
|\Delta\Theta| \;=\; 2\, N_{\mathrm{mod}}\, d\, r,
\qquad
N_{\mathrm{mod}} \;=\; L_{\mathrm{layer}} \times |\mathcal{S}|
\tag{3-6}
$$

$N_{\mathrm{mod}}$는 LoRA를 적용한 weight 행렬의 총 개수이고, 계수 2는 $A$와 $B$ 두 장에서 온다 [lora §5.1]. 이 식이 $d$·$L_{\mathrm{layer}}$·$r$·$\mathcal{S}$ 넷에만 의존하고 모델 파라미터 총수에는 의존하지 않는다는 점이 중요하다 — **delta의 크기는 모델의 크기가 아니라 모델의 폭과 깊이가 정한다.** 바이트로 옮기면 $C_{\mathrm{cap}} = |\Delta\Theta| \times (\text{bytes/elem})$이며, dtype을 명시하지 않은 delta 크기 진술은 무의미하다.

논문의 실측 값이 이 식의 기준점이다. GPT-3 175B에 $r=4$로 $\mathcal{S}=\{\Theta^{(q)}, \Theta^{(v)}\}$를 적용하면 checkpoint가 350GB → 35MB, 약 10,000배 줄어든다 [lora §4.2]. task 100개를 얹어도 350GB + 35MB × 100 ≈ 354GB이고(이 책의 예시 계산), 독립 fine-tune이면 약 35TB다 [lora §4.2 각주 4]. **Θ-경로의 per-task 상태량은 base weights의 $10^{-4}$ 수준이다** — 이것이 이 장이 Part III에 넘기는 첫 번째 숫자다.

여기서 자주 혼동되는 두 개념을 분리한다.

> **정의.** **QLoRA**는 base $\Theta_0$를 4-bit로 양자화해 얼려 두고 그 위에 더 높은 정밀도의 LoRA delta를 학습하는 구성이다(Dettmers et al. 2023, arXiv:2305.14314). **양자화되는 것은 $\Theta_0$이지 $\Delta\Theta$가 아니다.** 따라서 QLoRA가 줄이는 것은 §03.1에서 본 쓰기 시점의 GPU 메모리이며, delta 아티팩트의 크기는 바꾸지 않는다.

<!-- TODO-VERIFY: QLoRA의 base 양자화 형식(NF4)과 double quantization·paged optimizer의 정확한 절 위치. 이 corpus에 원문이 vendored 되어 있지 않다. 확인 방법: papers/stc/에 2305.14314를 반입한 뒤 검색어 "4-bit NormalFloat", "double quantization". 수치는 확인 전까지 본문에 쓰지 않는다. -->

> **정의.** **양자화된 delta**는 그와 다른 축이다 — $\Delta\Theta$ **자체**를 낮은 dtype으로 저장·전송·적재하는 것이며, 바꾸는 대상은 $B_s$가 아니라 $C_{\mathrm{cap}}$이다.

이 구분이 왜 중요한지는 저장 바이트당 정보량을 계산해 보면 드러난다. 이 책의 실험 [X2 Q2]는 $d=4096$, $L_{\mathrm{layer}}=32$, $\mathcal{S}=\{\Theta^{(q)}, \Theta^{(v)}\}$, $r=16$인 dense-8B 앵커(파라미터 8,388,608개)에 대해 다음을 회계한다.

표 3-3. delta의 dtype별 저장 바이트당 정보량 [X2 Q2]. 분자는 [lm-memorization-capacity Fig. 6 캡션]의 3.64 bits/param을 파라미터 수에 곱한 값이고, 이 책은 이것을 회계용 참고값으로만 쓴다(BPP-EXTRAPOLATED — 무작위 균등 문자열을 from-scratch 학습시킨 소형 GPT에서 잰 값이며 LoRA delta 적용은 외삽이다). **원 논문은 이 값을 하한으로 못 박는다** — "we are only ever measuring a lower bound on model capacity" [lm-memorization-capacity §3.2]. 따라서 이 표의 어느 칸도 용량 상한이 아니며, 실제 밀도는 이보다 높을 수 있다. 이 값의 지위와 용량 상한 논의의 정본은 ch09가 소유한다.

| dtype | 저장 바이트 | 저장 MB | 저장 바이트당 bits |
|---|---|---|---|
| bf16 | 16,777,216 | 16.78 | 1.82 |
| int8 | 8,388,608 | 8.39 | 3.64 |
| int4 | 4,194,304 | 4.19 | 7.28 |

같은 앵커에서 E-경로의 텍스트 상태(14K 토큰, 3:1 압축 가정)는 2.67 bits/byte다. 두 경로의 비는 dtype에 따라 bf16 1.47, int8 0.73, int4 0.37이다 [X2 Q2]. 즉 **bf16에서는 텍스트가 바이트당 정보 밀도에서 앞서고, int8에서 순서가 뒤집히며, int4에서는 Θ가 텍스트의 2.7배가 된다.** 순서가 뒤집히는 지점이 bf16과 int8 사이에 있다는 것 — 이것이 이 절의 결론이다.

> **[해설]** 이 역전의 기제는 알고리즘이 아니라 표현이다. 파라미터 하나에 담기는 정보량의 측정된 **하한**이 3.64 bits인데 [lm-memorization-capacity Fig. 6 캡션, lm-memorization-capacity §3.2] bf16은 그것을 16 bits 자리에 넣는다. 낭비되는 것은 용량이 아니라 **저장 폭**이다. 그러므로 "Θ-경로는 바이트 비효율이다"라는 흔한 인상은 dtype을 고정했을 때만 참이며, 알고리즘의 성질이 아니다 [X2 F3].

사용자 수로 곱하면 이 선택이 어디에 나타나는지 보인다. 같은 8B 앵커에서 사용자 $10^6$명분의 웜 스토리지는 $r=16$·bf16에서 16.777 TB, int4로 내리면 4.194 TB, $r=64$·bf16이면 67.109 TB다 [X2 Q3]. 절대치는 회계이지 측정이 아니므로 자릿수와 방향으로만 읽어야 하지만, 방향은 분명하다 — **dtype 하나가 웜 계층 규모를 4배 움직인다.**

> **[평가]** 그런데 corpus의 어느 논문도 delta 양자화를 논의하지 않는다 [X2 F3]. LoRA 원문에는 delta의 load 시간도, 대역폭도, roofline도 한 번 나오지 않는다. Θ-경로를 실제 서빙에 올리려면 delta 양자화는 선택 사항이 아니라 전제인데, 그 전제를 다룬 문헌이 이 corpus에 없다. 이 공백은 Part III(ch26, ch29)가 직접 메운다.

---

## (state, update, cost) 정리

이 장의 개념을 세 층 프레임에 배치한다. 모든 행이 $\Theta$ 층이라는 것 자체가 이 장의 요약이다 — **fine-tuning과 PEFT는 $W$도 $E$도 만들지 않는다.**

표 3-4. 이 장의 개념과 세 층 프레임

| 개념 | 상태(어느 층) | 갱신 규칙 | 시간척도 | 비용 4종에서의 자리 |
|---|---|---|---|---|
| full fine-tuning | $\Theta$ 전체 | (U-$\Theta$), 제약 없음 | adaptation run 1회 | $C_{\mathrm{cap}} = \lvert \Theta_0 \rvert$ (GPT-3에서 350GB) |
| adapter | $\Theta$의 삽입 부분 | (U-$\Theta$), 삽입 파라미터만 | adaptation run 1회 | $L_w > 0$ — merge 불가, batch 1에서 +20.7~30.3% |
| LoRA (rank $r$) | $\Theta$의 rank-$r$ 부분공간 | (U-$\Theta$), $\nabla$가 $(A,B)$에만 | adaptation run 1회 | $C_{\mathrm{cap}} = 2N_{\mathrm{mod}}dr \times$ bytes/elem |
| merge | $\Theta$ (gradient 아님) | 규칙 밖 — 배포 시점의 산술 | 배포 1회 | $L_w \to 0$, shared-weight batching을 깨뜨림 |
| 분리된 delta 형태 | $\Theta_0$ 공유 + per-request $(A,B)$ | 갱신 아님 — 적용 형태 | 요청마다 | $L_w$ 논문에 없음. FLOPs 비 $2r/d$(이 책의 산술) |
| QLoRA | $\Theta_0$의 저장 표현 | 갱신 아님 — 양자화 | 쓰기 시점 | $B_s$의 메모리 항 |
| 양자화된 delta | $\Delta\Theta$의 저장 표현 | 갱신 아님 — 양자화 | 배포·보관 시점 | $C_{\mathrm{cap}}$의 정보 밀도(bf16 1.82 → int4 7.28 bits/byte) |
| $W$ (fast weights) | — | **없다** | — | 이 장의 어떤 방법도 test-time에 상태를 움직이지 않는다 |
| $E$ (external store) | — | **없다** | — | LoRA 원문에 retrieval·vector store 언급이 없다 |

비용 4종을 LoRA 기준으로 채우면 다음과 같다. **빈칸을 추정으로 메우지 않는다.**

표 3-5. LoRA의 비용 4종

| 기호 | 값 | 출처 |
|---|---|---|
| $B_s$ | **논문에 없음.** sleep 예산 개념이 없고 $\mathrm{gen}(\cdot)$이 항등이다. 보고되는 것은 학습 VRAM 1.2TB → 350GB와 throughput 32.5 → 43.1 tokens/s per V100뿐이며 총 FLOPs·토큰 수·wall-clock은 전부 없다 | [lora §4.2, lora §4.2 각주 5] |
| $L_w$ | merge된 형태에서 **0** ("by construction"). 분리된 delta 형태는 **논문에 없음** | [lora §4.1, lora §4.2 Limitations] |
| $C_{\mathrm{cap}}$ | $2N_{\mathrm{mod}}dr$ 파라미터. GPT-3 $r=4$, $\{\Theta^{(q)},\Theta^{(v)}\}$에서 350GB → 35MB (약 10,000배) | [lora §5.1, lora §4.2, lora §7.1] |
| $\rho$ | **논문에 없음.** 순차 적응·delta 누적·반복 write·forgetting 측정이 하나도 없다 | (부재. 원문에 forget·continual·sequential·catastrophic 어휘가 등장하지 않음) |

> **[평가]** 마지막 줄이 이 장이 Part II에 넘기는 가장 중요한 공백이다. LoRA는 한 task에 한 번 적응하고 끝나며, 두 개의 $\Delta\Theta$를 누적하거나 같은 모델에 반복해서 쓰는 실험이 없다. 가장 근접한 서술인 task switch(식 (3-5))조차 열화가 아니라 **완전 되돌리기**를 전제한다. 그러므로 "LoRA를 sleep-time write primitive로 쓸 수 있다"는 이 책의 명제는 **이 논문으로는 warrant되지 않는다.** 그것을 warrant할 증거는 ch09(용량)·ch20(SEAL)·ch21(반복 sleep 라운드)에서 따로 조달해야 하며, 조달되지 않으면 조달되지 않았다고 쓴다.

---

## Worked micro-example — delta 아티팩트의 크기를 손으로 계산한다

식 (3-6)을 실제로 돌린다. 모든 단계는 암산 가능하며, 계산기가 필요한 곳은 없다.

**1단계 — 네 값을 고정한다.** 앵커는 dense-8B다: $d = 4096$, $L_{\mathrm{layer}} = 32$ [X2 앵커]. 적용 집합은 표 3-2의 관행을 따라 $\mathcal{S} = \{\Theta^{(q)}, \Theta^{(v)}\}$, rank는 $r = 16$, 저장 dtype은 bf16으로 둔다.

**2단계 — 적용 행렬 개수.** $N_{\mathrm{mod}} = L_{\mathrm{layer}} \times |\mathcal{S}| = 32 \times 2 = 64$. 층마다 두 장씩, 32개 층이므로 64장이다.

**3단계 — 파라미터 수.** 식 (3-6)에 넣는다.

$$
|\Delta\Theta| = 2 \times 64 \times 4096 \times 16
$$

암산 경로는 이렇다. $2 \times 64 = 128$. $128 \times 4096 = 524{,}288$(= $2^{19}$). $524{,}288 \times 16 = 8{,}388{,}608$(= $2^{23}$). 즉 **8.39M 파라미터**다. 8B 모델 대비 약 $10^{-3}$이다.

**4단계 — 바이트.** dtype을 곱한다. bf16은 2 bytes/elem이므로 $8{,}388{,}608 \times 2 = 16{,}777{,}216$ bytes = **16.78 MB**. int8이면 8.39 MB, int4면 4.19 MB다 [X2 Q1].

**교차 검증 — 논문의 값이 나오는가.** 같은 식에 GPT-3을 넣는다: $d = 12{,}288$, $L_{\mathrm{layer}} = 96$ [lora §1, lora §7.1], $\mathcal{S} = \{\Theta^{(q)}, \Theta^{(v)}\}$, $r = 4$. $N_{\mathrm{mod}} = 96 \times 2 = 192$. $2 \times 192 = 384$, $384 \times 12{,}288 = 4{,}718{,}592$, $\times 4 = 18{,}874{,}368$ — **약 18.9M**이다. 논문이 같은 설정을 "18M parameters (roughly 35MB if stored in FP16), 96 layers"로 적은 것과 파라미터 수에서 일치한다 [lora §7.1]. 바이트는 $18{,}874{,}368 \times 2 = 37{,}748{,}736$ = 37.7 MB이므로 논문의 "35MB"는 반올림 표기이며, 논문의 "10,000배 감소"(350GB → 35MB, [lora §4.2])도 그 반올림 값을 쓴 것이다. 식 (3-6)이 준 계산값으로 다시 나누면 $350\,\mathrm{GB} / 37.7\,\mathrm{MB} \approx 9{,}300$배다(이 책의 예시 계산). 이 책은 계산값과 원문 표기를 이렇게 병기하고, 어느 쪽도 상대를 지우지 않는다.

**두 knob의 감도.** 3단계로 돌아가 $r$만 16 → 64로 올리면 4배가 되어 67.11 MB, $\mathcal{S}$만 $\{\Theta^{(q)},\Theta^{(v)}\}$ → $\{\Theta^{(q)},\Theta^{(k)},\Theta^{(v)},\Theta^{(o)}\}$로 늘리면 2배가 되어 33.55 MB다 [X2 Q1]. 두 knob은 식 (3-6)에 **곱으로** 들어가므로 둘 다 키우면 8배(134.22 MB)다. 표 3-2가 보인 "같은 예산이면 rank를 낮추고 행렬을 늘려라"는 이 곱셈을 고정한 채 배분만 바꾸는 조작이다.

**정보 밀도.** 8,388,608 파라미터에 3.64 bits/param을 곱하면 30,534,533 bits ≈ 3.82 MB다 — 이 책의 예시 계산이다(BPP-EXTRAPOLATED — 회계용 참고값이며, 3.64는 원 논문이 하한이라고 밝힌 값이므로 이 결과도 하한이다 [lm-memorization-capacity §3.2]. 정본 서술은 ch09) [X2 Q2]. bf16 저장 16.78 MB로 나누면 저장 바이트당 1.82 bits, int4 저장 4.19 MB로 나누면 7.28 bits다. **같은 정보를 4분의 1 바이트에 담는 것**이 delta 양자화가 하는 일의 전부다.

**서빙 산술.** 분리된 delta 형태에서 추가되는 토큰당 MAC은 base 대비 $2r/d = 32/4096 = 1/128 \approx 0.78\%$다(이 책의 산술). **사용자 스케일.** 16.78 MB를 사용자 $10^6$명에 곱하면 웜 스토리지 16.777 TB, int4면 4.194 TB다 [X2 Q3] — 절대치는 회계이므로 자릿수로 읽되, 4배의 차이가 dtype 하나에서 나온다는 순서 관계는 그대로 성립한다.

---

## 요약

- 가중치에 쓰는 것은 문맥에 쓰는 것으로 대체되지 않으며(GPT-3 few-shot 대 fine-tuning이 MNLI-m 40.6 대 89.5 [lora Table 8]), 그 대가로 full fine-tuning은 $\Theta_0$와 크기가 같은 checkpoint를 task마다 남긴다 — GPT-3 175B에서 350GB다 [lora §2, lora §4.2].
- adapter가 밀리는 축은 품질이 아니라 mergeability다. 순차 depth의 비용은 batch가 작을수록 커지며, batch 1 / seq 128에서 +20.7~30.3%다 [lora Table 1].
- LoRA는 (U-$\Theta$)를 rank-$r$ 부분공간으로 제한한 write primitive이고, $B_s$ knob도 라운드 첨자도 갖지 않는다 — sleep 절차가 아니라 쓰기 도구다 [lora §4.1, lora §2].
- [lora §7.1]의 "$\{\Theta^{(q)},\Theta^{(v)}\}$가 최선"이라는 결론은 자기 Table 5와 어긋난다. $\{\Theta^{(q)},\Theta^{(k)},\Theta^{(v)},\Theta^{(o)}\}$($r=2$)가 WikiSQL 73.7로 동률, MultiNLI 91.7로 우위다 [lora Table 5].
- merge는 $L_w=0$을 shared-weight batching과 맞바꾼 것이며, 분리된 delta 형태의 latency는 논문에 측정값이 없다 [lora §4.1, lora §4.2 Limitations].
- delta 아티팩트의 크기는 $2N_{\mathrm{mod}}dr$로 정확히 주어지고, 모델 파라미터 총수가 아니라 폭·깊이·rank·적용 집합이 결정한다 [lora §5.1].
- 저장 바이트당 정보 밀도는 bf16 1.82, int8 3.64, int4 7.28 bits/byte이며, 텍스트(3:1 압축 가정 2.67) 대비 순서가 bf16과 int8 사이에서 뒤집힌다 — 3.64 bits/param 하한 위에 세운 이 책의 회계다 [X2 Q2, lm-memorization-capacity §3.2].
- LoRA의 $\rho$는 논문에 없다. 반복 write·delta 누적·forgetting 측정이 하나도 없으므로 이 논문은 Θ-경로의 지속가능성을 warrant하지 못한다.

---

## 자가 점검 체크리스트

- [ ] full fine-tuning · adapter · LoRA를 "$\Theta$의 어느 부분집합을 어떻게 움직이는가"와 "배포 아티팩트로 무엇을 남기는가" 두 축으로 구분해 설명할 수 있다.
- [ ] $d$, $L_{\mathrm{layer}}$, $r$, $\mathcal{S}$만 주어졌을 때 $|\Delta\Theta|$와 그 바이트를 손으로 계산하고, $r$과 $|\mathcal{S}|$가 곱으로 들어간다는 것을 보일 수 있다.
- [ ] merge된 형태에서 $L_w = 0$이 성립하는 이유와, 그 대가로 무엇이 깨지는지를 GEMM shape 수준에서 설명할 수 있다.
- [ ] QLoRA가 양자화하는 대상과 "양자화된 delta"가 양자화하는 대상이 다르다는 것을, 각각 비용 4종 중 무엇을 바꾸는지로 말할 수 있다.
- [ ] [lora §7.1]의 결론과 [lora Table 5]의 수치가 어긋나는 지점을 지적하고, 표에서 실제로 유도되는 결론을 말할 수 있다.
- [ ] LoRA의 비용 4종에서 "논문에 없음"인 칸이 어디이고 왜 그것이 Θ-경로 판정에 결정적인지 설명할 수 있다.
- [ ] **Rosetta** — "batch로 weight 공유"라는 서빙의 대전제가 Θ-경로에서 깨지는 자리를, merge된 형태와 분리된 delta 형태 각각에 대해 inference 어휘로 옮길 수 있다.

---

## 다음 장으로

이 장은 $\Theta$에 **쓰는 도구**를 확정했다. 그런데 식 (3-4)에서 학습 집합 $\mathcal{D}$는 사람이 만들어 놓은 고정 dataset이었고, 그래서 표준형 (U-$\Theta$)의 $\mathcal{R}_k = \mathrm{gen}(\mathcal{H}_k; B_s)$에서 $\mathrm{gen}(\cdot)$이 항등으로 붕괴했다. sleep 라운드가 성립하려면 이 자리가 비어 있어서는 안 된다 — 유휴 시간에 예산 $B_s$를 써서 **무엇을 학습 신호로 삼을 것인가**가 정해져야 한다.

다음 장이 답할 질문은 그 자리의 가장 오래된 답이다: 모델의 출력을 다른 모델(또는 과거의 자신)의 학습 신호로 쓰는 법, 즉 distillation이다. ch04는 logit·feature·self-distillation과 upward distillation을 (U-$\Theta$)의 $\mathcal{R}_k$ 생성기로 재서술하고, 그것이 ch21의 Knowledge Seeding과 ch20의 self-edit으로 이어지는 경로를 연다. 이 장의 $(A, B, r)$은 그때 "무엇에 쓰는가"가 아니라 "**무엇을** 쓰는가"의 문제로 바뀐다.
