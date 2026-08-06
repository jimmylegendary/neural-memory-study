# ch06. 데이터를 만드는 법 — augmentation, synthetic, 그리고 붕괴

> **이 장의 목표** — 이 장을 마치면 다음을 할 수 있어야 한다.
> 1. 임의의 sleep 라운드 학습 집합 $\mathcal{R}_k$를 augmentation · synthetic data · self-generated data로 **분해**하고, 생성기가 누구인지로 셋을 판별한다.
> 2. $\mathcal{R}_k$의 구성 스케줄을 replace / accumulate / replace-multiple로 **분류**하고, 각 스케줄의 라운드당 열화율 $\rho_k$를 닫힌 형식으로 적는다.
> 3. "model collapse가 일어난다"는 진술을 **네 현상 중 어느 것인지 지정하지 않고는** 쓰지 않는다.
> 4. 자기 생성 $\mathcal{R}_k$로 (U-$\Theta$)를 반복할 때 무엇이 유계로 남고 무엇이 무너지는지 분리하고, 그 경계가 어느 전제 위에 서 있는지 말한다.
>
> **왜 필요한가** — Part II의 $\Theta$-경로 전체가 이 장을 요구한다. SEAL(ch20)은 모델이 자기 자신이 쓴 self-edit을 학습 집합으로 삼아 (U-$\Theta$)를 돌리고 [SEAL §3.1], `Language Models Need Sleep`(ch21)의 Dreaming은 자기 생성 샘플을 sleep 라운드의 $\mathcal{R}_k$로 쓴다. 두 논문 중 어느 쪽도 **그 루프를 $K$번 반복하면 $\Theta$가 어디로 가는가**에 대한 경계를 제시하지 않는다. corpus에서 그 경계를 닫힌 형식으로 가진 논문은 한 편뿐이고, 그 논문은 $\Theta$-경로의 어느 논문도 인용하지 않으며 $\Theta$-경로의 어느 논문도 그것을 인용하지 않는다. 이 장이 그 다리를 놓는다.
>
> **NM과의 관계** — backward pass, optimizer state, inner/outer loop, test-time learning은 전제한다(→ NM ch02). Neural Memory에는 이 장에 대응하는 장이 **없다**. NM은 $W$층만 다뤘고, $W$의 학습 데이터는 test-time 토큰 스트림 그 자체였다 — 만들 필요가 없었다. $\Theta$층으로 올라오는 순간 "무엇으로 학습하는가"가 처음으로 설계 변수가 된다. 이 장이 더하는 것이 정확히 그 변수다.

---

## 06.1 $\mathcal{R}_k$는 어디서 오는가

식 (U-$\Theta$)에는 이 책이 아직 열어보지 않은 상자가 하나 있다. $\mathrm{gen}(\mathcal{H}_k; B_s)$다.

$$
\Theta_{k+1} \;=\; \Theta_k \;-\; \eta_\Theta\, \nabla_\Theta\, \mathcal{L}\big(\mathcal{R}_k;\ \Theta_k\big),
\qquad
\mathcal{R}_k \;=\; \mathrm{gen}\big(\mathcal{H}_k;\ B_s\big)
$$

이 식이 말하는 것은 sleep 라운드가 두 부분으로 나뉜다는 사실이다. 오른쪽 절반($\mathrm{gen}$)은 학습 집합을 **만드는** 절차이고, 왼쪽 절반은 그것으로 $\Theta$를 **움직이는** 절차다. ch03(PEFT)과 ch05(RL)는 왼쪽 절반을 다뤘다. 이 장은 오른쪽 절반만 다룬다.

$\mathrm{gen}$의 산출물은 세 가지 출처를 갖는다. 이 장은 그 셋을 여기서 정의하고, 이후 모든 장은 참조만 한다.

> **data augmentation**은 이미 가진 데이터 항목에 **라벨을 보존하는 변환**을 적용해 항목 수를 늘리는 절차다. 원 항목이 반드시 존재하고, 변환이 라벨을 바꾸지 않는다는 것이 정의의 두 조건이다.

> **synthetic data**는 사람이 쓰지 않고 **모델이 생성한** 학습 데이터다. 원 항목이 없어도 된다.

> **self-generated data**는 synthetic data 중에서 **생성기가 지금 학습받는 모델 자신인** 것이다. 즉 $\mathrm{gen}(\cdot; B_s) = \mathrm{LM}(\cdot\,;\,\Theta_{k-1})$인 경우다.

세 정의는 포함 관계가 아니라 축이 다르다. augmentation은 "원 항목이 있는가"를 묻고, synthetic은 "사람이 썼는가"를 묻고, self-generated는 "생성기가 누구인가"를 묻는다. 실무의 파이프라인은 대개 셋 중 둘 이상에 동시에 걸친다.

표 6-1. $\mathcal{R}_k$의 출처와 그 결과

| 출처 | 생성기 | 원 항목 필요 | 루프가 닫히는가 | $B_s$가 붙는 곳 |
|---|---|---|---|---|
| 사람이 쓴 데이터 | 사람 | — | 아니오 | 없음(수집 비용은 계산 예산이 아니다) |
| data augmentation (고전형) | 결정론적 변환 | 예 | 아니오 | CPU dataloader |
| synthetic data (타 모델) | 다른 모델 $\Theta^{\text{ext}}$ | 아니오 | 아니오 | 타 모델의 decode |
| **self-generated data** | $\Theta_{k-1}$ 자신 | 아니오 | **예** | 자기 모델의 decode |

마지막 행만 루프가 닫힌다. $\Theta_{k-1}$이 만든 데이터로 $\Theta_k$를 만들고, $\Theta_k$가 만든 데이터로 $\Theta_{k+1}$을 만든다. 이 되먹임이 이 장 나머지의 주제이고, Part III의 반증 축 하나가 여기서 나온다.

이 장의 주 논문은 Gerstgrasser 외, *Is Model Collapse Inevitable? Breaking the Curse of Recursion by Accumulating Real and Synthetic Data* (arXiv:2404.01413)이다. 이하 `[MCI]`로 인용한다.

<!-- TODO-VERIFY: 2404.01413의 게재 venue와 peer-review 상태. vendored 텍스트에는 arXiv 스탬프만 있다. 확인 방법: arXiv abstract 페이지의 comments 필드, 또는 제목 "Is Model Collapse Inevitable" 로 학회 proceedings 검색 -->

> **[해설]** systems 독자에게 이 절의 실질은 워크로드 하나가 새로 생긴다는 것이다. $\mathrm{gen}$이 self-generated이면 sleep 라운드의 앞 절반은 **대규모 batch decode job**이다 — 가중치가 HBM에 상주하고, 프롬프트가 스트리밍으로 들어오고, 출력이 스토리지로 나가는, 독자가 매일 보는 그 작업이다. 실제로 [MCI]는 데이터 생성을 vLLM으로 구현했다고 밝힌다 [MCI App. C]. 그것이 그 논문이 독자의 세계와 닿는 유일한 지점이다.

---

## 06.2 LLM에서 augmentation은 decode다

**augmentation의 정의는 vision에서 왔고, 그 정의의 두 조건 중 하나가 text에서 깨진다.** 깨지는 쪽은 "라벨을 보존하는 변환"이다.

vision에서 augmentation이 쓸모 있는 이유는 라벨을 보존하는 변환군이 **명시적으로 알려져 있기** 때문이다. 고양이 사진을 좌우 반전해도 고양이이고, 밝기를 10% 올려도 고양이다. 이 불변군은 도메인 지식에서 공짜로 온다. 비용도 싸다 — CPU dataloader에서 돌고, GPU는 그동안 backward를 돈다.

text에는 그런 불변군이 없다. 동의어 치환은 의미를 바꿀 수 있고, 역번역은 사실을 흘릴 수 있고, 문장 순서 교체는 논증을 파괴할 수 있다. 그래서 LLM 실무의 augmentation은 결정론적 변환을 포기하고 **모델에게 다시 쓰게 하는 것**으로 대체되었다. 그 순간 표 6-1의 두 번째 행과 세 번째·네 번째 행의 경계가 사라진다. "원 항목이 있는 augmentation"과 "원 항목을 조건으로 받은 synthetic data"는 구현상 같은 것이 된다.

SEAL의 self-edit이 정확히 그 경계 위에 있다. SQuAD passage 한 건을 주고 모델에게 그 passage의 **함의(implications)** 목록을 생성하게 한 뒤, 원 passage가 아니라 그 함의 문장들로 SFT를 돈다 [SEAL §3.2]. 원 항목이 있으므로 augmentation이고, 생성기가 모델이므로 synthetic data이며, 생성기가 학습 대상 자신이므로 self-generated data다. 세 이름이 모두 맞다.

> **[평가]** 세 이름이 모두 맞을 때 이 책은 **self-generated로 분류한다**. 이유는 분류의 목적이 이름 붙이기가 아니라 위험 판정이기 때문이다. §06.6이 보일 붕괴 경계는 "생성기가 누구인가"에만 걸리고 "원 항목이 있는가"에는 걸리지 않는다. 원 항목이 있다는 사실은 안전을 조금도 보장하지 않는다 — 원 항목은 조건일 뿐, 출력의 분포를 정하는 것은 $\Theta_{k-1}$이다.

systems 접점은 비용의 이동이다. 고전형 augmentation의 비용은 CPU dataloader에 있었고 GPU 시간에 잡히지 않았다. LLM augmentation의 비용은 **decode**에 있고, 그래서 $B_s$의 대부분을 차지한다. 문서 1건당 후보 $M$개를 뽑는 파이프라인이면 생성 비용은 그대로 $M$배다. 이 곱이 §06.4에서 다시 나온다.

---

## 06.3 자기 생성이 남의 생성을 이기는가

**답은 "설정에 따라 다르다"이고, 그 설정 의존성이 이 장이 쓰는 가장 중요한 실측 사실이다.**

SEAL은 이 질문을 정면으로 잰다. no-passage-in-context SQuAD v1.1에서 Qwen2.5-7B를 놓고, 같은 passage에 대해 (a) passage 원문으로만 학습, (b) 모델 자신이 생성한 함의로 학습(RL 없음), (c) GPT-4.1이 생성한 synthetic data로 학습, (d) 생성 정책을 RL로 학습한 뒤의 자기 생성(SEAL)을 비교한다. 세 열은 각각 passage 1건(LoRA), 200건 continued pretraining(full-FT), 2067건 continued pretraining(full-FT)이다 [SEAL Table 2].

표 6-2. 생성기에 따른 no-context SQuAD 정확도(%) ([SEAL Table 2] 그대로)

| 방법 | $n{=}1$ (LoRA) | CPT $n{=}200$ | CPT $n{=}2067$ |
|---|---|---|---|
| Base model | 32.7 | 32.7 | 29.0 |
| Train on Passage | 33.5 | 36.0 | 31.2 |
| Passage + 자기 생성(RL 없음) | 39.7 | 50.6 | 43.4 |
| Passage + GPT-4.1 synthetic | 46.3 | **59.4** | **49.2** |
| SEAL(RL로 학습된 자기 생성) | **47.0** | 58.2 | 46.4 |

이 표는 세 줄로 읽힌다. 첫째, **원문을 그대로 학습시키는 것은 거의 아무 일도 하지 않는다** — 32.7에서 33.5로 0.8점이다. 데이터를 만드는 절차 자체가 일의 대부분을 한다. 둘째, **무필터 자기 생성은 남의 생성에 진다** — 39.7 대 46.3. 셋째, 생성 정책을 RL로 학습하면 47.0으로 GPT-4.1을 0.7점 앞선다.

셋째 줄에는 조건이 붙는다. **그 역전은 세 열 중 한 열에서만 성립한다.** CPT 두 열에서는 GPT-4.1이 각각 59.4 대 58.2, 49.2 대 46.4로 이긴다. 논문 자신이 이 사실을 쓴다 [SEAL §4.2]. 초록과 서론은 GPT-4.1 비교를 범위 조건 없이 앞세운다 [SEAL Abstract, §1]. 이 caveat의 의무 장은 ch20이지만, 수치를 여기서 쓰는 이상 여기서도 병기한다.

같은 논문의 부록에는 더 불편한 행이 둘 있다. 첫째, self-edit 프롬프트를 7가지로 바꿔 본 실험에서 RL을 전혀 돌리지 않은 `implications-long`(49.3)과 `rewrite`(49.4)가 RL로 학습된 본문 헤드라인 47.0을 **이미 넘는다** [SEAL Table 10]. 프롬프트 한 줄이 RL 루프 전체보다 크다. 둘째, 학습된 생성 정책 대신 **구조화된 휴리스틱**으로 합성 데이터를 만드는 Entigraph가 가장 큰 규모에서 SEAL을 앞선다 — CPT $n{=}2067$에서 48.6 대 46.4다(문서당 생성 10건 대 5건이라 계산량이 맞춰져 있지 않고, 논문이 그 사실을 밝힌다) [SEAL Table 8, §B.9]. 규모가 커질수록 휴리스틱이 따라잡는 방향의 추세다.

> **[평가]** 표 6-2에서 이 장이 가져갈 결론은 "자기 생성이 낫다"가 아니라 **"형식이 생성기보다 먼저다"**이다. 33.5 → 39.7의 6.2점은 데이터의 *형식*을 원문에서 함의 목록으로 바꿔서 얻었고, 39.7 → 46.3의 6.6점은 *생성기 품질*로 얻었으며, 46.3 → 47.0의 0.7점은 *생성 정책 학습*으로 얻었다. 세 항의 크기 순서가 이 장의 실천 지침이다. $B_s$를 어디에 쓸지 정할 때 마지막 항부터 손대는 것은 순서가 틀렸다.

---

## 06.4 데이터 품질 필터링

**필터는 $\mathrm{gen}$의 절반이다.** 생성만으로 $\mathcal{R}_k$가 정해진다고 보는 것은 실제 파이프라인을 잘못 그린 것이다.

> **데이터 품질 필터링**은 생성된 후보 집합에서 $\mathcal{R}_k$에 실제로 넣을 부분집합을 고르는 절차다. 형식적으로 $\mathrm{gen}$을 두 단계로 쪼갠다: 후보 생성 $\tilde{\mathcal{R}}_k \sim \mathrm{LM}(\cdot\,|\,c;\ \Theta_{k-1})$, 채택 $\mathcal{R}_k = \{\,z \in \tilde{\mathcal{R}}_k \;:\; \phi(z) = 1\,\}$. 판정 함수 $\phi$가 필터다. 후보 하나를 가리키는 표본 기호는 $z$다 — $s$는 SGD step 첨자로 예약되어 있다.

$\phi$는 세 종류로 나뉜다.

1. **규칙·휴리스틱 필터.** 중복 제거, 길이 범위, 형식 검사, n-gram 오염 검사. 비용이 거의 0이고 판정이 결정론적이다.
2. **검증기 기반 필터.** 과제 점수, 실행 결과, 단위 테스트, 보상 모델. 판정이 정확하지만 비용이 후보마다 붙는다. 보상을 최적화하는 알고리즘 자체는 ch05가 소유한다 — 이 장은 그 보상을 **채택 판정으로 쓰는 것**만 다룬다.
3. **사람 필터.** 가장 정확하고 가장 비싸며, 유휴 시간에 상각되지 않는다.

SEAL의 $\phi$는 2번의 극단이다. 후보 self-edit으로 실제로 $\Theta$를 갱신하고, 갱신된 모델로 downstream 문항을 풀어, 점수가 오르면 채택한다 [SEAL Eq. 2]. 헤드라인 실험에서는 이보다 더 좁아서, 후보 $M$개 중 **가장 크게 개선한 하나만** 채택한다 [SEAL footnote 2]. 즉 $\phi$가 후보당 학습 1회 + 평가 1회를 요구한다.

그 $\phi$의 값이 얼마인지도 같은 논문이 잰다. 학습-후-평가 보상을 GPT-4.1 rubric 채점으로 바꾸면 정확도는 47.0에서 45.6으로 1.4점 떨어지고, RL 학습 시간은 약 6시간에서 약 5분으로 줄어든다 [SEAL Table 9]. 1.4점에 약 72배다.

> **[평가]** 이 ablation은 필터의 **비용–품질 곡선이 매우 평평하다**는 뜻이다. 가장 비싼 $\phi$가 가장 싼 $\phi$보다 1.4점 낫다면, $B_s$가 제약인 시스템에서 비싼 $\phi$를 고를 근거는 논문 안에 없다. 논문 자신도 proxy가 "true" 보상을 넘어설 수도 있다고 쓴다 [SEAL §B.10].

필터가 못 하는 것도 같은 표에 있다. 프롬프트를 주지 않고 모델이 스스로 형식을 정하게 한 `No-Prompt` 행은 13.8에서 출발해 RL 2라운드 뒤에도 18.9다 [SEAL Table 10]. **소스가 나쁘면 필터가 구제하지 못한다.** 필터는 분포를 자르는 연산이지 옮기는 연산이 아니다.

마지막으로, 다음 두 절이 다룰 붕괴 경계는 **필터가 없는 경우에만 증명되어 있다**. [MCI]의 $\mathrm{gen}$은 온도 0.3 또는 1.0의 무조건부 샘플링이고, 검증·보상·중복 제거·사람 검토가 전혀 없다 [MCI §2.1]. 필터링은 논문이 future work로 명시한다 [MCI §4]. 이 사실은 양쪽으로 작용한다 — 경계가 최악의 $\mathrm{gen}$을 덮는다는 뜻인 동시에, 필터가 있는 파이프라인(즉 배포된 모든 self-improvement 파이프라인)에 그 경계를 그대로 적용할 수 없다는 뜻이다.

systems 접점은 $B_s$의 곱셈이다. 후보 $M$개를 뽑아 1개를 채택하면 생성 예산은 $M$배이고, $\phi$가 검증기 기반이면 여기에 후보당 판정 비용이 더해진다. SEAL식 $\phi$에서는 그 판정 비용이 SFT 한 번 + 평가 한 번이므로, $B_s \approx M \times (\text{decode} + \text{SFT} + \text{eval})$이 된다. 상각식 (A)의 분자에 들어가는 것은 이 값이지 decode 항 하나가 아니다.

---

## 06.5 model collapse — 이름 하나에 현상 넷

**model collapse는 하나의 현상이 아니다. 이 절의 결론은 정의 하나가 아니라 "어느 뜻인지 지정하라"는 규율이다.**

> **model collapse**는 모델이 자기 계보가 생성한 데이터로 반복 학습될 때 성능이 라운드가 진행될수록 악화되는 현상이다. [MCI]의 조작적 정의는 이보다 좁고 날카롭다 — "model collapse란 model-data 루프의 반복이 늘어남에 따라 오차가 뚜렷하게 악화되는 것을 가리키고, model collapse를 피한다는 것은 그 반복에 대해 오차가 **유계로 남는 것**을 가리킨다" [MCI §2]. 이론 절에서는 더 좁혀서 "test error가 (어떤 속도로든) 무한대로 발산하는 상황"과 동일시한다 [MCI 각주 3].

정의가 "유계인가 발산인가"라는 점이 중요하다. 이 정의 아래에서는 오차가 늘어나도 수렴하면 붕괴가 아니고, 아무리 천천히 늘어도 발산하면 붕괴다. §06.6의 결과 전체가 이 이분법 위에 있다.

[MCI] 자신이 이 이름의 남용을 경고하며 네 현상을 분리한다 [MCI §4 Discussion].

표 6-3. model collapse라는 이름이 가리키는 네 현상 ([MCI §4] 분류)

| # | 현상 | 무엇이 관측되는가 |
|---|---|---|
| (0) | test error의 무계 폭증 | held-out loss가 라운드마다 오르고 수렴하지 않는다 |
| (1) | modal collapse | 생성 분포가 한두 개 mode로 수축한다 |
| (2) | uniformity로의 붕괴 | 생성 분포가 특징 없이 평평해진다 |
| (3) | artifact 증폭 | 앞선 합성 데이터가 남긴 결함이 라운드마다 커진다 |

> **[해설]** 이 네 항목은 서로를 함의하지 않는다. loss가 유계로 남으면서 mode가 사라질 수 있고, 그 반대도 가능하다. §06.7에서 보겠지만 [MCI]의 VAE 실험이 정확히 그 경우다 — (0)은 완화되는데 (1)은 그대로 일어난다. 따라서 이 책은 "붕괴한다/안 한다"를 번호 없이 쓰지 않는다. Part II와 Part III에서 $\Theta$-경로 논문의 주장을 검증할 때에도 같은 규율을 적용한다.

systems 접점은 모니터링이다. 프로덕션에서 관측 가능한 것은 대개 (0)뿐이다 — held-out perplexity는 대시보드에 있지만 생성 분포의 mode 수는 없다. 그래서 (1)과 (3)은 **알람이 울리지 않는 붕괴**다. §06.9에서 이 함정으로 돌아온다.

---

## 06.6 accumulate vs replace — 스케줄이 수렴을 정한다

**루프의 안정성은 학습 규칙이 아니라 $\mathcal{R}_k$를 구성하는 스케줄이 정한다.** 이것이 [MCI]의 결과이고, 이 장이 Part III에 넘기는 핵심 명제다.

> **replace**는 라운드마다 이전 학습 집합을 버리고 새로 생성한 것으로 바꾸는 스케줄이다: $\mathcal{R}_k = \mathrm{gen}(\mathcal{H}_{k-1}; B_s)$, $|\mathcal{R}_k| = T$.
> **accumulate**는 이전 학습 집합을 버리지 않고 새 생성분을 덧붙이는 스케줄이다: $\mathcal{R}_k = \mathcal{R}_{k-1} \cup \mathrm{gen}(\mathcal{H}_{k-1}; B_s)$, $|\mathcal{R}_k| = kT$. 라운드 1의 실데이터 $\mathcal{R}_1$은 영원히 남는다.
> ($T$는 라운드당 추가되는 표본 수다. 이 장에서 계속 그 뜻으로 쓴다.)

세 번째 스케줄이 하나 더 있다. **replace-multiple**은 실데이터를 버리되 합성 코퍼스를 $kT$까지 키우는 스케줄로, 데이터 양을 accumulate와 맞춘 통제군이다 [MCI App. E].

> **[평가]** 여기서 이 책의 프레임을 한 번 정정한다. **ch01 §01.3이 세운 (U-$\Theta$)의 $\mathcal{R}_k = \mathrm{gen}(\mathcal{H}_k; B_s)$는 글자 그대로 replace다.** 표준형을 그대로 반복하는 시스템은 [MCI]가 발산한다고 증명한 바로 그 레짐에 있다. 표준형은 스케줄을 지정하지 않았고, 지정하지 않은 채로 읽으면 replace로 읽힌다. 따라서 스케줄은 (U-$\Theta$)의 세부가 아니라 $\eta_\Theta$와 나란한 1급 파라미터이며, 이 책은 이후 $\Theta$-경로 장에서 그렇게 다룬다.

[MCI]의 이론은 잘 지정된 ridgeless 선형 회귀에서 전개된다. 입력 차원 $d_{\text{in}}$, 참 파라미터 $\Theta^{*}$, 라벨 잡음 분산 $\sigma^2$. 라운드마다 이전 모델로 타깃을 합성할 때 새 잡음 $\mathcal{E}_i \sim \mathcal{N}(0, \sigma^2 I_T)$가 주입된다. accumulate에서 파라미터의 정확한 재귀는 다음과 같다 [MCI Thm. 1, Eq. 2].

$$
\Theta_k \;=\; \Theta^{*} \;+\; (X^{\top}X)^{-1}X^{\top}\!\left(\sum_{i=1}^{k}\frac{\mathcal{E}_i}{i}\right)
\tag{6-1}
$$

이 식이 결과 전체를 이미 말한다. **라운드 $i$에 주입된 잡음은 계수 $1/i$로 파라미터에 도달한다** — 그것이 코퍼스에 들어간 시점에 $i$개 블록 중 하나였기 때문이다. 오차는 제곱이므로 기여는 $1/i^2$가 되고, $\sum 1/i^2$가 수렴하므로 총합이 유계로 남는다 [MCI §3.2]. 결론은 다음 세 줄이다.

$$
\mathcal{L}_{\text{test}}^{\text{acc}}(\Theta_k) = \frac{\sigma^{2} d_{\text{in}}}{T-d_{\text{in}}-1}\sum_{i=1}^{k}\frac{1}{i^{2}} \;\le\; \frac{\sigma^{2} d_{\text{in}}}{T-d_{\text{in}}-1}\cdot\frac{\pi^{2}}{6}
\tag{6-2}
$$

$$
\mathcal{L}_{\text{test}}^{\text{rep}}(\Theta_k) = \frac{\sigma^{2} d_{\text{in}}}{T-d_{\text{in}}-1}\times k
\qquad\quad
\mathcal{L}_{\text{test}}^{\text{rep-mult}}(\Theta_k) = \frac{\sigma^{2} d_{\text{in}}}{T-d_{\text{in}}-1}\sum_{i=1}^{k}\frac{1}{i} \approx \frac{\sigma^{2} d_{\text{in}}}{T-d_{\text{in}}-1}\log k
\tag{6-3}
$$

(6-2)는 [MCI Eq. 3](Theorem 2), (6-3)의 왼쪽은 [MCI Eq. 4](Dohmatob 외의 선행 결과를 표기 통일해 재진술한 것), 오른쪽은 [MCI Eq. 11]이다. 성립 조건은 $T \ge d_{\text{in}}+2$, 등방 공분산 $\Sigma = I_{d_{\text{in}}}$, ridge 계수 0, $X$ 열 full rank다 [MCI Thm. 2].

$\pi^2/6 \approx 1.6449$다. **accumulate 루프를 영원히 돌린 총 열화가 한 라운드 벌점의 1.645배를 넘지 않는다.** 이 책의 프레임이 각 논문에 요구하는 $\rho$ 칸을, corpus에서 유일하게 측정치가 아니라 공식으로 채우는 것이 이 결과다. 세 닫힌 형식을 1차 차분하면(이 책의 산술이며 추가 가정은 없다) 라운드당 열화율이 나온다.

$$
\rho_k \;=\; \frac{\sigma^{2}d_{\text{in}}}{T-d_{\text{in}}-1}\cdot
\begin{cases}
1 & \text{replace}\\
1/k & \text{replace-multiple}\\
1/k^{2} & \text{accumulate}
\end{cases}
\tag{6-4}
$$

설계 규칙은 여기서 바로 읽힌다. **반복되는 (U-$\Theta$) 루프는 $\sum_k \rho_k$가 수렴할 때에만 지속 가능하다.** [MCI]의 정의상 $\log k$ 발산도 붕괴로 센다 [MCI 각주 3] — replace-multiple은 데이터를 accumulate만큼 늘리고도 붕괴한다. **데이터를 더 만드는 것은 기제가 아니다. 실데이터를 계속 보유하는 것이 기제다.**

실측은 이론과 같은 방향이다. TinyStories(470M tokens)에서 라운드마다 새 모델을 초기화해 1 epoch씩 학습하고 [MCI §2.1], GPT-2 9M의 held-out cross-entropy를 재면 라운드 1에서 1.82이고, 라운드 4에서 accumulate 1.74(−0.07), replace 2.39(+0.58), 라운드 10 replace 2.91(+1.09)이다. 데이터 양을 맞춘 replace-multiple도 라운드 4에서 2.18(+0.36)로 악화된다 [MCI Table 2]. 가장 큰 모델(본문은 125M, Table 2와 Fig. 2 범례는 126M으로 표기한다 — 논문 내부가 1M 어긋나 있어 양쪽을 그대로 옮긴다)에서는 1.71 → accumulate 1.59(−0.12), replace 2.23(+0.53)이다 [MCI Table 2].

accumulate 열의 부호에는 단서가 붙는다. **표의 어느 행도 계산량이 맞춰져 있지 않다.** "1 epoch"은 accumulate에서 $kT$개 표본에 대한 1 epoch이므로 gradient step 수가 replace보다 많고, 논문 자신이 그림 캡션에서 그 사실을 밝힌다 [MCI Fig. 3 캡션 주석]. 따라서 1.82 → 1.74처럼 accumulate가 **좋아지는** 방향의 delta 일부는 코퍼스가 커지고 최적화가 늘어난 결과이며, 정규화에 쓸 계산량 수치는 논문 어디에도 없다. 이 장이 accumulate에서 가져가는 것은 개선이 아니라 **유계성**이다.

디코딩 온도가 이 그림을 크게 흔든다. 같은 GPT-2 9M을 온도 0.3으로 생성하면 replace는 라운드 4에서 5.82(+4.00), 라운드 10에서 9.85(+8.04)로 무너지는 반면 accumulate는 1.75(−0.06)로 사실상 불변이다 [MCI Table 2]. 낮은 온도가 더 나쁘다. 그리고 온도 0(greedy) — 프로덕션 기본값 — 은 검증되지 않았고, 논문이 future work로 명시한다 [MCI §4].

systems 접점은 비용의 비대칭이다. accumulate에서 생성은 라운드마다 $T$로 일정하지만 **학습은 $|\mathcal{R}_k| = kT$로 커진다.** $K$라운드 총 학습량은 $\Theta(K^2T)$이고 replace는 $\Theta(KT)$다. 논문 자신의 5라운드 스윕을 이 책의 산술로 환산하면 accumulate는 $470\text{M} \times (1{+}2{+}3{+}4{+}5) = 7.05\text{B}$ training tokens, replace는 $470\text{M}\times 5 = 2.35\text{B}$이며 생성량은 양쪽 모두 $470\text{M}\times 4 = 1.88\text{B}$로 같다. 논문은 이 세 합계 중 어느 것도 쓰지 않는다 — 비대칭을 말로만 두 번 언급한다 [MCI Fig. 3 캡션 주석; MCI App. E].

---

## 06.7 헤드라인이 성립하지 않는 자리

**[MCI]의 헤드라인은 세 모달리티 중 하나에서 성립하지 않는다. 이 책은 자기 논지에 유리한 논문에도 같은 자를 댄다.**

CelebA 위의 VAE 실험에서 논문은 이렇게 쓴다: "언어 모델링과 달리, 데이터를 accumulate할 때에도 test error는 반복 횟수에 따라 **증가한다**(데이터를 replace할 때보다 훨씬 느리기는 하지만)" [MCI §2.3]. 논문 자신의 정의 — 붕괴를 피한다는 것은 오차가 유계로 남는 것 [MCI §2] — 을 그대로 적용하면, 이 실험은 붕괴 회피를 보이지 않았다. 그럼에도 해당 그림의 캡션은 "Data Accumulation Avoids Model Collapse in Variational Autoencoders for Image Generation"이고 [MCI Fig. 5], 초록은 "다른 종류의 실데이터 위의 심층 생성 모델에서도 유사한 결과를 얻었다"고 쓴다 [MCI Abstract]. 캡션과 본문 측정이 어긋난다.

같은 실험에 두 번째 사실이 붙는다. accumulate 아래에서도 생성 다양성은 떨어진다 — "성별처럼 데이터셋의 주요 변동 축은 여전히 표현하지만, 안경이나 액세서리 같은 데이터 다양체의 부차적 축의 세부는 더 이상 생성하지 못하는 것으로 보인다" [MCI §2.3]. 부록은 반대 방향의 사실을 하나 더 준다: 같은 모델의 **재구성**은 안경과 모자를 유지한다 [MCI App. D]. 즉 encoder–decoder 능력은 살아남고 **샘플링 분포의 꼬리만 죽는다**.

> **[평가]** 표 6-3의 언어로 옮기면 이것이 정확히 (0)은 완화되고 (1)은 일어나는 경우다. 그리고 sleep 라운드를 반복하는 시스템에게는 이쪽이 더 나쁜 이야기다 — $\mathrm{gen}$은 재구성이 아니라 **샘플링**을 쓰기 때문이다. 라운드 $k$의 샘플 분포에서 사라진 꼬리는 라운드 $k+1$의 학습 집합에 없고, 그 손실은 자기 증폭한다. held-out loss는 그동안 완만하게 오를 뿐이다 [MCI §2.3].

논문이 스스로 인정하는 외부 반례도 있다. "Martínez 외(2023a)는 다소 상반된 증거를 찾았다 — 훨씬 작은 데이터셋 위의 다른 아키텍처에서는 데이터를 accumulate해도 성능이 빠르게 열화한다"이고, 왜 이런 불일치가 생기는지는 future work로 남긴다 [MCI §2.3]. 초록의 일반화 주장("여러 모델 크기·아키텍처·하이퍼파라미터에 걸쳐 성립" [MCI Abstract])에는 문헌 안에 이름이 붙은 반례가 있고 이 논문은 그것을 재현하지도 반박하지도 않는다.

증거의 성격도 한 항목에서 다르다. 부록의 근거 하나는 새 실험이 아니라 **남의 그림을 다시 읽은 것**이다. Alemohammad 외(2023)의 Figure 7을 그대로 옮겨 싣고, 원저자들이 "실데이터를 고정해 두는 것은 불가피한 열화를 늦출 뿐"이라고 내린 결론에 대해 "자세히 보면 오차가 낮은 값에서 평탄해진다"고 반박한다 [MCI App. A, Fig. 8]. 이 책은 이것을 증거가 아니라 **해석 분쟁**으로 인용한다 — 저자들이 돌리지 않은 실험의 곡선에 대한 판독이기 때문이다.

세 번째 모달리티도 이론과 형태가 다르다. 분자 conformation diffusion에서 replace는 "합성 데이터로 학습한 첫 라운드에서 크게 나빠진 뒤 이후 라운드에서 실질적으로 더 악화되지 않는다" [MCI §2.2]. 식 (6-3)의 왼쪽은 $k$에 **선형**인 증가를 예측한다. 한 번 떨어지고 평평해지는 곡선은 그 예측이 아니며, 논문은 이 불일치를 논평하지 않는다.

마지막으로 언어 모델 실험의 "실데이터"는 실데이터가 아니다. TinyStories를 논문은 "GPT-3.5/4가 생성한 470M 토큰의 유치원 수준 짧은 이야기 데이터셋"이라고 소개한다 [MCI §2.1]. 보호받는 분포 자체가 모델 출력이고, 엔트로피가 낮으며 구조가 단거리다. 웹 규모의 무거운 꼬리 분포에서 $1/i^2$ 논증이 살아남는지는 검증되지 않았다.

---

## 06.8 경계의 전제 — 이 결과를 SEAL과 Dreaming에 옮길 수 있는가

**옮길 수 없다. 그리고 그 사실이 이 장이 Part III에 넘기는 반증 축이다.**

식 (6-2)가 성립하는 설정은 다음을 모두 가정한다. (i) 라운드마다 $\Theta_k$를 **처음부터 다시 학습한다** — 이론에서는 닫힌 해 $\Theta_k = \tilde X_k^{\dagger}\tilde Y_k$이고, 실험에서는 "새로 초기화된 모델을 사전학습한다" [MCI §2.1, §3.1]. 그래서 $\eta_\Theta$는 이론에 아예 등장하지 않고, optimizer state도 checkpoint 연속성도 catastrophic forgetting도 모델링되지 않는다. (ii) 공변량이 동결되어 있다 — accumulate된 설계 행렬은 **같은 $X$를 $k$번 쌓은 것**이고 타깃만 재생성된다 [MCI §3.1]. 즉 정리가 경계 짓는 것은 누적된 **라벨 잡음**이지 입력 분포의 이동이 아니다. (iii) 모델 클래스가 참 함수를 포함하고(well-specified), ridge 계수가 0이며, 공분산이 등방이다 [MCI Thm. 2]. test error가 $\mathbb{E}\lVert\Theta - \Theta^{*}\rVert^2_{\Sigma}$로 정의되므로 bias 항이 분석 전체에 없다 [MCI Eq. 1]. (iv) $\mathrm{gen}$이 무필터다(§06.4).

> **[평가]** 네 전제는 $\Theta$-경로의 모든 시스템에서 깨진다. SEAL은 살아 있는 checkpoint를 LoRA로 증분 편집하고 [SEAL §3.2], 보상으로 후보를 고르며 [SEAL Eq. 2], 입력(생성된 함의 문장)을 매 라운드 새로 만든다. `Language Models Need Sleep`의 Dreaming도 마찬가지다. (6-2)의 기제 — 라운드 $i$의 데이터가 코퍼스의 $1/i$이므로 잡음이 $1/i$로 들어온다 — 는 **이전 데이터가 아니라 이전 파라미터가 운반자일 때 대응물이 없다.** 따라서 이 정리는 그 시스템들을 보증하지도, 반증하지도 않는다. 이 책이 하지 말아야 할 일은 $\pi^2/6$을 안전 증서로 인용하는 것이다.
>
> 반대 방향으로도 한 칸을 깎아야 한다. 논문은 자기 설정을 "합성 데이터가 통제 없이 인터넷에 쏟아지는 미래를 상정한, 어떤 의미에서 최대로 비관적인" 것이라고 부른다 [MCI §1]. 그러나 accumulate에서 원 실데이터는 규정상 모든 라운드의 코퍼스에 절대량 그대로 남고, 이론에서는 실공변량 $X$가 매 라운드 문자 그대로 재사용된다 [MCI §3.1]. **보호 성분을 제거할 수 없게 만들어 둔 설정은 최악의 경우가 아니라 유리한 구성이다.** 인터넷에는 2023년 이전 사람 코퍼스가 이후 모든 학습 집합에 줄지 않은 크기로 계속 포함된다고 보장하는 주체가 없다. 이 논문을 "인터넷 규모의 안심 근거"로 옮겨 쓰는 것은 이 구성을 지우는 일이다.

측정 자체의 한계도 기록해 둔다. 시드·반복·오차막대·신뢰구간이 전혀 없어 Table 2의 모든 곡선과 항목이 단일 실행이다. downstream 벤치마크도 없어 모든 정량 주장이 held-out loss이고, 정성 증거는 손으로 고른 생성문 여섯 개뿐이다 [MCI Table 1]. FLOPs·GPU-hour·wall-clock·bytes·latency 중 어느 것도 보고되지 않으며, 논문 전체의 유일한 비용 진술은 "계산적으로 비싼 시도"라는 형용사다 [MCI §2.1]. 그리고 실증 상한은 126M 파라미터 × 470M 토큰 × 5 accumulate 라운드인데 [MCI §2.1], 논문이 서론에서 동기로 든 규모는 Llama 1/2/3의 1.4T / 2T / 15T 토큰이다 [MCI §1]. 200라운드의 긴 지평선은 10차원 선형 장난감에만 존재한다 [MCI App. F].

이 corpus에서 가장 깨끗한 침묵도 여기 있다. [MCI]는 지속학습 문헌을 **하나도** 인용하지 않는다 — EWC도, GEM도, Deep Generative Replay도, LoRA도, 모델 편집도 없다. 반대 방향도 같다: $\Theta$-경로의 어느 논문도 이 결과를 인용하지 않는다. `Language Models Need Sleep`은 반복 self-distillation이 "training collapse"를 일으킬 수 있다는 위험을 관련 연구 논의에서 직접 다루면서도, 그 위험을 2026년 on-policy self-distillation 문헌(He 외, Wang 외)에 귀속시키고 2024년 model collapse 문헌에는 닿지 않는다(원문 위치는 ch21이 소유한다). **두 공동체가 같은 위험을 발견해 서로 다른 이름을 붙이고 서로를 0번 인용한다.** 이 다리는 논문이 아니라 이 책이 놓는다. ch11의 판별식과 ch22의 반증 축, ch30의 반증 조건이 그것을 이어받는다.

<!-- TODO-VERIFY: SEAL(2506.10943)·LM Need Sleep(2606.03979)·SCM(2604.20943)의 최신 arXiv 판이 2404.01413을 인용하는지. 확인 방법: papers/ 전체에 grep "Gerstgrasser" 및 "2404.01413"(현재 이 논문 자신 외 0건), 그 뒤 각 논문의 최신 판 참고문헌 재확인 -->

---

## 06.9 그래서 decode에서 무엇이 바뀌는가

**decode 자체는 아무것도 바뀌지 않는다. 그리고 그것이 이 장의 비용이 다른 데 숨어 있다는 뜻이다.**

sleep 라운드의 산출물은 아키텍처도 파라미터 수도 그대로인 $\Theta$다 [MCI §2.1]. 추가 상태도, KV cache 변화도, 검색 hop도, 토큰당 오버헤드도 없다. $L_w$는 정의상 갱신 전후가 같다. 이 칸은 "논문에 없음"이 아니라 **구조적으로 0**이다. $E$-경로와 비교했을 때 $\Theta$-경로의 구조적 강점이 정확히 이것이고, 그래서 비용을 다른 데서 찾아야 한다.

찾으면 세 곳에 있다.

**첫째, sleep 트래픽은 성격이 반대인 두 작업이다.** 생성은 $\Theta_{k-1}$에서 코퍼스를 뽑아내는 batch decode — memory-bandwidth-bound, 가중치 HBM 상주, 출력은 스토리지로 스트리밍 — 이고, 학습은 $k$에 선형으로 커지는 코퍼스를 순차 읽는 compute-bound 패스다. 어느 쪽도 지연에 민감하지 않고 어느 쪽도 캐시 계층에서 흥미로운 일을 하지 않는다. 압력은 스토리지 용량과 순차 읽기 대역폭에 걸린다.

**둘째, $C_{\text{cap}}$이 둘로 쪼개진다.** accumulate 아래에서 지속되는 상태는 가중치 delta가 아니라 **보존된 코퍼스**다. 가중치 상태는 모델 크기에 고정($\Theta(1)$)이고, 데이터 상태는 라운드 수에 선형($\Theta(k)$)이며 버려지지 않는다. 이 책의 산술로 크기를 보면 470M tokens를 uint16 id로 저장할 때 라운드당 약 0.94 GB, 5라운드 누적 약 4.7 GB다(논문은 어떤 바이트 수치도 보고하지 않는다). 절대값은 작지만 **모양**이 요점이다. ch01 §01.5가 세운 $C_{\text{cap}}$ 칸은 첫 번째만 가정한다. $\Theta$-경로 장들이 두 칸을 모두 요구하는 이유가 이 논문이다(→ ch09, ch26).

> **[평가]** 셋째, **batching은 살아남는다 — 단, 이 레짐에서만.** Rosetta의 "batch로 weight 공유 → $\Theta$-경로에서 깨짐" 행은 **per-user** 가중치 delta에 대한 진술이다. [MCI]의 루프는 전체 인구에 대해 라운드당 하나의 전역 $\Theta$를 만들므로 shared-weight batching이 깨지지 않는다. SEAL식 per-context self-edit만 깨뜨린다. 따라서 이 책은 **인구 수준 $\Theta$-경로**와 **사용자 수준 $\Theta$-경로**를 구분해서 부른다. 이 구분 없이 "$\Theta$-경로는 batching을 깬다"고 쓰면 절반이 틀린다.

마지막으로 모니터링 함정 하나. §06.7의 VAE 결과는 **집계 loss가 유계인 채로 샘플링 분포의 꼬리가 사라질 수 있음**을 보인다. held-out perplexity를 보는 대시보드는 이때 울리지 않는다. sleep 루프의 출력이 다음 라운드의 입력이 되는 시스템에서 꼬리 손실은 자기 증폭하므로, 관측 지표에 **분포 다양성 측정치**가 없으면 (1)번 붕괴는 조용히 진행된다.

---

## (state, update, cost) 정리

이 장의 개념은 전부 (U-$\Theta$)의 오른쪽 절반, 즉 $\mathrm{gen}(\mathcal{H}_k; B_s)$ 안에 있다. $W$도 $E$도 등장하지 않는다.

표 6-4. 이 장의 개념을 세 층 프레임에 놓기

| 개념 | state | update | cost |
|---|---|---|---|
| data augmentation | $\mathcal{R}_k$의 항목 수를 늘림 | (U-$\Theta$) 이전 단계. $\Theta$·$W$·$E$ 불변 | $B_s$: 고전형은 CPU, LLM형은 decode |
| synthetic data (타 모델) | $\mathcal{R}_k$ 전체를 대체 가능 | (U-$\Theta$) 이전 단계. 루프 안 닫힘 | $B_s$가 타 모델 decode로 외부화 |
| self-generated data | $\mathcal{R}_k = \mathrm{LM}(\cdot;\Theta_{k-1})$ | (U-$\Theta$)가 자기 출력을 되먹임 | $B_s$: 자기 decode. $\rho$가 여기서 발생 |
| 데이터 품질 필터링 | $\mathcal{R}_k \subseteq \tilde{\mathcal{R}}_k$ | $\mathrm{gen}$을 생성+채택 2단계로 분해 | $B_s \approx M\times(\text{decode}+\phi\text{ 판정})$ |
| accumulate vs replace | $|\mathcal{R}_k| = kT$ 대 $T$ | (U-$\Theta$)의 1급 스케줄 파라미터 | $\rho_k \propto 1/k^2$ 대 $1$; 총 학습 $\Theta(K^2T)$ 대 $\Theta(KT)$; $C_{\text{cap}}$에 데이터 꼬리 |
| model collapse | $\Theta_k$의 열화 | (U-$\Theta$) 반복의 극한 거동 | $\rho$ 칸 자체. $\sum_k\rho_k$ 수렴 여부가 판정 |

$L_w$는 이 장의 어느 개념에 대해서도 구조적으로 0이다. $B_s$와 $\rho$가 이 장이 채우는 두 칸이고, $C_{\text{cap}}$은 accumulate에서만 움직인다.

---

## Worked micro-example — 1차원 가우시안으로 세 스케줄을 손으로 따라가기

식 (6-1)~(6-4)를 암산으로 재현할 수 있는 가장 작은 설정을 만든다. 이 절의 모든 숫자는 이 책의 산술이다.

**설정.** 진짜 분포는 $p^{*} = \mathcal{N}(0,\,1)$이다. 모델은 파라미터가 평균 하나뿐이다: $\Theta_k = \hat\mu_k$, 예측 분포는 $\mathcal{N}(\hat\mu_k, 1)$. 학습은 학습 집합의 표본평균을 취하는 것이고(이것이 이 모델의 $\arg\min$이다), 생성은 $\mathcal{N}(\hat\mu_{k-1},1)$에서 뽑는 것이다. 라운드당 표본 수는 $T = 100$. 라운드 1은 실데이터 100개로 시작한다.

test error는 [MCI Eq. 1]과 같은 형태로 잡는다. 모델 클래스가 참 분포를 포함하므로 bias가 없고 순수 분산이다.

$$
\mathcal{L}_{\text{test}}(\Theta_k) \;=\; \mathbb{E}\big[(\hat\mu_k - 0)^2\big] \;=\; \mathrm{Var}[\hat\mu_k]
$$

표본 100개의 평균이 갖는 분산은 $\sigma^2/T = 1/100 = 0.01$이다. **이 0.01이 한 라운드 벌점이고, 이후 모든 숫자가 그 배수다.**

**단계 1 — replace.** 라운드 $k$는 $\hat\mu_{k-1}$에서 뽑은 100개만으로 학습한다. 그 100개의 표본평균은 $\hat\mu_{k-1} + \bar\varepsilon_k$이고 $\mathrm{Var}[\bar\varepsilon_k] = 0.01$이다. 따라서

$$
\hat\mu_k \;=\; \hat\mu_{k-1} + \bar\varepsilon_k \;=\; \sum_{i=1}^{k}\bar\varepsilon_i,
\qquad \mathrm{Var}[\hat\mu_k] = 0.01\,k .
$$

**random walk다.** 매 라운드 벌점 0.01이 그대로 쌓인다. 이것이 식 (6-3)의 왼쪽이다.

**단계 2 — accumulate.** 라운드 $k$의 코퍼스는 블록 $k$개($100k$개 표본)이고, $\hat\mu_k$는 블록 평균 $m_j$들의 평균이다. $m_1 = \bar\varepsilon_1$이고 $j\ge2$에 대해 $m_j = \hat\mu_{j-1} + \bar\varepsilon_j$다. 손으로 두 줄만 전개해 보면 계수가 보인다.

$$
\hat\mu_2 = \tfrac{1}{2}\big(\bar\varepsilon_1 + (\hat\mu_1 + \bar\varepsilon_2)\big) = \bar\varepsilon_1 + \tfrac{1}{2}\bar\varepsilon_2,
\qquad
\hat\mu_3 = \tfrac{1}{3}\big(m_1+m_2+m_3\big) = \bar\varepsilon_1 + \tfrac{1}{2}\bar\varepsilon_2 + \tfrac{1}{3}\bar\varepsilon_3 .
$$

일반형은 $\hat\mu_k = \sum_{i=1}^{k}\bar\varepsilon_i / i$ 이고, 이것이 식 (6-1)이다. 잡음이 제곱으로 들어가므로

$$
\mathrm{Var}[\hat\mu_k] = 0.01\sum_{i=1}^{k}\frac{1}{i^2} \;\le\; 0.01\times\frac{\pi^2}{6} = 0.016449 .
$$

**단계 3 — replace-multiple.** 실데이터를 버리되 라운드 $k$에 $100k$개를 $\hat\mu_{k-1}$에서 뽑는다. 블록 평균의 분산이 $0.01/k$이므로 $\mathrm{Var}[\hat\mu_k] = 0.01\sum_{i\le k} 1/i$ — 조화급수이고 $\log k$로 발산한다.

표 6-5. 세 스케줄의 $\mathcal{L}_{\text{test}}(\Theta_k)$ ($\sigma^2{=}1$, $T{=}100$, 이 책의 산술)

| $k$ | replace | replace-multiple | accumulate | accumulate의 실데이터 비율 |
|---|---|---|---|---|
| 1 | 0.0100 | 0.0100 | 0.0100 | 1 |
| 2 | 0.0200 | 0.0150 | 0.0125 | 1/2 |
| 3 | 0.0300 | 0.01833 | 0.01361 | 1/3 |
| 4 | 0.0400 | 0.02083 | 0.01424 | 1/4 |
| 5 | 0.0500 | 0.02283 | 0.01464 | 1/5 |
| 10 | 0.1000 | 0.02929 | 0.01550 | 1/10 |
| 100 | 1.0000 | 0.05187 | 0.01635 | 1/100 |
| $\infty$ | $\infty$ | $\infty$ | **0.016449** | 0 |

$k=100$ 행이 이 표의 요점이다. replace의 오차 1.00은 **참 분포의 분산과 같다** — 100라운드를 돌고 나면 모델의 평균 추정이 실표본 하나보다 나을 것이 없다. 같은 라운드에서 accumulate는 0.01635이고 영원히 0.016449를 넘지 않는다. 약 61배 차이가 학습 규칙이 아니라 스케줄 하나에서 나온다.

**단계 4 — 혼합 비율을 고정하면.** 이제 질문을 바꾼다. 코퍼스 크기를 $T=100$으로 **고정한 채** 매 라운드 실데이터 비율을 $\lambda$로 유지하면(이 장에서만 쓰는 기호다) 분포는 어디로 가는가. 라운드 $k$의 학습 집합은 새 실표본 $\lambda T$개와 $\hat\mu_{k-1}$에서 뽑은 합성 표본 $(1-\lambda)T$개다. 표본평균을 그대로 쓰면

$$
\hat\mu_k \;=\; (1-\lambda)\,\hat\mu_{k-1} \;+\; u_k,
\qquad
\mathrm{Var}[u_k] \;=\; \frac{\lambda^2}{\lambda T} + \frac{(1-\lambda)^2}{(1-\lambda)T} \;=\; \frac{1}{T} \;=\; 0.01 .
$$

계수 $(1-\lambda)$의 AR(1)이므로 정상 분산이 존재한다.

$$
\mathrm{Var}_\infty \;=\; \frac{0.01}{1-(1-\lambda)^2} \;=\; \frac{0.01}{\lambda(2-\lambda)}
\tag{6-5}
$$

$\lambda = 1$이면 0.0100(실데이터만), $\lambda = 0.5$이면 0.01333, $\lambda = 0.3$이면 0.01961, $\lambda = 0.1$이면 0.05263, $\lambda = 0.01$이면 0.50251이고, $\lambda \to 0$에서 발산한다. **$\lambda$가 0보다 크기만 하면 분포는 유계인 자리에 멈추고, 정확히 0일 때만 random walk가 된다.** 실데이터의 역할은 참 평균으로 끌어당기는 복원력이며, 그 세기가 $\lambda$다.

두 결과를 겹쳐 놓으면 이 장에서 가장 반직관적인 사실이 나온다. accumulate에서 실데이터 비율은 $1/k \to 0$으로 **사라진다.** 그런데도 오차는 0.016449에 멈춘다. 식 (6-5)를 뒤집어 풀면 그 값은 실데이터 비율을 영구히 $\lambda \approx 0.37$로 고정한 것과 같은 자리다. **중요한 것은 실데이터의 비율이 0으로 가느냐가 아니라 얼마나 빨리 가느냐다.** $1/k$로 줄어드는 것은 충분히 느리다.

> **[평가]** 두 처방의 가격표는 전혀 다르고, 이것이 systems 독자에게 실제 선택지다. accumulate는 유계성을 **계산**으로 산다 — 코퍼스가 $kT$로 커지므로 총 학습량이 $\Theta(K^2T)$이고, 보존 코퍼스가 $C_{\text{cap}}$의 데이터 꼬리로 남는다. 고정 $\lambda$는 유계성을 **새 실데이터 공급**으로 산다 — 라운드마다 $\lambda T$개의 신선한 사람 데이터가 계속 들어와야 하고, 대신 코퍼스 크기와 학습량은 $\Theta(KT)$로 평평하다. 후자는 [MCI]가 첫 번째 open question으로 남긴 레짐이고 [MCI §4], corpus의 어느 논문도 답하지 않는다. 배포된 시스템은 대부분 후자에 있다.

---

## 요약

- (U-$\Theta$)의 절반은 $\mathrm{gen}(\mathcal{H}_k;B_s)$이고, 이 장은 그 절반만 다뤘다. data augmentation·synthetic data·self-generated data는 포함 관계가 아니라 서로 다른 질문(원 항목이 있는가 / 사람이 썼는가 / 생성기가 누구인가)에 답하는 세 축이다.
- LLM에서 augmentation은 결정론적 변환이 아니라 decode다. 그래서 비용이 CPU dataloader에서 GPU로 옮겨 왔고, augmentation과 synthetic data의 경계가 구현상 사라졌다. 셋 중 둘 이상에 걸치면 이 책은 self-generated로 분류한다.
- 데이터를 만드는 절차가 일의 대부분을 한다. no-context SQuAD에서 원 passage 학습은 32.7 → 33.5에 그치고, 함의 형식으로 바꾸면 39.7, 생성기를 GPT-4.1로 바꾸면 46.3, 생성 정책을 RL로 학습하면 47.0이다 [SEAL Table 2]. 세 증분의 크기 순서가 $B_s$ 배분의 순서다. 그리고 자기 생성이 남의 생성을 이긴다는 주장은 그 세 열 중 한 열에서만 성립한다 — CPT $n{=}200$과 $n{=}2067$에서는 GPT-4.1이 각각 59.4 대 58.2, 49.2 대 46.4로 이긴다 [SEAL Table 2, §4.2].
- 필터 $\phi$는 $\mathrm{gen}$의 나머지 절반이고 $B_s$의 곱셈 인자다. 가장 비싼 $\phi$(실제 학습-후-평가)와 가장 싼 $\phi$(rubric 채점)의 차이는 1.4점이고 계산은 약 72배다 [SEAL Table 9]. 필터는 분포를 자를 뿐 옮기지 못한다 — 소스가 나쁘면 구제되지 않는다 [SEAL Table 10].
- model collapse는 네 현상의 이름이며, 번호를 지정하지 않은 붕괴 진술은 이 책에서 결함이다 [MCI §4].
- 반복 (U-$\Theta$)의 안정성은 학습 규칙이 아니라 $\mathcal{R}_k$의 스케줄이 정한다. $\rho_k$는 replace에서 상수, replace-multiple에서 $1/k$, accumulate에서 $1/k^2$이고, 지속 가능 조건은 $\sum_k \rho_k < \infty$다. accumulate의 총 열화는 한 라운드 벌점의 $\pi^2/6 \approx 1.6449$배를 넘지 않는다 [MCI Eq. 3, Eq. 4, Eq. 11].
- 그 경계는 VAE/CelebA에서 성립하지 않는다 — accumulate에서도 test error가 증가한다고 논문이 본문에 쓰고 [MCI §2.3], 같은 그림의 캡션은 반대로 단정한다 [MCI Fig. 5]. 집계 loss가 유계인 채로 샘플링 분포의 꼬리가 사라지는 것이 이 장이 넘기는 가장 위험한 실패 양식이다.
- 경계의 네 전제(처음부터 재학습, 공변량 동결, well-specified·ridgeless, 무필터 생성)는 $\Theta$-경로의 모든 시스템에서 깨진다. $\pi^2/6$은 SEAL과 Dreaming의 안전 증서가 아니다.

---

## 자가 점검 체크리스트

- [ ] 내가 보고 있는 파이프라인의 $\mathcal{R}_k$를 표 6-1의 네 행 중 하나로 분류할 수 있는가. 특히 생성기가 학습 대상 모델 자신인지 아닌지를 말할 수 있는가.
- [ ] 그 파이프라인의 스케줄이 replace인지 accumulate인지 replace-multiple인지 말할 수 있는가. 말할 수 없다면 그것은 설계가 정해지지 않았다는 뜻이다.
- [ ] $\rho_k$를 적고 $\sum_k \rho_k$의 수렴 여부를 판정할 수 있는가. 판정할 수 없다면 무엇이 빠져서인가.
- [ ] "붕괴한다"고 쓸 때 표 6-3의 네 번호 중 어느 것인지 매번 지정했는가.
- [ ] 필터 $\phi$가 $B_s$를 몇 배로 만드는지 계산했는가($M$개 생성 + 후보당 판정 비용).
- [ ] 식 (6-2)를 인용하기 전에 네 전제(처음부터 재학습·공변량 동결·well-specified/ridgeless·무필터)가 내 시스템에서 성립하는지 확인했는가.
- [ ] **Rosetta**: "warm-up / 사전 컴파일 ↔ sleep 라운드" 대응에서 sleep 라운드만 갖는 성질을 말할 수 있는가 — warm-up은 품질을 바꾸지 않지만 sleep 라운드는 바꾸며, **나쁜 쪽으로도 바꾼다**. 그리고 "checkpoint 크기 ↔ $C_{\text{cap}}$" 대응이 accumulate에서 왜 깨지는지(상태가 가중치가 아니라 보존 코퍼스로 옮겨간다) 설명할 수 있는가.

---

## 다음 장으로

이 장은 $\mathcal{R}_k$를 **무엇으로 채울 것인가**에 답했고, 그 답의 유효 범위가 전제 (i) — 라운드마다 $\Theta_k$를 처음부터 다시 학습한다 — 에 걸려 있음을 보였다. 실제 시스템은 그렇게 하지 않는다. 살아 있는 checkpoint를 라운드마다 증분 갱신한다. 그 순간 이 장의 계산에 없던 실패 양식이 등장한다: 새 데이터를 배우면서 **이전에 알던 것을 잃는 것**이다.

ch07이 그 실패를 다룬다. 그리고 그 대응책 중 하나가 이 장의 accumulate와 사촌이라는 사실 — 옛 데이터를 코퍼스에 계속 섞어 넣는 것 — 이 두 장을 잇는다. ch07은 그 사촌 관계를 정확히 어디까지 밀 수 있는지, 그리고 옛 데이터를 실물로 보관하지 않고도 같은 효과를 낼 수 있는지를 묻는다.
