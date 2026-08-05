# ch08. CLS 이론과 생물학적 sleep — 은유의 출처

> **이 장의 목표** — 이 장을 마치면 (1) Complementary Learning Systems가 무엇을 주장하고 무엇을 주장하지 않는지 한 문단으로 말할 수 있고, (2) SWS와 REM을 서로 다른 outer loss를 가진 두 개의 오프라인 라운드로 번역할 수 있으며, (3) 생물학적 replay와 deep learning의 replay를 갈라 놓을 수 있고, (4) 이 책의 corpus가 생물학에서 실제로 이송한 것과 이름만 빌린 것을 구분해 말할 수 있다.
>
> **왜 필요한가** — Part II의 두 장이 이 장을 직접 요구한다. ch21 `Language Models Need Sleep`(2606.03979)은 §2 전체를 SWS/REM 분업과 hippocampus–neocortex 대화의 서술에 쓰고 그 위에 자기 기제를 세운다 [LM Need Sleep §2]. ch14 `Sleep-time Compute`(2504.13171)은 제목과 본문에서 sleep이라는 단어를 170회 쓰면서(vendored 원문 전문 계수, 08.5) 생물학 문헌을 한 편도 인용하지 않는다. 같은 이름 아래 두 논문이 서로 다른 것을 하고 있다는 판정은 생물학 쪽 원본을 알아야만 내릴 수 있다. ch07의 replay·interleaving도 계보상 여기서 출발한다.
>
> **NM과의 관계** — Neural Memory ch17이 Google Sleep 논문의 lifecycle을 다루면서 sleep이라는 단어를 이미 썼지만 생물학 배경은 한 문단으로 넘겼다. 그 배경을 이 장이 소유한다. inner/outer loop, backward pass, optimizer state는 전제한다(→ NM ch02). 이 장이 더하는 것은 **오프라인 위상이 무엇을 근거로 필요하다고 주장되었는가**의 원출처, 그리고 그 근거가 LLM으로 이송될 때 어디서 끊기는가의 경계선이다.

---

## 08.1 왜 두 시스템인가 — Complementary Learning Systems

**Complementary Learning Systems(CLS)**는 하나의 학습 시스템이 빠른 개별 경험 기억과 느린 일반화를 동시에 감당할 수 없으므로 뇌가 두 개의 상보적 시스템을 가진다고 설명하는 이론이다. McClelland, McNaughton, O'Reilly가 1995년 *Psychological Review*에 낸 논문이 그 원전이다(이 책의 corpus에서는 vendored 원문이 아니라 후속 논문들의 인용으로만 접근한다 — 아래 모든 CLS 서술은 인용하는 쪽 논문에 귀속시킨다).

**hippocampus/neocortex 분업**은 CLS가 제시하는 그 두 시스템의 역할 분배다. Deep Generative Replay는 이렇게 요약한다: "The hippocampal system rapidly encodes recent experiences, and the memory trace that lasts for a short period is reactivated during sleep or conscious and unconscious recall. The memory is consolidated in the neocortex through the activation synchronized with multiple replays of the encoded experience" [DGR §1, L50-59]. ch21의 `LM Need Sleep`은 같은 분업을 더 날카롭게 쓴다: "The hippocampus serves as a high-fidelity temporary storage system, capable of rapidly encoding specific daily experiences. In contrast, the neocortex is a vast, long-term repository better suited for the gradual learning of generalized rules and semantic knowledge from these experiences" [LM Need Sleep §2].

이 분업의 요점은 속도가 아니라 **간섭**이다. 빠르게 쓰는 시스템은 새 항목이 옛 항목을 덮어쓰는 것을 감수해야 하고, 느리게 쓰는 시스템은 여러 경험에 걸친 통계를 뽑아내는 대신 한 번의 경험을 즉시 담지 못한다. 두 요구를 한 파라미터 집합에 얹으면 둘 다 망가진다. CLS는 그 충돌을 아키텍처로 푼 것이다.

> **[해설]** 이 책의 세 층으로 읽으면 hippocampus는 $E$(external store) 자리에, neocortex는 $\Theta$(slow weights) 자리에 놓인다. 빠른 쓰기·낮은 간섭·제한된 용량은 $E$의 성질이고, 느린 쓰기·높은 일반화·배포 아티팩트의 변경은 $\Theta$의 성질이다. **다만 이 대응은 이 책의 재구성이지 CLS의 주장이 아니다.** CLS는 1995년의 심리학·연결주의 이론이고 external store도 slow weights도 그 어휘에 없다. 이 장은 대응을 세우되 대응을 증거로 쓰지 않는다.

CLS가 남긴 실용적 처방은 **interleaving**이다. 여러 과제를 하나의 학습 집합에 섞어 제시하면 순차 학습이 일으키는 덮어쓰기를 피할 수 있다. PLOS 논문은 이 처방의 출처를 명시적으로 CLS로 지목한다: "interleaved training was originally construed to be an approximation to what the brain may be doing during sleep to consolidate memories; spontaneously reactivating memories from multiple interfering tasks in an interleaved manner" [PLOS Sleep Introduction, L106-108].

여기서 이 책 전체를 지배하는 잔여물이 하나 남는다. 같은 논문이 곧바로 이어 쓴다: "explicit use of interleaved training, in contrast to memory consolidation during biological sleep, imposes the stringent constraint that the original training data be perpetually stored for later use and combined with new data to retrain the network" [PLOS Sleep Introduction, L109-113]. **interleaving은 작동하지만 과거 데이터의 영구 저장을 요구한다.** CLS는 뇌가 그 저장 없이 같은 일을 한다고 말했고, interleaving은 그 주장을 저장이 있는 형태로만 재현했다. 이 간극이 Θ-경로 전체의 문제의식이다.

> **[해설]** systems 독자에게 이 잔여물은 익숙한 모양이다. 과거 데이터의 영구 저장은 곧 per-user 또는 per-task 상태량이고, 식 (A)의 분자에 들어가는 것이 아니라 $C_{\text{cap}}$에 들어간다. interleaving을 그대로 LLM에 옮기면 사용자마다 원본 대화를 통째로 보관해야 한다는 뜻이 된다 — 이것은 성능 문제가 아니라 저장·프라이버시·수명 관리 문제다. 이후 모든 replay 계열이 하려는 일은 이 $C_{\text{cap}}$을 줄이거나 없애는 것이다(→ ch07).

---

## 08.2 sleep의 두 상태 — SWS와 REM

**SWS(slow-wave sleep)**는 NREM sleep의 가장 깊은 단계로, 동기화된 고진폭·저주파 신경 활동으로 특징지어지는 상태다. **REM(rapid eye movement) sleep**은 각성 상태에 가까운 고주파·저진폭 뇌파로 특징지어지고 꿈과 가장 강하게 연관되는 상태다. 두 정의 모두 [LM Need Sleep §2]의 서술을 따른다.

기능 분업에 대한 주장은 다음과 같이 귀속시킨다. [LM Need Sleep §2]는 SWS에 두 기능을 배정한다 — synaptic homeostasis("globally downscales synaptic strengths to counteract the net increase in connectivity from waking experiences")와 memory consolidation("the transformation of fragile, recent experiences into stable, long-term knowledge"). 그리고 consolidation이 raw data의 replay가 아니라고 명시한다: "this transfer does not simply replay raw data; instead, it re-architects the knowledge acquired during waking hours, extracting abstractions and integrating them into a cohesive semantic network" [LM Need Sleep §2]. REM에는 다른 기능을 배정한다 — "the selective strengthening of newly formed synapses and the integration of new information with pre-existing emotional and semantic networks" [LM Need Sleep §2].

> **[해설]** 이 문단이 이 장에서 systems 독자에게 가장 값싼 이득이다. **SWS와 REM은 "sleep 라운드 하나"가 아니라 목적함수가 다른 두 라운드다.** 프레임 어휘로 옮기면, 하나의 $k$ 안에서 $\mathcal{L}$이 두 번 바뀐다 — 식 (U-$\Theta$)의 $\mathcal{L}(\mathcal{R}_k; \Theta_k)$에서 $\mathcal{L}$이 위상의 함수가 된다. 그래서 생물학을 실제로 인용하는 논문들은 예외 없이 **두 개 이상의 sleep 목적함수**를 갖는다: ch21은 Knowledge Seeding과 Dreaming 두 단계를(두 단계의 정의는 → ch21), 08.4의 PAD는 NREM과 REM에 서로 다른 outer loss를 둔다. 반대로 sleep을 이름으로만 쓰는 논문은 오프라인 위상이 하나다. 오프라인 위상의 **개수**가 은유를 실제로 이송했는지 아닌지의 값싼 판별기다.

한 가지를 이 장에서 못 박는다. SWS/REM 분업 서술의 근거는 신경과학 문헌이고, 이 책의 corpus는 그 문헌을 직접 갖고 있지 않다. corpus가 가진 것은 두 편의 **계산 모델**이다 — PLOS 논문(08.3)과 PAD(08.4). 둘 다 동물 실험이 아니라 시뮬레이션이며, 생물학적 권위는 인용으로만 확보된다. PLOS 논문은 자기가 모델링한 것이 REM뿐임을 밝힌다: "In vivo, activity of the neocortical neurons during REM sleep is low-synchronized and similar to baseline awake activity. Therefore, to simulate REM sleep-like activity in the model, the rewarded STDP rule was replaced by unsupervised STDP" [PLOS Sleep Results, L292-296]. **NREM은 이 논문에서 모델링되지 않는다.** 그런데 이 논문이 인용하는 declarative memory consolidation 문헌은 NREM에 그 기능을 배정한다. 즉 corpus 안에서 SWS 쪽 증거는 계산 모델로 뒷받침되지 않은 채 인용으로만 흘러간다.

---

## 08.3 생물학적 replay — 데이터 없이 다시 도는 것

**생물학적 replay**는 각성 중 활동했던 신경 앙상블이 오프라인 상태에서 저장된 감각 입력 없이 자발적으로 재활성화되어 시냅스를 변화시키는 현상이다. 이 정의에서 결정적인 부분은 "저장된 입력 없이"다. deep learning의 replay는 같은 이름을 쓰지만 다른 것을 가리킨다. PLOS 논문이 두 용법을 직접 갈라 놓는다 — deep learning 쪽 replay를 "storing a subset of previous veridical inputs and mixing them with more recent inputs to update the networks"로 규정하고, 자기 기여를 "the data intensive process of storing old data and using them for retraining can be avoided" 로 규정한다 [PLOS Sleep Discussion, L671-677].

> **[평가]** 이 구분은 용어 위생의 문제가 아니라 이 책의 판정에 직접 들어간다. **"replay를 한다"는 서술은 층도 데이터 출처도 특정하지 않으므로 그 자체로는 정보가 없다.** 어느 층에 쓰는지($\Theta$인가 $E$인가), $\mathcal{R}_k$를 어디서 얻는지(저장된 원본인가, 생성 모델인가, 모델 자신의 자발 활동인가)를 밝히지 않은 replay 주장은 이 책에서 미완성 서술로 취급한다.

### 08.3.1 PLOS 실험이 실제로 보인 것

corpus에서 오프라인 위상이 각성 위상이 만들 수 없는 가중치 상태를 만든다는 것을 가장 직접적으로 보인 것은 Golden·Delanois·Sanda·Bazhenov의 PLOS Computational Biology 논문(2022, doi 10.1371/journal.pcbi.1010628)이다. 이 논문은 수식에 번호를 붙이지 않으므로, 아래 인용은 절 이름과 vendored 원문 줄번호를 병기한다.

설정은 작다. 842개의 spiking 뉴런, 7×7 입력층 · 28×28 은닉층 · 3×3 출력층, 연속적으로 학습되는 시냅스는 은닉→출력의 흥분성 6272개뿐이다 [PLOS Sleep Methods: Network structure, L743-755; L995]. 과제는 두 개의 상보적 foraging 과제이고 성능 지표는 pattern discriminability로 chance가 0.5다 [PLOS Sleep Results, L166-167]. 모든 수치는 최소 10회 시행의 평균 ± 표준편차다 [L168-169].

네 조건의 결과가 이 장의 뼈대다.

| 조건 | Task 1 | Task 2 | 과거 데이터 저장 |
|---|---|---|---|
| Task 1 단독 학습 후 | 0.70 ± 0.02 | 0.53 ± 0.02 | — |
| Sequential (T1→T2) | 0.52 ± 0.02 | 0.69 ± 0.03 | 불필요 |
| Interleaved$_{T1,T2}$ (고전 interleaving) | 0.68 ± 0.03 | 0.65 ± 0.04 | **필요** |
| Interleaved$_{S,T2}$ (sleep 교대) | **0.70 ± 0.03** | **0.68 ± 0.05** | 불필요 |

표 8-1. PLOS Sleep의 핵심 네 조건 [Results, L228-229, L275-276, L282, L304-305].

Sequential 학습은 옛 과제를 chance로 떨어뜨린다(0.70 → 0.52). sleep을 새 과제 학습과 100 movement cycle 단위로 교대시키면 옛 과제가 전혀 떨어지지 않는다(0.70 → 0.70). 그리고 결정적으로, "Importantly, no training on Task 1 was performed at any time during Interleaved$_{S,T2}$" [PLOS Sleep Results, L303-304]. 옛 과제 데이터는 한 번도 다시 제시되지 않았다.

기제에 대한 논문 자신의 설명은 다음과 같다: "Having a sleep phase after a short period of Task 2 training enables spontaneous forward replay between hidden and output layers (H->O) that preferentially benefits the strongest synapses. Thus, if Task 1 synapses are still strong enough to maintain replay, they are replayed and weights are increased" [PLOS Sleep Results, L307-311]. 각성 중에는 입력층이 현재 과제의 통계를 은닉층에 강제하므로 형성되는 상관은 새 과제의 것이다. 입력을 끊고 무구조 noise로 대체해야만 **가중치 행렬 자신이** 어떤 상관을 되살릴지 고를 수 있다. 그 선택은 가중치만의 성질이고, 그래서 각성 절차로는 재현되지 않는다.

이 지점에서 계보상의 공백을 하나 기록한다. 저장 없이 $\mathcal{R}_k$를 만든다는 목표에는 두 개의 독립된 해법이 있다 — 생성 모델을 따로 학습시켜 과거 분포를 그리게 하는 것(Deep Generative Replay, → ch07)과, 모델 자신을 noise로 돌려 나오는 활동을 그대로 쓰는 것(이 논문)이다. PLOS Sleep은 **DGR을 인용하지 않는다.** 참고문헌 77편은 대부분 신경과학·심리학·spiking network 모델링과 고전 연결주의 망각 문헌이고, 기계학습 쪽 인용은 EWC [Results, L546-547], deep learning replay 리뷰와 지속학습 문헌 [Introduction, L102-105; Discussion, L673-674], meta-learning 한 건 [Discussion, L647-648], 그리고 같은 그룹이 이 기제를 feedforward ANN에 옮긴 자기 인용 [Discussion, L677-679]에 한정된다. 같은 문제의 두 해법이 서로를 보지 않고 나란히 자란 것이며, 이 침묵은 Part II의 경로 간 침묵 지도에서 되풀이되는 형태의 첫 사례다(→ ch11).

sleep을 정적 보호와 구분하는 통제도 있다. Task 1 학습 후 상위 x% 시냅스를 얼려 두는 baseline은 1%에서 0.54/0.68, 5%에서 0.65/0.61, 10%에서 0.70/0.53을 낸다 [PLOS Sleep S6 Fig, L1181-1189]. 세 점 모두 두 축을 맞바꾸는 곡선 위에 있고, sleep의 0.70/0.68은 세 점 전부를 두 축에서 동시에 지배한다. 논문은 이 baseline을 EWC와 같은 계열로 명시적으로 연결한다 [PLOS Sleep Results, L517-547] — 다만 EWC 자체는 실행하지 않는다(→ ch07).

### 08.3.2 이 실험에 불리한 사실 — 반드시 함께 읽어야 하는 것

**강한 baseline이 방법과 거의 같다.** 표 8-1에서 Interleaved$_{T1,T2}$는 0.68/0.65이고 sleep은 0.70/0.68이다. 차이는 0.02와 0.03이며 표준편차는 0.03–0.05다. 논문은 본문에서 sleep이 고전 interleaving을 "exceeding"한다고 쓰지만 [PLOS Sleep Results, L305-306], Discussion에서 스스로 정정한다: "Although classical interleaved training of the old and new tasks showed similar performance results in our model as interleaving new task training with sleep, **we believe the latter to be superior on the following theoretical grounds**" [PLOS Sleep Discussion, L654-655, 강조는 이 책]. 이어지는 근거는 전부 이론적이다 — 고전 interleaving은 각 주기마다 단일 과제 비용함수를 쓰므로 이중 과제 최적점 주위에서 진동하는 반면 sleep은 두 과제가 섞인 replay를 지원할 수 있다는 것 [L655-670]. 그런데 replay 내용을 실제로 분석한 실험은 이 논문에 없다. 혼합 replay 주장은 인용으로 수입된다.

> **[평가]** 이 논문이 **실증한** 우위는 정확도가 아니라 **저장의 제거**다. 과거 데이터가 이미 있다면 이 논문은 sleep을 택할 정확도상의 이유를 하나도 제공하지 않는다. Θ-경로를 옹호할 때 이 논문을 근거로 들 수 있는 것은 "가중치만으로 $\mathcal{R}_k$를 만들 수 있다"까지이고, "그렇게 만든 $\mathcal{R}_k$가 저장된 데이터보다 낫다"까지는 아니다. 이 두 문장을 붙여 쓰는 서술은 이 책에서 결함이다.

나머지 제약도 함께 기록한다. 이 논문에는 **어떤 통계 검정도 없다** — 모든 수치가 10회 이상 시행의 평균 ± 표준편차이고, "significantly"라는 단어를 최소 세 번 쓰면서 검정 이름도 p-값도 제시하지 않는다 [L168-169; L418, L444, L506]. 과제는 **두 개뿐이며** 같은 네 방향 어휘에서 뽑은 대칭 상보 과제다 — 논문 자신이 유사성을 특징으로 강조한다 [L233-237]. 따라서 반복 sleep 라운드에 걸친 열화율 $\rho$는 **측정되지 않았다**: 세 번째 과제도, 두 번째 Interleaved 위상도 실행되지 않았다. 보호되는 성능 밴드는 chance 0.5에서 천장 0.70까지 **0.20 폭**이고 조건별 표준편차가 0.02–0.05로 밴드의 10–25%다. 과제·환경·평가 지표가 모두 저자 제작이며 표준 지속학습 벤치마크는 하나도 쓰이지 않았다 [L166-167, L1037-1095]. 그리고 sleep은 **복원하지 않고 보존한다** — "a single episode of new task training using reinforcement learning could quickly erase old memories to the point that they cannot be recovered by subsequent sleep" [PLOS Sleep Discussion, L639-640]. 각성 손상이 아직 부분적인 창 안에서만 오프라인 위상이 각성 위상이 못 하는 일을 한다.

마지막으로 비용이다. 이 논문은 **FLOPs·wall-clock·메모리 사용량·에너지를 하나도 보고하지 않는다.** $B_s$는 계산량이 아니라 시뮬레이션 시간으로만 존재한다 — 한 sleep 구간은 100 movement cycle이고 이는 직전 각성 구간과 정확히 같아 duty ratio가 1:1이다 [PLOS Sleep Results, L301-303; Methods: Simulated sleep, L985-986]. sleep 길이·비율·교대 횟수에 대한 sweep은 없다. §6.4의 규칙대로, 비용 4종이 비어 있으므로 이 논문을 근거로 "효율적"이라고 쓸 수 없다.

> **[해설]** 그럼에도 systems 독자가 이 논문에서 가져갈 구조적 사실이 하나 있다. sleep 위상을 구동하는 데 필요한 각성 유래 상태는 은닉 뉴런 784개의 목표 발화율 스칼라인데, Uniform-Noise Sleep 통제에서 이것을 **집단 평균 하나로 축약해도 결과가 유지된다**(0.67 ± 0.05 / 0.69 ± 0.03) [PLOS Sleep Results, L355-362; S4E/F, L1144-1150]. 즉 이 모델에서 각성→sleep 경계를 넘는 상태는 $O(1)$ 스칼라 하나로 줄어든다. 이것이 이 논문에서 나오는 **유일한** 용량 주장이고, gradient로 학습되는 모델로의 이송은 이 논문이 보이지 않았다.

---

## 08.4 dreaming과 표현학습 — 재생이 아니라 생성

**dreaming과 표현학습**은 오프라인 위상이 과거 경험을 재생하는 대신 **경험한 적 없는 입력을 생성**하고 그 생성물로 학습함으로써 표현의 질을 높인다는 가설이다. 08.3의 replay가 "있었던 것을 다시 돌린다"였다면 이쪽은 "없었던 것을 만들어 돌린다"다. Deperrois·Petrovici·Senn·Jordan의 *eLife* 논문 `Learning cortical representations through perturbed and adversarial dreaming`(2022, doi 10.7554/eLife.76384, 이하 PAD)이 이 가설의 계산 모델이다.

출발점은 관찰 하나다. Wake–Sleep·Helmholtz machine 계열은 오프라인 위상을 생성 모델 개선에 쓰지 않고 각성 중 입력 재구성에만 학습시키는데, "most dreams during REM sleep exhibit realistic imagery beyond past sensory experience" [PAD Introduction, L84-86]. 꿈은 겪은 것의 재생이 아니다. PAD는 그 사실을 잡음이 아니라 **계산의 원재료**로 삼는다.

### 08.4.1 한 사이클에 세 개의 outer loss

PAD의 한 wake–sleep 사이클 $k$는 세 위상으로 구성되고 위상마다 목적함수가 다르다. 논문 자신의 표현으로 "the three brain states only differ in their objective function and the presence or absence of external input" [PAD Results, L275-276].

- **Wake**: 실제 입력 $x_k$를 encoder에 통과시켜 latent을 얻고, 그것을 두 슬롯짜리 hippocampal buffer에 쓴다. 동시에 재구성 손실 + latent prior 정규화 + discriminator의 "real" 항으로 파라미터를 갱신한다 [PAD Eq. 1-5].
- **NREM**: buffer의 latent을 generator로 되그려 **가림(occlusion)을 씌운 뒤**, 그 훼손된 이미지를 다시 encode한 결과가 원래 latent과 같아지도록 encoder만 갱신한다 [PAD Eq. 6]. 저장된 latent이 정답이고 generator가 입력을 만든다.
- **REM**: 현재와 직전 사이클의 latent 두 개와 가우시안 noise를 볼록 결합해 $z' = \tfrac14 z + \tfrac14 z_{\text{old}} + \tfrac12 \epsilon$을 만들고, 그것을 generator로 그린 뒤 discriminator에게 "internal"로 판정하도록 학습시키면서 **generator에는 부호를 뒤집어 상승 방향으로** 갱신한다 [PAD Eq. 7-8, Figure 2c]. 논문의 핵심 생물학적 예측이 이 부호 전환이다: "identical local errors lead to opposing weight changes between Wake and REM sleep" [PAD Figure 7 caption].

> **[해설]** 프레임으로 옮기면 PAD는 (U-E)를 한 번, (U-$\Theta$)를 세 번 도는 시스템이다. 그런데 (U-E)의 $S(c_k; B_s)$는 encoder forward 한 번이고 $\mathrm{wr}$은 2슬롯 shift register이며, **$\mathrm{ret}(E,q)$가 존재하지 않는다** — 읽기는 질의가 아니라 라운드 첨자로만 이루어진다. 즉 PAD의 $E$는 검색 절반이 잘려 나간 store로, 오직 $\mathcal{R}_k$를 만들기 위해서만 존재한다. 그리고 평가 시점에는 $\hat y = W_{\text{ro}}\,\mathrm{enc}(x)$ 한 번의 forward가 전부다 — $W$층이 없고 $E$도 읽지 않는다. **오프라인 위상이 벌어들인 것이 전부 $\Theta$에 흡수되어 있으므로 wake 비용이 정확히 0만큼 늘어난다.** 이것이 상각 논증이 가장 좋아하는 모양이지만, 논문은 어느 쪽 비용도 재지 않는다.

> **[평가]** PAD가 자기 기여로 내세우는 것은 부호 전환이라는 생물학적 예측이다 — 각성과 REM에서 동일한 국소 오차가 반대 방향의 가중치 변화를 낳는다는 것. 계산 쪽에서 보면 이것은 한 파라미터 블록에 대한 gradient의 부호를 뒤집는 것, 즉 표준 GAN 학습 절차 그 자체다. 새 연산도, 새 커널도, 추가 상태도 없다. 이 사실을 이 책은 약점이 아니라 **이송 가능성의 지표**로 읽는다: 생물학 쪽에서 "각성과 sleep은 다른 종류의 가소성을 쓴다"로 서술되는 것이 계산 쪽에서는 "오프라인 라운드는 목적함수와 부호만 다르다"로 환원된다. 반대로 말하면, 이 라인이 생물학에서 가져올 수 있는 것은 대개 **스케줄과 목적함수의 배치**이지 새로운 갱신 primitive가 아니다. ch21의 두 단계 sleep이 기존 distillation과 RL 조각의 재배치로 구성되는 것도 같은 이유다(→ ch04, ch21).

이 논문의 기호는 이 책의 예약 기호와 정면 충돌한다. 원문을 대조해 읽을 독자를 위해 세 행만 못 박는다.

| 원 논문 기호 | 원문에서의 뜻 | 이 책 표기 |
|---|---|---|
| $E$, $E_z$, $E_d$ | encoder 신경망과 그 두 출력 [PAD Methods: Network architecture] | $\Theta^{\mathrm{enc}}$, $\Theta^{\mathrm{disc}}$ — 예약된 $E$(external store)는 **논문이 이름을 주지 않은 hippocampal buffer**에 배정 |
| $W$ | 선형 readout 가중치 $W \in \mathbb{R}^{10\times 256}$ [PAD Eq. 10] | $W_{\text{ro}}$ — 모델이 얼려진 뒤 학습되는 **평가용 probe**이며 fast weights가 아니다 |
| epoch | 데이터셋 1회 통과 [PAD Methods] | 한 wake–sleep 사이클 $k$는 **미니배치 하나**다. 논문 그림의 x축은 epoch이므로 $k$의 수는 그보다 훨씬 많다 |

표 8-2. PAD 표기 대응 — 예약 기호를 덮어쓰는 두 충돌과 시간 단위 충돌.

<!-- TODO-VERIFY: 한 epoch에 포함된 wake–sleep 사이클 수(= 미니배치 수). b=64 [PAD Methods, L1283]와 "One training epoch is defined by the number of mini-batches necessary to cover the whole dataset" [PAD Methods, L1620-1621]는 확인했으나, 학습 집합 크기는 vendored 원문에 인쇄되어 있지 않다 — 2026-08-06 감사에서 papers/stc/STC-T13.txt 를 "training set" / "50,000" / "images from" 으로 검색한 결과 L1278의 "usual split into a training set and a smaller test set"뿐이다. 남은 확인 방법: 게재본(doi 10.7554/eLife.76384) Methods 'Datasets' 절 또는 코드 저장소 github.com/NicoZenith/PAD 의 dataloader. 확인 전까지 B_s의 절대 수치를 본문에 쓰지 않는다. -->

### 08.4.2 무엇이 측정되었고 무엇이 손봐졌는가

평가는 표현의 **선형 분리도**다 — 모델을 얼린 뒤 latent 위에 선형 분류기를 학습시켜 얻은 테스트 정확도로 정의한다 [PAD Methods: Linear separability]. epoch 50 시점, 초기값 4종에 대한 평균 ± SEM은 다음과 같다 [PAD Appendix 1—table 1].

| 데이터셋 | PAD | w/o memory mix | w/o REM | w/o NREM | Wake only |
|---|---|---|---|---|---|
| CIFAR-10 | 58.25 ± 0.70 | 53.87 ± 0.85 | 46.00 ± 0.43 | 58.00 ± 0.34 | 42.25 ± 0.54 |
| SVHN | 78.92 ± 0.40 | 60.87 ± 5.07 | 42.30 ± 1.51 | 73.25 ± 0.22 | 41.93 ± 0.65 |

표 8-3. PAD와 병리 조건의 최종 선형 분리도 [PAD Appendix 1—table 1].

REM을 제거하면 두 데이터셋 모두 크게 무너진다(58.25 → 46.00, 78.92 → 42.30). REM을 두되 여러 기억의 혼합도 noise도 없이 단일 기억만 재생하면 역시 떨어진다(53.87, 60.87) [PAD Results, L547-549]. 논문의 결론은 여기서 나온다 — 표현을 만드는 것은 오프라인이라는 사실 자체가 아니라 **오프라인에서 무엇을 생성하는가**다.

그러나 표를 그대로 읽으면 논문의 두 번째 주장은 훨씬 약하다. **CIFAR-10에서 w/o NREM은 58.00 ± 0.34로 PAD의 58.25 ± 0.70과 오차 안에서 같다.** NREM의 기여는 깨끗한 이미지의 선형 분리도가 아니라 가림이 있을 때의 견고성으로만 나타난다 [PAD Results, L571-577]. 논문은 이 구분을 지키지만, 표 없이 요약만 읽으면 "네 구성요소가 모두 기여한다"로 읽히기 쉽다.

더 무거운 사실이 Methods에 있다. **절제 대조군은 손봐진 상태로 비교된다.** REM을 제거한 모델은 "we observed a decrease of linear separability after some (>25) epochs"라는 이유로 encoder 출력에 $\epsilon \sim \mathcal{N}(0,\,0.5 I)$를 더한 변형 재구성 손실로 학습되고 [PAD Eq. 12], 나아가 "we reduced the effect of NREM by scaling down its loss with a factor of 0.5"라는 조정까지 받는다 [PAD Methods: Modifications specific to pathological models]. 논문은 이 조정을 공정 비교를 위한 것이라고 밝히며 숨기지 않는다.

> **[평가]** 조정을 밝힌 것은 정직하다. 그러나 결과의 지위는 달라진다. 표 8-3의 w/o REM 열은 "REM을 뺀 PAD"가 아니라 **"REM을 빼고 저자가 손본 다른 모델"**이다. 이 책이 절제 수치를 인용할 때는 그 사실을 함께 옮긴다. 그리고 PLOS 논문과 마찬가지로 PAD에도 **검정 이름이나 p-값이 없다** — 본문은 "significantly"를 반복해서 쓰지만 vendored 원문 어디에도 검정이 없다. 두 편의 생물학 논문 모두, 이 라인의 근거로 인용되는 바로 그 논문들이, 유의성 주장을 검정 없이 한다.

**저자 자신의 통제가 논문 제목의 성분을 지지하지 않는다.** PAD의 REM은 혼합된 episodic memory와 자발 활동을 함께 주입해 구동되는데, 저자는 그 혼합이 필요 없다고 보고한다: "we do not observe significant differences between using a combination of episodic memories with spontaneous activity or only using spontaneous activity" [PAD Discussion, L1034-1036; Appendix 1—figure 4]. 표 8-3의 w/o memory mix는 혼합만 없앤 조건이 아니라 **noise까지 없애고 단일 기억만 되돌린** 조건이다 [PAD Results, L547-549]. 두 결과를 함께 읽으면 REM에서 결정적인 성분은 기억의 혼합이 아니라 자발 활동의 주입이다. 논문 제목이 가리키는 성분(mixed episodes로부터의 dreaming)은 통제 대비 효과를 보이지 않았다.

**단계 순서도 무관하다.** "our model does not show significant differences in performance when the order of sleep phases is switched" [PAD Discussion, L1046-1047; Appendix 1—figure 5]. 저자는 생물학 쪽에서 NREM→REM 순서가 중요하다고 가설된다는 점(sequential hypothesis)을 인정하면서, 모델의 순서 독립성을 단계별 시냅스 변화가 작기 때문으로 돌린다 [PAD Discussion, L1050-1055].

**표현 품질의 지표는 하나뿐이다.** 측정된 것은 linear readout 정확도 하나이고, 저자는 이것을 "an obvious simplification with regard to cortical processing"이라고 인정한다 [PAD Discussion, L1039-1041]. downstream 과제 성능은 측정되지 않았다. 그 하나뿐인 지표의 절대값도 낮다 — CIFAR-10에서 약 59%다 [PAD Results, L534; Figure 4c].

> **[평가]** 세 사실을 합치면 이 논문을 인용할 수 있는 범위가 좁아진다. PAD는 **기제의 존재 증명**이지 성능 주장이 아니다. 절대 성능을 근거로 이 라인의 유망함을 말하는 데 이 논문을 쓰는 것은 오용이다. 그리고 sleep 단계를 개수와 순서까지 모사하려는 LLM 설계에 대해 이 논문은 순서의 필요성을 지지하지 않는다. 08.2에서 오프라인 위상의 **개수**를 판별기로 쓴 것은 은유가 실제로 이송되었는지를 재는 지표이지, 개수와 순서가 성능을 낳는다는 주장이 아니다.

비용은 다시 비어 있다. PAD는 FLOPs·GPU 시간·wall-clock·메모리 사용량을 어느 것도 보고하지 않는다. $C_{\text{cap}}$은 구조적으로 **2슬롯**으로 고정되고 보존 기간은 정확히 한 사이클이다 — `Z_old ← Z`가 매 사이클 덮어쓴다 [PAD Algorithm 1]. 누적되는 것은 아무것도 없다. $\rho$에 가장 가까운 관측은 방향성뿐이다: NREM이 없는 조건에서 "linear separability tends to decrease after many training epochs" [PAD Appendix 1, 'Linear classification performance'] — 즉 오프라인 위상 하나를 빼면 반복 라운드가 성능을 **깎기 시작한다**.

---

## 08.5 은유의 이송 한계

**은유의 이송 한계**란, 생물학적 sleep 서술에서 계산 시스템으로 실제로 옮겨지는 구조적 주장과, 이름만 옮겨지고 근거는 남겨진 부분 사이의 경계를 말한다. 이 절이 이 장의 결론이다.

먼저 무엇이 이송되는지 정리한다. 08.3과 08.4가 지지하는 것은 다음 네 문장이며, 각각 **적어도 하나의 계산 모델에서 실증되었다**.

| 이송되는 구조적 주장 | 근거 |
|---|---|
| 식 (U-$\Theta$)의 $\mathcal{R}_k$는 데이터셋일 필요가 없다. 이 설정에서는 모델을 noise로 돌려 만든 스트림으로 충분했고, $\mathcal{H}_k$는 $\Theta_k$ 안에만 있어도 된다 | PLOS Sleep Results, L300-311 |
| 각성→오프라인 경계를 넘는 보조 상태는 $O(1)$ 스칼라 하나로 줄어든다 | PLOS Sleep S4E/F, L1144-1150 |
| 오프라인 위상의 가치는 "오프라인"이라는 스케줄이 아니라 **거기서 무엇을 생성하는가**에서 나온다. 단 PAD에서 결정적인 성분은 기억의 혼합이 아니라 자발 활동의 주입이다 | PAD Appendix 1—table 1 (w/o memory mix 53.87 vs PAD 58.25) + Appendix 1—figure 4(혼합 유무는 차이 없음) |
| 오프라인 위상이 하나가 아니라 목적함수가 다른 둘일 때 서로 다른 성질(정확도 / 섭동 견고성)이 갈라져 나온다 | PAD Results, L534-577; Appendix 1—table 1 |

표 8-4. 생물학 두 편에서 이송 가능한 구조적 주장.

이송되지 않는 것은 더 길다. PLOS 모델에는 **loss도 gradient도 optimizer도 backward pass도 없다** — 갱신은 국소 eligibility trace를 낀 Hebbian 곱과 열 합을 고정하는 정규화 사영이고, 학습률 $\eta_\Theta$는 독립된 존재조차 아니다 [PLOS Sleep Methods: Synaptic plasticity, L861-963]. 그 기제는 transformer가 갖지 않은 장치에 의존한다: spike timing이라는 상관 신호, 뉴런별 목표 발화율 항상성, 열 합을 보존하는 heterosynaptic 정규화, 균형 잡힌 전방 억제. 논문이 sleep의 우위를 설명할 때 기대는 것이 바로 이 정규화 제약이다 [L509-517]. 제약을 빼면 그 설명에는 대체물이 없다. 규모도 이송되지 않는다 — 연속적으로 가소적인 시냅스 6272개, 과제 2개, 밴드 0.20이다. PAD는 gradient를 쓰지만 언어도 토큰도 attention도 없고, 저장은 2슬롯이며, 두 논문 모두 **replay·dream의 내용이 의미상 옳은지를 한 번도 검사하지 않는다**. LLM에서 이 검사 누락에 대응하는 실패 — 유창하지만 거짓인 생성물로 $\Theta$를 갱신하는 것 — 이 Θ-경로의 중심 위험이며(→ ch06), 생물학 쪽 두 편은 그것에 대해 아무 말도 하지 않는다.

여기서 이 책이 서술 대상으로 삼는 사실을 못 박는다. **E-경로의 창시 논문은 제목에 sleep을 빌려 쓰면서 생물학·CLS 문헌을 한 편도 인용하지 않는다.** `Sleep-time Compute`(2504.13171)의 참고문헌은 22편이고 [Letta STC References, pp.13-15] 그 안에 McClelland도, hippocampal replay도, sleep consolidation 신경과학도, dreaming 문헌도, EWC/replay 계열도 없다. 본문은 sleep이라는 단어를 170회 쓰지만(vendored 원문 전문, 대소문자 무시 계수) 그 단어를 도입하는 문장은 순전히 운영적이다: "inference is done between interactions with the model while it would otherwise be idle in sleep-time" [Letta STC §1]. **여기서 sleep은 consolidation이 아니라 유휴를 뜻한다.**

대조가 선명하다. Θ-경로의 `Language Models Need Sleep`(2606.03979)은 §2 한 절 전체를 SWS/REM 분업과 hippocampus–neocortex 대화에 쓰고, synaptic homeostasis와 systems consolidation을 각각 인용과 함께 세운 뒤 자기 기제를 그 위에 얹는다 [LM Need Sleep §2]. 같은 단어를 쓰는 두 논문 중 하나는 생물학을 논증에 넣었고 다른 하나는 이름만 가져왔다.

> **[평가]** 이 차이를 완곡하게 처리하지 않는다. 이름이 같다는 것은 **공유된 이론적 약속이 아니다.** `Sleep-time Compute`에 생물학 인용이 없는 것은 연대가 강제한 침묵이 아니다 — CLS는 1995년, PLOS는 2022년, PAD는 2022년이고 이 논문은 2025년이다. 선택된 침묵이다. 그리고 그 선택은 정당하다: 이 논문이 하는 일은 질의 전에 문맥에 대해 추론을 미리 돌려 텍스트를 만들어 두는 것이고, 여기에 CLS의 두 시스템 논증은 필요하지 않다. **결함은 인용의 부재가 아니라 이름의 재사용이다.** 서로 다른 세 경로가 같은 단어를 쓰기 때문에 독자는 그것들이 같은 가설의 변형이라고 읽게 되고, 이 책이 판별식을 따로 세워야 하는 이유가 정확히 그것이다(→ ch11).

반대 방향의 침묵도 기록하되 등급을 구분한다. PLOS Sleep과 PAD는 E-경로도 W-경로도 인용하지 않는다. 그러나 두 편 모두 2022년 논문이고 이 책 corpus의 E-경로 논문은 전부 2023년 이후다. **이쪽 침묵은 연대가 강제한 것이지 선택된 것이 아니다.** 두 종류를 섞어 세면 침묵 지도가 무의미해진다 — 이 책은 연대가 가능하게 한 인용이 실제로 이루어졌는지만 발견으로 취급한다(→ ch11, ch27). 그 기준을 적용하면 이 장의 두 생물학 논문은 무죄이고, `Sleep-time Compute`은 그렇지 않다.

> **[해설]** Rosetta로 옮기면 차이가 즉시 드러난다. `Sleep-time Compute`의 오프라인 위상은 **warm-up 또는 사전 컴파일**과 같은 자리에 있다 — 유휴 시간에 미리 해 두고, 배포 아티팩트는 그대로다. 반면 CLS·PLOS·PAD가 말하는 consolidation은 **모델 재배포 1회**와 같은 자리에 있다 — 끝나고 나면 서빙되는 가중치가 다른 것이 된다. 이 둘은 상각 구조도, 배칭 가능성도, 롤백 절차도 같지 않다. 앞의 것은 캐시 무효화 문제이고 뒤의 것은 배포 문제다. 같은 단어 하나가 이 구분을 덮고 있다.

---

## (state, update, cost) 정리

이 장의 개념을 세 층 프레임에 배치한다. 비용 칸은 **논문이 실제로 보고한 것만** 적는다.

| 개념 | 이 책의 층 | 갱신식 | 시간척도 | 비용에 대해 알려진 것 |
|---|---|---|---|---|
| hippocampus (CLS) | $E$에 대응(이 책의 재구성) | — | 경험 단위 | 원전 미확보. corpus 인용으로만 접근 |
| neocortex (CLS) | $\Theta$에 대응(이 책의 재구성) | (U-$\Theta$) | 다수 replay에 걸쳐 | 동일 |
| interleaving 처방 | $\Theta$ | (U-$\Theta$), $\mathcal{R}_k$ = 저장된 원본 | 학습 라운드 | $C_{\text{cap}}$ = 과거 데이터 전체(정성적 서술) |
| SWS | $\Theta$ | (U-$\Theta$), 목적 = homeostasis + consolidation | sleep 라운드 | 논문에 없음 |
| REM | $\Theta$ | (U-$\Theta$), 목적 = 선택적 강화 + 통합 | sleep 라운드 | 논문에 없음 |
| 생물학적 replay | $\Theta$ | (U-$\Theta$), $\mathcal{R}_k = \mathrm{gen}(\Theta_k;\,B_s)$ — 저장 없음 | sleep 라운드 $k$ | $B_s$: 100 movement cycle, wake와 1:1. FLOPs·시간·바이트 전부 논문에 없음 |
| PLOS sleep 위상 | $\Theta$만 ($W=\varnothing$, $E=\varnothing$) | (U-$\Theta$), gradient 아님 | $k$ = 교대 1회 | $L_w$ 변화 없음(구조적). $\rho$ 미측정(과제 2개) |
| PAD Wake write | $E$ (2슬롯) | (U-E), $S(c;B_s)$ = forward 1회 | 사이클 $k$ | $C_{\text{cap}}$ = 2 latent 슬롯, 보존 1 사이클 |
| PAD NREM | $\Theta^{\mathrm{enc}}$ | (U-$\Theta$), $\mathcal{R}_k$ = 저장 latent + 가림 | 사이클 $k$ | 논문에 없음 |
| PAD REM | $\Theta^{\mathrm{disc}}$ 하강 / $\Theta^{\mathrm{gen}}$ **상승** | (U-$\Theta$), 부호 반전 | 사이클 $k$ | 논문에 없음 |

표 8-5. 이 장의 개념과 세 층 프레임의 대응.

두 가지가 이 표에서 바로 읽힌다. 첫째, 이 장의 개념 중 $W$층에 놓이는 것이 **하나도 없다.** 생물학 쪽 두 모델 모두 fast weights를 갖지 않는다 — 질의 시점에 움직이는 상태가 없고, 오프라인에서 번 것은 전부 $\Theta$에 흡수된다. W-경로는 생물학에서 오지 않았다(→ ch17). 둘째, 비용 칸의 대부분이 "논문에 없음"이다. **이 라인의 생물학적 뿌리는 비용 회계를 하나도 물려주지 않았다.** 식 (A)의 어느 항도 이 장의 논문들에서 조달할 수 없다.

---

## Worked micro-example — 밴드 정규화 점수와 상각 불능

PLOS Sleep의 네 조건을 손으로 다시 계산해, "sleep이 interleaving을 이겼다"는 문장이 얼마나 큰 주장인지 재어 본다. 필요한 수치는 표 8-1과 S6뿐이고 계산은 나눗셈 네 번이다.

**1단계 — 밴드 정규화.** 이 과제의 chance는 0.50이고 단일 과제 천장은 Task 1이 0.70, Task 2가 0.69다 [PLOS Sleep Results, L166-167, L228-229, L276]. 따라서 보호 대상 밴드는 각각 0.20과 0.19다. 정규화 점수를 $g(x) = (x - 0.50)/(\text{천장} - 0.50)$으로 두면 $g=1$이 "단일 과제 학습과 같음", $g=0$이 "chance"다.

| 조건 | Task 1 $g$ | Task 2 $g$ | 합 |
|---|---|---|---|
| Sequential | $0.02/0.20 = 0.10$ | $0.19/0.19 = 1.00$ | 1.10 |
| Interleaved$_{T1,T2}$ | $0.18/0.20 = 0.90$ | $0.15/0.19 = 0.79$ | 1.69 |
| Interleaved$_{S,T2}$ | $0.20/0.20 = 1.00$ | $0.18/0.19 = 0.95$ | 1.95 |
| Freeze 5% | $0.15/0.20 = 0.75$ | $0.11/0.19 = 0.58$ | 1.33 |

Sequential은 옛 과제 밴드의 90%를 잃는다. sleep은 0%를 잃는다. freezing은 두 축을 맞바꿔 1.33에 머문다.

**2단계 — 표준편차를 같은 축으로 옮긴다.** Interleaved$_{S,T2}$의 표준편차는 Task 1이 0.03, Task 2가 0.05다. 같은 나눗셈을 적용하면 $0.03/0.20 = 0.15$ 밴드, $0.05/0.19 = 0.26$ 밴드다. 그런데 sleep과 고전 interleaving의 합 차이는 $1.95 - 1.69 = 0.26$ 밴드다. **두 방법의 격차가 한 조건의 1$\sigma$와 같은 자릿수다.** 논문이 본문에서 "exceeding"이라고 쓰고 Discussion에서 우위를 이론적 근거로 돌린 이유가 이 한 줄에 들어 있다 [L305-306 vs L654-655].

**3단계 — 그러면 무엇이 교환되었는가.** 두 방법의 차이는 정확도가 아니라 저장이다. Interleaved$_{T1,T2}$는 Task 1 환경에 대한 접근을 계속 유지해야 한다 [L282-286]. Interleaved$_{S,T2}$가 각성에서 sleep으로 넘기는 것은 은닉 뉴런 784개의 목표 발화율이 전부이고, Uniform-Noise 통제에서는 집단 평균 1개로 줄어든다 [L982-985; L1144-1150]. **논문은 dtype도 바이트도 밝히지 않으므로** 여기서부터는 이 책의 산수다: fp32(4 B/elem)로 가정하면 $784 \times 4\,\mathrm{B} = 3{,}136\,\mathrm{B} \approx 3.1$ KiB이고, 통제군은 4 B다. 환경 전체와 3.1 KiB 사이의 비교는 자릿수 비교조차 되지 않는다.

**4단계 — 상각식에 넣어 본다.** 식 (A)는 $C_{\text{avg}} = C_{\text{wake}} + C_{\text{sleep}}/N_q$다. duty ratio가 1:1이므로 한 주기에서 $C_{\text{sleep}} = C_{\text{wake}}$다 [L301-303]. 그런데 이 논문에는 질의 스트림도, 문맥도, 사용자도 없어 $N_q$가 **정의되지 않는다**. 정의하지 않은 채 $N_q = 1$로 놓으면 $C_{\text{avg}} = 2\,C_{\text{wake}}$다. 손익분기 $N_q^*$도 구할 수 없다 — 구하려면 sleep 없이 같은 품질을 내는 wake 예산이 있어야 하는데 이 논문에 그 대안은 없다(Sequential은 품질이 다른 결과다).

> **[평가]** 4단계의 결론을 그대로 쓴다. 이 논문에서 sleep은 **총 계산을 두 배로 쓰고, wake 지연은 그대로 두고, 저장 바이트를 환경 전체에서 킬로바이트 이하로 바꾼다.** 교환은 sleep ↔ wake가 아니라 **계산 ↔ 저장**이다. Part III에서 세 경로를 비교할 때 이 축 배치를 잊지 않는다 — 생물학적 조상은 상각 증거를 하나도 제공하지 않았고, 제공한 것은 상태 용량 논증이다.

---

## 요약

- CLS는 속도 논증이 아니라 **간섭 논증**이다. 빠른 쓰기와 느린 일반화를 한 파라미터 집합에 얹으면 둘 다 망가지므로 뇌가 두 시스템을 갖는다는 것이며, 그 처방인 interleaving은 과거 데이터의 영구 저장을 요구한다.
- SWS와 REM은 하나의 sleep이 아니라 **목적함수가 다른 두 오프라인 라운드**다. 오프라인 위상의 개수는 어떤 논문이 은유를 실제로 이송했는지 판별하는 값싼 지표다.
- 생물학적 replay는 저장된 입력 없이 일어나는 자발적 재활성화이고, deep learning의 replay는 저장된 원본을 섞는 것이다. 두 용법은 이름만 같다.
- PLOS Sleep은 과거 과제 데이터를 한 번도 다시 쓰지 않고 옛 과제 성능을 0.70에서 0.70으로 유지했다. 이것이 corpus에서 오프라인 위상의 필요성을 지지하는 가장 직접적인 증거다.
- 같은 논문에서 데이터를 저장하는 고전 interleaving은 0.68/0.65로 sleep의 0.70/0.68과 표준편차 안에서 같고, 논문은 우위를 **실증이 아니라 이론적 근거**로 돌린다. 이 논문이 실증한 것은 정확도 우위가 아니라 저장의 제거다.
- PAD는 오프라인의 가치가 스케줄이 아니라 **생성물**에 있음을 보인다. REM을 빼면 CIFAR-10 선형 분리도가 58.25에서 46.00으로 무너진다. 다만 저자 자신의 통제에서 기억의 **혼합 자체는 필요하지 않고**(자발 활동만으로도 차이 없음), sleep 단계의 **순서도 무관하다**.
- 두 생물학 논문 모두 검정 이름도 p-값도 없이 "유의하게"를 쓰고, 계산 비용을 어느 형태로도 보고하지 않는다. PAD의 절제 대조군은 저자가 손본 상태로 비교되고, 표현 품질 지표는 linear readout 하나뿐이며 CIFAR-10 절대 성능은 약 59%다 — 성능 근거로 인용하면 오용이다.
- E-경로의 창시 논문은 sleep을 170회 쓰면서 생물학·CLS 문헌을 한 편도 인용하지 않고, 그 단어를 consolidation이 아니라 **유휴**의 뜻으로 도입한다. 이름의 공유는 이론의 공유가 아니다.

## 자가 점검 체크리스트

- [ ] CLS의 두 시스템이 왜 필요한지를 "속도"가 아니라 "간섭"으로 설명할 수 있다.
- [ ] 어떤 논문이 "replay한다"고 쓸 때, **어느 층에 쓰는지**와 **$\mathcal{R}_k$를 어디서 얻는지** 두 질문을 먼저 던진다.
- [ ] 표 8-1의 네 조건에서 PLOS Sleep이 실증한 것과 실증하지 않은 것을 각각 한 문장으로 말할 수 있다.
- [ ] 표 8-3에서 w/o NREM 열이 CIFAR-10에서 PAD와 오차 안에서 같다는 사실을 지적할 수 있고, 그럼에도 NREM의 기여가 어디서 나타나는지 말할 수 있다.
- [ ] 어떤 논문이 sleep이라는 단어를 쓸 때 그것이 유휴를 뜻하는지 consolidation을 뜻하는지를 **참고문헌으로** 판별할 수 있다.
- [ ] 이 장의 개념 중 $W$층에 놓이는 것이 하나도 없는 이유를 설명할 수 있다.
- [ ] (Rosetta) `Sleep-time Compute`의 오프라인 위상은 **warm-up / 사전 컴파일** 자리에, CLS·PLOS·PAD의 consolidation은 **모델 재배포 1회** 자리에 놓인다는 것을 설명할 수 있다. 앞의 것은 캐시 무효화 문제이고 뒤의 것은 배포 문제다.

## 다음 장으로

이 장은 오프라인 위상이 저장 없이 $\mathcal{R}_k$를 만들 수 있음을 보였다. 그러나 두 질문이 열린 채 남는다. 첫째, 그렇게 만든 $\mathcal{R}_k$를 몇 번이나 반복해서 넣을 수 있는가 — PLOS Sleep은 과제 두 개에서 멈췄고 $\rho$를 재지 않았으며, PAD는 오프라인 위상 하나를 빼면 반복 라운드가 성능을 깎기 시작한다는 방향성만 보고한다. 둘째, 더 앞선 질문으로, $\Theta$에 애초에 **얼마나** 넣을 수 있는가.

ch09가 두 번째 질문을 받는다. 파라미터당 몇 bit가 저장되는지, 그 상한이 Θ-경로의 $C_{\text{cap}}$을 어디서 끊는지, 그리고 그 상한이 이 장이 말한 "저장을 계산으로 바꾼다"는 교환을 어디까지 허용하는지를 다룬다. 첫 번째 질문은 반증 축인 ch22가, 그리고 두 생물학 논문의 정면 재독은 ch23이 받는다.

