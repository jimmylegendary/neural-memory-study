# 생물학·계산적 Sleep 원문 감사

- 감사일: 2026-07-25
- 범위: 1983--2026년 sleep, replay, pseudorehearsal, dreaming,
  synaptic-homeostasis 계보
- 증거 경계: 원 논문·최종 venue·보충자료·공식 코드와 공개적으로 접근 가능한
  manuscript
- 시간 규칙: 확인 가능한 최초 공개일을 사용하며, preprint와 최종 출판을 분리
- 상태: evidence registry 입력 전 감사. 안정 ID는 registry 가동 뒤 부여한다.

## 핵심 판정

1. NREM replay가 기억 공고화에 기여한다는 증거는 강하다. 그러나
   `NREM = 해마 기억을 피질로 파일 이동`이라는 등식은 성립하지 않는다. Quiet
   wake에서도 replay가 발생하고, 상세 episodic trace가 장기적으로도 해마에 의존할
   수 있다는 multiple-trace/transformation 계열도 강하다.
2. REM이 불필요한 기억을 선택적으로 삭제하거나 pruning한다는 주장은 확립되지
   않았다. Crick--Mitchison의 reverse-learning은 역사적으로 중요한 가설이지
   현재의 합의된 생물학적 메커니즘이 아니다.
3. 2022년 PLOS Computational Biology 논문은 “sleep이 memory manifold를
   수학적으로 최적화함을 증명”하지 않았다. 두 과제 SNN simulation에서
   residual synaptic trace를 이용한 sleep-like interleaving이 catastrophic
   forgetting을 줄였고, PCA·kernel PCA·SVM으로 표현 공간을 사후 분석했다.
4. 감사한 sleep/replay 시스템은 대부분 완전히 지워진 기억을 복원하지 않는다.
   Frozen teacher, generator, latent buffer, activation statistic, residual
   synapse, 실제 sample 가운데 하나에 과거 정보가 남아 있다.
5. 무한 수명 용량, 이름 지정 삭제, provenance, rollback, privacy, LLM-scale
   lifecycle을 함께 해결한 연구는 없다.
6. 2026 Surprise-Gated Memory 연구 노트는 surprise admission과 명시적
   offline replay를 결합한 중요한 소규모 반례를 제공한다. 다만 전체 과거 replay는
   유한 수명 해법이 아니며, 최근 window만 replay하면 no-replay보다 나빠질 수 있다.

## 시간순 계보

| 최초 공개 | 연구 | 최종 상태 | 직접적인 의미 |
|---|---|---|---|
| 1983 | Crick & Mitchison, [The function of dream sleep](https://doi.org/10.1038/304111a0) | *Nature* | REM random activation과 anti-learning이라는 추측적 reverse-learning 가설 |
| 1995 | Robins, [Catastrophic Forgetting, Rehearsal and Pseudorehearsal](https://doi.org/10.1080/09540099550039318) | *Connection Science* | 원자료 대신 old-network 함수 출력을 replay |
| 2003 | Tononi & Cirelli, [Sleep and synaptic homeostasis](https://doi.org/10.1016/j.brainresbull.2003.09.004) | *Brain Research Bulletin* | wake potentiation--sleep downscaling 가설 |
| 2014 | Tononi & Cirelli, [Sleep and the Price of Plasticity](https://doi.org/10.1016/j.neuron.2013.12.025) | *Neuron* | 선택적·활동의존적 renormalization으로 정교화 |
| 2016 | Kumaran et al., [What Learning Systems do Intelligent Agents Need?](https://doi.org/10.1016/j.tics.2016.05.004) | *Trends in Cognitive Sciences* | 현대화된 complementary-learning-systems 이론 |
| 2017-05-24 | Shin et al., [Continual Learning with Deep Generative Replay](https://arxiv.org/abs/1705.08690) | NeurIPS 2017 | generator + frozen solver의 pseudo-experience replay |
| 2018-10-29 | Fachechi et al., [Dreaming Neural Networks](https://arxiv.org/abs/1810.12217) | *Neural Networks* 2019 | Hopfield spurious attractor를 projection/pseudo-inverse 방향으로 제거 |
| 2019-03-07 | González et al., [Can sleep protect memories from catastrophic forgetting?](https://doi.org/10.1101/569038) | *eLife* 2020 | NREM-like replay가 잔존 sequence trace를 파괴 임계점 이전에 회복 |
| 2019-07-01 | Golden et al., [bioRxiv precursor](https://doi.org/10.1101/688622) | PLOS Comp Bio 2022 | sleep-like interleaving과 joint synaptic representation |
| 2019-08-01 | Krishnan et al., [Biologically Inspired Sleep Algorithm](https://arxiv.org/abs/1908.02240) | arXiv/후속 발표 | activation statistic과 residual weight를 이용한 ANN--SNN sleep |
| 2019-12-20 | Tadros et al., [Sleep-like Unsupervised Replay](https://openreview.net/forum?id=r1xGnA4Kvr) | ICLR 2020 | 같은 operator의 robustness 효과와 정확도 손실 |
| 2020-08-13 | van de Ven et al., [Brain-inspired replay](https://doi.org/10.1038/s41467-020-17866-2) | *Nature Communications* | generator·feedback·context gating의 결합 |
| 2021-09-09 | Deperrois et al., [Perturbed and adversarial dreaming](https://doi.org/10.7554/eLife.76384) | *eLife* 2022 | wake/NREM/REM objective를 분리한 PAD |
| 2022-02-01 | Singh, Norman & Schapiro, [bioRxiv precursor](https://doi.org/10.1101/2022.01.31.478475) | *PNAS* 2022 | hippocampus--neocortex 자율 replay와 NREM/REM 교대 계산 모델 |
| 2022-11-18 | Golden et al., [Sleep prevents catastrophic forgetting](https://doi.org/10.1371/journal.pcbi.1010628) | *PLOS Computational Biology* | 블로그가 지칭한 실제 2022 논문 |
| 2022-12-15 | Tadros et al., [Sleep replay consolidation](https://doi.org/10.1038/s41467-022-34938-7) | *Nature Communications* | 여러 continual benchmark의 ANN--SNN SRC |
| 2025-09-08 | Tutuncuoglu, [Dream-Augmented Neural Networks](https://doi.org/10.2139/ssrn.5402490) | SSRN | 상세 검증이 불가능한 abstract-only 저증거 주장 |
| 2026-06-30 | Mouchon, [Surprise as a Signal for Plasticity and Metacognition](https://arxiv.org/abs/2606.31495) | 단독 저자 arXiv research note | surprise-gated episodic write와 periodic full replay를 slow linear/prototype memory로 공고화; strict experimental sleep이지만 코드·완전 설정·lifetime cap 없음 |

계보 혼동을 피해야 한다. `10.1101/569038`은 González--Sokolov--
Krishnan--Bazhenov의 thalamocortical sequence 연구이며,
`10.1101/688622`가 Golden의 2022 PLOS 연구 선공개판이다.

## 1. Crick--Mitchison 1983

Salk Institute 연구진이 제시한 명시적으로 추측적인 이론이다.

- `W`: Hebbian storage가 attractor와 기생적 mode를 축적한다.
- `S`: REM 중 뇌간의 quasi-random stimulation이 parasitic mode를
  활성화하고, 활성 connection을 anti-Hebbian하게 약화한다.
- `Q`: 다음 wake에서 recall과 network stability를 관찰한다고 가정한다.

새 dataset, optimizer, 동물·사람 실험은 없다. 저자도 직접 검증이 어렵고 REM
deprivation 및 당시 기억 증거가 단순한 설명과 맞지 않을 수 있음을 인정했다.
따라서 이는 unlearning/pruning 계보의 출발점이지만, REM이 실제 기억을 전역
삭제한다는 증거가 아니다.

## 2. Robins 1995 Pseudorehearsal

- 소속: University of Otago.
- 모델: 32--16--32 backprop network.
- optimizer: learning rate .3, momentum .5.
- 순서: 기존 pattern 20개 뒤 새 pattern 10개; 전체 실험 50회.
- 완료 기준: 모든 출력이 target의 6% 이내.

`W`에서 원 과제를 학습하고, 경계에서 이전 network를 frozen teacher로
보존한다. `S`에 해당하는 단계에서는 random 32-bit input을 teacher에 넣어
연속 pseudo-target을 만들고 새 data와 interleave한다. Pseudopopulation
8/32/128개를 비교했으며 대체로 큰 집합이 old-task retention에 유리했다.

Raw data를 저장하지 않지만 과거 함수가 frozen network에 남아 있다. Mapping이
random input 주변으로 일반화되지 않으면 pseudorehearsal도 실패한다. 저자는
sleep analogy를 “fanciful speculation” 수준으로 한정했다. 이는 generative
replay의 공학적 선조이지 생물학적 sleep model이 아니다.

## 3. Tononi--Cirelli Synaptic Homeostasis

2003년 원 가설은 네 연결고리를 제시한다.

1. Wake learning이 평균적인 synaptic potentiation을 만든다.
2. Potentiation이 sleep pressure와 slow-wave activity를 높인다.
3. NREM slow wave가 synapse를 downscale한다.
4. Downscaling이 에너지·공간·signal-to-noise와 다음 날 학습능력을 회복한다.

2014년 업데이트는 모든 synapse의 동일 비율 축소에서 선택적·활동의존적
down-selection/renormalization으로 이동했다.

후속 근거로 de Vivo et al. 2017은 생쥐 cortex의 6,920개 synapse를 조사해
sleep 후 axon-spine interface가 평균 약 18% 작아졌다고 보고했다. 감소는
작은·중간 synapse 약 80%에 집중됐고 가장 큰 약 20%는 상대적으로 보존됐다.
Diering et al.의 분자 결과도 이를 지지한다.

반면 Chauvette et al. 2012 등은 slow-wave sleep 중 cortical responsiveness와
synaptic strength가 강화될 수 있음을 보였다. 따라서 SHY는 강한 통합 가설이나
보편적 전역 감소 법칙은 아니며, REM pruning 이론도 아니다.

## 4. Kumaran--Hassabis--McClelland 2016 CLS Update

Google DeepMind/UCL--Stanford 계열의 이론·리뷰다.

- `W`: hippocampus/MTL이 episode를 빠르게 저장한다.
- `B`: replay 대상은 novelty뿐 아니라 task relevance, goal, schema
  consistency에 따라 우선순위화될 수 있다.
- `S`: hippocampal replay를 cortical representation과 interleave해 느린 구조
  학습을 돕는다.
- `Q`: 시간이 지나 cortex 또는 hippocampus--cortex 공동회로로 retrieval한다.

Replay는 sleep에 국한되지 않으며 quiet wake에도 발생한다. Literal copy가 아니라
재조합되거나 goal-weighted된 통계일 수 있다. Schema-consistent 정보는 cortex도
빠르게 학습할 수 있으므로 “해마는 항상 빠르고 피질은 항상 느리다”는 절대 규칙도
아니다.

Multiple-trace/transformation 관점에서는 cortex가 gist와 semantic structure를
획득해도 상세 episodic trace가 계속 hippocampus-dependent일 수 있다. 따라서
시스템 설계에서 장기기억 전환은 literal move보다
`replicate -> transform -> re-index -> dependency change`로 표현하는 편이
정확하다.

## 5. Shin et al. 2017 Deep Generative Replay

- 소속: MIT와 SK T-Brain.
- 경계: 이전 generator와 solver를 frozen snapshot으로 보존.
- replay: frozen generator가 old input을 만들고 frozen solver가 soft label을
  부여한다.
- update: 실제 new data와 replay를 섞어 새 generator와 solver를 공동 학습.
- destination: 새 generator와 solver weights.

Solver의 MNIST 1→5 transfer 단계별 정확도는
98.81, 98.64, 98.58, 98.53, 98.56%다. Permuted MNIST,
MNIST↔SVHN, disjoint MNIST class pair도 평가했다.

Generator가 과거 분포를 완벽히 재현한다는 이상 조건에서 joint training에
가까워지지만 실제 병목은 generator drift와 recursive approximation error다.
Raw data가 없더라도 `generator + frozen solver`가 압축된 parametric memory다.
Named deletion, provenance, rollback, privacy는 없다. 공식 재현 코드와 optimizer
세부가 부족하다는 당시 review 지적도 있었다.

## 6. Fachechi--Agliari--Barra Dreaming Neural Networks

Binary random pattern을 저장하는 Hopfield network의 통계물리 모델이다.

- `W`: Hebbian coupling으로 \(P\)개 pattern을 저장한다.
- `S`: dream iteration이 mixture/spurious attractor를 약화하고 pure pattern을
  강화한다.
- 충분한 반복에서 coupling은 pseudo-inverse/projection matrix 방향으로
  수렴한다.

표준 Hebbian Hopfield의 critical load는
\(\alpha=P/N\approx0.138\text{--}0.14\)다. Dream limit의 replica-symmetric
계산은 약 1.07, symmetry breaking을 고려한 현실적 해석은
\(\alpha\approx1\)에 가깝다. Field standard deviation은 대략
\(0.224t^{-0.998}\)로 감소한다.

이는 random-pattern associative-memory toy의 결과다. 실제 episode의 의미를
판단해 삭제한 것이 아니라 mixture attractor를 제거했다. Pattern 정보도 최종
projection coupling에 계속 들어 있으므로 LLM lifetime capacity로 직접
일반화할 수 없다.

## 7. González et al. 2019--2020

- 모델: 약 500 neuron의 biophysical thalamocortical spiking system.
- `W`: ordered sequence S1 뒤 겹치는 reverse/interfering sequence S1* 학습.
- `S`: NREM/N3-like Up-state의 spontaneous replay와 balanced STDP.
- `Q`: sequence completion/recall.

Figure에서 읽을 수 있는 대략적 값은 sleep 후 S1 약 65--70%, 새 S2 약 65%,
S1* 약 50%, interference 직후 old S1 약 20%다. 이는 table의 exact value가
아니라 visual estimate이므로 원문 인용 시 그렇게 표시해야 한다.

S1*을 계속 학습해 old trace가 완전히 파괴되면 sleep도 복원하지 못했다. 핵심은
망각된 기억을 무에서 다시 만든 것이 아니라 행동상 무너졌어도 synapse에 잔존한
trace를 파괴 임계점 전에 재조직한 것이다.
[공식 코드](https://github.com/o2gonzalez/sequenceLearningSleepCode)는
C++/Makefile 중심이며 문서화가 제한적이다.

## 8. Krishnan et al. 2019 ANN--SNN Sleep

- 소속: UCSD/New Mexico Tech; DARPA Lifelong Learning Machines 맥락.
- `W`: bias 없는 ReLU ANN을 SGD/backprop으로 순차 학습.
- `B`: ANN을 SNN으로 변환하고 과거 task input unit별 평균 activation을 보존.
- `S`: 그 평균을 rate로 사용한 Poisson stimulation과 modified STDP.
- `B`: SNN을 다시 ANN으로 변환.

“입력을 저장하지 않는다”는 말은 raw sample을 저장하지 않는다는 뜻이다. 과거
data에서 계산한 mean-activation vector는 명시적으로 남긴다.

MNIST 순차학습의 마지막 all-task accuracy는 약 19%, 마지막 sleep 뒤 약 69%다.
명시적 run 하나는 19.12%→70.41%이고 task별 최종 결과는
95.08/61.61/56.40/85.75/51.03%다. Catastrophic forgetting이 발생한
68개 applicable run 가운데 58개, 즉 85%에서 sleep이 이를 줄였고 10개는
실패했다. Generalization도 약 20%→50%로 개선됐다.

논문은 hidden weight에 old information이 일부 남아 있을 때만 동작하고, 완전히
사라진 정보는 회복할 수 없다고 명시한다. 공개 코드는 확인되지 않았다.

## 9. Tadros et al. ICLR 2020 Robustness

동일한 ANN→SNN→Poisson replay→STDP→ANN operator를 continual memory가
아닌 robustness에 적용했다.

- MNIST architecture: `[784,1200,1200,10]`.
- learning rate .1, momentum .5, dropout .2, 2 epochs.
- sleep 뒤 clean MNIST accuracy는 약 80%로 떨어지고 digit 8·9는 사실상
  잊혔다.
- 최소 세 network를 사용했지만 dataset·attack별 hyperparameter가 균일하지
  않고, 일부는 genetic optimization으로 FGSM에 맞춰졌다.

| Dataset | Attack | Control | Sleep |
|---|---|---:|---:|
| Patches | FGSM | .0175 | .2025 |
| Patches | Boundary | .2971 | .3515 |
| MNIST | FGSM median epsilon | .09 | .22 |
| MNIST | DeepFool perturbation | .0042, success 96.46% | .0484, success 86.38% |
| MNIST | JSMA | .0477, success 95.56% | .0059, success 72.97% |
| MNIST | Boundary | .0525 | .0488 |
| CUB | FGSM | .055 | .06 |
| CUB | Boundary | .9751 | .9957 |

일부 공격에는 강해졌지만 전부는 아니다. Aggregate robustness가 상승하면서 특정
class가 소실될 수 있다는 결과는 sleep update에 per-memory canary, checkpoint,
rollback이 필요함을 보여준다. CUB는 frozen ResNet-50 embedding을 사용했다.
공식 코드는 확인되지 않았다.

## 10. van de Ven et al. 2020 Brain-Inspired Replay

VAE decoder와 classifier feedback을 통합한 replay-through-feedback,
conditional-Gaussian-mixture latent, context gating, internal replay를
결합한다.

- `B`: old model snapshot을 보존.
- update: latent에서 old sample을 만들고 soft-target distillation으로 current
  task와 interleave.
- Adam `beta1=.9`, `beta2=.999`; 실험별 2,000/5,000 iterations.
- 5회 또는 10회 반복.

BI-R+SI는 100-task permuted MNIST `0.904±0.005`,
class-incremental CIFAR-100 `0.344±0.002`다. Replay-through-feedback를
제거하면 각각 `0.892±0.004`, `0.334±0.002`다. Standard generative
replay는 permuted MNIST 약 15 task 이후 성능 저하가 커졌고 BI-R이 이를
개선했지만 joint training에는 못 미친다. CIFAR는 frozen convolutional feature를
사용한다.

[공식 코드](https://github.com/GMvandeVen/brain-inspired-replay)가 공개돼
있다. 별도 wall-clock sleep scheduler가 아니라 task-transition replay이며,
generator weight가 externalized parametric memory 역할을 한다.

## 11. Deperrois et al. PAD

- 소속: Bern--Heidelberg.
- 구성: encoder/discriminator \(E\), generator \(G\), 작은 latent buffer.
- `W`: 실제 CIFAR-10/SVHN으로 reconstruction·KL·discriminator 학습하고
  latent \(z\)를 저장.
- `NREM`: 저장한 \(z\)로 image를 만들고 block occlusion을 적용한 뒤 원 latent를
  복원.
- `REM`: current latent, past latent, Gaussian noise를
  \(\lambda=\lambda'=0.5\)로 혼합해 adversarial learning.
- Adam `beta1=.5`, `beta2=.999`, learning rate `.0002`, batch 64.
- Linear probe: SGD learning rate `.2`, 20 epochs.

4 seeds, epoch 50:

| Condition | CIFAR-10 | SVHN |
|---|---:|---:|
| PAD | 58.25±0.70 | 78.92±0.40 |
| No latent mixing | 53.87±0.85 | 60.87±5.07 |
| No REM | 46.00±0.43 | 42.30±1.51 |
| No NREM | 58.00±0.34 | 73.25±0.22 |
| Wake only | 42.25±0.54 | 41.93±0.65 |

설계 안에서는 NREM이 occlusion robustness, REM이 semantic separability에
기여한다. 그러나 이는 NREM/REM 역할을 가정해 만든 모델의 ablation이지, 뇌가
동일한 역할 분담을 한다는 증거가 아니다. Continual task sequence도 아니며 latent
buffer의 lifetime capacity와 삭제를 평가하지 않는다.
[코드](https://github.com/NicoZenith/PAD)는 공개됐지만 repository는 현재
archived 상태다.

## 12. Singh--Norman--Schapiro PNAS 2022

- 최초 공개: [bioRxiv, 2022-02-01](https://doi.org/10.1101/2022.01.31.478475).
- 최종 논문: [*A model of autonomous interactions between hippocampus and
  neocortex driving sleep-dependent memory
  consolidation*](https://doi.org/10.1073/pnas.2123432119), *PNAS*,
  2022-10-24.
- 공개 [코드](https://github.com/schapirolab/SinghNormanSchapiro_PNAS22)는
  2022-08-30에 deposit됐다.

모델은 빠른 hippocampal C-HORSE 계와 느린 overlapping neocortical 계를
결합한다. Wake에서 새 satellite-category 구조와 episode를 학습한 뒤 외부
input을 제거한다. Sleep은 random activity로 시작하고, short-term synaptic
depression이 현재 attractor를 약화해 다음 attractor로 자율 전환시킨다. 안정된
attractor phase를 target으로, oscillation으로 왜곡된 phase를 prediction으로
삼는 contrastive Hebbian/error-driven update가 neocortical weight를 바꾼다.

- **NREM:** hippocampus와 neocortex가 결합돼 hippocampus가 최근의 약한
  neocortical attractor를 고충실도로 reinstatement한다.
- **REM:** 단순화된 lesion으로 hippocampal influence를 끊어 neocortex가 이미
  강한 remote attractor를 더 자유롭게 재생한다.
- **Simulation 1:** 15 exemplar·3 category, wake 뒤 30,000-cycle NREM.
- **Simulation 2:** 기존 Env 1을 neocortex에 overtrain한 뒤 Env 2를 새로
  학습하고, 10,000-cycle NREM/REM block을 다섯 번 교대하는 조건을 NREM-only
  및 REM-only 조건과 비교한다.
- 두 simulation 결과는 각각 100개 random initialization 평균이다.

이 연구의 중요한 공학적 선행은 replay item과 순서를 외부 scheduler가 매
trial 지정하지 않아도 내부 dynamics가 학습 trial을 생성한다는 점이다. 다만
sleep 시작, stage 종류, block 길이와 순서는 실험자가 정한다. learned scheduler,
event admission, 유한 lifetime capacity, raw-state accounting, deletion,
provenance, rollback은 없다.

`C1=Y`, `C2=Y`, `C3=Y`인 strict **experimental lifecycle**이지만, synthetic
category simulation이지 동물 실험이나 deployed learning system은 아니다.
NREM/REM 분업은 모델에 구현된 계산 가설이며, 뇌가 정확히 같은 objective와
disconnection을 사용한다는 증명이 아니다.

## 13. Golden et al. PLOS 2022

이 논문이 블로그의 “2022년 PLOS” 주장에 가장 직접적으로 대응한다.

- custom visual-foraging feed-forward SNN.
- input→hidden은 unsupervised STDP.
- hidden→output은 reward-modulated STDP.
- 서로 보완적인 두 task만 사용.
- `W`: Task 1 뒤 Task 2를 학습.
- `S`: 외부 input을 끄고 training 중 평균 firing rate를 사용한 Poisson
  stimulation으로 hidden neuron을 자발 활성화; hidden→output에는
  unsupervised STDP.
- cadence: sleep 100 movement cycles와 Task 2 100 cycles를 반복.
- 표본: 10개 simulated network.

| Condition | Task 1 | Task 2 |
|---|---:|---:|
| Task 1 only | .70±.02 | .53±.02 |
| Sequential Task 2 | .52±.02 | .69±.03 |
| Explicit Task1/2 interleave | .68±.03 | .65±.04 |
| Sleep/Task2 interleave | .70±.03 | .68±.05 |

Hidden neuron 70% pruning에도 어느 정도 회복력이 있었지만 이는 장기 다과제
capacity 법칙이 아니다. Old Task 1 synapse가 충분히 남아 있을 때만 회복됐고
Task 2를 오래 지속해 trace가 파괴되면 irreversible했다.

논문의 “manifold”는 10개 simulated network의 population-activity point set을
PCA/kernel PCA/SVM과 거리로 비교한 것이다. Theorem, storage-optimum proof,
manifold-capacity scaling이 아니다. Peer reviewer도 Figure 7 geometry 해석,
dimensionality, multi-task capacity, novelty를 문제 삼았다. Code/data는 공개
repository가 아니라 요청 시 제공 수준이었다.

허용되는 결론은 다음으로 좁아야 한다.

> 두 과제 SNN에서 sleep-like spontaneous activity와 STDP를 interleave하면
> joint training과 유사한 표현 분리를 만들며 residual old-task performance를
> 보존했다.

“PLOS가 sleep의 manifold 최적화를 수학적으로 증명했다”는 문장은 허용되지 않는다.

## 14. Tadros et al. Nature Communications 2022 SRC

Krishnan 2019의 ANN--SNN 변환을 여러 continual benchmark로 확장했다.

- `W`: ReLU/no-bias ANN, SGD momentum, cross-entropy.
- `B`: ANN→SNN 변환과 과거 task input별 mean activation 저장.
- `S`: mean activation 기반 Poisson replay와 local Hebbian plasticity.
- `B`: 다시 ANN으로 변환.
- 5 trials, mean±SD.

| Method | MNIST | Fashion | Multimodal | CUB task pair | CIFAR |
|---|---:|---:|---:|---:|---:|
| Sequential | 19.49±.002 | 19.67±.003 | 47.18±.002 | 5.32 / 95.41 | 19.01±.002 |
| SRC | 48.47±5.03 | 41.68±5.04 | 61.33±.015 | 63.2 / 45.4 | 44.55±1.45 |
| 0.75% rehearsal + SRC | 86.47±1.061 | 67.818±3.64 | 83.18±1.91 | 56.55 / 38.05 | 58.24±.561 |
| Joint | 98.02±.006 | 87.86±.005 | 90.05±.0028 | 85.49 / 79.15 | 72.43±.002 |

SRC 단독은 sequential보다 낫지만 joint와 큰 격차가 있고 일부 조건에서는 OWM
같은 baseline이 더 낫다. Raw rehearsal 0.75%를 더했을 때 성능이 크게 상승한다는
점은 순수 intrinsic sleep보다 작은 episodic buffer의 가치가 크다는 근거다.

Frozen feature extractor와 2--5 task 수준이며 endless capacity를 검증하지
않는다. [공식 코드](https://github.com/tmtadros/SleepReplayConsolidation)는
공개돼 있다.

## 15. DANN SSRN 2025

- 제목: *Dream-Augmented Neural Networks: Harnessing Synthetic Sleep for
  Continual Learning and Zero Forgetting*.
- 단독 저자: Bekir Tolga Tutuncuoglu.
- 작성일 2025-08-02, SSRN 게시 2025-09-08, 20쪽.
- SSRN metadata의 “IEEE”는 검증 가능한 연구기관 affiliation이나 IEEE
  publication을 뜻하지 않는다.

공개적으로 감사할 수 있었던 것은 abstract와 metadata뿐이다. SSRN PDF는 접근
제한이 있었고 Semantic Scholar/OpenAlex/ResearchGate에도 검증 가능한 full
text가 없었다. Abstract는 latent-only dream, vision/NLP, forgetting 최대 60%
감소, zero-shot/domain drift 대응, zero forgetting을 향한 경로를 주장한다.

Architecture, dataset, forgetting metric, baseline, seed, uncertainty,
ablation, code/data는 확인할 수 없다. 현재 증거 등급은 non-peer-reviewed,
abstract-only 최하위다. “60%”나 “zero forgetting”을 확정된 결과로 사용할 수
없다.

## 16. Mouchon 2026 Surprise-Gated Memory

**출처와 상태.** Louis Mouchon,
[*Surprise as a Signal for Plasticity and
Metacognition*](https://arxiv.org/abs/2606.31495), arXiv v1 최초 공개
2026-06-30. PDF는 스스로 *research note*와 두 proof-of-concept system으로
규정한다. 단독 저자·독립 연구이며 peer review는 확인되지 않는다. 논문과 arXiv
record는 code/data URL, executable algorithm, optimizer·epoch·batch 설정을
제공하지 않는다.

### System 1: surprise-gated periodic replay

고정 DINOv2 또는 I-JEPA encoder 위에서 masked view로 full embedding을 예측하는
작은 JEPA-style predictor의 smooth-L1 residual을 surprise로 쓴다. Base
distribution에서 threshold를 보정해 high-surprise embedding만 non-parametric
hippocampal buffer에 쓰고, 실제 classifier인 slow linear readout은 periodic sleep
때 interleaved replay로만 갱신한다. Encoder는 바뀌지 않는다.

1000 ImageNet class를 20-class task 50개로 순차 제시한 뒤 최초 5개 task를
측정했다.

| Backbone/arm | no replay | full-replay sleep | iid oracle | 해석 |
|---|---:|---:|---:|---|
| DINOv2 | 65.8 | 83.5 | 84.4 | sleep `+17.7 pp`; single run |
| I-JEPA | 25.9 | 77.2 | 78.5 | sleep `+51.3 pp`; single run |

별도 DINOv2 ablation에서 full replay는 `83.7`, surprise-gated write는 저장
수를 절반으로 줄이면서 `84.2`, class당 3 example은 `78.8`이다. 이 차이들은
single-run variation 안일 수 있다. 반대로 최근 5 task만 replay하면 DINOv2
`41.2`, I-JEPA `0.0`으로 no-replay보다 나쁘다. 이는 **bounded recent replay가
안전한 forgetting policy라는 가정의 직접 반례**다.

`C1=Y,C2=Y,C3=Y`: 완료된 task 뒤 replay가 slow readout을 만들고 이후 task가
이를 사용하므로 `B/S` **EXPERIMENTAL PASS**다. 그러나 full interleaving은 class와
task가 늘수록 replay state와 누적 work가 자란다. “sleep 뒤 small core로 trim”의
정확 알고리즘·크기·보존 기준도 공개되지 않았다. Class당 3 example조차 1000
class에서 3000 trace이고 open-ended class stream에서는 선형 성장한다.

### System 2: one-shot fact를 fast store에서 slow prototype으로

고정 SigLIP novelty detector, BGE-M3 text retriever, Qwen2.5-VL 7B 또는
Gemma 4 12B speaker를 사용한다. Calibrated surprise가 `<.35/.35-.65/>.65`일
때 assert/hedge/ask-user register를 고른다. 사용자가 novel concept를 설명하면
embedding--text pair를 hippocampus에 쓰고, 명령 또는 fast-store-full trigger의
sleep이 prototype/familiarity state를 neocortex로 합친 뒤 hippocampus를 비운다.
Base VLM weight는 고정이다.

- known/novel AUROC: external detector `.966±.024`, verbalized confidence
  `.618±.060`, token-logit confidence `.292±.043`; 15+15 concept를 8회
  resample했다.
- 50개 fact를 one-shot 학습하고 sleep, fast-store clear, dialogue clear 뒤
  slow store에는 `50/50`이 남고 answer recall은 `99.2±0.9%`(5 seed);
  context 없는 base model은 `0%`다.
- 그러나 text benchmark 네 차원 중 세 차원은 in-session context 때문에 모든
  system이 통과했고, correction은 backbone당 3개뿐이다. 50-fact 실험도 한
  backbone·좁은 fact pool이며 scaling sweep이 아니다.

이 경로도 평가 질의 전 fast state를 실제로 비우고 slow state만 재사용하므로 좁은
`B/S` **EXPERIMENTAL PASS**다. 다만 “buffer full” threshold, prototype merge,
router training, collision/expiry/delete, slow-store cap, replay compute, source
provenance, rollback은 재현 가능한 수준으로 명시되지 않았다. 따라서 surprise는
유망한 admission/trigger signal이지 검증된 lifetime-capacity controller가 아니다.

### 연구 프로그램에 주는 판정

1. surprise gate는 arrival rate와 stored traces를 줄일 수 있지만, 유익성 개선과
   같은 주장이 아니다; 보고된 `84.2` 대 `83.7`은 single-run 동률로 읽어야 한다.
2. full-history replay가 retention을 지킨 결과와 recent-window replay가 파괴적인
   결과를 함께 사용해야 한다. 전자만 인용하면 cumulative \(O(T^2)\) sleep work를
   숨긴다.
3. frozen representation은 encoder plasticity 문제를 피할 뿐, linear readout
   interference, prototype growth, retrieval collision, governance를 해결하지 않는다.
4. 독립 재현 전에는 LLM-scale sleep이나 production metacognition의 근거로
   승격하지 않는다.

## 생물학적으로 말할 수 있는 범위

### 비교적 강한 주장

- Hippocampal replay는 NREM slow-wave/ripple 상태에서 관찰된다.
- Wilson & McNaughton 1994는 wake 경험의 ensemble pattern 재활성화를
  보였다.
- Rasch et al. 2007은 학습 odor cue를 SWS에서 다시 제시했을 때 hippocampal
  declarative memory가 향상됐고 REM·wake cueing에서는 같은 효과가 없었다.
- Girardeau et al. 2009는 sharp-wave ripple 방해가 rat spatial-memory
  performance를 저하시킴을 보였다.
- Ngo et al. 2013은 phase-locked slow-wave stimulation으로 human
  declarative consolidation을 높였다.

### 확정 사실처럼 쓰면 안 되는 주장

- Replay는 quiet wake에도 발생하므로 sleep 독점 operator가 아니다.
- “해마 기억이 피질로 이동한다”보다 network dependency와 representation이
  재구성된다고 말해야 한다.
- REM-specific manipulation이 일부 object-place·contextual-fear memory를
  손상시켰지만 그것이 synaptic pruning을 뜻하지는 않는다.
- REM과 NREM 모두 dream report가 존재한다.
- REM emotional-memory 특이성에는 null과 task-dependent heterogeneity가 많다.

따라서 `NREM=transfer`, `REM=prune/recombine`은 유용한 계산 설계 가설이지만
established neuroscience fact가 아니다.

## No-Free-Resurrection 경계

시점 \(t\)의 전체 잔존 상태를 다음처럼 두자.

\[
\Omega_t=(\theta_t,\phi_t,B_t,s_t,M_t,\mathrm{metadata}_t)
\]

여기서 각 항은 model weights, generator/teacher, episodic buffer, sufficient
statistics, external memory, provenance다.

과거 task \(T\)와 \(\Omega_t\) 사이의 정보가 완전히 사라지고 sleep randomness
\(U\)가 \(\Omega_t\)를 조건으로 \(T\)와 독립이면
\(I(T;U\mid\Omega_t)=0\)이므로 data-processing inequality와 chain rule에
따라

\[
I(T;f(\Omega_t,U))
\le I(T;\Omega_t,U)
=I(T;\Omega_t)+I(T;U\mid\Omega_t)
=0.
\]

새 외부 관측이나 task-correlated prior가 없다면 random dream은 잃어버린
task-specific mapping을 정확히 복원할 수 없다.

| Method | 실제로 남긴 과거 정보 |
|---|---|
| Robins | frozen old network의 함수 |
| DGR / BI-R | generator와 frozen solver |
| Krishnan / SRC | per-input mean activation과 residual weight |
| González / Golden | interference 뒤 남은 subthreshold synaptic trace |
| PAD | latent buffer와 generator |
| Singh et al. | hippocampal/neocortical weights와 sleep-start attractor dynamics |
| Fachechi | 전체 pattern을 포함한 coupling |
| Rehearsal | 실제 old sample |
| External memory | text/vector/graph object와 metadata |

그러므로 “sleep이 지워진 기억을 되살렸다”보다
“behaviorally inaccessible하지만 물리적으로 남은 trace를 다시 접근 가능하게
만들었다”가 정확하다.

## Neural Memory 연구 가설로의 변환

### Recoverability-margin scheduling

고정 시각 sleep 대신 old-task margin, gradient conflict, retrieval confidence,
representational overlap을 이용해 trace가 파괴 임계점에 접근하는지 감지한다.
기억이 완전히 사라진 뒤 compute를 늘리는 방식보다 파괴 전 intervention이
효율적이라는 가설이다.

### Hybrid consolidation dominance

SRC에 raw rehearsal 0.75%를 더했을 때의 큰 상승은 순수 parametric sleep보다
작은 provenance-aware episodic buffer와 replay의 결합이 retention/compute
Pareto에서 우월할 가능성을 제시한다.

### External-first, parameter-last

Provenance, named delete, rollback이 필요한 기억은 외부 저장을 기본으로 한다.
반복적으로 유용하고 압축 이득과 canary 안전성이 검증된 구조만 adapter 또는
shared weight로 승격한다.

### Consolidation safety gate

ICLR 2020처럼 aggregate robustness가 올라가면서 특정 class가 사실상 사라질 수
있다. 따라서 sleep candidate에는 per-memory canary, shadow checkpoint,
counterfactual retrieval test, rollback trigger가 필요하다.

### Total-memory accounting

“Raw data가 없다”를 memory-free로 세지 않는다. Raw buffer, generator, frozen
teacher, statistics, optimizer state, residual trace, external index, lineage를
동일한 lifetime budget에 포함한다.

### No-free-resurrection scaling law

Sleep scaling은 compute만의 함수가 아니다. Retained-information budget이
recoverability의 전제다. Recoverability가 0에 가까워지는 regime에서는 sleep
FLOPs를 늘려도 past-task recovery가 포화하거나 실패해야 한다. 이 가설은
direct mutual information을 관측하기 어려우므로 calibrated canary margin,
old-task probe, teacher agreement, source availability를 proxy로 검증한다.

## 남은 원문 질문

1. 몇 task 또는 몇 memory에서 recursive replay drift가 급격히 커지는가?
2. 어느 trace가 recoverability threshold에 가까운지 online으로 측정할 수
   있는가?
3. Parametric memory와 external memory의 최적 분배는 무엇인가?
4. 다른 사용자·세션 기억의 오염을 어떻게 canary로 검출하는가?
5. Weight-level consolidation 뒤 named deletion을 어떻게 집행하는가?
6. Sleep 결과가 나쁠 때 어느 version으로 rollback하는가?
7. Consolidation compute와 serving compute 사이 data movement를 어떻게
   최소화하는가?

이 질문들은 biological stage label보다 memory object의 잔존 정보, 변환 매질,
governance, lifetime budget을 직접 계측해야 답할 수 있다.
