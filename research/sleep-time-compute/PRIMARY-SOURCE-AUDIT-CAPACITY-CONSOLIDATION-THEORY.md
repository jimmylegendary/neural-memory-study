# Sleep-Time Compute 원문 감사 — 용량·공고화·망각 이론 계보

- 감사일: 2026-07-25
- 범위: 순차학습 간섭, complementary learning systems, bounded synapse,
  multi-timescale consolidation, rehearsal/generative replay, parameter isolation,
  loss of plasticity, LLM knowledge-capacity measurement, parametric/external
  allocation
- 목적: “sleep-time에 계속 배우면 언젠가 기억 용량을 초과하는가?”를 단일
  `parameter count` 문제가 아닌 측정 가능한 여러 포화 모드로 분해
- 상태: canonical registry 입력 전 원문 감사. 이 문서의 식과 수치는 해당
  논문의 benchmark 가정 안에서만 유효하다.

최근 capacity/externalization 논문의 마지막 원문 감사에는 아래의 정확한 PDF
snapshot을 사용했다. digest는 검토한 byte를 식별할 뿐, 재배포 권리를 뜻하지
않는다.

| 연구 | 검토한 원문 | SHA-256 |
|---|---|---|
| Morris et al., *How much do language models memorize?* | arXiv `2505.24832` v3 PDF | `ce0148ea694019714c4c3d5d8bb3a09b2f1e7c402f377f9db50cae705e32b123` |
| Zhao et al., *Limited Memory Language Models* | arXiv `2505.15962` v3 PDF | `0947232c9969db74026320dce1f1ec4b05f11bea268cc2e1e1baa93c5fd18197` |
| Ujváry et al., *LaCy* | arXiv `2602.12005` v4 PDF | `d2de2b7dd0b27dc26080ac6f88bd1145150f9642224a43fe51208faaa2d64e89` |

## 1. 먼저 분리해야 하는 다섯 가지 용량

이 문헌에서 `memory capacity`는 같은 뜻으로 쓰이지 않는다. 적어도 다음을
분리해야 한다.

1. **storage capacity**: 물리적으로 몇 bit, exemplar, slot, expert 또는 parameter
   state를 보유할 수 있는가.
2. **retention capacity**: 새 기억이 계속 들어올 때 과거 trace가 retrieval
   threshold 위에 얼마나 오래 남는가.
3. **addressable capacity**: 정보가 어딘가 남아 있어도 실제 query와 readout이
   정확히 꺼낼 수 있는가.
4. **plasticity capacity**: 기존 지식을 가진 상태에서 새 목적을 계속 학습할
   수 있는가.
5. **governance capacity**: source 수정, consent 철회, selective delete,
   provenance, rollback을 감당할 수 있는가.

따라서 “weights에 fact가 남아 있다”, “외부 store가 줄었다”, “old-task accuracy가
보존됐다”는 서로 다른 관찰이다. 어느 하나도 나머지를 자동으로 증명하지 않는다.

본 연구에서는 다음 total-memory accounting을 사용한다.

```text
B_total
  = B_raw_episode
  + B_external_active
  + B_external_archive
  + B_latent_or_KV
  + B_parametric_delta
  + B_optimizer_and_importance_state
  + B_index_graph_metadata
  + B_verification_recovery
```

이는 아래 논문들의 정리를 그대로 옮긴 식이 아니라, 서로 다른 매질로 압력이
이동하는 것을 빠뜨리지 않기 위한 본 연구의 회계 정의다.

## 2. 시간순 원문 지도

| 최초 공개 | 연구 | 원문에서 입증한 것 | sleep-time compute에 허용되는 해석 |
|---:|---|---|---|
| 1989 | McCloskey & Cohen, [Catastrophic Interference in Connectionist Networks](https://doi.org/10.1016/S0079-7421%2808%2960536-8) | 분산 weight를 순차적으로 학습할 때 새 list가 이전 list를 빠르게 훼손할 수 있음을 고전 connectionist 실험으로 보임 | “배포 후 weight update도 pretraining과 같은 방식으로 하면 된다”는 기본 가정을 부정하는 출발점 |
| 1990 | Ratcliff, [Connectionist Models of Recognition Memory](https://doi.org/10.1037/0033-295X.97.2.285) | sequential learning에서 잘 학습한 정보의 급격한 망각과, 학습량에 따른 old/new discrimination의 비단조성을 함께 지적 | 평균 old-task accuracy뿐 아니라 discrimination과 calibration도 sleep transition 전후 측정해야 함 |
| 1995 | McClelland, McNaughton & O’Reilly, [Complementary Learning Systems](https://web.stanford.edu/~jlmcc/papers/McCMcNaughtonOReilly95.pdf) | 빠른 hippocampal acquisition과 느린 neocortical learning을 분리하고, recent memory reinstatement를 기존 경험과 interleave해야 structured knowledge를 보호할 수 있다고 논증 | fast external/episodic tier와 slow shared/parametric tier의 이론적 basis. “수면 중에만 일어난다”거나 NREM/REM 대응을 확정한 논문은 아님 |
| 1995 | Robins, [Catastrophic Forgetting, Rehearsal and Pseudorehearsal](https://doi.org/10.1080/09540099550039318) | old examples rehearsal과, random input에 대한 현재 network output을 pseudo-pattern으로 보존하는 방법을 분석 | raw data가 없을 때 teacher-generated replay가 가능한 초기 근거. teacher 오류까지 보존한다는 위험도 같이 남음 |
| 2005 | Fusi, Drew & Abbott, [Cascade Models of Synaptically Stored Memories](https://doi.org/10.1016/j.neuron.2005.02.001) | 서로 다른 plasticity를 가진 metaplastic state cascade가 simple binary synapse보다 훨씬 긴 power-law-like trace를 만든다는 ideal-observer 분석 | fast→slow state가 하나의 weight 뒤에 숨은 multi-timescale auxiliary state일 수 있음을 보임 |
| 2007 | Fusi & Abbott, [Limits on the Memory Storage Capacity of Bounded Synapses](https://www.columbia.edu/cu/neurotheory/Larry/FusiNatNeuro07.pdf) | bounded efficacy와 ongoing plasticity만으로는 trace가 빠르게 overwrite됨. hard bound의 상태 수 증가 이득은 potentiation/depression의 정밀한 균형에 의존 | parameter 수만 늘리지 않고 각 parameter의 값 정밀도만 늘리는 것은 robust lifetime 해법이 아님 |
| 2013 | Roxin & Fusi, [Efficient Partitioning of Memory Systems](https://doi.org/10.1371/journal.pcbi.1003146) | plasticity가 다른 여러 stage로 기억을 보내면 lifetime 이득이 stage 수에 비례할 수 있고, replay copy에는 speed–accuracy trade-off가 생김 | external→latent→adapter→shared tiering과 짧은 transfer window의 직접 이론 precedent |
| 2013 | Lahiri & Ganguli, [A Memory Frontier for Complex Synapses](https://papers.nips.cc/paper_files/paper/2013/file/7f24d240521d99071c93af3917215ef7-Paper.pdf) | \(N\) synapse, 각 \(M\) internal state의 Markov model에서 initial SNR, memory-curve area, lifetime의 upper bound를 유도 | “내부 상태를 늘리면 무한 용량”이 아니라 state complexity도 회계에 포함해야 한다는 경계 |
| 2016 | Benna & Fusi, [Computational Principles of Synaptic Memory Consolidation](https://doi.org/10.1038/nn.4401) | bidirectional fast/slow variables와 \(t^{-1/2}\) trace로 lifetime을 거의 \(N\)에 비례하게 개선; 필요한 dynamical-variable 수는 lifetime의 로그로 증가 | 단방향 distillation보다 bidirectional slow-state feedback이 중요할 수 있다는 이론 가설 |
| 2016 | Rusu et al., [Progressive Neural Networks](https://arxiv.org/abs/1606.04671) | 과거 column을 freeze하고 새 task마다 column을 추가해 forgetting을 구조적으로 차단하고 lateral transfer를 사용 | forgetting을 capacity growth로 바꾸는 해법. lifetime system에서는 column 수와 serving cost가 계속 증가 |
| 2017 | Kirkpatrick et al., [Elastic Weight Consolidation](https://doi.org/10.1073/pnas.1611835114) | old-task posterior를 diagonal-Fisher Gaussian으로 근사하고 중요한 weight의 이동을 quadratic penalty로 억제 | consolidation auxiliary state가 Fisher/anchor 형태일 수 있음. diagonal local approximation과 task switch 의존성을 그대로 기록해야 함 |
| 2017 | Zenke, Poole & Ganguli, [Synaptic Intelligence](https://proceedings.mlr.press/v70/zenke17a.html) | training trajectory에서 parameter별 loss-reduction 기여를 online 누적해 이후 변경을 억제 | Fisher 계산 없이 wake 중 importance를 모으고 boundary에서 확정하는 basis. task-free lifetime 해법은 아님 |
| 2017 | Shin et al., [Deep Generative Replay](https://papers.nips.cc/paper_files/paper/2017/hash/0efbe98067c6c73dba1250d2beaa81f9-Abstract.html) | generator가 만든 old input과 old solver target을 새 task data와 섞어 generator와 solver를 함께 갱신 | dream/replay data generation의 직접 basis. generator 품질이 누적 지식의 병목이며 raw-free가 information-free를 뜻하지 않음 |
| 2018 | Mallya & Lazebnik, [PackNet](https://openaccess.thecvf.com/content_cvpr_2018/html/Mallya_PackNet_Adding_Multiple_CVPR_2018_paper.html) | task 후 pruning으로 빈 weight를 만들고 surviving weight를 freeze해 후속 task를 pack | 고정 tensor 안의 free-slot 소비 방식. 빈 weight가 소진되면 재압축·증설·망각 중 하나가 필요 |
| 2018 | Yoon et al., [Dynamically Expandable Networks](https://arxiv.org/abs/1708.01547) | selective retraining, unit split/duplication, timestamping, 필요 시 network expansion | capacity exhaustion을 감지해 확장하는 선행 방식이지만 unbounded growth와 migration 문제를 남김 |
| 2018 | Serrà et al., [Hard Attention to the Task](https://proceedings.mlr.press/v80/serra18a.html) | task별 hard mask로 이미 사용된 unit을 보호; 논문은 forgetting rate를 45–80% 줄였다고 보고 | sparse ownership과 interference isolation precedent. task identity와 remaining free capacity가 필요 |
| 2019 | Chaudhry et al., [On Tiny Episodic Memories](https://arxiv.org/abs/1902.10486) | 네 supervised CL benchmark에서 task별 class당 한 exemplar만 둔 단순 replay가 여러 전용 CL 방식보다 강했고 7–17% gain을 보고 | 아주 작은 raw anchor도 중요하며 “모든 것을 weights로 승격”할 필요가 없다는 경험적 근거. workshop/preprint 범위에 한정 |
| 2021 | Doan et al., [NTK Overlap Matrix](https://proceedings.mlr.press/v130/doan21a.html) | NTK regime에서 task alignment와 forgetting error를 연결하고 projection 계열 방법을 분석 | 단순 task count보다 gradient/function overlap을 interference pressure로 측정해야 한다는 basis |
| 2023 | Lin et al., [Theory on Forgetting and Generalization](https://openreview.net/forum?id=t9oOGDbIpV) | overparameterized linear model에서 arbitrary task sequence의 forgetting/generalization 식을 유도하고 parameterization, similarity, ordering 효과를 분석 | “더 큰 model이면 해결”이 아니라 order·similarity와 함께 scaling surface를 추정해야 함 |
| 2024 | Allen-Zhu & Li, [Knowledge Capacity Scaling Laws](https://arxiv.org/abs/2404.05405) | 통제된 factual-tuple 생성과 retrieval 조건에서 약 2 knowledge bits/parameter 및 architecture·training·quantization·MoE·SNR 효과를 보고 | 자연어 사실의 보편 상수가 아니라, 모수 승격 arm이 같은 통제 workload에서 넘어야 할 capacity baseline |
| 2024 | Dohare et al., [Loss of Plasticity in Deep Continual Learning](https://doi.org/10.1038/s41586-024-07711-7) | standard deep learning이 장기 순차학습에서 새 task 학습능력 자체를 잃을 수 있고, 작은 비율의 low-utility unit 재초기화가 이를 완화함 | old-memory forgetting과 new-learning incapacity를 분리하고, sleep에 recycle/reset operator를 넣어야 할 근거 |
| 2025 | Morris et al., [How much do language models memorize?](https://arxiv.org/abs/2505.24832) | synthetic uniform sequence로 generalization을 제거한 GPT-style 실험에서 약 3.6 bits/parameter의 측정 lower bound와 capacity saturation/grokking·membership-inference scaling을 보고 | 특정 precision·architecture·data protocol의 parametric information benchmark. 3.6을 일반 lifetime 기억 상수로 사용하지 않음 |
| 2025 | Zhao et al., [Limited Memory Language Models](https://arxiv.org/abs/2505.15962) | pretraining target의 factual value를 external DB call로 대체하고 loss에서 반환값을 mask해, factual knowledge를 editable DB로 의도적으로 offload | 배포 후 sleep은 아니지만 parametric/external destination을 data construction 단계에서 학습시키는 직접 선행 |
| 2026 | Ujváry et al., [LaCy](https://arxiv.org/abs/2602.12005) | token loss와 문법적 factuality를 함께 사용해 제한된 SLM이 factual token을 학습할지 `<CALL>`할지 pretraining 중 선택 | loss나 “용량을 비웠다”는 직관만으로 승격/외부화할 수 없다는 allocator 선행; sleep phase와 실제 freed-bit 측정은 없음 |

## 3. 첫 계보: 왜 fast/slow system과 interleaving이 필요한가

### 3.1 1989–1990: 문제는 단순히 “오래되면 희미해진다”가 아니다

McCloskey–Cohen과 Ratcliff의 핵심은 새 data가 이전 data 뒤에 들어올 때
distributed weights가 급격히 새 해에 맞춰진다는 것이다. Ratcliff는 순차학습이
다음 두 실패를 동시에 만든다고 정리했다.

- 잘 학습한 old information이 new learning 뒤 빠르게 사라짐;
- studied item과 novel item을 구분하는 성능이 학습량에 따라 감소하거나
  비단조적으로 변함.

따라서 sleep benchmark는 retention accuracy만 보고 끝낼 수 없다. 최소한 다음
transition-level 관찰이 필요하다.

```text
old-item recall
new-item acquisition
old/new discrimination
confidence calibration
source/version discrimination
```

### 3.2 CLS 1995: fast store는 archive가 아니라 teacher다

CLS의 원문 논리는 다음과 같다.

1. hippocampal system은 한 번의 경험도 빠르게 encode한다.
2. neocortical system은 경험 ensemble의 structure를 발견하려면 느리고
   interleaved하게 배워야 한다.
3. recent hippocampal memory를 reinstatement하면 새 경험을 기존 경험과 섞어
   neocortex에 통합할 수 있다.

이것은 본 연구의 `wake capture → sleep replay → slow durable view`에 가장 중요한
초기 basis다. 다만 세 가지 과장을 금지한다.

- 원문은 LLM parameter update를 검증하지 않았다.
- reinstatement가 수면에서만 일어난다고 제한하지 않았다.
- hippocampus=vector DB, neocortex=base model 같은 일대일 등치를 입증하지 않았다.

### 3.3 rehearsal과 pseudorehearsal의 정보 경계

Robins의 pseudorehearsal과 이후 generative replay는 raw old examples가 없어도
현재 network의 function을 random/prototypical input 위에서 재현할 수 있다는
아이디어다. 하지만 pseudo-data는 무에서 old truth를 복원하지 않는다. teacher,
generator, weight, statistic 중 어딘가에 남은 trace를 다시 읽는다.

따라서 본 연구의 retained state를

\[
\Omega_t =
(\theta_t, G_t, T_t, B_t, S_t, M_t, Z_t)
\]

로 둔다. 각각 deployed weights, generator, teacher, replay buffer, sufficient
statistics, external memory, provenance/recovery metadata다. 삭제된 사실 \(X\)에
대해 \(I(X;\Omega_t)=0\)이고 새 randomness \(U\)가 \(X\)와 독립이면, data
processing inequality에 의해 이후 변환만으로 \(X\)를 정확히 되살릴 수 없다.
이것은 본 연구가 새 theorem이라고 주장하는 것이 아니라 standard information
boundary를 lifecycle 전체에 적용한 것이다.

## 4. 둘째 계보: bounded state에서는 왜 계속 overwrite되는가

### 4.1 Fusi–Abbott 2007의 정확한 benchmark

논문은 \(n\)개 bounded synapse가 ongoing random plasticity rate \(r\) 아래 한
memory trace를 얼마나 오래 유지하는지 ideal observer로 본다.

- memory signal은 tracked event가 potentiated/depressed한 synapse 집단과 현재
  state의 overlap이다.
- noise는 ongoing plasticity가 만드는 fluctuation이다.
- trace가 exponential time constant \(\tau\)로 감소하면 \(n\) synapse를 함께
  읽어도 detectability는 대략 \(O(\tau \log n)\)까지만 늘어난다.

hard bound에서 step size를 \(a\), 허용되는 discrete state 수를 대략 \(1/a\)라
두면:

\[
\tau \approx \frac{1}{r a^2}
\]

라는 quadratic 이득은 potentiation과 depression이 정밀하게 balanced된 경우에만
성립한다. 조금이라도 지속적인 imbalance가 있으면 작은 \(a\)의 추가 이득이
사라지고 lifetime은 \(a\)와 무관한 regime으로 포화된다. soft bound는 fine
tuning 없이 state 수에 선형 이득을 주지만, synapse 수에 대한 logarithmic
detectability 문제는 없애지 못한다.

이 결과에서 허용되는 시스템 결론은 다음뿐이다.

- weight precision 또는 slot 내부 state 수만 키우는 것은 robust lifetime
  scaling의 충분조건이 아니다.
- 실제 stream의 update bias, source frequency, tenant skew를 함께 측정해야 한다.

LLM의 factual capacity가 위 식을 그대로 따른다고 주장하면 안 된다. benchmark가
random uncorrelated binary modifications와 ideal readout이기 때문이다.

### 4.2 2005 cascade: 하나의 weight 뒤에 time scale을 여러 개 둔다

Fusi–Drew–Abbott는 plasticity가 큰 상단 state와 변화 확률이 작은 하단
metaplastic state를 cascade로 연결했다. 원 논문의 특정 구성은 초기에
\(t^{-3/4}\) 형태의 SNR decay와, 그 뒤 slowest transition probability가 정하는
exponential cutoff를 보고한다. state 수를 늘리면 cutoff가 지수적으로 늦어지는
대신 initial trace가 감소한다.

중요한 감사 주의점이 있다. 2005 원문의 특정 mean-field fit은
\(t_{\max}\propto N_{\mathrm{syn}}^{2/3}\) 항을 제시하지만, Fusi의 2017 review와
Benna–Fusi 2016은 cascade family를 더 보수적인 \(1/t\) decay와
\(\sqrt{N}\)-scale lifetime 계열로 요약한다. 이는 model variant, optimality
조건, finite cascade, complexity accounting의 차이를 섞으면 생기는
불일치다. 본 연구에서는 어느 한 exponent를 범용 biological/LLM scaling law로
승격하지 않고 다음만 사용한다.

- power-law trace는 exponential trace보다 많은 time scale을 유용하게 만든다.
- slow state 수와 precision도 공짜가 아니다.
- initial strength와 lifetime 사이 Pareto frontier가 남는다.

### 4.3 Lahiri–Ganguli 2013의 upper bound

이 논문은 \(N\)개의 synapse와 synapse당 \(M\)개의 finite internal Markov state를
가정한다. ideal observer 기준으로:

\[
\mathrm{SNR}(0) \le \sqrt{N}
\]

이고, memory curve의 area와 lifetime은 일반적으로
\(O(\sqrt{NM})\) 경계를 넘지 못한다고 유도한다. threshold \(\epsilon\)에서
논문이 제시한 envelope lifetime은

\[
\tau(\epsilon)
\le
\frac{\sqrt{N(M-1)}}{\epsilon e r}.
\]

late-time envelope는 \(O(\sqrt{NM}/(rt))\)이고, bound에 가까운 구조는
nearest-neighbor linear chain이다. 이 정리는 다음 조건에 한정된다.

- finite-state Markov synapse;
- random, uncorrelated memory modifications;
- direct synaptic-state ideal readout;
- fixed candidate plasticity rate.

따라서 external text/graph memory나 transformer의 정확한 upper bound가 아니다.
대신 auxiliary state complexity를 숨기고 “고정 parameter 수”라고 부르면 안
된다는 강한 회계 원칙을 준다.

### 4.4 Benna–Fusi 2016: bidirectional fast/slow coupling

Benna–Fusi는 synaptic weight에 해당하는 빠른 variable과 더 느린 hidden
variables를 bidirectional chain으로 연결한다. random uncorrelated stream에서
bounded variance를 유지하면서 가능한 가장 느린 유용한 trace가 대략

\[
r(t)\sim t^{-1/2}
\]

임을 논증한다. tuned chain은 다음 성질을 갖는다.

- lifetime은 고정 complexity에서 \(N\)에 선형으로 늘다가 longest time scale에
  포화한다.
- longest time scale을 \(N\)에 맞춰 조정하면 lifetime은
  \(N/\log N\)에 가깝게 scaling한다.
- 필요한 dynamical-variable 수 \(m\)은 longest lifetime의 로그로 증가한다.
- initial SNR은 대략 \(\sqrt{N}\) scale을 유지하되 \(m\)에 따라
  \(1/\sqrt{m}\)로 감소한다.
- slower variables는 faster variables보다 낮은 precision으로도 유사한 성능을
  낼 수 있다.

Lahiri–Ganguli의 \(M\)과 Benna–Fusi의 \(m\)은 같은 complexity 단위가 아니다.
전자는 finite Markov states의 총수이고, 후자는 여러 값을 갖는 coupled
dynamical variables의 수다. \(m\)개 variable의 joint state 수는 discretization에
따라 훨씬 크게 늘 수 있으므로 두 scaling을 직접 모순이라고 읽으면 안 된다.

LLM system에 옮길 때 이 논문은 theorem이 아니라 다음 design hypotheses를 준다.

- one-way `fast adapter → slow adapter` distillation만이 아니라 slow state가
  fast update를 regularize하는 feedback가 필요할 수 있다.
- time scale이 느릴수록 낮은 update frequency와 낮은 precision/storage tier를
  사용할 수 있다.
- auxiliary optimizer, importance, router state까지 total capacity에 포함해야
  한다.

## 5. 셋째 계보: machine continual learning의 네 가지 압력 이동

### 5.1 regularization은 기억을 없애지 않고 “움직이지 못하는 방향”을 늘린다

EWC는 task A posterior를 diagonal Gaussian으로 근사하고 Fisher diagonal을
parameter importance로 쓴다. task B에서 최소화하는 목적은:

\[
\mathcal{L}(\theta)
=
\mathcal{L}_B(\theta)
+
\frac{\lambda}{2}
\sum_i F_i
(\theta_i-\theta^{*}_{A,i})^2.
\]

장점은 raw old data 없이 old solution 주변의 중요한 방향을 보호한다는 것이다.
그러나 원문 자체가 밝힌 범위는 중요하다.

- posterior를 factorized Gaussian으로 근사한다.
- diagonal Fisher와 point estimate를 쓴다.
- Fisher는 task switch에서 계산한다.
- Atari system은 task-recognition model, game-specific gains/biases, task별 replay
  buffer도 함께 사용한다.
- plain EWC가 ten separate DQN 성능에는 도달하지 못했고, nullspace 중요도를
  과소평가하는 현상도 보고했다.

SI는 같은 목적을 training path integral에서 얻은 local importance로 근사한다.
importance \(\Omega^\mu_k\), old reference \(\tilde\theta_k\)를 쓰는 surrogate는:

\[
\tilde{\mathcal L}_\mu
=
\mathcal L_\mu
+
c \sum_k
\Omega^\mu_k(\theta_k-\tilde\theta_k)^2.
\]

이 계열은 memory pressure를 다음으로 옮긴다.

```text
raw replay bytes 감소
→ anchor/importance state 증가
→ protected parameter directions 증가
→ future plastic subspace 감소 가능
```

마지막 화살표는 EWC/SI 원문이 일반 lifetime theorem으로 증명한 것은 아니다.
본 benchmark에서 직접 측정할 가설이다.

### 5.2 replay는 parameter pressure를 data/generator pressure로 옮긴다

Deep Generative Replay의 solver loss는 current data와 generated old input에 대한
old solver target을 혼합한다.

\[
\mathcal L_i
=
\rho\,\mathbb E_{(x,y)\sim D_i}
[\ell(S_i(x),y)]
+
(1-\rho)\,\mathbb E_{x'\sim G_{i-1}}
[\ell(S_i(x'),S_{i-1}(x'))].
\]

generator도 current real input과 old generated input의 혼합으로 새로 학습한다.
논문의 preliminary MNIST chain에서는 solver 1→5 accuracy가
98.81→98.56%였고, 저자들은 optimal generator이면 joint training과 동등하다고
설명한다. 동시에 다음 한계를 명시한다.

- efficacy는 generator quality에 크게 의존한다.
- current/old mix ratio가 task importance를 결정한다.
- generator와 old solver 자체가 retained information이다.

따라서 synthetic dream은 저장을 제거하지 않고 `raw bytes → generator/teacher
capacity + generation compute + model-error debt`로 바꾼다.

Chaudhry et al.의 tiny episodic memory 결과는 이 교환의 반대편을 보여준다.
class당 한 old exemplar만 있어도 네 benchmark에서 7–17% gain을 보고했다.
작은 canonical anchor와 generator를 함께 쓰는 hybrid가 generator-only보다
강할 수 있다는 실험 가설을 만든다.

### 5.3 isolation과 expansion은 interference를 free-slot pressure로 바꾼다

| 방식 | 보호 메커니즘 | 새로 소비하는 것 | 포화 시 남는 문제 |
|---|---|---|---|
| Progressive Nets | old column freeze, lateral connection | column·activation·serving compute | task 수에 따른 구조 성장 |
| PackNet | prune 후 surviving weight freeze | 남은 free weights와 task mask | free weight exhaustion, task-ID routing |
| DEN | selective retrain, split/duplicate, expand | 새 unit와 timestamp metadata | migration, unbounded growth |
| HAT | task별 learned hard mask | unused units와 mask | mask overlap, free capacity, task identity |
| sparse memory/expert 계열 | event에 해당하는 slot/expert만 update | slot/expert pool | routing collision, pool exhaustion, recycle safety |

이 계열은 zero-forgetting을 얻을 수 있어도 zero-cost가 아니다. 본 연구에서는
`protected`, `free`, `recyclable`, `quarantined` parameter/slot fraction을 cycle마다
추적한다.

### 5.4 loss of plasticity는 catastrophic forgetting과 별개다

Dohare et al.은 다음을 명시적으로 분리한다.

- catastrophic forgetting: old example을 다시 보지 않을 때 old performance가
  저하되는 문제;
- loss of plasticity: new or repeated objective를 학습할 능력 자체가 장기 training
  뒤 저하되는 문제.

Continual ImageNet에서 standard network는 초기 task에서 최대 약 88%를 보이지만
2,000번째 task에는 여러 step size에서 shallow/linear 수준으로 떨어졌다.
class-incremental CIFAR-100에서는 dormant unit 증가와 stable-rank 감소가
동반됐다. continual backpropagation은 작은 비율의 low-utility unit을 계속
random reinitialize해 plasticity를 유지했고, L2와 Shrink-and-Perturb도 weight
magnitude를 낮게 유지해 완화했다. 실험은 30-run 평균을 보고한다.

sleep system에는 다음 별도 probe가 필요하다.

```text
retention probe: 과거를 아직 아는가?
plasticity probe: 같은 compute로 새 것을 여전히 배울 수 있는가?
reacquisition probe: 잃은 것을 canonical data로 다시 배울 수 있는가?
```

reset이 과거 정보를 보존하는 마법은 아니다. replay buffer 또는 canonical
external memory와 결합하지 않은 reset은 retention을 훼손할 수 있다.

### 5.5 LLM의 bits/parameter와 의도적 externalization은 무엇을 말하는가

두 capacity 논문은 서로 다른 통제 문제를 측정한다.

- Allen-Zhu & Li는 synthetic factual tuples를 학습하고 flexible extraction을
  평가해 약 2 knowledge bits/parameter를 보고한다. 결과는 training duration,
  architecture, int8 quantization, sparsity/MoE, data signal-to-noise, domain prefix
  조건에 따라 달라진다.
- Morris et al.은 uniform random sequence로 generalization을 제거하고
  unintended memorization과 generalization을 정보론적으로 분리한다. 500K부터
  1.5B parameter까지 수백 개 GPT-style model을 훈련했으며, bfloat16 설정의
  plateau를 약 3.5–3.6 bits/parameter, fp32 평균을 약 3.83으로 측정한다.

두 수치는 모순되는 보편 상수를 제시한 것이 아니다. 저장 대상, query/readout,
training regime, precision, architecture, 그리고 무엇을 “knowledge bit”로
세는지가 다르다. 특히 Morris et al.의 수치는 generalization을 제거한 실험에서
모델이 실제로 담은 정보의 측정 lower bound다. 이론적 maximum, 자연어 fact 수,
배포 후 sequential-update capacity와 같지 않다.

본 연구에서는 이를 다음과 같이 사용한다.

\[
\eta_{\text{param}} =
\frac{I(Y; \theta_{\text{candidate}}\mid Q,\theta_{\text{base}})}
     {\Delta B_{\text{param}}}
\quad\text{bits/added parameter-bit}
\]

이 식은 직접 측정 가능한 controlled opaque-payload arm의 효율 정의다. 자연어
benchmark에서는 \(I\)를 정확히 안다고 가정하지 않는다. 같은 payload와 query를
external row, adapter, expert, shared-weight arm에 넣고, held-out addressability,
retention, delete, interference를 함께 비교한다. 2 또는 3.6이라는 선행값은
universal prior가 아니라 sanity range와 falsification anchor다.

LMLM과 LaCy는 capacity를 늘리는 대신 **무엇을 parameter에 넣지 않을지**
학습한다.

- LMLM은 54.6M factual triplet 기반 annotator pipeline으로 pretraining text에
  DB calls를 넣고, 반환 factual value와 `<db_end>`를 language-model loss에서
  제외한다. DB edit/delete가 knowledge control surface가 된다.
- LaCy는 spaCy 기반 factuality와 현재 token loss를 결합해 높은-loss factual
  token의 일부를 `<CALL>` target으로 바꾼다. 334M/1.27B GPT-2-style models를
  3B-token Wikipedia-derived corpus에서 반복 훈련하고, 334M model과 Llama
  3.2 1B cascade에서 FactScore `22.71` versus baseline `15.89`를 보고한다.
  training-time calling은 inference-only calling보다 강하지만, empirical
  call rate 약 22.4%에서 약 79.7%로 올려도 FactScore는 `22.71→25.56`으로
  완만히 증가한다.

LaCy의 가장 중요한 negative result는 externalization이 tested scale의 NLU를
유의하게 높이지도 낮추지도 않았다는 점이다. “사실을 외부로 빼면 그만큼 추론
용량이 생긴다”는 직관은 입증되지 않았다. facts보다 넓은 token을 offload하면
NLU가 악화된다. LMLM과 LaCy 모두 predeployment pretraining 연구이며,
session experience를 sleep에서 재배치하지 않는다. 또한 LaCy는 실제 freed
information bits를 측정하지 않는다.

따라서 sleep allocator의 목적은 단순한 high-loss offload나 parameter fill이
아니다. 최소한 다음 utility를 past-only feature로 예측해야 한다.

```text
expected future reuse
compression/generalization gain
addressability after compilation
external lookup latency and failure
volatility, correction, and delete fan-out
interference and plasticity cost
provenance and rollback reserve
```

## 6. 네 가지 포화 모드와 대응 메커니즘

### 6.1 물리적 포화

증상:

- replay/archive bytes가 계속 증가;
- graph node/edge와 invalidated version이 reclaim되지 않음;
- adapter, expert, mask, column 수가 단조 증가;
- optimizer/Fisher/provenance가 모델 delta보다 커짐.

가능한 operator:

- deduplicate, exact/semantic coalesce;
- active/archive tiering;
- bounded reservoir 또는 utility-weighted coreset;
- prune/quantize/offload;
- adapter merge 후 검증된 predecessor archive;
- low-utility slot recycle;
- retention SLA에 따른 deliberate forgetting.

### 6.2 간섭 포화

증상:

- old-task loss 또는 source-confusion 증가;
- task-gradient/NTK overlap 증가;
- 같은 slot/expert에 unrelated memory가 충돌;
- consolidation 횟수가 늘수록 utility가 비단조 감소.

가능한 operator:

- interleaved raw/latent/generative replay;
- importance regularization;
- orthogonal/projected gradients;
- sparse routing과 parameter isolation;
- external-first, repeated-use verified memory만 parametric promotion;
- fresh-base rebuild 후 canary/rollback.

### 6.3 plasticity 포화

증상:

- 새 task의 fixed-budget learning gain 감소;
- dormant/saturated unit 증가;
- activation/gradient stable rank 감소;
- weight magnitude 증가;
- fresh initialization보다 기존 model의 acquisition이 열등.

가능한 operator:

- low-utility unit recycle 또는 plasticity injection;
- weight shrink/L2;
- optimizer-state reset;
- adapter rollover;
- verified canonical replay와 fresh-base reconstruction;
- slow tier 보호와 fast tier reset을 분리.

### 6.4 governance 포화

증상:

- source correction가 derived summary/graph/adapter/weight에 전파되지 않음;
- delete debt와 stale replica가 증가;
- 어떤 source가 어떤 parameter artifact에 들어갔는지 알 수 없음;
- rollback 가능한 checkpoint 수가 storage budget을 초과.

가능한 operator:

- raw append-only provenance를 canonical truth로 유지;
- derived memory를 materialized view로 취급;
- source→evidence→artifact dependency DAG;
- tombstone과 revocation fan-out;
- immutable candidate + atomic publish pointer;
- retrain/unlearn/rebuild 경로와 expiry.

### 6.5 finite memory와 dense coverage의 긴장

[Memento 2](https://arxiv.org/abs/2512.22716)는 frozen LLM 주위의 external case,
Parzen policy, \(Q\)를 갱신하는 wake-learning theory이지만, external-memory
capacity가 value error로 들어가는 방식을 명시한다. Corollary 15의 조건부 bound는
다음 꼴이다.

\[
\lVert V^\star-V^{\pi_M}\rVert_\infty
\le
\frac{2R_{\max}}{(1-\gamma)^2}
\left[\epsilon_{\mathrm{LLM}}(r_M)+\delta_M\right].
\]

\(r_M\)은 memory coverage radius, \(\delta_M\)은 retrieval error에 해당한다. 이
bound가 적용되려면 local consistency와 coverage 외에도 bounded reward,
\(\gamma<1\), evaluation 동안 stationary memory dynamics, finite current memory가
필요하다. two-timescale convergence는 martingale/bounded-iterate 조건과 compact
attractor까지 추가한다.

이것은 fixed-capacity 해법이 아니라 오히려 연구 공백을 드러낸다. 일반 연속 상태
공간에서 \(r_M\to0\)인 dense coverage와 **항상 finite한 고정 budget**을 동시에
요구하려면 state distribution의 intrinsic dimension/compactness, value-aware
coreset, merge/quantization distortion, eviction 뒤 error 증가를 연결하는 별도
covering argument가 필요하다. Memento 2에는 그 mechanism, 실험, fixed-budget
forgetting theorem이 없다. 따라서 이 bound는 다음을 정당화하지 않는다.

- memory item 수를 늘리면 무조건 value error가 단조 감소한다;
- finite memory가 open-ended deployment를 일정 distortion으로 덮는다;
- retrieval top-\(k\) 또는 prompt cap이 lifetime capacity를 bound한다;
- wake-time update가 deferred sleep의 이점을 증명한다.

## 7. Sleep-time scaling law가 세어야 할 축

기존 synapse theory의 \(N\), \(M\), \(r\)을 LLM parameter count에 바로 대응시키지
않는다. 대신 다음 empirical surface를 추정한다.

\[
Q =
f(
P,\,
B_r,\,
B_e,\,
B_l,\,
B_p,\,
C_s,\,
R,\,
H,\,
V,\,
\Delta,\,
K
)
\]

여기서:

- \(P\): shared/base parameter와 trainable local parameter;
- \(B_r\): retained raw episodic bytes;
- \(B_e\): active external text/vector/graph bytes;
- \(B_l\): latent/KV bytes;
- \(B_p\): adapter/expert와 importance/optimizer bytes;
- \(C_s\): cycle당 sleep FLOPs 또는 accelerator-seconds;
- \(R\): admitted memory의 valid future reuse count;
- \(H\): memory-type/task heterogeneity;
- \(V\): fact/preference volatility;
- \(\Delta\): source correction·delete·rollback pressure;
- \(K\): wake/sleep cycle 수.

종속변수 \(Q\)는 하나의 QA score가 아니라 다음 vector다.

```text
retention, acquisition, addressability, calibration,
interference, plasticity, staleness, corruption,
delete completeness, rollback recovery,
latency, energy, bytes moved
```

### 7.1 검증할 crossover

1. reuse가 낮거나 volatility/delete risk가 높으면 raw/external memory가
   parametric promotion보다 우세할 것이다.
2. reuse가 높고 source가 안정적이며 procedure가 압축 가능하면 sleep compiler의
   amortized cost가 retrieval cost를 추월할 수 있다.
3. replay buffer가 아주 작아도 generator-only보다 recoverability margin을 크게
   높일 수 있다.
4. parameter isolation은 초기에 강하지만 free-slot fraction이 임계치 아래로
   내려가면 recycle/rebuild 없는 성능이 급격히 악화될 것이다.
5. retention이 유지돼도 acquisition gain과 representation rank가 먼저 감소할 수
   있다.
6. 같은 total bytes에서 fast→slow stage 수를 늘리는 이득은 metadata, transfer
   error, read amplification 때문에 무한히 선형으로 지속되지 않을 것이다.

### 7.2 queue stability도 memory capacity다

arrival rate가 sleep service보다 높으면 store가 정확해도 backlog가 무한히
증가한다. 본 연구의 최소 queue 조건은:

\[
\lambda\,
\mathbb E[c_{\text{admitted}}]
<
\mu_{\text{sleep}},
\]

여기서 \(c_{\text{admitted}}\)는 capture가 아니라 verify, transform, publish,
recovery를 포함한 event당 실제 service demand다. 이 식은 새로운 memory theorem이
아니라 안정 queue의 필요조건을 sleep service에 적용한 것이다. batch efficiency,
priority, retries, tenant fairness가 있으면 실제 충분조건은 별도로 추정해야 한다.

## 8. 본 연구의 capacity safety gate

각 sleep candidate는 다음을 모두 보고해야 한다.

```text
input snapshot digest
canonical raw bytes retained
derived active/archive bytes
trainable and auxiliary parameter bytes
free/recyclable slot fraction
sleep FLOPs and bytes moved
pre/post retention
pre/post acquisition gain
source and deletion coverage
candidate-versus-incumbent delta
rollback artifact and expiry
```

다음 중 하나면 자동 승격하지 않는다.

- old/new aggregate score는 올랐지만 특정 protected slice가 허용치 아래로 하락;
- acquisition gain이 fresh-control보다 계속 낮음;
- source coverage 또는 deletion dependency가 불완전;
- total bytes는 늘었는데 active-store 감소만 “compression”으로 보고;
- synthetic replay의 teacher/generator/version digest가 없음;
- free slot과 recovery reserve가 안전선 아래;
- sleep backlog의 estimated load가 service capacity 이상;
- raw trace를 없앤 뒤 retained state 어디에도 recoverability evidence가 없음.

## 9. 허용 주장과 금지 주장

### 허용

- bounded, continually modified state에는 stability–plasticity trade-off가 있다.
- multi-timescale state, system partition, replay, regularization, isolation,
  growth, recycle은 그 trade-off를 서로 다른 자원으로 이동시킨다.
- catastrophic forgetting을 줄여도 physical capacity, plasticity, addressability,
  governance 문제가 남을 수 있다.
- sleep-time compute의 용량은 model weights만이 아니라 raw/external/latent/
  parametric/auxiliary/recovery state 전체로 세야 한다.
- canonical trace가 완전히 사라진 뒤 독립 randomness만으로 특정 기억을
  무손실 복구할 수는 없다.

### 금지

- Benna–Fusi의 \(N/\log N\)을 transformer factual capacity scaling law라고 직접
  인용.
- biological synapse state를 LLM parameter, expert, vector row에 일대일 대응.
- EWC/SI가 task-free long-lifetime 또는 user-specific deletion을 해결한다고 주장.
- generative replay가 old information을 저장하지 않는다고 주장.
- network expansion이나 sparse slots를 무한 capacity라고 표현.
- active-store reduction을 total-storage reduction 또는 legal erasure로 표현.
- old-task retention만으로 model이 계속 배울 수 있다고 결론.

## 10. 이 계보에서 직접 파생되는 연구 질문

1. **Recoverability margin**: candidate memory를 제거하기 전 retained state
   \(\Omega\)가 future task에 필요한 정보를 얼마나 남겼는지 어떤 measurable
   proxy로 판정할 수 있는가?
2. **Bidirectional consolidation**: slow external/parametric state가 fast wake
   learner의 update geometry를 regularize하면 one-way distillation보다
   retention–plasticity frontier가 개선되는가?
3. **Capacity pressure vector**: bytes, interference, plasticity, staleness,
   governance debt 중 어떤 pressure가 먼저 saturation을 예고하는가?
4. **Recycle safety**: low-utility parameter/slot의 reset 전후 raw anchor와
   canary가 어느 정도 있어야 destructive forgetting 없이 plasticity를
   회복하는가?
5. **Stage-count crossover**: fast/medium/slow tier를 추가할 때 얻는 lifetime
   이득이 read amplification과 copy error를 넘어서는 범위는 어디까지인가?
6. **Teacher corruption compounding**: raw replay, latent replay, generated replay의
   cycle 수에 따른 error exponent가 어떻게 다른가?
7. **Promotion reversibility**: external memory를 LoRA/expert/shared weight로
   승격한 뒤 source correction와 delete의 full fan-out cost는 reuse 이득을
   언제 상쇄하는가?
8. **Retention–plasticity decoupling**: old recall이 정상인 동안 acquisition
   slope가 먼저 무너지는 early-warning signature를 만들 수 있는가?
9. **Coverage–budget law**: 고정 external-memory budget에서 state distribution의
   intrinsic dimension, coreset/quantization distortion, retrieval error가
   \(r_M\)과 downstream value error를 어떻게 제한하는가?

## 11. 이론 읽기 순서

1. CLS 1995: fast/slow system과 interleaving의 이유.
2. Fusi–Drew–Abbott 2005, Fusi–Abbott 2007: bounded state와 overwrite.
3. Roxin–Fusi 2013: system partition과 replay-copy speed/accuracy.
4. Lahiri–Ganguli 2013: finite internal-state upper bound.
5. Benna–Fusi 2016: bidirectional multi-timescale dynamics와 complexity accounting.
6. EWC, SI, DGR 2017: regularization, local importance, generated replay의 세
   parametric consolidation basis.
7. Progressive Nets, PackNet, DEN, HAT: capacity growth/isolation의 비용.
8. Tiny Episodic Memory: 아주 작은 canonical anchor의 가치.
9. NTK Overlap 2021과 Lin et al. 2023: task similarity/order/geometry.
10. Dohare et al. 2024: retention과 plasticity를 분리하고 reset/recycle을 평가.
11. Allen-Zhu & Li 2024와 Morris et al. 2025: 서로 다른 controlled definition의
    parametric bit-capacity를 비교하고 universal constant로 오독하지 않기.
12. LMLM 2025와 LaCy 2026: 무엇을 weight에 쓰지 않을지를 data/target
    construction으로 학습하는 parametric/external allocation.
13. Memento 2 2025/2026: finite-memory value bound의 조건과 dense-coverage
    가정이 실제 eviction/coreset mechanism 없이 남기는 긴장.

## 12. 감사 결론

“계속 sleep training을 하면 memory capacity를 넘는가?”의 답은 **그렇다. 다만
먼저 넘는 자원이 하나가 아니다.**

- fixed shared weights는 interference 또는 protected-direction pressure에 부딪힌다.
- sparse slot/expert와 task masks는 free-pool exhaustion에 부딪힌다.
- generator replay는 generator error와 teacher-state capacity에 부딪힌다.
- raw/text/vector/graph memory는 storage, retrieval, staleness, deletion debt에
  부딪힌다.
- 모든 매질은 service backlog와 recovery reserve에 부딪힌다.
- retention이 남아 있어도 model은 plasticity를 먼저 잃을 수 있다.

따라서 credible sleep-time system은 “한 번의 consolidation algorithm”이 아니라
다음 closed loop여야 한다.

```text
measure pressure
→ choose destination/operator
→ transform under provenance
→ verify retention + plasticity + governance
→ atomically publish or roll back
→ recycle/demote/archive
→ remeasure total state and queue load
```

이 closed loop와 cross-medium capacity surface가 기존 continual-learning
알고리즘을 나열하는 review를 넘어서는 본 연구의 핵심 연구 대상이다.
