# 초록 (Abstract)

여러 과제를 해결하는 포괄적 인공지능을 순차 훈련하면 파국적 망각이 발생한다. 과거 데이터를 모두 replay하면 이를 완화할 수 있지만 큰 memory가 필요하고, privacy나 접근 제한 때문에 실제 환경에서 불가능할 수 있다. 이 논문은 primate hippocampus의 생성적 성질에서 영감을 받아 **Deep Generative Replay(DGR)** 를 제안한다.

DGR은 과거 input을 생성하는 deep generative model인 **generator**와 과제를 푸는 **solver**의 협력적 dual-model 구조다. generator가 이전 과제의 pseudo-data를 만들고, 이전 solver가 그 pseudo-data의 target을 제공한다. 이 쌍을 새 과제 실제 데이터와 섞어 다음 generator와 solver를 학습한다. 저자들은 image-classification의 여러 sequential-learning 설정에서 이 방법을 평가한다.

# 1. 서론 (Introduction)

사람은 평생 새 skill을 학습하면서 과거 memory를 심각하게 손상시키지 않는다. neural network는 새 objective에 맞춰 같은 parameter를 갱신하기 때문에 이전 input–output mapping을 빠르게 잊을 수 있다. episodic replay buffer는 효과적이지만 과거 raw sample을 계속 보관해야 한다.

Complementary Learning Systems 이론은 빠르게 경험을 encoding하는 hippocampus와 천천히 통계 구조를 학습하는 neocortex를 구분한다. hippocampal reactivation은 단순한 record 재생보다 유연하다. 기억을 변형하거나 함께 활성화하면 경험하지 않은 조합도 만들 수 있다는 관찰은 hippocampus를 고정 buffer보다 generative model에 가깝게 볼 동기를 준다.

# 2. 관련 연구 (Related Work)

## 2.1 비교 방법

논문은 sequential fine-tuning, Learning without Forgetting(LwF), elastic/regularization 계열, episodic replay를 비교한다. LwF는 새 data에서 이전 model의 output을 distill하여 old task를 보호하지만, 새 domain이 과거 domain과 겹치지 않으면 old input distribution을 보지 못한다. raw replay는 이 문제를 줄이지만 storage를 요구한다.

## 2.2 Complementary Learning Systems

CLS에서 빠른 episodic system은 recent experience를 저장하고, 반복 reactivation이 slow system의 weight에 knowledge를 통합한다. DGR은 이 구조를 엄밀한 생물학적 model로 구현하지 않는다. generator를 pseudo-hippocampal system, solver를 task knowledge를 가진 system으로 대응시켜 replay의 계산적 기능을 가져온다.

## 2.3 심층 생성 모델

GAN과 VAE 같은 deep generative model은 관측 data distribution에서 새 sample을 생성할 수 있다. generator가 과거 distribution을 충분히 근사한다면 raw data 없이도 rehearsal set을 만들 수 있다. 하지만 generator 자체도 새 domain을 학습할 때 과거 distribution을 잊을 수 있으므로 generator에도 replay가 필요하다.

# 3. 생성 재생 (Generative Replay)

## 3.1 제안 방법

시점 i-1의 scholar는 generator G_(i-1)와 solver S_(i-1)로 구성된다. 새 task i의 real sample (x_i, y_i)가 오면, 이전 generator에서 pseudo-input hat-x를 뽑고 이전 solver에서 pseudo-label hat-y를 얻는다.

\[
(\hat x,\hat y),\quad \hat x\sim G_{i-1},\quad \hat y=S_{i-1}(\hat x).
\]

새 solver (S_i)는 real pair와 replay pair의 mixture로 학습한다. 새 generator (G_i)도 새 real data와 과거 generated data를 함께 학습해 누적 distribution을 유지한다. replay ratio는 새 knowledge의 plasticity와 old knowledge의 stability를 조절한다.

solver target은 hard class label 또는 이전 solver의 soft output일 수 있다. soft target은 old decision boundary에 대한 더 많은 정보를 전달하는 distillation 역할을 한다. generator와 solver가 함께 다음 세대 scholar가 되므로, raw data를 보존하지 않고도 이전 task를 다른 model에 “가르칠” 수 있다.

## 3.2 예비 실험

preliminary experiment는 이전 domain sample이 없는 LwF가 왜 실패할 수 있는지 보인다. 새 domain input에 대한 old-model output만 맞추면, old domain의 function을 제약하는 signal이 약하다. DGR은 generated old-domain input을 제공해 그 영역에서 solver를 직접 제약한다.

# 4. 실험 (Experiments)

## 4.1 독립 과제의 순차 학습

서로 다른 image-classification task를 순서대로 학습한다. naive sequential training은 이전 task accuracy가 급락한다. real-data replay는 upper reference로 높은 성능을 보이지만 raw storage가 필요하다. DGR은 generated sample만으로 이 reference에 가까운 retention을 보이며 LwF보다 안정적이다.

## 4.2 새 domain 학습

동일하거나 관련된 label space를 갖지만 input domain이 달라지는 조건을 시험한다. generator가 이전 domain style을 재현하고 solver가 old target을 부여한다. generated image quality가 불완전해도 decision에 필요한 분포를 충분히 보존하면 forgetting을 줄일 수 있다.

## 4.3 새 class 학습

class-incremental setting에서는 label set이 단계마다 늘어난다. 새 class data만 학습하면 classifier가 최신 class 쪽으로 bias된다. DGR은 old class pseudo-example을 섞어 output balance를 유지한다. 평가에서는 task identity를 알려주는 조건과 알려주지 않는 조건의 차이를 구분한다.

# 5. 논의 (Discussion)

DGR은 raw past data를 저장하지 않고 pseudo-data를 생성한다는 장점이 있다. 그러나 generator parameter 자체가 과거 sample에 관한 정보를 담으므로 privacy가 자동으로 보장되는 것은 아니다. generator가 training example을 그대로 memorize하거나 추출 가능하게 만들 수 있고, generation error가 세대마다 누적될 수 있다.

성능은 generator가 old input distribution을 얼마나 충실히 재현하는지에 의존한다. 복잡한 high-resolution 또는 language distribution에서는 2017년 실험보다 어려운 문제가 된다. task 수가 늘면 generated mixture의 tail mode가 사라지는 **generative forgetting**과 compute 증가가 생길 수 있다.

논문은 sleep-time이라는 용어를 사용하지 않지만, 새 경험을 받는 단계와 internally generated replay로 과거 지식을 rehearsal하는 단계를 분리한 선행 구조다. 다만 replay는 training loop 안에서 이루어지며, 실제 service의 idle window·rollback·deletion·capacity policy는 다루지 않는다.

# 원문 구조 안내

뒤 원문 부록에는 scholar architecture, algorithm, preliminary comparison, independent/domain/class-incremental 실험, Figure 1–7과 Table 1, reference 및 appendix를 v3 PDF 그대로 제공한다.
