# 초록 (Abstract)

transformer 기반 대규모 언어 모델은 long-horizon 과제에 점점 더 많이 쓰이지만, attention 기제는 context 길이에 대해 잘 확장되지 않는다. 이 논문은 이 문제를 다루기 위해 **잠(sleep)과 유사한 consolidation 기제**를 연구한다. 모델은 key-value cache를 비우기 전에 주기적으로 최근 context를 지속적인 fast weight로 변환한다. 이 잠 동안 모델은 축적된 context 위에서 **N번의 오프라인 recurrent pass**를 수행하며, **학습된 local rule**을 통해 state-space model(SSM) block 안의 fast weight를 갱신한다. 추론 시점에서 보면 이는 추가 계산을 잠 쪽으로 옮기면서도 깨어 있을 때(wake-time)의 예측 지연시간은 그대로 보존하는 방식이다.

저자들은 cellular automaton과 multi-hop graph retrieval을 포함한 통제된 합성 과제, 그리고 일반 transformer와 SSM-attention hybrid 모델이 모두 실패하는 현실적인 수학 추론 과제에서 이 방법을 시험한다. 그리고 sleep duration N을 늘리면 성능이 개선되며, **더 깊은 추론을 요구하는 예제일수록 이득이 가장 크다**는 것을 보인다.

# 1. 서론 (Introduction)

LLM은 보통 transformer 아키텍처[58]에 기반하며, context를 attention cache에 저장했다가 필요할 때 과거 token을 꺼내 쓴다. 이 기억 기제는 성능의 핵심이지만 확장성이 나쁘다. 전체 attention 계산량은 context 길이에 대해 이차로 늘고, cache memory는 선형으로 늘어난다.

최근의 효율적 sequence model[48, 20, 18, 2]은 고정 크기의 fast weight memory[61, 17, 49]를 full self-attention 사이에 끼워 넣어 이 비용을 완화한다. 이 hybrid 설계는 서로 보완하는 두 형태의 기억을 결합한다. 최근 token에 고충실도로 접근하는 attention과, 활성 context window 너머의 정보를 압축해 담는 weight 기반 기억이다. hybrid 모델은 이제 대규모 frontier 모델 사이에서도 흔하다[56].

그러나 **확장 가능한 기억(scalable memory)이 곧 확장 가능한 추론(scalable reasoning)은 아니다.** fast weight memory가 장거리 recall을 지원할 수는 있지만[48], KV cache에 더 이상 존재하지 않는 token 위에서 깊은 계산까지 지원할 수 있는지는 분명하지 않다. 저자들은 저장해야 할 정보량을 고정한 채 요구되는 추론 깊이만 늘려도, 동일한 token budget 아래에서 vanilla SSM-attention hybrid 모델의 성능이 저하된다는 것을 발견한다. 이는 병목이 선행 연구[32, 2]가 지목한 단순한 memory capacity가 아니라, **축출된(evicted) context를 유용한 내부 상태로 변환하는 데 쓸 수 있는 계산량**임을 시사한다.

**잠(Sleep).** 동물에서 단기 기억이 장기 기억으로 이전되는 과정은 hippocampal replay[38]가 뒷받침하는 것으로 여겨지며, 특히 잠 중에 일어난다고 본다[47]. 이 국면에서 단기 hippocampus 기억이 재활성화되어 피질 시냅스 가중치로 통합된다. 잠은 동물이 외부 자극에 반응하지 못하게 만들기 때문에, 그 비용을 정당화할 만큼의 인지적 이득을 제공해야 한다고 볼 수 있다[47]. 저자들은 이 생물학적 과정에서 착안해 context-window 기억을 지속적 가중치로 이전하는 방법을 제안한다. 추론 중 모델의 context window가 가득 차면 모델은 "잠"에 들어가, 축적된 context 위로 여러 번의 forward pass를 수행하며 학습된 local rule을 통해 fast weight를 재귀적으로 갱신한다. 동물의 잠과 마찬가지로 이 국면 동안 모델은 외부 input token을 전혀 받지 않는다. consolidation이 끝나면 context window를 비우고, 갱신된 fast weight를 지닌 채 작동을 재개한다. 학습 시에는 잠 이후의 과제 성능을 최대화하도록 전체 과정을 통과하는 backpropagation으로 end-to-end 최적화한다.

이 아키텍처는 depth-recurrent 또는 looped neural network에 관한 결과[25, 19, 4]에서도 동기를 얻는다. 선행 연구는 동적 깊이 모델이 순차 추론 과제에서 고정 깊이 모델을 능가할 수 있고, 예측에 쓰는 계산량을 키움으로써 고정 깊이 모델이 풀지 못하는 어려운 문제 사례를 풀 수 있음을 보였다. **저자들의 핵심 통찰은 recurrence를 예측만이 아니라 기억 consolidation에도 쓸 수 있다는 것이다.** 관측한 token을 쓸모 있는 weight memory로 변환하는 일 자체가 사소하지 않은 계산이며, 한 번의 pass로 달성 가능하리라는 보장이 없다. 실제로 gradient descent를 비롯한 많은 학습 알고리즘은 반복적인 weight 갱신을 통해 개선된다. 따라서 fast weight 형성 단계에 recurrent 계산을 더 많이 배정하면, 모델은 context를 이후 예측을 뒷받침하는 표현으로 변환할 단계를 더 많이 갖게 된다. 저자들은 recurrence의 깊이, 즉 sleep duration을 늘리면 잠 이후의 추론이 개선됨을 보인다. 기존 looped 모델과 달리 이 모델은 **예측 시점에 loop를 돌 필요가 없다.** 추가 계산은 이미 이후의 단일 pass 예측을 뒷받침할 fast weight를 형성하는 데 소모되었기 때문이다.

저자들은 모델이 이미 축출된 context에 관한 질문을 단 한 번의 forward pass만으로 답해야 하는, 신중히 설계된 합성 과제 위에서 LLM sleep을 도입하고 평가한다. 이 합성 과제들은 memory load를 고정한 채 추론 깊이만 바꿀 수 있게 해 주며, sleep-time 계산이 일시적 context를 이후 추론을 뒷받침하는 fast weight로 변환할 수 있는지를 깔끔하게 스트레스 테스트한다. 기여는 다음과 같이 정리된다.

- 통제된 설정에서, 문제의 추론 깊이가 증가하면 Gated Delta Net(GDN) 같은 vanilla State-Space Model이 **충분한 fast weight 용량을 갖고 있음에도** 실패함을 보인다.
- recurrent 계산과 fast weight memory block을 결합한 아키텍처를 제안하고, 이 아키텍처에서 recursion 횟수를 늘리면 GDN 대비 성능이 개선됨을 보인다. **가장 깊은 추론을 요구하는 문제 사례에서 이득이 가장 크다.**
- 사전학습된 LLM 초기화를 사용해 자연어 수학 추론 데이터셋인 GSM-Infinite에서 아키텍처의 유효성을 추가로 검증한다.

전체적으로 이 결과들은 **잠과 유사한 오프라인 recurrence가 축출된 context를 이후 추론을 뒷받침하는 weight로 조직화할 수 있다**는 중심 주장을 지지한다.

# 2. 관련 연구 (Related Work)

**Fast weight와 linear recurrent neural network.** linear RNN 또는 SSM은 sequence 길이에 따라 이차로 커지는 KV cache 대신 online fast weight memory를 유지하는 것으로 볼 수 있다. 이 관점에서 linear attention은 고정 크기의 행렬값 state 위에서 key-value 매핑을 쓰고 조회하는 recurrent 갱신에 해당한다[34, 49]. 최근 변형들은 delta-rule 갱신과 gate로 이 기억을 개선해 더 선택적인 쓰기·덮어쓰기·망각을 가능하게 했다[60–62, 17]. 이 기제들이 최근의 효율적 hybrid 언어 모델[26, 45]의 바탕이며, linear network가 recall·throughput·memory 사이에서 유리한 tradeoff를 제공하는 이유를 설명해 준다. 다만 고정된 memory 크기 때문에 정확한 복사와 검색에서는 여전히 full attention에 못 미치는 경우가 있다고 선행 연구[2, 32]가 지적했다. 저자들은 이들과 달리, **저장할 정보량을 고정하더라도 과제를 푸는 데 필요한 추론 깊이가 증가하면 그런 모델들이 실패할 수 있음**을 보인다.

**Context compression.** test time에 긴 context를 다루기 위해 문맥 정보를 응축하는 방법이 여럿 있다. Ge et al.[23]은 언어 모델로 긴 context를 더 짧은 hidden state 시퀀스로 압축해 원래의 긴 context 대신 언어 모델에 넘긴다. Eyuboglu et al.[22]은 오프라인 self-study로 full-context cache를 대체할 수 있는 작은 KV cache를 학습한다. 이 계열은 오프라인 계산을 한 번 써서 긴 context를 나중에 재사용 가능한 compact state로 바꾼다는 목표를 저자들과 공유한다. 다만 이들은 attention context에 남는 것을 짧게 만드는 반면, 저자들의 방법은 **축출된 context를 weight 기반 기억으로 이전한다.**

**Context distillation.** context distillation[52, 3]은 활성 context를 모델 가중치로 증류하는 것을 목표로 한다. context 없는 모델이 context를 가진 teacher를 모방하거나[52, 3, 9], context를 재구성하거나[13], 그 이어질 내용을 예측하거나[9, 13], 그에 관한 질문에 답하도록[54, 10, 9] 학습시키는 방식이다. 저자들의 방법은 미리 정의된 loss에 대해 gradient descent를 하는 대신, **학습된 recurrent forward pass**를 사용해 context를 weight로 이전한다.

**Test-time training.** Tandon et al.[55]은 full attention을 sliding-window attention으로 바꾸고 MLP layer의 일부에 대해 test-time gradient 갱신을 수행한다. 추론 시점에 이들의 방법은 관측된 context에 대해 표준 cross-entropy loss를 최적화하며, 장거리 정보를 full KV cache가 아니라 임시적인 parameter 갱신에 저장한다. 각 context chunk를 증류할 때 gradient step은 한 번만 수행한다. 반면 저자들의 방법은 **학습된 recurrent forward pass를 memory-update rule로 사용**하므로, 고정된 스칼라 목적함수에 대한 1-step gradient descent에 대응할 필요가 없는 더 유연한 형태의 consolidation이 가능하다. 또한 이들은 주로 일반 web-text 데이터의 perplexity로 평가하는데, 여기서는 검색 요구와 추론 요구가 뒤엉켜 있다. 저자들은 대신 추론 깊이와 문제 길이를 독립적으로 통제하는 합성 과제를 사용하여, **추론 깊이가 증가할 때 추가적인 sleep-time 계산이 가장 유익함**을 보인다. Zhang et al.[63]은 현재 context chunk로부터 모델 가중치를 갱신하는 LoRA adapter를 붙이고 이를 강화학습(RL) 설정에서 평가한다. 저자들과 달리 이들의 방법은 chunk당 가중치를 **한 번만** 갱신한다.

**Depth-recurrent 모델.** 언어 모델의 깊이를 늘리면 표현력이 커진다는 것은 알려져 있다[40]. depth-recurrence는 transformer 모델의 깊이를 늘리는 한 방법이며, 이들을 Turing complete로 만드는 한 방법이기도 하다[19]. 나아가 이런 모델의 깊이는 적응적일 수 있다[25, 21, 50, 5]. 최근 연구는 이런 깊이 적응형 언어 모델을 처음부터 학습하는 방식[24, 65]과 post-training 목적함수로 쓰는 방식[39] 양쪽으로 대규모까지 확장했다. depth-recurrent 모델을 어떻게 학습하는 것이 최선인지에 대한 상세 분석은 recurrent depth를 training compute와 함께 확장해야 한다고 제안한다[46, 51].

**오프라인 계획(Offline planning).** 구조화된 환경에서의 성공적 계획은 새로 관측한 정보를 이전 상태에 관한 기억과 결합할 것을 요구하는 경우가 많다. 오랜 관점은 동물이 이 통합을 선택 시점에 online으로 수행한다는 것이었다[57, 42]. 그러나 선택 시점에 먼 기억을 통합하는 일은 시간이 많이 들 수 있고, 과제를 하지 않는 휴지기의 오프라인 계획이 그 비용을 상각할 수 있다[42]. 이 관점에 부합하게 Momennejad et al.[42]은 휴식 중 오프라인 replay의 신경학적 증거가 사람 피험자의 계획 성능 향상을 예측함을 보인다. 최근 기계학습 쪽 연구도 인공 신경망으로 관련 기제를 연구한다. Lin et al.[35]은 LLM이 사용자로부터 예상되는 질문을 생성하고 그것을 풀기 위해 필요한 양을 미리 계산하게 함으로써 오프라인 계산을 확장하는 방식을 제안한다. Chalvidal et al.[12]은 RL 환경에서 single-layer network를 학습시켜 재귀적인 Hebbian 유사 weight 갱신이 빠른 적응을 뒷받침함을 보인다. 이 논문에서 저자들은 **잠과 유사한 오프라인 국면에서 fast weight를 재귀적으로 갱신하면, 엄격한 예측 국면 지연시간 제약을 지키면서도 축출된 context에 대한 추론이 개선됨**을 보인다.

**잠(Sleep).** 여러 기계학습 방법이 생물학적 잠에서 영감을 얻었다. 에이전트가 모델이 생성한 trajectory[28, 53, 27]나 replay buffer[41]로 학습하는 RL 방법들이 여기 속한다. wake-sleep과 contrastive divergence 방법도 생성 모델을 학습하는 오프라인 국면에 잠의 비유를 쓴다[30, 11]. Lin et al.[35]은 자신들의 오프라인 계획 국면을 "sleep"이라 부른다. Behrouz et al.[6]은 RL, parameter expansion, teacher-student distillation, 합성 데이터 생성에 기반한, 잠에서 영감을 얻은 언어 모델 기억 consolidation 방법을 연구한다.

> **원문 각주 2.** 해당 OpenReview 투고는 저자들의 원래 제목과 겹치는 "Language Models Need Sleep"이라는 제목을 갖고 있다. 저자들은 제목을 정할 당시 이를 알지 못했고, 혼동을 피하기 위해 자신들의 제목을 수정했다.

# 3. 예비 사항 (Preliminaries)

## 3.1 Sequence mixer

**Attention.** Softmax attention[58]은 각 token이 query-key 유사도에 따라 이전 token들로부터 정보를 가져오는 sequence-mixing 연산이다. timestep t의 token 표현 x_t에 대해 다음과 같이 정의한다.

\[
q_t = W_Q x_t,\qquad k_t = W_K x_t,\qquad v_t = W_V x_t. \tag{1}
\]

여기서 (q_t, k_t, v_t \in \mathbb{R}^d)는 **열벡터**이고, (W_Q, W_K, W_V)는 shape가 호환되는 학습된 projection 행렬이다. self-attention은 이전의 모든 key (k_t)와 value (v_t)를 (K_t=[k_1,\dots,k_t]^{\top}\in\mathbb{R}^{t\times d}), (V_t=[v_1,\dots,v_t]^{\top}\in\mathbb{R}^{t\times d})에 저장한 뒤 다음을 계산한다.

\[
o_t = V_t^{\top}\,\mathrm{softmax}\!\left(\frac{K_t q_t}{\sqrt{d}}\right). \tag{2}
\]

이로써 (x_t)는 이전의 어떤 token에도 attend할 수 있지만, 크기가 sequence 길이에 따라 선형으로 커지는 KV cache (K_t)와 (V_t)를 저장해야 한다.

**Linear recurrent layer.** 반면 많은 SSM 계열 아키텍처를 포함하는 linear recurrent layer는 과거를 **고정 크기 fast-weight state**에 저장한다. 단순한 Mamba2 방식[17]의 갱신은 gated Hebbian 유사 외적 규칙[29, 49]으로 쓸 수 있다.

\[
S_t = \alpha_t S_{t-1} + \beta_t\, v_t k_t^{\top},\qquad o_t = S_t q_t. \tag{3}
\]

여기서 (\alpha_t \in (0,1))은 데이터 의존적 forget gate, (\beta_t \in (0,1))은 데이터 의존적 input gate이며, **둘 다 (x_t)로부터 계산된다.** KV cache (K_t), (V_t)와 달리 fast-weight (S_t)는 (t)에 따라 크기가 커지지 않는다. 이는 linear recurrent layer를 더 memory-효율적으로 만들지만 동시에 더 손실적(lossy)으로 만든다. 과거 token들이 고정 크기의 weight 기반 기억으로 압축되어야 하기 때문이다. **실험에서 저자들은 이 갱신에 delta-rule 보정을 더한 Gated Delta Network(GDN)를 사용한다. 다만 구체적인 갱신 규칙 자체는 논의에 중요하지 않다.**

언어 모델에서 sequence-mixing layer는 normalization, residual connection, MLP layer와 결합해 하나의 block을 이룬다. sequence-mixing layer가 attention인 block을 (B_\ell^{\mathrm{attn}}), linear recurrent layer인 block을 (B_\ell^{\mathrm{ssm}})으로 쓴다.

예컨대 attention만 쓰는 언어 모델은 embedding layer와 output projection 사이에 attention block을 (D)번 쌓아 만든다.

\[
\mathrm{Embed} \to B_0^{\mathrm{attn}} \to \cdots \to B_\ell^{\mathrm{attn}} \to B_{\ell+1}^{\mathrm{attn}} \to \cdots \to B_{D-1}^{\mathrm{attn}} \to \mathrm{OutProj}. \tag{4}
\]

**Hybrid 모델.** 최근의 hybrid sequence model[48, 20, 18, 2]은 고정 크기 fast-weight memory를 가진 SSM block[61, 17, 49]을 self-attention layer 사이에 끼워 넣어 그 비용을 완화한다. 예를 들면 다음과 같다.

\[
\mathrm{Embed} \to B_0^{\mathrm{attn}} \to B_1^{\mathrm{ssm}} \to B_2^{\mathrm{attn}} \to B_3^{\mathrm{ssm}} \to \cdots \to B_{D-1}^{\mathrm{attn}} \to \mathrm{OutProj}. \tag{5}
\]

## 3.2 합성 추론 과제

먼저 통제된 설정에서 변경 사항을 이해하기 위해 두 개의 합성 과제를 연구한다.

**Rule 110.** Rule 110[15]은 고정된 국소 전이 규칙에 따라 이진 문자열을 발전시키는 단순한 1차원 이진 cellular automaton이다. (t)단계 뒤의 Rule 110을 예측하는 일반 문제는 P-complete이며[43], 알려진 효율적인 일반 병렬 지름길이 없다. 따라서 신경망이 (t)번째 상태를 예측하도록 학습시키는 것은 모델이 **깊은 순차 계산을 수행할 수 있는지**를 보는 좋은 시험이 된다.

**Depo.** Depo는 언어 모델의 추론 깊이를 평가하기 위해 Allen-Zhu[1]가 도입한 multi-hop 지식 검색 과제다. 각 sequence는 뒤섞인 유향 cycle과 그에 뒤따르는 query들로 구성된다. 각 query는 시작 노드에서 (k)개의 나가는 edge를 따라간 뒤 도달하는 노드를 묻고, (k)가 클수록 더 깊은 그래프 순회를 요구한다.

이 과제들은 sequence 길이를 고정한 채 추론 요구량을 바꿀 수 있게 하여, 모델의 **추론 능력을 정보 검색 능력으로부터 분리**한다.

# 4. 동기 예제: attention-SSM hybrid 모델은 더 이상 attend할 수 없는 context에 대해 추론할 수 있는가?

attention-SSM hybrid 모델은 흔히 fast-weight memory가 제한된 attention window를 보완할 수 있다는 발상[48], 즉 과거 token이 직접 접근 불가능해진 뒤 그 정보를 압축해 둔다는 발상으로 정당화된다. 이 절에서 저자들은 이 hybrid 기제가 **실패하는** 경우를 탐색한다.

cellular automaton Rule 110[15]에 기반한 예제를 보자. 이 설정에서 모델은 각각 Rule 110의 초기 상태를 나타내는, 서로 독립인 길이 24의 이진 문자열 4개로 학습된다. 여기서는 문자 단위 tokenizer를 쓴다(즉 '0'과 '1'이 token이다). 네 상태는 서로 무관하다(즉 이전 상태를 전개해 얻은 것이 아니다). 길이 (T := 24 \times 4 = 96)인 네 이진 문자열을 모두 처리한 뒤, 모델은 나중에 (t)번 전이한 각 상태의 **첫 비트**를 예측해야 한다. 상태들 뒤에 4개의 label token이 따라오므로 전체 sequence 길이 (T)는 100이다. 예시 sequence는 다음 형태다.

```
0101…1101 | 1101…1000 | 1101…0110 | 0011…0110 |  1      0      1      0
 state0        state1        state2        state3   label0 label1 label2 label3
        ← 각 상태 24비트 →                          ← answer token 4개 →
```

첫 answer token(label0)은 state0(`0101…1101`)을 (t)번 전개한 뒤 그 첫 비트를 취해 얻으며, 나머지도 마찬가지다. (t)는 이 과제를 푸는 데 요구되는 **추론 깊이를 통제**한다. (t=0)(전개 없음)이면 단순한 첫 비트 검색 과제가 되고, (t)가 커질수록 과제는 어려워진다.

SSM이 과거 정보를 제공함으로써 self-attention을 보완할 수 있는지를 스트레스 테스트하기 위해, 저자들은 엄격한 context window 크기와 **hard-eviction 제약**을 부과한다. 24 token마다 context window를 비우며 이를 (L = 24)로 표기한다. 이는 모델이 한 번에 오직 하나의 상태만 context에서 볼 수 있고, 다음 상태로 넘어가기 전에 KV cache (K_t)와 (V_t)가 완전히 축출되므로 **이 정보를 fast weight (S_t)에 온전히 부호화해야 함**을 뜻한다. hard eviction 경계는 `|`로 표시한다.

이 hard eviction 제약은 sequence를 자연스럽게 뚜렷한 두 국면으로 나눈다.

- **consolidation 국면** (예시 sequence의 처음 96 token): 이 동안 모델은 context를 fast weight (S_t)에 부호화해야 한다.
- **prediction 국면** (예시 sequence의 마지막 4 token): 이 동안 모델은 answer token을 예측한다.

저자들은 **prediction 국면 지연시간 제약**을 부과한다. prediction 국면 동안 각 answer token은 **표준 forward pass 한 번으로** 예측된다. 추가 loop나 chain-of-thought token은 예측 지연시간을 늘리므로 금지된다. 따라서 label을 예측하는 데 필요한 모든 정보는 prediction 국면이 시작되기 전에 이미 fast weight로 consolidate되어 있어야 한다.

이 hard eviction 제약 아래에서 표준 transformer는 무작위 추측보다 나을 수 없다. 예측이 이루어지기 전에 KV cache가 파괴되기 때문이다. SSM 또는 attention-SSM hybrid 모델은 초기 상태들을 fast weight에 저장할 수 있으므로 무작위 추측보다 나을 수 있다. 예컨대 이 과제를 푸는 한 가지 방법은 context가 가득 찼을 때 (t)단계 상태 전개를 한 번 시뮬레이션하고, 전개된 각 상태의 첫 비트를 fast weight에 저장한 뒤 예측 시점에 그 비트를 꺼내는 것이다.

그러나 **Figure 2a**는 4-layer GDN-attention hybrid 모델(attention → GDN → attention → GDN 배치)의 성능이 (t)가 커짐에 따라 급격히 떨어짐을 보여 준다. 이 하락은 선행 연구[32, 2]가 규명한 memory-capacity 한계 때문이 아니다. 저자들은 sequence 길이 (T)를 고정한 채 (t)만 바꾸었기 때문이다. 대신 그 어려움은 automaton을 (t)단계 시뮬레이션하는 데 필요한 **깊은 순차 계산**에서 오며, 고정 깊이 모델은 이 계산을 확장할 수 없다.

**과제 실패에 관하여.** 모델이 어떤 과제에서 실패하거나 성능이 저하된다고 말할 때, 저자들은 그 아키텍처가 무한한 데이터·계산·학습 시간으로도 결코 그 과제를 배울 수 없다는 뜻으로 말하는 것이 아니다. 주장은 **고정된 training-token budget 아래의 성능**에 관한 것이다. 이 budget이 통제된 설정이 중요한 이유는, 추론 집약적 데이터가 web 규모 corpus에서도 희소하기 때문이다. budget이 통제된 합성 과제는 대규모 사전학습에서 관측되는 현상과 부합하는 경향성을 더 이르고 더 뚜렷하게 드러낼 수 있다[1].

# 5. LLM Sleep: 오프라인 재귀적 기억 통합

이제 위 예제에 대한 해법을 소개한다. 저자들은 LLM 학습 중에 **잠**을 도입한다. 이 잠에서 모델은 context window가 가득 차 attention layer에서 token을 축출하기 전에, consolidation 국면 동안 recursion을 수행한다. 이렇게 하면 prediction 국면 지연시간 제약을 지키면서도 깊은 추론 과제(예: 동기 예제의 큰 (t))를 다룰 계산을 확장할 수 있다. 예컨대 (D)개 block 전체에 대해 loop를 돌면 다음과 같다.

\[
\mathrm{Embed} \to \Bigl( B_0^{\mathrm{attn}} \to B_1^{\mathrm{ssm}} \to \cdots \to B_{D-1}^{\mathrm{attn}} \Bigr)^{\times N} \to \mathrm{OutProj} \tag{6}
\]

여기서 위첨자 (\times N)은 아키텍처 위를 (N)번 looped pass 한다는 뜻이다.

**Algorithm 1: hard eviction을 사용하는 LLM sleep 학습.**

```
입력: token x, loss mask m, window 크기 L, sleep pass 수 N
 1: SSM fast weight S를 0으로 초기화
 2: x와 m을 길이가 최대 L인 겹치지 않는 chunk들로 분할
 3: 각 token chunk c와 그 loss mask m_c에 대해 반복:
 4:     h ← Embed(c)
 5:     if m_c 가 전부 0이면:                     ▷ consolidation 국면
 6:         for n = 1, …, N:
 7:             h, S ← Blocks(h, S)
 8:         end for
 9:     else:                                     ▷ prediction 국면
10:         h, S ← Blocks(h, S)
11:         L ← MaskedCE(OutProj(h), c, m_c)      ▷ masked cross entropy loss
12:     end if
13: end for
14: L을 backpropagate하고 optimizer step을 수행
```

**Figure 1**이 아키텍처를 자세히 설명한다. 저자들은 고정된 context-window 크기 (L)을 가지며 (L) token마다 attention cache가 완전히 축출되는 SSM-attention hybrid 모델에서 시작한다. (L) token마다 KV cache를 축출하기 **전에**, 모델은 (N)번의 recurrent pass를 수행해 식 (3)에 따라 SSM block 내부의 fast weight를 반복적으로 갱신한다. (N = 1)이면 vanilla SSM-attention hybrid 모델로 환원된다. 모델이 fast weight를 반복적으로 갱신하는 이 국면을 **sleep**이라 부른다.

fast weight를 재귀적으로 정련한 뒤 KV cache가 축출되고 다음 (L) token이 처리된다. 전체 context를 처리한 뒤, 모델은 정련된 기억과 현재 context에 기반해 **단 한 번의 forward pass로** 답을 예측한다. 모델은 다른 depth-recurrent 모델[19, 25]과 유사하게 식 (6)에 나타난 전체 계산 그래프를 통과하는 backpropagation으로 예측 오차를 최소화하도록 학습된다. **재귀적으로 정련된 feature vector를 통해 gradient가 흐르는 기존 depth-recurrent 모델과 달리, 여기서는 잠 이후 정련된 feature를 버리기 때문에 gradient가 정련된 fast weight를 통해 흐른다.** Algorithm 1이 학습 절차를 요약한다.

> **Figure 1 설명.** eviction 경계에서 SSM-attention hybrid는 attention cache를 버리기 전에 현재 context 위로 (N)번의 오프라인 recurrent pass를 수행한다. 이 recurrent pass들이 SSM block의 fast weight를 갱신하여, 이후의 예측이 wake-time looping 없이도 consolidate된 context를 사용할 수 있게 한다.

# 6. 실험 (Experiments)

실험은 (N)을 늘려 구현한 **더 긴 잠**이, attention cache에 더 이상 존재하지 않는 상태들에 대해 더 깊은 추론을 뒷받침하는 fast weight를 만들어 내는지를 시험한다. 이는 축출된 token을 저장하는 것 이상을 요구한다. 모델은 과거 context를 fast weight (S_t)에 부호화하되, cache가 비워진 뒤에도 사소하지 않은 계산을 뒷받침하는 형태로 부호화해야 하며, 그러면서도 예측 시점에는 단 한 번의 forward pass만 사용해야 한다. 저자들은 이 질문을 점점 더 어려운 설정에 걸쳐 평가한다. 먼저 cellular automaton 과제는 rollout step (t)를 바꿔, 축출된 각 상태에 대해 요구되는 추론의 깊이를 분리한다. 다음으로 Depo 과제[1]는 더 어려운 압축 문제를 더한다. 모델은 조각난 그래프를 fast weight에 부호화하고 나중에 그 위에서 **본 적 없는** multi-hop query에 답해야 한다. 마지막으로 GSM-Infinite[64]를 고려하며, 여기서는 사전학습된 Jet-Nemotron 2B[26]와 Ouro 1.4B[65]를 합성 수학 추론 데이터셋에서 fine-tuning한다.

**실험 세부 사항.** McLeish et al.[39]을 따라 모든 실험에 Muon optimizer를 사용한다. AdamW learning rate는 5e−5로 고정하고 Muon learning rate만 tuning한다. 4절과 6.1절에서는 hidden dimension (d = 256)의 4-layer GDN-attention hybrid 모델을 사용한다. Muon learning rate는 (N = 1) 모델에서 tuning하여 **no-loop 기준선에 유리하게** 하고, 선택된 값 2e−3을 모든 looped 모델에 사용한다. 6.2절에서는 Jet-Nemotron 아키텍처[26]를 사용하는데, 이는 일부 attention layer를 Jet layer로 교체하여 Qwen 2.5 1.5B에서 fine-tuning한 SSM-attention hybrid 모델이며, Jet layer는 GDN의 고정 convolution 대신 dynamic convolution을 쓴다. Allen-Zhu[1]의 작은 모델 크기에 대략 맞추기 위해 hidden dimension (d = 512)의 10-layer 모델을 처음부터 학습한다. tuning 규약은 위와 동일하게 적용한다. (N = 1) 기준선에서 Muon learning rate를 tuning하고 looped 모델에는 2e−3을 사용한다. 6.3절에서는 사전학습된 Ouro 1.4B[65]와 Jet-Nemotron 2B[26] 모델을 사용하며, McLeish et al.[39]을 따라 Muon learning rate를 1e−3으로 둔다. automaton 실험은 A6000 GPU-day 1일 미만을 요구한다. Depo와 GSM-Infinite 실험은 run당 대략 1–2 H100 GPU-day를 요구한다. batch size는 automaton 512, Depo 128, GSM-Infinite 256을 쓴다. 공정한 비교를 위해 random seed를 고정하여 모든 run이 정확히 동일한 데이터 순서를 사용하도록 보장한다.

## 6.1 과제: Cellular automaton

4절에서 (t)가 클 때 vanilla SSM-attention hybrid 모델이 automaton 과제에서 실패하는 모습을 보았다. hybrid 모델은 기억 consolidation을 수행할 때 계산을 확장할 수 없기 때문이다. **Figure 2b**에서는 Figure 2a와 동일한 아키텍처, 즉 attention → GDN → attention → GDN 배치의 4-layer GDN-attention hybrid 모델을 사용한다. 저자들의 방법은 여기에 5절에서 논의한 consolidation 국면의 '잠'을 추가로 사용하며, 이때 recurrence로 fast weight를 반복 갱신한다. 여기서는 **2회에서 4회까지의 recurrent 갱신**을 연구한다.

이 looped hybrid 아키텍처를, 상당한 추론 계산을 요구하며 비재귀 아키텍처에게 도전적인 설정인 (t = 32)에서 학습한다. Figure 2b에서 "2 loops", "3 loops", "4 loop"는 모델이 기억 consolidation에 잠을 사용함을 뜻하고, "no loop"가 기준선이다. Figure 2b는 **non-looped 모델이 거의 5B training token 이후에도 exact accuracy 약 10%에 그쳐 무작위 추측에 가깝게 머무름**을 보여 준다. 오프라인 pass를 추가하면 동일 token budget 아래에서 학습 속도와 최종 정확도가 모두 개선된다. **2 loops는 약 20% 정확도를 달성하고, 3 loops와 4 loops는 30%를 넘는다.** 이 run들 사이에서 context 길이, eviction 규칙, prediction 국면 계산이 모두 고정되어 있으므로, 개선은 **잠 동안의 추가적인 consolidation-time 계산에서 온다.**

> **Figure 2 설명.** (N)을 늘리면 cellular automaton에서 성능이 개선된다. **왼쪽(2a):** 각 곡선은 동기 예제 절과 같이 hybrid attention-SSM 아키텍처에 대한 서로 다른 rollout step (t)를 나타낸다. (t)를 늘리면 vanilla attention-GDN hybrid 모델에게 과제가 어려워진다. 4-step과 8-step run은 더 일찍 수렴하므로 early stop한다. (표시된 곡선군은 4, 8, 10, 12, 14, 16 step이며 x축은 0M–40M training token, y축은 25%–100%.) **오른쪽(2b):** 도전적인 추론 과제 ((t = 32))에서, 오프라인 sleep loop를 추가하면 단일 pass wake-time 예측을 유지하면서도 정확도가 개선된다. (x축 0B–4B training token, y축 0%–50%, "Random guessing" 기준선 표시.)

## 6.2 과제: Depo

다음으로 Allen-Zhu[1]가 도입한 (k)-hop 지식 검색 과제인 Depo에서 평가한다. 각 sequence는 뒤섞인 유향 cycle과 그에 뒤따르는 query들로 구성되며, 각 query는 시작 노드에서 (k)개의 나가는 edge를 따라간 뒤 도달하는 노드를 묻는다. (k)가 클수록 더 깊은 그래프 순회를 요구한다. Allen-Zhu[1]는 SSM이 **context를 저장하기에 충분한 fast weight 용량을 가지고 있음에도** 이 과제에서 transformer보다 실질적으로 못한다는 것을 보인다. 이는 병목이 저장만이 아니라, **저장된 edge를 이후의 multi-hop 검색을 뒷받침하는 표현으로 조직화하는 일**임을 시사한다[44]. Depo의 예시 sequence는 다음 형태다.

```
b->a, f->l, … | … | …, e->b | 1 hop after a: c  …  4 hops after e: d
←        뒤섞인 유향 cycle        →   ←         query와 answer        →
```

여기서 `|`는 eviction 경계를 나타내고, 원문에서 붉은 글씨는 answer token을 나타낸다.

저자들의 설정에서 각 cycle은 최대 75개 노드를 포함하고 최대 300 token에 걸친다. 더 짧은 사례는 test와 train 양쪽에서 300 token으로 left-padding한다. 그 뒤에 query-answer 부분이 따라오는데, 최대 60 token에 걸친 **10개의 query-answer 쌍**으로 이루어져 전체 sequence 길이는 (T = 360)이 된다. 모델의 window 크기는 (L = 75)이므로 **각 cycle은 네 개의 cache window에 걸쳐 조각난다.** 모델이 query의 답을 예측할 때 cycle context는 이미 KV cache에서 축출된 상태다. Depo는 두 가지 이유로 cellular automaton 과제보다 어렵다. 첫째, 각 cycle이 네 개의 cache window에 걸쳐 조각나는 반면 각 automaton 상태는 하나의 window 안에 들어간다. 둘째, automaton 과제에서는 (t)가 고정인 반면 여기서는 (k)와 시작 노드가 예제마다 무작위로 뽑히므로 모델이 **query-agnostic한 표현을 형성해야 한다.**

Depo에서 (k)는 과제 난이도를 통제한다. (k)가 클수록 답을 복원하기 위해 더 긴 multi-hop 순회를 수행해야 하므로 query가 더 어려워진다. Allen-Zhu[1]를 따라 학습 중에는 (k)를 ([1, 16])에서 균등 표집하고, held-out 예제에서 (k = \{1, 2, 4, 8, 16\})에 대한 test loss를 측정한다. **Figure 3**은 학습 step에 따른 held-out 예제의 test loss를 보여 주며, 각 subplot은 hop 수 (k \in \{1, 2, 4, 8, 16\})에 대응하고 각 곡선은 오프라인 loop 수 (N \in \{1, 2, 4\})인 모델들을 비교한다. **오프라인 loop 수를 늘리면 4-hop 이상을 요구하는 query에서 학습 속도가 개선됨**을 볼 수 있다. 1-loop 모델은 4-hop 이상의 어려운 query에서 거의 진전을 보이지 못하고, 2-loop 모델도 마찬가지로 8-hop 이상에서 정체한다. 저자들의 training budget 안에서는 **오직 4-loop 모델만이 가장 어려운 16-hop 과제에서 개선되기 시작한다.**

> **Figure 3 설명.** (N)을 늘리면 Depo에서 성능이 개선된다. (k)-hop 지식 검색 과제에서 4-layer GDN-attention hybrid의 test loss. 오프라인 loop를 추가하면 학습이 가속되며, 특히 추론 집약적인 더 높은 hop query에서 그렇다. (x축 0k–100k step, loss축 0–2, hop 수별 subplot.)

## 6.3 과제: GSM-Infinite

통제된 과제에서 관측된 경향이 사전학습된 LLM으로 확장되는지 시험하기 위해, GSM8K[14]를 본떠 만든 합성 추론 benchmark인 GSM-Infinite[64]에서 평가한다. GSM-Infinite는 통제된 분석이 가능할 만큼 구조적이면서도, 여기에 학습하면 다른 과제에서의 추론 능력이 개선될 수 있을 만큼 현실적이다[33]. GSM-Infinite는 절차적으로 생성되므로 Kabra et al.[33]과 유사하게 동일 분포에서 서로 다른 학습·평가 데이터셋을 생성할 수 있다. 평가 집합은 **held-out 예제 1,600개**다. 이 데이터셋은 문제의 나머지 부분과 닮아 무시하기 어려운 distractor token을 추가함으로써 문제 길이를 통제하고, 문제를 푸는 데 필요한 산술 연산 수를 바꿈으로써 난이도를 통제한다. RULER[31] 같은 검색 중심 long-context 과제와 달리 GSM-Infinite에서는 단순한 retrieval-augmented 기준선이 실패하며[64], 이는 이 과제가 long-context 처리와 multi-step 추론을 **모두** 요구함을 가리킨다. GSM-Infinite는 추론에 최적화된 frontier 모델에게도 도전적이며, 요구되는 연산 수가 늘수록 이들의 정확도도 감소한다[64].

실험에서 각 문제는 **2,000–3,300 token**을 포함하고 연산 수는 ([1, 8])에서 균등 표집한다. 저자들은 **질문을 context 앞에 배치**하고 데이터에서 Chain-of-Thought trace를 제외하여, 모델이 오직 예측 시점의 단일 forward pass만으로 최종 답에 이르도록 강제한다. 이 순서는 모델이 긴 문제 context를 읽기 전에 query를 갖게 하여, 질문과 관련된 정보를 선택적으로 consolidate하고 filler token은 무시할 수 있게 한다. 모델의 context-window 크기는 (L = 2000)으로 설정하므로, 문제 전체가 활성 context window에 들어가지 않으며 모델은 예측 시점에 문제 context의 대부분에 attend할 수 없다.

사전학습 모델로부터 이 방법을 실체화하는 상보적인 두 방법이 있다. SSM-attention hybrid에서 시작해 sleep time recurrence로 fine-tuning하거나, depth-recurrent 모델에서 시작해 SSM memory layer를 추가하는 것이다. 저자들은 hybrid인 **Jet-Nemotron 2B**[26]와 recurrent인 **Ouro 1.4B**[65]를 fine-tuning하며 두 방법을 모두 탐색한다. Jet-Nemotron은 일부 attention layer를 Jet layer로 교체하여 Qwen 2.5 1.5B에서 fine-tuning한 SSM-attention hybrid 모델이며, Jet layer는 GDN의 고정 convolution 대신 dynamic convolution을 쓴다. Ouro는 looped attention-only 모델이므로, 저자들은 **MLP layer가 없는 Jet layer 6개를 삽입**하여 Ouro에 fast weight memory를 부여하되 전체 parameter 수는 **10% 미만** 증가시킨다.

Jet의 경우 전체 28개 block 중 **가운데 14개 block**에 대해 loop를 돈다. 가운데 block만 looping하는 것은 depth-recurrence 모델에서 흔한 관행이다[39, 24]. Ouro의 경우 사전학습 방식[65]을 따라 **전체 block**에 대해 loop를 돈다. 합리적인 batch size를 쓰면서 학습 중 memory 비용을 관리 가능하게 유지하기 위해, Ouro에는 (N = \{1, 2, 4\})를 사용한다. Jet은 전체 block의 절반에 대해서만 loop를 돌므로 (\{1, 2, 4, 6\})을 사용한다.

**Figure 4**는 학습 step에 따른 정확도 추세를 보여 주며, 각 subplot은 문제를 푸는 데 필요한 연산 수(2에서 8까지)에 대응한다. 처음부터 사전학습한 실험의 경향이 더 현실적인 수학 추론 설정에서도 지속됨을 볼 수 있다. 더 쉬운 2-연산·4-연산 문제에서는 loop 수와 무관하게 정확도가 종종 포화에 가까워지며, 특히 Ouro보다 fast weight memory 용량이 큰 Jet에서 그렇다. 그러나 요구되는 연산 수가 늘수록 loop 수 사이의 격차가 벌어진다. 추가적인 오프라인 recurrence는 6-연산과 8-연산 설정에서 최종 정확도와 학습 속도를 모두 개선한다.

- **Jet**의 경우 six loops가 6-연산 문제의 최종 정확도를 **0.742 → 0.812**로, 8-연산 문제를 **0.351 → 0.388**로 개선한다.
- **Ouro**의 경우 four loops가 6-연산 문제의 최종 정확도를 **0.419 → 0.615**로, 8-연산 문제를 **0.210 → 0.272**로 개선한다.

격차는 Ouro에서 더 넓은데, 이는 그 모델의 depth-recurrent 사전학습을 반영하는 것일 수 있다. 이 결과들은 **sleep-time 계산이 현실적인 수학 추론 데이터에서, 그리고 사전학습된 LLM에서도 multi-step 추론을 뒷받침할 수 있음**을 시사한다.

> **Figure 4 설명.** (N)을 늘리면 GSM-Infinite에서 성능이 개선된다. 학습 step에 따른 GSM-Infinite 정확도. subplot은 문제가 요구하는 산술 연산 수로 예제를 묶고, 색은 cache eviction 전에 사용한 오프라인 loop 수 (N)을 나타낸다. loop를 추가하면 연산 수가 많은 어려운 문제에서 정확도 개선이 가장 뚜렷하다. single-loop 모델은 축출된 context를 유용한 fast weight로 조직화할 sleep-time 계산이 더 적기 때문이다.
>
> **(a) Jet-Nemotron 2B** ((N \in \{1, 2, 4, 6\}), x축 300–700 step). 패널에 인쇄된 값(내림차순) — op2: 0.985, 0.985, 0.980, 0.979. op4: 0.995, 0.994, 0.993, 0.992. op6: 0.812, 0.799, 0.753, 0.742 및 "9% improvement" 주석. op8: 0.388, 0.372, 0.370, 0.351 및 "11% improvement" 주석. 본문은 op6의 0.742 → 0.812와 op8의 0.351 → 0.388을 no loop 대비 6 loops의 값으로 명시한다.
>
> **(b) Ouro 1.4B** ((N \in \{1, 2, 4\}), x축 300–1000 step). 패널에 인쇄된 값(내림차순) — op2: 0.868, 0.863, 0.857. op4: 0.932, 0.923, 0.903. op6: 0.615, 0.484, 0.419 및 "47% improvement" 주석. op8: 0.272, 0.210, 0.209 및 "30% improvement" 주석. 본문은 op6의 0.419 → 0.615와 op8의 0.210 → 0.272를 no loop 대비 4 loops의 값으로 명시한다.
>
> (원문은 각 패널에 곡선 끝점 값만 인쇄하고 값과 loop 수의 대응은 표시하지 않는다. 대응이 본문에 명시된 것은 위에 적은 op6·op8의 끝점뿐이다.)

## 6.4 Sliding-window eviction

지금까지는 모델의 context window가 가득 찰 때마다 완전히 축출된다고 가정했다. 대신 **sliding-window eviction** 전략을 쓸 수도 있다. 잠 이후 모델은 attention cache에 가장 최근의 (L - 1) token을 유지하고 더 오래된 token만 축출한다. 이는 추론 시점의 peak memory를 늘리지 않는다. sliding-window attention(SWA)에서처럼 활성 context는 여전히 (L) token으로 상한이 정해지기 때문이다. (N = 1)이면 이는 표준 SWA-SSM hybrid 모델[48]로 환원되고, (N > 1)이면 모델은 오래된 context가 attention cache를 떠나기 전에 추가적인 재귀적 consolidation을 수행한다.

저자들은 이 전략을 (L = 512)로 GSM-Infinite에서 평가하며, 이때 전체 sequence 길이 (T)는 window 크기의 대략 **4–6배**다. Ouro 1.4B를 (N \in \{1, 2, 4\})로 fine-tuning한다. 선행 연구[8]의 관측과 유사하게, 모델에 sliding-window KV cache에 대한 접근을 허용하면 새로 삽입한 Jet layer가 충분히 활용되지 않을 수 있음을 발견한다. 그래서 먼저 **한 epoch 동안 Jet layer만 warm up하고, 그 뒤 두 epoch 동안 전체 모델을 학습한다.** 이 SSM-only warm-up 단계는 attention-only 모델을 attention-SSM hybrid로 변환할 때 표준적인 절차다[59, 7, 26]. 그리고 (N > 1)일 때 **warm-up 단계에 hard eviction을 쓰는 것이 모델이 fast weight를 정련하는 법을 배우는 데 결정적**임을 발견한다.

**Figure 5**는 학습 step에 따른 정확도를 보여 주며, no loop로 표시된 곡선이 (N = 1)인 SWA-SSM hybrid 기준선에 해당한다. (N)을 늘리면 모든 연산 수에서 정확도가 개선되어 Figure 4의 경향과 일치한다. window 크기가 (L = 2000)이던 Figure 4와 달리, 이 기준선은 **추론 부담이 가장 적어 오히려 distractor token 아래에서의 검색을 더 직접적으로 압박하는 2-연산 문제에서조차 성능이 나쁘다.** 반면 loop를 쓰면 정확도가 **0.596에서 0.905로, 52% 개선**된다. 이는 활성 attention window가 sequence 길이보다 몇 배 작을 때, 더 긴 sleep duration이 multi-step 추론뿐 아니라 **관련 context의 압축과 검색에도 도움이 됨**을 시사한다.

> **Figure 5 설명.** sliding-window eviction을 쓴 GSM-Infinite에서 (N)을 늘리면 정확도가 개선된다. 학습 step에 따른 GSM-Infinite 정확도. window 크기 (L = 512)로 Ouro 1.4B를 fine-tuning하고 sleep pass (N \in \{1, 2, 4\})를 비교한다. (x축 0–600 step.) 패널에 인쇄된 값(내림차순) — op2: 0.905, 0.823, 0.596 및 "52% improvement" 주석. op4: 0.926, 0.892, 0.839 및 "10% improvement" 주석. op6: 0.320, 0.265, 0.251 및 "27% improvement" 주석. op8: 0.137, 0.118, 0.116 및 "18% improvement" 주석. 본문은 op2에 대해 0.596 → 0.905를 no loop 대비 loop 사용 값으로 명시한다.

## 6.5 학습 throughput

여기서는 이 방법이 SWA-SSM hybrid 기준선 대비 학습 throughput(초당 처리 token 수)에 어떤 영향을 주는지 분석한다. 6.3절의 Ouro 1.4B 모델을 사용한다.

**Context window에 걸친 recurrence.** 모든 token 위치를 병렬로 처리할 수 있는 표준 teacher-forced transformer 학습과 달리, 저자들의 학습은 context window에 걸쳐 recurrent하다. window (j+1)을 처리하려면 모델이 먼저 window (j) 처리를 마치고 fast weight를 정련하는 (N)번의 sleep pass를 수행해야 하기 때문이다. 갱신된 fast weight가 window (j+1)을 처리하는 데 쓰이는 state가 되어, window 사이에 순차적 의존성이 생긴다. 이는 sequence 축을 따른 완전한 병렬화를 막는다. 그러나 이 sequence-축 병렬성의 상실이 반드시 벽시계 학습 시간을 저해하지는 않는다. **window 크기 (L)이 GPU를 포화시킬 만큼 충분히 크면 그렇고**, (T)와 (L)이 모두 큰 long-context 학습 체제에서 그런 일이 일어날 수 있다(Figure 6a).

**Recurrent-depth 비용.** 또한 다른 depth-recurrent 모델과 마찬가지로 **학습 비용은 recurrent step 수 (N)에 대해 대략 선형으로 증가한다**(Figure 6b). 그러나 실험에서 보듯이 recurrence를 늘리면 비재귀 모델 대비 과제 성능이 일관되게 개선된다.

> **Figure 6 설명.** context window에 걸친 recurrence는 학습 overhead가 미미하지만, recurrent depth는 비용을 선형으로 늘린다. NVIDIA H200 GPU 1대에서의 학습 throughput 비교. sequence 길이는 12,000으로 설정. **(a)** window 크기 (L)이 충분히 클 때, context window에 걸친 순차성은 완전 병렬 기준선 대비 throughput을 유의미하게 바꾸지 않는다. **(b)** throughput은 대략 (N)에 반비례한다. 각 설정마다 GPU 이용률을 최적화하도록 batch size를 tuning한다. (b)는 out-of-memory 오류를 막기 위해 context chunk 축에 걸친 activation checkpointing을 추가로 사용한다. FlashAttention 2[16]를 사용한다. (window 크기 1K / 2K / 4K; (a)의 y축 눈금 30 / 20 / 10 k tokens/s, (b)의 y축 눈금 20 / 10 / 5 k tokens/s.)

# 7. 논의와 한계 (Discussion and Limitations)

저자들의 방법은 추가적인 recurrent 계산을 consolidation 국면으로 옮김으로써 단일 pass prediction 국면 지연시간을 보존하지만, **이 이득은 공짜가 아니다.** 학습 중에는 (N)배 더 깊은 forward pass와 backward pass를 수행해야 하며, 이는 학습을 느리고 불안정하게 만들 수 있다. 이 문제들을 다루는 것은 recurrent-depth 학습에서 활발한 주제이며, implicit gradient[4], truncated backpropagation through time[24, 39], 그리고 학습을 안정화하는 여러 기법[46, 24] 등이 가능한 접근으로 거론된다.

잠은 학습을 context 차원과 depth 차원 양쪽으로 순차적으로 만든다. 그러나 **바로 그 순차성이 저자들이 고려한 과제들에서 이득이 나오는 이유이기도 하다.** 그 과제들의 해법 자체가 순차적이기 때문이다. 현대 기계학습이 겨냥하는 추론·시뮬레이션·의사결정 문제 다수가 이런 성질을 지니는 것으로 보인다[37]. 본질적으로 순차적인 과제를 완전히 병렬적인 계산으로 풀려 하면 **깨지기 쉬운 지름길 해법**을 조장하게 된다[36, 37].

# 8. 결론 (Conclusion)

저자들은 모델이 해당 context를 attention cache에서 축출하기 전에, 여러 번의 재귀적 forward pass를 수행해 fast weight를 반복적으로 정련하는 잠 유사 과정을 제안한다. vanilla attention-SSM hybrid 모델과 달리, 잠은 모델이 **더 이상 attend할 수 없는 과거 context에 대해 깊이 추론할 수 있게** 한다. 통제된 합성 과제들과 더 현실적인 수학 추론 benchmark 전반에서, recursion 횟수(즉 sleep duration)를 늘리면 축출된 context에 대해 깊은 순차 계산을 수행하는 모델의 능력이 개선됨을 보인다.

# Broader Impact

이 연구는 언어 모델의 기억 consolidation과 추론을 다루며, 이는 더 유능한 long-context 시스템을 만드는 데 중요한 재료다. 기여는 주로 방법론적이며, 통제된 합성 과제와 중간 규모의 사전학습 모델에서 평가되었다. 따라서 이 분야의 다른 연구를 넘어서는 위험이 있으리라 예상하지 않는다.

# 감사의 글 (Acknowledgements)

넉넉한 GPU 자원을 제공해 준 Modal에 감사한다.

# 참고문헌과 원문 구성에 관하여

원문은 참고문헌 [1]–[65]로 끝나며, **부록(appendix)은 존재하지 않는다.** 본문은 총 15쪽이고 표는 없다. Figure 1과 Algorithm 1은 6쪽, Figure 2와 Figure 3은 8쪽, Figure 4와 Figure 5는 10쪽, Figure 6은 11쪽에 있다. 번호가 붙은 수식은 (1)–(6) 여섯 개다. 뒤의 원문 보존 부록은 이 6개 figure, Algorithm 1, 6개 수식, 참고문헌 전체를 고정 v3 PDF 그대로 제공한다.

---

*원문: Sangyun Lee, Sean McLeish, Tom Goldstein, Giulia Fanti, "Do Language Models Need Sleep? Offline Recurrence for Improved Online Inference", arXiv:2605.26099v3 [cs.CL], 2026-06-05. DOI: 10.48550/arXiv.2605.26099. 라이선스: CC BY 4.0 (http://creativecommons.org/licenses/by/4.0/). 이 한국어 의역본은 CC BY 4.0에 따른 2차적 저작물이며 원저자 표시를 유지한다.*
