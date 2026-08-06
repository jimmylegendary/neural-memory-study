# 초록 (Abstract)

이 논문은 model이 한 data point에 관해 얼마나 많은 정보를 “아는지” 추정하는 방법을 제안하고 modern language model의 capacity를 측정한다. memorization을 두 부분으로 분리한다. **unintended memorization**은 특정 dataset 자체에 관한 정보이고, **generalization**은 실제 data-generation process에 관한 정보다. generalization을 제거하면 model의 total memorization을 계산할 수 있고, 이를 capacity estimate로 사용한다.

GPT family의 empirical capacity는 약 **3.6 bits per parameter**로 추정된다. dataset size를 늘리며 model을 학습하면 capacity가 차기 전까지 memorization이 증가한다. capacity가 채워진 뒤에는 “grokking”이 시작되어 model이 generalize하면서 unintended memorization이 감소한다. 500K–1.5B parameter 범위의 수백 transformer를 훈련하고 capacity, data size, membership inference를 연결하는 scaling law를 제시한다.

# 1. 서론 (Introduction)

수십억 parameter model이 수조 token을 학습할 때, parameter보다 훨씬 큰 dataset의 정보를 무엇까지 보존할 수 있는지가 문제다. 특정 string을 재생할 수 있는지, training membership을 공격자가 추론할 수 있는지, 일반 language pattern을 배웠는지는 서로 다른 현상이다.

논문은 memorization을 compression으로 본다. model weight와 reference distribution을 함께 사용해 특정 dataset을 얼마나 짧게 기술할 수 있는지가 “model이 아는 bit”다. 쉽게 예측 가능한 language는 model이 training example을 직접 저장하지 않아도 압축된다. 따라서 true distribution에 관한 generalization과 특정 sample의 unintended information을 분리해야 한다.

# 2. Intended와 Unintended Memorization

## 2.1 통계적 관점

dataset (X)와 trained model Θ 사이 mutual information은 model parameter가 training set에 관해 담는 정보의 양을 나타낸다. 그러나 underlying concept 또는 data-generating distribution에 관한 knowledge도 포함한다. 저자들은 oracle/reference model로 generalizable structure를 condition하여 sample-specific component를 분리한다.

## 2.2 Kolmogorov complexity를 이용한 측정

Kolmogorov complexity는 string을 생성하는 가장 짧은 program 길이다. 직접 계산할 수 없으므로 practical compressor와 model likelihood를 사용해 code length를 근사한다. 어떤 sequence를 baseline compressor보다 trained model이 더 짧게 encode한다면 그 차이를 model이 제공한 information으로 본다.

\[
\text{memorized bits}(x;\theta)
\approx L_{\mathrm{reference}}(x)-L_{\theta}(x).
\]

reference가 true distribution의 regularity를 이미 설명하도록 만들면 남은 차이가 unintended memorization estimate가 된다. 논문은 estimator bias와 finite compressor overhead에 대한 bound와 correction을 appendix에서 유도한다.

## 2.3 Compression estimator

token log-likelihood를 arithmetic/code length로 바꾸고 metadata와 compressor cost를 포함한다. random sequence처럼 generalization할 구조가 없는 dataset에서는 memorized bit가 model capacity를 직접 드러낸다. natural text에서는 oracle/reference model로 일반 language structure를 제거한다.

# 3. Memorization Capacity

## 3.1 Capacity 정의

model capacity는 가능한 training setup에서 model이 보존할 수 있는 최대 dataset-specific information으로 정의한다. parameter precision의 raw bit 수와 같지 않다. optimization과 architecture가 모든 weight bit를 독립 storage로 쓰지 못하므로 empirical usable capacity를 측정해야 한다.

## 3.2 Synthetic sequence 실험

uniform random data는 일반화 가능한 pattern이 거의 없으므로 training loss 감소가 memorization을 나타낸다. 다양한 parameter 수의 GPT-style transformer를 dataset size가 증가하도록 훈련한다. 작은 dataset에서는 거의 모든 bit를 저장하지만 dataset이 커지면 memorized bit가 plateau에 도달한다.

plateau를 parameter 수로 나누면 architecture와 precision 조건에서 약 3.6 bits/parameter가 나온다. 원문은 setting에 따라 약 3–4 bits 범위를 논의한다. 이는 모든 deployed LLM에 고정된 자연상수가 아니라 해당 model family와 training procedure의 empirical law다.

# 4. Unintended Memorization과 Generalization의 분리

natural text는 반복되는 grammar와 semantics가 있어 model이 unseen text에도 낮은 loss를 낸다. dataset size가 capacity보다 작을 때 model은 sample-specific detail과 general structure를 함께 담는다. data가 늘어 capacity pressure가 생기면 memorization이 계속 증가하지 않고 generalizable rule에 parameter를 쓰는 편이 유리해진다.

저자들은 capacity가 차는 전환점 부근에서 train/test dynamics가 바뀌고 grokking-like generalization이 나타난다고 보고한다. unintended memorization은 감소할 수 있지만 total predictive performance는 좋아진다. “더 많이 학습했는데 덜 외운다”는 현상은 model이 finite capacity를 general structure로 재배치한다는 해석이다.

# 5. Memorization과 Membership

## 5.1 Synthetic와 text data의 membership

membership inference는 한 example이 training set에 있었는지 model output으로 판단한다. sample-specific memorization이 많을수록 train example의 likelihood advantage가 커져 공격이 쉬워질 수 있다. 그러나 extractability, loss gap, membership AUC는 동일 metric이 아니므로 논문은 information measure와의 관계를 따로 분석한다.

## 5.2 Membership scaling law

model size와 dataset size를 바꾸어 membership signal이 어떻게 scale하는지 functional form을 fit한다. capacity 대비 dataset information load가 주요 variable이 된다. 작은 model에 너무 큰 dataset을 주면 개별 sample을 저장할 여유가 줄고, 큰 model이나 작은 dataset에서는 membership signal이 커진다.

저자들은 작은 model에서 얻은 law를 더 큰 model(GPT2-XL, 약 1.56B parameter)에 검증하며, 예측이 대체로 실제 F1의 1.5 point 이내에 든다고 보고한다. 여기서 나오는 배포 관련 결론은, parameter당 token 비율이 10² 이상인 현대 언어 모델이라면 이 법칙상 membership inference 점수가 0.5로 예측된다는 것이다 — 즉 이 정식화 안에서는 loss 기반 membership inference가 통계적으로 유의하게 성립하지 않는다.

extraction 쪽에서도 유사한 전환이 나타난다. training set이 아주 작을 때는 32-token prefix의 100%가 추출 가능하지만, 중복 제거된 dataset이 충분히 커지면 extraction rate가 0으로 가지는 않되 test extraction rate와 거의 같아진다. 즉 그 지점에서 성공한 추출은 모두 generalization으로 설명된다. 예측은 특정 architecture·optimizer·data distribution의 범위에서 평가되며, arbitrary production model의 privacy risk를 단일 수치로 보장하지 않는다.

# 6. 관련 연구 (Related Work)

extractable memorization은 prompt에서 training sequence를 verbatim 생성할 수 있는지를 본다. counterfactual memorization은 한 sample을 포함/제외했을 때 output이 얼마나 달라지는지 측정한다. perplexity 기반 방법은 흔한 string과 memorized string을 구분한다. 이 논문은 이들을 compression/information 관점에서 비교하고 exact bit quantity를 목표로 한다.

# 7. 결론 (Conclusion)

새 정의는 model이 dataset에 관해 아는 information을 bit로 측정하고, transformer LM의 empirical capacity를 추정한다. 연구는 extraction·F1·membership이 model/data scale에 따라 어떻게 변하는지 분석하고 membership scaling law를 더 큰 model에서 검증한다. 결과는 capacity가 차면 memorization과 generalization의 배분이 달라진다는 관점을 제공한다.

# 한계 (Limitations)

main experiment는 특정 synthetic/text dataset, GPT-style architecture, optimizer와 precision에 의존한다. 3.6 bits/parameter를 다른 modality, MoE, quantized model, post-training, continual update에 그대로 적용할 수 없다. Kolmogorov complexity는 근사하며 reference model과 compressor 선택이 estimate에 영향을 준다. membership scaling은 공격자의 정보와 query access가 달라지면 변한다.

이 capacity는 “유용한 사실 몇 개”와 직접 같지 않다. bit를 저장해도 질문으로 retrieve하거나 여러 사실을 compose하지 못할 수 있다. 뒤 원문 보존 부록은 estimator proof, synthetic/text scaling plot, membership fit, 500K–1.5B 실험과 appendix limitation을 v3 그대로 제공한다.
