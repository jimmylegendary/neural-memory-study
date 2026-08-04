# 초록 (Abstract)

지속 학습은 학습 이후에도 언어 모델이 새로운 사실을 계속 가중치에 기록하고 축적할 수 있다고 가정한다. 이 논문은 그 가정이 실제로 성립하는지를, 만들어낸 사실을 Qwen3 계열 모델에 기록한 직후부터 이후 20–100회의 추가 기록 뒤까지 추적하여 시험한다. 평가는 recall, paraphrase, application, composition, counterfactual의 다섯 질문 유형을 사용하며, 같은 사실을 원래 모델의 prompt에 넣은 조건을 reference로 삼는다.

훈련 데이터의 **폭(breadth)** 이 기록되는 지식의 종류를 결정했다. 사실 문장 하나만 반복한 bare-statement training은 암송에 가까운 결과를 만들었다. 다양한 재진술을 사용한 study training은 정답 결론을 직접 보여주지 않고도 recitation-to-use gap을 27.4점에서 5.4점으로 줄였다. 20회의 순차 기록 뒤 bare-statement 사실의 정확도는 1%였지만, 폭넓은 study data로 기록한 사실은 46%를 유지했다.

행동적으로 잊힌 사실이 가중치에서 지워진 것은 아니었다. 잊힌 사실은 최초 기록이 더한 log-probability 상승의 상당 부분을 유지했고, bare-statement 조건의 오답 중 70%는 가장 최근에 기록한 사실을 포함했다. 잊힌 study fact를 prompt에 다시 넣으면 해당 질문 정확도가 77–80%로 회복되었다. 일반 능력 손상은 원래 모델과의 KL divergence에 따라 정렬되었다. 저자들이 시험한 어느 intervention도 이전 사실을 안정적으로 도달 가능하게 만들지 못했으며, 사실의 합성이나 장기 생존이 필요한 조건에서 context가 weights보다 신뢰 가능한 채널이었다.

# 1. 서론 (Introduction)

새 사실은 context 또는 weights라는 두 경로로 모델에 들어갈 수 있다. context는 즉시 사용할 수 있지만 prompt의 수명과 context budget에 제한된다. weight write는 prompt 밖에 남지만, 그 지식이 다양한 질문에 사용 가능한지, 뒤의 write를 견디는지, 다른 능력을 손상시키지 않는지가 불명확하다.

논문은 “사실 하나를 잘 외웠는가”만 보지 않고 세 요구조건을 분리한다. 첫째, 각 write가 단순 문장 재현이 아니라 사용할 수 있는 지식을 만들어야 한다. 둘째, 반복 write 중 일반 능력이 보존되어야 한다. 셋째, 먼저 기록한 사실이 나중에도 질문으로부터 **reachable** 해야 한다. 실험은 이 세 조건을 동일한 invented-fact 환경에서 연결한다.

# 2. 모델이 사실을 사용할 수 있는지 측정하기

## 2.1 사실 생성과 다섯 질문 유형

평가 사실은 사전학습에서 보았을 가능성을 피하기 위해 발명한다. 예를 들어 “Zorvathine은 영하 10도 아래에서 녹는 금속” 같은 비정상 속성을 만든다. 한 사실에서 다음 다섯 유형의 held-out 질문을 생성한다.

1. **Recall**: 원 사실을 직접 말하게 한다.
2. **Paraphrase**: 다른 표현으로 같은 속성을 묻는다.
3. **Application**: 그 속성을 이용해 행동이나 결과를 도출한다.
4. **Composition**: 사실을 다른 지식과 결합한다.
5. **Counterfactual**: 통상 지식과 충돌하는 조건에서 새 사실을 우선해 추론한다.

질문 생성 과정 전체에서 held-out된 문항만 downstream 평가에 쓴다. 같은 judge와 같은 조건에서 비교 가능한 run만 결합한다. 원래 모델에 사실을 prompt로 제공한 성능은 weight write가 도달할 수 있는 in-context reference를 제공한다.

## 2.2 이중 채점과 entailment gap

어떤 답은 요청된 결론을 직접 말하지 않고 그 결론을 함의하는 사실만 반복한다. 논문은 이를 구분하기 위해 strict와 lenient 두 채점 정책을 사용한다. strict는 요청된 결론·값·선택을 명시해야 정답으로 본다. lenient는 목표를 엄밀히 함의하는 사실 진술도 인정한다.

\[
\text{entailment gap}=\text{lenient accuracy}-\text{strict accuracy}.
\]

큰 gap은 모델이 사실을 저장했더라도 그것을 질문에 맞게 사용하는 대신 암송에 머문다는 뜻이다. 두 정책은 독립적으로 판정되므로 작은 음수 차이는 judge noise 범위로 취급하고 큰 차이만 해석한다.

# 3. 훈련 데이터의 폭이 write의 지식 종류를 정한다

bare-statement training은 새 사실 문장을 직접 학습한다. study training은 같은 사실을 다양한 문맥·재진술·관계로 표현하지만 평가 결론을 답으로 주지는 않는다. 동일 optimization budget에서 비교하면 폭넓은 데이터가 application, composition, counterfactual을 크게 높이고 entailment gap을 줄인다.

96-step 조건에서 broader data는 application을 21점, composition을 18점, counterfactual use를 29점 높였지만, bare statement에 optimization step만 더하는 것은 이 능력을 개선하지 못했다. study training은 composition 60%, bare statement는 40%, fact-in-prompt ceiling은 83%였다. 가장 좋은 weight-based 방법인 context distillation은 70%였지만 in-context ceiling에는 미치지 못했다.

논문은 text objective 외에도 offline context distillation과 online context distillation, 두 KL 방향, LoRA와 full fine-tuning, RL 변형을 비교한다. 목표가 달라도 training-data breadth가 write의 사용 가능성을 좌우한다는 패턴은 유지된다. RL 변형은 때때로 reward를 올리면서도 사실 자체를 안정적으로 설치하지 못해 첫 요구조건에서 실패했다.

# 4. 지식의 종류가 순차 write 뒤 생존을 좌우한다

각 사실을 adapter에 기록하고 merge한 뒤 다음 사실을 기록하는 과정을 20회, 일부 실험에서는 100회까지 반복한다. 20 write 뒤 bare-statement 사실은 1%만 유지되었고 study fact는 46%를 유지했다. study fact의 장기 retention은 100 write 부근에서 25–28% 수준으로 plateau를 보였다. factor 조건 전체에서 초기 entailment gap은 생존과 음의 상관(ρ = −0.526)을 보였다.

이는 “처음 잘 외운 사실이면 나중에도 남는다”라는 단순 설명과 다르다. 처음 만들어진 지식이 질문에 따라 사용할 수 있는 형태인지가 후속 간섭에 대한 생존을 예측한다. 하지만 broader data도 reachability 문제를 해결하지는 못한다. retention을 높일 뿐, 계속되는 write에서 모든 이전 사실을 안정적으로 보존하지 못한다.

# 5. 망각이 파괴하는 것: 저장이 아니라 접근

행동 정확도가 0에 가까워진 사실의 checkpoint를 분석하면, 최초 write가 더한 drift-corrected log-probability lift의 57–67%가 남아 있다. 즉 해당 token sequence의 확률 흔적이 완전히 삭제된 것은 아니다. 질문을 넣었을 때 올바른 흔적에 도달하지 못한다.

bare-statement write에서 잊힌 사실에 관한 오답의 70%는 가장 최근에 기록된 사실을 끌어온다. 새 write가 오래된 정보를 물리적으로 덮어쓰기만 한 것이 아니라, 질문이 최신 write의 내용으로 잘못 routing되는 현상이다. 잊힌 study fact의 문장을 prompt에 다시 제공하면 77–80%로 회복된다. 같은 weights 안에 남아 있던 지식을 context cue가 다시 reachable하게 만든다.

두 사실을 각각 성공적으로 기록해도 둘을 함께 조합하는 질문에는 약하다. 이 결과도 저장량보다 on-demand retrieval과 binding이 병목이라는 해석과 일치한다. 논문은 이를 **question-keyed storage**라고 표현한다.

# 6. 반복 write가 치르는 비용

새 사실을 기록할수록 unrelated capability가 얼마나 손상되는지 측정한다. 손상의 크기는 원래 모델의 출력 분포에서 얼마나 멀어졌는지를 나타내는 KL divergence와 함께 움직인다. frozen reference를 이용한 distillation이나 local drift penalty는 일반 능력 보존에는 도움을 줄 수 있다.

그러나 capability 보존과 old-fact reachability는 같은 문제가 아니다. 원래 모델과 가까운 분포를 유지해도 이전 사실을 묻는 질문이 정확한 사실로 routing된다는 보장은 없다. 저자들이 각 write의 gradient와 local curvature 등 정확한 지역 측정을 이용한 intervention을 시험했지만, incoming write가 일으키는 간섭을 안정적으로 제어하지 못했다.

# 7. 생성과 간섭에 대한 인과 시험

논문은 data breadth, objective, LoRA rank, model scale, write order를 바꾸어 초기 지식 형성과 후속 간섭을 분리한다. bare-statement gap은 4B와 8B 모델, LoRA와 full fine-tuning에서 재현되었다. study data의 어떤 개별 성분이 효과를 만드는지는 완전히 분해되지 않았고, study training이 recall과 paraphrase에서는 때로 낮은 이유도 남은 질문이다.

후속 write를 bare statement로 할지 study data로 할지 바꾸어도 먼저 기록된 사실에 대한 간섭이 사라지지 않는다. 즉 이전 write를 더 잘 만드는 것과 다음 write가 가하는 충돌을 막는 것은 별도의 intervention을 요구한다. 본문과 부록은 validity screen, judge audit, step 수, rank, model size, cue phrasing에 대한 절제 실험을 제공한다.

# 8. 관련 연구 (Related Work)

논문은 knowledge editing, continual fine-tuning, catastrophic forgetting, model merging, distillation, parameter-efficient adaptation, in-context learning을 비교한다. 기존 연구가 한 번의 edit 성공률이나 general benchmark 보존을 주로 측정했다면, 이 연구는 동일 사실을 처음 write한 순간부터 다수의 후속 write 뒤까지 추적하고 context channel을 고정 ceiling으로 둔다.

# 9. 논의 (Discussion)

실험은 발명된 소수 사실, Qwen3 계열, 특정 adapter merge schedule과 자동 judge에 한정된다. 모든 형태의 장기 pretraining이나 대규모 continual corpus로 곧바로 일반화할 수 없다. log-probability 흔적이 남았다는 결과는 완전한 의미 표현이 그대로 저장되었다는 증명이 아니며, 관측한 checkpoint와 질문 집합 안에서 access failure가 유력하다는 증거다.

결론은 weight update가 언제나 무용하다는 것이 아니다. broader training data는 훨씬 사용 가능한 지식을 만들고, frozen reference나 KL 제약은 capability drift를 줄인다. 그러나 여러 사실의 composition과 뒤이은 write 생존을 동시에 요구할 때, 이 논문이 시험한 weight-write 방법 중 어느 것도 context reference의 신뢰성에 도달하지 못했다. 따라서 저자들은 신뢰성이 필요한 사실 채널로 context를 유지하고, weight write는 그 운영 경계를 명확히 측정해야 한다고 결론짓는다.

# 부록 안내

원문 부록에는 dataset certification, judge의 strict/lenient audit, training step·rank·model scale factorization, 100-write 장기 run, cued phrasing, reconstruction과 log-probability 분석, local-control intervention의 상세 결과가 있다. 뒤의 원문 보존 부록은 모든 figure·table·equation과 수치표를 v2 PDF 그대로 싣는다.
