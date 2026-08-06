# 초록 (Abstract)

테스트 시점 계산(test-time compute)의 확장은 대규모 언어 모델이 어려운 문제를 풀게 하는 핵심 수단으로 부상했지만, 높은 지연시간과 추론 비용을 수반한다. 이 논문은 질의가 제시되기 전에 모델이 문맥에 관해 오프라인으로 “생각”하게 하는 **sleep-time compute**를 제안한다. 앞으로 사용자가 물을 법한 질의를 예상하고 유용한 중간 산출물을 미리 계산하면, 실제 질의 시 필요한 계산을 크게 줄일 수 있다는 생각이다.

저자들은 두 추론 과제를 상태 보존형으로 바꾼 Stateful GSM-Symbolic과 Stateful AIME을 만든다. 같은 정확도를 얻는 데 필요한 test-time compute가 두 과제에서 약 **5배** 줄었다. sleep-time compute의 양을 늘리면 Stateful GSM-Symbolic 정확도는 최대 **13%**, Stateful AIME 정확도는 최대 **18%** 더 높아졌다. 하나의 문맥에 여러 관련 질의를 붙인 Multi-Query GSM-Symbolic에서는 오프라인 계산을 질의들 사이에 상각해 평균 질의 비용을 **2.5배** 낮췄다. 효과의 크기는 미래 질의가 얼마나 예측 가능한지와 잘 연관되었다. 마지막으로 실제적인 agentic software-engineering 과제에 적용하는 사례를 제시한다.

# 1. 서론 (Introduction)

추론 모델은 답을 내기 전에 더 많은 token과 검색 분기를 사용함으로써 성능을 높인다. 그러나 사용자가 질의를 낸 뒤에 이 계산을 모두 수행하면 응답 지연과 비용이 직접 증가한다. 반대로 사용자나 에이전트의 문맥은 질의보다 먼저 알려지는 경우가 많다. 저장소, 문서 집합, 장기간 대화 이력처럼 비교적 안정된 문맥은 먼저 도착하고, 그 문맥에 관한 구체적인 질문은 나중에 도착한다.

논문은 이 시간적 비대칭을 이용한다. 원시 문맥을 그대로 저장해 두었다가 매번 질의와 함께 처리하는 대신, 질의가 없는 시간에 모델이 문맥을 다시 읽고 계산·요약·추론 결과를 **학습된 문맥(learned context)** 으로 기록한다. 이후 질의가 오면 원시 문맥과 학습된 문맥을 함께 사용한다. 여기서 “learned”는 모델 가중치 갱신을 뜻하지 않는다. 이 논문의 기본 방법은 텍스트 형태의 외부 기억을 미리 만드는 추론 절차다.

핵심 질문은 세 가지다. 첫째, 오프라인 계산이 정확도–온라인 계산 Pareto frontier를 실제로 이동시키는가. 둘째, sleep-time budget을 늘리면 그 frontier가 계속 개선되는가. 셋째, 하나의 문맥을 여러 질의가 공유할 때 미리 계산한 비용을 충분히 상각할 수 있는가.

# 2. 관련 연구 (Related Work)

논문은 test-time scaling, self-consistency, search와 verifier를 이용한 추론, 문맥 압축, retrieval과 장기 에이전트 기억을 연결한다. 기존 test-time scaling은 질의 이후에 계산을 늘린다. caching은 동일 계산의 재사용을 겨냥하지만, 미래 질의에 도움이 될 새로운 추론 결과를 능동적으로 만들지는 않는다. 요약이나 retrieval은 문맥 길이를 줄이거나 관련 조각을 찾지만, 아직 묻지 않은 질의에 대비해 다단계 계산을 수행하는 것과는 구별된다.

sleep-time compute는 이들 방법을 배제하지 않는다. 원시 문맥의 검색 결과를 입력으로 받아 오프라인 추론을 할 수도 있고, 생성된 learned context를 retrieval index에 저장할 수도 있다. 논문이 분리해 측정하는 축은 “질의가 오기 전”에 사용한 계산량이다.

# 3. Sleep-time Compute

원시 문맥을 c, 나중에 도착하는 질의를 q, sleep-time 계산 예산을 B_s, test-time 계산 예산을 B_t라고 하자. sleep 단계는 아직 q를 보지 않은 채 c를 처리하여 learned context인 hat-c를 만든다.

\[
\hat c = S(c; B_s), \qquad \hat y = A(c,\hat c,q;B_t).
\]

목표는 (B_s)를 늘렸을 때 같은 정확도에 필요한 (B_t)가 줄거나, 같은 (B_t)에서 정확도가 높아지는지를 보는 것이다. 계산을 단순히 앞당긴 것만으로는 총비용이 절감되지 않을 수 있으므로, 논문은 한 문맥에 대한 질의 수 (N_q)도 명시적으로 고려한다. 질의당 평균 비용은 대략 다음과 같이 상각된다.

\[
C_{\mathrm{avg}} = C_{\mathrm{test}} + \frac{C_{\mathrm{sleep}}}{N_q}.
\]

오프라인 모델은 가능한 질문을 직접 나열하는 데 그치지 않고, 문맥 속 수량 관계를 계산하고, 파생 사실을 만들고, 후속 답변에 필요한 중간 결과를 기록한다. 예컨대 “800개 공 중 1/4이 테니스공, 그 절반이 남색, 그중 1/10에 표식”이라는 문맥이라면 각 단계의 수량과 최종 수량을 미리 계산한다. 나중에 어느 중간 단계가 질문되어도 짧은 온라인 응답으로 답할 수 있다.

# 4. 실험 설정 (Experimental Setup)

## 4.1 데이터셋

**Stateful GSM-Symbolic**은 수학 문장제의 배경 문맥과 구체 질의를 분리한다. 모델은 sleep 단계에서 배경만 받고, test 단계에서 질문을 받는다. **Stateful AIME**도 어려운 수학 문제에서 질의 전에 사용할 수 있는 상태 문맥을 구성한다. **Multi-Query GSM-Symbolic**은 같은 문맥에서 파생되는 여러 질문을 묶어, 한 번 만든 learned context를 몇 번 재사용할 수 있는지 측정한다.

SWE 사례에서는 software repository와 pull-request 설명처럼 큰 상태를 먼저 제공하고, 모델이 코드 구조와 잠재적으로 중요한 feature를 미리 분석한 뒤 실제 작업을 수행하게 한다. 이 평가는 수학 benchmark의 통제된 조건에서 벗어나 agent lifecycle에 적용 가능성을 살펴보기 위한 사례 연구다.

## 4.2 모델과 기준선

GSM-Symbolic에는 GPT-4o-mini와 GPT-4o를 사용한다. AIME에는 o1, o3-mini, Claude Sonnet 3.7 Extended Thinking, DeepSeek-R1처럼 test-time compute를 늘렸을 때 성능이 개선되는 모델을 포함한다. 주 기준선은 (c)와 (q)를 test time에 처음 함께 보는 표준 설정이다. 문맥만 보고 정답을 추측할 수 있는지를 확인하기 위해 context-only 기준선도 둔다.

sleep-time budget과 test-time budget을 각각 변화시키고 accuracy와 token/compute 비용을 함께 표시한다. 따라서 어떤 조건이 단지 총 계산량을 더 쓴 것인지, 온라인 지연을 줄인 것인지, 질의당 총비용까지 낮춘 것인지 구분할 수 있다.

# 5. 실험 결과 (Experiments and Results)

## 5.1 정확도–test-time compute Pareto frontier

learned context를 제공하면 같은 정확도에 도달하는 데 필요한 test-time compute가 Stateful GSM-Symbolic과 Stateful AIME에서 약 5분의 1로 줄었다. 중요한 비교는 sleep 계산을 더한 시스템이 단순 기준선보다 총 token을 많이 썼는지가 아니라, 사용자가 기다리는 구간의 계산량과 정확도 사이 frontier가 이동했는가이다. 결과는 여러 모델에서 그 이동을 보인다.

다만 저자들은 이 5배가 **낮은 test-time budget 구간의 이야기**임을 본문에서 명시한다. Stateful GSM-Symbolic의 높은 test-time budget에서는 오히려 test-time-only 기준선이 근소하게 앞선다. 저자들은 표준 조건의 prompt에는 해당 질문에 관련된 내용만 들어 있어 방해 정보가 적기 때문일 수 있다는 가설을 제시하지만 이를 시험하지는 않는다. 모델별로도 균일하지 않아 o1은 이득이 제한적이었고 그에 대한 설명은 제시되지 않는다.

같은 test-time token 예산에서 pass@k 병렬 스케일링과도 비교한다. 저자들은 모든 과제·모델에서 sleep-time compute가 pass@k를 일관되게 능가한다고 보고하며, pass@k가 test time에 정답 verifier에 oracle로 접근한다는 비현실적 이점을 갖는 baseline이므로 이를 넘어서는 것이 의미 있는 개선이라고 스스로 근거를 붙인다. 이 비교는 각 과제의 가장 낮은 sequential compute 조건에만 적용되었다.

## 5.2 Sleep-time compute의 스케일링

sleep 단계에 더 큰 budget을 주면 learned context가 더 많은 파생 관계와 중간 해답을 포함한다. Stateful GSM-Symbolic에서는 최대 13%, Stateful AIME에서는 최대 18%의 정확도 상승이 보고된다. 다만 모든 문맥에서 단조롭게 같은 이득이 나오는 것은 아니다. 미래 질문과 무관한 계산을 많이 만들면 비용만 늘 수 있다.

## 5.3 여러 질의에 걸친 상각

Multi-Query GSM-Symbolic에서 동일 learned context를 관련 질문들이 공유하면 sleep 비용이 분산된다. 본문이 보고하는 2.5배는 **문맥당 질의가 10개일 때 단일 질의 기준선 대비 최대치**이며, 생성 token만을 센 비용 모형에서 얻은 값이다. 저자들은 문맥당 질문 수가 적으면 총비용 관점에서 sleep-time compute가 불리해진다고 그림 설명에 명시한다. 한 문맥당 질의가 하나뿐이면 오프라인 비용을 회수하기 어렵고, 반복적으로 조회되는 사용자·문서·repository 상태일수록 경제성이 커진다는 뜻이다.

## 5.4 질의 예측 가능성

저자들은 어떤 문맥에서 sleep-time compute가 잘 작동하는지를 분석한다. 미래 질의의 분포가 문맥으로부터 어느 정도 예측될 때, 미리 계산한 항목이 실제 질문과 겹칠 확률이 높다. 반대로 가능한 질문 공간이 넓거나 질의가 문맥과 약하게 연결되면 learned context가 적중하지 않는다. 논문은 질의 예측 가능성과 sleep-time 효과 사이의 상관을 보고하며, 이 변수를 방법의 적용 조건으로 제시한다.

# 6. Agentic SWE 사례 (SWE-Features)

software-engineering 에이전트는 문제를 받은 뒤 repository를 탐색하고, 관련 파일을 읽고, 변경 계획을 세우며, 테스트 실패에 따라 다시 탐색한다. 논문은 repository 상태를 미리 분석하여 구현에 유용할 수 있는 feature와 메모를 만들고, 실제 문제 해결 시 이를 사용하는 절차를 시험한다. 오프라인 산출물은 “rethink memory”에 저장되고 이후 단계에서 검색된다.

보고된 이득은 낮은 test-time budget에서 test-time token 약 1.5배 감소 수준이며, budget이 높아지면 test-time compute만 쓰는 쪽이 더 나을 수 있다고 저자들이 명시한다. 높은 budget 조건에서는 표준 조건의 precision이 더 높고 recall은 비슷했는데, sleep-time compute를 쓴 agent가 더 많은 파일을 탐색한 뒤 더 많은 파일을 수정하는 경향 때문일 수 있다고 해석한다. 평가 지표는 수정한 파일 집합과 정답 파일 집합 사이의 F1이며, GitHub에서 수집한 PR이라 사용할 수 있는 테스트가 마땅치 않아 테스트는 채점에 쓰이지 않는다.

이 사례는 완전한 자동 사전 해결이 아니라, 반복되는 codebase 이해 비용을 질의 이전으로 옮기는 시도다. 수학 과제보다 문맥과 질문의 불확실성이 크기 때문에 결과는 proof-of-concept로 제시되며, 일반적인 SWE 성능 향상을 확정하는 실험으로 해석되지 않는다.

# 7. 논의와 한계 (Discussion and Limitations)

sleep-time compute가 가능하려면 문맥이 질의보다 먼저 존재해야 하고, 그 문맥이 sleep 이후에도 충분히 안정적이어야 한다. 문맥이 바뀌면 learned context를 무효화하거나 갱신해야 한다. 미래 질의가 예측 불가능하면 유용한 계산을 선별하기 어렵다. 오프라인 생성물이 잘못된 추론을 포함하면 이후 답변이 그 오류를 재사용할 수 있으므로 provenance와 재검증이 필요하다.

온라인 지연 감소와 총비용 감소는 서로 다른 주장이다. 한 문맥에 적은 수의 질의만 오면 sleep 비용을 상각하지 못한다. 반대로 반복 질의가 많고 문맥 reuse가 높으면 미리 계산한 비용이 여러 요청에 분산된다. 따라서 적절한 sleep budget은 예상 질의 수, 질의 분포, 문맥 변화율, learned context의 크기를 함께 고려해야 한다.

이 논문의 sleep은 가중치 update, continual learning, 생물학적 NREM/REM 모사를 포함하지 않는다. 모델 외부 텍스트 기억을 오프라인으로 합성하는 inference-time 기법이다. 저자들이 입증한 범위는 통제된 reasoning benchmark와 제한된 SWE 사례이며, 장기적인 개인화·망각·삭제·보안·모델 용량 문제는 후속 연구 대상으로 남는다.

# 부록의 구성

원문 부록에는 각 benchmark prompt, Stateful AIME 예시, Multi-Query GSM-Symbolic 구성, SWE-Features 세부 절차와 prompt, 예측 가능한 질문과 예측하기 어려운 질문의 예, rethink memory 구현, 연도별 AIME 결과가 포함된다. 뒤의 원문 보존 부록은 이 모든 appendix와 26개 figure 및 표를 고정 v1 PDF 그대로 제공한다.
