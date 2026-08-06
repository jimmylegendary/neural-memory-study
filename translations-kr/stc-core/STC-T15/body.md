# 초록 (Abstract)

pre-trained LLM에 새로운 지식을 계속 반영할 필요가 있지만, full retraining과 supervised fine-tuning은 비용과 catastrophic forgetting 문제가 있다. ICL과 RAG는 parameter를 바꾸지 않지만 context budget, 반복 처리 비용, retrieval fragmentation에 제약을 받는다. 이 논문은 LoRA를 교체·조합 가능한 **parametric knowledge memory**로 보고 storage capacity, internalization, multi-module scaling, long-context reasoning의 design space를 체계적으로 측정한다.

하나의 architecture를 제안하기보다 LoRA memory가 작동하는 범위와 실패 조건을 지도화한다. 결론은 LoRA가 ICL/RAG를 대체하는 단일 해법이 아니라, 반복 조회되는 안정된 지식을 parameter에 압축해 inference cost를 줄일 수 있는 보완 축이라는 것이다.

# 1. 서론 (Introduction)

LoRA는 base weight (W)를 고정하고 low-rank delta를 학습한다.

\[
W' = W + \Delta W,
\qquad \Delta W = BA,
\qquad \operatorname{rank}(\Delta W)\le r.
\]

task adaptation에 쓰이던 adapter를 factual/document knowledge store로 사용하면 module을 train, swap, route, merge할 수 있다. 그러나 rank를 늘리면 저장량이 비례해 늘어나는지, 한 module의 finite limit은 무엇인지, 여러 module이 문맥을 나누어 저장할 때 cross-chunk reasoning이 가능한지는 불명확하다.

논문은 PhoneBook 같은 controlled synthetic dataset과 PaperQA·NarrativeQA·QuALITY·100K+ long-context benchmark를 사용해 15개 질문(Q1–Q15)을 순서대로 답한다.

# 2. 관련 연구 (Related Work)

ICL은 원문을 prompt에 넣어 높은 fidelity를 유지하지만 매 query마다 long prefill이 필요하다. RAG는 relevant chunk만 가져와 cost를 줄이지만 embedding과 top-k가 전체 document의 관계를 분절할 수 있다. model editing과 continual-learning adapter는 update를 parameter에 넣지만 storage capacity와 composition을 체계적으로 측정하지 않은 경우가 많다.

SEAL과 deep-context distillation은 synthetic data나 outer-loop를 이용해 knowledge를 adapter에 넣는다. 이 논문은 그런 pipeline의 한 component로 LoRA를 쓰는 데서 멈추지 않고 LoRA 자체의 intrinsic capacity와 serving trade-off를 측정한다.

# 3. 평가 설정

**PhoneBook**은 key–value fact 수를 임의로 늘려 exact retrieval capacity를 측정한다. **CounterFact**는 사실 편집형 지식을 efficacy 점수로 재고, Q1–Q3의 rank·용량·효율 sweep은 PhoneBook과 CounterFact 두 축에서 함께 수행된다. **PaperQA**는 scientific document knowledge를 factual/relational 질문으로 평가하며, pretraining contamination을 줄이려고 최근 학회(NeurIPS 2024·ICLR 2025·ICML 2025) 논문 15편에서 총 450개 QA pair를 계층적으로 구성했다. **NarrativeQA**와 **QuALITY**는 긴 document의 multi-hop와 global coherence를 요구한다. ∞Bench와 LongBench v2의 100K+ token subset도 추가한다.

efficacy score, exact/QA accuracy, long-document metric과 parameter 수, training token, rank, module 수, routing/merge cost를 함께 기록한다. single LoRA와 multi-LoRA, closed-book, ICL, RAG, hybrid를 비교한다.

# 4. Single LoRA를 Memory Unit으로 보기

## Q1. Capacity는 rank와 함께 scale하는가?

rank를 늘리면 trainable parameter와 capacity ceiling이 올라간다. 그러나 gain은 parameter cost에 비례하지 않는다. 낮거나 중간 rank가 높은 rank보다 parameter당 더 많은 usable knowledge를 저장하는 구간이 있다. 따라서 최고 rank를 자동 선택하는 것은 비효율적일 수 있다.

## Q2. Finite capacity limit이 있는가?

한 LoRA가 학습할 fact를 계속 늘리면 performance가 일정 지점 이후 saturation하거나 감소한다. optimization step을 더 주는 것만으로 limit이 사라지지 않는다. data volume, rank, base model, target module, training format이 effective capacity를 함께 정한다.

## Q3. Highest rank가 항상 효율적인가?

큰 rank는 raw capacity를 높이지만 training·storage·load cost도 늘고 parameter efficiency가 non-monotonic하다. 논문은 target workload에 맞는 “right-sized” rank를 선택하라고 결론짓는다.

# 5. 하나의 LoRA 최적화 (Optimizing a Single LoRA)

## Q4–Q5. Synthetic data와 format mixture

raw document text만 continued fine-tuning하는 것은 factual retrieval에 information density가 낮다. QA pair, summary, rewrite처럼 task-aligned synthetic format은 같은 training budget에서 지식을 더 효과적으로 internalize하며, 효과 순서는 QA > Summary > Rewrite > Original이었다. 다만 저자들은 QA의 우위가 평가 과제 자체가 QA 형식이라는 구조적 정렬에 일부 기인할 수 있다고 스스로 단서를 단다.

format을 섞는 효과는 **model에 따라 갈렸다**. Llama-3.1-8B에서는 Original+Summary+Rewrite+QA의 가장 포괄적인 혼합이 최고점을 냈지만, Qwen3-8B에서는 Summary+QA가 최고였고 전체 혼합은 그에 근접한 수준에 머물렀다. Qwen에서는 Original+QA가 QA 단독보다 오히려 약간 낮았다. 즉 “같은 내용을 다양한 관점으로 보여주면 도움이 된다”는 방향성은 두 model에서 공통이지만, 최적 혼합이 하나로 고정되지는 않는다.

이는 synthetic data가 새로운 사실을 창작해야 한다는 뜻이 아니다. source content를 다양한 supervision view로 바꿔 model이 질문–답, 관계, global summary에 접근하게 한다. generator가 source를 잘못 해석하면 그 error도 adapter에 들어간다.

## Q6–Q7. Base model scale과 generator quality

base model의 capability와 scale은 같은 LoRA rank가 지식을 internalize하는 정도에 영향을 준다. synthetic-data generator가 source의 depth와 breadth를 얼마나 정확히 표현하는지도 upper bound가 된다. 더 큰 generator가 항상 모든 downstream setting에서 비례 이득을 주는 것은 아니지만, 누락·오답이 있는 synthetic curriculum은 capacity를 낭비한다.

추가 ablation은 layer와 module placement를 비교한다. early layer와 FFN에 LoRA를 적용한 조건이 attention-only 또는 late-layer placement보다 memorization이 강하고 saturation이 늦는 경향을 보였다. DoRA·PiSSA의 이득은 model/setting 의존적이며 standard LoRA보다 일관된 우위를 보이지 않았다.

# 6. Multi-LoRA System으로 확장

## Q8. 여러 작은 LoRA

한 module의 finite capacity와 낮은 rank의 parameter efficiency를 이용해 knowledge를 여러 adapter로 partition한다. perfect routing을 가정한 proof-of-concept에서는 여러 작은 module이 한 큰 module보다 total capacity를 효율적으로 늘릴 수 있다.

## Q9. Routing error

실제 query가 어느 module의 knowledge를 필요로 하는지 router가 골라야 한다. top-k routing이 틀리면 필요한 fact가 parameter에 있어도 접근하지 못한다. 결과는 ideal routing의 이득이 “줄어드는” 정도가 아니었다 — embedding 기반 실제 router는 두 base model 모두에서 oracle routing보다 낮은 것은 물론, **지식 전체를 담은 단일 LoRA baseline보다도 낮았다**. 저자들은 잘못 선택된 고도로 특화된 module이, 전체 지식을 담은 비특화 단일 module보다 오히려 더 해로울 수 있다고 정리한다. token 단위 router(Arrow·SpectR·LAG)도 dataset과 retrieval depth 전반에서 embedding baseline을 일관되게 앞서지 못했다. 따라서 retrieval metric과 answer metric을 분리해 보아야 한다.

## Q10–Q11. Merging과 interference

여러 LoRA를 동시에 merge하면 더 많은 knowledge를 사용할 수 있지만 parameter delta 사이 interference와 signal dilution이 생긴다. merge 연산자로는 Linear, CAT, CAT-1/√N, TIES, DARE-Linear, DARE-TIES를 비교한다. TIES가 가장 높아 단일 LoRA에 필적하고 Linear가 의외로 강한 baseline이며, DARE는 확률적 drop이 memorization 설정에서 중요한 정보를 버려 뒤처지고, 소박한 CAT은 크게 무너진다(CAT-1/√N이 그 격차를 대부분 회복하므로 CAT의 주 실패 원인은 rank 확장이 아니라 scale 불일치다).

merge 개수 자체의 효과는 trade-off라기보다 **단조 감소**다. query가 실제로 유래한 ground-truth module만 골라 $N_m = 1 \to 5$로 늘려도 성능은 $N_m = 1$에서 최고이고 이후 계속 떨어진다. 필요한 지식이 모두 포함되어 있어도 module을 더 붙이는 행위 자체가 저장된 정보를 희석·간섭시킨다는 뜻이다. 따라서 논문은 conflict-aware merge를 small high-confidence set에만 적용하라고 권한다.

# 7. Long-document와 Hybrid Memory

## Q12. Multi-hop QA

document를 chunk별 LoRA에 나누면 각 module 내부 지식은 저장되지만 질문이 여러 chunk를 연결할 때 contextual continuity가 끊긴다. NarrativeQA 같은 multi-hop task에서 routing과 merge noise가 modular capacity의 이득을 상쇄할 수 있다.

## Q13. External context의 보완

single/multi-LoRA에 ICL 또는 RAG context를 추가하면 parameter에 압축되며 빠진 detail을 원문 evidence가 보완한다. 특히 fragmented multi-LoRA에서 hybrid가 global coherence를 회복한다. 100K+ benchmark에서도 LoRA가 일률적으로 붕괴하지는 않았지만 setting 의존적이며 hybrid가 성능을 회복하거나 높이는 경우가 많았다.

## Q14. 여러 LoRA merge와 continuity

top-3 module merge는 대부분의 설정에서 top-1 selection보다 multi-hop synthesis를 개선했다. 외부 context와 결합했을 때 이점이 더 뚜렷했다. 하지만 merge 수를 무제한 늘리면 interference가 커지므로 routing confidence와 query type에 맞춰야 한다.

## Q15. 시간 비용

NarrativeQA 한 document의 30개 연속 질문에서 LoRA는 매번 수천 context token을 처리하지 않아 pure inference time이 짧았다. single LoRA는 한 번 load/attach하는 cost가 있다. multi-LoRA는 dynamic I/O와 merge overhead가 크므로 관련 module을 GPU에 preload했다. 이 조건에서 multi-LoRA total time은 ICL보다 낮았고, 남은 overhead는 adapter caching·prefetch·kernel optimization의 system 문제로 남았다.

# 8. 결론 (Conclusion)

**Single LoRA는 resource-constrained memory unit**이다. rank를 늘리면 finite capacity ceiling은 올라가지만 parameter efficiency는 비단조적이다. **Knowledge density**가 중요하며 raw text보다 task-aligned synthetic QA·summary·rewrite mixture가 효과적이다.

**Modularity는 total capacity를 늘리지만 system failure를 만든다.** imperfect routing이 이득을 없앨 수 있고, merge 수 증가는 recall과 interference를 교환한다. robust router와 interference-aware merge, adapter serving이 필요하다.

**Hybrid memory가 가장 견고하다.** LoRA는 안정된 knowledge base의 반복 접근 비용을 줄일 수 있지만, long/multi-hop에서 peak performance는 ICL/RAG의 external evidence와 결합할 때 나왔다. LoRA를 context memory의 대체재보다 보완적인 parametric axis로 보는 것이 논문의 최종 입장이다.

# 한계와 향후 연구

실험은 LoRA memory의 성질을 통제 조건에서 분리한 것이며 완전한 continual-update deployment가 아니다. daily news처럼 topic이 연결되고 fact가 수정되는 stream, deletion과 rollback, adapter versioning은 검증하지 않았다. large-scale multi-LoRA에는 routing, conflict-aware merge, caching, prefetch, kernel-level composition이 필요하다.

Samsung SDS 저자들이 포함된 이 연구는 LoRA knowledge memory의 운영 경계를 제시하지만, 모든 base model·data·rank에 공통인 보편 capacity law를 주장하지 않는다. 뒤 원문 보존 부록에는 Q1–Q15, 22개 figure, 7개 table, PhoneBook/PaperQA/100K+ 설정과 모든 appendix가 v5 그대로 포함된다.
