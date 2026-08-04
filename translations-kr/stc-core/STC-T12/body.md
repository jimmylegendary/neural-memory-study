# 초록 (Abstract)

LLM과 그 위의 agent는 KV cache, 긴 prompt, recurrent state, session history를 유지하는 데 점점 더 많은 compute와 memory를 쓴다. KV-cache compression, prompt compression, bounded architecture, agent-memory consolidation은 서로 다른 연구 분야처럼 발전했지만, 이 논문은 모두 하나의 **rate–distortion** 문제라고 주장한다. 제한된 budget 아래 context에서 유래한 정보 중 무엇을 어떤 fidelity로 남겨 downstream utility를 보존할 것인가라는 문제다.

저자들은 하나의 compaction objective와 layer-agnostic lower bound, 일곱 축 taxonomy를 제시한다. 분야 전반에서 importance signal은 attention magnitude 또는 recency에 치우쳐 있고, query가 알려지기 전에 되돌릴 수 없이 버린 정보가 나중에 필요해지는 공통 실패를 보인다고 정리한다. single-turn compression은 많이 측정되지만 agent가 실제로 겪는 반복 compaction의 오차 누적은 거의 측정되지 않는다. 이를 위해 COMPACT-Bench, 작은 reference experiment, open-problem agenda를 제안한다.

# 1. 서론 (Introduction)

긴 transformer의 KV cache, 길어진 prompt, fixed recurrent state, multi-session agent store는 서로 다른 위치에 있지만 모두 history (H)를 더 작은 representation (Z)로 바꾼다. compaction operator를 (C), compacted representation을 사용하는 operator를 (U)라 두면 다음과 같다.

\[
Z=C(H;B), \qquad \hat Y=U(Z,Q).
\]

B는 byte, token, state dimension, stored fact 수 같은 budget이다. 핵심 평가는 compression ratio만이 아니라 원 history를 썼을 때의 task output Y와 compacted output hat-Y 사이 distortion이다. 같은 budget에서 query에 맞추어 필요한 정보를 다시 가져올 수 있는지, 되돌릴 수 있는지, 여러 fidelity를 유지하는지가 성능을 좌우한다.

# 2. 통합된 Rate–Distortion 정식화

논문은 expected task loss와 storage/compute cost를 결합한다.

\[
\min_C\; \mathbb{E}_{Q,H}\bigl[d(Y,U(C(H),Q))\bigr]
+\lambda R(C(H)),\qquad R(C(H))\le B.
\]

query-conditioned information requirement를 (I^{\star}(Q)=I(Y;H\mid Q))라고 둔다. query-agnostic compactor가 (B) bit 이하 representation을 만들면 data-processing inequality와 Fano inequality로 error lower bound를 얻는다.

\[
P_e \ge \frac{H(Y\mid Q)-B-1}{\log |\mathcal{Y}|},
\qquad B<I^{\star}(Q).
\]

이 bound는 KV vector, gist token, recurrent state, agent note처럼 representation 종류와 무관하다. 필요한 정보량보다 budget이 작으면 어떤 compactor도 error를 피할 수 없다. query를 모른 채 압축하면 여러 가능한 (Q)를 대비해야 하므로 더 큰 rate가 필요하다.

논문은 세 속성을 강조한다. **P-rev**는 버린 원 정보를 다시 가져오거나 compaction을 되돌릴 수 있는가, **P-q**는 query를 본 뒤 선택을 조정할 수 있는가, **P-fid**는 lossless leaf와 lossy summary 같은 여러 fidelity를 함께 유지하는가이다.

# 3. 일곱 축 Taxonomy

방법을 다음 축으로 분류한다: 압축 unit(bit, token, head, span, vector, fact), lifecycle 위치(prefill, decode, serving, task 내부, task 사이), budget 종류, query-conditioning 시점, reversibility, fidelity level, learned/training-free 여부. 서로 다른 논문이 보고하는 “4배 compression”은 unit과 metric이 다르면 직접 비교할 수 없으므로 byte/token과 downstream accuracy를 함께 정규화해야 한다.

# 4. KV-cache Compaction

## 4.1 Eviction

token 또는 head의 importance를 attention score, recency, heavy hitter로 추정해 KV entry를 버린다. decode memory와 bandwidth를 줄이지만 query가 진행되며 중요도가 바뀌면 이미 버린 token을 복구하기 어렵다. prefix와 recent window를 보존하는 heuristic은 많은 task에 유용하지만 “중간의 특정 사실”을 나중에 묻는 경우 실패할 수 있다.

## 4.2 Quantization

모든 token을 남기되 key/value bit-width를 낮춘다. eviction보다 P-rev에 가깝지만 quantization error와 outlier 처리가 필요하다. 같은 memory budget에서 token 수와 bit fidelity 사이를 교환한다.

## 4.3 Low-rank, hidden-dimension, sharing과 merging

KV matrix를 factorize하거나 head/layer 사이 representation을 공유하고 유사 token을 merge한다. 구조적 redundancy를 이용하지만 어떤 subspace가 future query에 필요한지 모르면 rare direction을 손실할 수 있다. 논문은 output-error bound가 있는 방법을 agent summary의 stopping rule로 옮길 가능성을 제시한다.

# 5. Prompt와 Context Compaction

extractive 방법은 중요 span을 선택해 원문을 보존하고, abstractive 방법은 더 짧은 text로 다시 쓴다. learned prompt compressor와 gist vector는 token 수를 크게 줄이지만 사람이 읽고 복구하기 어려울 수 있다. query-aware selection은 특정 질문에는 효율적이지만 동일 compact context를 다른 질문에 재사용하기 어렵다.

lossy summary를 반복해서 다시 summary하면 작은 오류와 누락이 다음 단계 입력이 되어 self-reinforce될 수 있다. 원 chunk를 leaf tier에 남기고 위에 hierarchical summary를 만드는 RAPTOR·GraphRAG 유형은 multi-fidelity와 reversibility를 일부 제공한다.

# 6. Architecture-level Bounded Memory

segment recurrence, linear attention, state-space model은 history를 fixed-size state에 압축해 (O(L)) 또는 subquadratic compute를 얻는다. budget이 sequence length와 함께 늘지 않으므로 장기 recall에서 information bottleneck이 생긴다. Memory Caching처럼 과거 state checkpoint를 추가하면 capacity를 늘릴 수 있지만, 결국 cache 수와 access cost라는 새 budget이 생긴다.

bounded architecture는 compaction operator가 model training 안에 포함된다는 장점이 있다. 반면 state가 어떤 정보를 잃었는지 inspect·rollback하기 어렵고, query가 나중에 도착하면 write 시점의 query-agnostic compression을 되돌릴 수 없다.

# 7. Agent Long-Term Memory Compaction

agent는 trajectory를 trim·summarize하고, session 사이 fact를 text/vector/graph에 저장한다. MemGPT는 tier 간 paging, Mem0는 extraction/update, Zep은 temporal knowledge graph, hierarchical RAG는 multi-level summary를 사용한다. raw episode를 남기는 design은 P-rev를 높이고, derived fact만 남기는 design은 compact하지만 extraction error에 취약하다.

agent memory의 고유 문제는 동일 memory가 여러 번 consolidate된다는 점이다. 처음에는 정확한 note도 이후 summary·merge·conflict resolution을 거치며 변한다. single-shot recall benchmark는 이 compounding curve를 측정하지 않는다.

# 8. Trainable Sparse Attention

training-free KV heuristic은 pre-trained weight가 임의의 missing token을 견딜 것이라 기대한다. trainable sparse attention은 어떤 token/head를 남길지 학습하여 compaction과 model을 공동 최적화한다. learned selector가 training query distribution에 과적합하면 새로운 query에서 동일 failure가 날 수 있고, hard selection의 reversibility 문제는 남는다.

# 9. Cross-layer Compaction

실제 system은 prompt를 먼저 줄이고, prefill 뒤 KV를 quantize/evict하며, agent turn 사이 summary를 만든다. 각 layer를 따로 최적화하면 같은 information을 중복 보존하거나 모두 버릴 수 있다. 논문은 C1부터 Ck까지 이어지는 operator composition의 distortion을 공통 budget으로 측정해야 한다고 주장한다. 현재는 이를 지배하는 일반 composition law가 확립되지 않았다.

# 10. 계층 사이를 잇는 다섯 knob

unit, timing, query adaptivity, reversibility, fidelity라는 공통 knob를 쓰면 한 분야의 mechanism을 다른 분야로 옮길 수 있다. 예를 들어 query-aware KV retrieval은 agent store에서 “미리 삭제하지 말고 cheap index로 후보를 좁힌 뒤 query에 맞춰 원문을 fetch”하는 design으로 대응한다. KV output-error bound는 summary를 언제 멈출지 정하는 threshold가 될 수 있다.

# 11. System과 Serving Substrate

compaction은 GPU HBM, host memory, SSD/object store, network 사이 data movement를 바꾼다. reversible design은 lower tier의 lossless copy와 index를 요구하고, query-time fetch가 latency tail을 만든다. multi-fidelity tier는 hot summary와 cold raw episode의 consistency/version을 관리해야 한다.

batching과 cache reuse도 중요하다. session별 state를 자주 merge·rewrite하면 immutable prefix sharing이 깨질 수 있다. asynchronous compaction은 critical path를 피하지만, 완료 전·후 version을 query가 일관되게 보도록 publication protocol이 필요하다.

# 12. Compaction 이론

lower bound의 (I^{\star}(Q))는 실제 task에서 직접 알기 어렵다. attention magnitude와 recency는 이 값을 추정하는 proxy일 뿐이다. 논문은 marginal task utility에 따라 budget을 할당하고, query distribution이 변할 때 다시 추정하는 문제를 제시한다. multi-layer budget은 각 unit의 residual information need가 균형을 이루도록 배분하는 것이 이상적이지만 실용 estimator와 guarantee가 없다.

# 13. 평가와 COMPACT-Bench

COMPACT-Bench는 서로 다른 layer의 방법을 동일 accuracy–budget curve에 놓고, latency와 byte를 함께 기록하며, single-shot과 repeated compaction을 분리하도록 설계된다. query가 compression 전에 알려지는 조건과 나중에 알려지는 조건, reversible fetch 허용 여부, source distribution shift를 교차한다.

평가에는 exact recall, multi-hop reasoning, temporal update, abstention, end-task utility와 함께 source fidelity·rollback을 포함한다. 단일 compression ratio가 아니라 Pareto frontier와 error accumulation slope를 보고해야 한다.

# 14. Reference Experiment

저자들은 commodity GPU의 small open model에서 KV eviction, quantization, prompt compression, summarization을 하나의 budget axis에 놓는 reference-scale 실험을 수행한다. 절대 수치는 방법론의 예시이며 대규모 production result로 제시되지 않는다. repeated irreversible summarization은 cycle이 늘며 error가 커지고, retrieval-backed reversible condition은 더 평평한 curve를 보이는지를 시험한다.

# 15. Open Problems and Research Agenda

우선순위에는 모든 layer의 query-conditioned reversible multi-fidelity compaction, cross-layer budget allocation, repeated-compaction benchmark, lossless episodic tier에서 lossy semantic tier로의 promotion rule, cheap asynchronous self-curation, RL-stable memory policy, guarantee가 있는 adaptive allocation, evolving memory의 audit·rollback·revocation이 포함된다.

논문은 sleep-time compute를 critical path 밖의 asynchronous curation 사례로 언급한다. 중요한 조건은 단지 밤에 계산한다는 것이 아니라, raw source를 복구 가능하게 남기고, 어떤 query distribution을 위해 무엇을 lossy하게 승격했는지 기록하며, repeated error를 측정하는 것이다.

# 한계 (Limitations)

이 연구는 survey와 formalism, benchmark proposal이며 대규모 empirical validation을 제공하지 않는다. lower bound의 가장 단순한 형태는 query-agnostic compactor를 가정하고 (I^{\star}(Q))를 주어진 값으로 취급한다. 실제 task의 information need를 추정하는 일 자체가 open problem이다. cross-layer operator algebra는 아직 증명된 law가 아니라 empirical composition map이다. reference experiment는 single commodity GPU의 작은 model이라 절대 성능은 scale evidence가 아니다. 최신 preprint가 많아 독립 재현을 기다리는 수치도 있다.

# 16. 결론 (Conclusion)

KV token eviction, quantization, prompt pruning, bounded recurrent state, agent trajectory summary는 모두 budget 아래 task utility를 보존하는 rate–distortion decision이다. 같은 budget에서 되돌릴 수 있고 query에 맞춰 선택하는 operator가 유리하며, 반복 lossy compaction의 누적 error를 공통 benchmark로 측정해야 한다. 뒤 원문 부록은 taxonomy, bound, survey table, COMPACT-Bench와 reference experiment 전체를 보존한다.
