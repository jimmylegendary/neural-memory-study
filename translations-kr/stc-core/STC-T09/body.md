# 초록 (Abstract)

Zep은 AI agent를 위한 memory-layer service다. 핵심 구성요소 Graphiti는 대화 같은 비정형 data와 business data를 동적으로 합성하고, fact와 relation의 시간적 유효성을 유지하는 knowledge-graph engine이다. 논문은 MemGPT 팀이 사용한 Deep Memory Retrieval(DMR) benchmark에서 Zep이 94.8%, MemGPT가 93.4%를 기록했다고 보고한다.

더 복잡한 temporal reasoning을 포함한 LongMemEval에서는 baseline 대비 accuracy가 최대 18.5% 향상되고 response latency가 90% 감소했다고 보고한다. 특히 cross-session information synthesis와 long-term context maintenance를 enterprise use case로 강조한다.

# 1. 서론 (Introduction)

전통 RAG는 비교적 정적인 document corpus에서 query와 유사한 chunk를 찾는다. 장기 agent는 계속 변하는 대화, user preference, business entity, event와 그 관계를 다뤄야 한다. 과거에는 참이었지만 지금은 더 이상 참이 아닌 fact, 같은 이름을 가진 entity, 여러 session에 흩어진 relation을 단순 vector similarity만으로 처리하기 어렵다.

Zep은 raw episode를 보존하면서 semantic entity와 relation을 graph로 추출하고, valid time과 transaction time을 기록한다. retrieval은 raw evidence와 derived graph를 함께 사용한다. 논문은 이를 “non-lossy” episode store 위에 semantic layer를 만든다고 설명한다.

# 2. Knowledge Graph Construction

Graphiti의 graph를 G=(N, E, phi)로 두며, 세 계층의 subgraph를 사용한다.

1. **Episode subgraph (G_e)**: message, text, JSON 같은 raw input을 episode node로 저장한다. episode는 provenance를 보존하고, 추출된 semantic entity로 연결된다.
2. **Semantic entity subgraph (G_s)**: entity node와 entity 사이 fact/relation edge를 둔다. 새 episode의 mention을 기존 entity에 resolve하고, relation의 유효기간을 기록한다.
3. **Community subgraph (G_c)**: 밀접한 entity cluster를 community로 묶고 높은 수준의 summary를 제공한다.

episode를 삭제하거나 semantic extraction이 틀렸을 때 raw source로 돌아갈 수 있다는 점이 중요하다. derived edge는 episode provenance를 갖고, 동일 relation의 갱신과 contradiction을 시간 정보로 처리한다.

## 2.1 Entity extraction과 resolution

LLM은 현재 message에서 speaker와 주요 entity를 추출한다. relation이나 시간 자체를 entity로 만들지 않도록 prompt constraint를 둔다. 새 node가 기존 node와 같은 실제 entity인지 비교하고 duplicate이면 기존 UUID에 merge한다. 이름뿐 아니라 summary와 context를 함께 사용한다.

## 2.2 Relation extraction과 temporal validity

관계는 두 distinct entity 사이 edge로 표현한다. edge에는 relation type, 상세 fact, source episode, `valid_at`, `invalid_at` 같은 temporal field가 붙는다. “현재 ~이다”, “작년에 ~였다”, “이제 더 이상 ~가 아니다” 같은 표현에서 실제 event time을 계산한다.

새 fact가 기존 fact를 대체하면 old edge를 물리적으로 즉시 삭제하기보다 invalidation time을 기록한다. 이렇게 하면 “현재 상태” 질문과 “과거 어느 시점” 질문을 구분할 수 있다. 추론으로 date를 만들지 않고 source에 명시된 temporal cue만 쓰도록 extraction rule을 둔다.

## 2.3 Community construction

entity graph에서 관련 node를 cluster하고 community summary를 만든다. query가 특정 entity 이름을 정확히 포함하지 않아도 community-level context가 관련 영역을 찾는 데 도움을 준다. 새 episode가 들어오면 graph와 summary를 incrementally update한다.

# 3. Retrieval

Zep retrieval은 semantic similarity, keyword/BM25, graph traversal, recency와 temporal filter를 결합한다. query에서 entity를 찾고 관련 edge와 episode를 확장하며, 필요하면 community summary를 함께 반환한다. raw episode가 남아 있어 derived summary만으로 답하지 않고 근거 message를 prompt에 넣을 수 있다.

temporal question에는 event validity가 중요하다. “Alice가 지금 어디에 사는가”와 “2023년에 어디에 살았는가”는 같은 entity의 다른 edge를 선택해야 한다. cross-session synthesis는 서로 다른 episode의 relation을 graph path로 연결한다.

retrieval 결과의 최종 token budget은 여전히 제한된다. graph는 모든 정보를 model context에 넣는 대신 관련 evidence를 고르는 index다. 검색되지 않은 fact는 store에 있어도 answer에 쓰이지 않는다.

# 4. 평가 (Evaluation)

## 4.1 Deep Memory Retrieval

DMR은 긴 대화 중 특정 fact를 나중에 회상하는 과제다. 논문은 Zep 94.8%, MemGPT 93.4%를 보고한다. 두 수치 차이는 benchmark와 judge 설정 안에서 해석해야 하며, 서로 다른 전체 agent quality를 직접 의미하지 않는다.

## 4.2 LongMemEval

LongMemEval은 single-session fact lookup뿐 아니라 temporal reasoning, multi-session synthesis, knowledge update, abstention을 포함한다. Zep은 비교 baseline에 따라 accuracy를 최대 18.5% 높였고 latency를 90% 줄였다고 보고한다. graph에 미리 추출한 entity/relation을 재사용하여 query-time LLM processing을 줄이는 것이 latency 이득의 한 원인이다.

## 4.3 Enterprise 관점

논문은 ongoing conversation과 structured business data를 같은 graph에 넣는 use case를 강조한다. 동일 customer, ticket, product, organization을 entity resolution으로 연결하고, relation history를 유지한다. production system으로서 ingestion latency, retrieval latency, scale을 함께 고려한다.

# 5. 논의와 한계

graph construction에는 LLM extraction error가 들어갈 수 있다. 잘못 merge한 entity와 잘못된 temporal edge는 이후 여러 query에 영향을 준다. raw episode provenance가 correction을 가능하게 하지만 자동으로 오류를 없애지는 않는다. schema가 없는 open-world graph는 relation vocabulary와 community summary가 계속 변한다.

benchmark 결과는 Zep 저자들이 구현·평가한 production system의 보고다. system component별 독립 reproduction, cost accounting, privacy와 deletion, 매우 긴 lifetime에서 graph growth와 compaction은 추가 검증이 필요하다. “non-lossy”는 episode layer를 보존한다는 architecture claim이며, retrieval response가 모든 source detail을 항상 보존한다는 뜻은 아니다.

Zep은 model 외부 text/vector/graph memory 축의 대표다. ingestion 시 semantic synthesis를 수행하므로 query 밖의 background compute가 있지만, base-model weight를 sleep training하는 방법은 아니다. memory capacity는 graph/storage로 확장되는 대신 entity resolution, index maintenance, compaction과 provenance가 주요 system problem이 된다.

# 원문 구조 안내

뒤의 원문 보존 부록은 graph의 세 subgraph, ingestion prompt, entity de-duplication, temporal edge extraction, retrieval pipeline, DMR·LongMemEval table과 appendix를 v1 PDF 그대로 포함한다.
