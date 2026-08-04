# 초록 (Abstract)

LLM은 제한된 context window 때문에 장기 대화와 큰 문서 분석에 제약을 받는다. 이 논문은 전통 운영체제의 hierarchical memory가 physical memory와 disk 사이 paging으로 큰 virtual memory의 환상을 제공하는 데서 착안해 **virtual context management**를 제안한다. 이를 구현한 MemGPT는 여러 storage tier를 관리하여 제한된 context 안에서 더 긴 상태를 사용하게 한다.

저자들은 장기 대화와 document analysis에서 MemGPT를 평가한다. 핵심은 더 긴 transformer를 새로 훈련하는 것이 아니라, LLM이 function call을 통해 자기 context의 일부를 편집하고 외부 memory를 읽고 쓰며, 필요할 때 정보를 paging하는 system design이다.

# 1. 서론 (Introduction)

self-attention context를 직접 늘리면 compute와 memory cost가 커지고, 긴 context를 주더라도 model이 중간 정보를 안정적으로 사용하는 것은 별도 문제다. operating system은 작은 RAM과 큰 disk를 계층화하고, program에는 연속된 큰 address space처럼 보이게 한다. MemGPT는 LLM context를 제한된 main memory로, 외부 database를 secondary storage로 본다.

LLM은 고정된 prompt를 수동으로 잘라 받는 대신, 언제 무엇을 main context에서 내보내고 어떤 memory를 다시 불러올지 판단한다. system은 memory pressure와 event를 알려주고, model은 tool/function을 호출해 상태를 변경한다.

# 2. MemGPT (MemoryGPT)

## 2.1 Memory hierarchy

**Main context**는 현재 LLM call에 실제로 들어가는 token이다. system instruction, working context, 최근 대화, 핵심 memory가 포함된다. 이 공간은 작고 비싸다. 외부 tier는 크게 **recall storage**와 **archival storage**로 나뉜다.

recall storage는 과거 message와 event를 보존해 대화 기록을 시간·keyword 등으로 검색한다. archival storage는 model이 장기간 보존해야 한다고 판단한 note, fact, document chunk를 저장한다. core memory 또는 in-context memory에는 persona와 user에 관한 작은 고가치 state를 둔다.

## 2.2 Self-editing memory

모델은 function call로 core memory를 append, replace하고 archival memory를 insert/search한다. context에 모든 history를 넣지 않고, 현재 task에 필요한 항목을 검색해 main context로 가져온다. 이 방식은 memory controller를 별도 heuristic으로 완전히 고정하지 않고 LLM의 reasoning에 일부 맡긴다.

## 2.3 Function executor와 event loop

LLM이 tool call을 내면 function executor가 외부 state를 바꾸고 결과를 새 event로 돌려준다. model은 user message뿐 아니라 system event, memory warning, function result를 처리한다. context가 한계에 접근하면 system이 warning을 주어 model이 중요한 정보를 external tier로 옮기게 한다.

한 번의 user turn 안에서도 model이 여러 internal step과 memory operation을 수행할 수 있다. “heartbeat” 또는 continue event는 function call 뒤 computation을 이어가게 한다. 따라서 user-facing reply와 internal memory-management action을 분리한다.

## 2.4 Virtual context management

virtual context는 무한하지 않다. external storage가 커져도 retrieval과 selection을 통해 일부만 main context에 들어온다. illusion의 의미는 application이 context limit에 도달할 때 대화를 종료하는 대신, 오래된 state를 external tier로 page out하고 나중에 page in할 수 있다는 것이다.

# 3. 평가 (Evaluation)

## 3.1 장기 대화

perpetual chat setting에서 agent는 많은 turn에 걸쳐 user fact와 사건을 기억해야 한다. Deep Memory Retrieval(DMR) benchmark는 과거 대화에 심어 둔 정보를 나중에 질의한다. MemGPT는 fixed-context baseline보다 높은 retrieval 성능을 보이고, conversation opener 평가에서는 과거 관계를 이용해 자연스럽게 대화를 이어가는지를 본다.

평가가 보여주는 것은 model parameter에 user memory가 통합되었다는 것이 아니다. 외부 store에 기록된 항목을 적절히 검색하여 prompt 안에서 사용하는 능력이다. memory write가 빠졌거나 retrieval query가 틀리면 정보가 store에 있어도 답하지 못할 수 있다.

## 3.2 문서 분석

큰 document를 archival storage에 넣고 question에 따라 관련 passage를 검색한다. fixed-context baseline은 context limit만큼의 앞부분이나 선택된 chunk만 볼 수 있지만, MemGPT는 iterative search와 paging으로 다른 부분을 탐색할 수 있다. 여러 passage를 연결해야 하는 질문에서는 memory operation의 순서가 중요하다.

논문은 retrieval-only pipeline과 agentic management의 차이를 살핀다. MemGPT는 검색 결과를 읽고 추가 query를 만들거나 core memory를 수정할 수 있다. 반면 이 flexibility는 더 많은 LLM call과 latency를 유발할 수 있다.

# 4. 관련 연구 (Related Work)

long-context transformer, recurrent memory, retrieval-augmented generation, conversational agent memory, tool-using agent와 비교한다. long-context architecture는 model 내부 attention 범위를 늘린다. RAG는 query와 document corpus 사이 retrieval을 수행한다. MemGPT는 여러 tier와 self-editing event loop를 결합해 application lifecycle 전체의 context를 관리한다.

운영체제 비유는 설계 원리를 설명하지만 완전한 등가는 아니다. OS page fault와 replacement는 명시적 address와 deterministic semantics를 갖는다. LLM memory는 natural-language content를 확률적으로 검색·요약하고, write의 의미가 model output에 의존한다.

# 5. 논의와 한계

external memory는 model weight capacity를 소모하지 않고 수정·삭제·provenance 추적이 쉽다. 반면 retrieval latency, storage growth, stale fact, contradiction, privacy, access control이 새 system 문제로 생긴다. core memory가 잘못 편집되면 여러 turn에 걸쳐 오류가 지속될 수 있다.

model이 스스로 어떤 내용을 저장할지 결정하므로 memory admission quality가 base model과 prompt에 민감하다. context pressure warning이 너무 늦으면 중요한 detail이 손실되고, 너무 자주 memory operation을 하면 cost가 커진다. benchmark는 제한된 conversation/document task이며 lifetime-scale user state의 capacity와 deletion compliance를 검증하지 않는다.

MemGPT는 이후 Letta system과 sleep-time compute 연구의 기반이 되는 agent lifecycle을 제공하지만, 이 논문 자체의 memory transition은 주로 online write/search/paging이다. idle period에 model weight를 훈련하거나 장기간 memory를 자동 consolidation하는 training 절차를 제안하지 않는다.

# 6. 부록 (Appendix)

원문 부록은 system prompt, memory function schema, DMR와 conversation-opener judge prompt, document-analysis 설정과 예시를 제공한다. 뒤 원문 보존 부록에는 architecture diagram, context layout, task examples, 결과 table을 포함한 v2 전체가 있다.
