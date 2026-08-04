const P = (label, body, metric = "") => ({ label, body, metric });
const S = (pattern, section, title, status, source, refs, points, takeaway, extras = {}) => ({
  pattern, section, title, status, source, refs, points, takeaway, ...extras,
});

export const evidenceVerdictSlides = [
  S(5, "EVIDENCE", "Evidence는 이름이 아니라 성숙도 ladder로 읽는다", "방법", "Study §9 · date-frozen corpus",
    ["STC-S09", "STC-C001", "STC-C002", "STC-F005"],
    [P("E0", "idea·analogy·unverified claim"), P("E1", "paper proposition 또는 toy analysis"), P("E2", "controlled method experiment"), P("E3", "code·benchmark·independent reproduction"), P("E4", "production behavior·controls·rollout"), P("E5", "workload·failure·rollback·delete trace")],
    "Expected value가 커도 evidence maturity가 자동으로 높아지지 않는다."),

  S(22, "DIRECT EVIDENCE", "2025 STC는 ‘질의 전 reasoning’의 직접 이득을 보였다", "직접 근거", "Lin et al. 2025 · STC-T01",
    ["STC-T01", "STC-C002", "STC-C034"],
    [P("LATENCY", "동일 정확도의 test-time compute", "≈5×↓"), P("GSM", "sleep budget 증가 시 정확도", "+13% max"), P("AIME", "sleep budget 증가 시 정확도", "+18% max"), P("MULTI-QUERY", "질의당 평균 비용", "2.5×↓")],
    "효과는 future-query predictability와 context reuse가 높을 때 상각된다.", { bigNumbers: ["≈5×", "+13%", "+18%", "2.5×"] }),

  S(7, "DIRECT EVIDENCE", "2026 ‘Language Models Need Sleep’은 parametric sleep을 구성했다", "직접 근거", "Behrouz et al. 2026 · STC-T02",
    ["STC-T02", "STC-C039", "STC-C045"],
    [P("EXPAND", "smaller/fast memory에서 larger/slower capacity로"), P("SEED", "generalized distillation+RL imitation"), P("DREAM", "RL로 synthetic curriculum 생성·선별"), P("EVAL", "class-incremental·long-context·reasoning·knowledge incorporation"), P("SCOPE", "research-scale repeated cycles; production contract 미검증")],
    "Parametric sleep의 가장 직접적인 method evidence지만 fleet readiness 증거는 아니다."),

  S(24, "NEGATIVE EVIDENCE", "2026 continual fact study는 weight write의 reachability를 반박했다", "직접 반증", "STC-T03 · sequential fact writes",
    ["STC-T03", "STC-C039", "STC-C040", "STC-F015"],
    [P("BREADTH", "study data가 bare statement보다 application/composition 향상"), P("PLATEAU", "sequential write 뒤 old-fact survival이 낮은 수준에 정체"), P("MISROUTE", "잊힌 응답의 다수가 최신 fact로 routing"), P("RECOVER", "old fact를 context에 넣으면 77–80% 회복"), P("IMPLICATION", "stored와 behaviorally reachable은 다름")],
    "Weight에 흔적이 남았다는 것만으로 usable long-term memory가 아니다."),

  S(11, "DIRECT EVIDENCE", "Memory Caching은 RNN state reuse와 growing capacity를 연결한다", "직접 근거", "Behrouz et al. 2026 · STC-T11",
    ["STC-T11", "STC-C025", "STC-F004"],
    [P("CHECKPOINT", "segment 끝 recurrent memory state 저장"), P("AGGREGATE", "residual·gated·soup·sparse selective"), P("CAPACITY", "fixed RNN과 all-token KV 사이 연속체"), P("SERVING", "state reuse·routing·cache locality가 핵심"), P("GAP", "cache 증가가 access·invalidation cost를 다시 만듦")],
    "Subquadratic model도 오래 살려면 state cache policy가 필요하다."),

  S(16, "THEORY", "Rate–distortion은 모든 memory layer의 공통 한계를 준다", "직접 근거", "STC-T12 · layer-agnostic compaction",
    ["STC-T12", "STC-C028", "STC-C030", "STC-F009"],
    [P("RATE", "byte·token·state dimension·fact budget"), P("DISTORTION", "원 history 대비 downstream output 변화"), P("QUERY", "미래 query를 모르면 더 많은 rate 필요"), P("COMPOSE", "반복 compaction의 distortion이 누적"), P("REVERSIBLE", "cold raw copy가 hot compact state의 보험")],
    "Budget 아래에서는 모든 것을 기억할 수 없고 query uncertainty가 최저 비용을 정한다.", { equation: "R ≥ I*(Q)  or  D>0" }),

  S(18, "DIRECT EVIDENCE", "LoRA memory에도 finite capacity와 saturation knee가 있다", "직접 근거", "STC-T15 · LoRA as Knowledge Memory",
    ["STC-T15", "STC-C027", "STC-C029"],
    [P("RANK", "capacity ceiling은 늘지만 parameter efficiency는 비단조"), P("DATA", "task-aligned synthetic mixture가 raw text보다 효율적"), P("MODULE", "multi-LoRA가 total capacity를 늘림"), P("SYSTEM", "routing·merge·cache miss가 이득을 상쇄"), P("HYBRID", "long/multi-hop peak는 external evidence와 결합")],
    "Adapter는 유망한 compiled cache이지 무한한 truth store가 아니다.", { curveLabels: ["usable facts", "adapter bytes", "saturation"] }),

  S(24, "EVIDENCE HYGIENE", "DANN의 강한 문구는 현재 낮은 증거 등급으로 둔다", "경계", "DANN SSRN 2025 · abstract-only audit",
    ["STC-C031", "SRC-STC-0048", "STC-S16"],
    [P("CLAIM", "synthetic sleep·zero forgetting·vision/NLP"), P("ACCESS", "동결 시점 full text·code·tables 검증 불가"), P("RISK", "abstract 수치를 direct evidence처럼 인용할 수 없음"), P("USE", "hypothesis generator로만 유지"), P("UPGRADE", "full artifact·baseline·cycle curve 공개 필요")],
    "‘Zero forgetting’은 검증 가능한 repeated-cycle trace가 나올 때까지 결론이 아니다."),

  S(7, "INDUSTRY", "OpenAI Dreaming은 strict external lifecycle STC의 제품 신호다", "제품 근거", "Official ChatGPT memory releases · freeze 2026-08-05",
    ["STC-C038", "SRC-STC-0060", "SRC-STC-0063"],
    [P("INPUT", "여러 conversation의 accumulated memory"), P("BACKGROUND", "response path 밖 synthesis"), P("OUTPUT", "freshness·continuity·relevance를 반영한 memory state"), P("CONTROL", "user memory controls와 rollout"), P("BOUNDARY", "공개 evidence는 model-weight rewrite를 뜻하지 않음")],
    "Lifecycle adoption의 강한 신호와 parametric learning 주장은 분리해야 한다."),

  S(12, "INDUSTRY", "Letta는 sleeptime agent를 pre-query memory compiler로 구현한다", "코드 근거", "Letta sleeptime multi-agent lineage",
    ["STC-C038", "SRC-STC-0030", "SRC-STC-0064"],
    [P("FOREGROUND", "user-facing agent가 interaction 수행"), P("BUFFER", "shared memory block에 event 축적"), P("SLEEPER", "별도 agent가 memory를 정리·수정"), P("REUSE", "다음 query 전에 compiled token-space state 제공"), P("LIMIT", "학습 weight·independent production trace는 아님")],
    "External sleep operator는 이미 구현 가능하지만 품질·비용 검증은 별개다."),

  S(10, "INDUSTRY", "Mem0는 extraction·update·graph를 background memory로 묶는다", "제품/논문 근거", "Mem0 production-ready memory architecture",
    ["STC-C038", "SRC-STC-0034", "SRC-STC-0061"],
    [P("EXTRACT", "conversation에서 candidate memory 생성"), P("DECIDE", "add·update·delete·no-op"), P("STORE", "vector/graph와 metadata 갱신"), P("SERVE", "query-time retrieval로 context 축소"), P("EVIDENCE", "제품·benchmark 신호; weight consolidation 아님")],
    "현재 industry STC는 parameter보다 external memory management에 가깝다."),

  S(13, "INDUSTRY", "Zep은 memory를 temporal truth-maintenance 문제로 본다", "논문/제품 근거", "Zep/Graphiti temporal knowledge graph · STC-T09",
    ["STC-T09", "STC-C013", "STC-C038"],
    [P("EPISODE", "원 대화·event를 보존"), P("ENTITY", "stable node와 semantic relation"), P("TIME", "validity와 chronology"), P("CONFLICT", "새 assertion이 old를 supersede"), P("RETRIEVE", "current-state와 historical provenance를 함께 제공")],
    "장기기억은 단순 vector append가 아니라 시간에 따른 consistency 관리다."),

  S(11, "PORTFOLIO", "Google은 parametric sleep과 external self-evolution을 병렬 탐색한다", "산업/학계", "Google/DeepMind portfolio audit",
    ["STC-T02", "STC-T10", "SRC-STC-0054", "SRC-STC-0055"],
    [P("SLEEP", "Knowledge Seeding·Dreaming의 offline adaptation"), P("NESTED", "HOPE가 memory/optimizer를 update frequency로 해석"), P("REASONINGBANK", "성공·실패 trajectory에서 strategy 추출"), P("PRODUCTION", "공개 Gemini per-user weight sleep은 확인되지 않음"), P("OPEN", "isolation·rollback·delete·resource trace")],
    "Research convergence는 강하지만 product parametric adoption은 아직 미확인이다."),

  S(11, "PORTFOLIO", "Meta는 explicit memory와 self-editable agent program을 연구한다", "산업/학계", "PAHF · HyperAgents official research",
    ["SRC-STC-0044", "SRC-STC-0045", "SRC-STC-0057", "SRC-STC-0058"],
    [P("PAHF", "clarification·feedback을 per-user external memory에 반영"), P("HYPERAGENT", "meta-agent가 archive·memory·program을 수정"), P("SIGNAL", "experience-to-durable-artifact 연구"), P("BOUNDARY", "background biological sleep이나 consumer rollout 아님"), P("OPEN", "tenant·deletion·rollback·hardware trace")],
    "Self-improving agent와 sleep lifecycle은 겹치지만 동일하지 않다."),

  S(11, "PORTFOLIO", "Microsoft는 destination-neutral memory benchmark를 강화한다", "산업/학계", "LongMem · GenerativeAdapter · STATE-Bench",
    ["SRC-STC-0026", "SRC-STC-0027", "SRC-STC-0056", "SRC-STC-0067"],
    [P("EXTERNAL", "LongMem의 frozen backbone+side memory"), P("PARAMETRIC", "GenerativeAdapter의 data→adapter compilation"), P("BENCH", "STATE-Bench가 reliability·cost·UX를 평가"), P("BASELINE", "no-memory와 common interface 비교"), P("OPEN", "특정 mechanism의 production win은 아직 경쟁 중")],
    "업계는 ‘기억이 있는가’보다 실제 task value를 측정하는 방향으로 이동한다."),

  S(5, "TRAJECTORY", "학계는 네 계보가 lifecycle 관점에서 수렴한다", "통합 추론", "Study chronology · STC-F005",
    ["STC-S05", "STC-C016", "STC-C023", "STC-C028"],
    [P("REPLAY", "pseudorehearsal·generative replay·sleep SNN"), P("PLASTICITY", "regularization·gradient constraint·modularity"), P("NEURAL MEMORY", "TTT·Titans·HOPE·Memory Caching"), P("COMPACTION", "rate–distortion·summary·graph·adapter")],
    "주류가 되는 것은 하나의 sleep algorithm보다 lifecycle composition일 가능성이 높다."),

  S(15, "VERDICT", "여섯 핵심 주장 중 강한 결론은 두 개뿐이다", "조건부 결론", "Study §10 · six-verdict table",
    ["STC-S10", "STC-C038", "STC-C039", "STC-C040"],
    [P("NEW AXIS", "질의 전/후 reusable computation은 실재"), P("MAINSTREAM", "lifecycle은 likely, 명칭은 uncertain"), P("LONG CONTEXT", "특정 reuse workload에서만 우월"), P("CONTINUAL", "보조하지만 forgetting 해결 증거 부족"), P("PARAMETRIC", "external보다 보편 우월하지 않음"), P("DEVICE", "조건부로 큰 data-movement 기회")],
    "‘promising’은 workload·destination·evidence level을 붙여 말해야 한다."),

  S(9, "VERDICT", "가치는 reuse·stability·verifiability가 높은 workload에 집중된다", "판정", "Workload promisingness matrix · STC-F007",
    ["STC-S10", "STC-C034", "STC-C035", "STC-F007"],
    [P("CODEBASE", "반복 query·stable repository → 높음"), P("PERSONAL", "high reuse이나 privacy/volatility gate → 중간"), P("ENTERPRISE", "evidence·delete 요구가 높아 external 우세"), P("NEWS", "volatile truth → parametric 낮음"), P("ONE-SHOT", "상각 불가 → 낮음")],
    "Sleep compute은 future query가 반복되어 선불 비용을 회수할 때 가장 유리하다."),

  S(23, "MAINSTREAM", "주류화 여부는 다섯 시나리오로 추적한다", "전망", "Study §11 · scenario signals",
    ["STC-S11", "STC-C045", "STC-C046"],
    [P("EXTERNAL DEFAULT", "background summary·graph·index가 표준"), P("HYBRID", "external truth+adapter cache가 확산"), P("PARAMETRIC NICHE", "stable high-reuse domain만 weight/adapter"), P("BROAD PARAMETRIC", "반복 cycle·delete·isolation proof 필요"), P("STALL", "RAG/context frontier가 lifecycle gain을 흡수")],
    "달력 예측 대신 각 시나리오의 leading indicator와 falsifier를 본다."),

  S(14, "BASE CASE", "가장 가능성 높은 경로는 lifecycle mainstream·parameter niche다", "조건부 결론", "Date freeze 2026-08-05 · synthesis",
    ["STC-C038", "STC-C039", "STC-C045", "STC-C046"],
    [P("NOW", "external background synthesis와 memory policy 확산"), P("NEXT", "adapter·neural state를 high-reuse artifact로 추가"), P("LATER", "shared-weight promotion은 evidence가 축적된 subset"), P("NOT YET", "unbounded per-user self-training은 base case 아님")],
    "이 축은 주류가 될 수 있지만 대부분 ‘sleep’이라는 이름이나 full-weight training 형태는 아닐 것이다."),
];
