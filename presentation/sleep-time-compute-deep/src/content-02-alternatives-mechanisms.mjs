const P = (label, body, metric = "") => ({ label, body, metric });
const S = (pattern, section, title, status, source, refs, points, takeaway, extras = {}) => ({
  pattern, section, title, status, source, refs, points, takeaway, ...extras,
});

export const alternativeMechanismSlides = [
  S(9, "ALTERNATIVES", "문제마다 sleep의 strongest alternative가 다르다", "비교", "Study §6–§8 · STC-F006",
    ["STC-S06", "STC-S08", "STC-F006", "STC-C040"],
    [P("ACCESS", "RAG · long context · recurrent state"), P("RETENTION", "replay · regularization · modular growth"), P("PERSONALIZE", "prompt/prefix · adapter · external profile"), P("COMPACT", "summary · graph merge · distillation"), P("GOVERN", "versioned external memory · rebuild")],
    "Sleep은 문제 이름이 아니라 baseline 대비 lifecycle gain으로 평가해야 한다."),

  S(17, "ALTERNATIVES", "더 긴 inference가 memory lifecycle을 대체하지는 않는다", "대안", "Long-context and inference scaling frontier",
    ["STC-C007", "STC-C008", "STC-C009", "STC-F004"],
    [P("SCALE", "token budget·parallel samples·deliberation 증가"), P("GAIN", "현재 요청의 reasoning·evidence coverage 향상"), P("COST", "attention/KV traffic과 serving latency 증가"), P("GAP", "요청 종료 뒤 reusable state·deletion contract가 없음")],
    "Inference scaling은 wake capability이고 sleep은 cross-request reuse 문제다.", { curveLabels: ["per-request quality", "context/compute", "lifecycle gap"] }),

  S(6, "ALTERNATIVES", "Prompt compression은 가장 싼 ‘sleep-like’ baseline이다", "강한 대안", "Summary and context compression literature",
    ["STC-C010", "STC-C040", "STC-F006"],
    [P("INPUT", "긴 대화·문서를 compact representation으로 변환"), P("OUTPUT", "다음 wake의 context에 다시 삽입"), P("STRENGTH", "weight update 없이 cheap·reversible"), P("FAILURE", "세부 손실·summary drift·citation 단절")],
    "Parametric sleep은 먼저 좋은 compression+retrieval보다 낫다는 것을 보여야 한다."),

  S(10, "ALTERNATIVES", "Retrieval은 canonical truth와 reusable access를 분리한다", "강한 대안", "RAG · MemGPT · memory products",
    ["STC-C011", "STC-C012", "STC-C038", "STC-F007"],
    [P("STORE", "원문·episode를 audit 가능한 형태로 보존"), P("INDEX", "vector·keyword·metadata로 candidate 생성"), P("RERANK", "현재 query와 policy로 selection"), P("INJECT", "bounded context 또는 tool result로 제공"), P("UPDATE", "delete·correct·reindex가 weight보다 직접적")],
    "External retrieval은 현재 production governance frontier의 기본선이다."),

  S(13, "ALTERNATIVES", "Graph memory는 관계·시간·충돌을 명시적으로 보존한다", "강한 대안", "GraphRAG · Zep/Graphiti-style temporal graph",
    ["STC-C013", "STC-C038", "STC-F007"],
    [P("ENTITY", "사람·조직·artifact를 stable node로"), P("RELATION", "사건·근거·causal link를 edge로"), P("TIME", "valid-from/to와 episode sequence"), P("CONFLICT", "상충 assertion을 삭제 대신 병존"), P("QUERY", "multi-hop retrieval과 provenance trace")],
    "관계형 기억은 weight에 압축하기 전에 truth maintenance를 담당한다."),

  S(11, "ALTERNATIVES", "KV·prefix·recurrent state는 재사용 단위가 서로 다르다", "시스템", "Serving-state comparison · HOPE/NSTM context",
    ["STC-C023", "STC-C024", "STC-C025", "STC-F004"],
    [P("KV CACHE", "exact prefix의 layer activation 재사용"), P("PREFIX", "learned/compiled prompt state를 request에 부착"), P("RNN STATE", "history를 fixed-size recurrent state로 압축"), P("NEURAL MEMORY", "test-time update로 content-addressable state 생성"), P("SLEEP", "여러 state를 비동기 검증·merge")],
    "Serving 효율은 state reuse ratio와 invalidation granularity로 비교해야 한다."),

  S(14, "ALTERNATIVES", "External-first가 현재 production default인 이유", "판정", "STC-C038–C040 · product evidence",
    ["STC-C038", "STC-C039", "STC-C040", "STC-C046"],
    [P("REVERSIBLE", "원본·index·summary를 재생성 가능"), P("ADDRESSABLE", "memory ID로 inspect·edit·delete"), P("CHEAP", "per-user optimizer 없이 background transform"), P("SHAREABLE", "service와 model version 사이 독립"), P("LIMIT", "retrieval miss·latency·context tax")],
    "External memory의 약점은 분명하지만 operational contract가 성숙하다."),

  S(4, "ALTERNATIVES", "Continual learning은 update 위치를 네 층으로 나눈다", "대안", "Study §6 · continual-learning taxonomy",
    ["STC-C019", "STC-C020", "STC-C021", "STC-C027"],
    [P("DATA", "replay buffer와 curriculum"), P("LOSS", "regularization·constraint·distillation"), P("MODULE", "adapter·expert·progressive growth"), P("WEIGHT", "shared parameter rewrite")],
    "Sleep의 새로움은 technique보다 이들을 response 밖 lifecycle로 묶는 데 있다."),

  S(7, "ALTERNATIVES", "TTT·neural memory는 wake에서 state를 즉시 바꾼다", "직접 근거", "Titans · HOPE · test-time training",
    ["STC-C023", "STC-C024", "STC-C025", "STC-T08", "STC-T09"],
    [P("SIGNAL", "surprise·self-supervised loss를 현재 token에서 생성"), P("UPDATE", "memory/fast weight를 forward 중 변경"), P("USE", "같은 sequence의 다음 token이 즉시 재사용"), P("STRENGTH", "long-context를 bounded state로 압축"), P("GAP", "cross-session validate·rollback·delete는 별도")],
    "Wake update와 sleep consolidation은 dependency와 SLO가 다른 두 phase다."),

  S(15, "ALTERNATIVES", "Sleep이 이겨야 할 frontier는 이미 강하다", "비교", "STC-F006 · strongest-alternatives audit",
    ["STC-S08", "STC-F006", "STC-C040", "STC-C045"],
    [P("QUALITY", "long context·RAG·agentic retrieval"), P("RETENTION", "replay·regularization·modularity"), P("COST", "summary·cache·batch rebuild"), P("GOVERNANCE", "external truth·version·delete"), P("LATENCY", "wake TTT·recurrent memory")],
    "‘biology-inspired’는 frontier 우위의 증거가 아니다."),

  S(12, "MECHANISM", "Sleep pipeline은 여섯 operator와 두 gate로 구성된다", "제안", "Study §7 · STC-F008",
    ["STC-S07", "STC-F008", "STC-C041"],
    [P("ADMIT", "consent·novelty·risk로 event 선별"), P("BUILD", "replay·synthetic·counterexample batch"), P("TRANSFORM", "merge·distill·update"), P("VERIFY", "utility·retention·harm 평가"), P("PUBLISH", "atomic version promotion"), P("UNDO", "rollback·delete·rebuild")],
    "Admission gate와 promotion gate 사이가 sleep compute의 실제 범위다."),

  S(3, "MECHANISM", "Admission은 ‘많이 저장’이 아니라 expected value 판단이다", "제안", "Capacity-aware admission policy",
    ["STC-C029", "STC-C030", "STC-F009"],
    [P("UTILITY", "미래 재사용 확률×성공 개선"), P("NOVELTY", "기존 state와 중복되지 않는 정보"), P("RISK", "privacy·poison·conflict·copyright"), P("COST", "store·train·verify·serve·delete 비용")],
    "낮은 reuse와 높은 risk의 기억은 durable layer로 승격하지 않는다.", { equation: "admit if  E[ΔU] − λC − μR > 0" }),

  S(7, "MECHANISM", "Replay batch는 new·old·negative를 함께 담아야 한다", "학습", "Replay and continual-learning basis",
    ["STC-C019", "STC-C020", "bg:replay-buffer", "bg:hard-negative"],
    [P("NEW", "이번 wake에서 검증된 high-value episode"), P("OLD", "회귀를 막는 representative memory"), P("NEGATIVE", "혼동·거절·삭제·counterexample"), P("MIX", "task/user/time strata로 balanced sampling"), P("HOLDOUT", "promotion에 쓰지 않는 evaluation slice")],
    "Replay 없는 sleep update는 새 경험에 과적합하기 쉽다."),

  S(16, "MECHANISM", "Distillation은 memory를 줄이지만 entropy를 공짜로 없애지 못한다", "학습", "Knowledge distillation · compression",
    ["STC-C022", "STC-C028", "bg:distillation", "STC-F009"],
    [P("TEACHER", "원 episode+retrieval+larger model의 target"), P("STUDENT", "summary·adapter·compact state"), P("LOSS", "behavior·representation·retention objective"), P("TRADE", "compactness가 detail·calibration을 희생"), P("CHECK", "raw evidence로 reconstruct 가능한지")],
    "Compression ratio는 accuracy뿐 아니라 reversibility와 deletion cost로 제한된다.", { equation: "min  L_task + αL_retain + βC_state" }),

  S(8, "MECHANISM", "Dream data는 coverage를 늘리지만 오류도 증폭할 수 있다", "학습", "Synthetic replay · PAD · self-improvement",
    ["STC-C018", "STC-C031", "bg:synthetic-data", "bg:reward-model"],
    [P("PERTURB", "noise·occlusion·counterfactual로 robustness"), P("GENERATE", "희소 scenario·hard negative 보충"), P("CRITIQUE", "generator와 evaluator를 분리"), P("FILTER", "grounding·novelty·safety gate"), P("RISK", "model collapse·reward hacking·hallucinated memory")],
    "Dream은 truth source가 아니라 검증 전 candidate data다."),

  S(22, "MECHANISM", "Promotion은 utility·retention·harm 세 축을 동시에 통과해야 한다", "검증", "Benchmark blueprint · STC-F014",
    ["STC-C041", "STC-C042", "STC-F014", "STC-S15"],
    [P("UTILITY", "target task의 future success·latency 개선", "+ΔU"), P("RETENTION", "old task·identity·policy 회귀", "≤ budget"), P("HARM", "privacy·poison·bias·deletion failure", "0 critical"), P("COST", "joule·HBM byte·wall time·storage", "reported")],
    "한 축의 평균 개선으로 durable publication을 정당화할 수 없다.", { bigNumbers: ["+ΔU", "≤ε", "0"] }),

  S(21, "MECHANISM", "Publish는 파일 복사가 아니라 state transaction이다", "시스템", "STC-F011 · lifecycle transaction",
    ["STC-C041", "STC-C043", "STC-F011"],
    [P("PREPARE", "artifact·manifest·evaluation 서명"), P("VALIDATE", "schema·compatibility·policy gate"), P("COMMIT", "atomic pointer switch와 cache invalidate"), P("OBSERVE", "canary·shadow·regression telemetry"), P("ROLLBACK", "prior version으로 bounded recovery")],
    "Mutable memory는 model binary와 같은 release discipline이 필요하다."),

  S(6, "MECHANISM", "Parametric destination은 update 반경으로 구분한다", "설계", "Adapter-to-shared-weight promotion ladder",
    ["STC-C026", "STC-C027", "STC-C040", "STC-F010"],
    [P("FAST STATE", "한 sequence·session 안에서만 유효"), P("USER ADAPTER", "개인 scope와 명시적 version"), P("DOMAIN MODULE", "많은 사용자에서 검증된 reusable delta"), P("SHARED WEIGHT", "넓은 blast radius의 마지막 승격")],
    "재사용 범위가 넓을수록 evidence와 rollback budget도 커져야 한다."),

  S(7, "MECHANISM", "LoRA promotion에는 utility 외 네 가지 gate가 더 필요하다", "제안", "PEFT governance contract",
    ["STC-C027", "STC-C041", "STC-C046", "bg:lora"],
    [P("RETENTION", "base·old adapter capability 회귀 없음"), P("ROUTING", "어떤 user/domain에서 attach할지 명시"), P("COMPAT", "base/tokenizer/tool schema version 일치"), P("DELETE", "source episode 제거 시 rebuild 경로"), P("SECURITY", "poison·exfiltration red-team 통과")],
    "Parameter-efficient는 governance-efficient와 동의어가 아니다."),

  S(24, "MECHANISM", "Shared-weight sleep은 가장 강한 반증을 요구한다", "Red team", "Negative-evidence and safety audit",
    ["STC-C039", "STC-C040", "STC-C046", "STC-F015"],
    [P("INTERFERENCE", "unrelated capability가 조용히 변함"), P("UNLEARNING", "한 사용자의 삭제가 global rebuild로 확대"), P("POISON", "작은 malicious stream이 durable bias 생성"), P("IDENTITY", "개인 preference가 shared policy와 충돌"), P("ECONOMICS", "optimizer·validation cost가 reuse gain 초과")],
    "Per-user foundation-weight rewrite를 기본값으로 두지 않는다."),

  S(10, "MECHANISM", "External destination도 raw→derived 계층을 분리해야 한다", "설계", "External memory hierarchy",
    ["STC-C012", "STC-C013", "STC-C038", "STC-F010"],
    [P("L0 RAW", "immutable event·document·tool trace"), P("L1 FACT", "source-linked assertion와 temporal validity"), P("L2 INDEX", "vector·keyword·graph access path"), P("L3 SUMMARY", "task-oriented compiled view"), P("L4 CACHE", "hot query·prefix·state artifact")],
    "Derived memory는 canonical evidence를 가리켜야 고칠 수 있다."),

  S(18, "MECHANISM", "Capacity는 가득 찬 뒤가 아니라 knee 전에 관리한다", "이론", "Capacity and interference synthesis · STC-F009",
    ["STC-C028", "STC-C029", "STC-C030", "STC-F009"],
    [P("ADMIT", "expected reuse가 낮은 event drop"), P("DEDUP", "semantic·temporal duplicate merge"), P("COMPRESS", "raw→fact→summary→adapter 승격"), P("EVICT", "low-value derived state 제거"), P("REBUILD", "canonical truth에서 새 index/state 생성")],
    "용량 문제의 해법은 무한 weight가 아니라 value-aware lifecycle이다.", { curveLabels: ["retained utility", "state bytes", "interference knee"] }),

  S(20, "MECHANISM", "Delete는 모든 derived state를 따라가는 역방향 graph다", "거버넌스", "Provenance and erasure contract",
    ["STC-C044", "STC-C046", "STC-F012"],
    [P("REQUEST", "subject·source·scope를 resolve"), P("TRACE", "summary·index·adapter lineage 탐색"), P("INVALIDATE", "serving cache와 retrieval pointer 차단"), P("REBUILD", "남은 canonical evidence로 재생성"), P("ATTEST", "삭제 결과와 residual risk 기록")],
    "Lineage 없는 consolidation은 정확한 삭제를 약속할 수 없다."),

  S(19, "MECHANISM", "Sleep cadence는 queue value와 staleness SLO로 정한다", "시스템", "Scheduler model · STC-F013",
    ["STC-C032", "STC-C033", "STC-F013"],
    [P("TRIGGER", "idle·deadline·queue depth·risk event"), P("BATCH", "user/domain/model compatibility로 묶기"), P("BUDGET", "GPU·HBM·energy·validator quota"), P("PREEMPT", "wake burst가 오면 checkpoint 후 양보"), P("DEADLINE", "memory staleness와 delete SLA 우선")],
    "‘밤’이 아니라 workload slack과 value density가 sleep 시점을 결정한다."),
];
