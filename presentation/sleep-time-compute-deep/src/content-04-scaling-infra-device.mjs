const P = (label, body, metric = "") => ({ label, body, metric });
const S = (pattern, section, title, status, source, refs, points, takeaway, extras = {}) => ({
  pattern, section, title, status, source, refs, points, takeaway, ...extras,
});

export const scalingInfraDeviceSlides = [
  S(16, "SCALING", "Sleep scaling은 compute를 미래 query에 선불 배분하는 법칙이다", "가설", "Study §12 · STC-F016",
    ["STC-S12", "STC-C034", "STC-F016"],
    [P("SLEEP", "query 전 transform·training budget"), P("WAKE", "query가 도착한 뒤 latency-critical budget"), P("REUSE", "한 artifact가 서비스하는 future query 수"), P("GAIN", "quality·latency·cost frontier 이동")],
    "총 FLOP보다 누가 언제 기다리고 몇 번 재사용하는지가 새 축을 만든다.", { equation: "C̄ = C_wake + C_sleep / N_reuse" }),

  S(16, "SCALING", "순가치는 quality gain에서 다섯 비용을 뺀 값이다", "가설", "Decision rule · STC-C035",
    ["STC-C034", "STC-C035", "STC-C041", "STC-F016"],
    [P("VALUE", "future query 분포에서의 expected utility"), P("BUILD", "generate·train·compact 비용"), P("VERIFY", "retention·safety evaluation 비용"), P("SERVE", "lookup·adapter load·cache miss 비용"), P("GOV", "provenance·delete·rollback 비용")],
    "Sleep budget은 gain이 아니라 lifecycle net value로 결정한다.", { equation: "V = E_q[ΔU_q] − λC_build − αC_verify − βC_serve − γC_gov" }),

  S(17, "SCALING", "Compute scaling은 초기에 이득이 크고 뒤에는 포화한다", "검증 전 가설", "H1: sleep-compute response curve",
    ["STC-C036", "STC-H001", "STC-F016"],
    [P("LOW", "coverage 부족: 중요한 relation 미계산"), P("MID", "high-probability query를 빠르게 precompute"), P("HIGH", "중복·irrelevant dream·verification debt 증가"), P("SHIFT", "better policy가 curve를 위/왼쪽으로 이동")],
    "무제한 sleep FLOP이 아니라 policy-conditioned diminishing return을 예상한다.", { equation: "ΔU(C_s) ≈ A(1−e^{−kC_s}) − ρC_s", curveLabels: ["utility", "sleep compute", "saturation"] }),

  S(17, "SCALING", "Data scaling의 독립변수는 sample 수가 아니라 valid diversity다", "검증 전 가설", "H2: evidence-quality scaling",
    ["STC-C031", "STC-C036", "STC-H002"],
    [P("VALID", "grounded·consented·correct outcome"), P("DIVERSE", "task·time·failure mode coverage"), P("REPLAY", "old capability를 대표하는 support"), P("SYNTHETIC", "real support를 넓히는 candidate"), P("POISON", "오류율이 durable harm으로 누적")],
    "같은 token budget에서도 provenance와 coverage가 effective data를 결정한다.", { equation: "D_eff = D · q_valid · q_diverse · (1−q_dup)" }),

  S(18, "SCALING", "Capacity scaling에는 utility knee와 interference cliff가 있다", "검증 전 가설", "H3: capacity law · STC-F009",
    ["STC-C028", "STC-C029", "STC-H003", "STC-F009"],
    [P("GROW", "새 memory가 recall coverage를 높임"), P("KNEE", "중복·routing entropy가 marginal gain을 줄임"), P("CLIFF", "interference·cache miss·merge debt가 utility를 감소"), P("RESET", "compaction·eviction·module split으로 frontier 복구")],
    "용량은 parameter count가 아니라 reachable utility per byte로 측정한다.", { curveLabels: ["reachable utility", "durable bytes", "knee / cliff"] }),

  S(17, "SCALING", "Reuse와 predictability가 sleep economics의 곱셈항이다", "검증 전 가설", "H4: amortization law",
    ["STC-C034", "STC-C035", "STC-H004"],
    [P("REUSE", "같은 context/domain state가 다시 조회되는 횟수"), P("HIT", "precomputed artifact가 실제 query에 유용한 확률"), P("FRESH", "artifact가 invalidation 전 유효한 기간"), P("SHARE", "여러 tenant/query가 안전하게 공동 사용")],
    "One-shot·volatile·unpredictable workload에서는 sleep이 구조적으로 불리하다.", { equation: "amortized gain ∝ N_reuse · p_hit · p_fresh" }),

  S(19, "SCALING", "Cadence scaling은 staleness와 batch efficiency를 교환한다", "검증 전 가설", "H5: refresh-cadence law",
    ["STC-C032", "STC-C033", "STC-H005", "STC-F013"],
    [P("FAST", "fresh하지만 작은 batch·높은 launch/verify overhead"), P("SLOW", "좋은 batching이지만 stale memory 증가"), P("RISK", "delete·security event는 별도 urgent lane"), P("OPTIMUM", "workload별 cost+staleness minimum")],
    "고정 1시간 주기보다 deadline-aware queue가 합리적이다.", { equation: "T* = argmin_T  C_batch(T)+C_stale(T)+C_risk(T)" }),

  S(24, "SCALING", "Robustness cost를 빼면 scaling curve는 허상이다", "Red team", "Negative-evidence register",
    ["STC-C042", "STC-C044", "STC-C046", "STC-F015"],
    [P("RETENTION", "old capability regression"), P("COLLAPSE", "recursive synthetic-data tail loss"), P("POISON", "persistent backdoor·prompt injection"), P("PRIVACY", "derived state의 exposure·cross-tenant leak"), P("ERASURE", "delete propagation과 residual proof")],
    "안전·삭제·회귀 검증은 optional overhead가 아니라 scaling denominator다."),

  S(16, "SCALING", "External↔parametric crossover는 reuse threshold로 계산한다", "가설", "STC-F010 · destination crossover",
    ["STC-C035", "STC-C040", "STC-F010"],
    [P("EXTERNAL", "낮은 build, query마다 retrieval/prefill 비용"), P("PARAMETRIC", "높은 train/verify, 낮은 반복 serve 비용"), P("VOLATILITY", "수정이 잦으면 retrain·revalidate cost 증가"), P("THRESHOLD", "누적 serve 절감이 build+governance를 넘는 지점")],
    "Adapter 승격은 stable high-reuse subset에서만 경제성이 생긴다.", { equation: "N* = (C_train+C_verify+C_update)/(C_ext/query−C_param/query)" }),

  S(9, "SCALING", "Scaling law를 찾으려면 여섯 축을 독립 sweep한다", "실험 설계", "Scaling-law theory agenda",
    ["STC-S12", "STC-C036", "STC-F014"],
    [P("COMPUTE", "sleep FLOP·token·wall-time"), P("DATA", "valid diversity·replay ratio"), P("CAPACITY", "bytes·rank·modules·facts"), P("REUSE", "queries/context·hit probability"), P("VOLATILITY", "update/delete/conflict rate"), P("RISK", "poison/privacy/regression budget")],
    "한 benchmark의 최고점으로 법칙을 선언하지 않는다."),

  S(27, "SCALING", "현재 scaling law는 결과가 아니라 검증 가능한 hypothesis set이다", "근거 장부", "STC-H001–H005 · evidence boundary",
    ["STC-H001", "STC-H002", "STC-H003", "STC-H004", "STC-H005"],
    [P("H1", "compute diminishing return"), P("H2", "valid-diversity data law"), P("H3", "capacity knee/cliff"), P("H4", "reuse×predictability amortization"), P("H5", "cadence–staleness optimum"), P("STATUS", "all prospective; no universal exponent claimed")],
    "논문이 제안한 것은 power-law 숫자가 아니라 측정 좌표계다."),

  S(23, "SCALING", "다음 연구는 exponent보다 falsifier를 먼저 공개한다", "연구 질문", "Study §16 · falsification plan",
    ["STC-S16", "STC-C042", "STC-C045"],
    [P("HIT", "query predictability가 낮아도 gain이 유지되는가?"), P("CYCLE", "100+ consolidation 뒤 retention이 남는가?"), P("DELETE", "source 삭제가 모든 artifact에 bounded time으로 전파되는가?"), P("COST", "검증 포함 joule/query가 strongest baseline보다 낮은가?"), P("TRANSFER", "toy benchmark 밖 enterprise trace에서 재현되는가?")],
    "하나라도 실패하면 broad adoption 주장을 좁힌다."),

  S(20, "INFRA", "Infra는 wake·sleep·memory fabric의 세 plane으로 분리된다", "설계", "Study §13 · STC-F011",
    ["STC-S13", "STC-C041", "STC-F011"],
    [P("WAKE PLANE", "latency-critical inference·retrieval·fast update"), P("SLEEP PLANE", "batch transform·train·verify·publish"), P("MEMORY FABRIC", "canonical evidence·derived state·lineage·tiering"), P("CONTROL", "scheduler·policy·version·observability")],
    "Plane은 분리하되 artifact와 version contract로 연결한다."),

  S(7, "INFRA", "Wake cluster는 응답과 evidence capture를 끝까지 분리한다", "설계", "SYSTEM-INFRA-BLUEPRINT · wake path",
    ["STC-C041", "STC-C043", "STC-F011"],
    [P("READ", "model·adapter·retrieval snapshot pin"), P("INFER", "prefill/decode·tool use·optional TTT"), P("CAPTURE", "event·outcome·consent를 append-only WAL에"), P("ACK", "durable enqueue만 하고 training을 기다리지 않음"), P("SLO", "TTFT·ITL·tail latency를 우선")],
    "Wake가 sleep optimizer나 graph merge를 기다리면 causal boundary가 무너진다."),

  S(7, "INFRA", "Sleep cluster는 mixed inference+training workload다", "설계", "SYSTEM-INFRA-BLUEPRINT · sleep workers",
    ["STC-C032", "STC-C033", "STC-F013"],
    [P("GENERATE", "teacher inference·dream·counterexample"), P("TRAIN", "LoRA/full/fast-state update"), P("COMPACT", "summary·dedup·graph merge·index"), P("VERIFY", "held-out eval·red-team·delete rehearsal"), P("PUBLISH", "snapshot·manifest·atomic promotion")],
    "FLOP뿐 아니라 HBM traffic·checkpoint·metadata tail이 자원 모델에 들어간다."),

  S(13, "INFRA", "Memory fabric는 truth와 compiled artifact를 같은 graph로 잇는다", "설계", "STC-F010–F012",
    ["STC-C043", "STC-C044", "STC-F010", "STC-F012"],
    [P("CANONICAL", "raw event·document·tool outcome"), P("DERIVED", "fact·summary·vector·graph·KV/state"), P("PARAMETRIC", "adapter·expert·candidate checkpoint"), P("LINEAGE", "source→operator→artifact dependency"), P("CATALOG", "owner·scope·TTL·policy·compatibility")],
    "빠른 검색과 정확한 삭제는 동일한 lineage metadata를 공유한다."),

  S(20, "INFRA", "Data movement는 state locality와 delta transport로 줄인다", "설계", "STC-F011 · device-aware topology",
    ["STC-C033", "STC-C043", "STC-F011"],
    [P("CO-LOCATE", "replay shard와 target adapter를 같은 worker에"), P("DELTA", "raw corpus 대신 manifest·gradient/adapter delta 이동"), P("PIN", "immutable base와 hot snapshot을 HBM/DRAM에 유지"), P("STREAM", "cold evidence는 SSD/object tier에서 순차 공급"), P("CONTENT ID", "중복 artifact 전송 방지")],
    "Wake↔sleep 간 전체 state 복사는 scaling law를 data-movement law로 바꾼다."),

  S(19, "INFRA", "Scheduler는 가치·deadline·locality를 함께 최적화한다", "설계", "STC-F013 · consolidation scheduler",
    ["STC-C032", "STC-C033", "STC-F013"],
    [P("VALUE", "expected future utility per joule"), P("DEADLINE", "freshness·delete·security SLA"), P("LOCALITY", "이미 resident한 model/state/data"), P("BATCH", "compatible tenant·base·operator grouping"), P("PREEMPT", "wake surge 시 checkpoint-safe 양보")],
    "GPU utilization만 최대화하면 high-value memory가 stale해질 수 있다."),

  S(21, "INFRA", "Snapshot version은 model·memory·index를 하나의 read view로 묶는다", "설계", "STC-F011 · publication protocol",
    ["STC-C041", "STC-C043", "STC-F011"],
    [P("BUNDLE", "base+adapter+index+graph+policy manifest"), P("PIN", "한 request는 같은 version을 끝까지 사용"), P("CANARY", "새 bundle을 제한 traffic에 공개"), P("INVALIDATE", "cache와 prefix reuse key 갱신"), P("ROLLBACK", "pointer revert로 prior consistent view 복구")],
    "부분 publish는 stale index와 새 adapter가 섞이는 silent failure를 만든다."),

  S(24, "INFRA", "Persistent memory는 새로운 control-plane attack surface다", "보안", "Negative-evidence register · STC-F015",
    ["STC-C042", "STC-C046", "STC-F015"],
    [P("INGEST", "prompt injection이 replay candidate로 승격"), P("TRAIN", "poison이 adapter/weight에 durable imprint"), P("RETRIEVE", "cross-tenant·secret exfiltration"), P("PUBLISH", "malicious artifact·rollback blocking"), P("AUDIT", "output-only test가 hidden memory를 놓침")],
    "Sleep worker는 sandbox·signature·tenant isolation·provenance gate가 필요하다."),

  S(21, "INFRA", "Deletion SLA는 serving 차단과 physical rebuild를 분리한다", "거버넌스", "STC-F012 · erasure protocol",
    ["STC-C044", "STC-C046", "STC-F012"],
    [P("T0 BLOCK", "retrieval·adapter routing 즉시 차단"), P("TRACE", "derived lineage와 shared dependency 계산"), P("REBUILD", "remaining evidence로 index/state 재생성"), P("VERIFY", "canary query와 residue scan"), P("ATTEST", "logical·physical completion time 기록")],
    "빠른 logical delete와 느린 compaction을 하나의 모호한 약속으로 만들지 않는다."),

  S(22, "INFRA", "관측성은 FLOP보다 lifecycle accounting을 먼저 보여야 한다", "계측", "Benchmark blueprint · infra metrics",
    ["STC-C033", "STC-C041", "STC-F014"],
    [P("QUALITY", "future utility·retention·harm", "ΔU / ε"), P("COMPUTE", "teacher/train/eval FLOP", "FLOP"), P("TRAFFIC", "HBM↔DRAM↔SSD↔network", "bytes"), P("STATE", "raw/derived/adapter footprint", "bytes"), P("OPS", "queue age·publish·rollback·delete", "p95/p99")],
    "Memory-device 기회는 end-to-end metric에서만 확인된다.", { bigNumbers: ["ΔU", "J/query", "B/query", "p99"] }),

  S(9, "DEVICE", "Device opportunity는 destination과 operator의 교차점에 있다", "기회 지도", "Study §14 · STC-F017",
    ["STC-S14", "STC-C047", "STC-F017"],
    [P("TEXT/GRAPH", "small random I/O·metadata·snapshot"), P("VECTOR/KV", "high-read BW·quantize·tiering"), P("ADAPTER", "mixed read/write·optimizer·checkpoint"), P("SHARED", "large sequential training traffic"), P("CONTROL", "version·lineage·QoS telemetry")],
    "‘sleep accelerator’ 하나보다 계층별 data movement 병목이 더 구체적이다."),

  S(16, "DEVICE", "HBM 기회는 inference와 training traffic이 겹치는 구간이다", "기회", "Device opportunity matrix · HBM",
    ["STC-C047", "STC-C048", "STC-F017"],
    [P("READ", "base weight·teacher·replay activation"), P("WRITE", "gradient·optimizer·adapter delta·checkpoint"), P("MIX", "generate→train→eval phase 전환"), P("QOS", "wake inference와 sleep batch의 bandwidth isolation"), P("METRIC", "useful memory promotion당 HBM bytes·joule")],
    "Peak FLOP보다 mixed-phase bandwidth와 preemption recovery가 차별점이 된다."),

  S(20, "DEVICE", "CXL·pooled memory는 큰 mutable state의 residency 문제를 푼다", "기회", "Device opportunity matrix · CXL/DRAM",
    ["STC-C047", "STC-C048", "STC-F017"],
    [P("POOL", "adapter·optimizer·KV/state를 GPU 밖 공동 보유"), P("MIGRATE", "hot set만 HBM으로 promote"), P("SHARE", "immutable base와 domain artifact dedup"), P("CONSIST", "snapshot/version-aware cache coherence"), P("TAIL", "page fault·network hop p99 관리")],
    "용량 이득이 locality 손실로 상쇄되지 않는 workload boundary를 찾아야 한다."),

  S(13, "DEVICE", "SSD·object tier는 canonical truth와 rebuild bandwidth를 담당한다", "기회", "Device opportunity matrix · storage",
    ["STC-C044", "STC-C047", "STC-F017"],
    [P("LOG", "append-only episode·consent·outcome"), P("SNAPSHOT", "index·graph·adapter checkpoint"), P("STREAM", "replay batch를 sequential high-BW로 공급"), P("ENDURANCE", "frequent compaction·delete·checkpoint write"), P("RECOVERY", "corrupt derived state를 canonical truth에서 rebuild")],
    "Sleep infra는 read-mostly model serving보다 write amplification과 recovery가 크다."),

  S(12, "DEVICE", "Near-memory operator는 byte를 줄이는 지점에서만 가치가 있다", "기회", "Proposed device-side operators",
    ["STC-C047", "STC-C048", "STC-F017"],
    [P("FILTER", "metadata·consent·TTL로 replay candidate 제거"), P("DEDUP", "hash·embedding prefilter로 중복 억제"), P("QUANTIZE", "KV/vector/adapter state의 tier-aware 압축"), P("MERGE", "delta·small-state aggregation"), P("SCAN", "deletion lineage와 residue 검색")],
    "연산 offload가 아니라 host/GPU로 이동할 byte를 실제로 줄이는지가 기준이다."),

  S(28, "DEVICE", "Samsung이 우선 검증할 여섯 실험은 명확하다", "실험 장부", "Study §14–§15 · benchmark blueprint",
    ["STC-S14", "STC-S15", "STC-C047", "STC-C048"],
    [P("E1 HBM", "generate/train/eval phase별 bytes·joule"), P("E2 PREEMPT", "wake burst 중 checkpoint·resume overhead"), P("E3 TIER", "HBM↔CXL↔SSD hot-state policy와 p99"), P("E4 SNAPSHOT", "full vs delta checkpoint write amplification"), P("E5 CAPACITY", "state bytes 대비 utility knee"), P("E6 DELETE", "lineage scan·rebuild·attestation cost")],
    "Device roadmap은 broad STC adoption을 기다리지 않고 falsifiable workload로 시작할 수 있다."),

  S(25, "ROADMAP", "90일 benchmark는 baseline·cycle·device를 한 번에 묶는다", "실행", "BENCHMARK-EXPERIMENT-BLUEPRINT · STC-F014",
    ["STC-S15", "STC-C041", "STC-C042", "STC-F014"],
    [P("0–30", "external summary/RAG·LoRA·no-memory baseline 고정"), P("31–60", "reuse·volatility·capacity·100-cycle sweep"), P("61–75", "poison·delete·rollback·cross-tenant red-team"), P("76–90", "HBM/CXL/SSD trace와 system simulator"), P("OUTPUT", "Pareto frontier·falsifier·device spec proposal")],
    "같은 evidence stream과 query trace로 모든 destination을 비교한다."),

  S(26, "CLOSE", "결론: sleep은 모델 기능이 아니라 검증 가능한 memory economy다", "최종 결론", "Study synthesis · freeze 2026-08-05",
    ["STC-C038", "STC-C039", "STC-C040", "STC-C047"],
    [P("PROMISING", "background external transformation은 이미 실용적"), P("CONDITIONAL", "parametric consolidation은 stable high-reuse niche"), P("LAW", "reuse·predictability·capacity·governance가 scaling을 결정"), P("INFRA", "wake/sleep/memory planes와 atomic state contract"), P("DEVICE", "mixed traffic·tiering·snapshot·rebuild가 기회")],
    "External-first, parameter-last로 시작하고 evidence가 destination을 승격하게 한다."),
];
