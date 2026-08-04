const P = (label, body, metric = "") => ({ label, body, metric });
const S = (pattern, section, title, status, source, refs, points, takeaway, extras = {}) => ({
  pattern, section, title, status, source, refs, points, takeaway, ...extras,
});

export const foundationSlides = [
  S(1, "OPENING", "SLEEP–TIME COMPUTE", "통합 추론", "73 sources · 47 public claims · freeze 2026-08-05",
    ["STC-S01", "STC-C005", "STC-C040"],
    [P("WAKE", "요청 처리와 fast-state 포착"), P("SNAPSHOT", "검증 가능한 evidence generation"), P("SLEEP", "응답 밖 transformation"), P("MEMORY", "외부 truth와 reversible artifact")],
    "기억 lifecycle을 새로운 scaling·infra 축으로 읽는다.", { coverSubtitle: "장기기억·continual learning·background training을 하나의 lifecycle로 재구성" }),

  S(2, "THESIS", "핵심은 ‘자는 모델’이 아니라 기억의 이동 규칙이다", "통합 추론", "STC-S02 · claims C005/C006/C040",
    ["STC-S02", "STC-C005", "STC-C006", "STC-C040"],
    [P("언제", "배포 전 · wake/test-time · delayed sleep"), P("무엇을", "replay · distill · synthesize · merge · verify"), P("어디로", "text/vector/graph → state/adapter → shared weight")],
    "모든 이동은 provenance·capacity·rollback·deletion을 함께 가져야 한다."),

  S(3, "DEFINITION", "Strict sleep은 세 조건을 동시에 만족한다", "정의", "Operational definition · STC-C005",
    ["STC-S02", "STC-C005"],
    [P("CAUSAL BOUNDARY", "현재 사용자 응답의 critical path 밖"), P("LIFETIME INPUT", "누적 wake experience를 변환"), P("DURABLE REUSE", "검증한 state를 publish해 다음 wake가 재사용")],
    "Idle compute나 긴 reasoning만으로는 sleep이 아니다."),

  S(4, "FRAMING", "앞으로의 학습은 세 phase와 하나의 memory fabric으로 나뉜다", "통합 추론", "STC-S03 · training lifecycle",
    ["STC-S03", "bg:pretraining", "bg:test-time-training", "bg:sleep-state"],
    [P("PRE", "pretraining · post-training · 배포 전"), P("WAKE", "infer · retrieve · capture · fast update"), P("SLEEP", "replay · distill · compact · validate"), P("MEMORY", "canonical evidence · index · state · adapter")],
    "Fixed base와 session/domain mutable state를 version contract로 연결한다."),

  S(25, "ORIENTATION", "세미나는 정의에서 device experiment까지 한 방향으로 간다", "안내", "Study S02–S17 · Easy E01–E12",
    ["STC-S02", "STC-S17", "easy:E01", "easy:E12"],
    [P("01", "정의와 training background"), P("02", "근본 문제와 strongest alternatives"), P("03", "계보와 direct evidence"), P("04", "sleep mechanism과 data"), P("05", "promisingness와 scaling"), P("06", "infra·device·falsifier")],
    "답보다 먼저 비교 기준과 반증 조건을 고정한다."),

  S(14, "ANSWER FIRST", "현재의 최선은 external-first, parameter-last hybrid다", "조건부 결론", "STC-C038–C040",
    ["STC-C038", "STC-C039", "STC-C040", "STC-C046"],
    [P("RAW / EVENT", "감사·삭제 가능한 system-of-record"), P("TEXT / GRAPH", "충돌·시간·관계를 합성"), P("VECTOR / KV", "compact cache와 retrieval"), P("ADAPTER", "검증된 high-reuse subset만 promotion"), P("SHARED", "broad weight rewrite는 마지막 선택")],
    "Reversibility와 provenance가 promotion gate의 첫 조건이다."),

  S(15, "EVIDENCE", "Production 신호는 external sleep이 강하고 parametric sleep은 연구 단계다", "근거 판정", "freeze 2026-08-05 · STC-C038/C039",
    ["STC-C002", "STC-C038", "STC-C039"],
    [P("STORAGE", "외부 episode·graph는 제품 신호가 강함"), P("RETENTION", "작은 continual-learning 실험은 직접 증거"), P("ADDRESSABILITY", "weight에 남아도 access 실패 가능"), P("PLASTICITY", "새 학습과 보존은 충돌"), P("GOVERNANCE", "rollback·delete·security는 미성숙")],
    "Lifecycle adoption과 per-user foundation-weight sleep을 분리해 판단한다."),

  S(26, "SEMINAR CONTRACT", "오늘의 단위는 ‘검증 가능한 기억 lifecycle’이다", "안내", "Study figures F001–F015 · claim map",
    ["STC-F001", "STC-F015", "STC-C041"],
    [P("CAPTURE", "원 evidence와 consent를 남긴다"), P("TRANSFORM", "data·operator·version을 고정한다"), P("VERIFY", "utility·retention·harm을 함께 잰다"), P("REUSE / UNDO", "publish·observe·rollback·delete한다")],
    "External vs parametric은 진영이 아니라 destination routing decision이다."),

  S(12, "TRAINING PRIMER", "Training은 data를 미래 행동으로 변환하는 operator pipeline이다", "배경", "Training Background §1–§3",
    ["bg:data", "bg:objective", "bg:optimizer", "STC-S03"],
    [P("ADMIT", "어떤 experience를 학습 대상으로 넣을지"), P("BUILD", "batch·replay·synthetic data를 구성"), P("UPDATE", "loss·gradient·optimizer로 state를 변경"), P("VERIFY", "held-out utility와 old capability 확인"), P("PUBLISH", "serving artifact를 versioned 배포"), P("UNDO", "regression·delete 시 rollback/rebuild")],
    "Sleep은 이 pipeline을 response path 뒤로 옮긴 training lifecycle이다."),

  S(7, "TRAINING PRIMER", "한 번의 update는 forward→loss→backward→optimizer 순서다", "배경", "bg:forward-pass · loss · backpropagation · optimizer",
    ["bg:forward-pass", "bg:loss", "bg:backpropagation", "bg:optimizer"],
    [P("FORWARD", "입력과 현재 parameter로 prediction"), P("LOSS", "target과 prediction의 차이를 수치화"), P("BACKWARD", "각 parameter가 loss에 미친 기울기"), P("UPDATE", "optimizer state와 learning rate로 새 parameter")],
    "Inference-only보다 activation·gradient·optimizer-state traffic이 추가된다.", { numbers: ["x→ŷ", "L(ŷ,y)", "∇θL", "θ←θ−ηg"] }),

  S(3, "TRAINING PRIMER", "Objective·metric·evaluator는 같은 것이 아니다", "배경", "bg:objective · validation-set · reward-model",
    ["bg:objective", "bg:validation-set", "bg:reward-model"],
    [P("OBJECTIVE", "학습 중 직접 최적화하는 수치"), P("METRIC", "held-out에서 보고하는 성공·비용"), P("EVALUATOR", "memory·dream·answer의 품질 판정자")],
    "Evaluator가 틀리면 sleep은 오류를 durable state로 승격한다."),

  S(9, "TRAINING PRIMER", "Sleep data는 real·replay·synthetic·feedback을 함께 다룬다", "배경", "Training Background data map",
    ["bg:data-curation", "bg:replay-buffer", "bg:synthetic-data", "bg:provenance"],
    [P("REAL", "원 episode·문서·tool outcome"), P("REPLAY", "old capability 보호 example"), P("SYNTHETIC", "dream·counterfactual·hard negative"), P("FEEDBACK", "preference·correction·reward"), P("PROVENANCE", "source·transform·policy version")],
    "Raw sample count보다 valid·diverse·traceable evidence가 중요하다."),

  S(4, "TRAINING PRIMER", "Fine-tuning은 full weight부터 external transform까지 연속체다", "배경", "bg:fine-tuning · PEFT · adapter",
    ["bg:fine-tuning", "bg:peft", "bg:lora", "bg:external-memory"],
    [P("FULL", "모든 weight와 optimizer state를 갱신"), P("PEFT", "작은 trainable subset만 변경"), P("ADAPTER", "base와 분리된 serving delta"), P("EXTERNAL", "weight 없이 summary·graph·index 갱신")],
    "Update cost와 governance는 destination에 따라 달라진다."),

  S(16, "TRAINING PRIMER", "LoRA는 update를 작게 만들지만 training을 없애지는 않는다", "배경", "LoRA · STC-C027",
    ["bg:lora", "bg:adapter", "STC-C027"],
    [P("BASE", "공유 foundation weight는 frozen"), P("LOW-RANK", "ΔW=BA로 trainable parameter 축소"), P("STATE", "activation·gradient·optimizer는 여전히 필요"), P("SERVE", "adapter routing·cache·version 관리")],
    "작은 delta는 rollback 단위를 만들지만 sequential interference를 자동 해결하지 않는다.", { equation: "ΔW = B A,  rank(A,B) ≪ d" }),

  S(6, "TRAINING PRIMER", "Distillation은 behavior를 옮기지만 무엇을 버릴지도 결정한다", "배경", "bg:distillation · teacher/student",
    ["bg:distillation", "bg:teacher-model", "bg:student-model", "bg:kl-divergence"],
    [P("TEACHER", "긴 context·fast memory·old model의 output"), P("STUDENT", "adapter·slow memory·compact state"), P("SIGNAL", "logit/behavior divergence를 최소화"), P("RISK", "teacher error와 rare-tail loss도 압축")],
    "Compression fidelity는 downstream access와 counterexample로 검증한다."),

  S(13, "TRAINING PRIMER", "Replay와 augmentation은 retention data plane을 만든다", "배경", "STC-C019 · generated replay",
    ["STC-C019", "bg:replay-buffer", "bg:generated-replay", "bg:data-augmentation"],
    [P("RAW REPLAY", "fidelity는 높지만 storage·privacy가 큼"), P("CORESET", "대표 example만 남겨 byte 절감"), P("GENERATED", "generator가 old distribution을 근사"), P("AUGMENT", "counterfactual·rare case를 확장")],
    "Generated replay는 공짜 data가 아니라 bias가 있는 derived evidence다."),

  S(12, "TRAINING PRIMER", "RL-style sleep은 rollout→reward→update의 delayed loop다", "배경", "bg:policy-gradient · reward-model",
    ["bg:policy-gradient", "bg:reward-model", "bg:trajectory", "bg:credit-assignment"],
    [P("ROLL OUT", "dream이나 future action 후보 생성"), P("SCORE", "task utility·semantic·safety reward"), P("ASSIGN", "어떤 choice가 개선을 만들었는지"), P("UPDATE", "policy 또는 generator를 최적화"), P("VERIFY", "reward hacking과 old task regression 확인")],
    "Future utility가 늦게 보이므로 credit assignment가 핵심 병목이다."),

  S(15, "TRAINING PRIMER", "Continual learning은 retention과 plasticity를 동시에 요구한다", "배경", "STC-C020–C022",
    ["bg:continual-learning", "bg:catastrophic-forgetting", "STC-C020", "STC-C021"],
    [P("RETENTION", "old trace를 recall threshold 위에 유지"), P("PLASTICITY", "fresh control만큼 새 task를 학습"), P("ACCESS", "encoded information을 실제 answer에 적용"), P("CAPACITY", "finite state에서 interference를 관리"), P("GOVERNANCE", "source·delete·rollback을 보존")],
    "새 점수만 오르고 old access가 사라지면 continual learning이 아니다."),

  S(18, "TRAINING PRIMER", "Capacity는 parameter 수보다 utility가 꺾이는 경계다", "통합 추론", "STC-C032–C034/C045",
    ["STC-C032", "STC-C033", "STC-C034", "STC-C045"],
    [P("LOAD", "association 수×state budget"), P("INTERFERENCE", "새 update가 old access를 가림"), P("COMPACTION", "rate–distortion으로 detail 선택"), P("KNEE", "marginal utility가 급격히 감소")],
    "Static memorization capacity는 lifelong safe capacity가 아니다."),

  S(25, "TRAINING PRIMER", "Background paper는 mechanism을 읽기 위한 공통 언어다", "안내", "Training Background 57 pages",
    ["bg:forward-pass", "bg:lora", "bg:continual-learning", "bg:rlhf", "bg:versioning"],
    [P("B1", "data·forward·loss·gradient·optimizer"), P("B2", "fine-tuning·PEFT·LoRA·distillation"), P("B3", "replay·augmentation·continual learning"), P("B4", "RL·reward·credit assignment"), P("B5", "systems·version·rollback·observability")],
    "모든 training 용어는 PDF named destination으로 연결되어 있다."),

  S(16, "PROBLEM", "Long context는 작업대이지 장기기억 lifecycle이 아니다", "문제", "STC-S04 · STC-C026",
    ["STC-S04", "STC-C026", "bg:external-memory"],
    [P("READ", "선택한 evidence를 한 request에서 깊게 읽음"), P("WRITE", "durable update·merge·delete는 별도 문제"), P("COST", "매 query token·attention·retrieval 비용"), P("GOVERN", "source-of-truth와 validity를 유지해야 함")],
    "Context가 커져도 admission·conflict·staleness·deletion은 남는다.", { equation: "Context capacity ≠ durable memory policy" }),

  S(14, "PROBLEM", "Canonical truth와 compiled memory를 분리해야 한다", "통합 추론", "STC-C040/C046",
    ["STC-C040", "STC-C046"],
    [P("EVENT", "원 episode와 consent가 authoritative"), P("GRAPH", "time/version conflict를 구조화"), P("CACHE", "retrieval·state를 재사용"), P("ADAPTER", "고빈도 stable pattern만 compile"), P("WEIGHT", "shared model은 가장 느린 promotion")],
    "Derived state는 source evidence에서 재생성 가능해야 한다."),

  S(17, "PROBLEM", "Full attention의 reuse 비용이 길어질수록 state 압축이 매력적이다", "가설", "full attention vs recurrent state framing",
    ["STC-C025", "STC-C055", "background:attention"],
    [P("ATTENTION", "prefix load와 KV가 sequence에 따라 증가"), P("STATE", "fixed/learned state는 prefix를 압축"), P("CROSSOVER", "reuse와 sequence 길이가 break-even 결정"), P("CAVEAT", "압축 fidelity와 cache hit가 필요")],
    "단일 N보다 reuse×length×state-quality의 crossover atlas가 필요하다."),

  S(19, "PROBLEM", "Recurrent model도 state cache가 없으면 serving 이득을 잃는다", "문제", "Memory Caching · STC-C025",
    ["STC-C025", "STC-C055"],
    [P("KEY", "model/adapter version + exact prefix hash"), P("MATERIALIZE", "prefix에서 recurrent/neural state 생성"), P("REUSE", "동일 prefix request가 state를 공유"), P("INVALIDATE", "weight·schema·memory update가 cache를 무효화")],
    "Architecture의 O(L) 주장은 실제 trace hit ratio와 함께 검증해야 한다."),

  S(21, "PROBLEM", "Stale·conflicting fact는 update가 아니라 transaction 문제다", "문제", "STC-C010/C046",
    ["STC-C010", "STC-C046", "bg:provenance"],
    [P("SNAP", "valid-time을 포함한 evidence snapshot"), P("CAND", "merge·supersede 후보 generation"), P("VERIFY", "temporal conflict와 provenance 검사"), P("PUBLISH", "atomic generation으로 active view 전환")],
    "In-place overwrite보다 immutable generation과 lineage가 안전하다."),

  S(18, "PROBLEM", "Memory는 늘릴수록 좋다가 어느 순간 noise가 앞선다", "가설", "STC-C045 · finite-memory knee",
    ["STC-C045", "STC-F009"],
    [P("EARLY", "새 evidence가 answer utility를 빠르게 높임"), P("MIDDLE", "중복·retrieval noise가 marginal gain을 낮춤"), P("KNEE", "interference·compaction·delete cost가 급증"), P("AFTER", "admission·eviction·tiering이 capacity보다 중요")],
    "Device byte와 usable memory capacity를 같은 값으로 보지 않는다."),

  S(24, "PROBLEM", "Poison·privacy·delete는 memory medium마다 다른 잔여 위험을 남긴다", "한계", "STC-C037/C054",
    ["STC-C037", "STC-C054", "bg:unlearning"],
    [P("POISON", "오염된 record가 derived state로 확산"), P("PRIVACY", "raw replay와 user delta가 sensitive data를 보존"), P("DELETE", "summary·graph·cache·adapter까지 전파"), P("RECOVER", "lineage-driven invalidate와 rebuild")],
    "삭제가 source DB에서 끝나면 sleep lifecycle은 불완전하다."),

  S(23, "PROBLEM", "Research question은 여섯 개의 answerable 축으로 분해된다", "질문", "Study S04/S15",
    ["STC-S04", "STC-S15", "STC-F015"],
    [P("Q1", "long-context comprehension은 durable memory를 대체하는가"), P("Q2", "recurrent state를 prefix처럼 재사용할 수 있는가"), P("Q3", "prefill·decode·training의 효율 frontier는 무엇인가"), P("Q4", "continual retention과 governance를 같이 달성하는가"), P("Q5", "finite capacity에서 admission·compaction은 무엇인가"), P("Q6", "sleep이 새로운 scaling·infra 축이 되는가")],
    "각 질문은 strongest baseline과 falsifier를 가져야 한다."),

  S(5, "LINEAGE", "‘Sleep’이라는 이름보다 offline consolidation 계보가 먼저였다", "계보", "1995–2026 chronology",
    ["STC-S05", "STC-C016", "STC-C022"],
    [P("1995", "CLS · wake–sleep · pseudorehearsal"), P("2017", "deep generative replay · EWC"), P("2018", "Progress & Compress"), P("2022", "PLOS SNN · PAD · NREM/REM"), P("2023–25", "MemGPT · memory layers · test-time memory"), P("2026", "explicit LLM sleep · product dreaming")],
    "새로움은 오래된 operator를 long-lived deployment lifecycle로 묶는 데 있다."),

  S(6, "LINEAGE", "Complementary learning systems는 fast와 slow memory 역할을 나눈다", "이론", "McClelland et al. 1995",
    ["SRC-STC-0001", "STC-C017"],
    [P("FAST", "episode를 빠르게 포착하는 hippocampal 역할"), P("SLOW", "여러 episode에서 stable structure를 통합"), P("REPLAY", "offline reactivation으로 두 system 연결"), P("LIMIT", "biological plausibility는 fleet economics가 아님")],
    "Engineering hint는 stage 이름보다 fast/slow objective 분리다."),

  S(7, "LINEAGE", "Wake–Sleep은 generative model 학습 알고리즘에서 시작했다", "계보", "Hinton et al. 1995",
    ["SRC-STC-0002", "STC-S05"],
    [P("WAKE", "recognition model이 latent를 추론"), P("SLEEP", "generative sample로 recognition을 개선"), P("LEGACY", "현대 STC와 이름은 같지만 lifecycle 정의는 다름"), P("LESSON", "label보다 causal boundary를 확인")],
    "동일 용어가 동일 mechanism을 뜻하지 않는다."),

  S(12, "LINEAGE", "Pseudorehearsal과 generative replay는 old data를 재구성한다", "직접 근거", "Robins 1995 · Shin et al. 2017",
    ["STC-C019", "bg:generated-replay"],
    [P("ADMIT", "old distribution의 대표 trace 선택"), P("GENERATE", "stored exemplar 또는 pseudo-sample 재생"), P("MIX", "new data와 replay를 같은 batch에 결합"), P("UPDATE", "solver가 old/new를 함께 학습"), P("RISK", "generator bias와 privacy trade-off")],
    "Explicit sleep 없이도 retention 기능은 이미 강한 baseline이다."),

  S(8, "LINEAGE", "EWC와 GEM은 sleep 없이 update 충돌을 줄인다", "대안", "Kirkpatrick 2017 · Lopez-Paz 2017",
    ["STC-C020", "STC-C021", "bg:ewc", "bg:gem"],
    [P("EWC", "old-important parameter 이동에 Fisher penalty"), P("GEM", "old example loss가 증가하지 않도록 gradient projection"), P("STRENGTH", "retention 문제에 직접 대응"), P("LIMIT", "task 증가·buffer·governance debt")],
    "Sleep은 이 baseline보다 lifecycle utility에서 이겨야 한다."),

  S(7, "LINEAGE", "Progress & Compress는 active learning과 consolidation을 교대했다", "직접 근거", "Schwarz et al. ICML 2018",
    ["STC-C022", "SRC-STC-0008"],
    [P("ACTIVE", "새 task를 plastic column에서 학습"), P("PROGRESS", "old knowledge를 활용하며 빠르게 적응"), P("COMPRESS", "distillation로 knowledge base에 통합"), P("DIRECT LINK", "periodic refresh가 modern sleep의 전신")],
    "Wake/consolidate 분리는 2026년 이전부터 강한 대안이었다."),

  S(7, "DIRECT EVIDENCE", "2022 PLOS는 두 과제 SNN에서 sleep-like replay를 검증했다", "직접 근거", "Golden et al. PLOS CB 2022",
    ["STC-C016", "SRC-STC-0011"],
    [P("WAKE 1", "Task 1 reward-STDP"), P("WAKE 2", "Task 2가 old representation에 간섭"), P("SLEEP", "Poisson activation·unsupervised STDP"), P("LATER", "joint synaptic representation과 retention"), P("SCOPE", "842 neurons · 2 tasks · ≥10 initializations")],
    "LLM·다과제·deletion·fleet cost는 검증하지 않았다.", { bigNumbers: ["842", "2", "≥10"] }),

  S(6, "BIOLOGY", "NREM/REM alternation은 integration과 protection objective를 나눈다", "직접 근거", "PNAS hippocampus–neocortex model",
    ["STC-C017", "STC-T05"],
    [P("NREM", "replay-like autonomous dynamics로 새 정보 통합"), P("REM", "older cortical knowledge와의 관계 재구성"), P("INTERLEAVE", "phase alternation이 interference를 조절"), P("LIMIT", "model scale과 task abstraction이 제한적")],
    "Stage 모방보다 phase별 objective와 ablation이 중요하다."),

  S(8, "BIOLOGY", "PAD는 wake·perturbed replay·adversarial dreaming을 분리했다", "직접 근거", "Deperrois et al. eLife 2022",
    ["STC-C018", "STC-T13"],
    [P("WAKE", "episodic latent reconstruction"), P("NREM-LIKE", "noise·occlusion으로 robust replay"), P("REM-LIKE", "adversarial dream으로 representation 확장"), P("VALUE", "phase-specific operator 후보"), P("LIMIT", "modern LLM system evidence는 아님")],
    "Biological label은 mechanism evidence를 대체하지 않는다."),

  S(10, "LINEAGE", "계보는 agent external memory와 neural memory에서 합류한다", "통합 추론", "MemGPT·Titans·HOPE·product dreaming",
    ["STC-C011", "STC-C023", "STC-C024", "STC-C038"],
    [P("MEMGPT", "context pressure를 OS-like paging으로 해결"), P("TITANS", "test-time surprise로 neural memory 갱신"), P("HOPE", "multi-frequency optimization을 nested memory로 해석"), P("PRODUCT", "background synthesis를 durable user memory에 적용")],
    "이제 문제는 algorithm 하나가 아니라 serving lifecycle 전체다."),
];
