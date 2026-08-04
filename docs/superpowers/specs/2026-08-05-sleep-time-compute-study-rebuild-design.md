# Sleep-Time Compute 심층 연구·출판 프로그램 재설계

**작성일:** 2026-08-05
**상태:** 사용자 승인 내용을 반영한 구현 전 설계
**대상 저장소:** `neural-memory-study`
**근거 동결 시점:** 2026-08-05

## 1. 목적과 커뮤니케이션 과업

이 프로그램의 목적은 Sleep-Time Compute(STC)를 흥미로운 비유나 단일 논문 계보로 설명하는 데 있지 않다. 다음 질문에 최신 1차 자료와 반증 가능한 분석으로 답하는 학회급 연구 패키지를 만드는 것이 목적이다.

1. STC는 어떤 미해결 문제를 해결하려는가?
2. 그 문제는 기존의 RAG, external memory, long context, recurrent state, test-time training, continual learning, model editing, periodic post-training으로 해결할 수 없는가?
3. STC가 기존 접근법보다 우월하거나 열등해지는 조건은 무엇인가?
4. 학계와 산업계는 실제로 STC 방향으로 이동하고 있는가, 아니면 외부기억·검색·agent memory 쪽이 주류가 될 가능성이 더 높은가?
5. STC가 독립된 scaling axis라면 어떤 변수와 포화·간섭·재사용 법칙을 가져야 하는가?
6. 그 변화가 memory-device 기업에 어떤 기술·제품·연구 기회를 만드는가?

커뮤니케이션 과업은 다음 한 문장으로 고정한다.

> 이 연구를 읽은 ML 학습 배경이 제한적인 Samsung SAIT의 memory·system 연구자와 학술 심사자는 STC가 별도 연구·인프라 축으로 투자할 가치가 있는 조건, 아직 증거가 부족한 지점, 경쟁 접근법 대비 우위, 검증해야 할 scaling law, memory-device 기회를 근거와 함께 판단할 수 있어야 한다.

## 2. 현재 산출물에 대한 진단

기존 저장소에는 이미 대규모 원천조사, 1차 출처 감사, scaling-law agenda, infrastructure blueprint, 한국어 번역, 28장 STC deck이 있다. 문제는 원천자료의 양이 아니라 독자용 통합 구조다.

- 기존 reader-facing STC 모노그래프는 약 1.4만 단어로 많은 근거가 압축돼 있다.
- 기존 쉬운 설명 `G08`은 약 2,900단어로 mechanism 소개에 집중한다.
- 기존 STC deck은 28장으로, 145장 Neural Memory seminar와 비교하면 배경·경쟁 접근법·논문별 mechanism·반증 조건이 충분히 드러나지 않는다.
- 기존 모노그래프는 계보와 정의에는 강하지만 “정말 promising한가”, “주류가 될까”, “최선의 해결책인가”를 중심으로 전체 서사를 재배열하지 않았다.

따라서 기존 문서를 부분 증보하지 않는다. 기존 감사 자료를 evidence base로 보존하고, 전략적 질문 중심으로 pre-research부터 신규 작성한다.

## 3. 핵심 설계 원칙

### 3.1 Canonical evidence spine

모든 산출물은 하나의 evidence spine에서 파생한다.

```text
primary sources + official artifacts
              ↓
pre-research alignment / search map / unknown register
              ↓
verified citation pool + claim ledger + figure ledger
              ↓
full study paper ───────────────┐
       ↓                        │
conference paper                │
       ↓                        │
section easy papers             │
       ↓                        │
seminar deck                    │
                                │
training-background paper ──────┘ cross-PDF references

selected source papers → faithful Korean translation PDFs
```

Study paper, 쉬운 설명본, slide가 별도로 주장이나 수치를 만들지 않는다. 모두 canonical claim ID, citation ID, figure ID를 참조한다.

### 3.2 주장과 설명의 층위

모든 핵심 문장은 다음 중 하나로 표시·관리한다.

- **직접 사실:** 논문·공식 문서·공개 코드·측정 결과가 직접 지지한다.
- **저자 주장:** 원 논문 저자의 해석이나 전망이다.
- **본 연구의 종합:** 여러 근거를 결합한 분석이다.
- **가설:** 아직 검증이 필요한 반증 가능한 명제다.
- **시나리오:** 조건부 미래 전망이며 사실 예측으로 표현하지 않는다.

### 3.3 최신성·출처 우선순위

근거 동결일은 2026-08-05다. 출처 우선순위는 다음과 같다.

1. 동료심사 논문, 최종 학회 proceedings, 공식 benchmark와 코드
2. 고정 버전 preprint, 공식 연구 블로그와 기술 보고서
3. 공식 제품 문서, API 문서, release note, repository
4. 저자 발표·인터뷰
5. 제3자 기사와 커뮤니티 자료는 발견용으로만 사용하고 핵심 주장의 단독 근거로 사용하지 않는다.

최신 제품 기능과 배포 주장은 반드시 공식 문서 또는 공식 repository로 재검증한다.

## 4. 연구 단계

### 4.1 Phase 0 — Pre-research alignment

심층 리서치 전에 현재 지식과 사용자의 objective를 정렬한다. 다음 파일을 만든다.

- `research/sleep-time-compute/pre-research/ALIGNMENT.md`
- `research/sleep-time-compute/pre-research/QUESTION-TREE.md`
- `research/sleep-time-compute/pre-research/APPROACH-TAXONOMY.md`
- `research/sleep-time-compute/pre-research/SEED-CORPUS.json`
- `research/sleep-time-compute/pre-research/UNKNOWN-UNKNOWN-REGISTER.md`
- `research/sleep-time-compute/pre-research/SEARCH-AND-SATURATION-PLAN.md`
- `research/sleep-time-compute/pre-research/TRANSLATION-CANDIDATES.md`
- `research/sleep-time-compute/pre-research/FIGURE-SOURCE-PLAN.md`

Pre-research는 결론을 먼저 고정하지 않는다. 대신 다음 working hypotheses를 서로 경쟁시키는 형태로 둔다.

- **H-STC:** 비동기 consolidation은 pre-training·test-time compute와 구별되는 유효한 scaling axis가 된다.
- **H-EXT:** 장기기억은 대부분 external memory와 retrieval로 해결되고 parametric sleep은 제한적 역할만 갖는다.
- **H-HYBRID:** 외부기억을 기본으로 두고 일부 고재사용·고가치 기억만 parametric하게 승격하는 hybrid lifecycle이 지배적이다.
- **H-REFRESH:** 별도 STC보다 주기적 post-training·model refresh가 경제적으로 우세하다.
- **H-NICHE:** STC는 개인 agent, on-device adaptation, embodied systems 등 일부 workload에서만 주류가 된다.

### 4.2 Phase 1 — Deep research

다음 literature·product family를 모두 포함한다.

1. 생물학적 replay, complementary learning systems, wake–sleep, dreaming
2. continual learning: replay, regularization, gradient constraints, isolation, expansion, pruning
3. fine-tuning, continued pre-training, instruction tuning, PEFT, LoRA, adapter, MoE/expert growth
4. data curation, coreset, augmentation, synthetic data, generated replay, distillation
5. reinforcement learning, RLHF/RLAIF, policy learning, learned memory and sleep scheduling
6. meta-learning, bilevel optimization, test-time training/learning, fast weights
7. long context, KV cache, recurrent state, state-space and neural memory
8. RAG, vector memory, graph memory, agent memory, episodic and semantic external memory
9. model editing, knowledge editing, unlearning, rollback and provenance
10. Google/DeepMind, Meta, Microsoft, OpenAI, Letta/MemGPT, Mem0, Zep 및 관련 공개 계보
11. training·serving infrastructure, tiered memory, checkpoint, snapshot, CXL, HBM, storage and data movement

Veridraft deep-survey의 citation-graph snowballing을 사용하되, 검색 포화는 단순 논문 수가 아니라 다음 cluster별 신규 핵심 주장이 더 이상 나타나지 않는지로 판단한다.

- 문제 정의
- algorithm family
- systems implementation
- deployment evidence
- negative result
- capacity and forgetting
- security, deletion and rollback
- scaling evidence
- device implication

### 4.3 Unknown-unknown 최소화

다음 탐색을 별도로 수행한다.

- STC를 지지하는 검색과 반대하는 검색을 분리한다.
- `sleep`, `dream`, `consolidation`이라는 명칭을 쓰지 않는 기능적으로 동등한 연구를 찾는다.
- adjacent field를 검색한다: database compaction, cache lifecycle, continual robotics, federated personalization, on-device adaptation, autonomous-agent experience learning.
- 실패·붕괴·negative result·retraction·benchmark leakage를 우선 탐색하는 red-team pass를 둔다.
- 기업 제품 기능과 연구 prototype을 분리한다.
- STC 없이 동일 목적을 달성하는 strongest alternative를 각 문제마다 지정한다.
- taxonomy의 빈 칸과 설명되지 않는 artifact를 unknown 후보로 다시 검색한다.

## 5. Study paper 설계

### 5.1 출판 형태

하나의 LaTeX source tree에서 두 형식을 빌드한다.

- **Full technical study:** 약 45–70쪽, references와 상세 appendix 포함
- **Conference variant:** 약 12–18쪽 본문 + 별도 appendix

목표는 특정 venue의 acceptance를 보장하는 것이 아니라, venue template만 교체하면 제출 검토가 가능한 수준의 주장·근거·figure·재현성·한계 기술을 갖추는 것이다. venue가 확정되기 전에는 `systems-paper` profile을 사용한다.

예상 source tree:

```text
paper-kr/sleep-time-compute-study/
  main.tex
  conference.tex
  sections/
  figures/
  tables/
  bibliography.bib
  claims-map.json
  figure-ledger.json
  Makefile
```

빌드 결과:

```text
build/publications/SLEEP-TIME-COMPUTE-STRATEGIC-STUDY-KR.pdf
build/publications/SLEEP-TIME-COMPUTE-CONFERENCE-KR.pdf
build/publications/SLEEP-TIME-COMPUTE-CONFERENCE-APPENDIX-KR.pdf
```

### 5.2 Paper section

1. Abstract와 contribution
2. Operational definition과 evidence boundary
3. 필요한 training background의 최소 지도
4. 해결하려는 문제의 분해
5. 역사적·생물학적·계산적 계보
6. 경쟁 접근법의 최신 상태
7. STC algorithm·data·training mechanism
8. 문제별 strongest-alternative 비교
9. evidence maturity와 industry/academia trajectory
10. STC가 promising한 조건과 실패 조건
11. 조건부 mainstream scenario
12. candidate scaling-law family
13. wake–sleep–long-term-memory infrastructure
14. memory-device opportunity와 device requirement
15. benchmark·experiment·roadmap
16. threats to validity, limitations, falsification criteria
17. conclusion

### 5.3 독창적 기여의 경계

Study paper의 독창적 기여 후보는 다음이다.

- STC의 엄격한 operational definition과 exclusion test
- 문제 × memory medium × update phase × algorithm family taxonomy
- strongest-alternative를 포함한 conditional superiority matrix
- evidence maturity와 mainstream likelihood를 분리한 평가 프레임
- reuse·interference·capacity·data movement를 포함한 candidate scaling-law family
- memory-device opportunity를 workload primitive로 환원한 architecture map
- 검증 가능한 benchmark와 반증 조건

실험으로 지지되지 않은 관계는 `law`가 아니라 `candidate law` 또는 `hypothesis`로 명시한다.

## 6. Training Background Paper

### 6.1 목적

Study paper 독자는 backward와 optimizer의 개념은 일부 알 수 있으나 현대 training stack 전체를 안다고 가정하지 않는다. Study paper 본문은 STC 논증에 집중하고, 필요한 학습 개념은 별도 background paper로 연결한다.

예상 source tree:

```text
paper-kr/sleep-time-compute-training-background/
  main.tex
  sections/
  figures/
  glossary.tex
  bibliography.bib
  Makefile
```

빌드 결과:

```text
build/publications/TRAINING-BACKGROUND-FOR-SLEEP-TIME-COMPUTE-KR.pdf
```

### 6.2 대상과 설명 방식

- 대상: hardware·memory·system 연구자, ML training 입문자
- 각 개념을 `직관 → 최소 수식 → 작은 예 → STC에서 왜 필요한가 → systems cost` 순서로 설명한다.
- 용어집, notation map, 추천 학습 순서, 개념 의존성 지도를 제공한다.
- forward/backward/optimizer는 짧게 복습하되 전체 training pipeline과 연결한다.

### 6.3 Background section

1. 모델이 학습한다는 것: data, objective, loss, generalization
2. forward, backward, autodiff, optimizer 복습
3. batch, epoch, sampling, curriculum과 evaluation split
4. pre-training, continued pre-training, fine-tuning, instruction tuning
5. supervised learning, self-supervised learning, contrastive learning
6. distillation, teacher–student learning, KL divergence
7. PEFT, LoRA, adapter, prompt tuning, sparse expert와 MoE
8. data curation, filtering, augmentation, synthetic data, replay buffer, coreset
9. catastrophic forgetting, stability–plasticity, continual learning taxonomy
10. regularization, EWC, gradient projection, replay, isolation, growth, pruning
11. reinforcement learning 기초: MDP, reward, policy, value, credit assignment
12. RLHF/RLAIF와 policy optimization, learned trigger·routing·memory policy
13. meta-learning, bilevel optimization, fast weights, TTT
14. model editing, unlearning, provenance, rollback
15. training systems: precision, activation, optimizer state, checkpoint, distributed training
16. STC mapping: sleep data, objective, trainable state, update cadence, validation, publication

### 6.4 Cross-PDF linking contract

- Background의 모든 주요 개념에 안정적인 LaTeX `\label{bg:...}`을 부여한다.
- Study paper는 `xr-hyper`/`hyperref` 기반 external reference를 사용한다.
- Study paper에서 개념이 처음 등장할 때 `Background Paper §X.Y` 링크를 둔다.
- `\bgref{concept-id}` macro로 링크 형식을 통일한다.
- build script는 모든 `\bgref`가 실제 background label에 연결되는지 검사한다.
- PDF 페이지 번호를 직접 하드코딩하지 않고 section label을 사용한다.
- 쉬운 설명본도 동일 label registry를 사용한다.

## 7. Figure 설계

### 7.1 Canonical figure ledger

모든 figure에 다음 metadata를 둔다.

- figure ID
- source paper·version·page·original figure number
- source hash
- license와 사용 범위
- `direct-reuse`, `adapted`, `original`, `internal-only` 상태
- study/easy/translation/slide 사용 위치
- caption과 attribution
- extraction 또는 생성 명령

### 7.2 원 논문 figure

- 라이선스가 허용하는 figure는 원형을 유지해 발췌한다.
- 라이선스가 불명확한 figure는 public paper에서 직접 재사용하지 않고 핵심 관계를 새로 도식화한다.
- 내부 연구용 번역본은 원문 figure를 유지하되 배포 경계를 명시한다.
- figure의 축, 범례, 표본 수, metric, 조건을 caption과 본문에서 왜곡하지 않는다.
- 원 figure 번호와 출처 페이지를 기록한다.

### 7.3 새 figure

최소 다음 original figure를 계획한다.

1. pre/wake/sleep와 parametric/external/hybrid memory의 2D taxonomy
2. 문제–접근법–강점–실패 조건 지도
3. 1983–2026 chronology와 industry lineage
4. STC lifecycle과 artifact transition
5. strongest-alternative conditional decision map
6. evidence maturity × expected value matrix
7. candidate scaling surface와 saturation/interference regime
8. wake/sleep/memory fabric system architecture
9. data movement와 reuse break-even
10. memory-device opportunity stack

## 8. 주요 논문 한국어 번역

### 8.1 선정

Pre-research와 deep-survey 뒤 핵심 12–15편을 고른다. 다음 역할을 균형 있게 포함한다.

- 생물학·CLS·wake–sleep 기초
- generative replay·dreaming·continual learning
- external/agent memory
- TTT·fast weight·neural memory
- 명시적 STC·offline consolidation
- strongest competing alternative
- capacity·forgetting·unlearning·governance

### 8.2 번역 계약

- 목적은 자연스러운 한국어 번역이며 연구자의 의견을 섞지 않는다.
- 초록, section, equation, theorem, table, figure, caption의 순서와 번호를 유지한다.
- 수식은 재입력하되 의미·기호·번호를 바꾸지 않는다.
- figure는 원문을 유지하고 caption을 번역한다.
- 참고문헌의 서지정보와 논문 제목은 원문을 유지한다.
- 명백한 원문 오탈자는 임의 수정하지 않고 번역자 주석에 최소한으로 표시한다.
- 각 문서에 source version, PDF hash, section/equation/table/figure count를 기록한다.

예상 구조:

```text
translations-kr/stc-core/{paper-id}/
  main.tex
  figures/
  source-manifest.json
  translation-qa.json
```

PDF는 `build/publications/translations-kr/`에 생성한다.

## 9. Section별 쉬운 설명본

Study paper의 주요 section마다 독립적인 쉬운 설명 PDF를 만든다. Background paper와 달리 prerequisite 전반을 가르치지 않고 해당 section의 논증을 쉽게 풀어쓴다.

각 booklet은 다음 구조를 따른다.

1. 이 section이 답하는 질문
2. 먼저 알아야 할 세 문장
3. 직관적 설명
4. 작은 예와 반례
5. canonical figure 해설
6. 기존 방법과 STC 비교
7. 무엇이 아직 증명되지 않았나
8. Study paper 및 Training Background 링크

결과물:

- section별 약 10–12개 PDF
- 전체를 합친 `SLEEP-TIME-COMPUTE-EASY-COMPANION-KR.pdf`

## 10. Seminar deck

### 10.1 범위

- 현재 28장 deck의 typography, color, grid, visual hierarchy를 source design으로 유지한다.
- 기존 deck을 덮어쓰지 않고 심층 세미나용 새 파일을 만든다.
- 본 발표 약 60–75장, technical appendix 약 40–55장, 총 100–130장을 목표로 한다.
- 한 슬라이드에 많은 문장을 넣는 대신 논문·mechanism을 여러 단계로 분해한다.

결과물:

```text
presentation/sleep-time-compute/build/sleep-time-compute-deep-study.pptx
presentation/sleep-time-compute/build/sleep-time-compute-deep-study.pdf
```

### 10.2 Narrative arc

```text
왜 지금 필요한가
→ 어떤 문제가 실제로 남아 있는가
→ 기존 접근법은 어디까지 왔는가
→ STC는 무엇을 다르게 하는가
→ 증거는 충분한가
→ 어떤 조건에서 이기는가
→ scaling axis라면 무엇을 측정해야 하는가
→ memory-device 기업은 어디에서 가치를 만들 수 있는가
→ 무엇을 공동 검증해야 하는가
```

### 10.3 출처와 QA

- 모든 비자명한 주장과 외부 figure에 speaker-note `[Sources]` block을 둔다.
- Study paper claim ID와 figure ID를 note에 함께 기록한다.
- 기존 PPTX를 template source로 검사하고 source slide를 복제·편집한다.
- 모든 slide를 렌더링해 개별 full-size 검수한다.
- overlap, clipping, title wrap, font fallback, image crop, empty placeholder, source note를 자동·수동 검사한다.

## 11. Scaling-law 연구 설계

기존 scaling law가 확립됐다고 주장하지 않는다. 다음 관계를 분리해 candidate family로 제안한다.

- sleep compute와 품질 향상의 포화
- wake evidence 양·품질·다양성과 consolidation 성능
- replay coverage와 interference
- 새로 활성화되는 parametric capacity와 forgetting
- external memory read cost와 parametric promotion의 대체율
- state migration bytes와 training FLOPs의 교환
- 한 번의 sleep 결과가 재사용되는 query/session 수에 따른 amortization
- wake SLA 손실과 sleep throughput
- cumulative sleep에서 recursive distillation과 plasticity loss

최소 objective는 다음 비용을 함께 포함한다.

```text
net value
= downstream quality × future reuse
- sleep compute cost
- memory/data movement cost
- wake-service interference
- validation/publication cost
- forgetting, staleness and rollback risk
```

Analytical model, literature meta-data, 재현 가능한 작은 실험으로 지지되는 항과 전망 항을 분리한다.

## 12. Memory-device opportunity 설계

기회는 제품 이름이 아니라 workload primitive에서 도출한다.

- immutable base snapshot + versioned delta
- copy-on-write session state
- high-capacity adapter/expert paging
- high-endurance update tier
- HBM–CXL–host–SSD artifact placement
- content-addressed deduplication
- atomic publication and rollback
- provenance-preserving delete/unlearning
- near-memory search, merge, compaction and validation
- sleep-job checkpoint/restart
- tenant isolation and encryption
- lifecycle-aware prefetch and migration

각 기회는 다음과 연결한다.

- 어떤 STC mechanism이 요구하는가?
- 기존 RAG/serving에도 공통으로 필요한가?
- capacity, bandwidth, endurance, latency 중 무엇이 병목인가?
- STC가 주류가 되지 않아도 가치가 남는가?
- device company가 algorithm team과 공동 검증할 최소 benchmark는 무엇인가?

## 13. Veridraft 적용

Veridraft v0.1.0을 다음 용도로 사용한다.

- bundle ID: `sleep-time-compute-strategic-study-2026`
- 초기 profile: `systems-paper`
- deep-survey citation snowballing과 saturation report
- P2: faithful literature summary
- P1: 측정·계산된 정량 주장과 `result_refs`
- P3: 미래 device·특허 민감 가설, 기본적으로 내부 hold
- claim import 후 deterministic gate
- contribution–evidence mapping과 precision lint
- public PDF publish/redaction pass
- venue 확정 후 readiness와 AI disclosure 검사
- event hash-chain 검증

Veridraft가 문장의 의미적 충실성을 자동 보장한다고 간주하지 않는다. Gate는 source resolvability와 claim discipline을 강제하며, 내용 충실성은 원문 대조와 별도 audit로 검증한다.

## 14. 구현 단위와 의존성

이 프로그램은 하나의 구현 계획으로 처리하기에는 크므로 다음 하위 작업으로 분해한다. 각 작업은 별도 실행 계획과 검증 checkpoint를 갖되 canonical evidence spine을 공통 interface로 사용한다.

1. **Research spine:** pre-research, deep-survey, citation pool, unknown register, claim/figure ledger
2. **Academic publication:** Full/Conference Study Paper와 Training Background Paper, cross-PDF reference
3. **Faithful translations:** 12–15편 선정, 원문 구조 보존 번역, figure/equation/table QA
4. **Easy companions:** Study section별 booklet와 합본
5. **Seminar publication:** 기존 디자인을 유지한 PPTX/PDF와 speaker-note provenance
6. **Release and audit:** Veridraft, LaTeX/PDF, translation, presentation, repository 통합 검증

의존성은 다음과 같다.

```text
Research spine
 ├─→ Academic publication ─→ Easy companions
 ├─→ Faithful translations ─┘
 └─→ Academic publication ─→ Seminar publication
                         all ─→ Release and audit
```

Academic publication은 research spine이 동결되기 전에도 outline과 background 작성은 시작할 수 있지만, 핵심 결론·수치·figure는 gated claim을 통해서만 들어간다. 번역은 seed corpus의 우선순위가 확정된 뒤 시작한다. Easy와 Seminar는 Study Paper section·claim·figure ID가 안정된 뒤 작성한다.

## 15. 오류·불확실성 처리

- 출처가 충돌하면 하나로 합치지 않고 version·조건·metric 차이를 기록한다.
- figure 라이선스가 불명확하면 public reuse를 중단하고 adapted figure로 교체한다.
- 제품 기능을 확인할 수 없으면 `not publicly verified`로 표시한다.
- scaling fit이 불안정하면 law를 제시하지 않고 regime hypothesis와 필요한 데이터만 제시한다.
- 번역에서 문장이 모호하면 원문 병기 또는 번역자 주석을 사용하고 해석을 단정하지 않는다.
- cross-PDF link가 깨지면 publication gate를 통과시키지 않는다.
- venue가 정해지지 않은 상태에서는 페이지 제한이나 익명화 요건을 임의로 가정하지 않는다.
- 사용자 소유의 기존 untracked artifact를 수정·삭제·커밋하지 않는다.

## 16. 검증 계획

### 16.1 Research

- citation pool 중복·버전·철회 검사
- cluster별 citation saturation 검사
- 핵심 주장별 primary-source locator 검사
- 지지·반대 근거 균형 검사
- company deployment claim 공식 출처 검사
- claim ledger와 manuscript 문장 매핑 검사

### 16.2 LaTeX/PDF

- clean build에서 두 번 이상 재현
- undefined citation/reference 0
- overfull box와 잘린 figure 0
- 모든 figure caption·source·license 연결
- study→background external reference 전수 검사
- PDF metadata, bookmark, link, font embedding 검사

### 16.3 Translation

- section/equation/table/figure count 원문 대조
- 무작위 문단과 핵심 claim의 역대조
- 번역에 분석적 의견이 섞이지 않았는지 검사
- 원문 figure와 caption mapping 검사

### 16.4 Easy companion

- Study paper claim·figure와 불일치 0
- background prerequisite link 전수 검사
- 용어집 및 notation 일치 검사

### 16.5 Presentation

- source-template fidelity 검사
- 전 slide render와 개별 full-size 검수
- overflow·unintended overlap·empty placeholder 0
- 모든 외부 claim/asset note source 검사
- Study paper와 수치·figure·결론 일치 검사

### 16.6 Repository

- 기존 unit/integration test 유지
- Veridraft gate/readiness/events 통과
- build manifest와 output hash 기록
- 범위 밖 파일과 사용자 untracked artifact 보존

## 17. 완료 기준

다음이 모두 충족돼야 프로그램을 완료로 본다.

1. Pre-research alignment와 unknown-unknown register가 존재한다.
2. 최신 deep-survey와 saturation report가 존재한다.
3. Study full/conference/appendix LaTeX PDF가 재현 가능하게 빌드된다.
4. Training Background PDF와 cross-PDF link가 작동한다.
5. 선정된 12–15개 핵심 논문의 충실한 한국어 번역 PDF가 있다.
6. Study section별 쉬운 설명 PDF와 합본이 있다.
7. 기존 디자인을 유지한 100–130장 세미나 PPTX/PDF가 있다.
8. figure·claim·source·license ledger가 완전하다.
9. Veridraft gate와 publication QA 결과가 저장돼 있다.
10. 모든 산출물이 README/DELIVERABLES에서 한 번에 탐색 가능하다.
