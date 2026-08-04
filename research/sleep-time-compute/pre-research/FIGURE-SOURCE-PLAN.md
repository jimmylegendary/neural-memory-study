# Figure Source and Rights Plan

- principle: 논문 figure를 적극 활용하되, public artifact에는 explicit rights evidence가 있는 direct reuse만 허용한다.
- default: 관계·수치·축을 원 primary data에서 독립 redraw하고 source를 caption에 명시한다.
- shared use: Study Paper에서 확정한 figure ID를 Background, Easy, Seminar가 재사용한다.

## 1. Planned canonical figures

| Figure ID | Working title | Purpose | Source basis | Mode | Rights action |
|---|---|---|---|---|---|
| STC-F001 | Three learning clocks | pre/post, wake, sleep lifecycle | user framing + synthesis | original | project-owned SVG/PDF |
| STC-F002 | Memory-medium × update-time plane | parametric/latent/external axis | taxonomy synthesis | original | project-owned |
| STC-F003 | Evidence spine and artifact derivation | claim/source/figure traceability | program architecture | original | project-owned |
| STC-F004 | Complementary learning systems | fast store teaches slow store | McClelland 1995, Singh 2022 | redraw | verify paper values; no original artwork reuse |
| STC-F005 | Wake/NREM/REM objective factorization | biology analogy boundary | PAD and PNAS 2022 | redraw | use independent icons/shapes |
| STC-F006 | Golden PLOS two-task result | show actual limited empirical claim | PLOS 2022 result table | data-derived redraw | PLOS license evidence and numeric locator |
| STC-F007 | SRC versus joint/rehearsal | show sleep alone is not enough | Nature Communications 2022 table | data-derived redraw | license and table locator |
| STC-F008 | Explicit sleep algorithm lineage | DGDMN→PAD→Letta→Google Sleep | chronology registry | original timeline | project-owned |
| STC-F009 | STC strict lifecycle test | C1 accumulated input, C2 off-path transform, C3 later use | audit contract | original | project-owned |
| STC-F010 | External memory lifecycle | event→vector/graph→summary→retrieve | MemGPT, Mem0, Zep official artifacts | redraw | official logo use avoided by default |
| STC-F011 | Neural memory wake update | token/chunk update and fixed state | Titans/MIRAS/ATLAS/HOPE | redraw | equations and relationships only |
| STC-F012 | HOPE multi-frequency CMS | L1/L2/L3 update cadence | Nested Learning | redraw | use audited equations/values |
| STC-F013 | Parametric sleep training loop | curate→replay/dream→SFT/distill→validate→publish | Google Sleep + CL literature | original synthesis | project-owned |
| STC-F014 | Strongest-alternative decision tree | exact fact vs preference vs skill vs policy | problem–solution matrix | original | project-owned |
| STC-F015 | Five capacities | physical, functional, interference, serving, governance | capacity audit | original radial/stacked diagram | project-owned |
| STC-F016 | Capacity pressure migration | weights→buffer→generator→module/router→versions | EWC/replay/isolation synthesis | original flow | project-owned |
| STC-F017 | Sequential facts survival | prompt ceiling, LoRA/FT, composition | arXiv 2607.11020 | data-derived redraw | preprint figure rights; values/locators only |
| STC-F018 | Query-reuse crossover | compile cost vs per-query saving | Sleep-Time Compute + system model | hypothesis plot | mark as candidate, no fabricated measurement |
| STC-F019 | Candidate STC scaling surface | `C_s`, `M`, `R_q`, interference | scaling agenda | hypothesis figure | axes explicitly labeled unvalidated |
| STC-F020 | Lifetime utility objective | quality minus latency/compute/storage/risk | proposed theory | original equation diagram | project-owned |
| STC-F021 | Wake/sleep/control planes | reference infrastructure | systems audit | original architecture | project-owned |
| STC-F022 | Snapshot→candidate→canary→atomic publish | safe state transition | database/training release primitives | original sequence | project-owned |
| STC-F023 | Data movement roofline | source/model/optimizer/delta bytes | analytical systems model | data-derived/hypothesis | inputs and assumptions embedded |
| STC-F024 | Fleet crossover | shared, separate, shard-local, on-device | infrastructure scenarios | original | project-owned |
| STC-F025 | Device opportunity stack | HBM/CXL/DRAM/NVMe + version/provenance | device matrix | original | project-owned |
| STC-F026 | Promisingness vs evidence maturity | expected value separate from maturity | assessment rubric | original matrix | project-owned |
| STC-F027 | Mainstream scenario tree | broad/hybrid/external/niche/refresh | scenario analysis | original | probabilities labeled scenario bands |
| STC-F028 | Experiment matrix | matched arms, budgets, metrics | benchmark blueprint | original table/diagram | project-owned |
| STC-F029 | Training method map | SFT, LoRA, distillation, RL, replay, editing | Background Paper | original | project-owned |
| STC-F030 | End-to-end memory promotion | external source of truth with selective promotion/demotion | H-HYBRID | original | project-owned |

## 2. Direct-reuse candidate ledger

Direct reuse is provisional until `SOURCE-RIGHTS.json` records a reviewed license and exact evidence locator.

| Candidate source | Desired original element | Why direct reuse could help | Default if rights unclear |
|---|---|---|---|
| Golden et al., PLOS 2022 | architecture/result figure | blog claim verification | redraw data and mechanism |
| PAD, eLife 2022 | wake/NREM/REM schematic | objective factorization clarity | redraw independent schematic |
| Singh et al., PNAS 2022 | hippocampus–neocortex interaction | stage sequence clarity | redraw with source labels |
| Letta Sleep-Time Compute | primary/sleeptime agent architecture | defining lifecycle | redraw exact relationships |
| Titans/MIRAS/ATLAS/Nested Learning | memory block and CMS diagrams | connect prior Neural Memory study | reuse existing audited redraws first |
| Language Models Need Sleep | consolidation/training pipeline | central mechanism | redraw pipeline and data flow |
| July 2026 capacity papers | empirical curves | negative verdict | reproduce plotted values only if extractable and licensed |

## 3. Rights record fields

```json
{
  "figure_id": "STC-F006",
  "source_id": "SRC-STC-0000",
  "source_figure": "Figure 4",
  "mode": "data_derived_redraw",
  "license_id": "CC-BY-4.0",
  "license_evidence_url": "https://...",
  "license_evidence_locator": "Rights and permissions section",
  "reuse_allowed": true,
  "modification_allowed": true,
  "attribution_text": "...",
  "review_status": "reviewed",
  "reviewed_at": "2026-08-05"
}
```

빈 값이나 `publicly accessible`은 permission evidence로 인정하지 않는다.

## 4. Figure provenance rules

### 4.1 Data-derived redraw

- source table/figure의 exact numeric values와 locator를 ledger에 기록한다.
- axis range, aggregation, error bar, sample count를 원문과 동일하게 보존한다.
- visual style는 새로 만들되 결과 방향을 과장하도록 축을 자르지 않는다.
- 여러 source를 합친 그림은 각 trace/point의 source ID를 따로 기록한다.

### 4.2 Conceptual redraw

- source의 box position이나 artwork를 trace하지 않는다.
- mechanism의 input/operator/state/output relationship만 재구성한다.
- 원 저자의 interpretation과 본 연구 synthesis를 색·caption으로 분리한다.

### 4.3 Hypothesis figure

- empirical line처럼 보이는 smooth curve를 측정값으로 오인시키지 않는다.
- `candidate`, `conceptual`, `not measured` label을 figure 안과 caption에 둔다.
- equation assumption과 falsification measurement를 인접 text에 둔다.

## 5. Shared asset specification

각 figure는 다음 파일을 생성한다.

```text
figures/stc-study/STC-Fxxx/source.json
figures/stc-study/STC-Fxxx/figure.svg
figures/stc-study/STC-Fxxx/figure.pdf
figures/stc-study/STC-Fxxx/figure.png
figures/stc-study/STC-Fxxx/alt-text-ko.txt
figures/stc-study/STC-Fxxx/alt-text-en.txt
```

Study Paper는 PDF/SVG, Easy와 Seminar는 verified PNG/SVG를 사용한다. 별도 복사본을 편집하지 않고 canonical source에서 export한다.

## 6. Visual QA

모든 figure는 다음을 통과한다.

- grayscale에서 family와 evidence type을 구분 가능
- 최소 publication width에서 label readable
- Korean/English font embedding 확인
- equation과 source locator 일치
- axis/unit/sample count 누락 없음
- rights mode와 caption attribution 일치
- alt text가 관계·추세·불확실성을 설명
- direct reuse는 ledger `review_status=reviewed`, `reuse_allowed=true`

## 7. Slide integration

세미나 deck의 source note는 다음 형식을 사용한다.

```text
[Sources]
Claims: STC-C001, STC-C014
Figures: STC-F021
Primary: SRC-STC-0004 p.12 lines 431–455
Boundary: synthesis; deployment evidence not established
```

한 slide가 figure를 crop하거나 label을 제거하면 canonical figure와 동일한 claim boundary를 유지하는지 별도 visual QA한다.
