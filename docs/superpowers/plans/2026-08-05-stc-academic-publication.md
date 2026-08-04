# STC Academic Publication Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Full/Conference/Appendix Study Paper와 ML training 입문자를 위한 Training Background Paper를 native LaTeX로 작성하고 상호 참조 가능한 학회급 PDF로 빌드한다.

**Architecture:** Study와 Background는 독립 LaTeX project지만 공유 `concept-registry.json`, bibliography, canonical figure ledger를 사용한다. `xr-hyper` external labels로 Study의 첫 용어 사용을 Background section에 연결하고, build/QA script가 citation, label, figure, overflow, font, metadata를 검증한다.

**Tech Stack:** LuaLaTeX, Biber, latexmk, `xr-hyper`, `hyperref`, `cleveref`, BibLaTeX, Graphviz, Poppler, Python QA.

## Global Constraints

- Full Study는 45–70쪽 목표, Conference 본문은 12–18쪽 목표다.
- Background는 직관→최소 수식→예→STC relevance→systems cost 순서를 따른다.
- Study의 load-bearing claim은 Veridraft claim ID와 source locator를 갖는다.
- Study에서 background concept의 첫 사용은 `\bgref{concept-id}`로 연결한다.
- Candidate scaling law는 established law로 표현하지 않는다.
- Public figure는 license ledger가 허용한 direct reuse 또는 attributed adaptation만 사용한다.

---

### Task 1: Build the native LaTeX publication harness

**Files:**
- Create: `paper-kr/common/stc-common.sty`
- Create: `paper-kr/common/bibliography.bib`
- Create: `paper-kr/common/concept-registry.json`
- Create: `paper-kr/common/build_publications.py`
- Create: `paper-kr/common/qa_publications.py`
- Create: `paper-kr/common/tests/test_publication_qa.py`
- Create: `paper-kr/sleep-time-compute-study/Makefile`
- Create: `paper-kr/sleep-time-compute-training-background/Makefile`

**Interfaces:**
- Consumes: claim/figure/source registries
- Produces: `build_all(root: Path) -> BuildManifest`; `qa_pdf(path: Path) -> QAReport`

- [x] **Step 1: Write failing QA tests**

```python
def test_bgref_must_resolve_to_background_label(tmp_path):
    study = r"\bgref{lora} \bgref{missing-concept}"
    registry = {"lora": "bg:lora"}
    report = validate_cross_document_refs(study, registry)
    assert "missing-concept" in report.unresolved_concepts

def test_public_figure_requires_reuse_permission():
    record = {"id": "fig-x", "mode": "direct-reuse", "reuse_scope": "internal-only"}
    assert "public-reuse-not-authorized" in validate_figure(record, audience="public")
```

- [x] **Step 2: Run tests and confirm failure**

```bash
python3 -m pytest -q paper-kr/common/tests/test_publication_qa.py
```

- [x] **Step 3: Implement style, build, and QA**

The style must support Korean CJK fonts, BibLaTeX, external documents, claim margin markers, accessible color-independent callouts, figure source captions, and print-safe grayscale. Build order: Background twice with Biber, Study twice with external Background `.aux`, Conference and Appendix twice.

- [x] **Step 4: Run tests**

```bash
python3 -m pytest -q paper-kr/common/tests/test_publication_qa.py
```

- [x] **Step 5: Commit**

```bash
git add paper-kr/common paper-kr/sleep-time-compute-study/Makefile paper-kr/sleep-time-compute-training-background/Makefile
git commit -m "feat: add native LaTeX publication harness"
```

### Task 2: Write Training Background Paper foundations

**Files:**
- Create: `paper-kr/sleep-time-compute-training-background/main.tex`
- Create: `paper-kr/sleep-time-compute-training-background/sections/01-learning-objectives.tex`
- Create: `paper-kr/sleep-time-compute-training-background/sections/02-forward-backward-optimizer.tex`
- Create: `paper-kr/sleep-time-compute-training-background/sections/03-batch-sampling-evaluation.tex`
- Create: `paper-kr/sleep-time-compute-training-background/sections/04-training-regimes.tex`
- Create: `paper-kr/sleep-time-compute-training-background/sections/05-supervision-representations.tex`
- Create: `paper-kr/sleep-time-compute-training-background/sections/06-distillation.tex`
- Create: `paper-kr/sleep-time-compute-training-background/sections/07-peft-lora-adapters-moe.tex`
- Create: `paper-kr/sleep-time-compute-training-background/figures/learning-map.pdf`
- Create: `paper-kr/sleep-time-compute-training-background/figures/training-regimes.pdf`

**Interfaces:**
- Consumes: verified primary training references
- Produces: concept labels for objective, loss, gradient, optimizer, batch, epoch, generalization, pretraining, continued pretraining, fine-tuning, instruction tuning, self-supervision, distillation, KL, PEFT, LoRA, adapter, MoE

- [x] **Step 1: Define the concept dependency map**

Populate `concept-registry.json` with stable IDs, background labels, one-sentence definition, prerequisites, first Study section, and glossary term.

- [x] **Step 2: Write sections 1–3**

Use a scalar and a two-parameter example to connect objective, forward, backward, gradient, optimizer state, batch variance, epoch, validation, overfitting, and generalization. Explain memory/compute consequences without assuming calculus beyond slope.

- [x] **Step 3: Write sections 4–7**

Distinguish pre-training, continued pre-training, full fine-tuning, instruction tuning, preference optimization, PEFT, LoRA, adapters, sparse experts, and distillation by trainable state, data, objective, retained optimizer state, serving artifact, and failure mode.

- [x] **Step 4: Build conceptual figures**

`learning-map.pdf` must show data→objective→gradient→update→validation→publication. `training-regimes.pdf` must locate regimes by when learning occurs and how much state changes. Both require source IDs and text alternatives.

- [x] **Step 5: Compile and QA**

```bash
make -C paper-kr/sleep-time-compute-training-background background
python3 paper-kr/common/qa_publications.py --pdf build/publications/TRAINING-BACKGROUND-FOR-SLEEP-TIME-COMPUTE-KR.pdf --kind background
```

- [x] **Step 6: Commit**

```bash
git add paper-kr/common/concept-registry.json paper-kr/sleep-time-compute-training-background
git commit -m "docs: explain core training regimes for sleep-time compute"
```

### Task 3: Complete Training Background Paper

**Files:**
- Create: `paper-kr/sleep-time-compute-training-background/sections/08-data-augmentation-replay.tex`
- Create: `paper-kr/sleep-time-compute-training-background/sections/09-continual-learning.tex`
- Create: `paper-kr/sleep-time-compute-training-background/sections/10-stability-plasticity-methods.tex`
- Create: `paper-kr/sleep-time-compute-training-background/sections/11-rl-foundations.tex`
- Create: `paper-kr/sleep-time-compute-training-background/sections/12-rlhf-memory-policy.tex`
- Create: `paper-kr/sleep-time-compute-training-background/sections/13-meta-learning-ttt.tex`
- Create: `paper-kr/sleep-time-compute-training-background/sections/14-editing-unlearning.tex`
- Create: `paper-kr/sleep-time-compute-training-background/sections/15-training-systems.tex`
- Create: `paper-kr/sleep-time-compute-training-background/sections/16-stc-mapping.tex`
- Create: `paper-kr/sleep-time-compute-training-background/glossary.tex`
- Create: `paper-kr/sleep-time-compute-training-background/figures/continual-learning-map.pdf`
- Create: `paper-kr/sleep-time-compute-training-background/figures/rl-sleep-policy.pdf`
- Create: `paper-kr/sleep-time-compute-training-background/figures/training-memory-cost.pdf`

**Interfaces:**
- Consumes: sections 1–7 and concept registry
- Produces: complete Background `.aux` label surface for Study/Easy projects

- [x] **Step 1: Explain data and continual-learning foundations**

Cover curation, filtering, augmentation, synthetic data, coreset, replay buffer, generated replay, catastrophic forgetting, stability–plasticity, EWC/regularization, gradient projection, isolation, growth, pruning, and capacity exhaustion.

- [x] **Step 2: Explain RL and learned policies**

Cover state/action/reward/trajectory, return, policy/value, credit assignment, policy gradient, on/off-policy distinction, RLHF/RLAIF, preference data, policy optimization, and how RL could learn admission, trigger, routing, eviction, and consolidation policies. Keep algorithm names subordinate to concepts and identify unstable/offline-evaluation risks.

- [x] **Step 3: Explain meta-learning, TTT, editing, and systems**

Connect bilevel optimization, fast weights, TTT, model editing, unlearning, optimizer-state memory, activations, mixed precision, checkpointing, data parallelism, sharding, and publication/rollback to STC.

- [x] **Step 4: Write the end-to-end STC mapping**

Map every sleep operator to data, teacher, student, trainable state, loss/reward, optimizer, capacity destination, validation set, publication boundary, rollback artifact, and systems cost.

- [x] **Step 5: Compile and validate labels**

```bash
make -C paper-kr/sleep-time-compute-training-background clean background
python3 paper-kr/common/qa_publications.py --pdf build/publications/TRAINING-BACKGROUND-FOR-SLEEP-TIME-COMPUTE-KR.pdf --kind background --require-all-concepts
```

- [x] **Step 6: Commit**

```bash
git add paper-kr/sleep-time-compute-training-background paper-kr/common/concept-registry.json
git commit -m "paper: complete sleep-time compute training background"
```

### Task 4: Write Study Paper sections 1–8

**Files:**
- Create: `paper-kr/sleep-time-compute-study/main.tex`
- Create: `paper-kr/sleep-time-compute-study/sections/01-abstract-contributions.tex`
- Create: `paper-kr/sleep-time-compute-study/sections/02-operational-definition.tex`
- Create: `paper-kr/sleep-time-compute-study/sections/03-training-map.tex`
- Create: `paper-kr/sleep-time-compute-study/sections/04-problem-decomposition.tex`
- Create: `paper-kr/sleep-time-compute-study/sections/05-lineage.tex`
- Create: `paper-kr/sleep-time-compute-study/sections/06-alternative-frontier.tex`
- Create: `paper-kr/sleep-time-compute-study/sections/07-stc-mechanisms.tex`
- Create: `paper-kr/sleep-time-compute-study/sections/08-strongest-alternative-comparison.tex`
- Create: `paper-kr/sleep-time-compute-study/figures/taxonomy.pdf`
- Create: `paper-kr/sleep-time-compute-study/figures/chronology.pdf`
- Create: `paper-kr/sleep-time-compute-study/figures/problem-solution-map.pdf`
- Create: `paper-kr/sleep-time-compute-study/figures/stc-lifecycle.pdf`

**Interfaces:**
- Consumes: frozen Research Spine, Background label registry
- Produces: Study section IDs `STC-S01`…`STC-S08` and claim citations

- [x] **Step 1: Write definition and contribution boundary**

Define strict STC by causal boundary, lifetime input, persistent destination, deferred compute, and validation/publication. Explicitly exclude ordinary batching, within-query reasoning, passive cache eviction, unvalidated background summarization, and generic offline training without wake-derived state.

- [x] **Step 2: Write training prerequisite map with cross-PDF links**

Every first-use term among fine-tuning, continued pre-training, LoRA, adapter, distillation, replay, augmentation, RL, continual learning, meta-learning, TTT, editing, and unlearning must call `\bgref{...}`.

- [x] **Step 3: Write problem and lineage sections**

Separate static deployment, personalization, long-horizon agent experience, continual learning, bounded capacity, freshness, governance, and system cost. Build chronology from biological precursors through external memory, TTT/neural memory, Letta, explicit consolidation, and 2026 work.

- [x] **Step 4: Write alternative frontier and STC mechanisms**

For each family, report current strongest method, evidence status, cost model, capacity behavior, failure mode, and whether it is complementary or substitutive to STC.

- [x] **Step 5: Write strongest-alternative comparison**

Each problem must have a named non-STC baseline and conditional verdict; no straw-man comparison is allowed.

- [x] **Step 6: Build and inspect draft**

```bash
make -C paper-kr/sleep-time-compute-study full-draft
python3 paper-kr/common/qa_publications.py --pdf build/publications/SLEEP-TIME-COMPUTE-STRATEGIC-STUDY-KR.pdf --kind study --allow-incomplete-sections 9-17
```

- [x] **Step 7: Commit**

```bash
git add paper-kr/sleep-time-compute-study paper-kr/common
git commit -m "paper: draft sleep-time compute study foundations"
```

### Task 5: Write Study Paper sections 9–17 and original analysis

**Files:**
- Create: `paper-kr/sleep-time-compute-study/sections/09-evidence-trajectory.tex`
- Create: `paper-kr/sleep-time-compute-study/sections/10-promisingness.tex`
- Create: `paper-kr/sleep-time-compute-study/sections/11-mainstream-scenarios.tex`
- Create: `paper-kr/sleep-time-compute-study/sections/12-scaling-law.tex`
- Create: `paper-kr/sleep-time-compute-study/sections/13-infrastructure.tex`
- Create: `paper-kr/sleep-time-compute-study/sections/14-device-opportunity.tex`
- Create: `paper-kr/sleep-time-compute-study/sections/15-benchmark-roadmap.tex`
- Create: `paper-kr/sleep-time-compute-study/sections/16-limitations-falsifiers.tex`
- Create: `paper-kr/sleep-time-compute-study/sections/17-conclusion.tex`
- Create: `paper-kr/sleep-time-compute-study/figures/evidence-value-matrix.pdf`
- Create: `paper-kr/sleep-time-compute-study/figures/scaling-regimes.pdf`
- Create: `paper-kr/sleep-time-compute-study/figures/reuse-break-even.pdf`
- Create: `paper-kr/sleep-time-compute-study/figures/system-architecture.pdf`
- Create: `paper-kr/sleep-time-compute-study/figures/device-opportunity-stack.pdf`
- Create: `paper-kr/sleep-time-compute-study/tables/claim-table.tex`

**Interfaces:**
- Consumes: strategic synthesis matrices and Veridraft claims
- Produces: complete Full Study manuscript

- [ ] **Step 1: Separate evidence maturity from expected value**
- [ ] **Step 2: Present conditional mainstream scenarios with leading indicators and falsifiers**
- [ ] **Step 3: Derive candidate scaling-law identities, break-even equations, regimes, and empirical protocol**
- [ ] **Step 4: Derive system and device opportunities from workload primitives, including non-STC residual value**
- [ ] **Step 5: Write limitations, non-results, and falsification criteria**
- [ ] **Step 6: Reconcile every load-bearing sentence with `claim-table.tex`**
- [ ] **Step 7: Compile and run full QA**

```bash
make -C paper-kr/sleep-time-compute-study clean full
python3 paper-kr/common/qa_publications.py --pdf build/publications/SLEEP-TIME-COMPUTE-STRATEGIC-STUDY-KR.pdf --kind study --claim-map claims/stc-study/claim-map.json --figure-ledger claims/stc-study/figure-ledger.json
python3 build/overflow_gate.py build/publications/SLEEP-TIME-COMPUTE-STRATEGIC-STUDY-KR.pdf
```

- [ ] **Step 8: Commit**

```bash
git add paper-kr/sleep-time-compute-study
git commit -m "paper: complete strategic sleep-time compute study"
```

### Task 6: Build Conference and Appendix variants

**Files:**
- Create: `paper-kr/sleep-time-compute-study/conference.tex`
- Create: `paper-kr/sleep-time-compute-study/appendix.tex`
- Create: `paper-kr/sleep-time-compute-study/conference-guidelines.md`
- Create: `paper-kr/sleep-time-compute-study/reports/page-budget.json`

**Interfaces:**
- Consumes: Full Study sections and claims
- Produces: 12–18 page Conference PDF and separate Appendix PDF

- [ ] **Step 1: Select conference narrative**

Keep operational definition, strongest-alternative comparison, conditional verdict, candidate scaling law, device/system implications, limitations, and contribution table in the main paper. Move exhaustive chronology, paper catalog, derivations, extended tables, and benchmark details to appendix.

- [ ] **Step 2: Build both variants**

```bash
make -C paper-kr/sleep-time-compute-study conference appendix
```

- [ ] **Step 3: Verify page budget and references**

```bash
python3 paper-kr/common/qa_publications.py --pdf build/publications/SLEEP-TIME-COMPUTE-CONFERENCE-KR.pdf --kind conference --min-pages 12 --max-pages 18
python3 paper-kr/common/qa_publications.py --pdf build/publications/SLEEP-TIME-COMPUTE-CONFERENCE-APPENDIX-KR.pdf --kind appendix
```

- [ ] **Step 4: Run Veridraft manuscript gate**

```bash
python3 -m veridraft readiness --venue mlsys --data-dir .veridraft-stc-study > claims/stc-study/reports/readiness-mlsys.txt
python3 -m veridraft events --data-dir .veridraft-stc-study > claims/stc-study/reports/events.txt
```

- [ ] **Step 5: Commit**

```bash
git add paper-kr/sleep-time-compute-study claims/stc-study/reports build/publications/SLEEP-TIME-COMPUTE-*.pdf build/publications/TRAINING-BACKGROUND-FOR-SLEEP-TIME-COMPUTE-KR.pdf
git commit -m "paper: add conference and background publication set"
```
