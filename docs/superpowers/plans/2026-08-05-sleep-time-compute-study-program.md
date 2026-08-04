# Sleep-Time Compute Study Program Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 2026-08-05 기준으로 pre-research를 먼저 동결하고 deep research를 수행한 뒤, 학회급 Study Paper, Training Background Paper, 핵심 논문 번역 12–15편, section별 쉬운 설명 PDF, 기존 디자인을 유지한 112장 세미나 deck을 재현 가능하게 제작한다.

**Architecture:** 모든 산출물은 Research Spine의 source/claim/figure registry를 canonical interface로 사용한다. Academic Publication이 canonical section·claim·figure ID를 고정하고, Translation·Easy·Seminar가 이를 소비하며, Release Audit이 Veridraft와 artifact별 QA를 통합한다.

**Tech Stack:** Python 3.12+, JSON/JSON Schema, Veridraft v0.1.0, LaTeX/LuaLaTeX/Biber/latexmk, Pandoc, Poppler, Graphviz, `@oai/artifact-tool`, LibreOffice, Node.js, pytest.

## Global Constraints

- 근거 동결일은 `2026-08-05`다.
- 최신 주장에는 peer-reviewed paper, fixed-version preprint, official research/product/code source만 사용한다.
- Study Paper의 주장은 `direct fact`, `author claim`, `synthesis`, `hypothesis`, `scenario` 중 하나로 분류한다.
- Training Background Paper는 Study Paper와 `xr-hyper` 기반 cross-PDF reference를 제공한다.
- 번역은 의견을 넣지 않고 원문의 section/equation/table/figure 구조와 번호를 보존한다.
- 원 figure 재사용은 license ledger에 허용 범위가 기록된 경우에만 public artifact에 포함한다.
- 세미나 deck은 기존 28장 STC deck의 디자인을 source template로 사용하고 `@oai/artifact-tool`로 구현한다.
- 기존 `seminar/` 미추적 파일과 범위 밖 파일을 수정·삭제·커밋하지 않는다.
- 모든 정량 전망은 측정값과 hypothesis를 분리하고 Veridraft claim gate를 통과한다.

---

## Plan Set and Dependency Order

1. `2026-08-05-stc-research-spine.md`
2. `2026-08-05-stc-academic-publication.md`
3. `2026-08-05-stc-faithful-translations.md`
4. `2026-08-05-stc-easy-companions.md`
5. `2026-08-05-stc-seminar-deck.md`
6. `2026-08-05-stc-release-audit.md`

```text
Research Spine
 ├── Academic Publication ── Easy Companions
 ├── Faithful Translations ─┘
 └── Academic Publication ── Seminar Deck
                      all ── Release Audit
```

### Task 1: Establish the isolated baseline

**Files:**
- Verify: `docs/superpowers/specs/2026-08-05-sleep-time-compute-study-rebuild-design.md`
- Verify: all six implementation plans above

**Interfaces:**
- Consumes: commit `b233024`
- Produces: clean branch `codex/stc-study-rebuild` and baseline test ledger

- [ ] **Step 1: Confirm worktree isolation and branch**

Run:

```bash
git rev-parse --absolute-git-dir
git rev-parse --path-format=absolute --git-common-dir
git branch --show-current
git status --short --branch
```

Expected: linked worktree on `codex/stc-study-rebuild`, no tracked changes beyond plan files.

- [ ] **Step 2: Run the existing baseline suites**

Run:

```bash
uv run --project research/sleep-time-compute pytest -q research/sleep-time-compute/tests
python3 -m unittest -q experiments/E5-attn-vs-hope/test_model.py
python3 -m unittest -q experiments/E5-attn-vs-hope/sheet/test_workbook_source.py
(cd presentation/sleep-time-compute && npm test)
```

Expected: 736 Python STC tests, 38 analytical-model tests, 24 workbook tests, and all existing presentation tests pass.

- [ ] **Step 3: Record baseline**

Create `research/sleep-time-compute/program/baseline.json` with commit SHA, UTC timestamp, commands, pass counts, and `tracked_clean: true`.

- [ ] **Step 4: Commit plan set and baseline**

```bash
git add docs/superpowers/plans/2026-08-05-sleep-time-compute-study-program.md docs/superpowers/plans/2026-08-05-stc-*.md research/sleep-time-compute/program/baseline.json
git diff --cached --check
git commit -m "docs: plan sleep-time compute study program"
```

### Task 2: Execute Research Spine plan

**Files:**
- Plan: `docs/superpowers/plans/2026-08-05-stc-research-spine.md`

**Interfaces:**
- Consumes: existing audits, papers, registries, 2026-08-05 web sources
- Produces: `research/sleep-time-compute/program/research-spine.json`

- [ ] **Step 1: Execute every checkbox in the Research Spine plan**
- [ ] **Step 2: Verify `research-spine.json` has `status: "frozen"` and zero unresolved load-bearing claims**
- [ ] **Step 3: Commit the frozen research spine**

```bash
git add research/sleep-time-compute/pre-research research/sleep-time-compute/deep-research research/sleep-time-compute/program claims/stc-study veridraft.stc-study.config.json
git diff --cached --check
git commit -m "research: freeze sleep-time compute evidence spine"
```

### Task 3: Execute Academic Publication plan

**Files:**
- Plan: `docs/superpowers/plans/2026-08-05-stc-academic-publication.md`

**Interfaces:**
- Consumes: frozen `research-spine.json`
- Produces: Study, Conference, Appendix, and Training Background PDFs plus canonical section registry

- [ ] **Step 1: Execute every checkbox in the Academic Publication plan**
- [ ] **Step 2: Verify all Study→Background links and all claim/figure IDs**
- [ ] **Step 3: Commit paper sources and reproducible outputs**

```bash
git add paper-kr/sleep-time-compute-study paper-kr/sleep-time-compute-training-background build/publications/SLEEP-TIME-COMPUTE-STRATEGIC-STUDY-KR.pdf build/publications/SLEEP-TIME-COMPUTE-CONFERENCE-KR.pdf build/publications/SLEEP-TIME-COMPUTE-CONFERENCE-APPENDIX-KR.pdf build/publications/TRAINING-BACKGROUND-FOR-SLEEP-TIME-COMPUTE-KR.pdf
git diff --cached --check
git commit -m "paper: add strategic sleep-time compute study"
```

### Task 4: Execute Faithful Translations plan

**Files:**
- Plan: `docs/superpowers/plans/2026-08-05-stc-faithful-translations.md`

**Interfaces:**
- Consumes: `translation-selection.json`, vendored source PDFs and figure ledger
- Produces: 12–15 translation source trees and PDFs

- [ ] **Step 1: Execute every checkbox in the Faithful Translations plan**
- [ ] **Step 2: Verify source hashes and structural parity for all selected papers**
- [ ] **Step 3: Commit translation sources and PDFs**

```bash
git add translations-kr/stc-core build/publications/translations-kr
git diff --cached --check
git commit -m "docs: add faithful Korean sleep-time paper translations"
```

### Task 5: Execute Easy Companions plan

**Files:**
- Plan: `docs/superpowers/plans/2026-08-05-stc-easy-companions.md`

**Interfaces:**
- Consumes: canonical Study section registry, Background labels, figure ledger
- Produces: 12 section PDFs and one combined PDF

- [ ] **Step 1: Execute every checkbox in the Easy Companions plan**
- [ ] **Step 2: Verify every booklet maps to Study claims and Background references**
- [ ] **Step 3: Commit sources and PDFs**

```bash
git add easy/sleep-time-compute build/publications/SLEEP-TIME-COMPUTE-EASY-COMPANION-KR.pdf
git diff --cached --check
git commit -m "docs: add section-level sleep-time compute companions"
```

### Task 6: Execute Seminar Deck plan

**Files:**
- Plan: `docs/superpowers/plans/2026-08-05-stc-seminar-deck.md`

**Interfaces:**
- Consumes: Study claim/figure/section registry and existing 28-slide deck
- Produces: 112-slide PPTX/PDF with source notes and QA ledgers

- [ ] **Step 1: Execute every checkbox in the Seminar Deck plan**
- [ ] **Step 2: Verify 68 core slides and 44 appendix slides**
- [ ] **Step 3: Commit source, deck, PDF, and QA reports**

```bash
git add presentation/sleep-time-compute-deep-study
git diff --cached --check
git commit -m "slides: add deep sleep-time compute seminar"
```

### Task 7: Execute Release Audit plan and integrate

**Files:**
- Plan: `docs/superpowers/plans/2026-08-05-stc-release-audit.md`

**Interfaces:**
- Consumes: all five artifact families
- Produces: final manifest, hashes, Veridraft reports, updated indexes, merge-ready branch

- [ ] **Step 1: Execute every checkbox in the Release Audit plan**
- [ ] **Step 2: Run `git diff --check main...HEAD` and verify only approved paths changed**
- [ ] **Step 3: Commit release indexes and audit outputs**

```bash
git add README.md research/sleep-time-compute/README.md research/sleep-time-compute/DELIVERABLES.md build/publications/SHA256SUMS build/publications/STC-STUDY-RELEASE-MANIFEST.json claims/stc-study/reports
git diff --cached --check
git commit -m "release: publish sleep-time compute study program"
```

- [ ] **Step 4: Use `superpowers:finishing-a-development-branch`**

Run the complete verification suite, fast-forward merge to `main` only after confirming no overlap with the 52 pre-existing untracked files, push the feature branch and `main`, and verify local/remote SHA equality.
