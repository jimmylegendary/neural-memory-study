# STC Section-Level Easy Companions Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Study Paper의 전체 논증을 training 비전공자가 따라갈 수 있도록 12개 section-level 쉬운 설명 PDF와 한 권의 합본으로 재구성한다.

**Architecture:** `manifest.json`이 각 booklet을 Study section, claim ID, canonical figure, Background concept label에 연결한다. 기존 easy-booklet 시각 언어와 LuaLaTeX pipeline을 재사용하고, 새 builder가 cross-PDF link와 claim/figure consistency를 검사한다.

**Tech Stack:** Markdown/Pandoc, LuaLaTeX, existing easy template/Lua filters, Python QA, Poppler.

## Global Constraints

- Background Paper는 prerequisite를 가르치고 Easy Companion은 Study의 논증을 쉽게 설명한다. 두 역할을 섞지 않는다.
- 각 booklet은 질문→세 문장→직관→예/반례→figure→비교→미증명 지점→링크 순서다.
- Study의 결론·수치·figure를 변경하거나 새로 만들지 않는다.
- 모든 booklet은 독립 PDF와 합본에서 동일하게 렌더링돼야 한다.

---

### Task 1: Add manifest, build, and consistency QA

**Files:**
- Create: `easy/sleep-time-compute/manifest.json`
- Create: `easy/sleep-time-compute/build.py`
- Create: `easy/sleep-time-compute/qa.py`
- Create: `easy/sleep-time-compute/template.tex`
- Create: `easy/sleep-time-compute/tests/test_easy_qa.py`

**Interfaces:**
- Consumes: Study section registry, claim map, figure ledger, Background concept registry
- Produces: `build_booklet(booklet_id: str)` and `validate_booklet(booklet_id: str)`

- [x] **Step 1: Write failing consistency tests**

```python
def test_every_claim_reference_exists_in_study_registry():
    report = validate_manifest(
        booklet_claims=["STC-C001", "STC-C999"],
        study_claims={"STC-C001"},
    )
    assert report.codes == ("unknown-study-claim:STC-C999",)

def test_every_background_link_exists():
    report = validate_background_refs(["lora", "missing"], {"lora": "bg:lora"})
    assert report.codes == ("unknown-background-concept:missing",)
```

- [x] **Step 2: Run tests and confirm failure**

```bash
python3 -m pytest -q easy/sleep-time-compute/tests/test_easy_qa.py
```

- [x] **Step 3: Implement builder and QA**

The builder reuses `easy/build/callouts.lua`, `easy/build/mathfit.lua`, CJK fonts, and canonical figures. QA rejects unknown claims/figures/concepts, missing required section blocks, hard-coded PDF page references, missing source captions, overflow, and missing glyphs.

- [x] **Step 4: Run tests and commit**

```bash
python3 -m pytest -q easy/sleep-time-compute/tests/test_easy_qa.py
git add easy/sleep-time-compute
git commit -m "feat: add sleep-time easy companion pipeline"
```

### Task 2: Write booklets E01–E04

**Files:**
- Create: `easy/sleep-time-compute/booklets/E01-definition-and-training.md`
- Create: `easy/sleep-time-compute/booklets/E02-problems.md`
- Create: `easy/sleep-time-compute/booklets/E03-lineage.md`
- Create: `easy/sleep-time-compute/booklets/E04-alternative-frontier.md`

**Interfaces:**
- Consumes: Study sections 2–6 and Background labels
- Produces: introductory half of the easy narrative

- [x] **Step 1: Write E01**

Explain the strict wake/sleep boundary, what persists, why ordinary background batching is not STC, and how fine-tuning/distillation/replay differ. Link every training prerequisite to Background.

- [x] **Step 2: Write E02**

Use concrete scenarios for static deployment, personal agent memory, knowledge freshness, bounded capacity, deletion, and latency isolation. Include cases where no learning is needed.

- [x] **Step 3: Write E03**

Explain the lineage as changing solutions to the same stability–plasticity problem, not as a list of dates. Distinguish biological evidence from computational analogy.

- [x] **Step 4: Write E04**

Explain RAG/external memory, long context, recurrent state, TTT, continual learning, model editing, periodic refresh, and STC using a common “where is memory and when does it change?” frame.

- [x] **Step 5: Build and validate**

```bash
python3 easy/sleep-time-compute/build.py --booklets E01,E02,E03,E04
python3 easy/sleep-time-compute/qa.py --booklets E01,E02,E03,E04 --strict
```

- [x] **Step 6: Commit**

```bash
git add easy/sleep-time-compute/booklets easy/sleep-time-compute/manifest.json easy/sleep-time-compute/pdf
git commit -m "docs: explain sleep-time foundations and alternatives"
```

### Task 3: Write booklets E05–E08

**Files:**
- Create: `easy/sleep-time-compute/booklets/E05-mechanisms.md`
- Create: `easy/sleep-time-compute/booklets/E06-strongest-comparison.md`
- Create: `easy/sleep-time-compute/booklets/E07-evidence-and-trends.md`
- Create: `easy/sleep-time-compute/booklets/E08-promising-and-mainstream.md`

**Interfaces:**
- Consumes: Study sections 7–11
- Produces: mechanism and strategic judgment layer

- [x] **Step 1: Write E05**

Walk through one sleep job: admission, snapshot, data construction, teacher/student, loss or reward, optimizer, destination state, validation, publication, rollback. Use one numeric toy example.

- [x] **Step 2: Write E06**

For each target problem, compare STC to the strongest actual alternative and explain the conditional winner. Include at least one counterexample where STC is unnecessary or worse.

- [x] **Step 3: Write E07**

Separate paper mechanism, released code, product feature, and deployed evidence. Explain why more papers do not automatically mean mainstream adoption.

- [x] **Step 4: Write E08**

Explain evidence maturity versus expected value, five mainstream scenarios, leading indicators, and falsifiers without presenting a single unsupported probability.

- [x] **Step 5: Build, validate, and commit**

```bash
python3 easy/sleep-time-compute/build.py --booklets E05,E06,E07,E08
python3 easy/sleep-time-compute/qa.py --booklets E05,E06,E07,E08 --strict
git add easy/sleep-time-compute
git commit -m "docs: explain sleep-time evidence and strategic verdict"
```

### Task 4: Write booklets E09–E12

**Files:**
- Create: `easy/sleep-time-compute/booklets/E09-scaling-law.md`
- Create: `easy/sleep-time-compute/booklets/E10-infrastructure.md`
- Create: `easy/sleep-time-compute/booklets/E11-device-opportunity.md`
- Create: `easy/sleep-time-compute/booklets/E12-roadmap-and-falsifiers.md`

**Interfaces:**
- Consumes: Study sections 12–17
- Produces: scaling, systems, device, and roadmap layer

- [x] **Step 1: Write E09**

Explain why a candidate scaling law needs both benefit and cost. Derive reuse break-even with a simple “one sleep job reused by N sessions” example, then introduce saturation, interference, capacity, and state-migration terms.

- [x] **Step 2: Write E10**

Explain wake plane, snapshot boundary, sleep plane, validation/publication, long-term memory tier, versioning, rollback, tenant isolation, and data movement.

- [x] **Step 3: Write E11**

Map workload primitives to HBM, CXL/host memory, SSD, endurance, bandwidth, capacity, atomicity, deduplication, copy-on-write, near-memory operations. State which opportunities survive if STC remains niche.

- [x] **Step 4: Write E12**

Explain benchmark requirements, minimum experiments, negative controls, threats to validity, and evidence that would falsify broad STC adoption.

- [x] **Step 5: Build, validate, and commit**

```bash
python3 easy/sleep-time-compute/build.py --booklets E09,E10,E11,E12
python3 easy/sleep-time-compute/qa.py --booklets E09,E10,E11,E12 --strict
git add easy/sleep-time-compute
git commit -m "docs: explain sleep-time scaling and device opportunities"
```

### Task 5: Build the combined companion and run visual QA

**Files:**
- Create: `easy/sleep-time-compute/combined.tex`
- Create: `easy/sleep-time-compute/reports/qa.json`
- Create: `build/publications/SLEEP-TIME-COMPUTE-EASY-COMPANION-KR.pdf`
- Modify: `easy/README.md`

**Interfaces:**
- Consumes: E01–E12
- Produces: one combined PDF and artifact index

- [x] **Step 1: Build all individual and combined PDFs**

```bash
python3 easy/sleep-time-compute/build.py --all --combined
```

- [x] **Step 2: Run full-document QA**

```bash
python3 easy/sleep-time-compute/qa.py --all --combined --strict --render-all-pages
python3 build/overflow_gate.py build/publications/SLEEP-TIME-COMPUTE-EASY-COMPANION-KR.pdf
```

Expected: 12 individual PDFs, combined PDF, unresolved link 0, missing glyph 0, overflow 0.

- [x] **Step 3: Update README and commit**

```bash
git add easy build/publications/SLEEP-TIME-COMPUTE-EASY-COMPANION-KR.pdf
git commit -m "docs: publish sleep-time easy companion set"
```
