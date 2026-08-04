# STC Deep Seminar Deck Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 현재 STC deck의 디자인을 보존하면서 Study Paper의 전체 논증을 68장 본 발표와 44장 appendix, 총 112장 PPTX/PDF로 전달한다.

**Architecture:** 기존 `sleep-time-compute-research.pptx`의 모든 slide를 template inventory로 검사하고, output slide마다 source slide와 inherited edit target을 지정한다. `@oai/artifact-tool`로 starter deck을 import·편집·export하며, visible copy·speaker-note sources·claim/figure IDs는 `content.json`에서 공급한다.

**Tech Stack:** `@oai/artifact-tool`, template-following helpers, JavaScript ES modules, LibreOffice/Poppler, presentation QA scripts.

## Global Constraints

- 기존 28장 deck과 build source를 덮어쓰지 않는다.
- source master→layout→slide hierarchy와 typography를 보존한다.
- output은 정확히 112장: core 68, appendix 44.
- 외부 claim/asset마다 speaker-note `[Sources]` block이 있어야 한다.
- title 1줄, body 최소 16pt, unintended overlap·overflow·empty placeholder 0.
- 새 figure는 Study canonical figure를 사용하고 paper/deck 간 수치·caption 의미를 바꾸지 않는다.

---

### Task 1: Inspect the source deck and freeze the visual contract

**Files:**
- Create: `presentation/sleep-time-compute-deep-study/template-audit.txt`
- Create: `presentation/sleep-time-compute-deep-study/template-frame-map.json`
- Create: `presentation/sleep-time-compute-deep-study/deviation-log.txt`
- Create under temp build: `template-inspect.ndjson`, rendered source slides, layouts, manifest

**Interfaces:**
- Consumes: `presentation/sleep-time-compute/build/sleep-time-compute-research.pptx`
- Produces: inventory of all 28 source slides and 112-slide frame map

- [ ] **Step 1: Read required presentation API references**

Read completely:

```text
artifact_tool_docs/API_QUICK_START.md
artifact_tool_docs/api/API_DOCS.md
artifact_tool_docs/api/references/master.spec.md
artifact_tool_docs/api/references/layout.spec.md
artifact_tool_docs/api/references/inspect.md
artifact_tool_docs/api/references/cookbook/imported-deck.md
```

- [ ] **Step 2: Initialize artifact-tool workspace**

```bash
node "$SKILL_DIR/container_tools/setup_artifact_tool_workspace.mjs" --workspace "$TMP_DIR"
```

- [ ] **Step 3: Inspect every source slide**

```bash
node "$SKILL_DIR/template_following_scripts/inspect_template_deck.mjs" \
  --workspace "$TMP_DIR" \
  --pptx presentation/sleep-time-compute/build/sleep-time-compute-research.pptx
```

Review all 28 slide PNGs individually, layouts, `template-inspect.ndjson`, extracted media, font evidence, and placeholders.

- [ ] **Step 4: Write the visual contract**

`template-audit.txt` must document palette, fonts, sizes, title grid, source rail, page marker, divider patterns, figure frames, table patterns, note/source style, and placeholder rules.

- [ ] **Step 5: Map all 112 output slides**

Every output row requires output number, source slide number, narrative role, `duplicate-slide`, exact inherited `editTargets`, Study section ID, claim IDs, and figure IDs. Record why each source slide is reused or omitted.

- [ ] **Step 6: Validate and commit**

```bash
node "$SKILL_DIR/template_following_scripts/validate_template_plan.mjs" \
  --workspace "$TMP_DIR" \
  --pptx presentation/sleep-time-compute/build/sleep-time-compute-research.pptx \
  --map presentation/sleep-time-compute-deep-study/template-frame-map.json
git add presentation/sleep-time-compute-deep-study/template-audit.txt presentation/sleep-time-compute-deep-study/template-frame-map.json presentation/sleep-time-compute-deep-study/deviation-log.txt
git commit -m "slides: map deep seminar to source design"
```

### Task 2: Write the 112-slide content contract

**Files:**
- Create: `presentation/sleep-time-compute-deep-study/content.json`
- Create: `presentation/sleep-time-compute-deep-study/source-notes.txt`
- Create: `presentation/sleep-time-compute-deep-study/tests/test_content_contract.mjs`

**Interfaces:**
- Consumes: Study section/claim/figure registry
- Produces: audience-facing copy and note sources for every slide

- [ ] **Step 1: Write failing content tests**

```javascript
assert.equal(content.slides.length, 112);
assert.equal(content.slides.filter(s => s.section === "core").length, 68);
assert.equal(content.slides.filter(s => s.section === "appendix").length, 44);
for (const slide of content.slides) {
  assert.ok(slide.title.length > 0);
  assert.ok(slide.studySectionId);
  assert.ok(slide.claimIds.length > 0 || slide.role === "divider");
  assert.ok(slide.notes.includes("[Sources]"));
}
```

- [ ] **Step 2: Define the core sequence**

Core slide allocation:

```text
1–4 opening and decision question
5–12 training and memory background
13–20 problem decomposition
21–28 historical and company lineage
29–40 strongest alternative families
41–50 STC mechanisms and data/training
51–57 evidence, promisingness, mainstream scenarios
58–63 candidate scaling law and break-even
64–67 infrastructure and device opportunity
68 conclusion and research decision
```

- [ ] **Step 3: Define the appendix sequence**

Appendix allocation:

```text
69–80 paper-by-paper mechanism evidence
81–90 training method details
91–98 scaling derivations and sensitivity
99–105 system/device matrices
106–110 benchmark, negative controls, falsifiers
111–112 source, terminology, discussion map
```

- [ ] **Step 4: Populate every slide**

Each slide gets takeaway title, maximum three content beats, visual/figure instruction, claim IDs, exact source references, and presenter note. Do not include timing or authoring instructions in visible copy.

- [ ] **Step 5: Run tests and commit**

```bash
node --test presentation/sleep-time-compute-deep-study/tests/test_content_contract.mjs
git add presentation/sleep-time-compute-deep-study/content.json presentation/sleep-time-compute-deep-study/source-notes.txt presentation/sleep-time-compute-deep-study/tests
git commit -m "slides: define 112-slide seminar narrative"
```

### Task 3: Prepare starter deck and implement artifact-tool editor

**Files:**
- Create in temp: `template-starter.pptx`, starter renders/layouts/contact sheet
- Create: `presentation/sleep-time-compute-deep-study/src/build-deck.mjs`
- Create: `presentation/sleep-time-compute-deep-study/package.json`
- Create: `presentation/sleep-time-compute-deep-study/package-lock.json`
- Create: `presentation/sleep-time-compute-deep-study/tests/test_build.mjs`

**Interfaces:**
- Consumes: source deck, frame map, content contract, canonical figures
- Produces: editable 112-slide presentation object and PPTX

- [ ] **Step 1: Build the starter deck**

```bash
node "$SKILL_DIR/template_following_scripts/prepare_template_starter_deck.mjs" \
  --workspace "$TMP_DIR" \
  --pptx presentation/sleep-time-compute/build/sleep-time-compute-research.pptx \
  --map presentation/sleep-time-compute-deep-study/template-frame-map.json \
  --out "$TMP_DIR/template-starter.pptx" \
  --preview-dir "$TMP_DIR/template-starter-preview" \
  --layout-dir "$TMP_DIR/template-starter-layout" \
  --contact-sheet "$TMP_DIR/template-starter-contact-sheet.png"
```

- [ ] **Step 2: Write failing import/edit/export test**

Test that the build imports starter PPTX, preserves 112 slides and 16:9 dimensions, replaces all mapped title placeholders, retains required design furniture, and adds non-empty notes.

- [ ] **Step 3: Implement the artifact-tool editor**

Use `PresentationFile.importPptx`, edit mapped inherited elements by resolved IDs, replace inherited media frames, preserve masters/layouts, add `[Sources]` notes, and export with `PresentationFile.exportPptx`. Do not use `python-pptx`, PptxGenJS, or direct OOXML mutation.

- [ ] **Step 4: Run build tests**

```bash
(cd presentation/sleep-time-compute-deep-study && npm ci && npm test)
```

- [ ] **Step 5: Commit source**

```bash
git add presentation/sleep-time-compute-deep-study/src presentation/sleep-time-compute-deep-study/package*.json presentation/sleep-time-compute-deep-study/tests
git commit -m "feat: build deep seminar with artifact-tool"
```

### Task 4: Build and polish all slides

**Files:**
- Create: `presentation/sleep-time-compute-deep-study/build/sleep-time-compute-deep-study.pptx`
- Create: `presentation/sleep-time-compute-deep-study/build/sleep-time-compute-deep-study.pdf`
- Create: `presentation/sleep-time-compute-deep-study/build/renders/`
- Create: `presentation/sleep-time-compute-deep-study/build/contact-sheet.png`
- Create: `presentation/sleep-time-compute-deep-study/reports/slide-review.json`

**Interfaces:**
- Consumes: build source and canonical figures
- Produces: final deck and per-slide renders

- [ ] **Step 1: Build PPTX**

```bash
(cd presentation/sleep-time-compute-deep-study && npm run build)
```

- [ ] **Step 2: Export PDF and render all slides**

```bash
libreoffice --headless --convert-to pdf --outdir presentation/sleep-time-compute-deep-study/build presentation/sleep-time-compute-deep-study/build/sleep-time-compute-deep-study.pptx
python3 "$SKILL_DIR/container_tools/render_slides.py" presentation/sleep-time-compute-deep-study/build/sleep-time-compute-deep-study.pptx
python3 "$SKILL_DIR/container_tools/create_montage.py" --input_dir presentation/sleep-time-compute-deep-study/build/sleep-time-compute-deep-study --output_file presentation/sleep-time-compute-deep-study/build/contact-sheet.png
```

- [ ] **Step 3: Review every slide individually**

Record hierarchy, title wrapping, text fit, figure crop, table legibility, source rail, page marker, and notes for all 112 slides in `slide-review.json`. Fix every `fail` and rerender affected slides.

- [ ] **Step 4: Inspect deck-level pacing**

Use the contact sheet to verify adjacent silhouette variation, section transitions, evidence density, and that appendix is visually distinguishable but consistent.

- [ ] **Step 5: Commit polished artifact**

```bash
git add presentation/sleep-time-compute-deep-study/build presentation/sleep-time-compute-deep-study/reports/slide-review.json
git commit -m "slides: render and polish deep sleep-time seminar"
```

### Task 5: Run template fidelity and final deck QA

**Files:**
- Create: `presentation/sleep-time-compute-deep-study/reports/template-fidelity.json`
- Create: `presentation/sleep-time-compute-deep-study/reports/overflow.txt`
- Create: `presentation/sleep-time-compute-deep-study/reports/ooxml-qa.json`

**Interfaces:**
- Consumes: starter and final deck
- Produces: release-grade deck QA

- [ ] **Step 1: Run template fidelity**

```bash
node "$SKILL_DIR/template_following_scripts/check_template_fidelity.mjs" \
  --workspace "$TMP_DIR" \
  --starter-pptx "$TMP_DIR/template-starter.pptx" \
  --final-pptx presentation/sleep-time-compute-deep-study/build/sleep-time-compute-deep-study.pptx \
  --map presentation/sleep-time-compute-deep-study/template-frame-map.json \
  --starter-layout-dir "$TMP_DIR/template-starter-layout" \
  --final-layout-dir "$TMP_DIR/final-layout" \
  --edit-dir presentation/sleep-time-compute-deep-study
```

- [ ] **Step 2: Run overflow and OOXML tests**

```bash
python3 "$SKILL_DIR/container_tools/slides_test.py" presentation/sleep-time-compute-deep-study/build/sleep-time-compute-deep-study.pptx | tee presentation/sleep-time-compute-deep-study/reports/overflow.txt
(cd presentation/sleep-time-compute-deep-study && npm test)
```

Expected: slide count 112, notes 112, empty placeholders 0, missing relationships 0, invalid geometry 0, unintended overflow/overlap 0.

- [ ] **Step 3: Verify paper consistency**

Programmatically compare all slide claim IDs, numeric strings, figure IDs, and citations against Study registries; unresolved items must be zero.

- [ ] **Step 4: Commit final QA**

```bash
git add presentation/sleep-time-compute-deep-study
git commit -m "test: validate deep sleep-time seminar deck"
```
