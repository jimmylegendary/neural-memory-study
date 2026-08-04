# STC Faithful Korean Translations Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Research Spine이 선정한 핵심 논문 12–15편을 의견 없이 자연스러운 한국어로 번역하고 원문의 section, equation, table, figure, caption 구조를 보존한 LaTeX/PDF 세트를 만든다.

**Architecture:** `translation-selection.json`이 corpus를 고정하고, 각 paper directory가 source manifest, body source, figures, main LaTeX, QA report를 보유한다. 공통 builder는 원문과 번역본의 구조 count, source hash, figure mapping, missing glyph, overflow를 검사한다.

**Tech Stack:** Python, Pandoc, LuaLaTeX, Poppler/pdfimages, LaTeX, JSON, pytest.

## Global Constraints

- 번역 목적만 수행하며 분석·평가·Study Paper 결론을 삽입하지 않는다.
- 참고문헌 서지정보와 논문 제목은 원문을 유지한다.
- 원문 오탈자는 임의 수정하지 않고 최소 번역자 주석으로 표시한다.
- public redistribution rights가 불명확한 full translation과 figure는 `internal-only`로 표시한다.
- source PDF hash와 고정 version이 없는 논문은 build 대상에 들어갈 수 없다.

---

### Task 1: Add translation manifest, builder, and structural QA

**Files:**
- Create: `translations-kr/stc-core/manifest.json`
- Create: `translations-kr/stc-core/build_translations.py`
- Create: `translations-kr/stc-core/qa_translations.py`
- Create: `translations-kr/stc-core/common/translation.sty`
- Create: `translations-kr/stc-core/tests/test_translation_qa.py`

**Interfaces:**
- Consumes: `claims/stc-study/translation-selection.json`
- Produces: `build_translation(paper_id: str) -> TranslationBuild`; `qa_translation(paper_id: str) -> TranslationQA`

- [ ] **Step 1: Write failing structural tests**

```python
def test_translation_requires_source_hash_and_version(tmp_path):
    record = fixture_record(source_sha256="", version="")
    report = validate_record(record)
    assert {"missing-source-hash", "missing-source-version"} <= set(report.codes)

def test_equation_table_figure_counts_must_match_manifest(tmp_path):
    report = compare_structure(
        expected={"equations": 7, "tables": 3, "figures": 4},
        actual={"equations": 6, "tables": 3, "figures": 4},
    )
    assert report.codes == ("equation-count-mismatch",)
```

- [ ] **Step 2: Run tests and confirm failure**

```bash
python3 -m pytest -q translations-kr/stc-core/tests/test_translation_qa.py
```

- [ ] **Step 3: Implement deterministic builder and QA**

Each paper record must contain title, authors, year, venue/status, DOI/arXiv/version, source PDF path/hash, rights state, original section/equation/table/figure counts, translation path, and output PDF path. Builder must preserve labels and number figures/tables/equations according to source manifest.

- [ ] **Step 4: Run tests**

```bash
python3 -m pytest -q translations-kr/stc-core/tests/test_translation_qa.py
```

- [ ] **Step 5: Commit**

```bash
git add translations-kr/stc-core
git commit -m "feat: add faithful translation build and QA"
```

### Task 2: Freeze the selected corpus and extract source structures

**Files:**
- Modify: `translations-kr/stc-core/manifest.json`
- Create per selected paper ID recorded in `translation-selection.json`: `translations-kr/stc-core/$paper_id/source-manifest.json`
- Create per selected paper ID recorded in `translation-selection.json`: `translations-kr/stc-core/$paper_id/figures/`
- Create: `translations-kr/stc-core/reports/corpus-audit.json`

**Interfaces:**
- Consumes: exact IDs from `translation-selection.json`
- Produces: 12–15 source manifests with structural inventories

- [ ] **Step 1: Verify every selected source**

For each selected paper, validate local PDF hash against selection record, verify title/authors/version from DOI/arXiv/proceedings, and record rights status.

- [ ] **Step 2: Extract page images or embedded figures without altering content**

Use `pdfimages` or deterministic page crop instructions. Record original figure number, page, bounding box or embedded object index, resolution, and caption source.

- [ ] **Step 3: Inventory structural elements**

Count top-level and nested sections, numbered equations, tables, algorithms, theorem-like environments, figures, and appendices. Store count and source locators in each manifest.

- [ ] **Step 4: Verify corpus size and roles**

```bash
python3 translations-kr/stc-core/qa_translations.py --corpus-only
```

Expected: 12–15 papers; required roles from the design are all represented.

- [ ] **Step 5: Commit**

```bash
git add translations-kr/stc-core/manifest.json translations-kr/stc-core/*/source-manifest.json translations-kr/stc-core/*/figures translations-kr/stc-core/reports/corpus-audit.json
git commit -m "research: freeze core sleep-time translation corpus"
```

### Task 3: Migrate and verify existing Korean translations

**Files:**
- Create per reusable existing paper ID: `translations-kr/stc-core/$paper_id/main.tex`
- Create per reusable existing paper ID: `translations-kr/stc-core/$paper_id/body.tex`
- Create per reusable existing paper ID: `translations-kr/stc-core/$paper_id/translation-qa.json`
- Preserve: existing flat files under `translations-kr/*.md`

**Interfaces:**
- Consumes: existing Titans, MIRAS, ATLAS, TNT, Nested Learning, Memory Caching, Language Models Need Sleep, NSTM translations when selected
- Produces: structurally checked native-LaTeX wrappers and PDFs

- [ ] **Step 1: Convert reusable translation sources deterministically**

Use Pandoc only as a syntax conversion stage; review generated LaTeX for equations, tables, section hierarchy, figure placement, code blocks, and CJK punctuation. Keep legacy Markdown unchanged as provenance.

- [ ] **Step 2: Reconcile full versus abbreviated versions**

When both `FULL` and abbreviated files exist, use the full translation as body. Record omitted source sections as QA failures rather than silently accepting the abbreviated copy.

- [ ] **Step 3: Restore source figure and caption positions**

Insert each extracted figure after the same argument or subsection as the source. Translate captions without changing axes, legends, or metric names.

- [ ] **Step 4: Build and validate migrated papers**

```bash
python3 translations-kr/stc-core/build_translations.py --existing-only
python3 translations-kr/stc-core/qa_translations.py --existing-only --strict
```

- [ ] **Step 5: Commit**

```bash
git add translations-kr/stc-core build/publications/translations-kr
git commit -m "docs: migrate existing neural-memory translations to LaTeX"
```

### Task 4: Translate newly selected foundational and competing papers

**Files:**
- Create per new paper ID: `translations-kr/stc-core/$paper_id/main.tex`
- Create per new paper ID: `translations-kr/stc-core/$paper_id/body.tex`
- Create per new paper ID: `translations-kr/stc-core/$paper_id/translation-qa.json`

**Interfaces:**
- Consumes: source PDF/text, source manifest, figures
- Produces: faithful full-paper Korean translation

- [ ] **Step 1: Translate title metadata and abstract**

Keep the original title in metadata and provide a Korean title line. Preserve all claims, qualifiers, modality, and reported numbers.

- [ ] **Step 2: Translate every main section in source order**

Do not summarize. Preserve inline citations, footnotes, algorithms, definitions, theorem statements, limitations, and discussion. Technical terms use the project glossary but the first use includes the original term.

- [ ] **Step 3: Reproduce equations, tables, and captions**

Do not rename symbols. Preserve equation and table numbers. Translate table headings and notes while retaining metric names and numeric values verbatim.

- [ ] **Step 4: Translate appendices and limitations**

Appendix experiments, prompts, hyperparameters, and negative results are required; they may not be dropped for length.

- [ ] **Step 5: Run paper-scoped QA after each translation**

```bash
paper_id=$(python3 translations-kr/stc-core/select_next.py --selection translations-kr/stc-core/translation-selection.json)
python3 translations-kr/stc-core/build_translations.py --paper "$paper_id"
python3 translations-kr/stc-core/qa_translations.py --paper "$paper_id" --strict
```

Expected: structure parity, missing glyph 0, overflow 0, every figure mapped.

- [ ] **Step 6: Commit each completed paper separately**

```bash
git add "translations-kr/stc-core/$paper_id" "build/publications/translations-kr/$paper_id.pdf"
git commit -m "docs: translate $paper_id into Korean"
```

### Task 5: Run corpus-level translation audit and package

**Files:**
- Create: `translations-kr/stc-core/reports/structural-parity.json`
- Create: `translations-kr/stc-core/reports/source-rights.json`
- Create: `translations-kr/stc-core/reports/reverse-check.md`
- Create: `build/publications/translations-kr/SHA256SUMS`
- Modify: `translations-kr/README.md`

**Interfaces:**
- Consumes: all selected translations
- Produces: complete internal/public distribution matrix and reproducible hashes

- [ ] **Step 1: Run strict corpus QA**

```bash
python3 translations-kr/stc-core/qa_translations.py --all --strict --render-all-pages
```

- [ ] **Step 2: Reverse-check load-bearing passages**

For every selected paper, compare abstract, method definition, primary result, limitations, and conclusion against the source. Record source page/line and reviewer verdict in `reverse-check.md`.

- [ ] **Step 3: Generate rights-aware output index**

Mark each PDF `public`, `internal-only`, or `metadata-only`; do not publish restricted artifacts as public deliverables.

- [ ] **Step 4: Generate hashes and update README**

```bash
(cd build/publications/translations-kr && sha256sum *.pdf > SHA256SUMS)
```

- [ ] **Step 5: Commit**

```bash
git add translations-kr build/publications/translations-kr
git commit -m "docs: validate Korean sleep-time translation corpus"
```
