# STC Study Release Audit Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Research, papers, translations, easy companions, and seminar deck을 하나의 검증 가능한 release로 묶고 Veridraft·artifact QA·repository tests·hashes·indexes를 통과시킨다.

**Architecture:** `STC-STUDY-RELEASE-MANIFEST.json`이 모든 source/output/report hash와 audience boundary를 기록한다. Release checker는 artifact별 QA report를 읽어 실패를 집계하고, 모든 필수 gate가 통과한 경우에만 `release_status: ready`를 쓴다.

**Tech Stack:** Python, JSON, SHA-256, Veridraft, pytest, Git.

## Global Constraints

- Restricted or internal-only translation/figure를 public manifest에 public artifact로 표시하지 않는다.
- Existing untracked seminar assets are preserved and excluded from commits.
- A green test exit code and exact pass count are recorded; completion is never inferred.
- Feature branch and remote SHAs must be verified before and after main integration.

---

### Task 1: Add release manifest generator and tests

**Files:**
- Create: `research/sleep-time-compute/program/build_release_manifest.py`
- Create: `research/sleep-time-compute/tests/test_release_manifest.py`
- Create: `build/publications/STC-STUDY-RELEASE-MANIFEST.json`

**Interfaces:**
- Consumes: artifact files and QA reports
- Produces: `build_manifest(root: Path) -> dict`

- [ ] **Step 1: Write failing release tests**

```python
def test_release_fails_when_required_artifact_is_missing(tmp_path):
    report = build_manifest(tmp_path)
    assert report["release_status"] == "blocked"
    assert "missing-study-pdf" in report["blockers"]

def test_public_manifest_excludes_internal_translation(tmp_path):
    report = fixture_release(tmp_path, translation_scope="internal-only")
    assert report["public_artifacts"]["translations"] == []
```

- [ ] **Step 2: Run tests and confirm failure**

```bash
uv run --project research/sleep-time-compute pytest -q research/sleep-time-compute/tests/test_release_manifest.py
```

- [ ] **Step 3: Implement deterministic manifest generation**

Record relative path, SHA-256, bytes, pages/slides, source commit, audience, license state, builder, QA report, and pass status. Sort paths and serialize with stable indentation.

- [ ] **Step 4: Run tests and commit**

```bash
uv run --project research/sleep-time-compute pytest -q research/sleep-time-compute/tests/test_release_manifest.py
git add research/sleep-time-compute/program/build_release_manifest.py research/sleep-time-compute/tests/test_release_manifest.py
git commit -m "feat: add STC study release manifest"
```

### Task 2: Update artifact indexes and reader routes

**Files:**
- Modify: `research/sleep-time-compute/README.md`
- Modify: `research/sleep-time-compute/DELIVERABLES.md`
- Modify: `translations-kr/README.md`
- Modify: `easy/README.md`
- Create: `presentation/sleep-time-compute-deep-study/README.md`
- Modify or create: repository root `README.md`

**Interfaces:**
- Consumes: final artifact paths and audience boundaries
- Produces: one-click paths for Study, Background, Translations, Easy, Deck, Evidence, and QA

- [ ] **Step 1: Write audience-specific entry points**

Provide paths for academic reviewer, training beginner, seminar participant, translation reader, system/device researcher, and reproduction/QA reviewer.

- [ ] **Step 2: State the evidence and rights boundaries**

Explicitly distinguish public paper/deck from internal-only translations or figures and describe the 2026-08-05 source freeze.

- [ ] **Step 3: Add exact reproduction commands**

Include paper, background, translation, easy, deck, Veridraft, and release-manifest commands with expected outputs.

- [ ] **Step 4: Validate links**

Run a Markdown link checker over modified README/DELIVERABLES and ensure every local target exists.

- [ ] **Step 5: Commit**

```bash
git add README.md research/sleep-time-compute/README.md research/sleep-time-compute/DELIVERABLES.md translations-kr/README.md easy/README.md presentation/sleep-time-compute-deep-study/README.md
git commit -m "docs: index sleep-time compute study release"
```

### Task 3: Run Veridraft release gates

**Files:**
- Create: `claims/stc-study/reports/final-gate.txt`
- Create: `claims/stc-study/reports/readiness-mlsys.txt`
- Create: `claims/stc-study/reports/publish-public.txt`
- Create: `claims/stc-study/reports/events.txt`

**Interfaces:**
- Consumes: final Study PDF and claim bundle
- Produces: deterministic claim/readiness/egress audit

- [ ] **Step 1: Re-import and gate final bundle**

```bash
python3 -m veridraft --data-dir .veridraft-stc-study import-bundle claims/stc-study/bundle.json
python3 -m veridraft --data-dir .veridraft-stc-study gate sleep-time-compute-strategic-study-2026 | tee claims/stc-study/reports/final-gate.txt
```

- [ ] **Step 2: Run readiness and public egress**

```bash
python3 -m veridraft --data-dir .veridraft-stc-study readiness sleep-time-compute-strategic-study-2026 --venue mlsys | tee claims/stc-study/reports/readiness-mlsys-final.txt
python3 -m veridraft --data-dir .veridraft-stc-study publish sleep-time-compute-strategic-study-2026 --audience public --venue mlsys | tee claims/stc-study/reports/publish-public.txt
python3 -m veridraft --data-dir .veridraft-stc-study events | tee claims/stc-study/reports/events-final.txt
```

Expected: no blocked public claims, no confidentiality leak, event chain valid. Venue AI-policy text may require human sign-off and is recorded rather than falsely marked approved.

- [ ] **Step 3: Commit reports**

```bash
git add claims/stc-study/reports
git commit -m "audit: gate sleep-time study publication"
```

### Task 4: Run the complete technical and artifact QA suite

**Files:**
- Create: `build/publications/FINAL-QA.json`
- Update: `build/publications/SHA256SUMS`

**Interfaces:**
- Consumes: every artifact and test suite
- Produces: exact commands, exit codes, pass/fail counts, duration, report hashes

- [ ] **Step 1: Run Python and analytical tests**

```bash
uv run --project research/sleep-time-compute pytest -q research/sleep-time-compute/tests
python3 -m unittest -q experiments/E5-attn-vs-hope/test_model.py
python3 -m unittest -q experiments/E5-attn-vs-hope/sheet/test_workbook_source.py
python3 -m pytest -q paper-kr/common/tests translations-kr/stc-core/tests easy/sleep-time-compute/tests
```

- [ ] **Step 2: Run publication QA**

```bash
python3 paper-kr/common/build_publications.py --root . --target all --clean
python3 paper-kr/common/qa_publications.py --pdf build/publications/TRAINING-BACKGROUND-FOR-SLEEP-TIME-COMPUTE-KR.pdf --kind background --require-all-concepts
python3 paper-kr/common/qa_publications.py --pdf build/publications/SLEEP-TIME-COMPUTE-STRATEGIC-STUDY-KR.pdf --kind study --claim-map claims/stc-study/claim-map.json --figure-ledger claims/stc-study/figure-ledger.json
python3 paper-kr/common/qa_publications.py --pdf build/publications/SLEEP-TIME-COMPUTE-CONFERENCE-KR.pdf --kind conference --min-pages 12 --max-pages 18
python3 paper-kr/common/qa_publications.py --pdf build/publications/SLEEP-TIME-COMPUTE-CONFERENCE-APPENDIX-KR.pdf --kind appendix
python3 translations-kr/stc-core/qa_translations.py --all --strict --render-all-pages
python3 easy/sleep-time-compute/qa.py --all --combined --strict --render-all-pages
```

- [ ] **Step 3: Run presentation QA**

```bash
node --test presentation/sleep-time-compute-deep/tests/content.test.mjs
python3 presentation/sleep-time-compute-deep/tests/verify_deck_release.py
PYTHONPATH=build/deck-work/python-deps python3 "$SKILL_DIR/container_tools/slides_test.py" build/publications/SLEEP-TIME-COMPUTE-DEEP-SEMINAR-112SLIDES-KR.pptx --width 1280 --height 720
```

- [ ] **Step 4: Run program and link QA**

```bash
python3 research/sleep-time-compute/program/validate_program.py --all
python3 research/sleep-time-compute/program/build_release_manifest.py
```

- [ ] **Step 5: Generate hashes**

Generate `build/publications/SHA256SUMS` for every final PDF/PPTX/manifest and verify with `sha256sum -c`.

- [ ] **Step 6: Verify no placeholders or broken references**

```bash
python3 research/sleep-time-compute/program/validate_program.py --all --reject-authoring-markers --reject-latex-reference-errors
```

Expected: no authoring placeholders or LaTeX reference failures.

### Task 5: Build final manifest and verify repository scope

**Files:**
- Create or update: `build/publications/STC-STUDY-RELEASE-MANIFEST.json`
- Create: `build/publications/FINAL-QA.json`

**Interfaces:**
- Consumes: Task 4 results
- Produces: `release_status: ready`

- [ ] **Step 1: Build final manifest**

```bash
python3 research/sleep-time-compute/program/build_release_manifest.py --require-ready
```

- [ ] **Step 2: Verify scope and whitespace**

```bash
git diff --check main...HEAD
git diff --name-only main...HEAD
```

Expected changed paths are limited to approved specs/plans, STC research/program, claims, paper-kr, selected translation/easy paths, presentation deep-study, build/publications, and indexes.

- [ ] **Step 3: Verify user files remain untouched**

Compare the main worktree's 52 pre-existing untracked paths and hashes against the pre-merge snapshot. Overlap with branch changes must be zero.

- [ ] **Step 4: Commit final manifest**

```bash
git add build/publications/STC-STUDY-RELEASE-MANIFEST.json build/publications/FINAL-QA.json build/publications/SHA256SUMS
git commit -m "release: finalize sleep-time compute study artifacts"
```

### Task 6: Finish the branch and publish

**Files:**
- Verify only; no new content paths

**Interfaces:**
- Consumes: release-ready feature branch
- Produces: remote feature branch and fast-forwarded remote `main`

- [ ] **Step 1: Invoke `superpowers:verification-before-completion` and rerun its required fresh checks**
- [ ] **Step 2: Invoke `superpowers:finishing-a-development-branch`**
- [ ] **Step 3: Push `codex/stc-study-rebuild` and verify local/remote SHA**
- [ ] **Step 4: In the main worktree, fetch, verify tracked clean, verify zero untracked overlap, and fast-forward merge**
- [ ] **Step 5: Run smoke QA in main and push `main`**
- [ ] **Step 6: Verify local main SHA equals `origin/main` and report test counts, artifact paths, commit SHA, and preserved untracked count**
