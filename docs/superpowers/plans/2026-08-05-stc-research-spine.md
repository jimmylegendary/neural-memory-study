# STC Research Spine Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 기존 corpus와 2026-08-05 최신 1차 출처를 재조사해 STC의 타당성, 경쟁 접근법, 산업·학계 방향, candidate scaling law, device opportunity를 지지하거나 반박하는 canonical evidence spine을 동결한다.

**Architecture:** `pre-research/`는 질문과 탐색 경계를, `deep-research/`는 source/citation/claim/figure/translation selection을 보유한다. Python validator와 Veridraft bundle은 모든 load-bearing claim이 고정된 source와 locator를 갖고 evidence type과 공개 경계가 명시됐는지 검증한다.

**Tech Stack:** Python, JSON/JSONL, pytest, Veridraft, official web and paper sources, arXiv/DOI metadata.

## Global Constraints

- Source freeze: `2026-08-05`.
- Technical claims use primary sources; secondary sources are discovery-only.
- Supportive and disconfirming searches are separate required passes.
- Product deployment claims require official product docs, release notes, or repository evidence.
- No full-text paper is committed without an authoritative license record.

---

### Task 1: Add program registry schemas and validator

**Files:**
- Create: `research/sleep-time-compute/program/schema/research-spine.schema.json`
- Create: `research/sleep-time-compute/program/schema/source-record.schema.json`
- Create: `research/sleep-time-compute/program/schema/claim-record.schema.json`
- Create: `research/sleep-time-compute/program/schema/figure-record.schema.json`
- Create: `research/sleep-time-compute/program/schema/translation-selection.schema.json`
- Create: `research/sleep-time-compute/program/validate_program.py`
- Create: `research/sleep-time-compute/tests/test_program_validation.py`

**Interfaces:**
- Consumes: JSON files under `pre-research/` and `deep-research/`
- Produces: `validate_program(root: Path) -> ProgramValidationReport`

- [x] **Step 1: Write failing schema-validation tests**

```python
def test_research_spine_requires_frozen_sources_and_locators(tmp_path):
    root = write_program_fixture(tmp_path, source_status="candidate", locator="")
    report = validate_program(root)
    assert not report.success
    assert {d.code for d in report.diagnostics} >= {
        "source-not-frozen", "claim-missing-locator"
    }

def test_translation_selection_is_between_twelve_and_fifteen(tmp_path):
    root = write_program_fixture(tmp_path, translation_count=11)
    report = validate_program(root)
    assert "translation-count-out-of-range" in {d.code for d in report.diagnostics}
```

- [x] **Step 2: Run tests and confirm failure**

```bash
uv run --project research/sleep-time-compute pytest -q research/sleep-time-compute/tests/test_program_validation.py
```

Expected: import or assertion failure because validator and schemas do not exist.

- [x] **Step 3: Implement strict schemas and validator**

The validator must reject duplicate IDs, non-ISO dates, mutable URLs without version metadata, missing locator, unsupported claim type, public direct-reuse figures without license evidence, translation count outside 12–15, and a frozen spine with unresolved load-bearing claims.

- [x] **Step 4: Run focused and existing tests**

```bash
uv run --project research/sleep-time-compute pytest -q research/sleep-time-compute/tests/test_program_validation.py
uv run --project research/sleep-time-compute pytest -q research/sleep-time-compute/tests
```

Expected: focused tests pass; existing 736 tests remain green.

- [x] **Step 5: Commit**

```bash
git add research/sleep-time-compute/program research/sleep-time-compute/tests/test_program_validation.py
git commit -m "feat: add sleep-time research program validator"
```

### Task 2: Write pre-research alignment package

**Files:**
- Create: `research/sleep-time-compute/pre-research/ALIGNMENT.md`
- Create: `research/sleep-time-compute/pre-research/QUESTION-TREE.md`
- Create: `research/sleep-time-compute/pre-research/APPROACH-TAXONOMY.md`
- Create: `research/sleep-time-compute/pre-research/UNKNOWN-UNKNOWN-REGISTER.md`
- Create: `research/sleep-time-compute/pre-research/SEARCH-AND-SATURATION-PLAN.md`
- Create: `research/sleep-time-compute/pre-research/SEED-CORPUS.json`
- Create: `research/sleep-time-compute/pre-research/TRANSLATION-CANDIDATES.md`
- Create: `research/sleep-time-compute/pre-research/FIGURE-SOURCE-PLAN.md`

**Interfaces:**
- Consumes: existing audits, monograph, scaling agenda, Google meeting evidence
- Produces: hypothesis set `H-STC`, `H-EXT`, `H-HYBRID`, `H-REFRESH`, `H-NICHE`; seed source IDs; search clusters

- [x] **Step 1: Build the objective-to-question matrix**

`ALIGNMENT.md` must contain the user's eight decision questions, success criteria, intended audience, explicit exclusions, and the evidence required to answer each question.

- [x] **Step 2: Build the approach taxonomy**

`APPROACH-TAXONOMY.md` must compare external retrieval, long context/state, TTT/fast weights, continual-learning families, model editing, periodic refresh, explicit sleep consolidation, and hybrid promotion across memory medium, update timing, trainable state, data, capacity behavior, cost, governance, and strongest known result.

- [x] **Step 3: Build unknown-unknown and disconfirmation registers**

Every row must include `unknown_id`, why ordinary search misses it, discovery query/family, evidence needed, owner artifact, and closure condition. Required adjacent fields: database compaction, cache lifecycle, on-device adaptation, continual robotics, federated personalization, autonomous-agent experience learning, knowledge editing/unlearning.

- [x] **Step 4: Populate the seed corpus**

`SEED-CORPUS.json` must contain fixed IDs, title, authors, year, venue/status, DOI/arXiv/version, source URL, local artifact path if rights permit, cluster labels, supportive/disconfirming role, and translation candidacy.

- [x] **Step 5: Validate package**

```bash
python3 research/sleep-time-compute/program/validate_program.py --phase pre-research
python3 research/sleep-time-compute/program/validate_program.py --phase pre-research --reject-authoring-markers
```

Expected: validator passes; `rg` returns no matches.

- [x] **Step 6: Commit**

```bash
git add research/sleep-time-compute/pre-research
git commit -m "research: align sleep-time compute pre-research"
```

### Task 3: Re-run the deep survey and latest company/product audit

**Files:**
- Create: `research/sleep-time-compute/deep-research/SOURCE-REGISTRY.json`
- Create: `research/sleep-time-compute/deep-research/CITATION-POOL.json`
- Create: `research/sleep-time-compute/deep-research/SATURATION.json`
- Create: `research/sleep-time-compute/deep-research/LATEST-INDUSTRY-ACADEMIA-AUDIT.md`
- Create: `research/sleep-time-compute/deep-research/CHRONOLOGY.md`
- Create: `research/sleep-time-compute/deep-research/NEGATIVE-EVIDENCE.md`
- Create: `research/sleep-time-compute/deep-research/SOURCE-RIGHTS.json`

**Interfaces:**
- Consumes: `SEED-CORPUS.json`, official web sources, citation graph
- Produces: verified sources with fixed versions and cluster saturation metrics

- [x] **Step 1: Verify Veridraft and deep-survey prerequisites**

```bash
python3 -m veridraft --version
python3 -m veridraft status --config veridraft.stc-study.config.json
```

Expected: Veridraft `0.1.0`; configuration resolves. If Semantic Scholar access is unavailable, record `s2_unavailable` and run the documented DOI/arXiv/manual citation fallback without fabricating saturation.

- [x] **Step 2: Run forward/backward snowballing per cluster**

Use the Veridraft deep-survey scripts from `/home/jimmy/repos/veridraft/skills/literature-review-agent/scripts/` with `--max-rounds 4`, then verify every retained citation against DOI, arXiv, proceedings, or official repository metadata.

- [x] **Step 3: Audit 2025–2026 company and product lineages**

Required entities: Google/DeepMind, Meta, Microsoft, OpenAI, Letta/MemGPT, Mem0, Zep. Record separately: research mechanism, public implementation, product feature, deployment evidence, missing evidence, and last verified date.

- [x] **Step 4: Run negative-evidence search**

Search for collapse, catastrophic forgetting, recursive self-distillation degradation, replay scaling cost, data poisoning, stale memory, deletion failure, benchmark leakage, rollback, capacity exhaustion, and operational incidents. Each negative claim needs a fixed locator.

- [x] **Step 5: Freeze citation and source-rights registries**

Direct-reuse figures require `license_url`, `license_type`, `reuse_scope`, and source hash. Full-text artifacts without authoritative permission remain external references rather than committed copies.

- [x] **Step 6: Verify saturation and commit**

```bash
python3 research/sleep-time-compute/program/validate_program.py --phase deep-research
git add research/sleep-time-compute/deep-research
git commit -m "research: refresh sleep-time compute evidence through 2026-08-05"
```

### Task 4: Build the strategic comparison and conditional verdict

**Files:**
- Create: `research/sleep-time-compute/deep-research/PROBLEM-SOLUTION-MATRIX.md`
- Create: `research/sleep-time-compute/deep-research/STRONGEST-ALTERNATIVES.md`
- Create: `research/sleep-time-compute/deep-research/PROMISINGNESS-ASSESSMENT.md`
- Create: `research/sleep-time-compute/deep-research/MAINSTREAM-SCENARIOS.md`
- Create: `research/sleep-time-compute/deep-research/SCALING-LAW-HYPOTHESES.md`
- Create: `research/sleep-time-compute/deep-research/DEVICE-OPPORTUNITY-MATRIX.md`

**Interfaces:**
- Consumes: frozen citation pool and source registry
- Produces: paper-ready comparison rows and conditional conclusions

- [x] **Step 1: Compare each problem against its strongest non-STC solution**

Required problem rows: static deployment, personalization, agent experience, knowledge freshness, continual learning, bounded capacity, latency isolation, deletion/rollback, long-context reasoning, on-device adaptation.

- [x] **Step 2: Score evidence maturity separately from expected value**

Use ordinal fields with explicit rubrics: `problem_severity`, `quality_advantage`, `cost_advantage`, `evidence_maturity`, `deployment_fit`, `governance`, `industry_momentum`, `academic_momentum`. Do not average them into an unsupported single magic number.

- [x] **Step 3: Write conditional mainstream scenarios**

Required scenarios: external-memory dominant, hybrid promotion dominant, periodic-refresh dominant, STC niche, STC broad adoption. Each scenario needs triggers, leading indicators, falsifiers, and device implications.

- [x] **Step 4: Define candidate scaling-law family**

Separate measured identities, analytical break-even equations, fit candidates, and untested hypotheses. Include sleep FLOPs, effective wake evidence, replay diversity, activated capacity, interference, state-migration bytes, reuse count, wake-SLA cost, validation cost, recursive-distillation depth.

- [x] **Step 5: Derive device opportunities from workload primitives**

Every opportunity must state whether value survives if STC does not become mainstream, plus capacity/bandwidth/endurance/latency/atomicity/security requirements.

- [x] **Step 6: Validate and commit**

```bash
python3 research/sleep-time-compute/program/validate_program.py --phase synthesis
git add research/sleep-time-compute/deep-research
git commit -m "research: synthesize sleep-time compute strategic verdict"
```

### Task 5: Generate Veridraft claim and figure ledgers

**Files:**
- Create: `veridraft.stc-study.config.json`
- Create: `claims/stc-study/bundle.json`
- Create: `claims/stc-study/claim-map.json`
- Create: `claims/stc-study/figure-ledger.json`
- Create: `claims/stc-study/translation-selection.json`
- Create: `claims/stc-study/reports/gate.txt`
- Create: `claims/stc-study/reports/saturation.json`
- Create: `research/sleep-time-compute/program/research-spine.json`
- Create: `research/sleep-time-compute/program/build_claim_bundle.py`
- Create: `research/sleep-time-compute/tests/test_stc_claim_bundle.py`

**Interfaces:**
- Consumes: frozen source/citation/synthesis files
- Produces: Veridraft bundle `sleep-time-compute-strategic-study-2026` and frozen spine

- [x] **Step 1: Write failing bundle-generation tests**

```python
def test_every_load_bearing_claim_has_primary_source_and_locator():
    bundle = build_bundle(FIXTURE_ROOT)
    for claim in bundle["claims"]:
        if claim["load_bearing"]:
            assert claim["evidence"]
            assert all(e["locator"] for e in claim["evidence"])

def test_device_projections_are_held_p3_claims():
    bundle = build_bundle(FIXTURE_ROOT)
    device = [c for c in bundle["claims"] if "device" in c["tags"]]
    assert device and all(c["type"] == "P3" for c in device)
```

- [x] **Step 2: Implement deterministic bundle generation**

P1 requires `result_refs`; P2 literature claims require fixed source artifacts; P3 device projections default to internal hold. The same input must produce byte-identical sorted JSON.

- [x] **Step 3: Select 12–15 translation papers**

The selection file must include exact paper IDs and roles, source path/hash, license state, figure count, equation/table count when extractable, and `existing_translation` status.

- [x] **Step 4: Run tests and Veridraft gates**

```bash
uv run --project research/sleep-time-compute pytest -q research/sleep-time-compute/tests/test_stc_claim_bundle.py
python3 research/sleep-time-compute/program/build_claim_bundle.py
python3 -m veridraft import-bundle claims/stc-study/bundle.json --data-dir .veridraft-stc-study
python3 -m veridraft gate sleep-time-compute-strategic-study-2026 --data-dir .veridraft-stc-study | tee claims/stc-study/reports/gate.txt
```

Expected: zero blocked public Study claims; P3 held claims are reported, not leaked.

- [x] **Step 5: Freeze the spine**

`research-spine.json` must record hashes for all registries and `status: "frozen"`.

- [x] **Step 6: Commit**

```bash
git add veridraft.stc-study.config.json claims/stc-study research/sleep-time-compute/program research/sleep-time-compute/tests/test_stc_claim_bundle.py
git commit -m "research: gate sleep-time compute study claims"
```
