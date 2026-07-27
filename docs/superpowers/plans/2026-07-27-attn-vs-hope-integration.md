# Attn-vs-HOPE Integration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn the approved one-block Full-Attention+MoE versus HOPE design, the live `Attn vs HOPE` Google Sheet, and the Google-meeting question package into one reproducible, discoverable repository deliverable.

**Architecture:** A standard-library Python engine is the canonical numerical source and emits deterministic JSON/report outputs. A Node builder consumes those outputs and creates the planned 11-tab formula-driven workbook, while the user's live one-tab Sheet is retained separately as a versioned reference export with a manifest. Small repository indexes and answer-capture templates connect the analytical model to the meeting workflow.

**Tech Stack:** Python 3.11+ standard library and `unittest`; Node.js ESM with `@oai/artifact-tool`; OOXML/XLSX; Markdown and CSV.

## Global Constraints

- The binding analytical requirements and equations are the complete design in `docs/superpowers/specs/2026-07-27-attn-vs-hope-analytical-model-design.md`.
- One MAC is two FLOPs; all reported latencies are analytical roofline lower bounds.
- Effective bytes count HBM-boundary traffic only. FlashAttention never materializes the quadratic score/probability tensor in HBM, but still reads Q/K/V and writes the final output.
- Decode is one next-token invocation: KV reads are linear in current context and formulas must not multiply by OSL.
- Shipped HOPE has static `W_q` and no `M_q`; any adaptive-q case is labeled hypothetical and cannot affect default results.
- HOPE uses the stable partition `Titans forward -> CMS forward -> loss -> backward+update`. Shipped Titans forward is exactly static `W_q` plus `M_mem`; auxiliary target/prediction work is loss preparation. CMS online backward/update is excluded.
- Full Attention prefill/TTFT is one full-ISL causal invocation. HOPE operation rows are one 2K chunk, and only HOPE TTFT multiplies 64 chunks. Decode/ITL is one `Q=1` token with no OSL multiplier.
- LRD is used only by CMS. Capacity 64/128/256 controls router/state residency; compute uses top-k=1 active rank-64 A/B experts and HBM reads use a unique-active upper bound. Update periods remain 1K/5K/10K.
- Reference timing uses user-reviewed `4.614e15 FLOP/s` and `7.4e12 B/s`; the H100 twin supplies capacity/provenance only.
- Preserve the live Google Sheet's existing named inputs and Korean cell notes. Never replace the reference export with a generated workbook.
- Do not stage or modify unrelated worktree files. Do not commit or push from a subagent.

---

### Task 1: Tested analytical engine and deterministic reference outputs

**Files:**
- Create: `experiments/E5-attn-vs-hope/model.py`
- Create: `experiments/E5-attn-vs-hope/test_model.py`
- Create: `experiments/E5-attn-vs-hope/run.py`
- Create: `experiments/E5-attn-vs-hope/results.json`
- Create: `experiments/E5-attn-vs-hope/stdout.txt`
- Create: `experiments/E5-attn-vs-hope/README.md`

**Interfaces:**
- `Hardware`, `AttentionMoEConfig`, `MemorySpec`, `CMSLevel`, `HopeConfig`, and `Stage` are frozen dataclasses.
- `Stage` exposes exactly the fields required by design section 3.1 and computed properties `effective_hbm_bytes`, `arithmetic_intensity`, `compute_seconds`, `memory_seconds`, and `roofline_seconds`.
- Public builders: `build_attention_prefill`, `build_attention_decode`, `build_hope_prefill`, and `build_hope_decode` return ordered `list[Stage]` values.
- Public summaries: `summarize(stages, hardware)`, `summarize_by_group(stages, hardware)`, and `legacy_hope_proxy(d, chunk, element_bytes)` return JSON-serializable dictionaries. Group summaries expose Titans forward, CMS forward, forward total, loss, and backward+update without synthetic stages.
- `run.py` writes deterministic `results.json` and byte-for-byte stable `stdout.txt` from repository-relative inputs.

- [ ] **Step 1: Write failing tests for primitive invariants**

Add `unittest` cases that directly prove:

```python
def test_causal_pairs(self):
    self.assertEqual(causal_pair_count(batch=2, heads=4, query_tokens=3), 48)

def test_kv_traffic_counts_k_and_v(self):
    cfg = self.attention_config(query_tokens=1, context_tokens=8, kv_heads=2, head_dim=4)
    stages = build_attention_decode(cfg, self.hardware)
    flash = next(stage for stage in stages if stage.stage == "flash_decode_partials")
    self.assertEqual(flash.hbm_read_bytes, 2 * cfg.batch * 8 * 2 * 4 * cfg.kv_bytes)

def test_flash_attention_has_no_quadratic_write(self):
    flash = next(stage for stage in build_attention_prefill(self.prefill, self.hardware)
                 if stage.stage == "flash_attention")
    self.assertEqual(flash.temporary_write_bytes, 0)
    self.assertGreater(flash.mandatory_write_bytes, 0)

def test_flash_decode_partial_traffic_depends_on_splits(self):
    one = self.attention_config(decode_splits=1)
    many = self.attention_config(decode_splits=8)
    self.assertEqual(decode_partial_bytes(one), 0)
    self.assertGreater(decode_partial_bytes(many), 0)
```

- [ ] **Step 2: Run the primitive tests and verify RED**

Run: `python -m unittest -v experiments/E5-attn-vs-hope/test_model.py`

Expected: import failure for the missing `model.py` API, not a syntax or path error.

- [ ] **Step 3: Implement the dataclasses and primitive cost functions**

Implement the exact equations in design sections 4–6 with input validation for positive dimensions, byte widths, hardware rates, and efficiencies in `(0, 1]`. Model FlashDecode partial output/LSE traffic only when `decode_splits > 1`; never add an `ISL**2` HBM term.

- [ ] **Step 4: Run the primitive tests and verify GREEN**

Run: `python -m unittest -v experiments/E5-attn-vs-hope/test_model.py`

Expected: the primitive cases pass.

- [ ] **Step 5: Add failing tests for stage builders and totals**

Add cases for linear decode KV reads, context-independent KV append, unique-active-expert weight bytes, context-independent and batch-linear HOPE state, CMS forward cadence independent of update cadence, stagewise latency not below aggregate latency, shipped default absence of `M_q`, and the legacy closed form:

```python
def test_legacy_anchor(self):
    got = legacy_hope_proxy(d=2048, chunk=1, element_bytes=2)
    self.assertEqual(got["flops"], 1_082_187_776)
    self.assertEqual(got["hbm_bytes"], 679_510_016)

def test_stagewise_not_below_aggregate(self):
    total = summarize(build_attention_prefill(self.prefill, self.hardware), self.hardware)
    self.assertGreaterEqual(total["stagewise_latency_seconds"],
                            total["aggregate_latency_seconds"])
```

- [ ] **Step 6: Run the builder tests and verify RED**

Run the same `unittest` command and confirm failures identify missing builders or incorrect relationships.

- [ ] **Step 7: Implement Attention+MoE and HOPE builders**

Keep stages explicit enough to audit fusion and traffic. Decode retains normal/boundary/amortized event diagnostics but exposes deployment metrics as forward-only, online-overhead, and including-online ITL; CMS reads occur every token while update periods remain metadata. Include four scenarios: `paper_equation_lower_bound`, `shipped_credible_momentum`, `legacy_repo_proxy`, and `adaptive_q_hypothetical`, with the last one excluded from defaults.

- [ ] **Step 8: Run all E5 tests and verify GREEN**

Run: `python -m unittest -v experiments/E5-attn-vs-hope/test_model.py`

Expected: all cases pass with no warning or skipped test.

- [ ] **Step 9: Add the deterministic runner and verify its outputs**

`run.py` must use live-Sheet peak/BW authority plus the H100 twin capacity/provenance, write sorted/indented JSON, include all stage records/group summaries/TTFT/ITL metrics and QA checks, and print the same compact report to stdout and `stdout.txt`. Context/batch rows must carry explicit HOPE forward-primary and including-online-sensitivity decode seconds and flags; first results are labeled as satisfying grid points rather than interpolated thresholds.

Run twice:

```bash
python experiments/E5-attn-vs-hope/run.py
sha256sum experiments/E5-attn-vs-hope/results.json experiments/E5-attn-vs-hope/stdout.txt
python experiments/E5-attn-vs-hope/run.py
sha256sum experiments/E5-attn-vs-hope/results.json experiments/E5-attn-vs-hope/stdout.txt
```

Expected: both checksum pairs match.

- [ ] **Step 10: Document assumptions and reproduction**

The README must separate paper facts, implementation choices, and sensitivity parameters; explain sparse CMS, dependency groups, TTFT/ITL, mixed hardware authority, and stagewise versus aggregate roofline; link the design, H100 twin, legacy baseline, live-sheet archive, and Google-meeting package; and list the exact commands above.

---

### Task 2: Formula-driven 11-tab workbook builder and generated workbook

**Files:**
- Create: `experiments/E5-attn-vs-hope/sheet/build_workbook.mjs`
- Create: `experiments/E5-attn-vs-hope/sheet/test_workbook_source.py`
- Create: `experiments/E5-attn-vs-hope/sheet/FORMULA-MAP.md`
- Create: `experiments/E5-attn-vs-hope/sheet/Attn-vs-HOPE.xlsx`
- Create: `experiments/E5-attn-vs-hope/DESKTOP-HANDOFF.md`

**Interfaces:**
- Builder input: `../results.json` resolved from `import.meta.url`.
- Builder output: `sheet/Attn-vs-HOPE.xlsx` plus preview PNGs under an ignored `sheet/previews/` directory.
- Sheet names are exactly `00_Guide`, `01_Inputs`, `02_HW`, `10_AttnMoE_Prefill`, `11_AttnMoE_Decode`, `20_HOPE_Prefill`, `21_HOPE_Decode`, `30_Compare`, `40_Sweeps`, `90_QA`, and `99_Sources`.
- Every calculation cell is a formula referencing `01_Inputs` or `02_HW`; imported numerical results may appear only in clearly labeled QA/reference columns.

- [ ] **Step 1: Write failing structural tests**

The Python test opens the generated XLSX as ZIP/XML and asserts all 11 sheet names, no external workbook links, at least one formula in every calculation/compare/sweep sheet, quoted cross-sheet references, three charts or drawings, and no literal H100/model constants in calculation formulas.

```python
EXPECTED_SHEETS = ["00_Guide", "01_Inputs", "02_HW", "10_AttnMoE_Prefill",
                   "11_AttnMoE_Decode", "20_HOPE_Prefill", "21_HOPE_Decode",
                   "30_Compare", "40_Sweeps", "90_QA", "99_Sources"]
```

- [ ] **Step 2: Run the workbook tests and verify RED**

Run: `python -m unittest -v experiments/E5-attn-vs-hope/sheet/test_workbook_source.py`

Expected: failure because the builder/output is missing.

- [ ] **Step 3: Implement the builder**

Use only `@oai/artifact-tool`. Build compact input tables, the design section 8 stage columns, formula-driven summaries, HBM-fit checks, five formula-only HOPE dependency-group subtotals, explicit full-ISL TTFT and Q=1 ITL comparisons, and context/batch ITL sweeps. The primary prefill comparison scales every HOPE flow/latency metric to 64×2K TTFT while leaving resident state unscaled. Serving sweeps use HOPE forward ITL as primary and show including-online ITL separately. Add only the three approved charts: full-invocation latency diagnostics, roofline scatter, and context-length forward/sensitivity ITL. Use a restrained navy/blue/gray research style and preserve units in headers.

- [ ] **Step 4: Generate and render the workbook**

Run with the bundled runtime module path, then render each sheet. The builder must fail with an actionable message if `@oai/artifact-tool` cannot be imported. Visually inspect all sheets and correct clipping, formula errors, overlap, and unreadable number formats.

- [ ] **Step 5: Run structural tests and verify GREEN**

Run the test command from Step 2. Expected: all cases pass.

- [ ] **Step 6: Write formula lineage and Desktop handoff**

`FORMULA-MAP.md` maps every workbook section to design equations and source cells. `DESKTOP-HANDOFF.md` contains exact builder, inspect, render, export, Google-Sheets import, and post-import formula-error checks; it distinguishes this generated 11-tab workbook from the user's native one-tab live Sheet.

---

### Task 3: Live-Sheet archive, meeting capture templates, and repository navigation

**Files:**
- Preserve: `research/attn-vs-hope/Attn-vs-HOPE.xlsx`
- Create: `research/attn-vs-hope/README.md`
- Create: `research/attn-vs-hope/EXPORT-MANIFEST.json`
- Create: `research/attn-vs-hope/verify_snapshot.py`
- Create: `research/google-meeting/README.md`
- Create: `research/google-meeting/04-ANSWER-CAPTURE-LIVE.csv`
- Create: `research/google-meeting/05-ANSWER-CAPTURE-POSTMEETING.csv`
- Create: `research/google-meeting/EXTERNAL-ARTIFACTS.md`
- Create: `research/README.md`
- Modify: `research/google-meeting/02-GOOGLE-KICK-QUESTION-PLAYBOOK.md`
- Modify: `research/google-meeting/03-DEVICE-SOLUTION-BRIDGE.md`
- Modify: `docs/superpowers/specs/2026-07-27-attn-vs-hope-analytical-model-design.md`

**Interfaces:**
- Manifest records live Sheet ID `1BZLzsGgdE43GXAMtuhrg7btPRq94crQWjo5mb0VWNWk`, URL, update/export timestamps, SHA-256, sheet/range, counts, key outputs, and corrected cells.
- Snapshot verifier uses only the Python standard library and returns nonzero for checksum drift, missing named ranges, wrong formulas in corrected bound cells, stale `현재 64` note text, cached error values, or external workbook links.
- Both CSVs use UTF-8 and stable IDs from the playbook. Live columns: `question_id,axis,short_question,google_answer,evidence_or_metric,follow_up,device_hook,owner,status`. Post-meeting columns: `question_id,claim,evidence,confidence,implication,device_requirement,next_experiment,owner,due,status`.

- [ ] **Step 1: Write the failing snapshot verifier checks**

Tests embedded in `verify_snapshot.py --self-test` must corrupt one in-memory manifest value and one in-memory workbook relationship fixture and prove both are rejected before validating the real file.

- [ ] **Step 2: Run the verifier and establish RED**

Run: `python research/attn-vs-hope/verify_snapshot.py --self-test`

Expected: fail until manifest validation and OOXML checks exist.

- [ ] **Step 3: Implement verification and generate the manifest**

Validate one visible sheet `시트1`, used range `A1:P89`, 38 defined names, 872 formulas, 117 Korean notes/comments, zero cached formula errors, and zero external links. Record live validation of 872 formulas and zero effective errors separately from portable XLSX checks.

- [ ] **Step 4: Run the verifier and verify GREEN**

Run:

```bash
python research/attn-vs-hope/verify_snapshot.py --self-test
python research/attn-vs-hope/verify_snapshot.py
```

Expected: both exit zero and print the same workbook checksum as the manifest.

- [ ] **Step 5: Add meeting indexes and capture templates**

The meeting README gives reading order, six-axis coverage, KQ/BQ/DBQ ID counts, paper-version provenance, and links to E5/live-Sheet artifacts. Populate the live CSV with every KQ and BQ ID and the post-meeting CSV with every KQ ID; leave answer fields empty. Record the external MEMOIR locator `simulation-workbench@638d234` as unresolved provenance rather than inventing a URL.

- [ ] **Step 6: Add precise backlinks**

Link KQ-03 and DBQ-09/10 to E5 results and the live Sheet archive. Link the analytical design back to the meeting package and document that the native Sheet is the user-reviewed operational model while the generated 11-tab workbook is the reproducible analytical companion.

- [ ] **Step 7: Validate repository navigation**

Run a script that resolves every repository-relative Markdown link in the three package indexes and checks CSV header/ID uniqueness. Expected: zero missing targets and zero duplicate IDs.

---

## Final integration gate

- [ ] Run E5 Python tests, workbook structural tests, live snapshot verification, sleep-time-compute's full `uv run pytest -q`, Markdown-link validation, `git diff --check`, and an explicit staged-file audit.
- [ ] Confirm the live Google Sheet still has 872 formulas, 117 Korean notes, and zero effective formula errors.
- [ ] Confirm the generated workbook and the live export are different paths with different documented purposes.
- [ ] Commit once with an intentional file list and push `codex/stc-research-program`; verify the remote branch SHA equals local HEAD.
