# Veridraft tooling — findings + integration plan for the neural-memory study

Inspected: `/home/jimmy/repos/veridraft` on 2026-07-11.

## 1. Current state

- **Version**: `0.1.0` (pyproject), HEAD = `95b1286 release: v0.1.0`. Tests: **215 passed, 1 skipped** (`python3 -m pytest tests -q`). `verify.sh` = compile + suite + acceptance scenarios.
- Recent commits (newest first): v0.1.0 release; 14 adversarial-review hardening fixes; README/op-manifest wiring; **patent pre-draft prior-art memo**; **literature deep-survey mode (citation-graph snowballing to saturation)** — directly relevant here; reference-figure-extractor skill; patent-figure geometry gates; **preflight input-sufficiency gate + grounded auto-remediation**; **meta-info schema + forced ask-the-user gate before drafting**; evidence-gated marketing flow.
- Zero runtime deps (stdlib only); LLM engine (vendored PaperOrchestra under `skills/` + `paperorchestra/run.py`) is opt-in via config.

## 2. CLI entry points and gate set

Entry point: `veridraft` console script or `python3 -m veridraft` (`veridraft/cli.py`). Subcommands:

```
import-bundle <ref>          # ingest a bundle.json (claims+results ledger)
gate <bundle_id>             # run the evidence gate (authoritative, model-free)
assemble <bundle_id> --template T --guidelines G --title X --audience public|internal|counsel
draft <bundle_id>            # run the writing engine (minimal-latex default, paperorchestra opt-in)
review / reviews / venues    # review-readiness checklist + AI-review record validation
publish                      # egress: confidentiality decide() + redaction re-sweep over the PDF
readiness / sign-off-readiness  # venue LLM-policy -> disclosure text; human sign-off for AI-ban venues
status / events              # lifecycle board + hash-chained tamper-evident ledger
interlocks / release-interlock  # patent-first holds on P3 claims (human release only)
patentability / patent-prior-art / patent-idea / draft-patent / patent-review
adapters                     # list adapters + preflight
run <ref> --template --guidelines [--title --audience]   # full slice import->gate->...->publish
```

Gate layers (all in `veridraft/core/`):

- **`gate.py` — authoritative evidence gate.** Per-claim: evidence must be a typed, *resolvable* ref, fully anchored regex per kind:
  - `source_artifact`: `<path>@<7-40 hex commit>` OR `github://…`, `repo://…`, `https?://…`, `git://…`
  - `caw01_result`: `caw01://…`; `caw02_evidence`: `caw02://…`
  - `generated_text` / `prose_note` are **never** admissible (non-relaxable invariant).
  - Min-evidence floor per claim type from the gate profile (hard floor 1); `require_results` refuses to draft an evaluation from an empty/valueless result set; P3 (future-device) claims are disclosure-HELD for papers until `release-interlock`.
- **`lints.py` — advisory deterministic lints** (heuristic, tool gates): patent claim gates P1–P8 + P11 (PDF) + P12 (FTO/post-search); paper gates **D1** (every headline contribution → ≥1 VALIDATED datum), **D2** (venue-required experiments planned for marketed claims), **D4** (exploration-grade numbers print at ~2 sig figs / ratios, not 5–6); R\* review-record validation (stub/placeholder rejection); F\* figure gates. Aggregators: `paper_readiness()`, `patent_readiness()`.
- **Gate profiles** (`veridraft/profiles/`): `neurips-paper` (P1 needs a result_ref), `systems-paper` (result_ref optional for P1), `us-utility-patent`. Selected via `veridraft.config.json` → `{"gate_profile": "systems-paper"}`.
- Around drafting: PaperOrchestra runner adds **meta-info forced-ask gate** (exit 3 + `meta.questions.json` until venue/audience/anonymize are on record), **preflight input-sufficiency** gate with grounded auto-fill, schema-valid `outline.json` halt, S2-verified citations, orphan-citation/latex-sanity/anti-leakage gates, bounded refine loop, `provenance.json` hashes, render self-verify.

## 3. Initializing a claim manifest for the study

There is **no `init` subcommand** — the manifest is a hand-authored (or agent-authored) `bundle.json`. Canonical model to copy: `examples/taskops/bundle.json` (a real doc-sourced paper bundle) + `examples/taskops/caw03.config.json`. Shape:

```json
{
  "bundle_id": "neural-memory-study-2026",
  "boundary": "public",
  "provenance_manifest": { "exported_by": "manual", "topic": "...", "note": "..." },
  "claims": [
    {
      "claim_id": "c1", "type": "P1", "boundary": "public", "visibility": "team",
      "statement": "one precise, checkable assertion",
      "result_refs": ["r_scaling_1"],
      "evidence": [
        { "id": "c1e1", "kind": "source_artifact",
          "ref": "experiments/scaling/results.json@<commit>", "trust": 0.95 }
      ]
    }
  ],
  "results": [
    { "result_id": "r_scaling_1", "description": "...",
      "metrics": [ { "name": "...", "value": 1.4, "unit": "x" } ] }
  ]
}
```

- Claim types: **P1/P2** = method/tool claims (P1 ≈ quantitative/headline, P2 ≈ qualitative/integration); **P3** = future-device / patent-sensitive (paper-disclosure HELD by default — avoid unless intended).
- Recommended profile for this study: **`systems-paper`** (P1 doesn't force a result_ref, so faithful-summary claims can be P1 without fake numbers), or keep summary claims as P2 and reserve P1 for quantitative claims and use `neurips-paper` for the stricter floor. Suggest: **systems-paper + convention: every original quantitative claim carries `result_refs`** (D1 lint then enforces the datum mapping).
- Repo config, e.g. `/home/jimmy/repos/neural-memory-study/veridraft.config.json`:

```json
{
  "gate_profile": "systems-paper",
  "data_dir": ".veridraft",
  "adapters": {
    "writing_engine": { "id": "paperorchestra", "enabled": true, "config": {
      "po_command": ["python3", "/home/jimmy/repos/veridraft/paperorchestra/run.py",
                     "--config", "<repo>/po_backend.json", "--workspace", "{workspace}"] } }
  }
}
```

  (default `minimal-latex` engine works offline and needs nothing; the AI tier needs a `po_backend.json` — see `paperorchestra/config.example.json`. For this study set `"survey_depth": "deep"` and provide `s2_api_key`.)
- Precedent for a gate-manifest regeneration script: `/home/jimmy/repos/tiling-ir-survey/caw03/regenerate_manifests.py` and `.../caw03/veridraft-config.json` (paperorchestra + patent-llm wiring).

## 4. Integration workflow for the months-long 150–200p study

Reality check first: Veridraft's `draft` produces a *paper* (PaperOrchestra 5-agent pipeline or a deterministic skeleton), not a 200-page book. For this project use Veridraft primarily as the **claim ledger + deterministic gate + literature engine + egress governor** around chapter prose you (or chapter-wise engine runs) write. The gates are workspace/ledger-level, so this works cleanly.

### Repo layout

```
neural-memory-study/
  claims/bundle.json          # THE living claim ledger (single source of truth)
  veridraft.config.json       # profile + engine wiring
  po_backend.json             # LLM backend (deep survey_depth, s2 key)
  experiments/<name>/         # small scaling experiments: code + results.json (committed)
  latex/                      # the 150-200p manuscript (chapters)
  .veridraft/                 # harness data dir (imported bundles, ledger, events)
```

### When gates run

1. **On every ledger edit** (new/changed claim): `veridraft import-bundle claims/bundle.json && veridraft gate <bundle_id>`. Cheap (<1 s, offline). Make it a make target / pre-commit step; blocked claims list is the to-do list of missing warrants.
2. **Weekly / per-milestone**: run the D-lints over the current contribution list via a small `regenerate_manifest.py` (pattern above) calling `lints.map_contributions_to_evidence`, `lint_venue_experiments`, `lint_precision`, `paper_readiness()` — commit the resulting `manifest.paper.json` so gate status is versioned alongside the text.
3. **Per survey pass** (early months): run the **deep-survey** literature pipeline once the seed pool exists — `skills/literature-review-agent/scripts/snowball.py --seeds … --idea … --max-rounds 4` then `cluster_pool.py` → `expanded_pool.json` (S2-verified, cluster-labeled = your survey section skeleton) + `saturation.json` (coverage report — a *warrant that the survey is exhaustive*). Every citation in the manuscript should come from this verified pool; unverifiable citations get dropped/TODO by construction.
4. **Per chapter draft** (optional AI tier): `assemble` + `draft` on a per-chapter sub-bundle; assemble structurally excludes ungated claims, PO adds outline/citation/leakage/precision gates and `provenance.json`.
5. **At egress** (preprint/submission): `veridraft publish --audience public` (confidentiality decide() + redaction re-sweep over the actual PDF) and `veridraft readiness --venue <venue>` (LLM-policy disclosure text; hard-block + human sign-off if the venue prohibits AI text). `veridraft events` verifies the hash chain.

### Artifacts the gates need

- `claims/bundle.json` (claims + results) — the only mandatory input to `gate`.
- For `assemble`/`draft`: `template.tex` + `conference_guidelines.md` (any style memo works) — assemble emits `workspace/inputs/{idea.md, experimental_log.md}` *from the gated claims and real results only*.
- For deep survey: `idea.md` (scope statement), seed `citation_pool.json`, S2 key.
- For lints: contributions list (name → validated data), venue name, numeric values list.

### Warranting the two claim classes

**(a) Faithful-summary-of-paper claims** (the survey half):
- One claim per load-bearing summarized assertion (not per sentence — per assertion you'd be embarrassed to have wrong). Type **P2** (or P1 under systems-paper).
- Evidence = `source_artifact` pointing at the *surveyed paper itself*, which the ref-shape gate accepts in two forms:
  - URL scheme: `"ref": "https://arxiv.org/abs/2405.xxxxx"` (matches `^https?://\S+$`), or
  - vendored artifact at a commit: `"ref": "papers/vendored/smith2024.pdf@<commit>"` — **preferred**, since the repo already has `papers/`; pin the exact PDF you read.
- No `result_refs` needed. The *faithfulness* itself isn't machine-checked (gate checks resolvability, not semantics) — the discipline is: statement text must be checkable against the single cited artifact, and the deep-survey S2 verification guarantees the bibliography entries are real. Optionally add a second evidence ref (e.g. the paper's official code repo `github://…@commit`) for claims about artifacts rather than text.

**(b) Original quantitative claims backed by small experiments** (the scaling-analysis half):
- Each experiment lives in `experiments/<name>/` with committed code + a committed machine-readable `results.json` (or CSV). Commit before claiming — the commit hash *is* the warrant anchor.
- Claim type **P1** with `result_refs` into the bundle's `results[]` block, where every metric is transcribed verbatim from the committed results file. Evidence = `source_artifact` refs: `experiments/scaling/results.json@<commit>` (+ optionally the runner script `@<commit>`).
- `require_results` (on in all profiles) blocks drafting an evaluation with empty/valueless results; **D1** blocks any headline contribution lacking a VALIDATED datum; **D4** blocks 5–6-sig-fig printing of exploration-grade numbers — report ~2 sig figs / ratios in the manuscript, matching the small-experiment epistemic grade.
- If a claim extrapolates beyond what the experiments enable (classic scaling-analysis temptation), split it: the measured claim (P1, warranted) vs the projection (separate claim, ideally with 2 admissible refs à la the neurips P3 floor — and if it hints at a patentable device, type it P3 so the interlock holds it).

### Minimal driver (suggested)

```bash
# gate.sh at repo root
set -e
python3 -m veridraft import-bundle claims/bundle.json --data-dir .veridraft
python3 -m veridraft gate neural-memory-study-2026 --data-dir .veridraft
```

Run it in CI / pre-commit; the manuscript merges only when `blocked: (none)`.
