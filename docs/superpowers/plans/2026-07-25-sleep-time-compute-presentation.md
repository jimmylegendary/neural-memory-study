# Sleep-Time Compute Presentation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce an editable, evidence-locked, low-AI-feel sleep-time compute presentation and display-authoritative PDF without allowing slide production to alter or block the research release.

**Architecture:** A separate TypeScript package consumes a read-only G8 `presentation-handoff.json`, converts it into an editorial storyboard and typed scene graph, and renders true masters/placeholders, editable native shapes/charts, and hashed SVG evidence graphics with PptxGenJS. Automated OOXML, geometry, provenance, font, accessibility, and PDF checks precede target-PowerPoint review. The reproducible `generated.pptx` and sealed, allowlisted-postflight `release.pptx` are distinct artifacts.

**Design target:** `docs/superpowers/specs/2026-07-25-sleep-time-compute-presentation-design.md`.

**Tech Stack:** Node 25.6.1, npm 11.9.0, TypeScript 7.0.2, PptxGenJS 4.0.1, Ajv 8.20.0, Fontkit 2.0.4, JSZip 3.10.1, fast-xml-parser 5.10.1, `@resvg/resvg-js` 2.6.2, Vitest 4.1.10, `@types/node` 25.9.5, Python 3.14 with Matplotlib 3.11.1 in `uv.lock`, LibreOffice 24.2.7.2, Poppler 24.02.0, DocumentFormat.OpenXml 3.5.1 in a digest-pinned .NET 10 SDK container, veraPDF 1.30 in a digest-pinned container, and desktop Microsoft PowerPoint for terminal QA.

## Independence and Authority

- Logical package root is `presentation/sleep-time-compute/`.
- The package has its own `package-lock.json`, tests, build directory, and
  presentation-only branch/worktree.
- Research G0–G8 tests run without Node, LibreOffice, PowerPoint, veraPDF, or
  any presentation dependency.
- Tooling/capability spikes may run before G8. Factual release generation may
  consume only a G8-tagged handoff.
- A pre-G8 internal deck, if unavoidable, is watermarked
  `INTERNAL — NOT FOR EXPORT` on every slide and is rejected by release mode.
- Renderer source contains no factual slide copy. Facts, claims, numbers,
  citations, caveats, and chart data come only from the frozen handoff.
- Raw PptxGenJS output is not claimed byte-reproducible: the library writes
  current dates into core properties and embedded-chart XLSX metadata, and ZIP
  timestamps vary. Only the declared canonical OOXML/ZIP postprocessor may
  produce the reproducible `generated.pptx`; before that step, verification
  uses normalized part digests.
- PptxGenJS's Node SVG path cannot be trusted to synthesize a real PNG fallback
  in every case. Every SVG has a separately rendered, digest-bound
  `@resvg/resvg-js` PNG fallback; an `IMG_BROKEN` placeholder or missing
  fallback fails the build.
- PPTX failure cannot stale or fail the research release. A research handoff
  revision stales the deck, not vice versa.
- `seminar/build_deck.py`, `seminar/pptx_lib.py`, existing decks, previews, and
  equation assets are read-only references.
- The `claude-mem:wowerpoint` skill is not used: its kawaii NotebookLM/PDF
  output contract conflicts with the required editable, restrained,
  research-editorial PPTX.

## Command-Root Contract

Every `bash` block starts in a fresh shell at
`presentation/sleep-time-compute/`; no block inherits a previous working
directory or shell variable. The sole exception is Task 0: because it precedes
package creation, every Task 0 block starts at the repository root, proves that
root with `git rev-parse --show-toplevel`, and then changes to its reviewed
temporary directory. A block that needs another location resolves and changes
to it within that same block. Task 0's later spike blocks require the absolute
`STC_PPTX_SPIKE_DIR` as an explicit reviewed operator input and validate it
before changing directory; they never depend on a shell variable set by an
earlier block. Repository-level Git commands use `git -C ../..` outside Task 0
and the freshly resolved absolute repository root inside Task 0.

## Human Trust-Root Contract

Before any signed presentation subject exists, a human maintainer provisions
an Ed25519 trust root in its own earlier commit. It contains exactly one
distinct, pre-existing public key for each closed role:
`POWERPOINT_OPERATOR`, `CONTENT_OWNER`, and `VISUAL_EDITOR`. Automation never
generates or adds a reviewer key. Every signed subject binds the exact earlier
trust-root commit SHA and trust-root file digest, and every signature verifier
requires an explicit `--trusted-keys` path, re-resolves those bytes from the
bound earlier commit, and rejects current-worktree drift. A key or role supplied
by an attestation, a key first added in the outcome commit, an unknown key or
role, reuse of one key across roles, an expired/revoked key, or mutation of the
trust-root bytes fails closed.

## Durable Signed-Artifact Contract

`build/` is normally a disposable workspace, but a human signature may never
depend on an ignored, uncommitted leaf. The exact PPTX, PDF, contact-sheet, QA,
and render inputs named by representative, full-deck, postflight, or seal
subjects are therefore explicit exceptions: their task commit uses
path-scoped `git add -f` and proves each path is tracked. No broad ignored
directory is force-added. The signed subjects and `manifests/release.json`
provide the digest map; a clean checkout must reverify every approval and the
seal without a cache, temporary spike directory, or untracked `build/` file.
This contract also durably delivers the editable `release.pptx`.

## Frozen Handoff Contract

Evidence/publication owns the schema and self-contained bundle:

```text
research/sleep-time-compute/publication/handoff/{release_tag}/
  presentation-handoff.json
  presentation-handoff.schema.json
  release-manifest.json
  artifact-dag.jsonl
  source-index.jsonl
  claim-index.jsonl
  evidence-index.jsonl
  result-index.jsonl
  schemas/*.schema.json
  checksums.sha256
  assets/
```

The envelope has `schemaVersion`, `releaseTag`, `releaseManifestSha256`,
`artifactDagSha256`, `sourceIndexSha256`, `claimIndexSha256`,
`evidenceIndexSha256`, `resultIndexSha256`, `schemaSha256`, and `slides`.
Every slide contains exactly:

```text
id
order
kind
headline
audienceQuestion
takeaway
body
visualBrief
assets
sourceRefs
chartSpec
notes
altText
mustKeep
editorialFreedom
researchReleaseTag
```

Nested assets retain source asset ID, SHA-256, source IDs, license, role, crop,
alt text, claim/evidence IDs, artifact-DAG ID/path, and reuse owner.
`researchReleaseTag` equals the envelope tag. `chartSpec` is explicit or
`null`; no data are inferred in the presentation package.

Nested records are exactly:

```text
assets[]:
  assetId, sha256, sourceRefIds, license, role, crop, altText,
  researchClaimIds, evidenceIds, artifactId, relativePath, reuseOwner
sourceRefs[]:
  sourceRefId, url, citation, accessedAt, researchClaimIds, evidenceIds,
  resultIds, shortCaveat, longCaveat
chartSpec:
  null | {datasetAssetId, datasetSha256, units, transforms, filters, uncertainty}
mustKeep[]:
  {mustKeepId, field, text, canonicalTokenSha256}
editorialFreedom:
  {
    allowedLayoutOperations,
    textVariants,
    reorderConstraint
  }
editorialFreedom.allowedLayoutOperations[]:
  REFLOW | MOVE_WITHIN_SAFE_AREA | RESIZE_WITHIN_DECLARED_BOUNDS |
  SELECT_PRODUCER_VARIANT | REORDER_WITHIN_GROUP |
  CROP_TO_PRODUCER_BOX
editorialFreedom.textVariants[]:
  {variantId, field, text, canonicalTokenSha256}
editorialFreedom.reorderConstraint:
  null | {
    groupId, minOrder, maxOrder, precedesSlideIds, followsSlideIds
  }
```

The presentation package vendors the bundle’s schema bytes only for compilation
and verifies their digest. It never authors a competing schema. The four
indexes use the evidence-owned projections exactly:

```text
source-index: source ID, work/version identity, citation, URL, artifact digest,
  accessed date, rights disposition, short non-claim
claim-index: claim ID, exact statement/status/scope, caveats, typed support,
  counterevidence and result IDs, record digest
evidence-index: evidence/source/version/artifact IDs and digests, relation,
  role, short anchor/support summary, warrant/reproduction strength
result-index: result ID, estimand/unit, estimate/uncertainty, decision,
  logical-cell/resource vector, caveats, artifact digest
```

Validation resolves every reference exclusively within the bundle; access to
the live research registries is forbidden.

Canonical token hashing normalizes Unicode to NFC and line endings to LF but
preserves token order, punctuation, numbers, and citation markers. Only the
evidence producer may add a variant or reorder constraint. Unknown operations,
out-of-group reordering, a variant digest mismatch, or a selected variant that
drops a `mustKeep` record fails closed.

## Command Surface

`src/cli.ts` owns these exact commands:

```text
node dist/cli.js validate-handoff <bundle-dir>
node dist/cli.js build --handoff-bundle <bundle-dir> [--slide-ids-file <selector.json>] --output <generated.pptx>
node dist/cli.js qa --handoff <bundle-dir> --pptx <deck.pptx> --out <qa-dir> [--slide-ids-file <selector.json>] [--pdf-output <file.pdf> | --pdf-input <file.pdf>]
node dist/cli.js editorial-subject --pptx <slice.pptx> --powerpoint-pdf <powerpoint-export.pdf> --slide-ids-file <selector.json> --qa <qa-report.json> --contact-sheet <contact-sheet.png> --trusted-keys <trust-root.json> --trust-root-commit <commit-sha> --output <subject.json>
node dist/cli.js verify-editorial-approval --subject <subject.json> --powerpoint-operator-attestation <operator.json> --content-owner-attestation <content.json> --visual-editor-attestation <visual.json> --trusted-keys <trust-root.json> --output <approval.json>
node dist/cli.js full-deck-subject --pptx <generated.pptx> --provisional-pdf <generated.pdf> --contact-sheet <contact-sheet.png> --qa <qa-report.json> --pacing-rubric <rubric.json> --trusted-keys <trust-root.json> --trust-root-commit <commit-sha> --output <subject.json>
node dist/cli.js verify-full-deck-approval --subject <subject.json> --content-owner-attestation <content.json> --visual-editor-attestation <visual.json> --trusted-keys <trust-root.json> --output <approval.json>
node dist/cli.js validate-postflight --generated <generated.pptx> --release <release.pptx> --report <postflight.json>
node dist/cli.js compare-renders --generated <dir> --release <dir> --output <report.json>
node dist/cli.js powerpoint-postflight-subject --generated <generated.pptx> --release <release.pptx> --postflight <postflight.json> --reopen-report <reopen.json> --accessibility-report <accessibility.json> --release-pdf <release.pdf> --export-options <export-options.json> --trusted-keys <trust-root.json> --trust-root-commit <commit-sha> --output <subject.json>
node dist/cli.js verify-powerpoint-operator --subject <subject.json> --operator-attestation <operator.json> --trusted-keys <trust-root.json> --output <powerpoint-qa.json>
node dist/cli.js seal --generated <generated.pptx> --release <release.pptx> --release-pdf <release.pdf> --display-pdf <display.pdf> --qa <qa-dir> --postflight <postflight.json> --full-deck-approval <approval.json> --powerpoint-subject <subject.json> --powerpoint-operator-attestation <operator.json> --powerpoint-qa <powerpoint-qa.json> --trusted-keys <trust-root.json> --render-diff <render-diff.json> --output <release.json>
```

`npm run qa -- ...` is the sole tested alias for
`node dist/cli.js qa ...`; no other CLI aliases are allowed. The task that
introduces a command modifies
`src/cli.ts`, `package.json`, and `tests/cli.test.ts` before invoking it. Every
optional `--slide-ids-file` occurrence uses the same Task 3 selector schema and
digest semantics; all signature-bearing commands use the same canonical JSON
and immutable trust-root verifier.

## Editorial Contract

Each slide has one dominant hierarchy and one audience question. Use six visual
modes:

```text
assertion/opening
mechanism
evidence
comparison
synthesis/decision
section/reset
```

Typography:

```text
Noto Sans CJK KR       body and labels
Noto Serif CJK KR      restrained editorial display
Noto Sans Mono CJK KR  code, IDs, compact annotations
20–24 pt               normal projected body
18 pt                  hard floor except source/axis/legal labels
```

Use asymmetry, whitespace, deliberate crop/annotation, and pacing resets.
Matplotlib SVG is the default for scientific equations and plots. Native
PowerPoint charts are used only when audience editing matters. Mechanisms use
individually named native shapes. PptxGenJS has no assumed group-shape API;
grouping is excluded unless a dedicated OOXML spike proves it safe.

Reject slides based on repeated rounded-card grids, forced three-column
symmetry, pastel problem/solution boxes, dashboard density, generic icons,
decorative AI art, stock metaphors, unreadable paper thumbnails, automatic
highlighting, rainbow semantics, generic “What/Why/How” headings, shrink-to-fit,
edge-to-edge filling on every page, or mechanical badges/breadcrumbs.

---

### Task 0: Run disposable capability and editorial spikes

**Files:**
- Create before any spike subject: `docs/presentation/pptx-reviewer-trust-root.json`
- Create before any spike subject:
  `docs/presentation/pptx-reviewer-trust-root-approval.json`
- Create outside the repository: a temporary directory from `mktemp -d`
- Create temporarily: `package.json`, `package-lock.json`, `capability.mjs`,
  `scale.mjs`, `inspect.mjs`, `inspect-tools.mjs`, `postflight-diff.mjs`,
  `font-rights.mjs`, `editorial-subject.mjs`
- Create temporarily: `openxml-validator/{Dockerfile,OpenXmlValidator.csproj,Program.cs,packages.lock.json}`
- Create temporarily:
  `fonts/{manifest.json,OFL.txt,NotoSansCJKkr-Regular.otf,NotoSansCJKkr-Bold.otf,NotoSerifCJKkr-SemiBold.otf,NotoSansMonoCJKkr-Regular.otf}`
- Create temporarily:
  `editorial-slice-{subject,powerpoint-operator-attestation,content-owner-attestation,visual-editor-attestation}.json`
- Create temporarily: `capability-powerpoint-operator-attestation.json`
- Create after review: `docs/presentation/pptx-spike-report.md`
- Create after review: `docs/presentation/pptx-spike-manifest.json`
- Create after review:
  `docs/presentation/pptx-spike-fixtures/{capability,capability-font-embedded,capability-reading-order,capability-accessibility,capability-text-mutated,capability-chart-mutated,capability-geometry-mutated,capability-media-mutated}.pptx`
- Create after review:
  `docs/presentation/pptx-spike-fixtures/editorial-slice.{pptx,pdf}`
- Create after review:
  `docs/presentation/pptx-spike-fixtures/editorial-slice-slide-ids.json`
- Create after review:
  `docs/presentation/pptx-spike-fixtures/fonts/{manifest.json,OFL.txt,NotoSansCJKkr-Regular.otf,NotoSansCJKkr-Bold.otf,NotoSerifCJKkr-SemiBold.otf,NotoSansMonoCJKkr-Regular.otf}`
- Create after review:
  `docs/presentation/pptx-spike-fixtures/reviews/{capability-powerpoint-operator-attestation,editorial-slice-subject,editorial-slice-powerpoint-operator-attestation,editorial-slice-content-owner-attestation,editorial-slice-visual-editor-attestation}.json`
- Create after review:
  `docs/presentation/pptx-spike-fixtures/checksums.sha256`
- Create after review: `docs/presentation/pptx-spike.schema.json`
- Create after review: `docs/presentation/verify-pptx-spike.mjs`
- Create after review: `docs/presentation/verify-pptx-spike.test.mjs`

**Interfaces:**
- Produces: a go/fallback decision for PptxGenJS 4.0.1.
- Does not alter: repository lockfiles or research artifacts.
- Produces a self-contained, offline-verifiable synthetic fixture bundle. The
  Node standard-library verifier checks schema; the eight repo-relative
  postflight PPTX paths/digests; the editorial-slice PPTX, PDF, and slide-ID
  paths/digests; three nonempty allowed-change outcomes; four exact rejection
  codes; Open XML outcomes; target-PowerPoint reopen records; and Ed25519
  signatures for the PowerPoint operator, content owner, and visual editor.
  All three editorial signatures bind the exact persisted editorial-slice
  PPTX, PowerPoint-exported PDF, and producer-order slide-ID JSON—no contact
  sheet or unresolvable temporary render.
- The standard offline verifier always validates the immutable trust-root
  commit/digest and the exact four-font OFL manifest, font hashes,
  `OS/2.fsType`, and embedded-font provenance. There is no flag that skips
  trust or font-rights validation.

- [ ] **Step 0: Provision the reviewer trust root as an independent human boundary**

Pause before creating a spike deck or review subject. A human repository
maintainer provisions three pre-existing Ed25519 public keys with the exact,
non-overlapping roles `POWERPOINT_OPERATOR`, `CONTENT_OWNER`, and
`VISUAL_EDITOR`; automation must not generate these keys. The closed trust-root
record contains `schemaVersion`, `algorithm: "Ed25519"`, and exactly three
records with stable `keyId`, one of those roles, `publicKeySpkiBase64`,
`validFrom`, `validUntil`, and `revoked: false`. No private key enters the
repository. A second human
reviews its canonical bytes and records the repository's out-of-band
maintainer approval, trust-root SHA-256, reviewer identity, timestamp, and
`decision: "APPROVED"` in
`pptx-reviewer-trust-root-approval.json`.

Commit these two files alone before any signed subject exists:

```bash
STC_REPO_ROOT="$(git rev-parse --show-toplevel)"
test "$PWD" = "$STC_REPO_ROOT"
git -C "$STC_REPO_ROOT" add \
  docs/presentation/pptx-reviewer-trust-root.json \
  docs/presentation/pptx-reviewer-trust-root-approval.json
git -C "$STC_REPO_ROOT" diff --cached --check
git -C "$STC_REPO_ROOT" commit \
  -m "docs: provision presentation reviewer trust root"
STC_TRUST_ROOT_COMMIT="$(git -C "$STC_REPO_ROOT" rev-parse HEAD)"
git -C "$STC_REPO_ROOT" show \
  "$STC_TRUST_ROOT_COMMIT:docs/presentation/pptx-reviewer-trust-root.json" \
  | sha256sum
printf 'STC_TRUST_ROOT_COMMIT=%s\n' "$STC_TRUST_ROOT_COMMIT"
```

Every later review subject binds this trust-root commit SHA and file digest.
The verifier accepts only keys resolved from that trust-root commit and later
proves it is a strict ancestor of the distinct Task 0 Step 6 outcome commit. It
requires `--trusted-keys docs/presentation/pptx-reviewer-trust-root.json`.
`verify-pptx-spike.test.mjs` includes
`rejects_self_added_outcome_key`, `rejects_unknown_key_id`,
`rejects_key_or_role_reuse`, `rejects_current_trust_root_mutation`, and
`rejects_trust_root_commit_not_strictly_earlier`. A key embedded in a
manifest/attestation, a key added in the spike outcome commit, an unknown role,
one key assigned to two roles, an expired/revoked key, or a self-approved
trust-root mutation fails.

- [ ] **Step 1A: Create and print an isolated spike workspace, then stop**

Use `mktemp -d`; do not create dependencies under the research package. Pin:

```text
node 25.6.1
npm 11.9.0
pptxgenjs 4.0.1
fontkit 2.0.4
jszip 3.10.1
fast-xml-parser 5.10.1
@resvg/resvg-js 2.6.2
DocumentFormat.OpenXml 3.5.1
```

```bash
STC_REPO_ROOT="$(git rev-parse --show-toplevel)"
test "$PWD" = "$STC_REPO_ROOT"
STC_PPTX_SPIKE_DIR="$(mktemp -d)"
test "${STC_PPTX_SPIKE_DIR#/}" != "$STC_PPTX_SPIKE_DIR"
printf '%s\n' "$STC_PPTX_SPIKE_DIR"
```

Expected: the block prints one absolute directory and exits without changing
directory, installing dependencies, invoking Docker, or relying on the
variable in any later shell.

- [ ] **Step 1B: Create every spike source with an explicit `apply_patch` phase**

Set the task runner's patch target to the exact absolute directory printed by
Step 1A. In one or more explicit `apply_patch` calls, add complete,
non-placeholder implementations of:

```text
package.json
capability.mjs
scale.mjs
inspect.mjs
inspect-tools.mjs
postflight-diff.mjs
font-rights.mjs
editorial-subject.mjs
fonts/manifest.json
openxml-validator/Dockerfile
openxml-validator/OpenXmlValidator.csproj
openxml-validator/Program.cs
openxml-validator/packages.lock.json
```

Do not use `cat`, heredoc redirection, `tee`, `echo`, Python, or Node to write
these source files. Do not run a newly added program in this phase. The package
pins the four runtime dependencies above exactly; the Open XML project pins
`DocumentFormat.OpenXml` 3.5.1 with locked restore. `font-rights.mjs` parses the
SFNT table directory and `OS/2.fsType` itself. `editorial-subject.mjs`
canonicalizes exactly the PPTX, PowerPoint PDF, and slide-ID JSON leaves plus
the earlier trust-root commit/digest.

- [ ] **Step 1C: In a fresh repository-root shell, validate sources and run the toolchain**

The operator supplies the printed absolute path as
`STC_PPTX_SPIKE_DIR`; this block does not inherit it from Step 1A:

```bash
: "${STC_PPTX_SPIKE_DIR:?set to the absolute directory printed by Step 1A}"
STC_REPO_ROOT="$(git rev-parse --show-toplevel)"
test "$PWD" = "$STC_REPO_ROOT"
test "${STC_PPTX_SPIKE_DIR#/}" != "$STC_PPTX_SPIKE_DIR"
test -d "$STC_PPTX_SPIKE_DIR"
for STC_REQUIRED_SOURCE in \
  package.json \
  capability.mjs \
  scale.mjs \
  inspect.mjs \
  inspect-tools.mjs \
  postflight-diff.mjs \
  font-rights.mjs \
  editorial-subject.mjs \
  fonts/manifest.json \
  openxml-validator/Dockerfile \
  openxml-validator/OpenXmlValidator.csproj \
  openxml-validator/Program.cs \
  openxml-validator/packages.lock.json
do
  test -s "$STC_PPTX_SPIKE_DIR/$STC_REQUIRED_SOURCE" || exit 1
done
cd "$STC_PPTX_SPIKE_DIR"
npm install --package-lock-only --ignore-scripts
npm ci --ignore-scripts
test "$(node --version)" = "v25.6.1"
test "$(npm --version)" = "11.9.0"
node inspect-tools.mjs --output tool-versions.json
docker pull mcr.microsoft.com/dotnet/sdk:10.0
STC_DOTNET_BASE="$(
  docker image inspect mcr.microsoft.com/dotnet/sdk:10.0 \
    --format '{{index .RepoDigests 0}}'
)"
docker build \
  --build-arg DOTNET_BASE="$STC_DOTNET_BASE" \
  --tag stc-openxml-spike:3.5.1 \
  openxml-validator
docker image inspect stc-openxml-spike:3.5.1 \
  --format '{{.Id}}' \
  > openxml-validator-image.id
```

The temporary `.csproj` pins `DocumentFormat.OpenXml` to 3.5.1 and enables
NuGet lock-file restore. The Dockerfile uses `ARG DOTNET_BASE` followed by
`FROM ${DOTNET_BASE}` and `dotnet restore --locked-mode`. Record the resolved
base digest, built image ID, NuGet lock digest, npm lock digest, and program
digests in the spike manifest; a floating base tag or unlocked restore fails
the spike. Step 1C must fail before npm or Docker if any Step 1B file is
missing; no Task 0 command depends on a prior shell's current directory or
variables.

- [ ] **Step 2: Build an eight-slide capability deck**

Cover Korean, Latin, Greek, punctuation, superscripts, long titles, four font
roles, true master placeholders, SVG, native chart, notes, image/chart alt
text, individually named multi-shape diagram, internal/external links, and
document metadata.

The SVG slide embeds a nontrivial transparency/gradient fixture plus a PNG
rendered from the exact SVG bytes with `@resvg/resvg-js`. Inspection fails on
`IMG_BROKEN`, an empty or single-color placeholder fallback, a missing media
relationship, a dimension mismatch, or a fallback whose provenance digest does
not bind the SVG source.

Use only these four redistributable upstream font bytes and the byte-identical
SIL Open Font License 1.1 text. These values are normative inputs to both the
spike and package font manifests:

| Role | File / PostScript name | Upstream commit and path | SHA-256 | `OS/2.fsType` |
|---|---|---|---|---|
| body | `NotoSansCJKkr-Regular.otf` / `NotoSansCJKkr-Regular` | `notofonts/noto-cjk@523d033d6cb47f4a80c58a35753646f5c3608a78`, `Sans/OTF/Korean/NotoSansCJKkr-Regular.otf` | `6bcb2a0703aa137e874fc2dffa85f6c21ba9a67fa329e81b8c801663af7e992a` | `0x0000` |
| label/emphasis | `NotoSansCJKkr-Bold.otf` / `NotoSansCJKkr-Bold` | `notofonts/noto-cjk@523d033d6cb47f4a80c58a35753646f5c3608a78`, `Sans/OTF/Korean/NotoSansCJKkr-Bold.otf` | `26d0c6748500a0444844280b308f5b62c7ae92ac6c6ac88148e502dd211eb52a` | `0x0000` |
| display | `NotoSerifCJKkr-SemiBold.otf` / `NotoSerifCJKkr-SemiBold` | `notofonts/noto-cjk@9b0f1436e455d902de067a2501422e5dc71ad16b`, `Serif/OTF/Korean/NotoSerifCJKkr-SemiBold.otf` | `bce8d43887e2999c3af13802311fdc2b68f2c5b79d606ed3dd85293649932dcc` | `0x0000` |
| code/IDs | `NotoSansMonoCJKkr-Regular.otf` / `NotoSansMonoCJKkr-Regular` | `notofonts/noto-cjk@523d033d6cb47f4a80c58a35753646f5c3608a78`, `Sans/Mono/NotoSansMonoCJKkr-Regular.otf` | `d5afed9988a28ae96afb0f4791754d3c9f4f08d08477eb1a7d6e2d905679a472` | `0x0000` |

`fonts/OFL.txt` comes from
`notofonts/noto-cjk@523d033d6cb47f4a80c58a35753646f5c3608a78/LICENSE`,
has SHA-256
`6a73f9541c2de74158c0e7cf6b0a58ef774f5a780bf191f2d7ec9cc53efe2bf2`,
and is recorded as SPDX `OFL-1.1`. The Serif 2.003 license at
`Serif/LICENSE` has the same digest. Fetch and verify the files in a fresh
Task 0 shell:

```bash
: "${STC_PPTX_SPIKE_DIR:?set to the absolute directory printed by Step 1A}"
STC_REPO_ROOT="$(git rev-parse --show-toplevel)"
test "$PWD" = "$STC_REPO_ROOT"
test "${STC_PPTX_SPIKE_DIR#/}" != "$STC_PPTX_SPIKE_DIR"
test -s "$STC_PPTX_SPIKE_DIR/font-rights.mjs"
mkdir -p "$STC_PPTX_SPIKE_DIR/fonts"
curl -fL --proto '=https' \
  -o "$STC_PPTX_SPIKE_DIR/fonts/NotoSansCJKkr-Regular.otf" \
  "https://raw.githubusercontent.com/notofonts/noto-cjk/523d033d6cb47f4a80c58a35753646f5c3608a78/Sans/OTF/Korean/NotoSansCJKkr-Regular.otf"
curl -fL --proto '=https' \
  -o "$STC_PPTX_SPIKE_DIR/fonts/NotoSansCJKkr-Bold.otf" \
  "https://raw.githubusercontent.com/notofonts/noto-cjk/523d033d6cb47f4a80c58a35753646f5c3608a78/Sans/OTF/Korean/NotoSansCJKkr-Bold.otf"
curl -fL --proto '=https' \
  -o "$STC_PPTX_SPIKE_DIR/fonts/NotoSerifCJKkr-SemiBold.otf" \
  "https://raw.githubusercontent.com/notofonts/noto-cjk/9b0f1436e455d902de067a2501422e5dc71ad16b/Serif/OTF/Korean/NotoSerifCJKkr-SemiBold.otf"
curl -fL --proto '=https' \
  -o "$STC_PPTX_SPIKE_DIR/fonts/NotoSansMonoCJKkr-Regular.otf" \
  "https://raw.githubusercontent.com/notofonts/noto-cjk/523d033d6cb47f4a80c58a35753646f5c3608a78/Sans/Mono/NotoSansMonoCJKkr-Regular.otf"
curl -fL --proto '=https' \
  -o "$STC_PPTX_SPIKE_DIR/fonts/OFL.txt" \
  "https://raw.githubusercontent.com/notofonts/noto-cjk/523d033d6cb47f4a80c58a35753646f5c3608a78/LICENSE"
cd "$STC_PPTX_SPIKE_DIR"
node font-rights.mjs \
  --fonts-dir fonts \
  --manifest fonts/manifest.json \
  --license fonts/OFL.txt
```

Expected: the program recomputes every file/license digest, PostScript name,
and `OS/2.fsType=0x0000` and prints `PASS font_rights=OFL-1.1 fonts=4`.
The positive font case uses PowerPoint's full-font embedding option, not subset
embedding. A system-default, proprietary, differently hashed, or unknown font
may not enter any fixture, generated PPTX, release PPTX, QA bundle, or release
archive.

For native mechanism shapes, spike deterministic OOXML injection of object
name/title/description into each shape’s `p:cNvPr` using stable scene-node IDs.
Validate the injected package with the pinned Open XML image, target
PowerPoint, and the Accessibility Checker. If this fails, the affected
mechanism is one SVG with meaningful alt text; release mode never asserts
unsupported native-shape alt text.

- [ ] **Step 3: Run SVG/chart/raster and 30-slide scale comparisons**

Render the same evidence chart as Matplotlib SVG, native chart, and raster
fallback. Generate a 30-slide mixed-content deck. Inspect slide size, notes,
relationships, media, masters, placeholders, object names, and render
stability.

- [ ] **Step 4: Build a representative 12–15-slide editorial slice**

Use synthetic or already frozen non-result content. Generate exactly 13
identified slides with `scale.mjs --profile editorial`, not an unspecified
subset of `scale.pptx`. `inspect.mjs` emits the producer-order slide IDs and
their content digests to `editorial-slice-slide-ids.json`. An identified
target-PowerPoint operator opens the exact PPTX digest without repair and
exports `editorial-slice.pdf` from desktop PowerPoint. The PDF is not a
LibreOffice, Poppler, or synthetic substitute. Generate the PPTX and slide-ID
record in a fresh shell:

```bash
: "${STC_PPTX_SPIKE_DIR:?set to the absolute directory printed by Step 1A}"
STC_REPO_ROOT="$(git rev-parse --show-toplevel)"
test "$PWD" = "$STC_REPO_ROOT"
test "${STC_PPTX_SPIKE_DIR#/}" != "$STC_PPTX_SPIKE_DIR"
cd "$STC_PPTX_SPIKE_DIR"
node scale.mjs \
  --slides 13 \
  --profile editorial \
  --output editorial-slice.pptx
node inspect.mjs editorial-slice.pptx \
  --slide-ids-output editorial-slice-slide-ids.json \
  --output editorial-slice-inspection.json
test "$(node -e \
  "const x=require('./editorial-slice-slide-ids.json');process.stdout.write(String(x.slideIds.length))")" \
  = "13"
sha256sum editorial-slice.pptx editorial-slice-slide-ids.json \
  | LC_ALL=C sort > editorial-slice-pre-export.sha256
```

Obtain separate content-owner and human visual-editor decisions on narrative
pacing, hierarchy, typography, and the low-AI-feel rejection list. Each
canonical signed review subject has exactly three artifact leaves: the
editorial-slice PPTX, its PowerPoint-exported PDF, and the producer-order
slide-ID JSON. It additionally binds the earlier trust-root commit/digest and
`decision=PASS`; it does not bind or persist a fourth rendered derivative. The
PowerPoint operator, content owner, and visual editor use the three distinct
role keys, and no key may sign two roles.

Pause after the first block. The operator opens the exact PPTX digest and
exports `editorial-slice.pdf`. In a new repository-root shell, create the
canonical subject:

```bash
: "${STC_PPTX_SPIKE_DIR:?set to the absolute directory printed by Step 1A}"
: "${STC_TRUST_ROOT_COMMIT:?set to the earlier Step 0 trust-root commit}"
STC_REPO_ROOT="$(git rev-parse --show-toplevel)"
test "$PWD" = "$STC_REPO_ROOT"
test "${STC_PPTX_SPIKE_DIR#/}" != "$STC_PPTX_SPIKE_DIR"
git -C "$STC_REPO_ROOT" merge-base --is-ancestor \
  "$STC_TRUST_ROOT_COMMIT" HEAD
cd "$STC_PPTX_SPIKE_DIR"
sha256sum -c editorial-slice-pre-export.sha256
test -s editorial-slice.pdf
node editorial-subject.mjs \
  --pptx editorial-slice.pptx \
  --powerpoint-pdf editorial-slice.pdf \
  --slide-ids-file editorial-slice-slide-ids.json \
  --trusted-keys \
    "$STC_REPO_ROOT/docs/presentation/pptx-reviewer-trust-root.json" \
  --trust-root-commit "$STC_TRUST_ROOT_COMMIT" \
  --output editorial-slice-subject.json
```

Pause again. The PowerPoint operator, content owner, and visual editor inspect
those exact three subject-bound artifacts and independently sign the canonical
subject as `editorial-slice-powerpoint-operator-attestation.json`,
`editorial-slice-content-owner-attestation.json` and
`editorial-slice-visual-editor-attestation.json`. A review of `scale.pptx`, an
unspecified page range, a derived preview in place of the PDF, or an unpersisted
temporary render is invalid.

- [ ] **Step 5: Test every terminal allowlist class and negative control**

Open without a repair prompt. From the same generated capability deck, target
PowerPoint produces three separate positive cases:
`capability-font-embedded.pptx`, `capability-reading-order.pptx`, and
`capability-accessibility.pptx`. It also produces four deliberately forbidden
cases: `capability-text-mutated.pptx`, `capability-chart-mutated.pptx`,
`capability-geometry-mutated.pptx`, and `capability-media-mutated.pptx`.
The accessibility case exercises object name, title, description, and the
decorative marker. No file combines two cases.

```bash
: "${STC_PPTX_SPIKE_DIR:?export the reviewed Task 0 spike directory}"
STC_REPO_ROOT="$(git rev-parse --show-toplevel)"
test "$PWD" = "$STC_REPO_ROOT"
test "${STC_PPTX_SPIKE_DIR#/}" != "$STC_PPTX_SPIKE_DIR"
cd "$STC_PPTX_SPIKE_DIR"
node capability.mjs --output capability.pptx
node scale.mjs --slides 30 --output scale.pptx
node inspect.mjs capability.pptx scale.pptx --output inspection.json
mkdir -p renders
libreoffice \
  -env:UserInstallation=file:///tmp/stc-pptx-spike-lo \
  --headless --convert-to pdf --outdir renders capability.pptx
docker run --rm \
  --mount "type=bind,src=$STC_PPTX_SPIKE_DIR,dst=/work,readonly" \
  stc-openxml-spike:3.5.1 \
  /work/capability.pptx \
  > openxml-capability.json
sha256sum capability.pptx > capability-base.sha256
```

Pause here. An identified target-PowerPoint operator opens the exact
`capability.pptx` digest without repair, creates the three allowed and four
forbidden variants named above as separate files, reopens every variant, runs
Accessibility Checker/Reading Order where applicable, and signs a structured
operator record named `capability-powerpoint-operator-attestation.json` over
all eight PPTX digests and observed application/version results. Step 5
consumes the already exported and signed Step 4 editorial-slice artifacts
unchanged; it must not reopen, re-export, or re-sign them. Automation must not
synthesize any file or approval.

Only after all seven variants, all three editorial-slice files, the three
editorial attestations, and the separate capability-operator attestation exist,
run:

```bash
: "${STC_PPTX_SPIKE_DIR:?export the reviewed Task 0 spike directory}"
STC_REPO_ROOT="$(git rev-parse --show-toplevel)"
test "$PWD" = "$STC_REPO_ROOT"
test "${STC_PPTX_SPIKE_DIR#/}" != "$STC_PPTX_SPIKE_DIR"
cd "$STC_PPTX_SPIKE_DIR"
sha256sum -c capability-base.sha256
for STC_REQUIRED_PPTX in \
  capability-font-embedded.pptx \
  capability-reading-order.pptx \
  capability-accessibility.pptx \
  capability-text-mutated.pptx \
  capability-chart-mutated.pptx \
  capability-geometry-mutated.pptx \
  capability-media-mutated.pptx
do
  test -f "$STC_REQUIRED_PPTX" || exit 1
done
for STC_EDITORIAL_ARTIFACT in \
  editorial-slice.pptx \
  editorial-slice.pdf \
  editorial-slice-slide-ids.json
do
  test -f "$STC_EDITORIAL_ARTIFACT" || exit 1
done
for STC_EDITORIAL_RECORD in \
  editorial-slice-subject.json \
  editorial-slice-powerpoint-operator-attestation.json \
  editorial-slice-content-owner-attestation.json \
  editorial-slice-visual-editor-attestation.json
do
  test -s "$STC_EDITORIAL_RECORD" || exit 1
done
test -s capability-powerpoint-operator-attestation.json
sha256sum capability*.pptx | LC_ALL=C sort > capability-variants.sha256
sha256sum \
  editorial-slice.pptx \
  editorial-slice.pdf \
  editorial-slice-slide-ids.json \
  | LC_ALL=C sort > editorial-slice.sha256
for STC_ALLOWED_SPEC in \
  capability-font-embedded:FONT_EMBEDDING \
  capability-reading-order:READING_ORDER \
  capability-accessibility:ACCESSIBILITY_METADATA
do
  STC_ALLOWED_CASE="${STC_ALLOWED_SPEC%%:*}"
  STC_ALLOWED_CLASS="${STC_ALLOWED_SPEC#*:}"
  node postflight-diff.mjs \
    --generated capability.pptx \
    --release "$STC_ALLOWED_CASE.pptx" \
    --expect-allowed-class "$STC_ALLOWED_CLASS" \
    --output "$STC_ALLOWED_CASE-diff.json" || exit 1
  docker run --rm \
    --mount "type=bind,src=$STC_PPTX_SPIKE_DIR,dst=/work,readonly" \
    stc-openxml-spike:3.5.1 \
    "/work/$STC_ALLOWED_CASE.pptx" \
    > "$STC_ALLOWED_CASE-openxml.json" || exit 1
done
for STC_FORBIDDEN_SPEC in \
  capability-text-mutated:TEXT_MUTATION \
  capability-chart-mutated:CHART_DATA_MUTATION \
  capability-geometry-mutated:GEOMETRY_MUTATION \
  capability-media-mutated:MEDIA_MUTATION
do
  STC_FORBIDDEN_CASE="${STC_FORBIDDEN_SPEC%%:*}"
  STC_EXPECTED_CODE="${STC_FORBIDDEN_SPEC#*:}"
  docker run --rm \
    --mount "type=bind,src=$STC_PPTX_SPIKE_DIR,dst=/work,readonly" \
    stc-openxml-spike:3.5.1 \
    "/work/$STC_FORBIDDEN_CASE.pptx" \
    > "$STC_FORBIDDEN_CASE-openxml.json" || exit 1
  node postflight-diff.mjs \
    --generated capability.pptx \
    --release "$STC_FORBIDDEN_CASE.pptx" \
    --expect-rejection "$STC_EXPECTED_CODE" \
    --output "$STC_FORBIDDEN_CASE-diff.json" || exit 1
done
```

The target-PowerPoint no-repair result, all three post-edit reopens,
Accessibility Checker and Reading Order results, and visual-editor/content-owner
approvals are structured records keyed to exact PPTX digests. The manifest
records each expected/observed positive change class and negative stable error
code. A byte-identical positive or an unrelated negative failure does not
satisfy the spike; appending unstructured prose to the machine report cannot
change either result.

- [ ] **Step 6: Record the decision**

Use PptxGenJS only if the original and all three allowed cases pass Open XML and
target-PowerPoint reopening, all allowed diffs report exactly their required
nonempty change class, all four forbidden diffs report the expected stable
error code while their input PPTX files still pass Open XML,
Accessibility Checker/Reading Order and both human reviews pass, and every
recorded digest resolves. If any condition fails, stop this implementation plan
and obtain approval for a separately reviewed
`python-pptx 1.0.2` fallback design; do not imply that Tasks 1–6 remain valid
unchanged.

Create the JSON Schema, standard-library offline verifier and tests, signed
manifest, and report with `apply_patch`. The manifest maps all eight postflight
PPTX artifacts and all three editorial-slice artifacts to repo-relative
paths/digests, labels them `PROJECT_GENERATED_SYNTHETIC`, binds the
positive/negative reports and three signed human records, and contains no
temporary absolute path. Its editorial review subject resolves exactly the
PPTX, PowerPoint-exported PDF, and slide-ID JSON digests from the bundle. The
manifest also binds the earlier trust-root commit/digest and the exact
four-font/OFL rights manifest. The capability operator record and all four
editorial review records are named, independently resolvable signed leaves.
Then vendor the exact reviewed
bytes and verify before commit:

The verifier has one fail-closed phase distinction derived from Git state, not
a bypass flag. Before the outcome commit, when the exact manifest blob is not
yet present at `HEAD`, it validates the staged candidate, signatures, and
trust-root commit bytes but does not demand that the root be an ancestor of a
nonexistent outcome commit. After commit, when `HEAD` contains that exact
manifest blob, it additionally requires the trust-root commit to be a strict
ancestor. Step 6 runs both phases; offline reuse of a committed outcome always
takes the strict post-commit path.

```bash
: "${STC_PPTX_SPIKE_DIR:?export the reviewed Task 0 spike directory}"
: "${STC_TRUST_ROOT_COMMIT:?set to the earlier Task 0 Step 0 commit}"
STC_REPO_ROOT="$(git rev-parse --show-toplevel)"
test "$PWD" = "$STC_REPO_ROOT"
test "${STC_PPTX_SPIKE_DIR#/}" != "$STC_PPTX_SPIKE_DIR"
mkdir -p "$STC_REPO_ROOT/docs/presentation/pptx-spike-fixtures"
for STC_SPIKE_PPTX in \
  capability.pptx \
  capability-font-embedded.pptx \
  capability-reading-order.pptx \
  capability-accessibility.pptx \
  capability-text-mutated.pptx \
  capability-chart-mutated.pptx \
  capability-geometry-mutated.pptx \
  capability-media-mutated.pptx
do
  install -m 0644 \
    "$STC_PPTX_SPIKE_DIR/$STC_SPIKE_PPTX" \
    "$STC_REPO_ROOT/docs/presentation/pptx-spike-fixtures/$STC_SPIKE_PPTX"
done
for STC_EDITORIAL_ARTIFACT in \
  editorial-slice.pptx \
  editorial-slice.pdf \
  editorial-slice-slide-ids.json
do
  install -m 0644 \
    "$STC_PPTX_SPIKE_DIR/$STC_EDITORIAL_ARTIFACT" \
    "$STC_REPO_ROOT/docs/presentation/pptx-spike-fixtures/$STC_EDITORIAL_ARTIFACT"
done
mkdir -p "$STC_REPO_ROOT/docs/presentation/pptx-spike-fixtures/reviews"
for STC_REVIEW_RECORD in \
  capability-powerpoint-operator-attestation.json \
  editorial-slice-subject.json \
  editorial-slice-powerpoint-operator-attestation.json \
  editorial-slice-content-owner-attestation.json \
  editorial-slice-visual-editor-attestation.json
do
  install -m 0644 \
    "$STC_PPTX_SPIKE_DIR/$STC_REVIEW_RECORD" \
    "$STC_REPO_ROOT/docs/presentation/pptx-spike-fixtures/reviews/$STC_REVIEW_RECORD"
done
mkdir -p "$STC_REPO_ROOT/docs/presentation/pptx-spike-fixtures/fonts"
for STC_FONT_ARTIFACT in \
  manifest.json \
  OFL.txt \
  NotoSansCJKkr-Regular.otf \
  NotoSansCJKkr-Bold.otf \
  NotoSerifCJKkr-SemiBold.otf \
  NotoSansMonoCJKkr-Regular.otf
do
  install -m 0644 \
    "$STC_PPTX_SPIKE_DIR/fonts/$STC_FONT_ARTIFACT" \
    "$STC_REPO_ROOT/docs/presentation/pptx-spike-fixtures/fonts/$STC_FONT_ARTIFACT"
done
(
  cd "$STC_REPO_ROOT/docs/presentation/pptx-spike-fixtures"
  sha256sum \
    *.pptx \
    editorial-slice.pdf \
    editorial-slice-slide-ids.json \
    fonts/manifest.json \
    fonts/OFL.txt \
    fonts/NotoSansCJKkr-Regular.otf \
    fonts/NotoSansCJKkr-Bold.otf \
    fonts/NotoSerifCJKkr-SemiBold.otf \
    fonts/NotoSansMonoCJKkr-Regular.otf \
    reviews/capability-powerpoint-operator-attestation.json \
    reviews/editorial-slice-subject.json \
    reviews/editorial-slice-powerpoint-operator-attestation.json \
    reviews/editorial-slice-content-owner-attestation.json \
    reviews/editorial-slice-visual-editor-attestation.json \
    | LC_ALL=C sort > checksums.sha256
)
node --test "$STC_REPO_ROOT/docs/presentation/verify-pptx-spike.test.mjs"
node "$STC_REPO_ROOT/docs/presentation/verify-pptx-spike.mjs" \
  --manifest "$STC_REPO_ROOT/docs/presentation/pptx-spike-manifest.json" \
  --fixtures "$STC_REPO_ROOT/docs/presentation/pptx-spike-fixtures" \
  --trusted-keys \
    "$STC_REPO_ROOT/docs/presentation/pptx-reviewer-trust-root.json" \
  --offline
git -C "$STC_REPO_ROOT" add \
  docs/presentation/pptx-spike-report.md \
  docs/presentation/pptx-spike-manifest.json \
  docs/presentation/pptx-spike.schema.json \
  docs/presentation/verify-pptx-spike.mjs \
  docs/presentation/verify-pptx-spike.test.mjs \
  docs/presentation/pptx-spike-fixtures
if test -f "$STC_REPO_ROOT/research/sleep-time-compute/pyproject.toml"; then
  (
    cd "$STC_REPO_ROOT/research/sleep-time-compute"
    uv run stc sources staged-scan --repo-root "$STC_REPO_ROOT" --cached
  )
fi
git -C "$STC_REPO_ROOT" diff --cached --check
git -C "$STC_REPO_ROOT" commit \
  -m "docs: validate sleep-time presentation toolchain"
STC_SPIKE_OUTCOME_COMMIT="$(git -C "$STC_REPO_ROOT" rev-parse HEAD)"
test "$STC_TRUST_ROOT_COMMIT" != "$STC_SPIKE_OUTCOME_COMMIT"
git -C "$STC_REPO_ROOT" merge-base --is-ancestor \
  "$STC_TRUST_ROOT_COMMIT" "$STC_SPIKE_OUTCOME_COMMIT"
node "$STC_REPO_ROOT/docs/presentation/verify-pptx-spike.mjs" \
  --manifest "$STC_REPO_ROOT/docs/presentation/pptx-spike-manifest.json" \
  --fixtures "$STC_REPO_ROOT/docs/presentation/pptx-spike-fixtures" \
  --trusted-keys \
    "$STC_REPO_ROOT/docs/presentation/pptx-reviewer-trust-root.json" \
  --offline
```

Expected: explicit `renderer_decision`, known gaps, target-PowerPoint version,
all three human approvals, an offline-valid eight-PPTX postflight fixture set,
an exactly three-artifact editorial-slice review set, and a four-font OFL
rights set. No spike deck becomes a release artifact; the synthetic fixtures
exist only to replay the terminal postflight allowlist and editorial-gate
tests.

---

### Task 1: Scaffold the isolated presentation package

**Files:**
- Create: `presentation/sleep-time-compute/package.json`
- Create: `presentation/sleep-time-compute/package-lock.json`
- Create: `presentation/sleep-time-compute/pyproject.toml`
- Create: `presentation/sleep-time-compute/uv.lock`
- Create: `presentation/sleep-time-compute/.python-version`
- Create: `presentation/sleep-time-compute/tsconfig.json`
- Create: `presentation/sleep-time-compute/.nvmrc`
- Create: `presentation/sleep-time-compute/.gitignore`
- Create: `presentation/sleep-time-compute/README.md`
- Vendor from evidence fixture: `presentation/sleep-time-compute/vendor/presentation-handoff.schema.json`
- Create: `presentation/sleep-time-compute/src/contracts.ts`
- Create: `presentation/sleep-time-compute/src/handoff.ts`
- Create: `presentation/sleep-time-compute/src/cli.ts`
- Create: `presentation/sleep-time-compute/tests/handoff.test.ts`
- Create: `presentation/sleep-time-compute/tests/fixtures/g8-handoff/presentation-handoff.json`
- Create: `presentation/sleep-time-compute/tests/fixtures/g8-handoff/presentation-handoff.schema.json`
- Create: `presentation/sleep-time-compute/tests/fixtures/g8-handoff/release-manifest.json`
- Create: `presentation/sleep-time-compute/tests/fixtures/g8-handoff/artifact-dag.jsonl`
- Create: `presentation/sleep-time-compute/tests/fixtures/g8-handoff/source-index.jsonl`
- Create: `presentation/sleep-time-compute/tests/fixtures/g8-handoff/claim-index.jsonl`
- Create: `presentation/sleep-time-compute/tests/fixtures/g8-handoff/evidence-index.jsonl`
- Create: `presentation/sleep-time-compute/tests/fixtures/g8-handoff/result-index.jsonl`
- Create: `presentation/sleep-time-compute/tests/fixtures/g8-handoff/schemas/*.schema.json`
- Create: `presentation/sleep-time-compute/tests/fixtures/g8-handoff/checksums.sha256`
- Create: `presentation/sleep-time-compute/tests/fixtures/g8-handoff/assets/`
- Create: `presentation/sleep-time-compute/tests/cli.test.ts`

**Interfaces:**
- Produces: `validate-handoff` and typed immutable handoff records generated
  from the evidence-owned schema.

- [ ] **Step 1: Write failing consumer-contract tests**

Reject:

```text
non-G8 release
slide/envelope release-tag mismatch
duplicate or noncontiguous slide order
duplicate slide ID
missing notes, caveat, citation, alt text, or audience question
unknown field
missing or digest-mismatched asset
missing or digest-mismatched source/claim/evidence/result index
unresolved source/claim/evidence/result/artifact ID
missing/unknown asset license
chartSpec with no hashed data source
release-manifest, artifact-DAG, or schema digest mismatch
unknown editorial operation
variant ID/text/token-digest mismatch
mustKeep text/digest loss
reorder outside the producer-authored group or precedence constraints
bundle checksum failure
```

- [ ] **Step 2: Create the pinned package**

Pin every direct dependency exactly and commit the generated lockfile:

```bash
npm install --save-exact \
  pptxgenjs@4.0.1 \
  ajv@8.20.0 \
  fontkit@2.0.4 \
  jszip@3.10.1 \
  fast-xml-parser@5.10.1 \
  @resvg/resvg-js@2.6.2
npm install --save-dev --save-exact \
  typescript@7.0.2 \
  vitest@4.1.10 \
  @types/node@25.9.5
uv init --bare --python 3.14
uv add 'matplotlib==3.11.1'
uv lock
```

`package.json` uses no range operator for a direct dependency and `npm ci`
must not modify `package-lock.json`. `pyproject.toml` uses the exact
`matplotlib==3.11.1` requirement, `uv sync --frozen` must not modify
`uv.lock`, `.python-version` contains `3.14`, and `.nvmrc` contains `25.6.1`.
Use strict TypeScript and no implicit `any`.
`.gitignore` excludes `node_modules/`, `build/*.pptx`, rendered page images, and
temporary office profiles while retaining small manifests, checksums, and QA
reports. Later task commits use only the exact force-add exceptions enumerated
by the Durable Signed-Artifact Contract; the ignore rules remain protective for
all other generated files.

- [ ] **Step 3: Implement validation**

Validate with the vendored evidence-owned JSON Schema, first proving its digest
matches the bundle. Then run semantic checks for release-tag, release-manifest
G8 status, artifact-DAG resolution, all four index digests, filesystem
digests, the bundled schema inventory/digests, references, canonical token
hashes, `mustKeep`, variants, and reorder constraints. Temporarily hiding the
live research registry during the valid fixture test proves that resolution is
bundle-local.

```bash
npm ci
uv sync --frozen
npm run build
npm test
node dist/cli.js validate-handoff \
  tests/fixtures/g8-handoff
```

Expected: tests pass and CLI prints `PASS handoff`; each invalid fixture fails
with a stable machine-readable error code.

- [ ] **Step 4: Commit**

```bash
git -C ../.. add presentation/sleep-time-compute
git -C ../.. commit -m "presentation: add frozen handoff contract"
```

---

### Task 2: Build storyboard, scene graph, masters, and visual modes

**Files:**
- Create: `presentation/sleep-time-compute/src/scene.ts`
- Create: `presentation/sleep-time-compute/src/storyboard.ts`
- Create: `presentation/sleep-time-compute/src/theme.ts`
- Create: `presentation/sleep-time-compute/src/masters.ts`
- Create: `presentation/sleep-time-compute/src/modes/assertion.ts`
- Create: `presentation/sleep-time-compute/src/modes/mechanism.ts`
- Create: `presentation/sleep-time-compute/src/modes/evidence.ts`
- Create: `presentation/sleep-time-compute/src/modes/comparison.ts`
- Create: `presentation/sleep-time-compute/src/modes/synthesis.ts`
- Create: `presentation/sleep-time-compute/src/modes/section.ts`
- Create: `presentation/sleep-time-compute/tests/storyboard.test.ts`
- Create: `presentation/sleep-time-compute/tests/masters.test.ts`
- Create: `presentation/sleep-time-compute/tests/geometry.test.ts`

**Interfaces:**
- Produces: typed `Storyboard`, `SlideScene`, and named `SceneNode`.

- [ ] **Step 1: Write master and geometry tests**

Require 16:9 dimensions, `ko-KR` language metadata, real master/placeholders,
stable object names, bounded boxes, declared z-order, and the 18 pt floor.
Reject shrink-to-fit and undeclared overlaps.

- [ ] **Step 2: Implement the editorial transformation**

The storyboard may reorder a slide only within its
`editorialFreedom.reorderConstraint`, while satisfying every explicit
predecessor/successor edge. The renderer accepts only the enumerated operations
in the frozen contract. It may select a complete producer-authored
`textVariants[]` record or reflow it while preserving its canonical token hash;
it may not freely paraphrase, shorten, splice variants, or omit factual text.
A missing suitable variant returns to the evidence producer for a new handoff
revision. Every `mustKeep` record and caveat survives with its original field,
text, and token digest, and the build provenance records each selected
variant/operation/reorder.

- [ ] **Step 3: Implement six modes**

Every mode has one dominant hierarchy and at most one primary visual. Do not
implement a universal card-grid component. Color tokens map to information
roles, not arbitrary sections.

- [ ] **Step 4: Verify**

```bash
npm run build
npm test -- storyboard.test.ts masters.test.ts geometry.test.ts
```

Expected: every fixture slide maps to one mode and contains no geometry,
typography, provenance, variant/reorder, or `mustKeep` violation.

- [ ] **Step 5: Commit**

```bash
git -C ../.. add presentation/sleep-time-compute
git -C ../.. commit -m "presentation: add editorial scene system"
```

---

### Task 3: Implement deterministic PptxGenJS rendering

**Files:**
- Create: `presentation/sleep-time-compute/src/render.ts`
- Create: `presentation/sleep-time-compute/src/slice.ts`
- Create: `presentation/sleep-time-compute/src/accessibility.ts`
- Create: `presentation/sleep-time-compute/src/canonicalize.ts`
- Create: `presentation/sleep-time-compute/src/svgFallback.ts`
- Create: `presentation/sleep-time-compute/python/render_evidence.py`
- Create: `presentation/sleep-time-compute/tests/slice.test.ts`
- Create: `presentation/sleep-time-compute/tests/canonicalize.test.ts`
- Create: `presentation/sleep-time-compute/tests/accessibility.test.ts`
- Create: `presentation/sleep-time-compute/tests/svgFallback.test.ts`
- Modify: `presentation/sleep-time-compute/src/cli.ts`
- Modify: `presentation/sleep-time-compute/package.json`
- Modify: `presentation/sleep-time-compute/tests/cli.test.ts`

**Interfaces:**
- Produces: editable `build/generated.pptx`, a 12–15-slide editorial-slice
  build when `build --slide-ids-file PATH` is supplied, and build provenance.
- `--slide-ids-file` is an exact CLI contract. It accepts a closed JSON record
  binding the handoff digest and 12–15 unique handoff slide IDs. The renderer
  preserves producer order and all predecessor/successor constraints, rejects
  a foreign handoff digest, unknown/duplicate ID, mode-coverage gap, or
  requested reorder, and records the selector digest in build provenance.

- [ ] **Step 1: Write determinism and relationship tests**

Require stable ZIP order, timestamps, core metadata, object names, internal
relationships, media names, and checksums. Reject externally linked media.
Parser and contract tests require exactly:

```text
build --handoff-bundle DIR [--slide-ids-file JSON] --output PPTX
```

Test 11, 12, 15, and 16 IDs; only 12–15 pass. Test a stale handoff digest,
unknown/duplicate IDs, missing visual mode, and an order that differs from the
producer-authored order; each fails with a stable error code. A valid slice
contains exactly the declared IDs in producer order and leaves the handoff
bytes unchanged.

- [ ] **Step 2: Render native, editable objects**

Render text, individually named shapes, eligible native charts, notes, links,
and SVG evidence graphics. Use source figure preservation in notes/appendix
when the main slide redraws a relationship. Do not rasterize a whole slide.
For every SVG, render a real PNG from the same frozen SVG bytes with
`@resvg/resvg-js`, bind both digests in build provenance, and inject both
OOXML media paths explicitly. Reject `IMG_BROKEN`, a transparent/empty
fallback, SVG/PNG dimension disagreement, or an SVG whose PNG fallback was
generated from different bytes.
Evidence SVG subprocesses run only as
`uv run --frozen python python/render_evidence.py`; the build records the
Python, Matplotlib, `pyproject.toml`, and `uv.lock` digests and rejects a
different interpreter or imported Matplotlib version.

After PptxGenJS rendering, inject scene-node names and descriptions into native
shape `cNvPr` records deterministically. The injector may change accessibility
metadata only and is part of the reproducible generated build, not the terminal
PowerPoint postflight. Each scene node declares
`accessibilityRole=INFORMATIVE|DECORATIVE`; informative objects receive stable
name/title/description values, while decorative objects receive the supported
OOXML decorative marker and are excluded from meaningful reading order. A
missing role, duplicate object name, empty informative description, or
decorative object with factual text fails.

- [ ] **Step 3: Canonicalize OOXML**

Normalize ZIP entry order/timestamps and generated volatile metadata without
changing presentation semantics. This explicitly covers PptxGenJS core
property dates and the `new Date()` values in embedded-chart XLSX properties.
Record canonicalization rules, excluded volatile inputs, and pre/post part
digests. A raw-library digest is diagnostic only; the canonicalized package
digest is authoritative.

- [ ] **Step 4: Verify byte-for-byte reproducibility**

```bash
npm run build
npm test -- \
  canonicalize.test.ts \
  accessibility.test.ts \
  svgFallback.test.ts \
  slice.test.ts \
  cli.test.ts
node dist/cli.js build \
  --handoff-bundle tests/fixtures/g8-handoff \
  --output build/generated.pptx
sha256sum build/generated.pptx > /tmp/stc-pptx-first.sha256
node dist/cli.js build \
  --handoff-bundle tests/fixtures/g8-handoff \
  --output build/generated.pptx
sha256sum build/generated.pptx > /tmp/stc-pptx-second.sha256
diff -u /tmp/stc-pptx-first.sha256 /tmp/stc-pptx-second.sha256
```

Expected: CLI contract tests pass, the digest diff is empty, and undeclared
external relationships equal zero.

- [ ] **Step 5: Commit**

```bash
git -C ../.. add presentation/sleep-time-compute
git -C ../.. commit -m "presentation: render deterministic editable deck"
```

---

### Task 4: Add automated PPTX and PDF QA

**Files:**
- Create: `presentation/sleep-time-compute/src/qa/geometry.ts`
- Create: `presentation/sleep-time-compute/src/qa/content.ts`
- Create: `presentation/sleep-time-compute/src/qa/ooxml.ts`
- Create: `presentation/sleep-time-compute/src/qa/relationships.ts`
- Create: `presentation/sleep-time-compute/src/qa/fonts.ts`
- Create: `presentation/sleep-time-compute/src/qa/pdf.ts`
- Create: `presentation/sleep-time-compute/src/qa/provenance.ts`
- Create: `presentation/sleep-time-compute/python/render_diff.py`
- Create: `presentation/sleep-time-compute/qa/openxml-validator/Dockerfile`
- Create: `presentation/sleep-time-compute/qa/openxml-validator/OpenXmlValidator.csproj`
- Create: `presentation/sleep-time-compute/qa/openxml-validator/Program.cs`
- Create: `presentation/sleep-time-compute/qa/openxml-validator/packages.lock.json`
- Create: `presentation/sleep-time-compute/qa/verapdf/Dockerfile`
- Create: `presentation/sleep-time-compute/qa/validator-images.lock.json`
- Create: `presentation/sleep-time-compute/fonts/manifest.json`
- Vendor exact Task 0 bytes:
  `presentation/sleep-time-compute/fonts/{OFL.txt,NotoSansCJKkr-Regular.otf,NotoSansCJKkr-Bold.otf,NotoSerifCJKkr-SemiBold.otf,NotoSansMonoCJKkr-Regular.otf}`
- Create: `presentation/sleep-time-compute/manifests/toolchain.json`
- Create: `presentation/sleep-time-compute/manifests/assets.json`
- Create: `presentation/sleep-time-compute/tests/content.test.ts`
- Create: `presentation/sleep-time-compute/tests/fonts.test.ts`
- Create: `presentation/sleep-time-compute/tests/ooxml.test.ts`
- Create: `presentation/sleep-time-compute/tests/pdf.test.ts`
- Modify: `presentation/sleep-time-compute/src/cli.ts`
- Modify: `presentation/sleep-time-compute/package.json`
- Modify: `presentation/sleep-time-compute/tests/cli.test.ts`

**Interfaces:**
- Produces: machine-readable `build/qa/{run}/report.json`, a provisional
  generated-deck PDF, renders, contact sheet, font report, and provenance
  report.
- Owns the exact optional QA selector surface
  `qa ... [--slide-ids-file JSON]`. When present, it loads the Task 3 selector,
  rechecks its handoff digest, and requires the PPTX, PDF, rendered pages, and
  build provenance to contain exactly those IDs in producer order.

- [ ] **Step 1: Implement content and geometry lint**

Require:

```text
out_of_bounds=0
overflow=0
unapproved_overlaps=0
duplicate_or_missing_slide_ids=0
unresolved_assets_or_citations=0
missing_notes=0
missing_alt_text=0
type_below_floor=0
must_keep_loss=0
unapproved_editorial_operation=0
variant_or_reorder_violation=0
```

Preflight text fit uses the exact Noto font files/hashes in `fonts/manifest.json`
and Fontkit 2.0.4 `layout()` shaping, including GSUB/GPOS, glyph advances, the
declared size, and line spacing, with no shrink-to-fit. A startup assertion
checks the installed Fontkit package/version and every font digest before
measuring. LibreOffice renders every slide for pixel/clip detection; target
PowerPoint and human review remain the final overflow/line-wrap gate because
renderer metrics can differ.

The standard QA path always revalidates all four Task 0 font SHA-256 values,
PostScript names, `OS/2.fsType=0x0000`, the `OFL-1.1` manifest, and the exact
OFL text digest. It deobfuscates any OOXML embedded-font part before matching
it to the allowlist and rejects every unlisted font byte, proprietary font,
license omission, or manifest drift. There is no `--skip-fonts` mode.

- [ ] **Step 2: Implement OOXML and relationship validation**

Run a digest-pinned .NET 10 SDK image with
`DocumentFormat.OpenXml` exactly 3.5.1, NuGet `RestoreLockedMode=true`, and the
committed `packages.lock.json`. Inspect master/layout/slide relationships,
reject undeclared external links, verify informative/decorative accessibility
metadata, and verify that no repair output is substituted for the generated
file. The validator report records the base image digest, built image ID, SDK
package version/lock digest, target Office file-format version, input PPTX
digest, and every validation diagnostic.

- [ ] **Step 3: Implement LibreOffice/Poppler PDF checks**

Export 16:9 PDF through an isolated LibreOffice profile using the
`impress_pdf_Export` filter with `PDFUACompliance=true`,
`UseTaggedPDF=true`, and `EmbedStandardFonts=true`. Record the complete filter
JSON and LibreOffice profile digest. Check page count, 960×540-point page boxes,
embedded fonts, document language/title, tags, text extraction, reading order,
alt descriptions, and raster renders. Run veraPDF 1.30 from the image digest
recorded in `validator-images.lock.json`.

If veraPDF fails, label the PDF `visual-only`; do not claim PDF/UA conformance.
Machine conformance never replaces human reading-order and description review.

Emit presenter notes, source/derived asset inventory, font files and licenses
when redistributable (otherwise installation hashes), embedded-font-part
report, relationship report, OOXML report, render diff, contact sheet,
toolchain/container digests, and veraPDF report. Pin both validator image
digests in `validator-images.lock.json`; floating container tags fail QA.

- [ ] **Step 4: Run the full fixture QA**

```bash
npm run build
npm test -- \
  content.test.ts \
  fonts.test.ts \
  ooxml.test.ts \
  pdf.test.ts \
  cli.test.ts
npm run qa -- \
  --handoff tests/fixtures/g8-handoff \
  --pptx build/generated.pptx \
  --out build/qa/fixture \
  --pdf-output build/qa/fixture/fixture.pdf
```

`package.json` wires `npm run qa -- ...` exactly to
`node dist/cli.js qa ...`. Expected: all zero-error counters above, correct page
size/count, embedded PDF fonts, stable render, Open XML pass, and an explicit
PDF/UA result tied to `build/qa/fixture/fixture.pdf`.
Parser tests accept zero or one `--slide-ids-file`, reject duplicates, and QA
mutation tests reject a missing/extra/reordered slide, a selector/provenance
digest mismatch, a PDF page-count mismatch, and render order differing from
producer order.

- [ ] **Step 5: Commit**

```bash
git -C ../.. add presentation/sleep-time-compute
git -C ../.. commit -m "presentation: add pptx and pdf quality gates"
```

---

### Task 5: Build the real G8 deck and obtain editorial approval

**Files:**
- Consume read-only: `research/sleep-time-compute/publication/handoff/stc-paper-v1.0.0/`
- Consume read-only: `docs/presentation/pptx-reviewer-trust-root.json`
- Create: `presentation/sleep-time-compute/reports/representative-slice.json`
- Create: `presentation/sleep-time-compute/reports/representative-slice-subject.json`
- Create: `presentation/sleep-time-compute/reports/representative-slice-powerpoint-operator-attestation.json`
- Create: `presentation/sleep-time-compute/reports/representative-slice-content-owner-attestation.json`
- Create: `presentation/sleep-time-compute/reports/representative-slice-visual-editor-attestation.json`
- Create: `presentation/sleep-time-compute/reports/editorial-approval.json`
- Create: `presentation/sleep-time-compute/reports/full-deck-review-rubric.json`
- Create: `presentation/sleep-time-compute/reports/full-deck-subject.json`
- Create: `presentation/sleep-time-compute/reports/full-deck-content-owner-attestation.json`
- Create: `presentation/sleep-time-compute/reports/full-deck-visual-editor-attestation.json`
- Create: `presentation/sleep-time-compute/reports/full-deck-approval.json`
- Create: `presentation/sleep-time-compute/build/representative.pptx`
- Create:
  `presentation/sleep-time-compute/build/qa/representative/representative-powerpoint.pdf`
- Create: `presentation/sleep-time-compute/build/generated.pptx`
- Create: `presentation/sleep-time-compute/build/qa/generated/generated.pdf`
- Create: `presentation/sleep-time-compute/src/editorial_approval.ts`
- Create: `presentation/sleep-time-compute/tests/editorial_approval.test.ts`
- Modify: `presentation/sleep-time-compute/src/cli.ts`
- Modify: `presentation/sleep-time-compute/package.json`
- Modify: `presentation/sleep-time-compute/tests/cli.test.ts`

**Interfaces:**
- Consumes: actual G8 handoff and immutable assets.
- Produces: the reproducible generated parent deck and its provisional QA PDF;
  verified representative-slice and full-deck approvals; it does not yet
  produce the display-authoritative PDF.
- Owns the exact `editorial-subject`, `verify-editorial-approval`,
  `full-deck-subject`, and `verify-full-deck-approval` signatures in Command
  Surface. `editorial_approval.test.ts` and `cli.test.ts` pin their canonical
  leaf sets, required options, distinct-role trust checks, and mutation
  failures. The representative subject has five immutable artifact leaves:
  PPTX, PowerPoint-exported PDF, selector, QA report, and contact sheet.

- [ ] **Step 1: Render and approve the representative slice**

Render 12–15 slides selected to cover all visual modes and the densest
evidence. `representative-slice.json` binds the exact handoff digest and slide
IDs; its order must match producer order. Render and QA it before rendering the
full handoff:

```bash
npm run build
npm test -- editorial_approval.test.ts cli.test.ts
node dist/cli.js build \
  --handoff-bundle ../../research/sleep-time-compute/publication/handoff/stc-paper-v1.0.0 \
  --slide-ids-file reports/representative-slice.json \
  --output build/representative.pptx
```

Pause. The trusted `POWERPOINT_OPERATOR` opens that exact PPTX without repair
and exports `build/qa/representative/representative-powerpoint.pdf`. Then run
this fresh package-root block:

```bash
: "${STC_TRUST_ROOT_COMMIT:?set to the earlier Task 0 trust-root commit}"
npm run qa -- \
  --handoff ../../research/sleep-time-compute/publication/handoff/stc-paper-v1.0.0 \
  --pptx build/representative.pptx \
  --slide-ids-file reports/representative-slice.json \
  --out build/qa/representative \
  --pdf-input build/qa/representative/representative-powerpoint.pdf
node dist/cli.js editorial-subject \
  --pptx build/representative.pptx \
  --powerpoint-pdf build/qa/representative/representative-powerpoint.pdf \
  --slide-ids-file reports/representative-slice.json \
  --qa build/qa/representative/report.json \
  --contact-sheet build/qa/representative/contact-sheet.png \
  --trusted-keys ../../docs/presentation/pptx-reviewer-trust-root.json \
  --trust-root-commit "$STC_TRUST_ROOT_COMMIT" \
  --output reports/representative-slice-subject.json
```

Pause. The PowerPoint operator, content owner, and independent visual editor
sign the canonical subject with their distinct role keys. The latter two
inspect the exact
subject-bound PPTX, PowerPoint PDF, and slide-ID JSON plus narrative pacing,
hierarchy, typography, dense evidence, the exact QA report/contact sheet, and
the low-AI-feel rejection list.
They create separate Ed25519 attestations with distinct keys. Then verify all
three roles:

```bash
node dist/cli.js verify-editorial-approval \
  --subject reports/representative-slice-subject.json \
  --powerpoint-operator-attestation \
    reports/representative-slice-powerpoint-operator-attestation.json \
  --content-owner-attestation \
    reports/representative-slice-content-owner-attestation.json \
  --visual-editor-attestation \
    reports/representative-slice-visual-editor-attestation.json \
  --trusted-keys ../../docs/presentation/pptx-reviewer-trust-root.json \
  --output reports/editorial-approval.json
```

The verifier rehashes every direct subject leaf, requires all three
`decision=PASS` attestations, checks role/key separation and signatures, and proves the
subject handoff and selector digests match the render provenance. Failure stops
before the complete-deck command. The preceding QA invocation proves the PPTX
slide IDs, PowerPoint PDF page count, rendered-page order, and build-provenance
selector digest exactly match `representative-slice.json` in producer order.
Parser tests require exactly one `--qa` and `--contact-sheet`; changing either
leaf after signature fails verification.

- [ ] **Step 2: Render the complete deck**

```bash
: "${STC_TRUST_ROOT_COMMIT:?set to the earlier Task 0 trust-root commit}"
node dist/cli.js build \
  --handoff-bundle ../../research/sleep-time-compute/publication/handoff/stc-paper-v1.0.0 \
  --output build/generated.pptx
npm run qa -- \
  --handoff ../../research/sleep-time-compute/publication/handoff/stc-paper-v1.0.0 \
  --pptx build/generated.pptx \
  --out build/qa/generated \
  --pdf-output build/qa/generated/generated.pdf
node dist/cli.js full-deck-subject \
  --pptx build/generated.pptx \
  --provisional-pdf build/qa/generated/generated.pdf \
  --contact-sheet build/qa/generated/contact-sheet.png \
  --qa build/qa/generated/report.json \
  --pacing-rubric reports/full-deck-review-rubric.json \
  --trusted-keys ../../docs/presentation/pptx-reviewer-trust-root.json \
  --trust-root-commit "$STC_TRUST_ROOT_COMMIT" \
  --output reports/full-deck-subject.json
```

- [ ] **Step 3: Pause for separate full-deck approval and complete human QA**

The representative-slice approval does not approve unseen slides. After the
complete build above, the content owner and visual editor inspect the exact
full-deck subject and independently sign it. The subject binds
`generated.pptx`, its provisional PDF, all-page contact sheet, QA report,
pacing rubric, and earlier trust-root commit/digest. Record:

- contact-sheet pacing and hierarchy;
- target PowerPoint open with no repair, removed-content, font, or
  compatibility warning;
- target-PowerPoint render diff for SVG, chart, font, line wrap, and geometry;
- Accessibility Checker and Reading Order;
- representative screen-reader review;
- projector/reduced-zoom legibility;
- grayscale and color-vision review;
- independent content-owner and visual-editor sign-off.

Then verify in a new package-root block before any Task 6 postflight:

```bash
node dist/cli.js verify-full-deck-approval \
  --subject reports/full-deck-subject.json \
  --content-owner-attestation \
    reports/full-deck-content-owner-attestation.json \
  --visual-editor-attestation \
    reports/full-deck-visual-editor-attestation.json \
  --trusted-keys ../../docs/presentation/pptx-reviewer-trust-root.json \
  --output reports/full-deck-approval.json
```

Any missing slide, changed subject leaf, unknown/self-added key, role/key
reuse, trust-root drift, or non-`PASS` decision blocks Task 6.

- [ ] **Step 4: Freeze the generated parent**

Record generated PPTX/provisional PDF, handoff, toolchain, font/license,
render-set, contact-sheet, QA report, full-deck subject, both attestations, and
verified full-deck approval checksums. `generated.pptx` becomes the immutable
reproducibility parent for postflight. The provisional PDF is a QA artifact
only and may not be published as `display.pdf`.

- [ ] **Step 5: Commit the durable signed leaves and implementation**

```bash
git -C ../.. add -f \
  presentation/sleep-time-compute/build/representative.pptx \
  presentation/sleep-time-compute/build/generated.pptx \
  presentation/sleep-time-compute/build/qa/representative \
  presentation/sleep-time-compute/build/qa/generated
git -C ../.. add presentation/sleep-time-compute
for STC_DURABLE_LEAF in \
  presentation/sleep-time-compute/build/representative.pptx \
  presentation/sleep-time-compute/build/generated.pptx \
  presentation/sleep-time-compute/build/qa/representative/report.json \
  presentation/sleep-time-compute/build/qa/representative/contact-sheet.png \
  presentation/sleep-time-compute/build/qa/representative/representative-powerpoint.pdf \
  presentation/sleep-time-compute/build/qa/generated/report.json \
  presentation/sleep-time-compute/build/qa/generated/contact-sheet.png \
  presentation/sleep-time-compute/build/qa/generated/generated.pdf
do
  git -C ../.. ls-files --error-unmatch "$STC_DURABLE_LEAF"
done
git -C ../.. commit -m "presentation: validate G8 research deck"
```

The exact full-deck slide count is determined by the frozen handoff. The
renderer does not pad the deck to a target number.

---

### Task 6: Validate and seal terminal PowerPoint postflight

**Files:**
- Consume read-only: `docs/presentation/pptx-reviewer-trust-root.json`
- Consume read-only:
  `docs/presentation/pptx-reviewer-trust-root-approval.json`
- Consume read-only: `docs/presentation/pptx-spike-manifest.json`
- Consume read-only: `docs/presentation/pptx-spike.schema.json`
- Consume read-only: `docs/presentation/verify-pptx-spike.mjs`
- Consume read-only: `docs/presentation/pptx-spike-fixtures/*`
- Create: `presentation/sleep-time-compute/src/qa/postflight.ts`
- Create: `presentation/sleep-time-compute/src/qa/powerpoint_attestation.ts`
- Create: `presentation/sleep-time-compute/tests/postflight.test.ts`
- Create: `presentation/sleep-time-compute/tests/powerpoint_attestation.test.ts`
- Modify: `presentation/sleep-time-compute/src/cli.ts`
- Modify: `presentation/sleep-time-compute/package.json`
- Modify: `presentation/sleep-time-compute/tests/cli.test.ts`
- Create: `presentation/sleep-time-compute/build/release.pptx`
- Create: `presentation/sleep-time-compute/build/release.pdf`
- Create: `presentation/sleep-time-compute/build/display.pdf`
- Create: `presentation/sleep-time-compute/build/release-checksums.sha256`
- Create: `presentation/sleep-time-compute/build/qa/postflight.json`
- Consume verified: `presentation/sleep-time-compute/reports/full-deck-approval.json`
- Create: `presentation/sleep-time-compute/reports/powerpoint-reopen.json`
- Create: `presentation/sleep-time-compute/reports/powerpoint-accessibility.json`
- Create: `presentation/sleep-time-compute/reports/powerpoint-export-options.json`
- Create: `presentation/sleep-time-compute/reports/powerpoint-postflight-subject.json`
- Create: `presentation/sleep-time-compute/reports/powerpoint-postflight-operator-attestation.json`
- Create: `presentation/sleep-time-compute/reports/powerpoint-postflight-qa.json`
- Create: `presentation/sleep-time-compute/manifests/release.json`

**Interfaces:**
- Allows only: font embedding, reading order, object name/description, or
  accessibility metadata changes made in target desktop PowerPoint.
- Produces: a sealed `release.pptx`, a release-derived
  display-authoritative PDF, complete release QA, and a release manifest.
- Consumes the immutable Task 0 trust root and the verified Task 5 full-deck
  approval. The final seal independently verifies a `POWERPOINT_OPERATOR`
  signature over the complete postflight subject with explicit
  `--trusted-keys`; editorial approval cannot substitute for it.
- Owns the exact `powerpoint-postflight-subject`,
  `verify-powerpoint-operator`, and extended `seal` signatures in Command
  Surface. `powerpoint_attestation.test.ts`, `postflight.test.ts`, and
  `cli.test.ts` pin canonicalization, trusted role, required options, and every
  signed-leaf mutation failure.

- [ ] **Step 1: Write allowlist and rejection tests**

First prove that the repository-bundled capability evidence is complete:

```bash
node ../../docs/presentation/verify-pptx-spike.mjs \
  --manifest ../../docs/presentation/pptx-spike-manifest.json \
  --fixtures ../../docs/presentation/pptx-spike-fixtures \
  --trusted-keys ../../docs/presentation/pptx-reviewer-trust-root.json \
  --offline
```

The validator must prove:

- text nodes unchanged;
- chart data unchanged;
- media bytes unchanged;
- geometry unchanged;
- the multiset of canonical per-shape subtrees unchanged when only reading
  order changes;
- slide count/order/visibility and master/layout/theme semantics unchanged;
- only declared font parts and accessibility/name/description/order metadata
  changed.

Any factual, chart-data, geometry, or design delta fails.
Tests replay the three positive and four negative artifacts recorded by
`pptx-spike-manifest.json`; the positive cases must pass individually and every
positive must contain its recorded nonempty expected change class. Every
negative case must fail with its recorded expected stable error code. The
tests resolve only the repo-relative fixture paths in the offline-valid
manifest; they never read Task 0's temporary directory or a mutable CAS.
Removing a fixture, changing one byte without updating its signed manifest, or
substituting an absolute path fails before semantic allowlist evaluation. The
observed PowerPoint part changes constrain the semantic allowlist but never
authorize an otherwise unexplained changed part.
Seal tests also reject a release PDF whose recorded source-PPTX digest is not
the validated `release.pptx`, non-byte-identical release/display PDFs, a
nonpassing or digest-stale QA/postflight/human/render report, and any missing
direct input to the release manifest.
Parser tests require exactly one `--trusted-keys`, `--full-deck-approval`,
`--powerpoint-subject`, and `--powerpoint-operator-attestation` on `seal` and
reject duplicates. Mutation tests reject a self-added or unknown key, a
`CONTENT_OWNER`/`VISUAL_EDITOR` key reused as `POWERPOINT_OPERATOR`, a
trust-root commit that is not strictly earlier, current trust-root byte drift,
and mutation of any generated/release/postflight/reopen/accessibility/PDF/export
options leaf after signature.

- [ ] **Step 2: Implement OOXML package differencing**

Record both checksums, operator, PowerPoint version, timestamp, and a
part-by-part change log. Normalize ZIP entry order/timestamps before
comparison, then compare canonical semantic projections for text, chart
workbooks/caches, media, geometry, relationships, slide order/visibility,
masters/layouts/themes, notes, links, and accessibility metadata. No changed
OOXML part is ignored merely because PowerPoint commonly rewrites it.

- [ ] **Step 3: Apply the human PowerPoint postflight and validate**

An identified operator opens the exact `generated.pptx` digest in the target
desktop PowerPoint version, applies only the needed allowed edits, saves as
`release.pptx`, closes it, and reopens `release.pptx` without a repair,
removed-content, compatibility, or font warning. The operator then runs
Accessibility Checker and Reading Order and records results against the release
digest in `reports/powerpoint-reopen.json` and
`reports/powerpoint-accessibility.json` before any automated seal command runs.

```bash
npm run build
npm test -- \
  postflight.test.ts \
  powerpoint_attestation.test.ts \
  cli.test.ts
node dist/cli.js verify-full-deck-approval \
  --subject reports/full-deck-subject.json \
  --content-owner-attestation \
    reports/full-deck-content-owner-attestation.json \
  --visual-editor-attestation \
    reports/full-deck-visual-editor-attestation.json \
  --trusted-keys ../../docs/presentation/pptx-reviewer-trust-root.json \
  --output reports/full-deck-approval.json
node dist/cli.js validate-postflight \
  --generated build/generated.pptx \
  --release build/release.pptx \
  --report build/qa/postflight.json
```

Expected: `PASS`; otherwise discard the edited artifact and return to the
handoff/generator.

- [ ] **Step 4: Re-run the complete QA surface**

After the postflight validator passes, target PowerPoint exports
`build/release.pdf` from the reopened, postflight-validated `release.pptx` with
best-quality, document-properties, tagged-structure, and accessibility options
enabled. The operator records the PowerPoint version, source PPTX digest, PDF
digest, and complete export options in
`reports/powerpoint-export-options.json`. The signed verification result
`reports/powerpoint-postflight-qa.json` is produced only later from the
canonical subject and operator attestation; automation must not pre-create it.

Run Open XML validation, relationship/font scan, provenance/content/geometry
lint, LibreOffice comparison renders, PDF checks against the provided
PowerPoint-exported PDF, generated-versus-release render diff, and the recorded
target-PowerPoint no-repair reopen. Repeat Accessibility Checker, Reading
Order, and representative screen-reader review after the edit. Any unexplained
render pixel, line-wrap, chart, relationship, font, structure, or reading-order
delta fails.

```bash
npm run qa -- \
  --handoff ../../research/sleep-time-compute/publication/handoff/stc-paper-v1.0.0 \
  --pptx build/release.pptx \
  --out build/qa/release \
  --pdf-input build/release.pdf
node dist/cli.js compare-renders \
  --generated build/qa/generated/renders \
  --release build/qa/release/renders \
  --output build/qa/release-render-diff.json
```

- [ ] **Step 5: Seal the release**

First create the canonical operator subject. It binds the exact
`generated.pptx`, `release.pptx`, semantic postflight report, no-repair reopen
record, Accessibility Checker/Reading Order record, PowerPoint-exported
`release.pdf`, export-options record, and earlier trust-root commit/digest:

```bash
: "${STC_TRUST_ROOT_COMMIT:?set to the earlier Task 0 trust-root commit}"
node dist/cli.js powerpoint-postflight-subject \
  --generated build/generated.pptx \
  --release build/release.pptx \
  --postflight build/qa/postflight.json \
  --reopen-report reports/powerpoint-reopen.json \
  --accessibility-report reports/powerpoint-accessibility.json \
  --release-pdf build/release.pdf \
  --export-options reports/powerpoint-export-options.json \
  --trusted-keys ../../docs/presentation/pptx-reviewer-trust-root.json \
  --trust-root-commit "$STC_TRUST_ROOT_COMMIT" \
  --output reports/powerpoint-postflight-subject.json
```

Pause. The trusted `POWERPOINT_OPERATOR` signs that canonical subject. Then
verify it in a fresh package-root shell:

```bash
node dist/cli.js verify-powerpoint-operator \
  --subject reports/powerpoint-postflight-subject.json \
  --operator-attestation \
    reports/powerpoint-postflight-operator-attestation.json \
  --trusted-keys ../../docs/presentation/pptx-reviewer-trust-root.json \
  --output reports/powerpoint-postflight-qa.json
```

`generated.pptx` remains the reproducibility target. `release.pptx` is still a
release candidate until `seal` succeeds; that command promotes the PPTX and
its PowerPoint-exported `release.pdf` together. `display.pdf` is the
byte-identical publication name. The command re-resolves the postflight,
release QA, verified full-deck approval, PowerPoint subject/attestation/QA,
immutable trust-root commit and digest, render-diff, source-PPTX, and both PDF
digests, then writes `manifests/release.json` last. The manifest records both
PPTX checksums, both PDF names and shared checksum, semantic OOXML delta,
operator/tool version, handoff/index/toolchain/font digests, and every
revalidation and human-review report digest.

```bash
cp build/release.pdf build/display.pdf
cmp build/release.pdf build/display.pdf
node dist/cli.js seal \
  --generated build/generated.pptx \
  --release build/release.pptx \
  --release-pdf build/release.pdf \
  --display-pdf build/display.pdf \
  --qa build/qa/release \
  --postflight build/qa/postflight.json \
  --full-deck-approval reports/full-deck-approval.json \
  --powerpoint-subject reports/powerpoint-postflight-subject.json \
  --powerpoint-operator-attestation \
    reports/powerpoint-postflight-operator-attestation.json \
  --powerpoint-qa reports/powerpoint-postflight-qa.json \
  --trusted-keys ../../docs/presentation/pptx-reviewer-trust-root.json \
  --render-diff build/qa/release-render-diff.json \
  --output manifests/release.json
sha256sum build/release.pptx build/release.pdf build/display.pdf \
  > build/release-checksums.sha256
git -C ../.. add -f \
  presentation/sleep-time-compute/build/release.pptx \
  presentation/sleep-time-compute/build/release.pdf \
  presentation/sleep-time-compute/build/display.pdf \
  presentation/sleep-time-compute/build/release-checksums.sha256 \
  presentation/sleep-time-compute/build/qa/release \
  presentation/sleep-time-compute/build/qa/postflight.json \
  presentation/sleep-time-compute/build/qa/release-render-diff.json
git -C ../.. add presentation/sleep-time-compute
for STC_DURABLE_LEAF in \
  presentation/sleep-time-compute/build/release.pptx \
  presentation/sleep-time-compute/build/release.pdf \
  presentation/sleep-time-compute/build/display.pdf \
  presentation/sleep-time-compute/build/release-checksums.sha256 \
  presentation/sleep-time-compute/build/qa/postflight.json \
  presentation/sleep-time-compute/build/qa/release/report.json \
  presentation/sleep-time-compute/build/qa/release-render-diff.json
do
  git -C ../.. ls-files --error-unmatch "$STC_DURABLE_LEAF"
done
git -C ../.. commit -m "presentation: seal accessible release deck"
```

## Dependency Graph

```text
Task 0 spikes
  └── Task 1 handoff contract
        └── Task 2 editorial system
              └── Task 3 deterministic renderer
                    └── Task 4 automated QA

actual research G8 handoff ───────────────┐
Task 4 ──────────────────────────────────┴── Task 5 real deck
Task 5 representative approval → generated parent → full-deck approval
  └── target PowerPoint edit/reopen/accessibility
        └── semantic postflight validation
              └── target PowerPoint release.pdf export
                    └── complete release QA/render comparison
                          └── trusted POWERPOINT_OPERATOR attestation
                                └── Task 6 seal release.pptx + release.pdf/display.pdf
```

Tasks 0–4 may proceed independently of research results. Tasks 5–6 cannot begin
from factual content until the real G8 handoff exists.

## Definition of Done

- Every slide traces to one frozen G8 handoff revision.
- The representative slice binds exactly its PPTX, PowerPoint-exported PDF,
  producer-order slide-ID JSON, QA report, and contact sheet and has distinct
  PowerPoint-operator, content, and visual approval from the immutable earlier
  trust root.
- The complete generated deck has a later, separate content-owner and
  visual-editor approval over its PPTX, provisional PDF, all-page contact
  sheet, QA report, and pacing rubric; slice approval never approves unseen
  slides.
- The renderer invents no factual copy or chart data.
- The complete deck avoids every low-AI-feel rejection pattern.
- Type, contrast, geometry, overflow, provenance, notes, alt text, and
  reading-order gates pass.
- Target PowerPoint opens the generated deck without repair or compatibility
  warning, then reopens the postflight deck with the same result.
- Every font byte is one of the four exact OFL-1.1 Noto files, with verified
  SHA-256 and `OS/2.fsType=0x0000`; no proprietary font byte is vendored,
  embedded, or released.
- PDF/UA is claimed only when veraPDF and human accessibility review pass.
- `generated.pptx` is reproducible; `release.pptx` has only an allowlisted,
  logged terminal delta.
- `display.pdf` is byte-identical to the validated `release.pdf` exported from
  the sealed `release.pptx`; the provisional generated-deck PDF is never
  published under that name.
- The final postflight subject binds generated/release PPTX, postflight,
  reopen, accessibility, PowerPoint PDF/export-options, and immutable
  trust-root digests and is signed by the trusted `POWERPOINT_OPERATOR`; `seal`
  revalidates it with explicit `--trusted-keys`.
- Every signed direct leaf is explicitly tracked despite protective build
  ignores; a clean checkout can reverify the slice, full-deck, postflight, and
  seal records and contains the editable `release.pptx`.
- Any post-freeze factual change creates a new handoff revision and a new build.
