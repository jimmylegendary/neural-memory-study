# Desktop handoff: generated Attn-vs-HOPE workbook

This handoff concerns the reproducible **generated 11-tab workbook** at
`experiments/E5-attn-vs-hope/sheet/Attn-vs-HOPE.xlsx`.

It is not the user's native one-tab live Sheet archived at
`research/attn-vs-hope/Attn-vs-HOPE.xlsx`. Do not overwrite, import over, or
otherwise modify that live-Sheet archive. Import the generated workbook into a
new Google spreadsheet.

## Prerequisites

Run commands from the repository root. The only workbook authoring dependency is
the bundled `@oai/artifact-tool` runtime:

```bash
ARTIFACT_NODE_MODULES="<node_modules path returned by load_workspace_dependencies>"
test -d "$ARTIFACT_NODE_MODULES/@oai/artifact-tool"
ln -sfn "$ARTIFACT_NODE_MODULES" experiments/E5-attn-vs-hope/sheet/node_modules
```

In Codex Desktop, call `load_workspace_dependencies` for the spreadsheet
runtime and copy its returned `node_modules` path into the environment variable.
The path is host/runtime-version specific; no absolute home-directory path is
canonical or checked into this handoff.

The symlink is ignored. The builder emits an actionable error if the package
cannot be imported. Do not install or substitute another spreadsheet library.

## Build, inspect, render, and export

### Rebuild, inspect, render all tabs, and export the XLSX

```bash
node experiments/E5-attn-vs-hope/sheet/build_workbook.mjs --build \
  | tee experiments/E5-attn-vs-hope/sheet/inspect-build.log
```

This single reproducible command:

1. reads `experiments/E5-attn-vs-hope/results.json` through
   `new URL("../results.json", import.meta.url)`;
2. creates exactly the 11 named tabs;
3. inspects key comparison, timing, sweep, and QA ranges;
4. scans for formula errors;
5. renders every tab to the ignored `sheet/previews/` directory; and
6. exports `sheet/Attn-vs-HOPE.xlsx`.

The inspect logs and automatic `*.inspect.ndjson` dump are ignored.

### Inspect the already-exported workbook without rebuilding

```bash
node experiments/E5-attn-vs-hope/sheet/build_workbook.mjs --inspect-only \
  | tee experiments/E5-attn-vs-hope/sheet/inspect-final.log
```

Expected evidence includes:

- 11 sheet records and exactly three chart drawings;
- 16 `90_QA` reconciliation rows with status `PASS`; and
- `FORMULA_ERROR_SCAN` followed by `Cell search matched 0 entries.`

### Render the already-exported workbook without rebuilding

```bash
node experiments/E5-attn-vs-hope/sheet/build_workbook.mjs --render-only
```

Expected preview inventory:

```text
00_Guide.png
01_Inputs.png
02_HW.png
10_AttnMoE_Prefill.png
11_AttnMoE_Decode.png
20_HOPE_Prefill.png
21_HOPE_Decode.png
30_Compare.png
40_Sweeps.png
90_QA.png
99_Sources.png
```

### Export location and checksum

The `--build` command performs the export. Verify the exact artifact with:

```bash
sha256sum experiments/E5-attn-vs-hope/sheet/Attn-vs-HOPE.xlsx
```

Do not create alternate XLSX variants; rebuild the canonical path.

## Structural test

Run the test command from the task contract:

```bash
python -m unittest -v experiments/E5-attn-vs-hope/sheet/test_workbook_source.py
```

If the Desktop environment exposes Python only as `python3`, use the equivalent:

```bash
python3 -m unittest -v experiments/E5-attn-vs-hope/sheet/test_workbook_source.py
```

The suite reads the XLSX as OOXML ZIP/XML and verifies exact sheet names/order,
no external links, substantive/transitive formula lineage (including summary
row 3), absence of zero-multiplied lineage sentinels, formula-backed numeric
calculation cells, design-section 8 and comparison headers, complete current
reference-HW/model-constant coverage, position provenance, 16 formula-backed QA PASS
rows, a bounded OOXML formula-error scan, and the exact title/type/data purpose
of all three native charts.

## Google Sheets import

1. Open [Google Sheets](https://sheets.google.com) and create a blank spreadsheet.
2. Choose **File → Import → Upload**.
3. Upload
   `experiments/E5-attn-vs-hope/sheet/Attn-vs-HOPE.xlsx`.
4. Select **Create new spreadsheet**. Do not select a mode that replaces the
   user's existing live Sheet.
5. Confirm that all 11 tabs import in the documented order.
6. Confirm the two charts on `30_Compare` and the one chart on `40_Sweeps`.

Google Sheets may rewrite Excel formula syntax internally; the displayed values
and lineage checks below are the acceptance contract.

## Post-import checks in Google Sheets

### Formula errors

Use **Edit → Find and replace**, choose **All sheets**, enable **Search using
regular expressions**, and search for:

```text
#REF!|#DIV/0!|#VALUE!|#NAME\?|#N/A
```

Expected result: no matches.

### Reconciliation and key outputs

- `90_QA!F6:F21`: every cell must display `PASS`.
- `90_QA!A48:F54`: native operational outputs remain separate from generated
  formulas; nonzero deltas must read `ASSUMPTION DIFFERENCE`, not be hidden.
- `21_HOPE_Decode!B6:B8`: forward-only, online-overhead, and including-online
  ITL must remain populated and ordered consistently with their labels.
- `40_Sweeps!A6:G11` and `A17:G22`: context and batch sweeps, including the
  formula-derived crossover flags in column G, must remain populated.
- `30_Compare!A6:M9`: formulas must remain populated, including separate HBM
  read/write/total and ridge metrics.
- `30_Compare!A18:C23`: Full Attention and HOPE forward/including-online TTFT
  and ITL must remain populated; HOPE TTFT alone applies the chunk multiplier.
- `30_Compare!A13:B14`: context and batch crossover summaries must remain
  formula-linked to `40_Sweeps` (the current verified sweep displays
  `None in sweep` for both).

If a formula error or non-PASS QA row appears after import, keep the local XLSX
unchanged, record the exact tab/cell, and diagnose the Google-Sheets conversion
before applying any edit to the imported copy.

## Reproducibility boundary

- Read-only upstream input: `experiments/E5-attn-vs-hope/results.json`.
- Generated source/artifact: `experiments/E5-attn-vs-hope/sheet/`.
- Native one-tab live Sheet: `research/attn-vs-hope/` — separate and untouched.
- Formula lineage: `sheet/FORMULA-MAP.md`.
