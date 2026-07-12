# Build & QA method — lessons that must hold on every rebuild

Two process failures shipped defects in the first pass. Both are now fixed at the **method** level,
not just patched in the artifact.

## 1. The LaTeX log is NOT a valid overflow gate for this document
In a `luatexja` CJK document, long unbreakable **Latin/code runs** — slash/middot/colon-joined name
lists (`GLA/DeltaNet/GDN`, `GLA·DeltaNet·GDN·RWKV`), arXiv IDs (`arXiv:2501.00663`), code identifiers
(`update_in_place`, `mark_dirty/writeback`) — run past the right margin **without emitting an
`Overfull \hbox` warning**. The first pass trusted "0 overfull hbox" and shipped **33 pages** with
visible right-margin overflow. A 24-page visual sample also missed them.

**Method fix (mandatory, deterministic, full-document):**
- `build/overflow_gate.py <pdf>` rasterizes **every** page at 150 dpi and fails (exit 1) if any page
  has ink in the right-margin band (past 18.8 cm on a 21 cm page, 2.5 cm geometry margin). Run it on
  every build; it is the gate, not the LaTeX log.
- `build/breakable.lua` (pandoc `--lua-filter`) inserts `\allowbreak` after `/ · : _ .` in prose
  tokens and inline code so those runs wrap. This is the systematic fix (not per-instance patching).
- Wide **display equations** (equation body + trailing `\tag`/annotation exceeding `\linewidth`) are
  wrapped `\adjustbox{max width=\linewidth}{$\displaystyle … $}` with the `\tag` kept **outside** the
  box — shrinks only when needed, preserves tags/`cases`.
- Template safety net: `\emergencystretch=3em`, `\tolerance=2000`.

## 2. Inventory the tool's skills before using it
The first pass wrote a study/survey paper without ever checking Veridraft's skill set, and so missed
`reference-figure-extractor` — the skill that pulls **source-paper figures** (image · in-text
reference context · description, with third-party provenance) precisely for study/survey grounding.
A study paper explaining six papers should show those papers' own figures. Before using any tool,
list its skills (`ls ~/repos/veridraft/skills/`) and match them to the task.

## Canonical rebuild
```bash
# (re)assemble build/BOOK.md in reading order with absolute figure paths, then:
cd build
pandoc BOOK.md -o BOOK.pdf --pdf-engine=lualatex --template=template.tex \
  --lua-filter=breakable.lua --toc --top-level-division=chapter \
  --resource-path=/home/jimmy/repos/neural-memory-study:/home/jimmy/repos/neural-memory-study/figures
python3 overflow_gate.py BOOK.pdf          # MUST print PASS
grep -c 'Missing character' build_err.txt  # MUST be 0
```
