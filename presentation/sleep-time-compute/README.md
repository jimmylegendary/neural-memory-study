# Sleep-Time Compute research deck

This directory contains a generated, editable Korean PowerPoint deck and its
automated QA artifacts. The deck is a position/research-agenda presentation;
it does not present unrun benchmark experiments as results.

```bash
npm ci
npm test
npm run build
npm run qa
```

Render with LibreOffice:

```bash
libreoffice --headless --convert-to pdf --outdir build \
  build/sleep-time-compute-research.pptx
mkdir -p build/renders
pdftoppm -png -r 110 build/sleep-time-compute-research.pdf \
  build/renders/slide
python3 src/make-contact-sheet.py
```

The PowerPoint uses native text, shapes, connectors, and charts wherever
practical; it contains no flattened slide screenshots. All scientific plots in
this version are schematic and are labeled as theory candidates rather than
observations. Every slide has a speaker note that states its claim boundary.

`reports/build-qa.json` captures source-level bounds and typography checks.
`reports/ooxml-qa.json` checks the generated package for slide and note counts,
non-empty notes, missing internal relationships, external relationships,
invalid geometry presets, and 16:9 dimensions. The contact sheet is the
whole-deck visual review surface.

Automated Linux QA is diagnostic. A final Microsoft PowerPoint open/repair,
font, editability, reading-order, and accessibility review still requires a
human operator on the target PowerPoint version.
