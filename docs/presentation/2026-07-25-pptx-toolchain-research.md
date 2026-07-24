# Neural Memory PPTX Toolchain and Editorial Design Research

- Date: 2026-07-25
- Status: proposed separate presentation track
- Dependency: a frozen, hashed research `presentation-handoff.json`
- Non-goal: slide production must not block or alter the research evidence
  pipeline

## Decision

Use a manifest-driven TypeScript/PptxGenJS pipeline:

```text
frozen research handoff
→ typed slide manifest
→ editorial storyboard
→ PptxGenJS masters/placeholders/native shapes
  + Matplotlib SVG evidence graphics
→ geometry/accessibility lint
→ LibreOffice/Poppler render QA
→ target PowerPoint open/accessibility review
→ editable PPTX + display-authoritative PDF
```

Keep `python-pptx` as the low-migration fallback. PptxGenJS may be installed in
an isolated temporary spike workspace before approval; it enters the project
and lockfile only after the spikes pass. Do not make HTML/PDF-to-PPTX
or one-full-slide-image export the canonical path: those approaches weaken
editability, accessibility, reading order, and native PowerPoint behavior.

## Why the current deck feels generated

The July deck builder is reproducible, but its visual grammar and OOXML
structure are not suitable for the new research deck:

- `seminar/build_deck.py` generates 91 slides entirely from the `Blank` layout;
- `seminar/pptx_lib.py` manually positions nearly every title, card, arrow, and
  textbox rather than using true masters and placeholders;
- the current deck contains 998 shapes, including 90 pictures, but no native
  chart and no native table;
- median text density is 654 characters per slide and the maximum is 1,191;
- 658 text runs are below 12 pt and 1,223 are below 14 pt, with an 8.5 pt
  minimum;
- equations are cached as 300-dpi PNG files instead of scalable vectors;
- the repeated title strip, pastel rounded-card grid, equal visual weight, and
  repeated red/yellow/green callouts produce a template-driven rhythm;
- speaker notes are absent, image descriptions are filenames rather than useful
  alt text, and the blank layout weakens semantic structure and global editing;
- the package is visually wide but retains `screen4x3` slide-type metadata.

Current strengths should be preserved:

- no detected out-of-bounds object;
- source-image aspect ratios are preserved;
- citations exist;
- Noto CJK rendering is reliable in the tested Linux/LibreOffice environment;
- LibreOffice 24.2.7.2 reproduced the checked-in 91-page PDF pixel-for-pixel in
  a clean temporary rerender;
- the exported PDF is tagged and its fonts are subset-embedded.

## Toolchain comparison

| Option | Main advantages | Main weaknesses | Decision |
|---|---|---|---|
| PptxGenJS 4.x + TypeScript + SVG | Masters/placeholders, SVG, editable individual shapes/charts, notes, image/chart alt text, object names, typed manifest | No group-shape API; general shape alt text and font embedding remain gaps; OOXML needs validation and real PowerPoint testing | Recommended |
| `python-pptx` 1.0.2 redesign | Existing code/assets reusable; charts, tables, notes, groups available; lowest migration | Weaker SVG/accessibility surface; public API gaps; existing helpers encourage absolute blank-slide layout | Fallback |
| LibreOffice Impress as authoring source | Strong Linux manual editing and PDF export | Manual drift from evidence manifest; PPTX round-trip risk | Review/export only |

Local prerequisites already available:

- Node 25 and npm 11;
- `python-pptx` 1.0.2;
- LibreOffice 24.2.7.2;
- Poppler;
- Matplotlib;
- Noto Sans/Serif/Mono CJK.

The observed current registry release is 4.0.1. Evaluate that version in a
disposable spike first; after approval, the committed `package-lock.json`, not a
floating registry version, becomes authoritative.

## Editorial system

### Slide contract

Every slide has:

- stable slide ID;
- one claim headline;
- one audience question;
- one takeaway;
- evidence asset IDs and hashes;
- short visible source and full notes citation;
- speaker notes;
- meaningful alt text;
- explicit `mustKeep` and `editorialFreedom` fields.

### Visual modes

Use a small family of editorial modes, not one universal card template:

1. assertion/opening;
2. mechanism diagram;
3. evidence figure;
4. comparison;
5. synthesis/decision;
6. section/reset.

Each slide gets one dominant hierarchy. Use asymmetry, whitespace, deliberate
cropping and annotation, and periodic pacing resets. Preserve the source figure
or full provenance in notes/appendix even when the main slide redraws the one
relationship the audience must see.

### Typography

- Noto Sans CJK KR: primary body and labels;
- Noto Serif CJK KR: restrained editorial display moments;
- Noto Sans Mono CJK KR: code, identifiers, and compact technical annotations;
- normal projected body copy: 20–24 pt;
- 18 pt: hard floor except necessary source, axis, or legal labels;
- no shrink-to-fit as an overflow remedy.

Matplotlib SVG with a deck-specific style sheet is the default for equations and
scientific plots. Native PowerPoint charts are used when end-user data editing
matters. Explanatory mechanisms use native shapes so labels, arrows, and
individual elements remain editable. PptxGenJS 4.0.1 has no native group-shape
API: diagrams are either individually named shapes, a single SVG with useful alt
text, or—only after a dedicated spike—grouped by a validated OOXML
postprocessor/manual terminal postflight. The baseline never promises editable
native grouping.

## Low-AI-feel rejection rules

Reject a slide when it depends on:

- repeated rounded-card grids or forced three-column symmetry;
- pastel problem/solution/insight boxes on most pages;
- dashboard density and small type;
- six equally emphasized claims;
- generic icons, decorative AI art, or stock metaphors;
- full-paper screenshots reduced to unreadable thumbnails;
- automatic bold/highlight in every sentence;
- rainbow semantic colors without an information-mapping need;
- generic headings such as “What / Why / How,” “Journey,” “Key Takeaway,” or
  “Unlocking”;
- text overflow solved by font shrinking;
- edge-to-edge filling on every page;
- decorative badges, breadcrumbs, or evidence labels repeated mechanically.

## Pre-commit capability spikes

1. **Capability spike:** 6–8 synthetic slides covering Korean, Latin, Greek,
   master placeholders, SVG, native chart, notes, image/chart alt text,
   individually named multi-shape diagrams, links, and metadata. Test optional
   OOXML grouping separately rather than assuming library support.
2. **Font torture spike:** Korean punctuation, four font roles, superscripts,
   equations, long titles, and missing-glyph fallback in LibreOffice and target
   PowerPoint.
3. **SVG/chart spike:** compare Matplotlib SVG, native chart, and raster
   fallback across both renderers.
4. **Scale spike:** generate 30 mixed-content slides, validate OOXML, and open
   in PowerPoint without a repair dialog.
5. **Editorial spike:** obtain content-owner and human visual-editor approval on
   a representative 12–15-slide narrative slice before full generation.
6. **Postflight spike:** make one permitted desktop-PowerPoint font/accessibility
   change, inventory the OOXML delta, and verify that the release-artifact
   checksum/change-log protocol can distinguish metadata/accessibility changes
   from factual or visual edits.

## QA gates

### Automated

- zero out-of-bounds objects;
- zero text overflow;
- zero unintended overlap, with a narrow explicit allowlist;
- every frozen slide ID appears exactly once;
- every asset/citation resolves and its SHA-256 matches;
- every substantive slide has notes;
- every content visual has meaningful alt text;
- correct 16:9 page size and `ko-KR` language metadata;
- clean-environment build reproduces the generated preflight PPTX, declared
  slide count, and stable renders;
- all PDF fonts report embedded;
- PDF page count, page size, tags, text extraction, and image renders pass;
- PDF/UA export is explicitly enabled and a pinned veraPDF validator passes all
  machine-verifiable conformance rules; otherwise the PDF is labeled visual-only
  and cannot be called accessibility-conformant;
- no undeclared external relationship or linked media remains;
- font inspection verifies the expected `ppt/fonts` parts when font embedding
  is claimed;
- Open XML schema validation passes once the .NET validator container is
  pinned.

### Human

- contact-sheet pacing and hierarchy review;
- visual diffs against the approved representative slice;
- actual target PowerPoint opens with no repair, missing-font, or compatibility
  warning;
- the target PowerPoint version exports or renders every release slide, and its
  images are compared with the approved contact sheet for SVG, chart, font,
  line-wrap, and object-position drift;
- PowerPoint Accessibility Checker and Reading Order review;
- human PDF structure, logical reading-order, and alternative-description
  review, including a representative screen-reader pass; veraPDF cannot judge
  whether the order or descriptions are meaningful;
- 1920×1080 projector and reduced-zoom legibility check;
- grayscale and color-vision check;
- content owner and visual editor sign off independently.

The PDF is the display-authoritative fallback. For offline PPTX presentation,
either embed all licensed font characters in desktop PowerPoint or package and
preinstall the approved fonts.

Two PPTX artifacts resolve the reproducibility/postflight conflict:

- `generated.pptx` is produced byte-for-byte by the pinned clean build and is
  the reproducibility target;
- `release.pptx` may receive only an allowlisted desktop-PowerPoint postflight
  for validated font embedding, reading order, or accessibility metadata. It is
  a terminal artifact with SHA-256, operator/tool version, timestamp, and
  OOXML-part change log.

No factual copy, chart data, geometry, or design edit is allowed in the terminal
postflight. Any such change returns to the manifest/generator and creates a new
generated build. A scripted package inventory and render diff must show that the
postflight changed only allowlisted parts. The release PPTX is therefore
auditable and sealed, while reproducibility is claimed only for its generated
preflight parent.

Any PowerPoint repair prompt, compatibility repair, removed-content notice, or
need to “fix” the generated file fails the build. The defect must be corrected
in the generator or validated postprocessor and regenerated; repair output can
never become a release artifact.

## Frozen handoff schema

Minimum `presentation-handoff.json` shape:

```text
id
order
kind
headline
audienceQuestion
takeaway
body
visualBrief
assets[]:
  assetId
  sha256
  sourceRefIds[]
  license
  role
  crop
  altText
  researchClaimIds[]
  evidenceIds[]
sourceRefs[]:
  sourceRefId
  url
  citation
  accessedAt
  researchClaimIds[]
  evidenceIds[]
chartSpec:
  datasetAssetId
  datasetSha256
  units
  transforms[]
  filters[]
  uncertainty
notes
altText
mustKeep
editorialFreedom
researchReleaseTag
```

Any factual change after freeze requires a new manifest revision and hash.
Layout/editorial changes remain separate from research-claim changes.

## Outputs

- editable `.pptx`;
- display-authoritative `.pdf`;
- contact sheet;
- presenter-notes export;
- source/derived asset inventory with hashes and licenses;
- generator source, JSON Schema, theme/design tokens, and tests;
- `package-lock.json`, pinned Node/container definition, and exact build command;
- font files and licenses when redistributable, otherwise a font installation
  manifest with hashes;
- OOXML validation report;
- external-relationship and linked-media report;
- embedded-font-part report;
- geometry/overflow report;
- font report;
- render-diff report;
- accessibility checklist;
- build provenance with tool versions and handoff hash;
- `generated.pptx` checksum;
- terminal `release.pptx` checksum and OOXML postflight change log;
- target-PowerPoint render set and comparison report;
- veraPDF PDF/UA validation report.

## Definition of done

The PPTX track is complete only when:

1. the research handoff is frozen and every slide traces to it;
2. the representative editorial slice has separate content and visual approval;
3. factual copy was not invented during slide generation;
4. simple text, individual shapes, diagrams, and eligible charts remain
   editable to the degree declared per asset; native grouping is not claimed
   without a validated grouping path;
5. type, contrast, geometry, overflow, notes, alt text, and reading-order gates
   pass;
6. the PPTX opens in target PowerPoint without repair or compatibility warning;
7. the LibreOffice PDF passes page, font, tag, extraction, and PDF/UA
   validation, or is explicitly labeled non-conformant visual fallback;
8. a clean build reproduces `generated.pptx`; any manually postflighted
   `release.pptx` has an allowlisted OOXML delta, checksum, operator record, and
   change log;
9. post-freeze changes use a new manifest revision;
10. the final deck avoids all low-AI-feel rejection patterns above.

## Primary tool references

- [PptxGenJS introduction](https://gitbrent.github.io/PptxGenJS/docs/introduction/)
- [PptxGenJS masters and placeholders](https://gitbrent.github.io/PptxGenJS/docs/masters.html)
- [PptxGenJS speaker notes](https://gitbrent.github.io/PptxGenJS/docs/speaker-notes/)
- [PptxGenJS images, SVG, alt text, and sizing](https://gitbrent.github.io/PptxGenJS/docs/api-images/)
- [PptxGenJS charts](https://gitbrent.github.io/PptxGenJS/docs/api-charts.html)
- [PptxGenJS text fitting and language options](https://gitbrent.github.io/PptxGenJS/docs/api-text/)
- [PptxGenJS repair-error guidance](https://gitbrent.github.io/PptxGenJS/docs/needs-repair-errors/)
- [PptxGenJS font-embedding issue](https://github.com/gitbrent/PptxGenJS/issues/1378)
- [`python-pptx` documentation](https://python-pptx.readthedocs.io/en/latest/)
- [LibreOffice command-line conversion](https://help.libreoffice.org/latest/ug/text/shared/guide/start_parameters.html)
- [LibreOffice font embedding](https://help.libreoffice.org/latest/en-US/text/shared/01/prop_font_embed.html)
- [LibreOffice PDF and PDF/UA export parameters](https://help.libreoffice.org/latest/gl/text/shared/guide/pdf_params.html)
- [Microsoft PowerPoint accessibility guidance](https://support.microsoft.com/en-US/accessibility/powerpoint/make-your-powerpoint-presentations-accessible-to-people-with-disabilities)
- [Microsoft Reading Order guidance](https://support.microsoft.com/en-us/office/make-slides-easier-to-read-by-using-the-reading-order-pane-863b5c1c-4f19-45ec-96e6-93a6457f5e1c)
- [Microsoft presentation readability guidance](https://support.microsoft.com/en-us/powerpoint/tips-for-creating-and-delivering-an-effective-presentation)
- [Microsoft font picker and embedding behavior](https://support.microsoft.com/en-US/Office/fonts/use-the-modern-font-picker-in-office)
- [Microsoft Open XML SDK](https://github.com/dotnet/Open-XML-SDK)
- [Matplotlib vector export](https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.savefig.html)
- [veraPDF validation software](https://verapdf.org/software/)
