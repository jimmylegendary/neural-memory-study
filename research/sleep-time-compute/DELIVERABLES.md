# Sleep-Time Compute research deliverables

This index separates finished evidence synthesis and executable research design
from empirical outputs that have not been run. No benchmark result is implied
by a design, test fixture, or presentation schematic.

## Reader-facing synthesis

| Artifact | Purpose | Evidence status |
|---|---|---|
| [`SLEEP-TIME-COMPUTE-MONOGRAPH-KR.md`](SLEEP-TIME-COMPUTE-MONOGRAPH-KR.md) | Korean long-form monograph spanning chronology, memory media, training, capacity, scaling, systems, benchmark, and open questions | Evidence-backed synthesis plus visibly labeled proposals/hypotheses |
| [`SLEEP-TIME-COMPUTE-MONOGRAPH-KR.pdf`](../../build/publications/SLEEP-TIME-COMPUTE-MONOGRAPH-KR.pdf) / [`DOCX`](../../build/publications/SLEEP-TIME-COMPUTE-MONOGRAPH-KR.docx) / [`HTML`](../../build/publications/SLEEP-TIME-COMPUTE-MONOGRAPH-KR.html) | Reader and editable exports of the Korean monograph | Generated from the reviewed Markdown source |
| [`../../paper-en/sleep-time-compute/PAPER.md`](../../paper-en/sleep-time-compute/PAPER.md) | English position/review paper | Evidence-backed synthesis and falsifiable research agenda; no empirical PAPER-C result |
| [`SLEEP-TIME-COMPUTE-POSITION-PAPER-EN.pdf`](../../build/publications/SLEEP-TIME-COMPUTE-POSITION-PAPER-EN.pdf) / [`DOCX`](../../build/publications/SLEEP-TIME-COMPUTE-POSITION-PAPER-EN.docx) / [`HTML`](../../build/publications/SLEEP-TIME-COMPUTE-POSITION-PAPER-EN.html) | Reader and editable exports of the English paper | Generated from the reviewed Markdown source |
| [`../../dossier/SLEEP-TIME-COMPUTE-PRE-RESEARCH.md`](../../dossier/SLEEP-TIME-COMPUTE-PRE-RESEARCH.md) | Chronological pre-research dossier and reading order | Audited discovery record |
| [`../../presentation/sleep-time-compute/build/sleep-time-compute-research.pptx`](../../presentation/sleep-time-compute/build/sleep-time-compute-research.pptx) / [`PDF`](../../presentation/sleep-time-compute/build/sleep-time-compute-research.pdf) | Editable Korean presentation plus rendered review copy | 28 native-shape slides; all plots are marked schematic or unexecuted where applicable |
| [`PPTX contact sheet`](../../presentation/sleep-time-compute/build/sleep-time-compute-contact-sheet.png) / [`OOXML QA`](../../presentation/sleep-time-compute/reports/ooxml-qa.json) / [`build QA`](../../presentation/sleep-time-compute/reports/build-qa.json) | Whole-deck visual overview and machine validation | 28 slides, 28 non-empty notes, no missing internal relationship, no external relationship |

## Primary-source audit corpus

- [`EVIDENCE-INTAKE-WAVE-01.md`](EVIDENCE-INTAKE-WAVE-01.md) contains the
  chronological intake ledger and evidence-verified paper/claim cards.
- [`EVIDENCE-INTAKE-WAVE-02.md`](EVIDENCE-INTAKE-WAVE-02.md) extends the ledger
  with strict phase labels, negative evidence, and 2026 last-mile systems.
- [`PRIMARY-SOURCE-AUDIT-BIOLOGICAL-COMPUTATIONAL-SLEEP.md`](PRIMARY-SOURCE-AUDIT-BIOLOGICAL-COMPUTATIONAL-SLEEP.md)
- [`PRIMARY-SOURCE-AUDIT-EARLY-PRECURSORS.md`](PRIMARY-SOURCE-AUDIT-EARLY-PRECURSORS.md)
- [`PRIMARY-SOURCE-AUDIT-AGENT-MEMORY-SYSTEMS.md`](PRIMARY-SOURCE-AUDIT-AGENT-MEMORY-SYSTEMS.md)
- [`PRIMARY-SOURCE-AUDIT-CAPACITY-CONSOLIDATION-THEORY.md`](PRIMARY-SOURCE-AUDIT-CAPACITY-CONSOLIDATION-THEORY.md)
- [`PRIMARY-SOURCE-AUDIT-COMPANY-LINEAGES.md`](PRIMARY-SOURCE-AUDIT-COMPANY-LINEAGES.md)
- [`PRIMARY-SOURCE-AUDIT-JULY-2026.md`](PRIMARY-SOURCE-AUDIT-JULY-2026.md)
- [`PRIMARY-SOURCE-AUDIT-LAST-MILE-2026.md`](PRIMARY-SOURCE-AUDIT-LAST-MILE-2026.md)
- [`SYSTEMS-INFRA-PRIMARY-SOURCE-AUDIT.md`](SYSTEMS-INFRA-PRIMARY-SOURCE-AUDIT.md)

## Theory, benchmark, and systems

| Artifact | What it freezes |
|---|---|
| [`OPEN-QUESTIONS-ANSWERABILITY-MATRIX.md`](OPEN-QUESTIONS-ANSWERABILITY-MATRIX.md) | RQ1–RQ12, answerability, hypotheses, and non-answerable questions |
| [`TRAINING-DATA-METHODS-ATLAS.md`](TRAINING-DATA-METHODS-ATLAS.md) | Replay, distillation, dream data, RL, parametric isolation, and external consolidation |
| [`SCALING-LAWS-THEORY-AGENDA.md`](SCALING-LAWS-THEORY-AGENDA.md) | Identities, bounds, candidate laws, countermodels, holdouts, and claim-down rules |
| [`BENCHMARK-EXPERIMENT-BLUEPRINT.md`](BENCHMARK-EXPERIMENT-BLUEPRINT.md) | Matched 297-cell lifetime benchmark, global inference, capacity tracks, and falsification |
| [`SYSTEM-INFRA-BLUEPRINT.md`](SYSTEM-INFRA-BLUEPRINT.md) | Wake, snapshot, sleep, promotion, memory, serving, deletion, rollback, and data movement |
| [`SCIENTIFIC-RED-TEAM-AUDIT-2026-07-25.md`](SCIENTIFIC-RED-TEAM-AUDIT-2026-07-25.md) | 31 incorporated adversarial findings and their required mutation tests |

## Executable control plane

The Python package under this directory implements typed records, canonical
JSON/schema validation, durable registry primitives, and cryptographically
verified G7/G8 gate evaluation. The package is currently an implementation and
test substrate; the canonical research registries remain intentionally empty.

```bash
uv run pytest -q
uv run ruff check src tests
```

## Empirical status

The controlled GPU benchmark, capacity experiments, independent analyses, and
human release gates are preregistered designs, not completed experiments.
Until their exact G2–G8 artifacts exist:

- no PAPER-C hypothesis is reported as supported;
- no candidate scaling relation is called a measured law;
- no system architecture is claimed to dominate under matched resources;
- no novelty or priority claim is established by this bounded audit alone.
