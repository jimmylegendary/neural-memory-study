# Sleep-Time Compute research deliverables

This index separates finished evidence synthesis and executable research design
from empirical outputs that have not been run. No benchmark result is implied
by a design, test fixture, or presentation schematic.

**Current release:** 2026-08-05 source freeze. 아래 "학술 Study release"가 최신 통합 산출물이며, 뒤의 monograph/position-paper/28-slide 항목은 선행 연구 provenance로 유지한다.

## 프로젝트별 canonical package

최종 열람·전달본은 repository root의 [`deliverables/`](../../deliverables/README.md)에 모은다. 기존 `build/publications/`, `easy/`, `translations-kr/`, `presentation/` 경로는 생성 source와 QA provenance로 유지한다.

- [`deliverables/neural-memory/`](../../deliverables/neural-memory/README.md): 기존 Neural Memory 산출물 31개
- [`deliverables/sleep-time-compute/`](../../deliverables/sleep-time-compute/README.md): Sleep-Time Compute 산출물 46개
- [`manifest.json`](../../deliverables/sleep-time-compute/manifest.json): package path ↔ source path, byte size, SHA-256, translation rights
- [`SHA256SUMS`](../../deliverables/sleep-time-compute/SHA256SUMS): package file 46개의 무결성 목록

```bash
python3 scripts/package_deliverables.py --root . --output deliverables
python3 scripts/package_deliverables.py --root . --output deliverables --check
```

## 독자별 빠른 경로

| 목적 | 시작 파일 | 함께 볼 파일 |
|---|---|---|
| 학회형 검토 | [`Conference PDF`](../../deliverables/sleep-time-compute/study/SLEEP-TIME-COMPUTE-CONFERENCE-KR.pdf) | [`Appendix`](../../deliverables/sleep-time-compute/study/SLEEP-TIME-COMPUTE-CONFERENCE-APPENDIX-KR.pdf), [`상세 Study`](../../deliverables/sleep-time-compute/study/SLEEP-TIME-COMPUTE-STRATEGIC-STUDY-KR.pdf) |
| Training 사전지식 | [`Training Background`](../../deliverables/sleep-time-compute/study/TRAINING-BACKGROUND-FOR-SLEEP-TIME-COMPUTE-KR.pdf) | [`쉬운 설명 통합본`](../../deliverables/sleep-time-compute/easy/SLEEP-TIME-COMPUTE-EASY-COMPANION-KR.pdf) |
| Seminar | [`112-slide PPTX`](../../deliverables/sleep-time-compute/seminar/SLEEP-TIME-COMPUTE-DEEP-SEMINAR-112SLIDES-KR.pptx) | [`PDF`](../../deliverables/sleep-time-compute/seminar/SLEEP-TIME-COMPUTE-DEEP-SEMINAR-112SLIDES-KR.pdf), [`speaker/deck QA`](../../presentation/sleep-time-compute-deep/README.md) |
| 핵심 논문 의역 | [`15편 package`](../../deliverables/sleep-time-compute/translations/) | [`rights manifest`](../../translations-kr/stc-core/reports/source-rights.json), [`package manifest`](../../deliverables/sleep-time-compute/manifest.json) |
| System·device | [`SYSTEM-INFRA-BLUEPRINT.md`](SYSTEM-INFRA-BLUEPRINT.md) | [`SCALING-LAWS-THEORY-AGENDA.md`](SCALING-LAWS-THEORY-AGENDA.md), [`BENCHMARK-EXPERIMENT-BLUEPRINT.md`](BENCHMARK-EXPERIMENT-BLUEPRINT.md) |
| 재현·감사 | [`Package manifest`](../../deliverables/sleep-time-compute/manifest.json) | [`FINAL-QA.json`](../../deliverables/sleep-time-compute/manifests/publications/FINAL-QA.json), [`SHA256SUMS`](../../deliverables/sleep-time-compute/SHA256SUMS) |

## 학술 Study release

| Artifact | 역할 | 검증 범위 |
|---|---|---|
| [`TRAINING-BACKGROUND-FOR-SLEEP-TIME-COMPUTE-KR.pdf`](../../deliverables/sleep-time-compute/study/TRAINING-BACKGROUND-FOR-SLEEP-TIME-COMPUTE-KR.pdf) | training 비전공자용 사전 지식 | 57쪽, Study 개념 link의 target |
| [`SLEEP-TIME-COMPUTE-STRATEGIC-STUDY-KR.pdf`](../../deliverables/sleep-time-compute/study/SLEEP-TIME-COMPUTE-STRATEGIC-STUDY-KR.pdf) | 장문 분석 Study | 70쪽, 문헌·대안·promisingness·scaling·infra·device 통합 |
| [`SLEEP-TIME-COMPUTE-CONFERENCE-KR.pdf`](../../deliverables/sleep-time-compute/study/SLEEP-TIME-COMPUTE-CONFERENCE-KR.pdf) | 학회 제출 형식 본문 | 12쪽 |
| [`SLEEP-TIME-COMPUTE-CONFERENCE-APPENDIX-KR.pdf`](../../deliverables/sleep-time-compute/study/SLEEP-TIME-COMPUTE-CONFERENCE-APPENDIX-KR.pdf) | conference appendix | 30쪽 |
| [`SLEEP-TIME-COMPUTE-EASY-COMPANION-KR.pdf`](../../deliverables/sleep-time-compute/easy/SLEEP-TIME-COMPUTE-EASY-COMPANION-KR.pdf) | Study section별 쉬운 설명 | 62쪽, 12개 booklet 통합 |
| [`translations/`](../../deliverables/sleep-time-compute/translations/) | 핵심 논문 15편 구조 보존형 의역 | 579쪽, public 2 / internal-only 13 |
| [`SLEEP-TIME-COMPUTE-DEEP-SEMINAR-112SLIDES-KR.pptx`](../../deliverables/sleep-time-compute/seminar/SLEEP-TIME-COMPUTE-DEEP-SEMINAR-112SLIDES-KR.pptx) / [`PDF`](../../deliverables/sleep-time-compute/seminar/SLEEP-TIME-COMPUTE-DEEP-SEMINAR-112SLIDES-KR.pdf) | 상세 seminar | 112 slides, 112 source notes |

## Rights·evidence boundary

- PLOS STC-T04와 eLife STC-T13만 translation manifest에서 `public`이다. 나머지 13편은 `internal-only`이며 외부 배포 목록에서 제외된다.
- Study의 literature synthesis와 저자 제안·가설을 구별하고, benchmark·capacity 결과가 아직 실행되지 않은 곳은 design/preregistered 상태로 표시한다.
- 저자 작성 Study/Background/Easy/Deck도 source figure 및 인용 자산의 외부 공개 조건을 별도 검토해야 한다. `public` 표시는 자동적인 조직 외부 공개 승인이 아니다.

## 최신 release 재현

Repository root에서 다음 순서로 실행한다.

```bash
# Academic PDFs
python3 paper-kr/common/build_publications.py --root . --target all --clean
python3 paper-kr/common/qa_publications.py \
  --pdf build/publications/TRAINING-BACKGROUND-FOR-SLEEP-TIME-COMPUTE-KR.pdf \
  --kind background --require-all-concepts
python3 paper-kr/common/qa_publications.py \
  --pdf build/publications/SLEEP-TIME-COMPUTE-STRATEGIC-STUDY-KR.pdf \
  --kind study --claim-map claims/stc-study/claim-map.json \
  --figure-ledger claims/stc-study/figure-ledger.json
python3 paper-kr/common/qa_publications.py \
  --pdf build/publications/SLEEP-TIME-COMPUTE-CONFERENCE-KR.pdf \
  --kind conference --min-pages 12 --max-pages 18
python3 paper-kr/common/qa_publications.py \
  --pdf build/publications/SLEEP-TIME-COMPUTE-CONFERENCE-APPENDIX-KR.pdf \
  --kind appendix

# 15 translations and rights/structure QA
python3 translations-kr/stc-core/build_translations.py --all
python3 translations-kr/stc-core/qa_translations.py --all --strict --render-all-pages
python3 translations-kr/stc-core/audit_translations.py

# Easy companion
python3 easy/sleep-time-compute/build.py --all --combined
python3 easy/sleep-time-compute/qa.py --all --combined --strict --render-all-pages

# 112-slide deck (artifact-tool workspace가 준비된 Codex runtime)
node presentation/sleep-time-compute-deep/src/generate-frame-map.mjs
node presentation/sleep-time-compute-deep/src/prepare-starter.mjs
node presentation/sleep-time-compute-deep/src/build-deck.mjs
node --test presentation/sleep-time-compute-deep/tests/content.test.mjs
python3 presentation/sleep-time-compute-deep/tests/verify_deck_release.py

# Claim gates and release manifest
python3 -m veridraft --data-dir .veridraft-stc-study gate sleep-time-compute-strategic-study-2026
python3 research/sleep-time-compute/program/build_release_manifest.py --root . --require-ready
(cd build/publications && sha256sum -c SHA256SUMS)

# Neural Memory / Sleep-Time Compute를 같은 배포 루트에 분리
python3 scripts/package_deliverables.py --root . --output deliverables
python3 scripts/package_deliverables.py --root . --output deliverables --check
```

Expected build outputs are `build/publications/`의 7개 core PDF/PPTX, translation 15개 PDF, `FINAL-QA.json`, `STC-STUDY-RELEASE-MANIFEST.json`, `SHA256SUMS`다. Reader-facing package는 `deliverables/neural-memory/` 31개와 `deliverables/sleep-time-compute/` 46개 artifact로 구성한다. Deck authoring은 `@oai/artifact-tool`만 사용하며 LibreOffice는 PDF export/QA에만 사용한다.

## Reader-facing synthesis

| Artifact | Purpose | Evidence status |
|---|---|---|
| [`SLEEP-TIME-COMPUTE-MONOGRAPH-KR.md`](SLEEP-TIME-COMPUTE-MONOGRAPH-KR.md) | Korean long-form monograph spanning chronology, memory media, training, capacity, scaling, systems, benchmark, and open questions | Evidence-backed synthesis plus visibly labeled proposals/hypotheses |
| [`SLEEP-TIME-COMPUTE-MONOGRAPH-KR.pdf`](../../deliverables/sleep-time-compute/study/SLEEP-TIME-COMPUTE-MONOGRAPH-KR.pdf) / [`DOCX`](../../deliverables/sleep-time-compute/study/SLEEP-TIME-COMPUTE-MONOGRAPH-KR.docx) / [`HTML`](../../deliverables/sleep-time-compute/study/SLEEP-TIME-COMPUTE-MONOGRAPH-KR.html) | Reader and editable exports of the Korean monograph | Generated from the reviewed Markdown source |
| [`../../paper-en/sleep-time-compute/PAPER.md`](../../paper-en/sleep-time-compute/PAPER.md) | English position/review paper | Evidence-backed synthesis and falsifiable research agenda; no empirical PAPER-C result |
| [`SLEEP-TIME-COMPUTE-POSITION-PAPER-EN.pdf`](../../deliverables/sleep-time-compute/study/SLEEP-TIME-COMPUTE-POSITION-PAPER-EN.pdf) / [`DOCX`](../../deliverables/sleep-time-compute/study/SLEEP-TIME-COMPUTE-POSITION-PAPER-EN.docx) / [`HTML`](../../deliverables/sleep-time-compute/study/SLEEP-TIME-COMPUTE-POSITION-PAPER-EN.html) | Reader and editable exports of the English paper | Generated from the reviewed Markdown source |
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
