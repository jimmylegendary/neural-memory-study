# Sleep-Time Compute Deep Seminar — 112 slides

2026-08-05 source freeze의 상세 Study를 seminar로 설명하는 112-slide 한국어 deck이다. 기존 28-slide 사용 승인 design을 exact template로 가져오고, 28개 source frame을 네 차례 모두 재사용했다. 새 geometry primitive를 만들지 않고 inherited text와 speaker notes만 rewrite했다.

## 산출물

- [편집용 PPTX](../../build/publications/SLEEP-TIME-COMPUTE-DEEP-SEMINAR-112SLIDES-KR.pptx)
- [발표·검토용 PDF](../../build/publications/SLEEP-TIME-COMPUTE-DEEP-SEMINAR-112SLIDES-KR.pdf)
- [Deck manifest](reports/deck-manifest.json)
- [Release QA](reports/release-qa.json)
- [Template audit](reports/template-audit.md), [deviation log](reports/template-deviation-log.md), [text edit log](reports/text-edit-log.json)
- [7개 contact sheet](build/contact-sheets/)

각 slide에는 `[Sources]` speaker note가 있으며 112/112 source marker를 기계 검수한다. 그래프와 수치가 실험 결과가 아닌 경우 schematic, proposal, preregistered design의 상태 표기를 유지한다.

## 재현

Deck authoring은 presentations skill이 제공하는 `@oai/artifact-tool` runtime을 사용한다. `python-pptx`, PptxGenJS, direct OOXML은 authoring에 사용하지 않는다. LibreOffice는 PDF export와 QA에만 사용한다.

Repository root에서:

```bash
node presentation/sleep-time-compute-deep/src/generate-frame-map.mjs
node presentation/sleep-time-compute-deep/src/prepare-starter.mjs
node presentation/sleep-time-compute-deep/src/build-deck.mjs
node --test presentation/sleep-time-compute-deep/tests/content.test.mjs
python3 presentation/sleep-time-compute-deep/tests/verify_deck_release.py
```

`prepare-starter.mjs`는 승인 deck [`sleep-time-compute-research.pptx`](../sleep-time-compute/build/sleep-time-compute-research.pptx)를 import하고, import 과정의 10개 negative extent를 시각적으로 동등한 positive bbox + flip으로 정규화한다. `build-deck.mjs`는 starter의 112 slides를 content ledger로 rewrite하고 preview/layout/notes를 함께 출력한다.

Overflow 검사는 presentations skill의 `container_tools/slides_test.py`를 다음처럼 실행한다. Codex runtime version에 따라 `<presentations-skill>`의 absolute path만 달라질 수 있다.

```bash
PYTHONPATH=build/deck-work/python-deps \
python3 <presentations-skill>/container_tools/slides_test.py \
  build/publications/SLEEP-TIME-COMPUTE-DEEP-SEMINAR-112SLIDES-KR.pptx \
  --width 1280 --height 720
```

Expected gates: content test 4/4, slide XML 112, notes XML 112, source marker 112, placeholder 0, geometry mismatch 0, element inventory mismatch 0, overflow 0, PDF 112 pages.

Gitignored preview/layout intermediates가 없는 clean checkout에서는 committed PPTX만 검사하는 smoke mode를 사용한다. 이 mode는 report를 덮어쓰지 않는다.

```bash
python3 presentation/sleep-time-compute-deep/tests/verify_deck_release.py \
  --artifact-only --no-write-report
```

## 권리·공개 경계

Deck은 저자 작성 synthesis지만 인용 source와 일부 source-derived visual을 포함한다. 외부 공개 전 organizational legal/publication review가 필요하며, `internal-only` translation PDF나 source plate를 deck package와 함께 공개하지 않는다.
