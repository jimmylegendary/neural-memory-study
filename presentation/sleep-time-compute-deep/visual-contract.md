# Sleep-Time Compute Deep Seminar — Visual Contract

## Communication job

세미나가 끝날 때 Samsung SAIT의 연구·시스템·memory-device 연구자는 sleep-time compute를 anthropomorphic slogan이 아니라 검증 가능한 memory lifecycle로 이해하고, external-first hybrid 방향·scaling 질문·infra/device 실험을 토론할 수 있어야 한다.

## Source contract

- Visual source: `presentation/sleep-time-compute/build/sleep-time-compute-research.pptx` 28 slides.
- 모든 output slide는 source slide를 duplicate한 뒤 inherited element를 편집한다.
- Source palette, typography, line weight, chrome, footer rhythm, whitespace, 16:9 canvas를 보존한다.
- Source의 28개 pattern을 모두 적어도 한 번 재사용한다. Pattern 반복은 내용 역할에 맞춰 선택한다.
- 새 UI card grid, gradient, stock art, generated screenshot을 추가하지 않는다.

## Design tokens observed from the source

- Canvas: 1280×720, warm paper `#F6F3EC`.
- Primary ink: near-black `#171B1E`; accent families: cobalt, rust, teal with pale companion fills.
- Typeface: Noto Sans CJK KR for most narrative text, Noto Serif CJK KR for cover/editorial emphasis, Noto Sans Mono CJK KR for labels and evidence tags.
- Headline: one line, left aligned, approximately 32–36 px in imported geometry.
- Chrome: uppercase section label at upper left, slide ID upper right, evidence status under title, thin source/footer rule, page number lower right.
- Composition: one diagram or one comparison per slide, generous margins, flat shapes, no decorative iconography.

## Narrative and density rules

- 112 slides exactly; each has one claim and a takeaway-style title.
- Visible body copy is concise; full source IDs and presentation guidance live in speaker notes.
- Direct fact, synthesis, hypothesis, proposal, limitation are visibly distinct through the inherited status label.
- No title wraps. Non-chrome text is short enough to fit the inherited source box without reducing font size.
- Formula slides label status as identity, break-even model, metric proposal, or hypothesis; no unverified equation is called a law.

## Figure strategy

The deck preserves the source deck’s native editorial diagrams rather than flattening the Study figures into screenshots. The 15 Study figure concepts are mapped to inherited diagram patterns: lifecycle, timing×medium, chronology, comparison matrix, conditional verdict, hybrid architecture, accounting/hypothesis status, break-even, capacity knee, cadence frontier, state-migration roofline, device opportunities, failure controls, landscape, and falsifiable agenda.

## Source and notes policy

Every slide has a `[Sources]` speaker-note block containing one or more of: Study section IDs, public claim IDs, source IDs, Background concept labels, or product documentation identifiers. Notes never strengthen the underlying claim. The deck footer is a short human-readable route, while exact provenance remains in notes and the repository claim map.
