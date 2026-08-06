// Sleep-Time Compute v2 seminar — content ledger
//
// 2시간 발표 + 30분 Q&A. 내용의 출처는 study-kr-stc(30장)와 experiments/stc(X1–X4)이며,
// 렌더 하네스(build-deck.mjs / prepare-starter.mjs / normalize-imported-geometry.mjs)와
// 승인된 28-slide 시각 계약은 그대로 재사용한다. 이 파일은 텍스트와 발표 대본만 정의한다.
//
// 각 slide 객체의 계약:
//   id           슬라이드 식별자 (STC2-NNN)
//   pattern      재사용할 source slide 번호 (1..28)
//   section      chrome의 섹션 라벨 (<= 22자)
//   slideNumber  이 파일이 자동 부여
//   title        headline (<= 58자, 줄바꿈 없음)
//   status       evidence status 라벨 (<= 16자)
//   source       footer 한 줄 경로 (<= 105자)
//   takeaway     한 문장 결론 (<= 105자)
//   points       [{label, body, metric?}] — pattern이 요구하는 슬롯 수에 맞춘다
//   bigNumbers   숫자·짧은 문장 슬롯 (pattern마다 폭이 다름)
//   curveLabels  축·임계 라벨
//   equation     수식 슬롯이 있는 pattern에서만
//   refs         speaker note의 [Sources] 블록
//   script       발표 대본 (speaker note 본문). 슬라이드마다 반드시 있어야 한다.

import { orientation } from "./content-01-orientation.mjs";
import { background } from "./content-02-background.mjs";
import { discriminant } from "./content-03-discriminant.mjs";
import { paths } from "./content-04-paths.mjs";
import { experiments } from "./content-05-experiments.mjs";
import { verdict } from "./content-06-verdict.mjs";
import { limits } from "./content-07-limits.mjs";
import { qa } from "./content-08-qa.mjs";

export const deckMeta = {
  title: "Sleep-Time Compute v2 — 이름이 기제보다 넓다",
  output: "build/publications/SLEEP-TIME-COMPUTE-V2-SEMINAR-KR.pptx",
  freezeDate: "2026-08-06",
  template: "presentation/sleep-time-compute/build/sleep-time-compute-research.pptx",
  runtimeMinutes: { talk: 120, qa: 30 },
};

const sections = [
  orientation,
  background,
  discriminant,
  paths,
  experiments,
  verdict,
  limits,
  qa,
];

// pattern 12는 6개 원형 칩 옆에 번호가 붙는 도식이다. 슬라이드가 따로 숫자를 주지
// 않으면 1..6을 채워 넣는다(그 자리는 두 자리 이상을 담지 못한다).
const withPatternDefaults = (slide) =>
  slide.pattern === 12 && !slide.bigNumbers
    ? { ...slide, bigNumbers: ["1", "2", "3", "4", "5", "6"] }
    : slide;

export const slides = sections
  .flat()
  .map((slide, index) => ({ ...withPatternDefaults(slide), slideNumber: index + 1 }));

// ---- 내용 계약 검사 (build 전에 깨지면 즉시 실패한다) ----------------------
const seenIds = new Set();
for (const slide of slides) {
  const where = `${slide.id ?? "?"} (slide ${slide.slideNumber})`;
  if (!slide.id) throw new Error(`${where}: id가 없다`);
  if (seenIds.has(slide.id)) throw new Error(`${where}: id 중복`);
  seenIds.add(slide.id);
  if (!Number.isInteger(slide.pattern) || slide.pattern < 1 || slide.pattern > 28) {
    throw new Error(`${where}: pattern은 1..28이어야 한다`);
  }
  if (!slide.title || !slide.takeaway || !slide.status || !slide.source) {
    throw new Error(`${where}: title/takeaway/status/source가 모두 있어야 한다`);
  }
  if (!Array.isArray(slide.points) || slide.points.length === 0) {
    throw new Error(`${where}: points가 비어 있다`);
  }
  if (!Array.isArray(slide.refs) || slide.refs.length === 0) {
    throw new Error(`${where}: refs가 비어 있다`);
  }
  if (typeof slide.script !== "string" || slide.script.trim().length < 80) {
    throw new Error(`${where}: 발표 대본(script)이 없거나 너무 짧다`);
  }
}
