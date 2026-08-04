import { foundationSlides } from "./content-01-foundation.mjs";
import { alternativeMechanismSlides } from "./content-02-alternatives-mechanisms.mjs";
import { evidenceVerdictSlides } from "./content-03-evidence-verdict.mjs";
import { scalingInfraDeviceSlides } from "./content-04-scaling-infra-device.mjs";

const rawSlides = [
  ...foundationSlides,
  ...alternativeMechanismSlides,
  ...evidenceVerdictSlides,
  ...scalingInfraDeviceSlides,
];

export const deckMeta = {
  title: "Sleep-Time Compute",
  subtitle: "장기기억·continual learning·background training을 하나의 lifecycle로 재구성",
  freezeDate: "2026-08-05",
  slideCount: 112,
  template: "presentation/sleep-time-compute/build/sleep-time-compute-research.pptx",
  output: "build/publications/SLEEP-TIME-COMPUTE-DEEP-SEMINAR-112SLIDES-KR.pptx",
};

export const slides = rawSlides.map((slide, index) => ({
  ...slide,
  slideNumber: index + 1,
  id: `STC-DEEP-${String(index + 1).padStart(3, "0")}`,
}));

if (slides.length !== deckMeta.slideCount) {
  throw new Error(`Expected ${deckMeta.slideCount} slides, received ${slides.length}`);
}

for (const slide of slides) {
  if (!Number.isInteger(slide.pattern) || slide.pattern < 1 || slide.pattern > 28) {
    throw new Error(`${slide.id}: invalid source pattern ${slide.pattern}`);
  }
  if (!slide.title || !slide.section || !slide.takeaway || !slide.refs?.length || !slide.points?.length) {
    throw new Error(`${slide.id}: incomplete content record`);
  }
}

export default slides;
