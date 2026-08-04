import assert from "node:assert/strict";
import test from "node:test";

import { deckMeta, slides } from "../src/content.mjs";

test("seminar contains exactly 112 complete, uniquely identified slides", () => {
  assert.equal(slides.length, 112);
  assert.equal(deckMeta.slideCount, 112);
  assert.equal(new Set(slides.map((slide) => slide.id)).size, 112);
  assert.equal(new Set(slides.map((slide) => slide.title)).size, 112);
  assert.deepEqual(slides.map((slide) => slide.slideNumber), Array.from({ length: 112 }, (_, index) => index + 1));
});

test("all 28 approved template patterns are used", () => {
  assert.deepEqual(
    [...new Set(slides.map((slide) => slide.pattern))].sort((a, b) => a - b),
    Array.from({ length: 28 }, (_, index) => index + 1),
  );
});

test("every slide has bounded, sourced seminar content", () => {
  const placeholders = /\b(TODO|TBD|FIXME|LOREM|PLACEHOLDER|undefined|null)\b/i;
  for (const slide of slides) {
    assert.ok(slide.title.length <= 60, `${slide.id}: title too long`);
    assert.ok(slide.points.length >= 3 && slide.points.length <= 6, `${slide.id}: point count`);
    assert.ok(slide.refs.length >= 1, `${slide.id}: missing source`);
    assert.ok(slide.takeaway.length >= 12, `${slide.id}: weak takeaway`);
    const joined = JSON.stringify(slide);
    assert.equal(placeholders.test(joined), false, `${slide.id}: placeholder language`);
  }
});

test("evidence boundaries remain explicit", () => {
  const claims = JSON.stringify(slides);
  for (const required of ["external-first", "research-scale", "검증", "가설", "conditional", "evidence"]) {
    assert.ok(claims.toLowerCase().includes(required.toLowerCase()), `missing boundary term: ${required}`);
  }
});
