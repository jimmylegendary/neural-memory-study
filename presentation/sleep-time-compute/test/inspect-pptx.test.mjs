import assert from "node:assert/strict";
import test from "node:test";

import JSZip from "jszip";

import { inspectPptxBuffer } from "../src/inspect-pptx.mjs";

async function makeFixture({ broken = false } = {}) {
  const zip = new JSZip();
  zip.file(
    "[Content_Types].xml",
    '<?xml version="1.0"?><Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types"/>',
  );
  zip.file(
    "_rels/.rels",
    [
      '<?xml version="1.0"?>',
      '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">',
      '<Relationship Id="rId1" Type="officeDocument" Target="ppt/presentation.xml"/>',
      "</Relationships>",
    ].join(""),
  );
  zip.file(
    "ppt/presentation.xml",
    [
      '<?xml version="1.0"?>',
      '<p:presentation xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main">',
      '<p:sldSz cx="12192000" cy="6858000" type="screen16x9"/>',
      "</p:presentation>",
    ].join(""),
  );
  zip.file(
    "ppt/_rels/presentation.xml.rels",
    [
      '<?xml version="1.0"?>',
      '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">',
      '<Relationship Id="rId1" Type="slide" Target="slides/slide1.xml"/>',
      '<Relationship Id="rId2" Type="slide" Target="slides/slide2.xml"/>',
      '<Relationship Id="rId3" Type="hyperlink" Target="https://example.org/" TargetMode="External"/>',
      "</Relationships>",
    ].join(""),
  );

  for (let index = 1; index <= 2; index += 1) {
    const preset = broken && index === 2 ? "oval" : "roundRect";
    zip.file(
      `ppt/slides/slide${index}.xml`,
      [
        '<?xml version="1.0"?>',
        '<p:sld xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main" ',
        'xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main">',
        `<a:prstGeom prst="${preset}"/>`,
        "</p:sld>",
      ].join(""),
    );
    zip.file(
      `ppt/slides/_rels/slide${index}.xml.rels`,
      [
        '<?xml version="1.0"?>',
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">',
        `<Relationship Id="rId1" Type="notesSlide" Target="../notesSlides/notesSlide${index}.xml"/>`,
        broken && index === 2
          ? '<Relationship Id="rId2" Type="image" Target="../media/missing.png"/>'
          : "",
        "</Relationships>",
      ].join(""),
    );
    zip.file(
      `ppt/notesSlides/notesSlide${index}.xml`,
      [
        '<?xml version="1.0"?>',
        '<p:notes xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main" ',
        'xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main">',
        index === 2 && broken ? "<a:t>  </a:t>" : `<a:t>Speaker note ${index}</a:t>`,
        "</p:notes>",
      ].join(""),
    );
  }
  return zip.generateAsync({ type: "nodebuffer" });
}

test("accepts a complete wide deck and reports external relationships", async () => {
  const report = await inspectPptxBuffer(await makeFixture(), {
    expectedSlides: 2,
    expectedNotes: 2,
  });

  assert.equal(report.ok, true);
  assert.equal(report.summary.slideCount, 2);
  assert.equal(report.summary.notesSlideCount, 2);
  assert.equal(report.summary.nonEmptyNotesCount, 2);
  assert.equal(report.summary.externalRelationshipCount, 1);
  assert.equal(report.presentationSize.isWide, true);
  assert.deepEqual(report.failures, []);
});

test("rejects empty notes, invalid presets, and missing internal targets", async () => {
  const report = await inspectPptxBuffer(await makeFixture({ broken: true }), {
    expectedSlides: 2,
    expectedNotes: 2,
  });
  const failureCodes = new Set(report.failures.map(({ code }) => code));

  assert.equal(report.ok, false);
  assert.equal(failureCodes.has("EMPTY_NOTES"), true);
  assert.equal(failureCodes.has("INVALID_GEOMETRY_PRESET"), true);
  assert.equal(failureCodes.has("MISSING_RELATIONSHIP_TARGET"), true);
});
