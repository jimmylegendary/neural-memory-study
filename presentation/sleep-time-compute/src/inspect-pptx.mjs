import { createHash } from "node:crypto";
import fs from "node:fs/promises";
import path from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";

import JSZip from "jszip";

const EXPECTED_SLIDES = 28;
const EXPECTED_NOTES = 28;
const INVALID_GEOMETRY_PRESETS = new Set(["oval", "roundedRectangle"]);
const WIDE_ASPECT_RATIO = 16 / 9;
const WIDE_RATIO_TOLERANCE = 0.01;

function decodeXml(value) {
  return value
    .replaceAll("&quot;", '"')
    .replaceAll("&apos;", "'")
    .replaceAll("&lt;", "<")
    .replaceAll("&gt;", ">")
    .replaceAll("&amp;", "&")
    .replace(/&#(\d+);/g, (_, decimal) =>
      String.fromCodePoint(Number.parseInt(decimal, 10)),
    )
    .replace(/&#x([\da-f]+);/gi, (_, hexadecimal) =>
      String.fromCodePoint(Number.parseInt(hexadecimal, 16)),
    );
}

function parseAttributes(source) {
  const attributes = {};
  const expression = /([\w:.-]+)\s*=\s*(["'])([\s\S]*?)\2/g;
  for (const match of source.matchAll(expression)) {
    attributes[match[1]] = decodeXml(match[3]);
  }
  return attributes;
}

function numberedParts(names, expression) {
  return names
    .map((name) => {
      const match = name.match(expression);
      return match ? { name, number: Number.parseInt(match[1], 10) } : null;
    })
    .filter(Boolean)
    .sort((left, right) => left.number - right.number);
}

function expectedSequence(count) {
  return Array.from({ length: count }, (_, index) => index + 1);
}

function sameNumberSequence(actual, expected) {
  return (
    actual.length === expected.length &&
    actual.every((value, index) => value === expected[index])
  );
}

function relationshipSourcePart(relationshipPart) {
  if (relationshipPart === "_rels/.rels") {
    return "";
  }
  const match = relationshipPart.match(/^(.*)\/_rels\/([^/]+)\.rels$/);
  if (!match) {
    return null;
  }
  return path.posix.join(match[1], match[2]);
}

function safeDecodeUri(value) {
  try {
    return decodeURIComponent(value);
  } catch {
    return value;
  }
}

export function resolveRelationshipTarget(relationshipPart, target) {
  const sourcePart = relationshipSourcePart(relationshipPart);
  if (sourcePart === null) {
    return null;
  }
  const withoutFragment = target.split("#", 1)[0].split("?", 1)[0];
  const decoded = safeDecodeUri(withoutFragment);
  const normalized = decoded.startsWith("/")
    ? path.posix.normalize(decoded.slice(1))
    : path.posix.normalize(path.posix.join(path.posix.dirname(sourcePart), decoded));
  if (
    normalized === ".." ||
    normalized.startsWith("../") ||
    path.posix.isAbsolute(normalized)
  ) {
    return null;
  }
  return normalized;
}

function extractRelationships(xml) {
  const relationships = [];
  const expression =
    /<(?:[\w.-]+:)?Relationship\b([\s\S]*?)(?:\/>|>[\s\S]*?<\/(?:[\w.-]+:)?Relationship>)/g;
  for (const match of xml.matchAll(expression)) {
    relationships.push(parseAttributes(match[1]));
  }
  return relationships;
}

function extractNoteText(xml) {
  const fragments = [];
  const expression =
    /<(?:[\w.-]+:)?t(?:\s[^>]*)?>([\s\S]*?)<\/(?:[\w.-]+:)?t>/g;
  for (const match of xml.matchAll(expression)) {
    fragments.push(decodeXml(match[1].replace(/<[^>]+>/g, "")));
  }
  return fragments.join(" ").replace(/\s+/g, " ").trim();
}

function extractPresentationSize(xml) {
  const match = xml.match(/<(?:[\w.-]+:)?sldSz\b([\s\S]*?)(?:\/>|>)/);
  if (!match) {
    return {
      cx: null,
      cy: null,
      type: null,
      aspectRatio: null,
      isWide: false,
    };
  }
  const attributes = parseAttributes(match[1]);
  const cx = Number.parseInt(attributes.cx, 10);
  const cy = Number.parseInt(attributes.cy, 10);
  const aspectRatio =
    Number.isFinite(cx) && Number.isFinite(cy) && cy > 0 ? cx / cy : null;
  return {
    cx: Number.isFinite(cx) ? cx : null,
    cy: Number.isFinite(cy) ? cy : null,
    type: attributes.type ?? null,
    aspectRatio,
    isWide:
      aspectRatio !== null &&
      Math.abs(aspectRatio - WIDE_ASPECT_RATIO) <= WIDE_RATIO_TOLERANCE,
  };
}

function failure(code, message, details = {}) {
  return { code, message, ...details };
}

export async function inspectPptxBuffer(
  buffer,
  {
    expectedSlides = EXPECTED_SLIDES,
    expectedNotes = EXPECTED_NOTES,
    artifact = null,
  } = {},
) {
  const zip = await JSZip.loadAsync(buffer);
  const partNames = Object.values(zip.files)
    .filter((entry) => !entry.dir)
    .map((entry) => entry.name)
    .sort();
  const partNameSet = new Set(partNames);
  const failures = [];

  const slides = numberedParts(
    partNames,
    /^ppt\/slides\/slide(\d+)\.xml$/,
  );
  const notesSlides = numberedParts(
    partNames,
    /^ppt\/notesSlides\/notesSlide(\d+)\.xml$/,
  );
  if (slides.length !== expectedSlides) {
    failures.push(
      failure(
        "SLIDE_COUNT",
        `Expected ${expectedSlides} slide parts, found ${slides.length}.`,
        { expected: expectedSlides, actual: slides.length },
      ),
    );
  }
  const actualSlideNumbers = slides.map(({ number }) => number);
  if (!sameNumberSequence(actualSlideNumbers, expectedSequence(expectedSlides))) {
    failures.push(
      failure(
        "SLIDE_SEQUENCE",
        `Slide part numbers must be contiguous from 1 through ${expectedSlides}.`,
        { actual: actualSlideNumbers },
      ),
    );
  }
  if (notesSlides.length !== expectedNotes) {
    failures.push(
      failure(
        "NOTES_COUNT",
        `Expected ${expectedNotes} notesSlide parts, found ${notesSlides.length}.`,
        { expected: expectedNotes, actual: notesSlides.length },
      ),
    );
  }
  const actualNotesNumbers = notesSlides.map(({ number }) => number);
  if (!sameNumberSequence(actualNotesNumbers, expectedSequence(expectedNotes))) {
    failures.push(
      failure(
        "NOTES_SEQUENCE",
        `notesSlide part numbers must be contiguous from 1 through ${expectedNotes}.`,
        { actual: actualNotesNumbers },
      ),
    );
  }

  const emptyNotes = [];
  const noteTextLengths = {};
  for (const note of notesSlides) {
    const xml = await zip.file(note.name).async("string");
    const text = extractNoteText(xml);
    noteTextLengths[note.name] = text.length;
    if (text.length === 0) {
      emptyNotes.push(note.name);
    }
  }
  if (emptyNotes.length > 0) {
    failures.push(
      failure(
        "EMPTY_NOTES",
        `${emptyNotes.length} notesSlide part(s) contain no speaker-note text.`,
        { parts: emptyNotes },
      ),
    );
  }

  const relationshipParts = partNames.filter((name) => name.endsWith(".rels"));
  const externalRelationships = [];
  const missingInternalTargets = [];
  const invalidRelationshipParts = [];
  let internalRelationshipCount = 0;
  for (const relationshipPart of relationshipParts) {
    const xml = await zip.file(relationshipPart).async("string");
    for (const relationship of extractRelationships(xml)) {
      const record = {
        relationshipPart,
        id: relationship.Id ?? null,
        type: relationship.Type ?? null,
        target: relationship.Target ?? null,
      };
      if ((relationship.TargetMode ?? "").toLowerCase() === "external") {
        externalRelationships.push(record);
        continue;
      }
      internalRelationshipCount += 1;
      const resolvedTarget = relationship.Target
        ? resolveRelationshipTarget(relationshipPart, relationship.Target)
        : null;
      if (resolvedTarget === null) {
        invalidRelationshipParts.push(record);
      } else if (!partNameSet.has(resolvedTarget)) {
        missingInternalTargets.push({ ...record, resolvedTarget });
      }
    }
  }
  if (invalidRelationshipParts.length > 0) {
    failures.push(
      failure(
        "INVALID_RELATIONSHIP_TARGET",
        `${invalidRelationshipParts.length} internal relationship target(s) could not be resolved safely.`,
        { relationships: invalidRelationshipParts },
      ),
    );
  }
  if (missingInternalTargets.length > 0) {
    failures.push(
      failure(
        "MISSING_RELATIONSHIP_TARGET",
        `${missingInternalTargets.length} internal relationship target(s) are absent from the package.`,
        { relationships: missingInternalTargets },
      ),
    );
  }

  const invalidGeometryPresets = [];
  const geometryExpression =
    /<(?:[\w.-]+:)?prstGeom\b[^>]*\bprst\s*=\s*(["'])(oval|roundedRectangle)\1/gi;
  for (const partName of partNames.filter((name) => name.endsWith(".xml"))) {
    const xml = await zip.file(partName).async("string");
    for (const match of xml.matchAll(geometryExpression)) {
      if (INVALID_GEOMETRY_PRESETS.has(match[2])) {
        invalidGeometryPresets.push({
          part: partName,
          preset: match[2],
        });
      }
    }
  }
  if (invalidGeometryPresets.length > 0) {
    failures.push(
      failure(
        "INVALID_GEOMETRY_PRESET",
        `${invalidGeometryPresets.length} unsupported geometry preset occurrence(s) found.`,
        { occurrences: invalidGeometryPresets },
      ),
    );
  }

  const presentationPart = zip.file("ppt/presentation.xml");
  const presentationSize = presentationPart
    ? extractPresentationSize(await presentationPart.async("string"))
    : {
        cx: null,
        cy: null,
        type: null,
        aspectRatio: null,
        isWide: false,
      };
  if (!presentationPart) {
    failures.push(
      failure(
        "MISSING_PRESENTATION_PART",
        "ppt/presentation.xml is absent from the package.",
      ),
    );
  } else if (presentationSize.aspectRatio === null) {
    failures.push(
      failure(
        "MISSING_PRESENTATION_SIZE",
        "ppt/presentation.xml does not contain a valid p:sldSz element.",
      ),
    );
  } else if (!presentationSize.isWide) {
    failures.push(
      failure(
        "NOT_WIDE",
        "Presentation slide size is not within 0.01 of the 16:9 wide aspect ratio.",
        { aspectRatio: presentationSize.aspectRatio },
      ),
    );
  }

  return {
    schemaVersion: 1,
    artifact,
    sha256: createHash("sha256").update(buffer).digest("hex"),
    ok: failures.length === 0,
    expected: {
      slideCount: expectedSlides,
      notesSlideCount: expectedNotes,
      aspectRatio: "16:9",
    },
    summary: {
      packagePartCount: partNames.length,
      slideCount: slides.length,
      notesSlideCount: notesSlides.length,
      nonEmptyNotesCount: notesSlides.length - emptyNotes.length,
      relationshipPartCount: relationshipParts.length,
      internalRelationshipCount,
      externalRelationshipCount: externalRelationships.length,
      invalidGeometryPresetCount: invalidGeometryPresets.length,
    },
    presentationSize,
    noteTextLengths,
    emptyNotes,
    missingInternalTargets,
    invalidRelationshipTargets: invalidRelationshipParts,
    externalRelationships,
    invalidGeometryPresets,
    failures,
  };
}

function parseCliArguments(arguments_) {
  const directory = path.dirname(fileURLToPath(import.meta.url));
  const defaults = {
    input: path.resolve(directory, "../build/sleep-time-compute-research.pptx"),
    report: path.resolve(directory, "../reports/ooxml-qa.json"),
  };
  const parsed = { ...defaults };
  for (let index = 0; index < arguments_.length; index += 1) {
    const argument = arguments_[index];
    if (argument === "--input" || argument === "--report") {
      const value = arguments_[index + 1];
      if (!value) {
        throw new Error(`${argument} requires a path.`);
      }
      parsed[argument.slice(2)] = path.resolve(value);
      index += 1;
    } else if (argument === "--help" || argument === "-h") {
      parsed.help = true;
    } else {
      throw new Error(`Unknown argument: ${argument}`);
    }
  }
  return parsed;
}

async function writeReport(reportPath, report) {
  await fs.mkdir(path.dirname(reportPath), { recursive: true });
  await fs.writeFile(reportPath, `${JSON.stringify(report, null, 2)}\n`, "utf8");
}

async function main() {
  let arguments_;
  try {
    arguments_ = parseCliArguments(process.argv.slice(2));
  } catch (error) {
    console.error(`PPTX QA argument error: ${error.message}`);
    process.exitCode = 2;
    return;
  }
  if (arguments_.help) {
    console.log(
      "Usage: node src/inspect-pptx.mjs [--input PATH] [--report PATH]",
    );
    return;
  }

  const displayArtifact = path.relative(process.cwd(), arguments_.input);
  let report;
  try {
    const buffer = await fs.readFile(arguments_.input);
    report = await inspectPptxBuffer(buffer, {
      artifact: displayArtifact,
    });
  } catch (error) {
    report = {
      schemaVersion: 1,
      artifact: displayArtifact,
      ok: false,
      failures: [
        failure(
          "ARTIFACT_READ_ERROR",
          `Unable to read or parse the PPTX package: ${error.message}`,
        ),
      ],
    };
  }
  await writeReport(arguments_.report, report);
  const relativeReport = path.relative(process.cwd(), arguments_.report);
  if (!report.ok) {
    console.error(
      `PPTX OOXML QA failed with ${report.failures.length} finding(s). Report: ${relativeReport}`,
    );
    for (const finding of report.failures) {
      console.error(`- ${finding.code}: ${finding.message}`);
    }
    process.exitCode = 1;
    return;
  }
  console.log(
    `PPTX OOXML QA passed: ${report.summary.slideCount} slides, ` +
      `${report.summary.nonEmptyNotesCount} non-empty notes, ` +
      `${report.summary.externalRelationshipCount} external relationships. ` +
      `Report: ${relativeReport}`,
  );
}

const invokedAsScript =
  process.argv[1] &&
  pathToFileURL(path.resolve(process.argv[1])).href === import.meta.url;
if (invokedAsScript) {
  await main();
}
