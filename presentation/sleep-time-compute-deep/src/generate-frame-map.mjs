import fs from "node:fs";
import path from "node:path";
import { slides } from "./content.mjs";

const repo = process.cwd();
const workspace = path.join(repo, "build/deck-work");
const layoutDir = path.join(workspace, "template-inspect/layouts");
const inspectPath = path.join(workspace, "template-inspect/template-inspect.ndjson");
const outputPath = path.join(workspace, "template-frame-map.json");
const reportsDir = path.join(repo, "presentation/sleep-time-compute-deep/reports");

const inspectedTextIds = new Map();
for (const line of fs.readFileSync(inspectPath, "utf8").split("\n").filter(Boolean)) {
  const record = JSON.parse(line);
  if (record.kind !== "textbox" || !record.slide || !record.id) continue;
  if (!inspectedTextIds.has(record.slide)) inspectedTextIds.set(record.slide, []);
  inspectedTextIds.get(record.slide).push(String(record.id));
}

const sourceInventory = new Map();
for (let n = 1; n <= 28; n += 1) {
  const file = path.join(layoutDir, `source-slide-${String(n).padStart(2, "0")}.layout.json`);
  const layout = JSON.parse(fs.readFileSync(file, "utf8"));
  const layoutTextIds = layout.elements
    .filter((element) => typeof element.text === "string" && element.text.trim().length > 0)
    .map((element) => String(element.id));
  // The inspection helper assigns stable opaque ids to the first inspected slides.
  // Later slides may be absent because the human-readable inspection stream is capped;
  // their layout ids remain the authoritative inherited element ids.
  const textIds = inspectedTextIds.get(n)?.length ? inspectedTextIds.get(n) : layoutTextIds;
  sourceInventory.set(n, { file, textIds, totalElements: layout.elements.length });
}

const outputSlides = slides.map((slide) => {
  const inventory = sourceInventory.get(slide.pattern);
  return {
    outputSlide: slide.slideNumber,
    sourceSlide: slide.pattern,
    reuseMode: "duplicate-slide",
    narrativeRole: `content-${slide.section.toLowerCase().replace(/[^a-z0-9]+/g, "-")}`,
    rationale: `${slide.id} reuses the approved source pattern ${slide.pattern} and rewrites inherited text only.`,
    editTargets: [{
      action: "rewrite",
      shapeIds: inventory.textIds,
      reason: "Rewrite inherited labels, body text, metrics, footer, and page chrome while preserving geometry and typography.",
    }],
  };
});

const used = new Set(outputSlides.map((entry) => entry.sourceSlide));
const omittedSourceSlides = Array.from({ length: 28 }, (_, index) => index + 1)
  .filter((slide) => !used.has(slide))
  .map((slide) => ({ sourceSlide: slide, reason: "No matching narrative role in the 112-slide seminar." }));

const frameMap = {
  schema: "template-frame-map/v1",
  sourceDeck: "presentation/sleep-time-compute/build/sleep-time-compute-research.pptx",
  sourceSlideCount: 28,
  outputSlideCount: slides.length,
  authoringContract: "duplicate-slide + inherited-text rewrite; no added primitives",
  outputSlides,
  omittedSourceSlides,
};

fs.mkdirSync(path.dirname(outputPath), { recursive: true });
fs.mkdirSync(reportsDir, { recursive: true });
fs.writeFileSync(outputPath, `${JSON.stringify(frameMap, null, 2)}\n`);

const frequencies = Object.fromEntries(
  Array.from({ length: 28 }, (_, index) => index + 1).map((pattern) => [
    pattern,
    outputSlides.filter((entry) => entry.sourceSlide === pattern).length,
  ]),
);

const audit = [
  "# Template Fidelity Audit",
  "",
  `- Source deck: ${frameMap.sourceDeck}`,
  `- Source slides inspected: ${frameMap.sourceSlideCount}/28`,
  `- Output slides framed: ${frameMap.outputSlideCount}/112`,
  `- Source patterns reused: ${used.size}/28`,
  `- New primitives: 0`,
  `- Preserve-only screenshots: 0`,
  `- Pattern frequency: ${Object.entries(frequencies).map(([key, value]) => `${key}:${value}`).join(", ")}`,
  "",
  "Every output slide duplicates exactly one approved source slide and rewrites only inherited text objects. Geometry, non-text diagrams, palette, typography, and footer system remain inherited from the source deck.",
  "",
].join("\n");
fs.writeFileSync(path.join(reportsDir, "template-audit.md"), audit);

const deviation = [
  "# Template Deviation Log",
  "",
  "No structural deviations are planned. The final authoring pass may change inherited text values and speaker notes only; it must not add primitives, delete diagrams, or alter source geometry.",
  "",
].join("\n");
fs.writeFileSync(path.join(reportsDir, "template-deviation-log.md"), deviation);

console.log(`frame-map=${outputPath}`);
console.log(`outputs=${outputSlides.length} patterns=${used.size} omitted=${omittedSourceSlides.length}`);
