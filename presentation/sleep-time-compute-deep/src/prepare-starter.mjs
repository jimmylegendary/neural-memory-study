import fs from "node:fs/promises";
import path from "node:path";

import {
  ensureArtifactToolWorkspace,
  importArtifactTool,
  padSlideNumber,
  saveBlobToFile,
} from "/home/jimmy/.codex/plugins/cache/openai-primary-runtime/presentations/26.802.11031/skills/presentations/container_tools/artifact_tool_utils.mjs";
import { validateTemplatePlan } from "/home/jimmy/.codex/plugins/cache/openai-primary-runtime/presentations/26.802.11031/skills/presentations/template_following_scripts/validate_template_plan.mjs";
import { normalizeImportedNegativeExtents } from "./normalize-imported-geometry.mjs";

const repo = process.cwd();
const workspace = path.join(repo, "build/deck-work");
const sourcePath = path.join(repo, "presentation/sleep-time-compute/build/sleep-time-compute-research.pptx");
const mapPath = path.join(workspace, "template-frame-map.json");
const inspectPath = path.join(workspace, "template-inspect/template-inspect.ndjson");
const outputPath = path.join(workspace, "template-starter.pptx");
const previewDir = path.join(workspace, "template-starter-preview");
const layoutDir = path.join(workspace, "template-starter-layout");

await ensureArtifactToolWorkspace(workspace);
const { FileBlob, PresentationFile } = await importArtifactTool(workspace);
const frameMap = JSON.parse(await fs.readFile(mapPath, "utf8"));
const planCheck = await validateTemplatePlan({
  workspace,
  mapPath,
  inspectPath,
  sourceSlideCount: 28,
});
if (planCheck.status !== "pass") {
  throw new Error(`Template plan is not clean: ${planCheck.status} (${planCheck.issues.length} issues)`);
}

const presentation = await PresentationFile.importPptx(await FileBlob.load(sourcePath));
const sourceSlides = [...presentation.slides.items];
const geometryNormalization = normalizeImportedNegativeExtents(presentation);
if (geometryNormalization.length !== 10) {
  throw new Error(`Expected 10 imported negative extents; found ${geometryNormalization.length}`);
}

const entries = [...frameMap.outputSlides].sort((a, b) => a.outputSlide - b.outputSlide);
const starterSlides = entries.map((entry) => ({
  entry,
  slide: sourceSlides[entry.sourceSlide - 1].duplicate(),
}));

for (const slide of sourceSlides) slide.delete();
for (const [index, item] of starterSlides.entries()) item.slide.moveTo(index);

await fs.mkdir(previewDir, { recursive: true });
await fs.mkdir(layoutDir, { recursive: true });
const previews = [];
const layouts = [];
for (const [index, item] of starterSlides.entries()) {
  const n = padSlideNumber(index + 1);
  const previewPath = path.join(previewDir, `starter-slide-${n}.png`);
  const layoutPath = path.join(layoutDir, `starter-slide-${n}.layout.json`);
  await saveBlobToFile(await presentation.export({ slide: item.slide, format: "png", scale: 0.55 }), previewPath);
  await saveBlobToFile(await presentation.export({ slide: item.slide, format: "layout" }), layoutPath);
  previews.push(previewPath);
  layouts.push(layoutPath);
}

await fs.mkdir(path.dirname(outputPath), { recursive: true });
await (await PresentationFile.exportPptx(presentation)).save(outputPath);
const stat = await fs.stat(outputPath);
if (stat.size <= 0) throw new Error("Starter PPTX export is empty");

const manifest = {
  schema: "stc-template-starter/v1",
  sourcePptx: sourcePath,
  frameMap: mapPath,
  output: outputPath,
  outputBytes: stat.size,
  sourceSlideCount: sourceSlides.length,
  slideCount: starterSlides.length,
  planStatus: planCheck.status,
  geometryNormalization,
  authoringContract: "duplicate source slide, preserve geometry, rewrite inherited text only",
  slides: starterSlides.map(({ entry }, index) => ({
    outputSlide: index + 1,
    sourceSlide: entry.sourceSlide,
    narrativeRole: entry.narrativeRole,
    previewPath: previews[index],
    layoutPath: layouts[index],
  })),
};
await fs.writeFile(path.join(workspace, "template-starter.manifest.json"), `${JSON.stringify(manifest, null, 2)}\n`);
console.log(`starter=${outputPath}`);
console.log(`slides=${starterSlides.length} normalized-negative-extents=${geometryNormalization.length} bytes=${stat.size}`);
