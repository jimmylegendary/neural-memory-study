import fs from "node:fs/promises";
import path from "node:path";

import {
  ensureArtifactToolWorkspace,
  importArtifactTool,
  padSlideNumber,
  saveBlobToFile,
} from "/home/jimmy/.codex/plugins/cache/openai-primary-runtime/presentations/26.805.11740/skills/presentations/container_tools/artifact_tool_utils.mjs";
import { deckMeta, slides as contentSlides } from "./content.mjs";

const repo = process.cwd();
const workspace = path.join(repo, "build/deck-work");
const starterPath = path.join(workspace, "template-starter.pptx");
const outputPath = path.join(repo, deckMeta.output);
const previewDir = path.join(repo, "presentation/sleep-time-compute-deep/build/slides");
const layoutDir = path.join(repo, "presentation/sleep-time-compute-deep/build/layouts");
const reportDir = path.join(repo, "presentation/sleep-time-compute-deep/reports");

const clip = (value, max) => {
  const text = String(value ?? "").replace(/\s+/g, " ").trim();
  return text.length <= max ? text : `${text.slice(0, Math.max(1, max - 1)).trimEnd()}…`;
};

const indexFromName = (name) => {
  const match = name.match(/-(\d+)$/);
  return match ? Number(match[1]) : undefined;
};

function textPlanner(slide) {
  const points = slide.points;
  const labels = points.map((point) => point.label);
  const bodies = points.map((point) => point.body);
  const metrics = [
    ...(slide.bigNumbers || []),
    ...points.map((point) => point.metric).filter(Boolean),
  ];
  const curves = slide.curveLabels || [];
  const references = slide.refs || [];
  let labelCursor = 0;
  let bodyCursor = 0;
  let metricCursor = 0;
  let curveCursor = 0;
  let referenceCursor = 0;
  let takeawayUsed = false;

  const pointAt = (index) => points[(index ?? 0) % points.length];
  const nextLabel = () => labels[labelCursor++ % labels.length];
  const nextBody = () => bodies[bodyCursor++ % bodies.length];
  const nextMetric = () => metrics.length ? metrics[metricCursor++ % metrics.length] : String(metricCursor++ + 1).padStart(2, "0");
  const nextCurve = () => curves.length ? curves[curveCursor++ % curves.length] : nextLabel();
  const nextReference = () => references[referenceCursor++ % references.length];

  return (shapeName, original) => {
    const name = shapeName.toLowerCase();
    const index = indexFromName(name);

    if (name.endsWith("-section")) return clip(slide.section, 22);
    if (name.endsWith("-slide-id")) return `STC-${String(slide.slideNumber).padStart(3, "0")}`;
    if (name.endsWith("-headline")) return clip(slide.title, 58);
    if (name.endsWith("-status-label")) return clip(slide.status, 16);
    if (name.endsWith("-source-footer")) return clip(slide.source, 105);
    if (name.endsWith("-page-number")) return String(slide.slideNumber);

    if (name.includes("cover-title")) return "SLEEP–TIME\nCOMPUTE";
    if (name.includes("cover-korean")) return "기억 생애주기의 다음 인프라";
    if (name.includes("cover-subtitle")) return clip(slide.coverSubtitle || slide.takeaway, 105);
    if (name.includes("cover-stage-")) return clip(pointAt(index).label, 18);
    if (name.includes("cover-manifest")) return clip(slide.takeaway, 72);
    if (name.includes("cover-type")) return "DEEP STUDY / SEMINAR / DEVICE AGENDA";
    if (name.includes("cover-date")) return deckMeta.freezeDate.replaceAll("-", ".");

    if (/ref[ab]-num-/.test(name)) return String((index ?? 0) + 1).padStart(2, "0");
    if (/ref[ab]-text-/.test(name)) {
      if ((index ?? 0) < points.length) {
        const point = pointAt(index);
        return clip(`${point.label} · ${point.body}`, 92);
      }
      return clip(nextReference(), 45);
    }

    if (name.includes("redteam-31")) return String(points.length).padStart(2, "0");
    if (name.includes("redteam-findings")) return `${clip(slide.status, 11)}\nFINDINGS`;
    if (name.includes("redteam-domains")) return points.slice(0, 4).map((point) => clip(point.label, 12)).join("\n");
    if (name.includes("redteam-condition-")) return clip(pointAt(index).label.toUpperCase(), 28);
    if (name.includes("redteam-action-")) return clip(pointAt(index).body, 45);

    if (name.includes("phase-tag-")) return clip(pointAt(index).label, 16);
    if (name.includes("phase-ko-")) return clip(pointAt(index).body.split(/[·,;]/)[0], 16);
    if (name.includes("phase-body-")) return clip(pointAt(index).body, 48);
    if (name.includes("phase-invariant-text")) return clip(slide.takeaway, 92);

    if (name.includes("plos-tag-")) return clip(pointAt(index).label, 16);
    if (name.includes("plos-body-")) return clip(pointAt(index).body, 42);
    if (name.includes("plos-842")) return clip(nextMetric(), 6);
    if (name.includes("plos-scope")) return `${points.length} stages\n${references.length} refs`;
    if (name.includes("plos-seq-label")) return clip(points[0].label, 16);
    if (name.includes("plos-seq-num")) return clip(nextMetric(), 20);
    if (name.includes("plos-sleep-label")) return clip(points[Math.min(1, points.length - 1)].label, 16);
    if (name.includes("plos-sleep-num")) return clip(nextMetric(), 28);
    if (name.endsWith("plos-limit")) return "LIMIT";
    if (name.includes("plos-limit-text")) return clip(slide.takeaway, 78);

    if (/op-num-|step-num-|bench-.*num-|contract-num-|matrix-num-/.test(name)) return clip(nextMetric(), 18);
    if (/op-tag-|step-tag-|stage-|phase-label-|atlas-label-|matrix-label-|card-.*title|item-.*title/.test(name)) {
      return clip(pointAt(index).label, 28);
    }
    if (/op-body-|step-body-|atlas-body-|contract-body-|matrix-formula-|card-.*body|item-.*body/.test(name)) {
      return clip(pointAt(index).body, 92);
    }

    if (/axis-.*label|xlabel|ylabel|cross-label|threshold-label/.test(name)) return clip(nextCurve(), 28);
    if (/equation|formula/.test(name)) return clip(slide.equation || pointAt(index).body, 92);
    if (/metric|number|value|score|big|stat|pct|ratio|accuracy|latency|bench-297|knee-keff/.test(name)) return clip(nextMetric(), 22);

    if (/takeaway|thesis-bar-text|bottom|manifest|conclusion|verdict|nonresult|unidentified|status-text|center-text/.test(name)) {
      takeawayUsed = true;
      return clip(slide.takeaway, 105);
    }

    if (index !== undefined) {
      const point = pointAt(index);
      if (/body|answer|desc|detail|copy|evidence|note|text|why|result|claim|limit|risk|formula/.test(name)) {
        return clip(point.body, 92);
      }
      if (/label|title|tag|name|event|company|layer|lane|tier|node|category|type|capability|check|item/.test(name)) {
        return clip(point.label, 30);
      }
      return clip(`${point.label} · ${point.body}`, 70);
    }

    if (/body|answer|desc|detail|copy|evidence|note|why|result|claim|limit|risk|question/.test(name)) return clip(nextBody(), 92);
    if (/label|title|tag|name|event|company|layer|lane|tier|node|category|type|capability|header/.test(name)) return clip(nextLabel(), 30);
    if (/year|num|count/.test(name)) return clip(nextMetric(), 18);

    if (!takeawayUsed && original.length > 62) {
      takeawayUsed = true;
      return clip(slide.takeaway, 100);
    }
    return original.length > 28 ? clip(nextBody(), 92) : clip(nextLabel(), 30);
  };
}

const CJK = /[ᄀ-ᇿ⺀-꓏가-힣豈-﫿＀-｠]/;
const visualWidth = (text) => [...text].reduce((sum, ch) => sum + (CJK.test(ch) ? 1 : 0.52), 0);

// Split one replacement string over `count` paragraphs, balanced by visual width so the
// inherited box keeps its line count instead of overflowing or collapsing.
function splitAcrossLines(text, count) {
  if (count <= 1) return [text];
  const words = String(text).split(/\s+/).filter(Boolean);
  if (words.length <= count) {
    const padded = words.slice();
    while (padded.length < count) padded.push("");
    return padded;
  }
  const target = words.reduce((sum, word) => sum + visualWidth(word) + 0.52, 0) / count;
  const lines = [];
  let current = [];
  let accumulated = 0;
  for (const word of words) {
    const width = visualWidth(word) + 0.52;
    if (current.length && accumulated + width > target && lines.length < count - 1) {
      lines.push(current.join(" "));
      current = [word];
      accumulated = width;
    } else {
      current.push(word);
      accumulated += width;
    }
  }
  lines.push(current.join(" "));
  while (lines.length < count) lines.push("");
  return lines;
}

function notesFor(slide) {
  return [
    `${slide.id} — ${slide.title}`,
    `핵심 결론: ${slide.takeaway}`,
    "",
    ...(slide.script ? ["[발표 대본]", slide.script, ""] : []),
    ...slide.points.map((point) => `• ${point.label}: ${point.body}${point.metric ? ` (${point.metric})` : ""}`),
    "",
    "[Sources]",
    ...slide.refs,
  ];
}

await ensureArtifactToolWorkspace(workspace);
const { FileBlob, PresentationFile } = await importArtifactTool(workspace);
const presentation = await PresentationFile.importPptx(await FileBlob.load(starterPath));
if (presentation.slides.items.length !== contentSlides.length) {
  throw new Error(`Starter/content mismatch: ${presentation.slides.items.length}/${contentSlides.length}`);
}

const editLog = [];
for (const [index, slide] of presentation.slides.items.entries()) {
  const content = contentSlides[index];
  const planText = textPlanner(content);
  let textShapeCount = 0;
  for (const shape of slide.shapes.items) {
    if (!shape.text || typeof shape.text.toString !== "function") continue;
    const original = shape.text.toString();
    if (!original.trim()) continue;
    const replacement = planText(shape.name || "", original);
    // Inherited shapes may hold several paragraphs. `replace` only matches inside one
    // paragraph, so a whole-string replace silently leaves multi-paragraph shapes at
    // their source text. Rewrite paragraph by paragraph and keep the paragraph count.
    const originalLines = original.split("\n");
    if (originalLines.length > 1) {
      const parts = splitAcrossLines(replacement, originalLines.length);
      originalLines.forEach((line, lineIndex) => {
        if (line.length) shape.text.replace(line, parts[lineIndex]);
      });
    } else {
      shape.text.replace(original, replacement);
    }
    textShapeCount += 1;
    editLog.push({
      slide: index + 1,
      shapeId: String(shape.id),
      shapeName: shape.name || "",
      beforeChars: original.length,
      afterChars: replacement.length,
      replacement,
    });
  }
  if (textShapeCount === 0) throw new Error(`${content.id}: no inherited text shapes were rewritten`);
  slide.speakerNotes.textFrame.setText(notesFor(content));
  slide.speakerNotes.setVisible(true);
}

await fs.mkdir(path.dirname(outputPath), { recursive: true });
await fs.mkdir(previewDir, { recursive: true });
await fs.mkdir(layoutDir, { recursive: true });
await fs.mkdir(reportDir, { recursive: true });

const previewPaths = [];
const layoutPaths = [];
for (const [index, slide] of presentation.slides.items.entries()) {
  const n = padSlideNumber(index + 1);
  const previewPath = path.join(previewDir, `slide-${n}.png`);
  const layoutPath = path.join(layoutDir, `slide-${n}.layout.json`);
  await saveBlobToFile(await presentation.export({ slide, format: "png", scale: 1 }), previewPath);
  await saveBlobToFile(await presentation.export({ slide, format: "layout" }), layoutPath);
  previewPaths.push(previewPath);
  layoutPaths.push(layoutPath);
}

await (await PresentationFile.exportPptx(presentation)).save(outputPath);
const stat = await fs.stat(outputPath);
if (stat.size <= 0) throw new Error("Final PPTX export is empty");

const manifest = {
  schema: "stc-deep-seminar/v1",
  title: deckMeta.title,
  freezeDate: deckMeta.freezeDate,
  sourceTemplate: deckMeta.template,
  starter: starterPath,
  output: outputPath,
  outputBytes: stat.size,
  slideCount: contentSlides.length,
  editedTextShapeCount: editLog.length,
  speakerNotesCount: contentSlides.length,
  authoringTool: "@oai/artifact-tool",
  authoringContract: "inherited text rewrite and speaker-notes edit only; no new slide primitives",
  slides: contentSlides.map((content, index) => ({
    slide: index + 1,
    id: content.id,
    sourcePattern: content.pattern,
    title: content.title,
    refs: content.refs,
    previewPath: previewPaths[index],
    layoutPath: layoutPaths[index],
  })),
};
await fs.writeFile(path.join(reportDir, "deck-manifest.json"), `${JSON.stringify(manifest, null, 2)}\n`);
await fs.writeFile(path.join(reportDir, "text-edit-log.json"), `${JSON.stringify(editLog, null, 2)}\n`);
console.log(`deck=${outputPath}`);
console.log(`slides=${contentSlides.length} text-shapes=${editLog.length} notes=${contentSlides.length} bytes=${stat.size}`);
