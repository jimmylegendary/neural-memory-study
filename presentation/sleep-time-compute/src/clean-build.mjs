import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const sourceDir = path.dirname(fileURLToPath(import.meta.url));
const projectDir = path.resolve(sourceDir, "..");
const generatedFiles = [
  "build/sleep-time-compute-research.pptx",
  "build/sleep-time-compute-research.pdf",
  "build/contact-sheet.png",
  "reports/build-qa.json",
  "reports/ooxml-qa.json",
  "reports/render-qa.json",
];

for (const relativePath of generatedFiles) {
  const target = path.join(projectDir, relativePath);
  if (fs.existsSync(target)) fs.rmSync(target, { force: true });
}

process.stdout.write(`Removed ${generatedFiles.length} known generated targets when present.\n`);
