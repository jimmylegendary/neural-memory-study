import crypto from "node:crypto";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import PptxGenJS from "pptxgenjs";

const sourceDir = path.dirname(fileURLToPath(import.meta.url));
const projectDir = path.resolve(sourceDir, "..");
const buildDir = path.join(projectDir, "build");
const reportDir = path.join(projectDir, "reports");
const outputPath = path.join(buildDir, "sleep-time-compute-research.pptx");

fs.mkdirSync(buildDir, { recursive: true });
fs.mkdirSync(reportDir, { recursive: true });

const pptx = new PptxGenJS();
pptx.layout = "LAYOUT_WIDE";
pptx.author = "Neural Memory Research";
pptx.company = "Independent Research";
pptx.subject = "Sleep-time compute, continual learning, memory capacity, and systems infrastructure";
pptx.title = "Sleep-Time Compute: 기억 생애주기의 다음 인프라";
pptx.revision = "1";
pptx.theme = {
  headFontFace: "Noto Sans CJK KR",
  bodyFontFace: "Noto Sans CJK KR",
  lang: "ko-KR",
};

const W = 13.333;
const H = 7.5;
const SAFE = { x: 0.64, y: 0.42, w: 12.05, h: 6.67 };
const C = {
  paper: "F6F3EC",
  paper2: "ECE8DE",
  ink: "17191C",
  slate: "59616A",
  lightSlate: "A7ADB2",
  rule: "C9C4B8",
  cobalt: "244E8A",
  cobaltLight: "DDE6F2",
  rust: "9A442D",
  rustLight: "F0DED7",
  teal: "2F6B63",
  tealLight: "DDE9E4",
  white: "FFFFFF",
};
const FONT = {
  sans: "Noto Sans CJK KR",
  serif: "Noto Serif CJK KR",
  mono: "Noto Sans Mono CJK KR",
};

const slideMeta = [];
let current = null;

function idFor(name) {
  return `${current.id}-${name}`;
}

function register(kind, name, x, y, w, h, fontSize, role) {
  current.objects.push({ kind, name: idFor(name), x, y, w, h, fontSize, role });
}

function addText(slide, text, options, name, role = "body") {
  const defaults = {
    fontFace: FONT.sans,
    fontSize: 18,
    color: C.ink,
    margin: 0,
    breakLine: false,
    valign: "top",
    paraSpaceAfterPt: 0,
    objectName: idFor(name),
  };
  const merged = { ...defaults, ...options, objectName: idFor(name) };
  slide.addText(text, merged);
  register("text", name, merged.x, merged.y, merged.w, merged.h, merged.fontSize, role);
}

function addShape(slide, type, options, name, role = "visual") {
  const merged = { ...options, objectName: idFor(name) };
  slide.addShape(type, merged);
  register("shape", name, merged.x, merged.y, merged.w, merged.h, null, role);
}

function addLine(slide, x, y, w, h, color, width, name, arrow = "none", dash = "solid") {
  addShape(
    slide,
    pptx.ShapeType.line,
    {
      x,
      y,
      w,
      h,
      line: { color, width, endArrowType: arrow, dash },
    },
    name,
    "line",
  );
}

function rect(slide, x, y, w, h, fill, lineColor, name, transparency = 0) {
  addShape(
    slide,
    pptx.ShapeType.rect,
    {
      x,
      y,
      w,
      h,
      fill: { color: fill, transparency },
      line: { color: lineColor ?? fill, width: lineColor ? 1 : 0 },
    },
    name,
  );
}

function circle(slide, x, y, d, fill, lineColor, name) {
  addShape(
    slide,
    pptx.ShapeType.ellipse,
    {
      x,
      y,
      w: d,
      h: d,
      fill: { color: fill },
      line: { color: lineColor ?? fill, width: lineColor ? 1 : 0 },
    },
    name,
  );
}

function startSlide(id, section, title, status, source, noteBody, options = {}) {
  const slide = pptx.addSlide();
  slide.background = { color: options.background ?? C.paper };
  current = {
    id,
    section,
    title,
    status,
    source,
    objects: [],
    notes: "",
  };
  slideMeta.push(current);

  if (!options.cover) {
    addText(
      slide,
      section.toUpperCase(),
      { x: 0.66, y: 0.28, w: 3.3, h: 0.18, fontFace: FONT.mono, fontSize: 9.5, bold: true, color: C.slate, charSpacing: 1.6 },
      "section",
      "label",
    );
    addText(
      slide,
      id,
      { x: 11.73, y: 0.28, w: 0.92, h: 0.18, fontFace: FONT.mono, fontSize: 9.5, color: C.slate, align: "right" },
      "slide-id",
      "label",
    );
    addText(
      slide,
      title,
      { x: 0.66, y: 0.59, w: 12.0, h: 0.62, fontSize: options.titleSize ?? 30, bold: true, color: C.ink },
      "headline",
      "headline",
    );
  }

  if (status) {
    const statusColor = status === "근거" ? C.cobalt : status === "가설" ? C.rust : status === "제안" ? C.teal : C.slate;
    addText(
      slide,
      status,
      { x: 0.66, y: options.cover ? 0.56 : 1.28, w: 0.88, h: 0.24, fontFace: FONT.mono, fontSize: 10, bold: true, color: statusColor, align: "center", valign: "mid" },
      "status-label",
      "label",
    );
    addLine(slide, options.cover ? 1.68 : 1.62, options.cover ? 0.68 : 1.40, 0.62, 0, statusColor, 2.4, "status-rule");
  }

  if (!options.noFooter) {
    addLine(slide, 0.66, 7.08, 12.0, 0, C.rule, 0.7, "footer-rule");
    addText(
      slide,
      source,
      { x: 0.66, y: 7.15, w: 10.95, h: 0.16, fontSize: 8.5, color: C.slate },
      "source-footer",
      "citation",
    );
    addText(
      slide,
      `${slideMeta.length}`,
      { x: 11.96, y: 7.14, w: 0.7, h: 0.17, fontFace: FONT.mono, fontSize: 8.5, color: C.slate, align: "right" },
      "page-number",
      "citation",
    );
  }

  const notes = [
    `SLIDE_ID: ${id}`,
    `EVIDENCE_STATUS: ${status || "EDITORIAL"}`,
    `TAKEAWAY: ${title}`,
    `VISIBLE_SOURCE: ${source}`,
    "",
    noteBody,
  ].join("\n");
  slide.addNotes(notes);
  current.notes = notes;
  return slide;
}

function bandLabel(slide, text, x, y, w, color, name) {
  rect(slide, x, y, w, 0.38, color, null, `${name}-fill`);
  addText(slide, text, { x: x + 0.12, y: y + 0.08, w: w - 0.24, h: 0.22, fontSize: 13, bold: true, color: C.white }, `${name}-text`, "label");
}

function bulletList(slide, items, x, y, w, lineH, color, prefix, fontSize = 18) {
  items.forEach((item, index) => {
    circle(slide, x, y + index * lineH + 0.12, 0.09, color, color, `${prefix}-dot-${index}`);
    addText(
      slide,
      item,
      { x: x + 0.25, y: y + index * lineH, w: w - 0.25, h: lineH - 0.05, fontSize, color: C.ink, valign: "mid" },
      `${prefix}-item-${index}`,
      "body",
    );
  });
}

function axis(slide, x, y, w, h, xLabel, yLabel, name) {
  addLine(slide, x, y + h, w, 0, C.ink, 1.2, `${name}-x`, "triangle");
  addLine(slide, x, y + h, 0, -h, C.ink, 1.2, `${name}-y`, "triangle");
  addText(slide, xLabel, { x: x + w - 1.6, y: y + h + 0.16, w: 1.6, h: 0.2, fontSize: 11, color: C.slate, align: "right" }, `${name}-xlabel`, "label");
  addText(slide, yLabel, { x: x - 0.45, y: y - 0.12, w: 1.6, h: 0.2, fontSize: 11, color: C.slate, rotate: 270 }, `${name}-ylabel`, "label");
}

// 01 — cover
{
  const slide = startSlide(
    "STC-01",
    "thesis",
    "",
    "통합 추론",
    "Audited corpus cutoff · 2026-07-25",
    "This is an evidence-bounded position and research-agenda deck. It reports no new PAPER-C result. The deck uses only native PowerPoint text, shapes, and connectors.",
    { cover: true, noFooter: true },
  );
  addLine(slide, 0.76, 0.45, 11.8, 0, C.ink, 5, "cover-rule");
  addText(slide, "SLEEP–TIME\nCOMPUTE", { x: 0.78, y: 1.52, w: 5.25, h: 1.85, fontFace: FONT.serif, fontSize: 44, bold: true, color: C.ink, breakLine: true }, "cover-title", "headline");
  addText(slide, "기억 생애주기의 다음 인프라", { x: 0.82, y: 3.64, w: 5.4, h: 0.46, fontSize: 24, bold: true, color: C.teal }, "cover-korean", "headline");
  addText(
    slide,
    "wake에서 포착하고 · sleep에서 변환하고 · 장기기억에서 검증·검색·삭제한다",
    { x: 0.82, y: 4.28, w: 5.35, h: 0.86, fontSize: 18, color: C.slate, breakLine: true, valign: "mid" },
    "cover-subtitle",
    "body",
  );
  rect(slide, 7.04, 1.34, 5.06, 4.78, C.paper2, null, "cover-field");
  ["WAKE", "SNAPSHOT", "SLEEP", "MEMORY"].forEach((label, index) => {
    const y = 1.74 + index * 1.03;
    addText(slide, label, { x: 7.42, y, w: 1.42, h: 0.28, fontFace: FONT.mono, fontSize: 13, bold: true, color: index < 2 ? C.cobalt : index === 2 ? C.rust : C.teal }, `cover-stage-${index}`, "label");
    addLine(slide, 8.95, y + 0.14, 2.1, 0, C.ink, 1.3, `cover-arrow-${index}`, index < 3 ? "triangle" : "none");
  });
  addText(slide, "Evidence → candidate → verified manifest", { x: 7.42, y: 5.66, w: 4.08, h: 0.25, fontSize: 13, color: C.slate, align: "center" }, "cover-manifest", "label");
  addText(slide, "POSITION / REVIEW / RESEARCH AGENDA", { x: 0.82, y: 6.72, w: 5.6, h: 0.22, fontFace: FONT.mono, fontSize: 10, color: C.slate, charSpacing: 1.2 }, "cover-type", "label");
  addText(slide, "2026.07.25", { x: 10.94, y: 6.72, w: 1.2, h: 0.22, fontFace: FONT.mono, fontSize: 10, color: C.slate, align: "right" }, "cover-date", "label");
}

// 02 — central claim
{
  const slide = startSlide(
    "STC-02",
    "thesis",
    "핵심은 ‘자는 모델’이 아니라 기억의 이동 규칙이다",
    "통합 추론",
    "CLS · continual learning · Letta 2025 · Google Sleep 2026",
    "The phase label is useful only when it predicts operator timing, data visibility, publication, and later reuse. External versus parametric is a routing decision, not an ideology.",
  );
  addText(slide, "언제", { x: 0.8, y: 1.83, w: 1.05, h: 0.42, fontFace: FONT.serif, fontSize: 28, bold: true, color: C.cobalt }, "when", "headline");
  addText(slide, "무엇을", { x: 0.8, y: 3.02, w: 1.35, h: 0.42, fontFace: FONT.serif, fontSize: 28, bold: true, color: C.rust }, "what", "headline");
  addText(slide, "어디로", { x: 0.8, y: 4.21, w: 1.35, h: 0.42, fontFace: FONT.serif, fontSize: 28, bold: true, color: C.teal }, "where", "headline");
  addLine(slide, 2.35, 2.04, 8.5, 0, C.rule, 1, "when-line");
  addLine(slide, 2.35, 3.23, 8.5, 0, C.rule, 1, "what-line");
  addLine(slide, 2.35, 4.42, 8.5, 0, C.rule, 1, "where-line");
  addText(slide, "배포 전  |  wake / test-time  |  delayed sleep", { x: 2.6, y: 1.77, w: 7.95, h: 0.45, fontSize: 21, bold: true }, "when-answer", "body");
  addText(slide, "replay · distill · dream · merge · verify · forget", { x: 2.6, y: 2.96, w: 8.6, h: 0.45, fontSize: 21, bold: true }, "what-answer", "body");
  addText(slide, "text / vector / graph  →  latent / adapter  →  shared weight", { x: 2.6, y: 4.15, w: 9.25, h: 0.45, fontSize: 21, bold: true }, "where-answer", "body");
  rect(slide, 2.58, 5.24, 8.94, 0.8, C.ink, null, "thesis-bar");
  addText(slide, "모든 이동은 provenance · capacity · rollback · deletion을 함께 가져야 한다", { x: 2.85, y: 5.47, w: 8.4, h: 0.35, fontSize: 19, bold: true, color: C.white, align: "center" }, "thesis-bar-text", "body");
}

// 03 — operational definition
{
  const slide = startSlide(
    "STC-03",
    "definition",
    "strict sleep은 세 조건을 동시에 만족한다",
    "제안",
    "Operational definition synthesized from audited literature",
    "A job is strict sleep only when it is off the current action's causal critical path, consumes accumulated wake experience, and publishes durable state reused by a later wake. The definition is operational, not biological.",
  );
  const steps = [
    ["1", "CAUSAL BOUNDARY", "현재 사용자 action의\ncritical path 밖에서 실행"],
    ["2", "LIFETIME INPUT", "정적 pretraining corpus가 아니라\n누적 wake experience를 변환"],
    ["3", "DURABLE REUSE", "검증된 persistent state를 publish하고\n다음 wake가 재사용"],
  ];
  steps.forEach(([num, label, body], index) => {
    const x = 0.92 + index * 4.03;
    circle(slide, x, 2.05, 0.62, index === 0 ? C.cobalt : index === 1 ? C.rust : C.teal, null, `def-num-${index}`);
    addText(slide, num, { x: x, y: 2.17, w: 0.62, h: 0.24, fontSize: 15, bold: true, color: C.white, align: "center" }, `def-num-text-${index}`, "label");
    addText(slide, label, { x: x + 0.84, y: 2.06, w: 2.7, h: 0.25, fontFace: FONT.mono, fontSize: 12, bold: true, color: C.slate }, `def-label-${index}`, "label");
    addText(slide, body, { x: x + 0.84, y: 2.48, w: 2.92, h: 1.12, fontSize: 19, bold: true, breakLine: true }, `def-body-${index}`, "body");
    if (index < 2) addLine(slide, x + 3.38, 3.99, 0.92, 0, C.rule, 1.3, `def-link-${index}`, "triangle");
  });
  addLine(slide, 1.05, 4.34, 11.1, 0, C.ink, 1.2, "def-separator");
  addText(slide, "따라서", { x: 1.05, y: 4.69, w: 0.85, h: 0.28, fontFace: FONT.serif, fontSize: 18, bold: true, color: C.rust }, "therefore", "label");
  addText(slide, "online summary ≠ sleep   ·   offline corpus distill ≠ deployed sleep   ·   idle GPU ≠ memory consolidation", { x: 2.05, y: 4.62, w: 10.08, h: 0.48, fontSize: 18, bold: true }, "definition-nonclaims", "body");
  addText(slide, "이 정의는 scheduling label과 computational phase를 분리한다.", { x: 2.05, y: 5.48, w: 8.6, h: 0.35, fontSize: 20, color: C.teal, bold: true }, "definition-bottom", "body");
}

// 04 — three phases
{
  const slide = startSlide(
    "STC-04",
    "framing",
    "앞으로의 학습은 세 phase와 하나의 memory fabric으로 나뉜다",
    "통합 추론",
    "Sun et al. 2020 · Lin et al. 2025 · Behrouz et al. 2026",
    "Predeployment learning creates priors and update rules. Wake learning operates under latency and user-visible constraints. Sleep learning transforms a frozen snapshot. A separate memory fabric owns canonical and derived state.",
  );
  const xs = [0.84, 3.93, 7.02, 10.11];
  const phases = [
    ["PRE", "배포 전", "pretrain · post-train\nupdate rule 학습", C.slate],
    ["WAKE", "사용 중", "infer · retrieve\nTTT · capture", C.cobalt],
    ["SLEEP", "휴면 경계", "replay · distill\ncompact · validate", C.rust],
    ["MEMORY", "지속 상태", "canonical evidence\nviews · adapters", C.teal],
  ];
  phases.forEach(([tag, korean, body, color], index) => {
    addText(slide, tag, { x: xs[index], y: 1.92, w: 2.2, h: 0.25, fontFace: FONT.mono, fontSize: 12, bold: true, color }, `phase-tag-${index}`, "label");
    addText(slide, korean, { x: xs[index], y: 2.3, w: 2.2, h: 0.38, fontFace: FONT.serif, fontSize: 23, bold: true }, `phase-ko-${index}`, "headline");
    addLine(slide, xs[index], 2.89, 2.2, 0, color, 4, `phase-rule-${index}`);
    addText(slide, body, { x: xs[index], y: 3.26, w: 2.35, h: 1.12, fontSize: 18, breakLine: true }, `phase-body-${index}`, "body");
    if (index < 3) addLine(slide, xs[index] + 2.4, 3.0, 0.52, 0, C.ink, 1.1, `phase-arrow-${index}`, "triangle");
  });
  rect(slide, 1.3, 5.18, 10.73, 0.66, C.paper2, C.rule, "phase-invariant");
  addText(slide, "fixed base  +  session/user-local mutable state  +  shared slow state  +  external canonical truth", { x: 1.57, y: 5.38, w: 10.2, h: 0.27, fontSize: 17, bold: true, align: "center" }, "phase-invariant-text", "body");
}

// 05 — chronology
{
  const slide = startSlide(
    "STC-05",
    "lineage",
    "‘sleep-time’이라는 이름보다 계보가 먼저였다",
    "근거",
    "Hinton 1995 · Golden 2022 · MemGPT 2023 · Lin 2025 · Google Sleep 2026",
    "The river is chronological, not a single dependency chain. It separates replay and consolidation, external memory, latent compilation, and recurring parametric sleep.",
  );
  addLine(slide, 1.0, 3.83, 11.2, 0, C.ink, 2, "timeline");
  const events = [
    [1.05, "1995", "Wake–Sleep\nCLS", C.slate, true],
    [3.07, "2017–20", "replay · EWC\nTTT", C.cobalt, false],
    [5.1, "2022", "PLOS SNN\nPAD / replay", C.rust, true],
    [7.12, "2023–24", "MemGPT\nlatent memory", C.teal, false],
    [9.15, "2025", "Sleep-time\nLetta · SMF", C.cobalt, true],
    [11.18, "2026", "fast→slow\nparametric sleep", C.rust, false],
  ];
  events.forEach(([x, year, label, color, up], index) => {
    circle(slide, x, 3.63, 0.4, color, C.paper, `event-dot-${index}`);
    const y = up ? 1.82 : 4.45;
    addLine(slide, x + 0.2, up ? 2.98 : 4.03, 0, up ? 0.65 : 0.42, color, 1.2, `event-stem-${index}`);
    addText(slide, year, { x: x - 0.15, y, w: 1.45, h: 0.26, fontFace: FONT.mono, fontSize: 12, bold: true, color }, `event-year-${index}`, "label");
    addText(slide, label, { x: x - 0.15, y: y + 0.39, w: 1.62, h: 0.82, fontSize: 17, bold: true, breakLine: true }, `event-label-${index}`, "body");
  });
  addText(slide, "직선적 진보가 아니다", { x: 0.98, y: 5.82, w: 2.15, h: 0.3, fontFace: FONT.serif, fontSize: 18, bold: true, color: C.rust }, "chrono-warning", "label");
  addText(slide, "online replay · query-time compilation · external lifecycle · parametric consolidation은 timing과 destination이 다르다.", { x: 3.22, y: 5.76, w: 8.85, h: 0.48, fontSize: 18, bold: true }, "chrono-warning-text", "body");
}

// 06 — biology transfer boundary
{
  const slide = startSlide(
    "STC-06",
    "biology",
    "생물학은 operator 후보를 준다—역할표를 복사하라는 뜻은 아니다",
    "근거",
    "Wilson & McNaughton 1994 · Rasch et al. 2007 · Deperrois et al. 2022",
    "Evidence supports experience-related replay and some causal effects of cueing or rhythm. It does not establish the neat engineering slogan NREM=exact replay and REM=creative adversarial generation.",
  );
  bandLabel(slide, "전이 가능한 근거", 0.88, 1.78, 5.35, C.cobalt, "bio-left");
  bandLabel(slide, "전이하면 안 되는 비약", 7.08, 1.78, 5.35, C.rust, "bio-right");
  bulletList(slide, ["경험 관련 activity의 sleep replay", "cue·rhythm 조작이 다음날 recall을 변화", "fast episodic ↔ slow structured system", "replay·downscaling·recombination 후보"], 1.04, 2.48, 5.05, 0.78, C.cobalt, "bio-evidence", 18);
  bulletList(slide, ["hippocampus→cortex ‘파일 이동’", "NREM=정답 replay / REM=창의성의 확정", "동물 결과를 LLM lifetime law로 일반화", "sleep이면 이미 지워진 정보도 복원"], 7.24, 2.48, 5.05, 0.78, C.rust, "bio-nonclaim", 18);
  addLine(slide, 6.66, 1.84, 0, 4.52, C.rule, 1.2, "bio-divider");
  addText(slide, "ENGINEERING RULE", { x: 4.95, y: 5.95, w: 3.42, h: 0.24, fontFace: FONT.mono, fontSize: 10, bold: true, color: C.teal, align: "center" }, "bio-rule-label", "label");
  addText(slide, "stage 이름보다 data · objective · publication · later reuse를 기록한다", { x: 2.16, y: 6.25, w: 9.04, h: 0.36, fontSize: 20, bold: true, color: C.teal, align: "center" }, "bio-rule", "body");
}

// 07 — PLOS
{
  const slide = startSlide(
    "STC-07",
    "direct evidence",
    "2022 PLOS는 ‘학회’가 아니라 두 과제 SNN 저널 실험이다",
    "근거",
    "Golden et al. · PLOS Computational Biology 18(11), e1010628 · 2022",
    "Wake uses reward-modulated STDP; sleep disables external input, applies Poisson activation and unsupervised STDP, and moves toward a joint solution. Numbers are source-reported and scoped to the experiment.",
  );
  const nodes = [
    [0.92, "WAKE 1", "Task 1\nreward-STDP", C.cobalt],
    [3.3, "WAKE 2", "Task 2\ninterference", C.rust],
    [5.68, "SLEEP", "Poisson activation\nunsupervised STDP", C.teal],
    [8.55, "LATER WAKE", "joint weight\nrepresentation", C.cobalt],
  ];
  nodes.forEach(([x, tag, body, color], index) => {
    addText(slide, tag, { x, y: 1.93, w: 1.85, h: 0.2, fontFace: FONT.mono, fontSize: 10.5, bold: true, color }, `plos-tag-${index}`, "label");
    rect(slide, x, 2.28, 1.9, 1.25, index === 2 ? C.tealLight : index === 1 ? C.rustLight : C.cobaltLight, color, `plos-node-${index}`);
    addText(slide, body, { x: x + 0.15, y: 2.57, w: 1.6, h: 0.72, fontSize: 17, bold: true, align: "center", breakLine: true }, `plos-body-${index}`, "body");
    if (index < nodes.length - 1) addLine(slide, x + 1.97, 2.9, 0.95, 0, C.ink, 1.2, `plos-arrow-${index}`, "triangle");
  });
  rect(slide, 10.95, 1.82, 1.45, 2.2, C.ink, null, "plos-number-field");
  addText(slide, "842", { x: 11.08, y: 2.12, w: 1.2, h: 0.55, fontFace: FONT.serif, fontSize: 32, bold: true, color: C.white, align: "center" }, "plos-842", "headline");
  addText(slide, "neurons\n2 tasks\n≥10 inits", { x: 11.08, y: 2.82, w: 1.2, h: 0.85, fontSize: 13, color: C.white, align: "center", breakLine: true }, "plos-scope", "label");
  addText(slide, "순차 학습", { x: 1.0, y: 4.53, w: 1.3, h: 0.22, fontSize: 13, bold: true, color: C.rust }, "plos-seq-label", "label");
  addText(slide, "Task 1  0.52±0.02", { x: 2.35, y: 4.46, w: 2.62, h: 0.3, fontFace: FONT.mono, fontSize: 16, bold: true }, "plos-seq-num", "body");
  addText(slide, "wake↔sleep", { x: 1.0, y: 5.22, w: 1.3, h: 0.22, fontSize: 13, bold: true, color: C.teal }, "plos-sleep-label", "label");
  addText(slide, "Task 1/2  0.70±0.03 / 0.68±0.05", { x: 2.35, y: 5.15, w: 4.62, h: 0.3, fontFace: FONT.mono, fontSize: 16, bold: true }, "plos-sleep-num", "body");
  addLine(slide, 7.3, 4.38, 0, 1.56, C.rule, 1.2, "plos-limit-divider");
  addText(slide, "한계", { x: 7.68, y: 4.44, w: 0.65, h: 0.25, fontFace: FONT.serif, fontSize: 18, bold: true, color: C.rust }, "plos-limit", "label");
  addText(slide, "old trace가 완전히 사라진 뒤에는 복원하지 못했다.\nLLM · 다과제 · 장기용량 · deletion은 검증하지 않았다.", { x: 8.45, y: 4.39, w: 3.76, h: 1.05, fontSize: 17, breakLine: true }, "plos-limit-text", "body");
}

// 08 — PAD and DANN
{
  const slide = startSlide(
    "STC-08",
    "evidence tiers",
    "PAD는 계산 가설, DANN은 낮은 증거 branch다",
    "근거",
    "Deperrois et al. · eLife 2022 | Tutuncuoglu · SSRN 2025",
    "PAD gives NREM and REM different objectives in a representation-learning setup. DANN exists as a manuscript record, but the auditable evidence is insufficient for a zero-forgetting claim.",
  );
  addText(slide, "PAD", { x: 0.92, y: 1.9, w: 1.25, h: 0.45, fontFace: FONT.serif, fontSize: 29, bold: true, color: C.cobalt }, "pad-title", "headline");
  addText(slide, "Perturbed & Adversarial Dreaming", { x: 2.18, y: 1.97, w: 3.75, h: 0.28, fontSize: 16, bold: true }, "pad-full", "body");
  addLine(slide, 0.94, 2.55, 5.2, 0, C.cobalt, 3, "pad-rule");
  addText(slide, "NREM", { x: 1.02, y: 3.02, w: 1.0, h: 0.25, fontFace: FONT.mono, fontSize: 12, bold: true, color: C.cobalt }, "pad-nrem", "label");
  addText(slide, "episodic latent → occlusion → reconstruction", { x: 2.02, y: 2.96, w: 3.95, h: 0.38, fontSize: 18, bold: true }, "pad-nrem-text", "body");
  addText(slide, "REM", { x: 1.02, y: 3.85, w: 1.0, h: 0.25, fontFace: FONT.mono, fontSize: 12, bold: true, color: C.rust }, "pad-rem", "label");
  addText(slide, "memory mix + noise → adversarial dream", { x: 2.02, y: 3.79, w: 3.95, h: 0.38, fontSize: 18, bold: true }, "pad-rem-text", "body");
  addText(slide, "범위: CIFAR-10 / SVHN representation learning", { x: 1.02, y: 5.02, w: 4.9, h: 0.34, fontSize: 16, color: C.slate }, "pad-scope", "body");

  addLine(slide, 6.66, 1.78, 0, 4.56, C.rule, 1.2, "pad-dann-divider");
  addText(slide, "DANN", { x: 7.22, y: 1.9, w: 1.55, h: 0.45, fontFace: FONT.serif, fontSize: 29, bold: true, color: C.rust }, "dann-title", "headline");
  addText(slide, "Dream-Augmented Neural Networks", { x: 8.82, y: 1.97, w: 3.25, h: 0.28, fontSize: 16, bold: true }, "dann-full", "body");
  addLine(slide, 7.22, 2.55, 5.18, 0, C.rust, 3, "dann-rule");
  bulletList(slide, ["SSRN metadata·abstract 확인", "architecture/data/seed/code 독립 감사 불가", "‘zero forgetting’은 사용 금지", "방향성 아이디어로만 유지"], 7.34, 2.96, 4.86, 0.69, C.rust, "dann-list", 18);
  rect(slide, 7.34, 5.58, 4.82, 0.54, C.rustLight, C.rust, "dann-verdict");
  addText(slide, "EVIDENCE GRADE  ·  LOW / DIRECTIONAL", { x: 7.56, y: 5.76, w: 4.38, h: 0.22, fontFace: FONT.mono, fontSize: 11, bold: true, color: C.rust, align: "center" }, "dann-verdict-text", "label");
}

// 09 — phase × destination
{
  const slide = startSlide(
    "STC-09",
    "taxonomy",
    "phase와 기억 위치는 독립 축이다",
    "통합 추론",
    "MemGPT · Cartridges · SMF · Google Sleep · external-memory audits",
    "A method can run at wake or sleep and write to external, latent, adapter, or shared parameters. Conflating phase with destination creates false comparisons.",
  );
  const x0 = 2.1;
  const y0 = 2.02;
  const cw = 2.25;
  const rh = 1.03;
  const cols = ["EXTERNAL", "LATENT / KV", "ADAPTER", "SHARED WEIGHT"];
  const rows = ["WAKE", "SLEEP", "PREDEPLOY"];
  cols.forEach((label, index) => {
    addText(slide, label, { x: x0 + index * cw, y: 1.58, w: cw - 0.1, h: 0.26, fontFace: FONT.mono, fontSize: 10.5, bold: true, color: C.slate, align: "center" }, `matrix-col-${index}`, "label");
  });
  rows.forEach((label, row) => {
    addText(slide, label, { x: 0.75, y: y0 + row * rh + 0.36, w: 1.15, h: 0.22, fontFace: FONT.mono, fontSize: 11, bold: true, color: row === 0 ? C.cobalt : row === 1 ? C.rust : C.slate, align: "right" }, `matrix-row-${row}`, "label");
    for (let col = 0; col < cols.length; col += 1) {
      rect(slide, x0 + col * cw, y0 + row * rh, cw - 0.08, rh - 0.08, row === 1 ? C.rustLight : row === 0 ? C.cobaltLight : C.paper2, C.paper, `matrix-cell-${row}-${col}`);
    }
  });
  const items = [
    [0, 0, "Mem0 · Graphiti"],
    [0, 1, "TTT · Titans"],
    [0, 2, "fast-LoRA"],
    [1, 0, "Letta sleeper\nAuto-Dreamer"],
    [1, 1, "Cartridges\nOffline Recurrence"],
    [1, 2, "Doc→LoRA"],
    [1, 3, "Google Sleep"],
    [2, 0, "memory-policy\npretraining"],
    [2, 2, "adapter generator"],
    [2, 3, "SMF substrate"],
  ];
  items.forEach(([row, col, text], index) => {
    addText(slide, text, { x: x0 + col * cw + 0.12, y: y0 + row * rh + 0.25, w: cw - 0.32, h: 0.48, fontSize: 14.5, bold: true, align: "center", breakLine: true }, `matrix-item-${index}`, "label");
  });
  addText(slide, "핵심 질문", { x: 0.82, y: 5.54, w: 1.15, h: 0.26, fontFace: FONT.serif, fontSize: 18, bold: true, color: C.teal }, "matrix-question-label", "label");
  addText(slide, "같은 future utility를 만들 때 어느 phase × destination이 가장 적은 lifecycle cost와 가장 강한 reversibility를 가지는가?", { x: 2.14, y: 5.43, w: 9.76, h: 0.67, fontSize: 20, bold: true }, "matrix-question", "body");
}

// 10 — agent systems
{
  const slide = startSlide(
    "STC-10",
    "agent memory",
    "MemGPT·Letta·Mem0·Zep은 같은 ‘sleep’이 아니다",
    "근거",
    "Packer et al. 2023 · Lin et al. 2025 · Mem0/Zep audits + current product docs",
    "The slide separates paper, product, and implementation versions. It does not treat all background memory operations as a learned sleep phase.",
  );
  const systems = [
    ["2023", "MemGPT", "context pressure · recursive summary", "wake", "별도 sleeper 없음", C.cobalt],
    ["2025", "Letta paper", "pre-query context compilation", "sleep-time", "durable memory와 구분", C.rust],
    ["2025", "Letta Sleeper", "async core-memory editing", "product S", "paper benchmark와 구분", C.teal],
    ["2025", "Mem0 paper / OSS", "ADD · UPDATE · DELETE · NOOP", "mostly wake", "mutation과 phase를 구분", C.cobalt],
    ["NOW", "Mem0 Platform V3", "asynchronous ADD-only path", "async", "later-read 경로", C.cobalt],
    ["2025", "Graphiti OSS", "temporal graph invalidation", "ingest", "scheduler 미소유", C.teal],
    ["NOW", "Zep Cloud / Obs.", "async later-read / observer", "async / prop.", "Cloud PASS · operator 비공개", C.teal],
  ];
  systems.forEach(([year, name, mechanism, phase, boundary, color], index) => {
    const y = 1.63 + index * 0.67;
    addText(slide, year, { x: 0.8, y: y + 0.08, w: 0.65, h: 0.18, fontFace: FONT.mono, fontSize: 9.5, bold: true, color: C.slate }, `agent-year-${index}`, "label");
    addLine(slide, 1.56, y + 0.17, 0.72, 0, color, 3, `agent-year-rule-${index}`);
    addText(slide, name, { x: 2.48, y, w: 2.28, h: 0.29, fontSize: 16.2, bold: true }, `agent-name-${index}`, "body");
    addText(slide, mechanism, { x: 4.92, y: y + 0.01, w: 3.15, h: 0.32, fontSize: 14.1 }, `agent-mech-${index}`, "label");
    addText(slide, phase, { x: 8.18, y: y + 0.07, w: 1.42, h: 0.22, fontFace: FONT.mono, fontSize: 9.4, bold: true, color, align: "center" }, `agent-phase-${index}`, "label");
    addText(slide, boundary, { x: 9.75, y: y + 0.01, w: 2.37, h: 0.32, fontSize: 12.1, color: C.slate, align: "right" }, `agent-boundary-${index}`, "label");
  });
  addText(slide, "paper ≠ OSS ≠ cloud ≠ proprietary operator", { x: 7.95, y: 6.28, w: 4.17, h: 0.22, fontFace: FONT.mono, fontSize: 9.4, bold: true, color: C.rust, align: "right" }, "agent-version", "label");
}

// 11 — company lineages
{
  const slide = startSlide(
    "STC-11",
    "company lineages",
    "Google·Meta·Microsoft는 서로 다른 병목을 공략한다",
    "근거",
    "Audited public papers and implementations · company-lineage audit",
    "Affiliation labels follow audited public records. The three bands are parallel lineages, not claims about the companies as a whole.",
  );
  const bands = [
    [1.72, "GOOGLE / DEEPMIND", C.cobalt, "CLS · EWC · Titans/Miras", "fast→slow expert · synthetic dream", "experimental parametric"],
    [3.2, "META / FAIR", C.rust, "GEM · MAS · Product-Key Memory", "Memory Layers · sparse row update", "capacity & low-interference write"],
    [4.68, "MICROSOFT", C.teal, "LongMem · Generative Adapter", "HIMA · MEMENTO · RHO", "external schedule & harness"],
  ];
  bands.forEach(([y, label, color, history, currentText, verdict], index) => {
    addText(slide, label, { x: 0.82, y, w: 2.35, h: 0.25, fontFace: FONT.mono, fontSize: 11, bold: true, color }, `company-label-${index}`, "label");
    addLine(slide, 3.28, y + 0.14, 0.9, 0, color, 4, `company-start-${index}`);
    addText(slide, history, { x: 4.42, y: y - 0.04, w: 2.62, h: 0.37, fontSize: 17, bold: true }, `company-history-${index}`, "body");
    addLine(slide, 7.06, y + 0.14, 0.65, 0, C.rule, 1.2, `company-mid-${index}`, "triangle");
    addText(slide, currentText, { x: 7.92, y: y - 0.04, w: 2.68, h: 0.37, fontSize: 17, bold: true }, `company-current-${index}`, "body");
    addText(slide, verdict, { x: 10.77, y: y - 0.03, w: 1.5, h: 0.42, fontSize: 13, bold: true, color, align: "right" }, `company-verdict-${index}`, "label");
  });
  rect(slide, 0.82, 6.06, 11.45, 0.46, C.paper2, C.rule, "company-nonclaim");
  addText(slide, "비주장  ·  공개된 논문 계보를 회사의 단일 제품 전략·production deployment로 일반화하지 않는다", { x: 1.04, y: 6.2, w: 11.0, h: 0.2, fontSize: 14, color: C.slate, align: "center" }, "company-nonclaim-text", "label");
}

// 12 — operators
{
  const slide = startSlide(
    "STC-12",
    "training",
    "sleep은 하나의 optimizer가 아니라 operator pipeline이다",
    "통합 추론",
    "Replay · distillation · dream data · RL · isolation · external compaction",
    "The pipeline makes data selection, transformation, objective, validation, and publication explicit. It permits no-op and rollback.",
  );
  const ops = [
    ["ADMIT", "raw · rare · negative", C.cobalt],
    ["BUILD", "replay · coreset · dream", C.rust],
    ["UPDATE", "distill · RL · sparse write", C.teal],
    ["VERIFY", "utility · preservation · harm", C.cobalt],
    ["PUBLISH", "atomic manifest swap", C.teal],
    ["UNDO", "rollback · delete · rebuild", C.rust],
  ];
  ops.forEach(([tag, body, color], index) => {
    const angle = (Math.PI * 2 * index) / ops.length - Math.PI / 2;
    const cx = 6.6 + Math.cos(angle) * 3.65;
    const cy = 4.02 + Math.sin(angle) * 1.95;
    circle(slide, cx - 0.36, cy - 0.36, 0.72, color, C.paper, `op-node-${index}`);
    addText(slide, `${index + 1}`, { x: cx - 0.36, y: cy - 0.22, w: 0.72, h: 0.22, fontSize: 14, bold: true, color: C.white, align: "center" }, `op-num-${index}`, "label");
    addText(slide, tag, { x: cx - 1.12, y: cy + 0.52, w: 2.24, h: 0.22, fontFace: FONT.mono, fontSize: 11, bold: true, color, align: "center" }, `op-tag-${index}`, "label");
    addText(slide, body, { x: cx - 1.45, y: index === 0 ? 1.61 : cy + 0.84, w: 2.9, h: 0.42, fontSize: 14.5, align: "center" }, `op-body-${index}`, "label");
    const next = (index + 1) % ops.length;
    const nextAngle = (Math.PI * 2 * next) / ops.length - Math.PI / 2;
    const nx = 6.6 + Math.cos(nextAngle) * 3.65;
    const ny = 4.02 + Math.sin(nextAngle) * 1.95;
    addLine(slide, cx, cy, nx - cx, ny - cy, C.rule, 1.1, `op-link-${index}`, "triangle");
  });
  circle(slide, 5.5, 2.92, 2.2, C.ink, null, "op-center");
  addText(slide, "SLEEP\nTRANSACTION", { x: 5.68, y: 3.55, w: 1.84, h: 0.72, fontFace: FONT.mono, fontSize: 15, bold: true, color: C.white, align: "center", breakLine: true }, "op-center-text", "label");
}

// 13 — data construction
{
  const slide = startSlide(
    "STC-13",
    "sleep data",
    "좋은 dream은 미래 query utility를 보존한다",
    "제안",
    "PAD · Cartridges · SEAL · Google Dreaming · Auto-Dreamer",
    "A sleep-data manifest should make coverage, grounding, anti-collapse mixture, provenance, and future-task holdouts auditable.",
  );
  const layers = [
    [1.82, 9.8, "HOLDOUT FUTURE TASK", "dream likelihood가 아니라 다음 wake utility", C.ink],
    [2.68, 8.4, "GROUNDING", "source provenance · tool outcome · simulator", C.teal],
    [3.54, 7.0, "ANTI-COLLAPSE MIX", "raw · teacher · student · adversarial · random", C.cobalt],
    [4.4, 5.6, "NOVEL RECOMBINATION", "counterfactual · composition · edge case", C.rust],
    [5.26, 4.2, "COVERAGE", "old · rare · negative · correction", C.slate],
  ];
  layers.forEach(([y, w, label, body, color], index) => {
    const x = 6.67 - w / 2;
    rect(slide, x, y, w, 0.62, index === 0 ? C.ink : index === 1 ? C.tealLight : index === 2 ? C.cobaltLight : index === 3 ? C.rustLight : C.paper2, color, `data-layer-${index}`);
    addText(slide, label, { x: x + 0.18, y: y + 0.18, w: 1.8, h: 0.2, fontFace: FONT.mono, fontSize: 10.5, bold: true, color: index === 0 ? C.white : color }, `data-label-${index}`, "label");
    addText(slide, body, { x: x + 1.95, y: y + 0.1, w: w - 2.15, h: 0.42, fontSize: index < 2 ? 14.5 : 13.5, bold: true, color: index === 0 ? C.white : C.ink, align: "right" }, `data-body-${index}`, "label");
  });
  addText(slide, "teacher corruption과 omission은 sleep을 반복할수록 축적될 수 있다", { x: 2.5, y: 6.15, w: 8.3, h: 0.34, fontSize: 18, bold: true, color: C.rust, align: "center" }, "data-warning", "body");
}

// 14 — external vs parametric
{
  const slide = startSlide(
    "STC-14",
    "destination",
    "external-first, parameter-last는 방어 가능한 default다",
    "통합 추론",
    "Mem0/Zep · Cartridges · LoRA/SMF · Google Sleep · deletion audits",
    "The default is not a theorem. Promotion requires repeated validated reuse, low volatility, addressability, and a rollback or canonical recovery path.",
  );
  addLine(slide, 1.05, 4.27, 11.1, -1.66, C.ink, 2.2, "destination-diagonal", "triangle");
  const points = [
    [1.2, 4.18, "RAW / EVENT", "audit·delete ↑", C.cobalt],
    [3.5, 3.83, "TEXT / GRAPH", "semantic view", C.teal],
    [5.85, 3.46, "VECTOR / KV", "compact cache", C.teal],
    [8.15, 3.1, "ADAPTER", "user-local delta", C.rust],
    [10.55, 2.74, "SHARED WEIGHT", "reuse·generalize ↑", C.rust],
  ];
  points.forEach(([x, y, label, sub, color], index) => {
    circle(slide, x, y, 0.34, color, C.paper, `dest-point-${index}`);
    addText(slide, label, { x: x - 0.4, y: y - 0.68, w: 1.95, h: 0.23, fontFace: FONT.mono, fontSize: 10.5, bold: true, color }, `dest-label-${index}`, "label");
    addText(slide, sub, { x: x - 0.4, y: y + 0.5, w: 1.95, h: 0.22, fontSize: 13.5, color: C.slate }, `dest-sub-${index}`, "label");
  });
  addText(slide, "reversibility · provenance · correction", { x: 0.98, y: 5.36, w: 3.72, h: 0.26, fontSize: 16, bold: true, color: C.cobalt }, "dest-left-axis", "body");
  addText(slide, "reuse amortization · generalization", { x: 7.9, y: 1.7, w: 4.17, h: 0.24, fontSize: 14.5, bold: true, color: C.rust, align: "right" }, "dest-right-axis", "label");
  rect(slide, 2.28, 5.86, 8.76, 0.5, C.paper2, C.rule, "dest-rule");
  addText(slide, "PROMOTE only if  expected future value  >  compile + serve + interfere + govern + risk", { x: 2.52, y: 6.01, w: 8.28, h: 0.22, fontFace: FONT.mono, fontSize: 11.5, bold: true, align: "center" }, "dest-rule-text", "label");
}

// 15 — capacities
{
  const slide = startSlide(
    "STC-15",
    "capacity",
    "기억용량은 최소 다섯 개이고 먼저 차는 것이 다르다",
    "통합 추론",
    "bounded synapses · continual-learning bounds · agent-memory systems",
    "Storage is only one pressure. Retention, addressability, plasticity, governance, and service throughput can bind first.",
  );
  const caps = [
    ["01", "STORAGE", "bytes · rows · edges · slots", C.slate],
    ["02", "RETENTION", "old trace above recall threshold", C.cobalt],
    ["03", "ADDRESSABILITY", "find and apply under live budget", C.teal],
    ["04", "PLASTICITY", "learn new as well as fresh control", C.rust],
    ["05", "GOVERNANCE", "provenance · delete · rollback", C.ink],
  ];
  caps.forEach(([num, label, body, color], index) => {
    const y = 1.72 + index * 0.92;
    addText(slide, num, { x: 0.8, y: y + 0.12, w: 0.58, h: 0.22, fontFace: FONT.mono, fontSize: 11, bold: true, color }, `cap-num-${index}`, "label");
    rect(slide, 1.55, y, 2.18, 0.52, color, null, `cap-label-box-${index}`);
    addText(slide, label, { x: 1.72, y: y + 0.16, w: 1.84, h: 0.2, fontFace: FONT.mono, fontSize: 11, bold: true, color: C.white }, `cap-label-${index}`, "label");
    addText(slide, body, { x: 4.05, y: y + 0.1, w: 4.02, h: 0.3, fontSize: 17.5, bold: true }, `cap-body-${index}`, "body");
    const pressureW = [3.3, 4.0, 4.7, 3.8, 4.3][index];
    rect(slide, 8.35, y + 0.1, pressureW, 0.28, color, null, `cap-pressure-${index}`);
  });
  addText(slide, "first binding pressure", { x: 9.92, y: 6.2, w: 1.72, h: 0.22, fontFace: FONT.mono, fontSize: 10, color: C.rust, align: "right" }, "cap-binding", "label");
}

// 16 — no free resurrection
{
  const slide = startSlide(
    "STC-16",
    "information boundary",
    "근거가 사라지면 sleep도 특정 기억을 부활시킬 수 없다",
    "통합 추론",
    "Data-processing boundary · PLOS failure condition · replay literature",
    "If target information is absent from all retained answer-bearing state and independent of new randomness, later computation cannot reconstruct that specific target without new evidence.",
  );
  const stages = [
    [0.95, "TARGET X", C.cobalt],
    [3.2, "RETAINED Ωₜ", C.teal],
    [6.05, "SLEEP f(Ωₜ,U)", C.rust],
    [9.15, "LATER STATE", C.ink],
  ];
  stages.forEach(([x, label, color], index) => {
    rect(slide, x, 2.45, 2.05, 0.78, index === 3 ? C.ink : index === 2 ? C.rustLight : index === 1 ? C.tealLight : C.cobaltLight, color, `nfr-stage-${index}`);
    addText(slide, label, { x: x + 0.12, y: 2.71, w: 1.81, h: 0.24, fontFace: FONT.mono, fontSize: 12, bold: true, color: index === 3 ? C.white : color, align: "center" }, `nfr-label-${index}`, "label");
    if (index < 3) addLine(slide, x + 2.16, 2.83, 0.89, 0, C.ink, 1.3, `nfr-arrow-${index}`, "triangle");
  });
  addText(slide, "I(X; Ωₜ) = 0", { x: 3.17, y: 3.62, w: 2.15, h: 0.38, fontFace: FONT.serif, fontSize: 23, bold: true, color: C.rust, align: "center" }, "nfr-equation", "headline");
  addText(slide, "⇒ 특정 X의 복원 불가", { x: 6.2, y: 3.62, w: 2.45, h: 0.38, fontFace: FONT.serif, fontSize: 23, bold: true, color: C.rust, align: "center" }, "nfr-result", "headline");
  addLine(slide, 5.4, 3.82, 0.72, 0, C.rust, 1.5, "nfr-result-arrow", "triangle");
  rect(slide, 1.25, 4.66, 10.76, 1.16, C.paper2, C.rule, "nfr-carriers");
  addText(slide, "정보가 남는 곳", { x: 1.52, y: 4.95, w: 1.35, h: 0.28, fontSize: 17, bold: true, color: C.teal }, "nfr-carriers-label", "label");
  addText(slide, "raw event · replay buffer · teacher · generator · residual weight · adapter · checkpoint · archive · implicit KV", { x: 3.05, y: 4.85, w: 8.55, h: 0.5, fontSize: 17.5, bold: true, align: "center" }, "nfr-carriers-text", "body");
  addText(slide, "hidden carrier도 capacity·privacy·delete ledger에 포함한다", { x: 3.05, y: 5.44, w: 8.55, h: 0.23, fontSize: 14, color: C.slate, align: "center" }, "nfr-carriers-note", "label");
}

// 17 — scaling atlas
{
  const slide = startSlide(
    "STC-17",
    "scaling",
    "단일 N^α보다 crossover atlas가 더 가능성 높다",
    "가설",
    "Scaling-law agenda · all curves schematic / unmeasured",
    "Every curve is a proposal. A relation becomes a law only after held-out prediction, alternative-model competition, and scope-specific claim-down.",
  );
  axis(slide, 1.1, 2.05, 4.65, 3.55, "reuse / load", "value", "scale-left");
  addLine(slide, 1.3, 4.9, 3.85, -2.05, C.cobalt, 2.2, "scale-external");
  addLine(slide, 1.3, 2.75, 3.85, 2.15, C.rust, 2.2, "scale-parametric");
  circle(slide, 3.25, 3.76, 0.24, C.ink, C.ink, "scale-cross");
  addText(slide, "reuse break-even", { x: 3.5, y: 3.51, w: 1.75, h: 0.22, fontSize: 13, bold: true, color: C.ink }, "scale-cross-label", "label");
  addText(slide, "external", { x: 1.7, y: 4.48, w: 1.1, h: 0.2, fontSize: 12, color: C.cobalt }, "scale-external-label", "label");
  addText(slide, "compiled", { x: 4.36, y: 4.5, w: 1.1, h: 0.2, fontSize: 12, color: C.rust }, "scale-parametric-label", "label");
  addLine(slide, 6.55, 1.92, 0, 4.25, C.rule, 1.2, "scale-divider");
  const atlas = [
    ["CAPACITY KNEE", "association load × state budget", C.rust],
    ["CADENCE", "staleness × run cost × burst", C.cobalt],
    ["QUEUE", "arrival × service × clocks", C.teal],
    ["ROUTING", "module isolation × catalog tax", C.rust],
    ["RECOVERY", "compression × reversibility reserve", C.slate],
  ];
  atlas.forEach(([label, body, color], index) => {
    const y = 1.87 + index * 0.86;
    addText(slide, label, { x: 7.08, y, w: 1.82, h: 0.22, fontFace: FONT.mono, fontSize: 10.5, bold: true, color }, `atlas-label-${index}`, "label");
    addLine(slide, 9.06, y + 0.12, 0.55, 0, color, 3, `atlas-rule-${index}`);
    addText(slide, body, { x: 9.82, y: y - 0.03, w: 2.37, h: 0.3, fontSize: 15.5, bold: true }, `atlas-body-${index}`, "body");
  });
  addText(slide, "SCHEMATIC · NOT A RESULT", { x: 8.95, y: 6.25, w: 3.25, h: 0.22, fontFace: FONT.mono, fontSize: 10, bold: true, color: C.rust, align: "right" }, "scale-nonresult", "label");
}

// 18 — knee
{
  const slide = startSlide(
    "STC-18",
    "capacity knee",
    "capacity는 ‘최대 몇 개’보다 품질이 꺾이는 경계로 측정한다",
    "가설",
    "Threshold-anchored logistic candidate · schematic / unmeasured",
    "A knee is reported only when low/high brackets, monotonicity, single crossing, residual checks, and shape-constrained sensitivity agree. Otherwise the result is KNEE_UNIDENTIFIED.",
  );
  axis(slide, 1.25, 1.85, 7.05, 4.35, "association load  Nₐ", "retention", "knee-axis");
  addLine(slide, 1.42, 2.35, 1.1, 0.03, C.cobalt, 2.5, "knee-seg-1");
  addLine(slide, 2.52, 2.38, 1.1, 0.12, C.cobalt, 2.5, "knee-seg-2");
  addLine(slide, 3.62, 2.5, 1.1, 0.42, C.cobalt, 2.5, "knee-seg-3");
  addLine(slide, 4.72, 2.92, 1.1, 1.12, C.rust, 2.5, "knee-seg-4");
  addLine(slide, 5.82, 4.04, 1.1, 1.42, C.rust, 2.5, "knee-seg-5");
  addLine(slide, 6.92, 5.46, 0.9, 0.18, C.rust, 2.5, "knee-seg-6");
  addLine(slide, 1.25, 3.22, 7.05, 0, C.slate, 1, "knee-threshold", "none", "dash");
  addText(slide, "τret = 0.80", { x: 1.45, y: 3.02, w: 1.25, h: 0.2, fontFace: FONT.mono, fontSize: 10.5, color: C.slate }, "knee-threshold-label", "label");
  addLine(slide, 5.15, 3.16, 0, 2.74, C.rust, 1.4, "knee-marker", "none", "dash");
  addText(slide, "Keff", { x: 4.79, y: 5.9, w: 0.75, h: 0.24, fontFace: FONT.serif, fontSize: 18, bold: true, color: C.rust, align: "center" }, "knee-keff", "label");
  const checks = ["low/high bracket", "monotonic", "single crossing", "residual envelope", "shape sensitivity"];
  addText(slide, "KNEE AUTHORIZATION", { x: 9.02, y: 1.95, w: 2.8, h: 0.25, fontFace: FONT.mono, fontSize: 11, bold: true, color: C.rust }, "knee-check-title", "label");
  bulletList(slide, checks, 9.05, 2.45, 2.78, 0.61, C.rust, "knee-checks", 15.5);
  rect(slide, 8.98, 5.72, 2.96, 0.57, C.ink, null, "knee-unidentified");
  addText(slide, "else  KNEE_UNIDENTIFIED", { x: 9.13, y: 5.91, w: 2.66, h: 0.22, fontFace: FONT.mono, fontSize: 10.5, bold: true, color: C.white, align: "center" }, "knee-unidentified-text", "label");
}

// 19 — cadence and queue
{
  const slide = startSlide(
    "STC-19",
    "service law",
    "sleep clock은 정확도보다 queue와 staleness의 공동 문제다",
    "가설",
    "Cadence and queue candidates · schematic / unmeasured",
    "The analytic optimum is a restricted special case. Bursts, batch economies, deadlines, multiple clocks, and validation work can overturn it.",
  );
  axis(slide, 0.95, 1.98, 5.0, 3.88, "interval τ", "cost", "cadence-axis");
  addLine(slide, 1.25, 2.42, 0.8, 0.28, C.cobalt, 2.1, "cadence-down-1");
  addLine(slide, 2.05, 2.7, 0.8, 0.58, C.cobalt, 2.1, "cadence-down-2");
  addLine(slide, 2.85, 3.28, 0.8, 0.7, C.rust, 2.1, "cadence-up-1");
  addLine(slide, 3.65, 3.98, 0.8, 0.86, C.rust, 2.1, "cadence-up-2");
  addLine(slide, 4.45, 4.84, 0.85, 0.64, C.rust, 2.1, "cadence-up-3");
  circle(slide, 3.48, 3.68, 0.22, C.ink, C.ink, "cadence-opt");
  addText(slide, "τ*", { x: 3.28, y: 4.04, w: 0.62, h: 0.22, fontFace: FONT.serif, fontSize: 17, bold: true }, "cadence-opt-label", "label");
  addText(slide, "run cost", { x: 1.35, y: 2.95, w: 0.95, h: 0.2, fontSize: 11, color: C.cobalt }, "cadence-run", "label");
  addText(slide, "staleness", { x: 4.35, y: 4.31, w: 1.02, h: 0.2, fontSize: 11, color: C.rust }, "cadence-stale", "label");
  addLine(slide, 6.55, 1.85, 0, 4.42, C.rule, 1.2, "service-divider");
  addText(slide, "QUEUE CONDITION", { x: 7.05, y: 1.95, w: 2.08, h: 0.24, fontFace: FONT.mono, fontSize: 11, bold: true, color: C.teal }, "queue-title", "label");
  addText(slide, "ρs = λs E[Ss] / m  <  1", { x: 7.08, y: 2.55, w: 4.55, h: 0.5, fontFace: FONT.serif, fontSize: 27, bold: true, color: C.ink }, "queue-equation", "headline");
  bulletList(slide, ["mean-load 필요조건이지 tail 보장 아님", "GPU·IO·RAM·HBM을 typed vector로", "delete deadline과 p99 job age 포함", "clock 수가 늘면 catalog·routing tax 증가"], 7.08, 3.38, 4.82, 0.63, C.teal, "queue-list", 16.5);
  addText(slide, "HYPOTHESIS  ·  queue-before-storage", { x: 7.1, y: 6.15, w: 4.73, h: 0.22, fontFace: FONT.mono, fontSize: 10.5, bold: true, color: C.rust }, "queue-hyp", "label");
}

// 20 — infrastructure
{
  const slide = startSlide(
    "STC-20",
    "infrastructure",
    "세 logical plane과 하나의 control plane이 필요하다",
    "제안",
    "System infrastructure blueprint · no production superiority claim",
    "Logical ownership does not imply three physical clusters. A fleet may be shared or separated, but snapshot, publication, authorization, and deletion boundaries remain explicit.",
  );
  const columns = [
    [0.82, "WAKE PLANE", "infer · retrieve\ncapture · fast state", C.cobalt],
    [4.35, "SLEEP PLANE", "replay · dream\nFT · RL · compact", C.rust],
    [7.88, "MEMORY FABRIC", "text · vector · graph\nlatent · adapter", C.teal],
  ];
  columns.forEach(([x, label, body, color], index) => {
    rect(slide, x, 2.88, 2.75, 2.25, index === 0 ? C.cobaltLight : index === 1 ? C.rustLight : C.tealLight, color, `infra-plane-${index}`);
    addText(slide, label, { x: x + 0.2, y: 3.18, w: 2.35, h: 0.25, fontFace: FONT.mono, fontSize: 11.5, bold: true, color, align: "center" }, `infra-label-${index}`, "label");
    addText(slide, body, { x: x + 0.25, y: 3.85, w: 2.25, h: 0.8, fontSize: 18, bold: true, align: "center", breakLine: true }, `infra-body-${index}`, "body");
    if (index < 2) addLine(slide, x + 2.84, 4.0, 0.58, 0, C.ink, 1.2, `infra-arrow-${index}`, "triangle");
  });
  rect(slide, 1.62, 1.57, 8.95, 0.72, C.ink, null, "infra-control");
  addText(slide, "CONTROL PLANE  ·  admit / route / schedule / verify / publish / undo", { x: 1.94, y: 1.81, w: 8.32, h: 0.25, fontFace: FONT.mono, fontSize: 12.5, bold: true, color: C.white, align: "center" }, "infra-control-text", "label");
  addLine(slide, 3.08, 2.3, 0, 0.48, C.ink, 1.1, "infra-control-wake", "triangle");
  addLine(slide, 6.61, 2.3, 0, 0.48, C.ink, 1.1, "infra-control-sleep", "triangle");
  addLine(slide, 10.14, 2.3, 0, 0.48, C.ink, 1.1, "infra-control-memory", "triangle");
  rect(slide, 11.23, 1.57, 1.03, 3.56, C.paper2, C.rule, "infra-ledger");
  addText(slide, "RESOURCE\nLEDGER", { x: 11.38, y: 2.72, w: 0.73, h: 0.72, fontFace: FONT.mono, fontSize: 11, bold: true, color: C.slate, align: "center", breakLine: true, rotate: 270 }, "infra-ledger-text", "label");
  addText(slide, "latency-sensitive", { x: 0.95, y: 5.65, w: 2.42, h: 0.22, fontSize: 13, color: C.cobalt, align: "center" }, "infra-latency", "label");
  addText(slide, "throughput-oriented", { x: 4.52, y: 5.65, w: 2.42, h: 0.22, fontSize: 13, color: C.rust, align: "center" }, "infra-throughput", "label");
  addText(slide, "tiered persistent state", { x: 8.02, y: 5.65, w: 2.42, h: 0.22, fontSize: 13, color: C.teal, align: "center" }, "infra-tiered", "label");
}

// 21 — state transaction
{
  const slide = startSlide(
    "STC-21",
    "state transaction",
    "sleep 완료는 job 종료가 아니라 atomic publication이다",
    "제안",
    "ServingHead / MVCC / CAS / deletion-epoch design",
    "A sleep job may create text, graph, index, adapter, and policy artifacts. They become visible as one versioned manifest after validation; a partial multi-store update is not completion.",
  );
  const tx = [
    [0.88, "v17", "AUTHZ\nHEAD", C.cobalt],
    [3.42, "SNAP", "immutable\nwake snapshot", C.slate],
    [5.96, "CAND", "multi-store\ncandidate", C.rust],
    [8.5, "VERIFY", "utility · harm\nprovenance", C.teal],
    [11.04, "v18", "ATOMIC\nMANIFEST", C.ink],
  ];
  tx.forEach(([x, tag, body, color], index) => {
    addText(slide, tag, { x, y: 1.88, w: 1.35, h: 0.22, fontFace: FONT.mono, fontSize: 11, bold: true, color, align: "center" }, `tx-tag-${index}`, "label");
    rect(slide, x, 2.28, 1.55, 1.18, index === 4 ? C.ink : index === 2 ? C.rustLight : index === 3 ? C.tealLight : index === 1 ? C.paper2 : C.cobaltLight, color, `tx-node-${index}`);
    addText(slide, body, { x: x + 0.12, y: 2.57, w: 1.31, h: 0.62, fontSize: index === 0 ? 15 : 15.5, bold: true, color: index === 4 ? C.white : C.ink, align: "center", breakLine: true }, `tx-body-${index}`, "body");
    if (index < 4) addLine(slide, x + 1.62, 2.86, 0.82, 0, C.ink, 1.2, `tx-arrow-${index}`, "triangle");
  });
  addLine(slide, 8.95, 3.57, -3.4, 1.23, C.rust, 1.4, "tx-reject", "triangle", "dash");
  addText(slide, "REJECT / ROLLBACK", { x: 5.52, y: 4.9, w: 2.3, h: 0.22, fontFace: FONT.mono, fontSize: 10.5, bold: true, color: C.rust }, "tx-reject-label", "label");
  rect(slide, 1.48, 5.38, 10.25, 0.76, C.paper2, C.rule, "tx-invariants");
  addText(slide, "manifest generation + digest + authorization epoch + deletion epoch", { x: 1.8, y: 5.63, w: 9.62, h: 0.23, fontFace: FONT.mono, fontSize: 12, bold: true, align: "center" }, "tx-invariants-text", "label");
  addText(slide, "foreground request는 version을 pin하고, deletion은 같은 CAS word를 전진시킨다", { x: 2.0, y: 6.33, w: 9.35, h: 0.24, fontSize: 16, color: C.teal, align: "center" }, "tx-bottom", "body");
}

// 22 — benchmark
{
  const slide = startSlide(
    "STC-22",
    "benchmark",
    "297-cell benchmark는 lifecycle 선택을 비교한다",
    "제안",
    "BENCHMARK-EXPERIMENT-BLUEPRINT · proposed, preregistration-ready / unexecuted",
    "The benchmark is a design, not a result. It uses matched information, native resource envelopes, frozen TRAIN/CALIBRATION/TEST splits, and scored resource failures.",
  );
  addText(slide, "297", { x: 0.84, y: 1.83, w: 2.05, h: 1.02, fontFace: FONT.serif, fontSize: 54, bold: true, color: C.ink }, "bench-297", "headline");
  addText(slide, "logical cells", { x: 0.88, y: 2.83, w: 1.9, h: 0.28, fontSize: 17, bold: true, color: C.slate }, "bench-logical", "body");
  addLine(slide, 3.04, 2.3, 0.58, 0, C.ink, 1.4, "bench-arrow-1", "triangle");
  const matrices = [
    [3.78, "189", "ROUTING × MEDIUM", "6 deployable × 3 reuse × 3 pressure × 3 seeds = 162\n+ oracle on 27 operating cells", C.cobalt, C.cobaltLight],
    [8.18, "108", "SCHEDULE × OPERATOR", "2 schedules × 2 operators × 3 reuse\n× 3 pressure × 3 seeds", C.rust, C.rustLight],
  ];
  matrices.forEach(([x, num, label, formula, color, fill], index) => {
    rect(slide, x, 1.78, 3.88, 1.6, fill, color, `bench-matrix-${index}`);
    addText(slide, num, { x: x + 0.2, y: 1.98, w: 1.02, h: 0.45, fontFace: FONT.serif, fontSize: 29, bold: true, color }, `bench-matrix-num-${index}`, "headline");
    addText(slide, label, { x: x + 1.22, y: 2.05, w: 2.35, h: 0.26, fontFace: FONT.mono, fontSize: 10.3, bold: true, color }, `bench-matrix-label-${index}`, "label");
    addText(slide, formula, { x: x + 0.22, y: 2.55, w: 3.44, h: 0.57, fontSize: 12.8, bold: true, align: "center", breakLine: true }, `bench-matrix-formula-${index}`, "label");
  });
  addLine(slide, 0.88, 3.86, 11.35, 0, C.rule, 1.2, "bench-divider");
  const contracts = [
    ["48", "fixed blocks\n16 families × 3 seeds"],
    ["131", "global rows\n104 continuous + 27 RF"],
    ["20k", "blocked paired\nbootstrap"],
    ["ITT", "failures score zero"],
  ];
  contracts.forEach(([num, body], index) => {
    const x = 1.0 + index * 2.87;
    addText(slide, num, { x, y: 4.28, w: 1.25, h: 0.46, fontFace: FONT.serif, fontSize: 28, bold: true, color: index === 1 ? C.rust : index === 3 ? C.teal : C.cobalt }, `bench-contract-num-${index}`, "headline");
    addText(slide, body, { x: x + 1.13, y: 4.28, w: 1.48, h: 0.72, fontSize: 14.5, bold: true, breakLine: true }, `bench-contract-body-${index}`, "label");
  });
  rect(slide, 1.0, 5.75, 10.98, 0.58, C.ink, null, "bench-status");
  addText(slide, "UNEXECUTED  ·  proposed, preregistration-ready  ·  no PAPER-C result", { x: 1.25, y: 5.94, w: 10.48, h: 0.22, fontFace: FONT.mono, fontSize: 11.5, bold: true, color: C.white, align: "center" }, "bench-status-text", "label");
}

// 23 — open questions
{
  const slide = startSlide(
    "STC-23",
    "open questions",
    "리서치의 가치는 답보다 answerable question을 만드는 데 있다",
    "가설",
    "RQ1–RQ12 answerability matrix · H-STC and X-STC registry",
    "Questions are separated by what literature can answer, what matched experiments require, and what only long deployment can measure.",
  );
  const columns = [
    [0.82, "NOW", C.cobalt, ["phase 정의", "prior mechanism", "capacity necessity"]],
    [4.42, "MATCHED EXPERIMENT", C.rust, ["destination routing", "cadence advantage", "NREM/REM factorization"]],
    [8.02, "LONG DEPLOYMENT", C.teal, ["lifetime equilibrium", "governance debt", "cross-hardware law"]],
  ];
  columns.forEach(([x, tag, color, items], index) => {
    addText(slide, tag, { x, y: 1.8, w: 2.85, h: 0.25, fontFace: FONT.mono, fontSize: 11.5, bold: true, color }, `oq-tag-${index}`, "label");
    addLine(slide, x, 2.2, 2.9, 0, color, 4, `oq-rule-${index}`);
    bulletList(slide, items, x + 0.04, 2.62, 3.02, 0.76, color, `oq-list-${index}`, 17);
  });
  addLine(slide, 3.95, 1.74, 0, 3.48, C.rule, 1, "oq-div-1");
  addLine(slide, 7.55, 1.74, 0, 3.48, C.rule, 1, "oq-div-2");
  addText(slide, "확장 가설", { x: 0.84, y: 5.54, w: 1.16, h: 0.25, fontFace: FONT.serif, fontSize: 18, bold: true, color: C.rust }, "oq-hyp-title", "label");
  addText(slide, "queue-before-storage · usable-capacity release · routing knee · clock-count crossover · recovery-substrate floor", { x: 2.18, y: 5.43, w: 9.84, h: 0.68, fontSize: 18, bold: true }, "oq-hyp-list", "body");
}

// 24 — falsification
{
  const slide = startSlide(
    "STC-24",
    "red team",
    "좋은 sleep 연구는 실패했을 때도 무엇을 배웠는지 남긴다",
    "제안",
    "31-item scientific red-team audit · claim-down contract",
    "A null or falsified result remains publishable when the benchmark, resource accounting, and negative-result path are frozen before TEST.",
  );
  rect(slide, 0.85, 1.73, 3.12, 4.72, C.rust, null, "redteam-field");
  addText(slide, "31", { x: 1.2, y: 2.08, w: 2.42, h: 0.88, fontFace: FONT.serif, fontSize: 52, bold: true, color: C.white, align: "center" }, "redteam-31", "headline");
  addText(slide, "adversarial\nfindings", { x: 1.2, y: 3.08, w: 2.42, h: 0.68, fontSize: 20, bold: true, color: C.white, align: "center", breakLine: true }, "redteam-findings", "body");
  addLine(slide, 1.3, 4.08, 2.22, 0, C.white, 1, "redteam-rule");
  addText(slide, "statistics\ncapacity\nsystems\npublication", { x: 1.3, y: 4.38, w: 2.22, h: 1.18, fontFace: FONT.mono, fontSize: 12.5, color: C.white, align: "center", breakLine: true }, "redteam-domains", "label");
  const rules = [
    ["NO DELAY BENEFIT", "sleep-specific claim 철회"],
    ["NO HYBRID DOMINANCE", "matched frontier만 보고"],
    ["NO KNEE IDENTIFICATION", "KNEE_UNIDENTIFIED"],
    ["NO HELD-OUT PREDICTION", "curve, not law"],
    ["NO COMPLETE DELETE", "stage별 residual 보고"],
  ];
  rules.forEach(([condition, action], index) => {
    const y = 1.79 + index * 0.91;
    addText(slide, condition, { x: 4.48, y, w: 2.5, h: 0.22, fontFace: FONT.mono, fontSize: 10.5, bold: true, color: C.rust }, `redteam-condition-${index}`, "label");
    addLine(slide, 7.16, y + 0.12, 0.62, 0, C.rust, 1.3, `redteam-arrow-${index}`, "triangle");
    addText(slide, action, { x: 8.0, y: y - 0.04, w: 3.87, h: 0.32, fontSize: 17, bold: true }, `redteam-action-${index}`, "body");
  });
  addText(slide, "NULL ≠ FAILURE OF THE PROGRAM", { x: 5.55, y: 6.12, w: 5.82, h: 0.23, fontFace: FONT.mono, fontSize: 11, bold: true, color: C.teal, align: "right" }, "redteam-bottom", "label");
}

// 25 — roadmap
{
  const slide = startSlide(
    "STC-25",
    "research program",
    "다음 단계는 더 큰 모델이 아니라 더 단단한 lifecycle 실험이다",
    "제안",
    "Six-stage research agenda · empirical work remains pending",
    "The roadmap preserves negative outcomes and separates local deterministic reproducibility, confirmatory GPU work, dense scaling, systems exploration, and external validity.",
  );
  const stages = [
    ["01", "EVIDENCE", "chronology · source cards", C.cobalt],
    ["02", "LOCAL", "deterministic stream · fault", C.teal],
    ["03", "CONFIRM", "matched 297 cells", C.rust],
    ["04", "SCALE", "knees · held-outs", C.cobalt],
    ["05", "SYSTEMS", "queue · movement · fleet", C.teal],
    ["06", "VALIDITY", "public benchmarks · hardware", C.rust],
  ];
  stages.forEach(([num, tag, body, color], index) => {
    const x = 0.83 + index * 2.03;
    addText(slide, num, { x, y: 1.87, w: 0.42, h: 0.22, fontFace: FONT.mono, fontSize: 10.5, bold: true, color }, `road-num-${index}`, "label");
    addLine(slide, x, 2.29, 1.54, 0, color, 4, `road-rule-${index}`);
    addText(slide, tag, { x, y: 2.62, w: 1.54, h: 0.23, fontFace: FONT.mono, fontSize: 10.5, bold: true, color }, `road-tag-${index}`, "label");
    addText(slide, body, { x, y: 3.07, w: 1.62, h: 0.88, fontSize: 15.5, bold: true, breakLine: true }, `road-body-${index}`, "body");
    if (index < 5) addLine(slide, x + 1.62, 2.3, 0.28, 0, C.ink, 1, `road-arrow-${index}`, "triangle");
  });
  rect(slide, 1.17, 4.83, 10.65, 0.72, C.ink, null, "road-paper");
  addText(slide, "Minimum credible paper", { x: 1.42, y: 5.03, w: 2.15, h: 0.26, fontFace: FONT.serif, fontSize: 18, bold: true, color: C.white }, "road-paper-label", "label");
  addText(slide, "reproducible stream · matched destinations · predictive crossover or informative null · complete state · negative path", { x: 3.82, y: 5.02, w: 7.65, h: 0.3, fontSize: 15.5, bold: true, color: C.white, align: "center" }, "road-paper-text", "body");
  addText(slide, "hardware run과 human release gate는 아직 수행되지 않았다", { x: 3.24, y: 6.14, w: 6.88, h: 0.24, fontSize: 16, color: C.rust, bold: true, align: "center" }, "road-status", "body");
}

// 26 — conclusion
{
  const slide = startSlide(
    "STC-26",
    "conclusion",
    "Sleep의 단위는 검증 가능한 기억 lifecycle이다",
    "통합 추론",
    "Synthesis of the audited corpus and proposed research program",
    "Closing thesis: capture exact evidence at wake, transform against immutable snapshots, route by future value and risk, publish atomically, and preserve deletion and rollback.",
  );
  addText(slide, "CAPTURE", { x: 0.9, y: 2.0, w: 1.45, h: 0.26, fontFace: FONT.mono, fontSize: 12, bold: true, color: C.cobalt }, "con-capture", "label");
  addText(slide, "TRANSFORM", { x: 3.58, y: 2.0, w: 1.7, h: 0.26, fontFace: FONT.mono, fontSize: 12, bold: true, color: C.rust }, "con-transform", "label");
  addText(slide, "VERIFY", { x: 6.58, y: 2.0, w: 1.3, h: 0.26, fontFace: FONT.mono, fontSize: 12, bold: true, color: C.teal }, "con-verify", "label");
  addText(slide, "REUSE / UNDO", { x: 9.3, y: 2.0, w: 2.15, h: 0.26, fontFace: FONT.mono, fontSize: 12, bold: true, color: C.ink }, "con-reuse", "label");
  addLine(slide, 1.15, 2.72, 10.15, 0, C.ink, 2, "con-line", "triangle");
  [1.25, 4.1, 7.0, 10.0].forEach((x, index) => circle(slide, x, 2.48, 0.48, [C.cobalt, C.rust, C.teal, C.ink][index], C.paper, `con-dot-${index}`));
  addText(
    slide,
    "Wake가 exact event를 남기고\nSleep이 candidate를 만들며\nMemory fabric이 version을 보유하고\nControl plane이 publish와 undo를 결정한다",
    { x: 1.12, y: 3.46, w: 10.75, h: 1.65, fontFace: FONT.serif, fontSize: 25, bold: true, align: "center", breakLine: true, valign: "mid" },
    "con-statement",
    "headline",
  );
  rect(slide, 2.15, 5.66, 9.0, 0.55, C.teal, null, "con-final");
  addText(slide, "external vs parametric은 철학이 아니라 lifetime routing decision이다", { x: 2.4, y: 5.84, w: 8.5, h: 0.22, fontSize: 17, bold: true, color: C.white, align: "center" }, "con-final-text", "body");
}

// 27 — references A
{
  const slide = startSlide(
    "STC-27",
    "references",
    "핵심 1차 문헌 · foundations & direct sleep",
    "근거",
    "Full linked bibliography: PAPER.md and SLEEP-TIME-COMPUTE-MONOGRAPH-KR.md",
    "Reference slide 1. Full URLs and evidence boundaries are in the English paper and Korean monograph.",
    { titleSize: 28 },
  );
  const refs = [
    "Crick & Mitchison · Nature 1983 · dream reverse-learning hypothesis",
    "Hinton et al. · Science 1995 · Wake–Sleep algorithm",
    "McClelland et al. · 1995 · complementary learning systems",
    "Robins · 1995 · pseudorehearsal",
    "Shin et al. · NeurIPS 2017 · deep generative replay",
    "Kamra et al. · 2017 · Deep Generative Dual Memory Network",
    "Kemker & Kanan · ICLR 2018 · FearNet",
    "Schwarz et al. · ICML 2018 · Progress & Compress",
    "Sun et al. · ICML 2020 · test-time training",
    "Deperrois et al. · eLife 2022 · PAD",
    "Golden et al. · PLOS Computational Biology 2022 · SNN sleep",
    "Tadros et al. · Nature Communications 2022 · sleep-like replay",
    "Harun et al. · TMLR 2023 · SIESTA",
    "Packer et al. · 2023 · MemGPT",
  ];
  refs.forEach((ref, index) => {
    const col = index < 7 ? 0 : 1;
    const row = index % 7;
    const x = col === 0 ? 0.92 : 6.75;
    const y = 1.65 + row * 0.7;
    addText(slide, String(index + 1).padStart(2, "0"), { x, y: y + 0.06, w: 0.42, h: 0.19, fontFace: FONT.mono, fontSize: 9.5, bold: true, color: C.cobalt }, `refa-num-${index}`, "citation");
    addText(slide, ref, { x: x + 0.56, y, w: 5.15, h: 0.43, fontSize: 14.5, color: C.ink }, `refa-text-${index}`, "label");
  });
  addText(slide, "Links and exact evidence grades are retained in the companion paper/monograph.", { x: 2.42, y: 6.42, w: 8.5, h: 0.22, fontSize: 13.5, color: C.slate, align: "center" }, "refa-note", "label");
}

// 28 — references B
{
  const slide = startSlide(
    "STC-28",
    "references",
    "핵심 1차 문헌 · agent memory, capacity & systems",
    "근거",
    "83 linked primary sources in the English paper · 2026-07-25 cutoff",
    "Reference slide 2. This bounded bibliography is not evidence of novelty or non-existence of adjacent work.",
    { titleSize: 28 },
  );
  const refs = [
    "Lin et al. · 2025 · Sleep-time Compute",
    "Rasmussen et al. · 2025 · Zep / temporal knowledge graph",
    "Mem0 · ECAI 2025 · memory operators",
    "Berges et al. · ICML 2025 · Memory Layers at Scale",
    "Lin et al. · 2025 · Sparse Memory Finetuning",
    "Spens et al. · NeurIPS 2025 · learned offline processing",
    "Allen-Zhu & Li · 2024 · conditional knowledge capacity",
    "Dohare et al. · Nature 2024 · loss of plasticity",
    "Behrouz et al. · 2026 · Language Models Need Sleep",
    "Kerestecioglu et al. · 2026 · Human-Inspired Memory Architecture",
    "Auto-Dreamer · 2026 · learned external consolidation",
    "RecMem / TiMem / GAM · ACL 2026 · agent-memory lifecycle",
    "Useful Memories Become Faulty · 2026 · repeated rewrite counterevidence",
    "Agent Memory · 2026 · systems characterization",
  ];
  refs.forEach((ref, index) => {
    const col = index < 7 ? 0 : 1;
    const row = index % 7;
    const x = col === 0 ? 0.92 : 6.75;
    const y = 1.65 + row * 0.7;
    addText(slide, String(index + 15).padStart(2, "0"), { x, y: y + 0.06, w: 0.42, h: 0.19, fontFace: FONT.mono, fontSize: 9.5, bold: true, color: C.teal }, `refb-num-${index}`, "citation");
    addText(slide, ref, { x: x + 0.56, y, w: 5.15, h: 0.43, fontSize: 14.5, color: C.ink }, `refb-text-${index}`, "label");
  });
  rect(slide, 1.82, 6.18, 9.7, 0.43, C.ink, null, "refb-boundary");
  addText(slide, "BOUNDED AUDIT  ·  non-discovery is not novelty evidence", { x: 2.08, y: 6.31, w: 9.18, h: 0.18, fontFace: FONT.mono, fontSize: 10.5, bold: true, color: C.white, align: "center" }, "refb-boundary-text", "label");
}

const outOfBounds = [];
const minBodyFontViolations = [];
const missingNotes = [];
for (const meta of slideMeta) {
  if (!meta.notes.trim()) missingNotes.push(meta.id);
  for (const object of meta.objects) {
    const epsilon = 0.015;
    if (object.x < -epsilon || object.y < -epsilon || object.x + object.w > W + epsilon || object.y + object.h > H + epsilon) {
      outOfBounds.push({ slide: meta.id, ...object });
    }
    if (object.kind === "text" && object.role === "body" && object.fontSize < 15) {
      minBodyFontViolations.push({ slide: meta.id, name: object.name, fontSize: object.fontSize });
    }
  }
}

if (slideMeta.length !== 28 || outOfBounds.length || missingNotes.length || minBodyFontViolations.length) {
  const failure = {
    status: "FAIL",
    slideCount: slideMeta.length,
    outOfBounds,
    missingNotes,
    minBodyFontViolations,
  };
  fs.writeFileSync(path.join(reportDir, "build-qa.json"), `${JSON.stringify(failure, null, 2)}\n`);
  throw new Error(`Deck preflight failed: ${JSON.stringify(failure)}`);
}

await pptx.writeFile({ fileName: outputPath, compression: true });
const bytes = fs.readFileSync(outputPath);
const digest = crypto.createHash("sha256").update(bytes).digest("hex");
const report = {
  status: "PASS",
  artifact: path.relative(projectDir, outputPath),
  sha256: digest,
  slideCount: slideMeta.length,
  objectCount: slideMeta.reduce((sum, slide) => sum + slide.objects.length, 0),
  notesCount: slideMeta.filter((slide) => slide.notes.trim()).length,
  outOfBounds: 0,
  minBodyFontViolations: 0,
  background: C.paper,
  fontFaces: [FONT.sans, FONT.serif, FONT.mono],
  factualBoundary: "Position/review/research agenda; PAPER-C experiments unexecuted",
  generatedAt: new Date().toISOString(),
  slides: slideMeta.map(({ id, section, title, status, source, objects }) => ({
    id,
    section,
    title,
    status,
    source,
    objectCount: objects.length,
  })),
};
fs.writeFileSync(path.join(reportDir, "build-qa.json"), `${JSON.stringify(report, null, 2)}\n`);
process.stdout.write(`PASS slides=${report.slideCount} objects=${report.objectCount} sha256=${digest}\n`);
