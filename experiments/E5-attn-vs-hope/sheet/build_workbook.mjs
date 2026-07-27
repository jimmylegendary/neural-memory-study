#!/usr/bin/env node
/** Reproducible, formula-driven workbook companion for the verified E5 engine. */

import fs from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";

let artifactTool;
try {
  artifactTool = await import("@oai/artifact-tool");
} catch (error) {
  const here = path.dirname(fileURLToPath(import.meta.url));
  throw new Error(
    "Unable to import @oai/artifact-tool. Create a node_modules symlink at " +
      `${path.join(here, "node_modules")} pointing to the bundled runtime node_modules, ` +
      "then rerun this builder. Original error: " +
      (error instanceof Error ? error.message : String(error)),
  );
}

const { FileBlob, SpreadsheetFile, Workbook } = artifactTool;
const SCRIPT_DIR = path.dirname(fileURLToPath(import.meta.url));
const RESULTS_URL = new URL("../results.json", import.meta.url);
const OUTPUT_PATH = path.join(SCRIPT_DIR, "Attn-vs-HOPE.xlsx");
const PREVIEW_DIR = path.join(SCRIPT_DIR, "previews");

const SHEET_NAMES = [
  "00_Guide",
  "01_Inputs",
  "02_HW",
  "10_AttnMoE_Prefill",
  "11_AttnMoE_Decode",
  "20_HOPE_Prefill",
  "21_HOPE_Decode",
  "30_Compare",
  "40_Sweeps",
  "90_QA",
  "99_Sources",
];

const STAGE_HEADERS = [
  "Order",
  "Phase",
  "Component",
  "Stage",
  "Symbolic equation",
  "Executions/cadence",
  "FLOPs",
  "Required read B",
  "Mandatory write B",
  "Temporary/fusible write B",
  "Effective read B",
  "Effective write B",
  "Persistent state B",
  "AI",
  "Compute ms",
  "HBM ms",
  "Roofline ms",
  "Bound",
  "Evidence/assumption",
];

const COLORS = {
  navy: "#17324D",
  blue: "#2F6B9A",
  paleBlue: "#EAF2F8",
  ink: "#263746",
  gray: "#667788",
  line: "#CBD5DF",
  light: "#F4F6F8",
  white: "#FFFFFF",
  green: "#DDEFE3",
  amber: "#F5E8C8",
  red: "#F4DADA",
};

const results = JSON.parse(await fs.readFile(RESULTS_URL, "utf8"));

function colName(index) {
  let n = index;
  let name = "";
  while (n > 0) {
    n -= 1;
    name = String.fromCharCode(65 + (n % 26)) + name;
    n = Math.floor(n / 26);
  }
  return name;
}

function absCell(column, row) {
  return `$${column}$${row}`;
}

function qref(sheet, cell) {
  return `'${sheet}'!${cell}`;
}

function matrix(rows, columns, value = null) {
  return Array.from({ length: rows }, () => Array(columns).fill(value));
}

function writeTitle(sheet, endColumn, title, subtitle) {
  sheet.mergeCells(`A1:${endColumn}1`);
  sheet.getRange("A1").values = [[title]];
  sheet.getRange(`A1:${endColumn}1`).format = {
    fill: COLORS.navy,
    font: { bold: true, color: COLORS.white, size: 15 },
    verticalAlignment: "center",
  };
  sheet.getRange("A1").format.rowHeight = 28;
  if (subtitle) {
    sheet.mergeCells(`A2:${endColumn}2`);
    sheet.getRange("A2").values = [[subtitle]];
    sheet.getRange(`A2:${endColumn}2`).format = {
      fill: COLORS.paleBlue,
      font: { color: COLORS.ink, italic: true, size: 9 },
      wrapText: true,
      verticalAlignment: "center",
    };
    sheet.getRange("A2").format.rowHeight = 28;
  }
  sheet.showGridLines = false;
}

function styleHeader(range, fill = COLORS.blue) {
  range.format = {
    fill,
    font: { bold: true, color: COLORS.white, size: 9 },
    wrapText: true,
    verticalAlignment: "center",
    borders: { preset: "outside", style: "thin", color: COLORS.line },
  };
}

function styleBody(range) {
  range.format = {
    font: { color: COLORS.ink, size: 9 },
    verticalAlignment: "top",
    borders: {
      insideHorizontal: { style: "thin", color: "#E4E9EE" },
      bottom: { style: "thin", color: COLORS.line },
    },
  };
}

function writeKeyValueInputs(sheet) {
  writeTitle(
    sheet,
    "I",
    "01 · Model inputs",
    "Editable/raw model assumptions imported from results.json. Derived values live only in formula sheets.",
  );
  const attn = results.attention_moe.inputs.decode;
  const hopeInputs = results.hope_scenarios.shipped_credible_momentum.inputs;
  const hope = hopeInputs.decode;
  const hopePrefill = hopeInputs.prefill;
  const prefill = results.attention_moe.inputs.prefill;
  const rows = [
    ["batch", "Batch", attn.batch, "requests", "attention_moe.inputs.decode.batch", "Baseline request batch"],
    ["prefill_query_tokens", "Attention prefill Q", prefill.query_tokens, "tokens/request", "attention_moe.inputs.prefill.query_tokens", "Full ISL in one causal invocation"],
    ["prefill_context_tokens", "Attention prefill K", prefill.context_tokens, "tokens/request", "attention_moe.inputs.prefill.context_tokens", "Full ISL visible to FlashAttention"],
    ["hope_prefill_chunk_tokens", "HOPE prefill chunk", hopePrefill.query_tokens, "tokens/request", "hope_scenarios.shipped_credible_momentum.inputs.prefill.query_tokens", "One parallelizable operational chunk"],
    ["prefill_chunks", "HOPE prefill chunks", results.model_inputs.prefill_chunks, "chunks", "model_inputs.prefill_chunks", "Applied only when forming TTFT"],
    ["decode_query_tokens", "Decode Q", attn.query_tokens, "tokens/request", "attention_moe.inputs.decode.query_tokens", "Exactly one token"],
    ["decode_context_tokens", "Decode K", attn.context_tokens, "tokens/request", "attention_moe.inputs.decode.context_tokens", "Current context"],
    ["model_dim", "Model width D", attn.model_dim, "elements", "model_inputs.model_dim", "Shared model width"],
    ["head_dim", "Head width d_h", attn.head_dim, "elements", "model_inputs.head_dim", "Per-head width"],
    ["query_heads", "Query heads H_q", attn.query_heads, "heads", "attention_moe.inputs.decode.query_heads", "D / d_h"],
    ["kv_heads", "KV heads H_kv", attn.kv_heads, "heads", "attention_moe.inputs.decode.kv_heads", "No GQA in baseline"],
    ["activation_bytes", "Activation bytes", attn.activation_bytes, "B/element", "attention_moe.inputs.decode.activation_bytes", "HBM element width"],
    ["weight_bytes", "Static-weight bytes", attn.weight_bytes, "B/element", "attention_moe.inputs.decode.weight_bytes", "HBM element width"],
    ["kv_bytes", "KV bytes", attn.kv_bytes, "B/element", "attention_moe.inputs.decode.kv_bytes", "Cache element width"],
    ["partial_bytes", "Partial-output bytes", attn.partial_bytes, "B/element", "attention_moe.inputs.decode.partial_bytes", "Flash-Decode partial"],
    ["lse_bytes", "LSE bytes", attn.lse_bytes, "B/scalar", "attention_moe.inputs.decode.lse_bytes", "Flash-Decode LSE"],
    ["decode_splits", "Decode KV splits G", attn.decode_splits, "splits", "attention_moe.inputs.decode.decode_splits", "ceil(K / split tokens)"],
    ["kv_split_tokens", "Tokens per KV split", results.model_inputs.kv_split_tokens, "tokens", "model_inputs.kv_split_tokens", "Sweep split rule"],
    ["kv_reload_multiplier", "KV reload multiplier", attn.kv_reload_multiplier, "x", "attention_moe.inputs.decode.kv_reload_multiplier", "Ideal lower bound = 1"],
    ["softmax_flops_per_pair", "Softmax FLOPs/pair", attn.softmax_flops_per_pair, "FLOP/pair", "attention_moe.inputs.decode.softmax_flops_per_pair", "Explicit input"],
    ["experts", "Experts E", attn.experts, "experts", "model_inputs.experts", "MoE pool"],
    ["top_k", "Top-k r", attn.top_k, "experts/token", "model_inputs.top_k", "Routing multiplicity"],
    ["expert_hidden_dim", "Expert hidden F", attn.expert_hidden_dim, "elements", "model_inputs.expert_hidden_dim", "SwiGLU width"],
    ["expert_matrices", "Expert matrices", attn.expert_matrices, "matrices", "attention_moe.inputs.decode.expert_matrices", "3 = SwiGLU"],
    ["router_flops_per_token_expert", "Router selection cost", attn.router_flops_per_token_expert, "FLOP/token/expert", "attention_moe.inputs.decode.router_flops_per_token_expert", "Beyond router GEMM"],
    ["routing_metadata_bytes", "Routing metadata", 8, "B/route", "model.py: routing_metadata_bytes", "Visible engine convention"],
    ["active_unique_experts", "Active-expert override", attn.active_unique_experts ?? 0, "experts", "attention_moe.inputs.decode.active_unique_experts", "0 = uniform expectation"],
    ["expert_hbm_fraction", "Expert HBM fraction", attn.expert_hbm_fraction, "fraction", "attention_moe.inputs.decode.expert_hbm_fraction", "Residency sensitivity"],
    ["compute_efficiency", "Compute efficiency", attn.compute_efficiency, "fraction", "attention_moe.inputs.decode.compute_efficiency", "Stage default"],
    ["bandwidth_efficiency", "Bandwidth efficiency", attn.bandwidth_efficiency, "fraction", "attention_moe.inputs.decode.bandwidth_efficiency", "Stage default"],
    ["expert_activation_flops", "Expert activation FLOPs", attn.expert_activation_flops_per_element, "FLOP/element", "model_inputs.expert_activation_flops_per_element", "SwiGLU elementwise"],
    ["fuse_expert_intermediates", "Fuse expert intermediates", attn.fuse_expert_intermediates ? 1 : 0, "0/1", "model_inputs.fuse_expert_intermediates", "1 removes internal HBM traffic"],
    ["state_bytes", "Mutable-state bytes", hope.state_bytes, "B/element", "hope_scenarios.shipped_credible_momentum.inputs.decode.state_bytes", "Fast weights"],
    ["optimizer_bytes", "Optimizer bytes", hope.optimizer_bytes, "B/element", "hope_scenarios.shipped_credible_momentum.inputs.decode.optimizer_bytes", "Gradient/slot width"],
    ["sigma_flops", "Sigma FLOPs/hidden", hope.sigma_flops_per_hidden, "FLOP/hidden", "hope_scenarios.shipped_credible_momentum.inputs.decode.sigma_flops_per_hidden", "Shipped-credible scenario"],
    ["residual_flops", "Residual FLOPs/output", hope.residual_flops_per_output, "FLOP/output", "hope_scenarios.shipped_credible_momentum.inputs.decode.residual_flops_per_output", "Shipped-credible scenario"],
    ["chunk_cache_bytes", "Chunk cache/request", hope.chunk_cache_bytes_per_request, "B/request", "hope_scenarios.shipped_credible_momentum.inputs.decode.chunk_cache_bytes_per_request", "Persistent request cache"],
    ["prefill_position", "Prefill position p", hopePrefill.position, "tokens", "hope_scenarios.shipped_credible_momentum.inputs.prefill.position", "Imported boundary-count origin"],
    ["decode_position", "Decode position p", hope.position, "tokens", "hope_scenarios.shipped_credible_momentum.inputs.decode.position", "Imported timing origin; timing modes remain explicit"],
    ["main_memory_chunk", "Main-memory chunk", results.model_inputs.main_memory_chunk, "tokens", "model_inputs.main_memory_chunk", "M_mem period"],
    ["aux_memory_chunk", "Aux-memory chunk", results.model_inputs.aux_memory_chunk, "tokens", "model_inputs.aux_memory_chunk", "M_k/v/eta/alpha period"],
  ];
  sheet.getRange("A4:F4").values = [["Key", "Input", "Value", "Unit", "Engine path", "Notes"]];
  styleHeader(sheet.getRange("A4:F4"));
  sheet.getRange(`A5:F${4 + rows.length}`).values = rows.map((row) => row.slice(0, 2).concat(row.slice(2)));
  styleBody(sheet.getRange(`A5:F${4 + rows.length}`));
  sheet.getRange(`C5:C${4 + rows.length}`).format.numberFormat = "#,##0.############";
  const refs = {};
  rows.forEach((row, index) => {
    refs[row[0]] = qref("01_Inputs", absCell("C", 5 + index));
  });

  const memories = hope.memories;
  const memoryStart = 47;
  sheet.getRange(`A${memoryStart}:H${memoryStart}`).values = [[
    "Memory",
    "Input dim I",
    "Hidden H",
    "Output O",
    "Update chunk",
    "Momentum slots",
    "Update multiplier μ",
    "State read count",
  ]];
  styleHeader(sheet.getRange(`A${memoryStart}:H${memoryStart}`), COLORS.navy);
  sheet.getRange(`A${memoryStart + 1}:H${memoryStart + memories.length}`).values = memories.map((memory) => [
    memory.name,
    memory.input_dim,
    memory.hidden_dim,
    memory.output_dim,
    memory.update_chunk,
    memory.momentum_slots,
    memory.update_multiplier,
    memory.state_weight_read_count,
  ]);
  styleBody(sheet.getRange(`A${memoryStart + 1}:H${memoryStart + memories.length}`));
  const memoryRefs = {};
  memories.forEach((memory, index) => {
    const row = memoryStart + 1 + index;
    memoryRefs[memory.name] = {
      input: qref("01_Inputs", absCell("B", row)),
      hidden: qref("01_Inputs", absCell("C", row)),
      output: qref("01_Inputs", absCell("D", row)),
      chunk: qref("01_Inputs", absCell("E", row)),
      slots: qref("01_Inputs", absCell("F", row)),
      multiplier: qref("01_Inputs", absCell("G", row)),
      readCount: qref("01_Inputs", absCell("H", row)),
    };
  });

  const cms = hope.cms_levels;
  const cmsStart = 56;
  sheet.getRange(`A${cmsStart}:M${cmsStart}`).values = [[
    "CMS level",
    "Input dim I",
    "Output O",
    "Capacity C",
    "Low-rank R",
    "Top-k",
    "Router FLOPs/token/expert",
    "Update period",
    "BPTT span",
    "Optimizer slots",
    "Personalized (0/1)",
    "Gradient multiplier",
    "Routing metadata B",
  ]];
  styleHeader(sheet.getRange(`A${cmsStart}:M${cmsStart}`), COLORS.navy);
  sheet.getRange(`A${cmsStart + 1}:M${cmsStart + cms.length}`).values = cms.map((level) => [
    level.name,
    level.input_dim,
    level.output_dim,
    level.capacity,
    level.low_rank_dim,
    level.top_k,
    level.router_flops_per_token_expert,
    level.update_period,
    level.bptt_span,
    level.optimizer_slots,
    level.personalized ? 1 : 0,
    level.gradient_multiplier,
    level.routing_metadata_bytes,
  ]);
  styleBody(sheet.getRange(`A${cmsStart + 1}:M${cmsStart + cms.length}`));
  const cmsRefs = {};
  cms.forEach((level, index) => {
    const row = cmsStart + 1 + index;
    cmsRefs[level.name] = {
      input: qref("01_Inputs", absCell("B", row)),
      output: qref("01_Inputs", absCell("C", row)),
      capacity: qref("01_Inputs", absCell("D", row)),
      lowRank: qref("01_Inputs", absCell("E", row)),
      topK: qref("01_Inputs", absCell("F", row)),
      routerCost: qref("01_Inputs", absCell("G", row)),
      period: qref("01_Inputs", absCell("H", row)),
      bptt: qref("01_Inputs", absCell("I", row)),
      slots: qref("01_Inputs", absCell("J", row)),
      personalized: qref("01_Inputs", absCell("K", row)),
      multiplier: qref("01_Inputs", absCell("L", row)),
      metadata: qref("01_Inputs", absCell("M", row)),
    };
  });

  const contextValues = results.crossovers.context_sweep.map((item) => item.context_tokens);
  const batchValues = results.crossovers.batch_sweep.map((item) => item.batch);
  const sweepStart = 63;
  sheet.getRange(`A${sweepStart}:G${sweepStart}`).values = [["Sweep", ...contextValues]];
  styleHeader(sheet.getRange(`A${sweepStart}:G${sweepStart}`), COLORS.gray);
  sheet.getRange(`A${sweepStart + 1}:G${sweepStart + 1}`).values = [["Context tokens", ...contextValues]];
  sheet.getRange(`A${sweepStart + 2}:G${sweepStart + 2}`).values = [["Batch", ...batchValues]];
  styleBody(sheet.getRange(`A${sweepStart + 1}:G${sweepStart + 2}`));
  const sweepRefs = {
    contexts: contextValues.map((_, index) => qref("01_Inputs", absCell(colName(2 + index), sweepStart + 1))),
    batches: batchValues.map((_, index) => qref("01_Inputs", absCell(colName(2 + index), sweepStart + 2))),
  };

  sheet.freezePanes.freezeRows(4);
  sheet.getRange("A:A").format.columnWidth = 22;
  sheet.getRange("B:B").format.columnWidth = 24;
  sheet.getRange("C:C").format.columnWidth = 16;
  sheet.getRange("D:D").format.columnWidth = 18;
  sheet.getRange("E:E").format.columnWidth = 48;
  sheet.getRange("F:F").format.columnWidth = 38;
  sheet.getRange("G:M").format.columnWidth = 17;
  return { refs, memoryRefs, cmsRefs, sweepRefs };
}

function writeHardware(sheet) {
  writeTitle(
    sheet,
    "F",
    "02 · Hardware anchor",
    "User-reviewed live-Sheet peak/BW authority plus repository-twin HBM capacity. Calculation sheets reference these cells; no hardware constants are embedded in formulas.",
  );
  const h = results.hardware;
  const rows = [
    ["name", "Hardware", h.name, "text", "hardware.name", "Mixed authority is explicit"],
    ["peak_flops", "Peak compute", h.peak_flops_per_second, "FLOP/s", "hardware.peak_flops_per_second", "User-reviewed live-Sheet analytical input"],
    ["hbm_bandwidth", "HBM bandwidth", h.hbm_bandwidth_bytes_per_second, "B/s", "hardware.hbm_bandwidth_bytes_per_second", "User-reviewed live-Sheet analytical input"],
    ["hbm_capacity", "HBM capacity", h.hbm_capacity_bytes, "B", "hardware.hbm_capacity_bytes", "Repository H100 twin capacity only"],
    ["launch_seconds", "Kernel launch overhead", h.kernel_launch_overhead_seconds, "s/execution", "hardware.kernel_launch_overhead_seconds", "Zero: twin publishes no constant"],
    ["ms_per_second", "Milliseconds per second", 1000, "ms/s", "unit conversion", "Visible unit conversion"],
    ["bytes_per_gib", "Bytes per GiB", 1073741824, "B/GiB", "unit conversion", "2^30"],
    ["flops_per_tflop", "FLOPs per TFLOP", 1000000000000, "FLOP/TFLOP", "unit conversion", "10^12"],
  ];
  sheet.getRange("A4:F4").values = [["Key", "Hardware/input", "Value", "Unit", "Engine path", "Evidence"]];
  styleHeader(sheet.getRange("A4:F4"));
  sheet.getRange(`A5:F${4 + rows.length}`).values = rows;
  styleBody(sheet.getRange(`A5:F${4 + rows.length}`));
  sheet.getRange(`C5:C${4 + rows.length}`).format.numberFormat = "0.############E+00";
  const refs = {};
  rows.forEach((row, index) => {
    refs[row[0]] = qref("02_HW", absCell("C", 5 + index));
  });
  sheet.freezePanes.freezeRows(4);
  sheet.getRange("A:A").format.columnWidth = 22;
  sheet.getRange("B:B").format.columnWidth = 28;
  sheet.getRange("C:C").format.columnWidth = 42;
  sheet.getRange("D:D").format.columnWidth = 16;
  sheet.getRange("E:E").format.columnWidth = 44;
  sheet.getRange("F:F").format.columnWidth = 44;
  return refs;
}

function stage({ phase, component, name, equation, cadence, flops, read, mandatory, temporary, persistent, note }) {
  return { phase, component, name, equation, cadence, flops, read, mandatory, temporary, persistent, note };
}

function attentionStages(I, mode, overrides = {}) {
  const B = overrides.batch ?? I.batch;
  const Q = overrides.query ?? (mode === "prefill" ? I.prefill_query_tokens : I.decode_query_tokens);
  const K = overrides.context ?? (mode === "prefill" ? I.prefill_context_tokens : I.decode_context_tokens);
  const G = overrides.splits ?? (mode === "prefill" ? "1" : I.decode_splits);
  const N = `((${B})*(${Q}))`;
  const Dq = `((${I.query_heads})*(${I.head_dim}))`;
  const Dkv = `((${I.kv_heads})*(${I.head_dim}))`;
  const zero = "0";
  const one = "1";
  const modelAct = `((${N})*(${I.model_dim})*(${I.activation_bytes}))`;
  const qBytes = `((${N})*(${Dq})*(${I.activation_bytes}))`;
  const routed = `((${N})*(${I.top_k})*(${I.model_dim})*(${I.activation_bytes}))`;
  const metadata = `((${N})*(${I.top_k})*(${I.routing_metadata_bytes}))`;
  const hidden = `((${N})*(${I.top_k})*(${I.expert_hidden_dim})*(${I.activation_bytes}))`;
  const active = `(IF((${I.active_unique_experts})>0,(${I.active_unique_experts}),(${I.experts})*(1-POWER(1-1/(${I.experts}),(${N})*(${I.top_k})))))`;
  const expertWeight = `((${active})*(${I.model_dim})*(${I.expert_hidden_dim})*(${I.weight_bytes})*(${I.expert_hbm_fraction}))`;
  const expertFlops = `(2*(${N})*(${I.top_k})*(${I.model_dim})*(${I.expert_hidden_dim}))`;
  const unfused = (expr) => `(IF((${I.fuse_expert_intermediates})=1,0,(${expr})))`;
  const specs = [
    stage({ phase: mode, component: "attention", name: "qkv_projection", equation: "2ND(Dq+2Dkv)", cadence: one, flops: `(2*(${N})*(${I.model_dim})*((${Dq})+2*(${Dkv})))`, read: `((${N})*(${I.model_dim})*(${I.activation_bytes})+(${I.model_dim})*((${Dq})+2*(${Dkv}))*(${I.weight_bytes}))`, mandatory: zero, temporary: qBytes, persistent: zero, note: "Explicit Q materialization; K/V append follows." }),
    stage({ phase: mode, component: "attention", name: "kv_cache_append", equation: "2BQDkv bkv", cadence: one, flops: zero, read: zero, mandatory: `(2*(${N})*(${Dkv})*(${I.kv_bytes}))`, temporary: zero, persistent: `(2*(${B})*(${K})*(${Dkv})*(${I.kv_bytes}))`, note: "Append is invocation-local; state is full KV residency." }),
    stage({ phase: mode, component: "attention", name: "q_materialization_read", equation: "NDq ba", cadence: one, flops: zero, read: qBytes, mandatory: zero, temporary: zero, persistent: zero, note: "Fusible Q read remains explicit." }),
  ];
  if (mode === "prefill") {
    const pairs = `((${B})*(${I.query_heads})*(${Q})*((${Q})+1)/2)`;
    specs.push(stage({ phase: mode, component: "attention", name: "flash_attention", equation: "4Πdh + csoftmaxΠ", cadence: one, flops: `(4*(${pairs})*(${I.head_dim})+(${I.softmax_flops_per_pair})*(${pairs}))`, read: `(2*(${B})*(${K})*(${Dkv})*(${I.kv_bytes})*(${I.kv_reload_multiplier}))`, mandatory: qBytes, temporary: zero, persistent: zero, note: "Causal pairs; no quadratic score/probability HBM write." }));
  } else {
    const pairs = `((${B})*(${I.query_heads})*(${Q})*(${K}))`;
    const partials = `(IF((${G})=1,0,(${B})*(${Q})*(${I.query_heads})*(${G})*((${I.head_dim})*(${I.partial_bytes})+(${I.lse_bytes}))))`;
    specs.push(stage({ phase: mode, component: "attention", name: "flash_decode_partials", equation: "4BHqQKdh + csoftmaxBHqQK", cadence: one, flops: `(4*(${pairs})*(${I.head_dim})+(${I.softmax_flops_per_pair})*(${pairs}))`, read: `(2*(${B})*(${K})*(${Dkv})*(${I.kv_bytes})*(${I.kv_reload_multiplier}))`, mandatory: `(IF((${G})=1,${qBytes},0))`, temporary: partials, persistent: zero, note: "KV traffic is linear in K; split partials only for G>1." }));
    specs.push(stage({ phase: mode, component: "attention", name: "flash_decode_reduce", equation: "read partials; write O", cadence: `(IF((${G})>1,1,0))`, flops: zero, read: partials, mandatory: qBytes, temporary: zero, persistent: zero, note: "Optional split-local reduction." }));
  }
  specs.push(
    stage({ phase: mode, component: "attention", name: "output_projection", equation: "2NDqD", cadence: one, flops: `(2*(${N})*(${Dq})*(${I.model_dim}))`, read: `((${N})*(${Dq})*(${I.activation_bytes})+(${Dq})*(${I.model_dim})*(${I.weight_bytes}))`, mandatory: zero, temporary: modelAct, persistent: zero, note: "Projects attention output to model width." }),
    stage({ phase: mode, component: "moe", name: "moe_router", equation: "2NDE + crouterNE", cadence: one, flops: `(2*(${N})*(${I.model_dim})*(${I.experts})+(${I.router_flops_per_token_expert})*(${N})*(${I.experts}))`, read: `((${modelAct})+(${I.model_dim})*(${I.experts})*(${I.weight_bytes}))`, mandatory: zero, temporary: metadata, persistent: zero, note: "Router GEMM plus selection cost." }),
    stage({ phase: mode, component: "moe", name: "moe_dispatch", equation: "materialize top-k routes", cadence: one, flops: zero, read: `((${modelAct})+(${metadata}))`, mandatory: zero, temporary: routed, persistent: zero, note: "Dispatch traffic kept visible." }),
    stage({ phase: mode, component: "moe", name: "expert_up_projection", equation: "2NrDF", cadence: one, flops: expertFlops, read: `((${routed})+(${expertWeight}))`, mandatory: zero, temporary: unfused(hidden), persistent: zero, note: "Expected active-expert weight read." }),
    stage({ phase: mode, component: "moe", name: "expert_gate_projection", equation: "2NrDF", cadence: one, flops: expertFlops, read: `((${expertWeight})+${unfused(routed)})`, mandatory: zero, temporary: unfused(hidden), persistent: zero, note: "SwiGLU gate matrix." }),
    stage({ phase: mode, component: "moe", name: "expert_swiglu_activation", equation: "NrF cactivation", cadence: one, flops: `((${N})*(${I.top_k})*(${I.expert_hidden_dim})*(${I.expert_activation_flops}))`, read: unfused(`2*(${hidden})`), mandatory: zero, temporary: unfused(hidden), persistent: zero, note: "Elementwise traffic vanishes only under fusion." }),
    stage({ phase: mode, component: "moe", name: "expert_down_projection", equation: "2NrDF", cadence: one, flops: expertFlops, read: `((${expertWeight})+${unfused(hidden)})`, mandatory: zero, temporary: routed, persistent: zero, note: "Down matrix plus routed output." }),
    stage({ phase: mode, component: "moe", name: "moe_combine", equation: "combine top-k outputs", cadence: one, flops: zero, read: routed, mandatory: modelAct, temporary: zero, persistent: zero, note: "Mandatory block-output write." }),
  );
  return specs;
}

function hopeStages(I, memoryRefs, cmsRefs, phase, timing = "amortized") {
  const B = I.batch;
  const Q = phase === "prefill" ? I.hope_prefill_chunk_tokens : I.decode_query_tokens;
  const position = phase === "prefill" ? I.prefill_position : I.decode_position;
  const N = `((${B})*(${Q}))`;
  const zero = "0";
  const one = "1";
  const modelAct = `((${N})*(${I.model_dim})*(${I.activation_bytes}))`;
  const specs = [stage({ phase, component: "titans_forward", name: "static_q_projection", equation: "2ND²", cadence: one, flops: `(2*(${N})*(${I.model_dim})*(${I.model_dim}))`, read: `((${modelAct})+(${I.model_dim})*(${I.model_dim})*(${I.weight_bytes}))`, mandatory: zero, temporary: modelAct, persistent: `((${B})*(${I.chunk_cache_bytes}))`, note: "Shipped path: static q=xWq; Wq is not mutable state." })];

  const memoryDerived = [];
  for (const [name, memory] of Object.entries(memoryRefs)) {
    const params = `((${memory.input})*(${memory.hidden})+(${memory.hidden})*(${memory.output}))`;
    const fwd = `(2*(${N})*(${params})+(${N})*((${I.sigma_flops})*(${memory.hidden})+(${I.residual_flops})*(${memory.output})))`;
    const inputBytes = `((${N})*(${memory.input})*(${I.activation_bytes}))`;
    const outputBytes = `((${N})*(${memory.output})*(${I.activation_bytes}))`;
    const gradient = `((${B})*(${params})*(${I.optimizer_bytes}))`;
    const backward = `(2*(${N})*(${memory.input})*(${memory.hidden})+4*(${N})*(${memory.hidden})*(${memory.output}))`;
    memoryDerived.push({ name, memory, params, fwd, inputBytes, outputBytes, gradient, backward });
    if (name === "M_mem") {
      specs.push(stage({ phase, component: "titans_forward", name: `${name}_forward`, equation: "2NP + nonlinear terms", cadence: one, flops: fwd, read: `((${inputBytes})+(${B})*(${params})*(${I.state_bytes})*(${memory.readCount}))`, mandatory: zero, temporary: outputBytes, persistent: `((${B})*(${params})*((${I.state_bytes})+(${memory.slots})*(${I.optimizer_bytes})))`, note: "Actual shipped main-memory forward after static Wq." }));
    }
  }

  for (const [name, level] of Object.entries(cmsRefs)) {
    const sharing = `(IF((${level.personalized})=1,(${B}),1))`;
    const activeUnique = `(MIN((${level.capacity}),IF((${level.personalized})=1,(${Q})*(${level.topK}),(${N})*(${level.topK}))))`;
    const routerParams = `((${level.input})*(${level.capacity}))`;
    const expertAParams = `((${level.input})*(${level.lowRank}))`;
    const expertBParams = `((${level.lowRank})*(${level.output}))`;
    const poolParams = `((${routerParams})+(${level.capacity})*((${expertAParams})+(${expertBParams})))`;
    const routing = `((${N})*(${level.topK})*(${level.metadata}))`;
    const lowRankAct = `((${N})*(${level.topK})*(${level.lowRank})*(${I.activation_bytes}))`;
    specs.push(
      stage({ phase, component: "cms_forward", name: `${name}_router`, equation: "2ND·C + crouterN·C", cadence: one, flops: `(2*(${N})*(${level.input})*(${level.capacity})+(${level.routerCost})*(${N})*(${level.capacity}))`, read: `((${N})*(${level.input})*(${I.activation_bytes})+(${sharing})*(${routerParams})*(${I.state_bytes}))`, mandatory: zero, temporary: routing, persistent: `((${sharing})*(${poolParams})*((${I.state_bytes})+(${level.slots})*(${I.optimizer_bytes})))`, note: "Router compute is dense in capacity; capacity remains a residency dimension." }),
      stage({ phase, component: "cms_forward", name: `${name}_expert_a`, equation: "2N·topk·D·R", cadence: one, flops: `(2*(${N})*(${level.topK})*(${level.input})*(${level.lowRank}))`, read: `((${N})*(${level.input})*(${I.activation_bytes})+(${routing})+(${sharing})*(${activeUnique})*(${expertAParams})*(${I.state_bytes}))`, mandatory: zero, temporary: lowRankAct, persistent: zero, note: "HBM reads use the per-request unique-active upper bound min(C,Q·top-k)." }),
      stage({ phase, component: "cms_forward", name: `${name}_expert_b`, equation: "2N·topk·R·D", cadence: one, flops: `(2*(${N})*(${level.topK})*(${level.lowRank})*(${level.output}))`, read: `((${lowRankAct})+(${sharing})*(${activeUnique})*(${expertBParams})*(${I.state_bytes}))`, mandatory: zero, temporary: `((${N})*(${level.output})*(${I.activation_bytes}))`, persistent: zero, note: "Only top-k active B matrices are read; no capacity×LRD dense hidden flatten." }),
    );
  }

  // Stable partition: every target/prediction loss-preparation stage precedes
  // every gradient, DGD, and state-apply stage.
  for (const item of memoryDerived) {
    const { name, memory, params, fwd, inputBytes, outputBytes } = item;
    specs.push(
      stage({ phase, component: "loss", name: `${name}_target_forward`, equation: "Ffwd(N)", cadence: one, flops: fwd, read: `((${inputBytes})+(${B})*(${params})*(${I.state_bytes})*(${memory.readCount}))`, mandatory: zero, temporary: outputBytes, persistent: name === "M_mem" ? zero : `((${B})*(${params})*((${I.state_bytes})+(${memory.slots})*(${I.optimizer_bytes})))`, note: "Auxiliary self-target/loss preparation; not a direct block forward output." }),
      stage({ phase, component: "loss", name: `${name}_prediction_forward`, equation: "Ffwd(N)", cadence: one, flops: fwd, read: inputBytes, mandatory: zero, temporary: outputBytes, persistent: zero, note: "Prediction term for the inner loss." }),
    );
  }
  for (const item of memoryDerived) {
    const { name, memory, params, fwd, gradient, backward } = item;
    specs.push(
      stage({ phase, component: "backward_update", name: `${name}_weight_backward`, equation: "2NIH + 4NHO", cadence: one, flops: backward, read: `((${N})*((${memory.input})+(${memory.hidden})+(${memory.output}))*(${I.activation_bytes}))`, mandatory: zero, temporary: gradient, persistent: zero, note: "Weight-gradient backward; no input gradient." }),
      stage({ phase, component: "backward_update", name: `${name}_dgd_extra`, equation: "μFfwd − (2Ffwd+Fbwd)", cadence: one, flops: `((${memory.multiplier})*(${fwd})-(2*(${fwd})+(${backward})))`, read: gradient, mandatory: zero, temporary: gradient, persistent: zero, note: "Residual DGD calibration after all loss terms." }),
    );
    if (phase === "prefill" || timing !== "normal") {
      const cadence = phase === "prefill"
        ? `(INT(((${position})+(${Q}))/(${memory.chunk}))-INT((${position})/(${memory.chunk})))`
        : timing === "boundary"
          ? one
          : `(1/(${memory.chunk}))`;
      const apply = `((${B})*(1+(${memory.slots}))*(${params})*(${I.state_bytes}))`;
      specs.push(stage({ phase, component: "backward_update", name: `${name}_state_apply`, equation: "B(1+m)Pbs read + write", cadence, flops: zero, read: apply, mandatory: apply, temporary: zero, persistent: zero, note: phase === "prefill" ? "Boundary-count state RMW." : `${timing} decode state RMW.` }));
    }
  }
  return specs;
}

function assertStageNames(label, specs, engineStages) {
  const effectiveSpecs = specs.filter((item) => !item.name.endsWith("flash_decode_reduce") || engineStages.some((stageItem) => stageItem.stage === item.name));
  const workbookNames = effectiveSpecs.map((item) => item.name);
  const engineNames = engineStages.map((item) => item.stage);
  if (JSON.stringify(workbookNames) !== JSON.stringify(engineNames)) {
    throw new Error(`${label} stage names/order do not match results.json`);
  }
  const workbookComponents = effectiveSpecs.map((item) => item.component);
  const engineComponents = engineStages.map((item) => item.component);
  if (JSON.stringify(workbookComponents) !== JSON.stringify(engineComponents)) {
    throw new Error(`${label} stage groups/order do not match results.json`);
  }
}

function stageTimeFormula(spec, I, H) {
  const compute = `((${spec.cadence})*(${spec.flops})/((${I.compute_efficiency})*(${H.peak_flops}))*(${H.ms_per_second}))`;
  const memory = `((${spec.cadence})*((${spec.read})+(${spec.mandatory})+(${spec.temporary}))/((${I.bandwidth_efficiency})*(${H.hbm_bandwidth}))*(${H.ms_per_second}))`;
  return `(MAX(${compute},${memory})+(${spec.cadence})*(${H.launch_seconds})*(${H.ms_per_second}))`;
}

function writeStageSheet(sheet, title, subtitle, specs, I, H, headerRow = 5) {
  writeTitle(sheet, "S", title, subtitle);
  const dataStart = headerRow + 1;
  const dataEnd = dataStart + specs.length - 1;
  const summaryHeaders = ["Total FLOPs", "HBM read B", "HBM write B", "HBM total B", "Persistent state B", "AI FLOP/B", "Aggregate compute ms", "Aggregate HBM ms", "Aggregate latency ms", "Stagewise latency ms", "Bound", "HBM fit", "Ridge AI", "Ridge class"];
  sheet.getRange("A2:N2").unmerge();
  sheet.getRange("A2:N2").values = [summaryHeaders];
  styleHeader(sheet.getRange("A2:N2"), COLORS.gray);
  sheet.getRange("A3:N3").formulas = [[
    `=SUM(G${dataStart}:G${dataEnd})`,
    `=SUM(K${dataStart}:K${dataEnd})`,
    `=SUM(L${dataStart}:L${dataEnd})`,
    `=SUM(K${dataStart}:L${dataEnd})`,
    `=SUM(M${dataStart}:M${dataEnd})`,
    `=IF(SUM(K${dataStart}:L${dataEnd})=0,"",SUM(G${dataStart}:G${dataEnd})/SUM(K${dataStart}:L${dataEnd}))`,
    `=SUM(O${dataStart}:O${dataEnd})`,
    `=SUM(P${dataStart}:P${dataEnd})`,
    `=MAX(SUM(O${dataStart}:O${dataEnd}),SUM(P${dataStart}:P${dataEnd}))`,
    `=SUM(Q${dataStart}:Q${dataEnd})`,
    `=IF(SUM(O${dataStart}:O${dataEnd})>SUM(P${dataStart}:P${dataEnd}),"compute",IF(SUM(P${dataStart}:P${dataEnd})>SUM(O${dataStart}:O${dataEnd}),"memory","balanced"))`,
    `=IF(SUM(M${dataStart}:M${dataEnd})<=${H.hbm_capacity},"FIT","EXCEEDS")`,
    `=${H.peak_flops}/${H.hbm_bandwidth}`,
    `=IF(F3>=M3,"compute-side","memory-side")`,
  ]];
  styleBody(sheet.getRange("A3:N3"));
  sheet.getRange("A3:E3").format.numberFormat = "0.000E+00";
  sheet.getRange("F3:J3").format.numberFormat = "0.000";
  sheet.getRange("M3").format.numberFormat = "0.000";

  sheet.getRange(`A${headerRow}:S${headerRow}`).values = [STAGE_HEADERS];
  styleHeader(sheet.getRange(`A${headerRow}:S${headerRow}`));
  sheet.getRange(`A${headerRow}:S${headerRow}`).format.rowHeight = 34;
  const values = matrix(specs.length, 19);
  const formulas = matrix(specs.length, 19);
  specs.forEach((spec, index) => {
    const row = dataStart + index;
    values[index][1] = spec.phase;
    values[index][2] = spec.component;
    values[index][3] = spec.name;
    values[index][4] = spec.equation;
    values[index][18] = spec.note;
    formulas[index][0] = `=ROW()-${dataStart - 1}`;
    formulas[index][5] = `=${spec.cadence}`;
    formulas[index][6] = `=${spec.flops}`;
    formulas[index][7] = `=${spec.read}`;
    formulas[index][8] = `=${spec.mandatory}`;
    formulas[index][9] = `=${spec.temporary}`;
    formulas[index][10] = `=(${spec.cadence})*(${spec.read})`;
    formulas[index][11] = `=(${spec.cadence})*((${spec.mandatory})+(${spec.temporary}))`;
    formulas[index][12] = `=${spec.persistent}`;
    formulas[index][13] = `=IF(((${spec.cadence})*((${spec.read})+(${spec.mandatory})+(${spec.temporary})))=0,"",((${spec.cadence})*(${spec.flops}))/(((${spec.cadence})*((${spec.read})+(${spec.mandatory})+(${spec.temporary})))))`;
    formulas[index][14] = `=((${spec.cadence})*(${spec.flops})/((${I.compute_efficiency})*(${H.peak_flops}))*(${H.ms_per_second}))`;
    formulas[index][15] = `=((${spec.cadence})*((${spec.read})+(${spec.mandatory})+(${spec.temporary}))/((${I.bandwidth_efficiency})*(${H.hbm_bandwidth}))*(${H.ms_per_second}))`;
    formulas[index][16] = `=${stageTimeFormula(spec, I, H)}`;
    formulas[index][17] = `=IF(O${row}>P${row},"compute",IF(P${row}>O${row},"memory","balanced"))`;
  });
  sheet.getRange(`A${dataStart}:S${dataEnd}`).values = values;
  sheet.getRange(`A${dataStart}:S${dataEnd}`).formulas = formulas;
  styleBody(sheet.getRange(`A${dataStart}:S${dataEnd}`));
  sheet.getRange(`E${dataStart}:E${dataEnd}`).format.wrapText = true;
  sheet.getRange(`S${dataStart}:S${dataEnd}`).format.wrapText = true;
  sheet.getRange(`F${dataStart}:F${dataEnd}`).format.numberFormat = "0.000000";
  sheet.getRange(`G${dataStart}:M${dataEnd}`).format.numberFormat = "#,##0.00";
  sheet.getRange(`N${dataStart}:Q${dataEnd}`).format.numberFormat = "0.000000";
  sheet.freezePanes.freezeRows(headerRow);
  sheet.freezePanes.freezeColumns(4);
  sheet.getRange("O2:S3").format.fill = COLORS.white;
  const widths = [13, 10, 16, 27, 31, 16, 16, 16, 16, 18, 16, 16, 18, 12, 13, 13, 13, 12, 42];
  widths.forEach((width, index) => {
    sheet.getRange(`${colName(index + 1)}:${colName(index + 1)}`).format.columnWidth = width;
  });
  let groupSummary = null;
  if (specs.some((spec) => spec.component === "titans_forward")) {
    const groupHeader = dataEnd + 3;
    const groupStart = groupHeader + 1;
    const groupRows = [
      ["Titans forward", "titans_forward"],
      ["CMS forward", "cms_forward"],
      ["Forward total", "forward_total"],
      ["Loss", "loss"],
      ["Backward + update", "backward_update"],
    ];
    sheet.getRange(`A${groupHeader}:E${groupHeader}`).values = [[
      "Dependency group",
      "FLOPs",
      "HBM read B",
      "HBM write B",
      "Stagewise ms",
    ]];
    styleHeader(sheet.getRange(`A${groupHeader}:E${groupHeader}`), COLORS.gray);
    sheet.getRange(`A${groupStart}:A${groupStart + groupRows.length - 1}`).values = groupRows.map(([label]) => [label]);
    const groupFormula = (group, metricColumn) => group === "forward_total"
      ? `=SUMIF($C$${dataStart}:$C$${dataEnd},"titans_forward",$${metricColumn}$${dataStart}:$${metricColumn}$${dataEnd})+SUMIF($C$${dataStart}:$C$${dataEnd},"cms_forward",$${metricColumn}$${dataStart}:$${metricColumn}$${dataEnd})`
      : `=SUMIF($C$${dataStart}:$C$${dataEnd},"${group}",$${metricColumn}$${dataStart}:$${metricColumn}$${dataEnd})`;
    sheet.getRange(`B${groupStart}:E${groupStart + groupRows.length - 1}`).formulas = groupRows.map(([, group]) => [
      groupFormula(group, "G"),
      groupFormula(group, "K"),
      groupFormula(group, "L"),
      groupFormula(group, "Q"),
    ]);
    styleBody(sheet.getRange(`A${groupStart}:E${groupStart + groupRows.length - 1}`));
    sheet.getRange(`B${groupStart}:D${groupStart + groupRows.length - 1}`).format.numberFormat = "0.000E+00";
    sheet.getRange(`E${groupStart}:E${groupStart + groupRows.length - 1}`).format.numberFormat = "0.000000";
    groupSummary = Object.fromEntries(groupRows.map(([, group], index) => {
      const row = groupStart + index;
      return [group, {
        flops: qref(sheet.name, `$B$${row}`),
        read: qref(sheet.name, `$C$${row}`),
        write: qref(sheet.name, `$D$${row}`),
        stagewiseMs: qref(sheet.name, `$E$${row}`),
      }];
    }));
  }
  return {
    dataStart,
    dataEnd,
    summary: {
      flops: qref(sheet.name, "$A$3"),
      read: qref(sheet.name, "$B$3"),
      write: qref(sheet.name, "$C$3"),
      totalBytes: qref(sheet.name, "$D$3"),
      state: qref(sheet.name, "$E$3"),
      ai: qref(sheet.name, "$F$3"),
      aggregateComputeMs: qref(sheet.name, "$G$3"),
      aggregateHbmMs: qref(sheet.name, "$H$3"),
      aggregateMs: qref(sheet.name, "$I$3"),
      stagewiseMs: qref(sheet.name, "$J$3"),
      bound: qref(sheet.name, "$K$3"),
      fit: qref(sheet.name, "$L$3"),
      ridgeAi: qref(sheet.name, "$M$3"),
      ridgeClass: qref(sheet.name, "$N$3"),
    },
    rows: Object.fromEntries(specs.map((spec, index) => [spec.name, dataStart + index])),
    groups: groupSummary,
  };
}

function writeGuide(sheet) {
  writeTitle(sheet, "H", "Attn vs HOPE · reproducible workbook", "Generated 11-tab companion to the verified E5 analytical engine.");
  const rows = [
    ["Purpose", "Formula-driven analytical companion; edit assumptions only in 01_Inputs and 02_HW."],
    ["Scope", "One full-attention + MoE block versus shipped-credible HOPE; analytical lower bounds, not measured wall clock."],
    ["Lineage", "results.json supplies raw assumptions and QA references. Calculation, comparison, and sweep cells are workbook formulas."],
    ["Live Sheet distinction", "This generated 11-tab workbook is separate from the user's native one-tab live Sheet archived under research/attn-vs-hope; that artifact is not modified."],
    ["Primary latency", "Stagewise sequential sum: Σ(max(compute, HBM) + launch). Aggregate roofline is an optimistic diagnostic."],
    ["HOPE dependency", "Stable order is Titans forward → CMS forward → loss → backward + update. Group summaries contain no synthetic stages."],
    ["End-to-end timing", "Attention TTFT is one full-ISL invocation; HOPE TTFT multiplies per-chunk groups by 64. ITL is one Q=1 token and never multiplies OSL."],
    ["CMS sparsity", "Capacity 64/128/256 controls residency; compute uses router plus top-k=1 active rank-64 expert A/B matrices."],
    ["HBM fit", "Persistent request state is compared with device HBM capacity; model weights and runtime workspace are outside this fit flag."],
    ["Charts", "Only three approved native charts: stagewise latency, roofline scatter, and context decode/state sweep."],
    ["Rebuild", "Run build_workbook.mjs with the bundled @oai/artifact-tool runtime. See DESKTOP-HANDOFF.md."],
  ];
  sheet.getRange("A4:B4").values = [["Topic", "Guidance"]];
  styleHeader(sheet.getRange("A4:B4"));
  sheet.getRange(`A5:B${4 + rows.length}`).values = rows;
  styleBody(sheet.getRange(`A5:B${4 + rows.length}`));
  sheet.getRange(`B5:B${4 + rows.length}`).format.wrapText = true;
  sheet.getRange("A:A").format.columnWidth = 24;
  sheet.getRange("B:B").format.columnWidth = 96;
  sheet.freezePanes.freezeRows(4);
}

function writeHopeDecodeTiming(sheet, info, I, H, memoryRefs) {
  sheet.getRange("A5:E5").values = [["Per-token metric", "Stagewise ms", "Effective HBM B", "Persistent state B", "HBM fit"]];
  styleHeader(sheet.getRange("A5:E5"), COLORS.navy);
  const forward = info.groups.forward_total;
  const loss = info.groups.loss;
  const update = info.groups.backward_update;
  sheet.getRange("A6:A8").values = [["Forward-only ITL"], ["Online-overhead ITL"], ["Including-online ITL"]];
  sheet.getRange("B6:E8").formulas = [
    [`=${forward.stagewiseMs}`, `=(${forward.read})+(${forward.write})`, `=SUM(M${info.dataStart}:M${info.dataEnd})`, `=IF(SUM(M${info.dataStart}:M${info.dataEnd})<=${H.hbm_capacity},"FIT","EXCEEDS")`],
    [`=(${loss.stagewiseMs})+(${update.stagewiseMs})`, `=(${loss.read})+(${loss.write})+(${update.read})+(${update.write})`, `=SUM(M${info.dataStart}:M${info.dataEnd})`, `=IF(SUM(M${info.dataStart}:M${info.dataEnd})<=${H.hbm_capacity},"FIT","EXCEEDS")`],
    [`=B6+B7`, `=C6+C7`, `=SUM(M${info.dataStart}:M${info.dataEnd})`, `=IF(SUM(M${info.dataStart}:M${info.dataEnd})<=${H.hbm_capacity},"FIT","EXCEEDS")`],
  ];
  styleBody(sheet.getRange("A6:E8"));
  sheet.getRange("B6:B8").format.numberFormat = "0.000000";
  sheet.getRange("C6:D8").format.numberFormat = "0.000E+00";
  return {
    forwardItlMs: qref(sheet.name, "$B$6"),
    onlineOverheadItlMs: qref(sheet.name, "$B$7"),
    includingOnlineItlMs: qref(sheet.name, "$B$8"),
    forwardBytes: qref(sheet.name, "$C$6"),
    onlineOverheadBytes: qref(sheet.name, "$C$7"),
    includingOnlineBytes: qref(sheet.name, "$C$8"),
  };
}

function writeCompare(sheet, refs, I, H) {
  writeTitle(sheet, "AA", "30 · Architecture comparison", "Formula-linked section-8 metrics, sweep-derived crossover summaries, and the two approved comparison charts.");
  const headers = [
    "Architecture / phase",
    "Stagewise ms",
    "Aggregate ms",
    "FLOPs",
    "HBM read GiB",
    "HBM write GiB",
    "HBM total GiB",
    "AI FLOP/B",
    "Ridge AI",
    "Ridge class",
    "State GiB",
    "HBM fit",
    "Bound",
  ];
  sheet.getRange("A5:M5").values = [headers];
  styleHeader(sheet.getRange("A5:M5"));
  const cases = [
    ["Attention+MoE · prefill full ISL", refs.attnPrefill],
    ["HOPE shipped · prefill per 2K chunk", refs.hopePrefill],
    ["Attention+MoE · decode Q=1", refs.attnDecode],
    ["HOPE shipped · including-online Q=1", refs.hopeDecode],
  ];
  sheet.getRange("A6:A9").values = cases.map(([label]) => [label]);
  sheet.getRange("B6:M9").formulas = cases.map(([, item]) => [
    `=${item.stagewiseMs}`,
    `=${item.aggregateMs}`,
    `=${item.flops}`,
    `=${item.read}/${H.bytes_per_gib}`,
    `=${item.write}/${H.bytes_per_gib}`,
    `=${item.totalBytes}/${H.bytes_per_gib}`,
    `=${item.ai}`,
    `=${item.ridgeAi}`,
    `=${item.ridgeClass}`,
    `=${item.state}/${H.bytes_per_gib}`,
    `=${item.fit}`,
    `=${item.bound}`,
  ]);
  styleBody(sheet.getRange("A6:M9"));
  sheet.getRange("B6:C9").format.numberFormat = "0.000000";
  sheet.getRange("D6:D9").format.numberFormat = "0.00E+00";
  sheet.getRange("E6:I9").format.numberFormat = "0.000";
  sheet.getRange("K6:K9").format.numberFormat = "0.000";

  const firstCrossoverFormula = (startRow, endRow) => {
    let expression = '"None in sweep"';
    for (let row = endRow; row >= startRow; row -= 1) {
      expression = `IF(${qref("40_Sweeps", `$G$${row}`)}="YES",${qref("40_Sweeps", `$A$${row}`)},${expression})`;
    }
    return `=${expression}`;
  };
  sheet.getRange("A12:D12").values = [["Crossover summary", "First crossover", "Unit", "Rule"]];
  styleHeader(sheet.getRange("A12:D12"), COLORS.gray);
  sheet.getRange("A13:A14").values = [["Context crossover"], ["Batch crossover"]];
  sheet.getRange("B13:B14").formulas = [
    [firstCrossoverFormula(6, 11)],
    [firstCrossoverFormula(17, 22)],
  ];
  sheet.getRange("C13:D14").values = [
    ["tokens", "First sweep row where Attention decode ms >= HOPE decode ms"],
    ["requests", "First sweep row where Attention decode ms >= HOPE decode ms"],
  ];
  styleBody(sheet.getRange("A13:D14"));

  sheet.getRange("A17:C17").values = [["End-to-end metric", "ms", "Definition"]];
  styleHeader(sheet.getRange("A17:C17"), COLORS.gray);
  sheet.getRange("A18:A23").values = [
    ["Full Attention TTFT"],
    ["HOPE forward TTFT"],
    ["HOPE including-online TTFT"],
    ["Full Attention ITL"],
    ["HOPE forward ITL"],
    ["HOPE including-online ITL"],
  ];
  sheet.getRange("B18:B23").formulas = [
    [`=${refs.attnPrefill.stagewiseMs}`],
    [`=(${refs.hopePrefill.groups.forward_total.stagewiseMs})*(${I.prefill_chunks})`],
    [`=(${refs.hopePrefill.stagewiseMs})*(${I.prefill_chunks})`],
    [`=${refs.attnDecode.stagewiseMs}`],
    [`=${refs.hopeTiming.forwardItlMs}`],
    [`=${refs.hopeTiming.includingOnlineItlMs}`],
  ];
  sheet.getRange("C18:C23").values = [
    ["Full ISL causal invocation"],
    ["64 × (Titans + CMS forward per 2K chunk)"],
    ["64 × (forward + loss + backward/update per 2K chunk)"],
    ["One Q=1 Flash-Decode invocation"],
    ["One Q=1 Titans + CMS forward"],
    ["One Q=1 forward + amortized online overhead; no OSL multiplier"],
  ];
  styleBody(sheet.getRange("A18:C23"));
  sheet.getRange("B18:B23").format.numberFormat = "0.000000";

  sheet.getRange("O5:Q5").values = [["Case", "AI FLOP/B", "Effective TFLOP/s"]];
  styleHeader(sheet.getRange("O5:Q5"), COLORS.gray);
  sheet.getRange("O6:O9").values = cases.map(([label]) => [label]);
  sheet.getRange("P6:Q9").formulas = cases.map(([, item]) => [
    `=${item.ai}`,
    `=(${item.flops}/(${item.stagewiseMs}/${H.ms_per_second}))/${H.flops_per_tflop}`,
  ]);
  styleBody(sheet.getRange("O6:Q9"));
  sheet.getRange("P6:Q9").format.numberFormat = "0.000";

  const stageChart = sheet.charts.add("bar", sheet.getRange("A5:C9"));
  stageChart.title = "Stagewise vs aggregate latency (ms)";
  stageChart.hasLegend = true;
  stageChart.yAxis = { numberFormatCode: "0.000" };
  stageChart.setPosition("S4", "AA17");

  const rooflineChart = sheet.charts.add("scatter", sheet.getRange("P5:Q9"));
  rooflineChart.series.getItemAt(0).xFormula = qref("30_Compare", "$P$6:$P$9");
  rooflineChart.title = "Roofline scatter: AI vs effective TFLOP/s";
  rooflineChart.hasLegend = false;
  rooflineChart.xAxis = { numberFormatCode: "0.0" };
  rooflineChart.yAxis = { numberFormatCode: "0.0" };
  rooflineChart.setPosition("S19", "AA33");

  sheet.freezePanes.freezeRows(5);
  [38, 14, 14, 18, 15, 15, 15, 14, 13, 16, 14, 12, 12, 3, 45, 16, 20, 3].forEach((width, index) => {
    sheet.getRange(`${colName(index + 1)}:${colName(index + 1)}`).format.columnWidth = width;
  });
}

function writeSweeps(sheet, refs, I, H) {
  writeTitle(sheet, "AD", "40 · Context and batch sweeps", "All sweep outputs are formulas. Imported engine values appear only on 90_QA as references.");
  const helperStart = 8;
  const contextHeader = 5;
  const contextStart = 6;
  const contextEnd = contextStart + refs.sweepRefs.contexts.length - 1;
  sheet.getRange(`A${contextHeader}:G${contextHeader}`).values = [["Context tokens", "Attention ITL ms", "HOPE including-online ITL ms", "Attention state GiB", "HOPE state GiB", "Attn / HOPE ITL", "Attention >= HOPE?"]];
  styleHeader(sheet.getRange(`A${contextHeader}:G${contextHeader}`));
  const helperHeaders = attentionStages(I, "decode").map((item) => item.name);
  sheet.getRange(`${colName(helperStart)}${contextHeader}:${colName(helperStart + helperHeaders.length - 1)}${contextHeader}`).values = [helperHeaders];
  styleHeader(sheet.getRange(`${colName(helperStart)}${contextHeader}:${colName(helperStart + helperHeaders.length - 1)}${contextHeader}`), COLORS.gray);
  refs.sweepRefs.contexts.forEach((contextRef, index) => {
    const row = contextStart + index;
    const contextCell = `$A$${row}`;
    const splits = `(INT(((${contextCell})+(${I.kv_split_tokens})-1)/(${I.kv_split_tokens})))`;
    const specs = attentionStages(I, "decode", { context: contextCell, splits });
    sheet.getRange(`A${row}`).formulas = [[`=${contextRef}`]];
    specs.forEach((spec, stageIndex) => {
      sheet.getRange(`${colName(helperStart + stageIndex)}${row}`).formulas = [[`=${stageTimeFormula(spec, I, H)}`]];
    });
    const helperEnd = colName(helperStart + specs.length - 1);
    sheet.getRange(`B${row}:G${row}`).formulas = [[
      `=SUM(${colName(helperStart)}${row}:${helperEnd}${row})`,
      `=${refs.hopeTiming.includingOnlineItlMs}`,
      `=(2*(${I.batch})*(${contextCell})*(${I.kv_heads})*(${I.head_dim})*(${I.kv_bytes}))/${H.bytes_per_gib}`,
      `=${refs.hopeDecode.state}/${H.bytes_per_gib}`,
      `=B${row}/C${row}`,
      `=IF(B${row}>=C${row},"YES","NO")`,
    ]];
  });
  styleBody(sheet.getRange(`A${contextStart}:${colName(helperStart + helperHeaders.length - 1)}${contextEnd}`));
  sheet.getRange(`A${contextStart}:A${contextEnd}`).format.numberFormat = "#,##0";
  sheet.getRange(`B${contextStart}:F${contextEnd}`).format.numberFormat = "0.000000";

  const batchHeader = 16;
  const batchStart = 17;
  const batchEnd = batchStart + refs.sweepRefs.batches.length - 1;
  sheet.getRange(`A${batchHeader}:G${batchHeader}`).values = [["Batch", "Attention ITL ms", "HOPE including-online ITL ms", "Attention state GiB", "HOPE state GiB", "Attn / HOPE ITL", "Attention >= HOPE?"]];
  styleHeader(sheet.getRange(`A${batchHeader}:G${batchHeader}`));
  sheet.getRange(`${colName(helperStart)}${batchHeader}:${colName(helperStart + helperHeaders.length - 1)}${batchHeader}`).values = [helperHeaders];
  styleHeader(sheet.getRange(`${colName(helperStart)}${batchHeader}:${colName(helperStart + helperHeaders.length - 1)}${batchHeader}`), COLORS.gray);
  refs.sweepRefs.batches.forEach((batchRef, index) => {
    const row = batchStart + index;
    const batchCell = `$A$${row}`;
    const specs = attentionStages(I, "decode", { batch: batchCell });
    sheet.getRange(`A${row}`).formulas = [[`=${batchRef}`]];
    specs.forEach((spec, stageIndex) => {
      sheet.getRange(`${colName(helperStart + stageIndex)}${row}`).formulas = [[`=${stageTimeFormula(spec, I, H)}`]];
    });
    const helperEnd = colName(helperStart + specs.length - 1);
    const staticRow = refs.hopeDecodeInfo.rows.static_q_projection;
    const baselineStaticMs = qref("21_HOPE_Decode", `$Q$${staticRow}`);
    const otherExecutions = `(SUM(${qref("21_HOPE_Decode", `$F$${refs.hopeDecodeInfo.dataStart}:$F$${refs.hopeDecodeInfo.dataEnd}`)})-${qref("21_HOPE_Decode", `$F$${staticRow}`)})`;
    const otherLaunchMs = `((${otherExecutions})*(${H.launch_seconds})*(${H.ms_per_second}))`;
    const staticN = `((${batchCell})*(${I.decode_query_tokens}))`;
    const staticModelAct = `((${staticN})*(${I.model_dim})*(${I.activation_bytes}))`;
    const staticSpec = stage({
      phase: "decode",
      component: "titans",
      name: "static_q_projection",
      equation: "2ND²",
      cadence: "1",
      flops: `(2*(${staticN})*(${I.model_dim})*(${I.model_dim}))`,
      read: `((${staticModelAct})+(${I.model_dim})*(${I.model_dim})*(${I.weight_bytes}))`,
      mandatory: "0",
      temporary: staticModelAct,
      persistent: "0",
      note: "Static-q batch helper",
    });
    const staticAtBatchMs = stageTimeFormula(staticSpec, I, H);
    sheet.getRange(`B${row}:G${row}`).formulas = [[
      `=SUM(${colName(helperStart)}${row}:${helperEnd}${row})`,
      `=((${refs.hopeTiming.includingOnlineItlMs})-(${baselineStaticMs})-(${otherLaunchMs}))*(${batchCell})/(${I.batch})+(${otherLaunchMs})+(${staticAtBatchMs})`,
      `=(2*(${batchCell})*(${I.decode_context_tokens})*(${I.kv_heads})*(${I.head_dim})*(${I.kv_bytes}))/${H.bytes_per_gib}`,
      `=(${refs.hopeDecode.state})*(${batchCell})/(${I.batch})/${H.bytes_per_gib}`,
      `=B${row}/C${row}`,
      `=IF(B${row}>=C${row},"YES","NO")`,
    ]];
  });
  styleBody(sheet.getRange(`A${batchStart}:${colName(helperStart + helperHeaders.length - 1)}${batchEnd}`));
  sheet.getRange(`A${batchStart}:A${batchEnd}`).format.numberFormat = "#,##0";
  sheet.getRange(`B${batchStart}:F${batchEnd}`).format.numberFormat = "0.000000";

  const chart = sheet.charts.add("line", sheet.getRange(`A${contextHeader}:E${contextEnd}`));
  chart.title = "Context sweep: ITL ms and persistent state GiB";
  chart.hasLegend = true;
  chart.xAxis = { axisType: "textAxis" };
  chart.yAxis = { numberFormatCode: "0.000" };
  chart.setPosition("V4", "AD22");

  sheet.freezePanes.freezeRows(5);
  sheet.getRange("A:A").format.columnWidth = 18;
  sheet.getRange("B:F").format.columnWidth = 20;
  sheet.getRange("G:G").format.columnWidth = 20;
  sheet.getRange(`${colName(helperStart)}:${colName(helperStart + helperHeaders.length - 1)}`).format.columnWidth = 18;
}

function writeQA(sheet, refs, I, H) {
  writeTitle(sheet, "G", "90 · QA and engine reconciliation", "Engine QA is exact. Native live-Sheet outputs are shown separately; assumption differences are explicit and are not forced to match.");
  sheet.getRange("A5:G5").values = [["Check", "Workbook formula", "Engine reference (imported)", "Delta", "Tolerance", "Status", "Notes"]];
  styleHeader(sheet.getRange("A5:G5"));
  const shipped = results.hope_scenarios.shipped_credible_momentum;
  const checks = [
    ["Attention prefill FLOPs", refs.attnPrefill.flops, results.attention_moe.prefill.summary.flops, 0, "Exact integer"],
    ["Attention prefill effective HBM B", refs.attnPrefill.totalBytes, results.attention_moe.prefill.summary.effective_hbm_bytes, 0.01, "Formula reconstruction"],
    ["Attention prefill stagewise ms", refs.attnPrefill.stagewiseMs, results.attention_moe.prefill.summary.stagewise_latency_seconds * 1000, 1e-8, "Primary latency"],
    ["Attention decode FLOPs", refs.attnDecode.flops, results.attention_moe.decode.summary.flops, 0, "Exact integer"],
    ["Attention decode effective HBM B", refs.attnDecode.totalBytes, results.attention_moe.decode.summary.effective_hbm_bytes, 0.01, "Formula reconstruction"],
    ["Attention decode stagewise ms", refs.attnDecode.stagewiseMs, results.attention_moe.decode.summary.stagewise_latency_seconds * 1000, 1e-8, "Primary latency"],
    ["HOPE prefill FLOPs", refs.hopePrefill.flops, shipped.prefill.summary.flops, 0.01, "Shipped-credible momentum"],
    ["HOPE prefill effective HBM B", refs.hopePrefill.totalBytes, shipped.prefill.summary.effective_hbm_bytes, 0.01, "Shipped-credible momentum"],
    ["HOPE prefill stagewise ms", refs.hopePrefill.stagewiseMs, shipped.prefill.summary.stagewise_latency_seconds * 1000, 1e-8, "Primary latency"],
    ["HOPE forward-only ITL ms", refs.hopeTiming.forwardItlMs, shipped.metrics.decode_per_token.forward_itl_ms, 1e-8, "Q=1 Titans + sparse CMS forward"],
    ["HOPE online-overhead ITL ms", refs.hopeTiming.onlineOverheadItlMs, shipped.metrics.decode_per_token.online_overhead_itl_ms, 1e-8, "Loss + backward/amortized update"],
    ["HOPE including-online ITL ms", refs.hopeTiming.includingOnlineItlMs, shipped.metrics.decode_per_token.including_online_itl_ms, 1e-8, "No OSL multiplier"],
    ["Context sweep 1K Attention decode ms", qref("40_Sweeps", "$B$6"), results.crossovers.context_sweep[0].attention_decode_seconds * 1000, 1e-8, "Formula-driven sweep endpoint"],
    ["Context sweep 256K Attention decode ms", qref("40_Sweeps", "$B$11"), results.crossovers.context_sweep.at(-1).attention_decode_seconds * 1000, 1e-8, "Formula-driven sweep endpoint"],
    ["Batch sweep B=1 HOPE decode ms", qref("40_Sweeps", "$C$17"), results.crossovers.batch_sweep[0].hope_decode_seconds * 1000, 1e-8, "Static-q read is not batch-linear"],
    ["Batch sweep B=32 HOPE decode ms", qref("40_Sweeps", "$C$22"), results.crossovers.batch_sweep.at(-1).hope_decode_seconds * 1000, 1e-8, "Formula-driven sweep endpoint"],
  ];
  sheet.getRange(`A6:A${5 + checks.length}`).values = checks.map((item) => [item[0]]);
  sheet.getRange(`B6:B${5 + checks.length}`).formulas = checks.map((item) => [`=${item[1]}`]);
  sheet.getRange(`C6:C${5 + checks.length}`).values = checks.map((item) => [item[2]]);
  sheet.getRange(`D6:F${5 + checks.length}`).formulas = checks.map((item, index) => {
    const row = 6 + index;
    return [`=B${row}-C${row}`, `=${item[3]}`, `=IF(ABS(D${row})<=E${row},"PASS","REVIEW")`];
  });
  sheet.getRange(`G6:G${5 + checks.length}`).values = checks.map((item) => [item[4]]);
  styleBody(sheet.getRange(`A6:G${5 + checks.length}`));
  sheet.getRange(`B6:E${5 + checks.length}`).format.numberFormat = "0.0000000000E+00";

  const qaStart = 9 + checks.length;
  sheet.getRange(`A${qaStart}:C${qaStart}`).values = [["Upstream invariant", "Engine result", "Meaning"]];
  styleHeader(sheet.getRange(`A${qaStart}:C${qaStart}`), COLORS.gray);
  const qaRows = Object.entries(results.qa_checks);
  sheet.getRange(`A${qaStart + 1}:C${qaStart + qaRows.length}`).values = qaRows.map(([name, value]) => [name, value ? "PASS" : "FAIL", "Imported from results.json QA contract"]);
  styleBody(sheet.getRange(`A${qaStart + 1}:C${qaStart + qaRows.length}`));

  const nativeStart = qaStart + qaRows.length + 3;
  const native = results.native_sheet_reference.key_outputs_ms;
  const nativeRows = [
    ["Full Attention TTFT", native.full_attention_ttft.value_ms, refs.attnPrefill.stagewiseMs],
    ["HOPE forward TTFT", native.hope_forward_ttft.value_ms, `((${refs.hopePrefill.groups.forward_total.stagewiseMs})*(${I.prefill_chunks}))`],
    ["HOPE including-online TTFT", native.hope_including_online_ttft.value_ms, `((${refs.hopePrefill.stagewiseMs})*(${I.prefill_chunks}))`],
    ["Full Attention ITL", native.full_attention_itl.value_ms, refs.attnDecode.stagewiseMs],
    ["HOPE forward ITL", native.hope_forward_itl.value_ms, refs.hopeTiming.forwardItlMs],
    ["HOPE including-online ITL", native.hope_including_online_itl.value_ms, refs.hopeTiming.includingOnlineItlMs],
  ];
  sheet.getRange(`A${nativeStart}:F${nativeStart}`).values = [[
    "Native operational metric",
    "Native Sheet ms",
    "Generated companion ms",
    "Delta ms",
    "Reconciliation",
    "Why exact equality is not asserted",
  ]];
  styleHeader(sheet.getRange(`A${nativeStart}:F${nativeStart}`), COLORS.gray);
  sheet.getRange(`A${nativeStart + 1}:B${nativeStart + nativeRows.length}`).values = nativeRows.map(([label, nativeValue]) => [label, nativeValue]);
  sheet.getRange(`C${nativeStart + 1}:E${nativeStart + nativeRows.length}`).formulas = nativeRows.map(([, , generated], index) => {
    const row = nativeStart + 1 + index;
    return [`=${generated}`, `=C${row}-B${row}`, `=IF(ABS(D${row})<=0.001,"MATCH","ASSUMPTION DIFFERENCE")`];
  });
  sheet.getRange(`F${nativeStart + 1}:F${nativeStart + nativeRows.length}`).values = nativeRows.map(() => [results.native_sheet_reference.reconciliation_note]);
  styleBody(sheet.getRange(`A${nativeStart + 1}:F${nativeStart + nativeRows.length}`));
  sheet.getRange(`B${nativeStart + 1}:D${nativeStart + nativeRows.length}`).format.numberFormat = "0.000000";
  sheet.getRange(`F${nativeStart + 1}:F${nativeStart + nativeRows.length}`).format.wrapText = true;
  sheet.freezePanes.freezeRows(5);
  sheet.getRange("A:A").format.columnWidth = 38;
  sheet.getRange("B:F").format.columnWidth = 22;
  sheet.getRange("G:G").format.columnWidth = 34;
  sheet.getRange("F:F").format.columnWidth = 52;
}

function writeSources(sheet) {
  writeTitle(sheet, "E", "99 · Sources and provenance", "Paths and public URLs needed to reproduce or audit the analytical companion.");
  sheet.getRange("A5:E5").values = [["Source", "Kind", "Location / URL", "Used for", "Notes"]];
  styleHeader(sheet.getRange("A5:E5"));
  const rows = [
    ["Verified engine output", "Repository file", "experiments/E5-attn-vs-hope/results.json", "Raw inputs, QA references, stage/group-order assertion", "Schema v2 dependency and TTFT/ITL contract"],
    ["Analytical model", "Repository file", "experiments/E5-attn-vs-hope/model.py", "Equations and stage semantics", "Read-only upstream"],
    ["Approved design", "Repository file", results.sources.design, "Workbook sections and validation contract", "Sections 4–9"],
    ["Hardware authority", "Mixed source", results.sources.hardware.source, "Live-Sheet peak/BW; twin capacity/provenance only", results.sources.hardware.authority],
    ["Legacy HOPE proxy", "Repository file", results.sources.legacy, "Legacy algebra QA anchor", "Not used as primary timing"],
    ["Native live Sheet archive", "Repository file", results.sources.live_sheet_archive, "User-reviewed one-tab artifact", "Kept separate and untouched"],
    ["HOPE / Nested Learning", "Public paper", "https://arxiv.org/abs/2512.24695", "Architecture/equation evidence", "No official GPU kernel or wall-clock trace"],
    ["Meeting research package", "Repository directory", results.sources.google_meeting_package, "Evidence map and deployment questions", "Context only"],
  ];
  sheet.getRange(`A6:E${5 + rows.length}`).values = rows;
  styleBody(sheet.getRange(`A6:E${5 + rows.length}`));
  sheet.getRange(`C6:E${5 + rows.length}`).format.wrapText = true;
  sheet.freezePanes.freezeRows(5);
  [28, 20, 62, 40, 44].forEach((width, index) => {
    sheet.getRange(`${colName(index + 1)}:${colName(index + 1)}`).format.columnWidth = width;
  });
}

async function inspectWorkbook(workbook) {
  const overview = await workbook.inspect({ kind: "sheet,drawing", include: "id,name,type", maxChars: 5000 });
  console.log("INSPECT_OVERVIEW");
  console.log(overview.ndjson);
  const compare = await workbook.inspect({ kind: "table", range: "'30_Compare'!A5:Q22", include: "values,formulas", tableMaxRows: 24, tableMaxCols: 17, maxChars: 12000 });
  console.log("INSPECT_COMPARE");
  console.log(compare.ndjson);
  const timing = await workbook.inspect({ kind: "table", range: "'21_HOPE_Decode'!A5:E8", include: "values,formulas", tableMaxRows: 8, tableMaxCols: 5, maxChars: 5000 });
  console.log("INSPECT_HOPE_TIMING");
  console.log(timing.ndjson);
  const sweeps = await workbook.inspect({ kind: "table", range: "'40_Sweeps'!A5:G22", include: "values,formulas", tableMaxRows: 22, tableMaxCols: 7, maxChars: 8000 });
  console.log("INSPECT_SWEEPS");
  console.log(sweeps.ndjson);
  const qa = await workbook.inspect({ kind: "table", range: "'90_QA'!A5:G21", include: "values,formulas", tableMaxRows: 24, tableMaxCols: 7, maxChars: 9000 });
  console.log("INSPECT_QA");
  console.log(qa.ndjson);
  const errors = await workbook.inspect({ kind: "match", searchTerm: "#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A", options: { useRegex: true, maxResults: 300 }, summary: "final formula error scan", maxChars: 6000 });
  console.log("FORMULA_ERROR_SCAN");
  console.log(errors.ndjson);
}

async function renderAll(workbook) {
  await fs.mkdir(PREVIEW_DIR, { recursive: true });
  for (const sheetName of SHEET_NAMES) {
    const preview = await workbook.render({ sheetName, autoCrop: "all", scale: 1, format: "png" });
    const bytes = new Uint8Array(await preview.arrayBuffer());
    const target = path.join(PREVIEW_DIR, `${sheetName}.png`);
    await fs.writeFile(target, bytes);
    console.log(`RENDERED ${sheetName} ${bytes.length} bytes ${target}`);
  }
}

async function buildWorkbook() {
  const workbook = Workbook.create();
  const sheets = Object.fromEntries(SHEET_NAMES.map((name) => [name, workbook.worksheets.add(name)]));
  writeGuide(sheets["00_Guide"]);
  const { refs: I, memoryRefs, cmsRefs, sweepRefs } = writeKeyValueInputs(sheets["01_Inputs"]);
  const H = writeHardware(sheets["02_HW"]);

  const attnPrefillSpecs = attentionStages(I, "prefill");
  const attnDecodeSpecs = attentionStages(I, "decode");
  const hopePrefillSpecs = hopeStages(I, memoryRefs, cmsRefs, "prefill");
  const hopeDecodeSpecs = hopeStages(I, memoryRefs, cmsRefs, "decode", "amortized");
  assertStageNames("Attention prefill", attnPrefillSpecs, results.attention_moe.prefill.stages);
  assertStageNames("Attention decode", attnDecodeSpecs, results.attention_moe.decode.stages);
  assertStageNames("HOPE prefill", hopePrefillSpecs, results.hope_scenarios.shipped_credible_momentum.prefill.stages);
  assertStageNames("HOPE amortized decode", hopeDecodeSpecs, results.hope_scenarios.shipped_credible_momentum.decode.amortized.stages);

  const attnPrefillInfo = writeStageSheet(sheets["10_AttnMoE_Prefill"], "10 · Attention + MoE prefill", "Causal FlashAttention chunk followed by top-k SwiGLU MoE. All numeric calculation cells are formulas.", attnPrefillSpecs, I, H);
  const attnDecodeInfo = writeStageSheet(sheets["11_AttnMoE_Decode"], "11 · Attention + MoE decode", "One next-token Flash-Decode invocation; split-local partial traffic is explicit.", attnDecodeSpecs, I, H);
  const hopePrefillInfo = writeStageSheet(sheets["20_HOPE_Prefill"], "20 · HOPE prefill", "One 2K chunk in canonical Titans-forward → sparse-CMS-forward → loss → backward/update order.", hopePrefillSpecs, I, H);
  const hopeDecodeInfo = writeStageSheet(sheets["21_HOPE_Decode"], "21 · HOPE decode", "One Q=1 token; forward-only, online overhead, and including-online ITL remain separate.", hopeDecodeSpecs, I, H, 11);
  const hopeTiming = writeHopeDecodeTiming(sheets["21_HOPE_Decode"], hopeDecodeInfo, I, H, memoryRefs);

  const refs = {
    I,
    H,
    sweepRefs,
    attnPrefill: attnPrefillInfo.summary,
    attnDecode: attnDecodeInfo.summary,
    hopePrefill: { ...hopePrefillInfo.summary, groups: hopePrefillInfo.groups },
    hopeDecode: { ...hopeDecodeInfo.summary, groups: hopeDecodeInfo.groups },
    hopeDecodeInfo,
    hopeTiming,
  };
  writeSweeps(sheets["40_Sweeps"], refs, I, H);
  writeCompare(sheets["30_Compare"], refs, I, H);
  writeQA(sheets["90_QA"], refs, I, H);
  writeSources(sheets["99_Sources"]);
  return workbook;
}

async function loadExisting() {
  const input = await FileBlob.load(OUTPUT_PATH);
  return SpreadsheetFile.importXlsx(input);
}

const mode = process.argv[2] ?? "--build";
if (!["--build", "--inspect-only", "--render-only"].includes(mode)) {
  throw new Error("Usage: node build_workbook.mjs [--build|--inspect-only|--render-only]");
}

if (mode === "--inspect-only") {
  await inspectWorkbook(await loadExisting());
} else if (mode === "--render-only") {
  await renderAll(await loadExisting());
} else {
  const workbook = await buildWorkbook();
  await inspectWorkbook(workbook);
  await renderAll(workbook);
  const output = await SpreadsheetFile.exportXlsx(workbook);
  await output.save(OUTPUT_PATH);
  console.log(`EXPORTED ${OUTPUT_PATH}`);
}
