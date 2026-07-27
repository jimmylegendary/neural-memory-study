#!/usr/bin/env python3
"""Generate deterministic Attention+MoE versus HOPE reference outputs."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

from model import (
    AttentionMoEConfig,
    CMSLevel,
    Hardware,
    HopeConfig,
    MemorySpec,
    Stage,
    build_attention_decode,
    build_attention_prefill,
    build_hope_decode,
    build_hope_prefill,
    decode_partial_bytes,
    legacy_hope_proxy,
    summarize,
)


ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = Path(__file__).resolve().parent
H100_PATH = ROOT / "multiarch" / "twins" / "h100.json"
RESULTS_PATH = OUTPUT_DIR / "results.json"
STDOUT_PATH = OUTPUT_DIR / "stdout.txt"

# User-reviewed live-sheet inputs.  Prefill rows model one 2K-token chunk;
# ``prefill_chunks`` is applied only by downstream TTFT formulas.
MODEL_INPUTS = {
    "batch": 32,
    "model_dim": 8_192,
    "input_sequence_tokens": 131_072,
    "output_sequence_tokens": 8_192,
    "prefill_chunk_tokens": 2_048,
    "prefill_chunks": 64,
    "head_dim": 128,
    "experts": 128,
    "top_k": 1,
    "expert_hidden_dim": 16_384,
    "element_bytes": 1,
    "partial_bytes": 4,
    "kv_split_tokens": 2_048,
    "decode_splits": 64,
    "memory_hidden_dim": 8_192,
    "main_memory_chunk": 2_048,
    "aux_memory_chunk": 2_048,
    "cms_low_rank_dim": 64,
    "cms_capacities": [64, 128, 256],
    "cms_update_periods": [1_000, 5_000, 10_000],
    "cms_bptt_span": 4,
}


def _find_level(node: dict[str, Any], level_id: str) -> dict[str, Any]:
    if node.get("level_id") == level_id:
        return node
    for child in node.get("children", []):
        try:
            return _find_level(child, level_id)
        except KeyError:
            pass
    raise KeyError(level_id)


def load_h100() -> tuple[Hardware, dict[str, Any]]:
    twin = json.loads(H100_PATH.read_text(encoding="utf-8"))
    hbm = _find_level(twin, "hbm")
    mxu = _find_level(twin, "mxu")
    peak_flops = 2 * mxu["instances"] * mxu["peak_macs_per_s"]
    hardware = Hardware(
        name="NVIDIA H100 SXM5 (repository twin)",
        peak_flops_per_second=peak_flops,
        hbm_bandwidth_bytes_per_second=hbm["bandwidth_bps"],
        hbm_capacity_bytes=hbm["capacity_bytes"],
        # The twin does not publish a kernel-launch constant.  Zero preserves
        # an auditable analytical lower bound instead of inventing one.
        kernel_launch_overhead_seconds=0.0,
    )
    provenance = {
        "source": str(H100_PATH.relative_to(ROOT)),
        "hbm_level_id": hbm["level_id"],
        "compute_level_id": mxu["level_id"],
        "peak_derivation": "2 FLOP/MAC * instances * peak_macs_per_s",
        "peak_macs_per_s_per_instance": mxu["peak_macs_per_s"],
        "instances": mxu["instances"],
    }
    return hardware, provenance


def attention_config(
    *, batch: int, query_tokens: int, context_tokens: int
) -> AttentionMoEConfig:
    d = MODEL_INPUTS["model_dim"]
    head_dim = MODEL_INPUTS["head_dim"]
    heads = d // head_dim
    splits = (
        1
        if query_tokens > 1
        else math.ceil(context_tokens / MODEL_INPUTS["kv_split_tokens"])
    )
    return AttentionMoEConfig(
        batch=batch,
        query_tokens=query_tokens,
        context_tokens=context_tokens,
        model_dim=d,
        query_heads=heads,
        kv_heads=heads,
        head_dim=head_dim,
        activation_bytes=MODEL_INPUTS["element_bytes"],
        weight_bytes=MODEL_INPUTS["element_bytes"],
        kv_bytes=MODEL_INPUTS["element_bytes"],
        partial_bytes=MODEL_INPUTS["partial_bytes"],
        lse_bytes=MODEL_INPUTS["partial_bytes"],
        decode_splits=splits,
        kv_reload_multiplier=1.0,
        softmax_flops_per_pair=5,
        experts=MODEL_INPUTS["experts"],
        top_k=MODEL_INPUTS["top_k"],
        expert_hidden_dim=MODEL_INPUTS["expert_hidden_dim"],
        expert_matrices=3,
        router_flops_per_token_expert=1,
        active_unique_experts=None,
        expert_hbm_fraction=1.0,
        compute_efficiency=1.0,
        bandwidth_efficiency=1.0,
    )


def _memory_specs(momentum_slots: int, adaptive_q: bool) -> tuple[MemorySpec, ...]:
    d = MODEL_INPUTS["model_dim"]
    hidden = MODEL_INPUTS["memory_hidden_dim"]
    aux_chunk = MODEL_INPUTS["aux_memory_chunk"]
    main_chunk = MODEL_INPUTS["main_memory_chunk"]
    values = [
        MemorySpec("M_k", d, hidden, d, aux_chunk, momentum_slots, 4.0, 1.0),
        MemorySpec("M_v", d, hidden, d, aux_chunk, momentum_slots, 4.0, 1.0),
        MemorySpec("M_eta", d, hidden, 1, aux_chunk, momentum_slots, 4.0, 1.0),
        MemorySpec("M_alpha", d, hidden, 1, aux_chunk, momentum_slots, 4.0, 1.0),
        MemorySpec("M_mem", d, hidden, d, main_chunk, momentum_slots, 4.0, 1.0),
    ]
    if adaptive_q:
        values.append(
            MemorySpec(
                "M_q", d, hidden, d, aux_chunk, momentum_slots, 4.0, 1.0
            )
        )
    return tuple(values)


def _cms_levels(optimizer_slots: int) -> tuple[CMSLevel, ...]:
    d = MODEL_INPUTS["model_dim"]
    low_rank = MODEL_INPUTS["cms_low_rank_dim"]
    return tuple(
        CMSLevel(
            name=f"cms_l{index}",
            input_dim=d,
            # Flatten capacity low-rank experts into an equivalent aggregate
            # hidden width: P=2*D*(LRD*capacity).
            hidden_dim=low_rank * capacity,
            output_dim=d,
            update_period=period,
            bptt_span=MODEL_INPUTS["cms_bptt_span"],
            optimizer_slots=optimizer_slots,
            personalized=True,
            gradient_multiplier=1.0,
        )
        for index, (capacity, period) in enumerate(
            zip(
                MODEL_INPUTS["cms_capacities"],
                MODEL_INPUTS["cms_update_periods"],
                strict=True,
            ),
            start=1,
        )
    )


def hope_config(
    *,
    batch: int,
    query_tokens: int,
    context_tokens: int,
    momentum_slots: int,
    adaptive_q: bool,
    paper_lower_bound: bool,
) -> HopeConfig:
    return HopeConfig(
        batch=batch,
        query_tokens=query_tokens,
        context_tokens=context_tokens,
        model_dim=MODEL_INPUTS["model_dim"],
        activation_bytes=MODEL_INPUTS["element_bytes"],
        weight_bytes=MODEL_INPUTS["element_bytes"],
        state_bytes=MODEL_INPUTS["element_bytes"],
        optimizer_bytes=4,
        memories=_memory_specs(momentum_slots, adaptive_q),
        cms_levels=_cms_levels(momentum_slots),
        position=0,
        adaptive_q=adaptive_q,
        schedule="eager_gradient",
        # The live-sheet baseline excludes CMS online backward/update.
        include_cms_updates=False,
        chunk_cache_bytes_per_request=0,
        sigma_flops_per_hidden=0 if paper_lower_bound else 4,
        residual_flops_per_output=0 if paper_lower_bound else 1,
        compute_efficiency=1.0,
        bandwidth_efficiency=1.0,
    )


def _stage_record(order: int, stage: Stage) -> dict[str, Any]:
    intensity = stage.arithmetic_intensity
    return {
        "order": order,
        **asdict(stage),
        "effective_hbm_bytes": stage.effective_hbm_bytes,
        "arithmetic_intensity": intensity if math.isfinite(intensity) else None,
        "compute_seconds": stage.compute_seconds,
        "memory_seconds": stage.memory_seconds,
        "roofline_seconds": stage.roofline_seconds,
        "bound": (
            "compute"
            if stage.compute_seconds > stage.memory_seconds
            else "memory"
            if stage.memory_seconds > stage.compute_seconds
            else "balanced"
        ),
    }


def _case(stages: list[Stage], hardware: Hardware) -> dict[str, Any]:
    return {
        "summary": summarize(stages, hardware),
        "stages": [
            _stage_record(order, stage) for order, stage in enumerate(stages, 1)
        ],
    }


def _hope_scenario(
    hardware: Hardware,
    *,
    momentum_slots: int,
    adaptive_q: bool,
    paper_lower_bound: bool,
    included_in_defaults: bool,
) -> dict[str, Any]:
    prefill = hope_config(
        batch=MODEL_INPUTS["batch"],
        query_tokens=MODEL_INPUTS["prefill_chunk_tokens"],
        context_tokens=MODEL_INPUTS["prefill_chunk_tokens"],
        momentum_slots=momentum_slots,
        adaptive_q=adaptive_q,
        paper_lower_bound=paper_lower_bound,
    )
    decode = hope_config(
        batch=MODEL_INPUTS["batch"],
        query_tokens=1,
        context_tokens=MODEL_INPUTS["input_sequence_tokens"],
        momentum_slots=momentum_slots,
        adaptive_q=adaptive_q,
        paper_lower_bound=paper_lower_bound,
    )
    return {
        "included_in_defaults": included_in_defaults,
        "inputs": {"prefill": asdict(prefill), "decode": asdict(decode)},
        "prefill": _case(build_hope_prefill(prefill, hardware), hardware),
        "decode": {
            timing: _case(
                build_hope_decode(decode, hardware, timing=timing), hardware
            )
            for timing in ("normal", "boundary", "amortized")
        },
    }


def _crossovers(hardware: Hardware) -> dict[str, Any]:
    context_records = []
    for context in (1_024, 4_096, 16_384, 65_536, 131_072, 262_144):
        attention = summarize(
            build_attention_decode(
                attention_config(
                    batch=MODEL_INPUTS["batch"],
                    query_tokens=1,
                    context_tokens=context,
                ),
                hardware,
            ),
            hardware,
        )
        hope = summarize(
            build_hope_decode(
                hope_config(
                    batch=MODEL_INPUTS["batch"],
                    query_tokens=1,
                    context_tokens=context,
                    momentum_slots=1,
                    adaptive_q=False,
                    paper_lower_bound=False,
                ),
                hardware,
                timing="amortized",
            ),
            hardware,
        )
        context_records.append(
            {
                "context_tokens": context,
                "attention_decode_seconds": attention[
                    "stagewise_latency_seconds"
                ],
                "hope_decode_seconds": hope["stagewise_latency_seconds"],
                "attention_persistent_state_bytes": attention[
                    "persistent_state_bytes"
                ],
                "hope_persistent_state_bytes": hope["persistent_state_bytes"],
            }
        )

    batch_records = []
    for batch in (1, 2, 4, 8, 16, 32):
        attention = summarize(
            build_attention_decode(
                attention_config(
                    batch=batch,
                    query_tokens=1,
                    context_tokens=MODEL_INPUTS["input_sequence_tokens"],
                ),
                hardware,
            ),
            hardware,
        )
        hope = summarize(
            build_hope_decode(
                hope_config(
                    batch=batch,
                    query_tokens=1,
                    context_tokens=MODEL_INPUTS["input_sequence_tokens"],
                    momentum_slots=1,
                    adaptive_q=False,
                    paper_lower_bound=False,
                ),
                hardware,
                timing="amortized",
            ),
            hardware,
        )
        batch_records.append(
            {
                "batch": batch,
                "attention_decode_seconds": attention[
                    "stagewise_latency_seconds"
                ],
                "hope_decode_seconds": hope["stagewise_latency_seconds"],
                "attention_persistent_state_bytes": attention[
                    "persistent_state_bytes"
                ],
                "hope_persistent_state_bytes": hope["persistent_state_bytes"],
            }
        )
    first_crossover = next(
        (
            record["context_tokens"]
            for record in context_records
            if record["attention_decode_seconds"]
            >= record["hope_decode_seconds"]
        ),
        None,
    )
    return {
        "context_sweep": context_records,
        "batch_sweep": batch_records,
        "first_context_where_attention_decode_not_faster": first_crossover,
    }


def _qa_checks(
    hardware: Hardware,
    attention_prefill: list[Stage],
    attention_decode: list[Stage],
    scenarios: dict[str, Any],
    default_scenarios: list[str],
) -> dict[str, bool]:
    short_cfg = attention_config(batch=1, query_tokens=1, context_tokens=1_024)
    long_cfg = attention_config(batch=1, query_tokens=1, context_tokens=2_048)
    short_stages = build_attention_decode(short_cfg, hardware)
    long_stages = build_attention_decode(long_cfg, hardware)
    stage = lambda stages, name: next(item for item in stages if item.stage == name)

    hope_short = hope_config(
        batch=1,
        query_tokens=1,
        context_tokens=8,
        momentum_slots=1,
        adaptive_q=False,
        paper_lower_bound=False,
    )
    hope_long = replace(hope_short, context_tokens=1_000_000)
    hope_two = replace(hope_short, batch=2)
    state_short = summarize(
        build_hope_decode(hope_short, hardware), hardware
    )["persistent_state_bytes"]
    state_long = summarize(
        build_hope_decode(hope_long, hardware), hardware
    )["persistent_state_bytes"]
    state_two = summarize(
        build_hope_decode(hope_two, hardware), hardware
    )["persistent_state_bytes"]

    cadence_fast = replace(hope_short.cms_levels[0], update_period=1)
    cadence_slow = replace(hope_short.cms_levels[0], update_period=10_000)
    cms_fast_cfg = replace(
        hope_short, cms_levels=(cadence_fast, *hope_short.cms_levels[1:])
    )
    cms_slow_cfg = replace(
        hope_short, cms_levels=(cadence_slow, *hope_short.cms_levels[1:])
    )
    cms_fast_read = stage(
        build_hope_decode(cms_fast_cfg, hardware), "cms_l1_forward"
    ).hbm_read_bytes
    cms_slow_read = stage(
        build_hope_decode(cms_slow_cfg, hardware), "cms_l1_forward"
    ).hbm_read_bytes

    all_cases = [
        _case(attention_prefill, hardware),
        _case(attention_decode, hardware),
    ]
    for name, scenario in scenarios.items():
        if name == "legacy_repo_proxy":
            continue
        all_cases.append(scenario["prefill"])
        all_cases.extend(scenario["decode"].values())
    legacy = scenarios["legacy_repo_proxy"]["proxy"]
    shipped_stage_names = {
        record["stage"]
        for record in scenarios["shipped_credible_momentum"]["decode"]
        ["amortized"]["stages"]
    }
    prefill_flash = stage(attention_prefill, "flash_attention")
    return {
        "all_stagewise_not_below_aggregate": all(
            case["summary"]["stagewise_latency_seconds"]
            >= case["summary"]["aggregate_latency_seconds"]
            for case in all_cases
        ),
        "causal_flash_has_no_quadratic_temporary_write": (
            prefill_flash.temporary_write_bytes == 0
            and prefill_flash.mandatory_write_bytes > 0
        ),
        "decode_kv_read_is_linear_in_context": (
            stage(long_stages, "flash_decode_partials").hbm_read_bytes
            == 2 * stage(short_stages, "flash_decode_partials").hbm_read_bytes
        ),
        "kv_append_is_context_independent": (
            stage(long_stages, "kv_cache_append").mandatory_write_bytes
            == stage(short_stages, "kv_cache_append").mandatory_write_bytes
        ),
        "flash_decode_partials_zero_at_one_split": (
            decode_partial_bytes(replace(short_cfg, decode_splits=1)) == 0
            and decode_partial_bytes(replace(short_cfg, decode_splits=2)) > 0
        ),
        "hope_state_is_context_independent": state_short == state_long,
        "hope_state_is_batch_linear": state_two == 2 * state_short,
        "cms_forward_read_is_not_divided_by_update_period": (
            cms_fast_read == cms_slow_read and cms_fast_read > 0
        ),
        "shipped_default_has_no_M_q": not any(
            "M_q" in name for name in shipped_stage_names
        ),
        "adaptive_q_is_excluded_from_defaults": (
            "adaptive_q_hypothetical" not in default_scenarios
            and not scenarios["adaptive_q_hypothetical"][
                "included_in_defaults"
            ]
        ),
        "legacy_flops_anchor_matches": legacy["flops"] == 1_082_187_776,
        "legacy_hbm_anchor_matches": legacy["hbm_bytes"] == 679_510_016,
    }


def build_results() -> dict[str, Any]:
    hardware, hardware_provenance = load_h100()
    prefill_cfg = attention_config(
        batch=MODEL_INPUTS["batch"],
        query_tokens=MODEL_INPUTS["prefill_chunk_tokens"],
        context_tokens=MODEL_INPUTS["prefill_chunk_tokens"],
    )
    decode_cfg = attention_config(
        batch=MODEL_INPUTS["batch"],
        query_tokens=1,
        context_tokens=MODEL_INPUTS["input_sequence_tokens"],
    )
    attention_prefill = build_attention_prefill(prefill_cfg, hardware)
    attention_decode = build_attention_decode(decode_cfg, hardware)
    default_scenarios = [
        "paper_equation_lower_bound",
        "shipped_credible_momentum",
        "legacy_repo_proxy",
    ]
    scenarios: dict[str, Any] = {
        "paper_equation_lower_bound": _hope_scenario(
            hardware,
            momentum_slots=0,
            adaptive_q=False,
            paper_lower_bound=True,
            included_in_defaults=True,
        ),
        "shipped_credible_momentum": _hope_scenario(
            hardware,
            momentum_slots=1,
            adaptive_q=False,
            paper_lower_bound=False,
            included_in_defaults=True,
        ),
        "legacy_repo_proxy": {
            "included_in_defaults": True,
            "source": "multiarch/hope_block_dag.py",
            "proxy": legacy_hope_proxy(d=2_048, chunk=1, element_bytes=2),
        },
        "adaptive_q_hypothetical": _hope_scenario(
            hardware,
            momentum_slots=1,
            adaptive_q=True,
            paper_lower_bound=False,
            included_in_defaults=False,
        ),
    }
    qa_checks = _qa_checks(
        hardware,
        attention_prefill,
        attention_decode,
        scenarios,
        default_scenarios,
    )
    return {
        "schema_version": 1,
        "method": (
            "Analytical stagewise roofline lower bounds; one MAC is two FLOPs; "
            "effective bytes cross the HBM boundary only."
        ),
        "sources": {
            "design": (
                "docs/superpowers/specs/"
                "2026-07-27-attn-vs-hope-analytical-model-design.md"
            ),
            "hardware": hardware_provenance,
            "legacy": "multiarch/hope_block_dag.py",
            "live_sheet_archive": "research/attn-vs-hope/Attn-vs-HOPE.xlsx",
            "google_meeting_package": "research/google-meeting/",
        },
        "hardware": asdict(hardware),
        "model_inputs": MODEL_INPUTS,
        "default_scenarios": default_scenarios,
        "attention_moe": {
            "inputs": {
                "prefill": asdict(prefill_cfg),
                "decode": asdict(decode_cfg),
            },
            "prefill": _case(attention_prefill, hardware),
            "decode": _case(attention_decode, hardware),
        },
        "hope_scenarios": scenarios,
        "crossovers": _crossovers(hardware),
        "qa_checks": qa_checks,
    }


def _milliseconds(case: dict[str, Any]) -> float:
    return case["summary"]["stagewise_latency_seconds"] * 1_000


def compact_report(results: dict[str, Any]) -> str:
    hardware = results["hardware"]
    attention = results["attention_moe"]
    shipped = results["hope_scenarios"]["shipped_credible_momentum"]
    legacy = results["hope_scenarios"]["legacy_repo_proxy"]["proxy"]
    checks = results["qa_checks"]
    lines = [
        "Attn-vs-HOPE deterministic analytical reference",
        (
            f"Hardware: {hardware['name']} | peak "
            f"{hardware['peak_flops_per_second'] / 1e12:.3f} TFLOP/s | HBM "
            f"{hardware['hbm_bandwidth_bytes_per_second'] / 1e12:.3f} TB/s"
        ),
        (
            "Attention+MoE stagewise lower bound: "
            f"prefill-chunk {_milliseconds(attention['prefill']):.6f} ms | "
            f"decode {_milliseconds(attention['decode']):.6f} ms"
        ),
        (
            "HOPE shipped-credible stagewise lower bound: "
            f"prefill-chunk {_milliseconds(shipped['prefill']):.6f} ms | "
            f"decode normal {_milliseconds(shipped['decode']['normal']):.6f} ms | "
            f"boundary {_milliseconds(shipped['decode']['boundary']):.6f} ms | "
            f"amortized {_milliseconds(shipped['decode']['amortized']):.6f} ms"
        ),
        (
            f"Legacy proxy anchor: {legacy['flops']} FLOPs | "
            f"{legacy['hbm_bytes']} HBM bytes"
        ),
        (
            f"QA: {sum(checks.values())}/{len(checks)} PASS | "
            "latencies are analytical lower bounds, not wall-clock measurements"
        ),
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    results = build_results()
    report = compact_report(results)
    RESULTS_PATH.write_text(
        json.dumps(
            results,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    STDOUT_PATH.write_text(report, encoding="utf-8")
    print(report, end="")


if __name__ == "__main__":
    main()
