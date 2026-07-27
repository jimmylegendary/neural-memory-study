#!/usr/bin/env python3
"""Contract tests for the Attention+MoE versus HOPE analytical engine."""

from __future__ import annotations

import sys
import unittest
import hashlib
import json
import subprocess
import tempfile
from dataclasses import FrozenInstanceError, fields, replace
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parent))

from model import (  # noqa: E402
    AttentionMoEConfig,
    Hardware,
    build_attention_decode,
    build_attention_prefill,
    causal_pair_count,
    decode_partial_bytes,
)
import model as analytical_model  # noqa: E402


class PrimitiveInvariantTests(unittest.TestCase):
    def setUp(self) -> None:
        self.hardware = Hardware(
            name="test accelerator",
            peak_flops_per_second=1_000_000.0,
            hbm_bandwidth_bytes_per_second=100_000.0,
            hbm_capacity_bytes=1_000_000,
            kernel_launch_overhead_seconds=0.0,
        )
        self.prefill = self.attention_config(
            batch=2,
            query_tokens=4,
            context_tokens=4,
            query_heads=4,
            kv_heads=2,
            head_dim=4,
            model_dim=16,
        )

    @staticmethod
    def attention_config(**overrides: object) -> AttentionMoEConfig:
        values: dict[str, object] = {
            "batch": 1,
            "query_tokens": 1,
            "context_tokens": 8,
            "model_dim": 16,
            "query_heads": 4,
            "kv_heads": 2,
            "head_dim": 4,
            "activation_bytes": 2,
            "weight_bytes": 2,
            "kv_bytes": 2,
            "partial_bytes": 4,
            "lse_bytes": 4,
            "decode_splits": 1,
            "kv_reload_multiplier": 1.0,
            "softmax_flops_per_pair": 5,
            "experts": 8,
            "top_k": 2,
            "expert_hidden_dim": 32,
            "expert_matrices": 3,
            "router_flops_per_token_expert": 1,
            "active_unique_experts": None,
            "expert_hbm_fraction": 1.0,
            "compute_efficiency": 1.0,
            "bandwidth_efficiency": 1.0,
        }
        values.update(overrides)
        return AttentionMoEConfig(**values)

    def test_causal_pairs(self) -> None:
        self.assertEqual(causal_pair_count(batch=2, heads=4, query_tokens=3), 48)

    def test_kv_traffic_counts_k_and_v(self) -> None:
        cfg = self.attention_config(
            query_tokens=1, context_tokens=8, kv_heads=2, head_dim=4
        )
        stages = build_attention_decode(cfg, self.hardware)
        flash = next(
            stage for stage in stages if stage.stage == "flash_decode_partials"
        )
        self.assertEqual(
            flash.hbm_read_bytes,
            2 * cfg.batch * 8 * 2 * 4 * cfg.kv_bytes,
        )

    def test_flash_attention_has_no_quadratic_write(self) -> None:
        flash = next(
            stage
            for stage in build_attention_prefill(self.prefill, self.hardware)
            if stage.stage == "flash_attention"
        )
        self.assertEqual(flash.temporary_write_bytes, 0)
        self.assertGreater(flash.mandatory_write_bytes, 0)

    def test_flash_decode_partial_traffic_depends_on_splits(self) -> None:
        one = self.attention_config(decode_splits=1)
        many = self.attention_config(decode_splits=8)
        self.assertEqual(decode_partial_bytes(one), 0)
        self.assertGreater(decode_partial_bytes(many), 0)

    def test_query_width_can_differ_from_model_width(self) -> None:
        cfg = self.attention_config(model_dim=32, query_heads=4, head_dim=4)
        self.assertEqual(cfg.query_width, 16)


class BuilderAndSummaryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.hardware = Hardware(
            name="test accelerator",
            peak_flops_per_second=1_000_000.0,
            hbm_bandwidth_bytes_per_second=100_000.0,
            hbm_capacity_bytes=1_000_000_000,
            kernel_launch_overhead_seconds=0.0,
        )
        self.prefill = PrimitiveInvariantTests.attention_config(
            batch=2,
            query_tokens=4,
            context_tokens=4,
            model_dim=16,
            query_heads=4,
            kv_heads=2,
            head_dim=4,
        )

    @staticmethod
    def hope_config(**overrides: object):
        memory = analytical_model.MemorySpec
        level = analytical_model.CMSLevel
        memories = (
            memory("M_k", 16, 16, 16, 4, 1, 4.0, 1.0),
            memory("M_v", 16, 16, 16, 4, 1, 4.0, 1.0),
            memory("M_eta", 16, 16, 1, 4, 1, 4.0, 1.0),
            memory("M_alpha", 16, 16, 1, 4, 1, 4.0, 1.0),
            memory("M_mem", 16, 16, 16, 8, 1, 4.0, 1.0),
        )
        cms_levels = (
            level("cms_l1", 16, 8, 16, 1_000, 4, 1, True, 1.0),
            level("cms_l2", 16, 8, 16, 5_000, 8, 1, True, 1.0),
            level("cms_l3", 16, 8, 16, 10_000, 16, 1, True, 1.0),
        )
        values: dict[str, object] = {
            "batch": 1,
            "query_tokens": 1,
            "context_tokens": 8,
            "model_dim": 16,
            "activation_bytes": 2,
            "weight_bytes": 2,
            "state_bytes": 2,
            "optimizer_bytes": 4,
            "memories": memories,
            "cms_levels": cms_levels,
            "position": 0,
            "adaptive_q": False,
            "schedule": "eager_gradient",
            "include_cms_updates": False,
            "chunk_cache_bytes_per_request": 0,
            "sigma_flops_per_hidden": 4,
            "residual_flops_per_output": 1,
            "compute_efficiency": 1.0,
            "bandwidth_efficiency": 1.0,
        }
        values.update(overrides)
        return analytical_model.HopeConfig(**values)

    @staticmethod
    def stage(stages, name: str):
        return next(stage for stage in stages if stage.stage == name)

    def test_decode_kv_reads_are_linear_in_context(self) -> None:
        short = PrimitiveInvariantTests.attention_config(context_tokens=8)
        long = PrimitiveInvariantTests.attention_config(context_tokens=16)
        short_flash = self.stage(
            build_attention_decode(short, self.hardware), "flash_decode_partials"
        )
        long_flash = self.stage(
            build_attention_decode(long, self.hardware), "flash_decode_partials"
        )
        self.assertEqual(long_flash.hbm_read_bytes, 2 * short_flash.hbm_read_bytes)

    def test_kv_append_is_context_independent(self) -> None:
        short = PrimitiveInvariantTests.attention_config(context_tokens=8)
        long = PrimitiveInvariantTests.attention_config(context_tokens=4_096)
        short_append = self.stage(
            build_attention_decode(short, self.hardware), "kv_cache_append"
        )
        long_append = self.stage(
            build_attention_decode(long, self.hardware), "kv_cache_append"
        )
        self.assertEqual(
            short_append.mandatory_write_bytes,
            long_append.mandatory_write_bytes,
        )

    def test_expert_weight_bytes_use_unique_active_experts(self) -> None:
        one_request = PrimitiveInvariantTests.attention_config(
            batch=1,
            active_unique_experts=2,
            fuse_expert_intermediates=True,
        )
        four_requests = PrimitiveInvariantTests.attention_config(
            batch=4,
            active_unique_experts=2,
            fuse_expert_intermediates=True,
        )
        one = self.stage(
            build_attention_decode(one_request, self.hardware),
            "expert_gate_projection",
        )
        four = self.stage(
            build_attention_decode(four_requests, self.hardware),
            "expert_gate_projection",
        )
        self.assertEqual(one.hbm_read_bytes, 2 * 16 * 32 * 2)
        self.assertEqual(four.hbm_read_bytes, one.hbm_read_bytes)

    def test_swiglu_stages_keep_each_gemms_weights_with_its_flops(self) -> None:
        cfg = replace(
            self.prefill,
            active_unique_experts=2,
            expert_activation_flops_per_element=4,
        )
        stages = build_attention_prefill(cfg, self.hardware)
        names = [stage.stage for stage in stages]
        expected = [
            "moe_dispatch",
            "expert_up_projection",
            "expert_gate_projection",
            "expert_swiglu_activation",
            "expert_down_projection",
            "moe_combine",
        ]
        first = names.index("moe_dispatch")
        self.assertEqual(names[first : first + len(expected)], expected)

        up = self.stage(stages, "expert_up_projection")
        gate = self.stage(stages, "expert_gate_projection")
        activation = self.stage(stages, "expert_swiglu_activation")
        down = self.stage(stages, "expert_down_projection")
        self.assertEqual(up.flops, 16_384)
        self.assertEqual(gate.flops, 16_384)
        self.assertEqual(down.flops, 16_384)
        self.assertEqual(activation.flops, 2_048)
        self.assertEqual(up.hbm_read_bytes, 2_560)
        self.assertEqual(gate.hbm_read_bytes, 2_560)
        self.assertEqual(down.hbm_read_bytes, 3_072)
        self.assertEqual(up.temporary_write_bytes, 1_024)
        self.assertEqual(gate.temporary_write_bytes, 1_024)
        self.assertEqual(activation.hbm_read_bytes, 2_048)
        self.assertEqual(activation.temporary_write_bytes, 1_024)

    def test_fused_swiglu_keeps_weights_but_removes_intermediate_traffic(self) -> None:
        cfg = PrimitiveInvariantTests.attention_config(
            active_unique_experts=2,
            expert_activation_flops_per_element=4,
            fuse_expert_intermediates=True,
        )
        stages = build_attention_prefill(cfg, self.hardware)
        up = self.stage(stages, "expert_up_projection")
        gate = self.stage(stages, "expert_gate_projection")
        activation = self.stage(stages, "expert_swiglu_activation")
        down = self.stage(stages, "expert_down_projection")
        combine = self.stage(stages, "moe_combine")

        self.assertEqual(up.hbm_read_bytes, 2_112)
        self.assertEqual(gate.hbm_read_bytes, 2_048)
        self.assertEqual(down.hbm_read_bytes, 2_048)
        self.assertEqual(up.hbm_write_bytes, 0)
        self.assertEqual(gate.hbm_write_bytes, 0)
        self.assertEqual(activation.hbm_read_bytes, 0)
        self.assertEqual(activation.hbm_write_bytes, 0)
        self.assertEqual(down.temporary_write_bytes, 64)
        self.assertEqual(combine.hbm_read_bytes, 64)

    def test_attention_flop_equations_match_design(self) -> None:
        stages = build_attention_prefill(self.prefill, self.hardware)
        self.assertEqual(self.stage(stages, "qkv_projection").flops, 8_192)
        self.assertEqual(self.stage(stages, "flash_attention").flops, 1_680)
        self.assertEqual(self.stage(stages, "output_projection").flops, 4_096)
        self.assertEqual(self.stage(stages, "moe_router").flops, 2_112)
        self.assertEqual(self.stage(stages, "expert_up_projection").flops, 16_384)
        self.assertEqual(self.stage(stages, "expert_gate_projection").flops, 16_384)
        self.assertEqual(self.stage(stages, "expert_down_projection").flops, 16_384)

    def test_attention_summary_reports_full_kv_residency(self) -> None:
        total = analytical_model.summarize(
            build_attention_prefill(self.prefill, self.hardware), self.hardware
        )
        self.assertEqual(total["persistent_state_bytes"], 256)

    def test_hope_state_is_context_independent(self) -> None:
        short = self.hope_config(context_tokens=8)
        long = self.hope_config(context_tokens=1_000_000)
        short_total = analytical_model.summarize(
            analytical_model.build_hope_decode(short, self.hardware), self.hardware
        )
        long_total = analytical_model.summarize(
            analytical_model.build_hope_decode(long, self.hardware), self.hardware
        )
        self.assertEqual(
            short_total["persistent_state_bytes"],
            long_total["persistent_state_bytes"],
        )

    def test_hope_state_is_linear_in_batch(self) -> None:
        one = self.hope_config(batch=1)
        two = self.hope_config(batch=2)
        one_total = analytical_model.summarize(
            analytical_model.build_hope_decode(one, self.hardware), self.hardware
        )
        two_total = analytical_model.summarize(
            analytical_model.build_hope_decode(two, self.hardware), self.hardware
        )
        self.assertEqual(
            two_total["persistent_state_bytes"],
            2 * one_total["persistent_state_bytes"],
        )
        self.assertEqual(one_total["persistent_state_bytes"], 17_088)

    def test_cms_forward_read_is_independent_of_update_cadence(self) -> None:
        base = self.hope_config()
        fast = replace(base.cms_levels[0], update_period=1)
        slow = replace(base.cms_levels[0], update_period=10_000)
        fast_cfg = replace(base, cms_levels=(fast, *base.cms_levels[1:]))
        slow_cfg = replace(base, cms_levels=(slow, *base.cms_levels[1:]))
        fast_stage = self.stage(
            analytical_model.build_hope_decode(fast_cfg, self.hardware),
            "cms_l1_forward",
        )
        slow_stage = self.stage(
            analytical_model.build_hope_decode(slow_cfg, self.hardware),
            "cms_l1_forward",
        )
        self.assertGreater(fast_stage.hbm_read_bytes, 0)
        self.assertEqual(fast_stage.hbm_read_bytes, slow_stage.hbm_read_bytes)

    def test_hope_decode_reports_normal_boundary_and_amortized_costs(self) -> None:
        cfg = self.hope_config()
        totals = {
            timing: analytical_model.summarize(
                analytical_model.build_hope_decode(
                    cfg, self.hardware, timing=timing
                ),
                self.hardware,
            )
            for timing in ("normal", "boundary", "amortized")
        }
        self.assertLess(
            totals["normal"]["stagewise_latency_seconds"],
            totals["boundary"]["stagewise_latency_seconds"],
        )
        self.assertGreaterEqual(
            totals["amortized"]["stagewise_latency_seconds"],
            totals["normal"]["stagewise_latency_seconds"],
        )
        self.assertLessEqual(
            totals["amortized"]["stagewise_latency_seconds"],
            totals["boundary"]["stagewise_latency_seconds"],
        )

    def test_shipped_default_has_no_adaptive_query_memory(self) -> None:
        cfg = self.hope_config()
        self.assertFalse(cfg.adaptive_q)
        stage_names = {
            stage.stage
            for stage in analytical_model.build_hope_decode(cfg, self.hardware)
        }
        self.assertFalse(any("M_q" in name for name in stage_names))

    def test_all_titans_work_precedes_cms(self) -> None:
        cases = {
            "prefill": analytical_model.build_hope_prefill(
                self.hope_config(query_tokens=4), self.hardware
            ),
            "decode_boundary": analytical_model.build_hope_decode(
                self.hope_config(), self.hardware, timing="boundary"
            ),
        }
        for name, stages in cases.items():
            with self.subTest(case=name):
                last_titans = max(
                    index
                    for index, stage in enumerate(stages)
                    if stage.component in {"titans", "titans_update"}
                )
                first_cms = min(
                    index
                    for index, stage in enumerate(stages)
                    if stage.component == "cms"
                )
                self.assertLess(last_titans, first_cms)

    def test_legacy_anchor(self) -> None:
        got = analytical_model.legacy_hope_proxy(
            d=2048, chunk=1, element_bytes=2
        )
        self.assertEqual(got["flops"], 1_082_187_776)
        self.assertEqual(got["hbm_bytes"], 679_510_016)

    def test_stagewise_not_below_aggregate(self) -> None:
        total = analytical_model.summarize(
            build_attention_prefill(self.prefill, self.hardware), self.hardware
        )
        self.assertGreaterEqual(
            total["stagewise_latency_seconds"],
            total["aggregate_latency_seconds"],
        )


class DataModelTests(unittest.TestCase):
    def test_stage_serialized_fields_match_design(self) -> None:
        self.assertEqual(
            [field.name for field in fields(analytical_model.Stage)],
            [
                "phase",
                "component",
                "stage",
                "executions",
                "flops",
                "hbm_read_bytes",
                "hbm_write_bytes",
                "mandatory_write_bytes",
                "temporary_write_bytes",
                "persistent_state_delta_bytes",
                "compute_efficiency",
                "bandwidth_efficiency",
                "notes",
            ],
        )

    def test_stage_computed_properties_include_executions_and_launches(self) -> None:
        hardware = Hardware("tiny", 1_000.0, 100.0, 1_000, 0.01)
        stage = analytical_model.Stage(
            "decode",
            "test",
            "example",
            2,
            100,
            30,
            20,
            20,
            0,
            0,
            1.0,
            1.0,
            "hand-derived test fixture",
            hardware,
        )
        self.assertEqual(stage.effective_hbm_bytes, 100)
        self.assertEqual(stage.arithmetic_intensity, 2.0)
        self.assertAlmostEqual(stage.compute_seconds, 0.2)
        self.assertAlmostEqual(stage.memory_seconds, 1.0)
        self.assertAlmostEqual(stage.roofline_seconds, 1.02)

    def test_all_public_records_are_frozen(self) -> None:
        records = (
            Hardware("tiny", 1.0, 1.0, 1),
            PrimitiveInvariantTests.attention_config(),
            BuilderAndSummaryTests.hope_config().memories[0],
            BuilderAndSummaryTests.hope_config().cms_levels[0],
            BuilderAndSummaryTests.hope_config(),
        )
        for record in records:
            with self.subTest(record=type(record).__name__):
                with self.assertRaises(FrozenInstanceError):
                    record.batch = 99

    def test_memory_primitive_equations_match_design(self) -> None:
        spec = analytical_model.MemorySpec("toy", 2, 3, 4, 8, 0, 4.0, 1.0)
        self.assertEqual(analytical_model.memory_parameter_count(spec), 18)
        self.assertEqual(
            analytical_model.memory_forward_flops(
                spec,
                tokens=5,
                sigma_flops_per_hidden=2,
                residual_flops_per_output=1,
            ),
            230,
        )
        self.assertEqual(
            analytical_model.memory_weight_backward_flops(spec, tokens=5),
            300,
        )

    def test_update_boundary_count_handles_offset(self) -> None:
        self.assertEqual(
            analytical_model.update_boundary_count(
                position=3, query_tokens=6, update_chunk=4
            ),
            2,
        )


class ValidationTests(unittest.TestCase):
    def test_rejects_nonpositive_hardware_rate(self) -> None:
        with self.assertRaises(ValueError):
            Hardware("bad", 0.0, 1.0, 1)

    def test_rejects_nonfinite_hardware_rate(self) -> None:
        for bad_rate in (float("nan"), float("inf")):
            with self.subTest(rate=bad_rate):
                with self.assertRaises(ValueError):
                    Hardware("bad", bad_rate, 1.0, 1)

    def test_rejects_efficiency_above_one(self) -> None:
        with self.assertRaises(ValueError):
            PrimitiveInvariantTests.attention_config(compute_efficiency=1.01)

    def test_rejects_kv_reload_multiplier_below_one(self) -> None:
        with self.assertRaises(ValueError):
            PrimitiveInvariantTests.attention_config(kv_reload_multiplier=0.999)

    def test_attention_decode_rejects_multi_token_invocation(self) -> None:
        cfg = PrimitiveInvariantTests.attention_config(query_tokens=2)
        hardware = Hardware("tiny", 1.0, 1.0, 1)
        with self.assertRaises(ValueError):
            build_attention_decode(cfg, hardware)

    def test_memory_spec_rejects_zero_dimension(self) -> None:
        with self.assertRaises(ValueError):
            analytical_model.MemorySpec("bad", 0, 1, 1, 1, 0, 1.0, 1.0)

    def test_cms_level_rejects_zero_period(self) -> None:
        with self.assertRaises(ValueError):
            analytical_model.CMSLevel("bad", 1, 1, 1, 0, 1, 0, True, 1.0)


class DeterministicRunnerTests(unittest.TestCase):
    output_dir = Path(__file__).resolve().parent
    runner = output_dir / "run.py"
    results_path = output_dir / "results.json"
    stdout_path = output_dir / "stdout.txt"

    @staticmethod
    def digest(path: Path) -> str:
        return hashlib.sha256(path.read_bytes()).hexdigest()

    def test_runner_decouples_cms_state_from_titans_momentum(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_cwd:
            subprocess.run(
                [sys.executable, str(self.runner)],
                cwd=temporary_cwd,
                check=True,
                capture_output=True,
                text=True,
            )
        payload = json.loads(self.results_path.read_text(encoding="utf-8"))
        paper = payload["hope_scenarios"]["paper_equation_lower_bound"]
        shipped = payload["hope_scenarios"]["shipped_credible_momentum"]
        paper_levels = paper["inputs"]["decode"]["cms_levels"]
        shipped_levels = shipped["inputs"]["decode"]["cms_levels"]
        self.assertEqual(
            [level["optimizer_slots"] for level in paper_levels],
            [0, 0, 0],
        )
        self.assertEqual(shipped_levels, paper_levels)

        def cms_state(scenario: dict[str, object]) -> list[int]:
            stages = scenario["decode"]["normal"]["stages"]
            return [
                stage["persistent_state_delta_bytes"]
                for stage in stages
                if stage["component"] == "cms"
            ]

        self.assertEqual(cms_state(shipped), cms_state(paper))

    def test_runner_outputs_are_complete_and_byte_stable(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_cwd:
            first = subprocess.run(
                [sys.executable, str(self.runner)],
                cwd=temporary_cwd,
                check=True,
                capture_output=True,
                text=True,
            )
            first_hashes = (
                self.digest(self.results_path),
                self.digest(self.stdout_path),
            )
            second = subprocess.run(
                [sys.executable, str(self.runner)],
                cwd=temporary_cwd,
                check=True,
                capture_output=True,
                text=True,
            )
            second_hashes = (
                self.digest(self.results_path),
                self.digest(self.stdout_path),
            )

        self.assertEqual(first_hashes, second_hashes)
        self.assertEqual(first.stdout, second.stdout)
        self.assertEqual(first.stdout, self.stdout_path.read_text(encoding="utf-8"))

        payload = json.loads(self.results_path.read_text(encoding="utf-8"))
        self.assertEqual(
            set(payload["hope_scenarios"]),
            {
                "paper_equation_lower_bound",
                "shipped_credible_momentum",
                "legacy_repo_proxy",
                "adaptive_q_hypothetical",
            },
        )
        self.assertNotIn("adaptive_q_hypothetical", payload["default_scenarios"])
        self.assertTrue(all(payload["qa_checks"].values()))
        self.assertIn("stages", payload["attention_moe"]["prefill"])
        self.assertIn(
            "stages",
            payload["hope_scenarios"]["shipped_credible_momentum"]
            ["decode"]["amortized"],
        )
        self.assertEqual(
            self.results_path.read_text(encoding="utf-8"),
            json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True)
            + "\n",
        )


if __name__ == "__main__":
    unittest.main()
