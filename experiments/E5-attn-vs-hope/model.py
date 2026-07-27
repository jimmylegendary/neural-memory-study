#!/usr/bin/env python3
"""Auditable analytical costs for one Attention+MoE or HOPE block.

One multiply-accumulate is two FLOPs.  Byte counts in this module describe
traffic crossing the HBM boundary; on-chip traffic is intentionally absent.
"""

from __future__ import annotations

from dataclasses import InitVar, dataclass
from math import floor, inf, isfinite


def _require_positive(name: str, value: int | float) -> None:
    if isinstance(value, bool) or not isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and positive")


def _require_nonnegative(name: str, value: int | float) -> None:
    if isinstance(value, bool) or not isfinite(value) or value < 0:
        raise ValueError(f"{name} must be finite and nonnegative")


def _require_efficiency(name: str, value: float) -> None:
    if not 0 < value <= 1:
        raise ValueError(f"{name} must be in (0, 1]")


@dataclass(frozen=True)
class Hardware:
    """Single-device hardware anchor used by the stage roofline."""

    name: str
    peak_flops_per_second: float
    hbm_bandwidth_bytes_per_second: float
    hbm_capacity_bytes: int
    kernel_launch_overhead_seconds: float = 0.0

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("name must not be empty")
        _require_positive("peak_flops_per_second", self.peak_flops_per_second)
        _require_positive(
            "hbm_bandwidth_bytes_per_second", self.hbm_bandwidth_bytes_per_second
        )
        _require_positive("hbm_capacity_bytes", self.hbm_capacity_bytes)
        _require_nonnegative(
            "kernel_launch_overhead_seconds", self.kernel_launch_overhead_seconds
        )


@dataclass(frozen=True)
class AttentionMoEConfig:
    """Inputs for one full-attention block followed by a top-k MoE."""

    batch: int
    query_tokens: int
    context_tokens: int
    model_dim: int
    query_heads: int
    kv_heads: int
    head_dim: int
    activation_bytes: int
    weight_bytes: int
    kv_bytes: int
    partial_bytes: int
    lse_bytes: int
    decode_splits: int
    kv_reload_multiplier: float
    softmax_flops_per_pair: int
    experts: int
    top_k: int
    expert_hidden_dim: int
    expert_matrices: int
    router_flops_per_token_expert: int
    active_unique_experts: float | None
    expert_hbm_fraction: float
    compute_efficiency: float
    bandwidth_efficiency: float

    def __post_init__(self) -> None:
        positive_dimensions = (
            "batch",
            "query_tokens",
            "context_tokens",
            "model_dim",
            "query_heads",
            "kv_heads",
            "head_dim",
            "activation_bytes",
            "weight_bytes",
            "kv_bytes",
            "partial_bytes",
            "lse_bytes",
            "decode_splits",
            "experts",
            "top_k",
            "expert_hidden_dim",
            "expert_matrices",
        )
        for name in positive_dimensions:
            _require_positive(name, getattr(self, name))
        _require_positive("kv_reload_multiplier", self.kv_reload_multiplier)
        _require_nonnegative(
            "softmax_flops_per_pair", self.softmax_flops_per_pair
        )
        _require_nonnegative(
            "router_flops_per_token_expert",
            self.router_flops_per_token_expert,
        )
        if self.top_k > self.experts:
            raise ValueError("top_k must not exceed experts")
        if self.active_unique_experts is not None:
            _require_positive("active_unique_experts", self.active_unique_experts)
            if self.active_unique_experts > self.experts:
                raise ValueError("active_unique_experts must not exceed experts")
        if not 0 <= self.expert_hbm_fraction <= 1:
            raise ValueError("expert_hbm_fraction must be in [0, 1]")
        _require_efficiency("compute_efficiency", self.compute_efficiency)
        _require_efficiency("bandwidth_efficiency", self.bandwidth_efficiency)

    @property
    def query_width(self) -> int:
        return self.query_heads * self.head_dim

    @property
    def kv_width(self) -> int:
        return self.kv_heads * self.head_dim

    @property
    def processed_tokens(self) -> int:
        return self.batch * self.query_tokens


@dataclass(frozen=True)
class MemorySpec:
    """One request-local self-modifying Titans memory."""

    name: str
    input_dim: int
    hidden_dim: int
    output_dim: int
    update_chunk: int
    momentum_slots: int
    update_multiplier: float
    state_weight_read_count: float

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("name must not be empty")
        for name in ("input_dim", "hidden_dim", "output_dim", "update_chunk"):
            _require_positive(name, getattr(self, name))
        _require_nonnegative("momentum_slots", self.momentum_slots)
        _require_positive("update_multiplier", self.update_multiplier)
        _require_nonnegative(
            "state_weight_read_count", self.state_weight_read_count
        )


@dataclass(frozen=True)
class CMSLevel:
    """One sequential Continuum Memory System level."""

    name: str
    input_dim: int
    hidden_dim: int
    output_dim: int
    update_period: int
    bptt_span: int
    optimizer_slots: int
    personalized: bool
    gradient_multiplier: float

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("name must not be empty")
        for name in (
            "input_dim",
            "hidden_dim",
            "output_dim",
            "update_period",
            "bptt_span",
        ):
            _require_positive(name, getattr(self, name))
        _require_nonnegative("optimizer_slots", self.optimizer_slots)
        if not isinstance(self.personalized, bool):
            raise TypeError("personalized must be bool")
        _require_positive("gradient_multiplier", self.gradient_multiplier)


@dataclass(frozen=True)
class HopeConfig:
    """Inputs for self-modifying Titans followed by sequential CMS."""

    batch: int
    query_tokens: int
    context_tokens: int
    model_dim: int
    activation_bytes: int
    weight_bytes: int
    state_bytes: int
    optimizer_bytes: int
    memories: tuple[MemorySpec, ...]
    cms_levels: tuple[CMSLevel, ...]
    position: int = 0
    adaptive_q: bool = False
    schedule: str = "eager_gradient"
    include_cms_updates: bool = False
    chunk_cache_bytes_per_request: int = 0
    sigma_flops_per_hidden: int = 0
    residual_flops_per_output: int = 0
    compute_efficiency: float = 1.0
    bandwidth_efficiency: float = 1.0

    def __post_init__(self) -> None:
        for name in (
            "batch",
            "query_tokens",
            "context_tokens",
            "model_dim",
            "activation_bytes",
            "weight_bytes",
            "state_bytes",
            "optimizer_bytes",
        ):
            _require_positive(name, getattr(self, name))
        _require_nonnegative("position", self.position)
        _require_nonnegative(
            "chunk_cache_bytes_per_request", self.chunk_cache_bytes_per_request
        )
        _require_nonnegative(
            "sigma_flops_per_hidden", self.sigma_flops_per_hidden
        )
        _require_nonnegative(
            "residual_flops_per_output", self.residual_flops_per_output
        )
        _require_efficiency("compute_efficiency", self.compute_efficiency)
        _require_efficiency("bandwidth_efficiency", self.bandwidth_efficiency)
        if not self.memories:
            raise ValueError("memories must not be empty")
        if not self.cms_levels:
            raise ValueError("cms_levels must not be empty")
        if not all(isinstance(memory, MemorySpec) for memory in self.memories):
            raise TypeError("memories must contain MemorySpec values")
        if not all(isinstance(level, CMSLevel) for level in self.cms_levels):
            raise TypeError("cms_levels must contain CMSLevel values")
        if len({memory.name for memory in self.memories}) != len(self.memories):
            raise ValueError("memory names must be unique")
        if len({level.name for level in self.cms_levels}) != len(self.cms_levels):
            raise ValueError("CMS level names must be unique")
        if self.schedule not in {"eager_gradient", "deferred_replay"}:
            raise ValueError("schedule must be eager_gradient or deferred_replay")
        if not isinstance(self.adaptive_q, bool):
            raise TypeError("adaptive_q must be bool")
        if not isinstance(self.include_cms_updates, bool):
            raise TypeError("include_cms_updates must be bool")
        has_m_q = any(memory.name == "M_q" for memory in self.memories)
        if has_m_q != self.adaptive_q:
            raise ValueError(
                "M_q must be present exactly when adaptive_q is enabled"
            )

    @property
    def processed_tokens(self) -> int:
        return self.batch * self.query_tokens


def memory_parameter_count(spec: MemorySpec) -> int:
    """Return ``I*H + H*O`` parameters for a two-layer memory."""

    return (
        spec.input_dim * spec.hidden_dim
        + spec.hidden_dim * spec.output_dim
    )


def memory_forward_flops(
    spec: MemorySpec,
    tokens: int,
    sigma_flops_per_hidden: int = 0,
    residual_flops_per_output: int = 0,
) -> int:
    """Return the design-section 6.2 forward FLOPs for ``tokens`` rows."""

    _require_positive("tokens", tokens)
    _require_nonnegative("sigma_flops_per_hidden", sigma_flops_per_hidden)
    _require_nonnegative(
        "residual_flops_per_output", residual_flops_per_output
    )
    return (
        2 * tokens * memory_parameter_count(spec)
        + tokens
        * (
            sigma_flops_per_hidden * spec.hidden_dim
            + residual_flops_per_output * spec.output_dim
        )
    )


def memory_weight_backward_flops(spec: MemorySpec, tokens: int) -> int:
    """Return weight-gradient backward FLOPs without an input gradient."""

    _require_positive("tokens", tokens)
    return (
        2 * tokens * spec.input_dim * spec.hidden_dim
        + 4 * tokens * spec.hidden_dim * spec.output_dim
    )


def update_boundary_count(position: int, query_tokens: int, update_chunk: int) -> int:
    """Count update boundaries crossed by an invocation at ``position``."""

    _require_nonnegative("position", position)
    _require_positive("query_tokens", query_tokens)
    _require_positive("update_chunk", update_chunk)
    return floor((position + query_tokens) / update_chunk) - floor(
        position / update_chunk
    )


@dataclass(frozen=True)
class Stage:
    """One sequential analytical stage, bound to a hardware roofline.

    ``hardware`` is an ``InitVar`` rather than a serialized dataclass field, so
    ``dataclasses.fields(Stage)`` is exactly the section 3.1 stage schema.
    """

    phase: str
    component: str
    stage: str
    executions: int | float
    flops: int | float
    hbm_read_bytes: int | float
    hbm_write_bytes: int | float
    mandatory_write_bytes: int | float
    temporary_write_bytes: int | float
    persistent_state_delta_bytes: int | float
    compute_efficiency: float
    bandwidth_efficiency: float
    notes: str
    hardware: InitVar[Hardware]

    def __post_init__(self, hardware: Hardware) -> None:
        if not isinstance(hardware, Hardware):
            raise TypeError("hardware must be a Hardware instance")
        for name in ("phase", "component", "stage"):
            if not getattr(self, name).strip():
                raise ValueError(f"{name} must not be empty")
        _require_positive("executions", self.executions)
        for name in (
            "flops",
            "hbm_read_bytes",
            "hbm_write_bytes",
            "mandatory_write_bytes",
            "temporary_write_bytes",
            "persistent_state_delta_bytes",
        ):
            _require_nonnegative(name, getattr(self, name))
        if (
            self.mandatory_write_bytes + self.temporary_write_bytes
            != self.hbm_write_bytes
        ):
            raise ValueError(
                "mandatory_write_bytes + temporary_write_bytes must equal "
                "hbm_write_bytes"
            )
        _require_efficiency("compute_efficiency", self.compute_efficiency)
        _require_efficiency("bandwidth_efficiency", self.bandwidth_efficiency)
        object.__setattr__(self, "_hardware", hardware)

    @property
    def effective_hbm_bytes(self) -> int | float:
        return self.executions * (self.hbm_read_bytes + self.hbm_write_bytes)

    @property
    def arithmetic_intensity(self) -> float:
        if self.effective_hbm_bytes == 0:
            return inf
        return self.executions * self.flops / self.effective_hbm_bytes

    @property
    def compute_seconds(self) -> float:
        return (
            self.executions
            * self.flops
            / (self.compute_efficiency * self._hardware.peak_flops_per_second)
        )

    @property
    def memory_seconds(self) -> float:
        return self.effective_hbm_bytes / (
            self.bandwidth_efficiency
            * self._hardware.hbm_bandwidth_bytes_per_second
        )

    @property
    def roofline_seconds(self) -> float:
        return max(self.compute_seconds, self.memory_seconds) + (
            self.executions * self._hardware.kernel_launch_overhead_seconds
        )


def causal_pair_count(batch: int, heads: int, query_tokens: int) -> int:
    """Return ``B * H * Q * (Q + 1) / 2`` causal self-attention pairs."""

    for name, value in (
        ("batch", batch),
        ("heads", heads),
        ("query_tokens", query_tokens),
    ):
        _require_positive(name, value)
    return batch * heads * query_tokens * (query_tokens + 1) // 2


def decode_partial_bytes(config: AttentionMoEConfig) -> int:
    """Flash-Decode partial-output plus LSE bytes, or zero for one split."""

    if config.decode_splits == 1:
        return 0
    return (
        config.batch
        * config.query_tokens
        * config.query_heads
        * config.decode_splits
        * (config.head_dim * config.partial_bytes + config.lse_bytes)
    )


def _stage(
    hardware: Hardware,
    config: AttentionMoEConfig,
    *,
    phase: str,
    component: str,
    stage: str,
    flops: int | float = 0,
    hbm_read_bytes: int | float = 0,
    mandatory_write_bytes: int | float = 0,
    temporary_write_bytes: int | float = 0,
    persistent_state_delta_bytes: int | float = 0,
    notes: str,
) -> Stage:
    return Stage(
        phase=phase,
        component=component,
        stage=stage,
        executions=1,
        flops=flops,
        hbm_read_bytes=hbm_read_bytes,
        hbm_write_bytes=mandatory_write_bytes + temporary_write_bytes,
        mandatory_write_bytes=mandatory_write_bytes,
        temporary_write_bytes=temporary_write_bytes,
        persistent_state_delta_bytes=persistent_state_delta_bytes,
        compute_efficiency=config.compute_efficiency,
        bandwidth_efficiency=config.bandwidth_efficiency,
        notes=notes,
        hardware=hardware,
    )


def _attention_prefix(
    config: AttentionMoEConfig, hardware: Hardware, phase: str
) -> list[Stage]:
    n = config.processed_tokens
    d_q = config.query_width
    d_kv = config.kv_width
    q_bytes = n * d_q * config.activation_bytes
    kv_append_bytes = 2 * n * d_kv * config.kv_bytes
    kv_resident_bytes = (
        2
        * config.batch
        * config.context_tokens
        * d_kv
        * config.kv_bytes
    )
    projection_weights = config.model_dim * (d_q + 2 * d_kv) * config.weight_bytes
    return [
        _stage(
            hardware,
            config,
            phase=phase,
            component="attention",
            stage="qkv_projection",
            flops=2 * n * config.model_dim * (d_q + 2 * d_kv),
            hbm_read_bytes=n * config.model_dim * config.activation_bytes
            + projection_weights,
            temporary_write_bytes=q_bytes,
            notes="Q materialization is explicit; K/V append is the following stage.",
        ),
        _stage(
            hardware,
            config,
            phase=phase,
            component="attention",
            stage="kv_cache_append",
            mandatory_write_bytes=kv_append_bytes,
            persistent_state_delta_bytes=kv_resident_bytes,
            notes=(
                "Writes only this invocation's K/V append; the persistent-state "
                "column records full current KV residency."
            ),
        ),
        _stage(
            hardware,
            config,
            phase=phase,
            component="attention",
            stage="q_materialization_read",
            hbm_read_bytes=q_bytes,
            notes="Explicit fusible Q read into the FlashAttention kernel.",
        ),
    ]


def _active_expert_count(config: AttentionMoEConfig) -> float:
    if config.active_unique_experts is not None:
        return config.active_unique_experts
    routes = config.processed_tokens * config.top_k
    return config.experts * (1 - (1 - 1 / config.experts) ** routes)


def _attention_suffix(
    config: AttentionMoEConfig, hardware: Hardware, phase: str
) -> list[Stage]:
    n = config.processed_tokens
    output_input_bytes = n * config.query_width * config.activation_bytes
    model_activation_bytes = n * config.model_dim * config.activation_bytes
    routed_activation_bytes = (
        n * config.top_k * config.model_dim * config.activation_bytes
    )
    routing_metadata_bytes = n * config.top_k * 8
    active_experts = _active_expert_count(config)
    expert_weight_bytes = (
        active_experts
        * config.expert_matrices
        * config.model_dim
        * config.expert_hidden_dim
        * config.weight_bytes
        * config.expert_hbm_fraction
    )
    return [
        _stage(
            hardware,
            config,
            phase=phase,
            component="attention",
            stage="output_projection",
            flops=2 * n * config.query_width * config.model_dim,
            hbm_read_bytes=output_input_bytes
            + config.query_width * config.model_dim * config.weight_bytes,
            temporary_write_bytes=model_activation_bytes,
            notes="Projects the attention result back to model width.",
        ),
        _stage(
            hardware,
            config,
            phase=phase,
            component="moe",
            stage="moe_router",
            flops=(2 * n * config.model_dim * config.experts)
            + (
                config.router_flops_per_token_expert
                * n
                * config.experts
            ),
            hbm_read_bytes=model_activation_bytes
            + config.model_dim * config.experts * config.weight_bytes,
            temporary_write_bytes=routing_metadata_bytes,
            notes="Router GEMM plus configured per-token/expert selection cost.",
        ),
        _stage(
            hardware,
            config,
            phase=phase,
            component="moe",
            stage="moe_dispatch",
            hbm_read_bytes=model_activation_bytes + routing_metadata_bytes,
            temporary_write_bytes=routed_activation_bytes,
            notes=(
                "Materialized top-k dispatch; fusion remains visible as "
                "temporary traffic."
            ),
        ),
        _stage(
            hardware,
            config,
            phase=phase,
            component="moe",
            stage="expert_weight_read",
            hbm_read_bytes=expert_weight_bytes,
            notes=(
                f"Reads {active_experts:.6g} unique active experts after the "
                "configured HBM-residency fraction."
            ),
        ),
        _stage(
            hardware,
            config,
            phase=phase,
            component="moe",
            stage="expert_compute",
            flops=(
                2
                * n
                * config.top_k
                * config.expert_matrices
                * config.model_dim
                * config.expert_hidden_dim
            ),
            hbm_read_bytes=routed_activation_bytes,
            temporary_write_bytes=routed_activation_bytes,
            notes="Top-k expert matrices; one MAC is two FLOPs.",
        ),
        _stage(
            hardware,
            config,
            phase=phase,
            component="moe",
            stage="moe_combine",
            hbm_read_bytes=routed_activation_bytes,
            mandatory_write_bytes=model_activation_bytes,
            notes="Combines routed expert results and writes the block output.",
        ),
    ]


def build_attention_prefill(
    config: AttentionMoEConfig, hardware: Hardware
) -> list[Stage]:
    """Build the auditable FlashAttention portion of one causal prefill chunk."""

    stages = _attention_prefix(config, hardware, "prefill")
    pairs = causal_pair_count(
        config.batch, config.query_heads, config.query_tokens
    )
    kv_reads = (
        2
        * config.batch
        * config.context_tokens
        * config.kv_width
        * config.kv_bytes
        * config.kv_reload_multiplier
    )
    stages.append(
        _stage(
            hardware,
            config,
            phase="prefill",
            component="attention",
            stage="flash_attention",
            flops=(4 * pairs * config.head_dim)
            + (config.softmax_flops_per_pair * pairs),
            hbm_read_bytes=kv_reads,
            mandatory_write_bytes=(
                config.processed_tokens
                * config.query_width
                * config.activation_bytes
            ),
            notes=(
                "Causal FlashAttention; score/probability matrices are never "
                "materialized in HBM."
            ),
        )
    )
    stages.extend(_attention_suffix(config, hardware, "prefill"))
    return stages


def build_attention_decode(
    config: AttentionMoEConfig, hardware: Hardware
) -> list[Stage]:
    """Build the auditable Flash-Decode portion of one next-token invocation."""

    stages = _attention_prefix(config, hardware, "decode")
    pairs = (
        config.batch
        * config.query_heads
        * config.query_tokens
        * config.context_tokens
    )
    kv_reads = (
        2
        * config.batch
        * config.context_tokens
        * config.kv_width
        * config.kv_bytes
        * config.kv_reload_multiplier
    )
    partials = decode_partial_bytes(config)
    final_output = (
        config.processed_tokens * config.query_width * config.activation_bytes
    )
    stages.append(
        _stage(
            hardware,
            config,
            phase="decode",
            component="attention",
            stage="flash_decode_partials",
            flops=(4 * pairs * config.head_dim)
            + (config.softmax_flops_per_pair * pairs),
            hbm_read_bytes=kv_reads,
            mandatory_write_bytes=final_output if config.decode_splits == 1 else 0,
            temporary_write_bytes=partials,
            notes=(
                "KV traffic is linear in current context; split partials exist "
                "only when decode_splits > 1."
            ),
        )
    )
    if partials:
        stages.append(
            _stage(
                hardware,
                config,
                phase="decode",
                component="attention",
                stage="flash_decode_reduce",
                hbm_read_bytes=partials,
                mandatory_write_bytes=final_output,
                notes="Reduces split-local output/LSE records to the final output.",
            )
        )
    stages.extend(_attention_suffix(config, hardware, "decode"))
    return stages


def _hope_stage(
    hardware: Hardware,
    config: HopeConfig,
    *,
    phase: str,
    component: str,
    stage: str,
    executions: int | float = 1,
    flops: int | float = 0,
    hbm_read_bytes: int | float = 0,
    mandatory_write_bytes: int | float = 0,
    temporary_write_bytes: int | float = 0,
    persistent_state_delta_bytes: int | float = 0,
    notes: str,
) -> Stage:
    return Stage(
        phase=phase,
        component=component,
        stage=stage,
        executions=executions,
        flops=flops,
        hbm_read_bytes=hbm_read_bytes,
        hbm_write_bytes=mandatory_write_bytes + temporary_write_bytes,
        mandatory_write_bytes=mandatory_write_bytes,
        temporary_write_bytes=temporary_write_bytes,
        persistent_state_delta_bytes=persistent_state_delta_bytes,
        compute_efficiency=config.compute_efficiency,
        bandwidth_efficiency=config.bandwidth_efficiency,
        notes=notes,
        hardware=hardware,
    )


def _hope_forward_stages(
    config: HopeConfig, hardware: Hardware, phase: str
) -> list[Stage]:
    n = config.processed_tokens
    model_activation = n * config.model_dim * config.activation_bytes
    stages = [
        _hope_stage(
            hardware,
            config,
            phase=phase,
            component="titans",
            stage="static_q_projection",
            flops=2 * n * config.model_dim * config.model_dim,
            hbm_read_bytes=model_activation
            + config.model_dim * config.model_dim * config.weight_bytes,
            temporary_write_bytes=model_activation,
            persistent_state_delta_bytes=(
                config.batch * config.chunk_cache_bytes_per_request
            ),
            notes="Shipped HOPE uses static q=xW_q; W_q is not mutable state.",
        )
    ]
    for memory in config.memories:
        params = memory_parameter_count(memory)
        input_bytes = n * memory.input_dim * config.activation_bytes
        output_bytes = n * memory.output_dim * config.activation_bytes
        stages.append(
            _hope_stage(
                hardware,
                config,
                phase=phase,
                component="titans",
                stage=f"{memory.name}_forward",
                flops=memory_forward_flops(
                    memory,
                    n,
                    config.sigma_flops_per_hidden,
                    config.residual_flops_per_output,
                ),
                hbm_read_bytes=input_bytes
                + (
                    config.batch
                    * params
                    * config.state_bytes
                    * memory.state_weight_read_count
                ),
                temporary_write_bytes=output_bytes,
                persistent_state_delta_bytes=(
                    config.batch
                    * params
                    * (
                        config.state_bytes
                        + memory.momentum_slots * config.optimizer_bytes
                    )
                ),
                notes=(
                    "Request-local mutable memory forward; state_weight_read_count "
                    "controls HBM reloads independently of state apply."
                ),
            )
        )

    for level in config.cms_levels:
        params = (
            level.input_dim * level.hidden_dim
            + level.hidden_dim * level.output_dim
        )
        sharing = config.batch if level.personalized else 1
        stages.append(
            _hope_stage(
                hardware,
                config,
                phase=phase,
                component="cms",
                stage=f"{level.name}_forward",
                flops=2 * n * params,
                hbm_read_bytes=(
                    sharing * params * config.state_bytes
                    + n * level.input_dim * config.activation_bytes
                ),
                temporary_write_bytes=(
                    n * level.output_dim * config.activation_bytes
                ),
                persistent_state_delta_bytes=(
                    sharing
                    * params
                    * (
                        config.state_bytes
                        + level.optimizer_slots * config.optimizer_bytes
                    )
                ),
                notes=(
                    f"read_period=1 token; update_period={level.update_period} "
                    f"tokens; bptt_span={level.bptt_span}."
                ),
            )
        )
    return stages


def _memory_update_compute_stages(
    config: HopeConfig,
    hardware: Hardware,
    phase: str,
    memory: MemorySpec,
    *,
    tokens: int,
    executions: int | float,
) -> list[Stage]:
    params = memory_parameter_count(memory)
    forward_flops = memory_forward_flops(
        memory,
        tokens,
        config.sigma_flops_per_hidden,
        config.residual_flops_per_output,
    )
    backward_flops = memory_weight_backward_flops(memory, tokens)
    explicit_flops = (2 * forward_flops) + backward_flops
    calibrated_flops = memory.update_multiplier * forward_flops
    dgd_extra_flops = calibrated_flops - explicit_flops
    if dgd_extra_flops < 0:
        raise ValueError(
            f"{memory.name} update_multiplier is too small for the explicit "
            "target + prediction + weight-backward decomposition"
        )
    input_bytes = tokens * memory.input_dim * config.activation_bytes
    output_bytes = tokens * memory.output_dim * config.activation_bytes
    gradient_bytes = config.batch * params * config.optimizer_bytes
    common = {
        "hardware": hardware,
        "config": config,
        "phase": phase,
        "component": "titans_update",
        "executions": executions,
    }
    return [
        _hope_stage(
            **common,
            stage=f"{memory.name}_target_forward",
            flops=forward_flops,
            hbm_read_bytes=input_bytes
            + (
                config.batch
                * params
                * config.state_bytes
                * memory.state_weight_read_count
            ),
            temporary_write_bytes=output_bytes,
            notes="Explicit self-target forward term in the update decomposition.",
        ),
        _hope_stage(
            **common,
            stage=f"{memory.name}_prediction_forward",
            flops=forward_flops,
            hbm_read_bytes=input_bytes,
            temporary_write_bytes=output_bytes,
            notes="Explicit prediction forward term in the update decomposition.",
        ),
        _hope_stage(
            **common,
            stage=f"{memory.name}_weight_backward",
            flops=backward_flops,
            hbm_read_bytes=(
                tokens
                * (memory.input_dim + memory.hidden_dim + memory.output_dim)
                * config.activation_bytes
            ),
            temporary_write_bytes=gradient_bytes,
            notes="Weight-gradient backward only; no input-gradient FLOPs.",
        ),
        _hope_stage(
            **common,
            stage=f"{memory.name}_dgd_extra",
            flops=dgd_extra_flops,
            hbm_read_bytes=gradient_bytes,
            temporary_write_bytes=gradient_bytes,
            notes=(
                f"Residual DGD term calibrating the decomposition to "
                f"mu={memory.update_multiplier:g} times forward."
            ),
        ),
    ]


def _memory_apply_stage(
    config: HopeConfig,
    hardware: Hardware,
    phase: str,
    memory: MemorySpec,
    executions: int | float,
) -> Stage:
    params = memory_parameter_count(memory)
    apply_bytes = (
        config.batch
        * (1 + memory.momentum_slots)
        * params
        * config.state_bytes
    )
    return _hope_stage(
        hardware,
        config,
        phase=phase,
        component="titans_update",
        stage=f"{memory.name}_state_apply",
        executions=executions,
        hbm_read_bytes=apply_bytes,
        mandatory_write_bytes=apply_bytes,
        notes=(
            f"Request-local state RMW at period {memory.update_chunk}; "
            "optimizer buffers use the design apply-byte convention."
        ),
    )


def _cms_update_stages(
    config: HopeConfig,
    hardware: Hardware,
    phase: str,
    level: CMSLevel,
    executions: int | float,
) -> list[Stage]:
    params = (
        level.input_dim * level.hidden_dim
        + level.hidden_dim * level.output_dim
    )
    sharing = config.batch if level.personalized else 1
    gradient_bytes = sharing * params * config.optimizer_bytes
    apply_bytes = (
        sharing
        * (1 + level.optimizer_slots)
        * params
        * config.state_bytes
    )
    return [
        _hope_stage(
            hardware,
            config,
            phase=phase,
            component="cms_update",
            stage=f"{level.name}_gradient",
            executions=executions,
            flops=(
                level.gradient_multiplier
                * 2
                * config.batch
                * level.bptt_span
                * params
            ),
            hbm_read_bytes=(
                config.batch
                * level.bptt_span
                * level.input_dim
                * config.activation_bytes
            ),
            temporary_write_bytes=gradient_bytes,
            notes=(
                f"CMS gradient event; update_period={level.update_period}, "
                f"BPTT span={level.bptt_span}."
            ),
        ),
        _hope_stage(
            hardware,
            config,
            phase=phase,
            component="cms_update",
            stage=f"{level.name}_state_apply",
            executions=executions,
            hbm_read_bytes=apply_bytes,
            mandatory_write_bytes=apply_bytes,
            notes="CMS update-event state RMW; read cadence remains one token.",
        ),
    ]


def build_hope_prefill(config: HopeConfig, hardware: Hardware) -> list[Stage]:
    """Build one HOPE prefill chunk with explicit update-boundary events."""

    phase = "prefill"
    stages = _hope_forward_stages(config, hardware, phase)
    for memory in config.memories:
        boundaries = update_boundary_count(
            config.position, config.query_tokens, memory.update_chunk
        )
        if config.schedule == "eager_gradient":
            stages.extend(
                _memory_update_compute_stages(
                    config,
                    hardware,
                    phase,
                    memory,
                    tokens=config.processed_tokens,
                    executions=1,
                )
            )
        elif boundaries:
            stages.extend(
                _memory_update_compute_stages(
                    config,
                    hardware,
                    phase,
                    memory,
                    tokens=config.batch * memory.update_chunk,
                    executions=boundaries,
                )
            )
        if boundaries:
            stages.append(
                _memory_apply_stage(
                    config, hardware, phase, memory, boundaries
                )
            )
    if config.include_cms_updates:
        for level in config.cms_levels:
            boundaries = update_boundary_count(
                config.position, config.query_tokens, level.update_period
            )
            if boundaries:
                stages.extend(
                    _cms_update_stages(
                        config, hardware, phase, level, boundaries
                    )
                )
    return stages


def build_hope_decode(
    config: HopeConfig,
    hardware: Hardware,
    *,
    timing: str = "amortized",
) -> list[Stage]:
    """Build normal, simultaneous-boundary, or amortized HOPE decode costs."""

    if config.query_tokens != 1:
        raise ValueError("HOPE decode models exactly one next-token invocation")
    if timing not in {"normal", "boundary", "amortized"}:
        raise ValueError("timing must be normal, boundary, or amortized")
    phase = "decode"
    stages = _hope_forward_stages(config, hardware, phase)
    for memory in config.memories:
        if config.schedule == "eager_gradient":
            stages.extend(
                _memory_update_compute_stages(
                    config,
                    hardware,
                    phase,
                    memory,
                    tokens=config.processed_tokens,
                    executions=1,
                )
            )
        elif timing != "normal":
            stages.extend(
                _memory_update_compute_stages(
                    config,
                    hardware,
                    phase,
                    memory,
                    tokens=config.batch * memory.update_chunk,
                    executions=(
                        1 if timing == "boundary" else 1 / memory.update_chunk
                    ),
                )
            )
        if timing != "normal":
            stages.append(
                _memory_apply_stage(
                    config,
                    hardware,
                    phase,
                    memory,
                    1 if timing == "boundary" else 1 / memory.update_chunk,
                )
            )
    if config.include_cms_updates and timing != "normal":
        for level in config.cms_levels:
            stages.extend(
                _cms_update_stages(
                    config,
                    hardware,
                    phase,
                    level,
                    1 if timing == "boundary" else 1 / level.update_period,
                )
            )
    return stages


def summarize(stages: list[Stage], hardware: Hardware) -> dict[str, object]:
    """Aggregate ordered stages into JSON-serializable roofline totals."""

    if not stages:
        raise ValueError("stages must not be empty")
    if not all(isinstance(stage, Stage) for stage in stages):
        raise TypeError("stages must contain Stage values")
    if any(stage._hardware != hardware for stage in stages):
        raise ValueError("all stages must be bound to the supplied hardware")
    flops = sum(stage.executions * stage.flops for stage in stages)
    hbm_read_bytes = sum(
        stage.executions * stage.hbm_read_bytes for stage in stages
    )
    hbm_write_bytes = sum(
        stage.executions * stage.hbm_write_bytes for stage in stages
    )
    mandatory_write_bytes = sum(
        stage.executions * stage.mandatory_write_bytes for stage in stages
    )
    temporary_write_bytes = sum(
        stage.executions * stage.temporary_write_bytes for stage in stages
    )
    effective_hbm_bytes = hbm_read_bytes + hbm_write_bytes
    aggregate_compute = sum(stage.compute_seconds for stage in stages)
    aggregate_memory = sum(stage.memory_seconds for stage in stages)
    aggregate_latency = max(aggregate_compute, aggregate_memory)
    stagewise_latency = sum(stage.roofline_seconds for stage in stages)
    launch_seconds = sum(
        stage.executions * hardware.kernel_launch_overhead_seconds
        for stage in stages
    )
    if aggregate_compute > aggregate_memory:
        bound = "compute"
    elif aggregate_memory > aggregate_compute:
        bound = "memory"
    else:
        bound = "balanced"
    return {
        "stage_count": len(stages),
        "flops": flops,
        "hbm_read_bytes": hbm_read_bytes,
        "hbm_write_bytes": hbm_write_bytes,
        "effective_hbm_bytes": effective_hbm_bytes,
        "mandatory_write_bytes": mandatory_write_bytes,
        "temporary_write_bytes": temporary_write_bytes,
        "persistent_state_bytes": sum(
            stage.persistent_state_delta_bytes for stage in stages
        ),
        "arithmetic_intensity": (
            flops / effective_hbm_bytes if effective_hbm_bytes else None
        ),
        "aggregate_compute_seconds": aggregate_compute,
        "aggregate_memory_seconds": aggregate_memory,
        "aggregate_latency_seconds": aggregate_latency,
        "stagewise_latency_seconds": stagewise_latency,
        "launch_seconds": launch_seconds,
        "bound": bound,
    }


def legacy_hope_proxy(
    d: int, chunk: int, element_bytes: int
) -> dict[str, int]:
    """Reproduce the closed form implemented by ``multiarch/hope_block_dag.py``."""

    _require_positive("d", d)
    _require_positive("chunk", chunk)
    _require_positive("element_bytes", element_bytes)
    return {
        "d": d,
        "chunk": chunk,
        "element_bytes": element_bytes,
        "flops": 194 * chunk * d * d
        + 20 * chunk * d
        + 64 * d * d
        + 8 * d,
        "hbm_bytes": element_bytes * (81 * d * d + 8 * d),
    }
