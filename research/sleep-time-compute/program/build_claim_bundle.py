"""Build the frozen Sleep-Time Compute claim, figure, and translation ledgers.

The builder deliberately contains no network access.  Every source is resolved through
the date-frozen primary-source registry, and every generated JSON file is serialized in
one canonical form so that a source freeze can be reproduced byte-for-byte.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

BUNDLE_ID = "sleep-time-compute-strategic-study-2026"
SCHEMA_VERSION = "1.0.0"
SOURCE_FREEZE = "2026-08-05"


def _support(
    source_id: str, locator: str, relation: str = "supports"
) -> dict[str, str]:
    return {"source_id": source_id, "locator": locator, "relation": relation}


def _spec(
    claim_id: str,
    text: str,
    claim_type: str,
    veridraft_type: str,
    support: list[dict[str, str]],
    tags: list[str],
    *,
    confidence: str = "high",
    public: bool = True,
    status: str = "supported",
    load_bearing: bool = True,
    result_ref: str | None = None,
    scope_note: str = "",
) -> dict[str, Any]:
    return {
        "claim_id": claim_id,
        "text": text,
        "claim_type": claim_type,
        "veridraft_type": veridraft_type,
        "support": support,
        "tags": sorted(tags),
        "confidence": confidence,
        "public": public,
        "status": status,
        "load_bearing": load_bearing,
        "result_ref": result_ref,
        "scope_note": scope_note,
    }


def _claim_specs() -> list[dict[str, Any]]:
    """Return the load-bearing claims in stable ID order."""

    return [
        _spec(
            "STC-C001",
            "The date-frozen evidence audit contains 73 primary papers, official releases, "
            "documentation pages, and tagged repositories.",
            "direct_fact",
            "P1",
            [_support("SRC-STC-0014", "source registry audit; entry SRC-STC-0014")],
            ["audit", "corpus"],
            result_ref="R-AUDIT-001",
        ),
        _spec(
            "STC-C002",
            "By 2026-08-05, at least two public production systems described explicit "
            "background memory synthesis: OpenAI Dreaming and Mem0 Dream.",
            "direct_fact",
            "P1",
            [
                _support(
                    "SRC-STC-0052", "official release, opening and How dreaming works"
                ),
                _support("SRC-STC-0063", "official docs, Overview and How Dream works"),
            ],
            ["audit", "industry", "product"],
            result_ref="R-AUDIT-002",
        ),
        _spec(
            "STC-C003",
            "The 12-cluster search closes only as a bounded date snapshot: 2 clusters are "
            "bounded, 5 remain active frontiers, and 5 retain measurement gaps.",
            "direct_fact",
            "P1",
            [_support("SRC-STC-0036", "paper abstract and §5 limitations", "context")],
            ["audit", "saturation"],
            result_ref="R-AUDIT-003",
        ),
        _spec(
            "STC-C004",
            "A balanced 15-paper translation corpus spans explicit sleep, biological and "
            "computational precursors, continual learning, external memory, neural memory, "
            "capacity, and negative evidence.",
            "direct_fact",
            "P1",
            [
                _support(
                    "SRC-STC-0011", "full paper, title through Discussion", "context"
                )
            ],
            ["audit", "translation"],
            result_ref="R-AUDIT-004",
        ),
        _spec(
            "STC-C005",
            "Sleep-time compute is best defined by lifecycle, not by metaphor: computation "
            "is deferred beyond the latency-critical response and changes reusable state for "
            "later wake episodes.",
            "synthesis",
            "P2",
            [
                _support("SRC-STC-0014", "abstract; §1; Fig. 1"),
                _support("SRC-STC-0052", "official release, How dreaming works"),
                _support("SRC-STC-0063", "official docs, Synthesis mode"),
            ],
            ["definition", "lifecycle"],
        ),
        _spec(
            "STC-C006",
            "Update timing and memory medium are orthogonal axes: a sleep phase may update "
            "weights, adapters, recurrent state, text/vector stores, graphs, or mixtures.",
            "synthesis",
            "P2",
            [
                _support("SRC-STC-0014", "§2, sleep-time agent architecture"),
                _support("SRC-STC-0021", "§2-§3, self-modification and consolidation"),
                _support("SRC-STC-0016", "§2-§3, temporal graph memory"),
            ],
            ["definition", "taxonomy"],
        ),
        _spec(
            "STC-C007",
            "OpenAI Dreaming is production background synthesis over external user memory, "
            "not disclosed per-user foundation-weight training.",
            "direct_fact",
            "P2",
            [
                _support(
                    "SRC-STC-0052",
                    "official release, opening through How dreaming works",
                )
            ],
            ["external-memory", "industry", "openai"],
        ),
        _spec(
            "STC-C008",
            "Mem0 Dream performs scheduled background synthesis with provenance-preserving, "
            "additive outputs; merge and supersede are distinct ingest-time operations.",
            "direct_fact",
            "P2",
            [
                _support(
                    "SRC-STC-0063",
                    "official docs, Overview; Modes; Operational behavior",
                ),
                _support("SRC-STC-0064", "tag v2.0.13, dream-gate.ts"),
            ],
            ["external-memory", "industry", "mem0"],
        ),
        _spec(
            "STC-C009",
            "Letta's tagged sleeptime group schedules an asynchronous post-response agent "
            "that revises durable memory outside the response-critical path.",
            "direct_fact",
            "P2",
            [_support("SRC-STC-0059", "tag 0.16.8, sleeptime_multi_agent_v4.py")],
            ["external-memory", "industry", "letta"],
        ),
        _spec(
            "STC-C010",
            "Zep and Graphiti represent external memory as temporally versioned episodes, "
            "entities, and relations rather than as user-specific model-weight updates.",
            "direct_fact",
            "P2",
            [
                _support("SRC-STC-0016", "abstract; §2 Architecture; §3 Temporal KG"),
                _support("SRC-STC-0065", "commit aab852d, README overview"),
                _support("SRC-STC-0066", "official docs, Episodes"),
            ],
            ["external-memory", "graph", "zep"],
        ),
        _spec(
            "STC-C011",
            "MemGPT frames long-term agent memory as an operating-system-style hierarchy "
            "that pages information between constrained context and external storage.",
            "author_claim",
            "P2",
            [
                _support(
                    "SRC-STC-0013", "abstract; §2 Virtual Context Management; Fig. 1"
                )
            ],
            ["external-memory", "memgpt"],
        ),
        _spec(
            "STC-C012",
            "ReasoningBank stores reusable reasoning strategies extracted from experience; "
            "the public method is an external strategy memory rather than broad online weight "
            "consolidation.",
            "direct_fact",
            "P2",
            [
                _support("SRC-STC-0043", "abstract; §2-§3"),
                _support("SRC-STC-0054", "official Google Research post, Method"),
            ],
            ["external-memory", "google", "reasoning"],
        ),
        _spec(
            "STC-C013",
            "Meta's personalized-agent and HyperAgents work emphasizes feedback-conditioned "
            "agent behavior, program memory, and archives; public evidence does not establish "
            "fleet-scale per-user sleep updates to foundation weights.",
            "synthesis",
            "P2",
            [
                _support("SRC-STC-0057", "official Meta paper page, abstract"),
                _support("SRC-STC-0058", "official Meta paper page, abstract"),
            ],
            ["external-memory", "industry", "meta"],
            confidence="medium",
        ),
        _spec(
            "STC-C014",
            "Microsoft STATE-Bench evaluates agent memory across state tracking, adaptation, "
            "and efficiency, but is a benchmark rather than a disclosed sleep-training product.",
            "direct_fact",
            "P2",
            [
                _support(
                    "SRC-STC-0056", "official Microsoft post, benchmark dimensions"
                ),
                _support("SRC-STC-0067", "commit 4efcbf2, README tasks and metrics"),
            ],
            ["benchmark", "microsoft"],
        ),
        _spec(
            "STC-C015",
            "Language Models Need Sleep is direct evidence for parametric sleep: the model "
            "learns to self-modify and consolidate memories using replay-like training, but the "
            "reported regime remains research-scale rather than production deployment proof.",
            "synthesis",
            "P2",
            [
                _support(
                    "SRC-STC-0021",
                    "abstract; §2 Method; §4 Experiments; §6 Limitations",
                )
            ],
            ["google", "parametric", "sleep"],
        ),
        _spec(
            "STC-C016",
            "In a two-task spiking-network setting, interleaved sleep-like replay formed a "
            "joint synaptic representation and reduced catastrophic forgetting.",
            "author_claim",
            "P2",
            [_support("SRC-STC-0011", "abstract; Results, Figs. 2-7; Discussion")],
            ["biological-analogy", "continual-learning", "plos"],
            scope_note="Evidence is narrow: a spiking network and controlled task sequence, not an LLM fleet.",
        ),
        _spec(
            "STC-C017",
            "A hippocampus-neocortex model used autonomous offline dynamics and alternating "
            "NREM/REM regimes to integrate new information while protecting older knowledge.",
            "author_claim",
            "P2",
            [_support("SRC-STC-0010", "abstract; Fig. 1; Simulation 2; Fig. 4")],
            ["biological-analogy", "nrem-rem"],
        ),
        _spec(
            "STC-C018",
            "Perturbed and Adversarial Dreaming separates wake, NREM-like perturbed replay, "
            "and REM-like adversarial dreaming to learn cortical representations.",
            "author_claim",
            "P2",
            [_support("SRC-STC-0009", "abstract; Results; Figs. 1-9; Discussion")],
            ["biological-analogy", "dreaming"],
        ),
        _spec(
            "STC-C019",
            "Deep generative replay is a strong non-sleep-named baseline: a generator replays "
            "pseudo-data while the solver learns new data, reducing catastrophic forgetting.",
            "author_claim",
            "P2",
            [
                _support(
                    "SRC-STC-0003", "abstract; §2 Deep Generative Replay; Algorithm 1"
                )
            ],
            ["continual-learning", "replay"],
        ),
        _spec(
            "STC-C020",
            "Elastic Weight Consolidation protects parameters judged important to earlier "
            "tasks through a Fisher-weighted quadratic penalty, trading plasticity for stability.",
            "author_claim",
            "P2",
            [_support("SRC-STC-0002", "§2; Eq. 3; experiments")],
            ["continual-learning", "regularization"],
        ),
        _spec(
            "STC-C021",
            "Gradient Episodic Memory constrains updates using stored examples so that loss on "
            "prior tasks does not increase under the local constraint.",
            "author_claim",
            "P2",
            [_support("SRC-STC-0004", "§2; Eqs. 3-10; Algorithm 1")],
            ["continual-learning", "episodic-replay"],
        ),
        _spec(
            "STC-C022",
            "Progress & Compress alternates an active column with a consolidation phase that "
            "compresses knowledge into a reusable knowledge base, providing a direct periodic-"
            "refresh alternative to explicit sleep framing.",
            "author_claim",
            "P2",
            [_support("SRC-STC-0007", "abstract; §2 Progress; §3 Compress; Fig. 1")],
            ["continual-learning", "periodic-refresh"],
        ),
        _spec(
            "STC-C023",
            "Titans learns a neural memory state at test time using surprise-driven updates; "
            "this is wake-path fast-state learning, not by itself deferred cross-session sleep.",
            "synthesis",
            "P2",
            [_support("SRC-STC-0017", "§2 Neural Long-Term Memory; Eqs. 7-13; Fig. 1")],
            ["neural-memory", "test-time-learning", "wake"],
        ),
        _spec(
            "STC-C024",
            "Nested Learning/HOPE treats optimization processes at multiple update frequencies "
            "as nested memory levels, supplying a useful cadence model but not public evidence "
            "of deployed asynchronous sleep consolidation.",
            "synthesis",
            "P2",
            [
                _support(
                    "SRC-STC-0020",
                    "§2 Nested Learning; §4 HOPE; CMS frequency schedule",
                )
            ],
            ["google", "hope", "multi-timescale"],
        ),
        _spec(
            "STC-C025",
            "Memory Caching identifies a serving problem for recurrent memory models: reusable "
            "state has to be materialized, keyed, cached, and shared to recover prefix-cache-like "
            "efficiency.",
            "author_claim",
            "P2",
            [
                _support(
                    "SRC-STC-0022",
                    "abstract; §2 Memory Caching; §4 Serving Experiments",
                )
            ],
            ["inference", "memory-caching", "serving"],
        ),
        _spec(
            "STC-C026",
            "Retrieval-augmented generation keeps mutable knowledge outside model weights and "
            "conditions generation on retrieved passages, making it a strong update-cost and "
            "governance baseline.",
            "author_claim",
            "P2",
            [_support("SRC-STC-0031", "abstract; §2 Method; Fig. 1")],
            ["alternative", "external-memory", "rag"],
        ),
        _spec(
            "STC-C027",
            "LoRA constrains adaptation to low-rank weight deltas, reducing trainable parameter "
            "and optimizer-state cost but not eliminating sequential interference or lifecycle "
            "governance.",
            "synthesis",
            "P2",
            [
                _support("SRC-STC-0030", "abstract; §4 Method; Eq. 3"),
                _support(
                    "SRC-STC-0036", "§3 sequential fact-learning results", "qualifies"
                ),
            ],
            ["alternative", "lora", "parametric"],
        ),
        _spec(
            "STC-C028",
            "ROME localizes and edits factual associations in transformer MLP weights, showing "
            "targeted parametric change is possible but not proving stable unbounded continual "
            "accumulation.",
            "synthesis",
            "P2",
            [_support("SRC-STC-0032", "abstract; §3 Causal Tracing; §4 ROME")],
            ["alternative", "model-editing", "parametric"],
        ),
        _spec(
            "STC-C029",
            "MEMIT extends model editing to many memories, but mass editing remains a bounded "
            "batch intervention rather than a demonstrated lifelong sleep loop.",
            "synthesis",
            "P2",
            [_support("SRC-STC-0033", "abstract; §3 MEMIT; scaling experiments")],
            ["alternative", "model-editing", "parametric"],
        ),
        _spec(
            "STC-C030",
            "Generative Adapter compiles context into parameters in one forward pass, suggesting "
            "a latent-memory path whose value depends on reuse amortizing compilation and "
            "version-management cost.",
            "synthesis",
            "P2",
            [_support("SRC-STC-0027", "abstract; §3 Generative Adapter; experiments")],
            ["adapter", "alternative", "latent-compilation"],
        ),
        _spec(
            "STC-C031",
            "Self-Adapting Language Models generate their own adaptation data and update rules, "
            "bridging wake-time learning and later consolidation while introducing recursive-data "
            "quality risks.",
            "synthesis",
            "P2",
            [
                _support("SRC-STC-0029", "abstract; §2-§3 self-edits"),
                _support("SRC-STC-0041", "model-collapse results", "qualifies"),
            ],
            ["data-generation", "self-adaptation", "training"],
        ),
        _spec(
            "STC-C032",
            "Sequentially learned facts can remain encoded in weights yet become behaviorally "
            "unreachable; retention must therefore be measured by access and composition, not "
            "only by parameter storage.",
            "author_claim",
            "P2",
            [_support("SRC-STC-0036", "abstract; §4 Results; §5 Discussion")],
            ["capacity", "negative-evidence", "parametric"],
        ),
        _spec(
            "STC-C033",
            "Static memorization capacity measurements do not directly determine safe continual-"
            "learning capacity because sequential access, interference, and governance add "
            "independent constraints.",
            "synthesis",
            "P2",
            [
                _support("SRC-STC-0034", "abstract; capacity experiments"),
                _support("SRC-STC-0036", "§4-§5", "qualifies"),
            ],
            ["capacity", "negative-evidence"],
        ),
        _spec(
            "STC-C034",
            "A rate-distortion view makes memory compaction an explicit utility-budget tradeoff, "
            "but its proposed frontier is not yet a universal empirical scaling law for agents.",
            "synthesis",
            "P2",
            [
                _support(
                    "SRC-STC-0035",
                    "abstract; §2 Rate-Distortion Formulation; limitations",
                )
            ],
            ["capacity", "compaction", "theory"],
        ),
        _spec(
            "STC-C035",
            "Recursive training on generated data can collapse distribution tails when real-data "
            "support is lost, so sleep-generated replay cannot be treated as costless fresh data.",
            "author_claim",
            "P2",
            [_support("SRC-STC-0041", "main results; Figs. 1-3; Methods")],
            ["data-quality", "model-collapse", "negative-evidence"],
        ),
        _spec(
            "STC-C036",
            "Model collapse is not inevitable when real and synthetic data are accumulated under "
            "appropriate conditions; the relevant scaling variable is recursive depth and real-"
            "data anchoring, not a binary synthetic-data flag.",
            "author_claim",
            "P2",
            [_support("SRC-STC-0069", "abstract; theoretical results; experiments")],
            ["data-quality", "negative-evidence", "qualification"],
        ),
        _spec(
            "STC-C037",
            "External memories create an attack surface: poisoned records can steer agent behavior "
            "while remaining hard to notice, making provenance and revocation first-class design "
            "requirements.",
            "synthesis",
            "P2",
            [_support("SRC-STC-0042", "abstract; threat model; attack results")],
            ["governance", "memory-poisoning", "security"],
        ),
        _spec(
            "STC-C038",
            "As of the freeze date, public industry evidence is strongest for external-memory "
            "sleep and background synthesis, not broad per-user foundation-weight sleep.",
            "synthesis",
            "P2",
            [
                _support("SRC-STC-0052", "official OpenAI Dreaming release"),
                _support("SRC-STC-0063", "official Mem0 Dream docs"),
                _support("SRC-STC-0059", "tagged Letta sleeptime implementation"),
                _support("SRC-STC-0065", "tagged Graphiti repository"),
            ],
            ["industry", "promisingness", "verdict"],
        ),
        _spec(
            "STC-C039",
            "Broad per-user parametric sleep remains technically immature because public evidence "
            "does not jointly establish continual retention, rollback, deletion, security, and "
            "fleet-level serving economics.",
            "synthesis",
            "P2",
            [
                _support("SRC-STC-0021", "§4 experiments and §6 limitations"),
                _support("SRC-STC-0036", "§4-§5 sequential-learning limits"),
                _support("SRC-STC-0042", "memory-poisoning threat model"),
            ],
            ["negative-evidence", "parametric", "verdict"],
            confidence="medium",
        ),
        _spec(
            "STC-C040",
            "The most credible medium-term architecture is hybrid: keep an auditable external "
            "system of record, then selectively promote high-reuse, stable knowledge into "
            "reversible adapters or bounded parametric state.",
            "synthesis",
            "P2",
            [
                _support("SRC-STC-0052", "external synthesis lifecycle"),
                _support("SRC-STC-0027", "parameter compilation method"),
                _support("SRC-STC-0030", "low-rank adapter method"),
                _support("SRC-STC-0066", "episode provenance", "context"),
            ],
            ["architecture", "hybrid", "verdict"],
            confidence="medium",
        ),
        _spec(
            "STC-C041",
            "No source in the frozen corpus establishes a universal sleep-time scaling law; "
            "current equations should be labeled accounting identities, break-even models, fit "
            "candidates, or untested hypotheses.",
            "synthesis",
            "P2",
            [
                _support("SRC-STC-0034", "static memorization-capacity experiments"),
                _support("SRC-STC-0035", "rate-distortion proposal"),
                _support("SRC-STC-0050", "memory-statistics tradeoff theory"),
            ],
            ["scaling-law", "theory", "verdict"],
            confidence="high",
        ),
        _spec(
            "STC-C042",
            "The lifecycle may become mainstream without the term 'sleep-time compute' becoming "
            "the dominant label; current systems use dreaming, reflection, synthesis, memory, "
            "self-evolution, and consolidation language.",
            "synthesis",
            "P2",
            [
                _support("SRC-STC-0052", "title and product terminology"),
                _support("SRC-STC-0054", "title and product terminology"),
                _support("SRC-STC-0060", "reflection-launcher terminology"),
                _support("SRC-STC-0063", "Dream terminology"),
            ],
            ["mainstream", "terminology"],
        ),
        _spec(
            "STC-C043",
            "A useful candidate scaling variable is valid reusable evidence per unit sleep cost, "
            "not raw memories processed or raw sleep FLOPs.",
            "hypothesis",
            "P2",
            [
                _support("SRC-STC-0043", "strategy selection and reuse results"),
                _support("SRC-STC-0051", "RL over memory objective"),
                _support("SRC-STC-0035", "utility-budget formulation"),
            ],
            ["hypothesis", "scaling-law", "utility"],
            confidence="medium",
        ),
        _spec(
            "STC-C044",
            "The optimal sleep cadence should balance staleness, batching efficiency, interference, "
            "and wake-SLA impact; neither continuous updating nor arbitrarily long batching is "
            "generally optimal.",
            "hypothesis",
            "P2",
            [
                _support("SRC-STC-0020", "multi-frequency update framework"),
                _support("SRC-STC-0063", "scheduled synthesis cadence"),
                _support("SRC-STC-0022", "serving and state-reuse costs"),
            ],
            ["cadence", "hypothesis", "scaling-law"],
            confidence="medium",
        ),
        _spec(
            "STC-C045",
            "Finite memory systems should exhibit a capacity knee where marginal consolidation "
            "utility falls and interference, compaction, or deletion cost rises sharply.",
            "hypothesis",
            "P2",
            [
                _support("SRC-STC-0034", "memorization-capacity curves"),
                _support("SRC-STC-0035", "rate-distortion frontier"),
                _support("SRC-STC-0036", "sequential reachability degradation"),
            ],
            ["capacity", "hypothesis", "scaling-law"],
            confidence="medium",
        ),
        _spec(
            "STC-C046",
            "For derived memory, the external source-of-record plus provenance graph is a stronger "
            "governance basis than treating updated weights as the sole authoritative record.",
            "synthesis",
            "P2",
            [
                _support("SRC-STC-0053", "official memory controls"),
                _support("SRC-STC-0063", "provenance and non-destructive behavior"),
                _support("SRC-STC-0066", "episode provenance and temporal facts"),
            ],
            ["architecture", "governance", "provenance"],
        ),
        _spec(
            "STC-C047",
            "The DANN/SSRN record is not load-bearing evidence because the frozen source exposes "
            "insufficient methods and results for independent verification.",
            "direct_fact",
            "P2",
            [_support("SRC-STC-0039", "SSRN abstract and metadata landing page")],
            ["dann", "evidence-hygiene"],
            confidence="high",
            load_bearing=False,
            scope_note="Retained as a blog-named lead and explicit evidence-hygiene example.",
        ),
        _spec(
            "STC-C048",
            "A memory-device vendor could provide a provenance-first append-only store optimized "
            "for derived-memory lineage and selective rebuild.",
            "hypothesis",
            "P3",
            [
                _support("SRC-STC-0063", "provenance and non-destructive synthesis"),
                _support("SRC-STC-0066", "episodes and temporal provenance"),
            ],
            ["device", "governance", "provenance-store"],
            public=False,
            status="held",
            confidence="medium",
        ),
        _spec(
            "STC-C049",
            "Snapshot and copy-on-write primitives could reduce rollback cost when sleep jobs "
            "produce new external or parametric memory generations.",
            "hypothesis",
            "P3",
            [
                _support("SRC-STC-0059", "asynchronous sleeptime lifecycle"),
                _support("SRC-STC-0052", "multi-year synthesized memory lifecycle"),
            ],
            ["device", "rollback", "snapshot"],
            public=False,
            status="held",
            confidence="medium",
        ),
        _spec(
            "STC-C050",
            "High-endurance indexed memory tiers could target repeated graph, vector, and metadata "
            "rewrites from consolidation, merge, decay, and supersede operations.",
            "hypothesis",
            "P3",
            [
                _support("SRC-STC-0062", "memory decay lifecycle"),
                _support("SRC-STC-0063", "merge, supersede, and synthesis modes"),
                _support("SRC-STC-0065", "temporal graph updates"),
            ],
            ["device", "endurance", "index"],
            public=False,
            status="held",
            confidence="medium",
        ),
        _spec(
            "STC-C051",
            "Near-data preprocessing could accelerate deduplication, clustering, graph maintenance, "
            "and provenance filtering before expensive accelerator-side sleep training.",
            "hypothesis",
            "P3",
            [
                _support("SRC-STC-0052", "memory synthesis over large history"),
                _support("SRC-STC-0063", "synthesis workflow"),
            ],
            ["device", "near-data", "preprocessing"],
            public=False,
            status="held",
            confidence="low",
        ),
        _spec(
            "STC-C052",
            "A versioned adapter fabric could store, activate, merge, and retire per-user or per-"
            "domain low-rank deltas without duplicating full foundation weights.",
            "hypothesis",
            "P3",
            [
                _support("SRC-STC-0030", "LoRA low-rank update construction"),
                _support("SRC-STC-0027", "context-to-parameter adapter generation"),
                _support("SRC-STC-0021", "self-modification and consolidation"),
            ],
            ["adapter", "device", "parametric"],
            public=False,
            status="held",
            confidence="medium",
        ),
        _spec(
            "STC-C053",
            "A pooled tiered-memory system could unify context cache, recurrent state, vector index, "
            "temporal graph, and cold episode storage while minimizing state migration.",
            "hypothesis",
            "P3",
            [
                _support("SRC-STC-0013", "virtual context hierarchy"),
                _support("SRC-STC-0016", "temporal graph architecture"),
                _support("SRC-STC-0022", "memory-state caching"),
            ],
            ["device", "pooled-memory", "tiering"],
            public=False,
            status="held",
            confidence="medium",
        ),
        _spec(
            "STC-C054",
            "Secure delete-and-rebuild acceleration could become a differentiator because deleting "
            "a source episode must propagate through derived summaries, graphs, caches, and any "
            "promoted parametric state.",
            "hypothesis",
            "P3",
            [
                _support("SRC-STC-0053", "official memory controls and deletion"),
                _support("SRC-STC-0063", "provenance and non-destructive outputs"),
                _support("SRC-STC-0066", "episode-derived graph facts"),
            ],
            ["delete", "device", "governance"],
            public=False,
            status="held",
            confidence="medium",
        ),
        _spec(
            "STC-C055",
            "Recurrent and neural-memory serving creates an opportunity for a cache tier keyed by "
            "reusable model state rather than only token KV prefixes.",
            "hypothesis",
            "P3",
            [
                _support("SRC-STC-0022", "memory-caching serving design"),
                _support("SRC-STC-0020", "HOPE/CMS persistent state hierarchy"),
            ],
            ["cache", "device", "recurrent-state"],
            public=False,
            status="held",
            confidence="medium",
        ),
        _spec(
            "STC-C056",
            "On-device consolidation may be attractive where privacy, intermittent connectivity, "
            "and local personalization outweigh device thermal and endurance limits.",
            "hypothesis",
            "P3",
            [
                _support("SRC-STC-0044", "personalized-agent learning setup"),
                _support("SRC-STC-0045", "distributed agent program and archive"),
            ],
            ["device", "on-device", "privacy"],
            public=False,
            status="held",
            confidence="low",
        ),
        _spec(
            "STC-C057",
            "Memory-device evaluation tooling should report useful retained evidence, staleness, "
            "rollback time, state-migration bytes, write amplification, and wake-SLA interference "
            "rather than bandwidth alone.",
            "hypothesis",
            "P3",
            [
                _support("SRC-STC-0056", "STATE-Bench dimensions"),
                _support("SRC-STC-0067", "STATE-Bench repository metrics"),
                _support("SRC-STC-0022", "serving efficiency evaluation"),
            ],
            ["benchmark", "device", "measurement"],
            public=False,
            status="held",
            confidence="medium",
        ),
    ]


TRANSLATION_SPECS: tuple[dict[str, Any], ...] = (
    {
        "source_id": "SRC-STC-0014",
        "exact_version": "arXiv:2504.13171v1",
        "source_url": "https://arxiv.org/pdf/2504.13171v1",
        "source_path": None,
        "source_media": "PDF",
        "source_sha256": "050133427aa9cd015a6715d30f9e3cb5f316bfe70929815ad5cb63ec08cf14cc",
        "license_state": "arXiv license; figure redistribution not cleared",
        "figure_count": 26,
        "equation_count": None,
        "table_count": 1,
        "existing_translation": "missing",
        "role": ["direct-stc", "agent-lifecycle"],
    },
    {
        "source_id": "SRC-STC-0021",
        "exact_version": "arXiv:2606.03979v2",
        "source_url": "https://arxiv.org/pdf/2606.03979v2",
        "source_path": None,
        "source_media": "PDF",
        "source_sha256": "dab98367e19e99eb8838a6604afddb9ea85a940f2f41d8a1602e55977fb12d14",
        "license_state": "arXiv license; figure redistribution not cleared",
        "figure_count": 8,
        "equation_count": None,
        "table_count": 5,
        "existing_translation": "existing_v1_requires_v2_delta",
        "role": ["direct-stc", "parametric-sleep"],
    },
    {
        "source_id": "SRC-STC-0036",
        "exact_version": "arXiv:2607.11020v2",
        "source_url": "https://arxiv.org/pdf/2607.11020v2",
        "source_path": None,
        "source_media": "PDF",
        "source_sha256": "3d7fde31817d293afd64066174293fe6a925ce2650331550d66c38af3d6b3702",
        "license_state": "arXiv license; figure redistribution not cleared",
        "figure_count": 22,
        "equation_count": None,
        "table_count": 2,
        "existing_translation": "missing",
        "role": ["capacity", "negative-result", "parametric"],
    },
    {
        "source_id": "SRC-STC-0011",
        "exact_version": "PLOS Computational Biology 18(11):e1010628",
        "source_url": "https://journals.plos.org/ploscompbiol/article/file?id=10.1371/journal.pcbi.1010628&type=printable",
        "source_path": None,
        "source_media": "PDF",
        "source_sha256": "a5ac00a79026561ba06b429b706beeefca552286e70971567ba6038c1ea39035",
        "license_state": "CC BY 4.0 on PLOS article",
        "figure_count": 7,
        "equation_count": None,
        "table_count": 0,
        "existing_translation": "missing",
        "role": ["biological-computational", "continual-learning", "plos-target"],
    },
    {
        "source_id": "SRC-STC-0010",
        "exact_version": "PNAS 119(44):e2123432119; PMC9636926 HTML capture",
        "source_url": "https://pmc.ncbi.nlm.nih.gov/articles/PMC9636926/",
        "source_path": None,
        "source_media": "HTML",
        "source_sha256": "009024a0ce3f47d1c62aab6d91c4560f9af111918ce35013352ed189111e88c9",
        "license_state": "CC BY-NC-ND 4.0; exact figures require license review",
        "figure_count": 4,
        "equation_count": None,
        "table_count": 0,
        "existing_translation": "missing",
        "role": ["biological-computational", "nrem-rem"],
    },
    {
        "source_id": "SRC-STC-0005",
        "exact_version": "arXiv:1606.01164v2",
        "source_url": "https://arxiv.org/pdf/1606.01164v2",
        "source_path": "papers/external/1606.01164.pdf",
        "source_media": "PDF",
        "source_sha256": "fa30d11b475edaaebeaa55815cf98c32ada2806e510bc16ca55602f06f23c83d",
        "license_state": "arXiv license; figure redistribution not cleared",
        "figure_count": 5,
        "equation_count": None,
        "table_count": 0,
        "existing_translation": "missing",
        "role": ["dual-memory", "wake-sleep-precursor"],
    },
    {
        "source_id": "SRC-STC-0003",
        "exact_version": "arXiv:1705.08690v3",
        "source_url": "https://arxiv.org/pdf/1705.08690v3",
        "source_path": None,
        "source_media": "PDF",
        "source_sha256": "cd9cfc3c5b6e71fa3255a40037c6819f403415e2e4fcb539a83f9eda88067e10",
        "license_state": "arXiv license; figure redistribution not cleared",
        "figure_count": 7,
        "equation_count": None,
        "table_count": 1,
        "existing_translation": "missing",
        "role": ["continual-learning", "generative-replay"],
    },
    {
        "source_id": "SRC-STC-0013",
        "exact_version": "arXiv:2310.08560v2",
        "source_url": "https://arxiv.org/pdf/2310.08560v2",
        "source_path": None,
        "source_media": "PDF",
        "source_sha256": "9f674bcff69c86f11c813dcfad613d8841f5f8ed17979e3c4df06a91df7762e0",
        "license_state": "arXiv license; figure redistribution not cleared",
        "figure_count": 8,
        "equation_count": None,
        "table_count": 3,
        "existing_translation": "missing",
        "role": ["agent-memory", "external-memory"],
    },
    {
        "source_id": "SRC-STC-0016",
        "exact_version": "arXiv:2501.13956v1",
        "source_url": "https://arxiv.org/pdf/2501.13956v1",
        "source_path": None,
        "source_media": "PDF",
        "source_sha256": "d26f7eb599540e8b14d75e7efda58a07661abbb1b864e58323b5768475a15d42",
        "license_state": "arXiv license; figure redistribution not cleared",
        "figure_count": None,
        "equation_count": None,
        "table_count": 3,
        "existing_translation": "missing",
        "role": ["agent-memory", "external-memory", "temporal-graph"],
    },
    {
        "source_id": "SRC-STC-0020",
        "exact_version": "arXiv:2512.24695v1",
        "source_url": "https://arxiv.org/pdf/2512.24695v1",
        "source_path": "papers/2512.24695.pdf",
        "source_media": "PDF",
        "source_sha256": "e87a9ce82ff24e96f55b83fb9713a5d6fc9e4a1a232c12d42da49c10022ed891",
        "license_state": "arXiv license; figure redistribution not cleared",
        "figure_count": 12,
        "equation_count": None,
        "table_count": 9,
        "existing_translation": "existing_full",
        "role": ["hope", "multi-timescale", "neural-memory"],
    },
    {
        "source_id": "SRC-STC-0022",
        "exact_version": "arXiv:2602.24281v1",
        "source_url": "https://arxiv.org/pdf/2602.24281v1",
        "source_path": "papers/2602.24281.pdf",
        "source_media": "PDF",
        "source_sha256": "f4983c96c96400320b3d883e8d39f96ade91cae55e2bde04c4d99b06a8348e82",
        "license_state": "arXiv license; figure redistribution not cleared",
        "figure_count": 5,
        "equation_count": None,
        "table_count": 6,
        "existing_translation": "existing",
        "role": ["neural-memory", "serving", "state-cache"],
    },
    {
        "source_id": "SRC-STC-0035",
        "exact_version": "arXiv:2607.08032v1",
        "source_url": "https://arxiv.org/pdf/2607.08032v1",
        "source_path": None,
        "source_media": "PDF",
        "source_sha256": "6c19532535fbc3c748faa1a67b0e047581a6fc56b99b54d43e47097be8107ebd",
        "license_state": "arXiv license; figure redistribution not cleared",
        "figure_count": 6,
        "equation_count": None,
        "table_count": 6,
        "existing_translation": "missing",
        "role": ["capacity", "compaction", "rate-distortion"],
    },
    {
        "source_id": "SRC-STC-0009",
        "exact_version": "eLife 11:e76384",
        "source_url": "https://elifesciences.org/articles/76384.pdf",
        "source_path": None,
        "source_media": "PDF",
        "source_sha256": "ce7aef9099666b4b40208c83580d43b248d9129f818b53e91950c299790866f1",
        "license_state": "CC BY 4.0 on eLife article",
        "figure_count": 9,
        "equation_count": None,
        "table_count": 1,
        "existing_translation": "missing",
        "role": ["biological-computational", "dreaming"],
    },
    {
        "source_id": "SRC-STC-0034",
        "exact_version": "arXiv:2505.24832v3",
        "source_url": "https://arxiv.org/pdf/2505.24832v3",
        "source_path": None,
        "source_media": "PDF",
        "source_sha256": "ce0148ea694019714c4c3d5d8bb3a09b2f1e7c402f377f9db50cae705e32b123",
        "license_state": "arXiv license; figure redistribution not cleared",
        "figure_count": 17,
        "equation_count": None,
        "table_count": 5,
        "existing_translation": "missing",
        "role": ["capacity", "parametric-memory"],
    },
    {
        "source_id": "SRC-STC-0040",
        "exact_version": "arXiv:2603.01097v5",
        "source_url": "https://arxiv.org/pdf/2603.01097v5",
        "source_path": None,
        "source_media": "PDF",
        "source_sha256": "08b937f35edf84cb1ac2ab67b8ff4a49a8f4f1d71a7629418555c0fcca66ad68",
        "license_state": "arXiv license; figure redistribution not cleared",
        "figure_count": 22,
        "equation_count": None,
        "table_count": 7,
        "existing_translation": "missing",
        "role": ["adapter", "capacity", "parametric-memory"],
    },
)


FIGURE_SPECS: tuple[tuple[str, str, list[str]], ...] = (
    (
        "STC-F001",
        "Wake-test-sleep lifecycle and reusable-state boundary",
        ["SRC-STC-0014", "SRC-STC-0052"],
    ),
    (
        "STC-F002",
        "Update timing by memory-medium taxonomy",
        ["SRC-STC-0014", "SRC-STC-0021", "SRC-STC-0016"],
    ),
    ("STC-F003", "1995-2026 evidence chronology", ["SRC-STC-0001", "SRC-STC-0063"]),
    (
        "STC-F004",
        "Problem-to-solution comparison matrix",
        ["SRC-STC-0002", "SRC-STC-0031", "SRC-STC-0036"],
    ),
    (
        "STC-F005",
        "Conditional promisingness verdict",
        ["SRC-STC-0052", "SRC-STC-0063", "SRC-STC-0021"],
    ),
    (
        "STC-F006",
        "Hybrid system-of-record and selective-promotion architecture",
        ["SRC-STC-0030", "SRC-STC-0066"],
    ),
    (
        "STC-F007",
        "Measured identities, break-even models, and hypotheses",
        ["SRC-STC-0035", "SRC-STC-0050"],
    ),
    (
        "STC-F008",
        "Sleep-cost versus later-reuse break-even",
        ["SRC-STC-0027", "SRC-STC-0052"],
    ),
    ("STC-F009", "Finite-memory capacity knee", ["SRC-STC-0034", "SRC-STC-0036"]),
    (
        "STC-F010",
        "Cadence frontier across staleness and batching",
        ["SRC-STC-0020", "SRC-STC-0063"],
    ),
    (
        "STC-F011",
        "State-migration roofline for wake and sleep clusters",
        ["SRC-STC-0022", "SRC-STC-0059"],
    ),
    ("STC-F012", "Memory-device opportunity map", ["SRC-STC-0056", "SRC-STC-0065"]),
    (
        "STC-F013",
        "Failure modes and lifecycle controls",
        ["SRC-STC-0042", "SRC-STC-0053"],
    ),
    (
        "STC-F014",
        "Industry and academia landscape at the source freeze",
        ["SRC-STC-0054", "SRC-STC-0057", "SRC-STC-0056"],
    ),
    (
        "STC-F015",
        "Falsifiable research and benchmark agenda",
        ["SRC-STC-0036", "SRC-STC-0056", "SRC-STC-0050"],
    ),
)


def _canonical_bytes(payload: object) -> bytes:
    return (
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")


def _sha_payload(payload: object) -> str:
    return hashlib.sha256(_canonical_bytes(payload)).hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _build_results(
    source_count: int, cluster_statuses: Counter[str], translation_count: int
) -> list[dict[str, Any]]:
    return [
        {
            "result_id": "R-AUDIT-001",
            "description": "Frozen primary-source registry count at the source-freeze date.",
            "metrics": [
                {
                    "name": "frozen_primary_sources",
                    "value": source_count,
                    "unit": "sources",
                }
            ],
        },
        {
            "result_id": "R-AUDIT-002",
            "description": "Explicit public production background-synthesis systems.",
            "metrics": [
                {
                    "name": "production_background_synthesis_systems",
                    "value": 2,
                    "unit": "systems",
                }
            ],
        },
        {
            "result_id": "R-AUDIT-003",
            "description": "Manual cluster-closure status after four search passes.",
            "metrics": [
                {
                    "name": "bounded_clusters",
                    "value": cluster_statuses["bounded"],
                    "unit": "clusters",
                },
                {
                    "name": "active_frontiers",
                    "value": cluster_statuses["active_frontier"],
                    "unit": "clusters",
                },
                {
                    "name": "measurement_gaps",
                    "value": cluster_statuses["measurement_gap"],
                    "unit": "clusters",
                },
            ],
        },
        {
            "result_id": "R-AUDIT-004",
            "description": "Structurally balanced faithful-translation selection.",
            "metrics": [
                {
                    "name": "selected_translation_papers",
                    "value": translation_count,
                    "unit": "papers",
                }
            ],
        },
    ]


def _translation_selection() -> dict[str, Any]:
    selected: list[dict[str, Any]] = []
    for index, spec in enumerate(TRANSLATION_SPECS, start=1):
        item = dict(spec)
        item.update(
            {
                "paper_id": f"STC-T{index:02d}",
                "selection_reason": "Fills a load-bearing stratum in the balanced evidence corpus.",
                "status": "selected",
                "count_method": (
                    "Numbered figure/table labels extracted from the frozen source; equation "
                    "count remains null until source-structure parity extraction."
                ),
            }
        )
        selected.append(item)
    return {"schema_version": SCHEMA_VERSION, "selected": selected}


def _figure_ledger() -> dict[str, Any]:
    figures = []
    evidence_url = (
        "https://github.com/jimmylegendary/neural-memory-study/"
        "blob/main/claims/stc-study/figure-ledger.json"
    )
    for figure_id, title, source_ids in FIGURE_SPECS:
        figures.append(
            {
                "figure_id": figure_id,
                "title": title,
                "mode": "redrawn",
                "public": True,
                "source_ids": source_ids,
                "license": {
                    "license_id": "original-author-created-redraw",
                    "evidence_url": evidence_url,
                    "evidence_locator": f"figure-ledger entry {figure_id}",
                    "reuse_allowed": True,
                    "review_status": "reviewed",
                },
                "caption": "Original synthesis figure; underlying factual claims retain source citations.",
                "local_path": f"research/sleep-time-compute/study-paper/figures/{figure_id.lower()}.pdf",
            }
        )
    return {"schema_version": SCHEMA_VERSION, "figures": figures}


def build_artifacts(repo_root: Path) -> dict[str, dict[str, Any]]:
    """Build every canonical ledger without mutating the repository."""

    repo_root = Path(repo_root)
    deep = repo_root / "research/sleep-time-compute/deep-research"
    source_registry = _read_json(deep / "SOURCE-REGISTRY.json")
    saturation_source = _read_json(deep / "SATURATION.json")
    s2_status = _read_json(deep / "S2-STATUS.json")
    sources = {source["source_id"]: source for source in source_registry["sources"]}

    specs = _claim_specs()
    missing_sources = sorted(
        {
            support["source_id"]
            for spec in specs
            for support in spec["support"]
            if support["source_id"] not in sources
        }
    )
    if missing_sources:
        raise ValueError(f"claim specs reference missing source IDs: {missing_sources}")

    claim_map_claims: list[dict[str, Any]] = []
    bundle_claims: list[dict[str, Any]] = []
    for spec in specs:
        mapped = {
            "claim_id": spec["claim_id"],
            "text": spec["text"],
            "claim_type": spec["claim_type"],
            "status": spec["status"],
            "public": spec["public"],
            "load_bearing": spec["load_bearing"],
            "support": spec["support"],
            "confidence": spec["confidence"],
            "scope_note": spec["scope_note"],
            "tags": spec["tags"],
        }
        claim_map_claims.append(mapped)

        evidence: list[dict[str, Any]] = []
        for index, support in enumerate(spec["support"], start=1):
            if spec["veridraft_type"] == "P1":
                ref = f"caw01://{BUNDLE_ID}/{spec['result_ref']}"
                kind = "caw01_result"
            else:
                ref = sources[support["source_id"]]["canonical_url"]
                kind = "source_artifact"
            evidence.append(
                {
                    "id": f"{spec['claim_id']}-E{index:02d}",
                    "kind": kind,
                    "ref": ref,
                    "trust": 0.95 if support["relation"] == "supports" else 0.85,
                    "source_id": support["source_id"],
                    "locator": support["locator"],
                    "relation": support["relation"],
                }
            )
        bundle_claims.append(
            {
                "claim_id": spec["claim_id"],
                "type": spec["veridraft_type"],
                "boundary": "public" if spec["public"] else "internal",
                "visibility": "team" if spec["public"] else "private",
                "statement": spec["text"],
                "result_refs": [spec["result_ref"]] if spec["result_ref"] else [],
                "evidence": evidence,
                "load_bearing": spec["load_bearing"],
                "tags": spec["tags"],
            }
        )

    translation_selection = _translation_selection()
    figure_ledger = _figure_ledger()
    statuses = Counter(cluster["status"] for cluster in saturation_source["clusters"])
    results = _build_results(
        len(source_registry["sources"]),
        statuses,
        len(translation_selection["selected"]),
    )
    bundle = {
        "bundle_id": BUNDLE_ID,
        "boundary": "mixed_public_and_internal_held",
        "provenance_manifest": {
            "exported_by": "stc deterministic claim-bundle builder",
            "source_freeze": SOURCE_FREEZE,
            "source_registry_sha256": hashlib.sha256(
                (deep / "SOURCE-REGISTRY.json").read_bytes()
            ).hexdigest(),
            "note": (
                "P1 claims resolve to declared audit results; P2 claims resolve to frozen "
                "primary source artifacts; P3 device projections are internal/private and "
                "remain patent-first held."
            ),
        },
        "claims": sorted(bundle_claims, key=lambda item: item["claim_id"]),
        "results": sorted(results, key=lambda item: item["result_id"]),
    }
    claim_map = {
        "schema_version": SCHEMA_VERSION,
        "claims": sorted(claim_map_claims, key=lambda item: item["claim_id"]),
    }
    saturation_report = {
        "schema_version": SCHEMA_VERSION,
        "source_freeze": SOURCE_FREEZE,
        "status": saturation_source["global_status"],
        "source_count": len(source_registry["sources"]),
        "citation_count": len(_read_json(deep / "CITATION-POOL.json")["citations"]),
        "cluster_count": len(saturation_source["clusters"]),
        "cluster_status_counts": dict(sorted(statuses.items())),
        "semantic_scholar_status": s2_status.get("status", "unknown"),
        "saturation_claim": "bounded date snapshot; universality and citation saturation are not claimed",
    }

    registry_paths = {
        "sources": "research/sleep-time-compute/deep-research/SOURCE-REGISTRY.json",
        "claims": "claims/stc-study/claim-map.json",
        "figures": "claims/stc-study/figure-ledger.json",
        "translations": "claims/stc-study/translation-selection.json",
    }
    registry_hashes = {
        "sources": hashlib.sha256(
            (deep / "SOURCE-REGISTRY.json").read_bytes()
        ).hexdigest(),
        "claims": _sha_payload(claim_map),
        "figures": _sha_payload(figure_ledger),
        "translations": _sha_payload(translation_selection),
        "bundle": _sha_payload(bundle),
        "saturation": _sha_payload(saturation_report),
    }
    research_spine = {
        "schema_version": SCHEMA_VERSION,
        "status": "frozen",
        "source_freeze": SOURCE_FREEZE,
        "registries": registry_paths,
        "registry_hashes": registry_hashes,
        "unresolved_load_bearing_claims": [],
    }
    return {
        "bundle": bundle,
        "claim_map": claim_map,
        "figure_ledger": figure_ledger,
        "translation_selection": translation_selection,
        "saturation": saturation_report,
        "research_spine": research_spine,
    }


def write_artifacts(repo_root: Path) -> list[Path]:
    """Write all generated ledgers with canonical JSON formatting."""

    repo_root = Path(repo_root)
    artifacts = build_artifacts(repo_root)
    paths = {
        "bundle": repo_root / "claims/stc-study/bundle.json",
        "claim_map": repo_root / "claims/stc-study/claim-map.json",
        "figure_ledger": repo_root / "claims/stc-study/figure-ledger.json",
        "translation_selection": repo_root
        / "claims/stc-study/translation-selection.json",
        "saturation": repo_root / "claims/stc-study/reports/saturation.json",
        "research_spine": repo_root
        / "research/sleep-time-compute/program/research-spine.json",
    }
    written: list[Path] = []
    for key, path in paths.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(_canonical_bytes(artifacts[key]))
        written.append(path)
    return written


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parents[3],
        help="repository root (default: inferred from this file)",
    )
    args = parser.parse_args()
    for path in write_artifacts(args.repo_root):
        print(path.relative_to(args.repo_root))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
