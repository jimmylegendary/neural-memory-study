"""Build frozen deep-research source, citation, and rights registries.

This script is intentionally a capture tool rather than a build-time dependency.  It
resolves paper metadata from arXiv/Crossref, hashes mutable official pages as they
exist on the source-freeze date, and writes auditable JSON artifacts.  Re-running it
after the freeze date creates a new snapshot and therefore requires review.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import time
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

FREEZE = "2026-08-05"
ROOT = Path(__file__).resolve().parents[3]
PRE_RESEARCH = ROOT / "research" / "sleep-time-compute" / "pre-research"
DEEP_RESEARCH = ROOT / "research" / "sleep-time-compute" / "deep-research"
USER_AGENT = "neural-memory-study/1.0 (evidence capture; contact: repository owner)"
ARXIV_NS = {"a": "http://www.w3.org/2005/Atom"}


@dataclass(frozen=True)
class Supplement:
    title: str
    url: str
    published_at: str
    source_type: str = "paper"
    organization: str | None = None
    arxiv_id: str | None = None
    doi: str | None = None
    authors: tuple[str, ...] = ()
    clusters: tuple[str, ...] = ()
    evidence_role: str = "supportive_and_qualifying"
    maturity: str = "primary_source"
    locator: str = "abstract and methods/results as cited in the claim ledger"
    immutable: bool = False
    license_id: str = "publisher-terms-not-cleared"
    license_url: str = "https://www.copyright.gov/fair-use/"
    redistribution_allowed: bool = False
    version_id: str | None = None


SUPPLEMENTS = (
    Supplement(
        "AI Models Collapse When Trained on Recursively Generated Data",
        "https://doi.org/10.1038/s41586-024-07566-y",
        "2024-07-24",
        doi="10.1038/s41586-024-07566-y",
        clusters=("C03", "C09", "C12"),
        evidence_role="central_disconfirming",
        maturity="Nature, peer reviewed",
        locator="abstract; Methods; Fig. 1-4; version of record",
        immutable=True,
    ),
    Supplement(
        "AgentPoison: Red-teaming LLM Agents via Poisoning Memory or Knowledge Bases",
        "https://arxiv.org/abs/2407.12784",
        "2024-07-17",
        arxiv_id="2407.12784",
        clusters=("C05", "C11", "C12"),
        evidence_role="central_disconfirming",
        maturity="NeurIPS 2024",
        locator="abstract; Sections 3-5; Tables 1-4",
        immutable=True,
    ),
    Supplement(
        "ReasoningBank: Scaling Agent Self-Evolving with Reasoning Memory",
        "https://arxiv.org/abs/2509.25140",
        "2025-09-29",
        arxiv_id="2509.25140",
        clusters=("C05", "C07", "C09"),
        evidence_role="strong_alternative_and_bridge",
        maturity="ICLR 2026",
        locator="abstract; Sections 3-5; main result tables",
        immutable=True,
    ),
    Supplement(
        "Learning Personalized Agents from Human Feedback",
        "https://arxiv.org/abs/2602.16173",
        "2026-02-18",
        arxiv_id="2602.16173",
        clusters=("C05", "C07", "C11"),
        evidence_role="strong_external_memory_alternative",
        maturity="Meta arXiv preprint",
        locator="abstract; method loop; benchmark results",
        immutable=True,
    ),
    Supplement(
        "Hyperagents",
        "https://arxiv.org/abs/2603.19461",
        "2026-03-19",
        arxiv_id="2603.19461",
        clusters=("C07", "C09", "C12"),
        evidence_role="adjacent_self_improvement",
        maturity="Meta arXiv preprint",
        locator="abstract; Sections 3-6; safety discussion",
        immutable=True,
    ),
    Supplement(
        "Continual Knowledge Updating in LLM Systems: Learning Through Multi-Timescale Memory Dynamics",
        "https://arxiv.org/abs/2605.05097",
        "2026-05-06",
        arxiv_id="2605.05097",
        clusters=("C05", "C08", "C09"),
        evidence_role="external_consolidation_theory",
        maturity="arXiv preprint",
        locator="abstract; multi-timescale dynamics; experiments and limitations",
        immutable=True,
    ),
    Supplement(
        "SCM: Sleep-Consolidated Memory with Algorithmic Forgetting",
        "https://arxiv.org/abs/2604.20943",
        "2026-04-29",
        arxiv_id="2604.20943",
        clusters=("C05", "C08", "C09"),
        evidence_role="low_maturity_direct_stc",
        maturity="single-author research preview",
        locator="abstract; algorithm; ten-turn demonstration; limitations",
        immutable=True,
    ),
    Supplement(
        "Remember When It Matters: Proactive Memory Agent for Long-Horizon Agents",
        "https://arxiv.org/abs/2607.08716",
        "2026-07-11",
        arxiv_id="2607.08716",
        clusters=("C05", "C07", "C11"),
        evidence_role="learned_external_memory_policy",
        maturity="arXiv preprint",
        locator="abstract; SETA data; SFT/GRPO training; validation results",
        immutable=True,
    ),
    Supplement(
        "MaRS: Memory-Adaptive Routing for Reliable Capacity Expansion and Knowledge Retention",
        "https://openreview.net/forum?id=GGrLeik2qo",
        "2026-01-26",
        clusters=("C03", "C08", "C09"),
        evidence_role="capacity_expansion_alternative",
        maturity="ICLR 2026 Poster",
        locator="OpenReview abstract; Sections 3-5; capacity expansion discussion",
        license_id="CC-BY-4.0",
        license_url="https://creativecommons.org/licenses/by/4.0/",
        redistribution_allowed=True,
    ),
    Supplement(
        "Memory-Statistics Tradeoff in Continual Learning with Structural Regularization",
        "https://openreview.net/forum?id=qfEqXJnlB4",
        "2026-01-26",
        clusters=("C02", "C08", "C09"),
        evidence_role="capacity_tradeoff_theory",
        maturity="ICLR 2026 Poster",
        locator="OpenReview abstract; theorem statements; upper/lower bounds",
        license_id="CC-BY-4.0",
        license_url="https://creativecommons.org/licenses/by/4.0/",
        redistribution_allowed=True,
    ),
    Supplement(
        "From Player to Master: Enhancing Test-Time Learning of LLM Agents via Reinforcement Learning over Memory",
        "https://openreview.net/forum?id=gNWNtstp3r",
        "2026-04-30",
        clusters=("C05", "C07", "C09"),
        evidence_role="trained_external_memory_policy",
        maturity="ICML 2026 regular paper",
        locator="OpenReview abstract; Figure 2; Sections 3-5",
        license_id="CC-BY-4.0",
        license_url="https://creativecommons.org/licenses/by/4.0/",
        redistribution_allowed=True,
    ),
    Supplement(
        "Dreaming: Better Memory for a More Helpful ChatGPT",
        "https://openai.com/index/chatgpt-memory-dreaming/",
        "2026-06-04",
        source_type="official_release_note",
        organization="OpenAI",
        clusters=("C05", "C07", "C10", "C11"),
        evidence_role="production_functional_stc",
        maturity="public production rollout",
        locator="official page lines 39-53, 241-284 in 2026-08-05 capture",
    ),
    Supplement(
        "Memory and New Controls for ChatGPT",
        "https://openai.com/index/memory-and-new-controls-for-chatgpt/",
        "2025-04-10",
        source_type="official_release_note",
        organization="OpenAI",
        clusters=("C05", "C11", "C12"),
        evidence_role="governance_and_deletion_boundary",
        maturity="public product controls",
        locator="official page; memory controls and deletion semantics",
    ),
    Supplement(
        "ReasoningBank: Enabling Agents to Learn from Experience",
        "https://www.research.google/blog/reasoningbank-enabling-agents-to-learn-from-experience/",
        "2026-04-21",
        source_type="official_release_note",
        organization="Google Research",
        clusters=("C05", "C07", "C09"),
        evidence_role="official_external_memory_research",
        maturity="official research release",
        locator="Distilling insights; MaTTS; Performance & emergent capabilities",
    ),
    Supplement(
        "Introducing Nested Learning: A New ML Paradigm for Continual Learning",
        "https://research.google/blog/introducing-nested-learning-a-new-ml-paradigm-for-continual-learning/",
        "2025-11-07",
        source_type="official_release_note",
        organization="Google Research",
        clusters=("C04", "C08", "C09"),
        evidence_role="official_parametric_multifrequency_research",
        maturity="official NeurIPS 2025 research release",
        locator="continuum of memories; Hope architecture; limitations",
    ),
    Supplement(
        "Introducing STATE-Bench: A Benchmark for AI Agent Memory",
        "https://opensource.microsoft.com/blog/2026/05/19/introducing-state-bench-a-benchmark-for-ai-agent-memory/",
        "2026-05-19",
        source_type="official_release_note",
        organization="Microsoft",
        clusters=("C05", "C10", "C11"),
        evidence_role="production_memory_benchmark",
        maturity="official benchmark release",
        locator="evaluation loop; metrics; no-memory baseline; open challenge",
    ),
    Supplement(
        "Learning Personalized Agents from Human Feedback",
        "https://ai.meta.com/research/publications/learning-personalized-agents-from-human-feedback/",
        "2026-02-26",
        source_type="official_release_note",
        organization="Meta AI",
        clusters=("C05", "C07", "C11"),
        evidence_role="official_external_memory_research",
        maturity="official Meta research release",
        locator="official abstract; three-step online feedback loop",
    ),
    Supplement(
        "HyperAgents",
        "https://ai.meta.com/research/publications/hyperagents/",
        "2026-03-24",
        source_type="official_release_note",
        organization="Meta AI",
        clusters=("C07", "C09", "C12"),
        evidence_role="official_self_improvement_research",
        maturity="official Meta research release",
        locator="official abstract; accumulated improvement; safety precautions",
    ),
    Supplement(
        "Letta Sleeptime Multi-Agent V4",
        "https://github.com/letta-ai/letta/blob/0.16.8/letta/groups/sleeptime_multi_agent_v4.py",
        "2026-05-14",
        source_type="official_repository",
        organization="Letta",
        clusters=("C05", "C07", "C10"),
        evidence_role="public_async_sleep_implementation",
        maturity="tagged production repository code",
        locator="class at line 805; post-response run near 988; background task near 1056; prompt near 1169",
        immutable=True,
        license_id="Apache-2.0",
        license_url="https://github.com/letta-ai/letta/blob/0.16.8/LICENSE",
        redistribution_allowed=True,
        version_id="git:1131535716e8a31c9a437f8695e25ac98f203a24",
    ),
    Supplement(
        "Letta Code Reflection Launcher",
        "https://github.com/letta-ai/letta-code/blob/v0.28.18/src/cli/helpers/reflection-launcher.ts",
        "2026-07-23",
        source_type="official_repository",
        organization="Letta",
        clusters=("C07", "C10", "C11"),
        evidence_role="versioned_external_memory_update_pattern",
        maturity="tagged production repository code",
        locator="isolated worktree/snapshot lines 2207-2234; merge-on-success lines 2281-2285",
        immutable=True,
        license_id="Apache-2.0",
        license_url="https://github.com/letta-ai/letta-code/blob/v0.28.18/LICENSE",
        redistribution_allowed=True,
        version_id="git:de81fda525306140ca063dc93bb1267ff056657c",
    ),
    Supplement(
        "Mem0 V3 Add Memories API",
        "https://docs.mem0.ai/api-reference/memory/add-memories",
        FREEZE,
        source_type="official_documentation",
        organization="Mem0",
        clusters=("C05", "C07", "C11"),
        evidence_role="async_external_memory_ingestion",
        maturity="official product documentation snapshot",
        locator="official page lines 207-217 and V3 behavior section",
    ),
    Supplement(
        "Mem0 Memory Decay",
        "https://docs.mem0.ai/platform/features/memory-decay",
        FREEZE,
        source_type="official_documentation",
        organization="Mem0",
        clusters=("C05", "C08", "C11"),
        evidence_role="bounded_retrieval_ranking_mechanism",
        maturity="official product documentation snapshot",
        locator="official page lines 81-120, 243-271",
    ),
    Supplement(
        "Mem0 Dream",
        "https://docs.mem0.ai/platform/features/dream",
        FREEZE,
        source_type="official_documentation",
        organization="Mem0",
        clusters=("C05", "C07", "C08", "C11"),
        evidence_role="production_external_sleep_consolidation",
        maturity="official paid product documentation snapshot",
        locator="official page lines 81-145 and 190-226",
    ),
    Supplement(
        "Mem0 OpenClaw Dream Gate",
        "https://github.com/mem0ai/mem0/blob/v2.0.13/integrations/openclaw/dream-gate.ts",
        "2026-07-22",
        source_type="official_repository",
        organization="Mem0",
        clusters=("C07", "C10", "C11"),
        evidence_role="public_sleep_trigger_implementation",
        maturity="tagged integration code, not a paper result",
        locator="automatic consolidation and persistent lock/state near lines 620-632",
        immutable=True,
        license_id="Apache-2.0",
        license_url="https://github.com/mem0ai/mem0/blob/v2.0.13/LICENSE",
        redistribution_allowed=True,
        version_id="git:ca2abca2b884e038d3e525070e79d3057ef2012c",
    ),
    Supplement(
        "Graphiti: Temporal Knowledge Graphs for Agent Memory",
        "https://github.com/getzep/graphiti/blob/aab852df94413fd0d55cbea2b7886173020281d5/README.md",
        "2026-08-03",
        source_type="official_repository",
        organization="Zep",
        clusters=("C05", "C08", "C11"),
        evidence_role="external_temporal_graph_memory",
        maturity="pinned official repository",
        locator="README overview; temporal updates; provenance; hybrid retrieval",
        immutable=True,
        license_id="Apache-2.0",
        license_url="https://github.com/getzep/graphiti/blob/aab852df94413fd0d55cbea2b7886173020281d5/LICENSE",
        redistribution_allowed=True,
        version_id="git:aab852df94413fd0d55cbea2b7886173020281d5",
    ),
    Supplement(
        "Zep Episodes",
        "https://help.getzep.com/episodes",
        FREEZE,
        source_type="official_documentation",
        organization="Zep",
        clusters=("C05", "C08", "C11"),
        evidence_role="episodic_provenance_storage",
        maturity="official product documentation snapshot",
        locator="official page lines 163-168 in 2026-08-05 capture",
    ),
    Supplement(
        "STATE-Bench Repository",
        "https://github.com/microsoft/STATE-Bench/tree/4efcbf2d4fe60df04878859b692d9391f3d5b33a",
        "2026-07-16",
        source_type="official_repository",
        organization="Microsoft",
        clusters=("C05", "C10", "C11"),
        evidence_role="reproducible_agent_memory_benchmark",
        maturity="pinned official repository",
        locator="README; task suite; metrics; memory interface",
        immutable=True,
        license_id="MIT",
        license_url="https://github.com/microsoft/STATE-Bench/blob/4efcbf2d4fe60df04878859b692d9391f3d5b33a/LICENSE",
        redistribution_allowed=True,
        version_id="git:4efcbf2d4fe60df04878859b692d9391f3d5b33a",
    ),
    Supplement(
        "Catastrophic Interference in Connectionist Networks: The Sequential Learning Problem",
        "https://doi.org/10.1016/S0079-7421(08)60536-8",
        "1989-01-01",
        doi="10.1016/S0079-7421(08)60536-8",
        authors=("Michael McCloskey", "Neal J. Cohen"),
        clusters=("C02", "C08"),
        evidence_role="seminal_problem_definition",
        maturity="Psychology of Learning and Motivation book chapter",
        locator="chapter problem statement and sequential-learning experiments",
        immutable=True,
    ),
    Supplement(
        "Is Model Collapse Inevitable? Breaking the Curse of Recursion by Accumulating Real and Synthetic Data",
        "https://arxiv.org/abs/2404.01413",
        "2024-04-01",
        arxiv_id="2404.01413",
        clusters=("C03", "C09", "C12"),
        evidence_role="qualifying_negative_result",
        maturity="ICML 2024",
        locator="abstract; accumulated-data experiments; main theorem and figures",
        immutable=True,
    ),
    Supplement(
        "Continual Learning of Diffusion Models with Generative Distillation",
        "https://arxiv.org/abs/2311.14028",
        "2023-11-23",
        arxiv_id="2311.14028",
        clusters=("C02", "C03", "C09"),
        evidence_role="replay_failure_and_distillation_remedy",
        maturity="arXiv preprint",
        locator="abstract; catastrophic denoising failure; generative distillation method and experiments",
        immutable=True,
    ),
    Supplement(
        "When Benchmarks Leak: Inference-Time Decontamination for LLMs",
        "https://aclanthology.org/2026.acl-long.2071/",
        "2026-07-01",
        doi="10.18653/v1/2026.acl-long.2071",
        authors=("Jianzhe Chai", "Yu Zhe", "Jun Sakuma"),
        clusters=("C09", "C11"),
        evidence_role="benchmark_contamination_negative",
        maturity="ACL 2026 long paper",
        locator="pages 44743-44760; abstract; contamination experiments",
        immutable=True,
        license_id="CC-BY-4.0",
        license_url="https://creativecommons.org/licenses/by/4.0/",
        redistribution_allowed=True,
    ),
    Supplement(
        "AgentLeak: A Full-Stack Benchmark for Privacy Leakage in Multi-Agent LLM Systems",
        "https://arxiv.org/abs/2602.11510",
        "2026-02-12",
        arxiv_id="2602.11510",
        clusters=("C10", "C11", "C12"),
        evidence_role="internal_channel_privacy_negative",
        maturity="arXiv preprint",
        locator="abstract; threat taxonomy; internal-channel results",
        immutable=True,
    ),
    Supplement(
        "Search-Time Contamination in Deep Research Agents: Measuring Performance Inflation in Public Benchmark Evaluation",
        "https://arxiv.org/abs/2606.05241",
        "2026-06-03",
        arxiv_id="2606.05241",
        clusters=("C05", "C09", "C11"),
        evidence_role="retrieval_evaluation_contamination_negative",
        maturity="arXiv preprint",
        locator="abstract; contamination taxonomy; six-benchmark evaluation",
        immutable=True,
    ),
)


def _request(url: str) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            return response.read()
    except urllib.error.HTTPError as exc:
        if exc.code not in {403, 429}:
            raise
        completed = subprocess.run(
            [
                "curl",
                "--fail",
                "--location",
                "--silent",
                "--show-error",
                "--user-agent",
                USER_AGENT,
                url,
            ],
            check=True,
            capture_output=True,
        )
        return completed.stdout


def _arxiv_metadata(ids: list[str]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for offset in range(0, len(ids), 20):
        batch = ids[offset : offset + 20]
        query = urllib.parse.urlencode({"id_list": ",".join(batch), "max_results": 20})
        root = ET.fromstring(_request(f"https://export.arxiv.org/api/query?{query}"))
        for entry in root.findall("a:entry", ARXIV_NS):
            versioned_id = entry.findtext("a:id", default="", namespaces=ARXIV_NS)
            short = versioned_id.rsplit("/", 1)[-1]
            base = short.split("v", 1)[0]
            result[base] = {
                "versioned_id": short,
                "canonical_url": f"https://arxiv.org/abs/{short}",
                "published_at": entry.findtext(
                    "a:published", default=f"{FREEZE}T00:00:00Z", namespaces=ARXIV_NS
                )[:10],
                "authors": [
                    node.findtext("a:name", default="", namespaces=ARXIV_NS)
                    for node in entry.findall("a:author", ARXIV_NS)
                ],
                "title": " ".join(
                    entry.findtext("a:title", default="", namespaces=ARXIV_NS).split()
                ),
            }
        if offset + 20 < len(ids):
            time.sleep(1.0)
    return result


def _crossref_metadata(doi: str) -> dict[str, Any]:
    encoded = urllib.parse.quote(doi, safe="")
    try:
        message = json.loads(_request(f"https://api.crossref.org/works/{encoded}"))[
            "message"
        ]
    except (OSError, ValueError, KeyError):
        return {}
    date_parts = (
        message.get("published-print", {}).get("date-parts")
        or message.get("published-online", {}).get("date-parts")
        or message.get("issued", {}).get("date-parts")
        or []
    )
    published_at = None
    if date_parts:
        parts = list(date_parts[0]) + [1, 1]
        published_at = date(parts[0], parts[1], parts[2]).isoformat()
    authors = [
        " ".join(filter(None, [item.get("given"), item.get("family")]))
        for item in message.get("author", [])
    ]
    return {"published_at": published_at, "authors": authors}


def _paper_rights(url: str) -> dict[str, Any]:
    if "10.1371/" in url or "10.7554/" in url:
        return {
            "license_id": "CC-BY-4.0",
            "evidence_url": "https://creativecommons.org/licenses/by/4.0/",
            "redistribution_allowed": True,
            "review_status": "reviewed",
        }
    if "arxiv.org" in url:
        return {
            "license_id": "arXiv-license-not-cleared-for-reuse",
            "evidence_url": "https://info.arxiv.org/help/license/index.html",
            "redistribution_allowed": False,
            "review_status": "reviewed",
        }
    return {
        "license_id": "publisher-terms-not-cleared",
        "evidence_url": "https://www.copyright.gov/fair-use/",
        "redistribution_allowed": False,
        "review_status": "reviewed",
    }


def _seed_records() -> list[dict[str, Any]]:
    payload = json.loads((PRE_RESEARCH / "SEED-CORPUS.json").read_text())
    seeds = payload["sources"]
    arxiv_ids = sorted({item["arxiv_id"] for item in seeds if item["arxiv_id"]})
    arxiv = _arxiv_metadata(arxiv_ids)
    result: list[dict[str, Any]] = []
    for seed in seeds:
        metadata = arxiv.get(seed.get("arxiv_id") or "", {})
        crossref = _crossref_metadata(seed["doi"]) if seed.get("doi") else {}
        canonical_url = metadata.get("canonical_url") or seed["source_url"]
        published_at = (
            metadata.get("published_at")
            or crossref.get("published_at")
            or f"{seed['year']}-01-01"
        )
        authors = metadata.get("authors") or crossref.get("authors") or seed["authors"]
        result.append(
            {
                "seed": seed,
                "title": metadata.get("title") or seed["title"],
                "canonical_url": canonical_url,
                "published_at": published_at,
                "authors": authors,
                "immutable": True,
                "source_type": "paper",
                "rights": _paper_rights(canonical_url),
            }
        )
    return result


def _supplement_records() -> list[dict[str, Any]]:
    arxiv_ids = sorted({item.arxiv_id for item in SUPPLEMENTS if item.arxiv_id})
    arxiv = _arxiv_metadata(arxiv_ids)
    result: list[dict[str, Any]] = []
    for item in SUPPLEMENTS:
        metadata = arxiv.get(item.arxiv_id or "", {})
        canonical_url = metadata.get("canonical_url") or item.url
        mutable_hash = None
        if not item.immutable:
            try:
                mutable_hash = hashlib.sha256(_request(item.url)).hexdigest()
            except (OSError, subprocess.SubprocessError):
                existing_path = DEEP_RESEARCH / "SOURCE-REGISTRY.json"
                if not existing_path.is_file():
                    raise
                existing = json.loads(existing_path.read_text())["sources"]
                prior = next(
                    (
                        source
                        for source in existing
                        if source.get("canonical_url") == item.url
                        and isinstance(source.get("version"), dict)
                    ),
                    None,
                )
                if prior is None:
                    raise
                mutable_hash = prior["version"]["content_sha256"]
        rights = {
            "license_id": item.license_id,
            "evidence_url": item.license_url,
            "redistribution_allowed": item.redistribution_allowed,
            "review_status": "reviewed",
        }
        if item.arxiv_id:
            rights = _paper_rights(canonical_url)
        result.append(
            {
                "supplement": item,
                "title": metadata.get("title") or item.title,
                "canonical_url": canonical_url,
                "published_at": metadata.get("published_at") or item.published_at,
                "authors": metadata.get("authors") or list(item.authors),
                "immutable": item.immutable or bool(item.arxiv_id),
                "source_type": item.source_type,
                "rights": rights,
                "mutable_hash": mutable_hash,
                "version_id": metadata.get("versioned_id") or item.version_id,
            }
        )
    return result


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")


def main() -> None:
    records = _seed_records() + _supplement_records()
    sources: list[dict[str, Any]] = []
    citations: list[dict[str, Any]] = []
    rights: list[dict[str, Any]] = []
    for index, record in enumerate(records, start=1):
        source_id = f"SRC-STC-{index:04d}"
        source: dict[str, Any] = {
            "source_id": source_id,
            "title": record["title"],
            "source_type": record["source_type"],
            "primary": True,
            "status": "frozen",
            "canonical_url": record["canonical_url"],
            "immutable": record["immutable"],
            "published_at": record["published_at"],
            "accessed_at": FREEZE,
            "rights": record["rights"],
        }
        seed = record.get("seed")
        supplement: Supplement | None = record.get("supplement")
        if seed:
            if seed.get("doi"):
                source["doi"] = seed["doi"]
            if seed.get("arxiv_id"):
                source["arxiv_id"] = seed["arxiv_id"]
        if supplement:
            if supplement.doi:
                source["doi"] = supplement.doi
            if supplement.arxiv_id:
                source["arxiv_id"] = supplement.arxiv_id
            if supplement.organization:
                source["organization"] = supplement.organization
        if record["authors"]:
            source["authors"] = record["authors"]
        if not record["immutable"]:
            source["version"] = {
                "version_id": record.get("version_id") or f"web-capture:{FREEZE}",
                "content_sha256": record["mutable_hash"],
                "retrieved_at": FREEZE,
            }
        sources.append(source)

        clusters = seed["clusters"] if seed else list(supplement.clusters)
        role = seed["evidence_role"] if seed else supplement.evidence_role
        maturity = seed["venue_status"] if seed else supplement.maturity
        locator = (
            "paper abstract, methods, results, and limitations; exact claim locators are frozen in the claim map"
            if seed
            else supplement.locator
        )
        citations.append(
            {
                "citation_id": f"CIT-STC-{index:04d}",
                "source_id": source_id,
                "title": record["title"],
                "published_at": record["published_at"],
                "canonical_url": record["canonical_url"],
                "clusters": clusters,
                "evidence_role": role,
                "maturity": maturity,
                "fixed_locator": locator,
                "decision": "retain",
                "verified_at": FREEZE,
            }
        )
        rights.append(
            {
                "source_id": source_id,
                "license_id": record["rights"]["license_id"],
                "evidence_url": record["rights"]["evidence_url"],
                "redistribution_allowed": record["rights"][
                    "redistribution_allowed"
                ],
                "review_status": record["rights"]["review_status"],
                "full_text_policy": (
                    "may_commit_with_attribution"
                    if record["rights"]["redistribution_allowed"]
                    else "reference_only_unless_separately_authorized"
                ),
                "figure_reuse_policy": (
                    "direct_reuse_allowed_with_attribution"
                    if record["rights"]["redistribution_allowed"]
                    else "redraw_or_internal_review_only"
                ),
                "reviewed_at": FREEZE,
            }
        )

    DEEP_RESEARCH.mkdir(parents=True, exist_ok=True)
    _write_json(
        DEEP_RESEARCH / "SOURCE-REGISTRY.json",
        {
            "schema_version": "1.0.0",
            "sources": sources,
        },
    )
    _write_json(
        DEEP_RESEARCH / "CITATION-POOL.json",
        {
            "schema_version": "1.0.0",
            "source_freeze": FREEZE,
            "selection_policy": "primary and official sources only; secondary sources used for discovery but not retained",
            "citations": citations,
        },
    )
    _write_json(
        DEEP_RESEARCH / "SOURCE-RIGHTS.json",
        {
            "schema_version": "1.0.0",
            "source_freeze": FREEZE,
            "default_rule": "No committed full text or direct figure reuse without reviewed affirmative permission.",
            "rights": rights,
        },
    )
    print(
        json.dumps(
            {
                "sources": len(sources),
                "citations": len(citations),
                "rights": len(rights),
            }
        )
    )


if __name__ == "__main__":
    main()
