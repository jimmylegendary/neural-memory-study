#!/usr/bin/env python3
"""Deterministically build the twelve STC easy companions and combined PDF."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
CORE = ROOT / "easy/sleep-time-compute"
MANIFEST_PATH = CORE / "manifest.json"
WORK = ROOT / "build/easy-companion"
INDIVIDUAL_OUT = CORE / "pdf"
PUBLICATION_OUT = ROOT / "build/publications"
EPOCH = "1785888000"


@dataclass(frozen=True)
class Artifact:
    booklet_id: str
    output: str
    pages: int
    sha256: str


def _json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def page_count(path: Path) -> int:
    out = subprocess.check_output(["pdfinfo", str(path)], text=True, errors="replace")
    match = re.search(r"^Pages:\s+(\d+)", out, flags=re.MULTILINE)
    if not match:
        raise RuntimeError(f"could-not-read-page-count:{path}")
    return int(match.group(1))


def load_catalogs() -> tuple[dict, dict, dict, dict]:
    manifest = _json(MANIFEST_PATH)
    claims = {x["claim_id"]: x for x in _json(ROOT / "claims/stc-study/claim-map.json")["claims"]}
    figures = {x["figure_id"]: x for x in _json(ROOT / "claims/stc-study/figure-ledger.json")["figures"]}
    concepts = {x["concept_id"]: x for x in _json(ROOT / "paper-kr/common/concept-registry.json")["concepts"]}
    return manifest, claims, figures, concepts


def record_for(booklet_id: str) -> dict:
    manifest, _, _, _ = load_catalogs()
    for record in manifest["booklets"]:
        if record["id"] == booklet_id:
            return record
    raise KeyError(booklet_id)


def _link_target(kind: str, output: Path) -> str:
    filename = (
        "TRAINING-BACKGROUND-FOR-SLEEP-TIME-COMPUTE-KR.pdf"
        if kind == "background"
        else "SLEEP-TIME-COMPUTE-STRATEGIC-STUDY-KR.pdf"
    )
    target = PUBLICATION_OUT / filename
    return os.path.relpath(target, output.parent).replace(os.sep, "/")


def prepare_markdown(record: dict, output: Path, combined: bool = False) -> str:
    _, claims, figures, concepts = load_catalogs()
    body = (CORE / record["file"]).read_text(encoding="utf-8")
    background_target = _link_target("background", output)
    study_target = _link_target("study", output)

    def bg_replace(match: re.Match[str]) -> str:
        concept_id, label = match.group(1), match.group(2)
        concept = concepts[concept_id]
        destination = concept["label"]
        return f"[{label}]({background_target}#nameddest={destination})"

    def study_replace(match: re.Match[str]) -> str:
        section_id, label = match.group(1), match.group(2)
        return f"[{label}]({study_target}#nameddest={section_id})"

    def claim_replace(match: re.Match[str]) -> str:
        claim_id = match.group(1)
        claim = claims[claim_id]
        confidence = claim["confidence"]
        return f"**{claim_id}** · {claim['claim_type']} · confidence {confidence}"

    def figure_replace(match: re.Match[str]) -> str:
        figure_id = match.group(1)
        figure = figures[figure_id]
        image = ROOT / "paper-kr/sleep-time-compute-study/figures" / f"{figure_id.lower()}.pdf"
        caption = (
            f"{figure_id}. {figure['title']}. 원 Study의 저자 작성 합성 figure이며, "
            "사실 주장의 인용은 원 Study와 figure ledger를 따른다."
        )
        return f"![{caption}]({image.as_posix()})"

    body = re.sub(r"\{\{BG:([a-z0-9-]+)\|([^}]+)\}\}", bg_replace, body)
    body = re.sub(r"\{\{STUDY:(STC-S\d+)\|([^}]+)\}\}", study_replace, body)
    body = re.sub(r"\{\{CLAIM:(STC-C\d+)\}\}", claim_replace, body)
    body = re.sub(r"\{\{FIG:(STC-F\d+)\}\}", figure_replace, body)
    if re.search(r"\{\{(?:BG|STUDY|CLAIM|FIG):", body):
        raise RuntimeError(f"unresolved-placeholder:{record['id']}")
    if combined:
        return f"\\hypertarget{{easy:{record['id']}}}{{}}\n\n{body}\n\n\\clearpage\n"
    return body


def _env() -> dict[str, str]:
    env = os.environ.copy()
    env.update({"SOURCE_DATE_EPOCH": EPOCH, "FORCE_SOURCE_DATE": "1", "TZ": "UTC"})
    return env


def _pandoc(source: Path, destination: Path, title: str, booklet_id: str, include_route: bool = False) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    command = [
        "pandoc",
        str(source),
        "--from=markdown+raw_tex+tex_math_dollars",
        "--template",
        str(CORE / "template.tex"),
        "--lua-filter",
        str(ROOT / "easy/build/callouts.lua"),
        "--lua-filter",
        str(ROOT / "easy/build/mathfit.lua"),
        "--pdf-engine=lualatex",
        "--pdf-engine-opt=-interaction=nonstopmode",
        "--pdf-engine-opt=-halt-on-error",
        "--toc",
        "-V",
        f"title={title}",
        "-V",
        f"booklet-id={booklet_id}",
        "--resource-path",
        str(ROOT),
    ]
    if include_route:
        command.extend(["--include-before-body", str(CORE / "combined.tex")])
    command.extend(["-o", str(destination)])
    log = WORK / f"{destination.stem}.log"
    with log.open("w", encoding="utf-8") as stream:
        subprocess.run(command, cwd=ROOT, env=_env(), stdout=stream, stderr=subprocess.STDOUT, check=True, timeout=900)


def build_booklet(record: dict) -> Artifact:
    destination = INDIVIDUAL_OUT / f"{record['id']}-{record['slug']}.pdf"
    prepared = WORK / f"{record['id']}.md"
    prepared.write_text(prepare_markdown(record, destination), encoding="utf-8")
    _pandoc(prepared, destination, record["title"], record["id"])
    return Artifact(record["id"], str(destination.relative_to(ROOT)), page_count(destination), sha256(destination))


def build_combined(records: list[dict]) -> Artifact:
    destination = PUBLICATION_OUT / "SLEEP-TIME-COMPUTE-EASY-COMPANION-KR.pdf"
    prepared = WORK / "combined.md"
    parts = [prepare_markdown(record, destination, combined=True) for record in records]
    prepared.write_text("\n".join(parts), encoding="utf-8")
    _pandoc(prepared, destination, "Sleep-Time Compute 쉬운 설명본", "E01–E12", include_route=True)
    return Artifact("COMBINED", str(destination.relative_to(ROOT)), page_count(destination), sha256(destination))


def select_records(spec: str | None) -> list[dict]:
    manifest, _, _, _ = load_catalogs()
    if not spec:
        return manifest["booklets"]
    wanted = {item.strip() for item in spec.split(",") if item.strip()}
    records = [r for r in manifest["booklets"] if r["id"] in wanted]
    missing = wanted - {r["id"] for r in records}
    if missing:
        raise SystemExit(f"unknown-booklet:{','.join(sorted(missing))}")
    return records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--booklets", help="Comma-separated IDs")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--combined", action="store_true")
    args = parser.parse_args()
    WORK.mkdir(parents=True, exist_ok=True)
    INDIVIDUAL_OUT.mkdir(parents=True, exist_ok=True)
    records = select_records(None if args.all else args.booklets)
    artifacts = [build_booklet(record) for record in records]
    if args.combined:
        if len(records) != 12:
            raise SystemExit("combined-build-requires-all-12-booklets")
        artifacts.append(build_combined(records))
    manifest_out = {
        "schema_version": "1.0.0",
        "source_freeze": "2026-08-05",
        "artifacts": [asdict(item) for item in artifacts],
    }
    (CORE / "reports/build-manifest.json").write_text(
        json.dumps(manifest_out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    checksum_lines = [f"{item.sha256}  {item.output}" for item in artifacts]
    (CORE / "reports/SHA256SUMS").write_text("\n".join(checksum_lines) + "\n", encoding="utf-8")
    for item in artifacts:
        print(f"{item.booklet_id}: {item.pages} pages {item.sha256}")


if __name__ == "__main__":
    main()
