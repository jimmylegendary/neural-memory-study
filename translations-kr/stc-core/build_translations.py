#!/usr/bin/env python3
"""Build the rights-aware STC Korean translation corpus reproducibly."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import re
import shutil
import subprocess
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
CORE = ROOT / "translations-kr/stc-core"
MANIFEST = CORE / "manifest.json"
OUT = ROOT / "build/publications/translations-kr"
WORK = ROOT / "build/translation"
EPOCH = "1785888000"


@dataclass(frozen=True)
class TranslationBuild:
    paper_id: str
    output: str
    pages: int
    sha256: str
    source_sha256: str
    source_pages: int | None
    source_plate_mode: str
    rights: str


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def page_count(path: Path) -> int:
    text = subprocess.check_output(["pdfinfo", str(path)], text=True)
    match = re.search(r"^Pages:\s+(\d+)", text, re.MULTILINE)
    if not match:
        raise RuntimeError(f"Could not determine page count: {path}")
    return int(match.group(1))


def load_manifest() -> dict:
    return json.loads(MANIFEST.read_text(encoding="utf-8"))


def record_for(paper_id: str) -> dict:
    for record in load_manifest()["papers"]:
        if record["paper_id"] == paper_id:
            return record
    raise KeyError(paper_id)


def latex_escape(text: str) -> str:
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(char, char) for char in text)


def _source_manifest(record: dict) -> dict:
    source = ROOT / record["source_path"]
    actual_hash = sha256(source)
    if actual_hash != record["source_sha256"]:
        raise ValueError(f"source-hash-mismatch:{record['paper_id']}:{actual_hash}")
    result = {
        "paper_id": record["paper_id"],
        "title": record["original_title"],
        "authors": record["authors"],
        "version": record["version"],
        "source_path": record["source_path"],
        "source_type": record["source_type"],
        "source_sha256": actual_hash,
        "selection_sha256": record.get("selection_sha256", actual_hash),
        "source_exception": record.get("source_exception"),
        "rights": record["rights"],
        "license_note": record["license_note"],
        "expected": record["expected"],
    }
    if record["source_type"] == "pdf":
        result["pages"] = page_count(source)
        result["source_plate_mode"] = "complete-pdf-pages"
        extracted = subprocess.check_output(["pdftotext", "-layout", str(source), "-"], text=True, errors="replace")
        source_pages = extracted.split("\f")
        patterns = {
            "abstract": [r"^\s*(Abstract|A BSTRACT)\s*$"],
            "method": [r"^\s*\d*\.?\s*(Method|Methods|Proposed Method|Model|Framework|Sleep-time Compute|Knowledge Graph Construction)\b"],
            "primary-result": [r"^\s*\d*\.?\s*(Experiment|Experiments|Results|Evaluation|Model Simulations|Model Capacity)\b"],
            "limitations": [r"^\s*(Limitations|Limitation and Future Work|Discussion and Limitations|Discussion)\b"],
            "conclusion": [r"^\s*\d*\.?\s*(Conclusion|Discussion and conclusions|Discussion)\b"],
        }
        locators = {}
        for name, candidates in patterns.items():
            for page_number, page_text in enumerate(source_pages, start=1):
                if any(re.search(candidate, page_text, re.IGNORECASE | re.MULTILINE) for candidate in candidates):
                    locators[name] = {"page": page_number, "locator": f"source PDF p.{page_number}"}
                    break
            if name not in locators:
                locators[name] = {"page": None, "locator": "not separately titled; inspect complete source plate"}
        result["load_bearing_locators"] = locators
        result["figure_pages"] = [
            index
            for index, page_text in enumerate(source_pages, start=1)
            if re.search(r"\b(?:Figure|Fig\.)\s*\d+", page_text, re.IGNORECASE)
        ]
        result["table_pages"] = [
            index
            for index, page_text in enumerate(source_pages, start=1)
            if re.search(r"\bTable\s*\d+", page_text, re.IGNORECASE)
        ]
    else:
        tree = ET.parse(source)
        captions = []
        sections = []
        for passage in tree.getroot().iter("passage"):
            infons = {item.attrib.get("key"): item.text for item in passage.findall("infon")}
            text = " ".join((passage.findtext("text") or "").split())
            if infons.get("type") == "fig_caption":
                captions.append({"file": infons.get("file"), "caption": text})
            if infons.get("type", "").startswith("title") or infons.get("section_type") in {"TITLE", "ABSTRACT"}:
                if text and text not in sections:
                    sections.append(text)
        result["pages"] = None
        result["source_plate_mode"] = "official-bioc-fulltext-plus-original-figures"
        result["figure_captions"] = captions
        result["section_inventory"] = sections
        result["load_bearing_locators"] = {
            "abstract": {"page": None, "locator": "official BioC ABSTRACT passages"},
            "method": {"page": None, "locator": "official BioC Methods passages"},
            "primary-result": {"page": None, "locator": "official BioC Model Simulations passages"},
            "limitations": {"page": None, "locator": "official BioC Discussion passages"},
            "conclusion": {"page": None, "locator": "official BioC Discussion closing passages"},
        }
    return result


def prepare_record(record: dict) -> dict:
    source_manifest = _source_manifest(record)
    paper_dir = CORE / record["paper_id"]
    paper_dir.mkdir(parents=True, exist_ok=True)
    path = paper_dir / "source-manifest.json"
    path.write_text(json.dumps(source_manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return source_manifest


def _appendix_tex(record: dict, source_manifest: dict, build_dir: Path) -> Path:
    appendix = build_dir / "source-plate.tex"
    lines = [
        r"\clearpage",
        r"\chapter*{원문 구조·그림 보존 부록}",
        r"\addcontentsline{toc}{chapter}{원문 구조·그림 보존 부록}",
        r"\begin{translationnotice}",
        r"아래 부록은 고정 원문의 equation, table, figure, caption과 section 배치를 대조하기 위한 검증판이다. ",
        latex_escape(record["license_note"]),
        r"\end{translationnotice}",
    ]
    if record["source_type"] == "pdf":
        source = (ROOT / record["source_path"]).resolve()
        lines.extend(
            [
                r"\clearpage",
                r"\includepdf[pages=-,pagecommand={\thispagestyle{plain}},fitpaper=true]{" + str(source) + "}",
            ]
        )
    else:
        captions = source_manifest.get("figure_captions", [])
        figure_dir = CORE / record["paper_id"] / "figures"
        for index, item in enumerate(captions, start=1):
            image = figure_dir / f"fig{index:02d}.jpg"
            if not image.exists():
                raise FileNotFoundError(f"missing-source-figure:{image}")
            lines.extend(
                [
                    r"\begin{figure}[p]",
                    r"\centering",
                    r"\includegraphics[width=0.94\textwidth,height=0.72\textheight,keepaspectratio]{" + str(image.resolve()) + "}",
                    r"\caption*{Original Figure " + str(index) + ": " + latex_escape(item["caption"]) + "}",
                    r"\end{figure}",
                    r"\clearpage",
                ]
            )
    appendix.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return appendix


def _env() -> dict[str, str]:
    env = os.environ.copy()
    texinputs = str((CORE / "common").resolve()) + os.pathsep + env.get("TEXINPUTS", "")
    env.update({"SOURCE_DATE_EPOCH": EPOCH, "FORCE_SOURCE_DATE": "1", "TZ": "UTC", "TEXINPUTS": texinputs})
    return env


def build_translation(paper_id: str) -> TranslationBuild:
    record = record_for(paper_id)
    source_manifest = prepare_record(record)
    body = ROOT / record["body_path"]
    if not body.exists():
        raise FileNotFoundError(f"missing-translation-body:{body}")
    build_dir = WORK / paper_id
    if build_dir.exists():
        shutil.rmtree(build_dir)
    build_dir.mkdir(parents=True)
    appendix = _appendix_tex(record, source_manifest, build_dir)
    prepared_body = build_dir / "prepared-body.md"
    body_text = body.read_text(encoding="utf-8")
    body_text = re.sub(r"(?m)^\\\[\s*$", "$$", body_text)
    body_text = re.sub(r"(?m)^\\\]\s*$", "$$", body_text)
    body_text = re.sub(
        r"!\[([^\]]*)\]\([^\n)]*\)",
        r"> **원문 그림 위치:** \1  \\\n+> 완전한 원문 그림과 caption은 뒤의 원문 구조·그림 보존 부록에서 대조할 수 있다.",
        body_text,
    )
    prepared_body.write_text(body_text, encoding="utf-8")
    OUT.mkdir(parents=True, exist_ok=True)
    destination = OUT / record["output"]
    command = [
        "pandoc",
        str(prepared_body),
        "--from=markdown+raw_tex+tex_math_dollars",
        "--template",
        str(CORE / "common/template.tex"),
        "--lua-filter",
        str(ROOT / "translations-kr/build/breakable.lua"),
        "--lua-filter",
        str(ROOT / "translations-kr/build/mathfit.lua"),
        "--include-after-body",
        str(appendix),
        "--pdf-engine=lualatex",
        "--pdf-engine-opt=-interaction=nonstopmode",
        "--pdf-engine-opt=-halt-on-error",
        "-V",
        f"paper-id={record['paper_id']}",
        "-V",
        f"korean-title={record['korean_title']}",
        "-V",
        f"original-title={record['original_title']}",
        "-V",
        f"author={record['authors']}",
        "-V",
        f"source-version={record['version']}",
        "-V",
        f"rights={record['rights']}",
        "-V",
        f"license-note={record['license_note']}",
        "--resource-path",
        str(ROOT),
        "-o",
        str(destination),
    ]
    log = build_dir / "pandoc.log"
    with log.open("w", encoding="utf-8") as stream:
        subprocess.run(command, cwd=ROOT, env=_env(), stdout=stream, stderr=subprocess.STDOUT, check=True, timeout=900)
    artifact = TranslationBuild(
        paper_id=paper_id,
        output=str(destination.relative_to(ROOT)),
        pages=page_count(destination),
        sha256=sha256(destination),
        source_sha256=source_manifest["source_sha256"],
        source_pages=source_manifest["pages"],
        source_plate_mode=source_manifest["source_plate_mode"],
        rights=record["rights"],
    )
    qa_path = CORE / paper_id / "translation-qa.json"
    qa_path.write_text(json.dumps(asdict(artifact), ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return artifact


def selected_records(existing_only: bool = False) -> list[dict]:
    records = load_manifest()["papers"]
    if existing_only:
        records = [record for record in records if record["body_kind"].startswith("existing")]
    return records


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--paper")
    group.add_argument("--existing-only", action="store_true")
    group.add_argument("--all", action="store_true")
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    records = [record_for(args.paper)] if args.paper else selected_records(args.existing_only)
    results = []
    if args.prepare_only:
        for record in records:
            prepare_record(record)
            print(f"{record['paper_id']}: source manifest prepared")
    elif len(records) > 1:
        with concurrent.futures.ThreadPoolExecutor(max_workers=min(4, len(records))) as pool:
            futures = {pool.submit(build_translation, record["paper_id"]): record["paper_id"] for record in records}
            for future in concurrent.futures.as_completed(futures):
                artifact = future.result()
                results.append(asdict(artifact))
                print(f"{artifact.paper_id}: {artifact.pages} pages {artifact.sha256}")
    else:
        artifact = build_translation(records[0]["paper_id"])
        results.append(asdict(artifact))
        print(f"{artifact.paper_id}: {artifact.pages} pages {artifact.sha256}")
    if results:
        index = OUT / "build-manifest.json"
        existing = []
        if index.exists() and args.paper:
            existing = json.loads(index.read_text(encoding="utf-8")).get("artifacts", [])
            ids = {item["paper_id"] for item in results}
            existing = [item for item in existing if item["paper_id"] not in ids]
        index.write_text(
            json.dumps({"success": True, "artifacts": sorted(existing + results, key=lambda item: item["paper_id"])}, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
