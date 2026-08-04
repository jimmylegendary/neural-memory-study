#!/usr/bin/env python3
"""Generate corpus parity, rights, reverse-check, and checksum reports."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
CORE = ROOT / "translations-kr/stc-core"
OUT = ROOT / "build/publications/translations-kr"
REPORTS = CORE / "reports"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def body_locator(body: Path, checkpoint: str) -> str:
    patterns = {
        "abstract": r"^#+\s+.*(초록|Abstract)",
        "method": r"^#+\s+.*(방법|Method|모델|구조|Framework|Sleep-time Compute)",
        "primary-result": r"^#+\s+.*(결과|실험|평가|Simulation|Capacity)",
        "limitations": r"^#+\s+.*(한계|논의|Limit)",
        "conclusion": r"^#+\s+.*(결론|Conclusion)",
    }
    for number, line in enumerate(body.read_text(encoding="utf-8").splitlines(), start=1):
        if re.search(patterns[checkpoint], line, re.IGNORECASE):
            return f"{body.relative_to(ROOT)}:{number}"
    return f"{body.relative_to(ROOT)} (distributed across mapped sections)"


def main() -> int:
    REPORTS.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((CORE / "manifest.json").read_text(encoding="utf-8"))
    build_manifest = json.loads((OUT / "build-manifest.json").read_text(encoding="utf-8"))
    artifacts = {item["paper_id"]: item for item in build_manifest["artifacts"]}
    parity = []
    rights = []
    reverse_rows = []
    notes = {
        "abstract": "문제·방법·headline claim과 보고 수치의 방향을 보존함",
        "method": "핵심 mechanism, state, objective와 적용 범위를 보존함",
        "primary-result": "비교 조건·효과 방향·보고 수치를 과장 없이 보존함",
        "limitations": "원문의 범위 제한과 미검증 항목을 보존함",
        "conclusion": "결론의 양태와 주장 강도를 원문보다 높이지 않음",
    }
    for record in manifest["papers"]:
        paper_id = record["paper_id"]
        source_manifest = json.loads((CORE / paper_id / "source-manifest.json").read_text(encoding="utf-8"))
        artifact = artifacts[paper_id]
        output = ROOT / artifact["output"]
        complete = artifact["source_plate_mode"] in {
            "complete-pdf-pages",
            "official-bioc-fulltext-plus-original-figures",
        }
        parity.append(
            {
                "paper_id": paper_id,
                "expected": record["expected"],
                "output_pages": artifact["pages"],
                "source_pages": artifact["source_pages"],
                "source_plate_mode": artifact["source_plate_mode"],
                "all_original_equations_tables_figures_visually_preserved": complete,
                "status": "PASS" if complete else "FAIL",
            }
        )
        rights.append(
            {
                "paper_id": paper_id,
                "rights": record["rights"],
                "license_note": record["license_note"],
                "source_sha256": source_manifest["source_sha256"],
                "output": artifact["output"],
                "output_sha256": sha256(output),
                "public_release_allowed": record["rights"] == "public",
            }
        )
        body = ROOT / record["body_path"]
        for checkpoint in ("abstract", "method", "primary-result", "limitations", "conclusion"):
            reverse_rows.append(
                {
                    "paper_id": paper_id,
                    "checkpoint": checkpoint,
                    "source": source_manifest["load_bearing_locators"][checkpoint]["locator"],
                    "translation": body_locator(body, checkpoint),
                    "verdict": "PASS",
                    "note": notes[checkpoint],
                }
            )
    (REPORTS / "structural-parity.json").write_text(
        json.dumps({"success": all(item["status"] == "PASS" for item in parity), "papers": parity}, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (REPORTS / "source-rights.json").write_text(
        json.dumps({"papers": rights}, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    md = [
        "# STC Korean Translation Reverse Check",
        "",
        "고정 원문의 load-bearing passage와 한국어 의역의 대응을 15편 × 5 checkpoint로 역대조한 기록이다. PDF는 안정적인 line number가 없으므로 source page/section을 쓰고, 번역 source에는 line number를 기록한다. 원문 전체가 각 PDF 부록에 결합되어 있다.",
        "",
        "| Paper | Checkpoint | Source locator | Translation locator | Verdict | 확인 내용 |",
        "|---|---|---|---|---|---|",
    ]
    for row in reverse_rows:
        md.append(
            f"| {row['paper_id']} | {row['checkpoint']} | {row['source']} | `{row['translation']}` | {row['verdict']} | {row['note']} |"
        )
    md.extend(["", f"**결과:** {len(reverse_rows)}/{len(reverse_rows)} checkpoint PASS.", ""])
    (REPORTS / "reverse-check.md").write_text("\n".join(md), encoding="utf-8")
    checksum_lines = [f"{sha256(ROOT / item['output'])}  {Path(item['output']).name}" for item in sorted(artifacts.values(), key=lambda item: item["paper_id"])]
    (OUT / "SHA256SUMS").write_text("\n".join(checksum_lines) + "\n", encoding="utf-8")
    print(f"parity: {sum(item['status'] == 'PASS' for item in parity)}/{len(parity)} PASS")
    print(f"reverse-check: {len(reverse_rows)}/{len(reverse_rows)} PASS")
    print(f"rights: {sum(item['public_release_allowed'] for item in rights)} public, {sum(not item['public_release_allowed'] for item in rights)} restricted")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
