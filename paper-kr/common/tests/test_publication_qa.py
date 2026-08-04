import importlib.util
import json
import subprocess
import re
from pathlib import Path


REPO_ROOT = Path(__file__).parents[3]
QA_PATH = REPO_ROOT / "paper-kr/common/qa_publications.py"
BUILDER_PATH = REPO_ROOT / "paper-kr/common/build_publications.py"


def _load_qa():
    assert QA_PATH.exists(), "publication QA has not been implemented"
    spec = importlib.util.spec_from_file_location("stc_publication_qa", QA_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_builder():
    assert BUILDER_PATH.exists(), "publication builder has not been implemented"
    spec = importlib.util.spec_from_file_location("stc_publication_builder", BUILDER_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_latex_environment_exposes_shared_bibliography_to_tex_and_bibtex():
    module = _load_builder()

    environment = module._latex_environment(REPO_ROOT)
    common = str((REPO_ROOT / "paper-kr/common").resolve())

    assert common in environment["TEXINPUTS"].split(":")
    assert common in environment["BIBINPUTS"].split(":")


def test_bgref_must_resolve_to_background_label():
    module = _load_qa()
    study = r"\bgref{lora} \bgref{missing-concept}"
    registry = {"lora": "bg:lora"}

    report = module.validate_cross_document_refs(study, registry)

    assert report.referenced_concepts == ("lora", "missing-concept")
    assert report.unresolved_concepts == ("missing-concept",)


def test_public_figure_requires_reuse_permission():
    module = _load_qa()
    record = {
        "figure_id": "STC-F999",
        "mode": "direct_reuse",
        "public": True,
        "license": {"reuse_allowed": False, "review_status": "pending"},
    }

    diagnostics = module.validate_figure(record, audience="public")

    assert "public-reuse-not-authorized" in diagnostics


def test_original_redraw_is_public_safe():
    module = _load_qa()
    record = {
        "figure_id": "STC-F001",
        "mode": "redrawn",
        "public": True,
        "license": {"reuse_allowed": True, "review_status": "reviewed"},
    }

    assert module.validate_figure(record, audience="public") == ()


def test_source_validation_rejects_unknown_claims_and_figures(tmp_path):
    module = _load_qa()
    tex = tmp_path / "section.tex"
    tex.write_text(
        r"\claim{STC-C001}\claim{STC-C999}\stcfigure{STC-F001}\stcfigure{STC-F999}",
        encoding="utf-8",
    )
    claim_map = {"claims": [{"claim_id": "STC-C001"}]}
    figure_ledger = {"figures": [{"figure_id": "STC-F001"}]}

    report = module.validate_source_tree(
        tmp_path,
        concept_registry={},
        claim_map=claim_map,
        figure_ledger=figure_ledger,
    )

    assert report.unknown_claims == ("STC-C999",)
    assert report.unknown_figures == ("STC-F999",)


def test_background_concept_anchors_are_collected_separately_from_study_refs(tmp_path):
    module = _load_qa()
    tex = tmp_path / "background.tex"
    tex.write_text(
        r"\concept{loss}{손실}\concept{gradient}{기울기}\concept{unknown}{미등록}",
        encoding="utf-8",
    )
    registry = {
        "concepts": [
            {"concept_id": "loss", "label": "bg:loss"},
            {"concept_id": "gradient", "label": "bg:gradient"},
        ]
    }

    report = module.validate_source_tree(
        tmp_path,
        concept_registry=registry,
        claim_map={"claims": []},
        figure_ledger={"figures": []},
    )

    assert report.declared_concepts == ("loss", "gradient", "unknown")
    assert report.unknown_declared_concepts == ("unknown",)


def test_concept_registry_is_dependency_closed(tmp_path):
    module = _load_qa()
    path = tmp_path / "concepts.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": "1.0.0",
                "concepts": [
                    {
                        "concept_id": "gradient",
                        "label": "bg:gradient",
                        "definition": "목적함수의 기울기",
                        "prerequisites": ["loss"],
                        "first_study_section": "STC-S03",
                        "glossary_term": "기울기",
                    }
                ],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    diagnostics = module.validate_concept_registry(path)

    assert "missing-prerequisite:gradient:loss" in diagnostics


def test_common_style_only_requires_installed_tex_packages():
    style = (REPO_ROOT / "paper-kr/common/stc-common.sty").read_text(encoding="utf-8")
    packages = []
    for line in style.splitlines():
        if not line.startswith(r"\RequirePackage"):
            continue
        names = line.rsplit("{", 1)[-1].rstrip("}").split(",")
        packages.extend(name.strip() for name in names)
    missing = [
        package
        for package in packages
        if subprocess.run(
            ["kpsewhich", f"{package}.sty"], capture_output=True, text=True
        ).returncode
        != 0
    ]

    assert missing == []


def test_background_figure_generator_builds_all_vector_pdfs(tmp_path):
    figure_builder = (
        REPO_ROOT
        / "paper-kr/sleep-time-compute-training-background/figures/build_figures.py"
    )
    spec = importlib.util.spec_from_file_location("stc_background_figures", figure_builder)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.OUT = tmp_path

    module.main()

    expected = {
        "learning-map.pdf",
        "training-regimes.pdf",
        "continual-learning-map.pdf",
        "rl-sleep-policy.pdf",
        "training-memory-cost.pdf",
    }
    assert {path.name for path in tmp_path.glob("*.pdf")} == expected
    for path in tmp_path.glob("*.pdf"):
        info = subprocess.run(
            ["pdfinfo", str(path)], check=True, capture_output=True, text=True
        ).stdout
        assert "Pages:           1" in info


def test_background_glossary_generator_covers_every_registered_concept(tmp_path):
    generator = (
        REPO_ROOT
        / "paper-kr/sleep-time-compute-training-background/build_glossary.py"
    )
    spec = importlib.util.spec_from_file_location("stc_background_glossary", generator)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    registry = REPO_ROOT / "paper-kr/common/concept-registry.json"
    output = tmp_path / "glossary.tex"

    module.build_glossary(registry, output)

    source = output.read_text(encoding="utf-8")
    concepts = json.loads(registry.read_text(encoding="utf-8"))["concepts"]
    assert source.count(r"\hypertarget{glossary:") == len(concepts)
    assert all(item["concept_id"] in source for item in concepts)


def test_source_bibliography_generator_covers_frozen_registry(tmp_path):
    generator = REPO_ROOT / "paper-kr/common/build_source_bibliography.py"
    spec = importlib.util.spec_from_file_location("stc_source_bibliography", generator)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    registry = (
        REPO_ROOT
        / "research/sleep-time-compute/deep-research/SOURCE-REGISTRY.json"
    )
    output = tmp_path / "source-registry.bib"

    module.build_bibliography(registry, output)

    source = output.read_text(encoding="utf-8")
    records = json.loads(registry.read_text(encoding="utf-8"))["sources"]
    assert source.count("@misc{srcstc") == len(records)
    assert all(item["source_id"].lower().replace("-", "") in source for item in records)
    assert r"official\_release\_note" in source
    assert "type=official_release_note" not in source


def test_study_figure_generator_builds_canonical_fifteen_vector_pdfs(tmp_path):
    figure_builder = (
        REPO_ROOT / "paper-kr/sleep-time-compute-study/figures/build_figures.py"
    )
    spec = importlib.util.spec_from_file_location("stc_study_figures", figure_builder)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.OUT = tmp_path

    module.main()

    expected = {f"stc-f{index:03d}.pdf" for index in range(1, 16)}
    assert {path.name for path in tmp_path.glob("*.pdf")} == expected
    for path in tmp_path.glob("*.pdf"):
        info = subprocess.run(
            ["pdfinfo", str(path)], check=True, capture_output=True, text=True
        ).stdout
        assert "Pages:           1" in info


def test_claim_table_generator_covers_every_frozen_claim(tmp_path):
    generator = REPO_ROOT / "paper-kr/sleep-time-compute-study/build_claim_table.py"
    spec = importlib.util.spec_from_file_location("stc_claim_table", generator)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    claim_map = REPO_ROOT / "claims/stc-study/claim-map.json"
    output = tmp_path / "claim-table.tex"

    module.build_claim_table(claim_map, output)

    source = output.read_text(encoding="utf-8")
    claims = json.loads(claim_map.read_text(encoding="utf-8"))["claims"]
    assert source.count(r"\hypertarget{claim-ledger:") == len(claims)
    assert all(item["claim_id"] in source for item in claims)


def test_concept_anchors_are_not_declared_inside_display_math():
    source_root = REPO_ROOT / "paper-kr/sleep-time-compute-training-background"
    violations = []
    display_math = re.compile(
        r"\\begin\{(?:equation\*?|align\*?|gather\*?)\}(.*?)"
        r"\\end\{(?:equation\*?|align\*?|gather\*?)\}",
        re.DOTALL,
    )
    for path in source_root.rglob("*.tex"):
        source = path.read_text(encoding="utf-8")
        if any(r"\concept{" in match.group(1) for match in display_math.finditer(source)):
            violations.append(str(path.relative_to(REPO_ROOT)))

    assert violations == []
