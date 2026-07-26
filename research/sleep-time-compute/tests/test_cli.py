import subprocess
import tomllib
from pathlib import Path

from stc_research import jsonl_store
from stc_research.cli import main

PACKAGE_ROOT = Path(__file__).parents[1]


def test_version_command(capsys):
    assert main(["version"]) == 0
    assert capsys.readouterr().out.strip() == "stc-research 0.1.0"


def test_package_metadata_and_readme_declare_linux_posix_only_support():
    metadata = tomllib.loads(
        (PACKAGE_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    )
    assert "Operating System :: POSIX :: Linux" in metadata["project"]["classifiers"]

    readme = (PACKAGE_ROOT / "README.md").read_text(encoding="utf-8")
    assert "Linux/POSIX-only" in readme
    assert "fcntl.flock" in readme
    assert "Windows is unsupported" in readme


def test_readme_documents_git_hash_and_writer_retry_contracts():
    readme = (PACKAGE_ROOT / "README.md").read_text(encoding="utf-8")
    normalized = " ".join(readme.split())
    assert "40 lowercase hexadecimal characters" in normalized
    assert "Git SHA-1 object IDs" in normalized
    assert "same persistent sibling lock" in normalized
    assert "CommitOutcomeUnknownError" in normalized
    assert "retry_token" in normalized
    assert "expected_current_digest" in normalized
    assert "canonical_jsonl_digest" in normalized
    assert "StaleWriteError" in normalized
    assert "0644" in normalized
    assert "permission bits" in normalized
    assert "setuid, setgid, and sticky bits" in normalized
    assert "symbolic links and non-regular files" in normalized
    assert "normalized lexical absolute path" in normalized
    assert "destination and referent bytes remain unchanged" in normalized
    assert "zero-byte coordination lock may be created" in normalized
    assert "ending exactly in `.lock`" in normalized
    assert "starting with `.stc-jsonl-`" in normalized
    assert "auto-created parent components" in normalized
    assert "before creating parent directories or materializing records" in normalized
    assert "zero-byte regular file" in normalized
    assert "`O_NOFOLLOW`" in normalized
    assert "outside the cooperative advisory-lock threat model" in normalized


def test_gitignore_covers_only_the_hashed_registry_temp_namespace():
    gitignore = (PACKAGE_ROOT / ".gitignore").read_text(encoding="utf-8")
    assert "registry/.stc-jsonl-*.tmp" in gitignore.splitlines()

    destination = PACKAGE_ROOT / "registry" / "sources.jsonl"
    hashed_temp = f"registry/{jsonl_store._temporary_prefix(destination)}orphan.tmp"
    ignored = subprocess.run(
        ["git", "check-ignore", "--quiet", hashed_temp],
        cwd=PACKAGE_ROOT,
        check=False,
    )
    legacy_unscoped_temp = subprocess.run(
        [
            "git",
            "check-ignore",
            "--quiet",
            "registry/.sources.jsonl.orphan.tmp",
        ],
        cwd=PACKAGE_ROOT,
        check=False,
    )

    assert ignored.returncode == 0
    assert legacy_unscoped_temp.returncode == 1
