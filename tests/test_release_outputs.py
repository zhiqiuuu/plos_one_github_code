"""Integrity tests for the code-only public release."""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_required_scripts_exist() -> None:
    required = {
        "01_data_preparation.py",
        "run_all.py",
        "run_revision_package.py",
        "generate_design_adjusted_figure.py",
        "export_public_aggregates.py",
    }
    assert required == {path.name for path in (ROOT / "code").glob("*.py")}


def test_release_files_exclude_restricted_row_level_data() -> None:
    forbidden_names = {"data_recoded.csv", "ttgintt4.csv", "atgintt4.csv", "btgintt4.csv", "ctgintt4.csv"}
    release_names = {path.name for path in ROOT.rglob("*") if path.is_file()}
    assert forbidden_names.isdisjoint(release_names)
    assert not (ROOT / "data").exists()
    assert not (ROOT / "results").exists()


def test_version_and_archival_instructions_exist() -> None:
    assert (ROOT / "VERSION").read_text(encoding="utf-8").strip()
    archive = (ROOT / "ARCHIVING.md").read_text(encoding="utf-8")
    assert "PENDING_ZENODO_ARCHIVAL" in archive
