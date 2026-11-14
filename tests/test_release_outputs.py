"""Integrity tests for the aggregate-only public release."""

import math
from pathlib import Path

import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[1]


def test_required_scripts_exist() -> None:
    required = {
        "01_data_preparation.py",
        "run_all.py",
        "run_revision_package.py",
        "run_khb_design.py",
        "run_nonresponse_indicator.py",
        "generate_design_adjusted_figure.py",
        "export_public_aggregates.py",
    }
    assert required == {path.name for path in (ROOT / "code").glob("*.py")}


def test_release_files_exclude_restricted_row_level_data() -> None:
    forbidden_names = {"data_recoded.csv", "ttgintt4.csv", "atgintt4.csv", "btgintt4.csv", "ctgintt4.csv"}
    release_names = {path.name for path in ROOT.rglob("*") if path.is_file()}
    assert forbidden_names.isdisjoint(release_names)
    assert not (ROOT / "data").exists()


def test_public_results_are_aggregate_and_present() -> None:
    output = ROOT / "results" / "output_final"
    assert (output / "B05" / "Table2_design_adjusted_association_paths.csv").exists()
    assert (output / "B02" / "khb_primary_brr.csv").exists()
    assert (output / "B02" / "khb_complete_case_brr.csv").exists()
    assert (output / "B02" / "khb_replicate_estimates.csv").exists()
    assert (output / "B03" / "nonresponse_indicator_models_brr.csv").exists()
    assert (output / "B03" / "nonresponse_indicator_replicates.csv").exists()
    assert (output / "B05" / "TableS5_KHB.csv").exists()
    assert (output / "B05" / "TableS6_nonresponse_indicator.csv").exists()
    assert (ROOT / "figures" / "Fig1_design_adjusted_association_paths.png").exists()
    forbidden_columns = {"IDCNTRY", "IDSCHOOL", "IDTEACH"}
    for table in output.rglob("*.csv"):
        header = table.read_text(encoding="utf-8").splitlines()
        if header:
            assert forbidden_columns.isdisjoint(set(header[0].split(",")))


def test_published_khb_brr_recalculates_and_obeys_identities() -> None:
    output = ROOT / "results" / "output_final" / "B02"
    summary = pd.concat(
        [
            pd.read_csv(output / "khb_primary_brr.csv"),
            pd.read_csv(output / "khb_complete_case_brr.csv"),
        ],
        ignore_index=True,
    )
    replicates = pd.read_csv(output / "khb_replicate_estimates.csv")
    expected_parameters = {"total_raw", "total_adjusted", "direct", "indirect", "rescaling"}
    assert set(summary["parameter"]) == expected_parameters
    counts = replicates.groupby(["analysis", "parameter"]).agg(
        rows=("replicate", "size"), unique_replicates=("replicate", "nunique")
    )
    assert (counts["rows"] == 100).all()
    assert (counts["unique_replicates"] == 100).all()
    assert replicates["status"].eq("converged").all()
    for row in summary.itertuples(index=False):
        values = replicates.loc[
            (replicates["analysis"] == row.analysis)
            & (replicates["parameter"] == row.parameter),
            "estimate",
        ]
        recalculated = math.sqrt(((values - row.estimate) ** 2).sum() / (100 * 0.5**2))
        assert row.brr_fay_se == pytest.approx(recalculated, abs=1e-12)
    wide = summary.pivot(index="analysis", columns="parameter", values="estimate")
    assert ((wide["total_adjusted"] - wide["direct"] - wide["indirect"]).abs() < 1e-10).all()
    assert ((wide["rescaling"] - wide["total_adjusted"] + wide["total_raw"]).abs() < 1e-10).all()


def test_published_nonresponse_results_match_locked_coding_checks() -> None:
    output = ROOT / "results" / "output_final" / "B03"
    input_checks = pd.read_csv(output / "nonresponse_indicator_input_anchor_checks.csv")
    model_checks = pd.read_csv(output / "nonresponse_indicator_model_anchor_checks.csv")
    diagnostics = pd.read_csv(output / "nonresponse_indicator_model_diagnostics.csv")
    replicates = pd.read_csv(output / "nonresponse_indicator_replicates.csv")
    assert input_checks["passed"].all()
    assert model_checks["passed"].all()
    b1 = diagnostics.loc[diagnostics["model"].eq("B1")].iloc[0]
    assert b1["matrix_rank"] == b1["n_design_columns"] == 59
    expected = {"A0": 2, "A1": 3, "B0": 1, "B1": 4}
    counts = replicates.groupby("model").size().to_dict()
    assert counts == {model: parameters * 100 for model, parameters in expected.items()}
    assert replicates["status"].eq("converged").all()


def test_version_and_archival_instructions_exist() -> None:
    assert (ROOT / "VERSION").read_text(encoding="utf-8").strip()
    archive = (ROOT / "ARCHIVING.md").read_text(encoding="utf-8")
    assert "PENDING_ZENODO_ARCHIVAL" in archive
