"""Lightweight integrity tests for committed, non-restricted release outputs."""

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def test_primary_paths_and_replicates() -> None:
    output = ROOT / "results" / "output_final"
    main = pd.read_csv(output / "B01" / "main_paths_brr.csv")
    replicates = pd.read_csv(output / "B01" / "brr_replicate_estimates.csv")
    assert set(main["parameter"]) == {"a", "b", "c", "c_prime"}
    assert len(replicates) == 400
    assert replicates["status"].eq("converged").all()


def test_public_data_excludes_linkage_keys() -> None:
    public_data = pd.read_csv(ROOT / "data" / "derived" / "public_system_summary.csv")
    forbidden = {"IDCNTRY", "IDSCHOOL", "IDTEACH"}
    assert forbidden.isdisjoint(public_data.columns)
    assert len(public_data) > 0
