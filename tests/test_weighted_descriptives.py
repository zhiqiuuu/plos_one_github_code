"""Unit tests for public Table S0 weighted distribution summaries."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))

from run_revision_package import weighted_descriptives  # noqa: E402


def test_weighted_descriptives_include_q35_items_and_central_moments() -> None:
    data = pd.DataFrame(
        {
            "ai_training": [0.0, 1.0, 1.0, 0.0],
            "ai_usage": [1.0, 0.0, 1.0, 0.0],
            "perceived_lesson": [1.0, 2.0, 4.0, np.nan],
            "perceived_adapt": [2.0, np.nan, 3.0, 4.0],
            "perceived_individual": [4.0, 4.0, np.nan, 1.0],
            "TCHWGT": [1.0, 2.0, 3.0, 4.0],
            "CNTRY": ["A", "A", "B", "B"],
        }
    )

    result = weighted_descriptives(data).set_index("variable")
    assert set(result.index) == {
        "ai_training",
        "ai_usage",
        "TT4G35A",
        "TT4G35B",
        "TT4G35C",
        "mediator",
    }
    assert {
        "weighted_mean",
        "weighted_sd",
        "weighted_skewness",
        "weighted_excess_kurtosis",
    }.issubset(result.columns)

    values = np.array([1.0, 2.0, 4.0])
    weights = np.array([1.0, 2.0, 3.0])
    mean = np.average(values, weights=weights)
    sd = np.sqrt(np.average((values - mean) ** 2, weights=weights))
    expected_skewness = np.average((values - mean) ** 3, weights=weights) / sd**3
    expected_excess_kurtosis = np.average((values - mean) ** 4, weights=weights) / sd**4 - 3.0
    item_a = result.loc["TT4G35A"]
    assert item_a["n_unweighted"] == 3
    assert item_a["weighted_mean"] == pytest.approx(mean)
    assert item_a["weighted_sd"] == pytest.approx(sd)
    assert item_a["weighted_skewness"] == pytest.approx(expected_skewness)
    assert item_a["weighted_excess_kurtosis"] == pytest.approx(expected_excess_kurtosis)
    assert "codes 5, 8, and 9 excluded" in item_a["valid_response_rule"]

    composite = result.loc["mediator"]
    assert composite["n_unweighted"] == 4
    assert composite["source_variable"] == "TT4G35A-C"
    assert "at least one valid item" in composite["valid_response_rule"]
