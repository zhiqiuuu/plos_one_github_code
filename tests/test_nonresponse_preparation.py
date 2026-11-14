"""Data-free checks for taskbook Q35 nonresponse coding."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "data_preparation", ROOT / "code" / "01_data_preparation.py"
)
assert SPEC and SPEC.loader
DATA_PREPARATION = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(DATA_PREPARATION)


def test_q35_dont_know_and_other_special_codes_remain_distinct() -> None:
    raw = pd.DataFrame(
        {
            "TT4G35A": [1, 5, 8, 9, 5],
            "TT4G35B": [2, 3, 5, 8, 5],
            "TT4G35C": [4, 5, 2, 9, 5],
        }
    )
    result = DATA_PREPARATION.add_nonresponse_indicators(raw)

    assert result[["DK_A", "DK_B", "DK_C"]].values.tolist() == [
        [0, 0, 0],
        [1, 0, 1],
        [0, 1, 0],
        [0, 0, 0],
        [1, 1, 1],
    ]
    assert result["DK_COUNT"].tolist() == [0, 2, 1, 0, 3]
    assert result["DK_ANY"].tolist() == [0, 1, 1, 0, 1]
    assert result["N_VALID"].tolist() == [3, 1, 1, 0, 0]
    assert result["NO_VALID"].tolist() == [0, 0, 0, 1, 1]
    assert result["OTHER_MISSING_ANY"].tolist() == [0, 0, 1, 1, 0]


def test_utility_recode_never_treats_five_as_a_scale_value() -> None:
    recoded = DATA_PREPARATION.recode_capability_item(pd.Series([1, 4, 5, 8, 9]))
    assert recoded.iloc[:2].tolist() == [1.0, 4.0]
    assert np.isnan(recoded.iloc[2:]).all()
