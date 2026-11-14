"""Data-free checks for explicit nonresponse-indicator model specifications."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))

from run_nonresponse_indicator import (  # noqa: E402
    MODEL_PARAMETERS,
    describe_nonresponse,
    ensure_indicators,
    fit_model,
    model_diagnostic,
)


def synthetic_nonresponse_data() -> pd.DataFrame:
    rng = np.random.default_rng(20261115)
    n = 480
    country = np.repeat(["A", "B", "C"], n // 3)
    training = rng.binomial(1, 0.48, n)
    raw_items = rng.choice([1, 2, 3, 4, 5, 8, 9], size=(n, 3), p=[0.12, 0.16, 0.22, 0.24, 0.18, 0.04, 0.04])
    # Guarantee a mix of all-valid, partial-valid, and zero-valid records.
    raw_items[:30] = np.array([5, 5, 5])
    raw_items[30:60] = np.array([1, 5, 8])
    valid_items = np.where(np.isin(raw_items, [1, 2, 3, 4]), raw_items, np.nan)
    mediator = pd.DataFrame(valid_items).mean(axis=1).to_numpy()
    mediator_for_y = np.nan_to_num(mediator, nan=2.5)
    probability = 1 / (1 + np.exp(-(-1.8 + 0.65 * training + 0.45 * mediator_for_y)))
    data = pd.DataFrame(
        {
            "CNTRY": country,
            "ai_training": training,
            "ai_usage": rng.binomial(1, probability),
            "TCHWGT": rng.uniform(0.5, 2.0, n),
            "TT4G35A": raw_items[:, 0],
            "TT4G35B": raw_items[:, 1],
            "TT4G35C": raw_items[:, 2],
            "perceived_lesson": valid_items[:, 0],
            "perceived_adapt": valid_items[:, 1],
            "perceived_individual": valid_items[:, 2],
        }
    )
    result = ensure_indicators(data)
    primary = result.loc[result["N_VALID"].ge(1)]
    mu = np.average(primary["mediator"], weights=primary["TCHWGT"])
    result["M_CENTER"] = result["mediator"].sub(mu).where(result["N_VALID"].ge(1), 0.0)
    return result


def test_all_four_taskbook_models_have_their_declared_parameters() -> None:
    data = synthetic_nonresponse_data()
    primary = data.loc[data["N_VALID"].ge(1)]
    inputs = {"A0": primary, "A1": primary, "B0": data, "B1": data}
    for model, sample in inputs.items():
        estimates = fit_model(sample, model, "TCHWGT")
        assert set(estimates) == set(MODEL_PARAMETERS[model])
        diagnostic = model_diagnostic(sample, model)
        assert diagnostic["fit_status"] == "converged"
        assert not diagnostic["rank_deficient"]


def test_descriptives_keep_dk_any_and_other_special_codes_separate() -> None:
    data = synthetic_nonresponse_data()
    cross, descriptives = describe_nonresponse(data)
    assert {"N_VALID", "DK_ANY", "DK_COUNT"}.issubset(cross.columns)
    assert (descriptives["section"] == "DK_ANY_by_X_or_Y").any()
    code_rows = descriptives.loc[descriptives["section"].eq("item_specific_raw_code")]
    assert {"dont_know_5", "other_missing_8", "other_missing_9"}.issubset(set(code_rows["category"]))
