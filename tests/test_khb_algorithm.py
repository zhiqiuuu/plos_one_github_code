"""Small, data-free equivalence checks for the KHB decomposition implementation."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm
import statsmodels.formula.api as smf


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))

from run_khb_design import brr_se, fit_khb_components, weighted_residualize_mediator  # noqa: E402


def synthetic_khb_data() -> pd.DataFrame:
    """Generate a non-separated weighted binary-outcome example with fixed effects."""
    rng = np.random.default_rng(20261115)
    n = 480
    country = np.repeat(["A", "B", "C"], n // 3)
    x = rng.binomial(1, 0.48, n)
    country_m = np.select([country == "B", country == "C"], [0.20, -0.15], default=0.0)
    mediator = 2.5 + 0.35 * x + country_m + rng.normal(0, 0.45, n)
    country_y = np.select([country == "B", country == "C"], [0.25, -0.18], default=0.0)
    probability = 1 / (1 + np.exp(-(-1.7 + 0.75 * x + 0.55 * mediator + country_y)))
    return pd.DataFrame(
        {
            "CNTRY": country,
            "ai_training": x,
            "mediator": mediator,
            "ai_usage": rng.binomial(1, probability),
            "TCHWGT": rng.uniform(0.5, 2.0, n),
        }
    )


def reference_khb_components(data: pd.DataFrame) -> dict[str, float]:
    """Independently write the three equations used in the task-book specification."""
    right = "ai_training + C(CNTRY)"
    reduced = smf.glm(
        f"ai_usage ~ {right}", data=data, family=sm.families.Binomial(), freq_weights=data["TCHWGT"]
    ).fit(maxiter=100, disp=0)
    projection = smf.wls(f"mediator ~ {right}", data=data, weights=data["TCHWGT"]).fit()
    residualised = data.assign(m_resid=np.asarray(projection.resid, dtype=float))
    full = smf.glm(
        f"ai_usage ~ {right} + mediator",
        data=residualised,
        family=sm.families.Binomial(),
        freq_weights=residualised["TCHWGT"],
    ).fit(maxiter=100, disp=0)
    adjusted = smf.glm(
        f"ai_usage ~ {right} + m_resid",
        data=residualised,
        family=sm.families.Binomial(),
        freq_weights=residualised["TCHWGT"],
    ).fit(maxiter=100, disp=0)
    total_raw = float(reduced.params["ai_training"])
    total_adjusted = float(adjusted.params["ai_training"])
    direct = float(full.params["ai_training"])
    return {
        "total_raw": total_raw,
        "total_adjusted": total_adjusted,
        "direct": direct,
        "indirect": total_adjusted - direct,
        "rescaling": total_adjusted - total_raw,
    }


def test_khb_matches_independent_residualisation_equations() -> None:
    data = synthetic_khb_data()
    observed = fit_khb_components(data, "TCHWGT")
    expected = reference_khb_components(data)
    for parameter, value in expected.items():
        assert observed[parameter] == pytest.approx(value, abs=1e-10)
    assert observed["direct"] + observed["indirect"] == pytest.approx(
        observed["total_adjusted"], abs=1e-12
    )
    residual, projection = weighted_residualize_mediator(data, "TCHWGT")
    assert residual.equals(data["mediator"] - projection.fittedvalues)


def test_fay_brr_formula() -> None:
    assert brr_se(2.0, [1.0, 3.0]) == pytest.approx(2.0)
