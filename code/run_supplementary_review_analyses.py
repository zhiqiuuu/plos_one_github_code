"""Supplementary analyses requested in PLOS ONE review.

This script does not overwrite the original pooled, unweighted analysis.  It
uses the TALIS teacher final weight, country fixed effects, and the 100
teacher Fay-BRR replicate weights for design-based uncertainty.  The reported
mediation quantity is a model-standardised probability-scale indirect
contrast, not the product of an OLS coefficient and a logit coefficient.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.imputation.mice import MICEData
from statsmodels.tools.sm_exceptions import ConvergenceWarning


ROOT = Path(__file__).resolve().parents[1]
DATA_FILE = ROOT / "data" / "processed" / "data_recoded.csv"
RAW_FILE = ROOT / "TALIS2024_teachers_NoESE_CSV" / "ttgintt4.csv"
OUT_DIR = ROOT / "analysis_output"
ITEMS = ["perceived_lesson", "perceived_adapt", "perceived_individual"]
RAW_ITEMS = ["TT4G35A", "TT4G35B", "TT4G35C"]
X = "ai_training"
Y = "ai_usage"
W = "TCHWGT"
MEDIATOR = "mediator"
FAY_FACTOR = 0.5
N_REPLICATES = 100
N_IMPUTATIONS = 20

warnings.simplefilter("ignore", ConvergenceWarning)


def add_replicate_weights(data: pd.DataFrame) -> pd.DataFrame:
    """Attach official teacher BRR replicate weights using stable TALIS IDs."""
    id_cols = ["IDCNTRY", "IDSCHOOL", "IDTEACH"]
    rep_cols = [f"TRWGT{i}" for i in range(1, N_REPLICATES + 1)]
    raw = pd.read_csv(
        RAW_FILE,
        sep=";",
        usecols=id_cols + rep_cols,
        dtype={column: "float32" for column in rep_cols},
        low_memory=False,
    )
    merged = data.merge(raw, on=id_cols, how="left", validate="one_to_one")
    if merged[rep_cols].isna().any().any():
        raise RuntimeError("Some processed records could not be matched to TALIS BRR weights.")
    return merged


def make_cluster_id(data: pd.DataFrame) -> pd.DataFrame:
    data = data.copy()
    data["school_cluster"] = data["IDCNTRY"].astype(str) + "_" + data["IDSCHOOL"].astype(str)
    return data


def fit_paths(data: pd.DataFrame, weight: str, mediator: str = MEDIATOR, extra: list[str] | None = None):
    """Fit weighted country-fixed-effect path models for one analytic sample."""
    extra = extra or []
    right = " + ".join([X, *extra, "C(CNTRY)"])
    a_model = smf.wls(
        f"{mediator} ~ {right}", data=data, weights=data[weight]
    ).fit()
    c_model = smf.glm(
        f"{Y} ~ {right}", data=data, family=sm.families.Binomial(), freq_weights=data[weight]
    ).fit(maxiter=100, disp=0)
    b_model = smf.glm(
        f"{Y} ~ {right} + {mediator}",
        data=data,
        family=sm.families.Binomial(),
        freq_weights=data[weight],
    ).fit(maxiter=100, disp=0)
    return a_model, c_model, b_model


def standardised_contrasts(data: pd.DataFrame, a_model, b_model, mediator: str, extra: list[str] | None = None):
    """Return model-standardised probability contrasts over observed covariates.

    These are associational standardised contrasts.  They are not labelled as
    causal ACME because the cross-sectional data do not identify causal paths.
    """
    extra = extra or []
    context_cols = ["CNTRY", *extra]
    context = data.groupby(context_cols, as_index=False)[W].sum()
    context["weight_share"] = context[W] / context[W].sum()

    m0_frame = context[context_cols].copy()
    m1_frame = context[context_cols].copy()
    m0_frame[X] = 0
    m1_frame[X] = 1
    m0 = np.asarray(a_model.predict(m0_frame))
    m1 = np.asarray(a_model.predict(m1_frame))

    def predict_y(training, med):
        frame = context[context_cols].copy()
        frame[X] = training
        frame[mediator] = med
        return np.asarray(b_model.predict(frame))

    y00 = predict_y(0, m0)
    y10 = predict_y(1, m0)
    y11 = predict_y(1, m1)
    weights = context["weight_share"].to_numpy()
    return {
        "indirect_probability_difference": float(np.sum(weights * (y11 - y10))),
        "direct_probability_difference": float(np.sum(weights * (y10 - y00))),
        "total_probability_difference": float(np.sum(weights * (y11 - y00))),
    }


def extract_paths(a_model, c_model, b_model, mediator: str) -> dict[str, float]:
    return {
        "a": float(a_model.params[X]),
        "c": float(c_model.params[X]),
        "b": float(b_model.params[mediator]),
        "c_prime": float(b_model.params[X]),
    }


def brr_se(estimate: float, replicates: list[float]) -> float:
    values = np.asarray(replicates, dtype=float)
    return float(np.sqrt(np.sum((values - estimate) ** 2) / (len(values) * (1 - FAY_FACTOR) ** 2)))


def brr_primary(data: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    """Estimate main model precision from TALIS 100 Fay-BRR teacher weights."""
    a_model, c_model, b_model = fit_paths(data, W)
    point_paths = extract_paths(a_model, c_model, b_model, MEDIATOR)
    point_effects = standardised_contrasts(data, a_model, b_model, MEDIATOR)
    names = ["a", "c", "b", "c_prime", *point_effects]
    rep_values = {name: [] for name in names}

    for index in range(1, N_REPLICATES + 1):
        rep_weight = f"TRWGT{index}"
        a_rep, c_rep, b_rep = fit_paths(data, rep_weight)
        rep_paths = extract_paths(a_rep, c_rep, b_rep, MEDIATOR)
        rep_effects = standardised_contrasts(data.assign(**{W: data[rep_weight]}), a_rep, b_rep, MEDIATOR)
        for name, value in {**rep_paths, **rep_effects}.items():
            rep_values[name].append(value)
        if index % 10 == 0:
            print(f"BRR replicate {index}/{N_REPLICATES} completed")

    estimates = {**point_paths, **point_effects}
    summary = {}
    for name, estimate in estimates.items():
        se = brr_se(estimate, rep_values[name])
        summary[name] = {
            "estimate": estimate,
            "brr_fay_se": se,
            "ci_lower": estimate - 1.96 * se,
            "ci_upper": estimate + 1.96 * se,
            "z": estimate / se if se else np.nan,
        }

    rows = []
    labels = {
        "a": "a: training -> perceived utility (linear scale)",
        "c": "c: training -> AI use (log odds)",
        "b": "b: perceived utility -> AI use (log odds)",
        "c_prime": "c-prime: training -> AI use conditional on mediator (log odds)",
        "indirect_probability_difference": "standardised indirect probability difference",
        "direct_probability_difference": "standardised direct probability difference",
        "total_probability_difference": "standardised total probability difference",
    }
    for name, result in summary.items():
        row = {"parameter": name, "description": labels[name], **result}
        if name in {"b", "c", "c_prime"}:
            row["odds_ratio"] = float(np.exp(result["estimate"]))
            row["or_ci_lower"] = float(np.exp(result["ci_lower"]))
            row["or_ci_upper"] = float(np.exp(result["ci_upper"]))
        rows.append(row)
    return summary, pd.DataFrame(rows)


def simple_model_comparison(data: pd.DataFrame) -> pd.DataFrame:
    """Show the impact of weighting and country fixed effects on point estimates."""
    rows = []
    specs = [
        ("unweighted_pooled", None, False),
        ("weighted_pooled", W, False),
        ("weighted_country_fixed_effects", W, True),
    ]
    for name, weight, fixed_effects in specs:
        if fixed_effects:
            a_model, c_model, b_model = fit_paths(data, W)
        else:
            if weight is None:
                a_model = smf.ols(f"{MEDIATOR} ~ {X}", data=data).fit()
                c_model = smf.glm(f"{Y} ~ {X}", data=data, family=sm.families.Binomial()).fit()
                b_model = smf.glm(f"{Y} ~ {X} + {MEDIATOR}", data=data, family=sm.families.Binomial()).fit()
            else:
                a_model = smf.wls(f"{MEDIATOR} ~ {X}", data=data, weights=data[W]).fit()
                c_model = smf.glm(f"{Y} ~ {X}", data=data, family=sm.families.Binomial(), freq_weights=data[W]).fit()
                b_model = smf.glm(f"{Y} ~ {X} + {MEDIATOR}", data=data, family=sm.families.Binomial(), freq_weights=data[W]).fit()
        paths = extract_paths(a_model, c_model, b_model, MEDIATOR)
        rows.append({
            "specification": name,
            "n": len(data),
            **paths,
            "c_odds_ratio": float(np.exp(paths["c"])),
            "b_odds_ratio": float(np.exp(paths["b"])),
            "c_prime_odds_ratio": float(np.exp(paths["c_prime"])),
        })
    return pd.DataFrame(rows)


def sensitivity_point(data: pd.DataFrame, label: str, mediator: str = MEDIATOR, extra: list[str] | None = None) -> dict:
    extra = extra or []
    needed = [X, Y, W, mediator, "CNTRY", *extra]
    sample = data.dropna(subset=needed).copy()
    a_model, c_model, b_model = fit_paths(sample, W, mediator=mediator, extra=extra)
    effects = standardised_contrasts(sample, a_model, b_model, mediator, extra=extra)
    return {"strategy": label, "n": len(sample), **extract_paths(a_model, c_model, b_model, mediator), **effects}


def run_missingness_sensitivity(data: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Compare requested complete-case and non-response handling strategies."""
    item_data = data[ITEMS]
    rows = []

    ge1 = data.loc[item_data.notna().sum(axis=1) >= 1].copy()
    ge1[MEDIATOR] = ge1[ITEMS].mean(axis=1)
    rows.append(sensitivity_point(ge1, "at_least_one_valid_item"))

    complete = data.loc[item_data.notna().sum(axis=1) == 3].copy()
    complete[MEDIATOR] = complete[ITEMS].mean(axis=1)
    rows.append(sensitivity_point(complete, "complete_case_all_three_items"))

    midpoint = data.copy()
    midpoint_items = midpoint[RAW_ITEMS].replace({5: 2.5, 8: np.nan, 9: np.nan})
    midpoint[MEDIATOR] = midpoint_items.mean(axis=1)
    midpoint = midpoint.loc[midpoint[MEDIATOR].notna()].copy()
    rows.append(sensitivity_point(midpoint, "dont_know_recoded_to_midpoint_2_5"))

    indicator = ge1.copy()
    indicator["dont_know_count"] = indicator[RAW_ITEMS].eq(5).sum(axis=1)
    rows.append(sensitivity_point(indicator, "explicit_dont_know_count_adjustment", extra=["dont_know_count"]))

    mi_rows = multiple_imputation(data)
    mi_frame = pd.DataFrame(mi_rows)
    mi_numeric = mi_frame.drop(columns=["strategy", "imputation"])
    mi_summary = mi_numeric.agg(["mean", "min", "max"]).T.reset_index().rename(columns={"index": "parameter"})
    return pd.DataFrame(rows), {"individual": mi_frame, "summary": mi_summary}


def multiple_imputation(data: pd.DataFrame) -> list[dict]:
    """PMM chained-equation imputations under a MAR working assumption."""
    imputation_items = data[ITEMS].copy()
    country = pd.get_dummies(data["CNTRY"], prefix="country", drop_first=True, dtype=float)
    mice_input = pd.concat([imputation_items, data[[X, Y]], country], axis=1)
    mice = MICEData(mice_input, perturbation_method="boot", k_pmm=20)
    country_terms = " + ".join(country.columns)
    for item in ITEMS:
        other_items = [name for name in ITEMS if name != item]
        mice.set_imputer(item, formula=" + ".join([X, Y, *other_items, country_terms]))
    mice.update_all(5)

    results = []
    for number in range(1, N_IMPUTATIONS + 1):
        mice.update_all(1)
        sample = data[["CNTRY", "IDCNTRY", "IDSCHOOL", W, X, Y]].copy()
        sample[MEDIATOR] = mice.data[ITEMS].mean(axis=1).to_numpy()
        result = sensitivity_point(sample, "MAR_multiple_imputation_PMM")
        result["imputation"] = number
        results.append(result)
        print(f"Multiple imputation {number}/{N_IMPUTATIONS} completed")
    return results


def data_checks(data: pd.DataFrame) -> dict:
    return {
        "processed_records": int(len(data)),
        "countries": int(data["CNTRY"].nunique()),
        "schools": int(data["school_cluster"].nunique()),
        "teacher_ids_duplicate": int(data.duplicated(["IDCNTRY", "IDTEACH"]).sum()),
        "training_valid": int(data[X].notna().sum()),
        "ai_use_valid": int(data[Y].notna().sum()),
        "valid_items_one_or_more": int((data[ITEMS].notna().sum(axis=1) >= 1).sum()),
        "valid_items_all_three": int((data[ITEMS].notna().sum(axis=1) == 3).sum()),
        "dont_know_by_item": {raw: int(data[raw].eq(5).sum()) for raw in RAW_ITEMS},
        "other_missing_codes_by_item": {raw: int(data[raw].isin([8, 9]).sum()) for raw in RAW_ITEMS},
        "weight_min": float(data[W].min()),
        "weight_max": float(data[W].max()),
    }


def descriptive_statistics(data: pd.DataFrame) -> pd.DataFrame:
    """Provide the requested item and composite univariate statistics."""
    rows = []
    variables = [X, Y, *ITEMS, MEDIATOR]
    labels = {
        X: "AI-related professional learning (0/1)",
        Y: "Any classroom AI use in past 12 months (0/1)",
        "perceived_lesson": "Perceived utility item A",
        "perceived_adapt": "Perceived utility item B",
        "perceived_individual": "Perceived utility item C",
        MEDIATOR: "Perceived utility composite (at least one valid item)",
    }
    for variable in variables:
        sample = data[[variable, W]].dropna()
        value = sample[variable].to_numpy(dtype=float)
        weight = sample[W].to_numpy(dtype=float)
        mean = np.average(value, weights=weight)
        variance = np.average((value - mean) ** 2, weights=weight)
        sd = np.sqrt(variance)
        skew = np.average((value - mean) ** 3, weights=weight) / sd ** 3 if sd else np.nan
        rows.append({
            "variable": variable,
            "description": labels[variable],
            "n_unweighted": len(sample),
            "weighted_mean": mean,
            "weighted_sd": sd,
            "weighted_skewness": skew,
            "minimum": value.min(),
            "maximum": value.max(),
        })
    return pd.DataFrame(rows)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    data = make_cluster_id(pd.read_csv(DATA_FILE))
    base = data.dropna(subset=[X, Y, W]).copy()
    base[MEDIATOR] = base[ITEMS].mean(axis=1)
    base = base.dropna(subset=[MEDIATOR]).copy()

    checks = data_checks(data)
    with (OUT_DIR / "data_validation.json").open("w", encoding="utf-8") as file:
        json.dump(checks, file, ensure_ascii=False, indent=2)
    descriptive_statistics(base).to_csv(OUT_DIR / "table_s0_descriptive_statistics.csv", index=False)

    brr_data = make_cluster_id(add_replicate_weights(data))
    brr_base = brr_data.dropna(subset=[X, Y, W]).copy()
    brr_base[MEDIATOR] = brr_base[ITEMS].mean(axis=1)
    brr_base = brr_base.dropna(subset=[MEDIATOR]).copy()
    brr, brr_table = brr_primary(brr_base)
    brr_table.to_csv(OUT_DIR / "table_s1_design_adjusted_paths.csv", index=False)
    simple_model_comparison(base).to_csv(OUT_DIR / "table_s2_weighting_comparison.csv", index=False)

    sensitivity, mi = run_missingness_sensitivity(data)
    sensitivity.to_csv(OUT_DIR / "table_s3_missing_data_sensitivity.csv", index=False)
    mi["individual"].to_csv(OUT_DIR / "table_s4_multiple_imputation_individual.csv", index=False)
    mi["summary"].to_csv(OUT_DIR / "table_s4_multiple_imputation_summary.csv", index=False)

    output = {
        "analysis_description": {
            "sample": "TALIS 2024 teacher data, AI-module analytic records",
            "weight": "TCHWGT final teacher weight",
            "fixed_effects": "55 education-system fixed effects",
            "variance": "100 teacher BRR replicate weights (TRWGT1-TRWGT100), Fay factor 0.5",
            "mediation_measure": "model-standardised probability-scale indirect contrast; associational, not causal ACME",
            "multiple_imputation": f"{N_IMPUTATIONS} predictive-mean-matching chained-equation imputations under a MAR working assumption",
        },
        "data_checks": checks,
        "primary_brr_results": brr,
    }
    with (OUT_DIR / "supplementary_analysis_results.json").open("w", encoding="utf-8") as file:
        json.dump(output, file, ensure_ascii=False, indent=2, allow_nan=False)


if __name__ == "__main__":
    main()
