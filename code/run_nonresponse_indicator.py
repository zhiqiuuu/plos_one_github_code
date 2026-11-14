"""Run explicit TT4G35 ``I don't know`` indicator sensitivity analyses.

The analysis keeps raw response codes 5 and 8/9 distinct.  It reports two
same-sample models (A0/A1, n=72,000) and two extended-work-sample models
(B0/B1, n=84,770), each with 100 Fay-BRR replicate fits.  Outputs are
aggregate only; restricted TALIS microdata remain outside the repository.
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import subprocess
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy.stats import norm

from run_revision_package import (
    FAY_FACTOR,
    ITEMS,
    N_REPLICATES,
    W,
    X,
    Y,
    attach_replicate_weights,
    brr_se,
    git_worktree_status,
    sha256_file,
    source_manifest,
)


ROOT = Path(__file__).resolve().parents[1]
RAW_ITEMS = ["TT4G35A", "TT4G35B", "TT4G35C"]
MODEL_PARAMETERS = {
    "A0": [X, "mediator"],
    "A1": [X, "mediator", "DK_ANY"],
    "B0": [X],
    "B1": [X, "M_CENTER", "DK_ANY", "NO_VALID"],
}
MODEL_ANCHORS = {
    ("A0", X): 1.3869881248112819,
    ("A0", "mediator"): 1.1591805215668596,
    ("A1", X): 1.3526302344831223,
    ("A1", "mediator"): 1.1733371187732868,
    ("A1", "DK_ANY"): -0.6128996541862266,
    ("B0", X): 1.665455530255135,
    ("B1", X): 1.3467085351387171,
    ("B1", "M_CENTER"): 1.1587195089851177,
    ("B1", "DK_ANY"): -0.6113140714193657,
    ("B1", "NO_VALID"): -1.7270557917878993,
}
LOCKED_COUNTS = {
    "DK_A": 15946,
    "DK_B": 17671,
    "DK_C": 19642,
    "DK_ANY": 23591,
    "OTHER_SPECIAL_A": 348,
    "OTHER_SPECIAL_B": 441,
    "OTHER_SPECIAL_C": 529,
    "OTHER_MISSING_ANY": 922,
}
LOCKED_N_VALID = {0: 12770, 1: 4714, 2: 6839, 3: 60447}
LOCKED_CENTERING_MU = 2.902023122631011
ANCHOR_TOLERANCE = 1e-6


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-file", type=Path, required=True)
    parser.add_argument("--processed-file", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20261114)
    return parser.parse_args()


def write_log(path: Path, message: str) -> None:
    line = f"{datetime.now(timezone.utc).isoformat()}Z {message}"
    print(line)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def git_revision() -> str:
    try:
        return subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unavailable"


def ensure_indicators(data: pd.DataFrame) -> pd.DataFrame:
    """Rebuild indicators from original item columns, never from recoded NaNs."""
    work = data.copy()
    for suffix, item in zip(("A", "B", "C"), RAW_ITEMS, strict=True):
        work[f"DK_{suffix}"] = work[item].eq(5).astype("int8")
    work["DK_COUNT"] = work[["DK_A", "DK_B", "DK_C"]].sum(axis=1).astype("int8")
    work["DK_ANY"] = work["DK_COUNT"].gt(0).astype("int8")
    work["N_VALID"] = work[RAW_ITEMS].isin([1, 2, 3, 4]).sum(axis=1).astype("int8")
    work["NO_VALID"] = work["N_VALID"].eq(0).astype("int8")
    work["OTHER_MISSING_ANY"] = work[RAW_ITEMS].isin([8, 9]).any(axis=1).astype("int8")
    work["mediator"] = work[ITEMS].mean(axis=1)
    return work


def weighted_mean(values: pd.Series, weights: pd.Series) -> float:
    mask = values.notna() & weights.notna()
    return float(np.average(values.loc[mask], weights=weights.loc[mask]))


def make_samples(processed: pd.DataFrame, raw_file: Path) -> tuple[pd.DataFrame, pd.DataFrame, float]:
    weighted = ensure_indicators(attach_replicate_weights(processed, raw_file))
    primary = weighted.loc[weighted["N_VALID"].ge(1)].copy()
    mu = weighted_mean(primary["mediator"], primary[W])
    weighted["M_CENTER"] = weighted["mediator"].sub(mu).where(weighted["N_VALID"].ge(1), 0.0)
    primary["M_CENTER"] = primary["mediator"] - mu
    return weighted, primary, mu


def formula_for(model: str) -> str:
    if model == "A0":
        rhs = f"{X} + mediator + C(CNTRY)"
    elif model == "A1":
        rhs = f"{X} + mediator + DK_ANY + C(CNTRY)"
    elif model == "B0":
        rhs = f"{X} + C(CNTRY)"
    elif model == "B1":
        rhs = f"{X} + M_CENTER + DK_ANY + NO_VALID + C(CNTRY)"
    else:
        raise ValueError(model)
    return f"{Y} ~ {rhs}"


def analysis_data(data: pd.DataFrame, model: str) -> pd.DataFrame:
    work = data.dropna(subset=[Y, X, W, "CNTRY"]).copy()
    if model in {"A0", "A1"}:
        work = work.dropna(subset=["mediator"])
    if work.empty:
        raise ValueError(f"{model}: no rows remain")
    return work


def fit_model(data: pd.DataFrame, model: str, weight: str) -> dict[str, float]:
    work = analysis_data(data, model)
    result = smf.glm(
        formula_for(model),
        data=work,
        family=sm.families.Binomial(),
        freq_weights=work[weight],
    ).fit(maxiter=100, disp=0)
    if not bool(getattr(result, "converged", True)):
        raise RuntimeError(f"{model}: weighted logistic model did not converge for {weight}")
    missing = [parameter for parameter in MODEL_PARAMETERS[model] if parameter not in result.params.index]
    if missing:
        raise ValueError(f"{model}: parameters absent from design matrix: {missing}")
    return {parameter: float(result.params[parameter]) for parameter in MODEL_PARAMETERS[model]}


def model_diagnostic(data: pd.DataFrame, model: str) -> dict[str, object]:
    """Make rank and convergence diagnostics explicit instead of silently dropping terms."""
    work = analysis_data(data, model)
    glm = smf.glm(
        formula_for(model), data=work, family=sm.families.Binomial(), freq_weights=work[W]
    )
    warning_names: list[str] = []
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = glm.fit(maxiter=100, disp=0)
        warning_names = sorted({warning.category.__name__ for warning in caught})
        status = "converged" if bool(getattr(result, "converged", True)) else "not_converged"
    except Exception as exc:
        status = "failed"
        warning_names = [repr(exc)]
    rank = int(np.linalg.matrix_rank(np.asarray(glm.exog)))
    columns = int(glm.exog.shape[1])
    return {
        "model": model,
        "n_unweighted": int(len(work)),
        "n_design_columns": columns,
        "matrix_rank": rank,
        "rank_deficient": bool(rank < columns),
        "fit_status": status,
        "fit_warnings": "; ".join(warning_names),
        "separation_or_convergence_issue": bool(status != "converged" or warning_names),
    }


def model_rows(model: str, point: dict[str, float], values: dict[str, list[float]], data: pd.DataFrame) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for parameter, estimate in point.items():
        se = brr_se(estimate, values[parameter])
        lo = estimate - 1.96 * se
        hi = estimate + 1.96 * se
        z_value = estimate / se if se else float("nan")
        rows.append(
            {
                "model": model,
                "n_unweighted": int(len(data)),
                "n_systems": int(data["CNTRY"].nunique()),
                "parameter": parameter,
                "estimate": estimate,
                "odds_ratio": math.exp(estimate),
                "brr_fay_se": se,
                "z": z_value,
                "p_two_sided": 2 * norm.sf(abs(z_value)) if math.isfinite(z_value) else float("nan"),
                "ci_lower_95": lo,
                "ci_upper_95": hi,
                "or_ci_lower_95": math.exp(lo),
                "or_ci_upper_95": math.exp(hi),
                "fay_factor": FAY_FACTOR,
                "n_replicates": N_REPLICATES,
                "inference_class": "Fay-BRR 95% CI",
                "reference_category": "DK_ANY=0 and NO_VALID=0 where applicable",
            }
        )
    return rows


def run_model(data: pd.DataFrame, model: str, log_file: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    point = fit_model(data, model, W)
    values = {parameter: [] for parameter in point}
    replicate_rows: list[dict[str, object]] = []
    failures: list[dict[str, object]] = []
    for replicate in range(1, N_REPLICATES + 1):
        weight = f"TRWGT{replicate}"
        try:
            estimates = fit_model(data, model, weight)
            for parameter, estimate in estimates.items():
                values[parameter].append(estimate)
                replicate_rows.append(
                    {
                        "model": model,
                        "replicate": replicate,
                        "parameter": parameter,
                        "estimate": estimate,
                        "weight": weight,
                        "status": "converged",
                    }
                )
        except Exception as exc:
            failures.append(
                {"model": model, "replicate": replicate, "weight": weight, "status": "failed", "error": repr(exc)}
            )
        if replicate % 5 == 0:
            write_log(log_file, f"{model}: completed Fay-BRR replicate {replicate}/{N_REPLICATES}")
    if failures:
        return pd.DataFrame(), pd.DataFrame(replicate_rows), pd.DataFrame(failures)
    return pd.DataFrame(model_rows(model, point, values, data)), pd.DataFrame(replicate_rows), pd.DataFrame(failures)


def describe_nonresponse(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    cross = (
        data.groupby(["N_VALID", "DK_ANY", "DK_COUNT"], dropna=False)
        .agg(n_unweighted=("CNTRY", "size"), weighted_n=(W, "sum"))
        .reset_index()
    )
    cross["weighted_percent_total"] = cross["weighted_n"] / data[W].sum() * 100
    rows: list[dict[str, object]] = []
    dk_any_all = data.loc[data["DK_ANY"].eq(1)]
    rows.append(
        {
            "section": "DK_ANY_overall",
            "stratifier": "all",
            "category": "all",
            "N_VALID": "not_applicable",
            "DK_ANY": 1,
            "DK_COUNT": "not_applicable",
            "source_item": "all_three_items",
            "response_code": "5_on_any_item",
            "n_unweighted": int(len(dk_any_all)),
            "weighted_n": float(dk_any_all[W].sum()),
            "weighted_percent": float(dk_any_all[W].sum() / data[W].sum() * 100),
        }
    )
    for row in cross.itertuples(index=False):
        rows.append(
            {
                "section": "N_VALID_by_DK_ANY_and_DK_COUNT",
                "stratifier": "all",
                "category": "all",
                "N_VALID": row.N_VALID,
                "DK_ANY": row.DK_ANY,
                "DK_COUNT": row.DK_COUNT,
                "source_item": "all_three_items",
                "response_code": "not_applicable",
                "n_unweighted": row.n_unweighted,
                "weighted_n": row.weighted_n,
                "weighted_percent": row.weighted_percent_total,
            }
        )
    for variable, label in ((X, "X"), (Y, "Y")):
        for category, subset in data.groupby(variable, dropna=False):
            denominator = subset[W].sum()
            dk_any = subset.loc[subset["DK_ANY"].eq(1)]
            rows.append(
                {
                    "section": "DK_ANY_by_X_or_Y",
                    "stratifier": label,
                    "category": category,
                    "N_VALID": "not_applicable",
                    "DK_ANY": 1,
                    "DK_COUNT": "not_applicable",
                    "source_item": "all_three_items",
                    "response_code": "5_on_any_item",
                    "n_unweighted": int(len(dk_any)),
                    "weighted_n": float(dk_any[W].sum()),
                    "weighted_percent": float(dk_any[W].sum() / denominator * 100) if denominator else float("nan"),
                }
            )
    for item in RAW_ITEMS:
        for code, code_label in ((5, "dont_know_5"), (8, "other_missing_8"), (9, "other_missing_9")):
            subset = data[item].eq(code)
            rows.append(
                {
                    "section": "item_specific_raw_code",
                    "stratifier": item,
                    "category": code_label,
                    "N_VALID": "not_applicable",
                    "DK_ANY": "not_applicable",
                    "DK_COUNT": "not_applicable",
                    "source_item": item,
                    "response_code": code,
                    "n_unweighted": int(subset.sum()),
                    "weighted_n": float(data.loc[subset, W].sum()),
                    "weighted_percent": float(data.loc[subset, W].sum() / data[W].sum() * 100),
                }
            )
    other = data.loc[data["OTHER_MISSING_ANY"].eq(1)]
    rows.append(
        {
            "section": "other_special_missing_any",
            "stratifier": "all",
            "category": "any_8_or_9",
            "N_VALID": "not_applicable",
            "DK_ANY": "not_applicable",
            "DK_COUNT": "not_applicable",
            "source_item": "all_three_items",
            "response_code": "8_or_9",
            "n_unweighted": int(len(other)),
            "weighted_n": float(other[W].sum()),
            "weighted_percent": float(other[W].sum() / data[W].sum() * 100),
        }
    )
    return cross, pd.DataFrame(rows)


def same_sample_changes(summaries: pd.DataFrame) -> pd.DataFrame:
    """Quantify A0/A1 changes without mislabelling them as causal effects."""
    a0 = summaries.loc[summaries["model"].eq("A0")].set_index("parameter")
    a1 = summaries.loc[summaries["model"].eq("A1")].set_index("parameter")
    rows: list[dict[str, object]] = []
    for parameter in (X, "mediator"):
        baseline = float(a0.loc[parameter, "estimate"])
        explicit = float(a1.loc[parameter, "estimate"])
        difference = explicit - baseline
        rows.append(
            {
                "parameter": parameter,
                "A0_estimate": baseline,
                "A1_estimate": explicit,
                "absolute_change_A1_minus_A0": difference,
                "percent_change_from_A0": difference / baseline * 100,
                "interpretation": "same-sample descriptive coefficient change; not a causal effect",
            }
        )
    return pd.DataFrame(rows)


def model_anchor_checks(summaries: pd.DataFrame) -> pd.DataFrame:
    indexed = summaries.set_index(["model", "parameter"])
    rows = []
    for (model, parameter), expected in MODEL_ANCHORS.items():
        observed = float(indexed.loc[(model, parameter), "estimate"])
        rows.append(
            {
                "model": model,
                "parameter": parameter,
                "expected_taskbook": expected,
                "observed": observed,
                "absolute_difference": abs(observed - expected),
                "tolerance": ANCHOR_TOLERANCE,
                "passed": abs(observed - expected) <= ANCHOR_TOLERANCE,
            }
        )
    return pd.DataFrame(rows)


def input_anchor_checks(data: pd.DataFrame, mu: float) -> pd.DataFrame:
    """Record locked raw-code counts before estimating any sensitivity model."""
    observed = {
        "DK_A": int(data["DK_A"].sum()),
        "DK_B": int(data["DK_B"].sum()),
        "DK_C": int(data["DK_C"].sum()),
        "DK_ANY": int(data["DK_ANY"].sum()),
        "OTHER_SPECIAL_A": int(data["TT4G35A"].isin([8, 9]).sum()),
        "OTHER_SPECIAL_B": int(data["TT4G35B"].isin([8, 9]).sum()),
        "OTHER_SPECIAL_C": int(data["TT4G35C"].isin([8, 9]).sum()),
        "OTHER_MISSING_ANY": int(data["OTHER_MISSING_ANY"].sum()),
    }
    rows = [
        {
            "check": name,
            "expected": expected,
            "observed": observed[name],
            "absolute_difference": abs(observed[name] - expected),
            "tolerance": 0,
            "passed": observed[name] == expected,
        }
        for name, expected in LOCKED_COUNTS.items()
    ]
    for n_valid, expected in LOCKED_N_VALID.items():
        actual = int(data["N_VALID"].eq(n_valid).sum())
        rows.append(
            {
                "check": f"N_VALID_{n_valid}",
                "expected": expected,
                "observed": actual,
                "absolute_difference": abs(actual - expected),
                "tolerance": 0,
                "passed": actual == expected,
            }
        )
    rows.append(
        {
            "check": "weighted_primary_mediator_mean",
            "expected": LOCKED_CENTERING_MU,
            "observed": mu,
            "absolute_difference": abs(mu - LOCKED_CENTERING_MU),
            "tolerance": ANCHOR_TOLERANCE,
            "passed": abs(mu - LOCKED_CENTERING_MU) <= ANCHOR_TOLERANCE,
        }
    )
    return pd.DataFrame(rows)


def add_same_sample_change_columns(summaries: pd.DataFrame, changes: pd.DataFrame) -> pd.DataFrame:
    result = summaries.copy()
    result["A0_estimate_for_same_sample_comparison"] = np.nan
    result["absolute_change_A1_minus_A0"] = np.nan
    result["percent_change_from_A0"] = np.nan
    for change in changes.itertuples(index=False):
        selector = result["model"].eq("A1") & result["parameter"].eq(change.parameter)
        result.loc[selector, "A0_estimate_for_same_sample_comparison"] = change.A0_estimate
        result.loc[selector, "absolute_change_A1_minus_A0"] = change.absolute_change_A1_minus_A0
        result.loc[selector, "percent_change_from_A0"] = change.percent_change_from_A0
    return result


def write_methods(path: Path, summaries: pd.DataFrame, mu: float) -> None:
    def sentence(model: str) -> str:
        rows = summaries.loc[summaries["model"].eq(model)]
        if rows.empty:
            return f"{model} was not available because one or more replicate models failed."
        pieces = [f"{row.parameter} beta={row.estimate:.4f} (SE={row.brr_fay_se:.4f})" for row in rows.itertuples()]
        return f"{model} (n={int(rows.n_unweighted.iloc[0]):,}): " + "; ".join(pieces) + "."

    text = f"""# Explicit nonresponse-indicator sensitivity analysis

## English Methods text

The original TT4G35A-C response codes were retained before recoding. For each
item, `5` (I don't know) was coded as a separate binary indicator, whereas 8
and 9 were retained as a distinct other-special-missing category. `DK_ANY`
indicates at least one code 5 and `DK_COUNT` counts the three code-5 responses.
`N_VALID` counts responses coded 1-4 and `NO_VALID` identifies respondents with
no valid utility item. The perceived-utility mean was unchanged: it used only
1-4 responses, and code 5 was never assigned a numeric scale value.

Models A0 and A1 were fitted in the same available-item sample (n=72,000),
with A1 adding `DK_ANY` to the baseline X, utility, and education-system fixed-
effect logistic model. Models B0 and B1 used the expanded X/Y/weight-valid
sample (n=84,770); B1 added `DK_ANY`, `NO_VALID`, and a centered observed-item
mean. The centering constant was the TCHWGT-weighted mean in the n=72,000
sample (mu={mu:.10f}); for N_VALID=0, the centered mean was set to zero solely
as a mathematical placeholder and does not represent a neutral response.
All models were re-estimated with each of 100 TALIS replicate weights. Fay-BRR
standard errors used factor 0.5 and 95% intervals used estimate +/- 1.96 SE.
This is a descriptive missingness sensitivity analysis and does not identify
MAR or MNAR mechanisms.

## English Results text (generated from the result CSV)

"""
    text += "\n".join(sentence(model) for model in ("A0", "A1", "B0", "B1"))
    path.write_text(text + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    if not args.raw_file.exists() or not args.processed_file.exists():
        raise FileNotFoundError("Raw and processed inputs are required")
    output = args.output_dir
    for name in ("B03", "B05", "B06"):
        (output / name).mkdir(parents=True, exist_ok=True)
    log_file = output / "B06" / "nonresponse_indicator_reproduction.log"
    log_file.write_text("", encoding="utf-8")
    write_log(log_file, "started explicit TT4G35 5/non-special-missing sensitivity analysis")
    processed = pd.read_csv(args.processed_file)
    weighted, primary, mu = make_samples(processed, args.raw_file)
    expected = LOCKED_N_VALID
    observed = weighted["N_VALID"].value_counts().sort_index().to_dict()
    if {int(k): int(v) for k, v in observed.items()} != expected:
        raise RuntimeError(f"N_VALID distribution mismatch: observed={observed}, expected={expected}")
    if len(weighted) != 84770 or len(primary) != 72000:
        raise RuntimeError(f"Locked sample mismatch: extended={len(weighted)}, primary={len(primary)}")
    input_checks = input_anchor_checks(weighted, mu)
    input_checks.to_csv(output / "B03" / "nonresponse_indicator_input_anchor_checks.csv", index=False)
    if not input_checks["passed"].all():
        raise RuntimeError("Raw TT4G35 coding or centering checks did not match the locked taskbook values.")

    cross, descriptives = describe_nonresponse(weighted)
    cross.to_csv(output / "B03" / "nonresponse_indicator_cross_distribution.csv", index=False)
    descriptives.to_csv(output / "B03" / "nonresponse_indicator_descriptives.csv", index=False)

    model_inputs = {"A0": primary, "A1": primary, "B0": weighted, "B1": weighted}
    diagnostics = pd.DataFrame(
        [model_diagnostic(model_inputs[model], model) for model in ("A0", "A1", "B0", "B1")]
    )
    diagnostics.to_csv(output / "B03" / "nonresponse_indicator_model_diagnostics.csv", index=False)
    if diagnostics["rank_deficient"].any() or diagnostics["fit_status"].ne("converged").any():
        raise RuntimeError("A nonresponse model is rank deficient or did not converge; inspect model diagnostics.")
    summary_frames: list[pd.DataFrame] = []
    replicate_frames: list[pd.DataFrame] = []
    failure_frames: list[pd.DataFrame] = []
    for model in ("A0", "A1", "B0", "B1"):
        write_log(log_file, f"starting {model} with n={len(model_inputs[model]):,}")
        summary, reps, failures = run_model(model_inputs[model], model, log_file)
        summary_frames.append(summary)
        replicate_frames.append(reps)
        failure_frames.append(failures)
    summaries = pd.concat(summary_frames, ignore_index=True)
    replicates = pd.concat(replicate_frames, ignore_index=True)
    failures = pd.concat(failure_frames, ignore_index=True)
    summaries.to_csv(output / "B03" / "nonresponse_indicator_models_brr.csv", index=False)
    replicates.to_csv(output / "B03" / "nonresponse_indicator_replicates.csv", index=False)
    failures.to_csv(output / "B03" / "nonresponse_indicator_failures.csv", index=False)
    if not failures.empty:
        raise RuntimeError(f"{len(failures)} nonresponse BRR replicate(s) failed; see failure CSV")

    anchors = model_anchor_checks(summaries)
    anchors.to_csv(output / "B03" / "nonresponse_indicator_model_anchor_checks.csv", index=False)
    if not anchors["passed"].all():
        raise RuntimeError("A nonresponse model did not reproduce its locked taskbook coefficient.")
    changes = same_sample_changes(summaries)
    changes.to_csv(output / "B03" / "nonresponse_indicator_same_sample_changes.csv", index=False)
    # A compact manuscript table retains all four models and the within-N=72,000 comparison.
    table_s6 = add_same_sample_change_columns(summaries, changes)
    table_s6.to_csv(output / "B05" / "TableS6_nonresponse_indicator.csv", index=False)
    write_methods(output / "B03" / "nonresponse_indicator_methods.md", summaries, mu)
    metadata = {
        "analysis": "explicit TT4G35 code-5 nonresponse sensitivity",
        "analysis_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "git_revision": git_revision(),
        "git_worktree_status": git_worktree_status(),
        "seed_recorded": args.seed,
        "python": sys.version,
        "platform": platform.platform(),
        "statsmodels": statsmodels.__version__,
        "n_extended": int(len(weighted)),
        "n_primary": int(len(primary)),
        "n_systems_extended": int(weighted["CNTRY"].nunique()),
        "n_systems_primary": int(primary["CNTRY"].nunique()),
        "weighted_centering_mu": mu,
        "n_replicates": N_REPLICATES,
        "fay_factor": FAY_FACTOR,
        "replicate_failures": int(len(failures)),
        "n_valid_distribution": {str(k): int(v) for k, v in observed.items()},
        "raw_file_name": args.raw_file.name,
        "raw_file_sha256": sha256_file(args.raw_file),
        "processed_file_sha256": sha256_file(args.processed_file),
        "analysis_script_sha256": sha256_file(Path(__file__)),
        "source_file_sha256": source_manifest(),
        "claim_boundary": "descriptive sensitivity analysis; does not identify MAR or MNAR",
    }
    (output / "B06" / "nonresponse_indicator_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    write_log(log_file, "completed explicit nonresponse analysis with 100 converged replicates per model")


if __name__ == "__main__":
    main()
