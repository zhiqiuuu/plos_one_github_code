"""Create the auditable PLOS ONE revision-analysis package.

The script expects the restricted TALIS teacher microdata outside this
repository. It writes only derived, non-identifying outputs to results/output_final.
Primary results are association-path estimates, not causal mediation effects.
"""

from __future__ import annotations

import argparse
import hashlib
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
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy.stats import norm, t as student_t
from statsmodels.imputation.mice import MICEData
from statsmodels.tools.sm_exceptions import ConvergenceWarning


ROOT = Path(__file__).resolve().parents[1]
ITEMS = ["perceived_lesson", "perceived_adapt", "perceived_individual"]
RAW_ITEMS = ["TT4G35A", "TT4G35B", "TT4G35C"]
X = "ai_training"
Y = "ai_usage"
W = "TCHWGT"
MEDIATOR = "mediator"
FAY_FACTOR = 0.5
N_REPLICATES = 100

warnings.simplefilter("ignore", ConvergenceWarning)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-file", type=Path, required=True, help="Restricted ttgintt4.csv file")
    parser.add_argument(
        "--processed-file",
        type=Path,
        default=ROOT / "data" / "processed" / "data_recoded.csv",
        help="Derived data created by code/01_data_preparation.py",
    )
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results" / "output_final")
    parser.add_argument("--n-imputations", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20261114, help="Random seed for chained-equation imputation")
    parser.add_argument("--reuse-brr", action="store_true", help="Reuse completed B01/B03 BRR outputs")
    return parser.parse_args()


def make_dirs(root: Path) -> dict[str, Path]:
    names = ("B01", "B02", "B03", "B04", "B05", "B06")
    folders = {name: root / name for name in names}
    for folder in folders.values():
        folder.mkdir(parents=True, exist_ok=True)
    return folders


def write_log(path: Path, message: str) -> None:
    line = f"{datetime.now(timezone.utc).isoformat()}Z {message}"
    print(line)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def sha256_file(path: Path) -> str:
    """Return a content checksum without publishing the restricted source file."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def git_revision() -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unavailable"


def source_manifest() -> dict[str, str]:
    files = [
        ROOT / "code" / "01_data_preparation.py",
        ROOT / "code" / "run_all.py",
        ROOT / "code" / "run_revision_package.py",
        ROOT / "code" / "run_khb_design.py",
        ROOT / "code" / "run_nonresponse_indicator.py",
        ROOT / "code" / "generate_design_adjusted_figure.py",
        ROOT / "code" / "export_public_aggregates.py",
        ROOT / "environment" / "requirements.lock",
    ]
    return {str(path.relative_to(ROOT)).replace("\\", "/"): sha256_file(path) for path in files}


def git_worktree_status() -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(ROOT), "status", "--porcelain"], text=True
        ).strip() or "clean"
    except (OSError, subprocess.CalledProcessError):
        return "unavailable"


def attach_replicate_weights(data: pd.DataFrame, raw_file: Path) -> pd.DataFrame:
    id_cols = ["IDCNTRY", "IDSCHOOL", "IDTEACH"]
    replicate_cols = [f"TRWGT{i}" for i in range(1, N_REPLICATES + 1)]
    raw = pd.read_csv(
        raw_file,
        sep=";",
        usecols=id_cols + replicate_cols,
        dtype={column: "float32" for column in replicate_cols},
        low_memory=False,
    )
    merged = data.merge(raw, on=id_cols, how="left", validate="one_to_one")
    if merged[replicate_cols].isna().any().any():
        raise RuntimeError("Processed records did not match all TALIS replicate weights.")
    return merged


def fit_paths(data: pd.DataFrame, weight: str) -> tuple[object, object, object]:
    right = f"{X} + C(CNTRY)"
    a_model = smf.wls(f"{MEDIATOR} ~ {right}", data=data, weights=data[weight]).fit()
    c_model = smf.glm(
        f"{Y} ~ {right}", data=data, family=sm.families.Binomial(), freq_weights=data[weight]
    ).fit(maxiter=100, disp=0)
    b_model = smf.glm(
        f"{Y} ~ {right} + {MEDIATOR}",
        data=data,
        family=sm.families.Binomial(),
        freq_weights=data[weight],
    ).fit(maxiter=100, disp=0)
    return a_model, c_model, b_model


def extract_paths(a_model: object, c_model: object, b_model: object) -> dict[str, float]:
    return {
        "a": float(a_model.params[X]),
        "c": float(c_model.params[X]),
        "b": float(b_model.params[MEDIATOR]),
        "c_prime": float(b_model.params[X]),
    }


def brr_se(point: float, replicate_values: list[float]) -> float:
    values = np.asarray(replicate_values, dtype=float)
    return float(math.sqrt(np.sum((values - point) ** 2) / (len(values) * (1 - FAY_FACTOR) ** 2)))


def brr_paths(data: pd.DataFrame, label: str, log_file: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    a_model, c_model, b_model = fit_paths(data, W)
    point = extract_paths(a_model, c_model, b_model)
    values = {name: [] for name in point}
    replicate_rows: list[dict[str, object]] = []
    failures: list[dict[str, object]] = []

    for replicate in range(1, N_REPLICATES + 1):
        weight = f"TRWGT{replicate}"
        try:
            a_rep, c_rep, b_rep = fit_paths(data, weight)
            estimates = extract_paths(a_rep, c_rep, b_rep)
            for parameter, estimate in estimates.items():
                values[parameter].append(estimate)
                replicate_rows.append({
                    "analysis": label,
                    "replicate": replicate,
                    "parameter": parameter,
                    "estimate": estimate,
                    "weight": weight,
                    "status": "converged",
                })
        except Exception as exc:  # Keep a machine-readable audit trail.
            failures.append({"analysis": label, "replicate": replicate, "weight": weight, "error": repr(exc)})
        if replicate % 5 == 0:
            write_log(log_file, f"{label}: completed BRR replicate {replicate}/{N_REPLICATES}")

    if failures:
        raise RuntimeError(f"{label}: {len(failures)} BRR replicates failed; inspect failure log.")

    labels = {
        "a": "AI-related professional learning -> perceived utility (linear composite)",
        "c": "AI-related professional learning -> AI use (log odds)",
        "b": "Perceived utility -> AI use conditional on learning (log odds)",
        "c_prime": "AI-related professional learning -> AI use conditional on utility (log odds)",
    }
    rows: list[dict[str, object]] = []
    for parameter, estimate in point.items():
        se = brr_se(estimate, values[parameter])
        z = estimate / se
        row = {
            "analysis": label,
            "parameter": parameter,
            "description": labels[parameter],
            "n_unweighted": len(data),
            "estimate": estimate,
            "brr_fay_se": se,
            "ci_lower_95": estimate - 1.96 * se,
            "ci_upper_95": estimate + 1.96 * se,
            "z": z,
            "p_two_sided": 2 * norm.sf(abs(z)),
            "fay_factor": FAY_FACTOR,
            "n_replicates": N_REPLICATES,
        }
        if parameter in {"b", "c", "c_prime"}:
            row["odds_ratio"] = math.exp(estimate)
            row["or_ci_lower_95"] = math.exp(row["ci_lower_95"])
            row["or_ci_upper_95"] = math.exp(row["ci_upper_95"])
        rows.append(row)
    return pd.DataFrame(rows), pd.DataFrame(replicate_rows), pd.DataFrame(failures)


def brr_c_only(data: pd.DataFrame, label: str, log_file: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Estimate c once when it is unaffected by imputed mediator items."""
    right = f"{X} + C(CNTRY)"
    point = float(
        smf.glm(f"{Y} ~ {right}", data=data, family=sm.families.Binomial(), freq_weights=data[W]).fit(
            maxiter=100, disp=0
        ).params[X]
    )
    values: list[float] = []
    rows: list[dict[str, object]] = []
    for replicate in range(1, N_REPLICATES + 1):
        weight = f"TRWGT{replicate}"
        estimate = float(
            smf.glm(f"{Y} ~ {right}", data=data, family=sm.families.Binomial(), freq_weights=data[weight]).fit(
                maxiter=100, disp=0
            ).params[X]
        )
        values.append(estimate)
        rows.append({"analysis": label, "replicate": replicate, "parameter": "c", "estimate": estimate, "weight": weight, "status": "converged"})
        if replicate % 5 == 0:
            write_log(log_file, f"{label}: completed BRR replicate {replicate}/{N_REPLICATES}")
    se = brr_se(point, values)
    summary = pd.DataFrame([{
        "analysis": label,
        "parameter": "c",
        "description": "AI-related professional learning -> AI use (log odds)",
        "n_unweighted": len(data),
        "estimate": point,
        "brr_fay_se": se,
        "ci_lower_95": point - 1.96 * se,
        "ci_upper_95": point + 1.96 * se,
        "fay_factor": FAY_FACTOR,
        "n_replicates": N_REPLICATES,
        "odds_ratio": math.exp(point),
        "or_ci_lower_95": math.exp(point - 1.96 * se),
        "or_ci_upper_95": math.exp(point + 1.96 * se),
    }])
    return summary, pd.DataFrame(rows)


def brr_imputed_mediator_paths(
    data: pd.DataFrame, label: str, log_file: Path
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run only paths whose estimates depend on the imputed mediator."""
    right = f"{X} + C(CNTRY)"
    a_model = smf.wls(f"{MEDIATOR} ~ {right}", data=data, weights=data[W]).fit()
    b_model = smf.glm(
        f"{Y} ~ {right} + {MEDIATOR}", data=data, family=sm.families.Binomial(), freq_weights=data[W]
    ).fit(maxiter=100, disp=0)
    point = {"a": float(a_model.params[X]), "b": float(b_model.params[MEDIATOR]), "c_prime": float(b_model.params[X])}
    values = {parameter: [] for parameter in point}
    rows: list[dict[str, object]] = []
    for replicate in range(1, N_REPLICATES + 1):
        weight = f"TRWGT{replicate}"
        a_rep = smf.wls(f"{MEDIATOR} ~ {right}", data=data, weights=data[weight]).fit()
        b_rep = smf.glm(
            f"{Y} ~ {right} + {MEDIATOR}", data=data, family=sm.families.Binomial(), freq_weights=data[weight]
        ).fit(maxiter=100, disp=0)
        estimates = {"a": float(a_rep.params[X]), "b": float(b_rep.params[MEDIATOR]), "c_prime": float(b_rep.params[X])}
        for parameter, estimate in estimates.items():
            values[parameter].append(estimate)
            rows.append({"analysis": label, "replicate": replicate, "parameter": parameter, "estimate": estimate, "weight": weight, "status": "converged"})
        if replicate % 5 == 0:
            write_log(log_file, f"{label}: completed BRR replicate {replicate}/{N_REPLICATES}")
    descriptions = {
        "a": "AI-related professional learning -> perceived utility (linear composite)",
        "b": "Perceived utility -> AI use conditional on learning (log odds)",
        "c_prime": "AI-related professional learning -> AI use conditional on utility (log odds)",
    }
    summary_rows: list[dict[str, object]] = []
    for parameter, estimate in point.items():
        se = brr_se(estimate, values[parameter])
        row = {
            "analysis": label, "parameter": parameter, "description": descriptions[parameter],
            "n_unweighted": len(data), "estimate": estimate, "brr_fay_se": se,
            "ci_lower_95": estimate - 1.96 * se, "ci_upper_95": estimate + 1.96 * se,
            "fay_factor": FAY_FACTOR, "n_replicates": N_REPLICATES,
        }
        if parameter in {"b", "c_prime"}:
            row.update({"odds_ratio": math.exp(estimate), "or_ci_lower_95": math.exp(row["ci_lower_95"]), "or_ci_upper_95": math.exp(row["ci_upper_95"])})
        summary_rows.append(row)
    return pd.DataFrame(summary_rows), pd.DataFrame(rows)


def weighted_mean(values: pd.Series, weights: pd.Series) -> float:
    mask = values.notna() & weights.notna()
    return float(np.average(values.loc[mask], weights=weights.loc[mask])) if mask.any() else float("nan")


def sample_flow(raw_file: Path, processed: pd.DataFrame) -> pd.DataFrame:
    required = ["TT4G21G", "TT4G35A", "TT4G35B", "TT4G35C", "TT4G36", W]
    raw = pd.read_csv(raw_file, sep=";", usecols=required, low_memory=False)
    module = raw["TT4G36"].isin([1, 2]) & raw["TT4G35B"].ne(8)
    xyw = module & raw["TT4G21G"].isin([1, 2]) & raw["TT4G36"].isin([1, 2]) & raw[W].notna()
    valid_items = raw[["TT4G35A", "TT4G35B", "TT4G35C"]].isin([1, 2, 3, 4]).sum(axis=1)
    stages = [
        ("raw_teacher_records", pd.Series(True, index=raw.index)),
        ("ai_module_administered", module),
        ("valid_training_usage_and_weight", xyw),
        ("at_least_one_valid_utility_item", xyw & valid_items.ge(1)),
        ("at_least_two_valid_utility_items", xyw & valid_items.ge(2)),
        ("all_three_utility_items_valid", xyw & valid_items.eq(3)),
    ]
    rows = [{"stage": name, "n_unweighted": int(mask.sum())} for name, mask in stages]
    result = pd.DataFrame(rows)
    result["retained_from_raw_pct"] = result["n_unweighted"] / len(raw) * 100
    result["matches_processed_base"] = result["n_unweighted"].eq(len(processed))
    return result


def coding_audit() -> pd.DataFrame:
    return pd.DataFrame([
        {"role": "X", "source_variable": "TT4G21G", "official_english_item": "Using artificial intelligence for teaching and learning.", "response_options": "1=Yes; 2=No", "special_codes_seen_or_expected": "6/8/9 are excluded from analysis", "analysis_coding": "1->1; 2->0; 6/8/9->missing", "sample_rule": "retain valid AI-module respondents with X, Y and TCHWGT observed", "questionnaire_page": 12},
        {"role": "M_item_1", "source_variable": "TT4G35A", "official_english_item": "Artificial intelligence helps teachers write or improve lesson plans.", "response_options": "1=Strongly disagree; 2=Disagree; 3=Agree; 4=Strongly agree; 5=I don't know", "special_codes_seen_or_expected": "5=I don't know; 8/9 are excluded special codes", "analysis_coding": "retain 1-4; 5/8/9->missing", "sample_rule": "included in the utility composite when observed; partial responses retained for sensitivity and MI", "questionnaire_page": 22},
        {"role": "M_item_2", "source_variable": "TT4G35B", "official_english_item": "Artificial intelligence enables teachers to adapt learning material to different students' abilities.", "response_options": "1=Strongly disagree; 2=Disagree; 3=Agree; 4=Strongly agree; 5=I don't know", "special_codes_seen_or_expected": "5=I don't know; 8/9 are excluded special codes", "analysis_coding": "retain 1-4; 5/8/9->missing", "sample_rule": "TT4G35B=8 identifies non-administered AI-module records and is excluded before analysis", "questionnaire_page": 22},
        {"role": "M_item_3", "source_variable": "TT4G35C", "official_english_item": "Artificial intelligence assists teachers in supporting students individually.", "response_options": "1=Strongly disagree; 2=Disagree; 3=Agree; 4=Strongly agree; 5=I don't know", "special_codes_seen_or_expected": "5=I don't know; 8/9 are excluded special codes", "analysis_coding": "retain 1-4; 5/8/9->missing", "sample_rule": "included in the utility composite when observed; partial responses retained for sensitivity and MI", "questionnaire_page": 22},
        {"role": "audit_only", "source_variable": "TT4G35F", "official_english_item": "Artificial intelligence enables students to misrepresent others' work as their own.", "response_options": "1=Strongly disagree; 2=Disagree; 3=Agree; 4=Strongly agree; 5=I don't know", "special_codes_seen_or_expected": "5 retained only as its questionnaire response; 8/9->missing", "analysis_coding": "not used in the primary mediator; 8/9->missing in the audit field", "sample_rule": "not part of the primary or missing-data mediator composite", "questionnaire_page": 22},
        {"role": "audit_only", "source_variable": "TT4G35G", "official_english_item": "Artificial intelligence makes recommendations that may not be appropriate or correct.", "response_options": "1=Strongly disagree; 2=Disagree; 3=Agree; 4=Strongly agree; 5=I don't know", "special_codes_seen_or_expected": "5 retained only as its questionnaire response; 8/9->missing", "analysis_coding": "not used in the primary mediator; 8/9->missing in the audit field", "sample_rule": "not part of the primary or missing-data mediator composite", "questionnaire_page": 22},
        {"role": "Y", "source_variable": "TT4G36", "official_english_item": "During the last 12 months, have you used artificial intelligence in your teaching or to facilitate student learning?", "response_options": "1=Yes; 2=No", "special_codes_seen_or_expected": "8/9 are excluded special codes", "analysis_coding": "1->1; 2->0; 8/9->missing", "sample_rule": "TT4G36=8 identifies non-administered AI-module records and is excluded before analysis", "questionnaire_page": 22},
    ])


def item_diagnostics(data: pd.DataFrame) -> pd.DataFrame:
    work = data.copy()
    work["valid_item_count"] = work[ITEMS].notna().sum(axis=1)
    rows: list[dict[str, object]] = []
    for count, subset in work.groupby("valid_item_count", dropna=False):
        row = {
            "valid_item_count": int(count),
            "n_unweighted": len(subset),
            "weighted_population_sum": float(subset[W].sum()),
            "weighted_training_mean": weighted_mean(subset[X], subset[W]),
            "weighted_ai_use_mean": weighted_mean(subset[Y], subset[W]),
            "education_systems": int(subset["CNTRY"].nunique()),
        }
        for raw_item in RAW_ITEMS:
            row[f"{raw_item}_dont_know_n"] = int(subset[raw_item].eq(5).sum())
            row[f"{raw_item}_other_missing_n"] = int(subset[raw_item].isin([8, 9]).sum())
        rows.append(row)
    return pd.DataFrame(rows).sort_values("valid_item_count")


def weighted_descriptives(data: pd.DataFrame) -> pd.DataFrame:
    """Create the public S0 table without retaining respondent-level values."""
    work = data.copy()
    work[MEDIATOR] = work[ITEMS].mean(axis=1)
    variables = [
        (X, "AI-related professional learning (1=yes)"),
        (Y, "AI use in teaching or student learning (1=yes)"),
        (MEDIATOR, "Perceived utility composite (1-4)") ,
    ]
    rows: list[dict[str, object]] = []
    for variable, label in variables:
        valid = work.loc[work[variable].notna() & work[W].notna(), [variable, W]]
        mean = weighted_mean(valid[variable], valid[W])
        variance = np.average((valid[variable] - mean) ** 2, weights=valid[W])
        rows.append({
            "variable": variable,
            "label": label,
            "n_unweighted": int(len(valid)),
            "weighted_population_sum": float(valid[W].sum()),
            "weighted_mean": mean,
            "weighted_sd": float(math.sqrt(variance)),
            "education_systems": int(work.loc[valid.index, "CNTRY"].nunique()),
            "inference_class": "weighted descriptive statistic; no confidence interval reported",
        })
    return pd.DataFrame(rows)


def make_mnar_composite(data: pd.DataFrame, delta: float) -> pd.Series:
    observed_mean = data[ITEMS].mean(axis=1)
    filled = data[ITEMS].copy()
    for item in ITEMS:
        filled[item] = filled[item].fillna((observed_mean + delta).clip(1, 4))
    return filled.mean(axis=1)


def point_sensitivity(data: pd.DataFrame, strategy: str, mediator: pd.Series) -> pd.DataFrame:
    sample = data.assign(**{MEDIATOR: mediator}).dropna(subset=[X, Y, W, MEDIATOR]).copy()
    a_model, c_model, b_model = fit_paths(sample, W)
    rows = []
    for parameter, estimate in extract_paths(a_model, c_model, b_model).items():
        rows.append({"strategy": strategy, "parameter": parameter, "n_unweighted": len(sample), "estimate": estimate})
    return pd.DataFrame(rows)


def rubin_mi_with_brr(
    data: pd.DataFrame, n_imputations: int, log_file: Path
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Combine MI uncertainty with Fay-BRR within-imputation variance.

    Each completed dataset is analysed with all 100 TALIS replicate weights.  The
    within-imputation variance is therefore design based; Rubin's rules then add
    between-imputation variance.  This is deliberately distinct from the main
    complete/available-case BRR estimates and from the MNAR point scenarios.
    """
    work = data.reset_index(drop=True)
    countries = pd.get_dummies(work["CNTRY"], prefix="country", drop_first=True, dtype=float)
    mice_input = pd.concat([work[ITEMS], work[[X, Y]], countries], axis=1)
    mice = MICEData(mice_input, perturbation_method="boot", k_pmm=20)
    country_terms = " + ".join(countries.columns)
    for item in ITEMS:
        other_items = [candidate for candidate in ITEMS if candidate != item]
        mice.set_imputer(item, formula=" + ".join([X, Y, *other_items, country_terms]))
    mice.update_all(5)

    individual: list[dict[str, object]] = []
    replicate_rows: list[pd.DataFrame] = []
    c_summary, c_replicates = brr_c_only(work, "MI_c_without_mediator", log_file)
    replicate_rows.append(c_replicates.assign(imputation="not_applicable"))
    for number in range(1, n_imputations + 1):
        mice.update_all(1)
        sample = work[["CNTRY", W, X, Y, *[f"TRWGT{i}" for i in range(1, N_REPLICATES + 1)]]].copy()
        sample[MEDIATOR] = mice.data[ITEMS].mean(axis=1).to_numpy()
        summary, replicates = brr_imputed_mediator_paths(sample, f"MI_{number:02d}", log_file)
        replicate_rows.append(replicates.assign(imputation=number))
        for row in summary.to_dict("records"):
            individual.append({
                "imputation": number,
                "parameter": row["parameter"],
                "n_unweighted": row["n_unweighted"],
                "estimate": row["estimate"],
                "brr_fay_se_within_imputation": row["brr_fay_se"],
                "brr_fay_variance_within_imputation": row["brr_fay_se"] ** 2,
            })
        write_log(log_file, f"B03: completed MI+BRR dataset {number}/{n_imputations}")

    individual_frame = pd.DataFrame(individual)
    rows: list[dict[str, object]] = []
    for parameter, group in individual_frame.groupby("parameter"):
        m = len(group)
        qbar = float(group["estimate"].mean())
        ubar = float(group["brr_fay_variance_within_imputation"].mean())
        between = float(group["estimate"].var(ddof=1))
        total = ubar + (1 + 1 / m) * between
        relative_increase = ((1 + 1 / m) * between / ubar) if ubar > 0 else float("inf")
        degrees_freedom = (m - 1) * (1 + 1 / relative_increase) ** 2 if relative_increase > 0 else float("inf")
        critical = student_t.ppf(0.975, degrees_freedom) if math.isfinite(degrees_freedom) else 1.96
        se = math.sqrt(total)
        rows.append({
            "parameter": parameter,
            "n_imputations": m,
            "n_unweighted_per_imputation": int(group["n_unweighted"].iloc[0]),
            "estimate_mi_brr_pooled": qbar,
            "se_mi_brr_pooled": se,
            "ci_lower_95": qbar - critical * se,
            "ci_upper_95": qbar + critical * se,
            "within_brr_variance_mean": ubar,
            "between_variance": between,
            "degrees_freedom": degrees_freedom,
            "inference_class": "MI plus Fay-BRR combined 95% CI",
            "variance_scope": "Mean within-imputation Fay-BRR variance plus Rubin between-imputation variance",
        })
    c_row = c_summary.iloc[0]
    rows.append({
        "parameter": "c",
        "n_imputations": 0,
        "n_unweighted_per_imputation": int(c_row["n_unweighted"]),
        "estimate_mi_brr_pooled": c_row["estimate"],
        "se_mi_brr_pooled": c_row["brr_fay_se"],
        "ci_lower_95": c_row["ci_lower_95"],
        "ci_upper_95": c_row["ci_upper_95"],
        "within_brr_variance_mean": c_row["brr_fay_se"] ** 2,
        "between_variance": 0.0,
        "degrees_freedom": float("nan"),
        "inference_class": "Fay-BRR 95% CI; c does not depend on imputed mediator items",
        "variance_scope": "Fay-BRR variance from the model without the mediator",
    })
    return pd.DataFrame(rows), individual_frame, pd.concat(replicate_rows, ignore_index=True)


def model_comparison(data: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    specifications = [
        ("unweighted_pooled", None, False),
        ("weighted_pooled", W, False),
        ("weighted_country_fixed_effects", W, True),
    ]
    for label, weight, fixed_effects in specifications:
        if fixed_effects:
            a_model, c_model, b_model = fit_paths(data, W)
        elif weight is None:
            a_model = smf.ols(f"{MEDIATOR} ~ {X}", data=data).fit()
            c_model = smf.glm(f"{Y} ~ {X}", data=data, family=sm.families.Binomial()).fit()
            b_model = smf.glm(f"{Y} ~ {X} + {MEDIATOR}", data=data, family=sm.families.Binomial()).fit()
        else:
            a_model = smf.wls(f"{MEDIATOR} ~ {X}", data=data, weights=data[W]).fit()
            c_model = smf.glm(f"{Y} ~ {X}", data=data, family=sm.families.Binomial(), freq_weights=data[W]).fit()
            b_model = smf.glm(f"{Y} ~ {X} + {MEDIATOR}", data=data, family=sm.families.Binomial(), freq_weights=data[W]).fit()
        row = {"specification": label, "n_unweighted": len(data), **extract_paths(a_model, c_model, b_model)}
        rows.append(row)
    return pd.DataFrame(rows)


def write_static_documents(folders: dict[str, Path]) -> None:
    (folders["B01"] / "model_specification.md").write_text(
        "# B01 模型与方差估计\n\n"
        "主模型使用最终教师权重 `TCHWGT`、教育系统固定效应 `C(CNTRY)`，并使用 `TRWGT1` 至 `TRWGT100` 的 Fay-BRR 复本权重。"
        "Fay 系数为 0.5；标准误公式为 `sqrt(sum((theta_r-theta)^2)/(R*(1-f)^2))`。"
        "模型同时保存每个复本和失败日志。报告的 a 路径为加权线性模型；b、c、c' 为加权逻辑回归的对数优势比。\n",
        encoding="utf-8",
    )
    (folders["B02"] / "association_only_decision.md").write_text(
        "# B02 方法决策记录\n\n"
        "审稿意见指出，线性 a 路径与 logit b 路径的系数乘积不能作为二元结局的间接效应。"
        "本次补充包因此采用**关联路径报告路线**：只报告设计校正的 a、b、c、c'，图中不画因果箭头，"
        "不报告旧版 `0.337`、`20.9%` 或 `0.0667` 作为正式间接效应。\n\n"
        "若作者决定保留量化中介，须另行实施完整的概率尺度 g-computation/中介模拟、对中介分布积分，"
        "并把全过程纳入 BRR 方差估计；该选择尚未由作者确认，故不在本包中声称完成。\n",
        encoding="utf-8",
    )
    (folders["B03"] / "mi_scope_and_mnar_note.md").write_text(
        "# B03 多重插补与 MNAR 说明\n\n"
        "多重插补采用 PMM 链式方程、20 个插补数据集，并按 Rubin 规则合并加权且含教育系统固定效应的模型协方差。"
        "该表明确标注为 MI 敏感性结果：其方差尚未与 TALIS BRR 复本方差二次合并，不能替代 B01 的正式设计校正推断。"
        "MNAR 情景将每位有至少一题有效回答的受访者，其缺失条目填为个人已观察条目均值加/减 0.5 分（范围截断为 1--4）；"
        "这是透明的情景敏感性检验，不是缺失机制已被识别的主张。\n",
        encoding="utf-8",
    )
    (folders["B02"] / "formal_statistical_methods.md").write_text(
        "# Statistical methods: association-path analysis\n\n"
        "The analysis estimates four cross-sectional association paths, rather than a causal or "
        "quantitative mediation effect. AI-related professional learning (TT4G21G) was coded 1 for "
        "Yes and 0 for No. The perceived-utility score was the mean of observed TT4G35A-C responses "
        "coded 1-4. AI use (TT4G36) was coded 1 for Yes and 0 for No. Teachers with valid exposure, "
        "outcome, and final teacher weight were retained; the primary available-item analysis required "
        "at least one valid perceived-utility item.\n\n"
        "The a path was estimated by weighted linear regression of the utility score on professional "
        "learning and education-system fixed effects. The c path was estimated by weighted logistic "
        "regression of AI use on professional learning and the same fixed effects. The b and c-prime "
        "paths were estimated jointly by weighted logistic regression of AI use on professional learning, "
        "utility, and fixed effects. Final teacher weights (TCHWGT) were used for point estimation.\n\n"
        "For complete/available-item analyses, standard errors and 95% confidence intervals use all 100 "
        "teacher replicate weights with Fay's balanced repeated replication (Fay factor 0.5). Sensitivity "
        "analyses compare at-least-one, at-least-two, and all-three valid utility items. The MNAR delta "
        "scenarios report point estimates only. Twenty chained-equation predictive-mean-matching imputations "
        "were also analysed with all 100 replicate weights within each imputation for paths involving the "
        "imputed mediator; within-imputation Fay-BRR variance and between-imputation variance were combined "
        "using Rubin's rules. The c path does not use the mediator and retains its direct Fay-BRR interval.\n\n"
        "Because the outcome model is logistic and the data are cross-sectional, no product-of-coefficients, "
        "proportion-mediated, or causal indirect effect is reported.\n",
        encoding="utf-8",
    )
    (folders["B03"] / "inference_key.md").write_text(
        "# Inference key\n\n"
        "- `TableS3_missingness_and_complete_case_sensitivity.csv`: at-least-one, at-least-two, and "
        "all-three item analyses have Fay-BRR 95% confidence intervals; MNAR delta scenarios have point "
        "estimates only.\n"
        "- `TableS4_multiple_imputation_design_brr.csv`: confidence intervals combine within-imputation "
        "Fay-BRR variance with between-imputation variance using Rubin's rules.\n"
        "- No table in this package labels a model-covariance-only confidence interval as design based.\n",
        encoding="utf-8",
    )


def write_static_documents(folders: dict[str, Path]) -> None:
    """Write readable, submission-ready method and inference notes."""
    (folders["B01"] / "model_specification.md").write_text(
        "# B01 Model specification and variance estimation\n\n"
        "Point estimates use the TALIS final teacher weight (`TCHWGT`) and fixed effects for all "
        "education systems (`C(CNTRY)`). The a path is a weighted linear model. The b, c and c-prime "
        "paths are weighted logistic models reported on the log-odds scale and, where appropriate, as "
        "odds ratios.\n\n"
        "Design-based standard errors use `TRWGT1` through `TRWGT100` with Fay balanced repeated "
        "replication (Fay factor 0.5): `sqrt(sum((theta_r-theta)^2)/(R*(1-f)^2))`. "
        "Replicate estimates and failures are retained as public model-level audit outputs.\n",
        encoding="utf-8",
    )
    (folders["B02"] / "association_only_decision.md").write_text(
        "# B02 Association-path reporting decision\n\n"
        "The final analysis reports only the four design-adjusted association paths a, b, c and c-prime. "
        "It does not report a coefficient-product indirect effect, a proportion mediated, or a causal "
        "mediation effect.\n\n"
        "A linear a path cannot be multiplied by a logistic b path and interpreted as an indirect effect "
        "for a binary outcome. Any future quantitative mediation analysis would require a separately "
        "specified probability-scale procedure and the full variance calculation inside the replicate "
        "design. It is intentionally outside this release.\n",
        encoding="utf-8",
    )
    (folders["B03"] / "mi_scope_and_mnar_note.md").write_text(
        "# B03 Multiple imputation and MNAR note\n\n"
        "Twenty predictive-mean-matching chained-equation imputations are generated for the three "
        "perceived-utility items. Paths involving the imputed mediator are analysed with all 100 Fay-BRR "
        "replicate weights within each completed dataset. The public MI table combines mean within-"
        "imputation Fay-BRR variance and between-imputation variance by Rubin's rules. The c path is "
        "unaffected by the imputed mediator and has a direct Fay-BRR interval.\n\n"
        "The MNAR delta scenarios replace missing utility items by each respondent's observed-item mean "
        "minus or plus 0.5, bounded to 1-4. They are transparent point-estimate sensitivity scenarios, "
        "not estimates of an identified missingness mechanism; no confidence interval is reported.\n",
        encoding="utf-8",
    )
    (folders["B02"] / "formal_statistical_methods.md").write_text(
        "# Statistical methods: association-path analysis\n\n"
        "The analysis estimates four cross-sectional association paths, rather than a causal or "
        "quantitative mediation effect. AI-related professional learning (TT4G21G) was coded 1 for "
        "Yes and 0 for No. The perceived-utility score was the mean of observed TT4G35A-C responses "
        "coded 1-4. AI use (TT4G36) was coded 1 for Yes and 0 for No. Teachers with valid exposure, "
        "outcome, and final teacher weight were retained; the primary available-item analysis required "
        "at least one valid perceived-utility item.\n\n"
        "The a path was estimated by weighted linear regression of the utility score on professional "
        "learning and education-system fixed effects. The c path was estimated by weighted logistic "
        "regression of AI use on professional learning and the same fixed effects. The b and c-prime "
        "paths were estimated jointly by weighted logistic regression of AI use on professional learning, "
        "utility, and fixed effects. Final teacher weights (TCHWGT) were used for point estimation.\n\n"
        "For complete/available-item analyses, standard errors and 95% confidence intervals use all 100 "
        "teacher replicate weights with Fay's balanced repeated replication (Fay factor 0.5). Sensitivity "
        "analyses compare at-least-one, at-least-two, and all-three valid utility items. The MNAR delta "
        "scenarios report point estimates only. Twenty chained-equation predictive-mean-matching imputations "
        "were also analysed with all 100 replicate weights within each imputation for paths involving the "
        "imputed mediator; within-imputation Fay-BRR variance and between-imputation variance were combined "
        "using Rubin's rules. The c path does not use the mediator and retains its direct Fay-BRR interval.\n\n"
        "Because the outcome model is logistic and the data are cross-sectional, no product-of-coefficients, "
        "proportion-mediated, or causal indirect effect is reported.\n",
        encoding="utf-8",
    )
    (folders["B03"] / "inference_key.md").write_text(
        "# Inference key\n\n"
        "- `TableS3_missingness_and_complete_case_sensitivity.csv`: at-least-one, at-least-two, and "
        "all-three item analyses have Fay-BRR 95% confidence intervals; MNAR delta scenarios have point "
        "estimates only.\n"
        "- `TableS4_multiple_imputation_design_brr.csv`: confidence intervals combine within-imputation "
        "Fay-BRR variance with between-imputation variance using Rubin's rules.\n"
        "- No table in this package labels a model-covariance-only confidence interval as design based.\n",
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    if not args.raw_file.exists():
        raise FileNotFoundError(args.raw_file)
    if not args.processed_file.exists():
        raise FileNotFoundError(args.processed_file)
    np.random.seed(args.seed)
    folders = make_dirs(args.output_dir)
    log_file = folders["B01"] / "reproduction.log"
    log_file.write_text("", encoding="utf-8")
    run_mode = "reuse_brr" if args.reuse_brr else "full_raw_to_results"
    write_log(log_file, f"started revision analysis package; run_mode={run_mode}")
    write_log(log_file, "raw input is external restricted TALIS teacher microdata; no row-level data will be written")

    write_log(log_file, "reading derived working file")
    processed = pd.read_csv(args.processed_file)
    processed["valid_item_count"] = processed[ITEMS].notna().sum(axis=1)
    coding_audit().to_csv(folders["B04"] / "coding_audit.csv", index=False)
    sample_flow(args.raw_file, processed).to_csv(folders["B04"] / "sample_flow.csv", index=False)
    item_diagnostics(processed).to_csv(folders["B03"] / "item_nonresponse_diagnostics.csv", index=False)

    write_log(log_file, "attaching 100 replicate weights from raw source")
    weighted = attach_replicate_weights(processed, args.raw_file)
    write_log(log_file, "replicate weights attached")
    primary = weighted.loc[weighted["valid_item_count"].ge(1)].copy()
    primary[MEDIATOR] = primary[ITEMS].mean(axis=1)
    if args.reuse_brr:
        main_table = pd.read_csv(folders["B01"] / "main_paths_brr.csv")
        main_replicates = pd.read_csv(folders["B01"] / "brr_replicate_estimates.csv")
        failure_file = folders["B01"] / "brr_failures.csv"
        try:
            failures = pd.read_csv(failure_file)
        except pd.errors.EmptyDataError:
            failures = pd.DataFrame(columns=["analysis", "replicate", "weight", "error"])
        write_log(log_file, "reused completed B01 BRR outputs")
    else:
        write_log(log_file, "starting primary Fay-BRR analysis")
        main_table, main_replicates, failures = brr_paths(primary, "at_least_one_valid_item_primary", log_file)
        main_table.to_csv(folders["B01"] / "main_paths_brr.csv", index=False)
        main_replicates.to_csv(folders["B01"] / "brr_replicate_estimates.csv", index=False)
        failures.to_csv(folders["B01"] / "brr_failures.csv", index=False)
    main_table.to_csv(folders["B01"] / "main_paths_brr.csv", index=False)
    main_replicates.to_csv(folders["B01"] / "brr_replicate_estimates.csv", index=False)
    failures.to_csv(folders["B01"] / "brr_failures.csv", index=False)
    model_comparison(primary).to_csv(folders["B01"] / "weighting_comparison.csv", index=False)

    if args.reuse_brr:
        sensitivity_frame = pd.read_csv(folders["B03"] / "missing_data_sensitivity.csv")
    else:
        write_log(log_file, "starting item-completeness Fay-BRR sensitivity analyses")
        sensitivity_tables: list[pd.DataFrame] = [main_table.assign(analysis="at_least_one_valid_item")]
        for label, selector in [
            ("at_least_two_valid_items", weighted["valid_item_count"].ge(2)),
            ("all_three_valid_items", weighted["valid_item_count"].eq(3)),
        ]:
            subset = weighted.loc[selector].copy()
            subset[MEDIATOR] = subset[ITEMS].mean(axis=1)
            summary, _, _ = brr_paths(subset, label, log_file)
            sensitivity_tables.append(summary)
        mnar_low = make_mnar_composite(weighted, -0.5)
        mnar_high = make_mnar_composite(weighted, 0.5)
        sensitivity_tables.extend([
            point_sensitivity(weighted, "MNAR_missing_items_minus_0_5", mnar_low),
            point_sensitivity(weighted, "MNAR_missing_items_plus_0_5", mnar_high),
        ])
        sensitivity_frame = pd.concat(sensitivity_tables, ignore_index=True, sort=False)
        sensitivity_frame.to_csv(folders["B03"] / "missing_data_sensitivity.csv", index=False)

    mi_summary, mi_individual, mi_replicates = rubin_mi_with_brr(weighted, args.n_imputations, log_file)
    mi_summary.to_csv(folders["B03"] / "multiple_imputation_design_brr.csv", index=False)
    mi_individual.to_csv(folders["B03"] / "multiple_imputation_individual.csv", index=False)
    mi_replicates.to_csv(folders["B03"] / "multiple_imputation_brr_replicates.csv", index=False)
    (folders["B03"] / "multiple_imputation_rubin_pooled.csv").unlink(missing_ok=True)
    write_static_documents(folders)

    main_table.to_csv(folders["B05"] / "Table2_design_adjusted_association_paths.csv", index=False)
    item_diagnostics(processed).to_csv(folders["B05"] / "Table1_item_response_and_sample_summary.csv", index=False)
    weighted_descriptives(primary).to_csv(folders["B05"] / "TableS0_weighted_descriptive_statistics.csv", index=False)
    main_table.assign(inference_class="Fay-BRR 95% CI").to_csv(
        folders["B05"] / "TableS1_primary_design_adjusted_paths.csv", index=False
    )
    model_comparison(primary).assign(
        inference_class="point estimates for specification comparison; no design-based CI"
    ).to_csv(folders["B05"] / "TableS2_weighting_and_fixed_effects_comparison.csv", index=False)
    sensitivity_frame.assign(
        inference_class=np.where(
            sensitivity_frame["strategy"].fillna("").str.startswith("MNAR"),
            "point estimate only; no CI",
            "Fay-BRR 95% CI",
        )
    ).to_csv(folders["B05"] / "TableS3_missingness_and_complete_case_sensitivity.csv", index=False)
    mi_summary.to_csv(folders["B05"] / "TableS4_multiple_imputation_design_brr.csv", index=False)
    manifest = pd.DataFrame([
        {"artifact": "B05/TableS0_weighted_descriptive_statistics.csv", "status": "weighted descriptive statistics; no CI"},
        {"artifact": "B05/TableS1_primary_design_adjusted_paths.csv", "status": "Fay-BRR 95% CI"},
        {"artifact": "B05/TableS2_weighting_and_fixed_effects_comparison.csv", "status": "point estimates only"},
        {"artifact": "B05/TableS3_missingness_and_complete_case_sensitivity.csv", "status": "BRR CI for item-completeness analyses; MNAR point estimates only"},
        {"artifact": "B05/TableS4_multiple_imputation_design_brr.csv", "status": "MI plus Fay-BRR combined 95% CI"},
        {"artifact": "B01/main_paths_brr.csv", "status": "final_design_adjusted_association_paths"},
        {"artifact": "B03/missing_data_sensitivity.csv", "status": "sensitivity"},
        {"artifact": "B03/multiple_imputation_design_brr.csv", "status": "sensitivity_design_variance_combined"},
        {"artifact": "B02/association_only_decision.md", "status": "author_decision_pending_for_quantitative_mediation"},
    ])
    manifest.to_csv(folders["B05"] / "release_manifest.csv", index=False)
    metadata = {
        "analysis_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "processed_records": int(len(processed)),
        "primary_records": int(len(primary)),
        "education_systems_primary": int(primary["CNTRY"].nunique()),
        "brr_replicate_failures": int(len(failures)),
        "n_imputations": args.n_imputations,
        "imputation_seed": args.seed,
        "python": sys.version,
        "platform": platform.platform(),
        "git_revision": git_revision(),
        "git_worktree_status": git_worktree_status(),
        "source_file_sha256": source_manifest(),
        "run_mode": run_mode,
        "raw_file_name": args.raw_file.name,
        "raw_file_bytes": args.raw_file.stat().st_size,
        "raw_file_sha256": sha256_file(args.raw_file),
        "primary_claim_scope": "association only; no final quantitative indirect effect",
    }
    (folders["B06"] / "run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    (folders["B06"] / "reproduction_audit.md").write_text(
        "# 可复现性审计\n\n"
        "本次运行读取仓库外的受限 TALIS 原始教师微观数据，仅在仓库中保留派生、非识别性结果。"
        "`B01/reproduction.log` 记录运行阶段；`B01/brr_replicate_estimates.csv` 保存 100 个复本的逐参数估计；"
        "`B01/brr_failures.csv` 保存失败记录。请先运行 `code/01_data_preparation.py`，再运行 `code/run_all.py`。\n",
        encoding="utf-8",
    )
    write_log(log_file, "completed revision analysis package")


if __name__ == "__main__":
    main()
