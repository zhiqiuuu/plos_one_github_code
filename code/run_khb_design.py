"""Run the taskbook-specified KHB-style decomposition with Fay-BRR inference.

The calculation is a statistical decomposition on the logistic-coefficient
scale. It is not a causal mediation analysis and it does not estimate an ACME
or a probability-scale indirect effect.
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import statsmodels
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy.stats import norm

from run_revision_package import (
    FAY_FACTOR,
    ITEMS,
    MEDIATOR,
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
ANCHOR_COEFFICIENTS = {
    ("primary_available_item", "total_raw"): 1.5417220924847053,
    ("primary_available_item", "total_adjusted"): 1.7081583978375159,
    ("primary_available_item", "direct"): 1.3869881248112819,
    ("primary_available_item", "indirect"): 0.3211702730262340,
    ("primary_available_item", "rescaling"): 0.1664363053528106,
    ("strict_complete_case", "total_raw"): 1.5639773810841924,
    ("strict_complete_case", "total_adjusted"): 1.7513663249691145,
    ("strict_complete_case", "direct"): 1.3862886054464232,
    ("strict_complete_case", "indirect"): 0.3650777195226913,
    ("strict_complete_case", "rescaling"): 0.1873889438849221,
}
FAILURE_COLUMNS = ["analysis", "replicate", "weight", "error"]
REPLICATE_COLUMNS = ["analysis", "replicate", "parameter", "estimate", "weight", "status"]
KHB_PARAMETERS = ["total_raw", "total_adjusted", "direct", "indirect", "rescaling"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-file", type=Path, required=True, help="Restricted TALIS ttgintt4.csv file")
    parser.add_argument("--processed-file", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20261115, help="Recorded for audit; analysis is deterministic")
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


def weighted_residualize_mediator(data: pd.DataFrame, weight: str) -> tuple[pd.Series, object]:
    """Residualize the mediator on X and education-system fixed effects."""
    projection = smf.wls(
        f"{MEDIATOR} ~ {X} + C(CNTRY)", data=data, weights=data[weight]
    ).fit()
    return data[MEDIATOR] - projection.fittedvalues, projection


def fit_logistic(formula: str, data: pd.DataFrame, weight: str) -> object:
    """Fit one replicate-specific weighted logistic model and require convergence."""
    result = smf.glm(
        formula,
        data=data,
        family=sm.families.Binomial(),
        freq_weights=data[weight],
    ).fit(maxiter=100, disp=0)
    if not bool(getattr(result, "converged", True)):
        raise RuntimeError(f"Weighted logistic model did not converge for {weight}: {formula}")
    return result


def fit_khb_components(data: pd.DataFrame, weight: str) -> dict[str, float]:
    """Fit taskbook reduced, full, and residualized logistic models."""
    reduced = fit_logistic(f"{Y} ~ {X} + C(CNTRY)", data, weight)
    mediator_residual, _ = weighted_residualize_mediator(data, weight)
    full = fit_logistic(f"{Y} ~ {X} + {MEDIATOR} + C(CNTRY)", data, weight)
    rescaled = fit_logistic(
        f"{Y} ~ {X} + mediator_residual + C(CNTRY)",
        data.assign(mediator_residual=mediator_residual),
        weight,
    )
    total_raw = float(reduced.params[X])
    total_adjusted = float(rescaled.params[X])
    direct = float(full.params[X])
    return {
        "total_raw": total_raw,
        "total_adjusted": total_adjusted,
        "direct": direct,
        "indirect": total_adjusted - direct,
        "rescaling": total_adjusted - total_raw,
    }


def component_description(parameter: str) -> str:
    descriptions = {
        "total_raw": "Reduced-model coefficient for AI-related professional learning",
        "total_adjusted": "Residualized-mediator KHB total coefficient",
        "direct": "Full-model coefficient for AI-related professional learning",
        "indirect": "KHB statistical indirect component: total_adjusted minus direct",
        "rescaling": "KHB rescaling component: total_adjusted minus total_raw",
    }
    return descriptions[parameter]


def summarise_brr(
    analysis: str,
    point: dict[str, float],
    replicate_values: dict[str, list[float]],
    data: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for parameter, estimate in point.items():
        se = brr_se(estimate, replicate_values[parameter])
        z = estimate / se
        rows.append(
            {
                "analysis": analysis,
                "parameter": parameter,
                "description": component_description(parameter),
                "coefficient_scale": "log odds",
                "n_unweighted": int(len(data)),
                "n_systems": int(data["CNTRY"].nunique()),
                "estimate": estimate,
                "brr_fay_se": se,
                "z": z,
                "p_two_sided": 2 * norm.sf(abs(z)),
                "ci_lower_95": estimate - 1.96 * se,
                "ci_upper_95": estimate + 1.96 * se,
                "fay_factor": FAY_FACTOR,
                "n_replicates": N_REPLICATES,
                "inference_class": "Fay-BRR 95% CI",
            }
        )
    return pd.DataFrame(rows)


def run_khb_brr(
    data: pd.DataFrame, analysis: str, log_file: Path
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    point = fit_khb_components(data, W)
    values = {parameter: [] for parameter in point}
    replicates: list[dict[str, object]] = []
    failures: list[dict[str, object]] = []
    for replicate in range(1, N_REPLICATES + 1):
        weight = f"TRWGT{replicate}"
        try:
            estimates = fit_khb_components(data, weight)
            for parameter, estimate in estimates.items():
                values[parameter].append(estimate)
                replicates.append(
                    {
                        "analysis": analysis,
                        "replicate": replicate,
                        "parameter": parameter,
                        "estimate": estimate,
                        "weight": weight,
                        "status": "converged",
                    }
                )
        except Exception as exc:
            failures.append(
                {"analysis": analysis, "replicate": replicate, "weight": weight, "error": repr(exc)}
            )
        if replicate % 5 == 0:
            write_log(log_file, f"{analysis}: completed BRR replicate {replicate}/{N_REPLICATES}")
    replicate_frame = pd.DataFrame(replicates, columns=REPLICATE_COLUMNS)
    failure_frame = pd.DataFrame(failures, columns=FAILURE_COLUMNS)
    if not failure_frame.empty:
        return pd.DataFrame(), replicate_frame, failure_frame
    return summarise_brr(analysis, point, values, data), replicate_frame, failure_frame


def identity_checks(summary: pd.DataFrame, replicates: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    point_wide = summary.pivot(index="analysis", columns="parameter", values="estimate")
    for analysis, values in point_wide.iterrows():
        rows.extend(
            [
                {
                    "analysis": analysis,
                    "level": "point",
                    "replicate": 0,
                    "identity": "total_adjusted_equals_direct_plus_indirect",
                    "absolute_error": abs(values["total_adjusted"] - values["direct"] - values["indirect"]),
                },
                {
                    "analysis": analysis,
                    "level": "point",
                    "replicate": 0,
                    "identity": "rescaling_equals_total_adjusted_minus_total_raw",
                    "absolute_error": abs(values["rescaling"] - values["total_adjusted"] + values["total_raw"]),
                },
            ]
        )
    wide_replicates = replicates.pivot_table(
        index=["analysis", "replicate"], columns="parameter", values="estimate", aggfunc="first"
    )
    for (analysis, replicate), values in wide_replicates.iterrows():
        rows.extend(
            [
                {
                    "analysis": analysis,
                    "level": "replicate",
                    "replicate": int(replicate),
                    "identity": "total_adjusted_equals_direct_plus_indirect",
                    "absolute_error": abs(values["total_adjusted"] - values["direct"] - values["indirect"]),
                },
                {
                    "analysis": analysis,
                    "level": "replicate",
                    "replicate": int(replicate),
                    "identity": "rescaling_equals_total_adjusted_minus_total_raw",
                    "absolute_error": abs(values["rescaling"] - values["total_adjusted"] + values["total_raw"]),
                },
            ]
        )
    result = pd.DataFrame(rows)
    result["tolerance"] = 1e-10
    result["passed"] = result["absolute_error"].le(result["tolerance"])
    return result


def anchor_checks(summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    indexed = summary.set_index(["analysis", "parameter"])
    for (analysis, parameter), expected in ANCHOR_COEFFICIENTS.items():
        observed = float(indexed.loc[(analysis, parameter), "estimate"])
        difference = abs(observed - expected)
        rows.append(
            {
                "analysis": analysis,
                "parameter": parameter,
                "expected_taskbook_coefficient": expected,
                "observed_khb_coefficient": observed,
                "absolute_difference": difference,
                "tolerance": 1e-6,
                "passed": difference <= 1e-6,
            }
        )
    return pd.DataFrame(rows)


def replicate_audit(replicates: pd.DataFrame) -> pd.DataFrame:
    """Confirm every requested coefficient has exactly 100 converged replicate fits."""
    rows: list[dict[str, object]] = []
    for analysis in ("primary_available_item", "strict_complete_case"):
        for parameter in KHB_PARAMETERS:
            subset = replicates.loc[
                replicates["analysis"].eq(analysis) & replicates["parameter"].eq(parameter)
            ]
            completed = int(subset["replicate"].nunique())
            rows.append(
                {
                    "analysis": analysis,
                    "parameter": parameter,
                    "expected_replicates": N_REPLICATES,
                    "converged_replicates": completed,
                    "failed_replicates": int(subset["status"].ne("converged").sum()),
                    "complete_fay_brr": completed == N_REPLICATES
                    and len(subset) == N_REPLICATES
                    and subset["status"].eq("converged").all(),
                }
            )
    return pd.DataFrame(rows)


def sample_checks(primary: pd.DataFrame, complete: pd.DataFrame) -> pd.DataFrame:
    result = pd.DataFrame(
        [
            {"analysis": "primary_available_item", "observed_n": len(primary), "expected_n": 72000},
            {"analysis": "strict_complete_case", "observed_n": len(complete), "expected_n": 60447},
        ]
    )
    result["passed"] = result["observed_n"].eq(result["expected_n"])
    return result


def _row(summary: pd.DataFrame, analysis: str, parameter: str) -> pd.Series:
    return summary.loc[
        summary["analysis"].eq(analysis) & summary["parameter"].eq(parameter)
    ].iloc[0]


def write_methods(output_dir: Path, summary: pd.DataFrame) -> None:
    primary_indirect = _row(summary, "primary_available_item", "indirect")
    primary_direct = _row(summary, "primary_available_item", "direct")
    complete_indirect = _row(summary, "strict_complete_case", "indirect")
    text = f"""# KHB statistical decomposition

## Analysis specification

The KHB-style decomposition used the taskbook-specified primary available-item
sample (n = 72,000) and strict complete-case sample (n = 60,447) separately.
In each sample, the reduced weighted logistic model was AI use on AI-related
professional learning and education-system fixed effects. The perceived-utility
composite was projected by weighted linear regression on the same covariates.
Its respondent-level residual was included in a residualized logistic model.
The full logistic model included the original perceived-utility composite. The
KHB statistical indirect component was beta_total_adjusted minus beta_direct,
and the rescaling component was beta_total_adjusted minus beta_total_raw.

Point estimates use TCHWGT. For each of 100 Fay-BRR replicate weights, the
mediator projection, residual, all three logistic models, and both differences
were re-estimated. Standard errors use
sqrt(sum((theta_r-theta)^2)/(100*(1-0.5)^2)); 95% intervals are estimate plus
or minus 1.96 standard errors. Coefficients are on the logistic log-odds scale.
This is a KHB statistical decomposition following Karlson, Holm, and Breen
(2012), not a causal mediation analysis: the cross-sectional design does not
establish temporal ordering or rule out unmeasured confounding.

## Manuscript-ready English Methods

For a supplementary non-linear decomposition, we applied a KHB-style
residualization procedure separately in the available-item and complete-case
samples. We estimated a weighted reduced logistic model for AI use, a weighted
linear projection of perceived utility on AI-related professional learning and
education-system fixed effects, a weighted full logistic model including
perceived utility, and a residualized logistic model including the projection
residual. The KHB indirect and rescaling components were defined as differences
between the residualized, full, and reduced professional-learning coefficients.
Point estimates used TALIS final teacher weights; all steps were repeated for
the 100 Fay-BRR weights (Fay factor 0.5). Results are reported on the log-odds
scale and are interpreted as cross-sectional statistical decompositions rather
than causal mediation effects.

## Manuscript-ready English Results

In the primary available-item sample, the KHB statistical indirect component
was {primary_indirect['estimate']:.3f} log odds (Fay-BRR SE
{primary_indirect['brr_fay_se']:.3f}, 95% CI
{primary_indirect['ci_lower_95']:.3f} to {primary_indirect['ci_upper_95']:.3f});
the corresponding direct component was {primary_direct['estimate']:.3f} log
odds (95% CI {primary_direct['ci_lower_95']:.3f} to
{primary_direct['ci_upper_95']:.3f}). In the strict complete-case analysis,
the KHB indirect component was {complete_indirect['estimate']:.3f} log odds
(95% CI {complete_indirect['ci_lower_95']:.3f} to
{complete_indirect['ci_upper_95']:.3f}). These estimates are statistical
decomposition components and are not reported as causal indirect effects.

## Implementation validation

khb_identity_checks.csv verifies both decomposition identities for each point
estimate and all 200 replicate analyses. Automated tests use a synthetic,
fixed-effect data set to verify weighted residual orthogonality to the
projection design and the two difference identities. This is an equivalence
check of the documented KHB residualization algorithm; it does not claim
validation against an unavailable external microdata implementation.
"""
    (output_dir / "B02" / "khb_methods.md").write_text(text, encoding="utf-8")


def write_metadata(
    output_dir: Path,
    args: argparse.Namespace,
    summaries: pd.DataFrame,
    failures: pd.DataFrame,
) -> None:
    analysis_sizes = (
        summaries.groupby("analysis")["n_unweighted"].first().to_dict()
        if not summaries.empty
        else {}
    )
    metadata = {
        "analysis": "KHB-style logistic-coefficient decomposition",
        "analysis_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "git_revision": git_revision(),
        "git_worktree_status": git_worktree_status(),
        "python": sys.version,
        "platform": platform.platform(),
        "statsmodels": statsmodels.__version__,
        "raw_file_name": args.raw_file.name,
        "raw_file_sha256": sha256_file(args.raw_file),
        "processed_file_sha256": sha256_file(args.processed_file),
        "source_file_sha256": source_manifest(),
        "seed_recorded": args.seed,
        "n_replicates": N_REPLICATES,
        "fay_factor": FAY_FACTOR,
        "replicate_failure_count": int(len(failures)),
        "analyses": analysis_sizes,
    }
    (output_dir / "B06" / "khb_run_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )


def main() -> None:
    args = parse_args()
    for required in (args.raw_file, args.processed_file):
        if not required.exists():
            raise FileNotFoundError(required)
    for folder in ("B02", "B05", "B06"):
        (args.output_dir / folder).mkdir(parents=True, exist_ok=True)
    log_file = args.output_dir / "B06" / "khb_reproduction.log"
    log_file.write_text("", encoding="utf-8")
    write_log(log_file, "started KHB taskbook analysis from external restricted TALIS microdata")
    processed = pd.read_csv(args.processed_file)
    required_columns = {"CNTRY", "N_VALID", W, X, Y, *ITEMS}
    missing = required_columns.difference(processed.columns)
    if missing:
        raise ValueError(f"Processed data are missing required KHB columns: {sorted(missing)}")
    weighted = attach_replicate_weights(processed, args.raw_file)
    weighted[MEDIATOR] = weighted[ITEMS].mean(axis=1)
    primary = weighted.loc[weighted["N_VALID"].ge(1)].copy()
    complete = weighted.loc[weighted["N_VALID"].eq(3)].copy()
    checks = sample_checks(primary, complete)
    checks.to_csv(args.output_dir / "B02" / "khb_sample_checks.csv", index=False)
    if not checks["passed"].all():
        raise RuntimeError("KHB sample counts do not match the locked taskbook definitions.")

    summary_frames: list[pd.DataFrame] = []
    replicate_frames: list[pd.DataFrame] = []
    failure_frames: list[pd.DataFrame] = []
    for analysis, sample in (("primary_available_item", primary), ("strict_complete_case", complete)):
        write_log(log_file, f"starting {analysis} KHB point estimate and BRR analysis")
        summary, replicates, failures = run_khb_brr(sample, analysis, log_file)
        summary_frames.append(summary)
        replicate_frames.append(replicates)
        failure_frames.append(failures)
    summaries = pd.concat(summary_frames, ignore_index=True)
    replicates = pd.concat(replicate_frames, ignore_index=True)
    failures = pd.concat(failure_frames, ignore_index=True)
    replicates.to_csv(args.output_dir / "B02" / "khb_replicate_estimates.csv", index=False)
    failures.to_csv(args.output_dir / "B02" / "khb_brr_failures.csv", index=False)
    if not failures.empty:
        write_metadata(args.output_dir, args, pd.DataFrame(), failures)
        raise RuntimeError(f"KHB BRR failed for {len(failures)} replicate(s); see khb_brr_failures.csv")

    primary_summary = summaries.loc[summaries["analysis"].eq("primary_available_item")]
    complete_summary = summaries.loc[summaries["analysis"].eq("strict_complete_case")]
    primary_summary.to_csv(args.output_dir / "B02" / "khb_primary_brr.csv", index=False)
    complete_summary.to_csv(args.output_dir / "B02" / "khb_complete_case_brr.csv", index=False)
    identities = identity_checks(summaries, replicates)
    identities.to_csv(args.output_dir / "B02" / "khb_identity_checks.csv", index=False)
    brr_audit = replicate_audit(replicates)
    brr_audit.to_csv(args.output_dir / "B02" / "khb_brr_audit.csv", index=False)
    anchors = anchor_checks(summaries)
    anchors.to_csv(args.output_dir / "B02" / "khb_anchor_checks.csv", index=False)
    if (
        not identities["passed"].all()
        or not anchors["passed"].all()
        or not brr_audit["complete_fay_brr"].all()
    ):
        raise RuntimeError("KHB identity or B01 anchor checks failed; inspect B02 validation files.")
    summaries.to_csv(args.output_dir / "B05" / "TableS5_KHB.csv", index=False)
    write_methods(args.output_dir, summaries)
    write_metadata(args.output_dir, args, summaries, failures)
    write_log(log_file, "completed KHB taskbook analysis with 100 converged Fay-BRR replicates per analysis")


if __name__ == "__main__":
    main()
