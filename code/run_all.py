"""One-command entry point for the revision-analysis package."""

from __future__ import annotations

import argparse
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def update_public_manifest(output_dir: Path) -> None:
    """Add supplementary artifacts after the legacy core runner writes its manifest."""
    manifest_path = output_dir / "B05" / "release_manifest.csv"
    if not manifest_path.exists():
        raise FileNotFoundError(manifest_path)
    manifest = pd.read_csv(manifest_path)
    additions = pd.DataFrame(
        [
            {"artifact": "B02/khb_primary_brr.csv", "status": "KHB-style decomposition; Fay-BRR 95% CI"},
            {"artifact": "B02/khb_complete_case_brr.csv", "status": "KHB-style decomposition; Fay-BRR 95% CI"},
            {"artifact": "B02/khb_replicate_estimates.csv", "status": "100 replicate coefficient estimates per sample"},
            {"artifact": "B05/TableS5_KHB.csv", "status": "KHB statistical decomposition; not causal mediation"},
            {"artifact": "B03/nonresponse_indicator_models_brr.csv", "status": "Q35 code-5 sensitivity; Fay-BRR 95% CI"},
            {"artifact": "B03/nonresponse_indicator_replicates.csv", "status": "100 replicate coefficient estimates per model"},
            {"artifact": "B05/TableS6_nonresponse_indicator.csv", "status": "Q35 nonresponse sensitivity; Fay-BRR 95% CI"},
        ]
    )
    existing = set(manifest["artifact"].astype(str))
    manifest = pd.concat(
        [manifest, additions.loc[~additions["artifact"].isin(existing)]], ignore_index=True
    )
    manifest.to_csv(manifest_path, index=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-file", type=Path, required=True)
    parser.add_argument("--n-imputations", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20261114)
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=None,
        help="Directory for an immutable reproduction run; defaults to results/reproductions/<UTC timestamp>",
    )
    args = parser.parse_args()
    run_dir = args.run_dir or (
        ROOT / "results" / "reproductions" / datetime.now(timezone.utc).strftime("run_%Y%m%dT%H%M%SZ")
    )
    run_dir = run_dir.resolve()
    processed_file = run_dir / "restricted_working" / "data_recoded.csv"
    output_dir = run_dir / "output"
    figure_dir = run_dir / "figures"
    aggregate_file = run_dir / "public_system_summary.csv"
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "code" / "01_data_preparation.py"),
            "--input-file",
            str(args.raw_file),
            "--output-file",
            str(processed_file),
        ],
        check=True,
    )
    update_public_manifest(output_dir)
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "code" / "run_khb_design.py"),
            "--raw-file",
            str(args.raw_file),
            "--processed-file",
            str(processed_file),
            "--output-dir",
            str(output_dir),
            "--seed",
            str(args.seed),
        ],
        check=True,
    )
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "code" / "run_nonresponse_indicator.py"),
            "--raw-file",
            str(args.raw_file),
            "--processed-file",
            str(processed_file),
            "--output-dir",
            str(output_dir),
            "--seed",
            str(args.seed),
        ],
        check=True,
    )
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "code" / "run_revision_package.py"),
            "--raw-file",
            str(args.raw_file),
            "--processed-file",
            str(processed_file),
            "--output-dir",
            str(output_dir),
            "--n-imputations",
            str(args.n_imputations),
            "--seed",
            str(args.seed),
        ],
        check=True,
    )
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "code" / "generate_design_adjusted_figure.py"),
            "--input-file",
            str(output_dir / "B01" / "main_paths_brr.csv"),
            "--output-dir",
            str(figure_dir),
        ],
        check=True,
    )
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "code" / "export_public_aggregates.py"),
            "--input-file",
            str(processed_file),
            "--output-file",
            str(aggregate_file),
        ],
        check=True,
    )
    print(f"Completed reproduction run at {run_dir}")


if __name__ == "__main__":
    main()
