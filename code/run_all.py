"""One-command entry point for the revision-analysis package."""

from __future__ import annotations

import argparse
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


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
