"""One-command entry point for the revision-analysis package."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-file", type=Path, required=True)
    parser.add_argument("--n-imputations", type=int, default=20)
    args = parser.parse_args()
    processed_file = ROOT / "data" / "processed" / "data_recoded.csv"
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
            str(ROOT / "results" / "output_final"),
            "--n-imputations",
            str(args.n_imputations),
        ],
        check=True,
    )
    subprocess.run([sys.executable, str(ROOT / "code" / "generate_design_adjusted_figure.py")], check=True)
    subprocess.run([sys.executable, str(ROOT / "code" / "export_public_aggregates.py")], check=True)


if __name__ == "__main__":
    main()
