"""Prepare the TALIS 2024 teacher file used by the supplementary analyses.

Expected input: TALIS2024_teachers_NoESE_CSV/ttgintt4.csv
Output: data/processed/data_recoded.csv
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_FILE = ROOT / "data" / "processed" / "data_recoded.csv"

CORE_VARIABLES = [
    "CNTRY", "IDCNTRY", "IDSCHOOL", "IDTEACH",
    "TT4G21G", "TT4G24G", "TT4G35A", "TT4G35B", "TT4G35C",
    "TT4G35F", "TT4G35G", "TT4G36", "TCHWGT",
    "T4TAGEGR", "TT4G04", "T4TYEXPE",
]

VALID_UTILITY_CODES = [1, 2, 3, 4]
OTHER_UTILITY_SPECIAL_CODES = [8, 9]


def recode_binary(series: pd.Series) -> pd.Series:
    """Map TALIS yes/no items (1/2) to 1/0 and special codes to missing."""
    return series.replace({1: 1, 2: 0, 6: np.nan, 8: np.nan, 9: np.nan})


def recode_capability_item(series: pd.Series) -> pd.Series:
    """Keep valid 1-4 capability responses; code 5, 8, and 9 as missing."""
    return series.where(series.isin(VALID_UTILITY_CODES), np.nan)


def add_nonresponse_indicators(data: pd.DataFrame) -> pd.DataFrame:
    """Create Q35 nonresponse indicators directly from the retained raw items."""
    raw_items = ["TT4G35A", "TT4G35B", "TT4G35C"]
    result = data.copy()
    for suffix, item in zip(("A", "B", "C"), raw_items, strict=True):
        result[f"DK_{suffix}"] = result[item].eq(5).astype("int8")
    result["DK_COUNT"] = result[["DK_A", "DK_B", "DK_C"]].sum(axis=1).astype("int8")
    result["DK_ANY"] = result["DK_COUNT"].gt(0).astype("int8")
    result["N_VALID"] = result[raw_items].isin(VALID_UTILITY_CODES).sum(axis=1).astype("int8")
    result["NO_VALID"] = result["N_VALID"].eq(0).astype("int8")
    result["OTHER_MISSING_ANY"] = result[raw_items].isin(OTHER_UTILITY_SPECIAL_CODES).any(axis=1).astype("int8")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-file", type=Path, required=True, help="Authorised TALIS ttgintt4.csv")
    parser.add_argument("--output-file", type=Path, default=OUTPUT_FILE)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.input_file.exists():
        raise FileNotFoundError(f"Raw TALIS file not found: {args.input_file}")

    data = pd.read_csv(args.input_file, sep=";", usecols=CORE_VARIABLES, low_memory=False)

    # Code 8 means that the AI module was not administered. Keep administered cases.
    data = data.loc[(data["TT4G36"] != 8) & (data["TT4G35B"] != 8)].copy()

    data = add_nonresponse_indicators(data)
    data["ai_training"] = recode_binary(data["TT4G21G"])
    data["ai_usage"] = recode_binary(data["TT4G36"])
    data["perceived_lesson"] = recode_capability_item(data["TT4G35A"])
    data["perceived_adapt"] = recode_capability_item(data["TT4G35B"])
    data["perceived_individual"] = recode_capability_item(data["TT4G35C"])
    data["concern_plagiarism"] = data["TT4G35F"].replace({8: np.nan, 9: np.nan})
    data["concern_incorrect"] = data["TT4G35G"].replace({8: np.nan, 9: np.nan})

    # Retain partial mediator responses for the later missing-data sensitivity analyses.
    cleaned = data.dropna(subset=["ai_training", "ai_usage", "TCHWGT"]).copy()
    args.output_file.parent.mkdir(parents=True, exist_ok=True)
    cleaned.to_csv(args.output_file, index=False)
    print(f"Saved {len(cleaned):,} records to {args.output_file}")


if __name__ == "__main__":
    main()
