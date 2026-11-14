"""Export non-identifying education-system aggregates for public release."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
INPUT_FILE = ROOT / "data" / "processed" / "data_recoded.csv"
OUTPUT_FILE = ROOT / "data" / "derived" / "public_system_summary.csv"
ITEMS = ["perceived_lesson", "perceived_adapt", "perceived_individual"]


def weighted_mean(group: pd.DataFrame, column: str) -> float:
    valid = group[[column, "TCHWGT"]].dropna()
    return float((valid[column] * valid["TCHWGT"]).sum() / valid["TCHWGT"].sum()) if len(valid) else float("nan")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-file", type=Path, default=INPUT_FILE)
    parser.add_argument("--output-file", type=Path, default=OUTPUT_FILE)
    args = parser.parse_args()
    data = pd.read_csv(args.input_file)
    data["valid_item_count"] = data[ITEMS].notna().sum(axis=1)
    rows = []
    for (system, count), group in data.groupby(["CNTRY", "valid_item_count"]):
        rows.append({
            "education_system": system,
            "valid_item_count": int(count),
            "n_unweighted": len(group),
            "weighted_population_sum": float(group["TCHWGT"].sum()),
            "weighted_training_mean": weighted_mean(group, "ai_training"),
            "weighted_ai_use_mean": weighted_mean(group, "ai_usage"),
        })
    args.output_file.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).sort_values(["education_system", "valid_item_count"]).to_csv(args.output_file, index=False)
    print(f"Saved public aggregate data to {args.output_file}")


if __name__ == "__main__":
    main()
