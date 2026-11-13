"""Export non-identifying education-system aggregates for public release."""

from __future__ import annotations

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
    data = pd.read_csv(INPUT_FILE)
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
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).sort_values(["education_system", "valid_item_count"]).to_csv(OUTPUT_FILE, index=False)
    print(f"Saved public aggregate data to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
