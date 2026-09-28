"""Check the public evidence package without requiring restricted source data."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
METRICS = ROOT / "outputs" / "metrics"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def rows(name: str) -> list[dict[str, str]]:
    with (METRICS / name).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def verify_structure() -> None:
    required = [
        "README.md",
        "MODEL_CARD.md",
        "src/lightgbm_model.py",
        "tests/test_forecast_governance.py",
        "dashboard/mockups/page_3.png",
    ]
    for relative in required:
        require((ROOT / relative).is_file(), f"Missing {relative}")

    forbidden = [".DS_Store", "supply_chain_data.csv"]
    for path in ROOT.rglob("*"):
        require(path.name not in forbidden, f"Restricted or unwanted file: {path}")


def verify_notebooks() -> None:
    notebooks = sorted((ROOT / "notebooks").glob("*.ipynb"))
    require(len(notebooks) == 4, "Expected four public notebooks")
    for path in notebooks:
        notebook = json.loads(path.read_text(encoding="utf-8"))
        errors = [
            output
            for cell in notebook.get("cells", [])
            for output in cell.get("outputs", [])
            if output.get("output_type") == "error"
        ]
        require(not errors, f"Saved error output in {path.name}")
    public_notebook = notebooks[-1].read_text(encoding="utf-8").lower()
    require("supply_chain_data.csv" not in public_notebook, "Private dataset reference remains")
    require("supp_supplier_quality" not in public_notebook, "Private derived table remains")


def verify_claims() -> None:
    overall = {
        row["model"]: row
        for row in rows("forecast_metrics_all_models.csv")
        if row["scope"] == "overall"
    }
    expected = {
        "global_lightgbm_recursive": (0.6527795578, 0.0845796996),
        "holt_winters_add_damped_7": (0.7299945925, 0.1078370270),
        "seasonal_naive_7": (0.8865218921, 0.1282980832),
    }
    for model, (rmsse, wape) in expected.items():
        require(model in overall, f"Missing results for {model}")
        require(math.isclose(float(overall[model]["rmsse"]), rmsse, abs_tol=1e-9), f"RMSSE changed for {model}")
        require(math.isclose(float(overall[model]["wape"]), wape, abs_tol=1e-9), f"WAPE changed for {model}")

    selection = rows("model_selection.csv")[0]
    require(selection["selected_model"] == "global_lightgbm_recursive", "Selected model changed")
    require(selection["lightgbm_series_wins_vs_seasonal_naive"] == "29", "Seasonal-naive win count changed")
    require(selection["lightgbm_series_wins_vs_holt_winters"] == "19", "Holt-Winters win count changed")

    leakage = rows("lightgbm_leakage_test.csv")
    require(len(leakage) == 4, "Expected four leakage checks")
    require(all(row["passed"].lower() == "true" for row in leakage), "Leakage check failed")
    require(all(float(row["maximum_absolute_difference"]) == 0 for row in leakage), "Masked predictions changed")
    require(all(int(row["n_predictions"]) == 840 for row in leakage), "Unexpected comparison count")


def main() -> None:
    verify_structure()
    verify_notebooks()
    verify_claims()
    print("PUBLIC REPOSITORY VERIFICATION PASSED")


if __name__ == "__main__":
    main()
