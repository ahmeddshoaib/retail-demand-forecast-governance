"""Build the validated 30-series M5 category-store panel."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from .config import EXPECTED_DAYS, EXPECTED_SERIES


ID_COLUMNS = ["cat_id", "store_id", "state_id"]


def _day_number(day_label: str) -> int:
    return int(day_label.split("_", maxsplit=1)[1])


def build_category_store_panel(
    sales_path: str | Path,
    calendar_path: str | Path,
    *,
    chunksize: int = 2500,
) -> pd.DataFrame:
    """Aggregate product-store rows without retaining the full wide matrix in memory."""
    sales_path = Path(sales_path)
    calendar_path = Path(calendar_path)

    header = pd.read_csv(sales_path, nrows=0)
    day_columns = sorted(
        (column for column in header.columns if column.startswith("d_")),
        key=_day_number,
    )
    expected_labels = [f"d_{day}" for day in range(1, EXPECTED_DAYS + 1)]
    if day_columns != expected_labels:
        raise ValueError("M5 evaluation file must contain the complete d_1-d_1941 history")

    totals: defaultdict[tuple[str, str, str], np.ndarray] = defaultdict(
        lambda: np.zeros(EXPECTED_DAYS, dtype=np.int64)
    )
    usecols = ID_COLUMNS + day_columns
    dtype = {column: "int32" for column in day_columns}
    dtype.update({column: "category" for column in ID_COLUMNS})

    for chunk in pd.read_csv(
        sales_path,
        usecols=usecols,
        dtype=dtype,
        chunksize=chunksize,
    ):
        grouped = chunk.groupby(ID_COLUMNS, observed=True)[day_columns].sum()
        for key, values in grouped.iterrows():
            totals[tuple(str(value) for value in key)] += values.to_numpy(dtype=np.int64)

    if len(totals) != EXPECTED_SERIES:
        raise ValueError(f"Expected {EXPECTED_SERIES} category-store series; found {len(totals)}")

    calendar = pd.read_csv(calendar_path, usecols=["date", "d"])
    calendar["day_number"] = calendar["d"].map(_day_number)
    calendar = calendar.loc[calendar["day_number"].between(1, EXPECTED_DAYS)].copy()
    if calendar["day_number"].nunique() != EXPECTED_DAYS:
        raise ValueError("Calendar does not contain one date for every observed M5 day")
    date_by_day = calendar.set_index("day_number")["date"]

    frames = []
    day_numbers = np.arange(1, EXPECTED_DAYS + 1)
    for (category, store, state), values in sorted(totals.items()):
        if np.any(values < 0):
            raise ValueError(f"Negative demand found for {category}_{store}")
        frames.append(
            pd.DataFrame(
                {
                    "series_id": f"{category}_{store}",
                    "category": category,
                    "store": store,
                    "state": state,
                    "day_number": day_numbers,
                    "d": [f"d_{day}" for day in day_numbers],
                    "date": pd.to_datetime(date_by_day.loc[day_numbers].to_numpy()),
                    "actual": values,
                }
            )
        )

    panel = pd.concat(frames, ignore_index=True)
    expected_rows = EXPECTED_SERIES * EXPECTED_DAYS
    if len(panel) != expected_rows:
        raise AssertionError(f"Expected {expected_rows} rows; found {len(panel)}")
    if panel.duplicated(["series_id", "day_number"]).any():
        raise AssertionError("Duplicate series-day keys found after aggregation")
    return panel.sort_values(["series_id", "day_number"]).reset_index(drop=True)

