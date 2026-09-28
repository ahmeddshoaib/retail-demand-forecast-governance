# Data setup

This public repository does not redistribute M5 source or processed demand records.

1. Obtain `calendar.csv` and `sales_train_evaluation.csv` from the official Kaggle **M5 Forecasting - Accuracy** competition.
2. Place both files in `data/raw/m5/`.
3. Create `data/processed/` and run Notebook 02 with `REBUILD_PANEL = True`.
4. Continue through Notebooks 03, 04 and 05 in order.

The complete sales file should contain 30,490 item-store rows and demand columns through `d_1941`. Notebook 02 aggregates it to 30 category-store series across 1,941 observed days (58,230 rows).

See [`../docs/data_provenance.md`](../docs/data_provenance.md) for the public-repository boundary.

