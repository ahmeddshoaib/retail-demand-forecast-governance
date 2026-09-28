# Explainable Category-Store Demand Forecasting and Forecast Governance

My MSc Business Analytics dissertation at Queen's University Belfast developed and tested a 28-day demand forecasting framework for category-store planning. The study uses 58,230 daily observations across 30 category-store series and compares a global recursive LightGBM model with seasonal-naive and Holt-Winters benchmarks through four ordered backtests.

I designed the research, engineered the forecasting pipeline, implemented recursive multi-step prediction, built the temporal evaluation and leakage controls, and translated the results into a five-page Power BI decision product. For GitHub, I rebuilt the submitted page layout as static images; the metrics are retained from the archived dissertation run.

![Forecast performance dashboard](dashboard/mockups/page_3.png)

## Executive result

| Outcome | Evidence |
|---|---:|
| Best model | Global recursive LightGBM |
| Overall RMSSE | **0.6528** |
| Overall WAPE | **8.46%** |
| Historical backtest forecasts | 10,080 |
| Series wins vs seasonal naive | **29 / 30** |
| Series wins vs Holt-Winters | **19 / 30** |
| Leakage masking checks | **4 / 4 passed** |

LightGBM produced the strongest overall result and won every evaluation fold. The analysis also retained the counter-evidence: Holt-Winters was better for 11 series and slightly stronger over forecast days 22–28. The recommendation was therefore to use LightGBM as the primary forecast while keeping series-level exceptions visible for review.

## Research contribution

The dissertation joins five parts of the forecasting problem that are often assessed separately:

1. a defined 28-day category-store planning decision;
2. time-ordered backtesting against credible baselines;
3. recursive feature generation that prevents future information entering predictions;
4. explicit selection, leakage and robustness controls;
5. a Power BI handoff designed for exception review rather than automated ordering.

Together, these components make the model choice traceable from raw history through evaluation, selection and management review.

## Analytical design

```text
M5 item-store demand
        |
        v
30 category-store series x 1,941 days
        |
        v
Four ordered 28-day backtests
        |
        +--> Seasonal naive (7-day)
        +--> Holt-Winters (fixed specification)
        +--> Global recursive LightGBM
                      |
                      v
      leakage tests + complexity gate + robustness checks
                      |
                      v
        28-day forward forecast + Power BI governance layer
```

### Evaluation controls

- Four non-overlapping temporal folds; no random train/test split.
- Training-only RMSSE scales and lagged/rolling features.
- Recursive multi-step forecasts, so each future step uses only information available at its origin.
- Post-origin target masking repeated across all four folds: 840 predictions compared per fold, maximum absolute difference `0.0`.
- A model-complexity gate specified before comparing final results.
- Performance broken down by fold, series and seven-day horizon band.

## Power BI decision product

The submitted Power BI implementation provides:

- an executive view of forward demand and model reliability;
- a category-store exception queue for human review;
- fold, series and horizon-level model comparison;
- model-selection, feature-reliance and leakage evidence;
- explicit boundaries: no SKU replenishment, purchase-order or causal claim.

| Executive review | Exceptions | Governance |
|---|---|---|
| ![Executive overview](dashboard/mockups/page_1.png) | ![Demand exceptions](dashboard/mockups/page_2.png) | ![Governance](dashboard/mockups/page_4.png) |

The submitted `.pbix` is intentionally not public because it embeds restricted data. The static images reproduce the implemented page structure, and the documented star schema records how the Power BI model was organised without redistributing source records.

## Repository guide

| Path | Purpose |
|---|---|
| `notebooks/02...04` | Executed preparation, evaluation and full-history forecasting workflow |
| `notebooks/05...` | Public Power BI preparation and governance workflow; private demonstrator removed |
| `src/` | Reusable data, model, evaluation, selection and governance modules |
| `scripts/` | Command-line entry points for each pipeline stage |
| `tests/` | Unit tests for metrics, paths, selection and governance logic |
| `outputs/metrics/` | Aggregate, non-row-level evidence used in this README |
| `docs/` | Research design, feature availability, provenance and dashboard schema |

## Reproduce the analysis

The raw and processed M5 records are not redistributed. Download the official M5 Forecasting - Accuracy files from Kaggle and place them under `data/raw/m5/` as described in [`data/README.md`](data/README.md).

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
python scripts/verify_public_repository.py
```

Run the notebooks in numerical order or use the command-line scripts. The repository includes the submitted aggregate metrics so the published claims can be checked without redistributing the competition data.

The public code uses the neutral seed `42`; the saved metrics are the evidence table from the archived submitted run. A complete rerun may therefore show small numerical variation while retaining the same evaluation design.

## Technical stack

Python · pandas · NumPy · statsmodels · LightGBM · scikit-learn · Jupyter · Power BI · temporal cross-validation · feature engineering · model governance

## Limitations

This is a historical forecasting study based on the M5 competition, not a live 2026 demand forecast. Residual bands are descriptive monitoring ranges, not calibrated prediction intervals. Feature importance describes model reliance, not causality.

## Author

**Muhammad Ahmed Shoaib**<br>
Business analytics, supply chain, forecasting and decision support.
