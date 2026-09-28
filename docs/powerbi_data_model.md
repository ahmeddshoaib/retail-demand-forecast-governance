# Power BI Data Model

The public dashboard model contains only the forecasting case.

## Forecasting tables

- `dim_date.csv`: one row per M5 calendar date from d_1 through d_1969.
- `dim_series.csv`: one row per category-store series.
- `dim_model.csv`: model names, families and complexity order.
- `fact_actual_demand.csv`: historical category-store actual demand.
- `fact_forecast_results.csv`: historical out-of-sample backtests, actuals and errors, labelled `Backtest`.
- `fact_forward_forecast.csv`: selected-model category-store forecasts for d_1942-d_1969, labelled `Forward forecast`.
- `fact_category_forward_forecast.csv`: executive category totals for the same 28-day forward period.
- `fact_forecast_monitoring.csv`: historical reliability and monitoring-priority summaries for the 30 series.
- `fact_forecast_metrics.csv`: pre-calculated overall, fold, series-fold and horizon metrics.
- `fact_model_selection.csv`: the pre-specified complexity-gate result.
- `fact_lightgbm_feature_importance.csv`: fold-level LightGBM gain and split importance.
- `fact_leakage_validation.csv`: post-origin masking test results.

Historical residual bands in the forward tables are descriptive monitoring ranges derived from the four backtests. They are not calibrated prediction intervals and must not be labelled as guaranteed confidence intervals.
