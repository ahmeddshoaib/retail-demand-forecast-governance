# Model card: global recursive LightGBM

## Intended use

Support a retail planning manager reviewing 28-day demand at category-store level, comparing stores and prioritising historically unreliable forecasts for human investigation.

## Not intended for

SKU ordering, safety-stock optimisation, supplier action, automated decisions, causal inference or live organisational claims.

## Training and evaluation

- Grain: 30 category-store series.
- History: 1,941 daily observations per series.
- Evaluation: four ordered, non-overlapping 28-day backtests.
- Comparators: seven-day seasonal naive and fixed damped additive Holt-Winters.
- Features: shifted lags, rolling summaries and calendar/event variables available at forecast origin.
- Selection: LightGBM must beat both baselines on overall RMSSE and seasonal naive on the majority of series.

## Results

Global recursive LightGBM achieved RMSSE `0.652780` and WAPE `0.084580`, beating seasonal naive on 29 of 30 series and Holt-Winters on 19 of 30. It passed four post-origin masking tests with no prediction change across 840 comparisons per fold.

## Known limitations

- Results are historical and specific to the M5 data-generating process.
- Holt-Winters remained stronger on 11 series and over days 22–28.
- Feature importance is descriptive and does not establish causal effects.
- Monitoring ranges are empirical residual summaries, not calibrated intervals.
- Operational value was not tested through a live deployment or user study.

