# Final Research Design

Status: Implemented and validated MSc dissertation project

## Working title

**Developing an Explainable Category-Store Demand Forecasting and Governance Dashboard for Retail Supply-Chain Planning**

## Primary research question

How can 28-day category-store demand forecasts be developed, evaluated and communicated through a transparent Power BI governance artefact for retail supply-chain planning?

## Supporting questions

1. Which of seasonal-naive, fixed Holt-Winters and recursive global LightGBM provides the strongest out-of-sample accuracy across 30 M5 category-store series?
2. How does forecast performance vary across evaluation folds, category-store series and seven-day horizon bands?
3. How can forecast outputs, historical reliability, model provenance and decision boundaries be communicated through Power BI?

## Objectives

1. Construct and validate 30 daily category-store demand series from the M5 evaluation history.
2. Compare seasonal-naive, fixed Holt-Winters and recursive global LightGBM using four ordered, non-overlapping 28-day backtests.
3. Prevent and test for forecast leakage through shifted demand features, post-origin masking, fixed hyperparameters and training-only evaluation scales.
4. Select the forecasting model using a rule specified before results and critically evaluate performance by fold, series and horizon.
5. Retrain the selected specification on all observed demand through d_1941 and produce a separate d_1942-d_1969 forward forecast.
6. Develop a Power BI governance artefact for forecast review, exception monitoring, model comparison, provenance and limitation communication.

## Intended contribution

The contribution is an applied and reproducible forecasting-governance artefact rather than a new forecasting algorithm or causal theory. It combines a leakage-controlled temporal evaluation, a pre-specified model-complexity gate, horizon-specific reliability evidence, a final full-history forecast and an explicit dashboard decision boundary.

## Primary user

A retail supply-chain planning manager who reviews category demand, compares stores, investigates forecast exceptions and monitors model reliability.

## Supported decisions

- category-store demand review over a 28-day planning horizon;
- category-total and store-level comparison;
- prioritisation of historically unreliable forecasts for human review;
- comparison of model performance and horizon behaviour;
- verification of model provenance and leakage controls.

## Unsupported decisions

- individual SKU replenishment quantities;
- purchase-order creation;
- safety-stock or inventory optimisation;
- supplier intervention based on M5 forecasts;
- causal explanation of demand changes;
- automated decisions or live organisational-performance claims;
- integration with unrelated operational datasets.
