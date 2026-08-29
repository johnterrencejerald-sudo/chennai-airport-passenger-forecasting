# Methodology

## Study aim

This study forecasts monthly domestic passenger traffic at Chennai International Airport and evaluates the Covid-19 disruption to passenger volumes. The operational motivation is to support resource planning, including staffing, queue management, and capacity planning.

## Data preparation

Monthly domestic aviation-traffic records from the OpenCity urban-data portal were used. Year and month fields were combined to form a monthly time index. Passenger values were aggregated by month to create time series for `Pax To Origin` and `Pax From Origin`.

Freight and mail variables were excluded from model development because they contained substantial missingness. Passenger and time variables had no missing values, and no duplicate rows were reported.

## Exploratory analysis

The exploratory analysis included descriptive summaries, correlation analysis, time-series plots, seasonal decomposition, residual checks, and ACF/PACF inspection. The analysis indicated a long-run upward trend before Covid-19, repeating annual seasonality, a sharp Covid-period traffic collapse, and subsequent recovery.

The strong positive association between `Pax To Origin` and `Pax From Origin` motivated the use of the latter as an external regressor in the SARIMAX model.

## Data split and evaluation

Observations were split chronologically into training, validation, and test periods. Model selection was guided by validation performance, and final forecasts were evaluated on held-out test observations.

Forecast accuracy was assessed using:

- Root mean squared error (RMSE)
- Mean absolute percentage error (MAPE)

## Forecasting models

### Holt–Winters

An additive Holt–Winters specification with level, trend, and 12-month seasonal components was fitted as a seasonal benchmark.

### SARIMA

A seasonal ARIMA model was selected using stationarity assessment, ACF/PACF inspection, and a grid search over candidate non-seasonal and seasonal orders.

### SARIMAX

A SARIMAX model extended the selected seasonal ARIMA specification by including `Pax From Origin` as an exogenous regressor. This model produced the lowest reported test-set RMSE and MAPE.

## Covid-19 impact analysis

Three complementary approaches were used:

1. A Chow test at April 2020 to test for a structural break.
2. SARIMAX intervention models with Covid-period dummy variables.
3. A counterfactual no-Covid forecast based on a model estimated using pre-Covid data.

The intervention analysis estimated a substantial and statistically significant reduction in domestic passenger traffic during the core Covid period. The counterfactual comparison indicated observed traffic was about 6% lower on average than the estimated no-Covid baseline over the Covid window.

## Reproducibility status

The original code, exact final model orders, data-processing scripts, and generated figures will be added in a subsequent update. The reproducible version will also add formal residual diagnostics, including a Ljung–Box test.
