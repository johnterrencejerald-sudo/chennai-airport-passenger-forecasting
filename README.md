# Chennai Airport Passenger Forecasting and Covid-19 Impact Analysis

A time-series forecasting and intervention-analysis study of monthly domestic passenger traffic at Chennai International Airport.

This research compares Holt–Winters, SARIMA, and SARIMAX models using monthly airport-traffic data from 2015 to June 2024. The study evaluates forecast accuracy on a time-ordered holdout period and assesses the structural disruption caused by the Covid-19 pandemic.

## Conference presentation and proceedings

This work was co-authored and presented at **NCDDI (2026)** and included in conference proceedings with ISBN **978-93-47331-92-3**.

> *Time Series Forecasting of Domestic Air Passenger Traffic at Chennai Airport with Covid-19 Impact Assessment.*

## Project objectives

- Forecast monthly domestic passenger traffic at Chennai International Airport.
- Compare seasonal time-series forecasting methods.
- Assess whether related passenger-flow information improves forecast accuracy.
- Quantify the Covid-19 disruption using structural-break, intervention, and counterfactual analyses.

## Data

- **Source:** [OpenCity urban-data portal](https://data.opencity.in/dataset/chennai-aviation-traffic-data)
- **Frequency:** Monthly
- **Coverage:** 2015 to June 2024
- **Target variable:** `Pax To Origin`
- **Exogenous predictor:** `Pax From Origin`
- **Excluded variables:** Freight and mail measures, owing to substantial missingness

The original aviation-traffic data were processed into monthly passenger time series. Year and month fields were combined into a monthly date index, and passenger totals were aggregated by month. See the [data dictionary](docs/data_dictionary.md) and [methodology note](docs/methodology.md) for details.

## Methods

The research used a chronological modelling workflow:

1. Data-quality checks for missing values and duplicate records.
2. Exploratory analysis of distributions, correlations, trend, seasonality, decomposition, ACF, and PACF.
3. Time-ordered training, validation, and test split.
4. Forecasting-model development and comparison using Holt–Winters, SARIMA, and SARIMAX.
5. Test-set evaluation using RMSE and MAPE.
6. Covid-19 impact assessment using a Chow structural-break test, SARIMAX intervention analysis, and a counterfactual no-Covid scenario.

## Forecasting performance

| Model | Test RMSE | Test MAPE |
|---|---:|---:|
| Holt–Winters | 59,389.77 passengers | 7.32% |
| SARIMA | 40,412.84 passengers | 4.69% |
| SARIMAX | 19,068.58 passengers | 2.45% |

**Best reported model:** SARIMAX. It incorporated `Pax From Origin` as an exogenous predictor and achieved the lowest reported test error.

## Covid-19 impact findings

- A Chow test at April 2020 indicated a statistically significant structural break in domestic passenger traffic.
- The intervention SARIMAX analysis estimated a Covid-period reduction of approximately 4–5 lakh passengers per month during April 2020–March 2021.
- The counterfactual analysis found actual passenger traffic to be about 6% lower on average than the estimated no-Covid baseline during the Covid period.

## Repository status

This repository currently documents the conference research project. Reproducible analysis code, data-access instructions, model specifications, and generated figures will be added in a forthcoming update.

```text
data/       Data-access instructions and processed analysis data
notebooks/  Reproducible exploratory analysis and forecasting workflow
src/        Python modules for preprocessing, modelling, and evaluation
outputs/    Forecast figures and model-comparison tables
docs/       Publication, data, and methodology documentation
```

## Limitations

- The analysis uses monthly aggregated data for one airport and does not capture daily or weekly operational patterns.
- Forecast quality depends on data completeness and the stability of seasonal relationships over time.
- The SARIMAX model uses `Pax From Origin` as an exogenous predictor; a fully prospective forecast would require future values or separate forecasts of that predictor.
- Formal residual diagnostics, including the Ljung–Box test, are planned for the reproducible-code version.

## Tools

Python, pandas, NumPy, matplotlib, statsmodels, scikit-learn

## Author

John Terrence Jerald  
MSc Statistics, Loyola College  
Interested in biostatistics, health analytics, time-series analysis, and applied statistical modelling.
