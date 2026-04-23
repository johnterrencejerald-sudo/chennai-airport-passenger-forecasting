# chennai-airport-passenger-forecasting
Time series forecasting of Chennai airport passenger traffic (2015-2024) using SARIMA, Holt-Winters, and Prophet models. Built with Python.
## Overview
This project analyzes and forecasts passenger traffic at Chennai International Airport 
using historical data from 2015 to 2024. The goal is to identify trends, seasonality, 
and generate accurate future forecasts using multiple time series models.

## Models Used
- **SARIMA** (Seasonal AutoRegressive Integrated Moving Average)
- **Holt-Winters Exponential Smoothing**
- **Prophet** (Facebook/Meta's forecasting library)
- **SARIMAX** (SARIMA with exogenous variables)

## Tools & Technologies
- Python (pandas, numpy, matplotlib, seaborn)
- statsmodels, scikit-learn, Prophet
- Jupyter Notebook

## Key Findings
- Identified clear seasonal patterns in passenger traffic
- COVID-19 period (2020–2021) treated as intervention/structural break
- SARIMA model achieved best performance based on MAPE and RMSE metrics
- Forecasts generated for 2024–2026 horizon

## Project Structure
├── data/ # Raw and processed passenger data
├── notebooks/ # Jupyter notebooks with analysis
├── reports/ # Final project report and visualizations
└── README.md
text

## Author
John Jerald — M.Sc Statistics, Loyola College Chennai  
[LinkedIn](https://www.linkedin.com/in/john-jerald-31a25a314)

