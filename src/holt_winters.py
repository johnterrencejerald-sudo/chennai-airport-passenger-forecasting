"""Fit and evaluate a Holt-Winters benchmark for Chennai Airport traffic."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing

try:
    from src.metrics import evaluate_forecast
    from src.sarima_model import split_series
except ModuleNotFoundError:
    from metrics import evaluate_forecast
    from sarima_model import split_series


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DATA_PATH = (
    PROJECT_ROOT / "data" / "processed" / "chennai_monthly_passengers.csv"
)
FIGURES_DIR = PROJECT_ROOT / "outputs" / "figures"
TABLES_DIR = PROJECT_ROOT / "outputs" / "tables"

TARGET_COLUMN = "Pax To Origin"
SEASONAL_PERIOD = 12


def load_target_series(path: Path = PROCESSED_DATA_PATH) -> pd.Series:
    """Load the processed monthly passenger-arrivals series."""
    monthly = pd.read_csv(path, parse_dates=["YearMonth"], index_col="YearMonth")
    target = monthly[TARGET_COLUMN].asfreq("MS")

    if target.isna().any():
        raise ValueError("The target series contains missing monthly values.")

    return target


def fit_holt_winters(train_validation: pd.Series):
    """Fit additive Holt-Winters level, trend, and seasonality components."""
    return ExponentialSmoothing(
        train_validation,
        trend="add",
        seasonal="add",
        seasonal_periods=SEASONAL_PERIOD,
    ).fit(optimized=True)


def save_outputs(
    series: pd.Series,
    train: pd.Series,
    validation: pd.Series,
    test: pd.Series,
    forecast: pd.Series,
    scores: dict[str, float],
) -> None:
    """Save Holt-Winters test metrics and forecast plot."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)

    pd.DataFrame([{"Model": "Holt-Winters", **scores}]).to_csv(
        TABLES_DIR / "holt_winters_test_metrics.csv",
        index=False,
    )

    plt.figure(figsize=(12, 6))
    plt.plot(series.index, series.values, color="black", label="Actual")
    plt.plot(
        forecast.index,
        forecast.values,
        color="purple",
        linestyle="--",
        marker="o",
        label="Holt-Winters forecast",
    )
    plt.axvspan(train.index[0], train.index[-1], color="green", alpha=0.08, label="Train")
    plt.axvspan(
        validation.index[0],
        validation.index[-1],
        color="tab:blue",
        alpha=0.06,
        label="Validation",
    )
    plt.axvspan(test.index[0], test.index[-1], color="red", alpha=0.06, label="Test")
    plt.title("Chennai Airport Passenger Traffic: Holt-Winters Forecast")
    plt.xlabel("Month")
    plt.ylabel("Passengers (Pax To Origin)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        FIGURES_DIR / "holt_winters_test_forecast.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()


def main() -> None:
    """Fit the Holt-Winters benchmark and evaluate the held-out test period."""
    series = load_target_series()
    train, validation, test = split_series(series)
    train_validation = pd.concat([train, validation])

    fitted_model = fit_holt_winters(train_validation)
    test_forecast = fitted_model.forecast(steps=len(test))
    test_forecast.name = "Holt-Winters Forecast"
    scores = evaluate_forecast(test, test_forecast)

    save_outputs(series, train, validation, test, test_forecast, scores)

    print("Holt-Winters test metrics:")
    print(scores)


if __name__ == "__main__":
    main()
