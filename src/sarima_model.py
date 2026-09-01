"""Select and evaluate a SARIMA model for Chennai Airport passenger traffic."""

from itertools import product
from pathlib import Path
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX

try:
    from src.metrics import evaluate_forecast
except ModuleNotFoundError:
    from metrics import evaluate_forecast


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DATA_PATH = (
    PROJECT_ROOT / "data" / "processed" / "chennai_monthly_passengers.csv"
)
FIGURES_DIR = PROJECT_ROOT / "outputs" / "figures"
TABLES_DIR = PROJECT_ROOT / "outputs" / "tables"

TARGET_COLUMN = "Pax To Origin"
SEASONAL_PERIOD = 12


def load_target_series(path: Path = PROCESSED_DATA_PATH) -> pd.Series:
    """Load the monthly passenger-arrivals series."""
    monthly = pd.read_csv(path, parse_dates=["YearMonth"], index_col="YearMonth")
    target = monthly[TARGET_COLUMN].asfreq("MS")

    if target.isna().any():
        raise ValueError("The target series contains missing monthly values.")

    return target


def split_series(
    series: pd.Series,
    train_fraction: float = 0.60,
    validation_fraction: float = 0.20,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Split a series chronologically into train, validation, and test sets."""
    if not 0 < train_fraction < 1:
        raise ValueError("train_fraction must be between 0 and 1.")
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be between 0 and 1.")
    if train_fraction + validation_fraction >= 1:
        raise ValueError("Train and validation fractions must sum to less than 1.")

    n_observations = len(series)
    train_end = int(n_observations * train_fraction)
    validation_end = int(n_observations * (train_fraction + validation_fraction))

    train = series.iloc[:train_end]
    validation = series.iloc[train_end:validation_end]
    test = series.iloc[validation_end:]

    if min(len(train), len(validation), len(test)) == 0:
        raise ValueError("One or more chronological data splits are empty.")

    return train, validation, test


def grid_search_sarima(
    train: pd.Series,
    validation: pd.Series,
) -> pd.DataFrame:
    """Evaluate candidate SARIMA specifications on the validation period."""
    parameter_grid = product(
        [0, 1, 2],
        [1],
        [0, 1, 2],
        [0, 1],
        [0, 1],
        [0, 1],
    )

    results: list[dict[str, object]] = []
    warnings.filterwarnings("ignore")

    for p, d, q, seasonal_p, seasonal_d, seasonal_q in parameter_grid:
        order = (p, d, q)
        seasonal_order = (seasonal_p, seasonal_d, seasonal_q, SEASONAL_PERIOD)

        try:
            fitted_model = SARIMAX(
                train,
                order=order,
                seasonal_order=seasonal_order,
                enforce_stationarity=False,
                enforce_invertibility=False,
            ).fit(disp=False)

            validation_forecast = fitted_model.forecast(steps=len(validation))
            scores = evaluate_forecast(validation, validation_forecast)
            results.append(
                {
                    "order": str(order),
                    "seasonal_order": str(seasonal_order),
                    **scores,
                    "AIC": fitted_model.aic,
                }
            )
        except (ValueError, np.linalg.LinAlgError):
            continue

    if not results:
        raise RuntimeError("No SARIMA candidates fitted successfully.")

    return pd.DataFrame(results).sort_values(["MAPE", "RMSE"]).reset_index(drop=True)


def parse_order(order_text: str) -> tuple[int, int, int]:
    """Convert a stored tuple string into a SARIMA order tuple."""
    values = order_text.strip("()").split(",")
    return tuple(int(value.strip()) for value in values)


def parse_seasonal_order(order_text: str) -> tuple[int, int, int, int]:
    """Convert a stored tuple string into a seasonal-order tuple."""
    values = order_text.strip("()").split(",")
    return tuple(int(value.strip()) for value in values)


def rolling_one_step_forecast(
    train_validation: pd.Series,
    test: pd.Series,
    order: tuple[int, int, int],
    seasonal_order: tuple[int, int, int, int],
) -> pd.Series:
    """Create rolling one-step-ahead SARIMA predictions over the test period."""
    history = train_validation.copy()
    predictions: list[float] = []

    warnings.filterwarnings("ignore")
    for timestamp, observed_value in test.items():
        fitted_model = SARIMAX(
            history,
            order=order,
            seasonal_order=seasonal_order,
            enforce_stationarity=False,
            enforce_invertibility=False,
        ).fit(disp=False)

        prediction = fitted_model.forecast(steps=1).iloc[0]
        predictions.append(prediction)
        history.loc[timestamp] = observed_value

    return pd.Series(predictions, index=test.index, name="SARIMA Forecast")


def save_outputs(
    series: pd.Series,
    train: pd.Series,
    validation: pd.Series,
    test: pd.Series,
    forecast: pd.Series,
    results: pd.DataFrame,
    test_scores: dict[str, float],
) -> None:
    """Save model-selection results, test metrics, and a forecast plot."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)

    results.to_csv(TABLES_DIR / "sarima_validation_results.csv", index=False)
    pd.DataFrame([{"Model": "SARIMA", **test_scores}]).to_csv(
        TABLES_DIR / "sarima_test_metrics.csv",
        index=False,
    )

    plt.figure(figsize=(12, 6))
    plt.plot(series.index, series.values, color="black", label="Actual")
    plt.plot(
        forecast.index,
        forecast.values,
        color="tab:orange",
        linestyle="--",
        marker="x",
        label="SARIMA rolling forecast",
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
    plt.title("Chennai Airport Passenger Traffic: SARIMA Rolling Forecast")
    plt.xlabel("Month")
    plt.ylabel("Passengers (Pax To Origin)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        FIGURES_DIR / "sarima_rolling_test_forecast.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()


def main() -> None:
    """Select a SARIMA model and evaluate it on held-out monthly observations."""
    series = load_target_series()
    train, validation, test = split_series(series)

    selection_results = grid_search_sarima(train, validation)
    best = selection_results.iloc[0]
    best_order = parse_order(best["order"])
    best_seasonal_order = parse_seasonal_order(best["seasonal_order"])

    train_validation = pd.concat([train, validation])
    test_forecast = rolling_one_step_forecast(
        train_validation,
        test,
        best_order,
        best_seasonal_order,
    )
    test_scores = evaluate_forecast(test, test_forecast)

    save_outputs(
        series,
        train,
        validation,
        test,
        test_forecast,
        selection_results,
        test_scores,
    )

    print("Selected SARIMA order:", best_order)
    print("Selected seasonal order:", best_seasonal_order)
    print("Validation metrics:")
    print(best[["MAE", "RMSE", "MAPE"]].to_dict())
    print("Test metrics:")
    print(test_scores)


if __name__ == "__main__":
    main()
