"""Fit and evaluate a SARIMAX model for Chennai Airport passenger traffic."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.stats.stattools import jarque_bera
from statsmodels.tsa.statespace.sarimax import SARIMAX

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
EXOGENOUS_COLUMN = "Pax From Origin"
ORDER = (2, 1, 2)
SEASONAL_ORDER = (0, 0, 1, 12)


def load_model_data(path: Path = PROCESSED_DATA_PATH) -> tuple[pd.Series, pd.Series]:
    """Load aligned target and external-regressor monthly series."""
    monthly = pd.read_csv(path, parse_dates=["YearMonth"], index_col="YearMonth")
    monthly = monthly.asfreq("MS")

    target = monthly[TARGET_COLUMN]
    exogenous = monthly[EXOGENOUS_COLUMN]

    if target.isna().any() or exogenous.isna().any():
        raise ValueError("Target or exogenous series contains missing monthly values.")

    return target, exogenous


def fit_sarimax(
    target: pd.Series,
    exogenous: pd.Series,
):
    """Fit the paper-aligned SARIMAX specification."""
    model = SARIMAX(
        target,
        exog=exogenous,
        order=ORDER,
        seasonal_order=SEASONAL_ORDER,
        enforce_stationarity=False,
        enforce_invertibility=False,
    )
    return model.fit(disp=False)


def calculate_diagnostics(fitted_model) -> pd.DataFrame:
    """Calculate Ljung-Box and Jarque-Bera diagnostics for model residuals."""
    residuals = fitted_model.resid.dropna()
    ljung_box = acorr_ljungbox(residuals, lags=[12, 24], return_df=True)
    jb_statistic, jb_p_value, skewness, kurtosis = jarque_bera(residuals)

    diagnostics = pd.DataFrame(
        {
            "Diagnostic": [
                "Ljung-Box Q(12)",
                "Ljung-Box Q(24)",
                "Jarque-Bera",
            ],
            "Statistic": [
                ljung_box.loc[12, "lb_stat"],
                ljung_box.loc[24, "lb_stat"],
                jb_statistic,
            ],
            "p_value": [
                ljung_box.loc[12, "lb_pvalue"],
                ljung_box.loc[24, "lb_pvalue"],
                jb_p_value,
            ],
            "Skewness": [None, None, skewness],
            "Kurtosis": [None, None, kurtosis],
        }
    )

    return diagnostics


def save_outputs(
    target: pd.Series,
    train: pd.Series,
    validation: pd.Series,
    test: pd.Series,
    forecast: pd.Series,
    test_scores: dict[str, float],
    diagnostics: pd.DataFrame,
) -> None:
    """Save evaluation tables, residual diagnostics, and a forecast figure."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)

    pd.DataFrame([{"Model": "SARIMAX", **test_scores}]).to_csv(
        TABLES_DIR / "sarimax_test_metrics.csv",
        index=False,
    )
    diagnostics.to_csv(TABLES_DIR / "sarimax_residual_diagnostics.csv", index=False)

    plt.figure(figsize=(12, 6))
    plt.plot(target.index, target.values, color="black", label="Actual")
    plt.plot(
        forecast.index,
        forecast.values,
        color="magenta",
        linestyle="--",
        marker="o",
        label="SARIMAX forecast",
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
    plt.title("Chennai Airport Passenger Traffic: SARIMAX Test Forecast")
    plt.xlabel("Month")
    plt.ylabel("Passengers (Pax To Origin)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        FIGURES_DIR / "sarimax_test_forecast.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()


def main() -> None:
    """Validate, evaluate, and diagnose the final SARIMAX model."""
    target, exogenous = load_model_data()
    train, validation, test = split_series(target)

    train_exogenous = exogenous.loc[train.index]
    validation_exogenous = exogenous.loc[validation.index]
    test_exogenous = exogenous.loc[test.index]

    validation_model = fit_sarimax(train, train_exogenous)
    validation_forecast = validation_model.forecast(
        steps=len(validation),
        exog=validation_exogenous,
    )
    validation_scores = evaluate_forecast(validation, validation_forecast)

    train_validation = pd.concat([train, validation])
    train_validation_exogenous = pd.concat([
        train_exogenous,
        validation_exogenous,
    ])
    final_model = fit_sarimax(train_validation, train_validation_exogenous)
    test_forecast = final_model.forecast(steps=len(test), exog=test_exogenous)
    test_forecast.name = "SARIMAX Forecast"
    test_scores = evaluate_forecast(test, test_forecast)
    diagnostics = calculate_diagnostics(final_model)

    save_outputs(
        target,
        train,
        validation,
        test,
        test_forecast,
        test_scores,
        diagnostics,
    )

    print("SARIMAX order:", ORDER)
    print("SARIMAX seasonal order:", SEASONAL_ORDER)
    print("Validation metrics:")
    print(validation_scores)
    print("Test metrics:")
    print(test_scores)
    print("Residual diagnostics:")
    print(diagnostics.to_string(index=False))


if __name__ == "__main__":
    main()
