"""Assess the Covid-19 disruption to Chennai Airport passenger traffic."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
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
COVID_START = pd.Timestamp("2020-04-01")
COVID_END = pd.Timestamp("2021-03-01")
PRE_COVID_END = pd.Timestamp("2020-02-01")


def load_model_data(path: Path = PROCESSED_DATA_PATH) -> tuple[pd.Series, pd.Series]:
    """Load aligned monthly target and passenger-flow predictor series."""
    monthly = pd.read_csv(path, parse_dates=["YearMonth"], index_col="YearMonth")
    monthly = monthly.asfreq("MS")

    target = monthly[TARGET_COLUMN]
    exogenous = monthly[EXOGENOUS_COLUMN]

    if target.isna().any() or exogenous.isna().any():
        raise ValueError("Target or exogenous series contains missing monthly values.")

    return target, exogenous


def build_seasonal_design(target: pd.Series) -> tuple[pd.Series, pd.DataFrame]:
    """Create a trend-and-month-dummy design matrix for the Chow test."""
    design = pd.DataFrame(index=target.index)
    design["time"] = np.arange(len(target), dtype=float)
    month_dummies = pd.get_dummies(target.index.month, prefix="month", drop_first=True)
    month_dummies.index = target.index

    predictors = pd.concat([design, month_dummies], axis=1).astype(float)
    predictors = sm.add_constant(predictors, has_constant="add")
    return target.astype(float), predictors


def chow_test(
    target: pd.Series,
    break_date: pd.Timestamp = COVID_START,
) -> dict[str, float]:
    """Run a Chow test for a break in trend and monthly seasonal effects."""
    response, predictors = build_seasonal_design(target)
    pre_break = predictors.index < break_date
    post_break = predictors.index >= break_date

    x_full, y_full = predictors.to_numpy(), response.to_numpy()
    x_pre, y_pre = predictors.loc[pre_break].to_numpy(), response.loc[pre_break].to_numpy()
    x_post, y_post = predictors.loc[post_break].to_numpy(), response.loc[post_break].to_numpy()

    number_of_parameters = x_full.shape[1]
    degrees_of_freedom = len(y_full) - 2 * number_of_parameters
    if degrees_of_freedom <= 0:
        raise ValueError("Not enough observations for the specified Chow-test design.")

    full_model = sm.OLS(y_full, x_full).fit()
    pre_model = sm.OLS(y_pre, x_pre).fit()
    post_model = sm.OLS(y_post, x_post).fit()

    rss_full = float(np.sum(full_model.resid**2))
    rss_split = float(np.sum(pre_model.resid**2) + np.sum(post_model.resid**2))
    f_statistic = ((rss_full - rss_split) / number_of_parameters) / (
        rss_split / degrees_of_freedom
    )
    p_value = float(stats.f.sf(f_statistic, number_of_parameters, degrees_of_freedom))

    return {
        "Break date": break_date.strftime("%Y-%m-%d"),
        "F statistic": f_statistic,
        "Numerator df": number_of_parameters,
        "Denominator df": degrees_of_freedom,
        "p value": p_value,
    }


def covid_dummy(index: pd.DatetimeIndex) -> pd.Series:
    """Create a dummy equal to one during the core Covid-19 period."""
    dummy = pd.Series(0.0, index=index, name="covid_dummy")
    dummy.loc[COVID_START:COVID_END] = 1.0
    return dummy


def fit_intervention_model(target: pd.Series, intervention: pd.Series):
    """Fit the SARIMAX intervention model with the Covid-period dummy."""
    model = SARIMAX(
        target,
        exog=intervention,
        order=ORDER,
        seasonal_order=SEASONAL_ORDER,
        enforce_stationarity=False,
        enforce_invertibility=False,
    )
    return model.fit(disp=False)


def intervention_analysis(target: pd.Series) -> tuple[pd.DataFrame, dict[str, float]]:
    """Estimate the Covid dummy effect and evaluate it on the held-out test period."""
    intervention = covid_dummy(target.index)
    train, validation, test = split_series(target)

    train_validation = pd.concat([train, validation])
    intervention_train_validation = intervention.loc[train_validation.index]
    intervention_test = intervention.loc[test.index]

    fitted_model = fit_intervention_model(
        train_validation,
        intervention_train_validation,
    )
    test_forecast = fitted_model.forecast(
        steps=len(test),
        exog=intervention_test,
    )
    metrics = evaluate_forecast(test, test_forecast)

    parameter_name = "covid_dummy"
    effect = pd.DataFrame(
        {
            "Parameter": [parameter_name],
            "Coefficient": [fitted_model.params.get(parameter_name, np.nan)],
            "Standard error": [fitted_model.bse.get(parameter_name, np.nan)],
            "p value": [fitted_model.pvalues.get(parameter_name, np.nan)],
            "Covid start": [COVID_START.strftime("%Y-%m-%d")],
            "Covid end": [COVID_END.strftime("%Y-%m-%d")],
        }
    )

    return effect, metrics


def counterfactual_analysis(
    target: pd.Series,
    exogenous: pd.Series,
) -> tuple[pd.Series, float]:
    """Estimate a pre-Covid SARIMAX counterfactual and Covid-period loss."""
    pre_covid_target = target.loc[:PRE_COVID_END]
    pre_covid_exogenous = exogenous.loc[:PRE_COVID_END]
    post_covid_target = target.loc[COVID_START:]
    post_covid_exogenous = exogenous.loc[COVID_START:]

    fitted_model = SARIMAX(
        pre_covid_target,
        exog=pre_covid_exogenous,
        order=ORDER,
        seasonal_order=SEASONAL_ORDER,
        enforce_stationarity=False,
        enforce_invertibility=False,
    ).fit(disp=False)

    counterfactual = fitted_model.forecast(
        steps=len(post_covid_target),
        exog=post_covid_exogenous,
    )
    counterfactual.index = post_covid_target.index
    counterfactual.name = "No-Covid Counterfactual"

    actual_covid = target.loc[COVID_START:COVID_END]
    counterfactual_covid = counterfactual.loc[COVID_START:COVID_END]
    percentage_loss = (counterfactual_covid - actual_covid) / counterfactual_covid * 100
    average_percentage_loss = float(percentage_loss.mean())

    return counterfactual, average_percentage_loss


def save_outputs(
    target: pd.Series,
    counterfactual: pd.Series,
    chow_results: dict[str, float],
    intervention_effect: pd.DataFrame,
    intervention_metrics: dict[str, float],
    average_percentage_loss: float,
) -> None:
    """Save Covid-analysis tables and an actual-versus-counterfactual figure."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)

    pd.DataFrame([chow_results]).to_csv(TABLES_DIR / "covid_chow_test.csv", index=False)
    intervention_effect.assign(**intervention_metrics).to_csv(
        TABLES_DIR / "covid_intervention_results.csv",
        index=False,
    )
    pd.DataFrame(
        [{"Covid start": COVID_START, "Covid end": COVID_END, "Average percentage loss": average_percentage_loss}]
    ).to_csv(TABLES_DIR / "covid_counterfactual_loss.csv", index=False)

    plt.figure(figsize=(12, 6))
    plt.plot(target.index, target.values, color="black", label="Actual Pax To Origin")
    plt.plot(
        counterfactual.index,
        counterfactual.values,
        color="red",
        linestyle="--",
        label="No-Covid counterfactual",
    )
    plt.axvspan(COVID_START, COVID_END, color="grey", alpha=0.2, label="Covid period")
    plt.title("Chennai Airport Passenger Traffic: Actual vs No-Covid Counterfactual")
    plt.xlabel("Month")
    plt.ylabel("Passengers (Pax To Origin)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        FIGURES_DIR / "covid_counterfactual.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()


def main() -> None:
    """Run structural-break, intervention, and counterfactual Covid analyses."""
    target, exogenous = load_model_data()

    chow_results = chow_test(target)
    intervention_effect, intervention_metrics = intervention_analysis(target)
    counterfactual, average_percentage_loss = counterfactual_analysis(target, exogenous)

    save_outputs(
        target,
        counterfactual,
        chow_results,
        intervention_effect,
        intervention_metrics,
        average_percentage_loss,
    )

    print("Chow-test results:")
    print(chow_results)
    print("Covid intervention effect:")
    print(intervention_effect.to_string(index=False))
    print("Intervention-model test metrics:")
    print(intervention_metrics)
    print(f"Average Covid-period loss: {average_percentage_loss:.2f}%")


if __name__ == "__main__":
    main()
