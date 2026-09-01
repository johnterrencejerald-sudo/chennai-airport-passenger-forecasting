"""Generate exploratory time-series analysis outputs for Chennai Airport data."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.seasonal import seasonal_decompose


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DATA_PATH = (
    PROJECT_ROOT / "data" / "processed" / "chennai_monthly_passengers.csv"
)
FIGURES_DIR = PROJECT_ROOT / "outputs" / "figures"


def load_monthly_data(path: Path = PROCESSED_DATA_PATH) -> pd.DataFrame:
    """Load the processed monthly passenger time series."""
    monthly = pd.read_csv(path, parse_dates=["YearMonth"], index_col="YearMonth")
    return monthly.asfreq("MS")


def save_figure(filename: str) -> None:
    """Save the current Matplotlib figure to the project outputs directory."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / filename, dpi=300, bbox_inches="tight")
    plt.close()


def plot_passenger_trends(monthly: pd.DataFrame) -> None:
    """Plot monthly passenger-flow measures over time."""
    ax = monthly[["Pax To Origin", "Pax From Origin"]].plot(
        figsize=(12, 6),
        linewidth=2,
    )
    ax.set_title("Monthly Domestic Passenger Traffic at Chennai Airport")
    ax.set_xlabel("Month")
    ax.set_ylabel("Passengers")
    ax.legend(["Pax To Origin", "Pax From Origin"])
    save_figure("passenger_traffic_trend.png")


def plot_passenger_correlation(monthly: pd.DataFrame) -> None:
    """Create a correlation heatmap for the monthly passenger measures."""
    correlation = monthly[["Pax To Origin", "Pax From Origin"]].corr()

    plt.figure(figsize=(6, 5))
    sns.heatmap(
        correlation,
        annot=True,
        cmap="coolwarm",
        vmin=-1,
        vmax=1,
        square=True,
        fmt=".3f",
    )
    plt.title("Correlation Between Monthly Passenger Measures")
    save_figure("passenger_correlation.png")


def plot_monthly_seasonality(monthly: pd.DataFrame) -> None:
    """Plot average passenger volume for each calendar month."""
    seasonal_mean = monthly["Pax To Origin"].groupby(monthly.index.month).mean()
    seasonal_mean.index = [
        "Jan", "Feb", "Mar", "Apr", "May", "Jun",
        "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
    ]

    ax = seasonal_mean.plot(kind="bar", figsize=(10, 5), color="steelblue")
    ax.set_title("Average Monthly Domestic Passenger Traffic: Pax To Origin")
    ax.set_xlabel("Calendar month")
    ax.set_ylabel("Average passengers")
    save_figure("monthly_seasonality.png")


def plot_decomposition(monthly: pd.DataFrame) -> None:
    """Create an additive seasonal decomposition of the arrivals series."""
    decomposition = seasonal_decompose(
        monthly["Pax To Origin"],
        model="additive",
        period=12,
    )
    figure = decomposition.plot()
    figure.set_size_inches(12, 9)
    figure.suptitle("Additive Seasonal Decomposition: Pax To Origin", y=1.02)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(
        FIGURES_DIR / "seasonal_decomposition.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(figure)


def plot_acf_pacf(monthly: pd.DataFrame) -> None:
    """Plot ACF and PACF for the first-differenced arrivals series."""
    differenced = monthly["Pax To Origin"].diff().dropna()

    figure, axes = plt.subplots(2, 1, figsize=(11, 8))
    plot_acf(differenced, lags=24, ax=axes[0])
    plot_pacf(differenced, lags=24, ax=axes[1], method="ywm")
    axes[0].set_title("ACF: First-Differenced Pax To Origin")
    axes[1].set_title("PACF: First-Differenced Pax To Origin")
    figure.tight_layout()

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    figure.savefig(
        FIGURES_DIR / "differenced_acf_pacf.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(figure)


def main() -> None:
    """Generate all core exploratory analysis figures."""
    monthly = load_monthly_data()

    plot_passenger_trends(monthly)
    plot_passenger_correlation(monthly)
    plot_monthly_seasonality(monthly)
    plot_decomposition(monthly)
    plot_acf_pacf(monthly)

    print(f"EDA figures saved to: {FIGURES_DIR}")


if __name__ == "__main__":
    main()
