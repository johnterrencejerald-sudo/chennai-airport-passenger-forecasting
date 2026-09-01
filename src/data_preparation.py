"""Prepare monthly Chennai Airport passenger time-series data."""

from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "chennai_aero_data.csv"
PROCESSED_DATA_PATH = (
    PROJECT_ROOT / "data" / "processed" / "chennai_monthly_passengers.csv"
)

REQUIRED_COLUMNS = {
    "Year",
    "Month",
    "Pax To Origin",
    "Pax From Origin",
}


def load_raw_data(path: Path = RAW_DATA_PATH) -> pd.DataFrame:
    """Load the raw Chennai aviation-traffic dataset."""
    data = pd.read_csv(path)

    missing_columns = REQUIRED_COLUMNS.difference(data.columns)
    if missing_columns:
        raise ValueError(
            f"Missing required columns: {sorted(missing_columns)}"
        )

    return data


def create_monthly_series(data: pd.DataFrame) -> pd.DataFrame:
    """Aggregate passenger traffic to a complete monthly time series."""
    prepared = data.copy()

    prepared["YearMonth"] = pd.to_datetime(
        prepared["Year"].astype(str)
        + "-"
        + prepared["Month"].astype(str).str.zfill(2)
        + "-01"
    )

    monthly = (
        prepared.groupby("YearMonth")[["Pax To Origin", "Pax From Origin"]]
        .sum()
        .sort_index()
        .asfreq("MS")
    )

    if monthly.isna().any().any():
        missing_months = monthly[monthly.isna().any(axis=1)].index.tolist()
        raise ValueError(
            "Missing passenger values after monthly aggregation for: "
            f"{missing_months}"
        )

    return monthly


def save_processed_data(
    monthly: pd.DataFrame,
    path: Path = PROCESSED_DATA_PATH,
) -> None:
    """Save the analysis-ready monthly passenger series."""
    path.parent.mkdir(parents=True, exist_ok=True)
    monthly.to_csv(path, index_label="YearMonth")


def main() -> None:
    """Create and save the processed monthly passenger dataset."""
    data = load_raw_data()
    monthly = create_monthly_series(data)
    save_processed_data(monthly)

    print("Processed data saved to:", PROCESSED_DATA_PATH)
    print("Date range:", monthly.index.min().date(), "to", monthly.index.max().date())
    print("Monthly observations:", len(monthly))
    print(monthly.head())


if __name__ == "__main__":
    main()
