"""Forecast-evaluation utilities for the Chennai Airport project."""

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error


def rmse(y_true, y_pred) -> float:
    """Return root mean squared error."""
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def mae(y_true, y_pred) -> float:
    """Return mean absolute error."""
    return float(mean_absolute_error(y_true, y_pred))


def mape(y_true, y_pred) -> float:
    """Return mean absolute percentage error as a percentage."""
    actual = np.asarray(y_true, dtype=float)
    predicted = np.asarray(y_pred, dtype=float)

    nonzero = actual != 0
    if not np.any(nonzero):
        raise ValueError(
            "MAPE cannot be calculated because all actual values are zero."
        )

    return float(
        np.mean(
            np.abs(
                (actual[nonzero] - predicted[nonzero]) / actual[nonzero]
            )
        )
        * 100
    )


def evaluate_forecast(y_true, y_pred) -> dict[str, float]:
    """Return common forecast-accuracy metrics."""
    return {
        "MAE": mae(y_true, y_pred),
        "RMSE": rmse(y_true, y_pred),
        "MAPE": mape(y_true, y_pred),
    }


def create_comparison_table(
    results: dict[str, dict[str, float]],
) -> pd.DataFrame:
    """Create a ranked model-comparison table from evaluation results."""
    comparison = pd.DataFrame(results).T
    comparison.index.name = "Model"

    return (
        comparison.sort_values("MAPE")
        .assign(Rank=lambda frame: range(1, len(frame) + 1))
        .reset_index()
    )
