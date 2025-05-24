#!/usr/bin/env python3
"""Cross-asset feature utilities."""

import pandas as pd
from statsmodels.tsa.stattools import coint


def compute_pair_correlation(
    data1: pd.DataFrame,
    data2: pd.DataFrame,
    column: str = "close",
) -> float:
    """Compute Pearson correlation of two aligned series."""
    series1 = data1[column]
    series2 = data2[column]
    aligned = pd.concat([series1, series2], axis=1).dropna()
    if aligned.empty:
        return float("nan")
    return aligned.iloc[:, 0].corr(aligned.iloc[:, 1])


def cointegration_score(
    data1: pd.DataFrame,
    data2: pd.DataFrame,
    column: str = "close",
) -> float:
    """Return the p-value from the Engle-Granger cointegration test."""
    series1 = data1[column]
    series2 = data2[column]
    aligned = pd.concat([series1, series2], axis=1).dropna()
    if aligned.shape[0] < 3:
        return float("nan")
    _, pvalue, _ = coint(aligned.iloc[:, 0], aligned.iloc[:, 1])
    return float(pvalue)


__all__ = ["compute_pair_correlation", "cointegration_score"]
