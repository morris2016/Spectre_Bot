#!/usr/bin/env python3
"""
Cross-Asset Feature Calculations

Provides utilities for analyzing relationships between two assets,
including rolling correlation and cointegration tests.
"""

from typing import Optional

"""Cross-asset feature utilities."""

import pandas as pd
from statsmodels.tsa.stattools import coint


"""Cross asset analysis utilities."""

from typing import Optional

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import coint


def compute_pair_correlation(
    series_a: pd.Series,
    series_b: pd.Series,
    window: int = 30,
    method: str = "pearson",
) -> pd.Series:
    """Return rolling correlation between two price series."""
    if len(series_a) != len(series_b):
        raise ValueError("Input series must have equal length")
    if window <= 1:
        raise ValueError("Window must be greater than 1")
    corr = series_a.rolling(window).corr(series_b, method=method)
    return corr


def cointegration_score(series_a: pd.Series, series_b: pd.Series) -> float:
    """Return p-value from the Engle-Granger cointegration test."""
    if len(series_a) != len(series_b):
        raise ValueError("Series length mismatch")
    _, p_value, _ = coint(series_a, series_b)
    return float(p_value)

__all__ = ["compute_pair_correlation", "cointegration_score"]


def _align_series(
    data1: pd.DataFrame,
    data2: pd.DataFrame,
    column: str
) -> Optional[tuple[pd.Series, pd.Series]]:
    if column not in data1.columns or column not in data2.columns:
        raise ValueError(f"Column '{column}' missing from input data")

    s1 = data1[column]
    s2 = data2[column]
    length = min(len(s1), len(s2))
    if length == 0:
        return None
    s1 = s1.iloc[-length:]
    s2 = s2.iloc[-length:]
    return s1, s2


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
    """Compute Pearson correlation for two assets."""
    series = _align_series(data1, data2, column)
    if series is None:
        return np.nan
    s1, s2 = series
    if s1.isna().all() or s2.isna().all():
        return np.nan
    return float(np.corrcoef(s1, s2)[0, 1])


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

    """Return Engle-Granger cointegration test p-value."""
    series = _align_series(data1, data2, column)
    if series is None:
        return np.nan
    s1, s2 = series
    s1 = s1.dropna()
    s2 = s2.dropna()
    min_len = min(len(s1), len(s2))
    if min_len < 2:
        return np.nan
    result = coint(s1.iloc[-min_len:], s2.iloc[-min_len:])
    return float(result[1])
