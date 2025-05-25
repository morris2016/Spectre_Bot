#!/usr/bin/env python3
"""Cross asset analysis utilities.

Utilities for analyzing relationships between two assets, including rolling
correlation and cointegration tests.
"""

from typing import Optional, Tuple

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import coint

__all__ = ["compute_pair_correlation", "cointegration_score"]


def _align_series(
    data1: pd.DataFrame,
    data2: pd.DataFrame,
    column: str,
) -> Optional[Tuple[pd.Series, pd.Series]]:
    """Return aligned series for the specified column."""
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
    """Compute Pearson correlation for two aligned asset series."""
    series = _align_series(data1, data2, column)
    if series is None:
        return float("nan")
    s1, s2 = series
    if s1.isna().all() or s2.isna().all():
        return float("nan")
    return float(np.corrcoef(s1, s2)[0, 1])


def cointegration_score(
    data1: pd.DataFrame,
    data2: pd.DataFrame,
    column: str = "close",
) -> float:
    """Return the Engle-Granger cointegration p-value for two assets."""
    series = _align_series(data1, data2, column)
    if series is None:
        return float("nan")
    s1, s2 = series
    s1 = s1.dropna()
    s2 = s2.dropna()
    min_len = min(len(s1), len(s2))
    if min_len < 2:
        return float("nan")
    result = coint(s1.iloc[-min_len:], s2.iloc[-min_len:])
    return float(result[1])
