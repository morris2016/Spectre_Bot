#!/usr/bin/env python3
"""Cross asset analysis utilities.

Utilities for analyzing relationships between two assets, including rolling
correlation and cointegration tests.
"""

from typing import Optional, Tuple, Union

import logging

import numpy as np
import pandas as pd
try:
    from statsmodels.tsa.stattools import coint  # type: ignore
    STATSMODELS_AVAILABLE = True
except Exception:  # pragma: no cover - optional dependency
    coint = None  # type: ignore
    STATSMODELS_AVAILABLE = False

__all__ = ["compute_pair_correlation", "cointegration_score"]


def _align_series(
    data1: Union[pd.DataFrame, pd.Series],
    data2: Union[pd.DataFrame, pd.Series],
    column: str,
) -> Optional[Tuple[pd.Series, pd.Series]]:
    """Return aligned series for the specified column or raw series."""
    if isinstance(data1, pd.Series) and isinstance(data2, pd.Series):
        length = min(len(data1), len(data2))
        if length == 0:
            return None
        return data1.iloc[-length:], data2.iloc[-length:]

    if not isinstance(data1, pd.DataFrame) or not isinstance(data2, pd.DataFrame):
        raise TypeError("data1 and data2 must both be Series or DataFrame")

    if column not in data1.columns or column not in data2.columns:
        raise ValueError(f"Column '{column}' missing from input data")

    s1 = data1[column]
    s2 = data2[column]
    length = min(len(s1), len(s2))
    if length == 0:
        return None
    return s1.iloc[-length:], s2.iloc[-length:]


def compute_pair_correlation(
    data1: Union[pd.DataFrame, pd.Series],
    data2: Union[pd.DataFrame, pd.Series],
    column: str = "close",
    window: int | None = None,
) -> Union[float, pd.Series]:
    """Compute Pearson correlation for two aligned asset series."""
    series = _align_series(data1, data2, column)
    if series is None:
        return float("nan")
    s1, s2 = series
    if window:
        return s1.rolling(window).corr(s2)
    if s1.isna().all() or s2.isna().all():
        return float("nan")
    return float(np.corrcoef(s1, s2)[0, 1])


def cointegration_score(
    data1: Union[pd.DataFrame, pd.Series],
    data2: Union[pd.DataFrame, pd.Series],
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
    if STATSMODELS_AVAILABLE and coint is not None:
        result = coint(s1.iloc[-min_len:], s2.iloc[-min_len:])
        return float(result[1])
    logging.getLogger(__name__).warning(
        "statsmodels not available; cointegration score set to NaN"
    )
    return float("nan")
