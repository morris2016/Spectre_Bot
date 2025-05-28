#!/usr/bin/env python3
"""Cross asset analysis utilities.

Utilities for analyzing relationships between two assets, including rolling
correlation and cointegration tests.
"""

from typing import Optional, Tuple

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
    data1: "pd.DataFrame | pd.Series",
    data2: "pd.DataFrame | pd.Series",
    column: str = "close",
    window: int | None = None,
) -> "float | pd.Series":
    """Compute correlation between two assets.

    If *window* is provided and inputs are :class:`~pandas.Series`, a rolling
    correlation series is returned. Otherwise a single Pearson correlation value
    is computed.
    """
    if isinstance(data1, pd.Series) and isinstance(data2, pd.Series):
        if window is not None:
            return data1.rolling(window).corr(data2)
        return float(np.corrcoef(data1, data2)[0, 1])

    if isinstance(data1, pd.DataFrame) and isinstance(data2, pd.DataFrame):
        series = _align_series(data1, data2, column)
        if series is None:
            return float("nan")
        s1, s2 = series
        if s1.isna().all() or s2.isna().all():
            return float("nan")
        return float(np.corrcoef(s1, s2)[0, 1])

    raise TypeError("compute_pair_correlation expects Series or DataFrame inputs")


def cointegration_score(
    data1: "pd.DataFrame | pd.Series",
    data2: "pd.DataFrame | pd.Series",
    column: str = "close",
) -> float:
    """Return the Engle-Granger cointegration p-value for two assets."""
    if isinstance(data1, pd.Series) and isinstance(data2, pd.Series):
        s1, s2 = data1, data2
    else:
        series = _align_series(data1, data2, column)
        if series is None:
            return float("nan")
        s1, s2 = series
    s1 = pd.Series(s1).dropna()
    s2 = pd.Series(s2).dropna()
    min_len = min(len(s1), len(s2))
    if min_len < 2:
        return float("nan")
    if STATSMODELS_AVAILABLE and coint is not None:
        result = coint(s1.iloc[-min_len:], s2.iloc[-min_len:])
        return float(result[1])

    logging.getLogger(__name__).warning(
        "statsmodels not available; falling back to correlation heuristic"
    )
    corr = np.corrcoef(s1.iloc[-min_len:], s2.iloc[-min_len:])[0, 1]
    return float(max(0.0, min(1.0, 1 - abs(corr))))
