#!/usr/bin/env python3
"""
Cross-Asset Feature Calculations

Provides utilities for analyzing relationships between two assets,
including rolling correlation and cointegration tests.
"""

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
