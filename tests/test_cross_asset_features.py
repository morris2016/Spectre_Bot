import numpy as np
import pandas as pd
from feature_service.features.cross_asset import compute_pair_correlation, cointegration_score


def build_series(n=100):
    index = pd.date_range("2020-01-01", periods=n, freq="D")
    s1 = pd.DataFrame({"close": np.arange(n)}, index=index)
    s2 = pd.DataFrame({"close": np.arange(n) + np.random.normal(0, 0.1, n)}, index=index)
    return s1, s2


def test_pair_correlation_high():
    s1, s2 = build_series()
    corr = compute_pair_correlation(s1, s2)
    assert corr > 0.99


def test_cointegration_pvalue_low():
    s1, s2 = build_series()
    pvalue = cointegration_score(s1, s2)
    assert pvalue < 0.05
