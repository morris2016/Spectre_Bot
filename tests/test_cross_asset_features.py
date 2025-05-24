import pandas as pd
from feature_service.features.cross_asset import compute_pair_correlation, cointegration_score


def test_pair_correlation_basic():
    s1 = pd.Series(range(30))
    s2 = pd.Series([x + 0.1 for x in range(30)])
    corr = compute_pair_correlation(s1, s2, window=5)
    assert isinstance(corr, pd.Series)
    assert corr.iloc[-1] > 0.9


def test_cointegration_score_range():
    s1 = pd.Series(range(30))
    s2 = pd.Series(range(30)) + 1
    pval = cointegration_score(s1, s2)
    assert 0 <= pval <= 1
