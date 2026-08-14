"""
Unit tests for the factor module
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
import pytest

from src.factors.base import winsorize_mad, zscore, preprocess_cross_section
from src.factors.fundamental import (
    EPFactor, BPFactor, SizeFactor, MomentumFactor, ROETTMFactor,
    MultiFactorCombiner
)


@pytest.fixture
def sample_dates():
    return pd.date_range("2023-01-02", periods=5, freq="B")


@pytest.fixture
def sample_stocks():
    return ["000001", "000002", "000003", "000004"]


def test_winsorize_mad():
    """MAD winsorization should suppress extreme values"""
    s = pd.Series([1, 2, 3, 4, 5, 100])
    result = winsorize_mad(s, n=3)
    assert result.max() < 100
    assert result.median() == s.median()


def test_zscore():
    """After normalization, mean should be 0 and std should be 1"""
    s = pd.Series(np.random.randn(100) * 5 + 10)
    result = zscore(s)
    assert abs(result.mean()) < 1e-10
    assert abs(result.std() - 1) < 1e-10


def test_preprocess_cross_section_nan(sample_dates):
    """Cross-section with NaN should keep NaN positions after processing"""
    s = pd.Series([1.0, 2.0, np.nan, 4.0, 5.0])
    result = preprocess_cross_section(s)
    assert result.isna().sum() == 1


def test_ep_factor():
    """EP = 1/PE; negative PE should be filtered out"""
    pe = pd.DataFrame({"A": [10.0, 20.0], "B": [-5.0, 40.0]})
    ep = EPFactor().compute(pe)
    assert ep.loc[0, "A"] == pytest.approx(0.1)
    assert np.isnan(ep.loc[0, "B"])
    assert ep.loc[1, "B"] == pytest.approx(0.025)


def test_bp_factor():
    """BP = 1/PB"""
    pb = pd.DataFrame({"A": [2.0, 4.0]})
    bp = BPFactor().compute(pb)
    assert bp.loc[0, "A"] == pytest.approx(0.5)
    assert bp.loc[1, "A"] == pytest.approx(0.25)


def test_size_factor():
    """Smaller market cap should score higher"""
    mv = pd.DataFrame({"A": [1e10], "B": [1e9]})
    size = SizeFactor().compute(mv)
    assert size.loc[0, "B"] > size.loc[0, "A"]


def test_momentum_factor(sample_stocks):
    """Momentum factor: should be NaN when the window is insufficient"""
    dates = pd.date_range("2023-01-02", periods=50, freq="B")
    ret = pd.DataFrame(np.random.randn(50, 4) * 0.01, index=dates, columns=sample_stocks)
    mom = MomentumFactor(long_window=20, skip_window=5).compute(ret)
    # first 20 rows have insufficient window
    assert mom.iloc[:20].isna().all().all()
    # later rows should have valid values
    assert mom.iloc[25:].notna().any().any()


def test_roe_ttm():
    """ROE TTM reconstruction: current YTD + prior annual - prior same-period YTD"""
    idx = pd.MultiIndex.from_tuples([
        ("000001", pd.Timestamp("2021-03-31")),
        ("000001", pd.Timestamp("2021-06-30")),
        ("000001", pd.Timestamp("2021-12-31")),
        ("000001", pd.Timestamp("2022-03-31")),
        ("000001", pd.Timestamp("2022-06-30")),
    ], names=["stock_code", "report_date"])
    # 2022Q1 TTM = 5 + 20 (prior annual) - 4 (prior Q1) = 21
    # 2022Q2 TTM = 8 + 20 - 9 (prior H1) = 19
    financial = pd.DataFrame({"roe": [4.0, 9.0, 20.0, 5.0, 8.0]}, index=idx)
    
    factor = ROETTMFactor(disclosure_lag={"q1": 30, "q2": 62, "q3": 31, "q4": 121})
    trading_dates = pd.date_range("2022-04-01", "2022-09-30", freq="B")
    panel = factor.compute(financial, trading_dates)
    
    # 2022Q1 report (3/31) with 30-day disclosure lag -> available on 4/30
    # (effective on the first following trading day)
    available_date = pd.Timestamp("2022-04-30")
    first_valid = panel["000001"].first_valid_index()
    assert first_valid >= available_date
    # TTM value after disclosure should be 21
    assert panel.loc[first_valid, "000001"] == pytest.approx(21.0)
    # H1 report (6/30) with 62-day lag -> available on 8/31, TTM = 19
    assert panel.loc[pd.Timestamp("2022-08-31"), "000001"] == pytest.approx(19.0)


def test_roe_no_lookahead():
    """ROE factor should have no value before the disclosure date (no lookahead)"""
    idx = pd.MultiIndex.from_tuples([
        ("000001", pd.Timestamp("2022-12-31")),
    ], names=["stock_code", "report_date"])
    financial = pd.DataFrame({"roe": [15.0]}, index=idx)
    
    factor = ROETTMFactor(disclosure_lag={"q1": 30, "q2": 62, "q3": 31, "q4": 121})
    trading_dates = pd.date_range("2023-01-02", "2023-04-28", freq="B")
    panel = factor.compute(financial, trading_dates)
    
    # annual report has a 121-day disclosure lag; no value before 4/28
    assert panel["000001"].isna().all()


def test_multi_factor_combiner(sample_dates, sample_stocks):
    """Combined score should be the weighted sum of normalized factors"""
    np.random.seed(42)
    panels = {
        "ep": pd.DataFrame(np.random.randn(5, 4), index=sample_dates, columns=sample_stocks),
        "bp": pd.DataFrame(np.random.randn(5, 4), index=sample_dates, columns=sample_stocks),
    }
    combiner = MultiFactorCombiner(weights={"ep": 0.5, "bp": 0.5})
    scores = combiner.combine(panels, list(sample_dates))
    
    assert scores.shape == (5, 4)
    # equal-weight combination of two normalized factors: cross-section mean ~ 0
    assert abs(scores.iloc[0].mean()) < 1e-10
