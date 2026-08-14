"""
因子模块单元测试
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
    """MAD去极值应压制极端值"""
    s = pd.Series([1, 2, 3, 4, 5, 100])
    result = winsorize_mad(s, n=3)
    assert result.max() < 100
    assert result.median() == s.median()


def test_zscore():
    """标准化后均值为0，标准差为1"""
    s = pd.Series(np.random.randn(100) * 5 + 10)
    result = zscore(s)
    assert abs(result.mean()) < 1e-10
    assert abs(result.std() - 1) < 1e-10


def test_preprocess_cross_section_nan(sample_dates):
    """含NaN的截面处理后仍保持NaN位置"""
    s = pd.Series([1.0, 2.0, np.nan, 4.0, 5.0])
    result = preprocess_cross_section(s)
    assert result.isna().sum() == 1


def test_ep_factor():
    """EP = 1/PE，负PE应被过滤"""
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
    """市值越小得分越高"""
    mv = pd.DataFrame({"A": [1e10], "B": [1e9]})
    size = SizeFactor().compute(mv)
    assert size.loc[0, "B"] > size.loc[0, "A"]


def test_momentum_factor(sample_stocks):
    """动量因子：窗口不足时应为NaN"""
    dates = pd.date_range("2023-01-02", periods=50, freq="B")
    ret = pd.DataFrame(np.random.randn(50, 4) * 0.01, index=dates, columns=sample_stocks)
    mom = MomentumFactor(long_window=20, skip_window=5).compute(ret)
    # 前20行窗口不足
    assert mom.iloc[:20].isna().all().all()
    # 后续应有有效值
    assert mom.iloc[25:].notna().any().any()


def test_roe_ttm():
    """ROE TTM还原：本期累计 + 上年年报 - 上年同期累计"""
    idx = pd.MultiIndex.from_tuples([
        ("000001", pd.Timestamp("2021-03-31")),
        ("000001", pd.Timestamp("2021-06-30")),
        ("000001", pd.Timestamp("2021-12-31")),
        ("000001", pd.Timestamp("2022-03-31")),
        ("000001", pd.Timestamp("2022-06-30")),
    ], names=["stock_code", "report_date"])
    # 2022Q1 TTM = 5 + 20(上年年报) - 4(上年Q1) = 21
    # 2022Q2 TTM = 8 + 20 - 9(上年半年报) = 19
    financial = pd.DataFrame({"roe": [4.0, 9.0, 20.0, 5.0, 8.0]}, index=idx)
    
    factor = ROETTMFactor(disclosure_lag={"q1": 30, "q2": 62, "q3": 31, "q4": 121})
    trading_dates = pd.date_range("2022-04-01", "2022-09-30", freq="B")
    panel = factor.compute(financial, trading_dates)
    
    # 2022Q1报告(3/31)披露滞后30天 -> 4/30可用（首个交易日生效）
    available_date = pd.Timestamp("2022-04-30")
    first_valid = panel["000001"].first_valid_index()
    assert first_valid >= available_date
    # 披露后TTM值应为21
    assert panel.loc[first_valid, "000001"] == pytest.approx(21.0)
    # 半年报(6/30)滞后62天 -> 8/31可用, TTM = 19
    assert panel.loc[pd.Timestamp("2022-08-31"), "000001"] == pytest.approx(19.0)


def test_roe_no_lookahead():
    """ROE因子在披露日前不应有值（无未来函数）"""
    idx = pd.MultiIndex.from_tuples([
        ("000001", pd.Timestamp("2022-12-31")),
    ], names=["stock_code", "report_date"])
    financial = pd.DataFrame({"roe": [15.0]}, index=idx)
    
    factor = ROETTMFactor(disclosure_lag={"q1": 30, "q2": 62, "q3": 31, "q4": 121})
    trading_dates = pd.date_range("2023-01-02", "2023-04-28", freq="B")
    panel = factor.compute(financial, trading_dates)
    
    # 年报披露滞后121天，4/28之前不应有值
    assert panel["000001"].isna().all()


def test_multi_factor_combiner(sample_dates, sample_stocks):
    """合成得分应为各因子标准化值的加权和"""
    np.random.seed(42)
    panels = {
        "ep": pd.DataFrame(np.random.randn(5, 4), index=sample_dates, columns=sample_stocks),
        "bp": pd.DataFrame(np.random.randn(5, 4), index=sample_dates, columns=sample_stocks),
    }
    combiner = MultiFactorCombiner(weights={"ep": 0.5, "bp": 0.5})
    scores = combiner.combine(panels, list(sample_dates))
    
    assert scores.shape == (5, 4)
    # 等权合成两个标准化因子，截面均值应接近0
    assert abs(scores.iloc[0].mean()) < 1e-10
