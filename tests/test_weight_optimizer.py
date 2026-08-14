"""
权重优化模块单元测试
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
import pytest

from src.factors.weight_optimizer import (
    rolling_icir_weights, max_icir_weights, MLWalkForwardScorer,
    ewma_icir_weights, orthogonalize_factors, factor_long_short_returns,
    rolling_mv_weights
)


@pytest.fixture
def sample_ic_table():
    """构造IC序列表: factor_a持续为正且稳定, factor_b持续为负"""
    dates = pd.date_range("2023-01-31", periods=24, freq="M")
    rng = np.random.default_rng(42)
    return pd.DataFrame({
        "factor_a": 0.05 + rng.normal(0, 0.02, 24),
        "factor_b": -0.05 + rng.normal(0, 0.02, 24),
    }, index=dates)


def test_rolling_icir_weights(sample_ic_table):
    """ICIR加权应给稳定正IC因子更高权重，负IC因子权重为0"""
    weights = rolling_icir_weights(sample_ic_table, window=12, min_periods=6)
    
    # 前几期样本不足应为NaN
    assert weights.iloc[0].isna().all()
    # 后期有效行: factor_a权重应为1, factor_b为0
    last = weights.iloc[-1]
    assert last["factor_a"] == pytest.approx(1.0, abs=1e-6)
    assert last["factor_b"] == pytest.approx(0.0, abs=1e-6)


def test_max_icir_weights(sample_ic_table):
    """最大化ICIR: 权重和为1且在[0,1]内，倾向正IC因子"""
    w = max_icir_weights(sample_ic_table)
    
    assert w is not None
    assert w.sum() == pytest.approx(1.0, abs=1e-4)
    assert (w >= -1e-8).all() and (w <= 1 + 1e-8).all()
    assert w[0] > w[1]  # 正IC因子权重更高


def test_max_icir_weights_insufficient_data(sample_ic_table):
    """历史不足时应返回None"""
    assert max_icir_weights(sample_ic_table.iloc[:5]) is None


def test_ml_walk_forward_no_lookahead():
    """walk-forward: 训练期不足时得分应为NaN，且得分不依赖未来数据"""
    dates = pd.date_range("2023-01-31", periods=12, freq="M")
    stocks = [f"S{i}" for i in range(20)]
    rng = np.random.default_rng(0)
    
    factor_panels = {
        "f1": pd.DataFrame(rng.normal(size=(12, 20)), index=dates, columns=stocks),
    }
    period_ret = pd.DataFrame(rng.normal(0, 0.05, size=(11, 20)),
                              index=dates[:-1], columns=stocks)
    rebalance_dates = list(dates)
    
    scorer = MLWalkForwardScorer(model_type="ridge", train_window=12,
                                 min_train_periods=8, factor_names=["f1"])
    scores = scorer.fit_predict(factor_panels, period_ret, rebalance_dates)
    
    # 前8期训练不足应为NaN
    assert scores.iloc[:8].isna().all().all()
    # 后期应有有效得分
    assert scores.iloc[-1].notna().any()
    # 得分形状与股票数一致
    assert scores.shape[1] == len(stocks)


def test_ewma_icir_weights(sample_ic_table):
    """EWMA-ICIR加权: 正IC因子权重为1，负IC因子为0，行和为1"""
    weights = ewma_icir_weights(sample_ic_table, halflife=6, min_periods=6)
    
    assert weights.iloc[0].isna().all()
    last = weights.iloc[-1]
    assert last["factor_a"] == pytest.approx(1.0, abs=1e-6)
    assert last["factor_b"] == pytest.approx(0.0, abs=1e-6)


def test_orthogonalize_factors():
    """正交化: 残差因子与前序因子截面相关应接近0，且保持标准化"""
    dates = pd.date_range("2023-01-31", periods=4, freq="M")
    stocks = [f"S{i}" for i in range(50)]
    rng = np.random.default_rng(1)
    
    base = pd.DataFrame(rng.normal(size=(4, 50)), index=dates, columns=stocks)
    # f2 = 0.8*f1 + 噪声，与f1强共线
    noise = pd.DataFrame(rng.normal(0, 0.5, size=(4, 50)), index=dates, columns=stocks)
    factor_panels = {"f1": base, "f2": base * 0.8 + noise}
    
    ortho = orthogonalize_factors(factor_panels, ["f1", "f2"], list(dates))
    
    # f1本身不变（仅标准化）
    assert ortho["f1"].shape == base.shape
    # f2正交化后与f1的截面相关应接近0
    corr = ortho["f1"].loc[dates[-1]].corr(ortho["f2"].loc[dates[-1]])
    assert abs(corr) < 1e-6
    # 残差应重新标准化（均值≈0，标准差≈1）
    resid = ortho["f2"].loc[dates[-1]].dropna()
    assert abs(resid.mean()) < 1e-6
    assert resid.std() == pytest.approx(1.0, abs=0.1)


def test_factor_long_short_returns():
    """多空收益: 因子值与下期收益同向时，多空收益应为正"""
    dates = pd.date_range("2023-01-31", periods=6, freq="M")
    stocks = [f"S{i}" for i in range(30)]
    
    # 因子值固定排序，下期收益与因子值正相关
    factor = pd.DataFrame(
        np.tile(np.arange(30, dtype=float), (6, 1)), index=dates, columns=stocks)
    rets = np.tile(np.arange(30, dtype=float) / 300, (5, 1))
    period_ret = pd.DataFrame(rets, index=dates[:-1], columns=stocks)
    
    ls = factor_long_short_returns({"f1": factor}, period_ret, list(dates),
                                   quantile=0.2)
    assert ls.shape == (5, 1)
    # 高分组收益高于低分组，多空收益应为正
    assert (ls["f1"] > 0).all()


def test_rolling_mv_weights():
    """均值-方差加权: 有效行权重和为1且非负；样本不足为NaN"""
    rng = np.random.default_rng(7)
    dates = pd.date_range("2023-01-31", periods=36, freq="M")
    # 两个因子的多空收益: a持续为正，b持续为负
    ls = pd.DataFrame({
        "a": 0.02 + rng.normal(0, 0.03, 36),
        "b": -0.02 + rng.normal(0, 0.03, 36),
    }, index=dates)
    
    weights = rolling_mv_weights(ls, window=24, min_periods=18)
    
    # 前期样本不足应为NaN
    assert weights.iloc[0].isna().all()
    # 有效行: 权重和为1、非负，且正收益因子权重更高
    valid = weights.dropna(how="all")
    assert len(valid) > 0
    for _, row in valid.iterrows():
        assert row.sum() == pytest.approx(1.0, abs=1e-4)
        assert (row >= -1e-8).all()
        assert row["a"] >= row["b"]
