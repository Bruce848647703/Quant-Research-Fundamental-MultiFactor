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
    rolling_icir_weights, max_icir_weights, MLWalkForwardScorer
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
