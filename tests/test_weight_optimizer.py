"""
Unit tests for the factor weight optimization module
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
    """Build an IC series table: factor_a is persistently positive and stable, factor_b is persistently negative"""
    dates = pd.date_range("2023-01-31", periods=24, freq="M")
    rng = np.random.default_rng(42)
    return pd.DataFrame({
        "factor_a": 0.05 + rng.normal(0, 0.02, 24),
        "factor_b": -0.05 + rng.normal(0, 0.02, 24),
    }, index=dates)


def test_rolling_icir_weights(sample_ic_table):
    """ICIR weighting should give a higher weight to the stable positive-IC factor and zero weight to the negative-IC factor"""
    weights = rolling_icir_weights(sample_ic_table, window=12, min_periods=6)
    
    # Early rows lack enough samples and should be NaN
    assert weights.iloc[0].isna().all()
    # For later valid rows: factor_a weight should be 1, factor_b should be 0
    last = weights.iloc[-1]
    assert last["factor_a"] == pytest.approx(1.0, abs=1e-6)
    assert last["factor_b"] == pytest.approx(0.0, abs=1e-6)


def test_max_icir_weights(sample_ic_table):
    """Max-ICIR: weights sum to 1 within [0,1] and favor the positive-IC factor"""
    w = max_icir_weights(sample_ic_table)
    
    assert w is not None
    assert w.sum() == pytest.approx(1.0, abs=1e-4)
    assert (w >= -1e-8).all() and (w <= 1 + 1e-8).all()
    assert w[0] > w[1]  # positive-IC factor gets a higher weight


def test_max_icir_weights_insufficient_data(sample_ic_table):
    """Should return None when history is insufficient"""
    assert max_icir_weights(sample_ic_table.iloc[:5]) is None


def test_ml_walk_forward_no_lookahead():
    """Walk-forward: scores should be NaN during the warm-up period, and never depend on future data"""
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
    
    # The first 8 periods lack training data and should be NaN
    assert scores.iloc[:8].isna().all().all()
    # Later periods should have valid scores
    assert scores.iloc[-1].notna().any()
    # Score shape matches the number of stocks
    assert scores.shape[1] == len(stocks)


def test_ewma_icir_weights(sample_ic_table):
    """EWMA-ICIR weighting: positive-IC factor weight 1, negative-IC factor 0, rows sum to 1"""
    weights = ewma_icir_weights(sample_ic_table, halflife=6, min_periods=6)
    
    assert weights.iloc[0].isna().all()
    last = weights.iloc[-1]
    assert last["factor_a"] == pytest.approx(1.0, abs=1e-6)
    assert last["factor_b"] == pytest.approx(0.0, abs=1e-6)


def test_orthogonalize_factors():
    """Orthogonalization: residual factors should be nearly uncorrelated with preceding factors, and stay standardized"""
    dates = pd.date_range("2023-01-31", periods=4, freq="M")
    stocks = [f"S{i}" for i in range(50)]
    rng = np.random.default_rng(1)
    
    base = pd.DataFrame(rng.normal(size=(4, 50)), index=dates, columns=stocks)
    # f2 = 0.8*f1 + noise, strongly collinear with f1
    noise = pd.DataFrame(rng.normal(0, 0.5, size=(4, 50)), index=dates, columns=stocks)
    factor_panels = {"f1": base, "f2": base * 0.8 + noise}
    
    ortho = orthogonalize_factors(factor_panels, ["f1", "f2"], list(dates))
    
    # f1 itself is unchanged (standardization only)
    assert ortho["f1"].shape == base.shape
    # Cross-sectional correlation between orthogonalized f2 and f1 should be ~0
    corr = ortho["f1"].loc[dates[-1]].corr(ortho["f2"].loc[dates[-1]])
    assert abs(corr) < 1e-6
    # Residuals should be re-standardized (mean ~0, std ~1)
    resid = ortho["f2"].loc[dates[-1]].dropna()
    assert abs(resid.mean()) < 1e-6
    assert resid.std() == pytest.approx(1.0, abs=0.1)


def test_factor_long_short_returns():
    """Long-short returns: should be positive when factor values are aligned with next-period returns"""
    dates = pd.date_range("2023-01-31", periods=6, freq="M")
    stocks = [f"S{i}" for i in range(30)]
    
    # Fixed factor ordering; next-period returns are positively correlated with factor values
    factor = pd.DataFrame(
        np.tile(np.arange(30, dtype=float), (6, 1)), index=dates, columns=stocks)
    rets = np.tile(np.arange(30, dtype=float) / 300, (5, 1))
    period_ret = pd.DataFrame(rets, index=dates[:-1], columns=stocks)
    
    ls = factor_long_short_returns({"f1": factor}, period_ret, list(dates),
                                   quantile=0.2)
    assert ls.shape == (5, 1)
    # The high-score group outperforms the low-score group, so long-short returns are positive
    assert (ls["f1"] > 0).all()


def test_rolling_mv_weights():
    """Mean-variance weighting: valid rows sum to 1 and are non-negative; insufficient samples yield NaN"""
    rng = np.random.default_rng(7)
    dates = pd.date_range("2023-01-31", periods=36, freq="M")
    # Long-short returns of two factors: a persistently positive, b persistently negative
    ls = pd.DataFrame({
        "a": 0.02 + rng.normal(0, 0.03, 36),
        "b": -0.02 + rng.normal(0, 0.03, 36),
    }, index=dates)
    
    weights = rolling_mv_weights(ls, window=24, min_periods=18)
    
    # Early rows lack samples and should be NaN
    assert weights.iloc[0].isna().all()
    # Valid rows: weights sum to 1, non-negative, and the positive-return factor gets a higher weight
    valid = weights.dropna(how="all")
    assert len(valid) > 0
    for _, row in valid.iterrows():
        assert row.sum() == pytest.approx(1.0, abs=1e-4)
        assert (row >= -1e-8).all()
        assert row["a"] >= row["b"]
