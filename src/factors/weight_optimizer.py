"""
Factor weight optimization module
1. Rolling ICIR weighting: weight proportional to rolling ICIR, automatically down-weights ineffective factors
2. EWMA-IC weighting: exponentially decayed ICIR weighting, more weight on recent signals
3. Max portfolio ICIR: constrained optimization based on the IC covariance matrix
4. Mean-variance weighting: factor long-short portfolio returns + Ledoit-Wolf shrunk covariance, maximize Sharpe
5. Factor orthogonalization: sequential regression residuals to remove collinearity before weighting
6. ML walk-forward: Ridge / histogram GBDT trained on rolling or expanding windows to predict next-period returns

All methods strictly use information available at each point in time (walk-forward), no lookahead bias
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Optional

from scipy.optimize import minimize
from sklearn.linear_model import Ridge
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.covariance import LedoitWolf

from .base import preprocess_cross_section


def rolling_icir_weights(ic_table: pd.DataFrame, window: int = 12,
                         min_periods: int = 6) -> pd.DataFrame:
    """
    Rolling ICIR weighting: weight = max(rolling ICIR, 0), normalized to sum to 1
    
    Args:
        ic_table: IC series table (index=rebalance date, columns=factor name)
        window: rolling window (months)
        min_periods: minimum number of observations
        
    Returns:
        weight table (same index as ic_table)
    """
    ic_mean = ic_table.rolling(window, min_periods=min_periods).mean()
    ic_std = ic_table.rolling(window, min_periods=min_periods).std()
    icir = ic_mean / ic_std
    
    weights = icir.clip(lower=0)
    row_sum = weights.sum(axis=1)
    weights = weights.div(row_sum, axis=0)
    
    # rows with no valid weights are filled with NaN; caller falls back to equal weight
    weights[row_sum == 0] = np.nan
    return weights


def max_icir_weights(ic_history: pd.DataFrame, shrinkage: float = 0.5) -> Optional[np.ndarray]:
    """
    Given IC history, solve for weights maximizing portfolio ICIR (mu'w / sqrt(w'Cov w)).
    Constraints: weights sum to 1, each weight in [0, 1].
    The covariance matrix is shrunk toward the identity matrix for small-sample stability.
    
    Args:
        ic_history: historical IC table (one row per period, one column per factor)
        shrinkage: covariance shrinkage coefficient (0~1)
        
    Returns:
        weight array; None if history is insufficient
    """
    ic_history = ic_history.dropna(how="all")
    if len(ic_history) < 12:
        return None
    
    mu = ic_history.mean().values
    cov = ic_history.cov().values
    
    # Ledoit-style shrinkage: Cov_reg = (1-s)*Cov + s*mean(diag)*I
    diag_mean = np.trace(cov) / len(cov)
    cov_reg = (1 - shrinkage) * cov + shrinkage * diag_mean * np.eye(len(mu))
    
    def neg_icir(w):
        port_ic = mu @ w
        port_var = w @ cov_reg @ w
        if port_var <= 0:
            return 0.0
        return -port_ic / np.sqrt(port_var)
    
    n = len(mu)
    constraints = [{"type": "eq", "fun": lambda w: w.sum() - 1}]
    bounds = [(0.0, 1.0)] * n
    x0 = np.ones(n) / n
    
    result = minimize(neg_icir, x0, method="SLSQP",
                      bounds=bounds, constraints=constraints,
                      options={"maxiter": 200})
    
    if not result.success:
        return None
    return result.x


def rolling_max_icir_weights(ic_table: pd.DataFrame, window: int = 36,
                             min_periods: int = 24, shrinkage: float = 0.5) -> pd.DataFrame:
    """
    Maximize portfolio ICIR within a rolling window, solved period by period
    
    Args:
        ic_table: IC series table
        window: rolling window (months)
        min_periods: minimum number of observations
        shrinkage: covariance shrinkage coefficient
        
    Returns:
        weight table (index=ic_table index, columns=factor name)
    """
    records = []
    for i in range(len(ic_table)):
        hist = ic_table.iloc[max(0, i - window + 1):i + 1]
        if len(hist.dropna(how="all")) >= min_periods:
            w = max_icir_weights(hist, shrinkage=shrinkage)
            records.append(w if w is not None else np.full(len(ic_table.columns), np.nan))
        else:
            records.append(np.full(len(ic_table.columns), np.nan))
    
    return pd.DataFrame(records, index=ic_table.index, columns=ic_table.columns)


def ewma_icir_weights(ic_table: pd.DataFrame, halflife: int = 6,
                      min_periods: int = 6) -> pd.DataFrame:
    """
    EWMA-ICIR weighting: estimate IC mean and volatility with exponential decay
    (half-life of halflife months); weight = max(EWMA-ICIR, 0), normalized.
    Compared to an equal-weight window, recent signals get higher weight.
    
    Args:
        ic_table: IC series table (index=rebalance date, columns=factor name)
        halflife: exponential decay half-life (months)
        min_periods: minimum number of observations
        
    Returns:
        weight table (same index as ic_table)
    """
    ewm_mean = ic_table.ewm(halflife=halflife, min_periods=min_periods).mean()
    ewm_std = ic_table.ewm(halflife=halflife, min_periods=min_periods).std()
    icir = ewm_mean / ewm_std
    
    weights = icir.clip(lower=0)
    row_sum = weights.sum(axis=1)
    weights = weights.div(row_sum, axis=0)
    weights[row_sum == 0] = np.nan
    return weights


def orthogonalize_factors(factor_panels: Dict[str, pd.DataFrame], order: List[str],
                          rebalance_dates: List[pd.Timestamp]) -> Dict[str, pd.DataFrame]:
    """
    Sequential orthogonalization: following `order`, each factor is regressed on the
    previously orthogonalized factors and the residual is taken, removing collinearity
    (e.g. size absorbing value/quality information); residuals are re-normalized.
    
    Args:
        factor_panels: {factor name: daily factor wide table} (raw values)
        order: orthogonalization order (factors to preserve first come earlier, e.g. size)
        rebalance_dates: list of rebalance dates
        
    Returns:
        dict of orthogonalized factor panels (daily, forward-filled between rebalance dates)
    """
    rb_index = pd.DatetimeIndex(rebalance_dates)
    cross_sections = {}
    for name in order:
        panel = factor_panels[name]
        cross = panel.reindex(panel.index.union(rb_index)).ffill().loc[rb_index]
        cross_sections[name] = cross.apply(preprocess_cross_section, axis=1)
    
    ortho_tables = {}
    for i, name in enumerate(order):
        if i == 0:
            ortho_tables[name] = cross_sections[name]
            continue
        
        prev_names = order[:i]
        rows = []
        for t in rb_index:
            y = cross_sections[name].loc[t]
            x_df = pd.DataFrame({p: ortho_tables[p].loc[t] for p in prev_names})
            valid = y.notna() & x_df.notna().all(axis=1)
            resid = pd.Series(np.nan, index=y.index)
            if valid.sum() > len(prev_names) + 2:
                beta, *_ = np.linalg.lstsq(x_df.loc[valid].values,
                                           y.loc[valid].values, rcond=None)
                resid.loc[valid] = y.loc[valid].values - x_df.loc[valid].values @ beta
            rows.append(preprocess_cross_section(resid))
        ortho_tables[name] = pd.DataFrame(rows, index=rb_index)
    
    # Rebalance-date cross sections + forward fill back to daily,
    # keeping the interface consistent with dynamic_weight_scores
    result = {}
    for name in order:
        daily_index = factor_panels[name].index.union(rb_index)
        result[name] = ortho_tables[name].reindex(daily_index).ffill()
    return result


def factor_long_short_returns(factor_panels: Dict[str, pd.DataFrame],
                              period_ret: pd.DataFrame,
                              rebalance_dates: List[pd.Timestamp],
                              quantile: float = 0.2) -> pd.DataFrame:
    """
    Build monthly long-short returns for each factor: each period, long the top
    `quantile` by factor value and short the bottom `quantile`.
    
    Args:
        factor_panels: {factor name: daily factor wide table}
        period_ret: period return wide table (index=rebalance dates[:-1])
        rebalance_dates: list of rebalance dates
        quantile: fraction on each side of the long-short split
        
    Returns:
        long-short return table (index=period_ret.index, columns=factor name)
    """
    rb_index = pd.DatetimeIndex(rebalance_dates)
    ls_records = {}
    for name, panel in factor_panels.items():
        cross = panel.reindex(panel.index.union(rb_index)).ffill().loc[period_ret.index]
        rets = {}
        for t in period_ret.index:
            x = cross.loc[t].dropna().sort_values()
            k = max(int(len(x) * quantile), 5)
            long_ret = period_ret.loc[t, list(x.index[-k:])].mean()
            short_ret = period_ret.loc[t, list(x.index[:k])].mean()
            rets[t] = long_ret - short_ret
        ls_records[name] = rets
    return pd.DataFrame(ls_records)


def rolling_mv_weights(ls_returns: pd.DataFrame, window: int = 36,
                       min_periods: int = 24) -> pd.DataFrame:
    """
    Mean-variance weighting (factor-return version): within a rolling window, use the
    realized returns of factor long-short portfolios with a Ledoit-Wolf shrunk covariance,
    and solve for Sharpe-maximizing weights (sum to 1, non-negative).
    Usage is consistent with IC tables: row t is computed from returns up to and including t,
    and is applied at the next rebalance date.
    
    Args:
        ls_returns: factor long-short return table
        window: rolling window (months)
        min_periods: minimum number of observations
        
    Returns:
        weight table; NaN for periods with non-positive expected portfolio return or
        failed solving (caller falls back to equal weight)
    """
    n = len(ls_returns.columns)
    records = []
    for i in range(len(ls_returns)):
        hist = ls_returns.iloc[max(0, i - window + 1):i + 1].dropna()
        if len(hist) < min_periods:
            records.append(np.full(n, np.nan))
            continue
        
        mu = hist.mean().values
        cov = LedoitWolf().fit(hist.values).covariance_
        
        def neg_sharpe(w):
            port_ret = mu @ w
            port_var = w @ cov @ w
            if port_var <= 0:
                return 0.0
            return -port_ret / np.sqrt(port_var)
        
        result = minimize(neg_sharpe, np.ones(n) / n, method="SLSQP",
                          bounds=[(0.0, 1.0)] * n,
                          constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1}],
                          options={"maxiter": 200})
        # Non-positive expected portfolio return means no usable alpha direction
        # in the window; fall back to equal weight
        if result.success and mu @ result.x > 0:
            records.append(result.x)
        else:
            records.append(np.full(n, np.nan))
    
    return pd.DataFrame(records, index=ls_returns.index, columns=ls_returns.columns)


def dynamic_weight_scores(factor_panels: Dict[str, pd.DataFrame], weights_df: pd.DataFrame,
                          rebalance_dates: List[pd.Timestamp],
                          factor_names: List[str] = None) -> pd.DataFrame:
    """
    Combine factor scores with time-varying weights.
    Weights applied at rebalance date T come only from IC information strictly before T
    (strict point-in-time alignment).
    
    Args:
        factor_panels: {factor name: daily factor wide table} (raw values)
        weights_df: weight table (index=rebalance dates for IC, columns=factor name)
        rebalance_dates: all rebalance dates
        factor_names: list of factor names (defaults to weights_df columns)
        
    Returns:
        combined score wide table (index=rebalance date, columns=stock code)
    """
    factor_names = factor_names or list(weights_df.columns)
    scores = None
    n_factors = len(factor_names)
    equal_w = np.ones(n_factors) / n_factors
    
    # Precompute normalized cross sections at rebalance dates for each factor
    normalized = {}
    for name in factor_names:
        panel = factor_panels[name]
        panel = panel.reindex(panel.index.union(pd.DatetimeIndex(rebalance_dates)))
        normalized[name] = panel.ffill().loc[rebalance_dates].apply(
            preprocess_cross_section, axis=1)
    
    for k, t in enumerate(rebalance_dates):
        # use only weights derived from IC information strictly before t
        past = weights_df.loc[weights_df.index < t]
        if not past.empty and past.iloc[-1].notna().any():
            w = past.iloc[-1].fillna(0).values
            w = w / w.sum() if w.sum() > 0 else equal_w
        else:
            w = equal_w
        
        row = sum(normalized[name].loc[t] * wi
                  for name, wi in zip(factor_names, w) if wi > 0)
        row_df = pd.DataFrame([row], index=[t])
        scores = row_df if scores is None else pd.concat([scores, row_df])
    
    return scores


class MLWalkForwardScorer:
    """
    Machine-learning walk-forward scorer.
    Rolling training: train a model on (factor cross-section -> next-period return)
    samples from the train_window months before T, then predict the next-period
    return of the cross section at T as the score.
    """
    
    def __init__(self, model_type: str = "ridge", train_window: int = 36,
                 min_train_periods: int = 24, factor_names: List[str] = None,
                 expanding: bool = False, clip_targets: bool = True):
        """
        Args:
            model_type: 'ridge' or 'gbdt'
            train_window: training window (months), ignored when expanding=True
            min_train_periods: minimum number of training periods
            factor_names: factor names used as model features
            expanding: use an expanding window (full history) instead of rolling
            clip_targets: winsorize training targets at the 1%/99% cross-section
                          quantiles to reduce heavy-tail impact
        """
        self.model_type = model_type
        self.train_window = train_window
        self.min_train_periods = min_train_periods
        self.factor_names = factor_names
        self.expanding = expanding
        self.clip_targets = clip_targets
    
    def _new_model(self):
        if self.model_type == "ridge":
            return Ridge(alpha=1.0)
        elif self.model_type == "gbdt":
            return HistGradientBoostingRegressor(
                max_iter=150, learning_rate=0.05, max_leaf_nodes=8,
                min_samples_leaf=50, l2_regularization=1.0,
                early_stopping=False, random_state=42
            )
        raise ValueError(f"Unknown model type: {self.model_type}")
    
    def _build_samples(self, factor_panels: Dict[str, pd.DataFrame],
                       period_ret: pd.DataFrame) -> pd.DataFrame:
        """Build the training sample long table: each row = (rebalance date, stock, factor z-values, next-period return)"""
        factor_names = self.factor_names
        frames = []
        
        for t in period_ret.index:
            cols = {}
            for name in factor_names:
                panel = factor_panels[name]
                if t in panel.index:
                    cols[name] = preprocess_cross_section(panel.loc[t])
            if not cols:
                continue
            x = pd.DataFrame(cols)
            x["period_date"] = t
            x["stock_code"] = x.index
            y = period_ret.loc[t].reindex(x.index)
            if self.clip_targets:
                # clip at cross-section 1%/99% quantiles to reduce heavy-tail interference
                y = y.clip(lower=y.quantile(0.01), upper=y.quantile(0.99))
            x["y"] = y
            frames.append(x)
        
        return pd.concat(frames, ignore_index=True)
    
    def fit_predict(self, factor_panels: Dict[str, pd.DataFrame],
                    period_ret: pd.DataFrame,
                    rebalance_dates: List[pd.Timestamp]) -> pd.DataFrame:
        """
        Walk-forward prediction, outputting cross-section scores for each rebalance date
        
        Returns:
            score wide table (index=rebalance date, columns=stock code);
            dates with insufficient training history are NaN
        """
        self.factor_names = self.factor_names or list(factor_panels.keys())
        samples = self._build_samples(factor_panels, period_ret)
        
        scores_rows = []
        for k, t in enumerate(rebalance_dates):
            # Training set: history strictly before t (expanding uses all history,
            # rolling limits to train_window)
            if self.expanding:
                train = samples[samples["period_date"] < t]
            else:
                train = samples[(samples["period_date"] < t) &
                                (samples["period_date"] >= t - pd.DateOffset(months=self.train_window))]
            n_periods = train["period_date"].nunique()
            
            if n_periods < self.min_train_periods:
                scores_rows.append(pd.Series(dtype=float, name=t))
                continue
            
            x_train = train[self.factor_names].values
            y_train = train["y"].values
            valid = ~np.isnan(y_train)
            
            model = self._new_model()
            if self.model_type == "ridge":
                # Ridge does not support NaN; fill with 0 (0 is the cross-section mean after zscore)
                x_fit = np.nan_to_num(x_train[valid], nan=0.0)
                model.fit(x_fit, y_train[valid])
            else:
                # histogram GBDT handles NaN natively
                model.fit(x_train[valid], y_train[valid])
            
            # predict the cross section at T
            x_t = pd.DataFrame({
                name: preprocess_cross_section(factor_panels[name].loc[t])
                if t in factor_panels[name].index else np.nan
                for name in self.factor_names
            })
            x_pred = x_t.values
            if self.model_type == "ridge":
                x_pred = np.nan_to_num(x_pred, nan=0.0)
            
            pred = model.predict(x_pred)
            scores_rows.append(pd.Series(pred, index=x_t.index, name=t))
        
        return pd.DataFrame(scores_rows)
