"""
因子权重优化模块
1. ICIR动态加权: 权重正比于滚动ICIR，自动降权无效因子
2. EWMA-IC加权: 指数衰减的ICIR加权，近期信号权重更高
3. 最大化组合ICIR: 基于因子IC协方差矩阵的约束优化
4. 均值-方差加权: 基于因子多空组合真实收益 + Ledoit-Wolf收缩协方差，最大化夏普
5. 因子正交化: 逐层回归取残差，消除因子共线后再加权
6. 机器学习walk-forward: Ridge回归 / 直方图GBDT 滚动或扩展窗口训练直接预测下期收益

所有方法严格使用时点之前的信息（walk-forward），避免未来函数
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
    滚动ICIR加权：权重 = max(滚动ICIR, 0)，归一化到和为1
    
    Args:
        ic_table: IC序列表（index=调仓日, columns=因子名）
        window: 滚动窗口（月）
        min_periods: 最少观测期数
        
    Returns:
        权重表（与ic_table同索引）
    """
    ic_mean = ic_table.rolling(window, min_periods=min_periods).mean()
    ic_std = ic_table.rolling(window, min_periods=min_periods).std()
    icir = ic_mean / ic_std
    
    weights = icir.clip(lower=0)
    row_sum = weights.sum(axis=1)
    weights = weights.div(row_sum, axis=0)
    
    # 无有效权重的行填NaN，由调用方回退为等权
    weights[row_sum == 0] = np.nan
    return weights


def max_icir_weights(ic_history: pd.DataFrame, shrinkage: float = 0.5) -> Optional[np.ndarray]:
    """
    给定历史IC序列，求最大化组合ICIR (mu'w / sqrt(w'Cov w)) 的权重
    约束: 权重和为1, 各权重在[0,1]之间
    协方差矩阵向单位阵收缩以增强小样本下的稳定性
    
    Args:
        ic_history: 历史IC表（每行一期, 每列一个因子）
        shrinkage: 协方差收缩系数 (0~1)
        
    Returns:
        权重数组；历史数据不足时返回None
    """
    ic_history = ic_history.dropna(how="all")
    if len(ic_history) < 12:
        return None
    
    mu = ic_history.mean().values
    cov = ic_history.cov().values
    
    # Ledoit式收缩: Cov_reg = (1-s)*Cov + s*diag均值*I
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
    滚动窗口内最大化组合ICIR，逐期求解权重
    
    Args:
        ic_table: IC序列表
        window: 滚动窗口（月）
        min_periods: 最少观测期数
        shrinkage: 协方差收缩系数
        
    Returns:
        权重表（index=ic_table索引, columns=因子名）
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
    EWMA-ICIR加权：指数衰减（半衰期halflife个月）估计IC均值与波动，
    权重 = max(EWMA-ICIR, 0) 归一化。相比等权窗口，近期信号权重更高
    
    Args:
        ic_table: IC序列表（index=调仓日, columns=因子名）
        halflife: 指数衰减半衰期（月）
        min_periods: 最少观测期数
        
    Returns:
        权重表（与ic_table同索引）
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
    逐层正交化：按order顺序，每个因子对前面已正交化的因子做截面回归取残差，
    消除因子间共线（如size吃掉价值/质量的信息），残差再标准化
    
    Args:
        factor_panels: {因子名: 日频因子宽表}（原始值）
        order: 正交化顺序（先保留的因子在前，如size优先）
        rebalance_dates: 调仓日期列表
        
    Returns:
        正交化后的因子面板dict（日频，调仓日之间前向填充）
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
    
    # 调仓日截面 + 前向填充回日频，保持与dynamic_weight_scores接口一致
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
    构造各因子的多空组合月度收益：每期按因子值取前quantile做多、后quantile做空
    
    Args:
        factor_panels: {因子名: 日频因子宽表}
        period_ret: 区间收益宽表（index=调仓日[:-1]）
        rebalance_dates: 调仓日期列表
        quantile: 多空两端各自的分位比例
        
    Returns:
        多空收益表（index=period_ret.index, columns=因子名）
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
    均值-方差加权（因子收益版）：滚动窗口内用因子多空组合的真实收益，
    Ledoit-Wolf收缩协方差，求解最大化夏普的权重（权重和为1, 非负）
    权重表与IC表用法一致：t行的权重由t及之前的收益算出，应用于下一调仓日
    
    Args:
        ls_returns: 因子多空收益表
        window: 滚动窗口（月）
        min_periods: 最少观测期数
        
    Returns:
        权重表；组合预期收益非正或求解失败的期为NaN（调用方回退等权）
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
        # 组合预期收益非正说明窗口内无可用alpha方向，回退等权
        if result.success and mu @ result.x > 0:
            records.append(result.x)
        else:
            records.append(np.full(n, np.nan))
    
    return pd.DataFrame(records, index=ls_returns.index, columns=ls_returns.columns)


def dynamic_weight_scores(factor_panels: Dict[str, pd.DataFrame], weights_df: pd.DataFrame,
                          rebalance_dates: List[pd.Timestamp],
                          factor_names: List[str] = None) -> pd.DataFrame:
    """
    按时点动态权重合成因子得分
    调仓日T使用的权重仅来自T之前的IC信息（严格时点对齐）
    
    Args:
        factor_panels: {因子名: 日频因子宽表}（原始值）
        weights_df: 权重表（index=IC对应调仓日, columns=因子名）
        rebalance_dates: 全部调仓日期
        factor_names: 因子名列表（默认取weights_df的列）
        
    Returns:
        合成得分宽表（index=调仓日, columns=股票代码）
    """
    factor_names = factor_names or list(weights_df.columns)
    scores = None
    n_factors = len(factor_names)
    equal_w = np.ones(n_factors) / n_factors
    
    # 预计算各因子在调仓日的标准化截面
    normalized = {}
    for name in factor_names:
        panel = factor_panels[name]
        panel = panel.reindex(panel.index.union(pd.DatetimeIndex(rebalance_dates)))
        normalized[name] = panel.ffill().loc[rebalance_dates].apply(
            preprocess_cross_section, axis=1)
    
    for k, t in enumerate(rebalance_dates):
        # 仅使用严格早于t的IC信息对应的权重
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
    机器学习walk-forward打分器
    滚动训练: 用T之前train_window个月的(因子截面 -> 下期收益)样本训练模型，
    预测T期截面的下期收益作为得分
    """
    
    def __init__(self, model_type: str = "ridge", train_window: int = 36,
                 min_train_periods: int = 24, factor_names: List[str] = None,
                 expanding: bool = False, clip_targets: bool = True):
        """
        Args:
            model_type: 'ridge' 或 'gbdt'
            train_window: 训练窗口（月），expanding=True时忽略
            min_train_periods: 最少训练期数
            factor_names: 参与建模的因子名
            expanding: 使用扩展窗口（全部历史）而非滚动窗口
            clip_targets: 训练目标按截面1%/99%分位缩尾，降低重尾影响
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
        raise ValueError(f"未知模型类型: {self.model_type}")
    
    def _build_samples(self, factor_panels: Dict[str, pd.DataFrame],
                       period_ret: pd.DataFrame) -> pd.DataFrame:
        """构建训练样本长表: 每行 = (调仓日, 股票, 各因子z值, 下期收益)"""
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
                # 截面1%/99%分位缩尾，降低重尾收益对回归的干扰
                y = y.clip(lower=y.quantile(0.01), upper=y.quantile(0.99))
            x["y"] = y
            frames.append(x)
        
        return pd.concat(frames, ignore_index=True)
    
    def fit_predict(self, factor_panels: Dict[str, pd.DataFrame],
                    period_ret: pd.DataFrame,
                    rebalance_dates: List[pd.Timestamp]) -> pd.DataFrame:
        """
        walk-forward预测，输出各调仓日截面得分
        
        Returns:
            得分宽表（index=调仓日, columns=股票代码），训练期不足的日期为NaN
        """
        self.factor_names = self.factor_names or list(factor_panels.keys())
        samples = self._build_samples(factor_panels, period_ret)
        
        scores_rows = []
        for k, t in enumerate(rebalance_dates):
            # 训练集: 严格早于t的历史期（扩展窗口用全部历史，滚动窗口限train_window）
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
                # Ridge不支持NaN，填充0（zscore后0即截面均值）
                x_fit = np.nan_to_num(x_train[valid], nan=0.0)
                model.fit(x_fit, y_train[valid])
            else:
                # 直方图GBDT原生支持NaN
                model.fit(x_train[valid], y_train[valid])
            
            # T期截面预测
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
