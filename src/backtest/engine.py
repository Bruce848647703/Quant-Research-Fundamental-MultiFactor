"""
多因子月度轮动回测引擎（向量化实现）
支持Top-N组合回测与分层（分位数）回测
"""
import numpy as np
import pandas as pd
from typing import Dict, List


class MultiFactorEngine:
    """多因子回测引擎"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.bt_config = config.get("backtest", {})
        self.commission = self.bt_config.get("commission", 0.0003)
        self.stamp_tax = self.bt_config.get("stamp_tax", 0.001)
        self.slippage = self.bt_config.get("slippage", 0.001)
        self.top_n = self.bt_config.get("top_n", 30)
        self.num_groups = self.bt_config.get("num_groups", 5)
        # 单边换手对应的交易成本（买入+卖出双边佣金/滑点，卖出印花税）
        self.turnover_cost = 2 * (self.commission + self.slippage) + self.stamp_tax
    
    def get_rebalance_dates(self, trading_dates: pd.DatetimeIndex,
                            start_date: str = None, end_date: str = None) -> List[pd.Timestamp]:
        """
        获取每月最后一个交易日作为调仓日
        
        Args:
            trading_dates: 全部交易日序列
            start_date: 起始日期 'YYYYMMDD'（可选）
            end_date: 结束日期 'YYYYMMDD'（可选）
            
        Returns:
            调仓日期列表
        """
        dates = pd.DatetimeIndex(trading_dates)
        if start_date:
            dates = dates[dates >= pd.to_datetime(start_date)]
        if end_date:
            dates = dates[dates <= pd.to_datetime(end_date)]
        
        series = pd.Series(dates, index=dates)
        month_end = series.groupby([dates.year, dates.month]).max()
        return list(month_end.values)
    
    def compute_period_returns(self, ret_wide: pd.DataFrame,
                               rebalance_dates: List[pd.Timestamp]) -> pd.DataFrame:
        """
        计算相邻调仓日之间的区间收益率
        
        Args:
            ret_wide: 日收益率宽表（index=交易日, columns=股票代码）
            rebalance_dates: 调仓日期列表
            
        Returns:
            区间收益宽表（index=调仓日[:-1], 每行表示该行日期到下一调仓日的收益）
        """
        # 停牌日按0收益处理
        cumret = (1 + ret_wide.fillna(0.0)).cumprod()
        
        period_rets = []
        for i in range(len(rebalance_dates) - 1):
            t0, t1 = rebalance_dates[i], rebalance_dates[i + 1]
            ret_period = cumret.loc[t1] / cumret.loc[t0] - 1
            period_rets.append(ret_period)
        
        return pd.DataFrame(period_rets, index=rebalance_dates[:-1])
    
    def _turnover(self, old_holdings: set, new_holdings: set) -> float:
        """计算换手率（换出股票数 / 原持仓数）"""
        if not old_holdings:
            return 1.0
        changed = len(old_holdings - new_holdings)
        return changed / max(len(old_holdings), 1)
    
    def run_portfolio(self, scores: pd.DataFrame, period_ret: pd.DataFrame,
                      rebalance_dates: List[pd.Timestamp]) -> Dict:
        """
        Top-N等权组合回测
        
        Args:
            scores: 合成得分宽表（index=调仓日, columns=股票代码）
            period_ret: 区间收益宽表
            rebalance_dates: 完整调仓日期列表（比period_ret多一个期末日期）
            
        Returns:
            结果字典: nav / holdings / turnover
        """
        nav_values = [1.0]
        holdings_history = []
        turnover_history = []
        
        old_holdings = set()
        
        for t in period_ret.index:
            # 当期选股：得分最高的top_n只
            score_t = scores.loc[t].dropna()
            n = min(self.top_n, len(score_t))
            new_holdings = set(score_t.nlargest(n).index)
            
            turnover = self._turnover(old_holdings, new_holdings)
            ret_t = period_ret.loc[t, list(new_holdings)].mean()
            cost = turnover * self.turnover_cost
            nav_values.append(nav_values[-1] * (1 + ret_t - cost))
            
            holdings_history.append(sorted(new_holdings))
            turnover_history.append(turnover)
            old_holdings = new_holdings
        
        # 净值对齐全部调仓日: nav[t_k]表示第k-1期结束时的净值
        nav = pd.Series(nav_values, index=rebalance_dates)
        
        return {
            "nav": nav,
            "holdings": holdings_history,
            "turnover": pd.Series(turnover_history, index=period_ret.index),
        }
    
    def run_groups(self, scores: pd.DataFrame, period_ret: pd.DataFrame,
                   rebalance_dates: List[pd.Timestamp], num_groups: int = None) -> Dict[str, pd.Series]:
        """
        分层回测：按得分分为num_groups组，各组等权
        
        Args:
            scores: 合成得分宽表
            period_ret: 区间收益宽表
            rebalance_dates: 完整调仓日期列表
            num_groups: 分组数量
            
        Returns:
            {组别名: 净值序列}，第1组得分最高
        """
        num_groups = num_groups or self.num_groups
        
        group_navs = {g: [1.0] for g in range(1, num_groups + 1)}
        old_groups = {}
        
        for t in period_ret.index:
            score_t = scores.loc[t].dropna()
            # 按得分排序后均分，第1组为得分最高组
            ranked = score_t.sort_values(ascending=False)
            group_lists = np.array_split(ranked.index.values, num_groups)
            
            for g in range(1, num_groups + 1):
                new_holdings = set(group_lists[g - 1])
                turnover = self._turnover(old_groups.get(g, set()), new_holdings)
                
                if new_holdings:
                    ret_t = period_ret.loc[t, list(new_holdings)].mean()
                else:
                    ret_t = 0.0
                cost = turnover * self.turnover_cost
                group_navs[g].append(group_navs[g][-1] * (1 + ret_t - cost))
                old_groups[g] = new_holdings
        
        return {
            f"group_{g}": pd.Series(vals, index=rebalance_dates)
            for g, vals in group_navs.items()
        }
