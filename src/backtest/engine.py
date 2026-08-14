"""
Multi-factor monthly-rotation backtest engine (vectorized implementation)
Supports Top-N portfolio backtest and quantile-group backtest
"""
import numpy as np
import pandas as pd
from typing import Dict, List


class MultiFactorEngine:
    """Multi-factor backtest engine"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.bt_config = config.get("backtest", {})
        self.commission = self.bt_config.get("commission", 0.0003)
        self.stamp_tax = self.bt_config.get("stamp_tax", 0.001)
        self.slippage = self.bt_config.get("slippage", 0.001)
        self.top_n = self.bt_config.get("top_n", 30)
        self.num_groups = self.bt_config.get("num_groups", 5)
        # transaction cost per unit of one-side turnover
        # (commission/slippage on both sides, stamp tax on sell)
        self.turnover_cost = 2 * (self.commission + self.slippage) + self.stamp_tax
    
    def get_rebalance_dates(self, trading_dates: pd.DatetimeIndex,
                            start_date: str = None, end_date: str = None) -> List[pd.Timestamp]:
        """
        Get the last trading day of each month as rebalance dates
        
        Args:
            trading_dates: full trading-day series
            start_date: start date 'YYYYMMDD' (optional)
            end_date: end date 'YYYYMMDD' (optional)
            
        Returns:
            list of rebalance dates
        """
        dates = pd.DatetimeIndex(trading_dates)
        if start_date:
            dates = dates[dates >= pd.to_datetime(start_date)]
        if end_date:
            dates = dates[dates <= pd.to_datetime(end_date)]
        
        series = pd.Series(dates, index=dates)
        month_end = series.groupby([dates.year, dates.month]).max()
        return [pd.Timestamp(d) for d in month_end.values]
    
    def compute_period_returns(self, ret_wide: pd.DataFrame,
                               rebalance_dates: List[pd.Timestamp]) -> pd.DataFrame:
        """
        Compute returns between consecutive rebalance dates
        
        Args:
            ret_wide: daily return wide table (index=trading day, columns=stock code)
            rebalance_dates: list of rebalance dates
            
        Returns:
            period return wide table (index=rebalance dates[:-1]; each row is the
            return from that date to the next rebalance date)
        """
        # treat suspended days as zero return
        cumret = (1 + ret_wide.fillna(0.0)).cumprod()
        
        period_rets = []
        for i in range(len(rebalance_dates) - 1):
            t0, t1 = rebalance_dates[i], rebalance_dates[i + 1]
            ret_period = cumret.loc[t1] / cumret.loc[t0] - 1
            period_rets.append(ret_period)
        
        return pd.DataFrame(period_rets, index=rebalance_dates[:-1])
    
    def _turnover(self, old_holdings: set, new_holdings: set) -> float:
        """Compute turnover (number of stocks sold / previous holdings count)"""
        if not old_holdings:
            return 1.0
        changed = len(old_holdings - new_holdings)
        return changed / max(len(old_holdings), 1)
    
    def run_portfolio(self, scores: pd.DataFrame, period_ret: pd.DataFrame,
                      rebalance_dates: List[pd.Timestamp],
                      inv_vol_panel: pd.DataFrame = None) -> Dict:
        """
        Top-N portfolio backtest
        
        Args:
            scores: combined score wide table (index=rebalance date, columns=stock code)
            period_ret: period return wide table
            rebalance_dates: full rebalance date list (one more date than period_ret)
            inv_vol_panel: daily volatility wide table (optional); when provided,
                           holdings are weighted by inverse volatility, otherwise equal weight
            
        Returns:
            result dict: nav / holdings / turnover
        """
        nav_values = [1.0]
        holdings_history = []
        turnover_history = []
        
        old_holdings = set()
        
        # sample volatility at rebalance dates and forward-fill
        vol_at_rb = None
        if inv_vol_panel is not None:
            rb_index = pd.DatetimeIndex(rebalance_dates)
            vol_at_rb = inv_vol_panel.reindex(
                inv_vol_panel.index.union(rb_index)).ffill()
        
        for t in period_ret.index:
            # stock selection for the period: top_n stocks with the highest scores
            score_t = scores.loc[t].dropna()
            if score_t.empty:
                # scores unavailable (e.g. ML training warm-up): keep old holdings, no cost
                if old_holdings:
                    ret_t = period_ret.loc[t, list(old_holdings)].mean()
                    nav_values.append(nav_values[-1] * (1 + ret_t))
                else:
                    nav_values.append(nav_values[-1])
                holdings_history.append(sorted(old_holdings))
                turnover_history.append(0.0)
                continue
            
            n = min(self.top_n, len(score_t))
            new_holdings = set(score_t.nlargest(n).index)
            
            turnover = self._turnover(old_holdings, new_holdings)
            holding_list = list(new_holdings)
            if vol_at_rb is not None:
                # inverse-volatility weighting: weight proportional to 1/recent volatility
                vols = vol_at_rb.loc[t, holding_list].clip(lower=1e-4)
                wts = (1.0 / vols)
                wts = wts / wts.sum()
                ret_t = (period_ret.loc[t, holding_list] * wts).sum()
            else:
                ret_t = period_ret.loc[t, holding_list].mean()
            cost = turnover * self.turnover_cost
            nav_values.append(nav_values[-1] * (1 + ret_t - cost))
            
            holdings_history.append(sorted(new_holdings))
            turnover_history.append(turnover)
            old_holdings = new_holdings
        
        # align NAV to all rebalance dates: nav[t_k] is the NAV at the end of period k-1
        nav = pd.Series(nav_values, index=rebalance_dates)
        
        return {
            "nav": nav,
            "holdings": holdings_history,
            "turnover": pd.Series(turnover_history, index=period_ret.index),
        }
    
    def run_groups(self, scores: pd.DataFrame, period_ret: pd.DataFrame,
                   rebalance_dates: List[pd.Timestamp], num_groups: int = None) -> Dict[str, pd.Series]:
        """
        Quantile-group backtest: split stocks into num_groups by score, equal weight within each group
        
        Args:
            scores: combined score wide table
            period_ret: period return wide table
            rebalance_dates: full rebalance date list
            num_groups: number of groups
            
        Returns:
            {group alias: NAV series}; group 1 has the highest scores
        """
        num_groups = num_groups or self.num_groups
        
        group_navs = {g: [1.0] for g in range(1, num_groups + 1)}
        old_groups = {}
        
        for t in period_ret.index:
            score_t = scores.loc[t].dropna()
            if score_t.empty:
                # scores unavailable: keep old holdings in each group
                for g in range(1, num_groups + 1):
                    holdings = old_groups.get(g, set())
                    ret_t = period_ret.loc[t, list(holdings)].mean() if holdings else 0.0
                    group_navs[g].append(group_navs[g][-1] * (1 + ret_t))
                continue
            
            # sort by score and split evenly; group 1 has the highest scores
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
