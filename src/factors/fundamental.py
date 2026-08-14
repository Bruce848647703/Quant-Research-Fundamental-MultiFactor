"""
Fundamental multi-factor module
Five factors: EP / BP / ROE(TTM) / Size / Momentum
"""
import numpy as np
import pandas as pd
from typing import Dict, List

from .base import BaseFactor, preprocess_cross_section


class EPFactor(BaseFactor):
    """Earnings yield factor EP = 1 / PE(TTM), valid only for PE > 0"""
    
    name = "ep"
    
    def compute(self, pe_wide: pd.DataFrame) -> pd.DataFrame:
        pe = pe_wide.where(pe_wide > 0)
        return 1.0 / pe


class BPFactor(BaseFactor):
    """Book-to-price factor BP = 1 / PB, valid only for PB > 0"""
    
    name = "bp"
    
    def compute(self, pb_wide: pd.DataFrame) -> pd.DataFrame:
        pb = pb_wide.where(pb_wide > 0)
        return 1.0 / pb


class SizeFactor(BaseFactor):
    """Small-cap factor Size = -ln(total market cap), smaller cap scores higher"""
    
    name = "size"
    
    def compute(self, mv_wide: pd.DataFrame) -> pd.DataFrame:
        mv = mv_wide.where(mv_wide > 0)
        return -np.log(mv)


class MomentumFactor(BaseFactor):
    """Momentum factor: return over the past long_window days, excluding the most recent skip days (12-1 momentum)"""
    
    name = "momentum"
    
    def __init__(self, long_window: int = 252, skip_window: int = 21):
        self.long_window = long_window
        self.skip_window = skip_window
    
    def compute(self, ret_wide: pd.DataFrame) -> pd.DataFrame:
        logret = np.log1p(ret_wide.fillna(0.0))
        # raw-data validity mask (used for availability filtering)
        valid_mask = ret_wide.notna().astype(float)
        
        cum = logret.cumsum()
        momentum = cum.shift(self.skip_window) - cum.shift(self.long_window)
        
        # invalidate when fewer than 80% of trading days in the window have data
        valid_count = valid_mask.rolling(self.long_window).sum()
        momentum = momentum.where(valid_count >= self.long_window * 0.8)
        return momentum


class ROETTMFactor(BaseFactor):
    """
    Profitability factor ROE(TTM)
    Reconstruct TTM from cumulative weighted ROE: current YTD + prior annual - prior same-period YTD.
    Aligned by financial-report disclosure lag (no lookahead bias).
    """
    
    name = "roe"
    
    # report month -> disclosure lag config key
    QUARTER_KEY = {3: "q1", 6: "q2", 9: "q3", 12: "q4"}
    
    def __init__(self, disclosure_lag: Dict[str, int] = None):
        self.disclosure_lag = disclosure_lag or {"q1": 30, "q2": 62, "q3": 31, "q4": 121}
    
    def _ttm_series(self, stock_df: pd.DataFrame) -> pd.DataFrame:
        """
        ROE TTM reconstruction for a single stock
        
        Args:
            stock_df: index=report_date, column=roe (cumulative weighted ROE)
            
        Returns:
            DataFrame with columns: report_date, available_date, roe_ttm
        """
        df = stock_df.sort_index()
        records = []
        
        for report_date, roe in df["roe"].items():
            year, month = report_date.year, report_date.month
            
            if month == 12:
                # annual report is already TTM
                roe_ttm = roe
            else:
                prev_annual = df.loc[df.index == pd.Timestamp(year - 1, 12, 31), "roe"]
                prev_same = df.loc[df.index == pd.Timestamp(year - 1, month, 1) + pd.offsets.MonthEnd(0), "roe"]
                if prev_annual.empty or prev_same.empty:
                    continue
                roe_ttm = roe + prev_annual.iloc[0] - prev_same.iloc[0]
            
            lag_days = self.disclosure_lag[self.QUARTER_KEY[month]]
            available_date = report_date + pd.Timedelta(days=lag_days)
            records.append({
                "report_date": report_date,
                "available_date": available_date,
                "roe_ttm": roe_ttm,
            })
        
        return pd.DataFrame(records)
    
    def compute(self, financial: pd.DataFrame, trading_dates: pd.DatetimeIndex) -> pd.DataFrame:
        """
        Args:
            financial: financial long table (stock_code, report_date) multi-index, column=roe
            trading_dates: trading-day series used to build the daily panel
            
        Returns:
            daily ROE(TTM) wide table (aligned by disclosure date, then forward-filled)
        """
        financial = financial.reset_index()
        ttm_list = []
        
        for code, stock_df in financial.groupby("stock_code"):
            ttm = self._ttm_series(stock_df.set_index("report_date"))
            if ttm.empty:
                continue
            ttm["stock_code"] = code
            ttm_list.append(ttm)
        
        if not ttm_list:
            return pd.DataFrame(index=trading_dates)
        
        ttm_all = pd.concat(ttm_list, ignore_index=True)
        ttm_all = ttm_all.sort_values(["available_date", "report_date"])
        
        # Drop records that are disclosed later but have an older report period
        # (e.g. annual report disclosed after next year's Q1 report),
        # preventing stale data from overwriting newer values during forward fill
        ttm_all = ttm_all[ttm_all["report_date"] == ttm_all["report_date"].cummax()]
        
        # Expand to a daily panel by disclosure date, align to trading days and forward-fill
        pivot = ttm_all.pivot_table(
            index="available_date", columns="stock_code",
            values="roe_ttm", aggfunc="last"
        )
        panel = pivot.reindex(trading_dates.union(pivot.index)).ffill()
        panel = panel.reindex(trading_dates)
        return panel


class MultiFactorCombiner:
    """Multi-factor combiner: cross-section winsorization + normalization, then weighted combination"""
    
    def __init__(self, weights: Dict[str, float]):
        self.weights = weights
    
    def combine(self, factor_panels: Dict[str, pd.DataFrame],
                rebalance_dates: List[pd.Timestamp]) -> pd.DataFrame:
        """
        Combine multi-factor scores on rebalance-date cross sections
        
        Args:
            factor_panels: {factor name: daily factor wide table} (raw values, unnormalized)
            rebalance_dates: list of rebalance dates
            
        Returns:
            combined score wide table (index=rebalance date, columns=stock code)
        """
        scores = None
        
        for name, weight in self.weights.items():
            if name not in factor_panels:
                print(f"Warning: factor {name} missing, skipped")
                continue
            
            panel = factor_panels[name]
            # take cross section at rebalance date (forward-fill missing dates)
            panel = panel.reindex(panel.index.union(pd.DatetimeIndex(rebalance_dates)))
            panel = panel.ffill().loc[rebalance_dates]
            
            # cross-section winsorization + normalization
            normalized = panel.apply(preprocess_cross_section, axis=1)
            
            if scores is None:
                scores = normalized * weight
            else:
                scores = scores.add(normalized * weight, fill_value=0)
        
        return scores
