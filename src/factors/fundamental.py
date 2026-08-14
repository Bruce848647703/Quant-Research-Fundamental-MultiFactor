"""
基本面多因子模块
EP / BP / ROE(TTM) / Size / Momentum 五因子
"""
import numpy as np
import pandas as pd
from typing import Dict, List

from .base import BaseFactor, preprocess_cross_section


class EPFactor(BaseFactor):
    """盈利收益率因子 EP = 1 / PE(TTM)，仅对PE>0有效"""
    
    name = "ep"
    
    def compute(self, pe_wide: pd.DataFrame) -> pd.DataFrame:
        pe = pe_wide.where(pe_wide > 0)
        return 1.0 / pe


class BPFactor(BaseFactor):
    """账面市值比因子 BP = 1 / PB，仅对PB>0有效"""
    
    name = "bp"
    
    def compute(self, pb_wide: pd.DataFrame) -> pd.DataFrame:
        pb = pb_wide.where(pb_wide > 0)
        return 1.0 / pb


class SizeFactor(BaseFactor):
    """小市值因子 Size = -ln(总市值)，市值越小得分越高"""
    
    name = "size"
    
    def compute(self, mv_wide: pd.DataFrame) -> pd.DataFrame:
        mv = mv_wide.where(mv_wide > 0)
        return -np.log(mv)


class MomentumFactor(BaseFactor):
    """动量因子：过去long_window日收益，剔除最近skip日（12-1动量）"""
    
    name = "momentum"
    
    def __init__(self, long_window: int = 252, skip_window: int = 21):
        self.long_window = long_window
        self.skip_window = skip_window
    
    def compute(self, ret_wide: pd.DataFrame) -> pd.DataFrame:
        logret = np.log1p(ret_wide.fillna(0.0))
        # 原始数据缺失标记（用于有效性过滤）
        valid_mask = ret_wide.notna().astype(float)
        
        cum = logret.cumsum()
        momentum = cum.shift(self.skip_window) - cum.shift(self.long_window)
        
        # 窗口内有效交易日不足80%的视为无效
        valid_count = valid_mask.rolling(self.long_window).sum()
        momentum = momentum.where(valid_count >= self.long_window * 0.8)
        return momentum


class ROETTMFactor(BaseFactor):
    """
    盈利能力因子 ROE(TTM)
    基于累计加权ROE还原TTM: 本期累计 + 上年年报 - 上年同期累计
    并按财报披露滞后做时点对齐（避免未来函数）
    """
    
    name = "roe"
    
    # 报告月份 -> 披露滞后配置键
    QUARTER_KEY = {3: "q1", 6: "q2", 9: "q3", 12: "q4"}
    
    def __init__(self, disclosure_lag: Dict[str, int] = None):
        self.disclosure_lag = disclosure_lag or {"q1": 30, "q2": 62, "q3": 31, "q4": 121}
    
    def _ttm_series(self, stock_df: pd.DataFrame) -> pd.DataFrame:
        """
        单只股票的ROE TTM还原
        
        Args:
            stock_df: index=report_date, column=roe（累计加权ROE）
            
        Returns:
            DataFrame with columns: report_date, available_date, roe_ttm
        """
        df = stock_df.sort_index()
        records = []
        
        for report_date, roe in df["roe"].items():
            year, month = report_date.year, report_date.month
            
            if month == 12:
                # 年报即为TTM
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
            financial: 财务长表 (stock_code, report_date) 多级索引, column=roe
            trading_dates: 交易日序列，用于生成日频面板
            
        Returns:
            日频ROE(TTM)宽表（按披露日期对齐后前向填充）
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
        
        # 剔除"披露时间更晚但报告期更旧"的记录（如年报披露晚于次年一季报的情形），
        # 避免前向填充时被旧报告期数据覆盖
        ttm_all = ttm_all[ttm_all["report_date"] == ttm_all["report_date"].cummax()]
        
        # 按披露日期展开为日频面板，再对齐交易日并前向填充
        pivot = ttm_all.pivot_table(
            index="available_date", columns="stock_code",
            values="roe_ttm", aggfunc="last"
        )
        panel = pivot.reindex(trading_dates.union(pivot.index)).ffill()
        panel = panel.reindex(trading_dates)
        return panel


class MultiFactorCombiner:
    """多因子合成器：截面去极值+标准化后按权重加权合成"""
    
    def __init__(self, weights: Dict[str, float]):
        self.weights = weights
    
    def combine(self, factor_panels: Dict[str, pd.DataFrame],
                rebalance_dates: List[pd.Timestamp]) -> pd.DataFrame:
        """
        在调仓日截面上合成多因子得分
        
        Args:
            factor_panels: {因子名: 因子日频宽表}（原始值，未标准化）
            rebalance_dates: 调仓日期列表
            
        Returns:
            合成得分宽表（index=调仓日期, columns=股票代码）
        """
        scores = None
        
        for name, weight in self.weights.items():
            if name not in factor_panels:
                print(f"警告: 因子 {name} 缺失，跳过")
                continue
            
            panel = factor_panels[name]
            # 取调仓日截面（缺失日期向前取最近值）
            panel = panel.reindex(panel.index.union(pd.DatetimeIndex(rebalance_dates)))
            panel = panel.ffill().loc[rebalance_dates]
            
            # 截面去极值 + 标准化
            normalized = panel.apply(preprocess_cross_section, axis=1)
            
            if scores is None:
                scores = normalized * weight
            else:
                scores = scores.add(normalized * weight, fill_value=0)
        
        return scores
