"""
因子预处理基础工具
MAD去极值 / zscore标准化 / 截面处理
"""
import numpy as np
import pandas as pd


def winsorize_mad(series: pd.Series, n: float = 5.0) -> pd.Series:
    """
    MAD去极值
    
    Args:
        series: 因子截面值
        n: MAD倍数阈值
        
    Returns:
        去极值后的序列
    """
    median = series.median()
    mad = (series - median).abs().median()
    upper = median + n * 1.4826 * mad
    lower = median - n * 1.4826 * mad
    return series.clip(lower=lower, upper=upper)


def zscore(series: pd.Series) -> pd.Series:
    """
    zscore标准化
    
    Args:
        series: 因子截面值
        
    Returns:
        标准化后的序列（均值0，标准差1）
    """
    std = series.std()
    if std == 0 or np.isnan(std):
        return series * 0.0
    return (series - series.mean()) / std


def preprocess_cross_section(series: pd.Series, n_mad: float = 5.0) -> pd.Series:
    """截面去极值 + 标准化"""
    return zscore(winsorize_mad(series, n=n_mad))


def preprocess_panel(panel: pd.DataFrame, n_mad: float = 5.0) -> pd.DataFrame:
    """
    对宽表面板逐行（每个截面日期）做去极值+标准化
    
    Args:
        panel: 宽表（index=日期, columns=股票代码）
        n_mad: MAD倍数阈值
        
    Returns:
        处理后的宽表
    """
    return panel.apply(preprocess_cross_section, axis=1, n_mad=n_mad)


class BaseFactor:
    """因子基类"""
    
    name = "base"
    direction = 1  # 1: 因子值越大越好; -1: 越小越好
    
    def compute(self, **kwargs) -> pd.DataFrame:
        """计算因子宽表（index=日期, columns=股票代码），由子类实现"""
        raise NotImplementedError
    
    def compute_normalized(self, **kwargs) -> pd.DataFrame:
        """计算并做截面标准化"""
        return preprocess_panel(self.compute(**kwargs))
