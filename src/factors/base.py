"""
Factor preprocessing utilities
MAD winsorization / zscore normalization / cross-section processing
"""
import numpy as np
import pandas as pd


def winsorize_mad(series: pd.Series, n: float = 5.0) -> pd.Series:
    """
    MAD winsorization
    
    Args:
        series: factor cross-section values
        n: MAD multiplier threshold
        
    Returns:
        winsorized series
    """
    median = series.median()
    mad = (series - median).abs().median()
    upper = median + n * 1.4826 * mad
    lower = median - n * 1.4826 * mad
    return series.clip(lower=lower, upper=upper)


def zscore(series: pd.Series) -> pd.Series:
    """
    zscore normalization
    
    Args:
        series: factor cross-section values
        
    Returns:
        normalized series (mean 0, std 1)
    """
    std = series.std()
    if std == 0 or np.isnan(std):
        return series * 0.0
    return (series - series.mean()) / std


def preprocess_cross_section(series: pd.Series, n_mad: float = 5.0) -> pd.Series:
    """Cross-section winsorization + normalization"""
    return zscore(winsorize_mad(series, n=n_mad))


def preprocess_panel(panel: pd.DataFrame, n_mad: float = 5.0) -> pd.DataFrame:
    """
    Apply winsorization + normalization row-wise (per cross-section date)
    
    Args:
        panel: wide table (index=date, columns=stock code)
        n_mad: MAD multiplier threshold
        
    Returns:
        processed wide table
    """
    return panel.apply(preprocess_cross_section, axis=1, n_mad=n_mad)


class BaseFactor:
    """Base class for factors"""
    
    name = "base"
    direction = 1  # 1: higher factor value is better; -1: lower is better
    
    def compute(self, **kwargs) -> pd.DataFrame:
        """Compute the factor wide table (index=date, columns=stock code), implemented by subclasses"""
        raise NotImplementedError
    
    def compute_normalized(self, **kwargs) -> pd.DataFrame:
        """Compute the factor and apply cross-section normalization"""
        return preprocess_panel(self.compute(**kwargs))
