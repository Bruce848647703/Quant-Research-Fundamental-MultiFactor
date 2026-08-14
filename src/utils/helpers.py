"""
工具函数模块
"""
import os
import yaml
import numpy as np
import pandas as pd
from typing import Dict


def load_config(config_path: str = None) -> Dict:
    """
    加载配置文件
    
    Args:
        config_path: 配置文件路径，默认为 config/config.yaml
        
    Returns:
        配置字典
    """
    if config_path is None:
        # 默认路径
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        config_path = os.path.join(base_dir, "config", "config.yaml")
    
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    
    return config


def ensure_dir(dir_path: str):
    """确保目录存在"""
    os.makedirs(dir_path, exist_ok=True)


def format_pct(value: float, decimals: int = 2) -> str:
    """格式化百分比"""
    return f"{value * 100:.{decimals}f}%"


def format_number(value: float, decimals: int = 2) -> str:
    """格式化数字（千分位）"""
    return f"{value:,.{decimals}f}"


def performance_metrics(nav: pd.Series, periods_per_year: int = 12) -> Dict:
    """
    计算净值序列的绩效指标
    
    Args:
        nav: 净值序列（期初为1）
        periods_per_year: 每年期数（月度=12，日度=252）
        
    Returns:
        指标字典: total_return / annual_return / annual_vol / sharpe / max_drawdown / calmar
    """
    nav = nav.dropna()
    returns = nav.pct_change().dropna()
    
    n_periods = len(nav)
    years = n_periods / periods_per_year
    
    total_return = nav.iloc[-1] / nav.iloc[0] - 1
    annual_return = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0.0
    annual_vol = returns.std() * np.sqrt(periods_per_year)
    sharpe = annual_return / annual_vol if annual_vol > 0 else 0.0
    
    # 最大回撤
    cummax = nav.cummax()
    drawdown = nav / cummax - 1
    max_drawdown = drawdown.min()
    
    calmar = annual_return / abs(max_drawdown) if max_drawdown < 0 else 0.0
    
    return {
        "total_return": float(total_return),
        "annual_return": float(annual_return),
        "annual_vol": float(annual_vol),
        "sharpe": float(sharpe),
        "max_drawdown": float(max_drawdown),
        "calmar": float(calmar),
    }
