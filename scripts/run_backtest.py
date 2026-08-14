#!/usr/bin/env python3
"""
回测运行脚本
一键运行基本面多因子月度轮动回测
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import yaml
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from typing import Dict

from src.data.data_loader import DataLoader
from src.factors.fundamental import (
    EPFactor, BPFactor, SizeFactor, MomentumFactor, ROETTMFactor,
    MultiFactorCombiner
)
from src.backtest.engine import MultiFactorEngine
from src.backtest.analyzers import compute_rank_ic, ic_summary, generate_report
from src.utils.helpers import load_config, ensure_dir, performance_metrics


def compute_factor_panels(config: Dict, panel: dict, financial: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """
    计算各因子日频宽表
    
    Args:
        config: 配置字典
        panel: 估值宽表字典 (close/ret/pe_ttm/pb/total_mv)
        financial: 财务长表
        
    Returns:
        {因子名: 因子宽表}
    """
    factor_config = config["factor"]
    trading_dates = panel["ret"].index
    
    print("计算 EP 因子...")
    ep = EPFactor().compute(panel["pe_ttm"])
    print("计算 BP 因子...")
    bp = BPFactor().compute(panel["pb"])
    print("计算 Size 因子...")
    size = SizeFactor().compute(panel["total_mv"])
    print("计算 Momentum 因子...")
    momentum = MomentumFactor(
        long_window=factor_config["momentum_long_window"],
        skip_window=factor_config["momentum_skip_window"]
    ).compute(panel["ret"])
    print("计算 ROE(TTM) 因子...")
    roe = ROETTMFactor(
        disclosure_lag=factor_config["disclosure_lag"]
    ).compute(financial, trading_dates)
    
    return {"ep": ep, "bp": bp, "roe": roe, "size": size, "momentum": momentum}


def plot_results(analysis: Dict, plot_dir: str):
    """绘制并保存回测图表"""
    ensure_dir(plot_dir)
    
    # 1. 组合 vs 基准净值曲线
    fig, ax = plt.subplots(figsize=(12, 6))
    analysis["portfolio_nav"].plot(ax=ax, label="Multi-Factor Portfolio")
    analysis["benchmark_nav"].plot(ax=ax, label="CSI 300 Benchmark")
    ax.set_title("Multi-Factor Portfolio vs CSI 300 (Monthly Rebalance)")
    ax.set_xlabel("Date")
    ax.set_ylabel("NAV")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(plot_dir, "nav_curve.png"), dpi=150)
    plt.close(fig)
    
    # 2. 分层回测净值曲线
    fig, ax = plt.subplots(figsize=(12, 6))
    for g, nav in analysis["group_navs"].items():
        label = f"{g} (top)" if g == "group_1" else (
            f"{g} (bottom)" if g == f"group_{len(analysis['group_navs'])}" else g)
        nav.plot(ax=ax, label=label)
    ax.set_title("Quantile Group NAV (Group 1 = Highest Score)")
    ax.set_xlabel("Date")
    ax.set_ylabel("NAV")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(plot_dir, "quantile_nav.png"), dpi=150)
    plt.close(fig)
    
    # 3. 因子月度IC柱状图
    ic_table = analysis["ic_table"]
    fig, axes = plt.subplots(1, len(ic_table.columns), figsize=(4 * len(ic_table.columns), 4),
                             sharey=True)
    for ax, name in zip(axes, ic_table.columns):
        ic = ic_table[name].dropna()
        ic.plot(kind="bar", ax=ax, width=0.8,
                color=np.where(ic.values >= 0, "#c0392b", "#27ae60"))
        ax.axhline(ic.mean(), color="black", linestyle="--", linewidth=1)
        ax.set_title(f"{name} (IC={ic.mean():.3f})")
        ax.tick_params(axis="x", rotation=45, labelsize=7)
    fig.suptitle("Monthly Rank IC by Factor")
    fig.tight_layout()
    fig.savefig(os.path.join(plot_dir, "factor_ic_monthly.png"), dpi=150)
    plt.close(fig)
    
    # 4. 因子累计IC曲线
    fig, ax = plt.subplots(figsize=(12, 6))
    ic_table.cumsum().plot(ax=ax)
    ax.set_title("Cumulative Rank IC by Factor")
    ax.set_xlabel("Date")
    ax.set_ylabel("Cumulative IC")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(plot_dir, "factor_ic_cumsum.png"), dpi=150)
    plt.close(fig)
    
    print(f"\n图表已保存至: {plot_dir}")


def main():
    """主函数"""
    print("基本面多因子回测系统")
    print("=" * 60)
    
    # 加载配置
    config = load_config()
    data_config = config["data"]
    results_config = config["results"]
    
    print("配置加载完成")
    print(f"  股票池: {data_config['universe']}")
    print(f"  回测范围: {data_config['start_date']} ~ {data_config['end_date']}")
    print(f"  因子: {config['factor']['name']}")
    print(f"  调仓频率: {config['backtest']['rebalance_freq']}")
    print()
    
    # ---------------- 数据准备 ----------------
    print("=" * 60)
    print("数据准备阶段")
    print("=" * 60)
    
    loader = DataLoader(
        raw_dir=data_config["raw_dir"],
        processed_dir=data_config["processed_dir"]
    )
    
    data = loader.load_data(
        start_date=data_config["start_date"],
        end_date=data_config["end_date"]
    )
    valuation, financial = data["valuation"], data["financial"]
    print(f"估值数据: {valuation.shape}, 财务数据: {financial.shape}")
    
    panel = loader.prepare_panel_data(valuation)
    print(f"面板数据: {panel['ret'].shape[0]} 个交易日, {panel['ret'].shape[1]} 只股票")
    
    # 基准指数
    bench_path = os.path.join(data_config["raw_dir"],
                              f"benchmark_{data_config['benchmark']}.parquet")
    benchmark = pd.read_parquet(bench_path)
    
    # ---------------- 因子计算 ----------------
    print("\n" + "=" * 60)
    print("因子计算阶段")
    print("=" * 60)
    
    factor_panels = compute_factor_panels(config, panel, financial)
    
    # 调仓日（每月最后一个交易日）
    engine = MultiFactorEngine(config)
    rebalance_dates = engine.get_rebalance_dates(
        panel["ret"].index,
        start_date=data_config["start_date"],
        end_date=data_config["end_date"]
    )
    print(f"\n调仓日数量: {len(rebalance_dates)} "
          f"({pd.Timestamp(rebalance_dates[0]).date()} ~ {pd.Timestamp(rebalance_dates[-1]).date()})")
    
    # 多因子合成
    combiner = MultiFactorCombiner(config["factor"]["weights"])
    scores = combiner.combine(factor_panels, rebalance_dates)
    print(f"合成得分矩阵: {scores.shape}")
    
    # ---------------- 回测阶段 ----------------
    print("\n" + "=" * 60)
    print("回测阶段")
    print("=" * 60)
    
    period_ret = engine.compute_period_returns(panel["ret"], rebalance_dates)
    
    # Top-N组合回测
    portfolio = engine.run_portfolio(scores, period_ret, rebalance_dates)
    portfolio_nav = portfolio["nav"]
    print(f"\n多头组合期末净值: {portfolio_nav.iloc[-1]:.4f}")
    
    # 分层回测
    group_navs = engine.run_groups(scores, period_ret, rebalance_dates)
    
    # 基准净值（对齐调仓日）
    bench_close = benchmark["close"].reindex(
        benchmark.index.union(pd.DatetimeIndex(rebalance_dates))
    ).ffill().loc[rebalance_dates]
    benchmark_nav = bench_close / bench_close.iloc[0]
    
    # IC分析
    print("\n计算因子Rank IC...")
    ic_table = compute_rank_ic(factor_panels, period_ret, rebalance_dates)
    ic_stat = ic_summary(ic_table)
    print(ic_stat.round(4))
    
    # ---------------- 结果输出 ----------------
    print("\n" + "=" * 60)
    print("结果输出阶段")
    print("=" * 60)
    
    analysis = {
        "config": config,
        "date_range": f"{pd.Timestamp(rebalance_dates[0]).date()} ~ "
                      f"{pd.Timestamp(rebalance_dates[-1]).date()}",
        "portfolio_nav": portfolio_nav,
        "benchmark_nav": benchmark_nav,
        "group_navs": group_navs,
        "ic_table": ic_table,
        "ic_summary": ic_stat,
        "portfolio_metrics": performance_metrics(portfolio_nav, periods_per_year=12),
        "benchmark_metrics": performance_metrics(benchmark_nav, periods_per_year=12),
        "group_metrics": {g: performance_metrics(nav, periods_per_year=12)
                          for g, nav in group_navs.items()},
        "turnover_avg": float(portfolio["turnover"].mean()),
    }
    
    # 文本报告
    ensure_dir(results_config["report_dir"])
    generate_report(analysis, results_config["report_dir"])
    
    # YAML结果（仅标量指标）
    yaml_results = {
        "date_range": analysis["date_range"],
        "portfolio_metrics": analysis["portfolio_metrics"],
        "benchmark_metrics": analysis["benchmark_metrics"],
        "group_metrics": analysis["group_metrics"],
        "ic_summary": ic_stat.to_dict(orient="index"),
        "turnover_avg": analysis["turnover_avg"],
    }
    results_path = os.path.join(results_config["report_dir"], "analysis_results.yaml")
    with open(results_path, "w", encoding="utf-8") as f:
        yaml.dump(yaml_results, f, allow_unicode=True, default_flow_style=False)
    print(f"\n分析结果已保存至: {results_path}")
    
    # 图表
    plot_results(analysis, results_config["plot_dir"])
    
    print("\n回测完成!")


if __name__ == "__main__":
    main()
