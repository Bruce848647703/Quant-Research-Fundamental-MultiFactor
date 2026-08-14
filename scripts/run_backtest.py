#!/usr/bin/env python3
"""
Backtest runner script
One-click run of the fundamental multi-factor monthly rotation backtest
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
    Compute daily wide tables for all factors
    
    Args:
        config: config dict
        panel: dict of valuation wide tables (close/ret/pe_ttm/pb/total_mv)
        financial: financial long table
        
    Returns:
        {factor name: factor wide table}
    """
    factor_config = config["factor"]
    trading_dates = panel["ret"].index
    
    print("Computing EP factor...")
    ep = EPFactor().compute(panel["pe_ttm"])
    print("Computing BP factor...")
    bp = BPFactor().compute(panel["pb"])
    print("Computing Size factor...")
    size = SizeFactor().compute(panel["total_mv"])
    print("Computing Momentum factor...")
    momentum = MomentumFactor(
        long_window=factor_config["momentum_long_window"],
        skip_window=factor_config["momentum_skip_window"]
    ).compute(panel["ret"])
    print("Computing ROE(TTM) factor...")
    roe = ROETTMFactor(
        disclosure_lag=factor_config["disclosure_lag"]
    ).compute(financial, trading_dates)
    
    return {"ep": ep, "bp": bp, "size": size, "momentum": momentum, "roe": roe}


def plot_results(analysis: Dict, plot_dir: str):
    """Generate and save backtest plots"""
    ensure_dir(plot_dir)
    
    # 1. portfolio vs benchmark NAV curves
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
    
    # 2. quantile-group NAV curves
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
    
    # 3. monthly factor IC bar charts
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
    
    # 4. cumulative factor IC curves
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
    
    print(f"\nPlots saved to: {plot_dir}")


def main():
    """Main function"""
    print("Fundamental Multi-Factor Backtest System")
    print("=" * 60)
    
    # load config
    config = load_config()
    data_config = config["data"]
    results_config = config["results"]
    
    print("Config loaded")
    print(f"  Universe: {data_config['universe']}")
    print(f"  Backtest range: {data_config['start_date']} ~ {data_config['end_date']}")
    print(f"  Factor: {config['factor']['name']}")
    print(f"  Rebalance frequency: {config['backtest']['rebalance_freq']}")
    print()
    
    # ---------------- data preparation ----------------
    print("=" * 60)
    print("Data Preparation")
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
    print(f"Valuation data: {valuation.shape}, financial data: {financial.shape}")
    
    panel = loader.prepare_panel_data(valuation)
    print(f"Panel data: {panel['ret'].shape[0]} trading days, {panel['ret'].shape[1]} stocks")
    
    # benchmark index
    bench_path = os.path.join(data_config["raw_dir"],
                              f"benchmark_{data_config['benchmark']}.parquet")
    benchmark = pd.read_parquet(bench_path)
    
    # ---------------- factor computation ----------------
    print("\n" + "=" * 60)
    print("Factor Computation")
    print("=" * 60)
    
    factor_panels = compute_factor_panels(config, panel, financial)
    
    # rebalance dates (last trading day of each month)
    engine = MultiFactorEngine(config)
    rebalance_dates = engine.get_rebalance_dates(
        panel["ret"].index,
        start_date=data_config["start_date"],
        end_date=data_config["end_date"]
    )
    print(f"\nNumber of rebalance dates: {len(rebalance_dates)} "
          f"({pd.Timestamp(rebalance_dates[0]).date()} ~ {pd.Timestamp(rebalance_dates[-1]).date()})")
    
    # multi-factor combination
    combiner = MultiFactorCombiner(config["factor"]["weights"])
    scores = combiner.combine(factor_panels, rebalance_dates)
    print(f"Combined score matrix: {scores.shape}")
    
    # ---------------- backtest ----------------
    print("\n" + "=" * 60)
    print("Backtest")
    print("=" * 60)
    
    period_ret = engine.compute_period_returns(panel["ret"], rebalance_dates)
    
    # Top-N portfolio backtest
    portfolio = engine.run_portfolio(scores, period_ret, rebalance_dates)
    portfolio_nav = portfolio["nav"]
    print(f"\nLong portfolio final NAV: {portfolio_nav.iloc[-1]:.4f}")
    
    # quantile-group backtest
    group_navs = engine.run_groups(scores, period_ret, rebalance_dates)
    
    # benchmark NAV (aligned to rebalance dates)
    bench_close = benchmark["close"].reindex(
        benchmark.index.union(pd.DatetimeIndex(rebalance_dates))
    ).ffill().loc[rebalance_dates]
    benchmark_nav = bench_close / bench_close.iloc[0]
    
    # IC analysis
    print("\nComputing factor Rank IC...")
    ic_table = compute_rank_ic(factor_panels, period_ret, rebalance_dates)
    ic_stat = ic_summary(ic_table)
    print(ic_stat.round(4))
    
    # ---------------- output ----------------
    print("\n" + "=" * 60)
    print("Results Output")
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
    
    # text report
    ensure_dir(results_config["report_dir"])
    generate_report(analysis, results_config["report_dir"])
    
    # YAML results (scalar metrics only)
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
    print(f"\nAnalysis results saved to: {results_path}")
    
    # plots
    plot_results(analysis, results_config["plot_dir"])
    
    print("\nBacktest complete!")


if __name__ == "__main__":
    main()
