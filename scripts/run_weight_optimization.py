#!/usr/bin/env python3
"""
权重优化对比脚本
对比 等权(基准) / ICIR动态加权 / 最大化组合ICIR / Ridge walk-forward / GBDT walk-forward
全部采用walk-forward方式，仅使用时点之前的信息
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))

import yaml
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from datetime import datetime

from src.data.data_loader import DataLoader
from src.factors.weight_optimizer import (
    rolling_icir_weights, rolling_max_icir_weights,
    dynamic_weight_scores, MLWalkForwardScorer
)
from src.backtest.engine import MultiFactorEngine
from src.backtest.analyzers import compute_rank_ic
from src.utils.helpers import load_config, ensure_dir, performance_metrics
from run_backtest import compute_factor_panels


def main():
    """主函数"""
    print("因子权重优化对比")
    print("=" * 60)
    
    config = load_config()
    data_config = config["data"]
    results_config = config["results"]
    wo_config = config.get("weight_optimization", {})
    
    # ---------------- 数据与因子 ----------------
    loader = DataLoader(
        raw_dir=data_config["raw_dir"],
        processed_dir=data_config["processed_dir"]
    )
    data = loader.load_data(data_config["start_date"], data_config["end_date"])
    valuation, financial = data["valuation"], data["financial"]
    panel = loader.prepare_panel_data(valuation)
    
    factor_panels = compute_factor_panels(config, panel, financial)
    factor_names = list(config["factor"]["weights"].keys())
    
    engine = MultiFactorEngine(config)
    rebalance_dates = engine.get_rebalance_dates(
        panel["ret"].index,
        start_date=data_config["start_date"],
        end_date=data_config["end_date"]
    )
    period_ret = engine.compute_period_returns(panel["ret"], rebalance_dates)
    
    # ---------------- IC序列 ----------------
    ic_table = compute_rank_ic(factor_panels, period_ret, rebalance_dates)
    
    # ---------------- 各方法打分 ----------------
    print("\n构建各权重方案的得分...")
    
    # 1. 等权（基准，与主回测一致）
    from src.factors.fundamental import MultiFactorCombiner
    scores_equal = MultiFactorCombiner(config["factor"]["weights"]).combine(
        factor_panels, rebalance_dates)
    
    # 2. ICIR动态加权
    icir_w = rolling_icir_weights(
        ic_table, window=wo_config.get("icir_window", 12))
    scores_icir = dynamic_weight_scores(factor_panels, icir_w,
                                        rebalance_dates, factor_names)
    
    # 3. 最大化组合ICIR（滚动窗口约束优化）
    maxicir_w = rolling_max_icir_weights(
        ic_table, window=wo_config.get("max_icir_window", 36),
        min_periods=wo_config.get("max_icir_min_periods", 24))
    scores_maxicir = dynamic_weight_scores(factor_panels, maxicir_w,
                                           rebalance_dates, factor_names)
    
    # 4. Ridge回归 walk-forward
    print("训练 Ridge walk-forward...")
    ridge = MLWalkForwardScorer(
        model_type="ridge",
        train_window=wo_config.get("ml_train_window", 36),
        min_train_periods=wo_config.get("ml_min_train_periods", 24),
        factor_names=factor_names
    )
    scores_ridge = ridge.fit_predict(factor_panels, period_ret, rebalance_dates)
    
    # 5. 直方图GBDT walk-forward
    print("训练 GBDT walk-forward...")
    gbdt = MLWalkForwardScorer(
        model_type="gbdt",
        train_window=wo_config.get("ml_train_window", 36),
        min_train_periods=wo_config.get("ml_min_train_periods", 24),
        factor_names=factor_names
    )
    scores_gbdt = gbdt.fit_predict(factor_panels, period_ret, rebalance_dates)
    
    # ---------------- 回测对比 ----------------
    print("\n运行回测...")
    methods = {
        "等权(基准)": scores_equal,
        "ICIR动态加权": scores_icir,
        "最大化组合ICIR": scores_maxicir,
        "Ridge walk-forward": scores_ridge,
        "GBDT walk-forward": scores_gbdt,
    }
    
    navs, metrics = {}, {}
    for name, scores in methods.items():
        result = engine.run_portfolio(scores, period_ret, rebalance_dates)
        navs[name] = result["nav"]
        m = performance_metrics(result["nav"], periods_per_year=12)
        m["avg_turnover"] = float(result["turnover"].mean())
        metrics[name] = m
    
    # 机器学习方法只统计训练期结束后的样本外表现
    ml_start = pd.Timestamp(rebalance_dates[wo_config.get("ml_min_train_periods", 24)])
    for name in ["Ridge walk-forward", "GBDT walk-forward"]:
        nav_oos = navs[name][navs[name].index >= ml_start]
        if len(nav_oos) > 12:
            nav_oos = nav_oos / nav_oos.iloc[0]
            mo = performance_metrics(nav_oos, periods_per_year=12)
            metrics[name]["oos_annual_return"] = mo["annual_return"]
            metrics[name]["oos_sharpe"] = mo["sharpe"]
            metrics[name]["oos_max_drawdown"] = mo["max_drawdown"]
    
    # ---------------- 结果输出 ----------------
    summary = pd.DataFrame(metrics).T
    summary = summary[["annual_return", "sharpe", "max_drawdown", "annual_vol",
                       "avg_turnover"] + [c for c in summary.columns if c.startswith("oos_")]]
    print("\n" + "=" * 60)
    print("【权重优化对比】(月度调仓, Top 30, 全区间)")
    print("=" * 60)
    print(summary.round(4).to_string())
    
    ensure_dir(results_config["report_dir"])
    ensure_dir(results_config["plot_dir"])
    
    # 净值对比图
    fig, ax = plt.subplots(figsize=(12, 6))
    for name, nav in navs.items():
        nav.plot(ax=ax, label=name)
    ax.set_title("Weight Optimization Comparison (Monthly Rebalance, Top 30)")
    ax.set_xlabel("Date")
    ax.set_ylabel("NAV")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(results_config["plot_dir"], "weight_optimization_nav.png"), dpi=150)
    plt.close(fig)
    
    # 文本报告
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = os.path.join(results_config["report_dir"],
                               f"weight_optimization_report_{timestamp}.txt")
    lines = [
        "=" * 60,
        "因子权重优化对比报告",
        "=" * 60,
        f"生成时间: {timestamp}",
        f"回测区间: {pd.Timestamp(rebalance_dates[0]).date()} ~ {pd.Timestamp(rebalance_dates[-1]).date()}",
        f"方法: {', '.join(methods.keys())}",
        "",
        summary.round(4).to_string(),
        "",
        "说明:",
        "- ICIR动态加权/最大化组合ICIR 仅使用调仓日之前的历史IC",
        "- Ridge/GBDT 为walk-forward: 滚动训练窗口预测下期收益, 前期为训练期",
        "- oos_* 为机器学习方法训练期结束后的样本外指标",
        "=" * 60,
    ]
    report_text = "\n".join(lines)
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_text)
    print(f"\n报告已保存至: {report_path}")
    
    # 权重序列样本（展示ICIR加权与最大化ICIR的最新权重）
    weights_path = os.path.join(results_config["report_dir"], "optimized_weights_latest.csv")
    latest_weights = pd.DataFrame({
        "icir_weight": icir_w.dropna(how="all").iloc[-1],
        "max_icir_weight": maxicir_w.dropna(how="all").iloc[-1],
        "equal_weight": 1.0 / len(factor_names),
    })
    latest_weights.to_csv(weights_path, encoding="utf-8-sig")
    print(f"最新优化权重已保存至: {weights_path}")
    print("\n权重优化对比完成!")


if __name__ == "__main__":
    main()
