#!/usr/bin/env python3
"""
权重优化对比脚本（v2）
v1方法: 等权(基准) / ICIR动态加权 / 最大化组合ICIR / Ridge walk-forward / GBDT walk-forward
v2改进: EWMA-IC加权 / 正交化+EWMA / 均值-方差(因子多空收益) /
        ML集成(扩展窗口+向等权收缩) / 逆波动率组内配权
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
    rolling_icir_weights, rolling_max_icir_weights, ewma_icir_weights,
    orthogonalize_factors, factor_long_short_returns, rolling_mv_weights,
    dynamic_weight_scores, MLWalkForwardScorer
)
from src.factors.base import preprocess_cross_section
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
    
    # ---------------- v2 改进方法 ----------------
    
    # 6. EWMA-IC加权（指数衰减，近期信号权重更高）
    ewma_w = ewma_icir_weights(
        ic_table, halflife=wo_config.get("ewma_halflife", 6))
    scores_ewma = dynamic_weight_scores(factor_panels, ewma_w,
                                        rebalance_dates, factor_names)
    
    # 7. 正交化后再EWMA加权（消除size共线）
    print("因子正交化...")
    ortho_order = wo_config.get("ortho_order", factor_names)
    ortho_panels = orthogonalize_factors(factor_panels, ortho_order, rebalance_dates)
    ic_table_ortho = compute_rank_ic(ortho_panels, period_ret, rebalance_dates)
    ewma_ortho_w = ewma_icir_weights(
        ic_table_ortho, halflife=wo_config.get("ewma_halflife", 6))
    scores_ortho = dynamic_weight_scores(ortho_panels, ewma_ortho_w,
                                         rebalance_dates, ortho_order)
    
    # 8. 均值-方差加权（因子多空组合真实收益 + Ledoit-Wolf收缩协方差）
    print("构造因子多空收益...")
    ls_returns = factor_long_short_returns(factor_panels, period_ret, rebalance_dates)
    mv_w = rolling_mv_weights(
        ls_returns, window=wo_config.get("mv_window", 36),
        min_periods=wo_config.get("mv_min_periods", 24))
    scores_mv = dynamic_weight_scores(factor_panels, mv_w,
                                      rebalance_dates, factor_names)
    
    # 9. ML集成：扩展窗口Ridge+GBDT，截面标准化后平均，再向等权得分收缩降方差
    print("训练 ML集成（扩展窗口 Ridge + GBDT）...")
    ridge_exp = MLWalkForwardScorer(
        model_type="ridge", expanding=True, clip_targets=True,
        min_train_periods=wo_config.get("ml_min_train_periods", 24),
        factor_names=factor_names)
    gbdt_exp = MLWalkForwardScorer(
        model_type="gbdt", expanding=True, clip_targets=True,
        min_train_periods=wo_config.get("ml_min_train_periods", 24),
        factor_names=factor_names)
    pred_ridge = ridge_exp.fit_predict(factor_panels, period_ret, rebalance_dates)
    pred_gbdt = gbdt_exp.fit_predict(factor_panels, period_ret, rebalance_dates)
    
    shrinkage = wo_config.get("ml_shrinkage", 0.3)
    scores_ensemble = []
    for t in pred_ridge.index:
        r_z = preprocess_cross_section(pred_ridge.loc[t])
        g_z = preprocess_cross_section(pred_gbdt.loc[t])
        ml_z = (r_z + g_z) / 2
        if ml_z.notna().any():
            eq_z = preprocess_cross_section(scores_equal.loc[t])
            row = (1 - shrinkage) * ml_z + shrinkage * eq_z
        else:
            row = ml_z
        scores_ensemble.append(pd.DataFrame([row], index=[t]))
    scores_ensemble = pd.concat(scores_ensemble)
    
    # 10. 逆波动率组内配权（分别叠加在等权与ML集成上）
    vol_panel = panel["ret"].rolling(wo_config.get("inv_vol_window", 63)).std()
    
    # ---------------- 回测对比 ----------------
    print("\n运行回测...")
    methods = {
        "等权(基准)": (scores_equal, None),
        "ICIR动态加权": (scores_icir, None),
        "最大化组合ICIR": (scores_maxicir, None),
        "Ridge walk-forward": (scores_ridge, None),
        "GBDT walk-forward": (scores_gbdt, None),
        "EWMA-IC加权": (scores_ewma, None),
        "正交化+EWMA": (scores_ortho, None),
        "均值-方差加权": (scores_mv, None),
        "ML集成(收缩)": (scores_ensemble, None),
        "等权+逆波动率配权": (scores_equal, vol_panel),
        "ML集成+逆波动率": (scores_ensemble, vol_panel),
    }
    
    navs, metrics = {}, {}
    for name, (scores, vol) in methods.items():
        result = engine.run_portfolio(scores, period_ret, rebalance_dates,
                                      inv_vol_panel=vol)
        navs[name] = result["nav"]
        m = performance_metrics(result["nav"], periods_per_year=12)
        m["avg_turnover"] = float(result["turnover"].mean())
        metrics[name] = m
    
    # 机器学习方法只统计训练期结束后的样本外表现
    ml_start = pd.Timestamp(rebalance_dates[wo_config.get("ml_min_train_periods", 24)])
    for name in ["Ridge walk-forward", "GBDT walk-forward", "ML集成(收缩)", "ML集成+逆波动率"]:
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
        "- ICIR/EWMA/均值-方差/最大化ICIR 仅使用调仓日之前的历史IC/收益",
        "- 正交化+EWMA: 因子先对size等前置因子回归取残差，消除共线后再EWMA加权",
        "- Ridge/GBDT/ML集成 为walk-forward: ML集成用扩展窗口+向等权得分收缩",
        "- 逆波动率配权: Top-N内按过去63日波动率倒数配权",
        "- oos_* 为机器学习方法训练期结束后的样本外指标",
        "=" * 60,
    ]
    report_text = "\n".join(lines)
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_text)
    print(f"\n报告已保存至: {report_path}")
    
    # 权重序列样本（展示各统计方法的最新权重）
    weights_path = os.path.join(results_config["report_dir"], "optimized_weights_latest.csv")
    latest_weights = pd.DataFrame({
        "icir_weight": icir_w.dropna(how="all").iloc[-1],
        "ewma_weight": ewma_w.dropna(how="all").iloc[-1],
        "max_icir_weight": maxicir_w.dropna(how="all").iloc[-1],
        "mv_weight": mv_w.dropna(how="all").iloc[-1].reindex(factor_names),
        "equal_weight": 1.0 / len(factor_names),
    })
    latest_weights.to_csv(weights_path, encoding="utf-8-sig")
    print(f"最新优化权重已保存至: {weights_path}")
    print("\n权重优化对比完成!")


if __name__ == "__main__":
    main()
