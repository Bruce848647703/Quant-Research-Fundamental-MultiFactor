#!/usr/bin/env python3
"""
Weight optimization comparison script (v2)
v1 methods: Equal Weight (baseline) / Rolling ICIR / Max Portfolio ICIR /
            Ridge walk-forward / GBDT walk-forward
v2 methods: EWMA-IC weighting / Orthogonalization+EWMA /
            Mean-Variance (factor long-short returns) /
            ML Ensemble (expanding window + shrinkage to equal weight) /
            Inverse-Volatility intra-portfolio weighting
All methods are walk-forward, using only information available at each point in time
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
    """Main function"""
    print("Factor Weight Optimization Comparison")
    print("=" * 60)
    
    config = load_config()
    data_config = config["data"]
    results_config = config["results"]
    wo_config = config.get("weight_optimization", {})
    
    # ---------------- data and factors ----------------
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
    
    # ---------------- IC series ----------------
    ic_table = compute_rank_ic(factor_panels, period_ret, rebalance_dates)
    
    # ---------------- scoring by each method ----------------
    print("\nBuilding scores for each weighting scheme...")
    
    # 1. Equal weight (baseline, consistent with the main backtest)
    from src.factors.fundamental import MultiFactorCombiner
    scores_equal = MultiFactorCombiner(config["factor"]["weights"]).combine(
        factor_panels, rebalance_dates)
    
    # 2. Rolling ICIR weighting
    icir_w = rolling_icir_weights(
        ic_table, window=wo_config.get("icir_window", 12))
    scores_icir = dynamic_weight_scores(factor_panels, icir_w,
                                        rebalance_dates, factor_names)
    
    # 3. Max portfolio ICIR (rolling-window constrained optimization)
    maxicir_w = rolling_max_icir_weights(
        ic_table, window=wo_config.get("max_icir_window", 36),
        min_periods=wo_config.get("max_icir_min_periods", 24))
    scores_maxicir = dynamic_weight_scores(factor_panels, maxicir_w,
                                           rebalance_dates, factor_names)
    
    # 4. Ridge regression walk-forward
    print("Training Ridge walk-forward...")
    ridge = MLWalkForwardScorer(
        model_type="ridge",
        train_window=wo_config.get("ml_train_window", 36),
        min_train_periods=wo_config.get("ml_min_train_periods", 24),
        factor_names=factor_names
    )
    scores_ridge = ridge.fit_predict(factor_panels, period_ret, rebalance_dates)
    
    # 5. Histogram GBDT walk-forward
    print("Training GBDT walk-forward...")
    gbdt = MLWalkForwardScorer(
        model_type="gbdt",
        train_window=wo_config.get("ml_train_window", 36),
        min_train_periods=wo_config.get("ml_min_train_periods", 24),
        factor_names=factor_names
    )
    scores_gbdt = gbdt.fit_predict(factor_panels, period_ret, rebalance_dates)
    
    # ---------------- v2 improved methods ----------------
    
    # 6. EWMA-IC weighting (exponential decay, more weight on recent signals)
    ewma_w = ewma_icir_weights(
        ic_table, halflife=wo_config.get("ewma_halflife", 6))
    scores_ewma = dynamic_weight_scores(factor_panels, ewma_w,
                                        rebalance_dates, factor_names)
    
    # 7. Orthogonalization followed by EWMA weighting (removes size collinearity)
    print("Orthogonalizing factors...")
    ortho_order = wo_config.get("ortho_order", factor_names)
    ortho_panels = orthogonalize_factors(factor_panels, ortho_order, rebalance_dates)
    ic_table_ortho = compute_rank_ic(ortho_panels, period_ret, rebalance_dates)
    ewma_ortho_w = ewma_icir_weights(
        ic_table_ortho, halflife=wo_config.get("ewma_halflife", 6))
    scores_ortho = dynamic_weight_scores(ortho_panels, ewma_ortho_w,
                                         rebalance_dates, ortho_order)
    
    # 8. Mean-variance weighting (factor long-short returns + Ledoit-Wolf covariance)
    print("Building factor long-short returns...")
    ls_returns = factor_long_short_returns(factor_panels, period_ret, rebalance_dates)
    mv_w = rolling_mv_weights(
        ls_returns, window=wo_config.get("mv_window", 36),
        min_periods=wo_config.get("mv_min_periods", 24))
    scores_mv = dynamic_weight_scores(factor_panels, mv_w,
                                      rebalance_dates, factor_names)
    
    # 9. ML ensemble: expanding-window Ridge+GBDT, averaged after cross-section
    #    normalization, then shrunk toward the equal-weight score to reduce variance
    print("Training ML ensemble (expanding-window Ridge + GBDT)...")
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
    
    # 10. Inverse-volatility intra-portfolio weighting (applied on top of
    #     equal weight and ML ensemble respectively)
    vol_panel = panel["ret"].rolling(wo_config.get("inv_vol_window", 63)).std()
    
    # ---------------- backtest comparison ----------------
    print("\nRunning backtests...")
    methods = {
        "Equal Weight (baseline)": (scores_equal, None),
        "Rolling ICIR": (scores_icir, None),
        "Max Portfolio ICIR": (scores_maxicir, None),
        "Ridge walk-forward": (scores_ridge, None),
        "GBDT walk-forward": (scores_gbdt, None),
        "EWMA-IC": (scores_ewma, None),
        "Ortho + EWMA": (scores_ortho, None),
        "Mean-Variance": (scores_mv, None),
        "ML Ensemble (shrinkage)": (scores_ensemble, None),
        "Equal Weight + InvVol": (scores_equal, vol_panel),
        "ML Ensemble + InvVol": (scores_ensemble, vol_panel),
    }
    
    navs, metrics = {}, {}
    for name, (scores, vol) in methods.items():
        result = engine.run_portfolio(scores, period_ret, rebalance_dates,
                                      inv_vol_panel=vol)
        navs[name] = result["nav"]
        m = performance_metrics(result["nav"], periods_per_year=12)
        m["avg_turnover"] = float(result["turnover"].mean())
        metrics[name] = m
    
    # for ML methods, also report out-of-sample performance after the training warm-up
    ml_start = pd.Timestamp(rebalance_dates[wo_config.get("ml_min_train_periods", 24)])
    for name in ["Ridge walk-forward", "GBDT walk-forward",
                 "ML Ensemble (shrinkage)", "ML Ensemble + InvVol"]:
        nav_oos = navs[name][navs[name].index >= ml_start]
        if len(nav_oos) > 12:
            nav_oos = nav_oos / nav_oos.iloc[0]
            mo = performance_metrics(nav_oos, periods_per_year=12)
            metrics[name]["oos_annual_return"] = mo["annual_return"]
            metrics[name]["oos_sharpe"] = mo["sharpe"]
            metrics[name]["oos_max_drawdown"] = mo["max_drawdown"]
    
    # ---------------- results output ----------------
    summary = pd.DataFrame(metrics).T
    summary = summary[["annual_return", "sharpe", "max_drawdown", "annual_vol",
                       "avg_turnover"] + [c for c in summary.columns if c.startswith("oos_")]]
    print("\n" + "=" * 60)
    print("[Weight Optimization Comparison] (monthly rebalance, Top 30, full period)")
    print("=" * 60)
    print(summary.round(4).to_string())
    
    ensure_dir(results_config["report_dir"])
    ensure_dir(results_config["plot_dir"])
    
    # NAV comparison plot
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
    
    # text report
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = os.path.join(results_config["report_dir"],
                               f"weight_optimization_report_{timestamp}.txt")
    lines = [
        "=" * 60,
        "Factor Weight Optimization Comparison Report",
        "=" * 60,
        f"Generated at: {timestamp}",
        f"Backtest period: {pd.Timestamp(rebalance_dates[0]).date()} ~ {pd.Timestamp(rebalance_dates[-1]).date()}",
        f"Methods: {', '.join(methods.keys())}",
        "",
        summary.round(4).to_string(),
        "",
        "Notes:",
        "- ICIR/EWMA/Mean-Variance/Max-ICIR use only IC/returns strictly before each rebalance date",
        "- Ortho + EWMA: factors are regressed on preceding factors (size first) and residuals are used, removing collinearity before EWMA weighting",
        "- Ridge/GBDT/ML Ensemble are walk-forward; ML Ensemble uses an expanding window and shrinks toward the equal-weight score",
        "- InvVol: holdings within Top-N are weighted by the inverse of trailing 63-day volatility",
        "- oos_* are out-of-sample metrics after the ML training warm-up",
        "=" * 60,
    ]
    report_text = "\n".join(lines)
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_text)
    print(f"\nReport saved to: {report_path}")
    
    # latest optimized weights (for the statistical methods)
    weights_path = os.path.join(results_config["report_dir"], "optimized_weights_latest.csv")
    latest_weights = pd.DataFrame({
        "icir_weight": icir_w.dropna(how="all").iloc[-1],
        "ewma_weight": ewma_w.dropna(how="all").iloc[-1],
        "max_icir_weight": maxicir_w.dropna(how="all").iloc[-1],
        "mv_weight": mv_w.dropna(how="all").iloc[-1].reindex(factor_names),
        "equal_weight": 1.0 / len(factor_names),
    })
    latest_weights.to_csv(weights_path, encoding="utf-8-sig")
    print(f"Latest optimized weights saved to: {weights_path}")
    print("\nWeight optimization comparison complete!")


if __name__ == "__main__":
    main()
