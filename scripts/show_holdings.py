#!/usr/bin/env python3
"""
最新持仓输出脚本
默认使用ML集成打分（扩展窗口 Ridge+GBDT walk-forward，向等权得分收缩），
输出当前应持有的股票组合；--equal-weight 可切换为原始等权多因子打分
"""
import sys
import os
import argparse
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))

import pandas as pd

from src.data.data_loader import DataLoader
from src.factors.fundamental import MultiFactorCombiner
from src.factors.weight_optimizer import MLWalkForwardScorer
from src.factors.base import preprocess_cross_section
from src.backtest.engine import MultiFactorEngine
from src.utils.helpers import load_config, ensure_dir
from run_backtest import compute_factor_panels


def ml_ensemble_scores(factor_panels: dict, period_ret: pd.DataFrame,
                       eval_dates: list, factor_names: list, config: dict,
                       scores_equal: pd.DataFrame) -> pd.DataFrame:
    """
    ML集成打分（walk-forward，严格无未来函数）
    扩展窗口训练 Ridge+GBDT 预测下期收益，截面标准化后平均，
    再按 weight_optimization.ml_shrinkage 比例向等权得分收缩降方差
    
    Args:
        factor_panels: 因子面板dict
        period_ret: 区间收益宽表（提供历史训练标签）
        eval_dates: 需要输出得分的日期列表
        factor_names: 因子名列表
        config: 全局配置
        scores_equal: 等权合成得分（收缩基准）
        
    Returns:
        得分宽表（index=eval_dates, columns=股票代码）
    """
    wo_config = config.get("weight_optimization", {})
    min_train = wo_config.get("ml_min_train_periods", 24)
    shrinkage = wo_config.get("ml_shrinkage", 0.3)
    
    ridge = MLWalkForwardScorer(model_type="ridge", expanding=True,
                                clip_targets=True, min_train_periods=min_train,
                                factor_names=factor_names)
    gbdt = MLWalkForwardScorer(model_type="gbdt", expanding=True,
                               clip_targets=True, min_train_periods=min_train,
                               factor_names=factor_names)
    pred_ridge = ridge.fit_predict(factor_panels, period_ret, eval_dates)
    pred_gbdt = gbdt.fit_predict(factor_panels, period_ret, eval_dates)
    
    rows = []
    for t in eval_dates:
        r_z = preprocess_cross_section(pred_ridge.loc[t])
        g_z = preprocess_cross_section(pred_gbdt.loc[t])
        ml_z = (r_z + g_z) / 2
        if ml_z.notna().any():
            eq_z = preprocess_cross_section(scores_equal.loc[t])
            row = (1 - shrinkage) * ml_z + shrinkage * eq_z
        else:
            row = ml_z
        rows.append(pd.DataFrame([row], index=[t]))
    return pd.concat(rows)


def build_holdings_table(scores: pd.DataFrame, date: pd.Timestamp,
                         name_map: dict) -> pd.DataFrame:
    """根据单日得分生成持仓表"""
    score_t = scores.loc[date].dropna().sort_values(ascending=False)
    table = pd.DataFrame({
        "stock_code": score_t.index,
        "stock_name": [name_map.get(c, "-") for c in score_t.index],
        "score": score_t.values.round(4),
    })
    table.insert(0, "rank", range(1, len(table) + 1))
    return table


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="最新持仓输出")
    parser.add_argument("--equal-weight", action="store_true",
                        help="使用等权多因子打分（默认: ML集成）")
    args = parser.parse_args()
    
    method = "等权多因子" if args.equal_weight else "ML集成(扩展窗口Ridge+GBDT, 收缩30%)"
    print("基本面多因子 - 最新持仓计算")
    print(f"打分方式: {method}")
    print("=" * 60)
    
    config = load_config()
    data_config = config["data"]
    results_config = config["results"]
    
    # 加载数据
    loader = DataLoader(
        raw_dir=data_config["raw_dir"],
        processed_dir=data_config["processed_dir"]
    )
    data = loader.load_data(
        start_date=data_config["start_date"],
        end_date=data_config["end_date"]
    )
    valuation, financial = data["valuation"], data["financial"]
    panel = loader.prepare_panel_data(valuation)
    
    # 股票名称映射
    stocks_df = loader.get_hs300_stocks()
    name_map = dict(zip(stocks_df["品种代码"], stocks_df["品种名称"]))
    
    # 计算因子面板
    trading_dates = panel["ret"].index
    factor_panels = compute_factor_panels(config, panel, financial)
    factor_names = list(config["factor"]["weights"].keys())
    
    latest_date = trading_dates[-1]
    engine = MultiFactorEngine(config)
    top_n = config["backtest"]["top_n"]
    
    # 1. 最近一次正式调仓日（月末）的持仓 —— 当前实际应持有
    rebalance_dates = engine.get_rebalance_dates(
        trading_dates,
        start_date=data_config["start_date"],
        end_date=data_config["end_date"]
    )
    last_rebalance = rebalance_dates[-1]
    
    # 等权得分（ML集成的收缩基准 / 或作为默认打分）
    combiner = MultiFactorCombiner(config["factor"]["weights"])
    eval_dates = sorted({last_rebalance, latest_date})
    scores_equal = combiner.combine(factor_panels, eval_dates)
    
    if args.equal_weight:
        scores_rebal, scores_latest = scores_equal, scores_equal
    else:
        print("训练 ML集成（扩展窗口 walk-forward）...")
        period_ret = engine.compute_period_returns(panel["ret"], rebalance_dates)
        scores_all = ml_ensemble_scores(
            factor_panels, period_ret, eval_dates,
            factor_names, config, scores_equal)
        scores_rebal = scores_all.loc[[last_rebalance]]
        scores_latest = scores_all.loc[[latest_date]]
    
    current = build_holdings_table(scores_rebal, last_rebalance, name_map)
    
    print(f"\n【当前持仓】最近调仓日: {last_rebalance.date()}, "
          f"持有至下一个月末调仓")
    print(current.head(top_n).to_string(index=False))
    
    # 2. 若按最新数据立即调仓的假设持仓（对比参考）
    if latest_date != last_rebalance:
        hypothetical = build_holdings_table(scores_latest, latest_date, name_map)
        
        print(f"\n【参考】若按最新数据({latest_date.date()})立即调仓:")
        print(hypothetical.head(top_n).to_string(index=False))
        
        cur_set = set(current.head(top_n)["stock_code"])
        hyp_set = set(hypothetical.head(top_n)["stock_code"])
        print(f"\n与当前持仓相比: 新进 {len(hyp_set - cur_set)} 只, "
              f"剔除 {len(cur_set - hyp_set)} 只")
        if hyp_set - cur_set:
            print(f"新进: {sorted(hyp_set - cur_set)}")
        if cur_set - hyp_set:
            print(f"剔除: {sorted(cur_set - hyp_set)}")
    
    # 保存结果
    ensure_dir(results_config["report_dir"])
    out_path = os.path.join(results_config["report_dir"], "holdings_latest.csv")
    current.head(top_n).to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"\n当前持仓已保存至: {out_path}")
    print("\n提示: 以上仅为因子模型输出，不构成投资建议")


if __name__ == "__main__":
    main()
