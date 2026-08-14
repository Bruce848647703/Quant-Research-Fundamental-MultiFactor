#!/usr/bin/env python3
"""
最新持仓输出脚本
基于最新数据计算多因子得分，输出当前应持有的股票组合
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pandas as pd

from src.data.data_loader import DataLoader
from src.factors.fundamental import (
    EPFactor, BPFactor, SizeFactor, MomentumFactor, ROETTMFactor,
    MultiFactorCombiner
)
from src.backtest.engine import MultiFactorEngine
from src.utils.helpers import load_config, ensure_dir


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
    print("基本面多因子 - 最新持仓计算")
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
    factor_config = config["factor"]
    trading_dates = panel["ret"].index
    factor_panels = {
        "ep": EPFactor().compute(panel["pe_ttm"]),
        "bp": BPFactor().compute(panel["pb"]),
        "size": SizeFactor().compute(panel["total_mv"]),
        "momentum": MomentumFactor(
            long_window=factor_config["momentum_long_window"],
            skip_window=factor_config["momentum_skip_window"]
        ).compute(panel["ret"]),
        "roe": ROETTMFactor(
            disclosure_lag=factor_config["disclosure_lag"]
        ).compute(financial, trading_dates),
    }
    
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
    
    combiner = MultiFactorCombiner(factor_config["weights"])
    scores_rebal = combiner.combine(factor_panels, [last_rebalance])
    current = build_holdings_table(scores_rebal, last_rebalance, name_map)
    
    print(f"\n【当前持仓】最近调仓日: {last_rebalance.date()}, "
          f"持有至下一个月末调仓")
    print(current.head(top_n).to_string(index=False))
    
    # 2. 若按最新数据立即调仓的假设持仓（对比参考）
    if latest_date != last_rebalance:
        scores_latest = combiner.combine(factor_panels, [latest_date])
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
