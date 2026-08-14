#!/usr/bin/env python3
"""
Latest holdings output script
Uses ML ensemble scoring by default (expanding-window Ridge+GBDT walk-forward,
shrunk toward the equal-weight score) and outputs the portfolio to hold now.
Use --equal-weight to switch to the plain equal-weight multi-factor score.
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
    ML ensemble scoring (walk-forward, strictly no lookahead)
    Train Ridge+GBDT on an expanding window to predict next-period returns,
    average after cross-section normalization, then shrink toward the
    equal-weight score by weight_optimization.ml_shrinkage to reduce variance.
    
    Args:
        factor_panels: dict of factor panels
        period_ret: period return wide table (provides historical training labels)
        eval_dates: list of dates to output scores for
        factor_names: list of factor names
        config: global config
        scores_equal: equal-weight combined score (shrinkage target)
        
    Returns:
        score wide table (index=eval_dates, columns=stock code)
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
    """Build a holdings table from single-date scores"""
    score_t = scores.loc[date].dropna().sort_values(ascending=False)
    table = pd.DataFrame({
        "stock_code": score_t.index,
        "stock_name": [name_map.get(c, "-") for c in score_t.index],
        "score": score_t.values.round(4),
    })
    table.insert(0, "rank", range(1, len(table) + 1))
    return table


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Latest holdings output")
    parser.add_argument("--equal-weight", action="store_true",
                        help="use equal-weight multi-factor scoring (default: ML ensemble)")
    args = parser.parse_args()
    
    method = "Equal-Weight Multi-Factor" if args.equal_weight else \
        "ML Ensemble (expanding-window Ridge+GBDT, 30% shrinkage)"
    print("Fundamental Multi-Factor - Latest Holdings")
    print(f"Scoring method: {method}")
    print("=" * 60)
    
    config = load_config()
    data_config = config["data"]
    results_config = config["results"]
    
    # load data
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
    
    # stock name mapping ("品种代码"/"品种名称" are raw Chinese column names from the API)
    stocks_df = loader.get_hs300_stocks()
    name_map = dict(zip(stocks_df["品种代码"], stocks_df["品种名称"]))
    
    # compute factor panels
    trading_dates = panel["ret"].index
    factor_panels = compute_factor_panels(config, panel, financial)
    factor_names = list(config["factor"]["weights"].keys())
    
    latest_date = trading_dates[-1]
    engine = MultiFactorEngine(config)
    top_n = config["backtest"]["top_n"]
    
    # 1. holdings at the most recent official rebalance date (month end)
    #    -- what should actually be held now
    rebalance_dates = engine.get_rebalance_dates(
        trading_dates,
        start_date=data_config["start_date"],
        end_date=data_config["end_date"]
    )
    last_rebalance = rebalance_dates[-1]
    
    # equal-weight score (shrinkage target for ML ensemble / or the default scorer)
    combiner = MultiFactorCombiner(config["factor"]["weights"])
    eval_dates = sorted({last_rebalance, latest_date})
    scores_equal = combiner.combine(factor_panels, eval_dates)
    
    if args.equal_weight:
        scores_rebal, scores_latest = scores_equal, scores_equal
    else:
        print("Training ML ensemble (expanding-window walk-forward)...")
        period_ret = engine.compute_period_returns(panel["ret"], rebalance_dates)
        scores_all = ml_ensemble_scores(
            factor_panels, period_ret, eval_dates,
            factor_names, config, scores_equal)
        scores_rebal = scores_all.loc[[last_rebalance]]
        scores_latest = scores_all.loc[[latest_date]]
    
    current = build_holdings_table(scores_rebal, last_rebalance, name_map)
    
    print(f"\n[Current Holdings] last rebalance: {last_rebalance.date()}, "
          f"hold until the next month-end rebalance")
    print(current.head(top_n).to_string(index=False))
    
    # 2. hypothetical holdings if rebalancing immediately with the latest data
    if latest_date != last_rebalance:
        hypothetical = build_holdings_table(scores_latest, latest_date, name_map)
        
        print(f"\n[Reference] if rebalancing immediately with the latest data ({latest_date.date()}):")
        print(hypothetical.head(top_n).to_string(index=False))
        
        cur_set = set(current.head(top_n)["stock_code"])
        hyp_set = set(hypothetical.head(top_n)["stock_code"])
        print(f"\nvs current holdings: {len(hyp_set - cur_set)} added, "
              f"{len(cur_set - hyp_set)} removed")
        if hyp_set - cur_set:
            print(f"Added: {sorted(hyp_set - cur_set)}")
        if cur_set - hyp_set:
            print(f"Removed: {sorted(cur_set - hyp_set)}")
    
    # save results
    ensure_dir(results_config["report_dir"])
    out_path = os.path.join(results_config["report_dir"], "holdings_latest.csv")
    current.head(top_n).to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"\nCurrent holdings saved to: {out_path}")
    print("\nNote: this is only the output of a factor model, not investment advice")


if __name__ == "__main__":
    main()
