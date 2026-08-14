"""
Factor analysis and backtest report module
Rank IC analysis / performance metrics / text report generation
"""
import os
import pandas as pd
from datetime import datetime
from typing import Dict, List

from ..utils.helpers import performance_metrics, format_pct


def compute_rank_ic(factor_panels: Dict[str, pd.DataFrame], period_ret: pd.DataFrame,
                    rebalance_dates: List[pd.Timestamp]) -> pd.DataFrame:
    """
    Compute monthly Rank IC (Spearman correlation) for each factor
    
    Args:
        factor_panels: {factor name: daily factor wide table} (raw values)
        period_ret: period return wide table (index=rebalance dates[:-1])
        rebalance_dates: full rebalance date list
        
    Returns:
        IC series table (index=rebalance date, columns=factor name)
    """
    ic_records = {}
    
    for name, panel in factor_panels.items():
        # align to rebalance dates (forward-fill missing)
        panel = panel.reindex(panel.index.union(pd.DatetimeIndex(rebalance_dates)))
        panel = panel.ffill().loc[rebalance_dates]
        
        ic_series = []
        for t in period_ret.index:
            x = panel.loc[t]
            y = period_ret.loc[t]
            valid = x.notna() & y.notna()
            if valid.sum() < 10:
                ic_series.append(float("nan"))
                continue
            # Rank IC: Pearson correlation of ranks
            ic = x[valid].rank().corr(y[valid].rank())
            ic_series.append(ic)
        
        ic_records[name] = ic_series
    
    return pd.DataFrame(ic_records, index=period_ret.index)


def ic_summary(ic_table: pd.DataFrame) -> pd.DataFrame:
    """
    IC statistics summary: IC mean / IC std / ICIR / IC positive ratio
    
    Args:
        ic_table: IC series table
        
    Returns:
        summary table
    """
    summary = pd.DataFrame({
        "IC_mean": ic_table.mean(),
        "IC_std": ic_table.std(),
        "ICIR": ic_table.mean() / ic_table.std(),
        "IC_positive_ratio": (ic_table > 0).mean(),
    })
    return summary


def generate_report(analysis: Dict, output_dir: str = "results/reports") -> str:
    """
    Generate the backtest text report
    
    Args:
        analysis: analysis result dict containing:
            config / portfolio_metrics / benchmark_metrics / group_metrics /
            ic_summary / turnover_avg / date_range
        output_dir: report output directory
        
    Returns:
        path of the report file
    """
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = os.path.join(output_dir, f"backtest_report_{timestamp}.txt")
    
    lines = []
    lines.append("=" * 60)
    lines.append("Fundamental Multi-Factor Strategy Backtest Report")
    lines.append("=" * 60)
    lines.append(f"Generated at: {timestamp}")
    lines.append(f"Backtest period: {analysis['date_range']}")
    lines.append(f"Universe: {analysis['config']['data']['universe']} (CSI 300)")
    lines.append(f"Rebalance frequency: {analysis['config']['backtest']['rebalance_freq']}")
    lines.append(f"Holdings: Top {analysis['config']['backtest']['top_n']}")
    lines.append(f"Factor weights: {analysis['config']['factor']['weights']}")
    lines.append("")
    
    # portfolio performance
    lines.append("-" * 60)
    lines.append("[Long Portfolio Performance]")
    lines.append("-" * 60)
    m = analysis["portfolio_metrics"]
    lines.append(f"Total return:     {format_pct(m['total_return'])}")
    lines.append(f"Annual return:    {format_pct(m['annual_return'])}")
    lines.append(f"Annual vol:       {format_pct(m['annual_vol'])}")
    lines.append(f"Sharpe ratio:     {m['sharpe']:.2f}")
    lines.append(f"Max drawdown:     {format_pct(m['max_drawdown'])}")
    lines.append(f"Calmar ratio:     {m['calmar']:.2f}")
    lines.append(f"Avg monthly turnover: {format_pct(analysis['turnover_avg'])}")
    lines.append("")
    
    # benchmark performance
    lines.append("-" * 60)
    lines.append("[Benchmark Performance (CSI 300)]")
    lines.append("-" * 60)
    b = analysis["benchmark_metrics"]
    lines.append(f"Total return:     {format_pct(b['total_return'])}")
    lines.append(f"Annual return:    {format_pct(b['annual_return'])}")
    lines.append(f"Max drawdown:     {format_pct(b['max_drawdown'])}")
    lines.append(f"Annual excess:    {format_pct(m['annual_return'] - b['annual_return'])}")
    lines.append("")
    
    # quantile-group backtest
    lines.append("-" * 60)
    lines.append("[Quantile-Group Annualized Returns] (group 1 = highest score)")
    lines.append("-" * 60)
    for g, gm in analysis["group_metrics"].items():
        lines.append(f"{g}: annualized {format_pct(gm['annual_return'])}, "
                     f"sharpe {gm['sharpe']:.2f}, max drawdown {format_pct(gm['max_drawdown'])}")
    lines.append("")
    
    # IC analysis
    lines.append("-" * 60)
    lines.append("[Factor Rank IC Statistics]")
    lines.append("-" * 60)
    ic_sum = analysis["ic_summary"]
    for name, row in ic_sum.iterrows():
        lines.append(f"{name:>10s}: IC mean {row['IC_mean']:+.4f}, "
                     f"ICIR {row['ICIR']:+.3f}, IC positive ratio {format_pct(row['IC_positive_ratio'], 1)}")
    lines.append("")
    lines.append("=" * 60)
    
    report_text = "\n".join(lines)
    
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_text)
    
    print(report_text)
    return report_path
