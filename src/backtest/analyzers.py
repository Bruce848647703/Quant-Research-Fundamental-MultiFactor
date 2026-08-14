"""
因子分析与回测报告模块
Rank IC分析 / 绩效统计 / 文本报告生成
"""
import os
import pandas as pd
from datetime import datetime
from typing import Dict, List

from ..utils.helpers import performance_metrics, format_pct


def compute_rank_ic(factor_panels: Dict[str, pd.DataFrame], period_ret: pd.DataFrame,
                    rebalance_dates: List[pd.Timestamp]) -> pd.DataFrame:
    """
    计算各因子的月度Rank IC（Spearman相关系数）
    
    Args:
        factor_panels: {因子名: 因子日频宽表}（原始值）
        period_ret: 区间收益宽表（index=调仓日[:-1]）
        rebalance_dates: 完整调仓日期列表
        
    Returns:
        IC序列表（index=调仓日, columns=因子名）
    """
    ic_records = {}
    
    for name, panel in factor_panels.items():
        # 对齐调仓日（缺失向前填充）
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
            # Rank IC: 秩相关的Pearson形式
            ic = x[valid].rank().corr(y[valid].rank())
            ic_series.append(ic)
        
        ic_records[name] = ic_series
    
    return pd.DataFrame(ic_records, index=period_ret.index)


def ic_summary(ic_table: pd.DataFrame) -> pd.DataFrame:
    """
    IC统计汇总: IC均值 / IC标准差 / ICIR / IC胜率
    
    Args:
        ic_table: IC序列表
        
    Returns:
        汇总表
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
    生成回测文本报告
    
    Args:
        analysis: 分析结果字典，包含:
            config / portfolio_metrics / benchmark_metrics / group_metrics /
            ic_summary / turnover_avg / date_range
        output_dir: 报告输出目录
        
    Returns:
        报告文件路径
    """
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = os.path.join(output_dir, f"backtest_report_{timestamp}.txt")
    
    lines = []
    lines.append("=" * 60)
    lines.append("基本面多因子策略回测报告")
    lines.append("=" * 60)
    lines.append(f"生成时间: {timestamp}")
    lines.append(f"回测区间: {analysis['date_range']}")
    lines.append(f"股票池: {analysis['config']['data']['universe']} (沪深300)")
    lines.append(f"调仓频率: {analysis['config']['backtest']['rebalance_freq']}")
    lines.append(f"持仓数量: Top {analysis['config']['backtest']['top_n']}")
    lines.append(f"因子权重: {analysis['config']['factor']['weights']}")
    lines.append("")
    
    # 组合绩效
    lines.append("-" * 60)
    lines.append("【多头组合绩效】")
    lines.append("-" * 60)
    m = analysis["portfolio_metrics"]
    lines.append(f"累计收益:   {format_pct(m['total_return'])}")
    lines.append(f"年化收益:   {format_pct(m['annual_return'])}")
    lines.append(f"年化波动:   {format_pct(m['annual_vol'])}")
    lines.append(f"夏普比率:   {m['sharpe']:.2f}")
    lines.append(f"最大回撤:   {format_pct(m['max_drawdown'])}")
    lines.append(f"卡玛比率:   {m['calmar']:.2f}")
    lines.append(f"平均月换手: {format_pct(analysis['turnover_avg'])}")
    lines.append("")
    
    # 基准绩效
    lines.append("-" * 60)
    lines.append("【基准绩效 (沪深300)】")
    lines.append("-" * 60)
    b = analysis["benchmark_metrics"]
    lines.append(f"累计收益:   {format_pct(b['total_return'])}")
    lines.append(f"年化收益:   {format_pct(b['annual_return'])}")
    lines.append(f"最大回撤:   {format_pct(b['max_drawdown'])}")
    lines.append(f"超额年化:   {format_pct(m['annual_return'] - b['annual_return'])}")
    lines.append("")
    
    # 分层回测
    lines.append("-" * 60)
    lines.append("【分层回测年化收益】(第1组得分最高)")
    lines.append("-" * 60)
    for g, gm in analysis["group_metrics"].items():
        lines.append(f"{g}: 年化 {format_pct(gm['annual_return'])}, "
                     f"夏普 {gm['sharpe']:.2f}, 最大回撤 {format_pct(gm['max_drawdown'])}")
    lines.append("")
    
    # IC分析
    lines.append("-" * 60)
    lines.append("【因子Rank IC统计】")
    lines.append("-" * 60)
    ic_sum = analysis["ic_summary"]
    for name, row in ic_sum.iterrows():
        lines.append(f"{name:>10s}: IC均值 {row['IC_mean']:+.4f}, "
                     f"ICIR {row['ICIR']:+.3f}, IC胜率 {format_pct(row['IC_positive_ratio'], 1)}")
    lines.append("")
    lines.append("=" * 60)
    
    report_text = "\n".join(lines)
    
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_text)
    
    print(report_text)
    return report_path
