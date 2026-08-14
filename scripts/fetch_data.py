#!/usr/bin/env python3
"""
数据下载脚本
批量下载沪深300成分股估值数据（东财估值分析）、财务指标（东财财务分析）与基准指数
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import yaml
from src.data.data_loader import DataLoader


def main():
    # 加载配置
    config_path = os.path.join(os.path.dirname(__file__), "..", "config", "config.yaml")
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    
    data_config = config["data"]
    
    print("=" * 60)
    print("A股基本面数据下载工具")
    print("=" * 60)
    print(f"股票池: {data_config['universe']} (沪深300)")
    print(f"回测范围: {data_config['start_date']} ~ {data_config['end_date']}")
    print(f"基准指数: {data_config['benchmark']}")
    print()
    
    loader = DataLoader(
        raw_dir=data_config["raw_dir"],
        processed_dir=data_config["processed_dir"]
    )
    
    # 获取成分股列表
    print("正在获取沪深300成分股列表...")
    stocks_df = loader.get_hs300_stocks()
    if stocks_df.empty:
        print("错误: 无法获取成分股列表")
        return
    
    stock_codes = stocks_df["品种代码"].tolist()
    print(f"共 {len(stock_codes)} 只成分股")
    print()
    
    # 批量下载估值与财务数据
    data = loader.fetch_all_stocks(
        stock_codes=stock_codes,
        save=True,
        start_date=data_config["start_date"],
        end_date=data_config["end_date"]
    )
    
    valuation = data["valuation"]
    print()
    if not valuation.empty:
        print(f"估值数据形状: {valuation.shape}")
        print(f"股票数量: {valuation.index.get_level_values('stock_code').nunique()}")
        print(f"日期范围: {valuation.index.get_level_values('date').min()} ~ "
              f"{valuation.index.get_level_values('date').max()}")
    
    # 下载基准指数
    print()
    print(f"正在下载基准指数 {data_config['benchmark']}...")
    benchmark = loader.fetch_benchmark(data_config["benchmark"])
    if not benchmark.empty:
        bench_path = os.path.join(data_config["raw_dir"],
                                  f"benchmark_{data_config['benchmark']}.parquet")
        benchmark.to_parquet(bench_path)
        print(f"基准数据已保存至: {bench_path}")
    
    print()
    print("下载完成!")


if __name__ == "__main__":
    main()
