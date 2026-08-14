#!/usr/bin/env python3
"""
Data download script
Batch download CSI 300 constituents' valuation data (Eastmoney valuation analysis),
financial indicators (Eastmoney financial analysis) and the benchmark index
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import yaml
from src.data.data_loader import DataLoader


def main():
    # load config
    config_path = os.path.join(os.path.dirname(__file__), "..", "config", "config.yaml")
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    
    data_config = config["data"]
    
    print("=" * 60)
    print("China A-share Fundamental Data Downloader")
    print("=" * 60)
    print(f"Universe: {data_config['universe']} (CSI 300)")
    print(f"Backtest range: {data_config['start_date']} ~ {data_config['end_date']}")
    print(f"Benchmark: {data_config['benchmark']}")
    print()
    
    loader = DataLoader(
        raw_dir=data_config["raw_dir"],
        processed_dir=data_config["processed_dir"]
    )
    
    # fetch constituent list
    print("Fetching CSI 300 constituent list...")
    stocks_df = loader.get_hs300_stocks()
    if stocks_df.empty:
        print("Error: failed to fetch the constituent list")
        return
    
    # "品种代码" is the raw Chinese column name returned by the API
    stock_codes = stocks_df["品种代码"].tolist()
    print(f"{len(stock_codes)} constituents in total")
    print()
    
    # batch download valuation and financial data
    data = loader.fetch_all_stocks(
        stock_codes=stock_codes,
        save=True,
        start_date=data_config["start_date"],
        end_date=data_config["end_date"]
    )
    
    valuation = data["valuation"]
    print()
    if not valuation.empty:
        print(f"Valuation data shape: {valuation.shape}")
        print(f"Number of stocks: {valuation.index.get_level_values('stock_code').nunique()}")
        print(f"Date range: {valuation.index.get_level_values('date').min()} ~ "
              f"{valuation.index.get_level_values('date').max()}")
    
    # download benchmark index
    print()
    print(f"Downloading benchmark index {data_config['benchmark']}...")
    benchmark = loader.fetch_benchmark(data_config["benchmark"])
    if not benchmark.empty:
        bench_path = os.path.join(data_config["raw_dir"],
                                  f"benchmark_{data_config['benchmark']}.parquet")
        benchmark.to_parquet(bench_path)
        print(f"Benchmark data saved to: {bench_path}")
    
    print()
    print("Download complete!")


if __name__ == "__main__":
    main()
