"""
数据获取与预处理模块
使用 akshare 获取A股估值数据（东财估值分析）与财务指标（东财财务分析指标）
"""
import os
import time
import pandas as pd
import akshare as ak
from typing import List


# 估值数据列名映射（东财估值分析接口）
VALUATION_COLUMNS = {
    "数据日期": "date",
    "当日收盘价": "close",
    "当日涨跌幅": "pct_change",  # 百分数形式，需 /100
    "总市值": "total_mv",
    "流通市值": "circ_mv",
    "PE(TTM)": "pe_ttm",
    "市净率": "pb",
    "市销率": "ps",
    "市现率": "pcf",
}


class DataLoader:
    """A股基本面数据加载器"""
    
    def __init__(self, raw_dir: str = "data/raw", processed_dir: str = "data/processed"):
        self.raw_dir = raw_dir
        self.processed_dir = processed_dir
        os.makedirs(raw_dir, exist_ok=True)
        os.makedirs(processed_dir, exist_ok=True)
    
    def get_hs300_stocks(self) -> pd.DataFrame:
        """获取沪深300成分股列表"""
        try:
            df = ak.index_stock_cons(symbol="000300")
            return df
        except Exception as e:
            print(f"获取沪深300成分股失败: {e}")
            return pd.DataFrame()
    
    def fetch_valuation(self, stock_code: str, max_retries: int = 3,
                        retry_delay: float = 2.0) -> pd.DataFrame:
        """
        获取单只股票日频估值数据（含重试机制）
        
        Args:
            stock_code: 股票代码，如 '600519'
            max_retries: 最大重试次数
            retry_delay: 重试间隔（秒）
            
        Returns:
            DataFrame with columns: date, close, pct_change, total_mv, pe_ttm, pb, ps, pcf
        """
        for attempt in range(max_retries):
            try:
                df = ak.stock_value_em(symbol=stock_code)
                if df is None or df.empty:
                    return pd.DataFrame()
                
                # 标准化列名
                df = df.rename(columns=VALUATION_COLUMNS)
                keep_cols = [c for c in VALUATION_COLUMNS.values() if c in df.columns]
                df = df[keep_cols].copy()
                
                df["date"] = pd.to_datetime(df["date"])
                # 涨跌幅为百分数形式，转换为小数收益率
                df["pct_change"] = df["pct_change"] / 100.0
                df["stock_code"] = stock_code
                df = df.set_index(["date", "stock_code"]).sort_index()
                return df
            except Exception as e:
                if attempt < max_retries - 1:
                    time.sleep(retry_delay * (attempt + 1))
                else:
                    print(f"\n获取 {stock_code} 估值数据失败(重试{max_retries}次): {e}")
                    return pd.DataFrame()
    
    def fetch_financial(self, stock_code: str, max_retries: int = 3,
                        retry_delay: float = 2.0) -> pd.DataFrame:
        """
        获取单只股票季度财务指标（含重试机制），仅保留ROE相关字段
        
        Args:
            stock_code: 股票代码，如 '600519'
            max_retries: 最大重试次数
            retry_delay: 重试间隔（秒）
            
        Returns:
            DataFrame with columns: report_date, roe (加权ROE，累计值)
        """
        # 财务接口要求带交易所后缀
        suffix = ".SH" if stock_code.startswith(("6", "9")) else ".SZ"
        symbol = stock_code + suffix
        
        for attempt in range(max_retries):
            try:
                df = ak.stock_financial_analysis_indicator_em(symbol=symbol, indicator="按报告期")
                if df is None or df.empty or "ROEJQ" not in df.columns:
                    return pd.DataFrame()
                
                df = df[["REPORT_DATE", "ROEJQ"]].copy()
                df.columns = ["report_date", "roe"]
                df["report_date"] = pd.to_datetime(df["report_date"])
                df = df.dropna(subset=["roe"])
                df["stock_code"] = stock_code
                df = df.set_index(["stock_code", "report_date"]).sort_index()
                return df
            except Exception as e:
                if attempt < max_retries - 1:
                    time.sleep(retry_delay * (attempt + 1))
                else:
                    print(f"\n获取 {stock_code} 财务数据失败(重试{max_retries}次): {e}")
                    return pd.DataFrame()
    
    def fetch_benchmark(self, symbol: str = "sh000300") -> pd.DataFrame:
        """获取基准指数日线数据"""
        try:
            df = ak.stock_zh_index_daily(symbol=symbol)
            df["date"] = pd.to_datetime(df["date"])
            df = df.set_index("date").sort_index()
            return df[["close"]]
        except Exception as e:
            print(f"获取基准指数 {symbol} 失败: {e}")
            return pd.DataFrame()
    
    def fetch_all_stocks(self, stock_codes: List[str], save: bool = True,
                         start_date: str = "", end_date: str = "") -> dict:
        """
        批量获取估值数据与财务数据
        
        Args:
            stock_codes: 股票代码列表
            save: 是否保存到文件
            start_date: 开始日期（仅用于文件命名）
            end_date: 结束日期（仅用于文件命名）
            
        Returns:
            {"valuation": 估值长表, "financial": 财务长表}
        """
        val_data, fin_data = [], []
        failed = []
        total = len(stock_codes)
        
        for i, code in enumerate(stock_codes):
            print(f"\r下载进度: {i+1}/{total} - {code}", end="", flush=True)
            
            val_df = self.fetch_valuation(code)
            if not val_df.empty:
                val_data.append(val_df)
            else:
                failed.append(code)
            time.sleep(0.3)
            
            fin_df = self.fetch_financial(code)
            if not fin_df.empty:
                fin_data.append(fin_df)
            time.sleep(0.3)
        
        print()
        print(f"估值数据成功: {len(val_data)}, 财务数据成功: {len(fin_data)}, 失败: {len(failed)}")
        if failed:
            print(f"失败股票: {failed[:10]}{'...' if len(failed) > 10 else ''}")
        
        valuation = pd.concat(val_data) if val_data else pd.DataFrame()
        financial = pd.concat(fin_data) if fin_data else pd.DataFrame()
        
        if save and not valuation.empty:
            val_path = os.path.join(self.raw_dir, f"valuation_{start_date}_{end_date}.parquet")
            valuation.to_parquet(val_path)
            print(f"估值数据已保存至: {val_path}")
            
            fin_path = os.path.join(self.raw_dir, f"financial_{start_date}_{end_date}.parquet")
            financial.to_parquet(fin_path)
            print(f"财务数据已保存至: {fin_path}")
        
        return {"valuation": valuation, "financial": financial}
    
    def load_data(self, start_date: str, end_date: str) -> dict:
        """
        加载已存储的数据，若不存在则从网络获取
        
        Args:
            start_date: 开始日期
            end_date: 结束日期
            
        Returns:
            {"valuation": 估值长表, "financial": 财务长表}
        """
        val_path = os.path.join(self.raw_dir, f"valuation_{start_date}_{end_date}.parquet")
        fin_path = os.path.join(self.raw_dir, f"financial_{start_date}_{end_date}.parquet")
        
        if os.path.exists(val_path) and os.path.exists(fin_path):
            print(f"从本地加载估值数据: {val_path}")
            print(f"从本地加载财务数据: {fin_path}")
            return {
                "valuation": pd.read_parquet(val_path),
                "financial": pd.read_parquet(fin_path),
            }
        
        print("本地数据不存在，开始从网络获取...")
        stocks_df = self.get_hs300_stocks()
        if stocks_df.empty:
            raise ValueError("无法获取沪深300成分股列表")
        
        stock_codes = stocks_df["品种代码"].tolist()
        return self.fetch_all_stocks(stock_codes, save=True,
                                     start_date=start_date, end_date=end_date)
    
    def prepare_panel_data(self, valuation: pd.DataFrame) -> dict:
        """
        准备面板数据：长表转宽表（保留全部历史，供因子计算预热窗口使用）
        
        Args:
            valuation: 估值长表 (date, stock_code) 多级索引
            
        Returns:
            宽表字典: close / ret / pe_ttm / pb / total_mv
        """
        df = valuation.reset_index()
        df["date"] = pd.to_datetime(df["date"])
        
        panel = {}
        for col in ["close", "pct_change", "pe_ttm", "pb", "total_mv"]:
            wide = df.pivot_table(index="date", columns="stock_code", values=col)
            wide = wide.sort_index()
            panel[col.replace("pct_change", "ret")] = wide
        
        return panel
