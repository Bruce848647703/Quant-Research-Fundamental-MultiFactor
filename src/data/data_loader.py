"""
Data fetching and preprocessing module
Uses akshare to fetch China A-share valuation data (Eastmoney valuation analysis)
and financial indicators (Eastmoney financial analysis indicators)
"""
import os
import time
import pandas as pd
import akshare as ak
from typing import List


# Valuation column name mapping (Eastmoney valuation analysis API).
# Keys are the raw Chinese column names returned by the API and must be kept as-is.
VALUATION_COLUMNS = {
    "数据日期": "date",
    "当日收盘价": "close",
    "当日涨跌幅": "pct_change",  # in percentage form, needs /100
    "总市值": "total_mv",
    "流通市值": "circ_mv",
    "PE(TTM)": "pe_ttm",
    "市净率": "pb",
    "市销率": "ps",
    "市现率": "pcf",
}


class DataLoader:
    """China A-share fundamental data loader"""
    
    def __init__(self, raw_dir: str = "data/raw", processed_dir: str = "data/processed"):
        self.raw_dir = raw_dir
        self.processed_dir = processed_dir
        os.makedirs(raw_dir, exist_ok=True)
        os.makedirs(processed_dir, exist_ok=True)
    
    def get_hs300_stocks(self) -> pd.DataFrame:
        """Get the CSI 300 constituent list"""
        try:
            df = ak.index_stock_cons(symbol="000300")
            return df
        except Exception as e:
            print(f"Failed to fetch CSI 300 constituents: {e}")
            return pd.DataFrame()
    
    def fetch_valuation(self, stock_code: str, max_retries: int = 3,
                        retry_delay: float = 2.0) -> pd.DataFrame:
        """
        Fetch daily valuation data for a single stock (with retry)
        
        Args:
            stock_code: stock code, e.g. '600519'
            max_retries: maximum number of retries
            retry_delay: delay between retries (seconds)
            
        Returns:
            DataFrame with columns: date, close, pct_change, total_mv, pe_ttm, pb, ps, pcf
        """
        for attempt in range(max_retries):
            try:
                df = ak.stock_value_em(symbol=stock_code)
                if df is None or df.empty:
                    return pd.DataFrame()
                
                # standardize column names
                df = df.rename(columns=VALUATION_COLUMNS)
                keep_cols = [c for c in VALUATION_COLUMNS.values() if c in df.columns]
                df = df[keep_cols].copy()
                
                df["date"] = pd.to_datetime(df["date"])
                # pct_change is in percentage form; convert to decimal return
                df["pct_change"] = df["pct_change"] / 100.0
                df["stock_code"] = stock_code
                df = df.set_index(["date", "stock_code"]).sort_index()
                return df
            except Exception as e:
                if attempt < max_retries - 1:
                    time.sleep(retry_delay * (attempt + 1))
                else:
                    print(f"\nFailed to fetch valuation data for {stock_code} "
                          f"after {max_retries} retries: {e}")
                    return pd.DataFrame()
    
    def fetch_financial(self, stock_code: str, max_retries: int = 3,
                        retry_delay: float = 2.0) -> pd.DataFrame:
        """
        Fetch quarterly financial indicators for a single stock (with retry),
        keeping only ROE-related fields
        
        Args:
            stock_code: stock code, e.g. '600519'
            max_retries: maximum number of retries
            retry_delay: delay between retries (seconds)
            
        Returns:
            DataFrame with columns: report_date, roe (weighted ROE, cumulative)
        """
        # the financial API requires an exchange suffix
        suffix = ".SH" if stock_code.startswith(("6", "9")) else ".SZ"
        symbol = stock_code + suffix
        
        for attempt in range(max_retries):
            try:
                # indicator="按报告期" ("by reporting period") is a required Chinese argument of the API
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
                    print(f"\nFailed to fetch financial data for {stock_code} "
                          f"after {max_retries} retries: {e}")
                    return pd.DataFrame()
    
    def fetch_benchmark(self, symbol: str = "sh000300") -> pd.DataFrame:
        """Fetch daily data of the benchmark index"""
        try:
            df = ak.stock_zh_index_daily(symbol=symbol)
            df["date"] = pd.to_datetime(df["date"])
            df = df.set_index("date").sort_index()
            return df[["close"]]
        except Exception as e:
            print(f"Failed to fetch benchmark index {symbol}: {e}")
            return pd.DataFrame()
    
    def fetch_all_stocks(self, stock_codes: List[str], save: bool = True,
                         start_date: str = "", end_date: str = "") -> dict:
        """
        Batch fetch valuation and financial data
        
        Args:
            stock_codes: list of stock codes
            save: whether to save to files
            start_date: start date (used for file naming only)
            end_date: end date (used for file naming only)
            
        Returns:
            {"valuation": valuation long table, "financial": financial long table}
        """
        val_data, fin_data = [], []
        failed = []
        total = len(stock_codes)
        
        for i, code in enumerate(stock_codes):
            print(f"\rDownload progress: {i+1}/{total} - {code}", end="", flush=True)
            
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
        print(f"Valuation OK: {len(val_data)}, financial OK: {len(fin_data)}, failed: {len(failed)}")
        if failed:
            print(f"Failed stocks: {failed[:10]}{'...' if len(failed) > 10 else ''}")
        
        valuation = pd.concat(val_data) if val_data else pd.DataFrame()
        financial = pd.concat(fin_data) if fin_data else pd.DataFrame()
        
        if save and not valuation.empty:
            val_path = os.path.join(self.raw_dir, f"valuation_{start_date}_{end_date}.parquet")
            valuation.to_parquet(val_path)
            print(f"Valuation data saved to: {val_path}")
            
            fin_path = os.path.join(self.raw_dir, f"financial_{start_date}_{end_date}.parquet")
            financial.to_parquet(fin_path)
            print(f"Financial data saved to: {fin_path}")
        
        return {"valuation": valuation, "financial": financial}
    
    def load_data(self, start_date: str, end_date: str) -> dict:
        """
        Load stored data; fetch from the network if not available
        
        Args:
            start_date: start date
            end_date: end date
            
        Returns:
            {"valuation": valuation long table, "financial": financial long table}
        """
        val_path = os.path.join(self.raw_dir, f"valuation_{start_date}_{end_date}.parquet")
        fin_path = os.path.join(self.raw_dir, f"financial_{start_date}_{end_date}.parquet")
        
        if os.path.exists(val_path) and os.path.exists(fin_path):
            print(f"Loading valuation data from local: {val_path}")
            print(f"Loading financial data from local: {fin_path}")
            return {
                "valuation": pd.read_parquet(val_path),
                "financial": pd.read_parquet(fin_path),
            }
        
        print("Local data not found, fetching from the network...")
        stocks_df = self.get_hs300_stocks()
        if stocks_df.empty:
            raise ValueError("Failed to fetch the CSI 300 constituent list")
        
        # "品种代码" is the raw Chinese column name returned by the API
        stock_codes = stocks_df["品种代码"].tolist()
        return self.fetch_all_stocks(stock_codes, save=True,
                                     start_date=start_date, end_date=end_date)
    
    def prepare_panel_data(self, valuation: pd.DataFrame) -> dict:
        """
        Prepare panel data: pivot long table to wide tables
        (keeps full history so factor computation has warm-up windows)
        
        Args:
            valuation: valuation long table with (date, stock_code) multi-index
            
        Returns:
            dict of wide tables: close / ret / pe_ttm / pb / total_mv
        """
        df = valuation.reset_index()
        df["date"] = pd.to_datetime(df["date"])
        
        panel = {}
        for col in ["close", "pct_change", "pe_ttm", "pb", "total_mv"]:
            wide = df.pivot_table(index="date", columns="stock_code", values=col)
            wide = wide.sort_index()
            panel[col.replace("pct_change", "ret")] = wide
        
        return panel
