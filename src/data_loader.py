"""
Data Loader Module for K-Quant System

Handles fetching, cleaning, and storing A-Share market data using AkShare.
Supports OHLCV data, adjusted factors, and fundamental data.
"""

from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import akshare as ak
import pandas as pd


# Data storage path
DATA_DIR = Path(__file__).parent.parent / "data"
DATA_DIR.mkdir(exist_ok=True)


def _get_stock_file_path(stock_code: str, data_type: str = "daily") -> Path:
    """Get the storage path for a stock's data file."""
    prefix = "sh" if stock_code.startswith("6") else "sz"
    full_code = f"{prefix}{stock_code}"
    return DATA_DIR / f"{full_code}_{data_type}.parquet"


def _standardize_stock_code(stock_code: str) -> str:
    """
    Standardize stock code to 6-digit format without exchange prefix.

    Args:
        stock_code: Raw stock code (e.g., "600519", "sh600519", "000001", "sz000001")

    Returns:
        6-digit stock code
    """
    code = str(stock_code).lower().strip()
    if code.startswith("sh") or code.startswith("sz"):
        code = code[2:]
    return code[-6:]


def download_daily_data(
    stock_code: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    force_reload: bool = False,
) -> pd.DataFrame:
    """
    Download daily OHLCV data for a single stock.

    Args:
        stock_code: 6-digit stock code (e.g., "600519")
        start_date: Start date in format "YYYYMMDD", defaults to 5 years ago
        end_date: End date in format "YYYYMMDD", defaults to today
        force_reload: If True, re-download even if data exists

    Returns:
        DataFrame with columns: date, open, high, low, close, volume, amount
    """
    stock_code = _standardize_stock_code(stock_code)
    file_path = _get_stock_file_path(stock_code, "daily")

    # Set default date range
    if end_date is None:
        end_date = datetime.now().strftime("%Y%m%d")
    if start_date is None:
        start_date = (datetime.now() - timedelta(days=365 * 5)).strftime("%Y%m%d")

    # Check for existing data
    if file_path.exists() and not force_reload:
        existing_df = pd.read_parquet(file_path)
        existing_df["date"] = pd.to_datetime(existing_df["date"])

        # Check if date range is covered
        existing_start = existing_df["date"].min().strftime("%Y%m%d")
        existing_end = existing_df["date"].max().strftime("%Y%m%d")

        if existing_start <= start_date and existing_end >= end_date:
            # Filter and return existing data
            mask = (existing_df["date"] >= start_date) & (existing_df["date"] <= end_date)
            return existing_df[mask].copy()

    # Download from AkShare
    try:
        df = ak.stock_zh_a_hist(
            symbol=stock_code,
            period="daily",
            start_date=start_date,
            end_date=end_date,
            adjust="qfq",  # Forward adjusted price (前复权)
        )
    except Exception as e:
        raise ValueError(f"Failed to download data for {stock_code}: {e}")

    if df.empty:
        raise ValueError(f"No data returned for {stock_code}")

    # Standardize column names
    df = df.rename(
        columns={
            "日期": "date",
            "开盘": "open",
            "最高": "high",
            "最低": "low",
            "收盘": "close",
            "成交量": "volume",
            "成交额": "amount",
            "振幅": "amplitude",
            "涨跌幅": "pct_change",
            "涨跌额": "change",
            "换手率": "turnover",
        }
    )

    # Ensure date column is datetime
    df["date"] = pd.to_datetime(df["date"])

    # Sort by date
    df = df.sort_values("date").reset_index(drop=True)

    # Save to parquet
    df.to_parquet(file_path, index=False)

    return df


def download_adjust_factor(stock_code: str, force_reload: bool = False) -> pd.DataFrame:
    """
    Download adjustment factors (复权因子) for a stock.

    Args:
        stock_code: 6-digit stock code
        force_reload: If True, re-download even if data exists

    Returns:
        DataFrame with columns: date, adjust_factor
    """
    stock_code = _standardize_stock_code(stock_code)
    file_path = _get_stock_file_path(stock_code, "adjust_factor")

    if file_path.exists() and not force_reload:
        return pd.read_parquet(file_path)

    try:
        df = ak.stock_zh_a_daily(
            symbol=stock_code,
            start_date="19900101",
            end_date=datetime.now().strftime("%Y%m%d"),
            adjust="qfq",
        )
    except Exception as e:
        raise ValueError(f"Failed to download adjust factor for {stock_code}: {e}")

    if df.empty:
        raise ValueError(f"No adjust factor data for {stock_code}")

    df = df.rename(columns={"date": "date", "factor": "adjust_factor"})
    df["date"] = pd.to_datetime(df["date"])
    df = df[["date", "adjust_factor"]].sort_values("date").reset_index(drop=True)

    df.to_parquet(file_path, index=False)
    return df


def download_fundamental_data(
    stock_code: str,
    force_reload: bool = False,
) -> pd.DataFrame:
    """
    Download fundamental data (valuation metrics) for a stock.

    Note: Real-time valuation data is fetched from spot market. For historical
    valuation data, consider using paid data sources or manual import.

    Args:
        stock_code: 6-digit stock code
        force_reload: If True, re-download even if data exists

    Returns:
        DataFrame with columns: date, pe_ttm, pb, market_cap, etc.
    """
    stock_code = _standardize_stock_code(stock_code)
    file_path = _get_stock_file_path(stock_code, "fundamental")

    if file_path.exists() and not force_reload:
        return pd.read_parquet(file_path)

    # Get current valuation from spot market
    try:
        spot_df = ak.stock_zh_a_spot_em()
        stock_info = spot_df[spot_df["代码"] == stock_code]

        if stock_info.empty:
            raise ValueError(f"No spot data for {stock_code}")

        # Create a single row DataFrame with current valuation
        row = stock_info.iloc[0]
        today = datetime.now().strftime("%Y-%m-%d")

        df = pd.DataFrame({
            "date": [pd.to_datetime(today)],
            "pe_ttm": [row.get("市盈率-动态", None)],
            "pe_static": [row.get("市盈率-静态", None)],
            "pb": [row.get("市净率", None)],
            "market_cap": [row.get("总市值", None)],
            "float_market_cap": [row.get("流通市值", None)],
            "turnover": [row.get("换手率", None)],
            "volume": [row.get("成交量", None)],
        })

    except Exception as e:
        raise ValueError(f"Failed to download fundamental data for {stock_code}: {e}")

    df.to_parquet(file_path, index=False)
    return df


def get_stock_list() -> pd.DataFrame:
    """
    Get the list of all A-Share stocks.

    Returns:
        DataFrame with columns: code, name, industry
    """
    try:
        df = ak.stock_zh_a_spot_em()
    except Exception as e:
        raise ValueError(f"Failed to get stock list: {e}")

    # Select and rename relevant columns
    df = df[["代码", "名称", "所属行业"]].copy()
    df.columns = ["code", "name", "industry"]

    # Standardize code
    df["code"] = df["code"].apply(_standardize_stock_code)

    return df.reset_index(drop=True)


def merge_data(
    daily_df: pd.DataFrame,
    fundamental_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    Merge daily price data with fundamental data using forward fill.

    Args:
        daily_df: Daily OHLCV data
        fundamental_df: Fundamental data (optional)

    Returns:
        Merged DataFrame
    """
    if fundamental_df is None or fundamental_df.empty:
        return daily_df

    # Ensure date columns are datetime
    daily_df["date"] = pd.to_datetime(daily_df["date"])
    fundamental_df["date"] = pd.to_datetime(fundamental_df["date"])

    # Merge on date
    merged = pd.merge_asof(
        daily_df.sort_values("date"),
        fundamental_df.sort_values("date"),
        on="date",
        direction="backward",
    )

    return merged


def load_stock_data(
    stock_code: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    include_fundamental: bool = True,
) -> pd.DataFrame:
    """
    Load stock data from local storage or download if not available.

    Args:
        stock_code: 6-digit stock code
        start_date: Start date "YYYYMMDD"
        end_date: End date "YYYYMMDD"
        include_fundamental: Whether to include fundamental data

    Returns:
        DataFrame with daily data and optional fundamental metrics
    """
    # Download or load daily data
    daily_df = download_daily_data(stock_code, start_date, end_date)

    # Load fundamental data if requested
    if include_fundamental:
        try:
            fund_df = download_fundamental_data(stock_code)
            # Filter to same date range
            if start_date:
                fund_df = fund_df[fund_df["date"] >= start_date]
            if end_date:
                fund_df = fund_df[fund_df["date"] <= end_date]
            return merge_data(daily_df, fund_df)
        except ValueError:
            # If fundamental data not available, return price data only
            pass

    return daily_df


def get_trade_calendar(start_date: Optional[str] = None, end_date: Optional[str] = None) -> pd.DataFrame:
    """
    Get A-Share trade calendar.

    Args:
        start_date: Start date "YYYYMMDD"
        end_date: End date "YYYYMMDD"

    Returns:
        DataFrame with trade dates
    """
    if end_date is None:
        end_date = datetime.now().strftime("%Y%m%d")
    if start_date is None:
        start_date = (datetime.now() - timedelta(days=365 * 2)).strftime("%Y%m%d")

    try:
        df = ak.tool_trade_date_hist_sina()
    except Exception as e:
        raise ValueError(f"Failed to get trade calendar: {e}")

    df["trade_date"] = pd.to_datetime(df["trade_date"])
    df = df.rename(columns={"trade_date": "date"})

    # Filter date range
    mask = (df["date"] >= start_date) & (df["date"] <= end_date)
    return df[mask].reset_index(drop=True)


if __name__ == "__main__":
    # Test: Download data for Kweichow Moutai (600519)
    print("Testing data loader with 600519 (Kweichow Moutai)...")

    test_code = "600519"
    start = "20240101"
    end = "20241231"

    print(f"\n1. Downloading daily data for {test_code}...")
    daily = download_daily_data(test_code, start, end)
    print(f"   Downloaded {len(daily)} rows")
    print(f"   Columns: {list(daily.columns)}")
    print(f"   Date range: {daily['date'].min()} to {daily['date'].max()}")

    print(f"\n2. Downloading fundamental data for {test_code}...")
    try:
        fund = download_fundamental_data(test_code)
        print(f"   Downloaded {len(fund)} rows")
        print(f"   Columns: {list(fund.columns)}")
    except ValueError as e:
        print(f"   Warning: {e}")

    print(f"\n3. Loading merged data for {test_code}...")
    merged = load_stock_data(test_code, start, end, include_fundamental=True)
    print(f"   Merged data shape: {merged.shape}")
    print(f"   Columns: {list(merged.columns)}")

    print(f"\n4. Testing cached data load...")
    cached = download_daily_data(test_code, start, end)
    print(f"   Loaded from cache: {len(cached)} rows")

    print("\n✅ Data loader test completed!")
