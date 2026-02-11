#!/usr/bin/env python3
"""
K-Quant Data Update Script

Manually update stock data with full overwrite.

Examples:
    python update_data.py --stock 601012                    # Update single stock
    python update_data.py --stock 601012 --start 20200101   # Update with custom start date
    python update_data.py --all                             # Update all stocks (slow!)
"""

import argparse
import sys
from pathlib import Path
from datetime import datetime, timedelta

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

import pandas as pd
from data_loader import (
    load_stock_data,
    download_daily_data,
    download_fundamental_data,
    get_stock_list,
    DATA_DIR,
)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="K-Quant Data Update Tool")
    parser.add_argument("--stock", type=str, default=None,
                        help="Stock code to update (e.g., 601012)")
    parser.add_argument("--all", action="store_true",
                        help="Update all stocks (warning: slow!)")
    parser.add_argument("--start", type=str, default=None,
                        help="Start date (YYYYMMDD, default: 5 years ago)")
    parser.add_argument("--end", type=str, default=None,
                        help="End date (YYYYMMDD, default: today)")
    parser.add_argument("--include-fundamental", action="store_true",
                        help="Also update fundamental data")
    parser.add_argument("--list", action="store_true",
                        help="List available stocks")

    return parser.parse_args()


def list_stocks():
    """List available stocks."""
    print("Fetching stock list...")
    df = get_stock_list()
    print(f"\nTotal stocks: {len(df)}")
    print("\nFirst 20 stocks:")
    print(df.head(20).to_string(index=False))
    return 0


def update_single_stock(stock_code: str, start_date: str, end_date: str, include_fundamental: bool = False):
    """Update data for a single stock."""
    print(f"\n{'='*60}")
    print(f"Updating data for {stock_code}")
    print(f"{'='*60}")

    try:
        # Delete existing data files for this stock
        prefix = "sh" if stock_code.startswith("6") else "sz"
        full_code = f"{prefix}{stock_code}"

        for suffix in ["daily", "fundamental", "adjust_factor"]:
            file_path = DATA_DIR / f"{full_code}_{suffix}.parquet"
            if file_path.exists():
                print(f"  Removing old file: {file_path.name}")
                file_path.unlink()

        # Download new data with force_reload=True
        print(f"\nDownloading daily data ({start_date} to {end_date})...")
        daily_df = download_daily_data(stock_code, start_date, end_date, force_reload=True)
        print(f"  ✓ Downloaded {len(daily_df)} rows")
        print(f"  Date range: {daily_df['date'].min()} to {daily_df['date'].max()}")

        if include_fundamental:
            print("\nDownloading fundamental data...")
            try:
                fund_df = download_fundamental_data(stock_code, force_reload=True)
                print(f"  ✓ Downloaded {len(fund_df)} rows")
            except Exception as e:
                print(f"  ⚠ Fundamental data not available: {e}")

        print(f"\n✅ {stock_code} updated successfully!")
        return True

    except Exception as e:
        print(f"\n❌ Failed to update {stock_code}: {e}")
        return False


def update_all_stocks(start_date: str, end_date: str, include_fundamental: bool = False):
    """Update all stocks (warning: very slow!)."""
    print("\nFetching stock list...")
    stock_list = get_stock_list()
    print(f"Total stocks to update: {len(stock_list)}")
    print("\n⚠️  This will take a long time! Press Ctrl+C to cancel.")
    print("Starting in 3 seconds...")
    import time
    time.sleep(3)

    success_count = 0
    fail_count = 0

    for idx, row in stock_list.iterrows():
        stock_code = row['code']
        print(f"\n[{idx+1}/{len(stock_list)}] ", end="")

        if update_single_stock(stock_code, start_date, end_date, include_fundamental):
            success_count += 1
        else:
            fail_count += 1

        # Progress summary every 10 stocks
        if (idx + 1) % 10 == 0:
            print(f"\n📊 Progress: {success_count} succeeded, {fail_count} failed")

    print(f"\n{'='*60}")
    print("Update completed!")
    print(f"  Success: {success_count}")
    print(f"  Failed: {fail_count}")
    print(f"{'='*60}")

    return fail_count == 0


def main():
    """Main entry point."""
    args = parse_args()

    # Handle list command
    if args.list:
        return list_stocks()

    # Set default dates
    if args.end is None:
        args.end = datetime.now().strftime("%Y%m%d")
    if args.start is None:
        args.start = (datetime.now() - timedelta(days=365*5)).strftime("%Y%m%d")

    print(f"\n{'='*60}")
    print("K-Quant Data Update Tool")
    print(f"{'='*60}")
    print(f"Date range: {args.start} to {args.end}")
    print(f"Include fundamental: {args.include_fundamental}")

    # Validate arguments
    if not args.stock and not args.all:
        print("\nError: Please specify --stock CODE or --all")
        print("Use --list to see available stocks")
        return 1

    if args.all:
        update_all_stocks(args.start, args.end, args.include_fundamental)
    else:
        success = update_single_stock(
            args.stock,
            args.start,
            args.end,
            args.include_fundamental
        )
        return 0 if success else 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
