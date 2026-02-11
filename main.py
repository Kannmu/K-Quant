#!/usr/bin/env python3
"""
K-Quant Main Entry Point

Run backtests from command line.

Examples:
    python main.py --stock 601012 --strategy ma --start 20230101 --end 20241231
    python main.py --stock 601012 --strategy rsi --cash 100000
    python main.py --stock 601012 --plot  # Generate visualization
"""

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from data_loader import load_stock_data
from backtest.engine import BacktestEngine, print_backtest_report
from strategy.templates import DoubleMovingAverageStrategy, RSIStrategy, MACDStrategy
from analysis.metrics import calculate_all_metrics, print_detailed_metrics
from analysis.plotting import create_full_report, plot_equity_curve, plot_drawdown


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="K-Quant Backtest System")
    parser.add_argument("--stock", type=str, default="601012",
                        help="Stock code (default: 601012 隆基绿能)")
    parser.add_argument("--strategy", type=str, default="ma",
                        choices=["ma", "rsi", "macd", "buyhold"],
                        help="Strategy to use (default: ma)")
    parser.add_argument("--start", type=str, default="20230101",
                        help="Start date (YYYYMMDD, default: 20230101)")
    parser.add_argument("--end", type=str, default="20241231",
                        help="End date (YYYYMMDD, default: 20241231)")
    parser.add_argument("--cash", type=float, default=100000,
                        help="Initial cash (default: 100000)")
    parser.add_argument("--fast", type=int, default=20,
                        help="Fast MA period (default: 20)")
    parser.add_argument("--slow", type=int, default=60,
                        help="Slow MA period (default: 60)")
    parser.add_argument("--verbose", action="store_true",
                        help="Print detailed trade log")
    parser.add_argument("--plot", action="store_true",
                        help="Show plots after backtest")
    parser.add_argument("--save-plot", type=str, default=None,
                        help="Save plot to file path")

    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()

    print(f"\n{'='*60}")
    print("K-Quant Backtest System")
    print(f"{'='*60}\n")

    # Load data
    print(f"Loading data for {args.stock}...")
    try:
        data = load_stock_data(args.stock, args.start, args.end, include_fundamental=False)
    except Exception as e:
        print(f"Error loading data: {e}")
        return 1

    print(f"  Loaded {len(data)} rows")
    print(f"  Date range: {data['date'].min().strftime('%Y-%m-%d')} to {data['date'].max().strftime('%Y-%m-%d')}")

    # Create strategy
    if args.strategy == "ma":
        strategy = DoubleMovingAverageStrategy(args.stock, fast_period=args.fast, slow_period=args.slow)
    elif args.strategy == "rsi":
        strategy = RSIStrategy(args.stock)
    elif args.strategy == "macd":
        strategy = MACDStrategy(args.stock)
    else:
        from strategy.base import BuyAndHoldStrategy
        strategy = BuyAndHoldStrategy(args.stock)

    print(f"\nStrategy: {strategy.name}")

    # Create engine and run
    engine = BacktestEngine(
        initial_cash=args.cash,
        verbose=args.verbose,
    )

    result = engine.run(strategy, data, args.stock)

    # Print basic report
    print_backtest_report(result)

    # Calculate and print detailed metrics
    metrics = calculate_all_metrics(
        result.equity_curve['total_value'],
        result.trades
    )
    print_detailed_metrics(metrics)

    # Plot if requested
    if args.plot or args.save_plot:
        import matplotlib.pyplot as plt

        print("\nGenerating plots...")

        # Prepare metrics dict for the report
        report_metrics = {
            'sharpe_ratio': metrics.sharpe_ratio,
            'total_return': metrics.total_return,
            'max_drawdown': metrics.max_drawdown,
            'win_rate': metrics.win_rate,
        }

        fig = create_full_report(
            result.equity_curve['total_value'],
            result.trades,
            title=f"{args.stock} - {strategy.name} Backtest Report",
            metrics=report_metrics
        )

        if args.save_plot:
            fig.savefig(args.save_plot, dpi=150, bbox_inches='tight')
            print(f"Plot saved to: {args.save_plot}")

        if args.plot:
            plt.show()

    return 0


if __name__ == "__main__":
    sys.exit(main())
