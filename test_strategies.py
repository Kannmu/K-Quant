#!/usr/bin/env python3
"""
Strategy Testing Script - Quick testing and comparison of multiple strategies

Usage:
    python test_strategies.py --stock 601012 --start 20240101 --end 20241231
    python test_strategies.py --stock 601012 --compare-all
    python test_strategies.py --stock 601012 --optimize ma
"""

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from data_loader import load_stock_data
from backtest.engine import BacktestEngine, print_backtest_report
from analysis.metrics import calculate_all_metrics, print_detailed_metrics
from analysis.plotting import create_full_report

# Import all strategies
from strategy.templates import (
    DoubleMovingAverageStrategy,
    RSIStrategy,
    MACDStrategy,
)
from strategy.base import BuyAndHoldStrategy
from strategy.advanced_templates import (
    TrendFollowingStrategy,
    MeanReversionStrategy,
    VolatilityBreakoutStrategy,
    MultiFactorStrategy,
    AdaptiveMomentumStrategy,
)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Strategy Testing Tool")
    parser.add_argument("--stock", type=str, default="601012",
                        help="Stock code (default: 601012)")
    parser.add_argument("--start", type=str, default="20240101",
                        help="Start date (YYYYMMDD)")
    parser.add_argument("--end", type=str, default="20241231",
                        help="End date (YYYYMMDD)")
    parser.add_argument("--cash", type=float, default=100000,
                        help="Initial cash (default: 100000)")

    # Strategy selection
    parser.add_argument("--strategy", type=str, default="ma",
                        choices=["ma", "rsi", "macd", "buyhold", "trend",
                                "meanrev", "volbreak", "multifactor", "adaptive"],
                        help="Strategy to test")

    # Compare all strategies
    parser.add_argument("--compare-all", action="store_true",
                        help="Compare all strategies")

    # Optimization
    parser.add_argument("--optimize", type=str,
                        choices=["ma", "rsi"],
                        help="Optimize strategy parameters")

    # Plotting
    parser.add_argument("--save-plot", type=str,
                        help="Save comparison plot to file")

    return parser.parse_args()


def get_strategy(name: str, stock_code: str):
    """Get strategy instance by name."""
    strategies = {
        "ma": DoubleMovingAverageStrategy(stock_code, 20, 60),
        "rsi": RSIStrategy(stock_code, period=14),
        "macd": MACDStrategy(stock_code),
        "buyhold": BuyAndHoldStrategy(stock_code),
        "trend": TrendFollowingStrategy(stock_code),
        "meanrev": MeanReversionStrategy(stock_code),
        "volbreak": VolatilityBreakoutStrategy(stock_code),
        "multifactor": MultiFactorStrategy(stock_code),
        "adaptive": AdaptiveMomentumStrategy(stock_code),
    }
    return strategies.get(name)


def run_single_backtest(strategy, data, stock_code, initial_cash, verbose=True):
    """Run single backtest and return results."""
    engine = BacktestEngine(initial_cash=initial_cash)
    result = engine.run(strategy, data, stock_code)

    if verbose:
        print_backtest_report(result)
        metrics = calculate_all_metrics(
            result.equity_curve['total_value'],
            result.trades
        )
        print_detailed_metrics(metrics)

    return result


def compare_all_strategies(stock_code, data, initial_cash, save_plot=None):
    """Compare all available strategies."""
    strategy_names = [
        ("MA Crossover", "ma"),
        ("RSI", "rsi"),
        ("MACD", "macd"),
        ("Buy & Hold", "buyhold"),
        ("Trend Following", "trend"),
        ("Mean Reversion", "meanrev"),
        ("Volatility Breakout", "volbreak"),
        ("Multi-Factor", "multifactor"),
        ("Adaptive Momentum", "adaptive"),
    ]

    results = {}

    print("\n" + "=" * 100)
    print("STRATEGY COMPARISON")
    print("=" * 100)
    print(f"{'Strategy':<25} {'Return':>10} {'CAGR':>10} {'Max DD':>10} "
          f"{'Sharpe':>10} {'Trades':>8} {'Win%':>8}")
    print("-" * 100)

    for name, key in strategy_names:
        try:
            strategy = get_strategy(key, stock_code)
            engine = BacktestEngine(initial_cash=initial_cash)
            result = engine.run(strategy, data, stock_code)
            results[name] = result

            print(f"{name:<25} {result.total_return:>10.2f} "
                  f"{result.annualized_return:>10.2f} {result.max_drawdown:>10.2f} "
                  f"{result.sharpe_ratio:>10.2f} {result.total_trades:>8} "
                  f"{result.win_rate:>8.2f}")

        except Exception as e:
            print(f"{name:<25} Error: {e}")

    print("=" * 100)

    # Find best strategy by Sharpe ratio
    if results:
        best_sharpe = max(results.items(), key=lambda x: x[1].sharpe_ratio)
        best_return = max(results.items(), key=lambda x: x[1].total_return)

        print(f"\n🏆 Best Sharpe Ratio: {best_sharpe[0]} ({best_sharpe[1].sharpe_ratio:.2f})")
        print(f"🏆 Best Total Return: {best_return[0]} ({best_return[1].total_return:.2f}%)")

        # Save plot if requested
        if save_plot:
            import matplotlib.pyplot as plt

            fig, axes = plt.subplots(2, 1, figsize=(14, 10))

            # Equity curves
            ax1 = axes[0]
            for name, result in results.items():
                equity = result.equity_curve['total_value']
                ax1.plot(equity.index, equity.values, label=name, linewidth=1.5)

            ax1.set_title(f"Strategy Comparison - {stock_code}", fontweight="bold")
            ax1.set_xlabel("Date")
            ax1.set_ylabel("Portfolio Value")
            ax1.legend(loc="upper left", fontsize=8)
            ax1.grid(True, alpha=0.3)
            ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"¥{x:,.0f}"))

            # Drawdown comparison
            ax2 = axes[1]
            for name, result in results.items():
                equity = result.equity_curve['total_value']
                peak = equity.cummax()
                drawdown = (equity - peak) / peak * 100
                ax2.fill_between(drawdown.index, drawdown.values, 0, alpha=0.2)
                ax2.plot(drawdown.index, drawdown.values, label=name, linewidth=1)

            ax2.set_title("Drawdown Comparison", fontweight="bold")
            ax2.set_xlabel("Date")
            ax2.set_ylabel("Drawdown (%)")
            ax2.grid(True, alpha=0.3)

            plt.tight_layout()
            fig.savefig(save_plot, dpi=150, bbox_inches='tight')
            print(f"\n📊 Comparison plot saved to: {save_plot}")

    return results


def optimize_strategy(stock_code, data, strategy_name, initial_cash):
    """Run parameter optimization for a strategy."""
    from strategy.optimizer import StrategyOptimizer

    print(f"\n🔧 Optimizing {strategy_name.upper()} strategy parameters...")
    print("=" * 60)

    if strategy_name == "ma":
        optimizer = StrategyOptimizer(DoubleMovingAverageStrategy, stock_code, initial_cash)
        param_grid = {
            'fast_period': [5, 10, 15, 20, 25, 30],
            'slow_period': [40, 50, 60, 70, 80],
        }
    elif strategy_name == "rsi":
        optimizer = StrategyOptimizer(RSIStrategy, stock_code, initial_cash)
        param_grid = {
            'period': [7, 14, 21],
            'oversold': [20, 25, 30, 35],
            'overbought': [65, 70, 75, 80],
        }
    else:
        print(f"Optimization not supported for {strategy_name}")
        return None

    result = optimizer.grid_search(
        param_grid=param_grid,
        data=data,
        metric='sharpe_ratio',
        verbose=True
    )

    print(f"\n✅ Optimization Complete!")
    print(f"Best parameters: {result.best_params}")
    print(f"Best Sharpe Ratio: {result.all_results[0].fitness:.4f}")

    # Show top 5 results
    print("\nTop 5 Parameter Sets:")
    print("-" * 80)
    for i, param_set in enumerate(result.all_results[:5], 1):
        print(f"{i}. Params: {param_set.params}")
        print(f"   Sharpe: {param_set.fitness:.4f} | "
              f"Return: {param_set.metrics['total_return']:.2f}% | "
              f"Max DD: {param_set.metrics['max_drawdown']:.2f}%")

    return result


def main():
    """Main entry point."""
    args = parse_args()

    print(f"\n{'='*60}")
    print("K-Quant Strategy Testing Tool")
    print(f"{'='*60}\n")

    # Load data
    print(f"Loading data for {args.stock}...")
    try:
        data = load_stock_data(args.stock, args.start, args.end, include_fundamental=False)
        print(f"  Loaded {len(data)} rows")
        print(f"  Date range: {data['date'].min().strftime('%Y-%m-%d')} to "
              f"{data['date'].max().strftime('%Y-%m-%d')}")
    except Exception as e:
        print(f"Error loading data: {e}")
        return 1

    # Handle different modes
    if args.compare_all:
        compare_all_strategies(args.stock, data, args.cash, args.save_plot)

    elif args.optimize:
        optimize_strategy(args.stock, data, args.optimize, args.cash)

    else:
        # Single strategy test
        strategy = get_strategy(args.strategy, args.stock)
        if strategy:
            print(f"\nTesting strategy: {strategy.name}")
            result = run_single_backtest(strategy, data, args.stock, args.cash)

            if args.save_plot:
                metrics = calculate_all_metrics(
                    result.equity_curve['total_value'],
                    result.trades
                )
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
                fig.savefig(args.save_plot, dpi=150, bbox_inches='tight')
                print(f"\n📊 Plot saved to: {args.save_plot}")
        else:
            print(f"Unknown strategy: {args.strategy}")
            return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
