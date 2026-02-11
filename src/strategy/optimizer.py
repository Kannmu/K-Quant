"""
Strategy Optimizer - Parameter optimization and backtesting utilities

Provides tools for:
- Parameter grid search
- Walk-forward optimization
- Multi-strategy comparison
- Benchmark comparison
"""

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple, Type, Any
from datetime import datetime
import itertools
import pandas as pd
import numpy as np
from pathlib import Path
import json

from .base import Strategy, Signal, SignalType, Context
from backtest.engine import BacktestEngine, BacktestResult


@dataclass
class ParameterSet:
    """Represents a set of strategy parameters."""
    name: str
    params: Dict[str, Any]
    fitness: float = 0.0
    metrics: Dict = field(default_factory=dict)


@dataclass
class OptimizationResult:
    """Results from parameter optimization."""
    strategy_class: Type[Strategy]
    stock_code: str
    best_params: Dict[str, Any]
    all_results: List[ParameterSet]
    optimization_metric: str


class StrategyOptimizer:
    """
    Strategy parameter optimizer.

    Performs grid search or random search over parameter space
to find optimal strategy parameters.

    Example:
        optimizer = StrategyOptimizer(DoubleMovingAverageStrategy, "601012")
        param_grid = {
            'fast_period': [10, 20, 30],
            'slow_period': [50, 60, 70]
        }
        result = optimizer.grid_search(param_grid, data, metric='sharpe_ratio')
    """

    def __init__(
        self,
        strategy_class: Type[Strategy],
        stock_code: str,
        initial_cash: float = 100000.0,
    ):
        self.strategy_class = strategy_class
        self.stock_code = stock_code
        self.initial_cash = initial_cash

    def grid_search(
        self,
        param_grid: Dict[str, List],
        data: pd.DataFrame,
        metric: str = 'sharpe_ratio',
        verbose: bool = False,
    ) -> OptimizationResult:
        """
        Perform grid search over parameter space.

        Args:
            param_grid: Dict mapping parameter names to lists of values
            data: OHLCV data for backtesting
            metric: Metric to optimize ('sharpe_ratio', 'total_return', 'calmar_ratio')
            verbose: Print progress

        Returns:
            OptimizationResult with best parameters
        """
        # Generate all parameter combinations
        param_names = list(param_grid.keys())
        param_values = list(param_grid.values())
        combinations = list(itertools.product(*param_values))

        if verbose:
            print(f"Testing {len(combinations)} parameter combinations...")

        results = []

        for i, values in enumerate(combinations):
            params = dict(zip(param_names, values))

            if verbose and i % 10 == 0:
                print(f"  Progress: {i}/{len(combinations)} - Testing {params}")

            # Create strategy with these parameters
            try:
                strategy = self._create_strategy(params)

                # Run backtest
                engine = BacktestEngine(initial_cash=self.initial_cash)
                result = engine.run(strategy, data, self.stock_code)

                # Extract metrics
                fitness = self._extract_metric(result, metric)

                param_set = ParameterSet(
                    name=f"{self.strategy_class.__name__}_{i}",
                    params=params,
                    fitness=fitness,
                    metrics={
                        'total_return': result.total_return,
                        'sharpe_ratio': result.sharpe_ratio,
                        'max_drawdown': result.max_drawdown,
                        'total_trades': result.total_trades,
                        'win_rate': result.win_rate,
                    }
                )
                results.append(param_set)

            except Exception as e:
                if verbose:
                    print(f"    Error with params {params}: {e}")
                continue

        # Sort by fitness (descending)
        results.sort(key=lambda x: x.fitness, reverse=True)

        best = results[0] if results else None

        return OptimizationResult(
            strategy_class=self.strategy_class,
            stock_code=self.stock_code,
            best_params=best.params if best else {},
            all_results=results,
            optimization_metric=metric,
        )

    def _create_strategy(self, params: Dict) -> Strategy:
        """Create strategy instance with given parameters."""
        # Always include stock_code as first parameter
        return self.strategy_class(self.stock_code, **params)

    def _extract_metric(self, result: BacktestResult, metric: str) -> float:
        """Extract optimization metric from backtest result."""
        metric_map = {
            'sharpe_ratio': result.sharpe_ratio,
            'total_return': result.total_return,
            'annualized_return': result.annualized_return,
            'calmar_ratio': abs(result.annualized_return / result.max_drawdown)
                if result.max_drawdown != 0 else 0,
            'win_rate': result.win_rate,
            'profit_factor': result.profit_factor,
        }
        return metric_map.get(metric, 0.0)


class StrategyComparator:
    """
    Compare multiple strategies on the same data.

    Example:
        comparator = StrategyComparator()
        comparator.add_strategy("MA20/60", DoubleMovingAverageStrategy("601012"))
        comparator.add_strategy("RSI", RSIStrategy("601012"))
        results = comparator.run_all(data, "601012")
        comparator.print_comparison()
    """

    def __init__(self, initial_cash: float = 100000.0):
        self.initial_cash = initial_cash
        self.strategies: Dict[str, Strategy] = {}
        self.results: Dict[str, BacktestResult] = {}

    def add_strategy(self, name: str, strategy: Strategy):
        """Add a strategy to compare."""
        self.strategies[name] = strategy

    def run_all(
        self,
        data: pd.DataFrame,
        stock_code: str,
        verbose: bool = False,
    ) -> Dict[str, BacktestResult]:
        """Run all strategies and collect results."""
        self.results = {}

        for name, strategy in self.strategies.items():
            if verbose:
                print(f"Running {name}...")

            engine = BacktestEngine(initial_cash=self.initial_cash)
            result = engine.run(strategy, data, stock_code)
            self.results[name] = result

        return self.results

    def print_comparison(self):
        """Print formatted comparison table."""
        if not self.results:
            print("No results to compare. Run run_all() first.")
            return

        print("\n" + "=" * 100)
        print("STRATEGY COMPARISON")
        print("=" * 100)

        # Header
        header = f"{'Strategy':<20} {'Return':>10} {'CAGR':>10} {'Max DD':>10} {'Sharpe':>10} {'Trades':>8} {'Win%':>8}"
        print(header)
        print("-" * 100)

        # Rows
        for name, result in self.results.items():
            row = f"{name:<20} {result.total_return:>10.2f} {result.annualized_return:>10.2f} " \
                  f"{result.max_drawdown:>10.2f} {result.sharpe_ratio:>10.2f} " \
                  f"{result.total_trades:>8} {result.win_rate:>8.2f}"
            print(row)

        print("=" * 100)

    def get_best_strategy(self, metric: str = 'sharpe_ratio') -> Tuple[str, BacktestResult]:
        """Get best performing strategy by metric."""
        if not self.results:
            raise ValueError("No results available")

        best_name = max(
            self.results.keys(),
            key=lambda k: getattr(self.results[k], metric, 0)
        )
        return best_name, self.results[best_name]

    def export_results(self, filepath: str):
        """Export comparison results to JSON."""
        export_data = {}
        for name, result in self.results.items():
            export_data[name] = {
                'total_return': result.total_return,
                'annualized_return': result.annualized_return,
                'max_drawdown': result.max_drawdown,
                'sharpe_ratio': result.sharpe_ratio,
                'total_trades': result.total_trades,
                'win_rate': result.win_rate,
                'profit_factor': result.profit_factor,
            }

        with open(filepath, 'w') as f:
            json.dump(export_data, f, indent=2)
        print(f"Results exported to {filepath}")


def run_benchmark_comparison(
    strategy: Strategy,
    data: pd.DataFrame,
    stock_code: str,
    benchmark_code: str = "000300",  # CSI 300
    initial_cash: float = 100000.0,
) -> Dict:
    """
    Compare strategy against a benchmark (e.g., CSI 300).

    Args:
        strategy: Strategy to test
        data: Stock data
        stock_code: Stock being traded
        benchmark_code: Benchmark index code
        initial_cash: Initial capital

    Returns:
        Dict with comparison metrics
    """
    from ...data_loader import load_stock_data

    # Run strategy backtest
    engine = BacktestEngine(initial_cash=initial_cash)
    strategy_result = engine.run(strategy, data, stock_code)

    # Load benchmark data
    try:
        benchmark_data = load_stock_data(
            benchmark_code,
            data['date'].min().strftime('%Y%m%d'),
            data['date'].max().strftime('%Y%m%d'),
            include_fundamental=False
        )

        # Calculate benchmark returns (buy and hold)
        benchmark_start = benchmark_data['close'].iloc[0]
        benchmark_end = benchmark_data['close'].iloc[-1]
        benchmark_return = (benchmark_end / benchmark_start - 1) * 100

        # Calculate benchmark equity curve
        benchmark_shares = initial_cash / benchmark_start
        benchmark_equity = benchmark_data['close'] * benchmark_shares
        benchmark_equity.index = pd.to_datetime(benchmark_data['date'])

    except Exception as e:
        print(f"Warning: Could not load benchmark data: {e}")
        benchmark_return = 0.0
        benchmark_equity = None

    return {
        'strategy_return': strategy_result.total_return,
        'benchmark_return': benchmark_return,
        'excess_return': strategy_result.total_return - benchmark_return,
        'strategy_sharpe': strategy_result.sharpe_ratio,
        'strategy_max_dd': strategy_result.max_drawdown,
        'strategy_equity': strategy_result.equity_curve['total_value'],
        'benchmark_equity': benchmark_equity,
    }


if __name__ == "__main__":
    # Test optimizer
    import sys
    sys.path.insert(0, '/Users/kannmu/Library/CloudStorage/OneDrive-Personal/Projects/K-Quant/src')

    from data_loader import load_stock_data
    from strategy.templates import DoubleMovingAverageStrategy

    print("Testing Strategy Optimizer with 601012...")
    print("=" * 60)

    # Load data
    data = load_stock_data("601012", "20230101", "20241231", include_fundamental=False)

    # Create optimizer
    optimizer = StrategyOptimizer(DoubleMovingAverageStrategy, "601012")

    # Define parameter grid
    param_grid = {
        'fast_period': [10, 20, 30],
        'slow_period': [50, 60, 70],
    }

    # Run optimization
    result = optimizer.grid_search(
        param_grid=param_grid,
        data=data,
        metric='sharpe_ratio',
        verbose=True
    )

    print(f"\nBest parameters: {result.best_params}")
    print(f"Best fitness ({result.optimization_metric}): "
          f"{result.all_results[0].fitness:.4f}")

    # Test comparator
    print("\n" + "=" * 60)
    print("Testing Strategy Comparator...")

    comparator = StrategyComparator()
    comparator.add_strategy("MA10/50", DoubleMovingAverageStrategy("601012", 10, 50))
    comparator.add_strategy("MA20/60", DoubleMovingAverageStrategy("601012", 20, 60))
    comparator.add_strategy("MA30/70", DoubleMovingAverageStrategy("601012", 30, 70))

    comparator.run_all(data, "601012", verbose=True)
    comparator.print_comparison()

    print("\n✅ Optimizer tests completed!")
