"""
Metrics Module - Performance calculation utilities

Calculates various trading performance metrics from backtest results.
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple


@dataclass
class PerformanceMetrics:
    """Complete performance metrics for a backtest."""
    # Returns
    total_return: float  # Total return percentage
    annualized_return: float  # CAGR
    volatility: float  # Annualized volatility

    # Risk
    max_drawdown: float  # Maximum drawdown percentage
    max_drawdown_duration: int  # Longest drawdown period in days
    calmar_ratio: float  # CAGR / Max DD

    # Risk-adjusted returns
    sharpe_ratio: float
    sortino_ratio: float

    # Trade statistics
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    avg_profit: float
    avg_loss: float
    profit_factor: float
    expectancy: float

    # Other
    beta: float  # Market beta (if benchmark provided)
    alpha: float  # Jensen's alpha
    information_ratio: float


def calculate_drawdown_series(equity: pd.Series) -> pd.Series:
    """Calculate drawdown series from equity curve."""
    peak = equity.cummax()
    drawdown = (equity - peak) / peak * 100
    return drawdown


def calculate_max_drawdown_duration(drawdown: pd.Series) -> int:
    """Calculate longest drawdown duration in days."""
    is_drawdown = drawdown < 0
    if not is_drawdown.any():
        return 0

    # Find drawdown periods
    changes = is_drawdown.astype(int).diff()
    start_dates = changes[changes == 1].index
    end_dates = changes[changes == -1].index

    # Handle open drawdown at end
    if len(start_dates) > len(end_dates):
        end_dates = end_dates.append(pd.Index([drawdown.index[-1]]))

    if len(start_dates) == 0:
        return 0

    durations = [(end - start).days for start, end in zip(start_dates, end_dates)]
    return max(durations) if durations else 0


def calculate_returns_metrics(equity: pd.Series, risk_free_rate: float = 0.03) -> Dict:
    """Calculate return-based metrics."""
    daily_returns = equity.pct_change().dropna()

    if len(daily_returns) < 2:
        return {
            'total_return': 0.0,
            'annualized_return': 0.0,
            'volatility': 0.0,
        }

    # Total return
    total_return = (equity.iloc[-1] / equity.iloc[0] - 1) * 100

    # Annualized return (CAGR)
    years = len(equity) / 252  # Trading days per year
    annualized_return = ((equity.iloc[-1] / equity.iloc[0]) ** (1 / years) - 1) * 100 if years > 0 else 0

    # Volatility (annualized)
    volatility = daily_returns.std() * np.sqrt(252) * 100

    return {
        'total_return': total_return,
        'annualized_return': annualized_return,
        'volatility': volatility,
        'daily_returns': daily_returns,
    }


def calculate_risk_metrics(equity: pd.Series, annualized_return: float,
                          daily_returns: pd.Series) -> Dict:
    """Calculate risk-based metrics."""
    # Max drawdown
    drawdown = calculate_drawdown_series(equity)
    max_drawdown = drawdown.min()
    max_drawdown_duration = calculate_max_drawdown_duration(drawdown)

    # Calmar ratio
    calmar_ratio = abs(annualized_return / max_drawdown) if max_drawdown != 0 else 0

    # Sharpe ratio
    excess_returns = daily_returns - 0.03 / 252  # Risk-free rate
    sharpe_ratio = (excess_returns.mean() / daily_returns.std() * np.sqrt(252)) \
                   if daily_returns.std() > 0 else 0

    # Sortino ratio (downside deviation)
    downside_returns = daily_returns[daily_returns < 0]
    downside_std = downside_returns.std() * np.sqrt(252)
    sortino_ratio = ((daily_returns.mean() * 252 - 0.03) / downside_std) \
                    if downside_std > 0 else 0

    return {
        'max_drawdown': max_drawdown,
        'max_drawdown_duration': max_drawdown_duration,
        'calmar_ratio': calmar_ratio,
        'sharpe_ratio': sharpe_ratio,
        'sortino_ratio': sortino_ratio,
        'drawdown_series': drawdown,
    }


def calculate_trade_metrics(trades: pd.DataFrame) -> Dict:
    """Calculate trade-based metrics."""
    if len(trades) == 0:
        return {
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'win_rate': 0.0,
            'avg_profit': 0.0,
            'avg_loss': 0.0,
            'profit_factor': 0.0,
            'expectancy': 0.0,
        }

    total_trades = len(trades)

    # Calculate P&L per trade (simplified matching)
    buy_trades = trades[trades['action'] == 'buy']
    sell_trades = trades[trades['action'] == 'sell']

    profits = []
    for _, sell in sell_trades.iterrows():
        matching_buys = buy_trades[buy_trades['date'] <= sell['date']]
        if len(matching_buys) > 0:
            buy = matching_buys.iloc[-1]
            profit = (sell['price'] - buy['price']) * sell['quantity']
            profits.append(profit)

    if len(profits) == 0:
        return {
            'total_trades': total_trades,
            'winning_trades': 0,
            'losing_trades': 0,
            'win_rate': 0.0,
            'avg_profit': 0.0,
            'avg_loss': 0.0,
            'profit_factor': 0.0,
            'expectancy': 0.0,
        }

    winning_trades = sum(1 for p in profits if p > 0)
    losing_trades = sum(1 for p in profits if p <= 0)
    win_rate = winning_trades / len(profits) * 100 if profits else 0

    winning_profits = [p for p in profits if p > 0]
    losing_profits = [p for p in profits if p <= 0]

    avg_profit = np.mean(winning_profits) if winning_profits else 0
    avg_loss = np.mean(losing_profits) if losing_profits else 0

    total_profit = sum(winning_profits)
    total_loss = abs(sum(losing_profits))
    profit_factor = total_profit / total_loss if total_loss > 0 else float('inf')

    # Expectancy = (Win% * Avg Win) - (Loss% * Avg Loss)
    win_pct = winning_trades / len(profits) if profits else 0
    loss_pct = losing_trades / len(profits) if profits else 0
    expectancy = (win_pct * avg_profit) - (loss_pct * abs(avg_loss))

    return {
        'total_trades': total_trades,
        'winning_trades': winning_trades,
        'losing_trades': losing_trades,
        'win_rate': win_rate,
        'avg_profit': avg_profit,
        'avg_loss': avg_loss,
        'profit_factor': profit_factor,
        'expectancy': expectancy,
    }


def calculate_all_metrics(equity: pd.Series, trades: pd.DataFrame,
                         benchmark: Optional[pd.Series] = None) -> PerformanceMetrics:
    """
    Calculate all performance metrics.

    Args:
        equity: Equity curve series
        trades: Trade history DataFrame
        benchmark: Optional benchmark equity series

    Returns:
        PerformanceMetrics object
    """
    # Return metrics
    returns_metrics = calculate_returns_metrics(equity)

    # Risk metrics
    risk_metrics = calculate_risk_metrics(
        equity,
        returns_metrics['annualized_return'],
        returns_metrics['daily_returns']
    )

    # Trade metrics
    trade_metrics = calculate_trade_metrics(trades)

    # Beta and Alpha (if benchmark provided)
    beta = 0.0
    alpha = 0.0
    information_ratio = 0.0

    if benchmark is not None and len(benchmark) == len(equity):
        strategy_returns = equity.pct_change().dropna()
        benchmark_returns = benchmark.pct_change().dropna()

        # Beta
        covariance = strategy_returns.cov(benchmark_returns)
        benchmark_variance = benchmark_returns.var()
        beta = covariance / benchmark_variance if benchmark_variance > 0 else 0

        # Alpha (Jensen's)
        alpha = (returns_metrics['annualized_return'] / 100 - 0.03 -
                beta * (benchmark_returns.mean() * 252 - 0.03)) * 100

        # Information Ratio
        tracking_error = (strategy_returns - benchmark_returns).std() * np.sqrt(252)
        information_ratio = ((returns_metrics['annualized_return'] -
                            benchmark_returns.mean() * 252 * 100) / (tracking_error * 100)) \
                           if tracking_error > 0 else 0

    return PerformanceMetrics(
        total_return=returns_metrics['total_return'],
        annualized_return=returns_metrics['annualized_return'],
        volatility=returns_metrics['volatility'],
        max_drawdown=risk_metrics['max_drawdown'],
        max_drawdown_duration=risk_metrics['max_drawdown_duration'],
        calmar_ratio=risk_metrics['calmar_ratio'],
        sharpe_ratio=risk_metrics['sharpe_ratio'],
        sortino_ratio=risk_metrics['sortino_ratio'],
        total_trades=trade_metrics['total_trades'],
        winning_trades=trade_metrics['winning_trades'],
        losing_trades=trade_metrics['losing_trades'],
        win_rate=trade_metrics['win_rate'],
        avg_profit=trade_metrics['avg_profit'],
        avg_loss=trade_metrics['avg_loss'],
        profit_factor=trade_metrics['profit_factor'],
        expectancy=trade_metrics['expectancy'],
        beta=beta,
        alpha=alpha,
        information_ratio=information_ratio,
    )


def print_detailed_metrics(metrics: PerformanceMetrics):
    """Print detailed performance metrics."""
    print("\n" + "=" * 60)
    print("DETAILED PERFORMANCE METRICS")
    print("=" * 60)

    print("\n📈 RETURNS:")
    print(f"  Total Return:        {metrics.total_return:>10.2f}%")
    print(f"  Annualized (CAGR):   {metrics.annualized_return:>10.2f}%")
    print(f"  Volatility:          {metrics.volatility:>10.2f}%")

    print("\n⚠️  RISK:")
    print(f"  Max Drawdown:        {metrics.max_drawdown:>10.2f}%")
    print(f"  Max DD Duration:     {metrics.max_drawdown_duration:>10} days")
    print(f"  Calmar Ratio:        {metrics.calmar_ratio:>10.2f}")

    print("\n📊 RISK-ADJUSTED RETURNS:")
    print(f"  Sharpe Ratio:        {metrics.sharpe_ratio:>10.2f}")
    print(f"  Sortino Ratio:       {metrics.sortino_ratio:>10.2f}")

    print("\n💰 TRADE STATISTICS:")
    print(f"  Total Trades:        {metrics.total_trades:>10}")
    print(f"  Winning Trades:      {metrics.winning_trades:>10}")
    print(f"  Losing Trades:       {metrics.losing_trades:>10}")
    print(f"  Win Rate:            {metrics.win_rate:>10.2f}%")
    print(f"  Average Profit:      {metrics.avg_profit:>10.2f}")
    print(f"  Average Loss:        {metrics.avg_loss:>10.2f}")
    print(f"  Profit Factor:       {metrics.profit_factor:>10.2f}")
    print(f"  Expectancy:          {metrics.expectancy:>10.2f}")

    if metrics.beta != 0:
        print("\n📉 BENCHMARK RELATIVE:")
        print(f"  Beta:                {metrics.beta:>10.2f}")
        print(f"  Alpha:               {metrics.alpha:>10.2f}%")
        print(f"  Information Ratio:   {metrics.information_ratio:>10.2f}")

    print("=" * 60)


if __name__ == "__main__":
    # Test metrics calculation
    import sys
    sys.path.insert(0, '/Users/kannmu/Library/CloudStorage/OneDrive-Personal/Projects/K-Quant/src')

    from data_loader import load_stock_data
    from strategy.templates import DoubleMovingAverageStrategy
    from backtest.engine import BacktestEngine

    print("Testing Metrics Module with 601012...")

    data = load_stock_data("601012", "20240101", "20241231")
    strategy = DoubleMovingAverageStrategy("601012")
    engine = BacktestEngine(initial_cash=100000)
    result = engine.run(strategy, data, "601012")

    metrics = calculate_all_metrics(
        result.equity_curve['total_value'],
        result.trades
    )

    print_detailed_metrics(metrics)
