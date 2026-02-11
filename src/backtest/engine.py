"""
Backtest Engine - Event-driven backtesting system

Main engine that orchestrates the backtest loop.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional

import pandas as pd

import sys
from pathlib import Path

# Add src to path for imports
src_path = Path(__file__).parent.parent
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

from strategy.base import Context, Signal, SignalType, Strategy
from backtest.broker import Broker, OrderType
from backtest.portfolio import Portfolio


@dataclass
class BacktestResult:
    """Complete backtest results."""
    strategy_name: str
    start_date: datetime
    end_date: datetime
    initial_cash: float
    final_value: float
    total_return: float  # Percentage
    annualized_return: float  # CAGR, percentage
    max_drawdown: float  # Percentage
    sharpe_ratio: float
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    avg_profit: float
    avg_loss: float
    profit_factor: float
    equity_curve: pd.DataFrame
    trades: pd.DataFrame
    positions_history: pd.DataFrame
    signals: List[Dict] = field(default_factory=list)


class BacktestEngine:
    """
    Event-driven backtesting engine.

    Simulates trading by iterating through historical data bar by bar,
    feeding data to the strategy, and executing signals through the broker.

    Example:
        engine = BacktestEngine(initial_cash=100000)
        result = engine.run(strategy, data)
        print(f"Total return: {result.total_return:.2f}%")
    """

    def __init__(
        self,
        initial_cash: float = 1000000.0,
        commission_rate: float = 0.0003,
        min_commission: float = 5.0,
        stamp_tax_rate: float = 0.0005,
        slippage_rate: float = 0.001,
        max_position_pct: float = 0.2,
        max_positions: int = 10,
        verbose: bool = False,
    ):
        self.initial_cash = initial_cash
        self.verbose = verbose

        # Initialize components
        self.broker = Broker(
            commission_rate=commission_rate,
            min_commission=min_commission,
            stamp_tax_rate=stamp_tax_rate,
            slippage_rate=slippage_rate,
        )
        self.portfolio = Portfolio(
            initial_cash=initial_cash,
            max_position_pct=max_position_pct,
            max_positions=max_positions,
        )

        # State
        self.strategy: Optional[Strategy] = None
        self.data: Optional[pd.DataFrame] = None
        self.current_date: Optional[datetime] = None
        self.signals_executed: List[Dict] = []

    def run(
        self,
        strategy: Strategy,
        data: pd.DataFrame,
        stock_code: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> BacktestResult:
        """
        Run backtest for a strategy on historical data.

        Args:
            strategy: Trading strategy instance
            data: OHLCV DataFrame with date index
            stock_code: Stock code being traded
            start_date: Backtest start date (optional)
            end_date: Backtest end date (optional)

        Returns:
            BacktestResult with performance metrics
        """
        # Initialize
        self.strategy = strategy
        self.data = data.copy()

        # Filter date range
        if start_date:
            self.data = self.data[self.data['date'] >= start_date]
        if end_date:
            self.data = self.data[self.data['date'] <= end_date]

        if len(self.data) == 0:
            raise ValueError("No data available for backtest")

        # Initialize strategy with full data
        strategy.init(data)

        if self.verbose:
            print(f"Starting backtest: {strategy.name}")
            print(f"Date range: {self.data['date'].min()} to {self.data['date'].max()}")
            print(f"Initial cash: {self.initial_cash:,.2f}")

        # Main backtest loop
        for idx in range(len(self.data)):
            row = self.data.iloc[idx]
            self.current_date = row['date']
            current_price = row['close']

            # Build context
            context = self._build_context(idx, current_price, stock_code)

            # Get signal from strategy
            signal = strategy.next(context)

            # Execute signal if present
            if signal:
                self._execute_signal(signal, current_price, stock_code)
                self.signals_executed.append({
                    'date': self.current_date,
                    'signal': signal.signal.name,
                    'price': signal.price,
                    'reason': signal.reason,
                })

            # Take portfolio snapshot with current prices
            prices = {stock_code: current_price}
            for code in self.portfolio.positions.keys():
                if code != stock_code:
                    # For simplicity, use last known price for other positions
                    # In real implementation, would need price data for all holdings
                    prices[code] = self.portfolio.positions[code].avg_cost

            self.portfolio.take_snapshot(self.current_date, prices)

        # Calculate results
        return self._calculate_results()

    def _build_context(self, idx: int, current_price: float, stock_code: str) -> Context:
        """Build context object for strategy."""
        # Historical data up to current bar
        hist_data = self.data.iloc[:idx + 1]

        # Current positions
        positions = {
            code: pos.quantity
            for code, pos in self.portfolio.positions.items()
        }

        # Current prices (simplified - single stock)
        current_prices = {stock_code: current_price}

        return Context(
            date=self.current_date,
            current_prices=current_prices,
            portfolio_value=self.portfolio.get_total_value(),
            cash=self.portfolio.cash,
            positions=positions,
            data=hist_data,
            indicators=self.strategy.indicators if self.strategy else {},
        )

    def _execute_signal(self, signal: Signal, current_price: float, stock_code: str):
        """Execute a trading signal."""
        if signal.signal == SignalType.HOLD:
            return

        # Determine order type
        order_type = OrderType.BUY if signal.signal == SignalType.BUY else OrderType.SELL

        # Determine quantity
        if signal.quantity > 0:
            quantity = signal.quantity
        elif order_type == OrderType.SELL:
            # Sell: close all positions
            position = self.portfolio.get_position(stock_code)
            quantity = position.quantity if position else 0
        else:
            # Buy: use 20% of portfolio value
            position_value = self.portfolio.get_total_value() * 0.2
            quantity = int(position_value / current_price)

        # Check position limits for buys
        if order_type == OrderType.BUY:
            can_buy, reason = self.portfolio.can_buy(stock_code, quantity, current_price)
            if not can_buy:
                if self.verbose:
                    print(f"  [Signal Rejected] {reason}")
                return

        # Check T+1 and holdings for sells
        if order_type == OrderType.SELL:
            can_sell, reason = self.portfolio.can_sell(stock_code, quantity, self.current_date)
            if not can_sell:
                if self.verbose:
                    print(f"  [Signal Rejected] {reason}")
                return

        # Prepare holdings info for broker
        current_holdings = {
            code: pos.quantity for code, pos in self.portfolio.positions.items()
        }
        holdings_buy_dates = {
            code: pos.buy_dates for code, pos in self.portfolio.positions.items()
        }

        # Submit order
        result = self.broker.submit_order(
            stock_code=stock_code,
            order_type=order_type,
            quantity=quantity,
            price=current_price,
            date=self.current_date,
            available_cash=self.portfolio.cash,
            current_holdings=current_holdings,
            holdings_buy_dates=holdings_buy_dates,
        )

        if result.success:
            # Update portfolio
            self.portfolio.update_from_fill(result)

            if self.verbose:
                action = "BOUGHT" if order_type == OrderType.BUY else "SOLD"
                print(f"  {action} {result.actual_quantity} shares @ {result.actual_price:.2f} "
                      f"(Total: {result.total_cost:,.2f})")

            # Notify strategy
            self.strategy.on_trade(signal, result.actual_price, result.actual_quantity)
        else:
            if self.verbose:
                print(f"  [Order Failed] {result.message}")

    def _calculate_results(self) -> BacktestResult:
        """Calculate backtest performance metrics."""
        equity_curve = self.portfolio.get_equity_curve()

        if len(equity_curve) == 0:
            raise ValueError("No equity data generated")

        # Basic metrics
        initial_value = self.initial_cash
        final_value = equity_curve['total_value'].iloc[-1]
        total_return = (final_value - initial_value) / initial_value * 100

        # Annualized return (CAGR)
        days = (equity_curve.index[-1] - equity_curve.index[0]).days
        years = days / 365.25
        if years > 0:
            annualized_return = ((final_value / initial_value) ** (1 / years) - 1) * 100
        else:
            annualized_return = 0

        # Max drawdown
        peak = equity_curve['total_value'].cummax()
        drawdown = (equity_curve['total_value'] - peak) / peak * 100
        max_drawdown = drawdown.min()

        # Sharpe ratio (simplified, assume 3% risk-free rate)
        if len(equity_curve) > 1:
            daily_returns = equity_curve['total_value'].pct_change().dropna()
            excess_returns = daily_returns - 0.03 / 252  # Daily risk-free rate
            if daily_returns.std() > 1e-10:  # Avoid division by near-zero
                sharpe_ratio = (excess_returns.mean() / daily_returns.std()) * (252 ** 0.5)
            else:
                sharpe_ratio = 0.0
        else:
            sharpe_ratio = 0.0

        # Trade statistics
        trades_df = self.broker.get_trade_history()
        if len(trades_df) > 0:
            total_trades = len(trades_df)

            # Calculate profit/loss per trade
            profits = []
            buy_trades = trades_df[trades_df['action'] == 'buy']
            sell_trades = trades_df[trades_df['action'] == 'sell']

            # Simplified: assume alternating buy/sell for same quantity
            winning_trades = 0
            losing_trades = 0
            total_profit = 0
            total_loss = 0

            for _, sell in sell_trades.iterrows():
                # Find matching buy (simplified)
                matching_buys = buy_trades[buy_trades['date'] <= sell['date']]
                if len(matching_buys) > 0:
                    buy = matching_buys.iloc[-1]
                    profit = sell['price'] - buy['price']
                    if profit > 0:
                        winning_trades += 1
                        total_profit += profit
                    else:
                        losing_trades += 1
                        total_loss += abs(profit)

            win_rate = winning_trades / (winning_trades + losing_trades) * 100 if (winning_trades + losing_trades) > 0 else 0
            avg_profit = total_profit / winning_trades if winning_trades > 0 else 0
            avg_loss = total_loss / losing_trades if losing_trades > 0 else 0
            profit_factor = total_profit / total_loss if total_loss > 0 else float('inf')
        else:
            total_trades = winning_trades = losing_trades = 0
            win_rate = avg_profit = avg_loss = profit_factor = 0

        # Build positions history
        positions_data = []
        for snap in self.portfolio.snapshots:
            for code, pos in snap.positions.items():
                # Get price from equity curve for this date (approximation)
                positions_data.append({
                    'date': snap.date,
                    'stock_code': code,
                    'quantity': pos.quantity,
                    'avg_cost': pos.avg_cost,
                    'market_value': pos.quantity * pos.avg_cost,  # Simplified
                })
        positions_df = pd.DataFrame(positions_data)

        return BacktestResult(
            strategy_name=self.strategy.name if self.strategy else "Unknown",
            start_date=equity_curve.index[0],
            end_date=equity_curve.index[-1],
            initial_cash=initial_value,
            final_value=final_value,
            total_return=total_return,
            annualized_return=annualized_return,
            max_drawdown=max_drawdown,
            sharpe_ratio=sharpe_ratio,
            total_trades=total_trades,
            winning_trades=winning_trades,
            losing_trades=losing_trades,
            win_rate=win_rate,
            avg_profit=avg_profit,
            avg_loss=avg_loss,
            profit_factor=profit_factor,
            equity_curve=equity_curve,
            trades=trades_df,
            positions_history=positions_df,
            signals=self.signals_executed,
        )


def print_backtest_report(result: BacktestResult):
    """Print formatted backtest report."""
    print("\n" + "=" * 60)
    print(f"BACKTEST REPORT: {result.strategy_name}")
    print("=" * 60)
    print(f"Period: {result.start_date.strftime('%Y-%m-%d')} to {result.end_date.strftime('%Y-%m-%d')}")
    print(f"Initial Capital: {result.initial_cash:,.2f}")
    print(f"Final Value: {result.final_value:,.2f}")
    print("-" * 60)
    print("PERFORMANCE METRICS:")
    print(f"  Total Return:      {result.total_return:>10.2f}%")
    print(f"  Annualized Return: {result.annualized_return:>10.2f}%")
    print(f"  Max Drawdown:      {result.max_drawdown:>10.2f}%")
    print(f"  Sharpe Ratio:      {result.sharpe_ratio:>10.2f}")
    print("-" * 60)
    print("TRADE STATISTICS:")
    print(f"  Total Trades:      {result.total_trades:>10}")
    print(f"  Winning Trades:    {result.winning_trades:>10}")
    print(f"  Losing Trades:     {result.losing_trades:>10}")
    print(f"  Win Rate:          {result.win_rate:>10.2f}%")
    print(f"  Avg Profit:        {result.avg_profit:>10.2f}")
    print(f"  Avg Loss:          {result.avg_loss:>10.2f}")
    print(f"  Profit Factor:     {result.profit_factor:>10.2f}")
    print("=" * 60)


if __name__ == "__main__":
    import sys
    sys.path.insert(0, '/Users/kannmu/Library/CloudStorage/OneDrive-Personal/Projects/K-Quant/src')

    from data_loader import load_stock_data
    from strategy.templates import DoubleMovingAverageStrategy

    print("Testing BacktestEngine with 601012 (隆基绿能)...")
    print("=" * 60)

    # Load data
    data = load_stock_data("601012", "20230101", "20241231", include_fundamental=False)
    print(f"\nLoaded {len(data)} rows of data")
    print(f"Date range: {data['date'].min()} to {data['date'].max()}")

    # Create strategy
    strategy = DoubleMovingAverageStrategy("601012", fast_period=20, slow_period=60)

    # Create engine and run
    engine = BacktestEngine(
        initial_cash=100000,
        verbose=True,
    )

    result = engine.run(strategy, data, "601012")

    # Print report
    print_backtest_report(result)

    # Show equity curve sample
    print("\nEquity Curve (first 5 and last 5):")
    print(result.equity_curve.head())
    print("...")
    print(result.equity_curve.tail())

    print("\n✅ Backtest engine test completed!")
