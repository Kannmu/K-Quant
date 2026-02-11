"""
Strategy Templates - Common trading strategies

Pre-built strategies for testing and reference.
"""

from typing import Optional

import pandas as pd

from .base import Context, Signal, SignalType, Strategy


class DoubleMovingAverageStrategy(Strategy):
    """
    Double Moving Average Crossover Strategy

    Buy when fast MA crosses above slow MA (golden cross).
    Sell when fast MA crosses below slow MA (death cross).

    Parameters:
        stock_code: Stock to trade
        fast_period: Fast MA period (default 20)
        slow_period: Slow MA period (default 60)
    """

    def __init__(
        self,
        stock_code: str,
        fast_period: int = 20,
        slow_period: int = 60,
    ):
        super().__init__(f"DoubleMA_{fast_period}_{slow_period}")
        self.stock_code = stock_code
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.fast_ma = None
        self.slow_ma = None
        self.in_position = False

    def _calculate_indicators(self):
        """Calculate fast and slow moving averages."""
        close = self.data['close']
        self.fast_ma = close.rolling(window=self.fast_period).mean()
        self.slow_ma = close.rolling(window=self.slow_period).mean()

        # Store in indicators dict
        self.indicators['fast_ma'] = self.fast_ma
        self.indicators['slow_ma'] = self.slow_ma

    def next(self, context: Context) -> Optional[Signal]:
        """
        Generate trading signals based on MA crossover.

        Returns BUY on golden cross, SELL on death cross.
        """
        # Need enough data for both MAs
        idx = len(context.data) - 1
        if idx < self.slow_period:
            return None

        # Get current and previous MA values
        fast_curr = self.fast_ma.iloc[idx]
        fast_prev = self.fast_ma.iloc[idx - 1]
        slow_curr = self.slow_ma.iloc[idx]
        slow_prev = self.slow_ma.iloc[idx - 1]

        # Check for valid values (not NaN)
        if pd.isna(fast_curr) or pd.isna(slow_curr):
            return None

        # Golden cross: fast crosses above slow
        golden_cross = fast_prev <= slow_prev and fast_curr > slow_curr

        # Death cross: fast crosses below slow
        death_cross = fast_prev >= slow_prev and fast_curr < slow_curr

        current_price = context.get_price(self.stock_code)
        has_position = context.has_position(self.stock_code)

        # Buy signal: golden cross and not in position
        if golden_cross and not has_position:
            return Signal(
                stock_code=self.stock_code,
                signal=SignalType.BUY,
                date=context.date,
                price=current_price,
                reason=f"Golden Cross: MA{self.fast_period}({fast_curr:.2f}) > MA{self.slow_period}({slow_curr:.2f})"
            )

        # Sell signal: death cross and in position
        if death_cross and has_position:
            return Signal(
                stock_code=self.stock_code,
                signal=SignalType.SELL,
                date=context.date,
                price=current_price,
                reason=f"Death Cross: MA{self.fast_period}({fast_curr:.2f}) < MA{self.slow_period}({slow_curr:.2f})"
            )

        return None


class RSIStrategy(Strategy):
    """
    RSI (Relative Strength Index) Strategy

    Buy when RSI is oversold (below threshold).
    Sell when RSI is overbought (above threshold).

    Parameters:
        stock_code: Stock to trade
        period: RSI calculation period (default 14)
        oversold: Oversold threshold (default 30)
        overbought: Overbought threshold (default 70)
    """

    def __init__(
        self,
        stock_code: str,
        period: int = 14,
        oversold: float = 30.0,
        overbought: float = 70.0,
    ):
        super().__init__(f"RSI_{period}")
        self.stock_code = stock_code
        self.period = period
        self.oversold = oversold
        self.overbought = overbought
        self.rsi = None

    def _calculate_rsi(self, close: pd.Series, period: int) -> pd.Series:
        """Calculate RSI indicator."""
        delta = close.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def _calculate_indicators(self):
        """Calculate RSI indicator."""
        self.rsi = self._calculate_rsi(self.data['close'], self.period)
        self.indicators['rsi'] = self.rsi

    def next(self, context: Context) -> Optional[Signal]:
        """Generate signals based on RSI levels."""
        idx = len(context.data) - 1
        if idx < self.period:
            return None

        rsi_curr = self.rsi.iloc[idx]
        rsi_prev = self.rsi.iloc[idx - 1]

        if pd.isna(rsi_curr):
            return None

        current_price = context.get_price(self.stock_code)
        has_position = context.has_position(self.stock_code)

        # Buy: RSI crosses above oversold level (leaving oversold)
        if rsi_prev <= self.oversold and rsi_curr > self.oversold and not has_position:
            return Signal(
                stock_code=self.stock_code,
                signal=SignalType.BUY,
                date=context.date,
                price=current_price,
                reason=f"RSI left oversold: {rsi_curr:.1f} (was {rsi_prev:.1f})"
            )

        # Sell: RSI crosses below overbought level (leaving overbought)
        if rsi_prev >= self.overbought and rsi_curr < self.overbought and has_position:
            return Signal(
                stock_code=self.stock_code,
                signal=SignalType.SELL,
                date=context.date,
                price=current_price,
                reason=f"RSI left overbought: {rsi_curr:.1f} (was {rsi_prev:.1f})"
            )

        return None


class MACDStrategy(Strategy):
    """
    MACD (Moving Average Convergence Divergence) Strategy

    Buy when MACD line crosses above signal line.
    Sell when MACD line crosses below signal line.

    Parameters:
        stock_code: Stock to trade
        fast: Fast EMA period (default 12)
        slow: Slow EMA period (default 26)
        signal: Signal line period (default 9)
    """

    def __init__(
        self,
        stock_code: str,
        fast: int = 12,
        slow: int = 26,
        signal: int = 9,
    ):
        super().__init__(f"MACD_{fast}_{slow}_{signal}")
        self.stock_code = stock_code
        self.fast = fast
        self.slow = slow
        self.signal = signal
        self.macd_line = None
        self.signal_line = None
        self.histogram = None

    def _calculate_indicators(self):
        """Calculate MACD components."""
        close = self.data['close']

        ema_fast = close.ewm(span=self.fast, adjust=False).mean()
        ema_slow = close.ewm(span=self.slow, adjust=False).mean()

        self.macd_line = ema_fast - ema_slow
        self.signal_line = self.macd_line.ewm(span=self.signal, adjust=False).mean()
        self.histogram = self.macd_line - self.signal_line

        self.indicators['macd'] = self.macd_line
        self.indicators['macd_signal'] = self.signal_line
        self.indicators['macd_hist'] = self.histogram

    def next(self, context: Context) -> Optional[Signal]:
        """Generate signals based on MACD crossover."""
        idx = len(context.data) - 1
        if idx < self.slow + self.signal:
            return None

        macd_curr = self.macd_line.iloc[idx]
        macd_prev = self.macd_line.iloc[idx - 1]
        signal_curr = self.signal_line.iloc[idx]
        signal_prev = self.signal_line.iloc[idx - 1]

        if pd.isna(macd_curr) or pd.isna(signal_curr):
            return None

        current_price = context.get_price(self.stock_code)
        has_position = context.has_position(self.stock_code)

        # Bullish crossover: MACD crosses above signal
        bullish = macd_prev <= signal_prev and macd_curr > signal_curr

        # Bearish crossover: MACD crosses below signal
        bearish = macd_prev >= signal_prev and macd_curr < signal_curr

        if bullish and not has_position:
            return Signal(
                stock_code=self.stock_code,
                signal=SignalType.BUY,
                date=context.date,
                price=current_price,
                reason=f"MACD bullish crossover: {macd_curr:.3f} > {signal_curr:.3f}"
            )

        if bearish and has_position:
            return Signal(
                stock_code=self.stock_code,
                signal=SignalType.SELL,
                date=context.date,
                price=current_price,
                reason=f"MACD bearish crossover: {macd_curr:.3f} < {signal_curr:.3f}"
            )

        return None


if __name__ == "__main__":
    # Test strategies with 601012
    import sys
    sys.path.insert(0, '/Users/kannmu/Library/CloudStorage/OneDrive-Personal/Projects/K-Quant/src')

    from data_loader import load_stock_data

    print("Testing Strategy Templates with 601012 (隆基绿能)...")
    print("=" * 60)

    # Load data
    data = load_stock_data("601012", "20240101", "20241231", include_fundamental=False)
    print(f"\nLoaded {len(data)} rows of data")

    # Test Double MA Strategy
    print("\n1. Testing DoubleMovingAverageStrategy (MA20/MA60)...")
    ma_strategy = DoubleMovingAverageStrategy("601012", fast_period=20, slow_period=60)
    ma_strategy.init(data)

    signals_found = 0
    for i in range(60, min(len(data), 200)):
        context = Context(
            date=data.iloc[i]['date'],
            current_prices={"601012": data.iloc[i]['close']},
            portfolio_value=100000,
            cash=100000,
            positions={},
            data=data.iloc[:i+1],
            indicators=ma_strategy.indicators
        )
        signal = ma_strategy.next(context)
        if signal:
            print(f"   {signal.date.strftime('%Y-%m-%d')}: {signal.signal.name} - {signal.reason}")
            signals_found += 1
            if signals_found >= 3:
                break

    # Test RSI Strategy
    print("\n2. Testing RSIStrategy (period=14)...")
    rsi_strategy = RSIStrategy("601012", period=14)
    rsi_strategy.init(data)

    signals_found = 0
    for i in range(14, min(len(data), 200)):
        has_pos = i > 100  # Simulate having position after some time
        context = Context(
            date=data.iloc[i]['date'],
            current_prices={"601012": data.iloc[i]['close']},
            portfolio_value=100000,
            cash=50000 if has_pos else 100000,
            positions={"601012": 1000} if has_pos else {},
            data=data.iloc[:i+1],
            indicators=rsi_strategy.indicators
        )
        signal = rsi_strategy.next(context)
        if signal:
            print(f"   {signal.date.strftime('%Y-%m-%d')}: {signal.signal.name} - {signal.reason}")
            signals_found += 1
            if signals_found >= 3:
                break

    # Test MACD Strategy
    print("\n3. Testing MACDStrategy (12/26/9)...")
    macd_strategy = MACDStrategy("601012")
    macd_strategy.init(data)

    signals_found = 0
    for i in range(35, min(len(data), 200)):
        context = Context(
            date=data.iloc[i]['date'],
            current_prices={"601012": data.iloc[i]['close']},
            portfolio_value=100000,
            cash=100000,
            positions={},
            data=data.iloc[:i+1],
            indicators=macd_strategy.indicators
        )
        signal = macd_strategy.next(context)
        if signal:
            print(f"   {signal.date.strftime('%Y-%m-%d')}: {signal.signal.name} - {signal.reason}")
            signals_found += 1
            if signals_found >= 3:
                break

    print("\n✅ Strategy templates test completed!")
