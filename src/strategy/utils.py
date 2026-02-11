"""
Strategy Utilities - Helper functions for strategy development

Provides common technical indicators, signal utilities, and helper functions
for developing trading strategies.
"""

from typing import Optional, Tuple, List
import pandas as pd
import numpy as np
from datetime import datetime


def calculate_sma(data: pd.Series, period: int) -> pd.Series:
    """Calculate Simple Moving Average."""
    return data.rolling(window=period).mean()


def calculate_ema(data: pd.Series, period: int) -> pd.Series:
    """Calculate Exponential Moving Average."""
    return data.ewm(span=period, adjust=False).mean()


def calculate_rsi(data: pd.Series, period: int = 14) -> pd.Series:
    """Calculate Relative Strength Index."""
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))


def calculate_macd(
    data: pd.Series,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    Calculate MACD indicator.

    Returns:
        Tuple of (MACD line, Signal line, Histogram)
    """
    ema_fast = calculate_ema(data, fast)
    ema_slow = calculate_ema(data, slow)
    macd_line = ema_fast - ema_slow
    signal_line = calculate_ema(macd_line, signal)
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


def calculate_bollinger_bands(
    data: pd.Series,
    period: int = 20,
    std_dev: float = 2.0
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    Calculate Bollinger Bands.

    Returns:
        Tuple of (Upper band, Middle band, Lower band)
    """
    middle = calculate_sma(data, period)
    std = data.rolling(window=period).std()
    upper = middle + (std * std_dev)
    lower = middle - (std * std_dev)
    return upper, middle, lower


def calculate_atr(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = 14
) -> pd.Series:
    """Calculate Average True Range."""
    high_low = high - low
    high_close = np.abs(high - close.shift())
    low_close = np.abs(low - close.shift())

    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = np.max(ranges, axis=1)

    return true_range.rolling(window=period).mean()


def calculate_volume_sma(volume: pd.Series, period: int = 20) -> pd.Series:
    """Calculate Volume Moving Average."""
    return volume.rolling(window=period).mean()


def detect_crossover(
    fast: pd.Series,
    slow: pd.Series,
    idx: int
) -> int:
    """
    Detect MA crossover at given index.

    Returns:
        1 for golden cross (fast crosses above slow)
        -1 for death cross (fast crosses below slow)
        0 for no crossover
    """
    if idx < 1:
        return 0

    fast_curr = fast.iloc[idx]
    fast_prev = fast.iloc[idx - 1]
    slow_curr = slow.iloc[idx]
    slow_prev = slow.iloc[idx - 1]

    if pd.isna(fast_curr) or pd.isna(slow_curr):
        return 0

    if fast_prev <= slow_prev and fast_curr > slow_curr:
        return 1  # Golden cross
    elif fast_prev >= slow_prev and fast_curr < slow_curr:
        return -1  # Death cross
    return 0


def detect_rsi_signal(
    rsi: pd.Series,
    idx: int,
    oversold: float = 30.0,
    overbought: float = 70.0
) -> int:
    """
    Detect RSI trading signal.

    Returns:
        1 for buy signal (RSI leaves oversold)
        -1 for sell signal (RSI leaves overbought)
        0 for no signal
    """
    if idx < 1:
        return 0

    curr = rsi.iloc[idx]
    prev = rsi.iloc[idx - 1]

    if pd.isna(curr):
        return 0

    if prev <= oversold and curr > oversold:
        return 1  # Leaving oversold - buy
    elif prev >= overbought and curr < overbought:
        return -1  # Leaving overbought - sell
    return 0


def detect_breakout(
    price: pd.Series,
    upper: pd.Series,
    lower: pd.Series,
    idx: int
) -> int:
    """
    Detect breakout from channels/bands.

    Returns:
        1 for upward breakout (price breaks above upper)
        -1 for downward breakout (price breaks below lower)
        0 for no breakout
    """
    if idx < 1:
        return 0

    curr_price = price.iloc[idx]
    prev_price = price.iloc[idx - 1]
    curr_upper = upper.iloc[idx]
    curr_lower = lower.iloc[idx]

    if pd.isna(curr_upper) or pd.isna(curr_lower):
        return 0

    if prev_price <= curr_upper and curr_price > curr_upper:
        return 1  # Upward breakout
    elif prev_price >= curr_lower and curr_price < curr_lower:
        return -1  # Downward breakout
    return 0


def calculate_position_size(
    portfolio_value: float,
    price: float,
    risk_pct: float = 0.2,
    max_position_value: Optional[float] = None
) -> int:
    """
    Calculate position size (number of shares) based on risk parameters.

    Args:
        portfolio_value: Current portfolio value
        price: Current stock price
        risk_pct: Percentage of portfolio to allocate (0.2 = 20%)
        max_position_value: Maximum position value in currency

    Returns:
        Number of shares (rounded to 100 for A-shares)
    """
    position_value = portfolio_value * risk_pct

    if max_position_value is not None:
        position_value = min(position_value, max_position_value)

    shares = int(position_value / price)
    # Round down to nearest 100 for A-shares
    return (shares // 100) * 100


def calculate_dynamic_position_size(
    portfolio_value: float,
    price: float,
    atr: float,
    risk_per_trade: float = 0.02,
    atr_multiplier: float = 2.0
) -> int:
    """
    Calculate position size based on volatility (ATR).

    Position size is adjusted so that risk per trade is limited.

    Args:
        portfolio_value: Current portfolio value
        price: Current stock price
        atr: Average True Range
        risk_per_trade: Risk per trade as percentage of portfolio
        atr_multiplier: Stop loss distance in ATR units

    Returns:
        Number of shares
    """
    risk_amount = portfolio_value * risk_per_trade
    stop_distance = atr * atr_multiplier

    if stop_distance <= 0:
        return 0

    shares = int(risk_amount / stop_distance)
    return (shares // 100) * 100


def is_trending_up(
    data: pd.Series,
    short_period: int = 10,
    long_period: int = 30,
    idx: int = -1
) -> bool:
    """Check if price is in uptrend (short MA above long MA)."""
    if len(data) < long_period:
        return False

    short_ma = calculate_sma(data, short_period)
    long_ma = calculate_sma(data, long_period)

    return short_ma.iloc[idx] > long_ma.iloc[idx]


def is_trending_down(
    data: pd.Series,
    short_period: int = 10,
    long_period: int = 30,
    idx: int = -1
) -> bool:
    """Check if price is in downtrend (short MA below long MA)."""
    if len(data) < long_period:
        return False

    short_ma = calculate_sma(data, short_period)
    long_ma = calculate_sma(data, long_period)

    return short_ma.iloc[idx] < long_ma.iloc[idx]


def filter_by_volume_spike(
    volume: pd.Series,
    volume_ma: pd.Series,
    idx: int,
    threshold: float = 2.0
) -> bool:
    """
    Check if volume spike is present (volume > threshold * volume_ma).

    Returns:
        True if volume spike detected
    """
    if pd.isna(volume_ma.iloc[idx]) or volume_ma.iloc[idx] == 0:
        return False
    return volume.iloc[idx] > volume_ma.iloc[idx] * threshold


class SignalBuffer:
    """
    Buffer for tracking recent signals to avoid over-trading.

    Example:
        buffer = SignalBuffer(min_bars_between_signals=5)
        if buffer.can_signal(current_idx):
            # Generate signal
            buffer.record_signal(current_idx)
    """

    def __init__(self, min_bars_between_signals: int = 5):
        self.min_bars = min_bars_between_signals
        self.last_signal_idx: Optional[int] = None

    def can_signal(self, current_idx: int) -> bool:
        """Check if enough bars have passed since last signal."""
        if self.last_signal_idx is None:
            return True
        return (current_idx - self.last_signal_idx) >= self.min_bars

    def record_signal(self, idx: int):
        """Record that a signal was generated at this index."""
        self.last_signal_idx = idx

    def reset(self):
        """Reset the buffer."""
        self.last_signal_idx = None


if __name__ == "__main__":
    # Test utilities
    import sys
    sys.path.insert(0, '/Users/kannmu/Library/CloudStorage/OneDrive-Personal/Projects/K-Quant/src')

    from data_loader import load_stock_data

    print("Testing Strategy Utilities...")
    print("=" * 60)

    # Load data
    data = load_stock_data("601012", "20240101", "20241231", include_fundamental=False)
    close = data['close']

    print("\n1. Testing Moving Averages...")
    sma20 = calculate_sma(close, 20)
    ema20 = calculate_ema(close, 20)
    print(f"   SMA20 (last): {sma20.iloc[-1]:.2f}")
    print(f"   EMA20 (last): {ema20.iloc[-1]:.2f}")

    print("\n2. Testing RSI...")
    rsi = calculate_rsi(close, 14)
    print(f"   RSI (last): {rsi.iloc[-1]:.2f}")

    print("\n3. Testing MACD...")
    macd, signal, hist = calculate_macd(close)
    print(f"   MACD (last): {macd.iloc[-1]:.4f}")
    print(f"   Signal (last): {signal.iloc[-1]:.4f}")

    print("\n4. Testing Bollinger Bands...")
    upper, middle, lower = calculate_bollinger_bands(close)
    print(f"   Upper (last): {upper.iloc[-1]:.2f}")
    print(f"   Middle (last): {middle.iloc[-1]:.2f}")
    print(f"   Lower (last): {lower.iloc[-1]:.2f}")

    print("\n5. Testing ATR...")
    atr = calculate_atr(data['high'], data['low'], close, 14)
    print(f"   ATR (last): {atr.iloc[-1]:.4f}")

    print("\n6. Testing Position Sizing...")
    shares = calculate_position_size(100000, close.iloc[-1], risk_pct=0.2)
    print(f"   Position size at {close.iloc[-1]:.2f}: {shares} shares")

    print("\n7. Testing Signal Buffer...")
    buffer = SignalBuffer(min_bars_between_signals=5)
    print(f"   Can signal at idx 10: {buffer.can_signal(10)}")
    buffer.record_signal(10)
    print(f"   Can signal at idx 12: {buffer.can_signal(12)}")
    print(f"   Can signal at idx 15: {buffer.can_signal(15)}")

    print("\n✅ Strategy utilities test completed!")
