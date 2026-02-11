"""
Advanced Strategy Templates - More sophisticated trading strategies

Pre-built strategies demonstrating various trading techniques:
- Multi-factor strategies
- Trend following with filters
- Mean reversion
- Volatility-based position sizing
"""

from typing import Optional
import pandas as pd
import numpy as np

from .base import Context, Signal, SignalType, Strategy
from .utils import (
    calculate_sma, calculate_ema, calculate_rsi, calculate_macd,
    calculate_bollinger_bands, calculate_atr, calculate_volume_sma,
    detect_crossover, detect_rsi_signal, filter_by_volume_spike,
    calculate_dynamic_position_size, is_trending_up, SignalBuffer
)


class TrendFollowingStrategy(Strategy):
    """
    Trend Following Strategy with Multiple Filters

    Uses moving average crossover as primary signal,
    filtered by trend direction and volume confirmation.

    Parameters:
        stock_code: Stock to trade
        fast_period: Fast MA period (default 20)
        slow_period: Slow MA period (default 50)
        trend_period: Long-term trend MA period (default 200)
        volume_factor: Volume spike threshold (default 1.5)
    """

    def __init__(
        self,
        stock_code: str,
        fast_period: int = 20,
        slow_period: int = 50,
        trend_period: int = 200,
        volume_factor: float = 1.5,
    ):
        super().__init__(f"TrendFollowing_{fast_period}_{slow_period}")
        self.stock_code = stock_code
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.trend_period = trend_period
        self.volume_factor = volume_factor

    def _calculate_indicators(self):
        """Calculate trend and volume indicators."""
        close = self.data['close']
        volume = self.data['volume']

        # Moving averages
        self.indicators['fast_ma'] = calculate_sma(close, self.fast_period)
        self.indicators['slow_ma'] = calculate_sma(close, self.slow_period)
        self.indicators['trend_ma'] = calculate_sma(close, self.trend_period)

        # Volume
        self.indicators['volume_sma'] = calculate_volume_sma(volume, 20)

    def next(self, context: Context) -> Optional[Signal]:
        """Generate signals with trend and volume filters."""
        idx = len(context.data) - 1

        # Need enough data for trend MA
        if idx < self.trend_period:
            return None

        fast = self.indicators['fast_ma']
        slow = self.indicators['slow_ma']
        trend = self.indicators['trend_ma']
        volume = self.data['volume']
        volume_sma = self.indicators['volume_sma']

        # Check for valid values
        if any(pd.isna(x.iloc[idx]) for x in [fast, slow, trend, volume_sma]):
            return None

        # Current price
        current_price = context.get_price(self.stock_code)
        has_position = context.has_position(self.stock_code)

        # Check trend direction
        price_above_trend = current_price > trend.iloc[idx]

        # Detect crossover
        crossover = detect_crossover(fast, slow, idx)

        # Volume confirmation
        volume_confirmed = filter_by_volume_spike(
            volume, volume_sma, idx, self.volume_factor
        )

        # Buy signal: golden cross, above trend, volume confirmed
        if crossover == 1 and price_above_trend and not has_position:
            return Signal(
                stock_code=self.stock_code,
                signal=SignalType.BUY,
                date=context.date,
                price=current_price,
                reason=f"Trend Follow Buy: Golden Cross above MA{self.trend_period}"
            )

        # Sell signal: death cross or price below trend
        if crossover == -1 and has_position:
            return Signal(
                stock_code=self.stock_code,
                signal=SignalType.SELL,
                date=context.date,
                price=current_price,
                reason=f"Trend Follow Sell: Death Cross"
            )

        return None


class MeanReversionStrategy(Strategy):
    """
    Mean Reversion Strategy using Bollinger Bands

    Buys when price touches lower band (oversold),
    sells when price touches upper band (overbought).

    Parameters:
        stock_code: Stock to trade
        period: Bollinger Bands period (default 20)
        std_dev: Standard deviation multiplier (default 2.0)
    """

    def __init__(
        self,
        stock_code: str,
        period: int = 20,
        std_dev: float = 2.0,
    ):
        super().__init__(f"MeanReversion_BB{period}")
        self.stock_code = stock_code
        self.period = period
        self.std_dev = std_dev
        self.signal_buffer = SignalBuffer(min_bars_between_signals=5)

    def _calculate_indicators(self):
        """Calculate Bollinger Bands."""
        upper, middle, lower = calculate_bollinger_bands(
            self.data['close'], self.period, self.std_dev
        )
        self.indicators['bb_upper'] = upper
        self.indicators['bb_middle'] = middle
        self.indicators['bb_lower'] = lower

    def next(self, context: Context) -> Optional[Signal]:
        """Generate mean reversion signals."""
        idx = len(context.data) - 1

        if idx < self.period:
            return None

        upper = self.indicators['bb_upper']
        lower = self.indicators['bb_lower']
        close = self.data['close']

        if any(pd.isna(x.iloc[idx]) for x in [upper, lower]):
            return None

        current_price = context.get_price(self.stock_code)
        has_position = context.has_position(self.stock_code)

        # Check buffer to avoid over-trading
        if not self.signal_buffer.can_signal(idx):
            return None

        # Buy: Price touches or breaks below lower band
        if close.iloc[idx] <= lower.iloc[idx] and not has_position:
            self.signal_buffer.record_signal(idx)
            return Signal(
                stock_code=self.stock_code,
                signal=SignalType.BUY,
                date=context.date,
                price=current_price,
                reason=f"Mean Reversion Buy: Price at lower BB ({current_price:.2f})"
            )

        # Sell: Price touches or breaks above upper band
        if close.iloc[idx] >= upper.iloc[idx] and has_position:
            self.signal_buffer.record_signal(idx)
            return Signal(
                stock_code=self.stock_code,
                signal=SignalType.SELL,
                date=context.date,
                price=current_price,
                reason=f"Mean Reversion Sell: Price at upper BB ({current_price:.2f})"
            )

        return None


class VolatilityBreakoutStrategy(Strategy):
    """
    Volatility Breakout Strategy using ATR

    Enters when price breaks out with high volatility.
    Uses ATR for position sizing.

    Parameters:
        stock_code: Stock to trade
        atr_period: ATR calculation period (default 14)
        breakout_multiplier: Breakout threshold in ATR units (default 2.0)
    """

    def __init__(
        self,
        stock_code: str,
        atr_period: int = 14,
        breakout_multiplier: float = 2.0,
    ):
        super().__init__(f"VolBreakout_ATR{atr_period}")
        self.stock_code = stock_code
        self.atr_period = atr_period
        self.breakout_multiplier = breakout_multiplier

    def _calculate_indicators(self):
        """Calculate ATR and moving average."""
        self.indicators['atr'] = calculate_atr(
            self.data['high'],
            self.data['low'],
            self.data['close'],
            self.atr_period
        )
        self.indicators['ma20'] = calculate_sma(self.data['close'], 20)

    def next(self, context: Context) -> Optional[Signal]:
        """Generate breakout signals."""
        idx = len(context.data) - 1

        if idx < max(self.atr_period, 20) + 1:
            return None

        atr = self.indicators['atr']
        ma20 = self.indicators['ma20']
        high = self.data['high']
        low = self.data['low']

        if pd.isna(atr.iloc[idx]) or pd.isna(ma20.iloc[idx]):
            return None

        current_price = context.get_price(self.stock_code)
        has_position = context.has_position(self.stock_code)

        # Breakout levels
        atr_value = atr.iloc[idx]
        upper_level = ma20.iloc[idx] + atr_value * self.breakout_multiplier
        lower_level = ma20.iloc[idx] - atr_value * self.breakout_multiplier

        # Buy: High breaks above upper level
        if high.iloc[idx] > upper_level and not has_position:
            return Signal(
                stock_code=self.stock_code,
                signal=SignalType.BUY,
                date=context.date,
                price=current_price,
                reason=f"Vol Breakout Buy: Price > MA20 + {self.breakout_multiplier}ATR"
            )

        # Sell: Low breaks below lower level
        if low.iloc[idx] < lower_level and has_position:
            return Signal(
                stock_code=self.stock_code,
                signal=SignalType.SELL,
                date=context.date,
                price=current_price,
                reason=f"Vol Breakout Sell: Price < MA20 - {self.breakout_multiplier}ATR"
            )

        return None


class MultiFactorStrategy(Strategy):
    """
    Multi-Factor Strategy combining multiple signals

    Combines:
    - Trend (price vs MA)
    - Momentum (RSI)
    - Volatility (ATR for position sizing)

    Requires all factors to align for signal generation.

    Parameters:
        stock_code: Stock to trade
        trend_ma: Trend moving average period (default 50)
        rsi_period: RSI period (default 14)
        rsi_oversold: RSI oversold threshold (default 35)
        rsi_overbought: RSI overbought threshold (default 65)
    """

    def __init__(
        self,
        stock_code: str,
        trend_ma: int = 50,
        rsi_period: int = 14,
        rsi_oversold: float = 35.0,
        rsi_overbought: float = 65.0,
    ):
        super().__init__(f"MultiFactor_T{trend_ma}_R{rsi_period}")
        self.stock_code = stock_code
        self.trend_ma = trend_ma
        self.rsi_period = rsi_period
        self.rsi_oversold = rsi_oversold
        self.rsi_overbought = rsi_overbought

    def _calculate_indicators(self):
        """Calculate all factor indicators."""
        close = self.data['close']

        # Trend factor
        self.indicators['trend_ma'] = calculate_sma(close, self.trend_ma)

        # Momentum factor
        self.indicators['rsi'] = calculate_rsi(close, self.rsi_period)

        # Volatility factor (ATR)
        self.indicators['atr'] = calculate_atr(
            self.data['high'], self.data['low'], close, 14
        )

    def next(self, context: Context) -> Optional[Signal]:
        """Generate signals based on multiple factors."""
        idx = len(context.data) - 1

        if idx < self.trend_ma:
            return None

        close = self.data['close']
        trend_ma = self.indicators['trend_ma']
        rsi = self.indicators['rsi']

        if any(pd.isna(x.iloc[idx]) for x in [trend_ma, rsi]):
            return None

        current_price = context.get_price(self.stock_code)
        has_position = context.has_position(self.stock_code)

        # Trend factor
        above_trend = close.iloc[idx] > trend_ma.iloc[idx]

        # RSI factor
        rsi_value = rsi.iloc[idx]
        rsi_oversold = rsi_value < self.rsi_oversold
        rsi_overbought = rsi_value > self.rsi_overbought

        # Buy: Above trend AND oversold RSI (pullback in uptrend)
        if above_trend and rsi_oversold and not has_position:
            return Signal(
                stock_code=self.stock_code,
                signal=SignalType.BUY,
                date=context.date,
                price=current_price,
                reason=f"Multi-Factor Buy: Trend up + RSI oversold ({rsi_value:.1f})"
            )

        # Sell: RSI overbought OR price below trend
        if has_position and (rsi_overbought or not above_trend):
            return Signal(
                stock_code=self.stock_code,
                signal=SignalType.SELL,
                date=context.date,
                price=current_price,
                reason=f"Multi-Factor Sell: {'RSI overbought' if rsi_overbought else 'Below trend'}"
            )

        return None


class AdaptiveMomentumStrategy(Strategy):
    """
    Adaptive Momentum Strategy

    Uses MACD for trend detection with adaptive parameters based on volatility.

    Parameters:
        stock_code: Stock to trade
        volatility_lookback: Period for volatility calculation (default 20)
        low_vol_adjustment: Parameter adjustment for low volatility (default 0.8)
        high_vol_adjustment: Parameter adjustment for high volatility (default 1.2)
    """

    def __init__(
        self,
        stock_code: str,
        volatility_lookback: int = 20,
        low_vol_adjustment: float = 0.8,
        high_vol_adjustment: float = 1.2,
    ):
        super().__init__(f"AdaptiveMomentum_V{volatility_lookback}")
        self.stock_code = stock_code
        self.volatility_lookback = volatility_lookback
        self.low_vol_adjustment = low_vol_adjustment
        self.high_vol_adjustment = high_vol_adjustment

    def _calculate_indicators(self):
        """Calculate volatility and adaptive MACD."""
        close = self.data['close']

        # Volatility (standard deviation of returns)
        returns = close.pct_change()
        self.indicators['volatility'] = returns.rolling(self.volatility_lookback).std()

        # Base MACD parameters
        self.indicators['macd'], self.indicators['macd_signal'], _ = calculate_macd(
            close, fast=12, slow=26, signal=9
        )

        # Adaptive parameters based on current volatility
        vol = self.indicators['volatility'].iloc[-1]
        avg_vol = self.indicators['volatility'].mean()

        if pd.notna(vol) and pd.notna(avg_vol) and avg_vol > 0:
            vol_ratio = vol / avg_vol

            if vol_ratio < 0.8:
                # Low volatility - faster signals
                fast = int(12 * self.low_vol_adjustment)
                slow = int(26 * self.low_vol_adjustment)
            elif vol_ratio > 1.2:
                # High volatility - slower signals
                fast = int(12 * self.high_vol_adjustment)
                slow = int(26 * self.high_vol_adjustment)
            else:
                fast, slow = 12, 26

            # Recalculate with adjusted parameters
            self.indicators['macd'], self.indicators['macd_signal'], _ = calculate_macd(
                close, fast=fast, slow=slow, signal=9
            )

    def next(self, context: Context) -> Optional[Signal]:
        """Generate adaptive momentum signals."""
        idx = len(context.data) - 1

        if idx < 35:  # Need enough data for MACD
            return None

        macd = self.indicators['macd']
        signal = self.indicators['macd_signal']

        if pd.isna(macd.iloc[idx]) or pd.isna(signal.iloc[idx]):
            return None

        current_price = context.get_price(self.stock_code)
        has_position = context.has_position(self.stock_code)

        macd_curr = macd.iloc[idx]
        macd_prev = macd.iloc[idx - 1]
        signal_curr = signal.iloc[idx]
        signal_prev = signal.iloc[idx - 1]

        # Bullish crossover
        bullish = macd_prev <= signal_prev and macd_curr > signal_curr
        # Bearish crossover
        bearish = macd_prev >= signal_prev and macd_curr < signal_curr

        if bullish and not has_position:
            return Signal(
                stock_code=self.stock_code,
                signal=SignalType.BUY,
                date=context.date,
                price=current_price,
                reason=f"Adaptive MACD Buy: Bullish crossover ({macd_curr:.3f})"
            )

        if bearish and has_position:
            return Signal(
                stock_code=self.stock_code,
                signal=SignalType.SELL,
                date=context.date,
                price=current_price,
                reason=f"Adaptive MACD Sell: Bearish crossover ({macd_curr:.3f})"
            )

        return None


if __name__ == "__main__":
    # Test advanced strategies
    import sys
    sys.path.insert(0, '/Users/kannmu/Library/CloudStorage/OneDrive-Personal/Projects/K-Quant/src')

    from data_loader import load_stock_data
    from backtest.engine import BacktestEngine

    print("Testing Advanced Strategy Templates...")
    print("=" * 60)

    # Load data
    data = load_stock_data("601012", "20240101", "20241231", include_fundamental=False)

    strategies = [
        ("Trend Following", TrendFollowingStrategy("601012")),
        ("Mean Reversion", MeanReversionStrategy("601012")),
        ("Volatility Breakout", VolatilityBreakoutStrategy("601012")),
        ("Multi-Factor", MultiFactorStrategy("601012")),
        ("Adaptive Momentum", AdaptiveMomentumStrategy("601012")),
    ]

    for name, strategy in strategies:
        print(f"\n{name}:")
        try:
            engine = BacktestEngine(initial_cash=100000)
            result = engine.run(strategy, data, "601012")
            print(f"  Return: {result.total_return:.2f}% | "
                  f"Sharpe: {result.sharpe_ratio:.2f} | "
                  f"Trades: {result.total_trades}")
        except Exception as e:
            print(f"  Error: {e}")

    print("\n✅ Advanced strategy templates test completed!")
