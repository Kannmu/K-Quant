"""
Strategy module for K-Quant System

Provides base classes, pre-built trading strategies, and utilities.
"""

from .base import Context, Signal, SignalType, Strategy, BuyAndHoldStrategy
from .templates import DoubleMovingAverageStrategy, MACDStrategy, RSIStrategy
from .utils import (
    calculate_sma, calculate_ema, calculate_rsi, calculate_macd,
    calculate_bollinger_bands, calculate_atr, calculate_volume_sma,
    detect_crossover, detect_rsi_signal, detect_breakout,
    calculate_position_size, calculate_dynamic_position_size,
    is_trending_up, is_trending_down, filter_by_volume_spike,
    SignalBuffer
)

# Note: optimizer module imports backtest modules, which may cause circular imports
# Import it separately when needed: from strategy.optimizer import StrategyOptimizer

__all__ = [
    # Base classes
    "Context",
    "Signal",
    "SignalType",
    "Strategy",
    "BuyAndHoldStrategy",
    # Templates
    "DoubleMovingAverageStrategy",
    "MACDStrategy",
    "RSIStrategy",
    # Utilities
    "calculate_sma",
    "calculate_ema",
    "calculate_rsi",
    "calculate_macd",
    "calculate_bollinger_bands",
    "calculate_atr",
    "calculate_volume_sma",
    "detect_crossover",
    "detect_rsi_signal",
    "detect_breakout",
    "calculate_position_size",
    "calculate_dynamic_position_size",
    "is_trending_up",
    "is_trending_down",
    "filter_by_volume_spike",
    "SignalBuffer",
]
