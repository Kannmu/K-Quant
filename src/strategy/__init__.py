"""
Strategy module for K-Quant System

Provides base classes and pre-built trading strategies.
"""

from .base import Context, Signal, SignalType, Strategy
from .templates import DoubleMovingAverageStrategy, MACDStrategy, RSIStrategy

__all__ = [
    "Context",
    "Signal",
    "SignalType",
    "Strategy",
    "DoubleMovingAverageStrategy",
    "MACDStrategy",
    "RSIStrategy",
]
