"""
Analysis module for K-Quant System

Provides performance metrics calculation and visualization.
"""

from .metrics import (
    PerformanceMetrics,
    calculate_all_metrics,
    calculate_drawdown_series,
    calculate_returns_metrics,
    calculate_risk_metrics,
    calculate_trade_metrics,
    print_detailed_metrics,
)
from .plotting import (
    create_full_report,
    plot_drawdown,
    plot_equity_curve,
    plot_monthly_returns,
    plot_trade_distribution,
)

__all__ = [
    # Metrics
    "PerformanceMetrics",
    "calculate_all_metrics",
    "calculate_drawdown_series",
    "calculate_returns_metrics",
    "calculate_risk_metrics",
    "calculate_trade_metrics",
    "print_detailed_metrics",
    # Plotting
    "plot_equity_curve",
    "plot_drawdown",
    "plot_monthly_returns",
    "plot_trade_distribution",
    "create_full_report",
]
