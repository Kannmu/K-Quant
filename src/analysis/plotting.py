"""
Plotting Module - Visualization for backtest results

Creates charts for equity curve, drawdown, and trade analysis.
"""

from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.gridspec import GridSpec


def plot_equity_curve(
    equity: pd.Series,
    benchmark: Optional[pd.Series] = None,
    title: str = "Equity Curve",
    figsize: tuple = (12, 6),
) -> plt.Figure:
    """
    Plot equity curve with optional benchmark.

    Args:
        equity: Portfolio value series
        benchmark: Optional benchmark series
        title: Chart title
        figsize: Figure size

    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Plot equity curve
    ax.plot(equity.index, equity.values, label="Strategy", linewidth=2, color="#2196F3")

    # Plot benchmark if provided
    if benchmark is not None:
        # Normalize benchmark to same starting value
        normalized_benchmark = benchmark / benchmark.iloc[0] * equity.iloc[0]
        ax.plot(benchmark.index, normalized_benchmark.values,
                label="Benchmark", linewidth=1.5, color="#757575", linestyle="--")

    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlabel("Date", fontsize=12)
    ax.set_ylabel("Portfolio Value", fontsize=12)
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)

    # Format y-axis as currency
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"¥{x:,.0f}"))

    plt.tight_layout()
    return fig


def plot_drawdown(
    equity: pd.Series,
    title: str = "Drawdown Analysis",
    figsize: tuple = (12, 4),
) -> plt.Figure:
    """
    Plot drawdown chart.

    Args:
        equity: Portfolio value series
        title: Chart title
        figsize: Figure size

    Returns:
        Matplotlib figure
    """
    # Calculate drawdown
    peak = equity.cummax()
    drawdown = (equity - peak) / peak * 100

    fig, ax = plt.subplots(figsize=figsize)

    # Fill area under drawdown curve
    ax.fill_between(drawdown.index, drawdown.values, 0,
                    color="#F44336", alpha=0.3, label="Drawdown")
    ax.plot(drawdown.index, drawdown.values, color="#D32F2F", linewidth=1)

    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlabel("Date", fontsize=12)
    ax.set_ylabel("Drawdown (%)", fontsize=12)
    ax.grid(True, alpha=0.3)

    # Add horizontal line at 0
    ax.axhline(y=0, color="black", linestyle="-", linewidth=0.5)

    # Format y-axis as percentage
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"{x:.1f}%"))

    # Annotate max drawdown
    max_dd = drawdown.min()
    max_dd_date = drawdown.idxmin()
    ax.annotate(f"Max DD: {max_dd:.2f}%",
                xy=(max_dd_date, max_dd),
                xytext=(10, -30),
                textcoords="offset points",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="red", alpha=0.7),
                arrowprops=dict(arrowstyle="->", color="red"),
                color="white",
                fontweight="bold")

    plt.tight_layout()
    return fig


def plot_monthly_returns(
    equity: pd.Series,
    title: str = "Monthly Returns",
    figsize: tuple = (12, 6),
) -> plt.Figure:
    """
    Plot monthly returns heatmap.

    Args:
        equity: Portfolio value series
        title: Chart title
        figsize: Figure size

    Returns:
        Matplotlib figure
    """
    # Calculate monthly returns
    monthly_returns = equity.resample('ME').last().pct_change() * 100
    monthly_returns = monthly_returns.dropna()

    # Create DataFrame for heatmap
    returns_df = pd.DataFrame({
        'year': monthly_returns.index.year,
        'month': monthly_returns.index.month,
        'return': monthly_returns.values
    })
    pivot_table = returns_df.pivot(index='year', columns='month', values='return')

    # Fill missing months with NaN
    pivot_table = pivot_table.reindex(columns=range(1, 13))

    fig, ax = plt.subplots(figsize=figsize)

    # Create heatmap
    im = ax.imshow(pivot_table.values, cmap='RdYlGn', aspect='auto',
                   vmin=-10, vmax=10)

    # Set ticks
    ax.set_xticks(range(12))
    ax.set_xticklabels(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                        'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
    ax.set_yticks(range(len(pivot_table.index)))
    ax.set_yticklabels(pivot_table.index)

    # Add values to cells
    for i in range(len(pivot_table.index)):
        for j in range(12):
            value = pivot_table.iloc[i, j]
            if not pd.isna(value):
                text_color = 'white' if abs(value) > 5 else 'black'
                ax.text(j, i, f'{value:.1f}%',
                        ha='center', va='center', color=text_color, fontsize=9)

    ax.set_title(title, fontsize=14, fontweight="bold")
    plt.colorbar(im, ax=ax, label='Return (%)')
    plt.tight_layout()

    return fig


def plot_trade_distribution(
    trades: pd.DataFrame,
    title: str = "Trade Analysis",
    figsize: tuple = (12, 5),
) -> plt.Figure:
    """
    Plot trade distribution histogram.

    Args:
        trades: Trade history DataFrame
        title: Chart title
        figsize: Figure size

    Returns:
        Matplotlib figure
    """
    if len(trades) == 0:
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, "No trades to analyze",
                ha='center', va='center', fontsize=14)
        return fig

    # Calculate P&L per trade
    buy_trades = trades[trades['action'] == 'buy']
    sell_trades = trades[trades['action'] == 'sell']

    pnl_list = []
    for _, sell in sell_trades.iterrows():
        matching_buys = buy_trades[buy_trades['date'] <= sell['date']]
        if len(matching_buys) > 0:
            buy = matching_buys.iloc[-1]
            pnl = (sell['price'] - buy['price']) * sell['quantity']
            pnl_list.append(pnl)

    if len(pnl_list) == 0:
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, "No completed trades",
                ha='center', va='center', fontsize=14)
        return fig

    pnl_series = pd.Series(pnl_list)

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Histogram
    ax1 = axes[0]
    colors = ['#4CAF50' if x > 0 else '#F44336' for x in pnl_series]
    ax1.hist(pnl_series, bins=20, edgecolor='black', color=colors, alpha=0.7)
    ax1.axvline(x=0, color='black', linestyle='--', linewidth=1)
    ax1.set_xlabel('P&L per Trade')
    ax1.set_ylabel('Frequency')
    ax1.set_title('Trade P&L Distribution')
    ax1.grid(True, alpha=0.3)

    # Cumulative P&L
    ax2 = axes[1]
    cumulative_pnl = pnl_series.cumsum()
    ax2.plot(range(len(cumulative_pnl)), cumulative_pnl.values,
             linewidth=2, color='#2196F3')
    ax2.fill_between(range(len(cumulative_pnl)), cumulative_pnl.values, 0,
                     alpha=0.3, color='#2196F3')
    ax2.axhline(y=0, color='black', linestyle='--', linewidth=1)
    ax2.set_xlabel('Trade Number')
    ax2.set_ylabel('Cumulative P&L')
    ax2.set_title('Cumulative Trade P&L')
    ax2.grid(True, alpha=0.3)

    fig.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()

    return fig


def create_full_report(
    equity: pd.Series,
    trades: pd.DataFrame,
    benchmark: Optional[pd.Series] = None,
    title: str = "Backtest Report",
    save_path: Optional[str] = None,
    metrics: Optional[Dict] = None,
) -> plt.Figure:
    """
    Create full report with multiple charts.

    Args:
        equity: Portfolio value series
        trades: Trade history DataFrame
        benchmark: Optional benchmark series
        title: Report title
        save_path: Optional path to save figure
        metrics: Optional pre-calculated metrics dict with sharpe_ratio, etc.

    Returns:
        Matplotlib figure
    """
    fig = plt.figure(figsize=(14, 10))
    gs = GridSpec(3, 2, figure=fig, hspace=0.3, wspace=0.3)

    # Calculate basic stats
    total_return = (equity.iloc[-1] / equity.iloc[0] - 1) * 100
    peak = equity.cummax()
    drawdown = (equity - peak) / peak * 100
    max_dd = drawdown.min()

    # Calculate Sharpe ratio if not provided
    if metrics is None or 'sharpe_ratio' not in metrics:
        daily_returns = equity.pct_change().dropna()
        if len(daily_returns) > 1 and daily_returns.std() > 0:
            excess_returns = daily_returns - 0.03 / 252
            sharpe = (excess_returns.mean() / daily_returns.std()) * (252 ** 0.5)
        else:
            sharpe = 0.0
    else:
        sharpe = metrics.get('sharpe_ratio', 0.0)

    # Calculate win rate if trades exist
    if len(trades) > 0:
        buy_trades = trades[trades['action'] == 'buy']
        sell_trades = trades[trades['action'] == 'sell']
        # Simple P&L calculation
        profits = []
        for _, sell in sell_trades.iterrows():
            matching_buys = buy_trades[buy_trades['date'] <= sell['date']]
            if len(matching_buys) > 0:
                buy = matching_buys.iloc[-1]
                profit = sell['price'] - buy['price']
                profits.append(profit)
        win_rate = sum(1 for p in profits if p > 0) / len(profits) * 100 if profits else 0.0
    else:
        win_rate = 0.0

    # Equity curve
    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(equity.index, equity.values, label="Strategy", linewidth=2, color="#2196F3")

    # Mark buy/sell signals on equity curve
    if len(trades) > 0:
        for _, trade in trades.iterrows():
            trade_date = pd.to_datetime(trade['date'])
            # Find closest date in equity index
            if trade_date in equity.index:
                equity_value = equity[trade_date]
                if trade['action'] == 'buy':
                    ax1.scatter(trade_date, equity_value, color='green', marker='^',
                               s=100, zorder=5, label='_nolegend_')
                else:
                    ax1.scatter(trade_date, equity_value, color='red', marker='v',
                               s=100, zorder=5, label='_nolegend_')

    if benchmark is not None:
        normalized_benchmark = benchmark / benchmark.iloc[0] * equity.iloc[0]
        ax1.plot(benchmark.index, normalized_benchmark.values,
                label="Benchmark", linewidth=1.5, color="#757575", linestyle="--")
    ax1.set_title("Equity Curve", fontweight="bold")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"¥{x:,.0f}"))

    # Drawdown
    ax2 = fig.add_subplot(gs[1, :])
    ax2.fill_between(drawdown.index, drawdown.values, 0,
                     color="#F44336", alpha=0.3)
    ax2.plot(drawdown.index, drawdown.values, color="#D32F2F", linewidth=1)
    ax2.set_title("Drawdown", fontweight="bold")
    ax2.set_ylabel("Drawdown (%)")
    ax2.grid(True, alpha=0.3)

    # Annotate max drawdown
    max_dd_date = drawdown.idxmin()
    ax2.annotate(f"Max DD: {max_dd:.2f}%",
                xy=(max_dd_date, max_dd),
                xytext=(10, -30),
                textcoords="offset points",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="red", alpha=0.7),
                arrowprops=dict(arrowstyle="->", color="red"),
                color="white",
                fontweight="bold")

    # Monthly returns heatmap
    ax3 = fig.add_subplot(gs[2, 0])
    monthly_returns = equity.resample('ME').last().pct_change() * 100
    monthly_returns = monthly_returns.dropna()

    if len(monthly_returns) > 0:
        returns_df = pd.DataFrame({
            'year': monthly_returns.index.year,
            'month': monthly_returns.index.month,
            'return': monthly_returns.values
        })
        pivot_table = returns_df.pivot(index='year', columns='month', values='return')
        pivot_table = pivot_table.reindex(columns=range(1, 13))

        # Dynamic color range based on data
        vmax = max(10, abs(pivot_table.values).max())
        vmin = -vmax

        im = ax3.imshow(pivot_table.values, cmap='RdYlGn', aspect='auto',
                       vmin=vmin, vmax=vmax)
        ax3.set_xticks(range(12))
        ax3.set_xticklabels(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                            'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'], fontsize=8)
        ax3.set_yticks(range(len(pivot_table.index)))
        ax3.set_yticklabels(pivot_table.index)
        ax3.set_title("Monthly Returns (%)", fontweight="bold")

        # Add values to cells
        for i in range(len(pivot_table.index)):
            for j in range(12):
                value = pivot_table.iloc[i, j]
                if not pd.isna(value):
                    text_color = 'white' if abs(value) > vmax * 0.5 else 'black'
                    ax3.text(j, i, f'{value:.1f}%',
                            ha='center', va='center', color=text_color, fontsize=7)

        plt.colorbar(im, ax=ax3, label='Return (%)')
    else:
        ax3.text(0.5, 0.5, "Insufficient data for monthly returns",
                ha='center', va='center', transform=ax3.transAxes)

    # Trade statistics
    ax4 = fig.add_subplot(gs[2, 1])
    ax4.axis('off')

    stats_text = f"""
Performance Summary
{'='*30}
Total Return:     {total_return:>10.2f}%
Max Drawdown:     {max_dd:>10.2f}%
Sharpe Ratio:     {sharpe:>10.2f}

Trade Statistics
{'='*30}
Total Trades:     {len(trades):>10}
Buy Trades:       {len(trades[trades['action']=='buy']):>10}
Sell Trades:      {len(trades[trades['action']=='sell']):>10}
Win Rate:         {win_rate:>10.2f}%
"""

    ax4.text(0.05, 0.95, stats_text, transform=ax4.transAxes,
            fontsize=10, verticalalignment='top', family='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

    fig.suptitle(title, fontsize=16, fontweight='bold', y=0.98)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Report saved to {save_path}")

    return fig


if __name__ == "__main__":
    import sys
    sys.path.insert(0, '/Users/kannmu/Library/CloudStorage/OneDrive-Personal/Projects/K-Quant/src')

    from data_loader import load_stock_data
    from strategy.templates import DoubleMovingAverageStrategy
    from backtest.engine import BacktestEngine

    print("Testing Plotting Module with 601012...")

    # Run backtest
    data = load_stock_data("601012", "20240101", "20241231")
    strategy = DoubleMovingAverageStrategy("601012")
    engine = BacktestEngine(initial_cash=100000)
    result = engine.run(strategy, data, "601012")

    # Create plots
    equity = result.equity_curve['total_value']

    print("\nGenerating plots...")

    # Test individual plots
    fig1 = plot_equity_curve(equity, title="601012 Double MA Strategy")
    fig2 = plot_drawdown(equity, title="601012 Drawdown Analysis")

    # Test full report
    fig3 = create_full_report(equity, result.trades, title="601012 Backtest Report")

    plt.show()

    print("\n✅ Plotting test completed!")
