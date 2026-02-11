# Project K-Quant: Personal A-Share Quantitative Trading Infrastructure for Kannmu

## 1. Project Overview
I need to build a lightweight, modular, and extensible quantitative trading system specifically for the **China A-Share market**. The system is for personal use, focusing on **Mid-to-Long term investment strategies** (holding periods from weeks to months).

**Goal:** To serve as a robust infrastructure for developing custom strategies, backtesting historical performance, and generating trading signals for future dates.

**Core Philosophy:**
*   **Simplicity:** No over-engineering. Use Python scripts and local storage.
*   **Extensibility:** Easy to add new logic without breaking the core.
*   **Accuracy:** Strictly adhere to A-Share trading rules (T+1, Tax, Board Lots).

## 2. Technology Stack
*   **Language:** Python 3.10+
*   **Data Source:** `AkShare` (Primary), `Baostock` (Secondary).
*   **Data Handling:** `Pandas` (Core), `Numpy`.
*   **Storage:** Local File System (`Parquet` format for speed) or `SQLite`.
*   **Visualization:** `Matplotlib` or `Plotly`.
*   **Backtesting Framework:** **Custom implementation** preferred (to strictly control A-share logic) OR a lightweight wrapper around `Backtrader`/`VectorBT`. *Decision: Let's build a clean, custom Event-Driven engine to ensure full understanding of the logic.*

## 3. System Architecture & Modules

The project should be structured as follows:

```text
quant_system/
├── data/                   # Storage for downloaded data (parquet/db)
├── src/
│   ├── data_loader.py      # Fetching and cleaning data
│   ├── strategy/           # Strategy logic
│   │   ├── base.py         # Abstract Strategy Class
│   │   ├── templates.py    # Common strategies (e.g., Moving Average, ROE-based)
│   ├── backtest/           # Backtesting engine
│   │   ├── engine.py       # Main loop
│   │   ├── broker.py       # Order execution simulation (Fees, T+1)
│   │   ├── portfolio.py    # Position & Cash management
│   ├── analysis/           # Performance metrics & plotting
│   ├── utils/              # Helper functions (Calendar, Date parsing)
├── main.py                 # Entry point for running backtests
├── run_scan.py             # Entry point for generating daily signals
└── requirements.txt
```

## 4. Detailed Functional Requirements

### 4.1. Data Module (`data_loader.py`)
*   **Function:** `download_history(stock_list, start_date, end_date)`
    *   Must fetch Daily OHLCV (Open, High, Low, Close, Volume).
    *   Must fetch **Adjusted Factors** (复权因子) to calculate "Forward Adjusted Prices" (前复权).
    *   Must fetch **Fundamental Data**: PE (TTM), PB, ROE, Market Cap.
*   **Storage:** Save data to `data/` directory. Check for existing data to avoid re-downloading.

### 4.2. Strategy Interface (`strategy/`)
*   Create an abstract base class `Strategy`.
*   **Methods:**
    *   `init(self, context)`: Define indicators/factors here.
    *   `next(self, context, bar)`: Called every day. Returns `Signal` (BUY/SELL/HOLD).
*   **Example Strategy:** A simple "Double Moving Average" strategy for testing.

### 4.3. Backtest Engine (`backtest/`)
*   **Broker Logic (`broker.py`):**
    *   **Commission:** 0.03% (buy/sell).
    *   **Stamp Duty:** 0.05% (Sell only) - **Important for A-Shares**.
    *   **Slippage:** Optional fixed percentage (e.g., 0.1%).
    *   **Lot Size:** Round down buy orders to nearest 100 shares.
    *   **T+1 Rule:** Assets bought today cannot be sold today.
*   **Engine Logic (`engine.py`):**
    *   Iterate through dates.
    *   Feed data to Strategy.
    *   Process orders via Broker.
    *   Update Portfolio.

### 4.4. Analysis Module (`analysis/`)
*   Calculate:
    *   **Total Return**.
    *   **Annualized Return (CAGR)**.
    *   **Max Drawdown** (Maximum % loss from peak).
    *   **Sharpe Ratio** (Risk-free rate = 3%).
*   **Visualization:** Plot the Equity Curve vs. Benchmark (e.g., CSI 300 index).

### 4.5. Signal Scanner (`run_scan.py`)
*   This script is for "Live" usage.
*   It should load the *latest* available data.
*   Run the selected strategy on a list of stocks.
*   Output a DataFrame: `Stock Code | Action (Buy/Sell) | Current Price | Reason`.

## 5. Development Steps (Instructions for AI)

Please implement the system in the following order. **Write code for one step at a time and ask for confirmation before proceeding.**

1.  **Step 1: Data Infrastructure.** Implement `data_loader.py` using AkShare to fetch daily bars and basic fundamentals for a single stock (e.g., "600519" Kweichow Moutai). Save to Parquet.
2.  **Step 2: Basic Backtest Skeleton.** Implement the `Portfolio` and `Broker` classes with A-Share specific fee/tax rules.
3.  **Step 3: Strategy & Engine.** Implement the `Strategy` base class and the `BacktestEngine` loop. Create a demo strategy (e.g., MA20 > MA60).
4.  **Step 4: Reporting.** Implement the metric calculations and a simple Matplotlib chart.
5.  **Step 5: Scanner.** Create the script to scan for tomorrow's buy signals.

## 6. Specific Constraints
*   **Error Handling:** Gracefully handle missing data or stock suspensions.
*   **Code Style:** Use Python Type Hints and Docstrings.
*   **Libraries:** Keep dependencies minimal (`pandas`, `akshare`, `matplotlib`).
