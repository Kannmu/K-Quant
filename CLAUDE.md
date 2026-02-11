# K-Quant Project Context

## Environment
- 使用 uv 作为 python 的包管理工具
- 虚拟环境名称为 `k-quant`

## Project Overview
- **目标**: 个人 A 股量化交易系统（中低频，持仓周期周/月级别）
- **数据源**: AkShare (主要), Baostock (备用)
- **存储**: Parquet 本地文件
- **核心原则**: 简洁、可扩展、严格遵循 A 股规则

## A-Share Trading Rules
- **佣金**: 0.03% (买卖双向)
- **印花税**: 0.05% (仅卖出)
- **滑点**: 固定 0.1%
- **手数**: 100 股整数倍（向下取整）
- **T+1**: 当日买入不可卖出

## Project Structure
```
quant_system/
├── data/                   # Parquet 数据存储
├── src/
│   ├── data_loader.py      # 数据获取模块
│   ├── strategy/           # 策略逻辑
│   ├── backtest/           # 回测引擎
│   └── analysis/           # 绩效分析
├── main.py                 # 回测入口
└── run_scan.py             # 每日信号扫描
```

## Key Modules
- `data_loader.py`: OHLCV 数据下载、复权、基本面数据、交易日历
- `backtest/broker.py`: A 股交易执行（佣金 0.03%、印花税 0.05%、滑点 0.1%、T+1）
- `backtest/portfolio.py`: 持仓管理、资金跟踪、权益曲线
- `backtest/engine.py`: 回测引擎主循环
- `strategy/base.py`: 策略抽象基类 (Strategy, Signal, Context)
- `strategy/templates.py`: 预置策略 (双均线、RSI、MACD)

## Test Stock
- **隆基绿能 (601012)**: 用于开发和测试的代表性股票

## Usage
```python
from backtest.engine import BacktestEngine
from strategy.templates import DoubleMovingAverageStrategy

engine = BacktestEngine(initial_cash=100000, verbose=True)
strategy = DoubleMovingAverageStrategy("601012", fast=20, slow=60)
result = engine.run(strategy, data, "601012")
```

## Development Progress
- [x] Step 1: 数据基础设施 (data_loader.py)
- [x] Step 2: 回测骨架 (Broker, Portfolio)
- [x] Step 3: 策略引擎 (Strategy, BacktestEngine)
- [ ] Step 4: 分析报告 (Metrics, Plotting)
- [ ] Step 5: 信号扫描 (run_scan.py)

## 迭代闭环

测试 ➜ 定位 ➜ 修复 ➜ 再测，飞轮不息。
