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
└── update_data.py          # 数据更新脚本
```

## Key Modules
- `data_loader.py`: OHLCV 数据下载、复权、基本面数据、交易日历
- `backtest/broker.py`: A 股交易执行（佣金 0.03%、印花税 0.05%、滑点 0.1%、T+1）
- `backtest/portfolio.py`: 持仓管理、资金跟踪、权益曲线
- `backtest/engine.py`: 回测引擎主循环
- `strategy/base.py`: 策略抽象基类 (Strategy, Signal, Context)
- `strategy/templates.py`: 预置策略 (双均线、RSI、MACD)
- `analysis/metrics.py`: 绩效指标计算 (Sharpe, Sortino, Calmar, Expectancy)
- `analysis/plotting.py`: 可视化图表 (权益曲线、回撤、月度收益)

## Test Stock
- **隆基绿能 (601012)**: 用于开发和测试的代表性股票

## Usage

### 数据更新 (手动执行)
```bash
# 更新单只股票数据（全量覆盖）
uv run python update_data.py --stock 601012

# 更新指定日期范围
uv run python update_data.py --stock 601012 --start 20200101 --end 20241231

# 查看股票列表
uv run python update_data.py --list
```

### 回测
```bash
# 双均线策略回测
uv run python main.py --stock 601012 --strategy ma --start 20240101 --end 20241231

# RSI 策略
uv run python main.py --stock 601012 --strategy rsi

# 保存分析报告
uv run python main.py --stock 601012 --strategy ma --save-plot report.png
```

### Python API
```python
# 数据获取
from data_loader import load_stock_data
data = load_stock_data("601012", "20240101", "20241231")

# 回测
from backtest.engine import BacktestEngine
from strategy.templates import DoubleMovingAverageStrategy

engine = BacktestEngine(initial_cash=100000, verbose=True)
strategy = DoubleMovingAverageStrategy("601012", fast=20, slow=60)
result = engine.run(strategy, data, "601012")

# 绩效分析
from analysis.metrics import calculate_all_metrics, print_detailed_metrics
metrics = calculate_all_metrics(result.equity_curve['total_value'], result.trades)
print_detailed_metrics(metrics)
```

## CLI Commands Summary
| 命令 | 用途 |
|------|------|
| `update_data.py --stock CODE` | 更新单只股票数据 |
| `update_data.py --list` | 列出所有股票 |
| `main.py --stock CODE --strategy STRAT` | 执行回测 |
| `main.py --stock CODE --save-plot PATH` | 回测并保存图表 |

## Development Progress
- [x] Step 1: 数据基础设施 (data_loader.py)
- [x] Step 2: 回测骨架 (Broker, Portfolio)
- [x] Step 3: 策略引擎 (Strategy, BacktestEngine)
- [x] Step 4: 分析报告 (Metrics, Plotting)
- [x] Step 5: 数据更新脚本 (update_data.py)

## 迭代闭环

测试 ➜ 定位 ➜ 修复 ➜ 再测，飞轮不息。
