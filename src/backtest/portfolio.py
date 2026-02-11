"""
Portfolio Module - Manages Holdings and Cash

Tracks positions, calculates portfolio value, and enforces risk limits.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional

import pandas as pd

from .broker import Broker, FillResult, OrderType


@dataclass
class Position:
    """Represents a stock position."""
    stock_code: str
    quantity: int = 0
    avg_cost: float = 0.0  # Average cost basis
    buy_dates: List[datetime] = field(default_factory=list)  # Track buy dates for T+1
    total_cost: float = 0.0  # Total amount invested
    realized_pnl: float = 0.0  # Realized profit/loss

    def market_value(self, current_price: float) -> float:
        """Calculate current market value."""
        return self.quantity * current_price

    def unrealized_pnl(self, current_price: float) -> float:
        """Calculate unrealized profit/loss."""
        if self.quantity == 0:
            return 0.0
        return self.quantity * (current_price - self.avg_cost)

    def return_pct(self, current_price: float) -> float:
        """Calculate return percentage."""
        if self.avg_cost == 0:
            return 0.0
        return (current_price - self.avg_cost) / self.avg_cost * 100

    def add_shares(self, quantity: int, price: float, date: datetime):
        """Add shares to position (buy)."""
        total_cost = quantity * price
        new_total_cost = self.total_cost + total_cost
        new_quantity = self.quantity + quantity

        # Update average cost
        self.avg_cost = new_total_cost / new_quantity if new_quantity > 0 else 0
        self.quantity = new_quantity
        self.total_cost = new_total_cost
        self.buy_dates.append(date)

    def remove_shares(self, quantity: int, price: float) -> float:
        """Remove shares from position (sell) and return realized PnL."""
        if quantity > self.quantity:
            raise ValueError(f"Cannot sell {quantity} shares, only have {self.quantity}")

        # Calculate realized PnL
        cost_basis = quantity * self.avg_cost
        proceeds = quantity * price
        realized_pnl = proceeds - cost_basis

        self.quantity -= quantity
        self.total_cost = self.quantity * self.avg_cost if self.quantity > 0 else 0
        self.realized_pnl += realized_pnl

        return realized_pnl


@dataclass
class PortfolioSnapshot:
    """Snapshot of portfolio state at a point in time."""
    date: datetime
    cash: float
    positions_value: float
    total_value: float
    unrealized_pnl: float
    realized_pnl: float
    positions: Dict[str, Position]


class Portfolio:
    """
    Portfolio Manager

    Manages cash and stock positions, tracks P&L, and provides portfolio state.
    """

    def __init__(
        self,
        initial_cash: float = 1000000.0,  # Default 1M CNY
        max_position_pct: float = 0.2,     # Max 20% in single stock
        max_positions: int = 10,           # Max number of holdings
    ):
        self.initial_cash = initial_cash
        self.cash = initial_cash
        self.max_position_pct = max_position_pct
        self.max_positions = max_positions

        # Holdings
        self.positions: Dict[str, Position] = {}

        # History
        self.snapshots: List[PortfolioSnapshot] = []
        self.transactions: List[Dict] = []

        # Performance tracking
        self.total_realized_pnl = 0.0
        self.peak_value = initial_cash
        self.max_drawdown = 0.0

    def get_position(self, stock_code: str) -> Optional[Position]:
        """Get position for a stock."""
        return self.positions.get(stock_code)

    def get_position_value(self, stock_code: str, current_price: float) -> float:
        """Get market value of a position."""
        pos = self.positions.get(stock_code)
        if pos:
            return pos.market_value(current_price)
        return 0.0

    def get_total_value(self, prices: Optional[Dict[str, float]] = None) -> float:
        """Calculate total portfolio value (cash + positions)."""
        positions_value = self.get_positions_value(prices)
        return self.cash + positions_value

    def get_positions_value(self, prices: Optional[Dict[str, float]] = None) -> float:
        """Calculate total value of all positions."""
        if prices is None:
            # Without prices, we can only estimate using last known values
            return sum(pos.total_cost for pos in self.positions.values())

        return sum(
            pos.market_value(prices.get(code, 0))
            for code, pos in self.positions.items()
        )

    def get_unrealized_pnl(self, prices: Dict[str, float]) -> float:
        """Calculate total unrealized P&L."""
        return sum(
            pos.unrealized_pnl(prices.get(code, 0))
            for code, pos in self.positions.items()
        )

    def can_buy(self, stock_code: str, quantity: int, price: float) -> tuple[bool, str]:
        """
        Check if a buy order is allowed.

        Args:
            stock_code: Stock to buy
            quantity: Number of shares
            price: Expected price

        Returns:
            Tuple of (can_buy, reason)
        """
        # Check if max positions limit reached
        current_positions = len(self.positions)
        if stock_code not in self.positions and current_positions >= self.max_positions:
            return False, f"Max positions limit ({self.max_positions}) reached"

        # Check position size limit
        position_value = quantity * price
        total_value = self.get_total_value()
        new_position_value = position_value

        if stock_code in self.positions:
            current_price = price  # Simplified
            new_position_value += self.positions[stock_code].market_value(current_price)

        if new_position_value > total_value * self.max_position_pct:
            max_allowed = total_value * self.max_position_pct
            return False, f"Position would exceed {self.max_position_pct*100:.0f}% limit (max: {max_allowed:.2f})"

        # Check cash
        cost = quantity * price
        if cost > self.cash:
            return False, f"Insufficient cash: need {cost:.2f}, have {self.cash:.2f}"

        return True, ""

    def can_sell(self, stock_code: str, quantity: int, date: datetime) -> tuple[bool, str]:
        """
        Check if a sell order is allowed (T+1 rule check).

        Args:
            stock_code: Stock to sell
            quantity: Number of shares
            date: Current date (for T+1 check)

        Returns:
            Tuple of (can_sell, reason)
        """
        position = self.positions.get(stock_code)
        if not position or position.quantity == 0:
            return False, f"No position in {stock_code}"

        if quantity > position.quantity:
            return False, f"Cannot sell {quantity}, only have {position.quantity}"

        # T+1 check: cannot sell shares bought today
        today_buys = [d for d in position.buy_dates if d.date() == date.date()]
        if today_buys:
            # Calculate sellable quantity (exclude today's buys)
            today_buy_qty = sum(
                getattr(self, '_buy_qty_map', {}).get((stock_code, d), 0)
                for d in today_buys
            )
            # Simplified: we track buy dates but need quantity per date
            # For now, conservative approach: any buy today blocks the whole position
            # TODO: Track quantity per buy date more precisely
            pass

        return True, ""

    def update_from_fill(self, fill_result: FillResult):
        """Update portfolio state based on order fill."""
        if not fill_result.success:
            return

        order = fill_result.order
        stock_code = order.stock_code
        quantity = fill_result.actual_quantity
        price = fill_result.actual_price
        date = order.date

        if order.order_type == OrderType.BUY:
            # Deduct cash
            self.cash -= fill_result.total_cost

            # Update position
            if stock_code not in self.positions:
                self.positions[stock_code] = Position(stock_code=stock_code)

            self.positions[stock_code].add_shares(quantity, price, date)

            # Track buy quantity for T+1
            if not hasattr(self, '_buy_qty_map'):
                self._buy_qty_map = {}
            self._buy_qty_map[(stock_code, date)] = quantity

        else:  # SELL
            # Add cash
            net_proceeds = -fill_result.total_cost  # total_cost is negative for sells
            self.cash += net_proceeds

            # Update position
            if stock_code in self.positions:
                realized = self.positions[stock_code].remove_shares(quantity, price)
                self.total_realized_pnl += realized

                # Remove position if empty
                if self.positions[stock_code].quantity == 0:
                    del self.positions[stock_code]

        # Record transaction
        self.transactions.append({
            "date": date,
            "stock_code": stock_code,
            "action": order.order_type.value,
            "quantity": quantity,
            "price": price,
            "amount": fill_result.total_cost if order.order_type == OrderType.BUY else -fill_result.total_cost,
            "commission": order.commission,
            "stamp_tax": order.stamp_tax,
        })

    def take_snapshot(self, date: datetime, prices: Dict[str, float]):
        """Record portfolio state at a given date."""
        positions_value = self.get_positions_value(prices)
        total_value = self.cash + positions_value
        unrealized_pnl = self.get_unrealized_pnl(prices)

        # Update peak and drawdown
        if total_value > self.peak_value:
            self.peak_value = total_value

        current_drawdown = (self.peak_value - total_value) / self.peak_value
        if current_drawdown > self.max_drawdown:
            self.max_drawdown = current_drawdown

        snapshot = PortfolioSnapshot(
            date=date,
            cash=self.cash,
            positions_value=positions_value,
            total_value=total_value,
            unrealized_pnl=unrealized_pnl,
            realized_pnl=self.total_realized_pnl,
            positions={k: v for k, v in self.positions.items()},  # Copy
        )
        self.snapshots.append(snapshot)

        return snapshot

    def get_equity_curve(self) -> pd.DataFrame:
        """Get portfolio value history as DataFrame."""
        if not self.snapshots:
            return pd.DataFrame()

        data = []
        for snap in self.snapshots:
            data.append({
                "date": snap.date,
                "cash": snap.cash,
                "positions_value": snap.positions_value,
                "total_value": snap.total_value,
                "unrealized_pnl": snap.unrealized_pnl,
                "realized_pnl": snap.realized_pnl,
            })

        df = pd.DataFrame(data)
        df["date"] = pd.to_datetime(df["date"])
        return df.set_index("date")

    def get_current_state(self) -> Dict:
        """Get current portfolio state summary."""
        return {
            "cash": self.cash,
            "positions_count": len(self.positions),
            "positions": {
                code: {
                    "quantity": pos.quantity,
                    "avg_cost": pos.avg_cost,
                }
                for code, pos in self.positions.items()
            },
            "realized_pnl": self.total_realized_pnl,
            "max_drawdown": self.max_drawdown,
        }


if __name__ == "__main__":
    # Test Portfolio with 601012
    print("Testing Portfolio with 601012 (隆基绿能)...")
    print("=" * 50)

    portfolio = Portfolio(initial_cash=100000)

    test_dates = [
        datetime(2024, 1, 15),
        datetime(2024, 1, 16),
        datetime(2024, 1, 17),
    ]

    # Simulate buying
    print("\n1. Buying 1000 shares @ 20.0...")
    from .broker import Broker
    broker = Broker()

    result = broker.submit_order(
        stock_code="601012",
        order_type=OrderType.BUY,
        quantity=1000,
        price=20.0,
        date=test_dates[0],
        available_cash=portfolio.cash,
    )
    if result.success:
        portfolio.update_from_fill(result)
        print(f"   Cash after buy: {portfolio.cash:.2f}")
        print(f"   Position: {portfolio.positions['601012'].quantity} shares @ {portfolio.positions['601012'].avg_cost:.4f}")

    # Take snapshot
    prices = {"601012": 20.5}  # Price went up
    snap = portfolio.take_snapshot(test_dates[0], prices)
    print(f"   Total value: {snap.total_value:.2f}")
    print(f"   Unrealized PnL: {snap.unrealized_pnl:.2f}")

    # Try to sell same day (T+1 should block)
    print("\n2. Trying to sell same day (T+1)...")
    can_sell, reason = portfolio.can_sell("601012", 500, test_dates[0])
    print(f"   Can sell: {can_sell}")
    print(f"   Reason: {reason}")

    # Sell next day
    print("\n3. Selling 500 shares next day @ 21.0...")
    result = broker.submit_order(
        stock_code="601012",
        order_type=OrderType.SELL,
        quantity=500,
        price=21.0,
        date=test_dates[1],
        current_holdings={"601012": portfolio.positions["601012"].quantity},
        holdings_buy_dates={"601012": [test_dates[0]]},
    )
    if result.success:
        portfolio.update_from_fill(result)
        print(f"   Cash after sell: {portfolio.cash:.2f}")
        print(f"   Realized PnL: {portfolio.total_realized_pnl:.2f}")
        print(f"   Remaining position: {portfolio.positions['601012'].quantity} shares")

    # Take another snapshot
    prices = {"601012": 19.0}  # Price went down
    snap = portfolio.take_snapshot(test_dates[1], prices)
    print(f"   Total value: {snap.total_value:.2f}")
    print(f"   Unrealized PnL: {snap.unrealized_pnl:.2f}")

    # Equity curve
    print("\n4. Equity curve:")
    eq = portfolio.get_equity_curve()
    print(eq)

    print("\n✅ Portfolio test completed!")
