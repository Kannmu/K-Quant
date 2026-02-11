"""
Broker Module - Simulates A-Share Order Execution

Handles order validation, fee calculation, and T+1 settlement rules.
"""

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional

import pandas as pd


class OrderType(Enum):
    """Order type enumeration."""
    BUY = "buy"
    SELL = "sell"


class OrderStatus(Enum):
    """Order execution status."""
    PENDING = "pending"
    FILLED = "filled"
    REJECTED = "rejected"
    CANCELLED = "cancelled"


@dataclass
class Order:
    """Represents a trading order."""
    stock_code: str
    order_type: OrderType
    quantity: int  # Must be multiple of 100 for A-shares
    price: float   # Limit price, or market price if 0
    date: datetime
    order_id: Optional[str] = None
    status: OrderStatus = OrderStatus.PENDING
    filled_price: float = 0.0
    filled_quantity: int = 0
    commission: float = 0.0
    stamp_tax: float = 0.0
    slippage_cost: float = 0.0
    reject_reason: str = ""

    def __post_init__(self):
        if self.order_id is None:
            self.order_id = f"{self.date.strftime('%Y%m%d')}_{self.stock_code}_{self.order_type.value}"


@dataclass
class FillResult:
    """Result of order execution."""
    order: Order
    success: bool
    actual_price: float
    actual_quantity: int
    total_cost: float  # Including fees
    message: str = ""


class Broker:
    """
    A-Share Broker Simulator

    Implements A-share specific rules:
    - Commission: 0.03% (min 5 CNY) on both buy/sell
    - Stamp Tax: 0.05% on sell only
    - Slippage: Fixed percentage (e.g., 0.1%)
    - Lot Size: Round down to nearest 100 shares
    - T+1: Cannot sell stocks bought today
    """

    def __init__(
        self,
        commission_rate: float = 0.0003,  # 0.03%
        min_commission: float = 5.0,       # Min 5 CNY
        stamp_tax_rate: float = 0.0005,   # 0.05% on sell
        slippage_rate: float = 0.001,     # 0.1%
    ):
        self.commission_rate = commission_rate
        self.min_commission = min_commission
        self.stamp_tax_rate = stamp_tax_rate
        self.slippage_rate = slippage_rate

        # Track executed orders
        self.orders: List[Order] = []
        self.trades: List[Dict] = []

    def _calculate_fees(self, order_type: OrderType, amount: float) -> tuple[float, float, float]:
        """
        Calculate trading fees for a given order.

        Returns:
            Tuple of (commission, stamp_tax, slippage_cost)
        """
        # Commission (both buy and sell)
        commission = max(amount * self.commission_rate, self.min_commission)

        # Stamp tax (sell only in A-shares)
        stamp_tax = amount * self.stamp_tax_rate if order_type == OrderType.SELL else 0.0

        # Slippage (applied to both buy and sell)
        slippage_cost = amount * self.slippage_rate

        return commission, stamp_tax, slippage_cost

    def _validate_lot_size(self, quantity: int) -> tuple[bool, int, str]:
        """
        Validate and adjust quantity to meet A-share lot size requirement.

        A-shares must be traded in multiples of 100 shares (1 lot).

        Returns:
            Tuple of (is_valid, adjusted_quantity, message)
        """
        if quantity <= 0:
            return False, 0, "Quantity must be positive"

        # Round down to nearest 100
        adjusted = (quantity // 100) * 100

        if adjusted == 0:
            return False, 0, f"Quantity {quantity} too small (min 100 shares)"

        if adjusted != quantity:
            return True, adjusted, f"Quantity rounded down from {quantity} to {adjusted}"

        return True, quantity, "Valid"

    def _check_t1_rule(
        self,
        order_type: OrderType,
        stock_code: str,
        current_date: datetime,
        holdings_buy_dates: Dict[str, List[datetime]]
    ) -> tuple[bool, str]:
        """
        Check T+1 rule - cannot sell stocks bought today.

        Args:
            order_type: Buy or Sell
            stock_code: Stock code
            current_date: Current trading date
            holdings_buy_dates: Dict mapping stock_code to list of buy dates

        Returns:
            Tuple of (can_trade, reason)
        """
        if order_type == OrderType.BUY:
            return True, ""

        # Check if any shares were bought today
        buy_dates = holdings_buy_dates.get(stock_code, [])
        today_buys = [d for d in buy_dates if d.date() == current_date.date()]

        if today_buys:
            return False, f"T+1 rule: Cannot sell {stock_code} bought today"

        return True, ""

    def submit_order(
        self,
        stock_code: str,
        order_type: OrderType,
        quantity: int,
        price: float,
        date: datetime,
        available_cash: float = float('inf'),
        current_holdings: Dict[str, int] = None,
        holdings_buy_dates: Dict[str, List[datetime]] = None,
    ) -> FillResult:
        """
        Submit and execute an order with A-share rules.

        Args:
            stock_code: 6-digit stock code
            order_type: BUY or SELL
            quantity: Number of shares (will be rounded to 100s)
            price: Target price (0 for market order)
            date: Trading date
            available_cash: Available cash for buying
            current_holdings: Current position holdings
            holdings_buy_dates: Track when each position was bought (for T+1)

        Returns:
            FillResult with execution details
        """
        if current_holdings is None:
            current_holdings = {}
        if holdings_buy_dates is None:
            holdings_buy_dates = {}

        # Create order
        order = Order(
            stock_code=stock_code,
            order_type=order_type,
            quantity=quantity,
            price=price,
            date=date,
        )

        # Validate lot size
        valid, adjusted_qty, msg = self._validate_lot_size(quantity)
        if not valid:
            order.status = OrderStatus.REJECTED
            order.reject_reason = msg
            self.orders.append(order)
            return FillResult(order, False, 0, 0, 0, msg)

        order.quantity = adjusted_qty
        if msg != "Valid":
            print(f"  [Broker] {msg}")

        # Check T+1 rule for sells
        if order_type == OrderType.SELL:
            can_sell, reason = self._check_t1_rule(
                order_type, stock_code, date, holdings_buy_dates
            )
            if not can_sell:
                order.status = OrderStatus.REJECTED
                order.reject_reason = reason
                self.orders.append(order)
                return FillResult(order, False, 0, 0, 0, reason)

        # Check holdings for sells
        if order_type == OrderType.SELL:
            held = current_holdings.get(stock_code, 0)
            if held < adjusted_qty:
                msg = f"Insufficient holdings: have {held}, want to sell {adjusted_qty}"
                order.status = OrderStatus.REJECTED
                order.reject_reason = msg
                self.orders.append(order)
                return FillResult(order, False, 0, 0, 0, msg)

        # Calculate execution price with slippage
        if price <= 0:
            msg = "Market orders not supported, please provide limit price"
            order.status = OrderStatus.REJECTED
            order.reject_reason = msg
            self.orders.append(order)
            return FillResult(order, False, 0, 0, 0, msg)

        # Apply slippage
        if order_type == OrderType.BUY:
            # Buy at higher price (slippage increases cost)
            execution_price = price * (1 + self.slippage_rate)
        else:
            # Sell at lower price (slippage reduces proceeds)
            execution_price = price * (1 - self.slippage_rate)

        order.filled_price = execution_price
        order.filled_quantity = adjusted_qty

        # Calculate gross amount
        gross_amount = execution_price * adjusted_qty

        # Calculate fees
        commission, stamp_tax, slippage_cost = self._calculate_fees(order_type, gross_amount)
        order.commission = commission
        order.stamp_tax = stamp_tax
        order.slippage_cost = slippage_cost

        # Total cost/proceeds
        if order_type == OrderType.BUY:
            total_cost = gross_amount + commission  # No stamp tax on buy
            # Check cash sufficiency
            if total_cost > available_cash:
                msg = f"Insufficient cash: need {total_cost:.2f}, have {available_cash:.2f}"
                order.status = OrderStatus.REJECTED
                order.reject_reason = msg
                self.orders.append(order)
                return FillResult(order, False, 0, 0, 0, msg)
        else:
            total_cost = gross_amount - commission - stamp_tax  # Net proceeds

        # Mark as filled
        order.status = OrderStatus.FILLED
        self.orders.append(order)

        # Record trade
        self.trades.append({
            "date": date,
            "stock_code": stock_code,
            "action": order_type.value,
            "quantity": adjusted_qty,
            "price": execution_price,
            "gross_amount": gross_amount,
            "commission": commission,
            "stamp_tax": stamp_tax,
            "total_cost": total_cost if order_type == OrderType.BUY else -total_cost,
        })

        return FillResult(order, True, execution_price, adjusted_qty, total_cost)

    def get_trade_history(self) -> pd.DataFrame:
        """Get trade history as DataFrame."""
        if not self.trades:
            return pd.DataFrame()
        return pd.DataFrame(self.trades)

    def get_order_summary(self) -> Dict:
        """Get summary statistics of orders."""
        if not self.orders:
            return {"total": 0, "filled": 0, "rejected": 0}

        filled = sum(1 for o in self.orders if o.status == OrderStatus.FILLED)
        rejected = sum(1 for o in self.orders if o.status == OrderStatus.REJECTED)

        return {
            "total": len(self.orders),
            "filled": filled,
            "rejected": rejected,
            "total_commission": sum(o.commission for o in self.orders),
            "total_stamp_tax": sum(o.stamp_tax for o in self.orders),
        }


if __name__ == "__main__":
    # Test broker with 601012 (LONGi Green Energy)
    print("Testing Broker with 601012 (隆基绿能)...")
    print("=" * 50)

    broker = Broker()

    test_date = datetime(2024, 1, 15)
    test_code = "601012"
    test_price = 20.0

    # Test 1: Buy order
    print("\n1. Testing BUY order (1000 shares @ 20.0)...")
    result = broker.submit_order(
        stock_code=test_code,
        order_type=OrderType.BUY,
        quantity=1000,
        price=test_price,
        date=test_date,
        available_cash=50000,
    )
    print(f"   Success: {result.success}")
    print(f"   Filled Qty: {result.actual_quantity}")
    print(f"   Filled Price: {result.actual_price:.4f}")
    print(f"   Total Cost: {result.total_cost:.2f}")
    if result.order.status == OrderStatus.FILLED:
        print(f"   Commission: {result.order.commission:.2f}")
        print(f"   Slippage: {result.order.slippage_cost:.2f}")

    # Test 2: Lot size adjustment
    print("\n2. Testing lot size adjustment (550 shares)...")
    result = broker.submit_order(
        stock_code=test_code,
        order_type=OrderType.BUY,
        quantity=550,
        price=test_price,
        date=test_date,
        available_cash=50000,
    )
    print(f"   Requested: 550, Filled: {result.actual_quantity}")

    # Test 3: Sell with T+1 check
    print("\n3. Testing SELL with T+1 rule...")
    holdings = {test_code: 1000}
    # Buy today
    today_buys = {test_code: [test_date]}

    result = broker.submit_order(
        stock_code=test_code,
        order_type=OrderType.SELL,
        quantity=500,
        price=test_price,
        date=test_date,
        current_holdings=holdings,
        holdings_buy_dates=today_buys,
    )
    print(f"   Success: {result.success}")
    print(f"   Message: {result.message}")

    # Test 4: Sell next day (should work)
    print("\n4. Testing SELL next day (T+1 satisfied)...")
    next_day = datetime(2024, 1, 16)
    result = broker.submit_order(
        stock_code=test_code,
        order_type=OrderType.SELL,
        quantity=500,
        price=test_price,
        date=next_day,
        current_holdings=holdings,
        holdings_buy_dates=today_buys,  # Bought on 1/15
    )
    print(f"   Success: {result.success}")
    if result.success:
        print(f"   Filled Price: {result.actual_price:.4f}")
        print(f"   Commission: {result.order.commission:.2f}")
        print(f"   Stamp Tax: {result.order.stamp_tax:.2f}")
        print(f"   Net Proceeds: {-result.total_cost:.2f}")

    # Test 5: Insufficient cash
    print("\n5. Testing insufficient cash...")
    result = broker.submit_order(
        stock_code=test_code,
        order_type=OrderType.BUY,
        quantity=10000,
        price=test_price,
        date=next_day,
        available_cash=1000,  # Not enough
    )
    print(f"   Success: {result.success}")
    print(f"   Message: {result.message}")

    # Summary
    print("\n" + "=" * 50)
    print("Order Summary:")
    summary = broker.get_order_summary()
    for key, value in summary.items():
        if isinstance(value, float):
            print(f"   {key}: {value:.2f}")
        else:
            print(f"   {key}: {value}")

    print("\n✅ Broker test completed!")
