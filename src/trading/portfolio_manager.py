"""
Portfolio Manager for live paper trading and real trading.
Handles position tracking, order execution, and P&L calculations.
"""

import json
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import logging
from dataclasses import dataclass, asdict
from enum import Enum

logger = logging.getLogger(__name__)


class OrderType(Enum):
    MARKET = "market"
    LIMIT = "limit" 
    STOP = "stop"


class OrderSide(Enum):
    BUY = "buy"
    SELL = "sell"


class OrderStatus(Enum):
    PENDING = "pending"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"


@dataclass
class Position:
    symbol: str
    quantity: float
    avg_price: float
    current_price: float = 0.0
    last_updated: str = ""
    
    @property
    def market_value(self) -> float:
        return self.quantity * self.current_price
    
    @property
    def unrealized_pnl(self) -> float:
        return (self.current_price - self.avg_price) * self.quantity
    
    @property
    def unrealized_pnl_percent(self) -> float:
        if self.avg_price == 0:
            return 0.0
        return (self.unrealized_pnl / (self.avg_price * abs(self.quantity))) * 100


@dataclass
class Order:
    id: str
    symbol: str
    side: OrderSide
    order_type: OrderType
    quantity: float
    price: Optional[float] = None
    status: OrderStatus = OrderStatus.PENDING
    created_at: str = ""
    filled_at: Optional[str] = None
    filled_price: Optional[float] = None
    filled_quantity: float = 0.0


@dataclass
class Trade:
    id: str
    symbol: str
    side: OrderSide
    quantity: float
    price: float
    timestamp: str
    commission: float = 0.0


class PortfolioManager:
    """Manages portfolio positions, orders, and trading operations."""
    
    def __init__(self, initial_balance: float = 10000.0, data_client=None, live_trading: bool = False):
        self.initial_balance = initial_balance
        self.data_client = data_client
        self.live_trading = live_trading
        
        # Portfolio state
        self.cash_balance = initial_balance
        self.positions: Dict[str, Position] = {}
        self.orders: Dict[str, Order] = {}
        self.trades: List[Trade] = []
        
        # Performance tracking
        self.daily_pnl = 0.0
        self.total_pnl = 0.0
        self.start_date = datetime.now().isoformat()
        
        # Storage
        self.portfolio_file = "portfolio_state.json"
        
        # Load existing portfolio if available
        self.load_portfolio()
    
    def get_current_price(self, symbol: str) -> float:
        """Get current market price for a symbol."""
        try:
            if self.data_client:
                # Get recent data point
                df = self.data_client.get_historical_data(symbol, "1d", "1m")
                if not df.empty:
                    return float(df['Close'].iloc[-1])
            
            # Fallback to mock prices for demo
            mock_prices = {
                'BTCUSDT': 63500.0,
                'ETHUSDT': 2650.0,
                'AAPL': 178.90,
                'MSFT': 412.45,
                'GOOGL': 2780.15,
                'AMZN': 145.30,
                'TSLA': 248.50
            }
            return mock_prices.get(symbol, 100.0)
            
        except Exception as e:
            logger.error(f"Error getting price for {symbol}: {e}")
            return 100.0  # Fallback price
    
    def update_positions_prices(self):
        """Update current prices for all positions."""
        for symbol, position in self.positions.items():
            current_price = self.get_current_price(symbol)
            position.current_price = current_price
            position.last_updated = datetime.now().isoformat()
    
    def place_order(self, symbol: str, side: OrderSide, quantity: float, 
                   order_type: OrderType = OrderType.MARKET, price: Optional[float] = None) -> str:
        """Place a new order."""
        
        order_id = f"{symbol}_{side.value}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        order = Order(
            id=order_id,
            symbol=symbol,
            side=side,
            order_type=order_type,
            quantity=quantity,
            price=price,
            created_at=datetime.now().isoformat()
        )
        
        # Validate order
        if not self._validate_order(order):
            order.status = OrderStatus.REJECTED
            self.orders[order_id] = order
            return order_id
        
        self.orders[order_id] = order
        
        # Execute immediately for market orders in paper trading
        if order_type == OrderType.MARKET and not self.live_trading:
            self._execute_order(order_id)
        elif self.live_trading:
            # TODO: Submit to actual broker API
            logger.info(f"Live trading order submitted: {order_id}")
        
        self.save_portfolio()
        return order_id
    
    def _validate_order(self, order: Order) -> bool:
        """Validate if order can be placed."""
        
        if order.side == OrderSide.BUY:
            # Check if we have enough cash
            current_price = self.get_current_price(order.symbol)
            required_cash = order.quantity * current_price
            
            if required_cash > self.cash_balance:
                logger.warning(f"Insufficient funds: need ${required_cash:.2f}, have ${self.cash_balance:.2f}")
                return False
                
        elif order.side == OrderSide.SELL:
            # Check if we have enough shares
            position = self.positions.get(order.symbol)
            if not position or position.quantity < order.quantity:
                available = position.quantity if position else 0
                logger.warning(f"Insufficient shares: need {order.quantity}, have {available}")
                return False
        
        return True
    
    def _execute_order(self, order_id: str):
        """Execute a pending order."""
        order = self.orders.get(order_id)
        if not order or order.status != OrderStatus.PENDING:
            return
        
        # Get execution price
        if order.order_type == OrderType.MARKET:
            execution_price = self.get_current_price(order.symbol)
        else:
            execution_price = order.price or self.get_current_price(order.symbol)
        
        # Calculate commission (0.1% for paper trading)
        commission = execution_price * order.quantity * 0.001
        
        # Execute the trade
        if order.side == OrderSide.BUY:
            self._execute_buy(order, execution_price, commission)
        else:
            self._execute_sell(order, execution_price, commission)
        
        # Update order status
        order.status = OrderStatus.FILLED
        order.filled_at = datetime.now().isoformat()
        order.filled_price = execution_price
        order.filled_quantity = order.quantity
        
        # Create trade record
        trade = Trade(
            id=f"trade_{len(self.trades) + 1}",
            symbol=order.symbol,
            side=order.side,
            quantity=order.quantity,
            price=execution_price,
            timestamp=datetime.now().isoformat(),
            commission=commission
        )
        self.trades.append(trade)
        
        logger.info(f"Order executed: {order.side.value} {order.quantity} {order.symbol} @ ${execution_price:.2f}")
    
    def _execute_buy(self, order: Order, price: float, commission: float):
        """Execute a buy order."""
        total_cost = (price * order.quantity) + commission
        
        # Update cash balance
        self.cash_balance -= total_cost
        
        # Update position
        if order.symbol in self.positions:
            # Add to existing position
            position = self.positions[order.symbol]
            total_quantity = position.quantity + order.quantity
            total_cost_basis = (position.avg_price * position.quantity) + (price * order.quantity)
            position.avg_price = total_cost_basis / total_quantity
            position.quantity = total_quantity
        else:
            # Create new position
            self.positions[order.symbol] = Position(
                symbol=order.symbol,
                quantity=order.quantity,
                avg_price=price,
                current_price=price,
                last_updated=datetime.now().isoformat()
            )
    
    def _execute_sell(self, order: Order, price: float, commission: float):
        """Execute a sell order."""
        total_proceeds = (price * order.quantity) - commission
        
        # Update cash balance
        self.cash_balance += total_proceeds
        
        # Update position
        if order.symbol in self.positions:
            position = self.positions[order.symbol]
            position.quantity -= order.quantity
            
            # Remove position if quantity is zero
            if position.quantity <= 0:
                del self.positions[order.symbol]
    
    def get_portfolio_value(self) -> float:
        """Calculate total portfolio value."""
        self.update_positions_prices()
        
        positions_value = sum(pos.market_value for pos in self.positions.values())
        return self.cash_balance + positions_value
    
    def get_daily_pnl(self) -> Tuple[float, float]:
        """Calculate daily P&L (absolute and percentage)."""
        current_value = self.get_portfolio_value()
        
        # For daily P&L, we'd need to track previous day's closing value
        # For now, calculate total return vs initial balance
        total_pnl = current_value - self.initial_balance
        total_pnl_percent = (total_pnl / self.initial_balance) * 100
        
        return total_pnl, total_pnl_percent
    
    def get_portfolio_summary(self) -> Dict:
        """Get comprehensive portfolio summary."""
        self.update_positions_prices()
        
        portfolio_value = self.get_portfolio_value()
        total_pnl, total_pnl_percent = self.get_daily_pnl()
        
        return {
            'portfolio_value': portfolio_value,
            'cash_balance': self.cash_balance,
            'positions_value': sum(pos.market_value for pos in self.positions.values()),
            'total_pnl': total_pnl,
            'total_pnl_percent': total_pnl_percent,
            'open_positions': len(self.positions),
            'pending_orders': len([o for o in self.orders.values() if o.status == OrderStatus.PENDING]),
            'total_trades': len(self.trades),
            'last_updated': datetime.now().isoformat()
        }
    
    def reset_portfolio(self):
        """Reset portfolio to initial state."""
        self.cash_balance = self.initial_balance
        self.positions.clear()
        self.orders.clear()
        self.trades.clear()
        self.save_state()
        self.logger.info("Portfolio reset to initial state")
    
    def save_portfolio(self):
        """Save portfolio state to file."""
        try:
            portfolio_data = {
                'cash_balance': self.cash_balance,
                'initial_balance': self.initial_balance,
                'positions': {k: asdict(v) for k, v in self.positions.items()},
                'orders': {k: asdict(v) for k, v in self.orders.items()},
                'trades': [asdict(t) for t in self.trades],
                'start_date': self.start_date,
                'last_saved': datetime.now().isoformat()
            }
            
            with open(self.portfolio_file, 'w') as f:
                json.dump(portfolio_data, f, indent=2, default=str)
                
        except Exception as e:
            logger.error(f"Error saving portfolio: {e}")
    
    def load_portfolio(self):
        """Load portfolio state from file."""
        try:
            if os.path.exists(self.portfolio_file):
                with open(self.portfolio_file, 'r') as f:
                    data = json.load(f)
                
                self.cash_balance = data.get('cash_balance', self.initial_balance)
                self.initial_balance = data.get('initial_balance', self.initial_balance)
                self.start_date = data.get('start_date', datetime.now().isoformat())
                
                # Load positions
                for symbol, pos_data in data.get('positions', {}).items():
                    self.positions[symbol] = Position(**pos_data)
                
                # Load orders
                for order_id, order_data in data.get('orders', {}).items():
                    # Convert string enums back to enum objects
                    order_data['side'] = OrderSide(order_data['side'])
                    order_data['order_type'] = OrderType(order_data['order_type'])
                    order_data['status'] = OrderStatus(order_data['status'])
                    self.orders[order_id] = Order(**order_data)
                
                # Load trades
                for trade_data in data.get('trades', []):
                    trade_data['side'] = OrderSide(trade_data['side'])
                    self.trades.append(Trade(**trade_data))
                
                logger.info(f"Portfolio loaded: ${self.get_portfolio_value():.2f}")
                
        except Exception as e:
            logger.error(f"Error loading portfolio: {e}")
            # Initialize with defaults
            self.cash_balance = self.initial_balance
    
    def reset_portfolio(self):
        """Reset portfolio to initial state."""
        self.cash_balance = self.initial_balance
        self.positions.clear()
        self.orders.clear()
        self.trades.clear()
        self.start_date = datetime.now().isoformat()
        
        # Remove saved file
        if os.path.exists(self.portfolio_file):
            os.remove(self.portfolio_file)
        
        logger.info("Portfolio reset to initial state")
    
    def buy(self, symbol: str, quantity: float, price: float = None) -> Optional[str]:
        """Convenience method to place a buy order."""
        order_type = OrderType.MARKET if price is None else OrderType.LIMIT
        return self.place_order(symbol, OrderSide.BUY, quantity, order_type, price)
    
    def sell(self, symbol: str, quantity: float, price: float = None) -> Optional[str]:
        """Convenience method to place a sell order.""" 
        order_type = OrderType.MARKET if price is None else OrderType.LIMIT
        return self.place_order(symbol, OrderSide.SELL, quantity, order_type, price)