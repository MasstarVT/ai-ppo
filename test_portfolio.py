#!/usr/bin/env python3
"""
Test script for PortfolioManager functionality
"""

import sys
import os

# Add the project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

from src.trading.portfolio_manager import PortfolioManager
from src.data.data_client import DataClient

def test_portfolio_manager():
    """Test basic PortfolioManager functionality."""
    print("🧪 Testing PortfolioManager...")
    
    # Initialize data client
    config = {
        'data_provider': 'binance',
        'symbols': ['BTCUSDT', 'ETHUSDT']
    }
    
    data_client = DataClient(config)
    
    # Initialize portfolio manager
    portfolio = PortfolioManager(
        initial_balance=10000.0,
        data_client=data_client,
        live_trading=False
    )
    
    print(f"✅ Portfolio initialized with ${portfolio.cash_balance:,.2f}")
    
    # Test getting current price
    try:
        btc_price = portfolio.get_current_price('BTCUSDT')
        print(f"✅ BTC Price: ${btc_price:.2f}")
    except Exception as e:
        print(f"❌ Error getting BTC price: {e}")
        return False
    
    # Test buy order
    try:
        order_id = portfolio.buy('BTCUSDT', 0.01, btc_price)
        if order_id:
            print(f"✅ Buy order placed: {order_id}")
        else:
            print("❌ Buy order failed")
            return False
    except Exception as e:
        print(f"❌ Error placing buy order: {e}")
        return False
    
    # Test portfolio summary
    try:
        summary = portfolio.get_portfolio_summary()
        print(f"✅ Portfolio Value: ${summary['portfolio_value']:,.2f}")
        print(f"✅ Open Positions: {summary['open_positions']}")
        print(f"✅ Total Trades: {summary['total_trades']}")
    except Exception as e:
        print(f"❌ Error getting portfolio summary: {e}")
        return False
    
    # Test reset functionality
    try:
        portfolio.reset_portfolio()
        print("✅ Portfolio reset successful")
    except Exception as e:
        print(f"❌ Error resetting portfolio: {e}")
        return False
    
    print("🎉 All tests passed!")
    return True

if __name__ == "__main__":
    success = test_portfolio_manager()
    sys.exit(0 if success else 1)