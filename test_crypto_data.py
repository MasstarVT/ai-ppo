#!/usr/bin/env python3
"""
Test script to verify cryptocurrency data fetching functionality.
"""

import sys
import os

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.data.data_client import DataClient

def test_crypto_data():
    """Test fetching cryptocurrency data."""
    print("🧪 Testing Cryptocurrency Data Fetching")
    print("=" * 50)
    
    # Create a minimal config
    config = {
        'tradingview': {'provider': 'yfinance'}
    }
    
    # Initialize data client
    data_client = DataClient(config)
    
    # Test crypto symbols
    crypto_symbols = ['BTC/USDT', 'ETH/USDT', 'ADA/USDT']
    stock_symbols = ['AAPL', 'MSFT']
    
    # Test each symbol
    for symbol in crypto_symbols + stock_symbols:
        print(f"\n📊 Testing {symbol}...")
        try:
            data = data_client.get_historical_data(symbol, period="5d", interval="1h")
            if not data.empty:
                print(f"✅ Success! Retrieved {len(data)} rows")
                print(f"   Date range: {data.index[0]} to {data.index[-1]}")
                print(f"   Columns: {list(data.columns)}")
                print(f"   Last price: ${data['Close'].iloc[-1]:.2f}")
            else:
                print(f"❌ No data retrieved for {symbol}")
        except Exception as e:
            print(f"❌ Error fetching {symbol}: {e}")
    
    print("\n" + "=" * 50)
    print("🎉 Crypto data test completed!")

if __name__ == "__main__":
    test_crypto_data()