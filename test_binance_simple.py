#!/usr/bin/env python3
"""
Simple test script to check what Binance actually returns.
"""

import os
import sys
sys.path.append(os.path.dirname(__file__))

def test_binance_simple():
    """Simple test to see what Binance returns for different periods."""
    
    try:
        from binance.client import Client as BinanceClient
    except ImportError:
        print("❌ Binance library not available")
        return
    
    # Get API keys from environment
    api_key = os.getenv('BINANCE_API_KEY')
    secret_key = os.getenv('BINANCE_SECRET_KEY')
    
    if not api_key or not secret_key or api_key == 'your_binance_api_key_here':
        print("❌ Binance API keys not set in environment")
        print("   Please set BINANCE_API_KEY and BINANCE_SECRET_KEY")
        return
    
    print("🔍 Testing Binance API directly...")
    
    # Initialize client
    client = BinanceClient(api_key, secret_key, tld='us')
    
    # Test different combinations
    tests = [
        ('BTCUSDT', '5m', '1 month ago UTC'),  # 1 month of 5-minute data
        ('BTCUSDT', '5m', '3 months ago UTC'), # 3 months of 5-minute data  
        ('BTCUSDT', '5m', '1 year ago UTC'),   # 1 year of 5-minute data
        ('BTCUSDT', '1h', '1 year ago UTC'),   # 1 year of hourly data
    ]
    
    for symbol, interval, start_str in tests:
        print(f"\n🧪 Test: {symbol}, {interval}, {start_str}")
        
        try:
            klines = client.get_historical_klines(symbol, interval, start_str)
            count = len(klines) if klines else 0
            
            if count > 0:
                # Calculate expected count
                if interval == '5m':
                    if 'month' in start_str:
                        if '1 month' in start_str:
                            expected = 30 * 24 * 12  # ~8,640 for 1 month
                        else:  # 3 months
                            expected = 90 * 24 * 12  # ~25,920 for 3 months
                    else:  # 1 year
                        expected = 365 * 24 * 12  # ~105,120 for 1 year
                elif interval == '1h':
                    expected = 365 * 24  # ~8,760 for 1 year
                
                print(f"   ✅ Got {count:,} data points")
                print(f"   📊 Expected: ~{expected:,}")
                print(f"   🎯 Accuracy: {count/expected*100:.1f}%")
                
                # Check first and last timestamps
                if klines:
                    first_ts = klines[0][0] / 1000  # Convert from ms
                    last_ts = klines[-1][0] / 1000
                    
                    from datetime import datetime
                    first_date = datetime.fromtimestamp(first_ts)
                    last_date = datetime.fromtimestamp(last_ts)
                    
                    print(f"   📅 Date range: {first_date} to {last_date}")
            else:
                print(f"   ❌ Got 0 data points")
                
        except Exception as e:
            print(f"   ❌ Error: {e}")

if __name__ == "__main__":
    test_binance_simple()