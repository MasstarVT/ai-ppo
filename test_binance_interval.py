#!/usr/bin/env python3
"""
Test script to debug Binance timeframe issue.
This will help us see exactly what's happening with the interval parameter.
"""

import os
import sys
sys.path.append(os.path.dirname(__file__))

from src.data.data_client import DataClient

def test_binance_intervals():
    """Test different intervals with Binance to see what we get."""
    
    # Load config
    import yaml
    config_path = os.path.join(os.path.dirname(__file__), 'config', 'config.yaml')
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Override to use Binance
    config['data_source'] = {'provider': 'binance'}
    
    print("🔍 Testing Binance intervals...")
    print(f"📁 Config file: {config_path}")
    print(f"🔧 Data provider: {config.get('data_source', {}).get('provider', 'NOT SET')}")
    print(f"⏱️  Config timeframe: {config.get('trading', {}).get('timeframe', 'NOT SET')}")
    print("-" * 80)
    
    # Initialize DataClient
    try:
        data_client = DataClient(config)
        print(f"✅ DataClient initialized with provider: {data_client.provider.name}")
    except Exception as e:
        print(f"❌ Failed to initialize DataClient: {e}")
        return
    
    # Test different intervals
    intervals_to_test = ['5m', '1h', '1d']
    symbol = 'BTCUSDT'
    period = '1y'
    
    for interval in intervals_to_test:
        print(f"\n🧪 Testing interval: {interval}")
        print(f"   Parameters: symbol={symbol}, period={period}, interval={interval}")
        
        try:
            data = data_client.get_historical_data(symbol, period, interval)
            
            if data is not None and not data.empty:
                print(f"   ✅ Got {len(data)} data points")
                print(f"   📊 Date range: {data.index[0]} to {data.index[-1]}")
                
                # Calculate expected vs actual
                if interval == '5m':
                    expected = 105120  # 5-min intervals in a year
                    print(f"   📈 Expected for 5m/1y: ~{expected:,}")
                elif interval == '1h':
                    expected = 8760   # hours in a year
                    print(f"   📈 Expected for 1h/1y: ~{expected:,}")
                elif interval == '1d':
                    expected = 365    # days in a year
                    print(f"   📈 Expected for 1d/1y: ~{expected:,}")
                
                print(f"   🎯 Accuracy: {len(data)/expected*100:.1f}% of expected")
                
            else:
                print(f"   ❌ No data returned")
                
        except Exception as e:
            print(f"   ❌ Error: {e}")
    
    print("\n" + "="*80)
    print("✅ Test completed! Check the debug output above.")

if __name__ == "__main__":
    test_binance_intervals()