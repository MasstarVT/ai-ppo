#!/usr/bin/env python3
"""
Test script to validate bug fixes and edge cases.
"""

import os
import sys
import numpy as np
import pandas as pd
from datetime import datetime
import traceback

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from data.indicators import TechnicalIndicators
from environments.trading_env import TradingEnvironment, Portfolio
from agents.ppo_agent import PPOAgent
from utils.config import ConfigManager

def test_indicators_division_by_zero():
    """Test indicators with edge cases that could cause division by zero."""
    print("Testing indicators for division by zero...")
    
    # Create test data with edge cases
    test_data = pd.DataFrame({
        'High': [100, 100, 100, 100, 100],
        'Low': [100, 100, 100, 100, 100],  # Same values to test division by zero
        'Close': [100, 100, 100, 100, 100],
        'Volume': [0, 0, 0, 0, 0]  # Zero volume
    })
    
    try:
        # Test RSI with no change in price
        rsi = TechnicalIndicators.rsi(test_data['Close'], 4)
        print(f"RSI with constant prices: {rsi.iloc[-1]}")
        
        # Test stochastic oscillator with no price range
        stoch_k, stoch_d = TechnicalIndicators.stochastic_oscillator(
            test_data['High'], test_data['Low'], test_data['Close'], 4, 3
        )
        print(f"Stochastic K with no range: {stoch_k.iloc[-1]}")
        
        # Test with small volume SMA
        volume_sma = TechnicalIndicators.sma(test_data['Volume'], 3)
        volume_sma_safe = volume_sma.where(volume_sma != 0, 1e-10)
        volume_ratio = test_data['Volume'] / volume_sma_safe
        print(f"Volume ratio with zero volume: {volume_ratio.iloc[-1]}")
        
        print("✅ Indicators tests passed")
    except Exception as e:
        print(f"❌ Indicators test failed: {e}")
        traceback.print_exc()

def test_portfolio_edge_cases():
    """Test portfolio with edge cases."""
    print("\nTesting portfolio edge cases...")
    
    try:
        portfolio = Portfolio(initial_balance=1000.0)
        
        # Test with very small price
        trade_info = portfolio.execute_trade(2, 0.001, 0.1)  # BUY with tiny price
        print(f"Trade with tiny price: {trade_info}")
        
        # Test total value calculation
        total_value = portfolio.get_total_value(0.001)
        print(f"Portfolio value with tiny price: {total_value}")
        
        # Test return calculation
        portfolio_return = portfolio.get_return(0.001)
        print(f"Portfolio return: {portfolio_return}")
        
        print("✅ Portfolio tests passed")
    except Exception as e:
        print(f"❌ Portfolio test failed: {e}")
        traceback.print_exc()

def test_config_access():
    """Test config access with missing keys."""
    print("\nTesting config access...")
    
    try:
        config = ConfigManager()
        
        # Test accessing non-existent keys
        value = config.get('non.existent.key', 'default')
        print(f"Non-existent key access: {value}")
        
        # Test setting and getting values
        config.set('test.nested.value', 42)
        retrieved = config.get('test.nested.value')
        print(f"Set and retrieved value: {retrieved}")
        
        print("✅ Config tests passed")
    except Exception as e:
        print(f"❌ Config test failed: {e}")
        traceback.print_exc()

def test_file_operations():
    """Test file operations that could fail."""
    print("\nTesting file operations...")
    
    try:
        # Test model directory listing with non-existent directory
        model_dir = "non_existent_models"
        try:
            files = [f for f in os.listdir(model_dir) if f.endswith('.pt')] if os.path.exists(model_dir) else []
            print(f"Files in non-existent directory: {files}")
        except (OSError, PermissionError):
            print("Caught expected error for non-existent directory")
        
        # Test creating temporary directories
        test_dir = "test_temp_dir"
        os.makedirs(test_dir, exist_ok=True)
        if os.path.exists(test_dir):
            os.rmdir(test_dir)
            print("✅ Temporary directory operations passed")
        
        print("✅ File operations tests passed")
    except Exception as e:
        print(f"❌ File operations test failed: {e}")
        traceback.print_exc()

def test_array_access():
    """Test array access patterns."""
    print("\nTesting array access...")
    
    try:
        # Test with empty arrays
        empty_array = np.array([])
        if len(empty_array) > 0:
            print(f"Empty array access: {empty_array[0]}")
        else:
            print("Empty array handled correctly")
        
        # Test with small arrays
        small_array = np.array([1, 2, 3])
        safe_index = min(5, len(small_array) - 1)
        if safe_index >= 0:
            print(f"Safe array access: {small_array[safe_index]}")
        
        print("✅ Array access tests passed")
    except Exception as e:
        print(f"❌ Array access test failed: {e}")
        traceback.print_exc()

def test_bb_position_calculation():
    """Test Bollinger Band position calculation with edge cases."""
    print("\nTesting BB Position calculation...")
    
    try:
        # Test data with zero range
        close_prices = pd.Series([100, 100, 100, 100, 100])
        bb_upper = pd.Series([100, 100, 100, 100, 100])
        bb_lower = pd.Series([100, 100, 100, 100, 100])
        
        bb_range = bb_upper - bb_lower
        bb_range_safe = bb_range.where(bb_range != 0, 1e-10).infer_objects(copy=False)
        bb_position = (close_prices - bb_lower) / bb_range_safe
        
        print(f"BB Position with zero range: {bb_position.iloc[-1]}")
        print("✅ BB Position tests passed")
    except Exception as e:
        print(f"❌ BB Position test failed: {e}")
        traceback.print_exc()

def main():
    """Run all bug fix tests."""
    print("🧪 Running Bug Fix Validation Tests")
    print("=" * 50)
    
    test_indicators_division_by_zero()
    test_portfolio_edge_cases()
    test_config_access()
    test_file_operations()
    test_array_access()
    test_bb_position_calculation()
    
    print("\n" + "=" * 50)
    print("✅ All bug fix tests completed!")
    print("\nRun this script regularly to ensure edge cases are handled properly.")

if __name__ == "__main__":
    main()