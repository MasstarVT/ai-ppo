#!/usr/bin/env python3
"""
Quick test script to demonstrate debug logging in action.
"""

import sys
import os

# Add debug logging
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from debug_config import setup_debug_logging

print("🧪 Testing debug logging functionality...")
setup_debug_logging()

# Test the enhanced training script debug output
print("\n🎯 Testing Training Script Debug Output:")
print("=" * 50)

try:
    from train_enhanced import continue_training
    print("✅ Training module imported successfully")
    print("📝 Debug messages will show model loading, training progress, and save operations")
except ImportError as e:
    print(f"⚠️ Could not import training module: {e}")

# Test data client debug output
print("\n📊 Testing Data Client Debug Output:")
print("=" * 50)

try:
    sys.path.append('src')
    from data.data_client import YahooFinanceClient
    
    print("✅ Data client imported successfully")
    client = YahooFinanceClient()
    print("📝 Debug messages will show API calls, data processing, and error handling")
    print("🔍 Try fetching data from the GUI to see detailed debug output")
except ImportError as e:
    print(f"⚠️ Could not import data client: {e}")

print("\n🎉 Debug Testing Complete!")
print("=" * 50)
print("🔍 Debug Features Available:")
print("  • Real-time console messages with timestamps")
print("  • Separate log files for different components:")
print("    - logs/gui_debug.log (GUI events)")
print("    - logs/training_debug.log (Training events)")
print("    - logs/data_debug.log (Data fetching events)")
print("  • Color-coded print statements for easy identification")
print("  • Detailed error messages and stack traces")
print("\n💡 Use the GUI to see debug messages in action!")