#!/usr/bin/env python3
"""
Test the specific import that's failing in the GUI context
"""

import os
import sys

# Simulate the same path addition as the GUI
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

print("Testing the problematic imports from the GUI context...")
print(f"Python path: {sys.path}")
print(f"Current working directory: {os.getcwd()}")

try:
    from utils import ConfigManager, create_default_config, format_currency, format_percentage
    print("✅ utils import successful")
    
    # Test the create_default_config function
    config = create_default_config()
    print("✅ create_default_config() works")
    print(f"Config keys: {list(config.keys())}")
    
except ImportError as e:
    print(f"❌ utils import failed: {e}")
    import traceback
    print("Full traceback:")
    traceback.print_exc()

try:
    # Test the ConfigManager
    config_manager = ConfigManager()
    print("✅ ConfigManager instantiation successful")
except Exception as e:
    print(f"❌ ConfigManager failed: {e}")
    import traceback
    print("Full traceback:")
    traceback.print_exc()