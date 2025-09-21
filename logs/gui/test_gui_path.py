#!/usr/bin/env python3
"""
Test the path setup from gui directory context
"""

import os
import sys

print("Testing path setup from gui directory context...")
print(f"Current working directory: {os.getcwd()}")
print(f"Script location: {os.path.dirname(os.path.abspath(__file__))}")

# Simulate what run_gui.py does
current_dir = os.path.dirname(os.path.abspath(__file__))
src_path = os.path.join(current_dir, '..', 'src')

print(f"Calculated src_path: {src_path}")
print(f"Absolute src_path: {os.path.abspath(src_path)}")
print(f"src_path exists: {os.path.exists(src_path)}")

# Add it to sys.path like the GUI does
sys.path.insert(0, src_path)
print(f"Updated Python path: {sys.path}")

try:
    from utils import ConfigManager, create_default_config
    print("✅ utils import successful from gui context")
except ImportError as e:
    print(f"❌ utils import failed from gui context: {e}")
    import traceback
    traceback.print_exc()