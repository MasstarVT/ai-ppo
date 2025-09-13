#!/usr/bin/env python3
"""
Test script to check all imports used in the GUI app
"""

import sys
import os

# Add src to path like the GUI does
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

print("Testing imports...")

try:
    import yaml
    print("✅ yaml imported successfully")
except ImportError as e:
    print(f"❌ yaml import failed: {e}")

try:
    import streamlit as st
    print("✅ streamlit imported successfully")
except ImportError as e:
    print(f"❌ streamlit import failed: {e}")

try:
    import pandas as pd
    print("✅ pandas imported successfully")
except ImportError as e:
    print(f"❌ pandas import failed: {e}")

try:
    import numpy as np
    print("✅ numpy imported successfully")
except ImportError as e:
    print(f"❌ numpy import failed: {e}")

try:
    import plotly.graph_objects as go
    print("✅ plotly imported successfully")
except ImportError as e:
    print(f"❌ plotly import failed: {e}")

try:
    from streamlit_option_menu import option_menu
    print("✅ streamlit_option_menu imported successfully")
except ImportError as e:
    print(f"❌ streamlit_option_menu import failed: {e}")

try:
    from st_aggrid import AgGrid, GridOptionsBuilder
    print("✅ st_aggrid imported successfully")
except ImportError as e:
    print(f"❌ st_aggrid import failed: {e}")

# Test the specific imports from our modules
try:
    from environments import TradingEnvironment
    print("✅ TradingEnvironment imported successfully")
except ImportError as e:
    print(f"❌ TradingEnvironment import failed: {e}")

try:
    from agents import PPOAgent
    print("✅ PPOAgent imported successfully")
except ImportError as e:
    print(f"❌ PPOAgent import failed: {e}")

try:
    from evaluation.backtesting import Backtester
    print("✅ Backtester imported successfully")
except ImportError as e:
    print(f"❌ Backtester import failed: {e}")

try:
    from utils import ConfigManager, create_default_config, format_currency, format_percentage
    print("✅ utils imported successfully")
except ImportError as e:
    print(f"❌ utils import failed: {e}")

print("\nImport test completed!")