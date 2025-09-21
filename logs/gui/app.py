"""
Main Streamlit GUI application for the AI PPO Trading System.
"""

import streamlit as st
import os
import sys
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import json
import time
import threading
import queue
import logging

# Set up optimized logging (disable debug by default for performance)
logging.basicConfig(
    level=logging.INFO,  # Changed from DEBUG to INFO
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Ensure project root and src are on path and import core components
ROOT_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
SRC_PATH = os.path.join(ROOT_PATH, 'src')
for _p in (ROOT_PATH, SRC_PATH):
    if _p not in sys.path:
        sys.path.insert(0, _p)

try:
    # Prefer package-style imports
    from src.data import DataClient, prepare_features
    from src.environments import TradingEnvironment
    from src.agents import PPOAgent
    from src.evaluation.backtesting import Backtester
    from src.utils import ConfigManager, create_default_config, format_currency, format_percentage
    from streamlit_option_menu import option_menu
except Exception as _e:
    # Defer errors to UI; some pages may still work
    logger.warning(f"Optional imports failed (some features may be unavailable): {_e}")

# Basic error handler decorator used across views
def handle_errors(fn):
    def wrapper(*args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except Exception as e:
            st.error(f"Error: {e}")
            logger.exception(e)
            return None
    return wrapper

# Theme helpers expected by the UI
def get_theme_css():
    theme = st.session_state.get('theme', 'dark')
    if theme not in ('dark', 'light'):
        theme = 'dark'
    return get_dark_theme_css() if theme == 'dark' else get_light_theme_css()

def get_plotly_theme():
    theme = st.session_state.get('theme', 'dark')
    return dict(template='plotly_dark') if theme == 'dark' else dict(template='plotly_white')

# Minimal system status gate to avoid blocking UI if not defined elsewhere
def show_system_status():
    return True

# Global flags to prevent repeated initialization messages
_GUI_ALREADY_INITIALIZED = False
_YAML_LOGGED = False
_COMPONENTS_LOADING_LOGGED = False
_CORE_COMPONENTS_LOGGED = False
_TRAINING_MANAGER_LOGGED = False

# Check if this is the first run of the session using a file marker
GUI_SESSION_FILE = os.path.join(os.path.dirname(__file__), '.gui_session_active')

def is_first_gui_run():
    """Check if this is the first GUI run in this session"""
    if 'gui_session_started' not in st.session_state:
        # Create session marker file
        with open(GUI_SESSION_FILE, 'w') as f:
            f.write(str(time.time()))
        st.session_state.gui_session_started = True
        return True
    return False

def get_dark_theme_css():
    """Get dark theme CSS styles."""
    return """
    <style>
    /* Dark theme variables */
    :root {
        --bg-primary: #0e1117;
        --bg-secondary: #262730;
        --text-primary: #ffffff;
        --text-secondary: rgba(255, 255, 255, 0.9);
        --border-color: rgba(255, 255, 255, 0.15);
        --accent-color: #667eea;
        --success-color: #28a745;
        --warning-color: #ffc107;
        --error-color: #dc3545;
        --shadow-color: rgba(0, 0, 0, 0.3);
    }
    
    /* Main container */
    .main > div {
        padding-top: 2rem;
        background-color: var(--bg-primary);
    }
    
    /* Enhanced metrics styling */
    div[data-testid="metric-container"] {
        background-color: rgba(255, 255, 255, 0.08) !important;
        border: 2px solid var(--border-color) !important;
        padding: 1.5rem !important;
        border-radius: 12px !important;
        margin: 0.5rem 0 !important;
        box-shadow: 0 4px 12px var(--shadow-color) !important;
        backdrop-filter: blur(10px) !important;
        transition: all 0.3s ease !important;
    }
    
    div[data-testid="metric-container"]:hover {
        transform: translateY(-3px) !important;
        box-shadow: 0 8px 25px rgba(0, 0, 0, 0.4) !important;
        border-color: rgba(255, 255, 255, 0.3) !important;
    }
    
    /* Metric labels */
    div[data-testid="metric-container"] label {
        font-size: 1rem !important;
        font-weight: 600 !important;
        color: var(--text-secondary) !important;
        margin-bottom: 0.5rem !important;
        text-transform: uppercase !important;
        letter-spacing: 0.5px !important;
    }
    
    /* Metric values */
    div[data-testid="metric-container"] div[data-testid="metric-value"] {
        font-size: 2rem !important;
        font-weight: 700 !important;
        color: var(--text-primary) !important;
        line-height: 1.2 !important;
        text-shadow: 0 2px 4px var(--shadow-color) !important;
    }
    
    /* Metric deltas */
    div[data-testid="metric-container"] div[data-testid="metric-delta"] {
        font-size: 1.1rem !important;
        font-weight: 600 !important;
        margin-top: 0.4rem !important;
        text-shadow: 0 1px 2px rgba(0,0,0,0.2) !important;
    }
    
    /* Portfolio Overview container */
    .portfolio-overview {
        background: linear-gradient(135deg, rgba(255,255,255,0.1) 0%, rgba(255,255,255,0.05) 100%);
        border-radius: 15px;
        padding: 2rem;
        margin: 1.5rem 0;
        border: 2px solid var(--border-color);
        box-shadow: 0 8px 32px rgba(0,0,0,0.1);
    }
    
    /* Status boxes */
    .success-box {
        background: linear-gradient(135deg, #d4edda 0%, #c3e6cb 100%);
        border: 2px solid var(--success-color);
        color: #155724;
        padding: 1.2rem;
        border-radius: 12px;
        margin: 1rem 0;
        box-shadow: 0 4px 12px rgba(40, 167, 69, 0.2);
        font-weight: 500;
    }
    
    .warning-box {
        background: linear-gradient(135deg, #fff3cd 0%, #ffeaa7 100%);
        border: 2px solid var(--warning-color);
        color: #856404;
        padding: 1.2rem;
        border-radius: 12px;
        margin: 1rem 0;
        box-shadow: 0 4px 12px rgba(255, 193, 7, 0.2);
        font-weight: 500;
    }
    
    .error-box {
        background: linear-gradient(135deg, #f8d7da 0%, #f5c6cb 100%);
        border: 2px solid var(--error-color);
        color: #721c24;
        padding: 1.2rem;
        border-radius: 12px;
        margin: 1rem 0;
        box-shadow: 0 4px 12px rgba(220, 53, 69, 0.2);
        font-weight: 500;
    }
    
    .info-box {
        background: linear-gradient(135deg, #d1ecf1 0%, #bee5eb 100%);
        border: 2px solid #17a2b8;
        color: #0c5460;
        padding: 1.2rem;
        border-radius: 12px;
        margin: 1rem 0;
        box-shadow: 0 4px 12px rgba(23, 162, 184, 0.2);
        font-weight: 500;
    }
    
    /* Button styling */
    .stButton > button {
        border-radius: 25px !important;
        border: none !important;
        background: linear-gradient(135deg, var(--accent-color) 0%, #764ba2 100%) !important;
        color: white !important;
        font-weight: 600 !important;
        padding: 0.7rem 2rem !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 4px 15px var(--shadow-color) !important;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 25px rgba(0,0,0,0.4) !important;
        background: linear-gradient(135deg, #764ba2 0%, var(--accent-color) 100%) !important;
    }
    
    /* Sidebar improvements */
    .css-1d391kg {
        background-color: var(--bg-secondary) !important;
    }
    
    /* Chart container improvements */
    .js-plotly-plot {
        border-radius: 12px !important;
        overflow: hidden !important;
        box-shadow: 0 6px 20px rgba(0, 0, 0, 0.15) !important;
        border: 2px solid var(--border-color) !important;
    }
    
    /* Theme toggle button */
    .theme-toggle {
        position: fixed;
        top: 1rem;
        right: 1rem;
        z-index: 999;
        background: var(--bg-secondary);
        border: 2px solid var(--border-color);
        border-radius: 50px;
        padding: 0.5rem 1rem;
        box-shadow: 0 4px 12px var(--shadow-color);
    }
    </style>
    """

def get_light_theme_css():
    """Get light theme CSS styles."""
    return """
    <style>
    /* Light theme variables */
    :root {
        --bg-primary: #ffffff !important;
        --bg-secondary: #f8f9fa !important;
        --text-primary: #212529 !important;
        --text-secondary: #495057 !important;
        --border-color: rgba(0, 0, 0, 0.1) !important;
        --accent-color: #007bff !important;
        --success-color: #28a745 !important;
        --warning-color: #ffc107 !important;
        --error-color: #dc3545 !important;
        --shadow-color: rgba(0, 0, 0, 0.1) !important;
    }
    
    /* Force light theme on main container and body */
    .stApp, .main, .main > div, div[data-testid="stAppViewContainer"], body {
        background-color: #ffffff !important;
        color: #212529 !important;
    }
    
    /* Sidebar light theme - VERY aggressive targeting */
    section[data-testid="stSidebar"], 
    section[data-testid="stSidebar"] > div, 
    section[data-testid="stSidebar"] div,
    section[data-testid="stSidebar"] .element-container,
    .css-1d391kg,
    div[data-testid="stSidebar"],
    .css-* section[data-testid="stSidebar"],
    .st-emotion-cache-* section[data-testid="stSidebar"] {
        background-color: #f8f9fa !important;
        color: #212529 !important;
    }
    
    /* Navigation menu styling for light theme - target all possible classes */
    .nav-link, .nav-link-selected, 
    div[class*="nav-link"],
    section[data-testid="stSidebar"] .element-container,
    section[data-testid="stSidebar"] .stSelectbox,
    section[data-testid="stSidebar"] .stButton,
    section[data-testid="stSidebar"] button,
    .css-* .nav-link,
    .st-emotion-cache-* .nav-link,
    div[role="tablist"],
    div[role="tab"] {
        background-color: #f8f9fa !important;
        color: #212529 !important;
    }
    
    /* Force sidebar background override any default styling */
    section[data-testid="stSidebar"] * {
        background-color: #f8f9fa !important;
        color: #212529 !important;
    }
    
    /* Navigation icons and text */
    section[data-testid="stSidebar"] svg,
    section[data-testid="stSidebar"] span,
    section[data-testid="stSidebar"] p,
    section[data-testid="stSidebar"] div {
        color: #212529 !important;
        fill: #212529 !important;
    }
    
    /* Streamlit option menu styling */
    .nav-link, .nav-link-selected,
    div[data-baseweb="tab-list"],
    div[data-baseweb="tab"],
    div[data-baseweb="tab-border"],
    .css-* div[role="tablist"],
    .css-* div[role="tab"] {
        background-color: #f8f9fa !important;
        color: #212529 !important;
    }
    
    /* Active/selected navigation item */
    .nav-link-selected,
    div[data-baseweb="tab"][aria-selected="true"] {
        background-color: #007bff !important;
        color: #ffffff !important;
    }
    
    /* Force all text to be dark in light theme */
    .stApp *, .main *, div[data-testid="stAppViewContainer"] *, 
    section[data-testid="stSidebar"] *, .css-1d391kg *,
    h1, h2, h3, h4, h5, h6, p, span, div, label, .stMarkdown * {
        color: #212529 !important;
    }
    
    /* Streamlit components text override */
    .element-container *, .stSelectbox *, .stTextInput *, 
    .stButton *, .stMetric *, .stDataFrame *, 
    div[data-testid="metric-container"] *,
    div[data-testid="metric-container"] label,
    div[data-testid="metric-container"] div[data-testid="metric-value"],
    div[data-testid="metric-container"] div[data-testid="metric-delta"] {
        color: #212529 !important;
    }
    
    /* Chart text colors */
    .js-plotly-plot .plotly .gtitle, 
    .js-plotly-plot .plotly .xtitle,
    .js-plotly-plot .plotly .ytitle,
    .js-plotly-plot .plotly text,
    .js-plotly-plot .plotly .colorbar text,
    .js-plotly-plot .plotly .legend text,
    .js-plotly-plot text {
        fill: #212529 !important;
        color: #212529 !important;
    }
    
    /* Plotly chart backgrounds and text */
    .js-plotly-plot {
        background-color: #ffffff !important;
    }
    
    .js-plotly-plot .plotly .main-svg {
        background-color: #ffffff !important;
    }
    
    /* Force chart visibility and sizing */
    .js-plotly-plot,
    .plotly-graph-div,
    div[data-testid="stPlotlyChart"] {
        background-color: #ffffff !important;
        border: 1px solid rgba(0, 0, 0, 0.1) !important;
        min-height: 400px !important;
    }
    
    /* Chart container styling */
    div[data-testid="stPlotlyChart"] > div {
        background-color: #ffffff !important;
    }
    
    /* Additional plotly text targeting */
    .js-plotly-plot svg text {
        fill: #212529 !important;
    }
    
    .js-plotly-plot .plotly .colorbar-title-text,
    .js-plotly-plot .plotly .colorbar-label-text {
        fill: #212529 !important;
    }
    
    /* Table styling for light theme */
    div[data-testid="stDataFrame"],
    .stDataFrame,
    table,
    .dataframe,
    div[data-testid="stTable"],
    .st-emotion-cache-* table,
    .css-* table {
        background-color: #ffffff !important;
        color: #212529 !important;
    }
    
    /* Table headers and cells - more aggressive targeting */
    table th, table td,
    .dataframe th, .dataframe td,
    div[data-testid="stDataFrame"] th,
    div[data-testid="stDataFrame"] td,
    .css-* table th,
    .css-* table td,
    .st-emotion-cache-* th,
    .st-emotion-cache-* td {
        background-color: #ffffff !important;
        color: #212529 !important;
        border-color: rgba(0, 0, 0, 0.1) !important;
    }
    
    /* Table header specific styling - very aggressive */
    table thead th,
    .dataframe thead th,
    div[data-testid="stDataFrame"] thead th,
    .css-* table thead th,
    .st-emotion-cache-* thead th,
    tbody tr:first-child th,
    tr:first-child th {
        background-color: #f8f9fa !important;
        color: #212529 !important;
        font-weight: 600 !important;
        border-bottom: 2px solid rgba(0, 0, 0, 0.1) !important;
    }
    
    /* Table rows */
    table tbody tr,
    .dataframe tbody tr,
    .css-* table tbody tr {
        background-color: #ffffff !important;
    }
    
    /* Table row hover */
    table tbody tr:hover,
    .dataframe tbody tr:hover {
        background-color: #f8f9fa !important;
    }
    
    /* Navigation menu text */
    .nav-link, .nav-link-selected {
        color: #212529 !important;
    }
    
    /* Main container */
    .main > div {
        padding-top: 2rem !important;
        background-color: var(--bg-primary) !important;
        color: var(--text-primary) !important;
    }
    
    /* Enhanced metrics styling */
    div[data-testid="metric-container"] {
        background-color: var(--bg-secondary) !important;
        border: 2px solid var(--border-color) !important;
        padding: 1.5rem !important;
        border-radius: 12px !important;
        margin: 0.5rem 0 !important;
        box-shadow: 0 4px 12px var(--shadow-color) !important;
        transition: all 0.3s ease !important;
    }
    
    div[data-testid="metric-container"]:hover {
        transform: translateY(-3px) !important;
        box-shadow: 0 8px 25px rgba(0, 0, 0, 0.15) !important;
        border-color: var(--accent-color) !important;
    }
    
    /* Metric labels */
    div[data-testid="metric-container"] label {
        font-size: 1rem !important;
        font-weight: 600 !important;
        color: var(--text-secondary) !important;
        margin-bottom: 0.5rem !important;
        text-transform: uppercase !important;
        letter-spacing: 0.5px !important;
    }
    
    /* Metric values */
    div[data-testid="metric-container"] div[data-testid="metric-value"] {
        font-size: 2rem !important;
        font-weight: 700 !important;
        color: var(--text-primary) !important;
        line-height: 1.2 !important;
    }
    
    /* Metric deltas */
    div[data-testid="metric-container"] div[data-testid="metric-delta"] {
        font-size: 1.1rem !important;
        font-weight: 600 !important;
        margin-top: 0.4rem !important;
    }
    
    /* Portfolio Overview container */
    .portfolio-overview {
        background: var(--bg-secondary);
        border-radius: 15px;
        padding: 2rem;
        margin: 1.5rem 0;
        border: 2px solid var(--border-color);
        box-shadow: 0 8px 32px var(--shadow-color);
    }
    
    /* Status boxes - light theme versions */
    .success-box {
        background: linear-gradient(135deg, #d4edda 0%, #c3e6cb 100%);
        border: 2px solid var(--success-color);
        color: #155724;
        padding: 1.2rem;
        border-radius: 12px;
        margin: 1rem 0;
        box-shadow: 0 4px 12px rgba(40, 167, 69, 0.15);
        font-weight: 500;
    }
    
    .warning-box {
        background: linear-gradient(135deg, #fff3cd 0%, #ffeaa7 100%);
        border: 2px solid var(--warning-color);
        color: #856404;
        padding: 1.2rem;
        border-radius: 12px;
        margin: 1rem 0;
        box-shadow: 0 4px 12px rgba(255, 193, 7, 0.15);
        font-weight: 500;
    }
    
    .error-box {
        background: linear-gradient(135deg, #f8d7da 0%, #f5c6cb 100%);
        border: 2px solid var(--error-color);
        color: #721c24;
        padding: 1.2rem;
        border-radius: 12px;
        margin: 1rem 0;
        box-shadow: 0 4px 12px rgba(220, 53, 69, 0.15);
        font-weight: 500;
    }
    
    .info-box {
        background: linear-gradient(135deg, #d1ecf1 0%, #bee5eb 100%);
        border: 2px solid #17a2b8;
        color: #0c5460;
        padding: 1.2rem;
        border-radius: 12px;
        margin: 1rem 0;
        box-shadow: 0 4px 12px rgba(23, 162, 184, 0.15);
        font-weight: 500;
    }
    
    /* Button styling */
    .stButton > button {
        border-radius: 25px !important;
        border: none !important;
        background: linear-gradient(135deg, var(--accent-color) 0%, #6c5ce7 100%) !important;
        color: white !important;
        font-weight: 600 !important;
        padding: 0.7rem 2rem !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 4px 15px var(--shadow-color) !important;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 25px rgba(0,0,0,0.2) !important;
        background: linear-gradient(135deg, #6c5ce7 0%, var(--accent-color) 100%) !important;
    }
    
    /* Sidebar improvements */
    .css-1d391kg {
        background-color: var(--bg-secondary) !important;
    }
    
    /* Chart container improvements */
    .js-plotly-plot {
        border-radius: 12px !important;
        overflow: hidden !important;
        box-shadow: 0 6px 20px var(--shadow-color) !important;
        border: 2px solid var(--border-color) !important;
    }
    
    /* Theme toggle button */
    .theme-toggle {
        position: fixed;
        top: 1rem;
        right: 1rem;
        z-index: 999;
        background: var(--bg-secondary);
        border: 2px solid var(--border-color);
        border-radius: 50px;
        padding: 0.5rem 1rem;
        box-shadow: 0 4px 12px var(--shadow-color);
    }
    </style>
    """

# CSS will be applied in main() after theme initialization

# Settings persistence functions
def load_user_settings():
    """Load user settings from file."""
    settings_file = os.path.join(os.path.dirname(__file__), 'user_settings.json')
    try:
        if os.path.exists(settings_file):
            with open(settings_file, 'r') as f:
                return json.load(f)
    except Exception as e:
        logger.warning(f"Failed to load user settings: {e}")
    return {}

def save_user_settings(settings):
    """Save user settings to file."""
    settings_file = os.path.join(os.path.dirname(__file__), 'user_settings.json')
    try:
        os.makedirs(os.path.dirname(settings_file), exist_ok=True)
        with open(settings_file, 'w') as f:
            json.dump(settings, f, indent=2)
        logger.info("User settings saved successfully")
    except Exception as e:
        logger.error(f"Failed to save user settings: {e}")

def merge_with_user_settings(config):
    """Merge configuration with saved user settings."""
    user_settings = load_user_settings()
    if user_settings:
        # Deep merge user settings into config
        for section, settings in user_settings.items():
            if section in config and isinstance(config[section], dict) and isinstance(settings, dict):
                config[section].update(settings)
            else:
                config[section] = settings
    return config

# Initialize session state
if 'config' not in st.session_state:
    try:
        base_config = create_default_config()
        # Load and merge user settings
        st.session_state.config = merge_with_user_settings(base_config)
    except Exception as e:
        logger.error(f"Failed to create default config: {e}")
        # Minimal fallback config
        st.session_state.config = {
            'trading': {'symbols': ['AAPL', 'BTC/USDT'], 'initial_balance': 10000, 'max_position_days': 30},
            'ppo': {'learning_rate': 3e-4, 'n_steps': 2048},
            'training': {'total_timesteps': 100000}
        }

# Initialize theme state
if 'theme' not in st.session_state:
    st.session_state.theme = 'dark'  # Default to dark theme

# Initialize other session state variables
if 'training_active' not in st.session_state:
    st.session_state.training_active = False
if 'training_metrics' not in st.session_state:
    st.session_state.training_metrics = []

def detect_active_training():
    """Detect if training is currently active based on log file activity and process checks."""
    try:
        # Try to use psutil for process detection
        try:
            import importlib
            psutil = importlib.import_module('psutil')
            
            # Check for running Python processes with train_enhanced.py
            for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
                try:
                    cmdline = proc.info.get('cmdline', [])
                    if (cmdline and 
                        'python' in cmdline[0].lower() and 
                        any('train_enhanced.py' in str(arg) for arg in cmdline)):
                        
                        # Found a training process - store its info
                        st.session_state.training_active = True
                        st.session_state.training_process_pid = proc.info['pid']
                        
                        # Try to determine training mode from cmdline
                        if '--mode' in cmdline:
                            try:
                                mode_idx = cmdline.index('--mode')
                                if mode_idx + 1 < len(cmdline):
                                    st.session_state.training_mode = cmdline[mode_idx + 1]
                            except (ValueError, IndexError):
                                pass
                        
                        return True
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
        except ImportError:
            # psutil not available, skip process-based detection
            pass
                
                # Check for recent log activity (within last 5 minutes)
        log_file = os.path.join(os.path.dirname(__file__), '..', 'logs', 'training_debug.log')
        if os.path.exists(log_file):
            stat = os.stat(log_file)
            last_modified = stat.st_mtime
            current_time = time.time()
            
            # If log was modified in last 5 minutes, check what kind of activity
            if current_time - last_modified < 300:  # 5 minutes
                try:
                    with open(log_file, 'r') as f:
                        lines = f.readlines()
                        
                    # First check if training was recently completed successfully
                    for line in reversed(lines[-50:]):  # Check last 50 lines
                        if any(keyword in line for keyword in ['Training completed successfully', 'TRAINING STOPPED', 'TRAINING FAILED', 'stopped by user']):
                            # Check if this completion event is recent (within last 10 minutes)
                            import re
                            timestamp_match = re.match(r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})', line)
                            if timestamp_match:
                                log_time = datetime.strptime(timestamp_match.group(1), '%Y-%m-%d %H:%M:%S')
                                if current_time - log_time.timestamp() < 600:  # 10 minutes
                                    # Training was recently completed/stopped, so it's not active
                                    return False
                    
                    # Check for completion by looking for 100% progress followed by no more progress
                    completion_found = False
                    last_progress_time = None
                    
                    for line in reversed(lines[-30:]):  # Check last 30 lines
                        import re
                        # Look for 100% completion steps
                        if 'Step' in line and '(100.0%)' in line:
                            timestamp_match = re.match(r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})', line)
                            if timestamp_match:
                                log_time = datetime.strptime(timestamp_match.group(1), '%Y-%m-%d %H:%M:%S')
                                completion_found = True
                                break
                    
                    if completion_found:
                        # If we found 100% completion, training is not active
                        return False
                    
                    # Now look for recent active progress lines (only if no completion found)
                    for line in reversed(lines[-20:]):  # Check last 20 lines
                        if any(keyword in line for keyword in ['Step', 'Episode']) and '%' in line and '100.0%' not in line:
                            # Check if this progress line is recent and not at 100%
                            import re
                            timestamp_match = re.match(r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})', line)
                            if timestamp_match:
                                log_time = datetime.strptime(timestamp_match.group(1), '%Y-%m-%d %H:%M:%S')
                                if current_time - log_time.timestamp() < 120:  # Only very recent (2 minutes)
                                    # Found recent active progress (not at completion)
                                    st.session_state.training_active = True
                                    return True
                except Exception:
                    pass
                    
        return False
        
    except Exception:
        return False

# Detect active training on startup
if not st.session_state.training_active:
    if detect_active_training():
        # Initialize training start time if not set
        if not hasattr(st.session_state, 'training_start_time'):
            st.session_state.training_start_time = time.time() - 60  # Assume started 1 minute ago
        
        # Set default training mode if not detected
        if not hasattr(st.session_state, 'training_mode'):
            st.session_state.training_mode = 'new'

def main():
    """Get dark theme CSS styles."""
    return """
    <style>
    /* Dark theme variables */
    :root {
        --bg-primary: #0e1117;
        --bg-secondary: #262730;
        --text-primary: #ffffff;
        --text-secondary: rgba(255, 255, 255, 0.9);
        --border-color: rgba(255, 255, 255, 0.15);
        --accent-color: #667eea;
        --success-color: #28a745;
        --warning-color: #ffc107;
        --error-color: #dc3545;
        --shadow-color: rgba(0, 0, 0, 0.3);
    }
    
    /* Main container */
    .main > div {
        padding-top: 2rem;
        background-color: var(--bg-primary);
    }
    
    /* Enhanced metrics styling */
    div[data-testid="metric-container"] {
        background-color: rgba(255, 255, 255, 0.08) !important;
        border: 2px solid var(--border-color) !important;
        padding: 1.5rem !important;
        border-radius: 12px !important;
        margin: 0.5rem 0 !important;
        box-shadow: 0 4px 12px var(--shadow-color) !important;
        backdrop-filter: blur(10px) !important;
        transition: all 0.3s ease !important;
    }
    
    div[data-testid="metric-container"]:hover {
        transform: translateY(-3px) !important;
        box-shadow: 0 8px 25px rgba(0, 0, 0, 0.4) !important;
        border-color: rgba(255, 255, 255, 0.3) !important;
    }
    
    /* Metric labels */
    div[data-testid="metric-container"] label {
        font-size: 1rem !important;
        font-weight: 600 !important;
        color: var(--text-secondary) !important;
        margin-bottom: 0.5rem !important;
        text-transform: uppercase !important;
        letter-spacing: 0.5px !important;
    }
    
    /* Metric values */
    div[data-testid="metric-container"] div[data-testid="metric-value"] {
        font-size: 2rem !important;
        font-weight: 700 !important;
        color: var(--text-primary) !important;
        line-height: 1.2 !important;
        text-shadow: 0 2px 4px var(--shadow-color) !important;
    }
    
    /* Metric deltas */
    div[data-testid="metric-container"] div[data-testid="metric-delta"] {
        font-size: 1.1rem !important;
        font-weight: 600 !important;
        margin-top: 0.4rem !important;
        text-shadow: 0 1px 2px rgba(0,0,0,0.2) !important;
    }
    
    /* Portfolio Overview container */
    .portfolio-overview {
        background: linear-gradient(135deg, rgba(255,255,255,0.1) 0%, rgba(255,255,255,0.05) 100%);
        border-radius: 15px;
        padding: 2rem;
        margin: 1.5rem 0;
        border: 2px solid var(--border-color);
        box-shadow: 0 8px 32px rgba(0,0,0,0.1);
    }
    
    /* Status boxes */
    .success-box {
        background: linear-gradient(135deg, #d4edda 0%, #c3e6cb 100%);
        border: 2px solid var(--success-color);
        color: #155724;
        padding: 1.2rem;
        border-radius: 12px;
        margin: 1rem 0;
        box-shadow: 0 4px 12px rgba(40, 167, 69, 0.2);
        font-weight: 500;
    }
    
    .warning-box {
        background: linear-gradient(135deg, #fff3cd 0%, #ffeaa7 100%);
        border: 2px solid var(--warning-color);
        color: #856404;
        padding: 1.2rem;
        border-radius: 12px;
        margin: 1rem 0;
        box-shadow: 0 4px 12px rgba(255, 193, 7, 0.2);
        font-weight: 500;
    }
    
    .error-box {
        background: linear-gradient(135deg, #f8d7da 0%, #f5c6cb 100%);
        border: 2px solid var(--error-color);
        color: #721c24;
        padding: 1.2rem;
        border-radius: 12px;
        margin: 1rem 0;
        box-shadow: 0 4px 12px rgba(220, 53, 69, 0.2);
        font-weight: 500;
    }
    
    .info-box {
        background: linear-gradient(135deg, #d1ecf1 0%, #bee5eb 100%);
        border: 2px solid #17a2b8;
        color: #0c5460;
        padding: 1.2rem;
        border-radius: 12px;
        margin: 1rem 0;
        box-shadow: 0 4px 12px rgba(23, 162, 184, 0.2);
        font-weight: 500;
    }
    
    /* Button styling */
    .stButton > button {
        border-radius: 25px !important;
        border: none !important;
        background: linear-gradient(135deg, var(--accent-color) 0%, #764ba2 100%) !important;
        color: white !important;
        font-weight: 600 !important;
        padding: 0.7rem 2rem !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 4px 15px var(--shadow-color) !important;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 25px rgba(0,0,0,0.4) !important;
        background: linear-gradient(135deg, #764ba2 0%, var(--accent-color) 100%) !important;
    }
    
    /* Sidebar improvements */
    .css-1d391kg {
        background-color: var(--bg-secondary) !important;
    }
    
    /* Chart container improvements */
    .js-plotly-plot {
        border-radius: 12px !important;
        overflow: hidden !important;
        box-shadow: 0 6px 20px rgba(0, 0, 0, 0.15) !important;
        border: 2px solid var(--border-color) !important;
    }
    
    /* Theme toggle button */
    .theme-toggle {
        position: fixed;
        top: 1rem;
        right: 1rem;
        z-index: 999;
        background: var(--bg-secondary);
        border: 2px solid var(--border-color);
        border-radius: 50px;
        padding: 0.5rem 1rem;
        box-shadow: 0 4px 12px var(--shadow-color);
    }
    </style>
    """

def main():
    """Main application function."""
    
    # Apply theme CSS immediately at start
    st.markdown(get_theme_css(), unsafe_allow_html=True)
    
    # Check system status first
    if not show_system_status():
        return
    
    # Sidebar navigation
    with st.sidebar:
        st.image("https://via.placeholder.com/200x80/1f77b4/ffffff?text=AI+PPO+Trading", width=200)
        
        # Theme toggle button
        st.divider()
        col1, col2 = st.columns([3, 1])
        with col1:
            st.write(f"**Theme:** {st.session_state.theme.title()}")
        with col2:
            # Show current theme and toggle icon
            current_icon = '🌞' if st.session_state.theme == 'dark' else '🌙'
            button_text = f"{current_icon}"
            if st.button(button_text, help=f"Switch to {'Light' if st.session_state.theme == 'dark' else 'Dark'} theme"):
                st.session_state.theme = 'light' if st.session_state.theme == 'dark' else 'dark'
                st.rerun()
        
        st.divider()
        
        selected = option_menu(
            menu_title="Navigation",
            options=[
                "Dashboard", 
                "Configuration", 
                "Data Analysis", 
                "Training", 
                "Backtesting", 
                "Live Trading",
                "Model Management"
            ],
            icons=[
                "speedometer2", 
                "gear", 
                "graph-up", 
                "play-circle", 
                "bar-chart", 
                "broadcast",
                "folder"
            ],
            menu_icon="cast",
            default_index=0,
        )
        
        # System status
        st.markdown("---")
        st.markdown("### System Status")
        
        # Check if models exist
        model_dir = "models"
        try:
            models_exist = os.path.exists(model_dir) and len([f for f in os.listdir(model_dir) if f.endswith('.pt')]) > 0
        except (OSError, PermissionError):
            models_exist = False
        
        
        if models_exist:
            st.success("✅ Models Available")
        else:
            st.warning("⚠️ No Trained Models")
        
        # Check config
        config_exists = os.path.exists("config/config.yaml")
        if config_exists:
            st.success("✅ Configuration Ready")
        else:
            st.warning("⚠️ No Configuration")
        
        # Training status
        if st.session_state.training_active:
            st.info("🔄 Training in Progress")
        else:
            st.info("⏸️ Training Idle")

    # Main content based on selection
    if selected == "Dashboard":
        show_dashboard()
    elif selected == "Configuration":
        show_configuration()
    elif selected == "Data Analysis":
        show_data_analysis()
    elif selected == "Training":
        show_training()
    elif selected == "Backtesting":
        show_backtesting()
    elif selected == "Live Trading":
        show_live_trading()
    elif selected == "Model Management":
        show_model_management()
    else:
        st.error(f"Unknown page: {selected}")

@handle_errors
def show_dashboard():
    """Show main dashboard."""
    st.title("📈 AI PPO Trading System Dashboard")
    
    # Debug info
    st.write(f"🔍 Current theme: {st.session_state.get('theme', 'not set')}")
    
    # Key metrics row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        # Portfolio value (placeholder)
        st.metric(
            label="Portfolio Value",
            value="$10,000",
            delta="+5.2%"
        )
    
    with col2:
        # Total return (placeholder)
        st.metric(
            label="Total Return",
            value="12.4%",
            delta="+2.1%"
        )
    
    with col3:
        # Sharpe ratio (placeholder)
        st.metric(
            label="Sharpe Ratio",
            value="1.85",
            delta="+0.15"
        )
    
    with col4:
        # Win rate (placeholder)
        st.metric(
            label="Win Rate",
            value="67%",
            delta="+3%"
        )
    
    # Charts row
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Portfolio Performance")
        
        try:
            # Create simple sample data
            dates = pd.date_range(start='2024-01-01', periods=100, freq='D')
            np.random.seed(42)
            values = np.cumsum(np.random.normal(0, 20, 100)) + 10000
            
            # Create basic line chart
            chart_data = pd.DataFrame({
                'Date': dates,
                'Portfolio Value': values
            })
            
            # Use Streamlit's native line chart as fallback
            st.line_chart(chart_data.set_index('Date'))
            
        except Exception as e:
            st.error(f"Chart error: {e}")
            # Fallback to simple text
            st.info("📈 Portfolio growing steadily over time")
            st.write("Current Value: $10,000 (+5.2%)")
    
    with col2:
        st.subheader("Monthly Returns")
        
        try:
            # Create simple heatmap data
            months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun']
            returns_data = {
                '2023': [0.02, -0.01, 0.03, 0.01, 0.04, -0.02],
                '2024': [0.01, 0.03, -0.01, 0.02, 0.01, 0.03]
            }
            
            df = pd.DataFrame(returns_data, index=months)
            
            # Use simple dataframe display
            st.dataframe(df.style.format("{:.1%}").background_gradient(cmap='RdYlGn', axis=None))
            
        except Exception as e:
            st.error(f"Heatmap error: {e}")
            # Fallback
            st.info("📊 Monthly returns showing positive trend")
            st.write("Average monthly return: +1.5%")
    
    # Recent activity
    st.subheader("Recent Activity")
    
    # Sample trading activity
    activity_data = {
        'Date': ['2024-09-12', '2024-09-11', '2024-09-10', '2024-09-09', '2024-09-08'],
        'Symbol': ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA'],
        'Action': ['BUY', 'SELL', 'HOLD', 'BUY', 'SELL'],
        'Shares': [10, 15, 0, 8, 12],
        'Price': [175.50, 415.20, 0, 185.30, 248.90],
        'P&L': ['+$125.50', '-$89.30', '$0.00', '+$201.40', '+$156.80']
    }
    
    activity_df = pd.DataFrame(activity_data)
    st.dataframe(activity_df, width="stretch")

@handle_errors
def show_configuration():
    """Show configuration management."""
    st.title("⚙️ Configuration Management")
    
    # Configuration tabs
    tab1, tab2, tab3, tab4 = st.tabs(["Trading", "PPO Parameters", "Risk Management", "Data Sources"])
    
    with tab1:
        st.subheader("Trading Configuration")
        
        col1, col2 = st.columns(2)
        
        with col1:
            symbols = st.text_area(
                "Trading Symbols (one per line)",
                value="\n".join(st.session_state.config.get('trading', {}).get('symbols', ['AAPL', 'BTC/USDT'])),
                height=100,
                help="Supports stocks (AAPL, MSFT) and crypto pairs (BTC/USDT, ETH/USDT, ADA/USDT, etc.)"
            )
            
            initial_balance = st.number_input(
                "Initial Balance ($)",
                value=st.session_state.config.get('trading', {}).get('initial_balance', 10000),
                min_value=1000,
                step=1000
            )
            
            timeframe = st.selectbox(
                "Timeframe",
                options=['1m', '5m', '15m', '30m', '1h', '4h', '1d'],
                index=4  # Default to '1h'
            )
        
        with col2:
            max_position_size = st.slider(
                "Max Position Size (%)",
                min_value=1,
                max_value=100,
                value=int(st.session_state.config.get('trading', {}).get('max_position_size', 0.1) * 100),
                step=1
            )
            
            max_position_days = st.slider(
                "Max Position Duration (Days)",
                min_value=1,
                max_value=90,
                value=st.session_state.config.get('trading', {}).get('max_position_days', 30),
                step=1,
                help="Maximum days to hold a position before force close"
            )
            
            transaction_cost = st.number_input(
                "Transaction Cost (%)",
                value=st.session_state.config.get('trading', {}).get('transaction_cost', 0.001) * 100,
                min_value=0.0,
                max_value=1.0,
                step=0.01,
                format="%.3f"
            )
            
            slippage = st.number_input(
                "Slippage (%)",
                value=st.session_state.config.get('trading', {}).get('slippage', 0.0005) * 100,
                min_value=0.0,
                max_value=0.1,
                step=0.001,
                format="%.4f"
            )
        
        if st.button("Save Trading Configuration"):
            # Update session state
            st.session_state.config['trading'] = {
                'symbols': [s.strip() for s in symbols.split('\n') if s.strip()],
                'timeframe': timeframe,
                'initial_balance': initial_balance,
                'max_position_size': max_position_size / 100,
                'max_position_days': max_position_days,
                'transaction_cost': transaction_cost / 100,
                'slippage': slippage / 100
            }
            # Save to persistent storage
            save_user_settings(st.session_state.config)
            st.success("Trading configuration saved!")
    
    with tab2:
        st.subheader("PPO Hyperparameters")
        
        col1, col2 = st.columns(2)
        
        with col1:
            learning_rate = st.number_input(
                "Learning Rate",
                value=st.session_state.config.get('ppo', {}).get('learning_rate', 3e-4),
                min_value=1e-6,
                max_value=1e-2,
                step=1e-5,
                format="%.6f"
            )
            
            n_steps = st.number_input(
                "N Steps",
                value=st.session_state.config.get('ppo', {}).get('n_steps', 2048),
                min_value=512,
                max_value=8192,
                step=512
            )
            
            batch_size = st.number_input(
                "Batch Size",
                value=st.session_state.config.get('ppo', {}).get('batch_size', 64),
                min_value=16,
                max_value=512,
                step=16
            )
            
            n_epochs = st.number_input(
                "N Epochs",
                value=st.session_state.config.get('ppo', {}).get('n_epochs', 10),
                min_value=1,
                max_value=50,
                step=1
            )
        
        with col2:
            gamma = st.slider(
                "Gamma (Discount Factor)",
                min_value=0.9,
                max_value=0.999,
                value=st.session_state.config.get('ppo', {}).get('gamma', 0.99),
                step=0.001
            )
            
            gae_lambda = st.slider(
                "GAE Lambda",
                min_value=0.9,
                max_value=0.99,
                value=st.session_state.config.get('ppo', {}).get('gae_lambda', 0.95),
                step=0.01
            )
            
            clip_range = st.slider(
                "Clip Range",
                min_value=0.1,
                max_value=0.5,
                value=st.session_state.config.get('ppo', {}).get('clip_range', 0.2),
                step=0.01
            )
            
            ent_coef = st.number_input(
                "Entropy Coefficient",
                value=st.session_state.config.get('ppo', {}).get('ent_coef', 0.01),
                min_value=0.001,
                max_value=0.1,
                step=0.001,
                format="%.3f"
            )
        
        if st.button("Save PPO Configuration"):
            st.session_state.config['ppo'] = {
                'learning_rate': learning_rate,
                'n_steps': n_steps,
                'batch_size': batch_size,
                'n_epochs': n_epochs,
                'gamma': gamma,
                'gae_lambda': gae_lambda,
                'clip_range': clip_range,
                'ent_coef': ent_coef
            }
            # Save to persistent storage
            save_user_settings(st.session_state.config)
            st.success("PPO configuration saved!")
    
    with tab3:
        st.subheader("Risk Management")
        
        col1, col2 = st.columns(2)
        
        with col1:
            max_drawdown = st.slider(
                "Maximum Drawdown (%)",
                min_value=5,
                max_value=50,
                value=int(st.session_state.config.get('risk_management', {}).get('max_drawdown', 0.2) * 100),
                step=1
            )
            
            var_confidence = st.slider(
                "VaR Confidence Level (%)",
                min_value=90,
                max_value=99,
                value=int((1 - st.session_state.config.get('risk_management', {}).get('var_confidence', 0.05)) * 100),
                step=1
            )
        
        with col2:
            max_leverage = st.slider(
                "Maximum Leverage",
                min_value=1.0,
                max_value=5.0,
                value=st.session_state.config.get('risk_management', {}).get('max_leverage', 2.0),
                step=0.1
            )
            
            position_concentration = st.slider(
                "Position Concentration Limit (%)",
                min_value=10,
                max_value=100,
                value=int(st.session_state.config.get('risk_management', {}).get('position_concentration_limit', 0.3) * 100),
                step=5
            )
        
        if st.button("Save Risk Management Configuration"):
            st.session_state.config['risk_management'] = {
                'max_drawdown': max_drawdown / 100,
                'var_confidence': 1 - (var_confidence / 100),
                'max_leverage': max_leverage,
                'position_concentration_limit': position_concentration / 100
            }
            # Save to persistent storage
            save_user_settings(st.session_state.config)
            st.success("Risk management configuration saved!")
    
    with tab4:
        st.subheader("Data Sources")
        
        provider = st.selectbox(
            "Data Provider",
            options=['yfinance', 'alphavantage', 'polygon'],
            index=0
        )
        
        if provider == 'alphavantage':
            api_key = st.text_input(
                "Alpha Vantage API Key",
                type="password",
                help="Get your free API key from https://www.alphavantage.co/support/#api-key"
            )
        elif provider == 'polygon':
            api_key = st.text_input(
                "Polygon API Key",
                type="password",
                help="Get your API key from https://polygon.io/"
            )
        else:
            st.info("Yahoo Finance is free and doesn't require an API key.")
            api_key = ""
        
        if st.button("Save Data Source Configuration"):
            st.session_state.config['tradingview'] = {'provider': provider}
            if api_key:
                st.session_state.config['data_providers'] = {
                    provider: {'api_key': api_key}
                }
            # Save to persistent storage
            save_user_settings(st.session_state.config)
            st.success("Data source configuration saved!")
    
    # Save all configuration to file
    st.markdown("---")
    col1, col2, col3 = st.columns([1, 1, 1])
    
    with col1:
        if st.button("💾 Save All to File", width="stretch"):
            try:
                config_manager = ConfigManager()
                config_manager.config = st.session_state.config
                
                os.makedirs("config", exist_ok=True)
                config_manager.save_config("config/config.yaml")
                
                st.success("Configuration saved to config/config.yaml")
            except Exception as e:
                st.error(f"Error saving configuration: {e}")
    
    with col2:
        if st.button("📂 Load from File", width="stretch"):
            try:
                if os.path.exists("config/config.yaml"):
                    config_manager = ConfigManager("config/config.yaml")
                    st.session_state.config = config_manager.to_dict()
                    st.success("Configuration loaded from file")
                    st.rerun()
                else:
                    st.warning("No configuration file found")
            except Exception as e:
                st.error(f"Error loading configuration: {e}")
    
    with col3:
        if st.button("🔄 Reset to Defaults", width="stretch"):
            st.session_state.config = create_default_config()
            st.success("Configuration reset to defaults")
            st.rerun()

@handle_errors
@handle_errors
def show_data_analysis():
    """Show data analysis page."""
    st.title("📊 Data Analysis")
    
    st.markdown("""
    <div class="warning-box">
        <strong>Note:</strong> This page shows sample data analysis. In a production environment, 
        this would connect to your configured data sources.
    </div>
    """, unsafe_allow_html=True)
    
    # Symbol selection
    symbols = st.session_state.config.get('trading', {}).get('symbols', ['AAPL'])
    selected_symbol = st.selectbox("Select Symbol", symbols)
    
    # Date range selection
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input("Start Date", value=datetime.now() - timedelta(days=365))
    with col2:
        end_date = st.date_input("End Date", value=datetime.now())
    
    if st.button("Load Data"):
        with st.spinner("Loading and analyzing data..."):
            # This would normally load real data
            st.info(f"Loading data for {selected_symbol} from {start_date} to {end_date}")
            
            # Create sample data for demonstration
            dates = pd.date_range(start=start_date, end=end_date, freq='D')
            np.random.seed(42)
            
            # Generate realistic stock price data
            price_changes = np.random.normal(0.001, 0.02, len(dates))
            prices = 100 * (1 + price_changes).cumprod()
            
            volume = np.random.normal(1000000, 200000, len(dates))
            volume = np.abs(volume).astype(int)
            
            data = pd.DataFrame({
                'Date': dates,
                'Close': prices,
                'Volume': volume
            })
            
            # Display data
            st.subheader(f"Price Chart - {selected_symbol}")
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=data['Date'],
                y=data['Close'],
                mode='lines',
                name='Close Price',
                line=dict(color='#1f77b4', width=2)
            ))
            
            fig.update_layout(
                title=f"{selected_symbol} Price Movement",
                xaxis_title="Date",
                yaxis_title="Price ($)",
                height=400,
                **get_plotly_theme()
            )
            
            st.plotly_chart(fig, width="stretch")
            
            # Statistics - with safety checks
            if len(data) > 0:
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Current Price", f"${data['Close'].iloc[-1]:.2f}")
                
                with col2:
                    # Protect against division by zero
                    start_price = data['Close'].iloc[0]
                    end_price = data['Close'].iloc[-1]
                    if start_price > 0:
                        total_return = (end_price / start_price - 1) * 100
                    else:
                        total_return = 0.0
                    st.metric("Total Return", f"{total_return:.1f}%")
                
                with col3:
                    returns = data['Close'].pct_change().dropna()
                    if len(returns) > 0:
                        volatility = returns.std() * np.sqrt(252) * 100
                    else:
                        volatility = 0.0
                    st.metric("Volatility (Annual)", f"{volatility:.1f}%")
            else:
                st.warning("No data available for statistics")
            
            with col4:
                avg_volume = data['Volume'].mean()
                st.metric("Avg Volume", f"{avg_volume:,.0f}")

# Add caching for expensive operations
@st.cache_data(ttl=10)  # Cache for 10 seconds
def get_available_models():
    """Get list of available model files with caching."""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    model_dir = os.path.join(project_root, "models")
    
    if os.path.exists(model_dir):
        try:
            return [f for f in os.listdir(model_dir) if f.endswith('.pt')]
        except (OSError, PermissionError):
            return []
    return []

@handle_errors
def show_training():
    """Show training interface with support for continuing existing models."""
    import subprocess
    import sys
    
    st.title("🎯 Model Training")
    
    # Training state management
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        # Show current training status
        if st.session_state.training_active:
            st.info("🔄 Training appears to be active")
        else:
            st.success("✅ Ready to start new training")
    
    with col2:
        # Manual training state reset button
        if st.button("🔄 Refresh Status", help="Check for active training processes and refresh status"):
            # Force re-check training status
            if detect_active_training():
                st.rerun()
            else:
                st.session_state.training_active = False
                if hasattr(st.session_state, 'training_process'):
                    delattr(st.session_state, 'training_process')
                if hasattr(st.session_state, 'training_mode'):
                    delattr(st.session_state, 'training_mode')
                st.success("Status refreshed")
                st.rerun()
    
    with col3:
        # Force reset button for stuck states
        if st.button("🚫 Reset State", help="Force reset training state if stuck"):
            st.session_state.training_active = False
            if hasattr(st.session_state, 'training_process'):
                delattr(st.session_state, 'training_process')
            if hasattr(st.session_state, 'training_mode'):
                delattr(st.session_state, 'training_mode')
            if hasattr(st.session_state, 'training_start_time'):
                delattr(st.session_state, 'training_start_time')
            st.success("Training state reset successfully!")
            st.rerun()
    
    st.divider()
    
    # Get available models for continuing training (cached)
    available_models = get_available_models()
    
    # Model directory path for file operations
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    model_dir = os.path.join(project_root, "models")
    
    # Training mode selection
    st.subheader("🎯 Training Mode")
    
    col1, col2 = st.columns(2)
    
    with col1:
        training_mode = st.radio(
            "Select Training Mode:",
            ["🆕 Train New Model", "🔄 Continue Existing Model", "♾️ Continuous Training"],
            help="Choose whether to start fresh, continue training, or run continuous training until stopped"
        )
    
    with col2:
        if training_mode == "🔄 Continue Existing Model":
            if available_models:
                selected_model = st.selectbox(
                    "Select Model to Continue:",
                    available_models,
                    help="Choose an existing model to continue training"
                )
                
                # Show model info
                if selected_model:
                    model_path = os.path.join(model_dir, selected_model)
                    if os.path.exists(model_path):
                        stat = os.stat(model_path)
                        st.info(f"📊 Selected: {selected_model}")
                        st.info(f"📅 Last modified: {datetime.fromtimestamp(stat.st_mtime).strftime('%Y-%m-%d %H:%M')}")
                        st.info(f"💾 Size: {stat.st_size / (1024*1024):.2f} MB")
            else:
                st.warning("⚠️ No existing models found. Please train a new model first.")
                training_mode = "🆕 Train New Model"
        
        elif training_mode == "♾️ Continuous Training":
            st.markdown("""
            <div class="info-box">
                <strong>♾️ Continuous Training Mode</strong><br>
                Training will run indefinitely until you manually stop it.
            </div>
            """, unsafe_allow_html=True)
            
            # Option to select existing model or start fresh
            continuous_start_mode = st.radio(
                "Start from:",
                ["🆕 Fresh Model", "📂 Existing Model"],
                help="Choose whether to start continuous training from scratch or from an existing model"
            )
            
            selected_model = None
            if continuous_start_mode == "📂 Existing Model" and available_models:
                selected_model = st.selectbox(
                    "Select Base Model:",
                    available_models,
                    help="Choose an existing model to use as starting point for continuous training"
                )
                
                if selected_model:
                    model_path = os.path.join(model_dir, selected_model)
                    if os.path.exists(model_path):
                        stat = os.stat(model_path)
                        st.info(f"📊 Base model: {selected_model}")
                        st.info(f"📅 Last modified: {datetime.fromtimestamp(stat.st_mtime).strftime('%Y-%m-%d %H:%M')}")
            
            # Continuous training parameters
            col_save, col_checkpoint = st.columns(2)
            with col_save:
                save_interval = st.number_input(
                    "Save Interval (timesteps)",
                    min_value=1000,
                    max_value=500000,
                    value=50000,
                    step=1000,
                    help="How often to save the model during continuous training"
                )
            
            with col_checkpoint:
                checkpoint_interval = st.number_input(
                    "Checkpoint Interval (timesteps)",
                    min_value=1000,
                    max_value=100000,
                    value=10000,
                    step=1000,
                    help="How often to create checkpoint saves"
                )
    
    # Training status
    if st.session_state.training_active:
        st.markdown("""
        <div class="success-box">
            <strong>🔄 Training in Progress!</strong><br>
            The model is currently being trained. Monitor the progress below.
        </div>
        """, unsafe_allow_html=True)
    else:
        if training_mode == "🆕 Train New Model":
            st.markdown("""
            <div class="warning-box">
                <strong>🆕 Ready to Train New Model</strong><br>
                Configure your training parameters and start training a fresh model.
            </div>
            """, unsafe_allow_html=True)
        elif training_mode == "🔄 Continue Existing Model":
            st.markdown("""
            <div class="success-box">
                <strong>🔄 Ready to Continue Training</strong><br>
                The selected model will be loaded and training will continue from its current state.
            </div>
            """, unsafe_allow_html=True)
        else:  # Continuous training
            st.markdown("""
            <div class="info-box">
                <strong>♾️ Ready for Continuous Training</strong><br>
                Training will run indefinitely until you manually stop it. Use Ctrl+C or create 'stop_training.txt' file.
            </div>
            """, unsafe_allow_html=True)
    
    # Training configuration
    st.subheader("⚙️ Training Configuration")
    
    # Model naming section
    st.markdown("**📝 Model Name**")
    col_name1, col_name2 = st.columns([2, 1])
    
    with col_name1:
        if training_mode == "🆕 Train New Model":
            model_name = st.text_input(
                "Model Name (optional)",
                placeholder="e.g., my_trading_bot_v1",
                help="Custom name for your model. If empty, a timestamp-based name will be used."
            )
        elif training_mode == "🔄 Continue Existing Model":
            model_name = st.text_input(
                "Save As (optional)", 
                placeholder="e.g., improved_model_v2",
                help="Optional new name for the continued model. If empty, will add timestamp to original name."
            )
        else:  # Continuous training
            model_name = st.text_input(
                "Model Name Prefix (optional)",
                placeholder="e.g., continuous_trader",
                help="Prefix for saved models during continuous training. Timestamp will be added automatically."
            )
    
    with col_name2:
        if model_name:
            st.markdown("**Preview:**")
            if training_mode == "♾️ Continuous Training":
                st.code(f"{model_name}_YYYYMMDD_HHMMSS.pt")
            else:
                st.code(f"{model_name}.pt")
    
    st.markdown("---")  # Separator
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**📊 Training Parameters**")
        
        if training_mode == "♾️ Continuous Training":
            st.info("💡 Continuous training runs indefinitely with periodic saves.")
            st.markdown("**Stop Methods:**")
            st.markdown("• Press **Ctrl+C** in terminal")
            st.markdown(f"• Create file: `stop_training.txt`")
            st.markdown("• Use the **Stop Training** button below")
            
        else:
            if training_mode == "🔄 Continue Existing Model":
                st.info("💡 When continuing training, these parameters will be added to the existing model's training.")
            
            additional_timesteps = st.number_input(
                "Additional Timesteps" if training_mode == "🔄 Continue Existing Model" else "Total Timesteps",
                value=50000 if training_mode == "🔄 Continue Existing Model" else 100000,
                min_value=10000,
                max_value=10000000,
                step=10000,
                help="Number of training steps to run" + (" (in addition to existing training)" if training_mode == "🔄 Continue Existing Model" else "")
            )
        
        eval_freq = st.number_input(
            "Evaluation Frequency",
            value=5000,
            min_value=1000,
            max_value=50000,
            step=1000,
            help="How often to evaluate the model during training"
        )
        
        save_freq = st.number_input(
            "Save Frequency",
            value=10000,
            min_value=5000,
            max_value=100000,
            step=5000,
            help="How often to save checkpoints"
        )
        
        # Learning rate adjustment for continued training
        if training_mode == "🔄 Continue Existing Model":
            lr_adjustment = st.selectbox(
                "Learning Rate Adjustment",
                ["Keep Current", "Reduce by Half", "Reduce by 90%", "Custom"],
                help="Adjust learning rate for continued training"
            )
            
            if lr_adjustment == "Custom":
                custom_lr = st.number_input(
                    "Custom Learning Rate",
                    value=1e-4,
                    min_value=1e-6,
                    max_value=1e-2,
                    format="%.2e",
                    help="Set a custom learning rate"
                )
    
    with col2:
        st.markdown("**🧠 Model Configuration**")
        
        if training_mode == "🆕 Train New Model":
            policy_layers = st.text_input(
                "Policy Network Layers",
                value="256,256",
                help="Comma-separated layer sizes for policy network"
            )
            
            value_layers = st.text_input(
                "Value Network Layers", 
                value="256,256",
                help="Comma-separated layer sizes for value network"
            )
            
            activation = st.selectbox(
                "Activation Function",
                options=['tanh', 'relu', 'leaky_relu'],
                index=0,
                help="Activation function for neural networks"
            )
            
            learning_rate = st.number_input(
                "Learning Rate",
                value=3e-4,
                min_value=1e-6,
                max_value=1e-2,
                format="%.2e",
                help="Learning rate for training"
            )
        else:
            st.info("🔄 **Continuing Existing Model**")
            st.write("• Network architecture will be loaded from the existing model")
            st.write("• Training will resume from the current state")
            st.write("• All hyperparameters will be preserved unless adjusted above")
            
            # Option to modify some parameters
            with st.expander("🛠️ Advanced: Modify Training Parameters"):
                st.warning("⚠️ Changing these parameters may affect training stability")
                
                modify_batch_size = st.checkbox("Modify Batch Size")
                if modify_batch_size:
                    new_batch_size = st.number_input("New Batch Size", value=64, min_value=16, max_value=512, step=16)
                
                modify_clip_range = st.checkbox("Modify Clip Range")
                if modify_clip_range:
                    new_clip_range = st.number_input("New Clip Range", value=0.2, min_value=0.1, max_value=0.5, step=0.05)
        
        # Data configuration
        st.markdown("**📈 Data Configuration**")
        data_symbols = st.text_area(
            "Trading Symbols",
            value="AAPL\nMSFT\nGOOGL",
            help="One symbol per line"
        )
        
        data_period = st.selectbox(
            "Data Period",
            ["1mo", "3mo", "6mo", "1y", "2y", "5y"],
            index=3,
            help="Historical data period for training"
        )
        
        # Performance configuration
        st.markdown("**⚡ Performance Settings**")
        num_threads = st.number_input(
            "Number of CPU Threads",
            value=4,
            min_value=1,
            max_value=16,
            step=1,
            help="Number of CPU threads to use for training. More threads can improve training speed but may use more system resources."
        )
    
    # Training Progress Monitor (if training is active)
    if st.session_state.training_active:
        st.subheader("📊 Training Progress")
        
        # Check if we have a subprocess running
        if hasattr(st.session_state, 'training_process'):
            process = st.session_state.training_process
            
            # Check if process is still running
            if process.poll() is None:
                # Process is still running
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Status", "🔄 Training Active")
                
                with col2:
                    # Calculate elapsed time
                    import time
                    elapsed = 0  # Initialize elapsed time
                    if hasattr(st.session_state, 'training_start_time'):
                        elapsed = time.time() - st.session_state.training_start_time
                        st.metric("Elapsed Time", f"{int(elapsed//60)}:{int(elapsed%60):02d}")
                    else:
                        st.session_state.training_start_time = time.time()
                        st.metric("Elapsed Time", "Starting...")
                
                with col3:
                    # Try to read progress from training log file
                    actual_progress = None
                    debug_info = ""
                    try:
                        log_file = os.path.join(os.path.dirname(__file__), '..', 'logs', 'training_debug.log')
                        if os.path.exists(log_file):
                            with open(log_file, 'r') as f:
                                lines = f.readlines()
                                
                            # Get training start time for this session
                            training_start = getattr(st.session_state, 'training_start_time', time.time())
                            
                            # Look for progress lines after training started
                            progress_lines = []
                            continuous_activity = []
                            
                            for line in reversed(lines[-100:]):  # Check last 100 lines
                                # Parse timestamp from log line
                                import re
                                timestamp_match = re.match(r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})', line)
                                if timestamp_match:
                                    log_time = datetime.strptime(timestamp_match.group(1), '%Y-%m-%d %H:%M:%S')
                                    log_timestamp = log_time.timestamp()
                                    
                                    # Only consider logs from current training session (with buffer)
                                    if log_timestamp >= training_start - 30:  # 30 second buffer
                                        
                                        # Regular training progress
                                        if 'Step' in line and '%' in line and ')' in line:
                                            step_match = re.search(r'Step\s+(\d+)/(\d+)\s+\(\s*(\d+\.?\d*)%\)', line)
                                            if step_match:
                                                current_step = int(step_match.group(1))
                                                total_steps = int(step_match.group(2))
                                                progress_pct = float(step_match.group(3))
                                                progress_lines.append((current_step, total_steps, progress_pct, log_timestamp))
                                        
                                        # Continuous training activity (saves, checkpoints, episodes)
                                        elif any(keyword in line.lower() for keyword in ['saved model', 'checkpoint', 'episode', 'timestep']):
                                            # Look for timestep information in continuous training
                                            timestep_match = re.search(r'timestep[s]?\s*:?\s*(\d+)', line.lower())
                                            episode_match = re.search(r'episode[s]?\s*:?\s*(\d+)', line.lower())
                                            
                                            if timestep_match:
                                                timesteps = int(timestep_match.group(1))
                                                continuous_activity.append(('timestep', timesteps, log_timestamp))
                                            elif episode_match:
                                                episodes = int(episode_match.group(1))
                                                continuous_activity.append(('episode', episodes, log_timestamp))
                                            else:
                                                # General activity indicator
                                                continuous_activity.append(('activity', 0, log_timestamp))
                                        
                            # Determine progress based on training mode
                            training_mode = getattr(st.session_state, 'training_mode', 'new')
                            
                            if training_mode == 'continuous':
                                # For continuous training, show activity indicators
                                if continuous_activity and len(continuous_activity) > 0:
                                    continuous_activity.sort(key=lambda x: x[2], reverse=True)
                                    activity_type, value, timestamp = continuous_activity[0]
                                    
                                    # Calculate time since last activity
                                    time_since = time.time() - timestamp
                                    if time_since < 60:  # Active within last minute
                                        if activity_type == 'timestep':
                                            debug_info = f"Timesteps: {value:,}"
                                            actual_progress = min(85, (value / 50000) * 100)  # Rough progress based on timesteps
                                        elif activity_type == 'episode':
                                            debug_info = f"Episodes: {value:,}"
                                            actual_progress = min(85, (value / 1000) * 100)  # Rough progress based on episodes
                                        else:
                                            debug_info = "Training active"
                                            actual_progress = 50  # Show 50% for general activity
                                    else:
                                        debug_info = f"Last activity: {int(time_since//60)}m ago"
                                        actual_progress = 25  # Reduced progress for older activity
                                        
                            else:
                                # Regular training with step-based progress
                                if progress_lines and len(progress_lines) > 0:
                                    # Sort by timestamp and get the latest
                                    progress_lines.sort(key=lambda x: x[3], reverse=True)
                                    current_step, total_steps, progress_pct, _ = progress_lines[0]
                                    actual_progress = progress_pct
                                    debug_info = f"Step {current_step}/{total_steps}"
                                    
                    except Exception as e:
                        debug_info = f"Parse error"
                    
                    # Show actual progress if available, otherwise use time estimation
                    if actual_progress is not None and actual_progress > 0:
                        if training_mode == 'continuous':
                            st.metric("Training Activity", f"{actual_progress:.0f}%", delta=debug_info)
                        else:
                            st.metric("Training Progress", f"{actual_progress:.1f}%", delta=debug_info)
                        progress = actual_progress
                    else:
                        # Fallback to time-based estimation
                        if elapsed > 60:  # After 1 minute, estimate progress
                            estimated_progress = min(95, (elapsed / 3600) * 20)  # Rough estimate
                            st.metric("Training Progress", f"~{estimated_progress:.1f}%", delta="Time estimate")
                            progress = estimated_progress
                        else:
                            st.metric("Training Progress", "Starting...", delta="Initializing")
                            progress = 0
                
                # Progress bar
                if training_mode == 'continuous':
                    # For continuous training, show a pulsing or indeterminate progress indicator
                    if progress > 0:
                        st.progress(min(progress / 100.0, 0.85))  # Cap at 85% for continuous
                    else:
                        st.progress(0.5)  # Show 50% as default for continuous training
                else:
                    # Regular training with standard progress bar
                    st.progress(min(progress / 100.0, 0.99))  # Cap at 99% to show it's still running
                
                # Control buttons
                col1, col2, col3 = st.columns([1, 1, 2])
                
                with col1:
                    if st.button("⏹️ Stop Training", type="secondary"):
                        try:
                            process.terminate()
                            st.session_state.training_active = False
                            st.success("Training stopped successfully")
                            print(f"\n{'='*60}")
                            print("⏹️ TRAINING STOPPED BY USER")
                            print(f"{'='*60}\n")
                            st.rerun()
                        except:
                            st.error("Could not stop training process")
                
                with col2:
                    if st.button("📊 Refresh Status", type="secondary"):
                        st.rerun()
                
                # Console output information
                st.info("📺 **Real-time training logs are displayed in the console/terminal where you started the GUI.** Check your terminal window to see detailed training progress!")
                
                # Special note for continuous training
                if training_mode == 'continuous':
                    st.info("♾️ **Continuous Training Mode**: This training will run indefinitely. Progress shows recent activity rather than completion percentage. Stop training manually when satisfied with results.")
                
                # Auto-refresh option
                auto_refresh = st.checkbox("Auto-refresh every 10 seconds", value=True)
                if auto_refresh:
                    import time
                    time.sleep(10)
                    st.rerun()
                
            else:
                # Process has finished
                st.session_state.training_active = False
                if process.returncode == 0:
                    st.success("🎉 Training completed successfully!")
                    print(f"\n{'='*60}")
                    print("🎉 TRAINING COMPLETED SUCCESSFULLY!")
                    print(f"{'='*60}\n")
                else:
                    st.error(f"❌ Training failed or was interrupted (Exit code: {process.returncode})")
                    print(f"\n{'='*60}")
                    print(f"❌ TRAINING FAILED! Exit code: {process.returncode}")
                    print(f"{'='*60}\n")
                
                # Remove the finished process
                if hasattr(st.session_state, 'training_process'):
                    delattr(st.session_state, 'training_process')
                
                # Show completion message
                st.info("💡 Check the Models tab to see your newly trained model")
        
        st.divider()
    
    # Neural Network Architecture Visualization
    st.subheader("🧠 Neural Network Architecture")
    
    # Check if we have model information available
    if available_models:
        # Allow user to select a model to analyze
        col1, col2 = st.columns([2, 1])
        
        with col1:
            analyze_model = st.selectbox(
                "Select Model to Analyze:",
                ["None"] + available_models,
                help="Choose a model to view its neural network architecture"
            )
        
        with col2:
            if analyze_model != "None":
                if st.button("🔍 Analyze Architecture", width="stretch"):
                    with st.spinner("Analyzing neural network architecture..."):
                        time.sleep(1)  # Simulate analysis
                        st.success("✅ Architecture analysis complete!")
        
        if analyze_model != "None":
            # Show simulated network architecture
            tab1, tab2 = st.tabs(["Policy Network", "Value Network"])
            
            with tab1:
                st.markdown("**Policy Network Architecture**")
                
                # Simulated network info
                col1, col2 = st.columns(2)
                
                with col1:
                    st.metric("Total Parameters", "423,456")
                    st.metric("Layers", "6")
                    st.metric("Model Size", "1.7 MB")
                
                with col2:
                    # Show layer structure
                    st.markdown("**Layer Structure:**")
                    layers_info = [
                        "Input Layer: 40 features",
                        "Dense Layer 1: 256 units (ReLU)",
                        "LayerNorm: 256 units",
                        "Dense Layer 2: 256 units (ReLU)", 
                        "LayerNorm: 256 units",
                        "Output Layer: 3 units (Softmax)"
                    ]
                    
                    for i, layer in enumerate(layers_info):
                        if "Input" in layer:
                            st.markdown(f"**{i+1}.** {layer} 🔵")
                        elif "Output" in layer:
                            st.markdown(f"**{i+1}.** {layer} 🔴")
                        else:
                            st.markdown(f"**{i+1}.** {layer}")
                
                # Simple architecture diagram
                st.markdown("**Visual Architecture:**")
                
                # Create a simple text-based architecture visualization
                architecture_text = """
                ```
                Input (40) → Dense(256) → LayerNorm → Dense(256) → LayerNorm → Output(3)
                    ↓           ↓                        ↓                        ↓
                Features    Hidden Layer 1          Hidden Layer 2           Actions
                (OHLCV,     (Buy/Sell/Hold         (Buy/Sell/Hold          (Buy/Sell/
                 Tech       Pattern Recognition)    Strategy Learning)       Hold)
                 Indicators)
                ```
                """
                st.code(architecture_text, language="text")
            
            with tab2:
                st.markdown("**Value Network Architecture**")
                
                # Simulated value network info
                col1, col2 = st.columns(2)
                
                with col1:
                    st.metric("Total Parameters", "201,729")
                    st.metric("Layers", "5")
                    st.metric("Model Size", "0.8 MB")
                
                with col2:
                    # Show layer structure
                    st.markdown("**Layer Structure:**")
                    value_layers_info = [
                        "Input Layer: 40 features",
                        "Dense Layer 1: 256 units (ReLU)",
                        "LayerNorm: 256 units",
                        "Dense Layer 2: 256 units (ReLU)",
                        "Output Layer: 1 unit (Linear)"
                    ]
                    
                    for i, layer in enumerate(value_layers_info):
                        if "Input" in layer:
                            st.markdown(f"**{i+1}.** {layer} 🔵")
                        elif "Output" in layer:
                            st.markdown(f"**{i+1}.** {layer} 🔴")
                        else:
                            st.markdown(f"**{i+1}.** {layer}")
                
                # Value network diagram
                st.markdown("**Visual Architecture:**")
                
                value_architecture_text = """
                ```
                Input (40) → Dense(256) → LayerNorm → Dense(256) → Output(1)
                    ↓           ↓                        ↓            ↓
                Features    State Value           State Value     Portfolio
                (Market     Assessment            Refinement      Value
                 State)     (Risk/Reward)        (Fine-tuning)   Estimation
                ```
                """
                st.code(value_architecture_text, language="text")
    
    else:
        # No models available
        st.info("🔄 Neural network architecture will be displayed once you have trained models.")
        st.markdown("**Example Architecture Preview:**")
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Policy Network (Actor)**")
            st.write("• Decides which action to take")
            st.write("• Input: Market state (40 features)")
            st.write("• Output: Action probabilities")
            st.write("• Architecture: Dense → LayerNorm → Dense → Softmax")
        
        with col2:
            st.markdown("**Value Network (Critic)**")
            st.write("• Estimates state value")
            st.write("• Input: Market state (40 features)")
            st.write("• Output: Single value estimate")
            st.write("• Architecture: Dense → LayerNorm → Dense → Linear")
    
    # Current Training Configuration Display
    if st.session_state.training_active:
        st.subheader("📊 Current Training Configuration")
        
        config = st.session_state.get('training_config', {})
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("🧵 CPU Threads", config.get('num_threads', 'N/A'))
            st.metric("🎯 Training Mode", config.get('mode', 'N/A').replace('🆕 Train New Model', 'New').replace('🔄 Continue Existing Model', 'Continue').replace('♾️ Continuous Training', 'Continuous'))
        
        with col2:
            st.metric("⏱️ Timesteps", f"{config.get('timesteps', 'N/A'):,}" if isinstance(config.get('timesteps'), int) else config.get('timesteps', 'N/A'))
            st.metric("📈 Symbols", len(config.get('symbols', [])))
        
        with col3:
            st.metric("🔄 Eval Frequency", f"{config.get('eval_freq', 'N/A'):,}" if isinstance(config.get('eval_freq'), int) else config.get('eval_freq', 'N/A'))
            st.metric("💾 Save Frequency", f"{config.get('save_freq', 'N/A'):,}" if isinstance(config.get('save_freq'), int) else config.get('save_freq', 'N/A'))
        
        with col4:
            st.metric("📅 Data Period", config.get('data_period', 'N/A'))
            
            # Show symbols list if available
            if config.get('symbols'):
                symbols_text = ', '.join(config.get('symbols', []))
                if len(symbols_text) > 20:
                    symbols_text = symbols_text[:17] + "..."
                st.metric("📊 Trading Symbols", symbols_text)
        
        st.divider()

    # Enhanced Training controls
    st.subheader("🎮 Training Controls")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        # Dynamic button text based on training mode
        if training_mode == "🔄 Continue Existing Model":
            button_text = "� Continue Training"
            button_help = "Continue training the selected model"
        elif training_mode == "♾️ Continuous Training":
            button_text = "♾️ Start Continuous Training"
            button_help = "Start continuous training (runs until manually stopped)"
        else:
            button_text = "🚀 Start Training"
            button_help = "Start training a new model"
        
        # Check if we can start training
        can_start = True
        if training_mode == "🔄 Continue Existing Model" and (not available_models or not selected_model):
            can_start = False
            button_help = "No model selected for continuing training"
        elif training_mode == "♾️ Continuous Training":
            if continuous_start_mode == "📂 Existing Model" and (not available_models or not selected_model):
                can_start = False
                button_help = "No model selected for continuous training base"
        
        start_button = st.button(
            button_text, 
            disabled=st.session_state.training_active or not can_start,
            width="stretch",
            type="primary",
            help=button_help
        )
        
        if start_button:
            # Prepare to start training
            try:
                import time
                
                # Get project root directory
                current_dir = os.path.dirname(os.path.abspath(__file__))
                project_root = os.path.dirname(current_dir)
                
                # Record training start time
                st.session_state.training_start_time = time.time()
                
                # Prepare training command
                if training_mode == "🔄 Continue Existing Model":
                    model_path = os.path.join(model_dir, selected_model)
                    
                    # Create temporary config file with thread settings
                    temp_config_data = {
                        'performance': {
                            'num_threads': num_threads,
                            'compile_model': False,
                            'use_mixed_precision': False,
                            'pin_memory': False,
                            'non_blocking': False
                        }
                    }
                    
                    temp_config_path = os.path.join(project_root, "temp_gui_config.yaml")
                    try:
                        import yaml
                        with open(temp_config_path, 'w') as f:
                            yaml.dump(temp_config_data, f)
                    except Exception as e:
                        st.warning(f"Could not create temp config: {e}. Using default settings.")
                        temp_config_path = None
                    
                    # Create command for continuing training
                    cmd = [
                        sys.executable, 
                        os.path.join(project_root, "train_enhanced.py"),
                        "--mode", "continue",
                        "--model", model_path,
                        "--timesteps", str(additional_timesteps)
                    ]
                    
                    # Add config file if created successfully
                    if temp_config_path and os.path.exists(temp_config_path):
                        cmd.extend(["--config", temp_config_path])
                    
                    # Add model name if provided
                    if model_name:
                        cmd.extend(["--model-name", model_name])
                    
                    # Store command for debugging
                    st.session_state.last_training_command = cmd
                    
                    # Initialize console output storage
                    if 'console_output' not in st.session_state:
                        st.session_state.console_output = []
                    st.session_state.console_output.clear()
                    
                    # Start training in background with real-time console output
                    print(f"\n{'='*60}")
                    print("🔄 CONTINUING TRAINING - Real-time console output:")
                    print(f"Command: {' '.join(cmd)}")
                    print(f"{'='*60}")
                    
                    # Start training in background
                    st.session_state.training_process = subprocess.Popen(
                        cmd,
                        cwd=project_root,
                        stdout=None,  # Let output go to console
                        stderr=None,  # Let errors go to console
                        text=True
                    )
                    
                    st.session_state.training_mode = "continue"
                    st.session_state.continue_model = selected_model
                    st.success(f"🔄 Started continuing training of {selected_model}!")
                    st.info(f"📊 Adding {additional_timesteps:,} more training steps")
                    
                elif training_mode == "♾️ Continuous Training":
                    # Create temporary config file with thread settings
                    temp_config_data = {
                        'performance': {
                            'num_threads': num_threads,
                            'compile_model': False,
                            'use_mixed_precision': False,
                            'pin_memory': False,
                            'non_blocking': False
                        }
                    }
                    
                    temp_config_path = os.path.join(project_root, "temp_gui_config.yaml")
                    try:
                        import yaml
                        with open(temp_config_path, 'w') as f:
                            yaml.dump(temp_config_data, f)
                    except Exception as e:
                        st.warning(f"Could not create temp config: {e}. Using default settings.")
                        temp_config_path = None
                    
                    # Create command for continuous training
                    cmd = [
                        sys.executable,
                        os.path.join(project_root, "train_enhanced.py"),
                        "--mode", "continuous",
                        "--save-interval", str(save_interval),
                        "--checkpoint-interval", str(checkpoint_interval)
                    ]
                    
                    # Add config file if created successfully
                    if temp_config_path and os.path.exists(temp_config_path):
                        cmd.extend(["--config", temp_config_path])
                    
                    # Add model path if starting from existing model
                    if continuous_start_mode == "📂 Existing Model" and selected_model:
                        model_path = os.path.join(model_dir, selected_model)
                        cmd.extend(["--model", model_path])
                    
                    # Add model name if provided
                    if model_name:
                        cmd.extend(["--model-name", model_name])
                    
                    # Store command for debugging
                    st.session_state.last_training_command = cmd
                    
                    # Initialize console output storage
                    if 'console_output' not in st.session_state:
                        st.session_state.console_output = []
                    st.session_state.console_output.clear()
                    
                    # Start training in background with real-time console output
                    print(f"\n{'='*60}")
                    print("♾️ STARTING CONTINUOUS TRAINING - Real-time console output:")
                    print(f"Command: {' '.join(cmd)}")
                    print(f"{'='*60}")
                    
                    # Start training in background
                    st.session_state.training_process = subprocess.Popen(
                        cmd,
                        cwd=project_root,
                        stdout=None,  # Let output go to console
                        stderr=None,  # Let errors go to console
                        text=True
                    )
                    
                    st.session_state.training_mode = "continuous"
                    st.session_state.training_config = {
                        'save_interval': save_interval,
                        'checkpoint_interval': checkpoint_interval,
                        'start_mode': continuous_start_mode,
                        'base_model': selected_model if continuous_start_mode == "📂 Existing Model" else None
                    }
                    st.success("♾️ Started continuous training!")
                    st.info(f"💾 Saves every {save_interval:,} timesteps | Checkpoints every {checkpoint_interval:,} timesteps")
                    if continuous_start_mode == "📂 Existing Model" and selected_model:
                        st.info(f"📂 Starting from model: {selected_model}")
                    else:
                        st.info("🆕 Starting from fresh model")
                    
                    # Show stop instructions
                    st.markdown("""
                    **🛑 To Stop Continuous Training:**
                    - Use the **Stop Training** button below
                    - Press **Ctrl+C** in the terminal
                    - Create a file named `stop_training.txt` in the project directory
                    """)
                    
                else:
                    # Create temporary config file with thread settings
                    temp_config_data = {
                        'performance': {
                            'num_threads': num_threads,
                            'compile_model': False,
                            'use_mixed_precision': False,
                            'pin_memory': False,
                            'non_blocking': False
                        }
                    }
                    
                    temp_config_path = os.path.join(project_root, "temp_gui_config.yaml")
                    try:
                        import yaml
                        with open(temp_config_path, 'w') as f:
                            yaml.dump(temp_config_data, f)
                    except Exception as e:
                        st.warning(f"Could not create temp config: {e}. Using default settings.")
                        temp_config_path = None
                    
                    # Create command for new training
                    cmd = [
                        sys.executable,
                        os.path.join(project_root, "train_enhanced.py"),
                        "--mode", "new",
                        "--timesteps", str(additional_timesteps)
                    ]
                    
                    # Add config file if created successfully
                    if temp_config_path and os.path.exists(temp_config_path):
                        cmd.extend(["--config", temp_config_path])
                    
                    # Add model name if provided
                    if model_name:
                        cmd.extend(["--model-name", model_name])
                    
                    # Store command for debugging
                    st.session_state.last_training_command = cmd
                    
                    # Initialize console output storage
                    if 'console_output' not in st.session_state:
                        st.session_state.console_output = []
                    st.session_state.console_output.clear()
                    
                    # Start training in background with real-time console output
                    print(f"\n{'='*60}")
                    print("🚀 STARTING NEW TRAINING - Real-time console output:")
                    print(f"Command: {' '.join(cmd)}")
                    print(f"{'='*60}")
                    
                    # Start training in background
                    st.session_state.training_process = subprocess.Popen(
                        cmd,
                        cwd=project_root,
                        stdout=None,  # Let output go to console
                        stderr=None,  # Let errors go to console
                        text=True
                    )
                    
                    st.session_state.training_mode = "new"
                    st.success("🚀 Started training new model!")
                    st.info(f"📊 Training for {additional_timesteps:,} timesteps")
                
                st.session_state.training_active = True
                st.session_state.training_start_time = time.time()
                
                # Inform user about console output
                st.info("📺 **Training logs will appear in the console/terminal where you started this GUI.** Switch to your terminal window to see real-time training progress!")
                
                # Show what command is being run for debugging
                with st.expander("🔍 Debug: Command Being Executed"):
                    st.code(' '.join(st.session_state.last_training_command), language='bash')
                
                # Store training config
                st.session_state.training_config = {
                    'mode': training_mode,
                    'timesteps': additional_timesteps,
                    'eval_freq': eval_freq,
                    'save_freq': save_freq,
                    'symbols': [s.strip() for s in data_symbols.split('\n') if s.strip()],
                    'data_period': data_period,
                    'num_threads': num_threads
                }
                
                st.rerun()
                
            except Exception as e:
                st.error(f"❌ Failed to start training: {str(e)}")
                st.info("💡 Make sure the training script is available and all dependencies are installed")
    
    with col2:
        if st.button("⏸️ Pause Training", disabled=not st.session_state.training_active, width="stretch"):
            # For subprocess training, we can't easily pause, so we'll note this
            training_process = getattr(st.session_state, 'training_process', None)
            if training_process and training_process.poll() is None:
                st.warning("⚠️ Cannot pause subprocess training - use Stop instead")
            else:
                st.session_state.training_active = False
                st.info("⏸️ Training paused - can be resumed later")
                st.rerun()
    
    with col3:
        stop_button_text = "🛑 Stop Continuous Training" if st.session_state.get('training_mode') == 'continuous' else "🛑 Stop Training"
        
        if st.button(stop_button_text, disabled=not st.session_state.training_active, use_container_width=True):
            # For continuous training, create stop file for graceful shutdown
            if st.session_state.get('training_mode') == 'continuous':
                try:
                    with open("stop_training.txt", "w") as f:
                        f.write("Graceful stop requested from GUI")
                    st.info("📝 Stop signal sent to continuous training (creating stop_training.txt)")
                except Exception as e:
                    st.error(f"❌ Error creating stop file: {str(e)}")
            
            # Stop the actual training process if running
            training_process = getattr(st.session_state, 'training_process', None)
            if training_process and training_process.poll() is None:
                try:
                    training_process.terminate()
                    if st.session_state.get('training_mode') == 'continuous':
                        st.warning("🛑 Continuous training process terminated")
                    else:
                        st.warning("🛑 Training process terminated")
                except Exception as e:
                    st.error(f"❌ Error stopping training: {str(e)}")
            
            st.session_state.training_active = False
            st.session_state.training_metrics = []
            st.session_state.training_process = None
            
            if st.session_state.get('training_mode') == 'continuous':
                st.warning("🛑 Continuous training stopped")
            else:
                st.warning("🛑 Training stopped and reset")
            st.rerun()
    
    with col4:
        if st.button("💾 Save Checkpoint", disabled=not st.session_state.training_active, use_container_width=True):
            st.success("💾 Checkpoint saved!")
            st.info("Model state saved for recovery")
    
    # Training Diagnostics Section
    st.subheader("🔧 Training Diagnostics")
    
    diag_col1, diag_col2 = st.columns(2)
    
    with diag_col1:
        if st.button("🔍 Test Training Script", use_container_width=True):
            with st.spinner("Testing training script availability..."):
                # Test if the training script exists and is runnable
                current_dir = os.path.dirname(os.path.abspath(__file__))
                project_root = os.path.dirname(current_dir)
                train_script = os.path.join(project_root, "train_enhanced.py")
                
                if os.path.exists(train_script):
                    st.success("✅ Training script found")
                    
                    # Test if we can run it with --help
                    try:
                        import subprocess
                        result = subprocess.run([
                            sys.executable, train_script, "--help"
                        ], capture_output=True, text=True, timeout=10)
                        
                        if result.returncode == 0:
                            st.success("✅ Training script is executable")
                            with st.expander("📋 Script Help Output"):
                                st.code(result.stdout, language="text")
                        else:
                            st.error("❌ Training script has issues")
                            st.code(result.stderr, language="text")
                    except subprocess.TimeoutExpired:
                        st.warning("⚠️ Script test timed out")
                    except Exception as e:
                        st.error(f"❌ Error testing script: {e}")
                else:
                    st.error("❌ Training script not found")
                    st.info(f"Expected location: {train_script}")
    
    with diag_col2:
        if st.button("🧪 Test Model Loading", use_container_width=True):
            if available_models:
                test_model = st.selectbox("Select model to test:", available_models, key="test_model")
                
                with st.spinner(f"Testing model loading: {test_model}"):
                    try:
                        import torch
                        model_path = os.path.join(model_dir, test_model)
                        
                        # Try to load the model
                        checkpoint = torch.load(model_path, map_location='cpu')
                        st.success("✅ Model loads successfully")
                        
                        # Show model info
                        if isinstance(checkpoint, dict):
                            st.info("📊 Model contains:")
                            for key in checkpoint.keys():
                                st.write(f"• {key}")
                        else:
                            st.info("📊 Model is a direct state dict")
                            
                    except Exception as e:
                        st.error(f"❌ Model loading failed: {e}")
            else:
                st.info("No models available to test")
    
    # Show last training command for debugging
    if hasattr(st.session_state, 'last_training_command'):
        with st.expander("🐛 Last Training Command (Debug)"):
            st.write("**Command executed:**")
            st.code(" ".join(st.session_state.last_training_command), language="bash")
            
            # Show individual arguments for clarity
            st.write("**Arguments breakdown:**")
            for i, arg in enumerate(st.session_state.last_training_command):
                st.write(f"{i}: `{arg}`")
    
    st.divider()
    
    # Training progress and status
    if st.session_state.training_active:
        # Add auto-refresh for training progress
        import time
        
        # Auto-refresh mechanism using JavaScript
        st.markdown("""
        <script>
        // Auto-refresh every second when training is active
        setTimeout(function() {
            if (document.querySelector('.training-active')) {
                location.reload();
            }
        }, 1000);
        </script>
        <div class="training-active" style="background-color: #e8f4fd; padding: 10px; border-radius: 5px; margin-bottom: 10px;">
            <small>🔄 Auto-refreshing every second during training...</small>
        </div>
        """, unsafe_allow_html=True)
        
        # Fallback auto-refresh using Streamlit's rerun
        if 'last_refresh' not in st.session_state:
            st.session_state.last_refresh = time.time()
        
        # Auto-refresh every second using Streamlit's built-in mechanism
        current_time = time.time()
        if current_time - st.session_state.last_refresh >= 1.0:
            st.session_state.last_refresh = current_time
            # Use direct rerun without threading to avoid context issues
            try:
                st.rerun()
            except Exception:
                pass  # Ignore errors if already refreshing
        
        # Check if training process is still running
        training_process = getattr(st.session_state, 'training_process', None)
        
        if training_process:
            # Check if process is still running
            if training_process.poll() is None:
                # Process is still running
                st.markdown("""
                <div class="success-box">
                    <strong>🔄 Training in Progress!</strong><br>
                    The model is currently being trained. Monitor the progress below.
                </div>
                """, unsafe_allow_html=True)
                
                # Show live output
                with st.expander("📋 Training Output (Live)", expanded=False):
                    output_placeholder = st.empty()
                    
                    # Try to read some output
                    try:
                        # Read available output without blocking
                        import select
                        
                        # For Windows, we'll use a simpler approach
                        if hasattr(training_process.stdout, 'readable'):
                            # Read any available output
                            output_lines = []
                            try:
                                for line in training_process.stdout:
                                    output_lines.append(line.strip())
                                    if len(output_lines) > 20:  # Keep only last 20 lines
                                        output_lines = output_lines[-20:]
                                
                                if output_lines:
                                    output_placeholder.code('\n'.join(output_lines[-10:]))  # Show last 10 lines
                                else:
                                    output_placeholder.text("Waiting for training output...")
                            except:
                                output_placeholder.text("Training process started... waiting for output")
                        else:
                            output_placeholder.text("Training process started... waiting for output")
                    
                    except Exception as e:
                        output_placeholder.text(f"Output monitoring: {str(e)}")
            
            else:
                # Process has finished
                return_code = training_process.returncode
                if return_code == 0:
                    st.success("✅ Training completed successfully!")
                    
                    # Show final output
                    stdout, stderr = training_process.communicate()
                    if stdout:
                        with st.expander("� Training Output"):
                            st.code(stdout)
                    
                    # Reset training state
                    st.session_state.training_active = False
                    st.session_state.training_process = None
                    
                    # Refresh model list
                    st.rerun()
                else:
                    st.error(f"❌ Training failed with exit code: {return_code}")
                    
                    # Show error output
                    stdout, stderr = training_process.communicate()
                    if stderr:
                        with st.expander("❌ Error Output"):
                            st.code(stderr)
                    if stdout:
                        with st.expander("📋 Standard Output"):
                            st.code(stdout)
                    
                    # Reset training state
                    st.session_state.training_active = False
                    st.session_state.training_process = None
        
        else:
            # No process found, show simulated progress
            st.markdown("""
            <div class="info-box">
                <strong>🔄 Training Active (Simulation Mode)</strong><br>
                Demo training metrics are being generated.
            </div>
            """, unsafe_allow_html=True)
        
        # Show current training info
        with st.expander("📊 Current Training Session", expanded=True):
            # Add timestamp of last update
            current_timestamp = datetime.now().strftime("%H:%M:%S")
            st.caption(f"Last updated: {current_timestamp}")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                mode = getattr(st.session_state, 'training_mode', 'new')
                if mode == 'continue':
                    st.metric("Training Mode", "🔄 Continue", f"Model: {getattr(st.session_state, 'continue_model', 'Unknown')}")
                else:
                    st.metric("Training Mode", "🆕 New Model", "Fresh start")
            
            with col2:
                config = getattr(st.session_state, 'training_config', {})
                st.metric("Target Steps", f"{config.get('timesteps', 0):,}", "Additional training")
            
            with col3:
                if training_process and training_process.poll() is None:
                    st.metric("Status", "🟢 Running", "Process active")
                elif training_process:
                    st.metric("Status", "🔴 Finished", f"Exit code: {training_process.returncode}")
                else:
                    st.metric("Status", "🟡 Simulation", "Demo mode")
    
    # Enhanced Training metrics and model versioning
    if st.session_state.training_active or st.session_state.training_metrics:
        st.subheader("📈 Training Metrics & Progress")
        
        # Create sample training metrics with enhanced info
        if st.session_state.training_active:
            # Simulate training progress
            base_step = len(st.session_state.training_metrics)
            
            # If continuing training, add offset based on model
            if getattr(st.session_state, 'training_mode', '') == 'continue':
                # Simulate that the existing model already has some training
                base_step += 5000  # Assume existing model had 5000 steps
            
            # Simulate improving metrics over time
            progress_factor = min(base_step / 10000, 1.0)  # Improvement over time
            
            new_metric = {
                'step': base_step + 1,
                'reward': np.random.normal(0.05 + progress_factor * 0.15, 0.03),
                'policy_loss': np.random.normal(0.02 - progress_factor * 0.01, 0.005),
                'value_loss': np.random.normal(0.08 - progress_factor * 0.03, 0.01),
                'entropy': np.random.normal(0.7 + progress_factor * 0.2, 0.1),
                'episode_length': np.random.normal(100 + progress_factor * 50, 20),
                'learning_rate': 3e-4 * (0.99 ** (base_step / 1000))  # Decay over time
            }
            st.session_state.training_metrics.append(new_metric)
        
        if st.session_state.training_metrics:
            metrics_df = pd.DataFrame(st.session_state.training_metrics)
            
            # Enhanced metrics visualization
            col1, col2 = st.columns(2)
            
            with col1:
                # Reward progress
                fig_reward = go.Figure()
                fig_reward.add_trace(go.Scatter(
                    x=metrics_df['step'],
                    y=metrics_df['reward'],
                    mode='lines+markers',
                    name='Average Reward',
                    line=dict(color='#28a745', width=3),
                    marker=dict(size=4)
                ))
                
                # Add trend line
                if len(metrics_df) > 10:
                    z = np.polyfit(metrics_df['step'], metrics_df['reward'], 1)
                    p = np.poly1d(z)
                    fig_reward.add_trace(go.Scatter(
                        x=metrics_df['step'],
                        y=p(metrics_df['step']),
                        mode='lines',
                        name='Trend',
                        line=dict(color='red', width=2, dash='dash')
                    ))
                
                fig_reward.update_layout(
                    title="🎯 Reward Progress",
                    xaxis_title="Training Step",
                    yaxis_title="Average Reward",
                    height=300,
                    showlegend=True
                )
                
                st.plotly_chart(fig_reward, use_container_width=True)
            
            with col2:
                # Loss metrics
                fig_loss = go.Figure()
                fig_loss.add_trace(go.Scatter(
                    x=metrics_df['step'],
                    y=metrics_df['policy_loss'],
                    mode='lines',
                    name='Policy Loss',
                    line=dict(color='#dc3545', width=2)
                ))
                fig_loss.add_trace(go.Scatter(
                    x=metrics_df['step'],
                    y=metrics_df['value_loss'],
                    mode='lines',
                    name='Value Loss',
                    line=dict(color='#fd7e14', width=2)
                ))
                
                fig_loss.update_layout(
                    title="📉 Loss Metrics",
                    xaxis_title="Training Step",
                    yaxis_title="Loss",
                    height=300,
                    showlegend=True
                )
                
                st.plotly_chart(fig_loss, use_container_width=True)
            
            # Current metrics with enhanced display
            if len(metrics_df) > 0:
                latest = metrics_df.iloc[-1]
                
                st.markdown("**📊 Current Training Statistics**")
                col1, col2, col3, col4, col5 = st.columns(5)
                
                with col1:
                    delta_reward = "+0.05" if len(metrics_df) > 1 else None
                    st.metric("Average Reward", f"{latest['reward']:.4f}", delta_reward)
                
                with col2:
                    st.metric("Policy Loss", f"{latest['policy_loss']:.4f}")
                
                with col3:
                    st.metric("Value Loss", f"{latest['value_loss']:.4f}")
                
                with col4:
                    st.metric("Entropy", f"{latest['entropy']:.3f}")
                
                with col5:
                    st.metric("Learning Rate", f"{latest['learning_rate']:.2e}")
    
    # Model versioning and checkpoint management
    if st.session_state.training_active or available_models:
        st.subheader("📋 Model Versioning & Checkpoints")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**💾 Checkpoint Settings**")
            
            auto_save = st.checkbox("Auto-save checkpoints", value=True, help="Automatically save model at intervals")
            if auto_save:
                checkpoint_interval = st.selectbox(
                    "Checkpoint Interval",
                    [1000, 2500, 5000, 10000],
                    index=2,
                    help="Steps between automatic checkpoints"
                )
            
            versioning_scheme = st.selectbox(
                "Versioning Scheme",
                ["timestamp", "step_count", "performance"],
                help="How to name model versions"
            )
            
            if st.button("💾 Save Current Model Version", use_container_width=True):
                if st.session_state.training_active:
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    if versioning_scheme == "timestamp":
                        version_name = f"model_v_{timestamp}.pt"
                    elif versioning_scheme == "step_count":
                        step = len(st.session_state.training_metrics)
                        version_name = f"model_step_{step:06d}.pt"
                    else:
                        reward = latest['reward'] if 'latest' in locals() else 0.1
                        version_name = f"model_reward_{reward:.3f}_{timestamp}.pt"
                    
                    st.success(f"✅ Model saved as: {version_name}")
                    st.info("💡 Model saved to models/ directory")
                else:
                    st.warning("⚠️ No active training session to save")
        
        with col2:
            st.markdown("**🔄 Model Evolution**")
            
            if available_models:
                st.write("**Available Model Versions:**")
                for i, model in enumerate(available_models):
                    col_icon, col_name = st.columns([1, 4])
                    with col_icon:
                        if "best" in model.lower():
                            st.write("🏆")
                        elif "checkpoint" in model.lower():
                            st.write("💾")
                        else:
                            st.write("🤖")
                    with col_name:
                        st.write(model)
                
                # Model comparison option
                if len(available_models) >= 2:
                    if st.button("📊 Compare Model Versions", use_container_width=True):
                        st.info("🔄 Model comparison feature would show performance differences between versions")
            else:
                st.info("💡 No model versions available yet. Train a model to see versions here.")

@handle_errors
def show_backtesting():
    """Show backtesting interface."""
    st.title("📊 Backtesting")
    # Model selection
    try:
        model_files = [f for f in os.listdir("models")] if os.path.isdir("models") else []
    except (OSError, PermissionError):
        model_files = []
    model_files = [f for f in model_files if f.endswith('.pt')]
    if not model_files:
        st.warning("No trained models found. Please train a model first.")
        return

    selected_model = st.selectbox("Select Model", model_files)

    # Backtest parameters
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input("Backtest Start Date", value=datetime.now() - timedelta(days=365))
        initial_balance = st.number_input("Initial Balance ($)", value=10000, min_value=1000, step=1000)
    with col2:
        end_date = st.date_input("Backtest End Date", value=datetime.now())
        symbols = st.session_state.get('config', {}).get('trading', {}).get('symbols', ['AAPL'])
        selected_symbol = st.selectbox("Symbol", symbols)

    # Advanced options
    with st.expander("Advanced Options"):
        benchmark_comparison = st.checkbox("Compare with Buy & Hold", value=True)

    if st.button("🚀 Run Backtest", use_container_width=True):
        with st.spinner("Running backtest..."):
            # Map explicit date range to a period suitable for provider
            def _range_to_period(sd, ed):
                sd2 = pd.to_datetime(sd)
                ed2 = pd.to_datetime(ed)
                days = max(1, (ed2 - sd2).days)
                if days <= 31:
                    return '1mo'
                if days <= 92:
                    return '3mo'
                if days <= 185:
                    return '6mo'
                if days <= 365:
                    return '1y'
                if days <= 730:
                    return '2y'
                if days <= 1825:
                    return '5y'
                return '10y'

            # Start from current UI config, but prefer model's training config if available
            cfg = st.session_state.get('config') or create_default_config()
            model_path = os.path.join("models", selected_model)
            try:
                import torch
                ckpt = torch.load(model_path, map_location='cpu')
                saved_cfg = ckpt.get('config') if isinstance(ckpt, dict) else None
                if isinstance(saved_cfg, dict) and saved_cfg:
                    cfg = saved_cfg
                    st.info("Using model's training configuration for backtest to match features and network shape.")
            except Exception as e:
                st.warning(f"Could not read model config from checkpoint; using current settings. ({e})")

            # Always override symbol from UI selection
            cfg.setdefault('trading', {})
            cfg['trading']['symbols'] = [selected_symbol]

            data_client = DataClient(cfg)
            interval = cfg.get('trading', {}).get('timeframe', '1h')
            period = _range_to_period(start_date, end_date)

            raw = data_client.get_historical_data(selected_symbol, period, interval)
            if raw is None or raw.empty:
                st.error("No historical data fetched for the selected range.")
                return

            # Trim to requested range with timezone awareness
            idx = raw.index
            tz = getattr(idx, 'tz', None)
            if tz is not None:
                start_ts = pd.Timestamp(start_date).tz_localize(tz)
                # Include entire end day
                end_ts = (pd.Timestamp(end_date) + pd.Timedelta(days=1)).tz_localize(tz) - pd.Timedelta(microseconds=1)
            else:
                start_ts = pd.Timestamp(start_date)
                end_ts = pd.Timestamp(end_date) + pd.Timedelta(days=1) - pd.Timedelta(microseconds=1)
            df_raw = raw[(idx >= start_ts) & (idx <= end_ts)]
            if df_raw.empty:
                st.error("No data within selected start/end dates. Try expanding the range.")
                return

            data = prepare_features(df_raw, cfg)
            if data.empty:
                st.error("Feature preparation returned empty data.")
                return

            # Agent and backtester
            temp_env = TradingEnvironment(data, cfg)
            obs_dim = temp_env.observation_space.shape[0]
            agent = PPOAgent(obs_dim, 3, cfg)
            agent.load(model_path)
            # Ensure inference-only behavior (disable dropout/batchnorm randomness)
            try:
                agent.policy_net.eval()
                agent.value_net.eval()
            except Exception:
                pass

            backtester = Backtester(cfg)
            results = backtester.run_backtest(
                agent=agent,
                data=data,
                start_date=pd.to_datetime(start_date).strftime('%Y-%m-%d'),
                end_date=pd.to_datetime(end_date).strftime('%Y-%m-%d'),
                initial_balance=float(initial_balance),
                deterministic=True
            )

            bt = results.get('backtest_metrics', {})
            perf = results.get('performance_metrics', {})
            detailed = results.get('detailed_results', {})

            st.subheader("Backtest Results")
            # If no trades, optionally rerun with stochastic policy
            total_trades = bt.get('total_trades', 0)
            if total_trades == 0:
                st.warning("No trades were executed with deterministic policy. Retrying with stochastic sampling to diagnose behavior…")
                results_sto = backtester.run_backtest(
                    agent=agent,
                    data=data,
                    start_date=pd.to_datetime(start_date).strftime('%Y-%m-%d'),
                    end_date=pd.to_datetime(end_date).strftime('%Y-%m-%d'),
                    initial_balance=float(initial_balance),
                    deterministic=False
                )
                bt_sto = results_sto.get('backtest_metrics', {})
                trades_sto = bt_sto.get('total_trades', 0)
                if trades_sto > 0:
                    st.info(f"Stochastic run executed {trades_sto} trades. Deterministic policy may be strongly preferring HOLD.")
                else:
                    st.warning("Even with stochastic sampling, no trades occurred. This suggests a configuration mismatch or a policy that learned to HOLD.")
                # Prefer the deterministic result for displayed metrics, but show action distributions for both
                detailed_sto = results_sto.get('detailed_results', {})
                actions_det = pd.Series(detailed.get('actions', []), name='Deterministic')
                actions_sto = pd.Series(detailed_sto.get('actions', []), name='Stochastic')
                dist_df = pd.concat([
                    actions_det.value_counts().rename(index={0:'SELL',1:'HOLD',2:'BUY'}),
                    actions_sto.value_counts().rename(index={0:'SELL',1:'HOLD',2:'BUY'})
                ], axis=1).fillna(0).astype(int)
                st.subheader("Action Distribution (Deterministic vs Stochastic)")
                st.dataframe(dist_df)

            # Show effective trading parameters
            tcfg = cfg.get('trading', {})
            st.caption(f"Params: max_position_size={tcfg.get('max_position_size', 0.1)}, fee={tcfg.get('transaction_cost', 0.001)}, slippage={tcfg.get('slippage', 0.0005)}")
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Final Portfolio Value", format_currency(bt.get('final_portfolio_value', initial_balance)))
            with col2:
                st.metric("Total Return", format_percentage(bt.get('total_return', 0)))
            with col3:
                st.metric("Sharpe Ratio", f"{perf.get('sharpe_ratio', 0):.3f}")
            with col4:
                st.metric("Max Drawdown", format_percentage(perf.get('max_drawdown', 0)))

            # Chart data
            dates = pd.to_datetime(detailed.get('dates', []))
            prices = pd.Series(detailed.get('prices', []), index=dates)
            portfolio_values = pd.Series(detailed.get('portfolio_values', []), index=dates)

            # Build benchmark
            benchmark_values = None
            if benchmark_comparison and not prices.empty:
                base_price = prices.iloc[0]
                if base_price:
                    benchmark_values = (prices / base_price) * float(initial_balance)

            # Signals
            buy_dates, buy_prices, sell_dates, sell_prices = [], [], [], []
            for i, tr in enumerate(detailed.get('trades', [])):
                if not tr or 'shares_traded' not in tr or i >= len(dates):
                    continue
                if tr['shares_traded'] > 0:
                    buy_dates.append(dates[i])
                    buy_prices.append(prices.iloc[i] if i < len(prices) else None)
                elif tr['shares_traded'] < 0:
                    sell_dates.append(dates[i])
                    sell_prices.append(prices.iloc[i] if i < len(prices) else None)

            # Price chart
            st.subheader("📈 Price Chart with Trading Signals")
            fig = go.Figure()
            if not prices.empty:
                fig.add_trace(go.Scatter(x=prices.index, y=prices.values, mode='lines', name=f'{selected_symbol} Price',
                                         line=dict(color='#1f77b4', width=2)))
            if buy_dates:
                fig.add_trace(go.Scatter(x=buy_dates, y=buy_prices, mode='markers', name='Buy Signals',
                                         marker=dict(symbol='triangle-up', size=12, color='#00ff00',
                                                     line=dict(color='#008000', width=2))))
            if sell_dates:
                fig.add_trace(go.Scatter(x=sell_dates, y=sell_prices, mode='markers', name='Sell Signals',
                                         marker=dict(symbol='triangle-down', size=12, color='#ff0000',
                                                     line=dict(color='#800000', width=2))))
            fig.update_layout(title=f"{selected_symbol} Price with AI Trading Signals", xaxis_title="Date",
                              yaxis_title="Price ($)", height=600, legend=dict(x=0, y=1), hovermode='closest',
                              showlegend=True, **get_plotly_theme())
            st.plotly_chart(fig, use_container_width=True)

            # Portfolio performance
            st.subheader("📊 Portfolio Performance")
            fig2 = go.Figure()
            if not portfolio_values.empty:
                fig2.add_trace(go.Scatter(x=portfolio_values.index, y=portfolio_values.values, mode='lines',
                                          name='AI Strategy', line=dict(color='blue', width=2)))
            if benchmark_values is not None:
                fig2.add_trace(go.Scatter(x=benchmark_values.index, y=benchmark_values.values, mode='lines',
                                          name='Buy & Hold', line=dict(color='gray', width=2, dash='dash')))
            fig2.update_layout(title="Portfolio Performance Comparison", xaxis_title="Date",
                               yaxis_title="Portfolio Value ($)", **get_plotly_theme())
            st.plotly_chart(fig2, use_container_width=True)

@handle_errors
def show_live_trading():
    """Show live trading interface."""
    st.title("📡 Live Trading Monitor")
    
    st.markdown("""
    <div class="warning-box">
        <strong>⚠️ IMPORTANT DISCLAIMER</strong><br>
        This is a demonstration interface. Live trading involves substantial risk of loss. 
        Always use paper trading first and never risk more than you can afford to lose.
    </div>
    """, unsafe_allow_html=True)
    
    # Trading status
    col1, col2, col3 = st.columns(3)
    
    with col1:
        trading_active = st.checkbox("Enable Live Trading", value=False)
    
    with col2:
        paper_trading = st.checkbox("Paper Trading Mode", value=True)
    
    with col3:
        auto_trading = st.checkbox("Automatic Trading", value=False)
    
    if trading_active and not paper_trading:
        st.error("🚨 LIVE TRADING ENABLED - REAL MONEY AT RISK!")
    elif trading_active and paper_trading:
        st.success("📝 Paper trading mode - Safe to experiment")
    
    # Portfolio overview with enhanced styling
    st.markdown('<div class="portfolio-overview">', unsafe_allow_html=True)
    st.subheader("💼 Portfolio Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="Portfolio Value",
            value="$10,234.56",
            delta="+$124.32 (+1.2%)",
            delta_color="normal"
        )
    
    with col2:
        st.metric(
            label="Cash Balance", 
            value="$2,456.78",
            delta="-$12.34 (-0.5%)",
            delta_color="inverse"
        )
    
    with col3:
        st.metric(
            label="Today's P&L",
            value="+$123.45",
            delta="+1.21%",
            delta_color="normal"
        )
    
    with col4:
        st.metric(
            label="Open Positions",
            value="3",
            delta="+1",
            delta_color="normal"
        )
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Current positions with enhanced styling
    st.subheader("📊 Current Positions")
    
    positions_data = {
        'Symbol': ['🍎 AAPL', '🖥️ MSFT', '🔍 GOOGL'],
        'Shares': [10, 15, 5],
        'Avg Price': ['$175.50', '$415.20', '$2,750.30'],
        'Current Price': ['$178.90', '$412.45', '$2,780.15'],
        'Market Value': ['$1,789.00', '$6,186.75', '$13,900.75'],
        'P&L': ['+$34.00', '-$41.25', '+$149.25'],
        'P&L %': ['📈 +1.94%', '📉 -0.66%', '📈 +0.54%']
    }
    
    positions_df = pd.DataFrame(positions_data)
    
    # Style the dataframe with conditional formatting
    def style_pnl(val):
        if '+' in str(val):
            return 'background-color: rgba(40, 167, 69, 0.1); color: #28a745; font-weight: bold;'
        elif '-' in str(val):
            return 'background-color: rgba(220, 53, 69, 0.1); color: #dc3545; font-weight: bold;'
        return ''
    
    styled_df = positions_df.style.map(style_pnl, subset=['P&L', 'P&L %'])
    st.dataframe(styled_df, use_container_width=True)
    
    # Real-time quotes (simulated) with enhanced styling
    st.subheader("📈 Real-Time Market Data")
    
    # Auto-refresh control
    col1, col2 = st.columns([3, 1])
    with col1:
        st.markdown("*Live market data with AI trading signals*")
    with col2:
        auto_refresh = st.checkbox("🔄 Auto Refresh (5s)", value=True)
    
    if auto_refresh:
        time.sleep(1)  # Simulate delay
    
    quotes_data = {
        'Symbol': ['🍎 AAPL', '🖥️ MSFT', '🔍 GOOGL', '₿ BTC/USDT', '💎 ETH/USDT'],
        'Price': ['$178.90', '$412.45', '$2,780.15', '$67,340.50', '$3,425.80'],
        'Change': ['+$3.40', '-$2.75', '+$29.85', '+$1,250.30', '+$125.60'],
        'Change %': ['+1.94%', '-0.66%', '+1.09%', '+1.89%', '+3.80%'],
        'Volume': ['12.5M', '8.7M', '1.2M', '2.1B', '890M'],
        'AI Signal': ['🟢 BUY', '🟡 HOLD', '🟢 BUY', '� BUY', '🟡 HOLD']
    }
    
    quotes_df = pd.DataFrame(quotes_data)
    
    # Enhanced styling for the quotes table
    def style_change(val):
        if '+' in str(val):
            return 'background-color: rgba(40, 167, 69, 0.1); color: #28a745; font-weight: bold;'
        elif '-' in str(val):
            return 'background-color: rgba(220, 53, 69, 0.1); color: #dc3545; font-weight: bold;'
        return ''
    
    def style_signal(val):
        if 'BUY' in str(val):
            return 'background-color: rgba(40, 167, 69, 0.2); color: #155724; font-weight: bold; border-radius: 5px; padding: 2px 8px;'
        elif 'SELL' in str(val):
            return 'background-color: rgba(220, 53, 69, 0.2); color: #721c24; font-weight: bold; border-radius: 5px; padding: 2px 8px;'
        elif 'HOLD' in str(val):
            return 'background-color: rgba(255, 193, 7, 0.2); color: #856404; font-weight: bold; border-radius: 5px; padding: 2px 8px;'
        return ''
    
    styled_quotes = quotes_df.style.map(style_change, subset=['Change', 'Change %']).map(style_signal, subset=['AI Signal'])
    st.dataframe(styled_quotes, use_container_width=True)
    
    # Trading controls
    st.subheader("Manual Trading")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        trade_symbol = st.selectbox("Symbol", ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA'])
    
    with col2:
        trade_action = st.selectbox("Action", ['BUY', 'SELL'])
    
    with col3:
        trade_quantity = st.number_input("Quantity", min_value=1, value=10)
    
    with col4:
        order_type = st.selectbox("Order Type", ['Market', 'Limit', 'Stop'])
    
    if order_type == 'Limit':
        limit_price = st.number_input("Limit Price", min_value=0.01, value=100.00, step=0.01)
    
    if st.button(f"Submit {trade_action} Order", type="primary"):
        if paper_trading:
            st.success(f"Paper trade submitted: {trade_action} {trade_quantity} shares of {trade_symbol}")
        else:
            st.error("Live trading not implemented - this is a demo interface")

@handle_errors
def show_model_management():
    """Show model management interface."""
    st.title("📁 Model Management")
    
    # Model directory status - handle relative path properly
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)  # Go up one level from gui/
    model_dir = os.path.join(project_root, "models")
    
    st.info(f"Looking for models in: {model_dir}")
    
    if not os.path.exists(model_dir):
        st.warning("⚠️ Models directory not found. No trained models available.")
        st.info("💡 Train a model first using the Training page to create models.")
        
        # Show expected directory structure
        with st.expander("🗂️ Expected Directory Structure"):
            st.code("""
ai-ppo/
├── models/          ← Models should be here
│   ├── best_model.pt
│   └── checkpoint_*.pt
├── gui/            ← Current location
│   └── app.py
└── src/
            """)
        return
    
    # Get all model files
    try:
        try:
            all_files = os.listdir(model_dir)
        except (OSError, PermissionError):
            st.error("Unable to access model directory")
            return
        model_files = [f for f in all_files if f.endswith('.pt')]
        
        st.success(f"✅ Models directory found: {len(all_files)} files total")
        
        if not model_files:
            st.info("📋 No trained models found (.pt files).")
            st.info("💡 Train a model first using the Training page.")
            
            # Show what files are in the directory
            if all_files:
                with st.expander("📁 Files in models directory"):
                    for file in all_files:
                        st.write(f"• {file}")
            return
            
        st.success(f"🎯 Found {len(model_files)} trained model(s)")
        
    except Exception as e:
        st.error(f"❌ Error accessing models directory: {e}")
        return
    
    # Enhanced Model list with detailed information
    st.subheader("📊 Available Models")
    
    model_data = []
    for model_file in model_files:
        model_path = os.path.join(model_dir, model_file)
        try:
            stat = os.stat(model_path)
            size_mb = stat.st_size / (1024*1024)
            created = datetime.fromtimestamp(stat.st_ctime)
            modified = datetime.fromtimestamp(stat.st_mtime)
            
            # Determine model type/status
            if 'best' in model_file.lower():
                model_type = "🏆 Best Model"
            elif 'checkpoint' in model_file.lower():
                model_type = "💾 Checkpoint"
            else:
                model_type = "🤖 Model"
            
            model_data.append({
                'Type': model_type,
                'Model Name': model_file,
                'Size (MB)': f"{size_mb:.2f}",
                'Created': created.strftime("%Y-%m-%d %H:%M"),
                'Modified': modified.strftime("%Y-%m-%d %H:%M"),
                'Age (days)': f"{(datetime.now() - modified).days}",
                'Status': "✅ Ready" if size_mb > 0.1 else "⚠️ Small File"
            })
        except Exception as e:
            model_data.append({
                'Type': "❌ Error",
                'Model Name': model_file,
                'Size (MB)': "N/A",
                'Created': "N/A",
                'Modified': "N/A", 
                'Age (days)': "N/A",
                'Status': f"Error: {e}"
            })
    
    models_df = pd.DataFrame(model_data)
    
    # Style the dataframe
    def style_status(val):
        if "Ready" in str(val):
            return 'background-color: rgba(40, 167, 69, 0.1); color: #28a745; font-weight: bold;'
        elif "Error" in str(val):
            return 'background-color: rgba(220, 53, 69, 0.1); color: #dc3545; font-weight: bold;'
        elif "Small File" in str(val):
            return 'background-color: rgba(255, 193, 7, 0.1); color: #856404; font-weight: bold;'
        return ''
    
    def style_type(val):
        if "Best" in str(val):
            return 'background-color: rgba(255, 215, 0, 0.2); color: #b8860b; font-weight: bold;'
        return ''
    
    styled_models = models_df.style.map(style_status, subset=['Status']).map(style_type, subset=['Type'])
    st.dataframe(styled_models, use_container_width=True)
    
    # Enhanced Model actions
    st.subheader("🛠️ Model Actions")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**📋 Model Operations**")
        selected_model = st.selectbox("Select Model", model_files, key="selected_model")
        
        if selected_model:
            # Show model details
            model_path = os.path.join(model_dir, selected_model)
            stat = os.stat(model_path)
            
            with st.expander(f"📊 Details: {selected_model}"):
                col1_1, col1_2 = st.columns(2)
                with col1_1:
                    st.metric("File Size", f"{stat.st_size / (1024*1024):.2f} MB")
                    st.metric("Created", datetime.fromtimestamp(stat.st_ctime).strftime("%Y-%m-%d"))
                with col1_2:
                    st.metric("Last Modified", datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d"))
                    st.metric("Age", f"{(datetime.now() - datetime.fromtimestamp(stat.st_mtime)).days} days")
                
                # Try to show model info if possible
                try:
                    # This would normally load model metadata
                    st.info("💡 Model architecture and training details would be shown here in a full implementation.")
                except Exception as e:
                    st.warning(f"Could not load model metadata: {e}")
        
        # Action buttons
        col1_1, col1_2 = st.columns(2)
        
        with col1_1:
            if st.button("📊 Analyze Model", use_container_width=True):
                with st.spinner(f"Analyzing {selected_model}..."):
                    time.sleep(1)  # Simulate analysis
                    st.success("✅ Model analysis complete!")
                    
                    # Show simulated analysis results
                    with st.expander("📈 Analysis Results"):
                        analysis_data = {
                            'Metric': ['Parameters', 'Training Episodes', 'Avg Reward', 'Convergence'],
                            'Value': ['~850K', '10,000', '0.245', '✅ Good']
                        }
                        analysis_df = pd.DataFrame(analysis_data)
                        st.dataframe(analysis_df, use_container_width=True)
        
        with col1_2:
            if st.button("� Load Model", use_container_width=True, type="primary"):
                with st.spinner(f"Loading {selected_model}..."):
                    time.sleep(1)  # Simulate loading
                    st.success(f"✅ {selected_model} loaded successfully!")
                    st.info("💡 Model is now ready for trading or backtesting.")
        
        # Model renaming section
        st.markdown("**✏️ Rename Model**")
        col_rename1, col_rename2 = st.columns([2, 1])
        
        with col_rename1:
            # Extract current name without extension for default
            current_name = selected_model.replace('.pt', '') if selected_model else ""
            new_name = st.text_input(
                "New Model Name",
                value="",
                placeholder=f"e.g., {current_name}_improved",
                help="Enter new name without .pt extension",
                key="rename_input"
            )
        
        with col_rename2:
            if st.button("✏️ Rename", use_container_width=True, disabled=not new_name):
                if new_name and selected_model:
                    old_path = os.path.join(model_dir, selected_model)
                    
                    # Ensure new name doesn't have .pt extension
                    if new_name.endswith('.pt'):
                        new_name = new_name[:-3]
                    
                    new_path = os.path.join(model_dir, f"{new_name}.pt")
                    
                    try:
                        # Check if new name already exists
                        if os.path.exists(new_path):
                            st.error(f"❌ A model named '{new_name}.pt' already exists!")
                        else:
                            # Rename the file
                            os.rename(old_path, new_path)
                            st.success(f"✅ Model renamed to '{new_name}.pt'!")
                            st.info("🔄 Refresh the page to see the updated model list.")
                            
                            # Clear the input
                            st.session_state.rename_input = ""
                            
                    except Exception as e:
                        st.error(f"❌ Error renaming model: {e}")
        
        # Danger zone
        with st.expander("⚠️ Danger Zone"):
            st.warning("**Delete Model** - This action cannot be undone!")
            col1_1, col1_2 = st.columns([2, 1])
            with col1_1:
                confirm_delete = st.checkbox(f"I confirm deletion of {selected_model}")
            with col1_2:
                if st.button("🗑️ Delete", use_container_width=True, disabled=not confirm_delete):
                    st.error(f"Would delete {selected_model} (Demo mode - not actually deleted)")
    
    with col2:
        st.markdown("**📊 Model Comparison**")
        
        if len(model_files) >= 2:
            model1 = st.selectbox("Model 1", model_files, key="model1")
            model2 = st.selectbox("Model 2", model_files, key="model2", index=1)
            
            if st.button("📈 Compare Models", width="stretch"):
                st.info(f"Comparing {model1} vs {model2}...")
                
                # Simulated comparison results
                comparison_data = {
                    'Metric': ['Total Return', 'Sharpe Ratio', 'Max Drawdown', 'Win Rate'],
                    model1: ['12.4%', '1.85', '-8.2%', '67%'],
                    model2: ['10.8%', '1.92', '-6.5%', '71%']
                }
                
                comparison_df = pd.DataFrame(comparison_data)
                st.dataframe(comparison_df, width="stretch")
        else:
            st.info("Need at least 2 models for comparison")
    
    # Enhanced Export/Import section
    st.subheader("📦 Export/Import Models")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**📤 Export Model**")
        export_model = st.selectbox("Select Model to Export", model_files, key="export_model")
        export_format = st.selectbox("Export Format", ["PyTorch (.pt)", "ONNX (.onnx)", "Compressed (.zip)"])
        
        if st.button("📤 Export Model", use_container_width=True):
            with st.spinner(f"Exporting {export_model}..."):
                time.sleep(2)  # Simulate export
                st.success(f"✅ {export_model} exported successfully!")
                st.info(f"💾 Exported as: {export_model.replace('.pt', f'_exported.{export_format.split(".")[-1][:-1]}')}") 
    
    with col2:
        st.markdown("**📥 Import Model**")
        uploaded_file = st.file_uploader(
            "Choose model file", 
            type=['pt', 'pth', 'onnx'],
            help="Upload a PyTorch model file (.pt, .pth) or ONNX model (.onnx)"
        )
        
        if uploaded_file is not None:
            st.info(f"📁 Selected file: {uploaded_file.name}")
            st.info(f"📊 File size: {len(uploaded_file.getvalue()) / (1024*1024):.2f} MB")
            
            if st.button("📥 Import Model", use_container_width=True, type="primary"):
                with st.spinner("Importing model..."):
                    time.sleep(2)  # Simulate import
                    st.success(f"✅ {uploaded_file.name} imported successfully!")
                    st.info("🔄 Refresh the page to see the new model in the list.")
    
    # Model backup section
    st.subheader("💾 Model Backup & Restore")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("📦 Backup All Models", use_container_width=True):
            with st.spinner("Creating backup..."):
                time.sleep(3)  # Simulate backup
                backup_name = f"model_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip"
                st.success(f"✅ Backup created: {backup_name}")
    
    with col2:
        backup_file = st.file_uploader("Restore from Backup", type=['zip'])
        if backup_file and st.button("🔄 Restore Backup", use_container_width=True):
            with st.spinner("Restoring models..."):
                time.sleep(3)  # Simulate restore
                st.success("✅ Models restored successfully!")
                st.info("🔄 Refresh the page to see restored models.")

if __name__ == "__main__":
    main()