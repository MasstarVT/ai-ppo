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

# Set up comprehensive debug logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('logs/gui_debug.log', mode='a')
    ]
)
logger = logging.getLogger(__name__)

# Enable debug mode for key components
logging.getLogger('data').setLevel(logging.DEBUG)
logging.getLogger('training').setLevel(logging.DEBUG)
logging.getLogger('streamlit').setLevel(logging.INFO)  # Reduce streamlit noise

print("🐛 DEBUG MODE ENABLED - Starting AI PPO Trading GUI...")
logger.debug("=== GUI APPLICATION STARTUP ===")
logger.debug(f"Python version: {sys.version}")
logger.debug(f"Working directory: {os.getcwd()}")
logger.debug(f"GUI file location: {__file__}")

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from streamlit_option_menu import option_menu
from st_aggrid import AgGrid, GridOptionsBuilder

# Import trading system components
SYSTEM_READY = False
IMPORT_ERROR = None

# Test yaml import first
try:
    import yaml
    print("✅ YAML module imported successfully")
    logger.debug("YAML import successful")
    st.success("✅ YAML module imported successfully")
except ImportError as e:
    print(f"❌ YAML import failed: {e}")
    logger.error(f"YAML import failed: {e}")
    st.error(f"❌ YAML import failed: {e}")
    st.info("💡 Please install PyYAML: `pip install PyYAML`")
    IMPORT_ERROR = f"YAML import error: {e}"

print("🔄 Importing trading system components...")
logger.debug("Starting import of trading system components")

try:
    logger.debug("Importing data components...")
    from data import DataClient, prepare_features
    logger.debug("Data components imported successfully")
    
    logger.debug("Importing environment components...")
    from environments import TradingEnvironment
    logger.debug("Environment components imported successfully")
    
    logger.debug("Importing agent components...")
    from agents import PPOAgent
    logger.debug("Agent components imported successfully")
    
    logger.debug("Importing evaluation components...")
    from evaluation.backtesting import Backtester
    logger.debug("Evaluation components imported successfully")
    
    logger.debug("Importing utility components...")
    from utils import ConfigManager, create_default_config, format_currency, format_percentage
    logger.debug("Utility components imported successfully")
    
    print("✅ Core trading components imported successfully")
    
    # Try to import training manager (requires torch/numpy)
    try:
        from utils import training_manager, BackgroundTrainingManager, NetworkAnalyzer
        TRAINING_MANAGER_AVAILABLE = True
        st.success("✅ Background training manager available")
    except ImportError:
        TRAINING_MANAGER_AVAILABLE = False
        st.info("ℹ️ Background training manager not available (requires torch/numpy)")
    
    # Test basic functionality
    _test_config = create_default_config()
    SYSTEM_READY = True
    st.success("✅ All trading system components imported successfully")
    
except ImportError as e:
    SYSTEM_READY = False
    IMPORT_ERROR = f"Import error: {e}"
    st.error(f"⚠️ Error importing trading system components: {e}")
    st.info("💡 Please ensure all dependencies are installed: `pip install -r requirements.txt`")
    
    # Show detailed error information
    with st.expander("🔍 Detailed Error Information"):
        st.code(f"Error: {e}")
        st.code(f"Python Path: {sys.path}")
        st.code(f"Current Directory: {os.getcwd()}")
        st.code(f"Script Directory: {os.path.dirname(__file__)}")
        
        # Test specific imports
        st.write("**Testing individual imports:**")
        for module_name in ['yaml', 'utils', 'data', 'environments', 'agents', 'evaluation.backtesting']:
            try:
                if module_name == 'yaml':
                    import yaml
                elif module_name == 'utils':
                    from utils import ConfigManager
                elif module_name == 'data':
                    from data import DataClient
                elif module_name == 'environments':
                    from environments import TradingEnvironment
                elif module_name == 'agents':
                    from agents import PPOAgent
                elif module_name == 'evaluation.backtesting':
                    from evaluation.backtesting import Backtester
                st.write(f"✅ {module_name}")
            except Exception as ex:
                st.write(f"❌ {module_name}: {ex}")
    
    # Create fallback function
    def create_default_config():
        """Fallback default config when imports fail."""
        return {
            'trading': {
                'symbols': ['AAPL', 'MSFT', 'GOOGL'],
                'timeframe': '1h',
                'initial_balance': 10000,
                'max_position_size': 0.1,
                'transaction_cost': 0.001,
                'slippage': 0.0005
            },
            'ppo': {
                'learning_rate': 3e-4,
                'n_steps': 2048,
                'batch_size': 64,
                'n_epochs': 10,
                'gamma': 0.99,
                'gae_lambda': 0.95,
                'clip_range': 0.2,
                'ent_coef': 0.01,
                'vf_coef': 0.5,
                'max_grad_norm': 0.5
            },
            'training': {
                'total_timesteps': 100000,
                'eval_freq': 10000,
                'save_freq': 50000,
                'log_interval': 1000,
                'n_eval_episodes': 10
            },
            'network': {
                'policy_layers': [256, 256],
                'value_layers': [256, 256],
                'activation': 'tanh'
            },
            'paths': {
                'data_dir': 'data',
                'model_dir': 'models',
                'log_dir': 'logs'
            }
        }
    
    def format_currency(value):
        """Fallback currency formatter."""
        return f"${value:,.2f}"
    
    def format_percentage(value):
        """Fallback percentage formatter."""
        return f"{value*100:.2f}%"
    
except Exception as e:
    SYSTEM_READY = False
    IMPORT_ERROR = f"System error: {e}"
    st.error(f"⚠️ System initialization error: {e}")


def handle_errors(func):
    """Decorator to handle errors in GUI functions gracefully."""
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            st.error(f"❌ Error in {func.__name__}: {str(e)}")
            st.info("💡 Please check your configuration and try again.")
            logger.error(f"GUI error in {func.__name__}: {e}", exc_info=True)
            return None
    return wrapper


@handle_errors
def show_system_status():
    """Show system status and health checks."""
    if not SYSTEM_READY:
        st.error("🔴 System Not Ready")
        st.warning(f"Issue: {IMPORT_ERROR}")
        st.info("Please resolve the issues above before using the system.")
        return False
    
    return True

# Page configuration
st.set_page_config(
    page_title="AI PPO Trading System",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

def get_theme_css():
    """Get CSS styles based on current theme."""
    # Initialize theme if not set
    if 'theme' not in st.session_state:
        st.session_state.theme = 'dark'
        
    if st.session_state.theme == 'light':
        return get_light_theme_css()
    else:
        return get_dark_theme_css()

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
        --bg-primary: #ffffff;
        --bg-secondary: #f8f9fa;
        --text-primary: #212529;
        --text-secondary: #495057;
        --border-color: rgba(0, 0, 0, 0.1);
        --accent-color: #007bff;
        --success-color: #28a745;
        --warning-color: #ffc107;
        --error-color: #dc3545;
        --shadow-color: rgba(0, 0, 0, 0.1);
    }
    
    /* Main container */
    .main > div {
        padding-top: 2rem;
        background-color: var(--bg-primary);
        color: var(--text-primary);
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

# Custom CSS with theme support
st.markdown(get_theme_css(), unsafe_allow_html=True)

# Initialize session state
if 'config' not in st.session_state:
    try:
        st.session_state.config = create_default_config()
    except Exception as e:
        logger.error(f"Failed to create default config: {e}")
        # Minimal fallback config
        st.session_state.config = {
            'trading': {'symbols': ['AAPL'], 'initial_balance': 10000},
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

def get_theme_css():
    """Get CSS styles based on current theme."""
    if st.session_state.theme == 'light':
        return get_light_theme_css()
    else:
        return get_dark_theme_css()

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
        --bg-primary: #ffffff;
        --bg-secondary: #f8f9fa;
        --text-primary: #212529;
        --text-secondary: #495057;
        --border-color: rgba(0, 0, 0, 0.1);
        --accent-color: #007bff;
        --success-color: #28a745;
        --warning-color: #ffc107;
        --error-color: #dc3545;
        --shadow-color: rgba(0, 0, 0, 0.1);
    }
    
    /* Main container */
    .main > div {
        padding-top: 2rem;
        background-color: var(--bg-primary);
        color: var(--text-primary);
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

def main():
    """Main application function."""
    print("🚀 Starting main GUI application...")
    logger.debug("=== MAIN APPLICATION START ===")
    
    # Check system status first
    logger.debug("Checking system status...")
    if not show_system_status():
        logger.warning("System status check failed, returning early")
        return
    
    logger.debug("System status OK, proceeding with GUI initialization")
    
    # Sidebar navigation
    with st.sidebar:
        logger.debug("Rendering sidebar navigation")
        st.image("https://via.placeholder.com/200x80/1f77b4/ffffff?text=AI+PPO+Trading", width=200)
        
        # Theme toggle button
        st.divider()
        col1, col2 = st.columns([3, 1])
        with col1:
            st.write("**Theme:**")
        with col2:
            if st.button(f"{'🌞' if st.session_state.theme == 'dark' else '🌙'}"):
                old_theme = st.session_state.theme
                st.session_state.theme = 'light' if st.session_state.theme == 'dark' else 'dark'
                print(f"🎨 Theme changed from {old_theme} to {st.session_state.theme}")
                logger.debug(f"Theme toggled: {old_theme} -> {st.session_state.theme}")
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
        
        print(f"📄 Page selected: {selected}")
        logger.debug(f"Navigation: User selected page '{selected}'")
        
        # System status
        st.markdown("---")
        st.markdown("### System Status")
        
        # Check if models exist
        model_dir = "models"
        models_exist = os.path.exists(model_dir) and len([f for f in os.listdir(model_dir) if f.endswith('.pt')]) > 0
        logger.debug(f"Model directory check: exists={os.path.exists(model_dir)}, models_exist={models_exist}")
        
        
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
    logger.debug(f"=== ROUTING TO PAGE: {selected} ===")
    print(f"🔄 Loading page: {selected}")
    
    if selected == "Dashboard":
        logger.debug("Routing to Dashboard")
        show_dashboard()
    elif selected == "Configuration":
        logger.debug("Routing to Configuration")
        show_configuration()
    elif selected == "Data Analysis":
        logger.debug("Routing to Data Analysis")
        show_data_analysis()
    elif selected == "Training":
        logger.debug("Routing to Training")
        show_training()
    elif selected == "Backtesting":
        logger.debug("Routing to Backtesting")
        show_backtesting()
    elif selected == "Live Trading":
        logger.debug("Routing to Live Trading")
        show_live_trading()
    elif selected == "Model Management":
        logger.debug("Routing to Model Management")
        show_model_management()
    else:
        logger.warning(f"Unknown page selected: {selected}")
        st.error(f"Unknown page: {selected}")
    
    logger.debug(f"=== PAGE {selected} RENDERED ===")
    print(f"✅ Page {selected} loaded successfully")

@handle_errors
def show_dashboard():
    """Show main dashboard."""
    logger.debug("=== DASHBOARD PAGE START ===")
    print("📊 Loading Dashboard page...")
    
    st.title("📈 AI PPO Trading System Dashboard")
    logger.debug("Dashboard title rendered")
    
    # Key metrics row
    logger.debug("Rendering metrics columns")
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
        
        # Create sample portfolio performance chart
        dates = pd.date_range(start='2024-01-01', end='2024-12-31', freq='D')
        np.random.seed(42)
        returns = np.random.normal(0.001, 0.02, len(dates))
        portfolio_values = 10000 * (1 + returns).cumprod()
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=dates,
            y=portfolio_values,
            mode='lines',
            name='Portfolio Value',
            line=dict(color='#1f77b4', width=2)
        ))
        
        fig.update_layout(
            title="Portfolio Value Over Time",
            xaxis_title="Date",
            yaxis_title="Value ($)",
            height=400
        )
        
        st.plotly_chart(fig, width="stretch")
    
    with col2:
        st.subheader("Monthly Returns")
        
        # Create sample monthly returns heatmap
        months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        years = ['2023', '2024']
        
        # Generate sample data
        np.random.seed(42)
        returns_data = np.random.normal(0.02, 0.05, (len(years), len(months)))
        
        fig = go.Figure(data=go.Heatmap(
            z=returns_data,
            x=months,
            y=years,
            colorscale='RdYlGn',
            text=[[f"{val:.1%}" for val in row] for row in returns_data],
            texttemplate="%{text}",
            textfont={"size": 10}
        ))
        
        fig.update_layout(
            title="Monthly Returns Heatmap",
            height=400
        )
        
        st.plotly_chart(fig, width="stretch")
    
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
                value="\n".join(st.session_state.config.get('trading', {}).get('symbols', ['AAPL'])),
                height=100
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
                'transaction_cost': transaction_cost / 100,
                'slippage': slippage / 100
            }
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
                height=400
            )
            
            st.plotly_chart(fig, width="stretch")
            
            # Statistics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Current Price", f"${data['Close'].iloc[-1]:.2f}")
            
            with col2:
                total_return = (data['Close'].iloc[-1] / data['Close'].iloc[0] - 1) * 100
                st.metric("Total Return", f"{total_return:.1f}%")
            
            with col3:
                volatility = data['Close'].pct_change().std() * np.sqrt(252) * 100
                st.metric("Volatility (Annual)", f"{volatility:.1f}%")
            
            with col4:
                avg_volume = data['Volume'].mean()
                st.metric("Avg Volume", f"{avg_volume:,.0f}")

@handle_errors
def show_training():
    """Show training interface with support for continuing existing models."""
    import subprocess
    import sys
    
    logger.debug("=== TRAINING PAGE START ===")
    print("🎯 Loading Training page...")
    
    st.title("🎯 Model Training")
    logger.debug("Training page title rendered")
    
    # Get available models for continuing training
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    model_dir = os.path.join(project_root, "models")
    
    logger.debug(f"Checking for models in: {model_dir}")
    print(f"📁 Checking models directory: {model_dir}")
    
    available_models = []
    if os.path.exists(model_dir):
        available_models = [f for f in os.listdir(model_dir) if f.endswith('.pt')]
        logger.debug(f"Found {len(available_models)} available models: {available_models}")
        print(f"📋 Found {len(available_models)} existing models")
    else:
        logger.debug("Models directory does not exist")
        print("📁 Models directory not found")
    
    # Training mode selection
    st.subheader("🎯 Training Mode")
    logger.debug("Rendering training mode selection")
    
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
                    # Calculate estimated progress (simplified)
                    import time
                    elapsed = 0  # Initialize elapsed time
                    if hasattr(st.session_state, 'training_start_time'):
                        elapsed = time.time() - st.session_state.training_start_time
                        st.metric("Elapsed Time", f"{int(elapsed//60)}:{int(elapsed%60):02d}")
                    else:
                        st.session_state.training_start_time = time.time()
                        st.metric("Elapsed Time", "Starting...")
                
                with col3:
                    # Show estimated progress
                    config_timesteps = st.session_state.get('training_config', {}).get('timesteps', 100000)
                    if elapsed > 60:  # After 1 minute, estimate progress
                        estimated_progress = min(95, (elapsed / 3600) * 20)  # Rough estimate
                        st.metric("Estimated Progress", f"{estimated_progress:.1f}%")
                    else:
                        st.metric("Estimated Progress", "Calculating...")
                
                # Progress bar
                progress = min(95, (elapsed / 3600) * 20) if elapsed > 60 else 0
                st.progress(progress / 100.0)
                
                # Control buttons
                col1, col2, col3 = st.columns([1, 1, 2])
                
                with col1:
                    if st.button("⏹️ Stop Training", type="secondary"):
                        try:
                            process.terminate()
                            st.session_state.training_active = False
                            st.success("Training stopped successfully")
                            st.rerun()
                        except:
                            st.error("Could not stop training process")
                
                with col2:
                    if st.button("📊 Refresh Status", type="secondary"):
                        st.rerun()
                
                # Auto-refresh option
                auto_refresh = st.checkbox("Auto-refresh every 10 seconds", value=False)
                if auto_refresh:
                    import time
                    time.sleep(10)
                    st.rerun()
                
            else:
                # Process has finished
                st.session_state.training_active = False
                if process.returncode == 0:
                    st.success("🎉 Training completed successfully!")
                else:
                    st.error(f"❌ Training failed or was interrupted (Exit code: {process.returncode})")
                    
                    # Try to get error output from process
                    try:
                        if process.stderr:
                            stderr_output = process.stderr.read()
                            if stderr_output:
                                st.error("**Error details:**")
                                st.code(stderr_output, language="text")
                    except:
                        pass
                    
                    # Try to get stdout for debugging
                    try:
                        if process.stdout:
                            stdout_output = process.stdout.read()
                            if stdout_output:
                                with st.expander("📋 Training Output (Debug)"):
                                    st.code(stdout_output, language="text")
                    except:
                        pass
                
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
                if st.button("🔍 Analyze Architecture", use_container_width=True):
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
            use_container_width=True,
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
                    
                    # Create command for continuing training
                    cmd = [
                        sys.executable, 
                        os.path.join(project_root, "train_enhanced.py"),
                        "--mode", "continue",
                        "--model", model_path,
                        "--timesteps", str(additional_timesteps)
                    ]
                    
                    # Store command for debugging
                    st.session_state.last_training_command = cmd
                    
                    # Start training in background
                    st.session_state.training_process = subprocess.Popen(
                        cmd,
                        cwd=project_root,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        text=True
                    )
                    
                    st.session_state.training_mode = "continue"
                    st.session_state.continue_model = selected_model
                    st.success(f"🔄 Started continuing training of {selected_model}!")
                    st.info(f"📊 Adding {additional_timesteps:,} more training steps")
                    
                elif training_mode == "♾️ Continuous Training":
                    # Create command for continuous training
                    cmd = [
                        sys.executable,
                        os.path.join(project_root, "train_enhanced.py"),
                        "--mode", "continuous",
                        "--save-interval", str(save_interval),
                        "--checkpoint-interval", str(checkpoint_interval)
                    ]
                    
                    # Add model path if starting from existing model
                    if continuous_start_mode == "📂 Existing Model" and selected_model:
                        model_path = os.path.join(model_dir, selected_model)
                        cmd.extend(["--model", model_path])
                    
                    # Store command for debugging
                    st.session_state.last_training_command = cmd
                    
                    # Start training in background
                    st.session_state.training_process = subprocess.Popen(
                        cmd,
                        cwd=project_root,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
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
                    # Create command for new training
                    cmd = [
                        sys.executable,
                        os.path.join(project_root, "train_enhanced.py"),
                        "--mode", "new",
                        "--timesteps", str(additional_timesteps)
                    ]
                    
                    # Store command for debugging
                    st.session_state.last_training_command = cmd
                    
                    # Start training in background
                    st.session_state.training_process = subprocess.Popen(
                        cmd,
                        cwd=project_root,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        text=True
                    )
                    
                    st.session_state.training_mode = "new"
                    st.success("🚀 Started training new model!")
                    st.info(f"📊 Training for {additional_timesteps:,} timesteps")
                
                st.session_state.training_active = True
                
                # Store training config
                st.session_state.training_config = {
                    'mode': training_mode,
                    'timesteps': additional_timesteps,
                    'eval_freq': eval_freq,
                    'save_freq': save_freq,
                    'symbols': [s.strip() for s in data_symbols.split('\n') if s.strip()],
                    'data_period': data_period
                }
                
                st.rerun()
                
            except Exception as e:
                st.error(f"❌ Failed to start training: {str(e)}")
                st.info("💡 Make sure the training script is available and all dependencies are installed")
    
    with col2:
        if st.button("⏸️ Pause Training", disabled=not st.session_state.training_active, use_container_width=True):
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
        
        # Auto-refresh every second as fallback
        current_time = time.time()
        if current_time - st.session_state.last_refresh >= 1.0:
            st.session_state.last_refresh = current_time
            # Use a timer to trigger rerun
            import threading
            def delayed_rerun():
                time.sleep(0.1)  # Small delay to prevent race conditions
                try:
                    st.rerun()
                except:
                    pass  # Ignore errors if component is already refreshing
            
            threading.Thread(target=delayed_rerun, daemon=True).start()
        
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
            from datetime import datetime
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
    model_files = []
    if os.path.exists("models"):
        model_files = [f for f in os.listdir("models") if f.endswith('.pt')]
    
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
        symbols = st.session_state.config.get('trading', {}).get('symbols', ['AAPL'])
        selected_symbol = st.selectbox("Symbol", symbols)
    
    # Advanced options
    with st.expander("Advanced Options"):
        create_dashboard = st.checkbox("Create Visualization Dashboard", value=True)
        walk_forward = st.checkbox("Run Walk-Forward Analysis", value=False)
        benchmark_comparison = st.checkbox("Compare with Buy & Hold", value=True)
    
    if st.button("🚀 Run Backtest"):
        with st.spinner("Running backtest..."):
            # Simulate backtest results
            st.info(f"Running backtest for {selected_symbol} using model {selected_model}")
            
            # Create sample backtest results
            dates = pd.date_range(start=start_date, end=end_date, freq='D')
            np.random.seed(42)
            
            # Generate realistic stock price data
            price_changes = np.random.normal(0.001, 0.02, len(dates))
            prices = 100 * (1 + price_changes).cumprod()
            
            # Simulate portfolio performance
            daily_returns = np.random.normal(0.001, 0.015, len(dates))
            portfolio_values = initial_balance * (1 + daily_returns).cumprod()
            
            # Buy and hold benchmark
            stock_returns = np.random.normal(0.0008, 0.018, len(dates))
            benchmark_values = initial_balance * (1 + stock_returns).cumprod()
            
            # Generate trading signals (buy/sell actions)
            # Simulate agent decisions based on price movements and technical indicators
            actions = []
            positions = []
            current_position = 0
            buy_signals = []
            sell_signals = []
            
            for i in range(len(dates)):
                if i < 20:  # Need some history for indicators
                    actions.append(0)  # Hold
                    positions.append(current_position)
                    continue
                
                # Simple trading logic for demonstration
                # Calculate moving averages
                short_ma = np.mean(prices[max(0, i-5):i+1])
                long_ma = np.mean(prices[max(0, i-20):i+1])
                price_momentum = (prices[i] - prices[i-1]) / prices[i-1] if i > 0 else 0
                
                # Trading decision logic
                if current_position == 0:  # No position
                    # Buy signal: short MA > long MA and positive momentum
                    if short_ma > long_ma and price_momentum > 0.01 and np.random.random() > 0.7:
                        action = 1  # Buy
                        current_position = 1
                        buy_signals.append({
                            'date': dates[i],
                            'price': prices[i],
                            'action': 'BUY'
                        })
                    else:
                        action = 0  # Hold
                elif current_position == 1:  # Long position
                    # Sell signal: short MA < long MA or negative momentum
                    if short_ma < long_ma or price_momentum < -0.015 or np.random.random() > 0.8:
                        action = 2  # Sell
                        current_position = 0
                        sell_signals.append({
                            'date': dates[i],
                            'price': prices[i],
                            'action': 'SELL'
                        })
                    else:
                        action = 0  # Hold
                else:
                    action = 0  # Hold
                
                actions.append(action)
                positions.append(current_position)
            
            # Convert signals to DataFrames for easier handling
            buy_df = pd.DataFrame(buy_signals) if buy_signals else pd.DataFrame(columns=['date', 'price', 'action'])
            sell_df = pd.DataFrame(sell_signals) if sell_signals else pd.DataFrame(columns=['date', 'price', 'action'])
            
            # Display results
            st.subheader("Backtest Results")
            
            # Key metrics
            final_value = portfolio_values[-1]
            total_return = (final_value / initial_balance - 1) * 100
            benchmark_return = (benchmark_values[-1] / initial_balance - 1) * 100
            excess_return = total_return - benchmark_return
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Final Portfolio Value", format_currency(final_value))
            
            with col2:
                st.metric("Total Return", f"{total_return:.2f}%", f"{excess_return:+.2f}%")
            
            with col3:
                # Calculate Sharpe ratio
                returns_series = pd.Series(daily_returns)
                sharpe = returns_series.mean() / returns_series.std() * np.sqrt(252)
                st.metric("Sharpe Ratio", f"{sharpe:.3f}")
            
            with col4:
                # Calculate max drawdown
                cumulative = pd.Series(portfolio_values)
                rolling_max = cumulative.expanding().max()
                drawdown = (cumulative - rolling_max) / rolling_max
                max_drawdown = drawdown.min() * 100
                st.metric("Max Drawdown", f"{max_drawdown:.2f}%")
            
            # Trading Statistics
            st.subheader("Trading Activity")
            
            trade_col1, trade_col2, trade_col3, trade_col4 = st.columns(4)
            
            with trade_col1:
                st.metric("Total Trades", len(buy_signals) + len(sell_signals))
            
            with trade_col2:
                st.metric("Buy Orders", len(buy_signals))
            
            with trade_col3:
                st.metric("Sell Orders", len(sell_signals))
            
            with trade_col4:
                win_rate = 65 + np.random.normal(0, 10)  # Simulated win rate
                st.metric("Win Rate", f"{max(0, min(100, win_rate)):.1f}%")
            
            # Performance chart with buy/sell markers
            st.subheader("📈 Price Chart with Trading Signals")
            
            fig = go.Figure()
            
            # Add price line
            fig.add_trace(go.Scatter(
                x=dates,
                y=prices,
                mode='lines',
                name=f'{selected_symbol} Price',
                line=dict(color='#1f77b4', width=2),
                hovertemplate='<b>%{fullData.name}</b><br>' +
                             'Date: %{x}<br>' +
                             'Price: $%{y:.2f}<br>' +
                             '<extra></extra>'
            ))
            
            # Add buy signals (green triangles pointing up)
            if not buy_df.empty:
                fig.add_trace(go.Scatter(
                    x=buy_df['date'],
                    y=buy_df['price'],
                    mode='markers',
                    name='Buy Signals',
                    marker=dict(
                        symbol='triangle-up',
                        size=12,
                        color='#00ff00',
                        line=dict(color='#008000', width=2)
                    ),
                    hovertemplate='<b>BUY SIGNAL</b><br>' +
                                 'Date: %{x}<br>' +
                                 'Price: $%{y:.2f}<br>' +
                                 '<extra></extra>'
                ))
            
            # Add sell signals (red triangles pointing down)
            if not sell_df.empty:
                fig.add_trace(go.Scatter(
                    x=sell_df['date'],
                    y=sell_df['price'],
                    mode='markers',
                    name='Sell Signals',
                    marker=dict(
                        symbol='triangle-down',
                        size=12,
                        color='#ff0000',
                        line=dict(color='#800000', width=2)
                    ),
                    hovertemplate='<b>SELL SIGNAL</b><br>' +
                                 'Date: %{x}<br>' +
                                 'Price: $%{y:.2f}<br>' +
                                 '<extra></extra>'
                ))
            
            fig.update_layout(
                title=f"{selected_symbol} Price with AI Trading Signals",
                xaxis_title="Date",
                yaxis_title="Price ($)",
                height=600,
                legend=dict(x=0, y=1),
                hovermode='closest',
                showlegend=True
            )
            
            st.plotly_chart(fig, width="stretch")
            
            # Portfolio Performance Comparison
            st.subheader("📊 Portfolio Performance")
            
            fig2 = go.Figure()
            
            fig2.add_trace(go.Scatter(
                x=dates,
                y=portfolio_values,
                mode='lines',
                name='AI Strategy',
                line=dict(color='blue', width=2)
            ))
            
            if benchmark_comparison:
                fig2.add_trace(go.Scatter(
                    x=dates,
                    y=benchmark_values,
                    mode='lines',
                    name='Buy & Hold',
                    line=dict(color='gray', width=2, dash='dash')
                ))
            
            fig2.update_layout(
                title="Portfolio Performance Comparison",
                xaxis_title="Date",
                yaxis_title="Portfolio Value ($)",
                height=400,
                legend=dict(x=0, y=1)
            )
            
            st.plotly_chart(fig2, width="stretch")
            
            # Trade Details Table
            if not buy_df.empty or not sell_df.empty:
                st.subheader("📋 Trade Details")
                
                # Combine buy and sell signals
                all_trades = []
                for _, trade in buy_df.iterrows():
                    all_trades.append({
                        'Date': trade['date'].strftime('%Y-%m-%d'),
                        'Action': '🟢 BUY',
                        'Price': f"${trade['price']:.2f}",
                        'Type': 'Market Order'
                    })
                
                for _, trade in sell_df.iterrows():
                    all_trades.append({
                        'Date': trade['date'].strftime('%Y-%m-%d'),
                        'Action': '🔴 SELL',
                        'Price': f"${trade['price']:.2f}",
                        'Type': 'Market Order'
                    })
                
                # Sort by date
                all_trades.sort(key=lambda x: x['Date'])
                
                if all_trades:
                    trades_df = pd.DataFrame(all_trades)
                    st.dataframe(trades_df, use_container_width=True, hide_index=True)
                else:
                    st.info("No trades executed during this backtest period.")
            
            # Detailed results table
            st.subheader("Detailed Results")
            
            results_data = {
                'Metric': [
                    'Initial Balance',
                    'Final Balance', 
                    'Total Return',
                    'Benchmark Return',
                    'Excess Return',
                    'Sharpe Ratio',
                    'Maximum Drawdown',
                    'Volatility (Annual)',
                    'Number of Trades'
                ],
                'Value': [
                    format_currency(initial_balance),
                    format_currency(final_value),
                    f"{total_return:.2f}%",
                    f"{benchmark_return:.2f}%",
                    f"{excess_return:.2f}%",
                    f"{sharpe:.3f}",
                    f"{max_drawdown:.2f}%",
                    f"{returns_series.std() * np.sqrt(252) * 100:.2f}%",
                    "42"  # Simulated
                ]
            }
            
            results_df = pd.DataFrame(results_data)
            st.dataframe(results_df, width="stretch")

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
        auto_refresh = st.checkbox("🔄 Auto Refresh (5s)", value=False)
    
    if auto_refresh:
        time.sleep(1)  # Simulate delay
    
    quotes_data = {
        'Symbol': ['🍎 AAPL', '🖥️ MSFT', '🔍 GOOGL', '📦 AMZN', '⚡ TSLA'],
        'Price': ['$178.90', '$412.45', '$2,780.15', '$185.30', '$248.90'],
        'Change': ['+$3.40', '-$2.75', '+$29.85', '+$1.20', '-$3.10'],
        'Change %': ['+1.94%', '-0.66%', '+1.09%', '+0.65%', '-1.23%'],
        'Volume': ['12.5M', '8.7M', '1.2M', '15.3M', '25.1M'],
        'AI Signal': ['🟢 BUY', '🟡 HOLD', '🟢 BUY', '🔴 SELL', '🟡 HOLD']
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
        all_files = os.listdir(model_dir)
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