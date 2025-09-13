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

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from streamlit_option_menu import option_menu
from st_aggrid import AgGrid, GridOptionsBuilder

# Import trading system components
try:
    from data import DataClient, prepare_features
    from environments import TradingEnvironment
    from agents import PPOAgent
    from evaluation.backtesting import Backtester
    from utils import ConfigManager, create_default_config, format_currency, format_percentage
except ImportError as e:
    st.error(f"Error importing trading system components: {e}")
    st.stop()

# Page configuration
st.set_page_config(
    page_title="AI PPO Trading System",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main > div {
        padding-top: 2rem;
    }
    .stMetric {
        background-color: #f0f2f6;
        border: 1px solid #e6e9ef;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .success-box {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        color: #155724;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .warning-box {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        color: #856404;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .error-box {
        background-color: #f8d7da;
        border: 1px solid #f5c6cb;
        color: #721c24;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'config' not in st.session_state:
    st.session_state.config = create_default_config()
if 'training_active' not in st.session_state:
    st.session_state.training_active = False
if 'training_metrics' not in st.session_state:
    st.session_state.training_metrics = []

def main():
    """Main application function."""
    
    # Sidebar navigation
    with st.sidebar:
        st.image("https://via.placeholder.com/200x80/1f77b4/ffffff?text=AI+PPO+Trading", width=200)
        
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
        models_exist = os.path.exists(model_dir) and len([f for f in os.listdir(model_dir) if f.endswith('.pt')]) > 0
        
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

def show_dashboard():
    """Show main dashboard."""
    st.title("📈 AI PPO Trading System Dashboard")
    
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
        
        st.plotly_chart(fig, use_container_width=True)
    
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
        
        st.plotly_chart(fig, use_container_width=True)
    
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
    st.dataframe(activity_df, use_container_width=True)

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
        if st.button("💾 Save All to File", use_container_width=True):
            try:
                config_manager = ConfigManager()
                config_manager.config = st.session_state.config
                
                os.makedirs("config", exist_ok=True)
                config_manager.save_config("config/config.yaml")
                
                st.success("Configuration saved to config/config.yaml")
            except Exception as e:
                st.error(f"Error saving configuration: {e}")
    
    with col2:
        if st.button("📂 Load from File", use_container_width=True):
            try:
                if os.path.exists("config/config.yaml"):
                    config_manager = ConfigManager("config/config.yaml")
                    st.session_state.config = config_manager.to_dict()
                    st.success("Configuration loaded from file")
                    st.experimental_rerun()
                else:
                    st.warning("No configuration file found")
            except Exception as e:
                st.error(f"Error loading configuration: {e}")
    
    with col3:
        if st.button("🔄 Reset to Defaults", use_container_width=True):
            st.session_state.config = create_default_config()
            st.success("Configuration reset to defaults")
            st.experimental_rerun()

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
            
            st.plotly_chart(fig, use_container_width=True)
            
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

def show_training():
    """Show training interface."""
    st.title("🎯 Model Training")
    
    # Training status
    if st.session_state.training_active:
        st.markdown("""
        <div class="success-box">
            <strong>Training in Progress!</strong><br>
            The model is currently being trained. Monitor the progress below.
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="warning-box">
            <strong>Ready to Train</strong><br>
            Configure your training parameters and start training a new model.
        </div>
        """, unsafe_allow_html=True)
    
    # Training configuration
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Training Parameters")
        
        total_timesteps = st.number_input(
            "Total Timesteps",
            value=100000,  # Reduced for demo
            min_value=10000,
            max_value=10000000,
            step=10000
        )
        
        eval_freq = st.number_input(
            "Evaluation Frequency",
            value=5000,
            min_value=1000,
            max_value=50000,
            step=1000
        )
        
        save_freq = st.number_input(
            "Save Frequency",
            value=10000,
            min_value=5000,
            max_value=100000,
            step=5000
        )
    
    with col2:
        st.subheader("Model Configuration")
        
        policy_layers = st.text_input(
            "Policy Network Layers",
            value="256,256",
            help="Comma-separated layer sizes"
        )
        
        value_layers = st.text_input(
            "Value Network Layers", 
            value="256,256",
            help="Comma-separated layer sizes"
        )
        
        activation = st.selectbox(
            "Activation Function",
            options=['tanh', 'relu', 'leaky_relu'],
            index=0
        )
    
    # Training controls
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🚀 Start Training", disabled=st.session_state.training_active, use_container_width=True):
            st.session_state.training_active = True
            st.success("Training started! (Simulated)")
            st.experimental_rerun()
    
    with col2:
        if st.button("⏸️ Pause Training", disabled=not st.session_state.training_active, use_container_width=True):
            st.session_state.training_active = False
            st.info("Training paused")
            st.experimental_rerun()
    
    with col3:
        if st.button("🛑 Stop Training", disabled=not st.session_state.training_active, use_container_width=True):
            st.session_state.training_active = False
            st.warning("Training stopped")
            st.experimental_rerun()
    
    # Training metrics (simulated)
    if st.session_state.training_active or st.session_state.training_metrics:
        st.subheader("Training Metrics")
        
        # Create sample training metrics
        if st.session_state.training_active:
            # Simulate new metrics
            new_metric = {
                'step': len(st.session_state.training_metrics) + 1,
                'reward': np.random.normal(0.1, 0.05),
                'policy_loss': np.random.normal(0.01, 0.005),
                'value_loss': np.random.normal(0.05, 0.01),
                'entropy': np.random.normal(0.8, 0.1)
            }
            st.session_state.training_metrics.append(new_metric)
        
        if st.session_state.training_metrics:
            metrics_df = pd.DataFrame(st.session_state.training_metrics)
            
            # Plot training metrics
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=metrics_df['step'],
                y=metrics_df['reward'],
                mode='lines',
                name='Average Reward',
                line=dict(color='green')
            ))
            
            fig.update_layout(
                title="Training Progress",
                xaxis_title="Training Step",
                yaxis_title="Average Reward",
                height=300
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Current metrics
            if len(metrics_df) > 0:
                latest = metrics_df.iloc[-1]
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Average Reward", f"{latest['reward']:.4f}")
                
                with col2:
                    st.metric("Policy Loss", f"{latest['policy_loss']:.4f}")
                
                with col3:
                    st.metric("Value Loss", f"{latest['value_loss']:.4f}")
                
                with col4:
                    st.metric("Entropy", f"{latest['entropy']:.3f}")

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
            
            # Simulate portfolio performance
            daily_returns = np.random.normal(0.001, 0.015, len(dates))
            portfolio_values = initial_balance * (1 + daily_returns).cumprod()
            
            # Buy and hold benchmark
            stock_returns = np.random.normal(0.0008, 0.018, len(dates))
            benchmark_values = initial_balance * (1 + stock_returns).cumprod()
            
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
            
            # Performance chart
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=dates,
                y=portfolio_values,
                mode='lines',
                name='AI Strategy',
                line=dict(color='blue', width=2)
            ))
            
            if benchmark_comparison:
                fig.add_trace(go.Scatter(
                    x=dates,
                    y=benchmark_values,
                    mode='lines',
                    name='Buy & Hold',
                    line=dict(color='gray', width=2, dash='dash')
                ))
            
            fig.update_layout(
                title="Backtest Performance Comparison",
                xaxis_title="Date",
                yaxis_title="Portfolio Value ($)",
                height=500,
                legend=dict(x=0, y=1)
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
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
            st.dataframe(results_df, use_container_width=True)

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
    
    # Portfolio overview
    st.subheader("Portfolio Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Portfolio Value", "$10,234.56", "+1.2%")
    
    with col2:
        st.metric("Cash Balance", "$2,456.78", "-0.5%")
    
    with col3:
        st.metric("Today's P&L", "+$123.45", "+1.21%")
    
    with col4:
        st.metric("Open Positions", "3", "+1")
    
    # Current positions
    st.subheader("Current Positions")
    
    positions_data = {
        'Symbol': ['AAPL', 'MSFT', 'GOOGL'],
        'Shares': [10, 15, 5],
        'Avg Price': [175.50, 415.20, 2750.30],
        'Current Price': [178.90, 412.45, 2780.15],
        'Market Value': [1789.00, 6186.75, 13900.75],
        'P&L': ['+$34.00', '-$41.25', '+$149.25'],
        'P&L %': ['+1.94%', '-0.66%', '+0.54%']
    }
    
    positions_df = pd.DataFrame(positions_data)
    st.dataframe(positions_df, use_container_width=True)
    
    # Real-time quotes (simulated)
    st.subheader("Real-Time Market Data")
    
    # Auto-refresh
    auto_refresh = st.checkbox("Auto Refresh (5s)", value=False)
    
    if auto_refresh:
        time.sleep(1)  # Simulate delay
    
    quotes_data = {
        'Symbol': ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA'],
        'Price': [178.90, 412.45, 2780.15, 185.30, 248.90],
        'Change': ['+3.40', '-2.75', '+29.85', '+1.20', '-3.10'],
        'Change %': ['+1.94%', '-0.66%', '+1.09%', '+0.65%', '-1.23%'],
        'Volume': ['12.5M', '8.7M', '1.2M', '15.3M', '25.1M'],
        'AI Signal': ['BUY', 'HOLD', 'BUY', 'SELL', 'HOLD']
    }
    
    quotes_df = pd.DataFrame(quotes_data)
    
    # Color code the signals
    def color_signal(val):
        if val == 'BUY':
            return 'color: green; font-weight: bold'
        elif val == 'SELL':
            return 'color: red; font-weight: bold'
        else:
            return 'color: gray'
    
    styled_df = quotes_df.style.applymap(color_signal, subset=['AI Signal'])
    st.dataframe(styled_df, use_container_width=True)
    
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

def show_model_management():
    """Show model management interface."""
    st.title("📁 Model Management")
    
    # Model directory status
    model_dir = "models"
    if not os.path.exists(model_dir):
        st.warning("Models directory not found. No trained models available.")
        return
    
    model_files = [f for f in os.listdir(model_dir) if f.endswith('.pt')]
    
    if not model_files:
        st.info("No trained models found. Train a model first using the Training page.")
        return
    
    st.success(f"Found {len(model_files)} trained models")
    
    # Model list
    st.subheader("Available Models")
    
    model_data = []
    for model_file in model_files:
        model_path = os.path.join(model_dir, model_file)
        stat = os.stat(model_path)
        
        model_data.append({
            'Model Name': model_file,
            'Size (MB)': f"{stat.st_size / (1024*1024):.2f}",
            'Created': datetime.fromtimestamp(stat.st_ctime).strftime("%Y-%m-%d %H:%M"),
            'Modified': datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M")
        })
    
    models_df = pd.DataFrame(model_data)
    st.dataframe(models_df, use_container_width=True)
    
    # Model actions
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Model Actions")
        selected_model = st.selectbox("Select Model", model_files)
        
        col1_1, col1_2 = st.columns(2)
        
        with col1_1:
            if st.button("📊 Analyze Model", use_container_width=True):
                st.info(f"Analyzing {selected_model}...")
                # This would normally load and analyze the model
                st.success("Model analysis complete (simulated)")
        
        with col1_2:
            if st.button("🗑️ Delete Model", use_container_width=True):
                if st.checkbox(f"Confirm deletion of {selected_model}"):
                    st.warning(f"Would delete {selected_model} (simulated)")
    
    with col2:
        st.subheader("Model Comparison")
        
        if len(model_files) >= 2:
            model1 = st.selectbox("Model 1", model_files, key="model1")
            model2 = st.selectbox("Model 2", model_files, key="model2", index=1)
            
            if st.button("📈 Compare Models", use_container_width=True):
                st.info(f"Comparing {model1} vs {model2}...")
                
                # Simulated comparison results
                comparison_data = {
                    'Metric': ['Total Return', 'Sharpe Ratio', 'Max Drawdown', 'Win Rate'],
                    model1: ['12.4%', '1.85', '-8.2%', '67%'],
                    model2: ['10.8%', '1.92', '-6.5%', '71%']
                }
                
                comparison_df = pd.DataFrame(comparison_data)
                st.dataframe(comparison_df, use_container_width=True)
        else:
            st.info("Need at least 2 models for comparison")
    
    # Export/Import
    st.subheader("Export/Import")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("📤 Export Selected Model", use_container_width=True):
            st.info(f"Exporting {selected_model}...")
            st.success("Model exported successfully (simulated)")
    
    with col2:
        uploaded_file = st.file_uploader("📥 Import Model", type=['pt'])
        if uploaded_file is not None:
            st.info(f"Would import {uploaded_file.name}")

if __name__ == "__main__":
    main()