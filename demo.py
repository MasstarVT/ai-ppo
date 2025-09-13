#!/usr/bin/env python3
"""
Demo script showing how to use the AI PPO Trading System.
This script demonstrates the basic workflow from data loading to model training and evaluation.
"""

import os
import sys
import logging

# Add src to path
sys.path.append('src')

from data import DataClient, prepare_features
from environments import TradingEnvironment
from agents import PPOAgent
from evaluation.backtesting import Backtester
from visualization.plots import TradingVisualizer
from utils import ConfigManager, setup_logging, create_default_config

def demo_data_loading():
    """Demonstrate data loading and feature preparation."""
    print("\n" + "="*50)
    print("DEMO 1: Data Loading and Feature Preparation")
    print("="*50)
    
    # Create default config
    config = create_default_config()
    
    # Initialize data client
    data_client = DataClient(config)
    
    # Fetch sample data
    print("Fetching sample data for AAPL...")
    data = data_client.get_historical_data('AAPL', period='6mo', interval='1d')
    
    if not data.empty:
        print(f"✓ Fetched {len(data)} data points")
        print(f"  Date range: {data.index[0]} to {data.index[-1]}")
        print(f"  Columns: {list(data.columns)}")
        
        # Prepare features
        print("\nPreparing features...")
        features = prepare_features(data, config)
        
        if not features.empty:
            print(f"✓ Prepared {len(features)} feature points")
            print(f"  Feature columns: {len(features.columns)}")
            print(f"  Sample features: {list(features.columns[:10])}")
        else:
            print("✗ No features prepared")
    else:
        print("✗ No data fetched")

def demo_environment():
    """Demonstrate trading environment."""
    print("\n" + "="*50)
    print("DEMO 2: Trading Environment")
    print("="*50)
    
    # Create config and fetch data
    config = create_default_config()
    data_client = DataClient(config)
    
    print("Setting up trading environment...")
    data = data_client.get_historical_data('AAPL', period='3mo', interval='1d')
    
    if data.empty:
        print("✗ No data available for environment demo")
        return
    
    features = prepare_features(data, config)
    
    if features.empty:
        print("✗ No features available for environment demo")
        return
    
    # Create environment
    env = TradingEnvironment(features, config)
    
    print(f"✓ Environment created")
    print(f"  Observation space: {env.observation_space.shape}")
    print(f"  Action space: {env.action_space.n} actions")
    print(f"  Initial balance: ${env.initial_balance:,.2f}")
    
    # Run a few random steps
    print("\nRunning random actions...")
    obs = env.reset()
    
    for step in range(5):
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)
        
        action_names = ['SELL', 'HOLD', 'BUY']
        print(f"  Step {step+1}: {action_names[action]} -> Reward: {reward:.4f}, "
              f"Portfolio: ${info['portfolio_value']:.2f}")
        
        if done:
            break

def demo_agent():
    """Demonstrate PPO agent."""
    print("\n" + "="*50)
    print("DEMO 3: PPO Agent")
    print("="*50)
    
    config = create_default_config()
    
    # Create dummy environment for agent dimensions
    obs_dim = 1000  # Example observation dimension
    action_dim = 3  # Sell, Hold, Buy
    
    print("Creating PPO agent...")
    agent = PPOAgent(obs_dim, action_dim, config)
    
    print(f"✓ Agent created")
    print(f"  Observation dimension: {obs_dim}")
    print(f"  Action dimension: {action_dim}")
    print(f"  Learning rate: {agent.lr}")
    print(f"  Network layers: {config['network']['policy_layers']}")
    
    # Test action prediction
    dummy_obs = np.zeros(obs_dim)
    action, log_prob, value = agent.get_action(dummy_obs)
    
    print(f"\nSample prediction:")
    print(f"  Action: {action}")
    print(f"  Log probability: {log_prob:.4f}")
    print(f"  Value estimate: {value:.4f}")

def demo_backtesting():
    """Demonstrate backtesting functionality."""
    print("\n" + "="*50)
    print("DEMO 4: Backtesting (Simplified)")
    print("="*50)
    
    config = create_default_config()
    
    print("Note: This is a simplified demo using random actions.")
    print("In practice, you would use a trained model.")
    
    # Create backtester
    backtester = Backtester(config)
    
    print("✓ Backtester initialized")
    print(f"  Configuration loaded")
    print(f"  Ready for backtesting with trained models")

def demo_config_system():
    """Demonstrate configuration system."""
    print("\n" + "="*50)
    print("DEMO 5: Configuration System")
    print("="*50)
    
    print("Creating configuration manager...")
    
    # Create default config
    config = create_default_config()
    
    # Save to file
    config_manager = ConfigManager()
    config_manager.config = config
    config_manager.save_config('demo_config.yaml')
    
    print("✓ Configuration saved to demo_config.yaml")
    
    # Load from file
    loaded_config = ConfigManager('demo_config.yaml')
    
    print("✓ Configuration loaded from file")
    print(f"  Trading symbols: {loaded_config.get('trading.symbols')}")
    print(f"  Initial balance: ${loaded_config.get('trading.initial_balance'):,}")
    print(f"  PPO learning rate: {loaded_config.get('ppo.learning_rate')}")
    
    # Clean up
    if os.path.exists('demo_config.yaml'):
        os.remove('demo_config.yaml')
        print("  Cleaned up demo config file")

def print_usage_instructions():
    """Print usage instructions."""
    print("\n" + "="*60)
    print("HOW TO USE THE AI PPO TRADING SYSTEM")
    print("="*60)
    
    print("""
1. SETUP (one-time):
   python setup.py
   
2. CONFIGURE:
   Edit config/config.yaml with your preferences:
   - Trading symbols (stocks to trade)
   - API credentials (if using paid data providers)
   - PPO hyperparameters
   - Risk management settings

3. TRAIN A MODEL:
   python src/train.py --config config/config.yaml
   
4. BACKTEST THE MODEL:
   python src/backtest.py --config config/config.yaml --model models/best_model.pt --create-dashboard
   
5. ANALYZE RESULTS:
   Check the backtest_results/ directory for:
   - Performance charts
   - Trading activity analysis
   - Risk metrics
   - Detailed trade logs

IMPORTANT WARNINGS:
⚠️  This is for educational purposes only
⚠️  Always test with paper trading first
⚠️  Never risk more than you can afford to lose
⚠️  Past performance ≠ future results
⚠️  Use proper risk management

ADVANCED FEATURES:
- Walk-forward analysis for strategy validation
- Multi-symbol portfolio optimization
- Custom reward functions
- Real-time data integration
- Risk-adjusted performance metrics
""")

def main():
    """Run all demos."""
    print("AI PPO Trading System - Demo")
    print("="*40)
    print("This demo shows the key components of the trading system.")
    print("Note: Some demos use simulated data due to API limitations.")
    
    try:
        # Import numpy for agent demo
        import numpy as np
        globals()['np'] = np
        
        # Run demos
        demo_config_system()
        demo_data_loading()
        demo_environment()
        demo_agent()
        demo_backtesting()
        
        # Print usage instructions
        print_usage_instructions()
        
        print("\n🎉 Demo completed successfully!")
        print("The system is ready for training and backtesting.")
        
    except ImportError as e:
        print(f"\n⚠️  Demo requires additional packages: {e}")
        print("Please run: pip install -r requirements.txt")
    except Exception as e:
        print(f"\n✗ Demo failed: {e}")
        print("Please check your setup and try again.")

if __name__ == "__main__":
    main()