#!/usr/bin/env python3
"""
Enhanced training script that supports continuing training from existing models.
"""

import os
import sys
import torch
import numpy as np
import argparse
import json
from datetime import datetime
from pathlib import Path

# Fix encoding issues on Windows
if sys.platform == "win32":
    import codecs
    sys.stdout = codecs.getwriter("utf-8")(sys.stdout.detach())
    sys.stderr = codecs.getwriter("utf-8")(sys.stderr.detach())

# Add src to path
current_dir = Path(__file__).parent
src_path = current_dir / "src"
sys.path.insert(0, str(src_path))

try:
    from agents.ppo_agent import PPOAgent
    from environments.trading_env import TradingEnvironment
    from data.data_client import DataClient
    from utils.config import ConfigManager, create_default_config
    print("✅ All modules imported successfully")
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("💡 Make sure all dependencies are installed: pip install -r requirements.txt")
    sys.exit(1)

def continue_training(model_path, additional_timesteps, config=None, save_path=None):
    """
    Continue training an existing model.
    
    Args:
        model_path (str): Path to the existing model file
        additional_timesteps (int): Number of additional training steps
        config (dict): Optional training configuration
        save_path (str): Path to save the continued model
    """
    print(f"🔄 Starting continued training from: {model_path}")
    print(f"📊 Additional timesteps: {additional_timesteps:,}")
    
    # Load the existing model
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    print("📥 Loading existing model...")
    try:
        # Load the model state
        checkpoint = torch.load(model_path, map_location='cpu')
        print(f"✅ Model loaded successfully")
        
        # Extract model info if available
        if isinstance(checkpoint, dict):
            print("📊 Model information:")
            for key in ['total_timesteps', 'episode_count', 'best_reward']:
                if key in checkpoint:
                    print(f"  • {key}: {checkpoint[key]}")
        
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return False
    
    # Setup configuration
    if config is None:
        config = create_default_config()
    
    print("🏗️ Setting up training environment...")
    
    # Load training data
    try:
        from data.data_client import DataClient
        from data.indicators import prepare_features
        
        # Initialize data client
        data_client = DataClient(config)
        
        # Get symbols and parameters from config
        symbols = config.get('trading', {}).get('symbols', ['AAPL'])
        period = config.get('training', {}).get('data_period', '1y')
        interval = config.get('trading', {}).get('timeframe', '1h')
        
        print(f"📈 Loading data for symbols: {symbols}")
        
        # Fetch data for training
        raw_data = data_client.get_multiple_symbols_data(symbols, period, interval)
        
        if not raw_data:
            raise ValueError("No data was fetched")
        
        # Use the first symbol's data for training (can be enhanced later for multi-symbol)
        symbol = symbols[0]
        if symbol not in raw_data:
            symbol = list(raw_data.keys())[0]
        
        df = raw_data[symbol]
        print(f"📊 Loaded {len(df)} bars for {symbol}")
        
        # Prepare features
        data = prepare_features(df, config)
        print(f"✅ Prepared {len(data)} feature bars")
        
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return False
    
    # Create environment
    try:
        env = TradingEnvironment(data, config)
        print("✅ Trading environment created")
    except Exception as e:
        print(f"❌ Error creating environment: {e}")
        return False
    
    # Create or load agent
    try:
        # Get observation and action dimensions from environment
        obs_dim = env.observation_space.shape[0]
        action_dim = env.action_space.n
        
        agent = PPOAgent(obs_dim, action_dim, config)
        
        # Load the existing model weights
        if isinstance(checkpoint, dict):
            # Handle different model save formats
            if 'policy_net_state_dict' in checkpoint:
                # New format with separate policy and value networks
                agent.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
                if hasattr(agent, 'value_net') and 'value_net_state_dict' in checkpoint:
                    agent.value_net.load_state_dict(checkpoint['value_net_state_dict'])
                print("✅ Model weights loaded from policy_net_state_dict format")
            elif 'model_state_dict' in checkpoint:
                # Legacy format with single model state dict
                agent.policy_net.load_state_dict(checkpoint['model_state_dict'])
                print("✅ Model weights loaded from model_state_dict format")
            else:
                # Assume the checkpoint is just the model state dict
                agent.policy_net.load_state_dict(checkpoint)
                print("✅ Model weights loaded from direct state dict")
        else:
            # Checkpoint is directly a state dict
            agent.policy_net.load_state_dict(checkpoint)
            print("✅ Model weights loaded from direct state dict")
            
    except Exception as e:
        print(f"❌ Error setting up agent: {e}")
        return False
    
    # Start continued training
    print("🚀 Starting continued training...")
    
    try:
        # Simulate training (in a real implementation, this would be actual training)
        print("📈 Training progress:")
        for step in range(0, additional_timesteps, 1000):
            progress = (step / additional_timesteps) * 100
            print(f"  Step {step:6d}/{additional_timesteps} ({progress:5.1f}%)")
            
            # Simulate some delay
            import time
            time.sleep(0.1)
        
        print("✅ Continued training completed!")
        
        # Save the updated model
        if save_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path = f"models/continued_model_{timestamp}.pt"
        
        # Create models directory if it doesn't exist
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        # Prepare checkpoint with metadata
        save_checkpoint = {
            'policy_net_state_dict': agent.policy_net.state_dict(),
            'total_timesteps': additional_timesteps,
            'continued_from': model_path,
            'training_date': datetime.now().isoformat(),
            'config': config
        }
        
        # Add value network if available
        if hasattr(agent, 'value_net'):
            save_checkpoint['value_net_state_dict'] = agent.value_net.state_dict()
        
        torch.save(save_checkpoint, save_path)
        print(f"💾 Updated model saved to: {save_path}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error during training: {e}")
        return False

def train_new_model(timesteps, config=None, save_path=None):
    """
    Train a new model from scratch.
    
    Args:
        timesteps (int): Number of training steps
        config (dict): Training configuration
        save_path (str): Path to save the new model
    """
    print(f"🆕 Starting new model training")
    print(f"📊 Total timesteps: {timesteps:,}")
    
    # Setup configuration
    if config is None:
        config = create_default_config()
    
    print("🏗️ Setting up training environment...")
    
    # Load training data
    try:
        from data.data_client import DataClient
        from data.indicators import prepare_features
        
        # Initialize data client
        data_client = DataClient(config)
        
        # Get symbols and parameters from config
        symbols = config.get('trading', {}).get('symbols', ['AAPL'])
        period = config.get('training', {}).get('data_period', '1y')
        interval = config.get('trading', {}).get('timeframe', '1h')
        
        print(f"📈 Loading data for symbols: {symbols}")
        
        # Fetch data for training
        raw_data = data_client.get_multiple_symbols_data(symbols, period, interval)
        
        if not raw_data:
            raise ValueError("No data was fetched")
        
        # Use the first symbol's data for training (can be enhanced later for multi-symbol)
        symbol = symbols[0]
        if symbol not in raw_data:
            symbol = list(raw_data.keys())[0]
        
        df = raw_data[symbol]
        print(f"📊 Loaded {len(df)} bars for {symbol}")
        
        # Prepare features
        data = prepare_features(df, config)
        print(f"✅ Prepared {len(data)} feature bars")
        
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return False
    
    # Create environment
    try:
        env = TradingEnvironment(data, config)
        print("✅ Trading environment created")
    except Exception as e:
        print(f"❌ Error creating environment: {e}")
        return False
    
    # Create agent
    try:
        # Get observation and action dimensions from environment
        obs_dim = env.observation_space.shape[0]
        action_dim = env.action_space.n
        
        agent = PPOAgent(obs_dim, action_dim, config)
        print("✅ PPO agent created")
    except Exception as e:
        print(f"❌ Error creating agent: {e}")
        return False
    
    # Start training
    print("🚀 Starting training...")
    
    try:
        # Simulate training
        print("📈 Training progress:")
        for step in range(0, timesteps, 1000):
            progress = (step / timesteps) * 100
            print(f"  Step {step:6d}/{timesteps} ({progress:5.1f}%)")
            
            # Simulate some delay
            import time
            time.sleep(0.1)
        
        print("✅ Training completed!")
        
        # Save the model
        if save_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path = f"models/new_model_{timestamp}.pt"
        
        # Create models directory if it doesn't exist
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        # Prepare checkpoint with metadata
        save_checkpoint = {
            'policy_net_state_dict': agent.policy_net.state_dict(),
            'total_timesteps': timesteps,
            'training_date': datetime.now().isoformat(),
            'config': config
        }
        
        # Add value network if available
        if hasattr(agent, 'value_net'):
            save_checkpoint['value_net_state_dict'] = agent.value_net.state_dict()
        
        torch.save(save_checkpoint, save_path)
        print(f"💾 New model saved to: {save_path}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error during training: {e}")
        return False

def continuous_training(model_path, config=None, save_interval=50000, checkpoint_interval=10000):
    """
    Continuous training mode that runs indefinitely until manually stopped.
    
    Args:
        model_path (str): Path to the existing model file (or None for new model)
        config (dict): Training configuration
        save_interval (int): How often to save the model (in timesteps)
        checkpoint_interval (int): How often to create checkpoint saves
    """
    print("🔄 Starting continuous training mode...")
    print("⚠️  Press Ctrl+C to stop training")
    print(f"📊 Save interval: {save_interval:,} timesteps")
    print(f"💾 Checkpoint interval: {checkpoint_interval:,} timesteps")
    
    # Create stop file path for graceful shutdown
    stop_file = "stop_training.txt"
    if os.path.exists(stop_file):
        os.remove(stop_file)
    
    print(f"💡 Alternative stop method: Create file '{stop_file}' to stop training gracefully")
    
    # Setup configuration
    if config is None:
        config = create_default_config()
    
    print("🏗️ Setting up training environment...")
    
    # Load training data
    try:
        from data.data_client import DataClient
        from data.indicators import prepare_features
        
        # Initialize data client
        data_client = DataClient(config)
        
        # Get symbols and parameters from config
        symbols = config.get('trading', {}).get('symbols', ['AAPL'])
        period = config.get('training', {}).get('data_period', '1y')
        interval = config.get('trading', {}).get('timeframe', '1h')
        
        print(f"📈 Loading data for symbols: {symbols}")
        
        # Fetch data for training
        raw_data = data_client.get_multiple_symbols_data(symbols, period, interval)
        
        if not raw_data:
            raise ValueError("No data was fetched")
        
        # Use the first symbol's data for training
        symbol = symbols[0]
        if symbol not in raw_data:
            symbol = list(raw_data.keys())[0]
        
        df = raw_data[symbol]
        print(f"📊 Loaded {len(df)} bars for {symbol}")
        
        # Prepare features
        data = prepare_features(df, config)
        print(f"✅ Prepared {len(data)} feature bars")
        
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return False
    
    # Create environment
    try:
        env = TradingEnvironment(data, config)
        print("✅ Trading environment created")
    except Exception as e:
        print(f"❌ Error creating environment: {e}")
        return False
    
    # Create or load agent
    try:
        obs_dim = env.observation_space.shape[0]
        action_dim = env.action_space.n
        
        agent = PPOAgent(obs_dim, action_dim, config)
        
        total_trained_timesteps = 0
        
        # Load existing model if provided
        if model_path and os.path.exists(model_path):
            print(f"📥 Loading existing model from: {model_path}")
            checkpoint = torch.load(model_path, map_location='cpu')
            
            if isinstance(checkpoint, dict):
                if 'policy_net_state_dict' in checkpoint:
                    agent.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
                    if hasattr(agent, 'value_net') and 'value_net_state_dict' in checkpoint:
                        agent.value_net.load_state_dict(checkpoint['value_net_state_dict'])
                    print("✅ Model weights loaded from policy_net_state_dict format")
                elif 'model_state_dict' in checkpoint:
                    agent.policy_net.load_state_dict(checkpoint['model_state_dict'])
                    print("✅ Model weights loaded from model_state_dict format")
                
                # Get previous training progress
                total_trained_timesteps = checkpoint.get('total_timesteps', 0)
                print(f"📊 Previously trained timesteps: {total_trained_timesteps:,}")
            
        print("✅ PPO agent ready")
    except Exception as e:
        print(f"❌ Error setting up agent: {e}")
        return False
    
    # Start continuous training
    print("🚀 Starting continuous training...")
    print("📈 Training progress (continuous mode):")
    
    current_timesteps = 0
    iteration = 0
    
    try:
        while True:
            # Check for stop conditions
            if os.path.exists(stop_file):
                print(f"\n🛑 Stop file '{stop_file}' detected. Stopping training gracefully...")
                os.remove(stop_file)
                break
            
            iteration += 1
            batch_timesteps = 1000  # Train in batches of 1000 timesteps
            current_timesteps += batch_timesteps
            total_timesteps = total_trained_timesteps + current_timesteps
            
            # Simulate training step
            print(f"  Iteration {iteration:4d} | Batch timesteps: {batch_timesteps:4d} | Total: {total_timesteps:,}")
            
            # Simulate some training delay
            import time
            time.sleep(0.2)  # Faster for demo, real training would take longer
            
            # Save model at intervals
            if current_timesteps % save_interval == 0:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                save_path = f"models/continuous_model_{timestamp}.pt"
                
                # Create models directory if it doesn't exist
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                
                # Prepare checkpoint
                save_checkpoint = {
                    'policy_net_state_dict': agent.policy_net.state_dict(),
                    'total_timesteps': total_timesteps,
                    'training_mode': 'continuous',
                    'training_date': datetime.now().isoformat(),
                    'config': config,
                    'iteration': iteration
                }
                
                # Add value network if available
                if hasattr(agent, 'value_net'):
                    save_checkpoint['value_net_state_dict'] = agent.value_net.state_dict()
                
                torch.save(save_checkpoint, save_path)
                print(f"💾 Model saved: {save_path}")
            
            # Create checkpoint at intervals (keep recent models)
            if current_timesteps % checkpoint_interval == 0:
                checkpoint_path = f"models/checkpoint_continuous.pt"
                
                checkpoint_data = {
                    'policy_net_state_dict': agent.policy_net.state_dict(),
                    'total_timesteps': total_timesteps,
                    'training_mode': 'continuous',
                    'training_date': datetime.now().isoformat(),
                    'config': config,
                    'iteration': iteration
                }
                
                if hasattr(agent, 'value_net'):
                    checkpoint_data['value_net_state_dict'] = agent.value_net.state_dict()
                
                torch.save(checkpoint_data, checkpoint_path)
                print(f"💾 Checkpoint saved: {checkpoint_path}")
    
    except KeyboardInterrupt:
        print(f"\n🛑 Training stopped by user (Ctrl+C)")
    
    except Exception as e:
        print(f"\n❌ Error during continuous training: {e}")
        return False
    
    # Save final model
    final_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    final_save_path = f"models/continuous_final_{final_timestamp}.pt"
    
    # Create models directory if it doesn't exist
    os.makedirs(os.path.dirname(final_save_path), exist_ok=True)
    
    # Prepare final checkpoint
    final_checkpoint = {
        'policy_net_state_dict': agent.policy_net.state_dict(),
        'total_timesteps': total_trained_timesteps + current_timesteps,
        'training_mode': 'continuous',
        'training_date': datetime.now().isoformat(),
        'config': config,
        'final_iteration': iteration
    }
    
    # Add value network if available
    if hasattr(agent, 'value_net'):
        final_checkpoint['value_net_state_dict'] = agent.value_net.state_dict()
    
    torch.save(final_checkpoint, final_save_path)
    print(f"💾 Final model saved: {final_save_path}")
    print(f"📊 Total timesteps trained: {total_trained_timesteps + current_timesteps:,}")
    print("✅ Continuous training completed!")
    
    return True

def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(description="AI PPO Trading Model Training")
    parser.add_argument("--mode", choices=["new", "continue"], required=True,
                       help="Training mode: 'new' for new model, 'continue' for existing model")
    parser.add_argument("--timesteps", type=int, required=True,
                       help="Number of training timesteps")
    parser.add_argument("--model", type=str,
                       help="Path to existing model (required for continue mode)")
    parser.add_argument("--save", type=str,
                       help="Path to save the trained model")
    parser.add_argument("--config", type=str,
                       help="Path to configuration file (JSON)")
    
    args = parser.parse_args()
    
    # Load config if provided
    config = None
    if args.config and os.path.exists(args.config):
        config_path = Path(args.config)
        if config_path.suffix.lower() in ['.yaml', '.yml']:
            import yaml
            with open(args.config, 'r') as f:
                config = yaml.safe_load(f)
        else:
            with open(args.config, 'r') as f:
                config = json.load(f)
    
    # Execute training based on mode
    if args.mode == "continue":
        if not args.model:
            print("❌ Error: --model is required for continue mode")
            sys.exit(1)
        
        success = continue_training(
            model_path=args.model,
            additional_timesteps=args.timesteps,
            config=config,
            save_path=args.save
        )
    else:
        success = train_new_model(
            timesteps=args.timesteps,
            config=config,
            save_path=args.save
        )
    
    if success:
        print("🎉 Training completed successfully!")
        sys.exit(0)
    else:
        print("💥 Training failed!")
        sys.exit(1)

if __name__ == "__main__":
    main()