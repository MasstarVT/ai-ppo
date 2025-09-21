#!/usr/bin/env python3
"""
Enhanced that supports continuing training from existing models.
"""

import os
import sys
import torch
import numpy as np
import argparse
import json
import logging
from datetime import datetime
from pathlib import Path

# Load environment variables from .env file FIRST
try:
    from dotenv import load_dotenv
    load_dotenv()
    print("✅ Loaded .env file with Binance credentials")
except ImportError:
    print("❌ dotenv not available, using system environment only")

# Setup proper logging for debugging crashes
log_dir = os.path.join(os.path.dirname(__file__), 'logs')
os.makedirs(log_dir, exist_ok=True)

# Configure logging to file and console
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(log_dir, 'training_debug.log')),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)
logger.info("=== LOGGING ENABLED FOR CRASH DEBUGGING ===")

print("=== Enhanced Training Script Starting...")

print("*** Enhanced Training Script Starting...")

# Simple encoding fix for Windows without buffer detachment
if sys.platform == "win32":
    import locale
    # Just set the locale, don't detach buffers
    try:
        locale.setlocale(locale.LC_ALL, 'en_US.UTF-8')
    except:
        pass  # Ignore if locale not available

# Add src to path
current_dir = Path(__file__).parent
src_path = current_dir / "src"
sys.path.insert(0, str(src_path))

print(">>> Loading modules...")

try:
    from agents.ppo_agent import PPOAgent
    from environments.trading_env import TradingEnvironment
    from data.data_client import DataClient
    from utils.config import ConfigManager, create_default_config
    print("[OK] All modules loaded")
except ImportError as e:
    print(f"[ERROR] Import error: {e}")
    sys.exit(1)

def configure_data_provider(config):
    """
    Auto-configure data provider based on symbols in config.
    
    Args:
        config (dict): Training configuration
        
    Returns:
        dict: Updated configuration with proper data provider
    """
    symbols = config.get('trading', {}).get('symbols', ['BTCUSDT'])
    
    # Auto-configure data provider for crypto symbols
    crypto_symbols = ['BTCUSDT', 'ETHUSDT', 'ADAUSDT', 'DOGEUSDT', 'XRPUSDT', 'LTCUSDT', 'BCHUSDT', 'LINKUSDT', 'DOTUSDT', 'UNIUSDT',
                      'BTC-USD', 'ETH-USD', 'ADA-USD', 'DOGE-USD', 'XRP-USD', 'LTC-USD', 'BCH-USD', 'LINK-USD', 'DOT-USD']
    
    if any(symbol in crypto_symbols for symbol in symbols):
        print(f"[AUTO] Detected crypto symbols: {symbols}")
        
        # Check if using YFinance format
        if any('-USD' in symbol for symbol in symbols):
            print(f"[AUTO] Configuring YFinance provider for crypto symbols")
            if 'data_source' not in config:
                config['data_source'] = {}
            config['data_source']['provider'] = 'yfinance'
        else:
            print(f"[AUTO] Configuring REAL Binance provider for crypto symbols")
            
            # Use Binance with REAL credentials
            if 'data_source' not in config:
                config['data_source'] = {}
            config['data_source']['provider'] = 'binance'
            
            # Load Binance credentials from environment or existing config
            if 'data_providers' not in config:
                config['data_providers'] = {}
            if 'binance' not in config['data_providers']:
                config['data_providers']['binance'] = {}
            
            # The DataClient will load credentials from .env file or environment variables
            print(f"[INFO] Using REAL Binance API for crypto symbols: {symbols}")
    else:
        print(f"[AUTO] Using default provider for stock symbols: {symbols}")
    
    return config

def rename_model(old_path, new_name):
    """
    Rename an existing model file.
    
    Args:
        old_path (str): Path to the existing model file
        new_name (str): New name for the model (without .pt extension)
    
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        if not os.path.exists(old_path):
            print(f"[ERROR] Model file not found: {old_path}")
            return False
        
        # Ensure new name doesn't have .pt extension
        if new_name.endswith('.pt'):
            new_name = new_name[:-3]
        
        # Create new path
        models_dir = os.path.dirname(old_path) or "models"
        new_path = os.path.join(models_dir, f"{new_name}.pt")
        
        # Check if new path already exists
        if os.path.exists(new_path):
            print(f"[ERROR] A model with name '{new_name}' already exists")
            return False
        
        # Rename the file
        os.rename(old_path, new_path)
        print(f"[SUCCESS] Model renamed from '{os.path.basename(old_path)}' to '{new_name}.pt'")
        return True
        
    except Exception as e:
        print(f"[ERROR] Error renaming model: {e}")
        return False


def get_network_config(args):
    """
    Get network configuration based on command line arguments.
    
    Args:
        args: Parsed command line arguments
        
    Returns:
        dict: Network configuration
    """
    # Predefined network sizes
    predefined_configs = {
        "small": {
            "policy_layers": [128, 128],
            "value_layers": [128, 128]
        },
        "medium": {
            "policy_layers": [256, 256], 
            "value_layers": [256, 256]
        },
        "large": {
            "policy_layers": [512, 512],
            "value_layers": [512, 512]
        },
        "extra-large": {
            "policy_layers": [1024, 512, 256],
            "value_layers": [1024, 512, 256]
        }
    }
    
    network_config = {
        "activation": args.activation
    }
    
    # Use predefined configuration if specified
    if args.network_size:
        if args.network_size in predefined_configs:
            config = predefined_configs[args.network_size]
            network_config.update(config)
            print(f"[INFO] Using {args.network_size} network configuration:")
            print(f"       Policy layers: {config['policy_layers']}")
            print(f"       Value layers: {config['value_layers']}")
        else:
            print(f"[WARNING] Unknown network size '{args.network_size}', using default")
    
    # Override with custom layer configurations if provided
    if args.policy_layers:
        network_config["policy_layers"] = args.policy_layers
        print(f"[INFO] Custom policy layers: {args.policy_layers}")
    
    if args.value_layers:
        network_config["value_layers"] = args.value_layers  
        print(f"[INFO] Custom value layers: {args.value_layers}")
    
    # Set defaults if nothing specified
    if "policy_layers" not in network_config:
        network_config["policy_layers"] = [256, 256]  # Default medium size
    if "value_layers" not in network_config:
        network_config["value_layers"] = [256, 256]   # Default medium size
    
    print(f"[INFO] Activation function: {network_config['activation']}")
    
    return network_config


def continue_training(model_path, additional_timesteps, config=None, save_path=None, network_config=None):
    """
    Continue training an existing model.
    
    Args:
        model_path (str): Path to the existing model file
        additional_timesteps (int): Number of additional training steps
        config (dict): Optional training configuration
        save_path (str): Path to save the continued model
    """
    logger.debug("=== CONTINUE TRAINING START ===")
    print(f">>> Starting continued training from: {model_path}")
    print(f"[INFO] Additional timesteps: {additional_timesteps:,}")
    
    logger.debug(f"Model path: {model_path}")
    logger.debug(f"Additional timesteps: {additional_timesteps}")
    logger.debug(f"Config provided: {config is not None}")
    logger.debug(f"Save path: {save_path}")
    
    # Load the existing model
    if not os.path.exists(model_path):
        logger.error(f"Model file not found: {model_path}")
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    print("[LOAD] Loading existing model...")
    logger.debug("Attempting to load model checkpoint")
    
    try:
        # Load the model state
        checkpoint = torch.load(model_path, map_location='cpu')
        print(f"[OK] Model loaded successfully")
        logger.debug(f"Model checkpoint loaded, type: {type(checkpoint)}")
        
        # Extract model info if available
        if isinstance(checkpoint, dict):
            print("[INFO] Model information:")
            logger.debug("Displaying model information:")
            for key in ['total_timesteps', 'episode_count', 'best_reward']:
                if key in checkpoint:
                    print(f"  - {key}: {checkpoint[key]}")
                    logger.debug(f"  Model info - {key}: {checkpoint[key]}")
        
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        print(f"[ERROR] Failed to load model: {e}")
        raise
        print(f"[ERROR] Error loading model: {e}")
        return False
    
    # Setup configuration
    if config is None:
        config = create_default_config()
    
    # Auto-configure data provider based on symbols
    config = configure_data_provider(config)
    
    # Apply network configuration if provided
    if network_config:
        if 'network' not in config:
            config['network'] = {}
        config['network'].update(network_config)
        print(f"[INFO] Applied custom network configuration")
        print(f"       Policy layers: {config['network']['policy_layers']}")
        print(f"       Value layers: {config['network']['value_layers']}")
        print(f"       Activation: {config['network']['activation']}")

    print("[SETUP] Setting up training environment...")
    
    # Load training data
    try:
        from data.data_client import DataClient
        from data.indicators import prepare_features
        
        # Initialize data client
        data_client = DataClient(config)
        
        # Get symbols and parameters from config
        symbols = config.get('trading', {}).get('symbols', ['AAPL'])
        period = config.get('training', {}).get('data_period', '1y')
        interval = config.get('trading', {}).get('timeframe', '5m')
        
        print(f"[DATA] Loading data for symbols: {symbols}")
        
        # Fetch data for training
        raw_data = data_client.get_multiple_symbols_data(symbols, period, interval)
        
        if not raw_data:
            raise ValueError("No data was fetched")
        
        # Use the first symbol's data for training (can be enhanced later for multi-symbol)
        symbol = symbols[0]
        if symbol not in raw_data:
            symbol = list(raw_data.keys())[0]
        
        df = raw_data[symbol]
        print(f"[INFO] Loaded {len(df)} bars for {symbol}")
        
        # Prepare features
        data = prepare_features(df, config)
        print(f"[OK] Prepared {len(data)} feature bars")
        
    except Exception as e:
        print(f"[ERROR] Error loading data: {e}")
        return False
    
    # Create environment
    try:
        env = TradingEnvironment(data, config)
        print("[OK] Trading environment created")
    except Exception as e:
        print(f"[ERROR] Error creating environment: {e}")
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
                print("[OK] Model weights loaded from policy_net_state_dict format")
            elif 'model_state_dict' in checkpoint:
                # Legacy format with single model state dict
                agent.policy_net.load_state_dict(checkpoint['model_state_dict'])
                print("[OK] Model weights loaded from model_state_dict format")
            else:
                # Assume the checkpoint is just the model state dict
                agent.policy_net.load_state_dict(checkpoint)
                print("[OK] Model weights loaded from direct state dict")
        else:
            # Checkpoint is directly a state dict
            agent.policy_net.load_state_dict(checkpoint)
            print("[OK] Model weights loaded from direct state dict")
            
    except Exception as e:
        print(f"[ERROR] Error setting up agent: {e}")
        return False
    
    # Start continued training
    print("[START] Starting continued training...")
    
    try:
        # Real PPO continued training loop
        print("[DATA] Training progress:")
        
        timestep = 0
        episode = 0
        
        while timestep < additional_timesteps:
            # Reset environment for new episode
            obs = env.reset()
            episode_reward = 0
            episode_length = 0
            
            # Run episode
            while not env.done and timestep < additional_timesteps:
                # Check if buffer is full and needs updating
                if agent.buffer.ptr >= agent.n_steps:
                    # Update agent and reset buffer
                    training_stats = agent.update()
                    if timestep % 1000 == 0:  # Log occasionally
                        print(f"  Updated agent - Policy Loss: {training_stats.get('policy_loss', 0):.4f}")
                
                # Get action from agent
                action, log_prob, value = agent.get_action(obs)
                
                # Take step in environment
                next_obs, reward, done, info = env.step(action)
                
                # Store experience in buffer
                agent.store_experience(obs, action, reward, value, log_prob, done)
                
                # Update tracking
                episode_reward += reward
                episode_length += 1
                timestep += 1
                obs = next_obs
                
                # Progress reporting
                if timestep % 100 == 0:
                    progress = (timestep / additional_timesteps) * 100
                    progress_msg = f"  Step {timestep:6d}/{additional_timesteps} ({progress:5.1f}%)"
                    print(progress_msg)
                    # Write progress to file directly to avoid logging buffer issues
                    try:
                        log_path = os.path.join(os.path.dirname(__file__), 'logs', 'training_debug.log')
                        with open(log_path, 'a', encoding='utf-8') as f:
                            f.write(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - INFO - {progress_msg}\n")
                            f.flush()
                    except Exception:
                        pass  # Ignore file writing errors to prevent training interruption
            
            # Finish episode
            agent.finish_episode(obs)
            episode += 1
            
            # Episode summary
            if episode % 10 == 0:
                episode_stats = env.get_episode_stats()
                total_return = episode_stats.get('total_return', 0)
                print(f"  Episode {episode}: Return {total_return:.2f}%, Reward {episode_reward:.2f}")
        
        # Final update if buffer has remaining data and is full
        if agent.buffer.ptr >= agent.n_steps:
            training_stats = agent.update()
        
        print("[OK] Continued training completed!")
        
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
        print(f"[SAVE] Updated model saved to: {save_path}")
        
        return True
        
    except Exception as e:
        print(f"[ERROR] Error during training: {e}")
        return False

def train_new_model(timesteps, config=None, save_path=None, network_config=None):
    """
    Train a new model from scratch.
    
    Args:
        timesteps (int): Number of training steps
        config (dict): Training configuration
        save_path (str): Path to save the new model
    """
    print(f"[NEW] Starting new model training")
    print(f"[INFO] Total timesteps: {timesteps:,}")
    
    # Setup configuration
    if config is None:
        config = create_default_config()
    
    # Auto-configure data provider based on symbols
    config = configure_data_provider(config)
    
    # Apply network configuration if provided
    if network_config:
        if 'network' not in config:
            config['network'] = {}
        config['network'].update(network_config)
        print(f"[INFO] Applied custom network configuration")
        print(f"       Policy layers: {config['network']['policy_layers']}")
        print(f"       Value layers: {config['network']['value_layers']}")
        print(f"       Activation: {config['network']['activation']}")

    print("[SETUP] Setting up training environment...")
    
    # Load training data
    try:
        from data.data_client import DataClient
        from data.indicators import prepare_features
        
        # Initialize data client
        data_client = DataClient(config)
        
        # Get symbols and parameters from config
        symbols = config.get('trading', {}).get('symbols', ['AAPL'])
        period = config.get('training', {}).get('data_period', '1y')
        interval = config.get('trading', {}).get('timeframe', '5m')
        
        print(f"[DATA] Loading data for symbols: {symbols}")
        
        # Fetch data for training
        raw_data = data_client.get_multiple_symbols_data(symbols, period, interval)
        
        if not raw_data:
            raise ValueError("No data was fetched")
        
        # Use the first symbol's data for training (can be enhanced later for multi-symbol)
        symbol = symbols[0]
        if symbol not in raw_data:
            symbol = list(raw_data.keys())[0]
        
        df = raw_data[symbol]
        print(f"[INFO] Loaded {len(df)} bars for {symbol}")
        
        # Prepare features
        data = prepare_features(df, config)
        print(f"[OK] Prepared {len(data)} feature bars")
        
    except Exception as e:
        print(f"[ERROR] Error loading data: {e}")
        return False
    
    # Create environment
    try:
        env = TradingEnvironment(data, config)
        print("[OK] Trading environment created")
    except Exception as e:
        print(f"[ERROR] Error creating environment: {e}")
        return False
    
    # Create agent
    try:
        # Get observation and action dimensions from environment
        obs_dim = env.observation_space.shape[0]
        action_dim = env.action_space.n
        
        agent = PPOAgent(obs_dim, action_dim, config)
        print("[OK] PPO agent created")
    except Exception as e:
        print(f"[ERROR] Error creating agent: {e}")
        return False
    
    # Start training
    print("[START] Starting training...")
    
    try:
        # Real PPO training loop
        print("[DATA] Training progress:")
        
        timestep = 0
        episode = 0
        
        while timestep < timesteps:
            # Reset environment for new episode
            obs = env.reset()
            episode_reward = 0
            episode_length = 0
            
            # Run episode
            while not env.done and timestep < timesteps:
                # Check if buffer is full and needs updating
                if agent.buffer.ptr >= agent.n_steps:
                    # Finish current path before updating
                    agent.finish_episode(obs)
                    
                    # Update agent and reset buffer
                    try:
                        training_stats = agent.update()
                        if timestep % 1000 == 0:  # Log occasionally
                            print(f"  Updated agent - Policy Loss: {training_stats.get('policy_loss', 0):.4f}")
                    except AssertionError as e:
                        # Buffer not full, skip this update
                        if timestep % 1000 == 0:
                            print(f"  Buffer not ready for update, skipping...")
                
                # Get action from agent
                action, log_prob, value = agent.get_action(obs)
                
                # Take step in environment
                next_obs, reward, done, info = env.step(action)
                
                # Store experience in buffer
                agent.store_experience(obs, action, reward, value, log_prob, done)
                
                # Update tracking
                episode_reward += reward
                episode_length += 1
                timestep += 1
                obs = next_obs
                
                # Progress reporting
                if timestep % 100 == 0:
                    progress = (timestep / timesteps) * 100
                    progress_msg = f"  Step {timestep:6d}/{timesteps} ({progress:5.1f}%)"
                    print(progress_msg)
                    # Write progress to file directly to avoid logging buffer issues
                    try:
                        log_path = os.path.join(os.path.dirname(__file__), 'logs', 'training_debug.log')
                        with open(log_path, 'a', encoding='utf-8') as f:
                            f.write(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - INFO - {progress_msg}\n")
                            f.flush()
                    except Exception:
                        pass  # Ignore file writing errors to prevent training interruption
            
            # Finish episode
            agent.finish_episode(obs)
            episode += 1
            
            # Episode summary
            if episode % 10 == 0:
                episode_stats = env.get_episode_stats()
                total_return = episode_stats.get('total_return', 0)
                print(f"  Episode {episode}: Return {total_return:.2f}%, Reward {episode_reward:.2f}")
        
        # Final update - always do this to handle any remaining data
        if agent.buffer.ptr > 0:
            print(f"  Final update - Buffer had {agent.buffer.ptr} experiences")
            # Use partial update for final buffer state (buffer is unlikely to be completely full)
            try:
                # For final update, always use partial update since buffer may not be completely full
                training_stats = agent.update_partial(force_final=True)
                print(f"    Policy Loss: {training_stats.get('policy_loss', 0):.4f}")
            except Exception as e:
                import traceback
                print(f"    Final update failed: {type(e).__name__}: {str(e)}")
                print(f"    Traceback: {traceback.format_exc()}")
                training_stats = {'policy_loss': 0, 'value_loss': 0}
        
        print("[OK] Training completed!")
        
        # Write completion message to log file
        try:
            log_path = os.path.join(os.path.dirname(__file__), 'logs', 'training_debug.log')
            with open(log_path, 'a', encoding='utf-8') as f:
                f.write(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - INFO - Training completed successfully\n")
                f.flush()
        except Exception:
            pass  # Ignore file writing errors
        
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
        print(f"[SAVE] New model saved to: {save_path}")
        
        return True
        
    except Exception as e:
        import traceback
        print(f"[ERROR] Error during training: {e}")
        print(f"[ERROR] Traceback: {traceback.format_exc()}")
        return False

def continuous_training(model_path, config=None, save_interval=50000, checkpoint_interval=10000, network_config=None, model_name=None):
    """
    Continuous training mode that runs indefinitely until manually stopped.
    
    Args:
        model_path (str): Path to the existing model file (or None for new model)
        config (dict): Training configuration
        save_interval (int): How often to save the model (in timesteps)
        checkpoint_interval (int): How often to create checkpoint saves
    """
    print(">>> Starting continuous training mode...")
    print("[WARNING]  Press Ctrl+C to stop training")
    print(f"[INFO] Save interval: {save_interval:,} timesteps")
    print(f"[SAVE] Checkpoint interval: {checkpoint_interval:,} timesteps")
    
    # Create stop file path for graceful shutdown
    stop_file = "stop_training.txt"
    if os.path.exists(stop_file):
        os.remove(stop_file)
    
    print(f"[INFO] Alternative stop method: Create file '{stop_file}' to stop training gracefully")
    
    # Setup configuration
    if config is None:
        config = create_default_config()
    
    # Auto-configure data provider based on symbols
    config = configure_data_provider(config)
    
    # Apply network configuration if provided
    if network_config:
        if 'network' not in config:
            config['network'] = {}
        config['network'].update(network_config)
        print(f"[INFO] Applied custom network configuration")
        print(f"       Policy layers: {config['network']['policy_layers']}")
        print(f"       Value layers: {config['network']['value_layers']}")
        print(f"       Activation: {config['network']['activation']}")

    print("[SETUP] Setting up training environment...")
    logger.info("Starting training environment setup")
    
    # Load training data
    try:
        logger.info("Importing data modules")
        from data.data_client import DataClient
        from data.indicators import prepare_features
        logger.info("Data modules imported successfully")
        
        # Initialize data client
        logger.info("Initializing DataClient")
        data_client = DataClient(config)
        logger.info("DataClient initialized successfully")
        
        # Get symbols and parameters from config
        symbols = config.get('trading', {}).get('symbols', ['AAPL'])
        period = config.get('training', {}).get('data_period', '1y')
        interval = config.get('trading', {}).get('timeframe', '5m')
        
        print(f"[DATA] Loading data for symbols: {symbols}")
        logger.info(f"Requesting data: symbols={symbols}, period={period}, interval={interval}")
        
        # Fetch data for training
        logger.info("Calling get_multiple_symbols_data...")
        raw_data = data_client.get_multiple_symbols_data(symbols, period, interval)
        logger.info(f"Data fetch completed, got {len(raw_data) if raw_data else 0} symbols")
        
        if not raw_data:
            raise ValueError("No data was fetched")
        
        # Use the first symbol's data for training
        symbol = symbols[0]
        if symbol not in raw_data:
            symbol = list(raw_data.keys())[0]
        
        df = raw_data[symbol]
        print(f"[INFO] Loaded {len(df)} bars for {symbol}")
        logger.info(f"Using symbol {symbol} with {len(df)} bars")
        
        # Prepare features
        logger.info("Preparing features...")
        data = prepare_features(df, config)
        print(f"[OK] Prepared {len(data)} feature bars")
        logger.info(f"Feature preparation complete: {len(data)} bars")
        
    except Exception as e:
        print(f"[ERROR] Error loading data: {e}")
        logger.error(f"Data loading failed: {e}", exc_info=True)
        return False
    
    # Create environment
    try:
        logger.info("Creating TradingEnvironment...")
        env = TradingEnvironment(data, config)
        print("[OK] Trading environment created")
        logger.info("TradingEnvironment created successfully")
    except Exception as e:
        print(f"[ERROR] Error creating environment: {e}")
        logger.error(f"Environment creation failed: {e}", exc_info=True)
        return False
    
    # Create or load agent
    try:
        logger.info("Getting environment dimensions...")
        obs_dim = env.observation_space.shape[0]
        action_dim = env.action_space.n
        logger.info(f"Environment dimensions: obs_dim={obs_dim}, action_dim={action_dim}")
        
        logger.info("Creating PPO agent...")
        agent = PPOAgent(obs_dim, action_dim, config)
        logger.info("PPO agent created successfully")
        
        total_trained_timesteps = 0
        
        # Load existing model if provided
        if model_path and os.path.exists(model_path):
            print(f"[LOAD] Loading existing model from: {model_path}")
            logger.info(f"Loading model from {model_path}")
            checkpoint = torch.load(model_path, map_location='cpu')
            
            if isinstance(checkpoint, dict):
                if 'policy_net_state_dict' in checkpoint:
                    agent.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
                    if hasattr(agent, 'value_net') and 'value_net_state_dict' in checkpoint:
                        agent.value_net.load_state_dict(checkpoint['value_net_state_dict'])
                    print("[OK] Model weights loaded from policy_net_state_dict format")
                    logger.info("Model loaded from policy_net_state_dict format")
                elif 'model_state_dict' in checkpoint:
                    agent.policy_net.load_state_dict(checkpoint['model_state_dict'])
                    print("[OK] Model weights loaded from model_state_dict format")
                    logger.info("Model loaded from model_state_dict format")
                
                # Get previous training progress
                total_trained_timesteps = checkpoint.get('total_timesteps', 0)
                print(f"[INFO] Previously trained timesteps: {total_trained_timesteps:,}")
                logger.info(f"Previous timesteps: {total_trained_timesteps}")
            
        print("[OK] PPO agent ready")
        logger.info("PPO agent setup complete")
    except Exception as e:
        print(f"[ERROR] Error setting up agent: {e}")
        logger.error(f"Agent setup failed: {e}", exc_info=True)
        return False
    
    # Start continuous training
    print("[START] Starting continuous training...")
    print("[DATA] Training progress (continuous mode):")
    logger.info("Starting continuous training loop")
    
    current_timesteps = 0
    iteration = 0
    
    try:
        while True:
            # Check for stop conditions
            if os.path.exists(stop_file):
                print(f"\n[STOP] Stop file '{stop_file}' detected. Stopping training gracefully...")
                logger.info("Stop file detected, graceful shutdown")
                os.remove(stop_file)
                break
            
            iteration += 1
            batch_timesteps = 1000  # Train in batches of 1000 timesteps
            current_timesteps += batch_timesteps
            total_timesteps = total_trained_timesteps + current_timesteps
            
            # Real training step - train for batch_timesteps
            print(f"  Iteration {iteration:4d} | Training {batch_timesteps} timesteps | Total: {total_timesteps:,}")
            logger.info(f"Starting iteration {iteration}, batch_timesteps={batch_timesteps}, total={total_timesteps}")
            
            # Reset environment and train for this batch
            batch_step = 0
            episode = 0
            
            while batch_step < batch_timesteps:
                logger.debug(f"Starting episode {episode+1}, batch_step={batch_step}")
                # Reset environment for new episode
                obs = env.reset()
                episode_reward = 0
                episode_length = 0
                
                # Run episode
                while not env.done and batch_step < batch_timesteps:
                    # Check if buffer is full and needs updating
                    if agent.buffer.ptr >= agent.n_steps:
                        # Update agent and reset buffer
                        training_stats = agent.update()
                    
                    # Get action from agent
                    action, log_prob, value = agent.get_action(obs)
                    
                    # Take step in environment
                    next_obs, reward, done, info = env.step(action)
                    
                    # Store experience in buffer
                    agent.store_experience(obs, action, reward, value, log_prob, done)
                    
                    # Update tracking
                    episode_reward += reward
                    episode_length += 1
                    batch_step += 1
                    obs = next_obs
                
                # Finish episode
                agent.finish_episode(obs)
                episode += 1
            
            # Final update if buffer has remaining data
            if agent.buffer.ptr > 0:
                try:
                    if agent.buffer.ptr >= agent.n_steps:
                        training_stats = agent.update()
                    else:
                        training_stats = agent.update_partial(force_final=True)
                    print(f"    Policy Loss: {training_stats.get('policy_loss', 0):.4f}, Episodes: {episode}")
                except Exception as e:
                    print(f"    Update failed: {e}")
                    training_stats = {'policy_loss': 0}
            
            # Save model at intervals
            if current_timesteps % save_interval == 0:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                if model_name:
                    save_path = f"models/{model_name}_{timestamp}.pt"
                else:
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
                print(f"[SAVE] Model saved: {save_path}")
            
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
                print(f"[SAVE] Checkpoint saved: {checkpoint_path}")
    
    except KeyboardInterrupt:
        print(f"\n[STOP] Training stopped by user (Ctrl+C)")
    
    except Exception as e:
        print(f"\n[ERROR] Error during continuous training: {e}")
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
    print(f"[SAVE] Final model saved: {final_save_path}")
    print(f"[INFO] Total timesteps trained: {total_trained_timesteps + current_timesteps:,}")
    print("[OK] Continuous training completed!")
    
    return True

def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(description="AI PPO Trading Model Training")
    parser.add_argument("--mode", choices=["new", "continue", "continuous"],
                       help="Training mode: 'new' for new model, 'continue' for existing model, 'continuous' for indefinite training")
    parser.add_argument("--timesteps", type=int, 
                       help="Number of training timesteps (not required for continuous mode)")
    parser.add_argument("--model", type=str,
                       help="Path to existing model (required for continue mode, optional for continuous)")
    parser.add_argument("--save", type=str,
                       help="Path to save the trained model")
    parser.add_argument("--config", type=str,
                       help="Path to configuration file (YAML or JSON)")
    parser.add_argument("--save-interval", type=int, default=50000,
                       help="Save interval for continuous mode (default: 50000 timesteps)")
    parser.add_argument("--checkpoint-interval", type=int, default=10000,
                       help="Checkpoint interval for continuous mode (default: 10000 timesteps)")
    
    # Neural network size configuration arguments
    parser.add_argument("--network-size", choices=["small", "medium", "large", "extra-large"], 
                       help="Predefined network size (small: [128,128], medium: [256,256], large: [512,512], extra-large: [1024,512,256])")
    parser.add_argument("--policy-layers", type=int, nargs='+', 
                       help="Custom policy network layer sizes (e.g., --policy-layers 512 256 128)")
    parser.add_argument("--value-layers", type=int, nargs='+',
                       help="Custom value network layer sizes (e.g., --value-layers 512 256 128)")
    parser.add_argument("--activation", choices=["relu", "tanh", "leaky_relu"], default="tanh",
                       help="Activation function for neural networks (default: tanh)")
    
    # Model naming arguments
    parser.add_argument("--model-name", type=str,
                       help="Custom name for the model (without .pt extension)")
    parser.add_argument("--rename-model", type=str, nargs=2, metavar=("OLD_PATH", "NEW_NAME"),
                       help="Rename an existing model: --rename-model path/to/model.pt new_name")
    
    args = parser.parse_args()
    
    # Handle model renaming if requested (no training mode needed)
    if args.rename_model:
        old_path, new_name = args.rename_model
        success = rename_model(old_path, new_name)
        if success:
            print("[SUCCESS] Model renamed successfully!")
        else:
            print("[FAILED] Model rename failed!")
        sys.exit(0 if success else 1)
    
    # Validate arguments for training modes
    if not args.mode:
        print("[ERROR] Error: --mode is required for training operations")
        sys.exit(1)
        
    if args.mode in ["new", "continue"] and not args.timesteps:
        print("[ERROR] Error: --timesteps is required for new and continue modes")
        sys.exit(1)
    
    if args.mode == "continue" and not args.model:
        print("[ERROR] Error: --model is required for continue mode")
        sys.exit(1)
    
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
    
    # Get network configuration from arguments
    network_config = get_network_config(args)
    
    # Generate save path with custom name if provided
    save_path = args.save
    if args.model_name and not save_path:
        save_path = f"models/{args.model_name}.pt"
    
    # Execute training based on mode
    if args.mode == "continue":
        success = continue_training(
            model_path=args.model,
            additional_timesteps=args.timesteps,
            config=config,
            save_path=save_path,
            network_config=network_config
        )
    elif args.mode == "continuous":
        success = continuous_training(
            model_path=args.model,
            config=config,
            save_interval=args.save_interval,
            checkpoint_interval=args.checkpoint_interval,
            network_config=network_config,
            model_name=args.model_name
        )
    else:  # new mode
        success = train_new_model(
            timesteps=args.timesteps,
            config=config,
            save_path=save_path,
            network_config=network_config
        )
    
    if success:
        print("[SUCCESS] Training completed successfully!")
        sys.exit(0)
    else:
        print("[FAILED] Training failed!")
        sys.exit(1)

if __name__ == "__main__":
    main()