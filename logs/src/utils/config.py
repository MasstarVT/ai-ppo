"""
Configuration management utilities.
"""

import yaml
import json
import os
from typing import Dict, Any, Optional
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class ConfigManager:
    """Manages configuration loading and validation."""
    
    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path
        self.config = {}
        
        if config_path and os.path.exists(config_path):
            self.load_config(config_path)
    
    def load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from file."""
        try:
            config_path = Path(config_path)
            
            if config_path.suffix.lower() == '.yaml' or config_path.suffix.lower() == '.yml':
                with open(config_path, 'r') as f:
                    self.config = yaml.safe_load(f)
            elif config_path.suffix.lower() == '.json':
                with open(config_path, 'r') as f:
                    self.config = json.load(f)
            else:
                raise ValueError(f"Unsupported config file format: {config_path.suffix}")
            
            logger.info(f"Loaded configuration from {config_path}")
            
            # Validate configuration
            self._validate_config()
            
            return self.config
            
        except Exception as e:
            logger.error(f"Error loading configuration: {e}")
            return {}
    
    def save_config(self, config_path: str, config: Optional[Dict] = None):
        """Save configuration to file."""
        try:
            config_to_save = config or self.config
            config_path = Path(config_path)
            
            # Create directory if it doesn't exist
            config_path.parent.mkdir(parents=True, exist_ok=True)
            
            if config_path.suffix.lower() == '.yaml' or config_path.suffix.lower() == '.yml':
                with open(config_path, 'w') as f:
                    yaml.dump(config_to_save, f, default_flow_style=False, indent=2)
            elif config_path.suffix.lower() == '.json':
                with open(config_path, 'w') as f:
                    json.dump(config_to_save, f, indent=2)
            else:
                raise ValueError(f"Unsupported config file format: {config_path.suffix}")
            
            logger.info(f"Saved configuration to {config_path}")
            
        except Exception as e:
            logger.error(f"Error saving configuration: {e}")
    
    def _validate_config(self):
        """Validate configuration parameters."""
        required_sections = ['trading', 'ppo', 'environment', 'network']
        
        for section in required_sections:
            if section not in self.config:
                logger.warning(f"Missing required configuration section: {section}")
        
        # Validate trading parameters
        trading_config = self.config.get('trading', {})
        if 'symbols' not in trading_config or not trading_config['symbols']:
            logger.warning("No trading symbols specified")
        
        if trading_config.get('initial_balance', 0) <= 0:
            logger.warning("Initial balance should be positive")
        
        # Validate PPO parameters
        ppo_config = self.config.get('ppo', {})
        if ppo_config.get('learning_rate', 0) <= 0:
            logger.warning("Learning rate should be positive")
        
        if ppo_config.get('gamma', 0) <= 0 or ppo_config.get('gamma', 1) >= 1:
            logger.warning("Gamma should be between 0 and 1")
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value using dot notation."""
        keys = key.split('.')
        value = self.config
        
        try:
            for k in keys:
                value = value[k]
            return value
        except (KeyError, TypeError):
            return default
    
    def set(self, key: str, value: Any):
        """Set configuration value using dot notation."""
        keys = key.split('.')
        config = self.config
        
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        
        config[keys[-1]] = value
    
    def update(self, updates: Dict[str, Any]):
        """Update configuration with new values."""
        def deep_update(base_dict, update_dict):
            for key, value in update_dict.items():
                if isinstance(value, dict) and key in base_dict and isinstance(base_dict[key], dict):
                    deep_update(base_dict[key], value)
                else:
                    base_dict[key] = value
        
        deep_update(self.config, updates)
    
    def to_dict(self) -> Dict[str, Any]:
        """Return configuration as dictionary."""
        return self.config.copy()


def setup_logging(config: Dict[str, Any], log_dir: str = "logs"):
    """Setup logging configuration."""
    try:
        logging_config = config.get('logging', {})
        log_level = logging_config.get('level', 'INFO').upper()
        log_to_file = logging_config.get('log_to_file', True)
        
        # Create log directory
        os.makedirs(log_dir, exist_ok=True)
        
        # Configure logging format
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        # Setup root logger
        root_logger = logging.getLogger()
        root_logger.setLevel(getattr(logging, log_level))
        
        # Clear existing handlers
        root_logger.handlers.clear()
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)
        
        # File handler
        if log_to_file:
            file_handler = logging.FileHandler(
                os.path.join(log_dir, 'trading_bot.log')
            )
            file_handler.setFormatter(formatter)
            root_logger.addHandler(file_handler)
        
        logger.info(f"Logging setup complete. Level: {log_level}")
        
    except Exception as e:
        print(f"Error setting up logging: {e}")


def create_default_config() -> Dict[str, Any]:
    """Create default configuration."""
    return {
        'trading': {
            'symbols': ['AAPL', 'MSFT', 'GOOGL'],
            'timeframe': '1h',
            'initial_balance': 10000,
            'max_position_size': 0.1,
            'max_position_days': 30,
            'transaction_cost': 0.001,
            'slippage': 0.0005
        },
        'tradingview': {
            'provider': 'yfinance'
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
        'network': {
            'policy_layers': [256, 256],
            'value_layers': [256, 256],
            'activation': 'tanh'
        },
        'training': {
            'total_timesteps': 1000000,
            'eval_freq': 10000,
            'n_eval_episodes': 10,
            'save_freq': 50000,
            'log_interval': 1000
        },
        'environment': {
            'lookback_window': 50,
            'normalize_observations': True,
            'reward_scaling': 1.0,
            'max_episode_steps': 1000
        },
        'indicators': {
            'sma_periods': [10, 20, 50],
            'ema_periods': [12, 26],
            'rsi_period': 14,
            'macd_periods': [12, 26, 9],
            'bollinger_period': 20,
            'bollinger_std': 2
        },
        'logging': {
            'level': 'INFO',
            'log_to_file': True,
            'use_tensorboard': True,
            'use_wandb': False
        },
        'paths': {
            'data_dir': 'data',
            'model_dir': 'models',
            'log_dir': 'logs'
        }
    }


def validate_paths(config: Dict[str, Any], create_dirs: bool = True) -> bool:
    """Validate and optionally create required directories."""
    try:
        paths = config.get('paths', {})
        
        required_dirs = [
            paths.get('data_dir', 'data'),
            paths.get('model_dir', 'models'),
            paths.get('log_dir', 'logs')
        ]
        
        all_exist = True
        
        for directory in required_dirs:
            if not os.path.exists(directory):
                if create_dirs:
                    os.makedirs(directory, exist_ok=True)
                    logger.info(f"Created directory: {directory}")
                else:
                    logger.warning(f"Directory does not exist: {directory}")
                    all_exist = False
        
        return all_exist
        
    except Exception as e:
        logger.error(f"Error validating paths: {e}")
        return False


def merge_configs(base_config: Dict[str, Any], override_config: Dict[str, Any]) -> Dict[str, Any]:
    """Merge two configuration dictionaries."""
    def deep_merge(base: Dict, override: Dict) -> Dict:
        result = base.copy()
        
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = deep_merge(result[key], value)
            else:
                result[key] = value
        
        return result
    
    return deep_merge(base_config, override_config)