"""
AI PPO Trading System
A reinforcement learning trading bot using Proximal Policy Optimization.
"""

__version__ = "1.0.0"
__author__ = "MasstarVT"
__description__ = "PPO-based trading bot with TradingView API integration"

from src.data import DataClient, TechnicalIndicators
from src.environments import TradingEnvironment
from src.agents import PPOAgent
from src.utils import ConfigManager

__all__ = [
    'DataClient',
    'TechnicalIndicators', 
    'TradingEnvironment',
    'PPOAgent',
    'ConfigManager'
]