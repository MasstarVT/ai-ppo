"""
Trading environment for reinforcement learning.
Implements a gymnasium-compatible environment for stock trading.
"""

import gymnasium as gym
from gymnasium import spaces
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import logging
from enum import IntEnum

logger = logging.getLogger(__name__)


class TradingAction(IntEnum):
    """Trading actions."""
    SELL = 0
    HOLD = 1
    BUY = 2


class Portfolio:
    """Manages portfolio state and calculations."""
    
    def __init__(self, initial_balance: float, transaction_cost: float = 0.001, slippage: float = 0.0005):
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.shares = 0
        self.transaction_cost = transaction_cost
        self.slippage = slippage
        self.total_trades = 0
        self.trade_history = []
        
    def execute_trade(self, action: int, price: float, max_position_size: float) -> Dict:
        """Execute a trading action."""
        trade_info = {
            'action': action,
            'price': price,
            'shares_before': self.shares,
            'balance_before': self.balance,
            'total_value_before': self.get_total_value(price)
        }
        
        if action == TradingAction.BUY and self.balance > 0:
            # Calculate maximum shares we can buy
            effective_price = price * (1 + self.slippage)  # Account for slippage
            
            # Protect against zero or very small prices
            if effective_price <= 1e-8:
                effective_price = 1e-8
            
            max_spend = self.balance * max_position_size
            max_shares = int(max_spend / effective_price)
            
            if max_shares > 0:
                cost = max_shares * effective_price
                transaction_fee = cost * self.transaction_cost
                total_cost = cost + transaction_fee
                
                if total_cost <= self.balance:
                    self.balance -= total_cost
                    self.shares += max_shares
                    self.total_trades += 1
                    
                    trade_info.update({
                        'shares_traded': max_shares,
                        'cost': total_cost,
                        'fee': transaction_fee
                    })
        
        elif action == TradingAction.SELL and self.shares > 0:
            # Sell all shares
            effective_price = price * (1 - self.slippage)  # Account for slippage
            
            # Protect against zero or very small prices
            if effective_price <= 1e-8:
                effective_price = 1e-8
            
            proceeds = self.shares * effective_price
            transaction_fee = proceeds * self.transaction_cost
            net_proceeds = proceeds - transaction_fee
            
            trade_info.update({
                'shares_traded': -self.shares,
                'proceeds': net_proceeds,
                'fee': transaction_fee
            })
            
            self.balance += net_proceeds
            self.shares = 0
            self.total_trades += 1
        
        # Update trade info with final state
        trade_info.update({
            'shares_after': self.shares,
            'balance_after': self.balance,
            'total_value_after': self.get_total_value(price)
        })
        
        self.trade_history.append(trade_info)
        return trade_info
    
    def get_total_value(self, current_price: float) -> float:
        """Get total portfolio value."""
        return self.balance + (self.shares * current_price)
    
    def get_return(self, current_price: float) -> float:
        """Get portfolio return."""
        current_value = self.get_total_value(current_price)
        return (current_value - self.initial_balance) / self.initial_balance
    
    def reset(self):
        """Reset portfolio to initial state."""
        self.balance = self.initial_balance
        self.shares = 0
        self.total_trades = 0
        self.trade_history = []


class TradingEnvironment(gym.Env):
    """
    A trading environment for reinforcement learning.
    
    The agent can take actions to buy, sell, or hold stocks.
    The observation space includes price data and technical indicators.
    The reward is based on portfolio performance and risk metrics.
    """
    
    def __init__(self, data: pd.DataFrame, config: Dict):
        super().__init__()
        
        self.data = data.copy()
        # Exclude Close_raw from observation features to preserve model input size
        self.feature_columns = [c for c in self.data.columns if c not in ['datetime', 'Close_raw']]
        self.config = config
        self.lookback_window = config.get('environment', {}).get('lookback_window', 50)
        self.max_episode_steps = config.get('environment', {}).get('max_episode_steps', 1000)
        
        # Trading parameters
        self.initial_balance = config.get('trading', {}).get('initial_balance', 10000)
        self.max_position_size = config.get('trading', {}).get('max_position_size', 0.1)
        self.max_position_days = config.get('trading', {}).get('max_position_days', 30)
        self.transaction_cost = config.get('trading', {}).get('transaction_cost', 0.001)
        self.slippage = config.get('trading', {}).get('slippage', 0.0005)
        
        # Position tracking for timeout
        self.position_start_step = None  # When current position started
        self.position_timeout_penalty = 0.01  # Penalty for forced position close
        
        # Initialize portfolio
        self.portfolio = Portfolio(self.initial_balance, self.transaction_cost, self.slippage)
        
        # Reward configuration
        env_cfg = config.get('environment', {})
        self.reward_mode = env_cfg.get('reward_mode', 'incremental')  # 'incremental' or 'cumulative'
        self.trade_penalty = float(env_cfg.get('trade_penalty', 0.0001))  # small penalty per trade action
        self.risk_penalty_scale = float(env_cfg.get('risk_penalty_scale', 2.0))
        
        # Define observation and action spaces
        self._setup_spaces()
        
        # Episode tracking
        self.current_step = 0
        self.episode_start_idx = 0
        self.done = False
        self._last_portfolio_value = self.initial_balance
        
        # Performance tracking
        self.episode_returns = []
        self.episode_sharpe_ratios = []
        self.max_drawdown = 0
        
        logger.info(f"Initialized trading environment with {len(data)} data points")
    
    def _setup_spaces(self):
        """Setup observation and action spaces."""
        # Action space: 0=Sell, 1=Hold, 2=Buy
        self.action_space = spaces.Discrete(3)
        
        # Observation space: market data + portfolio state
        # Market features: OHLCV + technical indicators
        market_features = len(self.feature_columns)
        
        # Portfolio features: balance, shares, total_value, position_ratio
        portfolio_features = 4
        
        # Total observation features for lookback window
        total_features = (market_features + portfolio_features) * self.lookback_window
        
        self.observation_space = spaces.Box(
            low=-np.inf, 
            high=np.inf, 
            shape=(total_features,), 
            dtype=np.float32
        )
        
        logger.info(f"Observation space shape: {self.observation_space.shape}")
    
    def _get_observation(self) -> np.ndarray:
        """Get current observation."""
        end_idx = self.episode_start_idx + self.current_step + 1
        start_idx = max(0, end_idx - self.lookback_window)
        
        # Validate indices
        if end_idx > len(self.data):
            logger.error(f"end_idx {end_idx} exceeds data length {len(self.data)}")
            end_idx = len(self.data)
        
        if start_idx >= end_idx:
            logger.error(f"Invalid slice: start_idx {start_idx} >= end_idx {end_idx}")
            start_idx = max(0, end_idx - 1)
        
        # Get market data (exclude Close_raw)
        try:
            market_data = self.data[self.feature_columns].iloc[start_idx:end_idx].values
        except IndexError as e:
            logger.error(f"IndexError in market data slice [{start_idx}:{end_idx}]: {e}")
            # Fallback: use last available data point
            market_data = self.data[self.feature_columns].iloc[-1:].values
        
        # Pad if we don't have enough historical data
        if len(market_data) < self.lookback_window:
            padding_rows = self.lookback_window - len(market_data)
            if len(market_data) > 0:
                padding = np.tile(market_data[0], (padding_rows, 1))
            else:
                # Emergency fallback: zeros
                padding = np.zeros((padding_rows, len(self.feature_columns)))
            market_data = np.vstack([padding, market_data])
        
        # Get current price for portfolio calculations (prefer raw)
        try:
            row = self.data.iloc[min(end_idx - 1, len(self.data) - 1)]
            current_price = row['Close_raw'] if 'Close_raw' in self.data.columns else row['Close']
        except IndexError:
            logger.error(f"Cannot access data at index {end_idx - 1}")
            # Fallback to last price
            last_row = self.data.iloc[-1]
            current_price = last_row['Close_raw'] if 'Close_raw' in self.data.columns else last_row['Close']
        
        # Portfolio state
        portfolio_state = np.array([
            self.portfolio.balance / self.initial_balance,  # Normalized balance
            self.portfolio.shares,  # Number of shares
            self.portfolio.get_total_value(current_price) / self.initial_balance,  # Normalized total value
            self.portfolio.shares * current_price / self.portfolio.get_total_value(current_price) if self.portfolio.get_total_value(current_price) > 0 else 0  # Position ratio
        ])
        
        # Repeat portfolio state for each time step in lookback window
        portfolio_data = np.tile(portfolio_state, (self.lookback_window, 1))
        
        # Combine market and portfolio data
        observation = np.hstack([market_data, portfolio_data])
        
        return observation.flatten().astype(np.float32)
    
    def _calculate_reward(self, action: int, trade_info: Dict) -> float:
        """Calculate reward for the current step."""
        # Safe index access
        current_idx = self.episode_start_idx + self.current_step
        if current_idx >= len(self.data):
            current_idx = len(self.data) - 1
            
        row = self.data.iloc[current_idx]
        current_price = row['Close_raw'] if 'Close_raw' in self.data.columns else row['Close']
        
        # Compute portfolio values
        current_value = self.portfolio.get_total_value(current_price)
        if self.reward_mode == 'incremental':
            prev_value = max(1e-8, self._last_portfolio_value)
            step_return = (current_value - prev_value) / prev_value
            reward = step_return
        else:
            # Fallback to cumulative return (legacy)
            reward = self.portfolio.get_return(current_price)
        
        # Penalize excessive trading
        if action != TradingAction.HOLD:
            reward -= self.trade_penalty
        
        # Reward for profitable trades
        if trade_info.get('shares_traded', 0) != 0:
            trade_return = (trade_info['total_value_after'] - trade_info['total_value_before']) / trade_info['total_value_before']
            reward += trade_return * 10  # Amplify trade returns
        
        # Risk adjustment
        if len(self.episode_returns) > 20:
            returns_series = pd.Series(self.episode_returns[-20:])
            volatility = returns_series.std()
            if volatility > 0:
                sharpe_ratio = returns_series.mean() / volatility
                reward += sharpe_ratio * 0.1
        
        # Penalize large drawdowns
        max_value = max([
            self.portfolio.get_total_value(
                self.data.iloc[i]['Close_raw'] if 'Close_raw' in self.data.columns else self.data.iloc[i]['Close']
            )
            for i in range(
                self.episode_start_idx,
                min(self.episode_start_idx + self.current_step + 1, len(self.data))
            )
        ])
        drawdown = (max_value - current_value) / max_value if max_value > 0 else 0
        
        if drawdown > 0.1:  # More than 10% drawdown
            reward -= drawdown * self.risk_penalty_scale
        
        return reward
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        """Execute one time step within the environment."""
        if self.done:
            raise ValueError("Episode has ended. Call reset() to start a new episode.")
        
        # Get current price with bounds checking
        current_idx = self.episode_start_idx + self.current_step
        if current_idx >= len(self.data):
            logger.error(f"Step index {current_idx} exceeds data length {len(self.data)}")
            current_idx = len(self.data) - 1
        
        row = self.data.iloc[current_idx]
        current_price = row['Close_raw'] if 'Close_raw' in self.data.columns else row['Close']
        
        # Check for position timeout and force close if needed
        force_close = False
        if self.portfolio.shares > 0 and self.position_start_step is not None:
            # Calculate position duration in steps (assuming hourly timeframe)
            steps_in_position = self.current_step - self.position_start_step
            # Convert max_position_days to steps (24 hours * max_days for hourly data)
            max_steps = self.max_position_days * 24
            
            if steps_in_position >= max_steps:
                # Force close position due to timeout
                action = TradingAction.SELL
                force_close = True
        
        # Track position start/end
        if action == TradingAction.BUY and self.portfolio.shares == 0:
            # Starting a new position
            self.position_start_step = self.current_step
        elif action == TradingAction.SELL and self.portfolio.shares > 0:
            # Closing position
            self.position_start_step = None
        
        # Execute trade
        trade_info = self.portfolio.execute_trade(action, current_price, self.max_position_size)
        
        # Calculate reward
        reward = self._calculate_reward(action, trade_info)
        
        # Apply penalty for forced position close
        if force_close:
            reward -= self.position_timeout_penalty
            trade_info['force_close'] = True
        
        # Update episode tracking
        # Update last portfolio value for next step's incremental reward
        self._last_portfolio_value = self.portfolio.get_total_value(current_price)
        self.current_step += 1
        portfolio_return = self.portfolio.get_return(current_price)
        self.episode_returns.append(portfolio_return)
        
        # Check if episode is done
        self.done = (
            self.current_step >= self.max_episode_steps or 
            self.episode_start_idx + self.current_step >= len(self.data) - 1 or
            self.portfolio.get_total_value(current_price) <= self.initial_balance * 0.5  # Stop if lose 50%
        )
        
        # Get next observation
        observation = self._get_observation()
        
        # Info dict
        info = {
            'portfolio_value': self.portfolio.get_total_value(current_price),
            'portfolio_return': portfolio_return,
            'balance': self.portfolio.balance,
            'shares': self.portfolio.shares,
            'action': action,
            'price': current_price,
            'trade_info': trade_info,
            'total_trades': self.portfolio.total_trades
        }
        
        return observation, reward, self.done, info
    
    def reset(self, start_idx: Optional[int] = None) -> np.ndarray:
        """Reset the environment to initial state."""
        # Reset portfolio
        self.portfolio.reset()
        
        # Reset episode tracking
        self.current_step = 0
        self.done = False
        self.episode_returns = []
        self._last_portfolio_value = self.initial_balance
        self.position_start_step = None  # Reset position tracking
        
        # Set episode start index
        if start_idx is not None:
            # Validate the provided start index
            min_start = self.lookback_window
            max_end = len(self.data) - 1
            max_start = max_end - self.max_episode_steps
            
            if start_idx < min_start or start_idx > max_start:
                logger.warning(f"Invalid start_idx {start_idx}, using random start")
                start_idx = None
            else:
                self.episode_start_idx = start_idx
        
        if start_idx is None:
            # Random start point, ensuring we have enough data for episode
            min_start = self.lookback_window
            required_length = self.max_episode_steps + self.lookback_window
            
            if len(self.data) < required_length:
                logger.warning(f"Insufficient data: {len(self.data)} < {required_length}")
                # Use what data we have, but adjust episode length
                self.episode_start_idx = min_start
                adjusted_max_steps = max(1, len(self.data) - min_start - 1)
                logger.warning(f"Reducing max_episode_steps from {self.max_episode_steps} to {adjusted_max_steps}")
                self.max_episode_steps = adjusted_max_steps
            else:
                max_start = len(self.data) - self.max_episode_steps - 1
                self.episode_start_idx = np.random.randint(min_start, max(min_start + 1, max_start))
        
        logger.debug(f"Reset environment. Episode start: {self.episode_start_idx}")
        
        return self._get_observation()
    
    def render(self, mode: str = 'human'):
        """Render the environment (for debugging)."""
        if mode == 'human':
            row = self.data.iloc[self.episode_start_idx + self.current_step]
            current_price = row['Close_raw'] if 'Close_raw' in row else row['Close']
            portfolio_value = self.portfolio.get_total_value(current_price)
            portfolio_return = self.portfolio.get_return(current_price)
            
            print(f"Step: {self.current_step}")
            print(f"Price: ${current_price:.2f}")
            print(f"Portfolio Value: ${portfolio_value:.2f}")
            print(f"Return: {portfolio_return:.2%}")
            print(f"Balance: ${self.portfolio.balance:.2f}")
            print(f"Shares: {self.portfolio.shares}")
            print("-" * 40)
    
    def get_episode_stats(self) -> Dict:
        """Get statistics for the current episode."""
        if not self.episode_returns:
            return {}
        
        returns_series = pd.Series(self.episode_returns)
        
        stats = {
            'total_return': returns_series.iloc[-1] if len(returns_series) > 0 else 0,
            'volatility': returns_series.std(),
            'sharpe_ratio': returns_series.mean() / returns_series.std() if returns_series.std() > 0 else 0,
            'max_drawdown': self._calculate_max_drawdown(),
            'total_trades': self.portfolio.total_trades,
            'final_balance': self.portfolio.balance,
            'final_shares': self.portfolio.shares
        }
        
        return stats
    
    def _calculate_max_drawdown(self) -> float:
        """Calculate maximum drawdown during episode."""
        if len(self.episode_returns) < 2:
            return 0
        
        values = [(1 + ret) * self.initial_balance for ret in self.episode_returns]
        peak = values[0]
        max_drawdown = 0
        
        for value in values:
            if value > peak:
                peak = value
            drawdown = (peak - value) / peak
            max_drawdown = max(max_drawdown, drawdown)
        
        return max_drawdown