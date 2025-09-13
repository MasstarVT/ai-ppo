"""
Main training script for the PPO trading agent.
"""

import os
import sys
import argparse
import logging
from typing import Dict, List, Optional
import numpy as np
import pandas as pd
from datetime import datetime

# Add src to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data import DataClient, TechnicalIndicators, prepare_features
from environments import TradingEnvironment
from agents import PPOAgent
from utils import (
    ConfigManager, setup_logging, set_seed, Timer, ProgressTracker,
    calculate_performance_metrics, format_currency, format_percentage,
    log_system_info
)

logger = logging.getLogger(__name__)


class TradingTrainer:
    """Main trainer class for the PPO trading agent."""
    
    def __init__(self, config: Dict):
        self.config = config
        self.data_client = None
        self.env = None
        self.agent = None
        self.training_data = {}
        
        # Training metrics
        self.episode_rewards = []
        self.episode_returns = []
        self.episode_sharpe_ratios = []
        self.training_metrics = []
        
        # Setup
        self._setup_environment()
        self._initialize_components()
    
    def _setup_environment(self):
        """Setup training environment."""
        # Set random seed
        seed = self.config.get('training', {}).get('seed', 42)
        set_seed(seed)
        
        # Setup logging
        log_dir = self.config.get('paths', {}).get('log_dir', 'logs')
        setup_logging(self.config, log_dir)
        
        # Log system info
        log_system_info()
        
        logger.info("Training environment setup complete")
    
    def _initialize_components(self):
        """Initialize training components."""
        try:
            # Initialize data client
            self.data_client = DataClient(self.config)
            logger.info("Data client initialized")
            
        except Exception as e:
            logger.error(f"Error initializing components: {e}")
            raise
    
    def load_data(self) -> Dict[str, pd.DataFrame]:
        """Load and prepare training data."""
        logger.info("Loading training data...")
        
        try:
            symbols = self.config.get('trading', {}).get('symbols', ['AAPL'])
            period = self.config.get('training', {}).get('data_period', '2y')
            interval = self.config.get('trading', {}).get('timeframe', '1h')
            
            # Fetch data for all symbols
            with Timer("Data fetching"):
                raw_data = self.data_client.get_multiple_symbols_data(
                    symbols, period, interval
                )
            
            if not raw_data:
                raise ValueError("No data was fetched for any symbol")
            
            # Prepare features for each symbol
            prepared_data = {}
            
            for symbol, df in raw_data.items():
                logger.info(f"Preparing features for {symbol} ({len(df)} bars)")
                
                try:
                    # Calculate technical indicators and prepare features
                    df_features = prepare_features(df, self.config)
                    
                    if not df_features.empty:
                        prepared_data[symbol] = df_features
                        logger.info(f"Prepared {len(df_features)} feature bars for {symbol}")
                    else:
                        logger.warning(f"No valid feature data for {symbol}")
                        
                except Exception as e:
                    logger.error(f"Error preparing features for {symbol}: {e}")
                    continue
            
            if not prepared_data:
                raise ValueError("No valid data after feature preparation")
            
            self.training_data = prepared_data
            logger.info(f"Data loading complete. Total symbols: {len(prepared_data)}")
            
            return prepared_data
            
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise
    
    def create_environment(self, data: pd.DataFrame) -> TradingEnvironment:
        """Create trading environment."""
        try:
            env = TradingEnvironment(data, self.config)
            logger.info(f"Created trading environment with observation space: {env.observation_space.shape}")
            return env
        except Exception as e:
            logger.error(f"Error creating environment: {e}")
            raise
    
    def create_agent(self, obs_dim: int, action_dim: int) -> PPOAgent:
        """Create PPO agent."""
        try:
            agent = PPOAgent(obs_dim, action_dim, self.config)
            logger.info("PPO agent created successfully")
            return agent
        except Exception as e:
            logger.error(f"Error creating agent: {e}")
            raise
    
    def train_episode(self, env: TradingEnvironment, agent: PPOAgent, episode_num: int) -> Dict:
        """Train a single episode."""
        obs = env.reset()
        episode_reward = 0
        episode_length = 0
        
        while not env.done:
            # Get action from agent
            action, log_prob, value = agent.get_action(obs)
            
            # Take step in environment
            next_obs, reward, done, info = env.step(action)
            
            # Store experience
            agent.store_experience(obs, action, reward, value, log_prob, done)
            
            # Update tracking
            episode_reward += reward
            episode_length += 1
            obs = next_obs
        
        # Finish episode
        agent.finish_episode(obs)
        
        # Get episode statistics
        episode_stats = env.get_episode_stats()
        episode_stats.update({
            'episode_reward': episode_reward,
            'episode_length': episode_length,
            'episode_num': episode_num
        })
        
        return episode_stats
    
    def evaluate_agent(self, agent: PPOAgent, n_episodes: int = 10) -> Dict:
        """Evaluate agent performance."""
        logger.info(f"Evaluating agent for {n_episodes} episodes...")
        
        eval_results = {
            'episode_returns': [],
            'episode_rewards': [],
            'episode_lengths': [],
            'portfolio_values': [],
            'sharpe_ratios': []
        }
        
        # Use random symbols and time periods for evaluation
        symbols = list(self.training_data.keys())
        
        for episode in range(n_episodes):
            try:
                # Select random symbol and data
                symbol = np.random.choice(symbols)
                data = self.training_data[symbol]
                
                # Create environment
                env = self.create_environment(data)
                
                # Run episode
                obs = env.reset()
                episode_reward = 0
                
                while not env.done:
                    action, _, _ = agent.get_action(obs, deterministic=True)
                    obs, reward, done, info = env.step(action)
                    episode_reward += reward
                
                # Store results
                episode_stats = env.get_episode_stats()
                eval_results['episode_returns'].append(episode_stats.get('total_return', 0))
                eval_results['episode_rewards'].append(episode_reward)
                eval_results['episode_lengths'].append(env.current_step)
                eval_results['portfolio_values'].append(env.portfolio.get_total_value(info['price']))
                eval_results['sharpe_ratios'].append(episode_stats.get('sharpe_ratio', 0))
                
            except Exception as e:
                logger.warning(f"Error in evaluation episode {episode}: {e}")
                continue
        
        # Calculate average metrics
        avg_metrics = {
            'avg_return': np.mean(eval_results['episode_returns']),
            'avg_reward': np.mean(eval_results['episode_rewards']),
            'avg_length': np.mean(eval_results['episode_lengths']),
            'avg_portfolio_value': np.mean(eval_results['portfolio_values']),
            'avg_sharpe_ratio': np.mean(eval_results['sharpe_ratios']),
            'std_return': np.std(eval_results['episode_returns']),
            'win_rate': len([r for r in eval_results['episode_returns'] if r > 0]) / len(eval_results['episode_returns'])
        }
        
        logger.info(f"Evaluation complete. Avg return: {format_percentage(avg_metrics['avg_return'])}, "
                   f"Win rate: {format_percentage(avg_metrics['win_rate'])}")
        
        return avg_metrics
    
    def train(self):
        """Main training loop."""
        logger.info("Starting PPO training...")
        
        try:
            # Load data
            data = self.load_data()
            
            # Get combined data for environment creation
            combined_data = pd.concat(data.values(), axis=0).sort_index()
            
            # Create environment and agent
            self.env = self.create_environment(combined_data)
            self.agent = self.create_agent(
                self.env.observation_space.shape[0], 
                self.env.action_space.n
            )
            
            # Training parameters
            total_timesteps = self.config.get('training', {}).get('total_timesteps', 1000000)
            eval_freq = self.config.get('training', {}).get('eval_freq', 10000)
            save_freq = self.config.get('training', {}).get('save_freq', 50000)
            log_interval = self.config.get('training', {}).get('log_interval', 1000)
            n_eval_episodes = self.config.get('training', {}).get('n_eval_episodes', 10)
            
            # Training loop
            timestep = 0
            episode = 0
            best_return = -np.inf
            
            progress = ProgressTracker(total_timesteps, "Training", log_interval)
            
            while timestep < total_timesteps:
                # Select random symbol and data for this episode
                symbol = np.random.choice(list(data.keys()))
                episode_data = data[symbol]
                
                # Create fresh environment for this episode
                env = self.create_environment(episode_data)
                
                # Train episode
                episode_stats = self.train_episode(env, self.agent, episode)
                
                # Update timestep counter
                timestep += episode_stats['episode_length']
                episode += 1
                
                # Store metrics
                self.episode_rewards.append(episode_stats['episode_reward'])
                self.episode_returns.append(episode_stats.get('total_return', 0))
                self.episode_sharpe_ratios.append(episode_stats.get('sharpe_ratio', 0))
                
                # Update agent (PPO update)
                if timestep % self.agent.n_steps < episode_stats['episode_length']:
                    training_stats = self.agent.update()
                    self.training_metrics.append(training_stats)
                
                # Progress tracking
                progress.update(episode_stats['episode_length'])
                
                # Logging
                if episode % (log_interval // 100) == 0:
                    recent_returns = self.episode_returns[-10:] if len(self.episode_returns) >= 10 else self.episode_returns
                    avg_return = np.mean(recent_returns) if recent_returns else 0
                    
                    logger.info(f"Episode {episode}: Return {format_percentage(avg_return)}, "
                               f"Reward {episode_stats['episode_reward']:.2f}, "
                               f"Length {episode_stats['episode_length']}")
                
                # Evaluation
                if timestep % eval_freq == 0:
                    eval_metrics = self.evaluate_agent(self.agent, n_eval_episodes)
                    
                    logger.info(f"Evaluation at timestep {timestep}:")
                    logger.info(f"  Avg Return: {format_percentage(eval_metrics['avg_return'])}")
                    logger.info(f"  Win Rate: {format_percentage(eval_metrics['win_rate'])}")
                    logger.info(f"  Avg Sharpe: {eval_metrics['avg_sharpe_ratio']:.3f}")
                    
                    # Save best model
                    if eval_metrics['avg_return'] > best_return:
                        best_return = eval_metrics['avg_return']
                        model_path = os.path.join(
                            self.config.get('paths', {}).get('model_dir', 'models'),
                            'best_model.pt'
                        )
                        self.agent.save(model_path)
                        logger.info(f"New best model saved with return: {format_percentage(best_return)}")
                
                # Periodic save
                if timestep % save_freq == 0:
                    model_path = os.path.join(
                        self.config.get('paths', {}).get('model_dir', 'models'),
                        f'model_timestep_{timestep}.pt'
                    )
                    self.agent.save(model_path)
            
            logger.info("Training completed successfully!")
            
            # Final evaluation
            final_metrics = self.evaluate_agent(self.agent, n_eval_episodes * 2)
            logger.info(f"Final evaluation: Return {format_percentage(final_metrics['avg_return'])}, "
                       f"Win Rate {format_percentage(final_metrics['win_rate'])}")
            
            return {
                'final_metrics': final_metrics,
                'training_history': {
                    'episode_returns': self.episode_returns,
                    'episode_rewards': self.episode_rewards,
                    'training_metrics': self.training_metrics
                }
            }
            
        except Exception as e:
            logger.error(f"Error during training: {e}")
            raise


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Train PPO trading agent')
    parser.add_argument('--config', type=str, required=True, help='Path to configuration file')
    parser.add_argument('--resume', type=str, help='Path to model checkpoint to resume from')
    parser.add_argument('--eval-only', action='store_true', help='Only run evaluation')
    
    args = parser.parse_args()
    
    try:
        # Load configuration
        config_manager = ConfigManager(args.config)
        config = config_manager.to_dict()
        
        # Create trainer
        trainer = TradingTrainer(config)
        
        if args.eval_only:
            # Load model and evaluate
            if not args.resume:
                raise ValueError("Model path required for evaluation")
            
            # Load data and create environment
            data = trainer.load_data()
            combined_data = pd.concat(data.values(), axis=0).sort_index()
            trainer.env = trainer.create_environment(combined_data)
            trainer.agent = trainer.create_agent(
                trainer.env.observation_space.shape[0],
                trainer.env.action_space.n
            )
            
            # Load model
            trainer.agent.load(args.resume)
            
            # Evaluate
            eval_metrics = trainer.evaluate_agent(trainer.agent, 50)
            
            print("Evaluation Results:")
            print(f"Average Return: {format_percentage(eval_metrics['avg_return'])}")
            print(f"Win Rate: {format_percentage(eval_metrics['win_rate'])}")
            print(f"Average Sharpe Ratio: {eval_metrics['avg_sharpe_ratio']:.3f}")
            
        else:
            # Train model
            if args.resume:
                # Load existing model
                trainer.load_data()
                combined_data = pd.concat(trainer.training_data.values(), axis=0).sort_index()
                trainer.env = trainer.create_environment(combined_data)
                trainer.agent = trainer.create_agent(
                    trainer.env.observation_space.shape[0],
                    trainer.env.action_space.n
                )
                trainer.agent.load(args.resume)
                logger.info(f"Resumed training from {args.resume}")
            
            # Start training
            results = trainer.train()
            
            print("Training completed!")
            print(f"Final Return: {format_percentage(results['final_metrics']['avg_return'])}")
            print(f"Final Win Rate: {format_percentage(results['final_metrics']['win_rate'])}")
    
    except Exception as e:
        logger.error(f"Training failed: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())