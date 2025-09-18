"""
Backtesting functionality for trading strategies.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import logging
from datetime import datetime, timedelta
import os

try:
    from src.environments import TradingEnvironment, Portfolio
    from src.data import DataClient, prepare_features
    from src.utils import calculate_performance_metrics, format_currency, format_percentage
except Exception:
    from environments import TradingEnvironment, Portfolio
    from data import DataClient, prepare_features
    from utils import calculate_performance_metrics, format_currency, format_percentage

logger = logging.getLogger(__name__)


class Backtester:
    """Comprehensive backtesting framework for trading strategies."""
    
    def __init__(self, config: Dict):
        self.config = config
        self.results = {}
        self.trades = []
        self.portfolio_history = []
        
    def run_backtest(
        self, 
        agent, 
        data: pd.DataFrame, 
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        initial_balance: float = 10000,
        deterministic: bool = True
    ) -> Dict[str, Any]:
        """
        Run comprehensive backtest on historical data.
        
        Args:
            agent: Trained trading agent
            data: Historical price data with features
            start_date: Start date for backtest (optional)
            end_date: End date for backtest (optional)
            initial_balance: Starting capital
            
        Returns:
            Dictionary containing backtest results
        """
        logger.info("Starting backtest...")
        
        try:
            # Filter data by date range if specified
            test_data = self._filter_data_by_date(data, start_date, end_date)
            
            if test_data.empty:
                raise ValueError("No data available for specified date range")
            
            logger.info(f"Backtesting on {len(test_data)} data points from {test_data.index[0]} to {test_data.index[-1]}")
            
            # Create environment
            env = TradingEnvironment(test_data, self.config)
            env.portfolio.initial_balance = initial_balance
            env.portfolio.balance = initial_balance
            
            # Run backtest
            obs = env.reset(start_idx=0)  # Start from beginning of data
            
            backtest_results = {
                'trades': [],
                'portfolio_values': [],
                'returns': [],
                'actions': [],
                'prices': [],
                'dates': [],
                'positions': [],
                'balances': []
            }
            
            step = 0
            while not env.done and step < len(test_data) - 1:
                # Get action from agent (deterministic)
                action, _, _ = agent.get_action(obs, deterministic=deterministic)
                
                # Execute action
                obs, reward, done, info = env.step(action)
                
                # Record results
                current_date = test_data.index[env.episode_start_idx + env.current_step - 1]
                portfolio_value = env.portfolio.get_total_value(info['price'])
                portfolio_return = env.portfolio.get_return(info['price'])
                
                backtest_results['trades'].append(info.get('trade_info', {}))
                backtest_results['portfolio_values'].append(portfolio_value)
                backtest_results['returns'].append(portfolio_return)
                backtest_results['actions'].append(action)
                backtest_results['prices'].append(info['price'])
                backtest_results['dates'].append(current_date)
                backtest_results['positions'].append(env.portfolio.shares)
                backtest_results['balances'].append(env.portfolio.balance)
                
                step += 1
            
            # Calculate performance metrics based on portfolio value changes
            portfolio_series = pd.Series(backtest_results['portfolio_values'], index=backtest_results['dates'])
            daily_returns = portfolio_series.pct_change().dropna()
            performance_metrics = calculate_performance_metrics(daily_returns)
            
            # Additional backtest metrics
            total_trades = len([t for t in backtest_results['trades'] if t.get('shares_traded', 0) != 0])
            winning_trades = len([t for t in backtest_results['trades'] 
                                if t.get('shares_traded', 0) != 0 and 
                                t.get('total_value_after', 0) > t.get('total_value_before', 0)])
            
            # Compute total return using initial balance as baseline
            total_return_calc = 0.0
            if backtest_results['portfolio_values']:
                pvn = backtest_results['portfolio_values'][-1]
                if initial_balance > 0:
                    total_return_calc = (pvn - initial_balance) / initial_balance

            backtest_metrics = {
                'start_date': backtest_results['dates'][0] if backtest_results['dates'] else None,
                'end_date': backtest_results['dates'][-1] if backtest_results['dates'] else None,
                'initial_balance': initial_balance,
                'final_balance': backtest_results['balances'][-1] if backtest_results['balances'] else initial_balance,
                'final_portfolio_value': backtest_results['portfolio_values'][-1] if backtest_results['portfolio_values'] else initial_balance,
                'total_return': total_return_calc,
                'total_trades': total_trades,
                'winning_trades': winning_trades,
                'win_rate': winning_trades / total_trades if total_trades > 0 else 0,
                'buy_and_hold_return': self._calculate_buy_and_hold_return(test_data),
                'excess_return': 0  # Will be calculated below
            }
            
            backtest_metrics['excess_return'] = backtest_metrics['total_return'] - backtest_metrics['buy_and_hold_return']
            
            # Combine all results
            final_results = {
                'backtest_metrics': backtest_metrics,
                'performance_metrics': performance_metrics,
                'detailed_results': backtest_results,
                'summary': self._create_summary(backtest_metrics, performance_metrics)
            }
            
            self.results = final_results
            
            logger.info(f"Backtest completed. Total return: {format_percentage(backtest_metrics['total_return'])}, "
                       f"Win rate: {format_percentage(backtest_metrics['win_rate'])}")
            
            return final_results
            
        except Exception as e:
            logger.error(f"Error during backtesting: {e}")
            raise
    
    def _filter_data_by_date(self, data: pd.DataFrame, start_date: Optional[str], end_date: Optional[str]) -> pd.DataFrame:
        """Filter data by date range."""
        filtered_data = data.copy()
        try:
            idx = filtered_data.index
            tz = getattr(idx, 'tz', None)
            start_ts = None
            end_ts = None
            if start_date:
                start_ts = pd.Timestamp(start_date)
            if end_date:
                # make end inclusive to the end of day
                end_ts = pd.Timestamp(end_date) + pd.Timedelta(days=1) - pd.Timedelta(microseconds=1)
            if tz is not None:
                if start_ts is not None and start_ts.tz is None:
                    start_ts = start_ts.tz_localize(tz)
                if end_ts is not None and end_ts.tz is None:
                    end_ts = end_ts.tz_localize(tz)
            # Apply filters safely
            if start_ts is not None:
                filtered_data = filtered_data[idx >= start_ts]
                idx = filtered_data.index
            if end_ts is not None:
                filtered_data = filtered_data[idx <= end_ts]
            return filtered_data
        except Exception as e:
            logger.warning(f"Date filtering fallback due to: {e}")
            return filtered_data
    
    def _calculate_buy_and_hold_return(self, data: pd.DataFrame) -> float:
        """Calculate buy and hold return for comparison."""
        if len(data) < 2:
            return 0
        # Prefer raw close price if present (pre-normalization)
        price_col = 'Close_raw' if 'Close_raw' in data.columns else 'Close'
        start_price = data[price_col].iloc[0]
        end_price = data[price_col].iloc[-1]
        
        return (end_price - start_price) / start_price
    
    def _create_summary(self, backtest_metrics: Dict, performance_metrics: Dict) -> str:
        """Create human-readable summary."""
        summary = f"""
BACKTEST SUMMARY
================
Period: {backtest_metrics['start_date']} to {backtest_metrics['end_date']}
Initial Balance: {format_currency(backtest_metrics['initial_balance'])}
Final Portfolio Value: {format_currency(backtest_metrics['final_portfolio_value'])}

RETURNS
=======
Total Return: {format_percentage(backtest_metrics['total_return'])}
Buy & Hold Return: {format_percentage(backtest_metrics['buy_and_hold_return'])}
Excess Return: {format_percentage(backtest_metrics['excess_return'])}
Annual Return: {format_percentage(performance_metrics.get('annual_return', 0))}

RISK METRICS
============
Volatility: {format_percentage(performance_metrics.get('volatility', 0))}
Sharpe Ratio: {performance_metrics.get('sharpe_ratio', 0):.3f}
Sortino Ratio: {performance_metrics.get('sortino_ratio', 0):.3f}
Max Drawdown: {format_percentage(performance_metrics.get('max_drawdown', 0))}
Calmar Ratio: {performance_metrics.get('calmar_ratio', 0):.3f}

TRADING ACTIVITY
================
Total Trades: {backtest_metrics['total_trades']}
Winning Trades: {backtest_metrics['winning_trades']}
Win Rate: {format_percentage(backtest_metrics['win_rate'])}
"""
        return summary
    
    def save_results(self, filepath: str):
        """Save backtest results to file."""
        try:
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            
            # Save detailed results to CSV
            if self.results and 'detailed_results' in self.results:
                detailed_df = pd.DataFrame(self.results['detailed_results'])
                detailed_df.to_csv(filepath.replace('.csv', '_detailed.csv'), index=False)
            
            # Save summary results
            summary_data = {}
            if 'backtest_metrics' in self.results:
                summary_data.update(self.results['backtest_metrics'])
            if 'performance_metrics' in self.results:
                summary_data.update(self.results['performance_metrics'])
            
            summary_df = pd.DataFrame([summary_data])
            summary_df.to_csv(filepath, index=False)
            
            logger.info(f"Backtest results saved to {filepath}")
            
        except Exception as e:
            logger.error(f"Error saving backtest results: {e}")
    
    def compare_strategies(self, results_list: List[Dict], strategy_names: List[str]) -> pd.DataFrame:
        """Compare multiple backtesting results."""
        try:
            comparison_data = []
            
            for i, results in enumerate(results_list):
                strategy_name = strategy_names[i] if i < len(strategy_names) else f"Strategy_{i+1}"
                
                metrics = {}
                metrics['Strategy'] = strategy_name
                
                # Add backtest metrics
                if 'backtest_metrics' in results:
                    metrics.update(results['backtest_metrics'])
                
                # Add performance metrics
                if 'performance_metrics' in results:
                    metrics.update(results['performance_metrics'])
                
                comparison_data.append(metrics)
            
            comparison_df = pd.DataFrame(comparison_data)
            
            # Reorder columns for better readability
            important_columns = [
                'Strategy', 'total_return', 'annual_return', 'volatility', 
                'sharpe_ratio', 'max_drawdown', 'win_rate', 'total_trades'
            ]
            
            available_columns = [col for col in important_columns if col in comparison_df.columns]
            other_columns = [col for col in comparison_df.columns if col not in available_columns]
            
            comparison_df = comparison_df[available_columns + other_columns]
            
            return comparison_df
            
        except Exception as e:
            logger.error(f"Error comparing strategies: {e}")
            return pd.DataFrame()


class WalkForwardAnalysis:
    """Walk-forward analysis for strategy validation."""
    
    def __init__(self, config: Dict):
        self.config = config
        self.results = []
    
    def run_analysis(
        self, 
        agent_class,
        data: pd.DataFrame,
        train_window: int = 252,  # Trading days
        test_window: int = 21,   # Trading days
        step_size: int = 21      # Days to step forward
    ) -> List[Dict]:
        """
        Run walk-forward analysis.
        
        Args:
            agent_class: Class to create trading agents
            data: Historical data
            train_window: Training window size in days
            test_window: Testing window size in days
            step_size: Step size for walking forward
            
        Returns:
            List of backtest results for each period
        """
        logger.info("Starting walk-forward analysis...")
        
        try:
            results = []
            start_idx = train_window
            
            while start_idx + test_window < len(data):
                # Define training and testing periods
                train_start = start_idx - train_window
                train_end = start_idx
                test_start = start_idx
                test_end = min(start_idx + test_window, len(data))
                
                train_data = data.iloc[train_start:train_end]
                test_data = data.iloc[test_start:test_end]
                
                logger.info(f"Training: {train_data.index[0]} to {train_data.index[-1]}")
                logger.info(f"Testing: {test_data.index[0]} to {test_data.index[-1]}")
                
                try:
                    # Create and train agent (simplified - in practice you'd train here)
                    agent = agent_class(test_data.shape[1], 3, self.config)
                    
                    # Run backtest on test period
                    backtester = Backtester(self.config)
                    period_results = backtester.run_backtest(agent, test_data)
                    
                    # Add period information
                    period_results['period'] = {
                        'train_start': train_data.index[0],
                        'train_end': train_data.index[-1],
                        'test_start': test_data.index[0],
                        'test_end': test_data.index[-1]
                    }
                    
                    results.append(period_results)
                    
                except Exception as e:
                    logger.warning(f"Error in walk-forward period: {e}")
                    continue
                
                # Move to next period
                start_idx += step_size
            
            self.results = results
            
            logger.info(f"Walk-forward analysis completed. {len(results)} periods analyzed.")
            
            return results
            
        except Exception as e:
            logger.error(f"Error in walk-forward analysis: {e}")
            return []
    
    def analyze_stability(self) -> Dict:
        """Analyze strategy stability across periods."""
        if not self.results:
            return {}
        
        try:
            returns = [r['backtest_metrics']['total_return'] for r in self.results]
            sharpe_ratios = [r['performance_metrics']['sharpe_ratio'] for r in self.results]
            max_drawdowns = [r['performance_metrics']['max_drawdown'] for r in self.results]
            
            stability_metrics = {
                'avg_return': np.mean(returns),
                'std_return': np.std(returns),
                'consistency_ratio': len([r for r in returns if r > 0]) / len(returns),
                'avg_sharpe': np.mean(sharpe_ratios),
                'std_sharpe': np.std(sharpe_ratios),
                'avg_max_drawdown': np.mean(max_drawdowns),
                'worst_drawdown': min(max_drawdowns),
                'periods_analyzed': len(self.results)
            }
            
            return stability_metrics
            
        except Exception as e:
            logger.error(f"Error analyzing stability: {e}")
            return {}