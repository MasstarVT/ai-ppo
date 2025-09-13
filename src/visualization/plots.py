"""
Visualization tools for trading analysis.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
import logging
import os
from datetime import datetime

logger = logging.getLogger(__name__)

# Set style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")


class TradingVisualizer:
    """Comprehensive visualization tools for trading analysis."""
    
    def __init__(self, config: Dict):
        self.config = config
        self.fig_size = config.get('visualization', {}).get('figure_size', (12, 8))
        self.dpi = config.get('visualization', {}).get('dpi', 100)
        
    def plot_portfolio_performance(
        self, 
        backtest_results: Dict, 
        save_path: Optional[str] = None,
        show_trades: bool = True
    ) -> plt.Figure:
        """Plot portfolio performance over time."""
        try:
            fig, axes = plt.subplots(3, 1, figsize=(15, 12), sharex=True)
            
            detailed_results = backtest_results.get('detailed_results', {})
            dates = pd.to_datetime(detailed_results.get('dates', []))
            portfolio_values = detailed_results.get('portfolio_values', [])
            prices = detailed_results.get('prices', [])
            actions = detailed_results.get('actions', [])
            
            if not dates or not portfolio_values:
                logger.warning("No data available for portfolio performance plot")
                return fig
            
            # Portfolio value over time
            axes[0].plot(dates, portfolio_values, label='Portfolio Value', linewidth=2)
            axes[0].set_title('Portfolio Performance Over Time', fontsize=14, fontweight='bold')
            axes[0].set_ylabel('Portfolio Value ($)', fontsize=12)
            axes[0].grid(True, alpha=0.3)
            axes[0].legend()
            
            # Add trade markers if requested
            if show_trades and actions:
                buy_dates = [dates[i] for i, action in enumerate(actions) if action == 2]  # Buy = 2
                sell_dates = [dates[i] for i, action in enumerate(actions) if action == 0]  # Sell = 0
                
                buy_values = [portfolio_values[i] for i, action in enumerate(actions) if action == 2]
                sell_values = [portfolio_values[i] for i, action in enumerate(actions) if action == 0]
                
                if buy_dates:
                    axes[0].scatter(buy_dates, buy_values, color='green', marker='^', s=50, alpha=0.7, label='Buy')
                if sell_dates:
                    axes[0].scatter(sell_dates, sell_values, color='red', marker='v', s=50, alpha=0.7, label='Sell')
                
                axes[0].legend()
            
            # Price chart
            axes[1].plot(dates, prices, label='Price', color='orange', linewidth=1.5)
            axes[1].set_title('Stock Price', fontsize=14, fontweight='bold')
            axes[1].set_ylabel('Price ($)', fontsize=12)
            axes[1].grid(True, alpha=0.3)
            axes[1].legend()
            
            # Returns
            returns = detailed_results.get('returns', [])
            if returns:
                axes[2].plot(dates, [r * 100 for r in returns], label='Cumulative Return (%)', color='purple', linewidth=2)
                axes[2].axhline(y=0, color='black', linestyle='--', alpha=0.5)
                axes[2].set_title('Cumulative Returns', fontsize=14, fontweight='bold')
                axes[2].set_ylabel('Return (%)', fontsize=12)
                axes[2].set_xlabel('Date', fontsize=12)
                axes[2].grid(True, alpha=0.3)
                axes[2].legend()
            
            plt.tight_layout()
            
            if save_path:
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
                logger.info(f"Portfolio performance plot saved to {save_path}")
            
            return fig
            
        except Exception as e:
            logger.error(f"Error plotting portfolio performance: {e}")
            return plt.figure()
    
    def plot_drawdown_analysis(self, backtest_results: Dict, save_path: Optional[str] = None) -> plt.Figure:
        """Plot drawdown analysis."""
        try:
            fig, axes = plt.subplots(2, 1, figsize=(15, 10), sharex=True)
            
            detailed_results = backtest_results.get('detailed_results', {})
            dates = pd.to_datetime(detailed_results.get('dates', []))
            portfolio_values = detailed_results.get('portfolio_values', [])
            
            if not dates or not portfolio_values:
                logger.warning("No data available for drawdown analysis")
                return fig
            
            # Calculate drawdowns
            portfolio_series = pd.Series(portfolio_values, index=dates)
            rolling_max = portfolio_series.expanding().max()
            drawdown = (portfolio_series - rolling_max) / rolling_max
            
            # Portfolio value with peak markers
            axes[0].plot(dates, portfolio_values, label='Portfolio Value', linewidth=2)
            axes[0].plot(dates, rolling_max, label='Previous Peak', linestyle='--', alpha=0.7)
            axes[0].set_title('Portfolio Value and Peak Analysis', fontsize=14, fontweight='bold')
            axes[0].set_ylabel('Portfolio Value ($)', fontsize=12)
            axes[0].grid(True, alpha=0.3)
            axes[0].legend()
            
            # Drawdown
            axes[1].fill_between(dates, drawdown * 100, 0, alpha=0.3, color='red', label='Drawdown')
            axes[1].plot(dates, drawdown * 100, color='red', linewidth=1)
            axes[1].set_title('Drawdown Analysis', fontsize=14, fontweight='bold')
            axes[1].set_ylabel('Drawdown (%)', fontsize=12)
            axes[1].set_xlabel('Date', fontsize=12)
            axes[1].grid(True, alpha=0.3)
            
            # Add max drawdown line
            max_drawdown = drawdown.min() * 100
            axes[1].axhline(y=max_drawdown, color='darkred', linestyle='--', 
                          label=f'Max Drawdown: {max_drawdown:.2f}%')
            axes[1].legend()
            
            plt.tight_layout()
            
            if save_path:
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
                logger.info(f"Drawdown analysis plot saved to {save_path}")
            
            return fig
            
        except Exception as e:
            logger.error(f"Error plotting drawdown analysis: {e}")
            return plt.figure()
    
    def plot_returns_distribution(self, backtest_results: Dict, save_path: Optional[str] = None) -> plt.Figure:
        """Plot returns distribution analysis."""
        try:
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            
            detailed_results = backtest_results.get('detailed_results', {})
            returns = detailed_results.get('returns', [])
            
            if not returns or len(returns) < 2:
                logger.warning("Insufficient data for returns distribution analysis")
                return fig
            
            # Calculate daily returns
            returns_series = pd.Series(returns)
            daily_returns = returns_series.pct_change().dropna()
            
            # Returns histogram
            axes[0, 0].hist(daily_returns, bins=50, alpha=0.7, edgecolor='black')
            axes[0, 0].set_title('Daily Returns Distribution', fontsize=12, fontweight='bold')
            axes[0, 0].set_xlabel('Daily Return')
            axes[0, 0].set_ylabel('Frequency')
            axes[0, 0].grid(True, alpha=0.3)
            
            # Q-Q plot
            from scipy import stats
            stats.probplot(daily_returns, dist="norm", plot=axes[0, 1])
            axes[0, 1].set_title('Q-Q Plot (Normal Distribution)', fontsize=12, fontweight='bold')
            axes[0, 1].grid(True, alpha=0.3)
            
            # Rolling volatility
            rolling_vol = daily_returns.rolling(window=20).std() * np.sqrt(252)  # Annualized
            axes[1, 0].plot(rolling_vol, linewidth=2)
            axes[1, 0].set_title('Rolling Volatility (20-day)', fontsize=12, fontweight='bold')
            axes[1, 0].set_ylabel('Annualized Volatility')
            axes[1, 0].grid(True, alpha=0.3)
            
            # Monthly returns heatmap
            if len(daily_returns) > 30:
                monthly_returns = daily_returns.resample('M').sum()
                monthly_returns.index = monthly_returns.index.to_period('M')
                
                # Create pivot table for heatmap
                monthly_data = monthly_returns.to_frame('returns')
                monthly_data['year'] = monthly_returns.index.year
                monthly_data['month'] = monthly_returns.index.month
                
                if len(monthly_data['year'].unique()) > 1:
                    pivot_table = monthly_data.pivot(index='year', columns='month', values='returns')
                    
                    sns.heatmap(pivot_table, annot=True, fmt='.2%', cmap='RdYlGn', center=0, ax=axes[1, 1])
                    axes[1, 1].set_title('Monthly Returns Heatmap', fontsize=12, fontweight='bold')
                else:
                    axes[1, 1].text(0.5, 0.5, 'Insufficient data\nfor monthly heatmap', 
                                  transform=axes[1, 1].transAxes, ha='center', va='center')
                    axes[1, 1].set_title('Monthly Returns Heatmap', fontsize=12, fontweight='bold')
            else:
                axes[1, 1].text(0.5, 0.5, 'Insufficient data\nfor monthly heatmap', 
                              transform=axes[1, 1].transAxes, ha='center', va='center')
                axes[1, 1].set_title('Monthly Returns Heatmap', fontsize=12, fontweight='bold')
            
            plt.tight_layout()
            
            if save_path:
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
                logger.info(f"Returns distribution plot saved to {save_path}")
            
            return fig
            
        except Exception as e:
            logger.error(f"Error plotting returns distribution: {e}")
            return plt.figure()
    
    def plot_trading_activity(self, backtest_results: Dict, save_path: Optional[str] = None) -> plt.Figure:
        """Plot trading activity analysis."""
        try:
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            
            detailed_results = backtest_results.get('detailed_results', {})
            actions = detailed_results.get('actions', [])
            trades = detailed_results.get('trades', [])
            positions = detailed_results.get('positions', [])
            dates = pd.to_datetime(detailed_results.get('dates', []))
            
            if not actions:
                logger.warning("No trading activity data available")
                return fig
            
            # Action distribution
            action_names = ['Sell', 'Hold', 'Buy']
            action_counts = [actions.count(i) for i in range(3)]
            
            axes[0, 0].pie(action_counts, labels=action_names, autopct='%1.1f%%', startangle=90)
            axes[0, 0].set_title('Action Distribution', fontsize=12, fontweight='bold')
            
            # Position over time
            if positions and dates:
                axes[0, 1].plot(dates, positions, linewidth=2, label='Position Size')
                axes[0, 1].set_title('Position Size Over Time', fontsize=12, fontweight='bold')
                axes[0, 1].set_ylabel('Shares Held')
                axes[0, 1].grid(True, alpha=0.3)
                axes[0, 1].legend()
            
            # Trade frequency (monthly)
            if trades and dates:
                # Count actual trades (non-zero shares traded)
                actual_trades = [t for t in trades if t.get('shares_traded', 0) != 0]
                if actual_trades and len(dates) > 30:
                    trade_dates = [dates[i] for i, t in enumerate(trades) if t.get('shares_traded', 0) != 0]
                    if trade_dates:
                        trade_series = pd.Series(1, index=trade_dates)
                        monthly_trades = trade_series.resample('M').sum()
                        
                        axes[1, 0].bar(range(len(monthly_trades)), monthly_trades.values)
                        axes[1, 0].set_title('Monthly Trading Frequency', fontsize=12, fontweight='bold')
                        axes[1, 0].set_ylabel('Number of Trades')
                        axes[1, 0].set_xlabel('Month')
                        axes[1, 0].grid(True, alpha=0.3)
            
            # Trade size distribution
            if trades:
                trade_sizes = [abs(t.get('shares_traded', 0)) for t in trades if t.get('shares_traded', 0) != 0]
                if trade_sizes:
                    axes[1, 1].hist(trade_sizes, bins=20, alpha=0.7, edgecolor='black')
                    axes[1, 1].set_title('Trade Size Distribution', fontsize=12, fontweight='bold')
                    axes[1, 1].set_xlabel('Shares Traded')
                    axes[1, 1].set_ylabel('Frequency')
                    axes[1, 1].grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            if save_path:
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
                logger.info(f"Trading activity plot saved to {save_path}")
            
            return fig
            
        except Exception as e:
            logger.error(f"Error plotting trading activity: {e}")
            return plt.figure()
    
    def plot_performance_comparison(
        self, 
        results_list: List[Dict], 
        strategy_names: List[str],
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """Plot performance comparison between strategies."""
        try:
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            
            # Extract metrics for comparison
            metrics_data = []
            for i, results in enumerate(results_list):
                strategy_name = strategy_names[i] if i < len(strategy_names) else f"Strategy_{i+1}"
                
                backtest_metrics = results.get('backtest_metrics', {})
                performance_metrics = results.get('performance_metrics', {})
                
                metrics_data.append({
                    'strategy': strategy_name,
                    'total_return': backtest_metrics.get('total_return', 0),
                    'volatility': performance_metrics.get('volatility', 0),
                    'sharpe_ratio': performance_metrics.get('sharpe_ratio', 0),
                    'max_drawdown': abs(performance_metrics.get('max_drawdown', 0)),
                    'win_rate': backtest_metrics.get('win_rate', 0)
                })
            
            if not metrics_data:
                logger.warning("No metrics data available for comparison")
                return fig
            
            strategies = [d['strategy'] for d in metrics_data]
            
            # Total returns comparison
            returns = [d['total_return'] * 100 for d in metrics_data]
            axes[0, 0].bar(strategies, returns)
            axes[0, 0].set_title('Total Returns Comparison', fontsize=12, fontweight='bold')
            axes[0, 0].set_ylabel('Total Return (%)')
            axes[0, 0].tick_params(axis='x', rotation=45)
            axes[0, 0].grid(True, alpha=0.3)
            
            # Risk-return scatter
            volatilities = [d['volatility'] * 100 for d in metrics_data]
            axes[0, 1].scatter(volatilities, returns, s=100, alpha=0.7)
            for i, strategy in enumerate(strategies):
                axes[0, 1].annotate(strategy, (volatilities[i], returns[i]), 
                                  xytext=(5, 5), textcoords='offset points')
            axes[0, 1].set_title('Risk-Return Profile', fontsize=12, fontweight='bold')
            axes[0, 1].set_xlabel('Volatility (%)')
            axes[0, 1].set_ylabel('Total Return (%)')
            axes[0, 1].grid(True, alpha=0.3)
            
            # Sharpe ratios
            sharpe_ratios = [d['sharpe_ratio'] for d in metrics_data]
            axes[1, 0].bar(strategies, sharpe_ratios)
            axes[1, 0].set_title('Sharpe Ratio Comparison', fontsize=12, fontweight='bold')
            axes[1, 0].set_ylabel('Sharpe Ratio')
            axes[1, 0].tick_params(axis='x', rotation=45)
            axes[1, 0].grid(True, alpha=0.3)
            
            # Max drawdowns
            max_drawdowns = [d['max_drawdown'] * 100 for d in metrics_data]
            axes[1, 1].bar(strategies, max_drawdowns, color='red', alpha=0.7)
            axes[1, 1].set_title('Maximum Drawdown Comparison', fontsize=12, fontweight='bold')
            axes[1, 1].set_ylabel('Max Drawdown (%)')
            axes[1, 1].tick_params(axis='x', rotation=45)
            axes[1, 1].grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            if save_path:
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
                logger.info(f"Performance comparison plot saved to {save_path}")
            
            return fig
            
        except Exception as e:
            logger.error(f"Error plotting performance comparison: {e}")
            return plt.figure()
    
    def create_dashboard(self, backtest_results: Dict, save_dir: str):
        """Create comprehensive dashboard with all visualizations."""
        try:
            os.makedirs(save_dir, exist_ok=True)
            
            logger.info("Creating trading dashboard...")
            
            # Generate all plots
            self.plot_portfolio_performance(
                backtest_results, 
                os.path.join(save_dir, 'portfolio_performance.png')
            )
            
            self.plot_drawdown_analysis(
                backtest_results,
                os.path.join(save_dir, 'drawdown_analysis.png')
            )
            
            self.plot_returns_distribution(
                backtest_results,
                os.path.join(save_dir, 'returns_distribution.png')
            )
            
            self.plot_trading_activity(
                backtest_results,
                os.path.join(save_dir, 'trading_activity.png')
            )
            
            # Create summary report
            summary_text = backtest_results.get('summary', 'No summary available')
            
            with open(os.path.join(save_dir, 'summary_report.txt'), 'w') as f:
                f.write(summary_text)
                f.write("\n\nDashboard generated on: ")
                f.write(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
            
            logger.info(f"Dashboard created successfully in {save_dir}")
            
        except Exception as e:
            logger.error(f"Error creating dashboard: {e}")