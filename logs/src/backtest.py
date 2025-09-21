"""
Backtesting script for evaluating trained models.
"""

import os
import sys
import argparse
import logging
from typing import Dict, List, Optional
import pandas as pd
from datetime import datetime

# Add src to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data import DataClient, prepare_features
from agents import PPOAgent
from evaluation.backtesting import Backtester, WalkForwardAnalysis
from visualization.plots import TradingVisualizer
from utils import ConfigManager, setup_logging, format_percentage, format_currency

logger = logging.getLogger(__name__)


def main():
    """Main backtesting function."""
    parser = argparse.ArgumentParser(description='Backtest trained PPO trading agent')
    parser.add_argument('--config', type=str, required=True, help='Path to configuration file')
    parser.add_argument('--model', type=str, required=True, help='Path to trained model')
    parser.add_argument('--symbol', type=str, help='Symbol to backtest (optional, uses config symbols if not specified)')
    parser.add_argument('--start-date', type=str, help='Start date for backtest (YYYY-MM-DD)')
    parser.add_argument('--end-date', type=str, help='End date for backtest (YYYY-MM-DD)')
    parser.add_argument('--initial-balance', type=float, default=10000, help='Initial balance for backtest')
    parser.add_argument('--output-dir', type=str, default='backtest_results', help='Output directory for results')
    parser.add_argument('--create-dashboard', action='store_true', help='Create visualization dashboard')
    parser.add_argument('--walk-forward', action='store_true', help='Run walk-forward analysis')
    
    args = parser.parse_args()
    
    try:
        # Load configuration
        config_manager = ConfigManager(args.config)
        config = config_manager.to_dict()
        
        # Setup logging
        log_dir = config.get('paths', {}).get('log_dir', 'logs')
        setup_logging(config, log_dir)
        
        logger.info("Starting backtesting...")
        logger.info(f"Model: {args.model}")
        logger.info(f"Config: {args.config}")
        
        # Initialize data client
        data_client = DataClient(config)
        
        # Determine symbols to test
        symbols = [args.symbol] if args.symbol else config.get('trading', {}).get('symbols', ['AAPL'])
        
        # Initialize backtester and visualizer
        backtester = Backtester(config)
        visualizer = TradingVisualizer(config)
        
        # Create output directory
        os.makedirs(args.output_dir, exist_ok=True)
        
        # Run backtests for each symbol
        all_results = []
        
        for symbol in symbols:
            logger.info(f"Backtesting {symbol}...")
            
            try:
                # Fetch data
                period = config.get('backtesting', {}).get('data_period', '5y')
                interval = config.get('trading', {}).get('timeframe', '1h')
                
                raw_data = data_client.get_historical_data(symbol, period, interval)
                
                if raw_data.empty:
                    logger.warning(f"No data available for {symbol}")
                    continue
                
                # Prepare features
                data = prepare_features(raw_data, config)
                
                if data.empty:
                    logger.warning(f"No valid feature data for {symbol}")
                    continue
                
                logger.info(f"Loaded {len(data)} data points for {symbol}")
                
                # Create agent and load model
                agent = PPOAgent(
                    obs_dim=len(data.columns) * config.get('environment', {}).get('lookback_window', 50),
                    action_dim=3,
                    config=config
                )
                agent.load(args.model)
                
                # Run backtest
                results = backtester.run_backtest(
                    agent=agent,
                    data=data,
                    start_date=args.start_date,
                    end_date=args.end_date,
                    initial_balance=args.initial_balance
                )
                
                # Add symbol information
                results['symbol'] = symbol
                all_results.append(results)
                
                # Print summary
                backtest_metrics = results.get('backtest_metrics', {})
                performance_metrics = results.get('performance_metrics', {})
                
                print(f"\n{symbol} BACKTEST RESULTS:")
                print("=" * 40)
                print(f"Total Return: {format_percentage(backtest_metrics.get('total_return', 0))}")
                print(f"Buy & Hold Return: {format_percentage(backtest_metrics.get('buy_and_hold_return', 0))}")
                print(f"Excess Return: {format_percentage(backtest_metrics.get('excess_return', 0))}")
                print(f"Sharpe Ratio: {performance_metrics.get('sharpe_ratio', 0):.3f}")
                print(f"Max Drawdown: {format_percentage(performance_metrics.get('max_drawdown', 0))}")
                print(f"Win Rate: {format_percentage(backtest_metrics.get('win_rate', 0))}")
                print(f"Total Trades: {backtest_metrics.get('total_trades', 0)}")
                
                # Save detailed results
                symbol_output_dir = os.path.join(args.output_dir, symbol)
                os.makedirs(symbol_output_dir, exist_ok=True)
                
                backtester.save_results(os.path.join(symbol_output_dir, 'backtest_results.csv'))
                
                # Create visualizations if requested
                if args.create_dashboard:
                    logger.info(f"Creating dashboard for {symbol}...")
                    visualizer.create_dashboard(results, symbol_output_dir)
                
                # Walk-forward analysis if requested
                if args.walk_forward:
                    logger.info(f"Running walk-forward analysis for {symbol}...")
                    
                    wfa = WalkForwardAnalysis(config)
                    wf_results = wfa.run_analysis(
                        agent_class=PPOAgent,
                        data=data,
                        train_window=252,  # 1 year
                        test_window=63,   # 3 months
                        step_size=21      # 1 month
                    )
                    
                    if wf_results:
                        stability_metrics = wfa.analyze_stability()
                        
                        print(f"\n{symbol} WALK-FORWARD ANALYSIS:")
                        print("=" * 40)
                        print(f"Periods Analyzed: {stability_metrics.get('periods_analyzed', 0)}")
                        print(f"Average Return: {format_percentage(stability_metrics.get('avg_return', 0))}")
                        print(f"Return Std Dev: {format_percentage(stability_metrics.get('std_return', 0))}")
                        print(f"Consistency Ratio: {format_percentage(stability_metrics.get('consistency_ratio', 0))}")
                        print(f"Average Sharpe: {stability_metrics.get('avg_sharpe', 0):.3f}")
                        
                        # Save walk-forward results
                        wf_df = pd.DataFrame([
                            {
                                'period': i,
                                'train_start': r['period']['train_start'],
                                'train_end': r['period']['train_end'], 
                                'test_start': r['period']['test_start'],
                                'test_end': r['period']['test_end'],
                                'return': r['backtest_metrics']['total_return'],
                                'sharpe': r['performance_metrics']['sharpe_ratio'],
                                'max_drawdown': r['performance_metrics']['max_drawdown']
                            }
                            for i, r in enumerate(wf_results)
                        ])
                        
                        wf_df.to_csv(os.path.join(symbol_output_dir, 'walk_forward_results.csv'), index=False)
                
            except Exception as e:
                logger.error(f"Error backtesting {symbol}: {e}")
                continue
        
        # Multi-symbol comparison if multiple symbols
        if len(all_results) > 1:
            logger.info("Creating multi-symbol comparison...")
            
            comparison_df = backtester.compare_strategies(
                all_results,
                [r['symbol'] for r in all_results]
            )
            
            comparison_df.to_csv(os.path.join(args.output_dir, 'symbol_comparison.csv'), index=False)
            
            # Create comparison plot
            if args.create_dashboard:
                visualizer.plot_performance_comparison(
                    all_results,
                    [r['symbol'] for r in all_results],
                    os.path.join(args.output_dir, 'performance_comparison.png')
                )
        
        # Overall summary
        if all_results:
            avg_return = sum(r['backtest_metrics']['total_return'] for r in all_results) / len(all_results)
            avg_sharpe = sum(r['performance_metrics']['sharpe_ratio'] for r in all_results) / len(all_results)
            avg_win_rate = sum(r['backtest_metrics']['win_rate'] for r in all_results) / len(all_results)
            
            print(f"\nOVERALL SUMMARY ({len(all_results)} symbols):")
            print("=" * 50)
            print(f"Average Return: {format_percentage(avg_return)}")
            print(f"Average Sharpe Ratio: {avg_sharpe:.3f}")
            print(f"Average Win Rate: {format_percentage(avg_win_rate)}")
            
            # Save overall summary
            summary_data = {
                'symbols_tested': len(all_results),
                'average_return': avg_return,
                'average_sharpe_ratio': avg_sharpe,
                'average_win_rate': avg_win_rate,
                'backtest_date': datetime.now().isoformat(),
                'model_path': args.model,
                'config_path': args.config
            }
            
            summary_df = pd.DataFrame([summary_data])
            summary_df.to_csv(os.path.join(args.output_dir, 'overall_summary.csv'), index=False)
        
        logger.info(f"Backtesting completed. Results saved to {args.output_dir}")
        
    except Exception as e:
        logger.error(f"Backtesting failed: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())