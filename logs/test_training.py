"""
Quick training test - runs for just a few episodes to verify everything works.
"""

import os
import sys

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def main():
    """Run quick training test."""
    print("Starting quick PPO Trading Agent test...")
    print("This will run for just a few episodes to verify everything works.")
    
    # Check if config exists
    config_path = "config/config.yaml"
    if not os.path.exists(config_path):
        print(f"Error: Configuration file '{config_path}' not found.")
        print("Please run 'setup_windows.bat' first to generate the config file.")
        return 1
    
    try:
        from train import TradingTrainer
        from utils import ConfigManager
        
        # Load configuration
        config_manager = ConfigManager(config_path)
        config = config_manager.to_dict()
        
        # Override config for quick test
        config['training']['total_timesteps'] = 5000  # Just 5000 steps
        config['training']['eval_freq'] = 2000  # Evaluate after 2000 steps
        config['training']['log_interval'] = 500  # Log every 500 steps
        
        print("Configuration loaded successfully")
        print(f"Training for {config['training']['total_timesteps']} timesteps")
        
        # Create trainer
        trainer = TradingTrainer(config)
        print("Trainer created successfully")
        
        # Start training
        results = trainer.train()
        
        print("\n" + "="*50)
        print("QUICK TEST COMPLETED SUCCESSFULLY!")
        print("="*50)
        print(f"Final Return: {results['final_metrics']['avg_return']:.2%}")
        print(f"Final Win Rate: {results['final_metrics']['win_rate']:.2%}")
        print("All systems are working correctly!")
        
        return 0
        
    except Exception as e:
        print(f"Test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit(main())