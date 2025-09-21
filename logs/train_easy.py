"""
Easy training script - automatically uses default config.
"""

import os
import sys

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def main():
    """Run training with default config."""
    print("Starting PPO Trading Agent Training...")
    print("Using default configuration: config/config.yaml")
    
    # Check if config exists
    config_path = "config/config.yaml"
    if not os.path.exists(config_path):
        print(f"Error: Configuration file '{config_path}' not found.")
        print("Please run 'setup_windows.bat' first to generate the config file.")
        return 1
    
    # Import and run the main training
    try:
        from train import main as train_main
        import argparse
        
        # Mock the args to use default config
        sys.argv = ['train_easy.py', '--config', config_path]
        return train_main()
        
    except Exception as e:
        print(f"Training failed: {e}")
        print("Please ensure all dependencies are installed by running 'setup_windows.bat'")
        return 1

if __name__ == "__main__":
    exit(main())