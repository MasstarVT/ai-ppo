#!/usr/bin/env python3
"""
Setup script for the AI PPO Trading System.
Run this script to set up the environment and test the installation.
"""

import os
import sys
import subprocess
import shutil
from pathlib import Path

def run_command(command, description):
    """Run a command and handle errors."""
    print(f"\n{description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✓ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ {description} failed:")
        print(f"  Error: {e.stderr}")
        return False

def check_python_version():
    """Check Python version."""
    print("Checking Python version...")
    version = sys.version_info
    if version.major == 3 and version.minor >= 8:
        print(f"✓ Python {version.major}.{version.minor}.{version.micro} is supported")
        return True
    else:
        print(f"✗ Python {version.major}.{version.minor}.{version.micro} is not supported. Please use Python 3.8+")
        return False

def create_virtual_environment():
    """Create virtual environment."""
    venv_path = Path("venv")
    if venv_path.exists():
        print("✓ Virtual environment already exists")
        return True
    
    return run_command("python -m venv venv", "Creating virtual environment")

def activate_and_install_dependencies():
    """Install dependencies in virtual environment."""
    # Determine activation script path based on OS
    if os.name == 'nt':  # Windows
        activate_script = "venv\\Scripts\\activate"
        python_exe = "venv\\Scripts\\python"
        pip_exe = "venv\\Scripts\\pip"
    else:  # Unix-like systems
        activate_script = "venv/bin/activate"
        python_exe = "venv/bin/python"
        pip_exe = "venv/bin/pip"
    
    # Install dependencies
    commands = [
        (f"{pip_exe} install --upgrade pip", "Upgrading pip"),
        (f"{pip_exe} install -r requirements.txt", "Installing dependencies"),
    ]
    
    for command, description in commands:
        if not run_command(command, description):
            return False
    
    return True

def create_config_file():
    """Create configuration file from template."""
    config_template = Path("config/config_template.yaml")
    config_file = Path("config/config.yaml")
    
    if config_file.exists():
        print("✓ Configuration file already exists")
        return True
    
    if config_template.exists():
        try:
            shutil.copy(config_template, config_file)
            print("✓ Configuration file created from template")
            print("  Please edit config/config.yaml with your API credentials and preferences")
            return True
        except Exception as e:
            print(f"✗ Failed to create configuration file: {e}")
            return False
    else:
        print("✗ Configuration template not found")
        return False

def test_imports():
    """Test importing main modules."""
    print("\nTesting module imports...")
    
    test_script = """
import sys
sys.path.append('src')

try:
    from data import DataClient
    print("✓ Data module imported successfully")
except ImportError as e:
    print(f"✗ Data module import failed: {e}")
    sys.exit(1)

try:
    from environments import TradingEnvironment
    print("✓ Environment module imported successfully")
except ImportError as e:
    print(f"✗ Environment module import failed: {e}")
    sys.exit(1)

try:
    from agents import PPOAgent
    print("✓ Agent module imported successfully")
except ImportError as e:
    print(f"✗ Agent module import failed: {e}")
    sys.exit(1)

try:
    from utils import ConfigManager
    print("✓ Utils module imported successfully")
except ImportError as e:
    print(f"✗ Utils module import failed: {e}")
    sys.exit(1)

print("✓ All modules imported successfully!")
"""
    
    # Write test script to temporary file
    with open("test_imports.py", "w") as f:
        f.write(test_script)
    
    try:
        # Determine python executable
        if os.name == 'nt':  # Windows
            python_exe = "venv\\Scripts\\python"
        else:  # Unix-like systems
            python_exe = "venv/bin/python"
        
        result = subprocess.run([python_exe, "test_imports.py"], 
                              capture_output=True, text=True, check=True)
        print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Import test failed: {e.stderr}")
        return False
    finally:
        # Clean up test file
        if os.path.exists("test_imports.py"):
            os.remove("test_imports.py")

def print_next_steps():
    """Print next steps for the user."""
    print("""
🎉 Setup completed successfully!

Next steps:
1. Edit config/config.yaml with your trading preferences and API credentials
2. Train a model: python src/train.py --config config/config.yaml
3. Run backtesting: python src/backtest.py --config config/config.yaml --model models/best_model.pt

Example commands:
- Train model: python src/train.py --config config/config.yaml
- Backtest model: python src/backtest.py --config config/config.yaml --model models/best_model.pt --create-dashboard
- Evaluate only: python src/train.py --config config/config.yaml --model models/best_model.pt --eval-only

Important notes:
⚠️  This is for educational purposes only. Never risk more than you can afford to lose.
⚠️  Always test with paper trading before using real money.
⚠️  Past performance does not guarantee future results.

Happy trading! 📈
""")

def main():
    """Main setup function."""
    print("AI PPO Trading System Setup")
    print("=" * 40)
    
    # Check Python version
    if not check_python_version():
        return 1
    
    # Create virtual environment
    if not create_virtual_environment():
        return 1
    
    # Install dependencies
    if not activate_and_install_dependencies():
        return 1
    
    # Create config file
    if not create_config_file():
        return 1
    
    # Test imports
    if not test_imports():
        print("\n⚠️  Import test failed. You may need to install additional dependencies.")
        print("   Try running: pip install -r requirements.txt")
    
    # Print next steps
    print_next_steps()
    
    return 0

if __name__ == "__main__":
    exit(main())