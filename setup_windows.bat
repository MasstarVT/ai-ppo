@echo off
:: Setup script for AI PPO Trading System
:: This script sets up the environment from scratch

title AI PPO Trading System - Setup

echo.
echo =====================================================
echo       AI PPO Trading System - Setup
echo =====================================================
echo.

:: Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    echo.
    echo Please install Python 3.8 or higher from:
    echo https://www.python.org/downloads/
    echo.
    echo Make sure to check "Add Python to PATH" during installation
    echo.
    pause
    exit /b 1
)

echo Python version check passed!
python --version

:: Check if we're in the correct directory
if not exist "setup.py" (
    echo ERROR: setup.py not found
    echo Please run this script from the ai-ppo project directory
    echo.
    pause
    exit /b 1
)

echo Directory check passed!

:: Create virtual environment
echo.
echo Creating virtual environment...
if exist "venv" (
    echo Virtual environment already exists, removing old one...
    rmdir /s /q venv
)

python -m venv venv
if errorlevel 1 (
    echo ERROR: Failed to create virtual environment
    echo Please ensure you have Python 3.8+ installed
    pause
    exit /b 1
)

echo Virtual environment created successfully!

:: Activate virtual environment
echo.
echo Activating virtual environment...
call venv\Scripts\activate.bat

:: Upgrade pip
echo.
echo Upgrading pip...
python -m pip install --upgrade pip

:: Install requirements
echo.
echo Installing project dependencies...
echo This may take a few minutes...
pip install -r requirements.txt

if errorlevel 1 (
    echo.
    echo ERROR: Failed to install some dependencies
    echo Please check your internet connection and try again
    echo.
    pause
    exit /b 1
)

:: Create necessary directories
echo.
echo Creating project directories...
if not exist "config" mkdir config
if not exist "models" mkdir models
if not exist "logs" mkdir logs
if not exist "data" mkdir data

echo Directories created successfully!

:: Run setup verification
echo.
echo Running setup verification...
python setup.py

:: Create default configuration
echo.
echo Creating default configuration...
python -c "from src.utils import create_default_config; create_default_config('config/config.yaml')" 2>nul
if errorlevel 1 (
    echo Warning: Could not create default config automatically
    echo You can create it later through the GUI
)

echo.
echo =====================================================
echo           Setup Complete!
echo =====================================================
echo.
echo Your AI PPO Trading System is now ready to use!
echo.
echo To get started:
echo 1. Double-click "quick_start.bat" to launch the Web GUI
echo 2. Or double-click "run_ai_ppo.bat" for full menu options
echo.
echo The Web GUI provides an easy-to-use interface for:
echo - Configuring trading parameters
echo - Training AI models
echo - Running backtests
echo - Monitoring live trading (paper mode)
echo.
echo =====================================================
echo.
pause