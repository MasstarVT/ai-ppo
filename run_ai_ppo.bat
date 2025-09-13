@echo off
:: AI PPO Trading System Launcher for Windows
:: This batch file sets up and runs the AI PPO Trading System

title AI PPO Trading System Launcher

echo.
echo =====================================================
echo       AI PPO Trading System Launcher
echo =====================================================
echo.

:: Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python 3.8+ from https://python.org
    echo.
    pause
    exit /b 1
)

echo [1/5] Python version check...
python --version

:: Check if we're in the correct directory
if not exist "setup.py" (
    echo ERROR: setup.py not found. Please run this script from the ai-ppo directory
    echo.
    pause
    exit /b 1
)

echo [2/5] Directory check passed

:: Create virtual environment if it doesn't exist
if not exist "venv" (
    echo [3/5] Creating virtual environment...
    python -m venv venv
    if errorlevel 1 (
        echo ERROR: Failed to create virtual environment
        pause
        exit /b 1
    )
) else (
    echo [3/5] Virtual environment already exists
)

:: Activate virtual environment
echo [4/5] Activating virtual environment...
call venv\Scripts\activate.bat

:: Install/upgrade requirements
echo [5/5] Installing/updating dependencies...
pip install --upgrade pip
pip install -r requirements.txt

if errorlevel 1 (
    echo ERROR: Failed to install dependencies
    echo.
    pause
    exit /b 1
)

:: Create necessary directories
if not exist "config" mkdir config
if not exist "models" mkdir models
if not exist "logs" mkdir logs
if not exist "data" mkdir data

:: Run setup to verify installation
echo.
echo Running system setup verification...
python setup.py

if errorlevel 1 (
    echo WARNING: Setup verification had issues, but continuing...
)

:: Create default config if it doesn't exist
if not exist "config\config.yaml" (
    echo.
    echo Creating default configuration...
    python -c "from src.utils import create_default_config; create_default_config('config/config.yaml')"
)

echo.
echo =====================================================
echo           Setup Complete!
echo =====================================================
echo.
echo Choose an option:
echo.
echo 1) Launch Web GUI (Recommended)
echo 2) Run Demo Script
echo 3) Train Model
echo 4) Run Backtesting
echo 5) Exit
echo.
set /p choice="Enter your choice (1-5): "

if "%choice%"=="1" goto launch_gui
if "%choice%"=="2" goto run_demo
if "%choice%"=="3" goto train_model
if "%choice%"=="4" goto run_backtest
if "%choice%"=="5" goto exit_script

echo Invalid choice. Launching Web GUI by default...
goto launch_gui

:launch_gui
echo.
echo =====================================================
echo          Launching Web GUI Dashboard
echo =====================================================
echo.
echo The web interface will open in your default browser
echo URL: http://localhost:8501
echo.
echo To stop the GUI, press Ctrl+C in this window
echo.
cd gui
python run_gui.py
goto end

:run_demo
echo.
echo =====================================================
echo             Running Demo Script
echo =====================================================
echo.
python demo.py
echo.
echo Demo completed. Press any key to continue...
pause >nul
goto menu

:train_model
echo.
echo =====================================================
echo            Starting Model Training
echo =====================================================
echo.
echo This will start training a PPO model. This may take a while...
echo Press Ctrl+C to stop training at any time.
echo.
python src/train.py
echo.
echo Training completed. Press any key to continue...
pause >nul
goto menu

:run_backtest
echo.
echo =====================================================
echo            Running Backtesting
echo =====================================================
echo.
python src/backtest.py
echo.
echo Backtesting completed. Press any key to continue...
pause >nul
goto menu

:menu
echo.
echo Returning to main menu...
echo.
echo Choose an option:
echo.
echo 1) Launch Web GUI (Recommended)
echo 2) Run Demo Script
echo 3) Train Model
echo 4) Run Backtesting
echo 5) Exit
echo.
set /p choice="Enter your choice (1-5): "

if "%choice%"=="1" goto launch_gui
if "%choice%"=="2" goto run_demo
if "%choice%"=="3" goto train_model
if "%choice%"=="4" goto run_backtest
if "%choice%"=="5" goto exit_script

echo Invalid choice. Please try again.
goto menu

:exit_script
echo.
echo Thank you for using AI PPO Trading System!
echo.

:end
echo.
echo Press any key to exit...
pause >nul