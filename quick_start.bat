@echo off
:: Quick launcher for AI PPO Trading System Web GUI
:: This is a simplified version that just launches the web interface

title AI PPO Trading System - Web GUI

echo.
echo =====================================================
echo       AI PPO Trading System - Quick Start
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

:: Check if we're in the correct directory
if not exist "setup.py" (
    echo ERROR: Please run this script from the ai-ppo directory
    echo.
    pause
    exit /b 1
)

:: Activate virtual environment if it exists
if exist "venv\Scripts\activate.bat" (
    echo Activating virtual environment...
    call venv\Scripts\activate.bat
)

:: Quick dependency check
echo Checking dependencies...
pip show streamlit >nul 2>&1
if errorlevel 1 (
    echo Installing missing dependencies...
    pip install -r requirements.txt
)

echo.
echo =====================================================
echo          Launching Web GUI Dashboard
echo =====================================================
echo.
echo The web interface will open at: http://localhost:8501
echo.
echo To stop the GUI, press Ctrl+C in this window
echo.

:: Launch the GUI
cd gui
python run_gui.py

echo.
echo GUI stopped. Press any key to exit...
pause >nul