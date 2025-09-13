@echo off
echo ==========================================
echo           PPO Trading Agent - Easy Training
echo ==========================================
echo.

REM Check if we're in the right directory
if not exist "config\config.yaml" (
    echo Error: config.yaml not found!
    echo Please run setup_windows.bat first to create the configuration.
    echo.
    pause
    exit /b 1
)

REM Activate virtual environment if it exists
if exist "venv\Scripts\activate.bat" (
    echo Activating virtual environment...
    call venv\Scripts\activate.bat
)

echo Starting training with default configuration...
echo Configuration file: config\config.yaml
echo.

REM Run the easy training script
python train_easy.py

if %ERRORLEVEL% neq 0 (
    echo.
    echo Training failed! Check the error messages above.
    echo.
    pause
    exit /b 1
)

echo.
echo Training completed successfully!
echo Check the models\ directory for saved models.
echo Check the logs\ directory for training logs.
echo.
pause