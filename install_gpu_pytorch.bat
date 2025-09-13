@echo off
echo ==========================================
echo     Install GPU-Enabled PyTorch for PPO Trading
echo ==========================================
echo.

echo This script will install PyTorch with CUDA support for faster GPU training.
echo.

REM Check for NVIDIA GPU
echo Checking for NVIDIA GPU...
nvidia-smi >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo.
    echo ❌ NVIDIA GPU or drivers not detected!
    echo.
    echo To use GPU acceleration, you need:
    echo   1. NVIDIA GPU ^(GTX 1060 or better recommended^)
    echo   2. NVIDIA drivers ^(latest version^)
    echo   3. CUDA toolkit ^(will be installed with PyTorch^)
    echo.
    echo If you have an NVIDIA GPU, please:
    echo   1. Install latest NVIDIA drivers from nvidia.com
    echo   2. Restart your computer
    echo   3. Run this script again
    echo.
    echo For now, continuing with CPU-only installation...
    pause
    goto :cpu_only
)

echo ✅ NVIDIA GPU detected!
nvidia-smi
echo.

REM Activate virtual environment if it exists
if exist "venv\Scripts\activate.bat" (
    echo Activating virtual environment...
    call venv\Scripts\activate.bat
    echo.
)

echo Installing PyTorch with CUDA support...
echo This may take several minutes...
echo.

REM Check CUDA compatibility and install appropriate version
echo Detecting CUDA compatibility...
for /f "tokens=*" %%i in ('nvidia-smi --query-gpu=driver_version --format=csv,noheader,nounits') do set DRIVER_VERSION=%%i
echo Driver version: %DRIVER_VERSION%

REM Install PyTorch with CUDA 12.1 (most compatible)
echo Installing PyTorch with CUDA 12.1...
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

if %ERRORLEVEL% neq 0 (
    echo.
    echo ❌ GPU PyTorch installation failed!
    echo Falling back to CPU version...
    goto :cpu_only
)

echo.
echo ✅ GPU PyTorch installation completed!
goto :test_installation

:cpu_only
echo Installing CPU-only PyTorch...
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

:test_installation
echo.
echo Testing PyTorch installation...
python -c "import torch; print('PyTorch version:', torch.__version__); print('CUDA available:', torch.cuda.is_available()); print('GPU count:', torch.cuda.device_count() if torch.cuda.is_available() else 0)"

if %ERRORLEVEL% neq 0 (
    echo.
    echo ❌ PyTorch installation test failed!
    echo Please check the error messages above.
    pause
    exit /b 1
)

echo.
echo ==========================================
echo           Installation Summary
echo ==========================================

python -c "
import torch
print('✅ PyTorch successfully installed!')
print(f'   Version: {torch.__version__}')
print(f'   CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'   GPU count: {torch.cuda.device_count()}')
    for i in range(torch.cuda.device_count()):
        print(f'   GPU {i}: {torch.cuda.get_device_name(i)}')
    print('')
    print('🚀 Your system is ready for GPU-accelerated training!')
    print('   Expected speedup: 3-10x faster than CPU')
else:
    print('')
    print('💻 CPU-only installation complete.')
    print('   Consider upgrading to GPU for faster training.')
print('')
print('Next steps:')
print('  1. Run: python gpu_test.py')
print('  2. Run: python test_training.py')
print('  3. Start training: python src/train.py')
"

echo.
pause