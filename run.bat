@echo off
setlocal EnableExtensions EnableDelayedExpansion
chcp 65001 >nul
cd /d "%~dp0"

set "SMOKE_TEST=0"
if /I "%~1"=="--smoke-test" set "SMOKE_TEST=1"

if not exist ".venv\Scripts\python.exe" (
    echo [INFO] The Python environment is missing. Running automatic setup...
    call "%~dp0setup.bat" --no-pause
    if errorlevel 1 goto :fail
)

rem A copied .venv is not portable. Let setup repair it and any missing package.
".venv\Scripts\python.exe" -c "import numpy,torch,transformers,safetensors,cv2,PyQt6,sounddevice,librosa,ultralytics,vosk,onnx,onnxruntime" >nul 2>nul
if errorlevel 1 (
    echo [INFO] Runtime dependencies are missing or the copied .venv is invalid. Repairing...
    call "%~dp0setup.bat" --no-pause
    if errorlevel 1 goto :fail
)

if not defined DETECTOR_VIDEO_DEVICE (
    set "DEVICE_PATH_FILE=%TEMP%\detector-danger-device-%RANDOM%-%RANDOM%.txt"
    ".venv\Scripts\python.exe" -c "import torch; print('cuda:0' if torch.cuda.is_available() else 'cpu')" >"!DEVICE_PATH_FILE!"
    if errorlevel 1 goto :fail
    set /p "DETECTOR_VIDEO_DEVICE="<"!DEVICE_PATH_FILE!"
    del /q "!DEVICE_PATH_FILE!" >nul 2>nul
    if not defined DETECTOR_VIDEO_DEVICE goto :fail
)
echo [OK] Video inference device: %DETECTOR_VIDEO_DEVICE%

".venv\Scripts\python.exe" "%~dp0tools\bootstrap.py" --check-only
if errorlevel 1 (
    echo [INFO] A model is missing, incomplete, or cannot be opened. Repairing models...
    ".venv\Scripts\python.exe" "%~dp0tools\bootstrap.py"
    if errorlevel 1 goto :model_fail
)

".venv\Scripts\python.exe" "%~dp0download_yamnet.py"
if errorlevel 1 goto :model_fail

".venv\Scripts\python.exe" "%~dp0download_ast.py"
if errorlevel 1 goto :model_fail

rem Do not silently start CPU inference when this PC has an NVIDIA adapter but
rem its driver/CUDA runtime is incompatible with PyTorch.
".venv\Scripts\python.exe" "%~dp0tools\doctor.py" --skip-resources --require-gpu-if-nvidia
if errorlevel 1 goto :fail

if "%SMOKE_TEST%"=="1" (
    echo [SMOKE] Loading every model and running one CPU-compatible inference...
    set "QT_QPA_PLATFORM=offscreen"
    ".venv\Scripts\python.exe" "%~dp0tools\doctor.py" --full
    if errorlevel 1 goto :fail
    echo [OK] Runtime smoke test passed.
    exit /b 0
)

".venv\Scripts\python.exe" "%~dp0run.py"
set "RUN_EXIT=%errorlevel%"
if not "%RUN_EXIT%"=="0" (
    echo.
    echo [ERROR] Detector Danger exited with an error. Run verify_install.bat for diagnostics.
    pause
)
exit /b %RUN_EXIT%

:model_fail
echo [ERROR] Model download or validation failed. Re-run run.bat to resume the download.

:fail
if defined DEVICE_PATH_FILE if exist "%DEVICE_PATH_FILE%" del /q "%DEVICE_PATH_FILE%" >nul 2>nul
echo [ERROR] Detector Danger could not start.
if "%SMOKE_TEST%"=="0" pause
exit /b 1
