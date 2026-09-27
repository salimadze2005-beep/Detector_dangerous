@echo off
setlocal EnableExtensions EnableDelayedExpansion
chcp 65001 >nul
cd /d "%~dp0"

set "NO_PAUSE=0"
set "SMOKE_TEST=0"
if /I "%~1"=="--no-pause" set "NO_PAUSE=1"
if /I "%~1"=="--smoke-test" (
    set "NO_PAUSE=1"
    set "SMOKE_TEST=1"
)

set "PYTHON_PATH_FILE=%TEMP%\detector-danger-python-%RANDOM%-%RANDOM%.txt"
if exist "%PYTHON_PATH_FILE%" del /q "%PYTHON_PATH_FILE%" >nul 2>nul

echo [1/7] Checking Windows prerequisites...
if "%SMOKE_TEST%"=="1" (
    "%SystemRoot%\System32\WindowsPowerShell\v1.0\powershell.exe" -NoLogo -NoProfile -ExecutionPolicy Bypass -File "%~dp0tools\windows_prerequisites.ps1" -PythonPathFile "%PYTHON_PATH_FILE%" -CheckOnly
) else (
    "%SystemRoot%\System32\WindowsPowerShell\v1.0\powershell.exe" -NoLogo -NoProfile -ExecutionPolicy Bypass -File "%~dp0tools\windows_prerequisites.ps1" -PythonPathFile "%PYTHON_PATH_FILE%"
)
if errorlevel 1 goto :fail
if not exist "%PYTHON_PATH_FILE%" (
    echo [ERROR] Prerequisite installer did not return a Python path.
    goto :fail
)
set /p "PYTHON_EXE="<"%PYTHON_PATH_FILE%"
del /q "%PYTHON_PATH_FILE%" >nul 2>nul
if not defined PYTHON_EXE goto :fail

"%PYTHON_EXE%" "%~dp0tools\bootstrap.py" --runtime-only
if errorlevel 1 goto :fail

if "%SMOKE_TEST%"=="1" (
    echo [SMOKE] Checking venv support and source syntax without changing the project...
    "%PYTHON_EXE%" -m venv --help >nul
    if errorlevel 1 goto :fail
    "%PYTHON_EXE%" -m compileall -q audio core tools ui video main.py run.py download_ast.py
    if errorlevel 1 goto :fail
    echo [OK] Bootstrap smoke test passed.
    exit /b 0
)

if exist ".venv\Scripts\python.exe" (
    ".venv\Scripts\python.exe" "%~dp0tools\bootstrap.py" --runtime-only >nul 2>nul
    if errorlevel 1 (
        set "VENV_BACKUP=.venv.not-portable-%RANDOM%"
        echo [INFO] The copied or incompatible .venv is being moved to !VENV_BACKUP! ...
        move /y ".venv" "!VENV_BACKUP!" >nul
        if errorlevel 1 goto :fail
    )
)

if not exist ".venv\Scripts\python.exe" (
    echo [2/7] Creating an isolated Python environment...
    "%PYTHON_EXE%" -m venv ".venv"
    if errorlevel 1 goto :fail
) else (
    echo [2/7] Existing Python environment is compatible.
)

set "PIP_DEFAULT_TIMEOUT=180"
set "PIP_RETRIES=10"
set "PIP_DISABLE_PIP_VERSION_CHECK=1"

echo [3/7] Installing Python dependencies. Interrupted pip downloads can be resumed...
".venv\Scripts\python.exe" -m ensurepip --upgrade
if errorlevel 1 goto :fail
".venv\Scripts\python.exe" -m pip install --upgrade pip wheel --retries 10 --timeout 180
if errorlevel 1 goto :fail
".venv\Scripts\python.exe" -m pip install --prefer-binary -r "%~dp0requirements.txt" --retries 10 --timeout 180
if errorlevel 1 goto :fail

echo [4/7] Selecting GPU or CPU runtime...
where nvidia-smi >nul 2>nul
if not errorlevel 1 (
    ".venv\Scripts\python.exe" -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)" >nul 2>nul
    if errorlevel 1 (
        echo [GPU] NVIDIA detected. Trying the official CUDA 12.6 PyTorch wheels...
        ".venv\Scripts\python.exe" -m pip install --upgrade --force-reinstall torch==2.9.1 torchvision==0.24.1 --index-url https://download.pytorch.org/whl/cu126 --retries 10 --timeout 180
        if errorlevel 1 (
            echo [WARN] CUDA PyTorch failed. Restoring the portable CPU runtime...
            ".venv\Scripts\python.exe" -m pip install --prefer-binary --force-reinstall torch==2.9.1 torchvision==0.24.1 --retries 10 --timeout 180
            if errorlevel 1 goto :fail
        )
    )
) else (
    echo [GPU] NVIDIA was not detected. The supported CPU fallback will be used.
)
".venv\Scripts\python.exe" -c "import torch; print('[OK] Device: '+(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'))"
if errorlevel 1 goto :fail
".venv\Scripts\python.exe" -m pip check
if errorlevel 1 goto :fail

echo [5/7] Checking and downloading all models...
".venv\Scripts\python.exe" "%~dp0tools\bootstrap.py"
if errorlevel 1 goto :fail
".venv\Scripts\python.exe" "%~dp0download_yamnet.py"
if errorlevel 1 goto :fail
".venv\Scripts\python.exe" "%~dp0download_ast.py"
if errorlevel 1 goto :fail

echo [6/7] Running installation diagnostics...
rem CPU-only PCs are supported. NVIDIA PCs must pass a real CUDA YOLO check so
rem an obsolete driver cannot silently run the detector on CPU.
".venv\Scripts\python.exe" "%~dp0tools\doctor.py" --full --require-gpu-if-nvidia
if errorlevel 1 goto :fail

echo [7/7] Running regression tests...
set "QT_QPA_PLATFORM=offscreen"
".venv\Scripts\python.exe" -m unittest discover -s "%~dp0tests" -p "test_*.py" -q
if errorlevel 1 goto :fail

echo.
echo [OK] Installation completed. Use run.bat to start Detector Danger.
if "%NO_PAUSE%"=="0" pause
exit /b 0

:fail
if exist "%PYTHON_PATH_FILE%" del /q "%PYTHON_PATH_FILE%" >nul 2>nul
echo.
echo [ERROR] Installation did not complete. Re-run setup.bat: downloads resume automatically.
if "%NO_PAUSE%"=="0" pause
exit /b 1
