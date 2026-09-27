@echo off
setlocal EnableExtensions
chcp 65001 >nul
cd /d "%~dp0"

set "NO_PAUSE=0"
if /I "%~1"=="--no-pause" set "NO_PAUSE=1"

echo === Detector Danger: repair Russian Vosk model ===

if not exist ".venv\Scripts\python.exe" (
    echo [INFO] The application environment is missing. Running setup first...
    call "%~dp0setup.bat" --no-pause
    if errorlevel 1 goto :fail
)

echo [INFO] Rebuilding model-ru from the verified official archive...
".venv\Scripts\python.exe" "%~dp0tools\bootstrap.py" --vosk-only --force
if errorlevel 1 goto :fail

".venv\Scripts\python.exe" -c "from core.vosk_model_path import vosk_runtime_path; from vosk import Model,SetLogLevel; SetLogLevel(-1); Model(vosk_runtime_path(r'%~dp0model-ru')); print('[OK] VOSK_MODEL_OK')"
if errorlevel 1 goto :fail

echo.
echo [OK] Vosk model is ready. You can now run run.bat.
if "%NO_PAUSE%"=="0" pause
exit /b 0

:fail
echo.
echo [ERROR] Vosk repair did not finish. Re-run this file to resume the download.
if "%NO_PAUSE%"=="0" pause
exit /b 1
