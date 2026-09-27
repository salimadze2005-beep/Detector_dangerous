@echo off
setlocal EnableExtensions
chcp 65001 >nul
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    echo [ERROR] Сначала запустите setup.bat.
    pause
    exit /b 1
)

echo Полная загрузка и проверка всех моделей может занять несколько минут...
".venv\Scripts\python.exe" "%~dp0tools\bootstrap.py"
if errorlevel 1 (
    pause
    exit /b 1
)
".venv\Scripts\python.exe" "%~dp0download_yamnet.py"
if errorlevel 1 (
    pause
    exit /b 1
)
".venv\Scripts\python.exe" "%~dp0download_ast.py"
if errorlevel 1 (
    pause
    exit /b 1
)

rem A CPU-only PC is supported. If NVIDIA is present, a real CUDA YOLO inference
rem is mandatory so an obsolete/broken driver cannot silently fall back to CPU.
".venv\Scripts\python.exe" "%~dp0tools\doctor.py" --full --require-gpu-if-nvidia
if errorlevel 1 (
    pause
    exit /b 1
)

".venv\Scripts\python.exe" -m pip check
if errorlevel 1 (
    pause
    exit /b 1
)

".venv\Scripts\python.exe" -m unittest discover -s "%~dp0tests" -p "test_*.py" -v
if errorlevel 1 (
    pause
    exit /b 1
)

".venv\Scripts\python.exe" -m unittest discover -s "%~dp0tests\rest" -p "test_*.py" -v
if errorlevel 1 (
    pause
    exit /b 1
)

echo.
echo [OK] Полная перепроверка AST-установки завершена успешно.
pause
