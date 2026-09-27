@echo off
setlocal EnableExtensions
cd /d "%~dp0"

rem setup.bat installs dependencies and downloads every required model, including PANNs.
call "%~dp0setup.bat" --no-pause
if errorlevel 1 (
    pause
    exit /b 1
)

call "%~dp0run.bat"
exit /b %errorlevel%
