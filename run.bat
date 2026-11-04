@echo off
title AI Shorts Maker Pro - Launcher
set "PYTHONIOENCODING=utf-8"
chcp 65001 >nul

echo ==========================================
echo    AI Shorts Maker Pro ^| Launcher
echo ==========================================
echo.
echo [+] Starting server...
echo [+] URL: http://localhost:7860
echo [+] Close this window to stop the server.
echo.

start http://localhost:7860
python main.py

if %errorlevel% neq 0 (
    echo.
    echo [!] Server stopped with ERROR CODE: %errorlevel%
    echo [!] Possible causes:
    echo     - Port 7860 is already in use
    echo     - Missing libraries, run install.bat
    echo     - Python is not in PATH
    echo.
    pause
)
pause