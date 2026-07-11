@echo off
title AI Shorts Maker Pro - Installer
echo ==========================================
echo    AI Shorts Maker Pro | Installation
echo ==========================================
echo.

echo [+] Step 1: Checking Python...
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [!] Python not found! Please install Python 3.10+ and add it to PATH.
    pause
    exit /b
)

echo [+] Step 2: Installing Python dependencies...
python -m pip install --upgrade pip
pip install -r requirements.txt

echo.
echo [+] Step 3: Checking Ollama...
ollama --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [!] Ollama not found! 
    echo [!] Please install it from https://ollama.com/
    echo [!] After installation, run this script again or pull model manually: ollama pull qwen2.5
) else (
    echo [+] Pulling AI model (qwen2.5)...
    ollama pull qwen2.5
)

echo.
echo [+] Step 4: Checking FFmpeg...
ffmpeg -version >nul 2>&1
if %errorlevel% neq 0 (
    echo [!] FFmpeg not found! 
    echo [!] Please install FFmpeg and add 'bin' folder to your system PATH.
)

echo.
echo ==========================================
echo    Installation Complete! 
echo    Use 'run.bat' to start the program.
echo ==========================================
pause