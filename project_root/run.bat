@echo off
setlocal EnableExtensions

REM =====================================================================
REM Offline setup:
REM 1) Place your GGUF model at: models\model.gguf
REM 2) Place llama.cpp executable at: bin\llama.exe
REM    Prebuilt Windows binaries: https://github.com/ggerganov/llama.cpp/releases
REM =====================================================================

set "ROOT_DIR=%~dp0"
cd /d "%ROOT_DIR%"

if not exist "%ROOT_DIR%python\python.exe" (
    echo [ERROR] Embedded Python not found: "%ROOT_DIR%python\python.exe"
    exit /b 1
)

if not exist "%ROOT_DIR%bin\llama.exe" (
    echo [ERROR] llama executable not found: "%ROOT_DIR%bin\llama.exe"
    exit /b 1
)

if not exist "%ROOT_DIR%models\model.gguf" (
    echo [ERROR] model file not found: "%ROOT_DIR%models\model.gguf"
    exit /b 1
)

REM Disable bytecode writes for read-only scenarios.
set "PYTHONDONTWRITEBYTECODE=1"
set "PYTHONUTF8=1"

"%ROOT_DIR%python\python.exe" "%ROOT_DIR%app\main.py"
exit /b %ERRORLEVEL%
