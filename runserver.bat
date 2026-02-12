@echo off
setlocal EnableExtensions EnableDelayedExpansion

REM ============================================================
REM runserver.bat - Starts the SkyrimNet Pocket-TTS API server
REM - Verifies .venv exists
REM - Activates venv
REM - Optionally sets env vars (uncomment as needed)
REM - Runs the FastAPI server entrypoint
REM ============================================================

goto :main

:die
echo.
echo [ERROR] %*
echo.
pause
exit /b 1

:main
echo.
echo === Starting SkyrimNet TTS API ===
echo.

REM Always run relative to this script's directory
cd /d "%~dp0" || call :die Failed to cd to "%~dp0"

set "VENV_DIR=%~dp0.venv"
set "PY=%VENV_DIR%\Scripts\python.exe"

if not exist "%PY%" (
  call :die Virtual environment not found. Run setup.bat first. Expected: "%PY%"
)

REM -------------------------
REM Optional environment vars
REM -------------------------
REM Uncomment/tweak as needed:

REM Force HF hub offline (use cached/local weights only)
REM set "HF_HUB_OFFLINE=1"

REM Pin to specific logical CPUs (overrides auto P-core detection if your app supports it)
REM set "PCORE_CPUS=0,1,2,3,4,5,6,7"

REM Cleanup tuning (if your server reads these env vars)
REM set "OUTPUT_CLEANUP_MAX_AGE_SECONDS=600"
REM set "OUTPUT_CLEANUP_INTERVAL_SECONDS=300"

REM Server bind options (if your app reads these env vars)
REM set "HOST=0.0.0.0"
REM set "PORT=7860"

echo [INFO] Using Python: "%PY%"
echo.

REM Activate venv for PATH / scripts (optional but nice)
call "%VENV_DIR%\Scripts\activate.bat" || call :die Failed to activate venv.

REM ---- Start server ----
REM Adjust the path to your server entrypoint if needed:
set "ENTRYPOINT=src\skyrimnet_api.py"
if not exist "%ENTRYPOINT%" (
  call :die Could not find entrypoint "%ENTRYPOINT%". Update ENTRYPOINT in runserver.bat.
)

echo [INFO] Running: %ENTRYPOINT%
echo.

"%PY%" "%ENTRYPOINT%"
set "RC=%errorlevel%"

if not "%RC%"=="0" (
  call :die Server exited with code %RC%
)

echo.
echo [OK] Server exited normally.
pause
exit /b 0
