@echo off
setlocal EnableExtensions EnableDelayedExpansion

REM ============================================================
REM setup.bat - Windows setup script for the server
REM 1) Verify Python 3.10-3.14 is installed and works
REM 2) Verify weights\tts_skyrimnet.safetensors exists
REM 3) Create .venv and pip install requirements
REM 4) Copy weights\* into .venv\Lib\site-packages\pocket_tts\config\
REM
REM On any error, the window will NOT close (it will pause).
REM ============================================================

goto :main

REM --- Helper: print error, pause, exit (keep at bottom, jumped to via CALL) ---
:die
echo.
echo [ERROR] %*
echo.
pause
exit /b 1

:main
echo.
echo === SkyrimNet TTS Server Setup ===
echo.

REM --- Step 2: Verify weights file exists early ---
set "WEIGHTS_FILE=%~dp0weights\tts_skyrimnet.safetensors"
if not exist "%WEIGHTS_FILE%" (
  call :die Missing required weights file: "%WEIGHTS_FILE%"
)
echo [OK] Found weights file: "%WEIGHTS_FILE%"

REM --- Step 1: Find a working Python in allowed versions ---
set "PY_CMD="
set "PY_VER="

REM Prefer the Windows Python Launcher if available
where py >nul 2>nul
if %errorlevel%==0 (
  for %%V in (3.14 3.13 3.12 3.11 3.10) do (
    py -%%V -c "import sys; assert sys.version_info[:2]==tuple(map(int,'%%V'.split('.')))" >nul 2>nul
    if !errorlevel!==0 (
      set "PY_CMD=py -%%V"
      set "PY_VER=%%V"
      goto :PY_FOUND
    )
  )
)

REM Fallback: try "python" if "py" not available
where python >nul 2>nul
if %errorlevel%==0 (
  for %%V in (3.14 3.13 3.12 3.11 3.10) do (
    python -c "import sys; assert sys.version_info[:2]==tuple(map(int,'%%V'.split('.')))" >nul 2>nul
    if !errorlevel!==0 (
      set "PY_CMD=python"
      set "PY_VER=%%V"
      goto :PY_FOUND
    )
  )
)

call :die Could not find a working Python 3.10-3.14. Install Python 3.10-3.14 and re-run.

:PY_FOUND
echo [OK] Using Python %PY_VER% via: %PY_CMD%

REM --- Step 3: Create venv + install requirements ---
set "VENV_DIR=%~dp0.venv"
if not exist "%VENV_DIR%\Scripts\python.exe" (
  echo.
  echo [INFO] Creating virtual environment in ".venv" ...
  %PY_CMD% -m venv "%VENV_DIR%"
  if errorlevel 1 call :die Failed to create venv in "%VENV_DIR%".
) else (
  echo.
  echo [OK] Virtual environment already exists: "%VENV_DIR%"
)

echo.
echo [INFO] Upgrading pip/setuptools/wheel ...
"%VENV_DIR%\Scripts\python.exe" -m pip install --upgrade pip setuptools wheel
if errorlevel 1 call :die Failed to upgrade pip tooling.

if not exist "%~dp0requirements.txt" (
  call :die requirements.txt not found in repo root: "%~dp0requirements.txt"
)

echo.
echo [INFO] Installing requirements ...
"%VENV_DIR%\Scripts\python.exe" -m pip install -r "%~dp0requirements.txt"
if errorlevel 1 call :die pip install -r requirements.txt failed.

REM --- Step 4: Copy weights to pocket_tts config directory inside venv ---
set "DST_DIR=%VENV_DIR%\Lib\site-packages\pocket_tts\config"
if not exist "%DST_DIR%" (
  call :die Destination folder not found: "%DST_DIR%". pocket_tts may not have installed correctly.
)

echo.
echo [INFO] Copying weights\* -> "%DST_DIR%"
where robocopy >nul 2>nul
if %errorlevel%==0 (
  robocopy "%~dp0weights" "%DST_DIR%" *.* /E /NFL /NDL /NJH /NJS /NC /NS
  REM Robocopy success codes are 0-7; failure is 8+
  if %errorlevel% GEQ 8 call :die robocopy failed with exit code %errorlevel%.
) else (
  xcopy "%~dp0weights\*" "%DST_DIR%\" /E /I /Y >nul
  if errorlevel 1 call :die xcopy failed copying weights to pocket_tts config.
)

REM Define the YAML file path
set "WEIGHTS_PATH=.venv\Lib\site-packages\pocket_tts\config\tts_skyrimnet.safetensors"
set "YAML_FILE=%DST_DIR%\skyrimnet.yaml"

REM Check if the YAML file exists
if not exist "%YAML_FILE%" (
  echo Error: YAML file not found at %YAML_FILE%
  exit /b 1
)

REM Update the YAML file with the new weights_path
powershell -Command "(Get-Content '%YAML_FILE%') -replace 'weights_path:.*', 'weights_path: %WEIGHTS_PATH%' | Set-Content '%YAML_FILE%'"

echo Updated weights_path in %YAML_FILE% to %WEIGHTS_PATH%

echo.
echo [OK] Setup complete.
echo     Next: use run_server.bat to start the API.
echo.
pause
exit /b 0
