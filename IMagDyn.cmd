@echo off
setlocal EnableExtensions EnableDelayedExpansion
REM IMagDyn - interactive menu (pass args for direct commands)
set "ROOT=%~dp0"
set "PYTHONPATH=%ROOT%;%PYTHONPATH%"

set "KEEP_OPEN=0"
if "%~1"=="" set "KEEP_OPEN=1"

set "ENVFILE=%ROOT%.imagdyn_env"
set "PREFENV="
if exist "%ENVFILE%" (
  set /p PREFENV=<"%ENVFILE%"
)
if "%PREFENV%"=="" if exist "%ROOT%.magdyn_env" (
  set /p PREFENV=<"%ROOT%.magdyn_env"
)

set "CONDA_BIN="
if defined CONDA_EXE if exist "%CONDA_EXE%" set "CONDA_BIN=%CONDA_EXE%"
if not defined CONDA_BIN if exist "D:\ProgramData\anaconda3\Scripts\conda.exe" set "CONDA_BIN=D:\ProgramData\anaconda3\Scripts\conda.exe"
if not defined CONDA_BIN if exist "%USERPROFILE%\anaconda3\Scripts\conda.exe" set "CONDA_BIN=%USERPROFILE%\anaconda3\Scripts\conda.exe"
if not defined CONDA_BIN if exist "%USERPROFILE%\miniconda3\Scripts\conda.exe" set "CONDA_BIN=%USERPROFILE%\miniconda3\Scripts\conda.exe"
if not defined CONDA_BIN if exist "C:\ProgramData\anaconda3\Scripts\conda.exe" set "CONDA_BIN=C:\ProgramData\anaconda3\Scripts\conda.exe"
if not defined CONDA_BIN if exist "%LOCALAPPDATA%\anaconda3\Scripts\conda.exe" set "CONDA_BIN=%LOCALAPPDATA%\anaconda3\Scripts\conda.exe"
if not defined CONDA_BIN if exist "%LOCALAPPDATA%\miniconda3\Scripts\conda.exe" set "CONDA_BIN=%LOCALAPPDATA%\miniconda3\Scripts\conda.exe"
if not defined CONDA_BIN (
  where conda >nul 2>&1
  if not errorlevel 1 (
    for /f "delims=" %%I in ('where conda 2^>nul') do (
      if not defined CONDA_BIN set "CONDA_BIN=%%I"
    )
  )
)

set "EC=0"
if not "%PREFENV%"=="" (
  if defined CONDA_BIN (
    echo Using preferred conda env: %PREFENV%
    echo conda: %CONDA_BIN%
    "%CONDA_BIN%" run -n %PREFENV% --no-capture-output python -m imagdyn %*
    set "EC=!ERRORLEVEL!"
    goto :finish
  ) else (
    echo Preferred env=%PREFENV% but conda.exe not found. Falling back to default python.
  )
)

where python >nul 2>&1
if errorlevel 1 (
  echo ERROR: python not found on PATH.
  echo Install Python or open Anaconda Prompt, then run IMagDyn.cmd again.
  set "EC=1"
  set "KEEP_OPEN=1"
  goto :finish
)

python -m imagdyn %*
set "EC=!ERRORLEVEL!"

:finish
if not "!EC!"=="0" (
  echo.
  echo Exit code: !EC!
  echo If this failed due to missing packages:
  echo   1^) Menu -^> Environment -^> set preferred env e.g. tf-gpu
  echo   2^) Or: conda activate tf-gpu then IMagDyn.cmd
  set "KEEP_OPEN=1"
)
if "!KEEP_OPEN!"=="1" (
  echo.
  pause
)
exit /b !EC!