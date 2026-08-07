@echo off
setlocal EnableExtensions EnableDelayedExpansion
REM magdyn — interactive menu (pass args for direct commands)
set "ROOT=%~dp0"
set "PYTHONPATH=%ROOT%;%PYTHONPATH%"

set "ENVFILE=%ROOT%.magdyn_env"
set "PREFENV="
if exist "%ENVFILE%" (
  set /p PREFENV=<"%ENVFILE%"
)

if not "!PREFENV!"=="" (
  where conda >nul 2>&1
  if not errorlevel 1 (
    echo Using preferred conda env: !PREFENV!
    conda run -n !PREFENV! --no-capture-output python -m magdyn %*
    exit /b !ERRORLEVEL!
  )
)

python -m magdyn %*
set "EC=!ERRORLEVEL!"
if not "!EC!"=="0" (
  echo.
  echo If this failed due to missing packages, activate an env then re-run:
  echo   conda activate ^<env_name^>
  echo   magdyn.cmd
  echo Or open the menu and choose Environment to set a preferred conda env.
)
exit /b !EC!
