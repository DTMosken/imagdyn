@echo off
REM Open IMagDyn viewer (starts HTTP server)
call "%~dp0IMagDyn.cmd" viewer %*
set EC=%ERRORLEVEL%
if not "%EC%"=="0" (
  echo.
  echo Viewer failed, exit code %EC%
  pause
)
exit /b %EC%
