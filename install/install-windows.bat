@echo off
rem Double click this file. It starts install-windows.ps1 from the same folder.
rem "-ExecutionPolicy Bypass" applies to this one run only and changes no system setting;
rem it is needed because Windows blocks downloaded .ps1 files by default.
setlocal
if not exist "%~dp0install-windows.ps1" (
    echo install-windows.ps1 must be in the same folder as this file.
    pause
    exit /b 1
)
rem Full path: works even when PATH is broken or trimmed.
set "PS_EXE=%SystemRoot%\System32\WindowsPowerShell\v1.0\powershell.exe"
if not exist "%PS_EXE%" set "PS_EXE=powershell.exe"
"%PS_EXE%" -NoProfile -ExecutionPolicy Bypass -File "%~dp0install-windows.ps1" %*
exit /b %ERRORLEVEL%
