@echo off
setlocal
title TypoFormer Installer
echo Launching TypoFormer 1-Click Installer...
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0Install-TyPoTaKer.ps1"
if %ERRORLEVEL% NEQ 0 (
    echo.
    echo Installation failed with exit code %ERRORLEVEL%.
    pause
) else (
    echo.
    echo Installation completed successfully!
    timeout /t 3 >nul
)
