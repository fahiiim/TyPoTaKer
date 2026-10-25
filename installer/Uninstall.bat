@echo off
setlocal
title TypoFormer Uninstaller
echo ==========================================================
echo               TypoFormer Desktop Uninstaller
echo ==========================================================
echo.
echo [1/4] Stopping any running TypoFormer processes...
taskkill /F /IM TyPoTaKer.Desktop.exe 2>nul
timeout /t 1 /nobreak >nul

echo [2/4] Removing Desktop and Start Menu shortcuts...
del /F /Q "%USERPROFILE%\Desktop\TypoFormer.lnk" 2>nul
del /F /Q "%APPDATA%\Microsoft\Windows\Start Menu\Programs\TypoFormer.lnk" 2>nul
del /F /Q "%USERPROFILE%\Desktop\TyPoTaKer.lnk" 2>nul
del /F /Q "%APPDATA%\Microsoft\Windows\Start Menu\Programs\TyPoTaKer.lnk" 2>nul

echo [3/4] Removing Windows startup registration...
reg delete "HKCU\Software\Microsoft\Windows\CurrentVersion\Run" /v "TypoFormer" /f 2>nul
reg delete "HKCU\Software\Microsoft\Windows\CurrentVersion\Run" /v "TyPoTaKer" /f 2>nul

echo [4/4] Deleting application files from LocalAppData...
if exist "%LOCALAPPDATA%\Programs\TypoFormer" (
    rmdir /S /Q "%LOCALAPPDATA%\Programs\TypoFormer"
)
if exist "%LOCALAPPDATA%\Programs\TyPoTaKer" (
    rmdir /S /Q "%LOCALAPPDATA%\Programs\TyPoTaKer"
)

echo.
echo ==========================================================
echo        TypoFormer has been successfully uninstalled.
echo ==========================================================
echo.
pause
