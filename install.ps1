<#
.SYNOPSIS
    TypoFormer 1-Click Shell Installer (Windows x64)
.DESCRIPTION
    Downloads, installs, and launches TypoFormer directly from PowerShell:
    irm https://raw.githubusercontent.com/fahiiim/TyPoTaKer/main/install.ps1 | iex
#>

[CmdletBinding()]
param(
    [string]$Version = "latest",
    [bool]$AutoStart = $true,
    [bool]$LaunchImmediately = $true
)

$ErrorActionPreference = "Stop"

Write-Host "==========================================================" -ForegroundColor Cyan
Write-Host "    TypoFormer - Offline AI Keystroke Companion Setup    " -ForegroundColor Green
Write-Host "==========================================================" -ForegroundColor Cyan
Write-Host ""

# 1. Architecture Check
if ($env:PROCESSOR_ARCHITECTURE -ne "AMD64" -and $env:PROCESSOR_ARCHITEW6432 -ne "AMD64") {
    Write-Error "TypoFormer requires a 64-bit Windows operating system (x64)."
    exit 1
}

# 2. Stop any running instances
Write-Host "[1/5] Checking for running instances..." -ForegroundColor White
Get-Process -Name "TyPoTaKer*", "TypoFormer*" -ErrorAction SilentlyContinue | ForEach-Object {
    Write-Host "      Closing process $($_.Name) (PID: $($_.Id))..." -ForegroundColor Gray
    Stop-Process -Id $_.Id -Force
}
Start-Sleep -Milliseconds 400

# 3. Setup Target Installation Directory
$InstallDir = Join-Path $env:LOCALAPPDATA "Programs\TypoFormer"
if (-not (Test-Path $InstallDir)) {
    New-Item -ItemType Directory -Path $InstallDir -Force | Out-Null
}

# 4. Acquire Application Files
Write-Host "[2/5] Preparing application payload..." -ForegroundColor White

$LocalPayload = $null
$ScriptDir = $PSScriptRoot

# Check if running locally within repository
if ($ScriptDir) {
    $Candidates = @(
        (Join-Path $ScriptDir "installer\payload"),
        (Join-Path $ScriptDir "payload"),
        (Join-Path $ScriptDir "app\bin\publish")
    )
    foreach ($c in $Candidates) {
        if (Test-Path (Join-Path $c "TyPoTaKer.Desktop.exe")) {
            $LocalPayload = $c
            break
        }
    }
}

if ($LocalPayload) {
    Write-Host "      Detected local payload: $LocalPayload" -ForegroundColor Green
    Write-Host "      Copying binaries and models to $InstallDir..." -ForegroundColor Gray
    robocopy $LocalPayload $InstallDir /E /IS /IT /NP /NDL /NFL /NJH /NJS | Out-Null
} else {
    # Remote execution (irm ... | iex) -> Download standalone distribution zip
    Write-Host "      Downloading TypoFormer standalone package..." -ForegroundColor Cyan
    $ZipDest = Join-Path $env:TEMP "TypoFormer-win-x64.zip"
    
    $DownloadUrls = @(
        "https://github.com/fahiiim/TyPoTaKer/releases/download/v1.0.0/TypoFormer-win-x64.zip",
        "https://github.com/fahiiim/TyPoTaKer/releases/latest/download/TypoFormer-win-x64.zip",
        "https://huggingface.co/fahiiim/TypoFormer/resolve/main/TypoFormer-win-x64.zip"
    )

    $Downloaded = $false
    $prevProgress = $ProgressPreference
    $ProgressPreference = "SilentlyContinue"
    [Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12

    foreach ($url in $DownloadUrls) {
        try {
            Write-Host "      Fetching from: $url" -ForegroundColor Gray
            try {
                Invoke-WebRequest -Uri $url -OutFile $ZipDest -UseBasicParsing -MaximumRedirection 10
            } catch {
                $wc = New-Object System.Net.WebClient
                $wc.Headers.Add("User-Agent", "Mozilla/5.0")
                $wc.DownloadFile($url, $ZipDest)
            }
            if ((Test-Path $ZipDest) -and (Get-Item $ZipDest).Length -gt 10000000) {
                $Downloaded = $true
                break
            }
        } catch {
            Write-Host "      Notice: $url not reachable yet, trying next source..." -ForegroundColor Yellow
        }
    }
    $ProgressPreference = $prevProgress

    if (-not $Downloaded) {
        Write-Error "Unable to download release archive. Please check internet connection or download manually from https://github.com/fahiiim/TyPoTaKer/releases."
        exit 1
    }

    Write-Host "      Extracting to: $InstallDir..." -ForegroundColor Gray
    Expand-Archive -Path $ZipDest -DestinationPath $InstallDir -Force
    Remove-Item -Path $ZipDest -Force -ErrorAction SilentlyContinue
}

# 5. Create Desktop and Start Menu Shortcuts
Write-Host "[3/5] Creating application shortcuts..." -ForegroundColor White
$TargetExe = Join-Path $InstallDir "TyPoTaKer.Desktop.exe"
$IconFile = Join-Path $InstallDir "typoformer.ico"

if (-not (Test-Path $TargetExe)) {
    Write-Error "Verification failed: Executable not found at $TargetExe"
    exit 1
}

$WshShell = New-Object -ComObject WScript.Shell

# Start Menu Shortcut
$StartMenuDir = [System.IO.Path]::Combine($env:APPDATA, "Microsoft\Windows\Start Menu\Programs")
$StartMenuLnk = Join-Path $StartMenuDir "TypoFormer.lnk"
$Shortcut = $WshShell.CreateShortcut($StartMenuLnk)
$Shortcut.TargetPath = $TargetExe
$Shortcut.WorkingDirectory = $InstallDir
if (Test-Path $IconFile) { $Shortcut.IconLocation = $IconFile }
$Shortcut.Description = "TypoFormer - 94M INT8 Causal LM and GEC Desktop Assistant"
$Shortcut.Save()
Write-Host "      [OK] Start Menu shortcut created: $StartMenuLnk" -ForegroundColor Gray

# Desktop Shortcut
$DesktopDir = [System.Environment]::GetFolderPath("Desktop")
$DesktopLnk = Join-Path $DesktopDir "TypoFormer.lnk"
$DesktopShortcut = $WshShell.CreateShortcut($DesktopLnk)
$DesktopShortcut.TargetPath = $TargetExe
$DesktopShortcut.WorkingDirectory = $InstallDir
if (Test-Path $IconFile) { $DesktopShortcut.IconLocation = $IconFile }
$DesktopShortcut.Description = "TypoFormer - AI Keystroke Companion"
$DesktopShortcut.Save()
Write-Host "      [OK] Desktop shortcut created: $DesktopLnk" -ForegroundColor Gray

# 6. Windows Logon Auto-Start
Write-Host "[4/5] Configuring startup options..." -ForegroundColor White
if ($AutoStart) {
    $RegPath = "HKCU:\Software\Microsoft\Windows\CurrentVersion\Run"
    Remove-ItemProperty -Path $RegPath -Name "TyPoTaKer" -ErrorAction SilentlyContinue
    Set-ItemProperty -Path $RegPath -Name "TypoFormer" -Value ('"' + $TargetExe + '"') -Force
    Write-Host "      [OK] Registered auto-start on Windows login." -ForegroundColor Gray
}

# Generate clean Uninstaller in the install directory
$UninstallBat = Join-Path $InstallDir "Uninstall.bat"
@"
@echo off
taskkill /F /IM TyPoTaKer.Desktop.exe /T 2>nul
reg delete "HKCU\Software\Microsoft\Windows\CurrentVersion\Run" /v "TypoFormer" /f 2>nul
del /q "%APPDATA%\Microsoft\Windows\Start Menu\Programs\TypoFormer.lnk" 2>nul
del /q "%USERPROFILE%\Desktop\TypoFormer.lnk" 2>nul
echo TypoFormer shortcuts and registry keys removed.
echo Deleting application files...
timeout /t 2 >nul
rmdir /s /q "$InstallDir" 2>nul
echo TypoFormer successfully uninstalled.
pause
"@ | Set-Content -Path $UninstallBat -Encoding ASCII

# 7. Complete & Launch
Write-Host "[5/5] Installation complete!" -ForegroundColor Green
Write-Host ""
Write-Host "==========================================================" -ForegroundColor Cyan
Write-Host "  TypoFormer is ready to use!" -ForegroundColor Green
Write-Host "   * Type anywhere in Windows (Word, Notepad, Browser, etc.)" -ForegroundColor White
Write-Host "   * Press [Tab] to accept corrections and autocompletions" -ForegroundColor White
Write-Host "   * Select any text and right-click to fix grammar instantly" -ForegroundColor White
Write-Host "   * Double-click tray icon to open the QuillBot Dashboard" -ForegroundColor White
Write-Host "==========================================================" -ForegroundColor Cyan
Write-Host ""

if ($LaunchImmediately) {
    Write-Host "Starting TypoFormer..." -ForegroundColor Cyan
    Start-Process -FilePath $TargetExe -WorkingDirectory $InstallDir
}
