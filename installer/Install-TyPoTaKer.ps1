<#
.SYNOPSIS
    1-Click Installer for TyPoTaKer - AI Keystroke Companion
.DESCRIPTION
    Installs TyPoTaKer into the user's local application folder ($env:LOCALAPPDATA\Programs\TyPoTaKer),
    creates Desktop and Start Menu shortcuts, sets up auto-start if desired, and launches the application.
#>

[CmdletBinding()]
param(
    [bool]$AutoStart = $true,
    [bool]$LaunchImmediately = $true
)

$ErrorActionPreference = "Stop"

Write-Host "==========================================================" -ForegroundColor Cyan
Write-Host "       TyPoTaKer - AI Keystroke Companion Installer       " -ForegroundColor Yellow
Write-Host "==========================================================" -ForegroundColor Cyan
Write-Host ""

# 1. Stop any currently running instance of TyPoTaKer
Write-Host "[1/5] Stopping any running TyPoTaKer instances..." -ForegroundColor White
Get-Process TyPoTaKer* -ErrorAction SilentlyContinue | ForEach-Object {
    Write-Host "      Stopping process $($_.Name) (PID: $($_.Id))..." -ForegroundColor Gray
    Stop-Process -Id $_.Id -Force
}
Start-Sleep -Milliseconds 500

# 2. Locate source payload
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RepoRoot = Split-Path -Parent $ScriptDir
$CandidatePaths = @(
    (Join-Path $ScriptDir "payload"),
    (Join-Path $ScriptDir "app\bin\publish"),
    (Join-Path $RepoRoot "app\bin\publish"),
    $ScriptDir
)

$SourceDir = $null
foreach ($path in $CandidatePaths) {
    if (Test-Path (Join-Path $path "TyPoTaKer.Desktop.exe")) {
        $SourceDir = $path
        break
    }
}

if (-not $SourceDir) {
    Write-Error "Could not locate TyPoTaKer.Desktop.exe! Please ensure app/bin/publish exists."
    exit 1
}

Write-Host "[2/5] Located payload at: $SourceDir" -ForegroundColor Green

# 3. Target Installation Directory
$InstallDir = Join-Path $env:LOCALAPPDATA "Programs\TyPoTaKer"
Write-Host "[3/5] Installing application to: $InstallDir" -ForegroundColor White

if (-not (Test-Path $InstallDir)) {
    New-Item -ItemType Directory -Path $InstallDir -Force | Out-Null
}

# Copy files using robocopy
Write-Host "      Copying binaries, models, and tokenizers..." -ForegroundColor Gray
robocopy $SourceDir $InstallDir /E /IS /IT /NP /NDL /NFL /NJH /NJS | Out-Null
Write-Host "      [OK] Installation files successfully copied." -ForegroundColor Green

# 4. Create Shortcuts (Desktop & Start Menu)
Write-Host "[4/5] Creating application shortcuts..." -ForegroundColor White

$WshShell = New-Object -ComObject WScript.Shell
$TargetExe = Join-Path $InstallDir "TyPoTaKer.Desktop.exe"

# Start Menu Shortcut
$StartMenuDir = [System.IO.Path]::Combine($env:APPDATA, "Microsoft\Windows\Start Menu\Programs")
$StartMenuLnk = Join-Path $StartMenuDir "TyPoTaKer.lnk"
$Shortcut = $WshShell.CreateShortcut($StartMenuLnk)
$Shortcut.TargetPath = $TargetExe
$Shortcut.WorkingDirectory = $InstallDir
$Shortcut.Description = "TyPoTaKer - AI Keystroke Companion and Autocomplete"
$Shortcut.Save()
Write-Host "      [OK] Created Start Menu Shortcut: $StartMenuLnk" -ForegroundColor Gray

# Desktop Shortcut
$DesktopDir = [System.Environment]::GetFolderPath("Desktop")
$DesktopLnk = Join-Path $DesktopDir "TyPoTaKer.lnk"
$DesktopShortcut = $WshShell.CreateShortcut($DesktopLnk)
$DesktopShortcut.TargetPath = $TargetExe
$DesktopShortcut.WorkingDirectory = $InstallDir
$DesktopShortcut.Description = "TyPoTaKer - AI Keystroke Companion"
$DesktopShortcut.Save()
Write-Host "      [OK] Created Desktop Shortcut: $DesktopLnk" -ForegroundColor Gray

# Optional Auto-start registry
if ($AutoStart) {
    Write-Host "      Registering TyPoTaKer to run automatically at logon..." -ForegroundColor Gray
    $RegPath = "HKCU:\Software\Microsoft\Windows\CurrentVersion\Run"
    Set-ItemProperty -Path $RegPath -Name "TyPoTaKer" -Value ('"' + $TargetExe + '"') -Force
    Write-Host "      [OK] Windows startup enabled." -ForegroundColor Gray
}

# 5. Launch Application
Write-Host "[5/5] Installation complete!" -ForegroundColor Green
Write-Host ""
Write-Host "----------------------------------------------------------" -ForegroundColor Cyan
Write-Host "  How to use TyPoTaKer:" -ForegroundColor Yellow
Write-Host "   1. Start typing in any Windows app (Word, Notepad, Browser)." -ForegroundColor White
Write-Host "   2. Press [Tab] or [Ctrl + Space] to accept completions." -ForegroundColor White
Write-Host "   3. Select any text and right-click to run Instant Grammar Fix." -ForegroundColor White
Write-Host "   4. Double-click the System Tray icon to open the Control Dashboard." -ForegroundColor White
Write-Host "----------------------------------------------------------" -ForegroundColor Cyan
Write-Host ""

if ($LaunchImmediately) {
    Write-Host "Launching TyPoTaKer now..." -ForegroundColor Cyan
    Start-Process -FilePath $TargetExe -WorkingDirectory $InstallDir
}
