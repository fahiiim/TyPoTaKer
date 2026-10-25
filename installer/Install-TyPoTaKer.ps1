<#
.SYNOPSIS
    1-Click Installer for TypoFormer - AI Keystroke Companion
.DESCRIPTION
    Installs TypoFormer into the user's local application folder ($env:LOCALAPPDATA\Programs\TypoFormer),
    creates Desktop and Start Menu shortcuts with the official TypoFormer icon, sets up auto-start, and launches the application.
#>

[CmdletBinding()]
param(
    [bool]$AutoStart = $true,
    [bool]$LaunchImmediately = $true
)

$ErrorActionPreference = "Stop"

Write-Host "==========================================================" -ForegroundColor Cyan
Write-Host "       TypoFormer - AI Keystroke Companion Installer       " -ForegroundColor Green
Write-Host "==========================================================" -ForegroundColor Cyan
Write-Host ""

# 1. Stop any currently running instance of TypoFormer or TyPoTaKer
Write-Host "[1/5] Stopping any running TypoFormer instances..." -ForegroundColor White
Get-Process -Name "TyPoTaKer*", "TypoFormer*" -ErrorAction SilentlyContinue | ForEach-Object {
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
$InstallDir = Join-Path $env:LOCALAPPDATA "Programs\TypoFormer"
Write-Host "[3/5] Installing application to: $InstallDir" -ForegroundColor White

if (-not (Test-Path $InstallDir)) {
    New-Item -ItemType Directory -Path $InstallDir -Force | Out-Null
}

# Copy files using robocopy
Write-Host "      Copying binaries, models, and tokenizers..." -ForegroundColor Gray
robocopy $SourceDir $InstallDir /E /IS /IT /NP /NDL /NFL /NJH /NJS | Out-Null

# Copy assets if available
$AssetsDir = Join-Path $RepoRoot "assets"
if (Test-Path $AssetsDir) {
    robocopy $AssetsDir $InstallDir "typoformer.ico" "typoformer_logo.png" /IS /IT /NP /NDL /NFL /NJH /NJS | Out-Null
}

Write-Host "      [OK] Installation files successfully copied." -ForegroundColor Green

# 4. Create Shortcuts (Desktop & Start Menu)
Write-Host "[4/5] Creating application shortcuts..." -ForegroundColor White

$WshShell = New-Object -ComObject WScript.Shell
$TargetExe = Join-Path $InstallDir "TyPoTaKer.Desktop.exe"
$IconFile = Join-Path $InstallDir "typoformer.ico"

# Clean up older TyPoTaKer shortcuts if present
Remove-Item (Join-Path $env:APPDATA "Microsoft\Windows\Start Menu\Programs\TyPoTaKer.lnk") -ErrorAction SilentlyContinue
Remove-Item (Join-Path ([System.Environment]::GetFolderPath("Desktop")) "TyPoTaKer.lnk") -ErrorAction SilentlyContinue

# Start Menu Shortcut
$StartMenuDir = [System.IO.Path]::Combine($env:APPDATA, "Microsoft\Windows\Start Menu\Programs")
$StartMenuLnk = Join-Path $StartMenuDir "TypoFormer.lnk"
$Shortcut = $WshShell.CreateShortcut($StartMenuLnk)
$Shortcut.TargetPath = $TargetExe
$Shortcut.WorkingDirectory = $InstallDir
if (Test-Path $IconFile) { $Shortcut.IconLocation = $IconFile }
$Shortcut.Description = "TypoFormer - 94M INT8 Causal LM and GEC Desktop Assistant"
$Shortcut.Save()
Write-Host "      [OK] Created Start Menu Shortcut: $StartMenuLnk" -ForegroundColor Gray

# Desktop Shortcut
$DesktopDir = [System.Environment]::GetFolderPath("Desktop")
$DesktopLnk = Join-Path $DesktopDir "TypoFormer.lnk"
$DesktopShortcut = $WshShell.CreateShortcut($DesktopLnk)
$DesktopShortcut.TargetPath = $TargetExe
$DesktopShortcut.WorkingDirectory = $InstallDir
if (Test-Path $IconFile) { $DesktopShortcut.IconLocation = $IconFile }
$DesktopShortcut.Description = "TypoFormer - AI Keystroke Companion"
$DesktopShortcut.Save()
Write-Host "      [OK] Created Desktop Shortcut: $DesktopLnk" -ForegroundColor Gray

# Optional Auto-start registry
if ($AutoStart) {
    Write-Host "      Registering TypoFormer to run automatically at logon..." -ForegroundColor Gray
    $RegPath = "HKCU:\Software\Microsoft\Windows\CurrentVersion\Run"
    # Remove old key if present
    Remove-ItemProperty -Path $RegPath -Name "TyPoTaKer" -ErrorAction SilentlyContinue
    Set-ItemProperty -Path $RegPath -Name "TypoFormer" -Value ('"' + $TargetExe + '"') -Force
    Write-Host "      [OK] Windows startup enabled." -ForegroundColor Gray
}

# 5. Launch Application
Write-Host "[5/5] Installation complete!" -ForegroundColor Green
Write-Host ""
Write-Host "----------------------------------------------------------" -ForegroundColor Cyan
Write-Host "  How to use TypoFormer:" -ForegroundColor Yellow
Write-Host "   1. Start typing in any Windows app (Word, Notepad, Browser)." -ForegroundColor White
Write-Host "   2. Press [Tab] or [Ctrl + Space] to accept completions." -ForegroundColor White
Write-Host "   3. Select any text and right-click to run Instant Grammar Fix." -ForegroundColor White
Write-Host "   4. Double-click the System Tray icon to open the Control Dashboard." -ForegroundColor White
Write-Host "----------------------------------------------------------" -ForegroundColor Cyan
Write-Host ""

if ($LaunchImmediately) {
    Write-Host "Launching TypoFormer now..." -ForegroundColor Cyan
    Start-Process -FilePath $TargetExe -WorkingDirectory $InstallDir
}
