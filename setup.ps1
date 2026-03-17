# XLA Face Privacy System — One-click Windows setup & launcher
# Run: Right-click → "Run with PowerShell"  OR  .\setup.ps1

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Write-Step { param($msg) Write-Host "`n==> $msg" -ForegroundColor Cyan }
function Write-OK { param($msg) Write-Host "    [OK] $msg" -ForegroundColor Green }
function Write-Warn { param($msg) Write-Host "    [!!] $msg" -ForegroundColor Yellow }
function Write-Fail { param($msg) Write-Host "`n[FAIL] $msg" -ForegroundColor Red ; Read-Host "Press Enter to exit" ; exit 1 }

$ROOT = $PSScriptRoot

# ─── 1. Check Python ─────────────────────────────────────────────────────────
Write-Step "Checking Python..."
$py = $null
foreach ($cmd in @("python", "python3", "py")) {
    try {
        $ver = & $cmd --version 2>&1
        if ($ver -match "Python (\d+)\.(\d+)") {
            $major = [int]$Matches[1]; $minor = [int]$Matches[2]
            if ($major -ge 3 -and $minor -ge 10) { $py = $cmd; break }
        }
    }
    catch {}
}

if (-not $py) {
    Write-Warn "Python 3.10+ not found. Attempting install via winget..."
    try {
        winget install --id Python.Python.3.11 --silent --accept-package-agreements --accept-source-agreements
        $env:PATH = [System.Environment]::GetEnvironmentVariable("PATH", "Machine") + ";" + [System.Environment]::GetEnvironmentVariable("PATH", "User")
        $py = "python"
        Write-OK "Python installed."
    }
    catch {
        Write-Fail "Could not install Python automatically.`nPlease install Python 3.10+ from https://python.org/downloads/ and re-run this script."
    }
}
Write-OK "Using: $(& $py --version)"

# ─── 2. Create virtual environment ──────────────────────────────────────────
Write-Step "Setting up virtual environment..."
$venvPath = Join-Path $ROOT ".venv"
if (-not (Test-Path $venvPath)) {
    & $py -m venv $venvPath
    Write-OK "Created .venv"
}
else {
    Write-OK ".venv already exists"
}

$pipExe = Join-Path $venvPath "Scripts\pip.exe"
$pythonExe = Join-Path $venvPath "Scripts\python.exe"

# ─── 3. Install dependencies ─────────────────────────────────────────────────
Write-Step "Installing Python packages (this takes a few minutes on first run)..."
& $pipExe install --upgrade pip --quiet
& $pipExe install -r (Join-Path $ROOT "requirements.txt")
Write-OK "All packages installed."

# ─── 4. Check ffmpeg ─────────────────────────────────────────────────────────
Write-Step "Checking ffmpeg..."
try {
    $null = & ffmpeg -version 2>&1
    Write-OK "ffmpeg found."
}
catch {
    Write-Warn "ffmpeg not found — clip recording will fall back to mp4v codec."
    Write-Warn "To install: winget install ffmpeg"
}

# ─── 5. Ensure data directories exist ────────────────────────────────────────
Write-Step "Preparing data directories..."
@("data\clips\blurred", "data\clips\secure", "data\members") | ForEach-Object {
    $dir = Join-Path $ROOT $_
    if (-not (Test-Path $dir)) { New-Item -ItemType Directory -Path $dir | Out-Null }
}
Write-OK "Data directories ready."

# ─── 6. Start server ─────────────────────────────────────────────────────────
Write-Step "Starting XLA server on http://localhost:8000 ..."
Write-Host ""
Write-Host "  Admin dashboard : http://localhost:8000/ui/admin/dashboard.html" -ForegroundColor Yellow
Write-Host "  User portal     : http://localhost:8000/ui/user/live-portal.html" -ForegroundColor Yellow
Write-Host "  API docs        : http://localhost:8000/docs" -ForegroundColor Yellow
Write-Host ""
Write-Host "  Press Ctrl+C to stop the server." -ForegroundColor Gray
Write-Host ""

Set-Location $ROOT
& $pythonExe -m uvicorn src.app_api:app --host 0.0.0.0 --port 8000 --reload
