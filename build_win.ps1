# ============================================================================
# build_win.ps1 — Build TEMseg.exe for Windows (one-dir bundle)
#
# Prerequisites:
#   - Python 3.11+ with the backend venv set up (uv sync done)
#   - Node.js + npm (for frontend build)
#   - PyInstaller: uv pip install pyinstaller
#   - weight_manifest.json in project root (URLs can be placeholders for build)
#   - temseg_icon.ico in project root (generate on mac via ./convert_icon.sh)
#
# Usage (PowerShell):
#   .\build_win.ps1                       # incremental build (fastest)
#   .\build_win.ps1 -SkipFrontend         # skip frontend rebuild (when only Python changed)
#   .\build_win.ps1 -Clean                # wipe build/ and dist/ first (use after spec changes)
#   .\build_win.ps1 -Clean -SkipFrontend  # clean PyInstaller but keep frontend
#
# If execution policy blocks the script:
#   powershell -ExecutionPolicy Bypass -File .\build_win.ps1
#
# Output:
#   dist\TEMseg\TEMseg.exe   (plus dist\TEMseg\_internal\ runtime tree)
# ============================================================================

param(
    [switch]$Clean,
    [switch]$SkipFrontend
)

$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ScriptDir

Write-Host "========================================"
Write-Host "  TEMseg Windows Build"
if ($Clean)        { Write-Host "  Mode: CLEAN" }
if ($SkipFrontend) { Write-Host "  Skipping frontend rebuild" }
Write-Host "========================================"

# -------------------------------------------
# Step 1: Check weight manifest + icon (cheap)
# -------------------------------------------
Write-Host ""
Write-Host "[1/4] Checking weight manifest..."

if (-not (Test-Path "weight_manifest.json")) {
    throw "weight_manifest.json not found in project root"
}
Write-Host "  Manifest found"

if (-not (Test-Path "temseg_icon.ico")) {
    Write-Warning "  temseg_icon.ico not found - build will use default PyInstaller icon"
}

# -------------------------------------------
# Step 2: Ensure PyInstaller is installed (cheap)
# -------------------------------------------
Write-Host ""
Write-Host "[2/4] Checking PyInstaller..."

uv run python -c "import PyInstaller" 2>$null
if ($LASTEXITCODE -ne 0) {
    Write-Host "  Installing PyInstaller..."
    uv pip install pyinstaller
    if ($LASTEXITCODE -ne 0) { throw "PyInstaller install failed" }
}
Write-Host "  PyInstaller OK"

# -------------------------------------------
# Step 3: Build frontend static export
# -------------------------------------------
Write-Host ""
if (-not $SkipFrontend) {
    Write-Host "[3/4] Building frontend..."

    if (-not (Test-Path "frontend\node_modules")) {
        Write-Host "  Installing npm dependencies..."
        Push-Location frontend
        npm install
        if ($LASTEXITCODE -ne 0) { Pop-Location; throw "npm install failed" }
        Pop-Location
    }

    Push-Location frontend
    npm run build
    if ($LASTEXITCODE -ne 0) { Pop-Location; throw "npm run build failed" }
    Pop-Location

    if (-not (Test-Path "frontend\out\index.html")) {
        throw "Frontend build failed - frontend\out\index.html not found"
    }
    Write-Host "  Frontend build OK"
} else {
    Write-Host "[3/4] Skipping frontend build (-SkipFrontend)"
    if (-not (Test-Path "frontend\out\index.html")) {
        throw "Cannot skip frontend - frontend\out\index.html doesn't exist yet, run without -SkipFrontend first"
    }
}

# -------------------------------------------
# Step 4: Run PyInstaller
# -------------------------------------------
Write-Host ""
Write-Host "[4/4] Running PyInstaller..."

if ($Clean) {
    Write-Host "  Clean build requested - wiping build\ and dist\ ..."
    if (Test-Path "build\TEMseg") { Remove-Item -Recurse -Force "build\TEMseg" }
    if (Test-Path "dist\TEMseg")  { Remove-Item -Recurse -Force "dist\TEMseg" }
    Write-Host "  This will take ~10 minutes..."
} else {
    Write-Host "  Incremental build - reusing build\ cache..."
    Write-Host "  (Pass -Clean for a full rebuild if spec/deps changed)"
}

uv run pyinstaller temseg.spec --noconfirm
if ($LASTEXITCODE -ne 0) { throw "PyInstaller failed" }

# -------------------------------------------
# Step 4b: Fix rsciio - PyInstaller fails to bundle its subdirectories
# Windows one-dir layout: dist\TEMseg\_internal\rsciio
# -------------------------------------------
$RsciioSrc = uv run python -c "import rsciio; from pathlib import Path; print(Path(rsciio.__file__).parent)" 2>$null
$RsciioSrc = $RsciioSrc.Trim()
$RsciioDest = "dist\TEMseg\_internal\rsciio"

if (Test-Path $RsciioSrc) {
    Write-Host "  Copying rsciio from $RsciioSrc ..."
    if (Test-Path $RsciioDest) { Remove-Item -Recurse -Force $RsciioDest }
    Copy-Item -Recurse $RsciioSrc $RsciioDest
    if (Test-Path "$RsciioDest\emd\specifications.yaml") {
        Write-Host "  rsciio\emd OK"
    } else {
        Write-Warning "  rsciio\emd\specifications.yaml still missing!"
    }
} else {
    Write-Warning "  Could not find rsciio source at $RsciioSrc"
}

# -------------------------------------------
# Done
# -------------------------------------------
if (Test-Path "dist\TEMseg\TEMseg.exe") {
    Write-Host ""
    Write-Host "========================================"
    Write-Host "  BUILD SUCCESSFUL"
    Write-Host "========================================"
    Write-Host ""
    Write-Host "  Output: dist\TEMseg\TEMseg.exe"
    $size = (Get-ChildItem -Recurse dist\TEMseg | Measure-Object -Property Length -Sum).Sum / 1MB
    Write-Host ("  Size:   {0:N1} MB" -f $size)
    Write-Host ""
    Write-Host "  Note: Model weights will be downloaded on first launch"
    Write-Host "  to %USERPROFILE%\AppData\Local\TEMseg\weights\"
    Write-Host ""
} else {
    throw "Build failed - dist\TEMseg\TEMseg.exe not found"
}
