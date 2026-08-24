# new window script
param(
    [switch]$Clean,
    [switch]$SkipFrontend,
    [switch]$Cuda
)

$ErrorActionPreference = "Stop"

# Repo root = two levels above scripts/windows
$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
Set-Location $RepoRoot

$SpecFile = if ($Cuda) { "temseg_cuda.spec" } else { "temseg.spec" }
$BaseDistName = if ($Cuda) { "TEMseg-cuda" } else { "TEMseg" }

# Git branch
$RawBranch = try {
    git rev-parse --abbrev-ref HEAD 2>$null
} catch {
    "unknown"
}

$Branch = ($RawBranch -replace '[^a-zA-Z0-9]', '-').Trim('-')
if (-not $Branch) { $Branch = "unknown" }

$DistName = "$BaseDistName-$Branch"

Write-Host "========================================"
Write-Host "  TEMseg Windows Build"
Write-Host "  Target: $(if ($Cuda) { 'CUDA (NVIDIA GPU)' } else { 'CPU-only' })"
Write-Host "  Branch: $Branch"
Write-Host "  Output: dist\$DistName"
if ($Clean) { Write-Host "  Mode: CLEAN" }
if ($SkipFrontend) { Write-Host "  Skipping frontend rebuild" }
Write-Host "========================================"

# --------------------------------------------------
# 1. Check files
# --------------------------------------------------

Write-Host ""
Write-Host "[1/5] Checking files..."

if (-not (Test-Path "weight_manifest.json")) {
    throw "weight_manifest.json not found"
}

if (-not (Test-Path $SpecFile)) {
    throw "$SpecFile not found"
}

if (-not (Test-Path "temseg_icon.ico")) {
    Write-Warning "temseg_icon.ico not found - using default icon"
}

Write-Host "  Files OK"

# --------------------------------------------------
# 2. Check ONNX Runtime
# --------------------------------------------------

Write-Host ""
Write-Host "[2/5] Checking onnxruntime..."

$OrtProviders = uv run python -c "import onnxruntime; print(','.join(onnxruntime.get_available_providers()))"

if ($Cuda) {
    if ($OrtProviders -notmatch "CUDAExecutionProvider") {
        throw @"
CUDA build requires onnxruntime-gpu.

Run:
  uv pip uninstall onnxruntime
  uv pip install onnxruntime-gpu
"@
    }

    Write-Host "  CUDAExecutionProvider detected"
}
else {
    if ($OrtProviders -match "CUDAExecutionProvider") {
        Write-Warning "onnxruntime-gpu detected. CPU build will still be attempted."
    }
    else {
        Write-Host "  CPU onnxruntime detected"
    }
}

# --------------------------------------------------
# 3. PyInstaller
# --------------------------------------------------

Write-Host ""
Write-Host "[3/5] Checking PyInstaller..."

uv run python -c "import PyInstaller" 2>$null

if ($LASTEXITCODE -ne 0) {
    Write-Host "  Installing PyInstaller..."
    uv pip install pyinstaller

    if ($LASTEXITCODE -ne 0) {
        throw "PyInstaller installation failed"
    }
}

Write-Host "  PyInstaller OK"

# --------------------------------------------------
# 4. Frontend
# --------------------------------------------------

Write-Host ""
Write-Host "[4/5] Frontend..."

if ($SkipFrontend) {
    if (-not (Test-Path "frontend\out\index.html")) {
        throw "frontend\out\index.html does not exist. Run without -SkipFrontend first."
    }

    Write-Host "  Skipped"
}
else {
    Push-Location frontend

    try {
        if (-not (Test-Path "node_modules")) {
            Write-Host "  Installing npm dependencies..."
            npm install

            if ($LASTEXITCODE -ne 0) {
                throw "npm install failed"
            }
        }

        Write-Host "  Building frontend..."
        npm run build

        if ($LASTEXITCODE -ne 0) {
            throw "npm run build failed"
        }
    }
    finally {
        Pop-Location
    }

    if (-not (Test-Path "frontend\out\index.html")) {
        throw "frontend build failed - frontend\out\index.html not found"
    }

    Write-Host "  Frontend OK"
}

# --------------------------------------------------
# 5. PyInstaller
# --------------------------------------------------

Write-Host ""
Write-Host "[5/5] Running PyInstaller..."

if ($Clean) {
    Write-Host "  Cleaning previous build..."

    if (Test-Path "build\$BaseDistName") {
        Remove-Item -Recurse -Force "build\$BaseDistName"
    }

    if (Test-Path "dist\$BaseDistName") {
        Remove-Item -Recurse -Force "dist\$BaseDistName"
    }

    if (Test-Path "dist\$DistName") {
        Remove-Item -Recurse -Force "dist\$DistName"
    }
}
else {
    Write-Host "  Incremental build"
}

uv run pyinstaller $SpecFile --noconfirm

if ($LASTEXITCODE -ne 0) {
    throw "PyInstaller failed"
}

# --------------------------------------------------
# Rename output
# --------------------------------------------------

if (Test-Path "dist\$BaseDistName") {
    if (Test-Path "dist\$DistName") {
        Remove-Item -Recurse -Force "dist\$DistName"
    }

    Rename-Item "dist\$BaseDistName" $DistName
}

# --------------------------------------------------
# Fix rsciio
# --------------------------------------------------

Write-Host ""
Write-Host "Fixing rsciio bundle..."

$RsciioSrc = uv run python -c "import rsciio; from pathlib import Path; print(Path(rsciio.__file__).parent)"

if ($RsciioSrc) {
    $RsciioSrc = $RsciioSrc.Trim()
}

$RsciioDest = "dist\$DistName\_internal\rsciio"

if ($RsciioSrc -and (Test-Path $RsciioSrc)) {

    Write-Host "  Source: $RsciioSrc"

    if (Test-Path $RsciioDest) {
        Remove-Item -Recurse -Force $RsciioDest
    }

    Copy-Item -Recurse $RsciioSrc $RsciioDest

    if (Test-Path "$RsciioDest\emd\specifications.yaml") {
        Write-Host "  rsciio OK"
    }
    else {
        Write-Warning "rsciio\emd\specifications.yaml missing"
    }
}
else {
    Write-Warning "Could not find rsciio"
}

# --------------------------------------------------
# Done
# --------------------------------------------------

$Exe = "dist\$DistName\TEMseg.exe"

if (-not (Test-Path $Exe)) {
    throw "Build failed - $Exe not found"
}

$Size = (
    Get-ChildItem -Recurse "dist\$DistName" |
    Where-Object { -not $_.PSIsContainer } |
    Measure-Object -Property Length -Sum
).Sum / 1MB

Write-Host ""
Write-Host "========================================"
Write-Host "  BUILD SUCCESSFUL"
Write-Host "========================================"
Write-Host ""
Write-Host "  Output: $Exe"
Write-Host ("  Size:   {0:N1} MB" -f $Size)
Write-Host ""
Write-Host "  Model weights download on first launch."
Write-Host "========================================"