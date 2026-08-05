# HBIA TrustAI Local Environment Start Script (Windows PowerShell)

Write-Host "================================================" -ForegroundColor Cyan
Write-Host "   HBIA TrustAI - Starting Local Full System    " -ForegroundColor Cyan
Write-Host "================================================" -ForegroundColor Cyan

$BackendDir = "$PSScriptRoot\backend"
$FrontendDir = "$PSScriptRoot\frontend"
$VenvPython = "$BackendDir\venv\Scripts\python.exe"
$FrontendNodeModules = "$FrontendDir\node_modules"

# Determine which Python to use
if (Test-Path $VenvPython) {
    $PythonExe = $VenvPython
    Write-Host "Using venv Python: $PythonExe" -ForegroundColor Green
} else {
    $PythonExe = "python"
    Write-Host "venv not found, using system Python" -ForegroundColor Yellow
}

# 1. Setup Backend Environment
Write-Host "`n[1/4] Checking Backend Python Environment..." -ForegroundColor Yellow
if (-not (Test-Path "$BackendDir\.env")) {
    if (Test-Path "$BackendDir\.env.example") {
        Write-Host "Creating backend\.env from example..." -ForegroundColor Gray
        Copy-Item "$BackendDir\.env.example" "$BackendDir\.env"
    }
}

Write-Host "Installing/Verifying backend Python packages..." -ForegroundColor Gray
& $PythonExe -m pip install -r "$BackendDir\requirements.txt" --quiet 2>$null

# 2. Setup Frontend Environment
Write-Host "`n[2/4] Checking Frontend Environment..." -ForegroundColor Yellow
if (-not (Test-Path "$FrontendDir\.env.local")) {
    Write-Host "Creating frontend\.env.local..." -ForegroundColor Gray
    Set-Content -Path "$FrontendDir\.env.local" -Value "NEXT_PUBLIC_API_URL=http://localhost:8000/api/v1"
}

if (-not (Test-Path $FrontendNodeModules)) {
    Write-Host "Installing frontend Node dependencies..." -ForegroundColor Gray
    Push-Location $FrontendDir
    npm install
    Pop-Location
}

# 3. Initialize Database
Write-Host "`n[3/4] Initializing Database Tables..." -ForegroundColor Yellow
Push-Location $BackendDir
& $PythonExe -c "import asyncio; from app.core.database import init_db; asyncio.run(init_db()); print('Database tables verified.')" 2>$null
Pop-Location

# 4. Start Services
Write-Host "`n[4/4] Starting Services..." -ForegroundColor Yellow
Write-Host " -> Backend API: http://localhost:8000 (API Docs: http://localhost:8000/api/v1/docs)" -ForegroundColor Green
Write-Host " -> Frontend Web: http://localhost:3000" -ForegroundColor Green

# Launch Backend as a background job in the SAME shell (more reliable than Start-Process)
$backendJob = Start-Job -ScriptBlock {
    param($dir, $py)
    Set-Location $dir
    & $py -m uvicorn app.main:app --host 0.0.0.0 --port 8000
} -ArgumentList $BackendDir, $PythonExe

Write-Host "`nBackend server starting as background job (ID: $($backendJob.Id))..." -ForegroundColor Cyan

# Wait a moment for backend to boot
Start-Sleep -Seconds 3

# Verify backend is up
try {
    $health = Invoke-RestMethod -Uri "http://localhost:8000/api/v1/health" -Method Get -TimeoutSec 5
    Write-Host "Backend health check: $($health.status) (v$($health.version))" -ForegroundColor Green
} catch {
    Write-Host "Backend may still be starting... Frontend will retry automatically." -ForegroundColor Yellow
}

# Launch Frontend in current window (blocking)
Write-Host "`nStarting Frontend (Next.js)..." -ForegroundColor Green
Push-Location $FrontendDir
npm run dev
Pop-Location

# Cleanup on exit
Write-Host "`nStopping backend job..." -ForegroundColor Yellow
Stop-Job -Job $backendJob -ErrorAction SilentlyContinue
Remove-Job -Job $backendJob -ErrorAction SilentlyContinue
