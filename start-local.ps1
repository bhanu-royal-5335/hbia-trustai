# HBIA TrustAI Local Environment Start Script (Windows PowerShell)

Write-Host "================================================" -ForegroundColor Cyan
Write-Host "   HBIA TrustAI - Starting Local Full System    " -ForegroundColor Cyan
Write-Host "================================================" -ForegroundColor Cyan

$BackendDir = "$PSScriptRoot\backend"
$FrontendDir = "$PSScriptRoot\frontend"
$VenvPython = "$BackendDir\venv\Scripts\python.exe"

# 1. Setup Backend Environment
Write-Host "`n[1/4] Checking Backend Python Environment..." -ForegroundColor Yellow
if (-not (Test-Path $VenvPython)) {
    Write-Host "Creating Python virtual environment in backend\venv..." -ForegroundColor Gray
    python -m venv "$BackendDir\venv"
}

if (-not (Test-Path "$BackendDir\.env")) {
    Write-Host "Creating backend\.env from example..." -ForegroundColor Gray
    Copy-Item "$BackendDir\.env.example" "$BackendDir\.env"
}

Write-Host "Installing/Verifying backend Python packages..." -ForegroundColor Gray
& $VenvPython -m pip install -r "$BackendDir\requirements.txt" --quiet

# 2. Setup Frontend Environment
Write-Host "`n[2/4] Checking Frontend Environment..." -ForegroundColor Yellow
if (-not (Test-Path "$FrontendDir\.env.local")) {
    Write-Host "Creating frontend\.env.local..." -ForegroundColor Gray
    Set-Content -Path "$FrontendDir\.env.local" -Value "NEXT_PUBLIC_API_URL=http://localhost:8000/api/v1"
}

# 3. Initialize Database
Write-Host "`n[3/4] Initializing Database Tables..." -ForegroundColor Yellow
Set-Location $BackendDir
& $VenvPython -c "import asyncio; from app.core.database import init_db; asyncio.run(init_db()); print('Database tables verified.')"

# 4. Start Services
Write-Host "`n[4/4] Starting Services..." -ForegroundColor Yellow
Write-Host " -> Backend API: http://localhost:8000 (API Docs: http://localhost:8000/api/v1/docs)" -ForegroundColor Green
Write-Host " -> Frontend Web: http://localhost:3000" -ForegroundColor Green

# Launch Backend in new PowerShell window
Start-Process powershell -ArgumentList "-NoExit", "-Command", "Set-Location '$BackendDir'; & '$VenvPython' -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload"

# Launch Frontend in current window
Set-Location $FrontendDir
npm run dev
