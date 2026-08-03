#!/usr/bin/env bash
# HBIA TrustAI Local Environment Start Script (Bash)

set -e

echo "================================================"
echo "   HBIA TrustAI - Starting Local Full System    "
echo "================================================"

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BACKEND_DIR="$ROOT_DIR/backend"
FRONTEND_DIR="$ROOT_DIR/frontend"
VENV_PYTHON="$BACKEND_DIR/venv/bin/python"

# 1. Setup Backend Environment
echo -e "\n[1/4] Checking Backend Python Environment..."
if [ ! -f "$VENV_PYTHON" ]; then
    echo "Creating Python virtual environment in backend/venv..."
    python3 -m venv "$BACKEND_DIR/venv"
fi

if [ ! -f "$BACKEND_DIR/.env" ]; then
    echo "Creating backend/.env from example..."
    cp "$BACKEND_DIR/.env.example" "$BACKEND_DIR/.env"
fi

echo "Installing/Verifying backend Python packages..."
"$VENV_PYTHON" -m pip install -r "$BACKEND_DIR/requirements.txt" --quiet

# 2. Setup Frontend Environment
echo -e "\n[2/4] Checking Frontend Environment..."
if [ ! -f "$FRONTEND_DIR/.env.local" ]; then
    echo "Creating frontend/.env.local..."
    echo "NEXT_PUBLIC_API_URL=http://localhost:8000/api/v1" > "$FRONTEND_DIR/.env.local"
fi

# 3. Initialize Database
echo -e "\n[3/4] Initializing Database Tables..."
cd "$BACKEND_DIR"
"$VENV_PYTHON" -c "import asyncio; from app.core.database import init_db; asyncio.run(init_db()); print('Database tables verified.')"

# 4. Start Services
echo -e "\n[4/4] Starting Services..."
echo " -> Backend API: http://localhost:8000 (API Docs: http://localhost:8000/api/v1/docs)"
echo " -> Frontend Web: http://localhost:3000"

# Start Backend API in background
cd "$BACKEND_DIR"
"$VENV_PYTHON" -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload &
BACKEND_PID=$!

trap "kill $BACKEND_PID" EXIT

# Start Frontend
cd "$FRONTEND_DIR"
npm run dev
