@echo off
title SentinelVision AI - Launcher
echo ========================================================
echo   SENTINELVISION AI - REAL-TIME SOC & THREAT INTELLIGENCE
echo ========================================================
echo.
echo [1/2] Starting FastAPI & AI Inference Engine (Port 8000)...
start "SentinelVision Backend" cmd /k "cd backend && python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload"

echo [2/2] Starting React + Vite SOC Command Center UI (Port 5173)...
start "SentinelVision Frontend" cmd /k "cd frontend && npm run dev"

echo.
echo SentinelVision AI initialized!
echo Open your browser at: http://localhost:5173
echo API Docs available at: http://localhost:8000/docs
echo ========================================================
pause
