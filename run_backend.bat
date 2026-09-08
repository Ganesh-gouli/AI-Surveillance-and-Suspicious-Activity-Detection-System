@echo off
title SentinelVision AI - Backend Service
echo Starting FastAPI & AI Inference Engine...
cd backend
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
pause
