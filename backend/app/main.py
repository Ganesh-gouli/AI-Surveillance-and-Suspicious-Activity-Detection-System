import os
import time
import asyncio
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from .database.session import init_db
from .api.routers import cameras, videos, detection, incidents, zones, analytics, system, models, fire, fighting, shooting
from .websocket.connection_manager import manager
from .ai.inference_pipeline import global_pipeline

app = FastAPI(
    title="SentinelVision AI",
    description="Real-Time Human Behaviour & Suspicious Activity Detection SOC Platform",
    version="2.4.0"
)

# CORS configuration
frontend_url_env = os.getenv("FRONTEND_URL", "").strip()
allowed_origins = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "http://localhost:3000",
    "http://127.0.0.1:3000",
]
if frontend_url_env:
    for origin in frontend_url_env.split(","):
        clean_origin = origin.strip().rstrip("/")
        if clean_origin and clean_origin not in allowed_origins:
            allowed_origins.append(clean_origin)

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_origin_regex=r"^https:\/\/.*\.vercel\.app$",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize clean database schema on startup
@app.on_event("startup")
def on_startup():
    init_db()
    port = os.getenv("PORT", "8000")
    print(f"SentinelVision AI Backend Engine fully operational on port {port}.")

# Mount API Routers
app.include_router(cameras.router, prefix="/api")
app.include_router(videos.router, prefix="/api")
app.include_router(fire.router, prefix="/api")
app.include_router(fighting.router, prefix="/api")
app.include_router(shooting.router, prefix="/api")
app.include_router(detection.router, prefix="/api")
app.include_router(incidents.router, prefix="/api")
app.include_router(zones.router, prefix="/api")
app.include_router(analytics.router, prefix="/api")
app.include_router(system.router, prefix="/api")
app.include_router(models.router, prefix="/api")

# Static files for snapshots and uploaded media
uploads_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "uploads")
results_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "results")
os.makedirs(uploads_dir, exist_ok=True)
os.makedirs(results_dir, exist_ok=True)
app.mount("/static", StaticFiles(directory=uploads_dir), name="static")
app.mount("/results", StaticFiles(directory=results_dir), name="results")

# WebSocket Channels
@app.websocket("/ws/live/{camera_id}")
async def websocket_live_camera_feed(websocket: WebSocket, camera_id: str):
    await manager.connect(websocket, channel=camera_id)
    try:
        while True:
            # Heartbeat check
            await asyncio.sleep(2.0)
            await websocket.send_json({
                "camera_id": camera_id,
                "status": "ONLINE",
                "timestamp": time.time(),
                "people_count": 0,
                "people": [],
                "active_alerts": []
            })
    except WebSocketDisconnect:
        manager.disconnect(websocket, channel=camera_id)
    except Exception:
        manager.disconnect(websocket, channel=camera_id)

@app.websocket("/ws/alerts")
async def websocket_alerts_stream(websocket: WebSocket):
    await manager.connect(websocket, channel="alerts")
    try:
        while True:
            await asyncio.sleep(5.0)
    except WebSocketDisconnect:
        manager.disconnect(websocket, channel="alerts")
    except Exception:
        manager.disconnect(websocket, channel="alerts")

@app.get("/")
def root():
    return {
        "platform": "SentinelVision AI",
        "status": "ONLINE",
        "model": "best_human_activity_v2.keras",
        "docs": "/docs"
    }
