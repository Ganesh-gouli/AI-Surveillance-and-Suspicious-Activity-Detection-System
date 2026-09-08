# SentinelVision AI 🛡️👁️
### Real-Time Human Behaviour & Suspicious Activity Intelligence Platform

[![Python 3.11](https://img.shields.io/badge/Python-3.11-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![React 18](https://img.shields.io/badge/React-18-61DAFB?style=for-the-badge&logo=react&logoColor=black)](https://react.dev)
[![TypeScript](https://img.shields.io/badge/TypeScript-5.3+-3178C6?style=for-the-badge&logo=typescript&logoColor=white)](https://typescriptlang.org)
[![Tailwind CSS](https://img.shields.io/badge/Tailwind_CSS-3.4-38B2AC?style=for-the-badge&logo=tailwind-css&logoColor=white)](https://tailwindcss.com)
[![TensorFlow / Keras](https://img.shields.io/badge/Keras-3.x-FF6F00?style=for-the-badge&logo=keras&logoColor=white)](https://keras.io)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.8+-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white)](https://opencv.org)

**SentinelVision AI** is an enterprise-grade Video Surveillance and Security Operations Center (SOC/VMS) platform designed for real-time computer vision analysis, multi-person tracking, 8-class human activity classification, temporal behaviour anomaly detection, restricted zone perimeter defense, and rapid threat incident response.

---

## 🌟 Key Capabilities & Features

1. **SOC Command Dashboard**: Real-time KPI monitors (Active Cameras, People Detected, Active Alerts, Today's Incidents, AI Confidence), live multi-camera quad wall, live threat feed ticker, and system telemetry.
2. **Multi-Camera Surveillance Wall**: 1x1, 2x2, 3x3, and Focus Grid layouts with real-time canvas bounding boxes, persistent tracking IDs (`Person #01`), 8-class activity tags (`RUNNING 94.2%`), and red glowing threat highlights (`FIGHTING 96.2%`).
3. **Live Webcam AI Vision Studio**: Direct browser `MediaDevices` video stream with live FPS, latency, detected people count, primary activity badge, posture confidence gauge, snapshot capture, and WebM recording.
4. **CCTV & IP Camera Fleet Management**: RTSP & HTTP stream management, password masking, connection latency tester, and status monitors (`ONLINE`, `OFFLINE`, `CONNECTING`, `ERROR`).
5. **Forensic Video Upload & Interactive AI Timeline**: Drag-and-drop video analysis (MP4, AVI, MOV, MKV), frame-by-frame processing, and an **Interactive AI Activity Timeline** (clicking timestamp badges like `00:52 - Fighting ⚠` seeks video immediately to that incident).
6. **Static Image Forensic Analyzer**: High-res multi-person posture/activity detection with raw JSON payload inspector.
7. **Security Incidents & Evidence Dossier**: Filterable/searchable audit table, incident detail modal with video playback, AI reasoning explanation, operator resolution notes, and PDF export.
8. **Restricted Zones & Vector Polygon Defense**: Interactive canvas polygon drawer on camera feeds with ray-casting point-in-polygon intrusion alerts.
9. **Spatial Heatmaps & AI Analytics**: 2D movement density heatmaps on camera viewpoints and Recharts analytics (hourly activities, suspicious incidents by day, 8-class distribution, peak threat hours).
10. **AI Models Hub & Pipeline Inspector**: Architecture flow diagram, EfficientNetV2B2 model inspector (87.38% test accuracy, 8 classes), Keras weights manager, and DEMO vs REAL AI switch.
11. **Tactical Web Audio Siren Synthesizer**: Native Web Audio API procedural sirens, alarm beeps, and synthetic voice announcements.

---

## 🧠 8 Human Activity Classes & Suspicious Behaviours

The AI Activity Classifier loads from `models/live_activity_classes.json`:
1. `cycling`
2. `drinking`
3. `eating`
4. `fighting` *(Suspicious / Critical)*
5. `running` *(Contextual / High severity in restricted zones)*
6. `sitting` *(Normal)*
7. `sleeping` *(Abnormal / Loitering in work zones)*
8. `using_laptop` *(Normal)*

### Suspicious Behaviour Rules:
- **Fighting / Violence**: Immediate CRITICAL alert with audible alarm and red HUD highlights.
- **Unauthorized Zone Intrusion**: Ray-casting point-in-polygon algorithm detecting when any tracked person enters user-drawn restricted zones.
- **Running in Restricted Area**: Fast velocity movement trigger (>4.5 m/s).
- **Loitering / Prolonged Presence**: Person remaining in secure area > configurable threshold (default: 30s).
- **Inactivity / Collapse**: Prolonged non-movement in non-resting zones (Sleeping/Collapse).
- **After-Hours Presence**: Detections occurring during scheduled off-hours.

---

## 🏗️ AI Service Architecture

```
Video / Webcam / RTSP Stream
          ↓
     OpenCV Ingestion
          ↓
YOLOv8 Person Detection (or HOG-NMS)
          ↓
Multi-Person Tracking (ByteTrack / IoU)
          ↓
Human Activity Classifier (EfficientNetV2B2 Keras)
          ↓
Temporal Behaviour & Ray-Casting Rule Engine
          ↓
     Alert Engine & Dispatcher
          ↓
FastAPI WebSockets & SQLite / PostgreSQL ORM
          ↓
React SOC Command Center UI
```

---

## 🚀 Quick Start Guide

### Automated One-Click Launch (Windows):
Double click `start_sentinelvision.bat` in the project root.

### Manual Launch:

#### 1. Backend Server (FastAPI):
```bash
cd backend
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```
- **API Root**: `http://localhost:8000`
- **Interactive Swagger Docs**: `http://localhost:8000/docs`

#### 2. Frontend Application (React + Vite):
```bash
cd frontend
npm install
npm run dev
```
- **Web App**: `http://localhost:5173`

---

## 📂 Project Structure

```
SentinelVision AI/
├── backend/
│   ├── app/
│   │   ├── ai/
│   │   │   ├── activity_classifier.py  # EfficientNetV2B2 Keras Classifier
│   │   │   ├── alert_engine.py         # Severity Classifier & Dispatcher
│   │   │   ├── behaviour_engine.py     # Temporal & Polygon Anomaly Engine
│   │   │   ├── inference_pipeline.py   # Unified AI Inference Pipeline
│   │   │   ├── person_detector.py      # YOLOv8 / HOG Person Detector
│   │   │   └── tracker.py              # IoU Multi-Person Tracker
│   │   ├── api/routers/                # REST API Endpoints
│   │   ├── database/                   # SQLAlchemy Models & Seed Data
│   │   ├── schemas/                    # Pydantic v2 Models
│   │   ├── websocket/                  # Live Stream & Alert WebSockets
│   │   └── main.py                     # FastAPI Application Entrypoint
│   └── requirements.txt
│
├── frontend/
│   ├── src/
│   │   ├── components/                 # Canvas Overlays, Video Player, Modals
│   │   ├── pages/                      # 11 Dedicated SOC Command Pages
│   │   ├── services/                   # REST API, WebSockets, Audio Alerts
│   │   ├── types/                      # TypeScript Definitions
│   │   ├── App.tsx                     # Main Router & Global State
│   │   ├── index.css                   # SOC Dark Theme & Scanline Filters
│   │   └── main.tsx
│   ├── package.json
│   ├── tailwind.config.js
│   └── vite.config.ts
│
├── models/
│   ├── best_human_activity_v2.keras    # Trained Keras Model Weights
│   ├── generate_sample_model.py        # Model Generator Script
│   └── live_activity_classes.json      # 8 Human Activity Classes
│
├── start_sentinelvision.bat            # Quick Start Launcher
└── README.md
```

---

## 🔒 Security & Roles

- **SOC Operator**: Live monitoring, PTZ control, incident review & resolution.
- **Security Administrator**: Camera provisioning, restricted zone vector drawing, detection sensitivity tuning.
- **RTSP Protection**: Sensitive camera credentials masked and secured on the backend.

---

## 🚀 Production Cloud Deployment (Vercel + Render)

The platform is designed to be deployed as a decoupled, full-stack cloud architecture:
- **Frontend SPA**: Hosted on **Vercel** (Global Edge CDN, automatic HTTPS)
- **Backend AI Engine**: Hosted on **Render** (FastAPI, WebSockets, Python 3.11)

```
                       GitHub Repository
                               |
            +------------------+------------------+
            |                                     |
            v                                     v
     Vercel (Frontend)                    Render (Backend)
    • Root: frontend/                    • Root: backend/
    • Framework: Vite                    • Runtime: Python 3.11
    • React 18 + TS SPA                  • FastAPI + Uvicorn ($PORT)
    • Output: dist/                      • YOLOv11 + MediaPipe + WebSockets
            |                                     |
            +============ HTTPS / WSS ============>
```

---

### Step 1: Deploy Backend to Render

1. Sign in to [Render](https://render.com) and click **New +** -> **Web Service**.
2. Connect your GitHub repository: `https://github.com/Ganesh-gouli/AI-Surveillance-and-Suspicious-Activity-Detection-System`.
3. Configure the service settings:
   - **Name**: `sentinelvision-ai-backend` (or your preferred name)
   - **Region**: Choose the region closest to you (e.g. Frankfurt, Oregon, Singapore)
   - **Root Directory**: `backend`
   - **Runtime**: `Python 3`
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `uvicorn app.main:app --host 0.0.0.0 --port $PORT`
4. In the **Environment Variables** section, add:
   - `PYTHON_VERSION`: `3.11.9`
   - `FRONTEND_URL`: `https://YOUR-FRONTEND.vercel.app` *(update once your Vercel URL is generated)*
5. Click **Create Web Service**. Wait for the build and deployment to complete.
6. Copy your public Render service URL (e.g., `https://sentinelvision-ai-backend.onrender.com`).

---

### Step 2: Deploy Frontend to Vercel

1. Sign in to [Vercel](https://vercel.com) and click **Add New...** -> **Project**.
2. Import your GitHub repository: `https://github.com/Ganesh-gouli/AI-Surveillance-and-Suspicious-Activity-Detection-System`.
3. Configure project settings:
   - **Framework Preset**: `Vite`
   - **Root Directory**: Click **Edit** and choose `frontend`
   - **Build Command**: `npm run build` *(default)*
   - **Output Directory**: `dist` *(default)*
4. In the **Environment Variables** section, add:
   - `VITE_API_URL`: `https://YOUR-BACKEND.onrender.com` *(your Render backend URL from Step 1)*
   - `VITE_WS_URL`: `wss://YOUR-BACKEND.onrender.com` *(note the `wss://` protocol)*
5. Click **Deploy**. Vercel will build and assign you a production domain (e.g., `https://sentinelvision-ai.vercel.app`).
6. *Final Step*: Return to Render, open **Environment**, and set `FRONTEND_URL` to your Vercel URL (`https://sentinelvision-ai.vercel.app`), then click **Save Changes**.

---

### ⚠️ Cloud Deployment Considerations & Limitations

| Feature | Local Environment | Render Cloud Environment | Production Recommendation |
| :--- | :--- | :--- | :--- |
| **Server Memory (RAM)** | Full Host RAM (8GB - 32GB) | **512 MB** (Render Free Tier) | Upgrading to **Render Starter (1GB RAM)** or **Standard (2GB RAM)** is recommended for simultaneous deep learning inference (PyTorch + Keras + YOLO). |
| **Model Weights** | Local `.keras` / `.pt` files | Graceful fallbacks (Auto YOLO download, MediaPipe Pose biomechanics) | For custom fine-tuned weights, store models on **Hugging Face Hub** or **AWS S3** and download during build. |
| **Storage Persistence** | Local Disk (`results/`, `uploads/`) | Ephemeral (cleared on restart) | Connect a **Render Disk** or use Cloud Object Storage (AWS S3, Supabase Storage) for permanent incident archiving. |
| **WebSockets** | `ws://localhost:8000` | Native `wss://` on port 443 | Fully supported. On Free tier, service sleeps after 15 min idle (cold boot takes 30-50s). |
| **Webcam Streaming** | Direct browser `MediaDevices` | Direct browser `MediaDevices` | **100% Functional** — Frames are encoded client-side in the browser and transmitted over HTTPS/WSS. |
| **Private CCTV/RTSP** | Local Subnet (e.g. `192.168.x.x`) | Cannot reach local LAN without tunnel | For remote CCTV cameras, use public RTSP streams, HLS feeds, or a secure tunnel (e.g. Cloudflare Tunnel / Tailscale). |
