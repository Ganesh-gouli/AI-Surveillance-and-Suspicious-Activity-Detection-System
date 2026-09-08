import os
import sys
import uuid
import time
import base64
import tempfile
import cv2
import numpy as np
from typing import Dict, Any, Optional, List
from fastapi import APIRouter, UploadFile, File, Form, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse

# Ensure root dir in path
WORKSPACE_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
if WORKSPACE_ROOT not in sys.path:
    sys.path.insert(0, WORKSPACE_ROOT)

from detection.fighting_detector import global_fighting_detector

router = APIRouter(prefix="/fighting", tags=["Fighting Detection AI"])

FIGHTING_JOBS: Dict[str, Dict[str, Any]] = {}
FIGHTING_EVENTS: List[Dict[str, Any]] = []

RESULTS_DIR = os.path.join(WORKSPACE_ROOT, "results")
PROCESSED_VIDEOS_DIR = os.path.join(RESULTS_DIR, "processed_videos")
os.makedirs(PROCESSED_VIDEOS_DIR, exist_ok=True)


def background_analyze_video(
    job_id: str,
    temp_input_path: str,
    output_video_path: str,
    filename: str,
    confidence_threshold: float,
    sample_rate: int
):
    try:
        def update_progress(pct: float):
            if job_id in FIGHTING_JOBS:
                FIGHTING_JOBS[job_id]["progress_percentage"] = pct

        analysis = global_fighting_detector.analyze_video(
            video_path=temp_input_path,
            output_video_path=output_video_path,
            confidence_threshold=confidence_threshold,
            sample_rate=sample_rate,
            progress_callback=update_progress
        )

        canonical_path = os.path.join(RESULTS_DIR, "fighting_detected_video.mp4")
        if os.path.exists(output_video_path):
            import shutil
            shutil.copy(output_video_path, canonical_path)

        result_data = {
            "status": "COMPLETED",
            "progress_percentage": 100.0,
            "filename": filename,
            "final_result": analysis["final_result"],
            "is_fighting": analysis["is_fighting"],
            "overall_confidence": analysis["overall_confidence"],
            "detected_action": analysis.get("detected_action", "Normal Activity"),
            "analyzed_frames": analysis["analyzed_frames"],
            "total_video_frames": analysis["total_video_frames"],
            "fighting_frames": analysis["fighting_frames"],
            "fighting_percentage": analysis["fighting_percentage"],
            "duration_sec": analysis["duration_sec"],
            "processing_time_sec": analysis["processing_time_sec"],
            "segments": analysis.get("segments", []),
            "timeline": analysis.get("timeline", [])[:100],
            "video_url": f"/api/fighting/video/{os.path.basename(output_video_path)}"
        }

        if job_id in FIGHTING_JOBS:
            FIGHTING_JOBS[job_id].update(result_data)

        # Log event if fighting detected
        if analysis["is_fighting"]:
            event = {
                "id": f"FIGHT-{uuid.uuid4().hex[:8].upper()}",
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "source": filename,
                "confidence": analysis["overall_confidence"],
                "action": analysis.get("detected_action", "Physical Combat"),
                "duration_sec": analysis["duration_sec"],
                "segments_count": len(analysis.get("segments", [])),
                "status": "CRITICAL ALERT"
            }
            FIGHTING_EVENTS.insert(0, event)

    except Exception as e:
        print(f"[Fighting Analysis Error] {e}")
        if job_id in FIGHTING_JOBS:
            FIGHTING_JOBS[job_id]["status"] = "FAILED"
            FIGHTING_JOBS[job_id]["error"] = str(e)
    finally:
        try:
            if os.path.exists(temp_input_path):
                os.remove(temp_input_path)
        except Exception:
            pass


@router.post("/analyze-video")
async def upload_and_analyze_video(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    confidence_threshold: float = Form(0.70),
    sample_rate: int = Form(2)
):
    """
    Upload an MP4/AVI/MOV video for complete MediaPipe Pose + BiLSTM temporal fighting detection.
    Renders annotated output video saved to results/processed_videos/.
    """
    allowed_exts = [".mp4", ".avi", ".mov", ".mkv", ".webm"]
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in allowed_exts:
        raise HTTPException(status_code=400, detail=f"Unsupported format {ext}. Allowed: {allowed_exts}")

    job_id = f"FIGHT-{uuid.uuid4().hex[:8].upper()}"

    temp_input = tempfile.NamedTemporaryFile(delete=False, suffix=ext)
    contents = await file.read()
    temp_input.write(contents)
    temp_input.close()

    output_filename = f"detected_fighting_{job_id}.mp4"
    output_video_path = os.path.join(PROCESSED_VIDEOS_DIR, output_filename)

    FIGHTING_JOBS[job_id] = {
        "job_id": job_id,
        "filename": file.filename,
        "status": "PROCESSING",
        "progress_percentage": 0.0,
        "created_at": time.time(),
        "message": "Video queued for MediaPipe pose extraction & BiLSTM inference."
    }

    background_tasks.add_task(
        background_analyze_video,
        job_id=job_id,
        temp_input_path=temp_input.name,
        output_video_path=output_video_path,
        filename=file.filename,
        confidence_threshold=confidence_threshold,
        sample_rate=sample_rate
    )

    return {
        "job_id": job_id,
        "filename": file.filename,
        "status": "PROCESSING",
        "message": "Video queued for MediaPipe Pose BiLSTM analysis."
    }


@router.get("/status/{job_id}")
def get_analysis_status(job_id: str):
    """Poll progress and metrics for a running or completed video analysis job."""
    if job_id not in FIGHTING_JOBS:
        raise HTTPException(status_code=404, detail="Job not found.")
    return FIGHTING_JOBS[job_id]


@router.get("/video/{filename}")
def stream_processed_video(filename: str):
    """Stream annotated fighting video to HTML5 video player."""
    candidate_paths = [
        os.path.join(PROCESSED_VIDEOS_DIR, filename),
        os.path.join(RESULTS_DIR, filename),
        os.path.join(RESULTS_DIR, "fighting_detected_video.mp4")
    ]
    video_path = None
    for cp in candidate_paths:
        if os.path.exists(cp):
            video_path = cp
            break

    if not video_path:
        raise HTTPException(status_code=404, detail=f"Video {filename} not found.")

    return FileResponse(
        video_path,
        media_type="video/mp4",
        filename=os.path.basename(video_path),
        headers={
            "Accept-Ranges": "bytes",
            "Cache-Control": "no-cache, no-store, must-revalidate",
            "Pragma": "no-cache",
            "Expires": "0"
        }
    )


@router.post("/detect-frame")
def detect_frame_live(payload: dict):
    """
    Low-latency frame-level inference for Live Webcam and CCTV RTSP streams.
    Synchronous execution allows FastAPI to dispatch to worker threadpool, avoiding event-loop blocking.
    """
    image_base64 = payload.get("image_base64", "")
    camera_id = payload.get("camera_id", "LIVE-WEBCAM")

    if not image_base64:
        raise HTTPException(status_code=400, detail="Missing image_base64")

    if "," in image_base64:
        image_base64 = image_base64.split(",", 1)[1]

    try:
        raw_bytes = base64.b64decode(image_base64)
        np_arr = np.frombuffer(raw_bytes, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if frame is None:
            raise ValueError("Failed to decode image frame")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image data: {e}")

    result = global_fighting_detector.detect_frame(frame, camera_id=camera_id)

    # Encode annotated frame back to JPEG base64 for frontend display (quality 65 for speed and low bandwidth)
    _, buffer = cv2.imencode(".jpg", result["annotated_frame"], [int(cv2.IMWRITE_JPEG_QUALITY), 65])
    annotated_b64 = base64.b64encode(buffer).decode("utf-8")

    return {
        "camera_id": camera_id,
        "is_fighting": result["is_fighting"],
        "fighting_confidence": result["fighting_confidence"],
        "consecutive_frames": result["consecutive_frames"],
        "persons_detected": result["persons_detected"],
        "trigger_alert": result["trigger_alert"],
        "status_text": result["status_text"],
        "annotated_frame": f"data:image/jpeg;base64,{annotated_b64}"
    }


@router.get("/events")
def get_fighting_events():
    """Retrieve history of confirmed fighting incidents."""
    return {"events": FIGHTING_EVENTS[:50]}


@router.get("/config")
def get_fighting_config():
    """Retrieve default fighting detector configuration."""
    return {
        "confidence_threshold": global_fighting_detector.confidence_threshold,
        "min_consecutive_frames": global_fighting_detector.min_consecutive_frames,
        "smooth_window": global_fighting_detector.smooth_window,
        "model_loaded": global_fighting_detector.is_loaded,
        "device": str(global_fighting_detector.device),
        "pipeline": "YOLOv11 Tracking + MediaPipe 33-Pose + Pelvis Normalization + BiLSTM"
    }
