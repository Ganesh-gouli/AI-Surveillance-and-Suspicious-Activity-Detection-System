import os
import uuid
import time
import base64
import tempfile
import cv2
import numpy as np
from typing import Dict, Any, Optional
from fastapi import APIRouter, UploadFile, File, Form, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse, StreamingResponse

from ...ai.fire_detector import global_fire_detector

router = APIRouter(prefix="/fire", tags=["Fire Detection AI"])

FIRE_ANALYSIS_JOBS: Dict[str, Dict[str, Any]] = {}
CAMERA_FIRE_TRACKER: Dict[str, int] = {}  # camera_id -> consecutive fire frame count

RESULTS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))),
    "results"
)
os.makedirs(RESULTS_DIR, exist_ok=True)


def background_analyze_video(
    job_id: str,
    temp_input_path: str,
    output_video_path: str,
    filename: str,
    confidence_threshold: float,
    frame_ratio: float,
    sample_rate: int
):
    try:
        def update_progress(pct: float):
            if job_id in FIRE_ANALYSIS_JOBS:
                FIRE_ANALYSIS_JOBS[job_id]["progress_percentage"] = pct

        analysis_result = global_fire_detector.analyze_video(
            video_path=temp_input_path,
            output_video_path=output_video_path,
            confidence_threshold=confidence_threshold,
            frame_ratio=frame_ratio,
            sample_rate=sample_rate,
            progress_callback=update_progress
        )

        # Standard file path for primary result
        canonical_result_path = os.path.join(RESULTS_DIR, "detected_fire_video.mp4")
        if os.path.exists(output_video_path) and output_video_path != canonical_result_path:
            import shutil
            shutil.copy(output_video_path, canonical_result_path)

        FIRE_ANALYSIS_JOBS[job_id].update({
            "status": "COMPLETED",
            "progress_percentage": 100.0,
            "filename": filename,
            "final_result": analysis_result["final_result"],
            "fire_detected": analysis_result["fire_detected"],
            "overall_confidence": analysis_result["overall_confidence"],
            "fire_frames": analysis_result["fire_frames"],
            "analyzed_frames": analysis_result["analyzed_frames"],
            "total_video_frames": analysis_result["total_video_frames"],
            "fire_percentage": analysis_result["fire_percentage"],
            "total_fire_boxes_detected": analysis_result.get("total_fire_boxes_detected", 0),
            "duration_sec": analysis_result["duration_sec"],
            "processing_time_sec": analysis_result["processing_time_sec"],
            "processed_video_url": f"/api/fire/video/{os.path.basename(output_video_path)}",
            "timeline": analysis_result.get("timeline", [])[:100]  # First 100 points for chart
        })

    except Exception as e:
        print(f"[Fire Analysis Error] {e}")
        if job_id in FIRE_ANALYSIS_JOBS:
            FIRE_ANALYSIS_JOBS[job_id]["status"] = "FAILED"
            FIRE_ANALYSIS_JOBS[job_id]["error"] = str(e)
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
    frame_ratio: float = Form(0.20),
    sample_rate: int = Form(5)
):
    """
    Upload a video file for genuine frame-by-frame Fire Detection AI analysis.
    Renders annotated output video with 🔥 FIRE DETECTED overlay saved to results/detected_fire_video.mp4.
    """
    allowed_exts = [".mp4", ".avi", ".mov", ".mkv", ".webm"]
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in allowed_exts:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported format '{ext}'. Allowed formats: MP4, AVI, MOV, MKV, WEBM."
        )

    job_id = f"FIRE-{uuid.uuid4().hex[:8].upper()}"

    # Save uploaded bytes to temp file
    temp_dir = tempfile.gettempdir()
    temp_input_path = os.path.join(temp_dir, f"{job_id}_in{ext}")
    output_video_path = os.path.join(RESULTS_DIR, f"detected_fire_{job_id}.mp4")

    contents = await file.read()
    with open(temp_input_path, "wb") as f:
        f.write(contents)

    FIRE_ANALYSIS_JOBS[job_id] = {
        "job_id": job_id,
        "filename": file.filename,
        "status": "PROCESSING",
        "progress_percentage": 0.0,
        "confidence_threshold": confidence_threshold,
        "frame_ratio": frame_ratio,
        "sample_rate": sample_rate,
        "created_at": time.time()
    }

    background_tasks.add_task(
        background_analyze_video,
        job_id,
        temp_input_path,
        output_video_path,
        file.filename,
        confidence_threshold,
        frame_ratio,
        sample_rate
    )

    return {
        "job_id": job_id,
        "filename": file.filename,
        "status": "PROCESSING",
        "message": "Video queued for PyTorch frame-by-frame fire detection."
    }


@router.get("/status/{job_id}")
def get_analysis_status(job_id: str):
    """Poll the status and metrics of a running or completed fire analysis job."""
    if job_id not in FIRE_ANALYSIS_JOBS:
        raise HTTPException(status_code=404, detail="Job not found.")
    return FIRE_ANALYSIS_JOBS[job_id]


@router.get("/video/{filename}")
def stream_processed_video(filename: str):
    """
    Stream the processed, annotated fire detection video directly to the HTML5 video player.
    Supports HTTP Range requests for seamless video scrubbing.
    """
    video_path = os.path.join(RESULTS_DIR, filename)
    # If not found with job id, fallback to canonical detected_fire_video.mp4
    if not os.path.exists(video_path):
        canonical = os.path.join(RESULTS_DIR, "detected_fire_video.mp4")
        if os.path.exists(canonical):
            video_path = canonical
        else:
            raise HTTPException(status_code=404, detail=f"Processed video {filename} not found.")

    return FileResponse(
        video_path,
        media_type="video/mp4",
        filename="detected_fire_video.mp4",
        headers={"Accept-Ranges": "bytes"}
    )


@router.post("/detect-frame")
async def detect_frame_live(payload: dict):
    """
    Low-latency (<25ms) frame-level inference for Live Camera surveillance.
    Triggers continuous fire alert if fire persists over consecutive frames.
    """
    image_base64 = payload.get("image_base64", "")
    camera_id = payload.get("camera_id", "LIVE-CAMERA")

    if not image_base64:
        raise HTTPException(status_code=400, detail="Missing image_base64")

    # Strip header if present
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

    # Predict with YOLO Fire & Smoke Object Detector + PyTorch
    is_fire, fire_prob, no_fire_prob, boxes = global_fire_detector.detect_boxes_and_classify(frame, yolo_conf=0.25)
    fire_conf_pct = round(fire_prob * 100, 1)

    # Track consecutive fire frames for the camera
    consecutive = CAMERA_FIRE_TRACKER.get(camera_id, 0)
    if is_fire and fire_prob >= 0.50:
        consecutive += 1
    else:
        consecutive = 0
    CAMERA_FIRE_TRACKER[camera_id] = consecutive

    # Continuous alert threshold: 3 or more consecutive fire frames
    continuous_alert = consecutive >= 3

    return {
        "camera_id": camera_id,
        "is_fire": is_fire,
        "fire_confidence": fire_conf_pct,
        "no_fire_confidence": round(no_fire_prob * 100, 1),
        "continuous_fire_alert": continuous_alert,
        "consecutive_fire_frames": consecutive,
        "boxes": boxes,
        "alert_text": "🔥🔥 FIRE ALERT 🔥🔥" if continuous_alert else ("🔥 FIRE DETECTED" if is_fire else "✅ NO FIRE")
    }


@router.get("/config")
def get_fire_config():
    """Retrieve default fire detection parameters."""
    return {
        "default_confidence_threshold": 0.70,
        "default_frame_ratio": 0.20,
        "default_sample_rate": 5,
        "model_loaded": global_fire_detector.is_custom_trained,
        "device": str(global_fire_detector.device)
    }
