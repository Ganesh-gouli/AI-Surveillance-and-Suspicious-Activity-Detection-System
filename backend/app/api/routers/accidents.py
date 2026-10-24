import os
import sys
import time
import uuid
import tempfile
from typing import Dict, Any, List, Optional

from fastapi import APIRouter, UploadFile, File, Form, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse, FileResponse
import cv2

# Ensure root dir in python path
WORKSPACE_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
if WORKSPACE_ROOT not in sys.path:
    sys.path.insert(0, WORKSPACE_ROOT)

from detection.industrial_accident_detector import global_industrial_detector

router = APIRouter(prefix="/accidents", tags=["Industrial Accident Detection"])

RESULTS_DIR = os.path.join(WORKSPACE_ROOT, "results")
PROCESSED_VIDEOS_DIR = os.path.join(RESULTS_DIR, "processed_videos")
ACCIDENTS_DIR = os.path.join(WORKSPACE_ROOT, "accidents")
os.makedirs(PROCESSED_VIDEOS_DIR, exist_ok=True)

# In-Memory Storage for Asynchronous Forensic Processing Jobs
ACCIDENT_JOBS: Dict[str, Dict[str, Any]] = {}
ACCIDENT_EVENTS: List[Dict[str, Any]] = []


def background_analyze_accident_video(
    job_id: str,
    temp_input_path: str,
    output_video_path: str,
    filename: str,
    confidence_threshold: float = 0.38,
    sample_rate: int = 4,
    enable_pose: bool = True
):
    """
    Executes deep industrial safety forensics in background worker thread.
    """
    try:
        def update_progress(pct: float):
            if job_id in ACCIDENT_JOBS:
                ACCIDENT_JOBS[job_id]["progress_percentage"] = pct

        analysis = global_industrial_detector.analyze_video(
            video_path=temp_input_path,
            output_video_path=output_video_path,
            confidence_threshold=confidence_threshold,
            sample_rate=sample_rate,
            enable_pose=enable_pose,
            progress_callback=update_progress
        )

        result_data = {
            "status": "COMPLETED",
            "progress_percentage": 100.0,
            "filename": filename,
            "processed_video_filename": os.path.basename(output_video_path),
            "video_url": f"/api/accidents/video/{os.path.basename(output_video_path)}",
            "final_result": analysis["final_result"],
            "is_accident": analysis["is_accident"],
            "accident_type": analysis["accident_type"],
            "overall_confidence": analysis["overall_confidence"],
            "hazard_detected": analysis["hazard_detected"],
            "vehicles_count": analysis["vehicles_count"],
            "fleet": analysis.get("fleet", []),
            "collision_intercept": analysis.get("collision_intercept", {}),
            "stack_alignment": analysis.get("stack_alignment", {}),
            "analyzed_frames": analysis["analyzed_frames"],
            "total_video_frames": analysis["total_video_frames"],
            "accident_frames": analysis["accident_frames"],
            "duration_sec": analysis["duration_sec"],
            "processing_time_sec": analysis["processing_time_sec"],
            "processing_fps": analysis["processing_fps"],
            "segments": analysis["segments"],
            "timeline": analysis["timeline"]
        }

        if job_id in ACCIDENT_JOBS:
            ACCIDENT_JOBS[job_id].update(result_data)

        # Record in EHS OSHA audit logs if an accident was verified
        if analysis["is_accident"]:
            event = {
                "id": f"EHS-INCIDENT-{uuid.uuid4().hex[:8].upper()}",
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "source": filename,
                "event_type": analysis["accident_type"],
                "hazard_detected": analysis["hazard_detected"],
                "vehicles_count": analysis["vehicles_count"],
                "confidence": analysis["overall_confidence"],
                "duration_sec": analysis["duration_sec"],
                "status": "CRITICAL_EHS_OPEN"
            }
            ACCIDENT_EVENTS.insert(0, event)
            if len(ACCIDENT_EVENTS) > 50:
                ACCIDENT_EVENTS.pop()

    except Exception as e:
        import traceback
        traceback.print_exc()
        if job_id in ACCIDENT_JOBS:
            ACCIDENT_JOBS[job_id].update({
                "status": "FAILED",
                "error": str(e)
            })
    finally:
        if "temp" in temp_input_path and os.path.exists(temp_input_path):
            try:
                os.remove(temp_input_path)
            except Exception:
                pass


@router.post("/analyze-video")
async def analyze_accident_video(
    background_tasks: BackgroundTasks,
    video: UploadFile = File(...),
    confidence_threshold: float = Form(0.38),
    sample_rate: int = Form(4),
    enable_pose: bool = Form(True)
):
    """
    Accepts CCTV surveillance video upload and queues asynchronous industrial accident analysis.
    """
    if not video.filename.lower().endswith((".mp4", ".avi", ".mov", ".mkv", ".webm")):
        raise HTTPException(status_code=400, detail="Invalid video format. Supported: mp4, avi, mov, mkv, webm.")

    job_id = f"ACCIDENT-JOB-{uuid.uuid4().hex[:8].upper()}"
    temp_dir = tempfile.gettempdir()
    temp_input_path = os.path.join(temp_dir, f"{job_id}_{video.filename}")

    with open(temp_input_path, "wb") as f:
        content = await video.read()
        f.write(content)

    output_filename = f"detected_accident_{job_id}.mp4"
    output_video_path = os.path.join(PROCESSED_VIDEOS_DIR, output_filename)

    ACCIDENT_JOBS[job_id] = {
        "job_id": job_id,
        "status": "PROCESSING",
        "progress_percentage": 5.0,
        "filename": video.filename,
        "confidence_threshold": confidence_threshold,
        "sample_rate": sample_rate,
        "enable_pose": enable_pose,
        "start_time": time.time()
    }

    background_tasks.add_task(
        background_analyze_accident_video,
        job_id=job_id,
        temp_input_path=temp_input_path,
        output_video_path=output_video_path,
        filename=video.filename,
        confidence_threshold=confidence_threshold,
        sample_rate=sample_rate,
        enable_pose=enable_pose
    )

    return {
        "job_id": job_id,
        "status": "QUEUED",
        "message": f"Industrial accident video analysis queued successfully for {video.filename}."
    }


@router.post("/analyze-sample")
async def analyze_sample_accident(
    background_tasks: BackgroundTasks,
    filename: str = Form(...),
    confidence_threshold: float = Form(0.38),
    sample_rate: int = Form(4),
    enable_pose: bool = Form(True)
):
    """
    Fast 1-click execution on preloaded sample videos from accidents/ folder.
    """
    video_path = os.path.join(ACCIDENTS_DIR, filename)
    if not os.path.exists(video_path):
        # Fallback check in dataset/no_fire
        fallback = os.path.join(WORKSPACE_ROOT, "dataset", "no_fire", filename)
        if os.path.exists(fallback):
            video_path = fallback
        else:
            raise HTTPException(status_code=404, detail=f"Sample video '{filename}' not found.")

    job_id = f"ACCIDENT-JOB-{uuid.uuid4().hex[:8].upper()}"
    output_filename = f"detected_accident_{job_id}.mp4"
    output_video_path = os.path.join(PROCESSED_VIDEOS_DIR, output_filename)

    ACCIDENT_JOBS[job_id] = {
        "job_id": job_id,
        "status": "PROCESSING",
        "progress_percentage": 5.0,
        "filename": filename,
        "confidence_threshold": confidence_threshold,
        "sample_rate": sample_rate,
        "enable_pose": enable_pose,
        "start_time": time.time()
    }

    background_tasks.add_task(
        background_analyze_accident_video,
        job_id=job_id,
        temp_input_path=video_path,
        output_video_path=output_video_path,
        filename=filename,
        confidence_threshold=confidence_threshold,
        sample_rate=sample_rate,
        enable_pose=enable_pose
    )

    return {
        "job_id": job_id,
        "status": "QUEUED",
        "message": f"Preloaded accident video '{filename}' analysis initialized."
    }


@router.get("/job/{job_id}")
async def get_accident_job_status(job_id: str):
    """
    Returns current analysis progress and completed results.
    """
    if job_id not in ACCIDENT_JOBS:
        raise HTTPException(status_code=404, detail="Job ID not found.")
    return ACCIDENT_JOBS[job_id]


@router.get("/video/{filename}")
async def get_accident_processed_video(filename: str):
    """
    Streams browser-native H.264 video with forensic telemetry HUD overlay.
    """
    video_path = os.path.join(PROCESSED_VIDEOS_DIR, filename)
    if not os.path.exists(video_path):
        raise HTTPException(status_code=404, detail="Processed video file not found.")

    return FileResponse(
        video_path,
        media_type="video/mp4",
        filename=filename,
        headers={"Accept-Ranges": "bytes"}
    )


@router.get("/samples")
async def list_sample_videos():
    """
    Lists available benchmark videos in accidents/ directory for immediate testing.
    """
    sample_list = []
    if os.path.exists(ACCIDENTS_DIR):
        for f in sorted(os.listdir(ACCIDENTS_DIR)):
            if f.endswith((".mp4", ".avi", ".mov")):
                full_p = os.path.join(ACCIDENTS_DIR, f)
                cap = cv2.VideoCapture(full_p)
                fps = cap.get(cv2.CAP_PROP_FPS) or 25
                cnt = cap.get(cv2.CAP_PROP_FRAME_COUNT)
                dur = round(cnt / fps, 1) if fps > 0 else 0.0
                cap.release()

                desc_map = {
                    "vidssave.com Forklift Accident_ The Blind Corner 480P.mp4": "Warehouse Forklift Blind Corner Collision Crash",
                    "WhatsApp Video 2026-10-23 at 11.54.39 PM.mp4": "Industrial Pallet & Product Cargo Stack Avalanche Collapse"
                }

                sample_list.append({
                    "filename": f,
                    "duration_sec": dur,
                    "title": desc_map.get(f, f"Industrial Accident Clip: {f}"),
                    "is_accident": True,
                    "accident_type": "FORKLIFT_VEHICLE_COLLISION" if "Forklift" in f else "CARGO_STACK_COLLAPSE"
                })

    # Add negative normal control
    norm_p = os.path.join(WORKSPACE_ROOT, "dataset", "no_fire", "normal_pedestrian_cctv.mp4")
    if os.path.exists(norm_p):
        sample_list.append({
            "filename": "normal_pedestrian_cctv.mp4",
            "duration_sec": 12.0,
            "title": "Normal Public Pedestrian Walking (False Positive Control)",
            "is_accident": False,
            "accident_type": "SAFE_NORMAL"
        })

    return {"samples": sample_list}


@router.get("/events")
async def get_accident_events():
    """
    Returns recorded industrial safety incidents for OSHA / EHS management.
    """
    return {"events": ACCIDENT_EVENTS}


@router.get("/metrics")
async def get_accident_metrics():
    """
    Returns verified model accuracy and forensic performance benchmarks.
    """
    return global_industrial_detector.model_metrics
