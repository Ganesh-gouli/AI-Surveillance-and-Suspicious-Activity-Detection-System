import os
import sys
import uuid
import time
import json
import tempfile
import cv2
import numpy as np
from typing import Dict, Any, Optional, List
from fastapi import APIRouter, UploadFile, File, Form, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse, StreamingResponse

# Ensure root dir in python path
WORKSPACE_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
if WORKSPACE_ROOT not in sys.path:
    sys.path.insert(0, WORKSPACE_ROOT)

from detection.atm_vandalism_detector import global_atm_detector

router = APIRouter(prefix="/atm", tags=["ATM Breaking & Vandalism Detection AI"])

ATM_JOBS: Dict[str, Dict[str, Any]] = {}
ATM_EVENTS: List[Dict[str, Any]] = []

RESULTS_DIR = os.path.join(WORKSPACE_ROOT, "results")
PROCESSED_VIDEOS_DIR = os.path.join(RESULTS_DIR, "processed_videos")
VIDEOS_DIR = os.path.join(WORKSPACE_ROOT, "VIDEOS")
os.makedirs(PROCESSED_VIDEOS_DIR, exist_ok=True)


def background_analyze_atm_video(
    job_id: str,
    temp_input_path: str,
    output_video_path: str,
    filename: str,
    confidence_threshold: float,
    sample_rate: int,
    enable_pose: bool = True
):
    try:
        def update_progress(pct: float):
            if job_id in ATM_JOBS:
                ATM_JOBS[job_id]["progress_percentage"] = pct

        analysis = global_atm_detector.analyze_video(
            video_path=temp_input_path,
            output_video_path=output_video_path,
            confidence_threshold=confidence_threshold,
            sample_rate=sample_rate,
            enable_pose=enable_pose,
            progress_callback=update_progress
        )

        canonical_path = os.path.join(RESULTS_DIR, "atm_detected_video.mp4")
        if os.path.exists(output_video_path):
            import shutil
            shutil.copy(output_video_path, canonical_path)

        result_data = {
            "status": "COMPLETED",
            "progress_percentage": 100.0,
            "filename": filename,
            "final_result": analysis["final_result"],
            "is_vandalism": analysis["is_vandalism"],
            "attack_type": analysis["attack_type"],
            "overall_confidence": analysis["overall_confidence"],
            "tool_detected": analysis["tool_detected"],
            "perpetrators_count": analysis["perpetrators_count"],
            "analyzed_frames": analysis["analyzed_frames"],
            "total_video_frames": analysis["total_video_frames"],
            "vandalism_frames": analysis["vandalism_frames"],
            "duration_sec": analysis["duration_sec"],
            "processing_time_sec": analysis["processing_time_sec"],
            "processing_fps": analysis.get("processing_fps", 0.0),
            "segments": analysis.get("segments", []),
            "timeline": analysis.get("timeline", [])[:150],
            "video_url": f"/api/atm/video/{os.path.basename(output_video_path)}"
        }

        if job_id in ATM_JOBS:
            ATM_JOBS[job_id].update(result_data)

        # Log event if ATM attack/vandalism detected
        if analysis["is_vandalism"]:
            event = {
                "id": f"ATM-BREACH-{uuid.uuid4().hex[:8].upper()}",
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "source": filename,
                "event_type": analysis["attack_type"],
                "tool_detected": analysis["tool_detected"],
                "perpetrators_count": analysis["perpetrators_count"],
                "confidence": analysis["overall_confidence"],
                "duration_sec": analysis["duration_sec"],
                "status": "CRITICAL_OPEN"
            }
            ATM_EVENTS.insert(0, event)
            if len(ATM_EVENTS) > 50:
                ATM_EVENTS.pop()

    except Exception as e:
        import traceback
        traceback.print_exc()
        if job_id in ATM_JOBS:
            ATM_JOBS[job_id].update({
                "status": "FAILED",
                "error": str(e)
            })
    finally:
        # Clean up temporary uploaded file if it was created in a temp dir
        if "temp" in temp_input_path and os.path.exists(temp_input_path):
            try:
                os.remove(temp_input_path)
            except Exception:
                pass


@router.post("/analyze-video")
async def analyze_atm_video(
    background_tasks: BackgroundTasks,
    video: UploadFile = File(...),
    confidence_threshold: float = Form(0.38),
    sample_rate: int = Form(3),
    enable_pose: bool = Form(True)
):
    """
    Accepts surveillance video upload and queues asynchronous ATM Vandalism Forensics.
    """
    if not video.filename.lower().endswith((".mp4", ".avi", ".mov", ".mkv", ".webm")):
        raise HTTPException(status_code=400, detail="Invalid video format. Supported: mp4, avi, mov, mkv, webm.")

    job_id = f"ATM-JOB-{uuid.uuid4().hex[:8].upper()}"
    temp_dir = tempfile.gettempdir()
    temp_input_path = os.path.join(temp_dir, f"{job_id}_{video.filename}")

    with open(temp_input_path, "wb") as f:
        content = await video.read()
        f.write(content)

    output_filename = f"detected_atm_{job_id}.mp4"
    output_video_path = os.path.join(PROCESSED_VIDEOS_DIR, output_filename)

    ATM_JOBS[job_id] = {
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
        background_analyze_atm_video,
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
        "message": "ATM breaking forensics queued successfully."
    }


@router.post("/analyze-sample")
async def analyze_sample_video(
    background_tasks: BackgroundTasks,
    filename: str = Form(...),
    confidence_threshold: float = Form(0.38),
    sample_rate: int = Form(3),
    enable_pose: bool = Form(True)
):
    """
    Fast 1-click execution on preloaded repository sample videos from VIDEOS/ folder.
    """
    video_path = os.path.join(VIDEOS_DIR, filename)
    if not os.path.exists(video_path):
        # Fallback check in dataset/no_fire
        fallback = os.path.join(WORKSPACE_ROOT, "dataset", "no_fire", filename)
        if os.path.exists(fallback):
            video_path = fallback
        else:
            raise HTTPException(status_code=404, detail=f"Sample video '{filename}' not found.")

    job_id = f"ATM-JOB-{uuid.uuid4().hex[:8].upper()}"
    output_filename = f"detected_atm_{job_id}.mp4"
    output_video_path = os.path.join(PROCESSED_VIDEOS_DIR, output_filename)

    ATM_JOBS[job_id] = {
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
        background_analyze_atm_video,
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
        "message": f"Preloaded ATM video '{filename}' analysis initialized."
    }


@router.get("/job/{job_id}")
async def get_atm_job_status(job_id: str):
    """
    Returns current analysis progress and completed results.
    """
    if job_id not in ATM_JOBS:
        raise HTTPException(status_code=404, detail="Job ID not found.")
    return ATM_JOBS[job_id]


@router.get("/video/{filename}")
async def get_atm_processed_video(filename: str):
    """
    Streams browser-native H.264 video with partial content range support.
    """
    video_path = os.path.join(PROCESSED_VIDEOS_DIR, filename)
    if not os.path.exists(video_path):
        canonical = os.path.join(RESULTS_DIR, "atm_detected_video.mp4")
        if os.path.exists(canonical):
            video_path = canonical
        else:
            raise HTTPException(status_code=404, detail="Video not found.")

    return FileResponse(
        video_path,
        media_type="video/mp4",
        filename=filename,
        headers={"Accept-Ranges": "bytes"}
    )


@router.get("/samples")
async def list_sample_videos():
    """
    Lists available benchmark videos in VIDEOS directory for immediate testing.
    """
    sample_list = []
    if os.path.exists(VIDEOS_DIR):
        for f in sorted(os.listdir(VIDEOS_DIR)):
            if f.endswith((".mp4", ".avi", ".mov")):
                full_p = os.path.join(VIDEOS_DIR, f)
                cap = cv2.VideoCapture(full_p)
                fps = cap.get(cv2.CAP_PROP_FPS) or 25
                cnt = cap.get(cv2.CAP_PROP_FRAME_COUNT)
                dur = round(cnt / fps, 1) if fps > 0 else 0.0
                cap.release()

                desc_map = {
                    "30a.mp4": "Violent Screen & Chassis Sledge/Pipe Smashing",
                    "30b.mp4": "Armed Intrusion & Kiosk Blunt Force Attack",
                    "5.mp4": "Coordinated Crowbar Vault Prying & Safe Breach",
                    "75.mp4": "Tow Chain Dislodgement & ATM Kiosk Yank",
                    "80.mp4": "Lower Cash Dispenser Tampering & Forced Entry",
                    "88.mp4": "Midnight Masked Rock/Hammer Violent Smashing"
                }

                sample_list.append({
                    "filename": f,
                    "duration_sec": dur,
                    "title": desc_map.get(f, f"ATM Surveillance Clip {f}"),
                    "is_attack": True,
                    "attack_type": "PHYSICAL_ATTACK"
                })

    # Add negative normal control
    norm_p = os.path.join(WORKSPACE_ROOT, "dataset", "no_fire", "normal_pedestrian_cctv.mp4")
    if os.path.exists(norm_p):
        sample_list.append({
            "filename": "normal_pedestrian_cctv.mp4",
            "duration_sec": 12.0,
            "title": "Normal Public Pedestrian Walking (False Positive Control)",
            "is_attack": False,
            "attack_type": "SAFE_NORMAL"
        })

    return {"samples": sample_list}


@router.get("/events")
async def get_atm_events():
    """
    Returns recorded ATM threat events for SOC incident management.
    """
    return {"events": ATM_EVENTS}


@router.get("/metrics")
async def get_atm_metrics():
    """
    Returns verified model accuracy and forensic performance benchmarks.
    """
    return global_atm_detector.model_metrics
