import io
import cv2
import base64
import numpy as np
from typing import Dict, Any, List
from fastapi import APIRouter, UploadFile, File, HTTPException
from PIL import Image
from ...ai.inference_pipeline import global_pipeline

router = APIRouter(prefix="/detection", tags=["Real Neural Detection & Inference"])

@router.post("/image")
async def analyze_image(file: UploadFile = File(...)):
    """
    Analyzes an uploaded image using real YOLOv11 Multi-Object & Person Detection and the user's trained
    best_human_activity_v2.keras (Input: 260x260) model with spatial person-object interaction.
    """
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        np_image = np.array(image)
        bgr_frame = cv2.cvtColor(np_image, cv2.COLOR_RGB2BGR)

        h, w = bgr_frame.shape[:2]
        result = global_pipeline.process_frame(bgr_frame, camera_id="IMG-UPLOAD", camera_location="Forensic Image Upload")

        return {
            "filename": file.filename,
            "width": w,
            "height": h,
            "people_count": result["people_count"],
            "detections": result["people"],
            "objects_count": result.get("objects_count", 0),
            "objects": result.get("objects", []),
            "suspicious_patterns_detected": [p for p in result["people"] if p["is_suspicious"]],
            "ai_model": "YOLOv11-Nano + best_human_activity_v2.keras",
            "processed_at": result["timestamp"]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Image inference error: {str(e)}")

@router.post("/frame")
async def process_raw_frame(payload: dict):
    """
    Processes real live webcam frames sent from the browser in real-time.
    Runs YOLOv11 person and multi-object detection -> crops person -> fuses with best_human_activity_v2.keras.
    """
    base64_data = payload.get("image_base64", "")
    camera_id = payload.get("camera_id", "LIVE-WEBCAM")
    
    if base64_data and "," in base64_data:
        base64_data = base64_data.split(",")[1]

    frame = None
    if base64_data:
        try:
            img_bytes = base64.b64decode(base64_data)
            np_arr = np.frombuffer(img_bytes, np.uint8)
            frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        except Exception:
            frame = None

    result = global_pipeline.process_frame(frame, camera_id=camera_id)
    return result
