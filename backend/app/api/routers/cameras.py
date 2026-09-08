from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from ...database.session import get_db
from ...database.models import Camera
from ...schemas.schemas import CameraCreate, CameraUpdate, CameraResponse

router = APIRouter(prefix="/cameras", tags=["Cameras"])

@router.get("", response_model=List[CameraResponse])
def get_all_cameras(db: Session = Depends(get_db)):
    """Retrieve all configured CCTV and IP cameras in the SOC network."""
    return db.query(Camera).all()

@router.get("/{camera_id}", response_model=CameraResponse)
def get_camera_by_id(camera_id: str, db: Session = Depends(get_db)):
    camera = db.query(Camera).filter(Camera.id == camera_id).first()
    if not camera:
        raise HTTPException(status_code=404, detail=f"Camera with ID {camera_id} not found.")
    return camera

@router.post("", response_model=CameraResponse, status_code=status.HTTP_201_CREATED)
def create_camera(payload: CameraCreate, db: Session = Depends(get_db)):
    # Generate unique ID if not provided
    cam_id = payload.id
    if not cam_id:
        count = db.query(Camera).count()
        cam_id = f"CAM-{count + 1:03d}"

    existing = db.query(Camera).filter(Camera.id == cam_id).first()
    if existing:
        raise HTTPException(status_code=400, detail=f"Camera {cam_id} already exists.")

    new_camera = Camera(
        id=cam_id,
        name=payload.name,
        stream_url=payload.stream_url,
        location=payload.location,
        camera_type=payload.camera_type,
        sensitivity=payload.sensitivity,
        fps=payload.fps,
        resolution=payload.resolution,
        status="ONLINE",
        ai_confidence=95.0,
        people_detected=0,
        active_threats=0,
        is_recording=True
    )
    db.add(new_camera)
    db.commit()
    db.refresh(new_camera)
    return new_camera

@router.put("/{camera_id}", response_model=CameraResponse)
def update_camera(camera_id: str, payload: CameraUpdate, db: Session = Depends(get_db)):
    camera = db.query(Camera).filter(Camera.id == camera_id).first()
    if not camera:
        raise HTTPException(status_code=404, detail="Camera not found.")

    for field, value in payload.dict(exclude_unset=True).items():
        setattr(camera, field, value)

    db.commit()
    db.refresh(camera)
    return camera

@router.delete("/{camera_id}")
def delete_camera(camera_id: str, db: Session = Depends(get_db)):
    camera = db.query(Camera).filter(Camera.id == camera_id).first()
    if not camera:
        raise HTTPException(status_code=404, detail="Camera not found.")
    db.delete(camera)
    db.commit()
    return {"message": f"Camera {camera_id} successfully decommissioned."}

@router.post("/{camera_id}/start")
def start_camera_feed(camera_id: str, db: Session = Depends(get_db)):
    camera = db.query(Camera).filter(Camera.id == camera_id).first()
    if not camera:
        raise HTTPException(status_code=404, detail="Camera not found.")
    camera.status = "ONLINE"
    db.commit()
    return {"status": "ONLINE", "camera_id": camera_id, "message": "Stream feed initialized."}

@router.post("/{camera_id}/stop")
def stop_camera_feed(camera_id: str, db: Session = Depends(get_db)):
    camera = db.query(Camera).filter(Camera.id == camera_id).first()
    if not camera:
        raise HTTPException(status_code=404, detail="Camera not found.")
    camera.status = "OFFLINE"
    camera.people_detected = 0
    camera.active_threats = 0
    db.commit()
    return {"status": "OFFLINE", "camera_id": camera_id, "message": "Stream feed paused."}

@router.post("/test-connection")
def test_stream_connection(payload: dict):
    """Simulates RTSP/HTTP handshake & latency test without exposing passwords."""
    url = payload.get("stream_url", "")
    if not url:
        raise HTTPException(status_code=400, detail="Stream URL is required.")
    
    # Secure test simulation
    is_valid = url.startswith("rtsp://") or url.startswith("http://") or url.startswith("https://")
    if is_valid:
        return {
            "success": True,
            "latency_ms": 24.8,
            "codec": "H.264 / AAC",
            "resolution": "1920x1080 @ 30fps",
            "bitrate": "4096 kbps",
            "message": "Handshake successful. Stream validated."
        }
    return {
        "success": False,
        "latency_ms": 0,
        "message": "Connection timed out. Verify network gateway & RTSP credentials."
    }
