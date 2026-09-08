from typing import List, Optional, Dict, Any
from datetime import datetime
from pydantic import BaseModel, Field

# Camera Schemas
class CameraBase(BaseModel):
    name: str
    stream_url: str
    location: str
    camera_type: str = "Fixed IP"
    sensitivity: float = 0.75

class CameraCreate(CameraBase):
    id: Optional[str] = None
    fps: int = 30
    resolution: str = "1920x1080"

class CameraUpdate(BaseModel):
    name: Optional[str] = None
    stream_url: Optional[str] = None
    location: Optional[str] = None
    camera_type: Optional[str] = None
    status: Optional[str] = None
    sensitivity: Optional[float] = None
    is_recording: Optional[bool] = None

class CameraResponse(CameraBase):
    id: str
    status: str
    fps: int
    resolution: str
    ai_confidence: float
    people_detected: int
    active_threats: int
    is_recording: bool
    created_at: datetime

    class Config:
        from_attributes = True

# Bounding Box & Detection
class BoundingBox(BaseModel):
    x: float
    y: float
    width: float
    height: float

class DetectedObject(BaseModel):
    label: str
    category: str = "general_object"
    confidence: float
    bbox: BoundingBox

class TrackedPerson(BaseModel):
    track_id: int
    bbox: BoundingBox
    activity: str
    confidence: float
    interacting_objects: List[DetectedObject] = []
    is_suspicious: bool = False
    threat_level: str = "NORMAL"  # NORMAL, LOW, MEDIUM, HIGH, CRITICAL
    first_seen: Optional[datetime] = None
    duration_sec: float = 0.0

class FrameDetectionResult(BaseModel):
    camera_id: str
    timestamp: datetime
    fps: float
    people_count: int
    people: List[TrackedPerson]
    objects_count: int = 0
    objects: List[DetectedObject] = []
    active_alerts: List[Dict[str, Any]]
    frame_base64: Optional[str] = None

# Restricted Zone Schemas
class Point(BaseModel):
    x: float
    y: float

class RestrictedZoneBase(BaseModel):
    camera_id: str
    name: str
    polygon: List[Point]
    severity: str = "CRITICAL"
    alert_type: str = "UNAUTHORIZED_ENTRY"
    is_active: bool = True
    color: str = "#ef4444"
    schedule: str = "24/7"

class RestrictedZoneCreate(RestrictedZoneBase):
    id: Optional[str] = None

class RestrictedZoneResponse(RestrictedZoneBase):
    id: str
    created_at: datetime

    class Config:
        from_attributes = True

# Incident Schemas
class IncidentBase(BaseModel):
    camera_id: str
    location: str
    activity: str
    severity: str  # LOW, MEDIUM, HIGH, CRITICAL
    confidence: float
    status: str = "OPEN"  # OPEN, INVESTIGATING, RESOLVED
    person_track_id: Optional[int] = None
    details: Optional[str] = None
    ai_reasoning: Optional[str] = None

class IncidentCreate(IncidentBase):
    id: Optional[str] = None
    snapshot_url: Optional[str] = None
    video_clip_url: Optional[str] = None

class IncidentUpdate(BaseModel):
    status: Optional[str] = None
    resolved_by: Optional[str] = None
    resolution_notes: Optional[str] = None

class ActivityTimelineItem(BaseModel):
    activity: str
    confidence: float
    time_offset_sec: float
    is_suspicious: bool
    interacting_objects: List[str] = []

class IncidentResponse(IncidentBase):
    id: str
    timestamp: datetime
    snapshot_url: Optional[str] = None
    video_clip_url: Optional[str] = None
    resolved_by: Optional[str] = None
    resolution_notes: Optional[str] = None
    created_at: datetime
    events: List[ActivityTimelineItem] = []

    class Config:
        from_attributes = True

# Video Analysis Schemas
class VideoAnalysisStatus(BaseModel):
    video_id: str
    status: str  # QUEUED, PROCESSING, COMPLETED, FAILED
    progress_percentage: float
    frames_processed: int
    total_frames: int
    duration_sec: float
    fps: float
    people_detected_total: int
    suspicious_events_count: int
    dominant_activity: str
    ai_confidence: float
    timeline: List[ActivityTimelineItem]
    incidents_generated: List[str]

# System & Model Telemetry
class SystemStatus(BaseModel):
    ai_engine_status: str = "ONLINE"
    model_mode: str = "REAL_AI"  # REAL_AI or DEMO_SIMULATION
    connected_cameras: int = 8
    total_cameras: int = 12
    total_people_detected: int = 37
    active_alerts: int = 4
    today_incidents: int = 18
    avg_ai_confidence: float = 94.6
    gpu_usage_percent: float = 42.5
    cpu_usage_percent: float = 28.3
    ram_usage_percent: float = 54.1
    inference_fps: float = 38.4
    websocket_clients: int = 1

class AIModelInfo(BaseModel):
    name: str
    version: str
    architecture: str
    classes: List[str]
    test_accuracy: float
    status: str
    model_path: str
    framework: str
