import datetime
from sqlalchemy import (
    Column, Integer, String, Float, Boolean, DateTime, Text, ForeignKey, JSON
)
from sqlalchemy.orm import declarative_base, relationship

Base = declarative_base()

class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    username = Column(String(64), unique=True, nullable=False, index=True)
    email = Column(String(128), unique=True, nullable=False)
    hashed_password = Column(String(256), nullable=False)
    role = Column(String(32), default="Security Operator")  # Admin, Security Operator, Viewer
    full_name = Column(String(128), default="Security Officer")
    badge_number = Column(String(32), default="SOC-8821")
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)

class Camera(Base):
    __tablename__ = "cameras"

    id = Column(String(32), primary_key=True, index=True)  # e.g., CAM-001
    name = Column(String(128), nullable=False)
    stream_url = Column(String(512), nullable=False)
    location = Column(String(128), nullable=False)
    camera_type = Column(String(32), default="Fixed IP")  # Fixed IP, PTZ, Dome, Thermal, Webcam
    status = Column(String(32), default="ONLINE")  # ONLINE, OFFLINE, CONNECTING, ERROR
    fps = Column(Integer, default=30)
    resolution = Column(String(32), default="1920x1080")
    ai_confidence = Column(Float, default=94.5)
    people_detected = Column(Integer, default=0)
    active_threats = Column(Integer, default=0)
    sensitivity = Column(Float, default=0.75)
    is_recording = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)

    # Relationships
    incidents = relationship("Incident", back_populates="camera_rel", cascade="all, delete-orphan")
    zones = relationship("RestrictedZone", back_populates="camera_rel", cascade="all, delete-orphan")

class Incident(Base):
    __tablename__ = "incidents"

    id = Column(String(32), primary_key=True, index=True)  # e.g., INC-1024
    camera_id = Column(String(32), ForeignKey("cameras.id"), nullable=False, index=True)
    timestamp = Column(DateTime, default=datetime.datetime.utcnow, index=True)
    location = Column(String(128), nullable=False)
    activity = Column(String(64), nullable=False)  # Fighting, Running in Restricted Zone, Loitering, Unauthorized Entry
    severity = Column(String(32), nullable=False, index=True)  # LOW, MEDIUM, HIGH, CRITICAL
    confidence = Column(Float, nullable=False)
    status = Column(String(32), default="OPEN", index=True)  # OPEN, INVESTIGATING, RESOLVED
    person_track_id = Column(Integer, nullable=True)
    details = Column(Text, nullable=True)
    ai_reasoning = Column(Text, nullable=True)
    snapshot_url = Column(String(512), nullable=True)
    video_clip_url = Column(String(512), nullable=True)
    resolved_by = Column(String(64), nullable=True)
    resolution_notes = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)

    camera_rel = relationship("Camera", back_populates="incidents")
    events = relationship("ActivityEvent", back_populates="incident_rel", cascade="all, delete-orphan")

class ActivityEvent(Base):
    __tablename__ = "activity_events"

    id = Column(Integer, primary_key=True, index=True)
    incident_id = Column(String(32), ForeignKey("incidents.id"), nullable=True)
    camera_id = Column(String(32), nullable=False, index=True)
    timestamp = Column(DateTime, default=datetime.datetime.utcnow)
    person_track_id = Column(Integer, nullable=False)
    activity = Column(String(64), nullable=False)
    confidence = Column(Float, nullable=False)
    bounding_box = Column(JSON, nullable=True)  # [x, y, w, h] or [x1, y1, x2, y2]
    is_suspicious = Column(Boolean, default=False)
    time_offset_sec = Column(Float, default=0.0)

    incident_rel = relationship("Incident", back_populates="events")

class Detection(Base):
    __tablename__ = "detections"

    id = Column(Integer, primary_key=True, index=True)
    camera_id = Column(String(32), nullable=False, index=True)
    timestamp = Column(DateTime, default=datetime.datetime.utcnow)
    track_id = Column(Integer, nullable=False)
    bbox = Column(JSON, nullable=False)  # {"x": 100, "y": 150, "width": 80, "height": 180}
    activity = Column(String(64), nullable=False)
    confidence = Column(Float, nullable=False)
    is_threat = Column(Boolean, default=False)
    threat_type = Column(String(64), nullable=True)

class RestrictedZone(Base):
    __tablename__ = "restricted_zones"

    id = Column(String(32), primary_key=True, index=True)  # e.g., ZONE-001
    camera_id = Column(String(32), ForeignKey("cameras.id"), nullable=False)
    name = Column(String(128), nullable=False)
    polygon = Column(JSON, nullable=False)  # [{"x": 120, "y": 80}, ...]
    severity = Column(String(32), default="CRITICAL")  # LOW, MEDIUM, HIGH, CRITICAL
    alert_type = Column(String(64), default="UNAUTHORIZED_ENTRY")
    is_active = Column(Boolean, default=True)
    color = Column(String(32), default="#ef4444")
    schedule = Column(String(64), default="24/7")  # 24/7, After-Hours (20:00-06:00)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)

    camera_rel = relationship("Camera", back_populates="zones")

class SystemLog(Base):
    __tablename__ = "system_logs"

    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime, default=datetime.datetime.utcnow)
    level = Column(String(32), default="INFO")  # INFO, WARNING, ERROR, CRITICAL
    component = Column(String(64), default="AI_ENGINE")
    message = Column(Text, nullable=False)
    metadata_info = Column(JSON, nullable=True)
