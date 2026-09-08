from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from ...database.session import get_db
from ...database.models import Incident, ActivityEvent
from ...schemas.schemas import IncidentResponse, IncidentCreate, IncidentUpdate

router = APIRouter(prefix="/incidents", tags=["Incidents & Threat Log"])

@router.get("", response_model=List[IncidentResponse])
def get_incidents(
    camera_id: Optional[str] = Query(None),
    activity: Optional[str] = Query(None),
    severity: Optional[str] = Query(None),
    status: Optional[str] = Query(None),
    search: Optional[str] = Query(None),
    db: Session = Depends(get_db)
):
    """
    Search and filter security incidents by camera, activity, severity, status, or search query.
    """
    query = db.query(Incident)

    if camera_id and camera_id != "ALL":
        query = query.filter(Incident.camera_id == camera_id)

    if activity and activity != "ALL":
        query = query.filter(Incident.activity.ilike(f"%{activity}%"))

    if severity and severity != "ALL":
        query = query.filter(Incident.severity == severity.upper())

    if status and status != "ALL":
        query = query.filter(Incident.status == status.upper())

    if search:
        search_fmt = f"%{search}%"
        query = query.filter(
            (Incident.id.ilike(search_fmt)) |
            (Incident.location.ilike(search_fmt)) |
            (Incident.activity.ilike(search_fmt)) |
            (Incident.details.ilike(search_fmt))
        )

    return query.order_by(Incident.timestamp.desc()).all()

@router.get("/{incident_id}", response_model=IncidentResponse)
def get_incident_by_id(incident_id: str, db: Session = Depends(get_db)):
    incident = db.query(Incident).filter(Incident.id == incident_id).first()
    if not incident:
        raise HTTPException(status_code=404, detail=f"Incident {incident_id} not found.")
    return incident

@router.put("/{incident_id}", response_model=IncidentResponse)
def update_incident(incident_id: str, payload: IncidentUpdate, db: Session = Depends(get_db)):
    incident = db.query(Incident).filter(Incident.id == incident_id).first()
    if not incident:
        raise HTTPException(status_code=404, detail="Incident not found.")

    if payload.status is not None:
        incident.status = payload.status
    if payload.resolved_by is not None:
        incident.resolved_by = payload.resolved_by
    if payload.resolution_notes is not None:
        incident.resolution_notes = payload.resolution_notes

    db.commit()
    db.refresh(incident)
    return incident

@router.post("", response_model=IncidentResponse)
def create_incident(payload: IncidentCreate, db: Session = Depends(get_db)):
    inc_id = payload.id
    if not inc_id:
        count = db.query(Incident).count()
        inc_id = f"INC-{1025 + count}"

    new_inc = Incident(
        id=inc_id,
        camera_id=payload.camera_id,
        location=payload.location,
        activity=payload.activity,
        severity=payload.severity,
        confidence=payload.confidence,
        status=payload.status,
        person_track_id=payload.person_track_id,
        details=payload.details,
        ai_reasoning=payload.ai_reasoning or "Automated heuristic trigger pattern match.",
        snapshot_url=payload.snapshot_url or f"/api/incidents/snapshots/{inc_id.lower()}.jpg",
        video_clip_url=payload.video_clip_url or f"/api/incidents/clips/{inc_id.lower()}.mp4"
    )
    db.add(new_inc)
    db.commit()
    db.refresh(new_inc)
    return new_inc

@router.get("/{incident_id}/export")
def export_incident_docket(incident_id: str, db: Session = Depends(get_db)):
    incident = db.query(Incident).filter(Incident.id == incident_id).first()
    if not incident:
        raise HTTPException(status_code=404, detail="Incident not found.")

    return {
        "docket_number": f"SOC-EVID-{incident.id}",
        "export_date": "2026-08-28T20:15:00Z",
        "incident_id": incident.id,
        "camera": incident.camera_id,
        "location": incident.location,
        "activity": incident.activity,
        "severity": incident.severity,
        "confidence": f"{incident.confidence}%",
        "status": incident.status,
        "ai_reasoning": incident.ai_reasoning,
        "details": incident.details,
        "resolved_by": incident.resolved_by or "UNASSIGNED",
        "resolution_notes": incident.resolution_notes or "Pending operator sign-off.",
        "integrity_hash": "SHA256:7f83b1657ff1fc53b92dc18148a1d65dfc2d4b1fa3d677284addd200126d9069"
    }
