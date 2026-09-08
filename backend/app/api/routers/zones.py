from typing import List
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from ...database.session import get_db
from ...database.models import RestrictedZone
from ...schemas.schemas import RestrictedZoneCreate, RestrictedZoneResponse
from ...ai.behaviour_engine import is_point_in_polygon

router = APIRouter(prefix="/zones", tags=["Restricted Zones"])

@router.get("", response_model=List[RestrictedZoneResponse])
def get_zones(camera_id: str = None, db: Session = Depends(get_db)):
    query = db.query(RestrictedZone)
    if camera_id:
        query = query.filter(RestrictedZone.camera_id == camera_id)
    return query.all()

@router.post("", response_model=RestrictedZoneResponse, status_code=status.HTTP_201_CREATED)
def create_zone(payload: RestrictedZoneCreate, db: Session = Depends(get_db)):
    zone_id = payload.id
    if not zone_id:
        count = db.query(RestrictedZone).count()
        zone_id = f"ZONE-{count + 1:03d}"

    polygon_dicts = [{"x": p.x, "y": p.y} for p in payload.polygon]

    new_zone = RestrictedZone(
        id=zone_id,
        camera_id=payload.camera_id,
        name=payload.name,
        polygon=polygon_dicts,
        severity=payload.severity,
        alert_type=payload.alert_type,
        is_active=payload.is_active,
        color=payload.color,
        schedule=payload.schedule
    )
    db.add(new_zone)
    db.commit()
    db.refresh(new_zone)
    return new_zone

@router.delete("/{zone_id}")
def delete_zone(zone_id: str, db: Session = Depends(get_db)):
    zone = db.query(RestrictedZone).filter(RestrictedZone.id == zone_id).first()
    if not zone:
        raise HTTPException(status_code=404, detail="Zone not found.")
    db.delete(zone)
    db.commit()
    return {"message": f"Restricted zone {zone_id} removed."}

@router.post("/test-point")
def test_point_containment(payload: dict):
    x = float(payload.get("x", 0))
    y = float(payload.get("y", 0))
    polygon = payload.get("polygon", [])
    
    inside = is_point_in_polygon(x, y, polygon)
    return {
        "x": x,
        "y": y,
        "is_breached": inside,
        "status": "UNAUTHORIZED_ENTRY" if inside else "SECURE"
    }
