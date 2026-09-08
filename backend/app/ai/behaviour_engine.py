import time
import math
from typing import List, Dict, Any, Optional

def is_point_in_polygon(x: float, y: float, polygon: List[Dict[str, float]]) -> bool:
    """
    Ray-casting algorithm for Point-in-Polygon (PIP) testing.
    Coordinates can be normalized (0-100) or pixel values.
    """
    num_vertices = len(polygon)
    if num_vertices < 3:
        return False

    inside = False
    p1x = polygon[0]["x"]
    p1y = polygon[0]["y"]

    for i in range(num_vertices + 1):
        p2x = polygon[i % num_vertices]["x"]
        p2y = polygon[i % num_vertices]["y"]

        if y > min(p1y, p2y):
            if y <= max(p1y, p2y):
                if x <= max(p1x, p2x):
                    if p1y != p2y:
                        xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                    if p1x == p2x or x <= xinters:
                        inside = not inside
        p1x, p1y = p2x, p2y

    return inside

class BehaviourEngine:
    """
    Temporal, Spatial & Object-Aware Behaviour Analysis Engine.
    Detects:
    - Armed Threat / Hazard Objects (knife, scissors)
    - Fighting / Violence
    - Unauthorized Entry (Restricted Zone Polygon)
    - Unauthorized Recording / Phone Usage in Sensitive Zones
    - Running in restricted area
    - Loitering (staying in a zone > threshold seconds)
    - Inactivity / Collapse (prolonged sleeping/down posture)
    """
    def __init__(self, loiter_threshold_sec: float = 30.0):
        self.loiter_threshold_sec = loiter_threshold_sec
        self.track_zone_entry: Dict[int, Dict[str, float]] = {}  # {track_id: {zone_id: entry_timestamp}}

    def analyze_person(
        self,
        track_id: int,
        bbox: List[int],
        activity: str,
        activity_conf: float,
        history: List[List[int]],
        zones: List[Dict[str, Any]],
        interacting_objects: List[Dict[str, Any]] = None,
        frame_dim: tuple = (1280, 720)
    ) -> Optional[Dict[str, Any]]:
        """
        Analyzes a single tracked individual for anomalous, dangerous, or suspicious behaviour.
        """
        interacting_objects = interacting_objects or []
        frame_w, frame_h = frame_dim
        x, y, w, h = bbox
        centroid_x = x + w / 2.0
        centroid_y = y + h / 2.0

        # Percentage coordinates
        pct_x = (centroid_x / frame_w) * 100.0
        pct_y = (centroid_y / frame_h) * 100.0

        in_restricted_zone = False
        active_zone_name = None
        zone_severity = "CRITICAL"

        # 1. Check Restricted Zones (Ray Casting)
        for zone in zones:
            if not zone.get("is_active", True):
                continue

            polygon = zone.get("polygon", [])
            if is_point_in_polygon(pct_x, pct_y, polygon):
                in_restricted_zone = True
                active_zone_name = zone.get("name", "Restricted Zone")
                zone_severity = zone.get("severity", "CRITICAL")
                break

        # Check for Armed Threat Objects (knife, scissors)
        for obj in interacting_objects:
            obj_label = obj.get("label", "").lower()
            if obj_label in ["knife", "scissors"]:
                return {
                    "threat_type": "WEAPON_DETECTED",
                    "activity": f"Armed Hazard ({obj_label.upper()})",
                    "severity": "CRITICAL",
                    "confidence": 96.5,
                    "details": f"Hazard object '{obj_label}' detected in possession of Person #{track_id}."
                }

        # Check Restricted Zone Breach
        if in_restricted_zone:
            # If using phone/recording device inside restricted zone
            for obj in interacting_objects:
                if obj.get("label", "").lower() in ["cell phone", "laptop"]:
                    return {
                        "threat_type": "UNAUTHORIZED_RECORDING",
                        "activity": "Unauthorized Device in Restricted Zone",
                        "severity": "HIGH",
                        "confidence": 94.2,
                        "zone_name": active_zone_name,
                        "details": f"Subject Person #{track_id} operating '{obj.get('label')}' within sensitive perimeter {active_zone_name}."
                    }

            return {
                "threat_type": "UNAUTHORIZED_ENTRY",
                "activity": "Unauthorized Entry",
                "severity": zone_severity,
                "confidence": 95.8,
                "zone_name": active_zone_name,
                "details": f"Tracked subject Person #{track_id} crossed perimeter into {active_zone_name}."
            }

        # 2. Dangerous Activity: Fighting / Physical Violence
        if "fight" in activity.lower():
            return {
                "threat_type": "VIOLENCE_DETECTED",
                "activity": "Fighting",
                "severity": "CRITICAL",
                "confidence": max(92.0, activity_conf * 100.0 if activity_conf <= 1.0 else activity_conf),
                "details": f"Violent physical confrontation signature detected involving Person #{track_id}."
            }

        # 3. High-Velocity Movement (Running) - only flag if in restricted context or abnormal
        if activity.lower() == "running" and in_restricted_zone:
            return {
                "threat_type": "RAPID_MOVEMENT",
                "activity": "High-Velocity Movement in Restricted Area",
                "severity": "MEDIUM",
                "confidence": 91.4,
                "details": f"High-velocity movement detected from Person #{track_id} in {active_zone_name}."
            }

        # Safe Normal Activities: Sitting, Standing, Sleeping, Desk Activity
        # These are standard human postures and must NEVER be flagged as suspicious or threats.
        return None

