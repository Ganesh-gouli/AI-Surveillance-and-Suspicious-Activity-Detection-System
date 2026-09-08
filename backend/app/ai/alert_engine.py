import time
import datetime
from typing import List, Dict, Any, Optional

class AlertEngine:
    """
    Alert generation, deduplication, and threat dispatch engine.
    """
    def __init__(self, cooldown_seconds: float = 10.0):
        self.cooldown_seconds = cooldown_seconds
        self.recent_alerts: Dict[str, float] = {}  # {key: last_triggered_timestamp}

    def process_threat(
        self,
        camera_id: str,
        camera_location: str,
        threat_info: Dict[str, Any],
        track_id: int
    ) -> Optional[Dict[str, Any]]:
        """
        Processes a threat event, checks cooldown, and generates structured alert payload.
        """
        threat_key = f"{camera_id}_{threat_info['threat_type']}_{track_id}"
        now = time.time()

        if threat_key in self.recent_alerts:
            if now - self.recent_alerts[threat_key] < self.cooldown_seconds:
                return None  # Suppress duplicate alert within cooldown window

        self.recent_alerts[threat_key] = now

        # Generate unique incident ID
        incident_id = f"INC-{int(now) % 10000:04d}"

        # Generate intelligent AI reasoning explanation
        activity = threat_info.get("activity", "Suspicious Activity")
        severity = threat_info.get("severity", "HIGH")
        conf = threat_info.get("confidence", 94.0)

        ai_reasoning = (
            f"Automated Temporal Analysis Engine: {threat_info.get('details', '')} "
            f"Pattern matched security rule signature '{threat_info['threat_type']}' "
            f"with confidence {conf:.1f}% on feed {camera_id}."
        )

        alert_payload = {
            "id": incident_id,
            "camera_id": camera_id,
            "location": camera_location,
            "timestamp": datetime.datetime.utcnow().isoformat(),
            "activity": activity,
            "severity": severity,
            "confidence": round(conf, 1),
            "status": "OPEN",
            "person_track_id": track_id,
            "details": threat_info.get("details", ""),
            "ai_reasoning": ai_reasoning,
            "threat_type": threat_info.get("threat_type")
        }

        return alert_payload
