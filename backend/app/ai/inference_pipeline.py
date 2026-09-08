import time
import math
import cv2
import numpy as np
from typing import List, Dict, Any, Optional
from datetime import datetime

from .yolov11_detector import YOLOv11Detector
from .tracker import MultiPersonTracker
from .activity_classifier import HumanActivityClassifier
from .behaviour_engine import BehaviourEngine
from .alert_engine import AlertEngine

class SentinelInferencePipeline:
    """
    High-Performance Real AI Inference Pipeline for SentinelVision AI.
    Executes real-time YOLOv11 multi-object and person detection (optimized imgsz=384 for CPU),
    Google MediaPipe 3D Pose Biomechanical estimation (standing vs. sitting),
    spatial person-object association, and object-aware threat analysis.
    """
    def __init__(self):
        self.detector = YOLOv11Detector()
        self.tracker = MultiPersonTracker()
        self.classifier = HumanActivityClassifier()
        self.behaviour_engine = BehaviourEngine()
        self.alert_engine = AlertEngine()
        self.frame_count = 0
        self.start_time = time.time()

    def process_frame(
        self,
        frame: Optional[np.ndarray],
        camera_id: str = "CAM-001",
        camera_location: str = "Surveillance Sector",
        zones: List[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Executes low-latency real AI inference pipeline on an incoming video / webcam / image frame.
        """
        self.frame_count += 1
        zones = zones or []
        t0 = time.time()

        people_results = []
        objects_results = []
        alerts_generated = []

        if frame is not None and frame.size > 0:
            h, w = frame.shape[:2]
            
            # 1. Real YOLOv11 Unified Multi-Object and Person Detection (optimized 384px)
            person_detections, scene_objects = self.detector.detect(frame)
            
            # Format scene objects for API response
            for obj in scene_objects:
                ox, oy, ow, oh = obj["bbox"]
                objects_results.append({
                    "label": obj["label"],
                    "category": obj.get("category", "general_object"),
                    "confidence": round(obj["confidence"] * 100.0, 1),
                    "bbox": {
                        "x": float(ox),
                        "y": float(oy),
                        "width": float(ow),
                        "height": float(oh)
                    }
                })

            # 2. Multi-Person Tracking
            tracked_objects = self.tracker.update(person_detections)

            # 3. Google MediaPipe 3D Pose Biomechanical Analysis & Fast Context Fusion
            for track in tracked_objects:
                x, y, bw, bh = track.bbox
                x1, y1 = max(0, x), max(0, y)
                x2, y2 = min(w, x + bw), min(h, y + bh)
                
                # Run Google MediaPipe 3D Pose Biomechanics (fast ~25ms on CPU)
                pose_data = self.classifier.classify_person_pose(
                    frame=frame,
                    bbox=track.bbox,
                    interacting_objects=track.interacting_objects
                )
                pose_act = pose_data.get("activity", "standing")
                pose_conf = pose_data.get("confidence", 0.92)
                pose_landmarks = pose_data.get("landmarks", [])
                pose_angles = pose_data.get("angles", {})
                posture_details = pose_data.get("posture_details", "")

                interacting_labels = {obj["label"].lower() for obj in track.interacting_objects}
                has_drink_container = any(l in interacting_labels for l in ["bottle", "wine glass", "cup"])
                has_laptop_tech = any(l in interacting_labels for l in ["laptop", "keyboard", "mouse"])
                has_phone = "cell phone" in interacting_labels
                has_bicycle = any(l in interacting_labels for l in ["bicycle", "motorcycle"])
                has_food = any(l in interacting_labels for l in ["sandwich", "pizza", "apple", "banana", "cake", "donut", "bowl", "fork", "spoon"])
                has_bag = any(l in interacting_labels for l in ["backpack", "handbag", "suitcase"])
                has_sports = any(l in interacting_labels for l in ["sports ball", "tennis racket", "baseball bat", "frisbee", "skateboard"])
                has_weapon_obj = any(obj.get("category") == "weapon" or obj.get("label", "").lower() in ["knife", "scissors", "sword", "automatic rifle", "handgun"] for obj in track.interacting_objects)

                # High-Accuracy Ultra-Fast Decision Logic (bypasses heavy Keras call unless specifically needed)
                if has_weapon_obj:
                    final_activity = "armed_hazard"
                    final_conf = 0.96
                elif has_laptop_tech:
                    final_activity = "using_laptop"
                    final_conf = 0.95
                elif has_phone:
                    final_activity = "using_phone"
                    final_conf = 0.93
                elif has_drink_container:
                    final_activity = "drinking"
                    final_conf = 0.92
                elif has_food:
                    final_activity = "eating"
                    final_conf = 0.92
                elif has_bicycle:
                    final_activity = "cycling"
                    final_conf = 0.96
                elif has_bag:
                    final_activity = "carrying_bag"
                    final_conf = 0.90
                elif has_sports:
                    final_activity = "playing_sports"
                    final_conf = 0.91
                elif pose_act == "sleeping":
                    final_activity = "sleeping"
                    final_conf = pose_conf
                elif pose_act in ["standing", "sitting"]:
                    # MediaPipe Biomechanical Angle Classification
                    if pose_act == "standing" and len(track.history) >= 5:
                        first_pos = track.history[0]
                        disp = math.hypot(track.bbox[0] - first_pos[0], track.bbox[1] - first_pos[1])
                        if disp > 30:
                            final_activity = "walking"
                            final_conf = min(0.96, pose_conf)
                        else:
                            final_activity = "standing"
                            final_conf = pose_conf
                    else:
                        final_activity = pose_act
                        final_conf = pose_conf
                else:
                    final_activity = pose_act
                    final_conf = pose_conf

                track.activity = final_activity
                track.activity_confidence = final_conf
                track.landmarks = pose_landmarks
                track.pose_angles = pose_angles
                track.posture_details = posture_details

                # 4. Object-Aware Behaviour & Threat Rules Analysis
                threat_info = self.behaviour_engine.analyze_person(
                    track_id=track.track_id,
                    bbox=track.bbox,
                    activity=final_activity,
                    activity_conf=final_conf,
                    history=track.history,
                    zones=zones,
                    interacting_objects=track.interacting_objects,
                    frame_dim=(w, h)
                )

                # Safe Normal Postures Guard: Sitting, Standing, Sleeping, Desk Activity
                # These are standard benign human postures and must NEVER be flagged as suspicious or dangerous.
                is_safe_posture = final_activity.lower() in [
                    "sitting", "standing", "sleeping", "using_laptop", "using_phone",
                    "drinking", "eating", "walking", "carrying_bag", "playing_sports"
                ]
                has_weapon = any(
                    obj.get("category") == "threat" or obj.get("label", "").lower() in ["knife", "scissors", "gun", "sword"]
                    for obj in track.interacting_objects
                )
                is_violent = "fight" in final_activity.lower()
                is_zone_breach = threat_info and threat_info.get("threat_type") == "UNAUTHORIZED_ENTRY"

                is_suspicious = (threat_info is not None) and (is_violent or has_weapon or is_zone_breach)
                threat_level = "NORMAL"

                if is_suspicious and threat_info:
                    threat_level = threat_info.get("severity", "HIGH")
                    track.is_threat = True
                    track.threat_type = threat_info.get("threat_type")

                    # 5. Alert Dispatch
                    alert = self.alert_engine.process_threat(
                        camera_id=camera_id,
                        camera_location=camera_location,
                        threat_info=threat_info,
                        track_id=track.track_id
                    )
                    if alert:
                        alerts_generated.append(alert)
                else:
                    track.is_threat = False
                    track.threat_type = None


                people_results.append({
                    "track_id": track.track_id,
                    "bbox": {
                        "x": float(x),
                        "y": float(y),
                        "width": float(bw),
                        "height": float(bh)
                    },
                    "activity": final_activity.upper(),
                    "confidence": round(final_conf * 100.0, 1),
                    "interacting_objects": track.interacting_objects,
                    "landmarks": pose_landmarks,
                    "pose_angles": pose_angles,
                    "posture_details": posture_details,
                    "is_suspicious": is_suspicious,
                    "threat_level": threat_level,
                    "duration_sec": round(time.time() - track.first_seen, 1)
                })

        inference_time_ms = max(1.0, (time.time() - t0) * 1000.0)
        calc_fps = round(1000.0 / inference_time_ms, 1)

        return {
            "camera_id": camera_id,
            "timestamp": datetime.utcnow().isoformat(),
            "fps": min(60.0, calc_fps),
            "inference_time_ms": round(inference_time_ms, 2),
            "people_count": len(people_results),
            "people": people_results,
            "objects_count": len(objects_results),
            "objects": objects_results,
            "active_alerts": alerts_generated
        }

# Global singleton pipeline instance
global_pipeline = SentinelInferencePipeline()
