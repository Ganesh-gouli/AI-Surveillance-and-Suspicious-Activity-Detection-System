import os
import time
from typing import Dict, Any, Optional, Tuple, List
import cv2
import numpy as np

try:
    from ultralytics import YOLO
    HAS_ULTRALYTICS = True
except ImportError:
    HAS_ULTRALYTICS = False

# Strict firearm target classes (excludes noisy knives, swords, bazookas, grenade launchers)
TARGET_FIREARMS = {
    "handgun", "pistol", "revolver", "smg", "automatic rifle", "rifle", "shotgun"
}


class WeaponDetector:
    """
    High-Precision Neural Handgun & Firearm Detector for SentinelVision AI.
    Optimized for live webcam feeds, forensic image analysis, and tactical alert systems.

    Features:
    1. Targeted Firearm Classification: Strictly focuses on Handguns, Pistols, Revolvers, and Firearms.
       Eliminates false alarms from ambient knives, swords, or full-frame bazooka hallucinations.
    2. Single Clean Bounding Box: Enforces single-winner Non-Maximum Suppression (NMS) so the user
       never sees duplicate, jittering, or multiple phantom gun boxes.
    3. Temporal EMA Coordinate Smoothing: Smoothly interpolates coordinates across frames (alpha=0.40)
       for rock-solid, fluid tracking without latency or jumping.
    4. Adaptive Geometry Verification: Rejects whole-screen false positives while perfectly recognizing
       both direct handgun images and handheld guns / phones held in front of the camera.
    """

    def __init__(self):
        self.base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
        self.primary_model = None
        self.secondary_model = None
        self.primary_names = {}
        self.secondary_names = {}
        self.smoothed_weapon_boxes: List[Dict[str, Any]] = []
        self.no_weapon_counter: int = 0
        self.frame_counter: int = 0
        self._load_models()

    def _load_models(self):
        if not HAS_ULTRALYTICS:
            print("[WeaponDetector] Warning: ultralytics is not installed.")
            return

        p1 = os.path.join(self.base_dir, "models", "best_weapon_detector.pt")
        p2 = os.path.join(self.base_dir, "models", "best_shooting_weapon_detector.pt")

        if os.path.exists(p1):
            try:
                print(f"[WeaponDetector] Loading Primary Weapon Model from: {p1}")
                self.primary_model = YOLO(p1)
                self.primary_names = self.primary_model.names if hasattr(self.primary_model, 'names') else {}
                print(f"[WeaponDetector] [OK] Primary model loaded with {len(self.primary_names)} classes.")
            except Exception as e:
                print(f"[WeaponDetector] Error loading primary model: {e}")

        if os.path.exists(p2):
            try:
                print(f"[WeaponDetector] Loading Secondary Weapon Model from: {p2}")
                self.secondary_model = YOLO(p2)
                self.secondary_names = self.secondary_model.names if hasattr(self.secondary_model, 'names') else {}
                print(f"[WeaponDetector] [OK] Secondary model loaded with {len(self.secondary_names)} classes.")
            except Exception as e:
                print(f"[WeaponDetector] Error loading secondary model: {e}")

    def _apply_nms_and_smoothing(
        self,
        candidate_boxes: List[Dict[str, Any]],
        iou_thresh: float = 0.25,
        alpha: float = 0.40
    ) -> List[Dict[str, Any]]:
        """
        1. Class-Agnostic Non-Maximum Suppression (NMS) to retain only the single best gun box.
        2. Temporal EMA Coordinate Smoothing for zero-lag, jitter-free visual tracking.
        """
        if not candidate_boxes:
            self.no_weapon_counter += 1
            if self.no_weapon_counter >= 2:
                self.smoothed_weapon_boxes = []
            return []

        self.no_weapon_counter = 0

        # Sort by confidence descending
        sorted_boxes = sorted(candidate_boxes, key=lambda b: b.get("confidence", 0.0), reverse=True)
        kept_boxes: List[Dict[str, Any]] = []

        while sorted_boxes:
            best = sorted_boxes.pop(0)
            kept_boxes.append(best)
            remaining: List[Dict[str, Any]] = []

            for b in sorted_boxes:
                xx1 = max(best["x1"], b["x1"])
                yy1 = max(best["y1"], b["y1"])
                xx2 = min(best["x2"], b["x2"])
                yy2 = min(best["y2"], b["y2"])
                w = max(0, xx2 - xx1)
                h = max(0, yy2 - yy1)
                inter = w * h
                area1 = best["width"] * best["height"]
                area2 = b["width"] * b["height"]
                union = area1 + area2 - inter
                iou = (inter / union) if union > 0 else 0.0

                if iou < iou_thresh:
                    remaining.append(b)

            sorted_boxes = remaining

        # Strict Single-Gun Consolidation: Only keep the highest-confidence winning box
        winner_box = kept_boxes[0]

        # Temporal EMA Smoothing across frames
        if self.smoothed_weapon_boxes:
            prev = self.smoothed_weapon_boxes[0]
            cx1 = (winner_box["x1"] + winner_box["x2"]) / 2.0
            cy1 = (winner_box["y1"] + winner_box["y2"]) / 2.0
            cx2 = (prev["x1"] + prev["x2"]) / 2.0
            cy2 = (prev["y1"] + prev["y2"]) / 2.0
            dist = ((cx1 - cx2) ** 2 + (cy1 - cy2) ** 2) ** 0.5

            if dist < 120:
                sx1 = int(prev["x1"] * (1 - alpha) + winner_box["x1"] * alpha)
                sy1 = int(prev["y1"] * (1 - alpha) + winner_box["y1"] * alpha)
                sx2 = int(prev["x2"] * (1 - alpha) + winner_box["x2"] * alpha)
                sy2 = int(prev["y2"] * (1 - alpha) + winner_box["y2"] * alpha)
                smoothed = {
                    "x1": sx1,
                    "y1": sy1,
                    "x2": sx2,
                    "y2": sy2,
                    "width": max(1, sx2 - sx1),
                    "height": max(1, sy2 - sy1),
                    "label": winner_box["label"],
                    "confidence": winner_box["confidence"],
                    "category": "weapon",
                    "is_weapon": True
                }
                self.smoothed_weapon_boxes = [smoothed]
                return [smoothed]

        self.smoothed_weapon_boxes = [winner_box]
        return [winner_box]

    def detect_weapons(
        self,
        frame_bgr: np.ndarray,
        conf_thresh: float = 0.28,
        imgsz: int = 448
    ) -> Tuple[bool, float, str, List[Dict[str, Any]]]:
        """
        Executes real-time accurate handgun and firearm detection with zero multi-box confusion.
        Returns:
            (is_weapon, top_confidence, top_weapon_type, formatted_objects)
        """
        if (self.primary_model is None and self.secondary_model is None) or frame_bgr is None or frame_bgr.size == 0:
            return False, 0.0, "", []

        self.frame_counter += 1
        h, w = frame_bgr.shape[:2]
        total_area = float(w * h)
        candidate_boxes = []

        # 1. Primary Model (best_weapon_detector.pt: high accuracy on closeups, phone screens, standalone guns)
        if self.primary_model is not None:
            try:
                results1 = self.primary_model(frame_bgr, imgsz=imgsz, conf=conf_thresh, verbose=False)
                for r in results1:
                    for b in r.boxes:
                        cls_id = int(b.cls[0].cpu())
                        raw_name = self.primary_names.get(cls_id, str(cls_id)).lower()
                        conf = float(b.conf[0].cpu())

                        if raw_name not in TARGET_FIREARMS:
                            continue

                        x1, y1, x2, y2 = map(int, b.xyxy[0].cpu().numpy().tolist())
                        bx1 = max(0, x1)
                        by1 = max(0, y1)
                        bx2 = min(w - 1, x2)
                        by2 = min(h - 1, y2)
                        bw = max(1, bx2 - bx1)
                        bh = max(1, by2 - by1)
                        area_ratio = (bw * bh) / total_area

                        # Reject ambient full-frame hallucinations (Automatic Rifle, Bazooka filling whole screen)
                        if area_ratio > 0.82 and raw_name in ("automatic rifle", "bazooka", "grenade launcher", "rifle"):
                            continue
                        if bw < 16 or bh < 16:
                            continue

                        display_label = "Handgun" if raw_name in ("handgun", "pistol", "revolver", "smg") else "Firearm"

                        candidate_boxes.append({
                            "x1": bx1,
                            "y1": by1,
                            "x2": bx2,
                            "y2": by2,
                            "width": bw,
                            "height": bh,
                            "label": display_label,
                            "confidence": round(conf, 3),
                            "category": "weapon",
                            "is_weapon": True
                        })
            except Exception as e:
                print(f"[WeaponDetector] Primary model error: {e}")

        # 2. Secondary Model (best_shooting_weapon_detector.pt: surveillance & distance firearm detection)
        if self.secondary_model is not None:
            try:
                results2 = self.secondary_model(frame_bgr, imgsz=384, conf=conf_thresh, verbose=False)
                for r in results2:
                    for b in r.boxes:
                        cls_id = int(b.cls[0].cpu())
                        raw_name = self.secondary_names.get(cls_id, str(cls_id)).lower()
                        conf = float(b.conf[0].cpu())

                        if raw_name not in TARGET_FIREARMS:
                            continue

                        x1, y1, x2, y2 = map(int, b.xyxy[0].cpu().numpy().tolist())
                        bx1 = max(0, x1)
                        by1 = max(0, y1)
                        bx2 = min(w - 1, x2)
                        by2 = min(h - 1, y2)
                        bw = max(1, bx2 - bx1)
                        bh = max(1, by2 - by1)
                        area_ratio = (bw * bh) / total_area

                        if area_ratio > 0.80 or bw < 16 or bh < 16:
                            continue

                        display_label = "Handgun" if raw_name in ("handgun", "pistol", "revolver", "smg") else "Firearm"

                        candidate_boxes.append({
                            "x1": bx1,
                            "y1": by1,
                            "x2": bx2,
                            "y2": by2,
                            "width": bw,
                            "height": bh,
                            "label": display_label,
                            "confidence": round(conf, 3),
                            "category": "weapon",
                            "is_weapon": True
                        })
            except Exception as e:
                print(f"[WeaponDetector] Secondary model error: {e}")

        # Consolidate candidate boxes to exactly 1 smoothed winning box
        consolidated = self._apply_nms_and_smoothing(candidate_boxes, iou_thresh=0.25, alpha=0.40)

        if not consolidated:
            return False, 0.0, "", []

        winner = consolidated[0]
        top_conf = round(float(winner["confidence"]) * 100.0, 1)
        top_type = winner["label"]

        formatted_objects = [{
            "label": winner["label"],
            "category": "weapon",
            "confidence": winner["confidence"],
            "bbox": [winner["x1"], winner["y1"], winner["width"], winner["height"]],
            "is_weapon": True
        }]

        return True, top_conf, top_type, formatted_objects


# Global Singleton Instance for fast reuse across API requests
global_weapon_detector = WeaponDetector()
