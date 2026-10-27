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


class WeaponDetector:
    """
    High-Performance Neural Weapon and Firearm Detector for SentinelVision AI.
    Combines fine-tuned YOLO weapon models (models/best_weapon_detector.pt and models/best_shooting_weapon_detector.pt)
    supporting all major firearm and tactical hazard classes:
    - Handgun (Pistol / Revolver / Semi-Automatic)
    - Automatic Rifle (Assault Rifle)
    - Shotgun
    - SMG (Submachine Gun)
    - Sniper Rifle
    - Knife & Bladed Hazards
    - Sword
    - Heavy Ordnance (Bazooka, Grenade Launcher)

    Features:
    1. Dual-model ensemble for ultra-high recall across close-up handgun images, phone screens, and CCTV distance.
    2. Class-Agnostic Non-Maximum Suppression (NMS) to eliminate duplicate or multi-class confusion (one clean bounding box).
    3. Temporal Exponential Moving Average (EMA) Coordinate Smoothing across frames for zero-lag, jitter-free visual tracking.
    4. Standalone & Handheld Recognition: Accurately detects guns whether shown directly, on a smartphone screen, or wielded by a person.
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
        iou_thresh: float = 0.35,
        alpha: float = 0.45
    ) -> List[Dict[str, Any]]:
        """
        1. Class-Agnostic Non-Maximum Suppression (NMS) to eliminate duplicate/conflicting weapon boxes.
        2. Temporal EMA Coordinate Smoothing for fluid, jitter-free visual tracking.
        """
        if not candidate_boxes:
            self.no_weapon_counter += 1
            if self.no_weapon_counter >= 3:
                self.smoothed_weapon_boxes = []
            return []

        self.no_weapon_counter = 0

        # 1. Non-Maximum Suppression (NMS) across all candidate boxes
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

        # 2. Temporal EMA Smoothing
        final_boxes: List[Dict[str, Any]] = []
        for box in kept_boxes:
            matched_prev = None
            for prev in self.smoothed_weapon_boxes:
                cx1, cy1 = (box["x1"] + box["x2"]) / 2.0, (box["y1"] + box["y2"]) / 2.0
                cx2, cy2 = (prev["x1"] + prev["x2"]) / 2.0, (prev["y1"] + prev["y2"]) / 2.0
                dist = ((cx1 - cx2) ** 2 + (cy1 - cy2) ** 2) ** 0.5
                if dist < 85:
                    matched_prev = prev
                    break

            if matched_prev:
                sx1 = int(matched_prev["x1"] * (1 - alpha) + box["x1"] * alpha)
                sy1 = int(matched_prev["y1"] * (1 - alpha) + box["y1"] * alpha)
                sx2 = int(matched_prev["x2"] * (1 - alpha) + box["x2"] * alpha)
                sy2 = int(matched_prev["y2"] * (1 - alpha) + box["y2"] * alpha)
                smoothed_box = {
                    "x1": sx1,
                    "y1": sy1,
                    "x2": sx2,
                    "y2": sy2,
                    "width": max(1, sx2 - sx1),
                    "height": max(1, sy2 - sy1),
                    "label": box["label"],
                    "confidence": box["confidence"],
                    "category": "weapon",
                    "is_weapon": True
                }
                final_boxes.append(smoothed_box)
            else:
                final_boxes.append(box)

        self.smoothed_weapon_boxes = final_boxes
        return final_boxes

    def detect_weapons(
        self,
        frame_bgr: np.ndarray,
        conf_thresh: float = 0.18,
        imgsz: int = 512
    ) -> Tuple[bool, float, str, List[Dict[str, Any]]]:
        """
        Executes real-time firearm and weapon detection with single-box consolidation and EMA smoothing.
        Returns:
            (is_weapon, top_confidence, top_weapon_type, formatted_objects)
        """
        if (self.primary_model is None and self.secondary_model is None) or frame_bgr is None or frame_bgr.size == 0:
            return False, 0.0, "", []

        self.frame_counter += 1
        h, w = frame_bgr.shape[:2]
        candidate_boxes = []

        models_to_run = []
        if self.primary_model is not None:
            models_to_run.append((self.primary_model, self.primary_names))
        if self.secondary_model is not None:
            models_to_run.append((self.secondary_model, self.secondary_names))

        try:
            for model, names_map in models_to_run:
                results = model(frame_bgr, imgsz=imgsz, conf=conf_thresh, verbose=False)
                for r in results:
                    for b in r.boxes:
                        cls_id = int(b.cls[0].cpu())
                        raw_name = names_map.get(cls_id, str(cls_id))
                        conf = float(b.conf[0].cpu())

                        # Standardize display labels
                        low_name = raw_name.lower()
                        if "handgun" in low_name or "pistol" in low_name or "revolver" in low_name:
                            display_label = "Handgun"
                        elif "rifle" in low_name:
                            display_label = "Assault Rifle"
                        elif "shotgun" in low_name:
                            display_label = "Shotgun"
                        elif "smg" in low_name:
                            display_label = "SMG"
                        elif "knife" in low_name:
                            display_label = "Knife"
                        elif "sword" in low_name:
                            display_label = "Sword"
                        else:
                            display_label = raw_name

                        x1, y1, x2, y2 = map(int, b.xyxy[0].cpu().numpy().tolist())
                        bx1 = max(0, x1)
                        by1 = max(0, y1)
                        bx2 = min(w - 1, x2)
                        by2 = min(h - 1, y2)
                        bw = max(1, bx2 - bx1)
                        bh = max(1, by2 - by1)

                        if bw < 12 or bh < 12:
                            continue

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
            print(f"[WeaponDetector] Inference error: {e}")
            return False, 0.0, "", []

        # Consolidate duplicate boxes across both models and smooth coordinates
        consolidated_boxes = self._apply_nms_and_smoothing(candidate_boxes, iou_thresh=0.35, alpha=0.45)

        if not consolidated_boxes:
            return False, 0.0, "", []

        # Sort by confidence descending
        consolidated_boxes.sort(key=lambda b: b["confidence"], reverse=True)
        top_box = consolidated_boxes[0]
        top_conf = round(float(top_box["confidence"]) * 100.0, 1)
        top_type = top_box["label"]

        # Format as standard scene objects: [x, y, w, h] bbox
        formatted_objects = []
        for b in consolidated_boxes:
            formatted_objects.append({
                "label": b["label"],
                "category": "weapon",
                "confidence": b["confidence"],
                "bbox": [b["x1"], b["y1"], b["width"], b["height"]],
                "is_weapon": True
            })

        return True, top_conf, top_type, formatted_objects


# Global Singleton Instance for fast reuse across API requests
global_weapon_detector = WeaponDetector()
