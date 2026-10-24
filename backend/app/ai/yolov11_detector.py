import os
import cv2
import numpy as np
import torch
from typing import List, Dict, Any, Tuple, Optional

# Ensure PyTorch weights_only compatibility for Ultralytics YOLO
try:
    _orig_torch_load = torch.load
    torch.load = lambda *args, **kwargs: _orig_torch_load(*args, **{**kwargs, 'weights_only': False})
except Exception:
    pass

# Whitelisted Normal Household & Table Objects
OBJECT_CATEGORIES = {
    # Electronics & Devices
    "laptop": "electronics",
    "cell phone": "electronics",
    "tv": "electronics",
    "remote": "electronics",
    "mouse": "electronics",
    "keyboard": "electronics",

    # Furniture (Home / Desk)
    "table": "furniture",
    "dining table": "furniture",
    "chair": "furniture",
    "couch": "furniture",

    # Tableware & Daily Living Items
    "cup": "tableware",
    "bottle": "tableware",
    "book": "living",
    "clock": "living",
}

NORMAL_HOME_CLASSES = {
    "cell phone", "laptop", "tv", "tvmonitor", "monitor",
    "dining table", "table", "chair", "couch",
    "cup", "bottle", "book", "remote", "mouse", "keyboard", "clock"
}


class YOLOv11Detector:
    """
    State-of-the-Art YOLOv11 Neural Detector for Home & Everyday Activity Surveillance.
    Strictly detects:
    - People (Class 0)
    - Normal Activities: Sitting, Walking, Running, Sleeping, Using/Watching Phone, Laptop, TV, Standing
    - Normal Household & Tabletop Items: Laptop, Phone, TV, Table, Chair, Cup, Bottle, Book, Mouse, Keyboard, Remote
    Weapons, guns, and knives are completely excluded.
    """
    def __init__(self, model_file: str = "yolo11n.pt", conf_threshold: float = 0.35):
        self.conf_threshold = conf_threshold
        self.model = None
        self.model_name = "YOLOv11-Nano (yolo11n.pt)"
        self.model_file = model_file
        self.classes_map = {}
        self.frame_counter = 0
        self._init_models()

    def _init_models(self):
        try:
            from ultralytics import YOLO

            root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
            
            yolo_candidates = [
                self.model_file,
                os.path.join(root_dir, self.model_file),
                os.path.join(root_dir, "models", self.model_file),
                "yolo11n.pt"
            ]
            resolved_yolo = "yolo11n.pt"
            for cand in yolo_candidates:
                if os.path.exists(cand):
                    resolved_yolo = os.path.abspath(cand)
                    break

            print(f"[YOLOv11 Detector] Initializing Ultralytics YOLOv11 from {resolved_yolo}...")
            self.model = YOLO(resolved_yolo)
            self.classes_map = self.model.names if hasattr(self.model, 'names') else {}
            print(f"[YOLOv11 Detector] [OK] Successfully initialized YOLOv11! Loaded {len(self.classes_map)} classes.")

        except Exception as e:
            print(f"[YOLOv11 Detector] Error initializing models: {e}. Attempting OpenCV fallback.")
            self.model = None
            try:
                self.hog = cv2.HOGDescriptor()
                self.hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
            except Exception:
                self.hog = None

    def detect(self, frame: np.ndarray, conf_threshold: Optional[float] = None, imgsz: int = 384) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
        """
        Executes unified detection of people and normal home/table items.
        Guns, knives, and weapons are strictly excluded.
        Returns:
            - people: List of detected persons with [x, y, w, h], confidence, and interacting_objects
            - objects: Whitelisted normal home/table objects
            - fire_info: Empty/clean status
        """
        if frame is None or frame.size == 0:
            return [], [], {"is_fire": False, "fire_confidence": 0.0, "boxes": []}

        self.frame_counter += 1
        h, w = frame.shape[:2]
        people = []
        objects = []
        effective_conf = conf_threshold if conf_threshold is not None else self.conf_threshold

        if self.model is not None:
            try:
                # Run YOLOv11 for people and normal home/table items
                results = self.model(frame, imgsz=imgsz, conf=effective_conf, verbose=False)
                for r in results:
                    boxes = r.boxes
                    if len(boxes) == 0:
                        continue
                    cls_ids = boxes.cls.int().cpu().numpy()
                    confs = boxes.conf.cpu().numpy()
                    xyxys = boxes.xyxy.int().cpu().numpy()

                    for cls_id, conf, (x1, y1, x2, y2) in zip(cls_ids, confs, xyxys):
                        bx = max(0, int(x1))
                        by = max(0, int(y1))
                        bw = min(w - bx, int(x2 - x1))
                        bh = min(h - by, int(y2 - y1))

                        raw_label = self.classes_map.get(int(cls_id), str(cls_id)).lower()

                        if raw_label == "person":
                            if bw > 15 and bh > 30:
                                people.append({
                                    "bbox": [bx, by, bw, bh],
                                    "confidence": round(float(conf), 3),
                                    "interacting_objects": []
                                })
                        elif raw_label in NORMAL_HOME_CLASSES:
                            # Normalize label names for display
                            if "tv" in raw_label or "monitor" in raw_label:
                                norm_label = "tv"
                            elif "table" in raw_label:
                                norm_label = "table"
                            else:
                                norm_label = raw_label

                            norm_cat = OBJECT_CATEGORIES.get(norm_label, "living")
                            if bw > 8 and bh > 8:
                                objects.append({
                                    "label": norm_label,
                                    "category": norm_cat,
                                    "confidence": round(float(conf), 3),
                                    "bbox": [bx, by, bw, bh]
                                })

                # Perform Spatial Person-Object Association (e.g. person holding phone or working on laptop)
                self._associate_person_objects(people, objects)

                return people, objects, {"is_fire": False, "fire_confidence": 0.0, "boxes": []}

            except Exception as e:
                print(f"[YOLOv11 Detector] Inference error: {e}")

        # Fallback to OpenCV HOG if model failed
        if hasattr(self, 'hog') and self.hog is not None:
            try:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame
                boxes, weights = self.hog.detectMultiScale(gray, winStride=(8, 8), padding=(8, 8), scale=1.05)
                for (bx, by, bw, bh), weight in zip(boxes, weights):
                    conf = float(weight[0]) if hasattr(weight, '__iter__') else float(weight)
                    if conf > 0.3:
                        people.append({
                            "bbox": [int(bx), int(by), int(bw), int(bh)],
                            "confidence": round(min(0.98, max(0.65, 0.70 + conf * 0.15)), 3),
                            "interacting_objects": []
                        })
            except Exception:
                pass

        return people, objects, {"is_fire": False, "fire_confidence": 0.0, "boxes": []}

    def _associate_person_objects(self, people: List[Dict[str, Any]], objects: List[Dict[str, Any]]):
        """
        Calculates spatial overlap and proximity between detected people and objects.
        Links interacting objects directly to the person dictionary.
        """
        for person in people:
            px, py, pw, ph = person["bbox"]

            # Expanded person bounding box to capture objects on lap, desk, or in hand
            margin_x = int(pw * 0.25)
            margin_y = int(ph * 0.20)
            exp_px1 = max(0, px - margin_x)
            exp_py1 = max(0, py - margin_y)
            exp_px2 = px + pw + margin_x
            exp_py2 = py + ph + margin_y

            for obj in objects:
                ox, oy, ow, oh = obj["bbox"]

                # Check bounding box intersection
                ix1 = max(exp_px1, ox)
                iy1 = max(exp_py1, oy)
                ix2 = min(exp_px2, ox + ow)
                iy2 = min(exp_py2, oy + oh)

                inter_w = max(0, ix2 - ix1)
                inter_h = max(0, iy2 - iy1)
                inter_area = inter_w * inter_h
                obj_area = max(1, ow * oh)

                overlap_ratio = inter_area / obj_area

                # Object within person proximity / interaction boundary
                if overlap_ratio > 0.15 or (ox >= exp_px1 and ox + ow <= exp_px2 and oy >= exp_py1 and oy + oh <= exp_py2):
                    person["interacting_objects"].append({
                        "label": obj["label"],
                        "category": obj["category"],
                        "confidence": round(obj["confidence"] * 100.0, 1),
                        "bbox": {
                            "x": float(ox),
                            "y": float(oy),
                            "width": float(ow),
                            "height": float(oh)
                        }
                    })

    def infer_posture_and_activity(
        self,
        bbox: List[int],
        interacting_objects: List[Dict[str, Any]],
        frame_dim: Tuple[int, int] = (1280, 720),
        velocity: float = 0.0,
        spine_incline: Optional[float] = None,
        is_standing_pose: bool = False
    ) -> Tuple[str, float]:
        """
        Infers strictly normal everyday activities using the user's explicit 4 rules:

        RULE 1:  Full body detected (stationary / low velocity)  → STANDING
        RULE 2:  Body moving slowly  (velocity >= 15 px/frame)   → WALKING
        RULE 3:  Body moving at speed (velocity >= 60 px/frame)  → RUNNING
        RULE 4:  Half visible body  (upper torso / webcam view)  → SITTING

        Device-interaction overrides (laptop, phone, TV) still apply when
        objects are spatially associated with the person.
        """
        x, y, w, h = bbox
        frame_w, frame_h = frame_dim
        aspect_ratio = float(h) / max(1.0, float(w))
        labels = {obj["label"].lower() for obj in interacting_objects}

        # ── 0. Sleeping Detection (horizontal body / recumbent) ──
        is_recumbent = (w > 1.25 * h) or (aspect_ratio < 0.85)
        is_inclined_sleep = (spine_incline is not None and spine_incline > 52.0)
        if is_recumbent or is_inclined_sleep:
            return "sleeping", 0.96

        # ── 1. Device-Interaction Overrides ──
        if any(l in labels for l in ["laptop", "keyboard", "mouse"]):
            return "using_laptop", 0.96

        if "cell phone" in labels:
            return "using_phone", 0.95

        if "tv" in labels:
            return "watching_tv", 0.94

        # ── 2. Determine body visibility: FULL vs HALF ──
        # "Full body" means the bounding box is tall enough to include legs:
        #   • MediaPipe 3D confirmed standing posture, OR
        #   • Vertical aspect ratio ≥ 1.50  (tall rectangle = full body), OR
        #   • Person's head is near the top AND their bbox covers >50% of frame height
        #   • Person's bbox extends into bottom 15% of frame (feet visible)
        head_near_top = (y < 0.22 * frame_h) and (h > 0.50 * frame_h)
        feet_near_bottom = (y + h) > (frame_h * 0.85)
        full_body_detected = (
            is_standing_pose
            or aspect_ratio >= 1.50
            or head_near_top
            or (feet_near_bottom and aspect_ratio >= 1.20)
        )

        # ── 3. Apply the strict 4-rule hierarchy ──
        if full_body_detected:
            # RULE 3: Fast movement → RUNNING
            if velocity >= 60.0:
                return "running", 0.96
            # RULE 2: Slow movement → WALKING
            if velocity >= 15.0:
                return "walking", 0.95
            # RULE 1: Stationary full body → STANDING
            return "standing", 0.96
        else:
            # RULE 4: Half visible body (upper torso / webcam crop) → SITTING
            # Even with some velocity (head sway etc.) half-body stays SITTING
            return "sitting", 0.94

    def infer_activity_from_objects(self, interacting_objects: List[Dict[str, Any]]) -> Optional[Tuple[str, float]]:
        """
        Normal object-specific activity inference.
        """
        if not interacting_objects:
            return None

        labels = {obj["label"].lower() for obj in interacting_objects}

        if any(l in labels for l in ["laptop", "keyboard", "mouse"]):
            return "using_laptop", 0.96

        if "cell phone" in labels:
            return "using_phone", 0.95

        if "tv" in labels:
            return "watching_tv", 0.94

        return None
