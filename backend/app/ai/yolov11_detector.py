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

# Whitelisted Object Categories requested for webcam surveillance
# Only: TV, Phone, Laptop, Fire, Gun, Knife
OBJECT_CATEGORIES = {
    # Electronics / Screens
    "laptop": "electronics",
    "cell phone": "electronics",
    "tv": "electronics",

    # Threats / Weapons
    "knife": "threat",
    "gun": "threat",

    # Fire Hazard
    "fire": "fire_hazard"
}

ALLOWED_OBJECTS = {"laptop", "cell phone", "tv", "knife", "gun", "fire"}

def fast_flame_check(frame: np.ndarray) -> bool:
    """
    Sub-1ms chromatic flame pre-filter.
    Analyzes color temperature: Fire exhibits dominant Red, moderate Green, low Blue.
    """
    if frame is None or frame.size == 0:
        return False
    try:
        small = cv2.resize(frame, (160, 90), interpolation=cv2.INTER_NEAREST)
        b = small[:, :, 0].astype(int)
        g = small[:, :, 1].astype(int)
        r = small[:, :, 2].astype(int)
        flame_mask = (r > 160) & (r > g) & (g >= b) & ((r - b) > 45)
        return int(np.count_nonzero(flame_mask)) > 15
    except Exception:
        return False


class YOLOv11Detector:
    """
    State-of-the-Art YOLOv11 Unified Neural Detector for SentinelVision AI.
    Optimized for high-speed, zero-confusion real-time detection of:
    - People (Class 0)
    - Basic Objects: Cell Phone, Laptop, TV
    - Threats: Guns (Handguns, Rifles, Shotguns) and Knives
    - Hazards: Fire / Flames
    """
    def __init__(self, model_file: str = "yolo11n.pt", conf_threshold: float = 0.35):
        self.conf_threshold = conf_threshold
        self.model = None
        self.model_name = "YOLOv11-Nano (yolo11n.pt)"
        self.model_file = model_file
        self.classes_map = {}
        self.weapon_model = None
        self.fire_model = None
        self.frame_counter = 0
        self._init_models()

    def _init_models(self):
        try:
            from ultralytics import YOLO

            root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
            
            # 1. Official YOLOv11 Model
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

            # 2. Specialized Weapon Detection Model (Guns, Rifles, Knives, Swords)
            weapon_candidates = [
                os.path.join(root_dir, "models", "best_shooting_weapon_detector.pt"),
                os.path.join(root_dir, "models", "best_weapon_detector.pt"),
                "models/best_shooting_weapon_detector.pt"
            ]
            for wcand in weapon_candidates:
                if os.path.exists(wcand):
                    try:
                        self.weapon_model = YOLO(wcand)
                        print(f"[YOLOv11 Detector] [OK] Specialized Weapon Model loaded from {wcand}.")
                        break
                    except Exception as we:
                        print(f"[YOLOv11 Detector] Weapon model note: {we}")

            # 3. Dedicated Fire Detection Model
            fire_candidates = [
                os.path.join(root_dir, "models", "yolov8_fire_box.pt"),
                "models/yolov8_fire_box.pt"
            ]
            for fcand in fire_candidates:
                if os.path.exists(fcand):
                    try:
                        self.fire_model = YOLO(fcand)
                        print(f"[YOLOv11 Detector] [OK] Fire & Flame Model loaded from {fcand}.")
                        break
                    except Exception as fe:
                        print(f"[YOLOv11 Detector] Fire model note: {fe}")

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
        Executes unified detection with zero-confusion object filtering.
        Returns:
            - people: List of detected persons with [x, y, w, h], confidence, and interacting_objects
            - objects: Whitelisted detected objects (laptop, cell phone, tv, knife, gun, fire)
            - fire_info: Fire detection status and bounding boxes
        """
        if frame is None or frame.size == 0:
            return [], [], {"is_fire": False, "fire_confidence": 0.0, "boxes": []}

        self.frame_counter += 1
        h, w = frame.shape[:2]
        people = []
        objects = []
        fire_boxes = []
        effective_conf = conf_threshold if conf_threshold is not None else self.conf_threshold

        if self.model is not None:
            try:
                # 1. Run YOLOv11 for people, phones, laptops, TVs, knives
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
                        elif raw_label in ["cell phone", "laptop", "tv", "tvmonitor", "monitor", "knife"]:
                            # Normalize label
                            norm_label = "tv" if "tv" in raw_label or "monitor" in raw_label else raw_label
                            norm_cat = OBJECT_CATEGORIES.get(norm_label, "electronics")
                            if bw > 8 and bh > 8:
                                objects.append({
                                    "label": norm_label,
                                    "category": norm_cat,
                                    "confidence": round(float(conf), 3),
                                    "bbox": [bx, by, bw, bh]
                                })

                # 2. Run Weapon Model (High-precision firearms & knives)
                if self.weapon_model is not None:
                    try:
                        w_results = self.weapon_model(frame, imgsz=imgsz, conf=max(0.25, effective_conf * 0.85), verbose=False)
                        for wr in w_results:
                            if len(wr.boxes) == 0:
                                continue
                            w_cls_ids = wr.boxes.cls.int().cpu().numpy()
                            w_confs = wr.boxes.conf.cpu().numpy()
                            w_xyxys = wr.boxes.xyxy.int().cpu().numpy()

                            for w_cls, w_conf, (wx1, wy1, wx2, wy2) in zip(w_cls_ids, w_confs, w_xyxys):
                                wbx = max(0, int(wx1))
                                wby = max(0, int(wy1))
                                wbw = min(w - wbx, int(wx2 - wx1))
                                wbh = min(h - wby, int(wy2 - wy1))
                                raw_w = self.weapon_model.names.get(int(w_cls), "weapon").lower()

                                # Classify into gun vs knife
                                if any(k in raw_w for k in ["knife", "sword", "dagger", "blade"]):
                                    norm_label = "knife"
                                    norm_cat = "threat"
                                else:
                                    norm_label = "gun"
                                    norm_cat = "threat"

                                if wbw > 10 and wbh > 10:
                                    # Avoid duplicate knife/gun overlapping existing object box
                                    is_dup = False
                                    for ex in objects:
                                        if ex["label"] == norm_label:
                                            ex_box = ex["bbox"]
                                            # Simple overlap check
                                            if abs(ex_box[0] - wbx) < 30 and abs(ex_box[1] - wby) < 30:
                                                is_dup = True
                                                break
                                    if not is_dup:
                                        objects.append({
                                            "label": norm_label,
                                            "category": norm_cat,
                                            "confidence": round(float(w_conf), 3),
                                            "bbox": [wbx, wby, wbw, wbh]
                                        })
                    except Exception:
                        pass

                # 3. Run Fire & Flame Model (Ultra-fast conditional inference)
                has_flame_colors = fast_flame_check(frame)
                should_run_fire = (self.fire_model is not None) and (has_flame_colors or (self.frame_counter % 20 == 0))
                if should_run_fire:
                    try:
                        f_results = self.fire_model(frame, imgsz=320, conf=0.30, verbose=False)
                        for fr in f_results:
                            if len(fr.boxes) == 0:
                                continue
                            f_cls_ids = fr.boxes.cls.int().cpu().numpy()
                            f_confs = fr.boxes.conf.cpu().numpy()
                            f_xyxys = fr.boxes.xyxy.int().cpu().numpy()

                            for f_cls, f_conf, (fx1, fy1, fx2, fy2) in zip(f_cls_ids, f_confs, f_xyxys):
                                fbx = max(0, int(fx1))
                                fby = max(0, int(fy1))
                                fbw = min(w - fbx, int(fx2 - fx1))
                                fbh = min(h - fby, int(fy2 - fy1))
                                f_label = self.fire_model.names.get(int(f_cls), "Fire")

                                if fbw > 8 and fbh > 8:
                                    objects.append({
                                        "label": "fire",
                                        "category": "fire_hazard",
                                        "confidence": round(float(f_conf), 3),
                                        "bbox": [fbx, fby, fbw, fbh]
                                    })
                                    fire_boxes.append({
                                        "x1": fbx,
                                        "y1": fby,
                                        "x2": fbx + fbw,
                                        "y2": fby + fbh,
                                        "width": fbw,
                                        "height": fbh,
                                        "label": f_label,
                                        "confidence": round(float(f_conf) * 100.0, 1),
                                        "is_fire": True
                                    })
                    except Exception:
                        pass

                # Perform Spatial Person-Object Association
                self._associate_person_objects(people, objects)

                # Assemble Fire Info
                is_fire = len(fire_boxes) > 0
                max_fire_conf = max([b["confidence"] for b in fire_boxes], default=0.0)
                fire_info = {
                    "is_fire": is_fire,
                    "fire_confidence": round(max_fire_conf, 1),
                    "boxes": fire_boxes
                }

                return people, objects, fire_info

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

            # Expanded person bounding box to capture objects held in hands or near body
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
        spine_incline: Optional[float] = None
    ) -> Tuple[str, float]:
        """
        Infers strictly from the basic human activity set:
        - STANDING
        - WALKING
        - RUNNING
        - SLEEPING
        - USING_LAPTOP (watching or using laptop)
        - USING_PHONE (watching or using phone)
        - WATCHING_TV (watching TV screen)
        - SITTING
        - (ARMED_HAZARD if carrying gun or knife)
        """
        x, y, w, h = bbox
        aspect_ratio = float(h) / max(1.0, float(w))
        labels = {obj["label"].lower() for obj in interacting_objects}

        # 1. Threat Overrides (Gun or Knife)
        if any(l in labels for l in ["gun", "knife"]):
            return "armed_hazard", 0.98

        # 2. Sleeping Detection (Horizontal posture or recumbent)
        is_recumbent = (w > 1.25 * h) or (aspect_ratio < 0.85)
        is_inclined_sleep = (spine_incline is not None and spine_incline > 52.0)
        if is_recumbent or is_inclined_sleep:
            return "sleeping", 0.96

        # 3. Device Usage / Watching Overrides
        if "laptop" in labels:
            return "using_laptop", 0.96

        if "cell phone" in labels:
            return "using_phone", 0.95

        if "tv" in labels:
            return "watching_tv", 0.94

        # 4. Locomotion Velocity Analysis (Running vs Walking vs Standing)
        if velocity >= 60.0:
            return "running", 0.95
        elif velocity >= 15.0:
            return "walking", 0.94

        # 5. Upright Stance vs Seated
        # Aspect ratio >= 1.65 indicates tall standing posture
        if aspect_ratio >= 1.65:
            return "standing", 0.94
        else:
            return "sitting", 0.93

    def infer_activity_from_objects(self, interacting_objects: List[Dict[str, Any]]) -> Optional[Tuple[str, float]]:
        """
        Object-specific basic activity inference.
        """
        if not interacting_objects:
            return None

        labels = {obj["label"].lower() for obj in interacting_objects}

        if any(l in labels for l in ["gun", "knife"]):
            return "armed_hazard", 0.98

        if "laptop" in labels:
            return "using_laptop", 0.96

        if "cell phone" in labels:
            return "using_phone", 0.95

        if "tv" in labels:
            return "watching_tv", 0.94

        return None
