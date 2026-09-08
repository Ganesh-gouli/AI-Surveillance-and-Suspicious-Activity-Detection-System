import os
import cv2
import numpy as np
import torch
from typing import List, Dict, Any, Tuple, Optional

# Ensure PyTorch 2.6 weights_only compatibility for Ultralytics YOLO
try:
    _orig_torch_load = torch.load
    torch.load = lambda *args, **kwargs: _orig_torch_load(*args, **{**kwargs, 'weights_only': False})
except Exception:
    pass

# Categorized COCO Objects Mapping
OBJECT_CATEGORIES = {
    # Electronics & Computing
    "laptop": "electronics",
    "cell phone": "electronics",
    "keyboard": "electronics",
    "mouse": "electronics",
    "tv": "electronics",
    "remote": "electronics",
    
    # Dining & Sustenance
    "bottle": "sustenance",
    "wine glass": "sustenance",
    "cup": "sustenance",
    "fork": "sustenance",
    "knife": "threat_or_dining",
    "spoon": "sustenance",
    "bowl": "sustenance",
    "banana": "sustenance",
    "apple": "sustenance",
    "sandwich": "sustenance",
    "orange": "sustenance",
    "broccoli": "sustenance",
    "carrot": "sustenance",
    "hot dog": "sustenance",
    "pizza": "sustenance",
    "donut": "sustenance",
    "cake": "sustenance",
    
    # Mobility & Vehicles
    "bicycle": "mobility",
    "motorcycle": "mobility",
    "skateboard": "mobility",
    "surfboard": "mobility",
    "car": "vehicle",
    "bus": "vehicle",
    
    # Luggage & Accessories
    "backpack": "luggage",
    "handbag": "luggage",
    "suitcase": "luggage",
    "umbrella": "accessory",
    "tie": "accessory",
    
    # Furniture & Seating Context
    "chair": "furniture",
    "couch": "furniture",
    "bed": "furniture",
    "dining table": "furniture",
    "bench": "furniture",
    
    # Sports & Recreation
    "sports ball": "sports",
    "baseball bat": "sports",
    "baseball glove": "sports",
    "tennis racket": "sports",
    "skis": "sports",
    "snowboard": "sports",
    "frisbee": "sports",
    
    # Threats / Security Hazards
    "scissors": "threat"
}

class YOLOv11Detector:
    """
    State-of-the-Art YOLOv11 Unified Neural Detector for SentinelVision AI.
    Performs real-time Person Detection (Class 0) and 80 COCO Object Detection simultaneously,
    followed by spatial Person-Object Interaction association, posture estimation (Sitting vs Standing),
    and contextual activity tagging.
    """
    def __init__(self, model_file: str = "yolo11n.pt", conf_threshold: float = 0.35):
        self.conf_threshold = conf_threshold
        self.model = None
        self.model_name = "YOLOv11-Nano (yolo11n.pt)"
        self.model_file = model_file
        self.classes_map = {}
        self._init_yolov11()

    def _init_yolov11(self):
        try:
            from ultralytics import YOLO
            
            # Resolve model path (root or models dir)
            root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
            candidates = [
                self.model_file,
                os.path.join(root_dir, self.model_file),
                os.path.join(root_dir, "models", self.model_file),
                "yolo11n.pt",
                "yolov8n.pt"  # Fallback
            ]
            
            resolved_path = "yolo11n.pt"
            for cand in candidates:
                if os.path.exists(cand):
                    resolved_path = os.path.abspath(cand)
                    break

            print(f"[YOLOv11 Detector] Initializing Ultralytics YOLO model from {resolved_path}...")
            self.model = YOLO(resolved_path)
            self.classes_map = self.model.names if hasattr(self.model, 'names') else {}
            print(f"[YOLOv11 Detector] [OK] Successfully initialized YOLOv11! Loaded {len(self.classes_map)} classes.")
        except Exception as e:
            print(f"[YOLOv11 Detector] Error initializing YOLOv11: {e}. Attempting OpenCV fallback.")
            self.model = None
            try:
                self.hog = cv2.HOGDescriptor()
                self.hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
            except Exception:
                self.hog = None

    def detect(self, frame: np.ndarray) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Executes unified YOLOv11 neural inference on a frame.
        Returns:
            - detected_people: List of persons with coordinates [x, y, w, h], confidence, and interacting_objects
            - detected_objects: List of all detected context objects with [x, y, w, h], label, category, confidence
        """
        if frame is None or frame.size == 0:
            return [], []

        h, w = frame.shape[:2]
        people = []
        objects = []

        if self.model is not None:
            try:
                # Run YOLOv11 with optimized 384x384 resolution for ultra-fast low-latency CPU detection
                results = self.model(frame, imgsz=384, conf=self.conf_threshold, verbose=False)
                
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

                        label = self.classes_map.get(int(cls_id), str(cls_id)).lower()

                        if label == "person":
                            if bw > 15 and bh > 30:  # Minimum human size filter
                                people.append({
                                    "bbox": [bx, by, bw, bh],
                                    "confidence": round(float(conf), 3),
                                    "interacting_objects": []
                                })
                        else:
                            if bw > 10 and bh > 10:
                                category = OBJECT_CATEGORIES.get(label, "general_object")
                                objects.append({
                                    "label": label,
                                    "category": category,
                                    "confidence": round(float(conf), 3),
                                    "bbox": [bx, by, bw, bh]
                                })


                # Perform Spatial Person-Object Association
                self._associate_person_objects(people, objects)

                return people, objects

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

        return people, objects

    def _associate_person_objects(self, people: List[Dict[str, Any]], objects: List[Dict[str, Any]]):
        """
        Calculates spatial overlap and proximity between detected people and objects.
        Links interacting objects directly to the person dictionary.
        """
        for person in people:
            px, py, pw, ph = person["bbox"]
            
            # Expanded person bounding box to capture objects held in hands or near body
            margin_x = int(pw * 0.20)
            margin_y = int(ph * 0.15)
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
                
                # If object is substantially within the person interaction zone
                if overlap_ratio > 0.20 or (ox >= exp_px1 and ox + ow <= exp_px2 and oy >= exp_py1 and oy + oh <= exp_py2):
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
        frame_dim: Tuple[int, int] = (1280, 720)
    ) -> Tuple[str, float]:
        """
        Infers whether a person is SITTING or STANDING or performing an object-grounded activity.
        Uses bounding box aspect ratio, vertical stance, and interacting objects.
        """
        x, y, w, h = bbox
        frame_w, frame_h = frame_dim
        aspect_ratio = float(h) / max(1.0, float(w))
        labels = {obj["label"].lower() for obj in interacting_objects}

        # 1. Direct Object Context Overrides
        if any(l in labels for l in ["laptop", "keyboard", "mouse"]):
            return "using_laptop", 0.95

        if "cell phone" in labels:
            return "using_phone", 0.93

        # Only classify drinking if an actual drink container is detected!
        if any(l in labels for l in ["bottle", "wine glass", "cup"]):
            return "drinking", 0.92

        if any(l in labels for l in ["sandwich", "pizza", "apple", "banana", "orange", "donut", "cake", "hot dog", "bowl", "fork", "spoon"]):
            return "eating", 0.92

        if any(l in labels for l in ["bicycle", "motorcycle"]):
            return "cycling", 0.96

        if any(l in labels for l in ["backpack", "handbag", "suitcase"]):
            return "carrying_bag", 0.90

        if any(l in labels for l in ["sports ball", "tennis racket", "baseball bat", "frisbee", "skateboard"]):
            return "playing_sports", 0.91

        if any(l in labels for l in ["knife", "scissors"]):
            return "armed_hazard", 0.96

        # 2. Furniture Interaction -> Sitting
        if any(l in labels for l in ["chair", "couch", "bench", "dining table", "bed"]):
            return "sitting", 0.94

        # 3. Posture Analysis: Sitting vs. Standing
        # Tall vertical aspect ratio (h/w >= 1.70) indicates upright standing posture
        # Lower aspect ratio or upper torso desktop webcam view indicates seated posture
        if aspect_ratio >= 1.70:
            return "standing", 0.92
        else:
            return "sitting", 0.91

    def infer_activity_from_objects(self, interacting_objects: List[Dict[str, Any]]) -> Optional[Tuple[str, float]]:
        """
        Object-specific activity inference.
        """
        if not interacting_objects:
            return None

        labels = {obj["label"].lower() for obj in interacting_objects}

        if any(l in labels for l in ["laptop", "keyboard", "mouse"]):
            return "using_laptop", 0.94

        if "cell phone" in labels:
            return "using_phone", 0.92

        # Drinking ONLY when a bottle/cup/wine glass is present
        if any(l in labels for l in ["bottle", "wine glass", "cup"]):
            return "drinking", 0.91

        if any(l in labels for l in ["sandwich", "pizza", "apple", "banana", "orange", "donut", "cake", "hot dog", "bowl", "fork", "spoon"]):
            return "eating", 0.92

        if any(l in labels for l in ["bicycle", "motorcycle"]):
            return "cycling", 0.96

        if any(l in labels for l in ["chair", "couch", "bench", "dining table"]):
            return "sitting", 0.93

        if any(l in labels for l in ["backpack", "handbag", "suitcase"]):
            return "carrying_bag", 0.89

        if any(l in labels for l in ["sports ball", "tennis racket", "baseball bat", "frisbee", "skateboard"]):
            return "playing_sports", 0.90

        if any(l in labels for l in ["knife", "scissors"]):
            return "armed_hazard", 0.95

        return None
