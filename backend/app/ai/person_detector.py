import os
import cv2
import numpy as np
import torch
from typing import List, Dict, Any
from .yolov11_detector import YOLOv11Detector

# Ensure PyTorch 2.6 weights_only compatibility for Ultralytics YOLO
try:
    _orig_torch_load = torch.load
    torch.load = lambda *args, **kwargs: _orig_torch_load(*args, **{**kwargs, 'weights_only': False})
except Exception:
    pass

class PersonDetector:
    """
    Real-Time Person Detector powered by YOLOv11.
    Accurately detects real humans and extracts bounding box coordinates (x, y, w, h).
    """
    def __init__(self, conf_threshold: float = 0.35):
        self.conf_threshold = conf_threshold
        self.detector = YOLOv11Detector(conf_threshold=conf_threshold)

    def detect(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """
        Runs real YOLOv11 neural detection on an image / video frame.
        Returns detected persons with coordinates [x, y, w, h], confidence, and interacting_objects.
        """
        people, _ = self.detector.detect(frame)
        return people
