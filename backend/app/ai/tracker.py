import time
from typing import List, Dict, Any, Tuple

class TrackedObject:
    def __init__(self, track_id: int, bbox: List[int], confidence: float, interacting_objects: List[Dict[str, Any]] = None):
        self.track_id = track_id
        self.bbox = bbox  # [x, y, w, h]
        self.confidence = confidence
        self.first_seen = time.time()
        self.last_seen = time.time()
        self.hits = 1
        self.time_since_update = 0
        self.history = [bbox]
        self.activity = "walking"
        self.activity_confidence = 0.90
        self.interacting_objects = interacting_objects or []
        self.is_threat = False
        self.threat_type = None

    def update(self, bbox: List[int], confidence: float, interacting_objects: List[Dict[str, Any]] = None):
        self.bbox = bbox
        self.confidence = confidence
        self.interacting_objects = interacting_objects or []
        self.last_seen = time.time()
        self.hits += 1
        self.time_since_update = 0
        self.history.append(bbox)
        if len(self.history) > 30:
            self.history.pop(0)

def calculate_iou(bbox1: List[int], bbox2: List[int]) -> float:
    """Calculates Intersection over Union (IoU) between two bounding boxes [x, y, w, h]."""
    x1, y1, w1, h1 = bbox1
    x2, y2, w2, h2 = bbox2

    xA = max(x1, x2)
    yA = max(y1, y2)
    xB = min(x1 + w1, x2 + w2)
    yB = min(y1 + h1, y2 + h2)

    inter_width = max(0, xB - xA)
    inter_height = max(0, yB - yA)
    inter_area = inter_width * inter_height

    box1_area = w1 * h1
    box2_area = w2 * h2
    union_area = box1_area + box2_area - inter_area

    if union_area == 0:
        return 0.0
    return inter_area / union_area

class MultiPersonTracker:
    """
    Multi-person tracker with IoU and centroid association.
    Maintains consistent track IDs (e.g. Person #01, Person #02).
    """
    def __init__(self, max_age: int = 15, iou_threshold: float = 0.3):
        self.max_age = max_age
        self.iou_threshold = iou_threshold
        self.next_track_id = 1
        self.tracks: Dict[int, TrackedObject] = {}

    def update(self, detections: List[Dict[str, Any]]) -> List[TrackedObject]:
        """
        Updates tracked objects with new frame detections.
        """
        # Increment time since update for all existing tracks
        for track in self.tracks.values():
            track.time_since_update += 1

        matched_detections = set()
        matched_tracks = set()

        # Match existing tracks with new detections via IoU
        for track_id, track in list(self.tracks.items()):
            best_iou = self.iou_threshold
            best_det_idx = -1

            for det_idx, det in enumerate(detections):
                if det_idx in matched_detections:
                    continue
                iou = calculate_iou(track.bbox, det["bbox"])
                if iou > best_iou:
                    best_iou = iou
                    best_det_idx = det_idx

            if best_det_idx >= 0:
                det = detections[best_det_idx]
                track.update(det["bbox"], det["confidence"], det.get("interacting_objects", []))
                matched_detections.add(best_det_idx)
                matched_tracks.add(track_id)

        # Create new tracks for unmatched detections
        for det_idx, det in enumerate(detections):
            if det_idx not in matched_detections:
                new_track = TrackedObject(
                    track_id=self.next_track_id,
                    bbox=det["bbox"],
                    confidence=det["confidence"],
                    interacting_objects=det.get("interacting_objects", [])
                )
                self.tracks[self.next_track_id] = new_track
                self.next_track_id += 1

        # Remove dead tracks that haven't been seen for max_age frames
        expired_ids = [t_id for t_id, t in self.tracks.items() if t.time_since_update > self.max_age]
        for t_id in expired_ids:
            del self.tracks[t_id]

        return list(self.tracks.values())
