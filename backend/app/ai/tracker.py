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
        self.velocity = 0.0
        self.activity_history = []
        self.activity = "standing"
        self.activity_confidence = 0.95
        self.interacting_objects = interacting_objects or []
        self.is_threat = False
        self.threat_type = None
        self.stationary_frames = 0
        self.motion_frames = 0

    def update(self, bbox: List[int], confidence: float, interacting_objects: List[Dict[str, Any]] = None):
        # Exponential Moving Average (EMA) position smoothing to stabilize bounding boxes
        alpha = 0.75
        self.bbox = [
            int(alpha * n + (1.0 - alpha) * o)
            for n, o in zip(bbox, self.bbox)
        ]
        self.confidence = round(float(0.7 * confidence + 0.3 * self.confidence), 3)
        self.interacting_objects = interacting_objects or []
        self.last_seen = time.time()
        self.hits += 1
        self.time_since_update = 0
        self.history.append(self.bbox)
        if len(self.history) > 30:
            self.history.pop(0)

        # Centroid displacement velocity tracking (pixels per frame)
        if len(self.history) >= 2:
            prev_box = self.history[-2]
            curr_cx = self.bbox[0] + self.bbox[2] / 2.0
            curr_cy = self.bbox[1] + self.bbox[3] / 2.0
            prev_cx = prev_box[0] + prev_box[2] / 2.0
            prev_cy = prev_box[1] + prev_box[3] / 2.0
            disp = ((curr_cx - prev_cx) ** 2 + (curr_cy - prev_cy) ** 2) ** 0.5
            self.velocity = 0.65 * disp + 0.35 * self.velocity

    def update_activity(self, raw_activity: str, raw_conf: float) -> Tuple[str, float]:
        """
        Smooths activity over a rolling window using confidence-weighted voting and hysteresis debouncing.
        Eliminates jitter, hesitation, and rapid flipping between sitting, walking, standing, and running.
        """
        self.activity_history.append((raw_activity, raw_conf))
        if len(self.activity_history) > 10:
            self.activity_history.pop(0)

        # For newly initialized tracks, immediately synchronize to detected activity
        if (self.hits <= 2 and len(self.activity_history) <= 2) or not hasattr(self, 'activity') or self.activity is None:
            self.activity = raw_activity
            self.activity_confidence = raw_conf
            return self.activity, self.activity_confidence

        # Motion vs. Stationary tracking
        is_raw_motion = raw_activity in ("walking", "running")
        if is_raw_motion:
            self.motion_frames += 1
            self.stationary_frames = 0
        else:
            self.stationary_frames += 1
            self.motion_frames = 0

        # Rapid locomotion response:
        # If moving fast (running), switch within 2 frames
        if raw_activity == "running":
            recent_runs = [a for a, _ in self.activity_history[-2:]]
            if recent_runs.count("running") >= 2:
                self.activity = "running"
                self.activity_confidence = raw_conf
                return self.activity, self.activity_confidence

        # If walking, transition smoothly within 2 frames
        if raw_activity == "walking" and self.activity in ("sitting", "standing"):
            recent_walks = [a for a, _ in self.activity_history[-2:]]
            if recent_walks.count("walking") >= 2:
                self.activity = "walking"
                self.activity_confidence = raw_conf
                return self.activity, self.activity_confidence

        # If currently walking, hold walking state across natural stride pauses
        # (require at least 4 consecutive stationary frames to drop back to standing/sitting)
        if self.activity == "walking" and not is_raw_motion:
            if self.stationary_frames < 4:
                return self.activity, self.activity_confidence

        # If currently running, hold running state for at least 3 frames
        if self.activity == "running" and raw_activity != "running":
            recent_acts = [a for a, _ in self.activity_history[-3:]]
            if "running" in recent_acts:
                return self.activity, self.activity_confidence

        # Posture Debouncing: require 3 frames of consistent evidence between sitting and standing
        if self.activity in ("sitting", "standing") and raw_activity in ("sitting", "standing"):
            recent_postures = [a for a, _ in self.activity_history[-4:]]
            if recent_postures.count(raw_activity) >= 3:
                self.activity = raw_activity
                self.activity_confidence = raw_conf
            return self.activity, self.activity_confidence

        # General confidence-weighted voting fallback
        votes: Dict[str, float] = {}
        for act, conf in self.activity_history:
            votes[act] = votes.get(act, 0.0) + conf

        best_act = max(votes, key=votes.get)
        best_weight = votes[best_act]

        if best_act != self.activity:
            recent_acts = [a for a, _ in self.activity_history[-3:]]
            if recent_acts.count(best_act) >= 2:
                self.activity = best_act
                self.activity_confidence = round(best_weight / max(1.0, len(self.activity_history)), 3)
        else:
            self.activity_confidence = round(best_weight / max(1.0, len(self.activity_history)), 3)

        return self.activity, self.activity_confidence

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
