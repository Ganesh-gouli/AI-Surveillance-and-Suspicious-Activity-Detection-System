import math
import time
from typing import List, Dict, Any, Optional, Tuple
import numpy as np


def compute_iou(boxA: List[int], boxB: List[int]) -> float:
    """Computes Intersection over Union (IoU) between two [x, y, w, h] boxes."""
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[0] + boxA[2], boxB[0] + boxB[2])
    yB = min(boxA[1] + boxA[3], boxB[1] + boxB[3])

    interW = max(0, xB - xA)
    interH = max(0, yB - yA)
    interArea = interW * interH

    boxAArea = max(1, boxA[2] * boxA[3])
    boxBArea = max(1, boxB[2] * boxB[3])

    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou


class TrackedWeapon:
    """Represents a weapon instance tracked over consecutive video frames."""
    def __init__(self, track_id: int, cls_id: int, cls_name: str, bbox: List[int], confidence: float, timestamp: float):
        self.track_id = track_id
        self.cls_id = cls_id
        self.cls_name = cls_name
        self.bbox = bbox  # [x, y, w, h]
        self.confidence = confidence
        self.start_time = timestamp
        self.last_seen = timestamp
        self.hits = 1
        self.time_since_update = 0

        # Trajectory & Kinematics
        cx = bbox[0] + bbox[2] / 2.0
        cy = bbox[1] + bbox[3] / 2.0
        self.trajectory = [(cx, cy, timestamp)]
        self.velocity = (0.0, 0.0)
        self.acceleration = (0.0, 0.0)
        self.peak_acceleration = 0.0

        # Association
        self.held_by_person_id: Optional[int] = None
        self.muzzle_flashes_detected = 0

    def update(self, bbox: List[int], confidence: float, cls_name: str, timestamp: float):
        self.time_since_update = 0
        self.hits += 1
        self.bbox = bbox
        self.confidence = 0.7 * confidence + 0.3 * self.confidence
        self.cls_name = cls_name

        cx = bbox[0] + bbox[2] / 2.0
        cy = bbox[1] + bbox[3] / 2.0

        if len(self.trajectory) >= 1:
            prev_cx, prev_cy, prev_t = self.trajectory[-1]
            dt = max(1e-3, timestamp - prev_t)
            vx = (cx - prev_cx) / dt
            vy = (cy - prev_cy) / dt

            # Compute jerk / acceleration
            prev_vx, prev_vy = self.velocity
            ax = (vx - prev_vx) / dt
            ay = (vy - prev_vy) / dt
            mag_a = math.hypot(ax, ay)
            if mag_a > self.peak_acceleration:
                self.peak_acceleration = mag_a

            self.velocity = (vx, vy)
            self.acceleration = (ax, ay)

        self.trajectory.append((cx, cy, timestamp))
        if len(self.trajectory) > 60:
            self.trajectory.pop(0)

        self.last_seen = timestamp


class WeaponTracker:
    """
    Real-time Multi-Object Tracker for detected weapons and armed persons.
    Implements IoU + Centroid distance association, temporal persistence filtering,
    and velocity/acceleration extraction for shooting recoil analysis.
    """
    def __init__(self, iou_threshold: float = 0.25, max_age: int = 6, min_hits: int = 2):
        self.iou_threshold = iou_threshold
        self.max_age = max_age
        self.min_hits = min_hits
        self.next_weapon_id = 1
        self.tracks: Dict[int, TrackedWeapon] = {}

    def update(self, detections: List[Dict[str, Any]], timestamp: float) -> List[TrackedWeapon]:
        """
        Updates weapon tracks with new detections from current frame.
        detections format: [{"bbox": [x,y,w,h], "confidence": float, "class_name": str, "class_id": int}]
        """
        # Increment time_since_update for all active tracks
        for track in self.tracks.values():
            track.time_since_update += 1

        unmatched_dets = list(range(len(detections)))
        unmatched_tracks = list(self.tracks.keys())

        matched_pairs = []

        # 1. Match by IoU and distance
        if len(self.tracks) > 0 and len(detections) > 0:
            for t_id in list(unmatched_tracks):
                track = self.tracks[t_id]
                best_iou = 0.0
                best_det_idx = -1

                for d_idx in unmatched_dets:
                    det = detections[d_idx]
                    iou = compute_iou(track.bbox, det["bbox"])
                    if iou > best_iou:
                        best_iou = iou
                        best_det_idx = d_idx

                if best_iou >= self.iou_threshold and best_det_idx != -1:
                    matched_pairs.append((t_id, best_det_idx))
                    unmatched_tracks.remove(t_id)
                    unmatched_dets.remove(best_det_idx)

        # 2. Update matched tracks
        for t_id, d_idx in matched_pairs:
            det = detections[d_idx]
            self.tracks[t_id].update(
                bbox=det["bbox"],
                confidence=det["confidence"],
                cls_name=det["class_name"],
                timestamp=timestamp
            )

        # 3. Create new tracks for unmatched detections
        for d_idx in unmatched_dets:
            det = detections[d_idx]
            new_track = TrackedWeapon(
                track_id=self.next_weapon_id,
                cls_id=det.get("class_id", 0),
                cls_name=det["class_name"],
                bbox=det["bbox"],
                confidence=det["confidence"],
                timestamp=timestamp
            )
            self.tracks[self.next_weapon_id] = new_track
            self.next_weapon_id += 1

        # 4. Remove dead tracks
        dead_ids = [t_id for t_id, track in self.tracks.items() if track.time_since_update > self.max_age]
        for d_id in dead_ids:
            del self.tracks[d_id]

        # 5. Return confirmed tracks that meet min_hits or high confidence
        confirmed = [
            track for track in self.tracks.values()
            if track.time_since_update == 0 and (track.hits >= self.min_hits or track.confidence >= 0.72)
        ]
        return confirmed
