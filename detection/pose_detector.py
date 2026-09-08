import os
import cv2
import numpy as np
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass
import mediapipe as mp
from ultralytics import YOLO

# Standard MediaPipe 33-landmark skeleton connection pairs
POSE_CONNECTIONS = [
    # Torso
    (11, 12), (11, 23), (12, 24), (23, 24),
    # Left Arm
    (11, 13), (13, 15), (15, 17), (15, 19), (15, 21), (17, 19),
    # Right Arm
    (12, 14), (14, 16), (16, 18), (16, 20), (16, 22), (18, 20),
    # Left Leg
    (23, 25), (25, 27), (27, 29), (27, 31), (29, 31),
    # Right Leg
    (24, 26), (26, 28), (28, 30), (28, 32), (30, 32),
    # Head/Face
    (0, 1), (1, 2), (2, 3), (3, 7), (0, 4), (4, 5), (5, 6), (6, 8), (9, 10),
    (0, 11), (0, 12)
]

@dataclass
class PersonPose:
    person_id: int
    bbox: List[int]  # [x1, y1, x2, y2]
    landmarks: np.ndarray  # shape (33, 4): [x, y, z, visibility] normalized to [0, 1] frame
    confidence: float
    has_pose: bool
    action: str = "Normal"
    action_confidence: float = 0.0


class MultiPersonPoseDetector:
    """
    Robust Multi-Person Pose Detector.
    Combines YOLOv11 person tracking with dedicated per-person MediaPipe Pose Landmark
    trackers (model_complexity=1, smooth_landmarks=True) on padded person bounding boxes.
    Eliminates multi-person tracker cross-talk, jitter, and limb clipping.
    """
    def __init__(
        self,
        yolo_model_path: str = "yolo11n.pt",
        min_detection_confidence: float = 0.35,
        min_tracking_confidence: float = 0.35,
        max_persons: int = 3
    ):
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        resolved_yolo = os.path.join(base_dir, yolo_model_path) if not os.path.exists(yolo_model_path) else yolo_model_path
        
        self.yolo = YOLO(resolved_yolo)
        self.max_persons = max_persons
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence
        self.mp_pose = mp.solutions.pose
        
        # Dedicated MediaPipe tracker instance per tracked person ID
        # Prevents temporal state corruption when processing multiple people
        self.pose_trackers: Dict[int, Any] = {}
        self.tracker_last_seen: Dict[int, int] = {}
        self.person_history: Dict[int, PersonPose] = {}
        self.frame_count: int = 0

    def _get_tracker(self, pid: int) -> Any:
        """Returns or creates a dedicated MediaPipe Pose instance for a person ID."""
        if pid not in self.pose_trackers:
            self.pose_trackers[pid] = self.mp_pose.Pose(
                static_image_mode=False,
                model_complexity=1,
                smooth_landmarks=True,
                min_detection_confidence=self.min_detection_confidence,
                min_tracking_confidence=self.min_tracking_confidence
            )
        self.tracker_last_seen[pid] = self.frame_count
        return self.pose_trackers[pid]

    def _cleanup_stale_trackers(self, max_idle_frames: int = 30):
        """Closes and releases MediaPipe instances for persons that left the scene."""
        stale_pids = [
            pid for pid, last_f in self.tracker_last_seen.items()
            if (self.frame_count - last_f) > max_idle_frames
        ]
        for pid in stale_pids:
            if pid in self.pose_trackers:
                try:
                    self.pose_trackers[pid].close()
                except Exception:
                    pass
                del self.pose_trackers[pid]
            self.tracker_last_seen.pop(pid, None)
            self.person_history.pop(pid, None)

    def reset(self):
        """Releases all MediaPipe tracker instances and resets tracking history."""
        for pid, tracker in list(self.pose_trackers.items()):
            try:
                tracker.close()
            except Exception:
                pass
        self.pose_trackers.clear()
        self.tracker_last_seen.clear()
        self.person_history.clear()
        self.frame_count = 0

    def process_frame(self, frame_bgr: np.ndarray) -> List[PersonPose]:
        """
        Detects people, extracts MediaPipe pose landmarks using dedicated per-person trackers,
        applies EMA coordinate smoothing, and returns list of PersonPose objects.
        """
        if frame_bgr is None or frame_bgr.size == 0:
            return []

        self.frame_count += 1
        h, w = frame_bgr.shape[:2]
        
        # Periodic cleanup of out-of-frame person trackers
        if self.frame_count % 15 == 0:
            self._cleanup_stale_trackers(max_idle_frames=30)
        
        # 1. Run YOLOv11 person tracking with ByteTrack
        results = self.yolo.track(
            frame_bgr,
            persist=True,
            verbose=False,
            classes=[0],
            conf=self.min_detection_confidence,
            tracker="bytetrack.yaml",
            imgsz=640
        )[0]

        detected_persons: List[PersonPose] = []

        if results.boxes is not None and len(results.boxes) > 0:
            boxes = results.boxes.xyxy.cpu().numpy()
            confs = results.boxes.conf.cpu().numpy()
            
            if results.boxes.id is not None:
                track_ids = results.boxes.id.int().cpu().numpy().tolist()
            else:
                track_ids = list(range(1, len(boxes) + 1))

            # Prioritize prominent actors by bounding box area
            areas = [(boxes[i][2] - boxes[i][0]) * (boxes[i][3] - boxes[i][1]) for i in range(len(boxes))]
            sorted_indices = np.argsort(areas)[::-1][:self.max_persons]

            for idx in sorted_indices:
                box = boxes[idx]
                pid = track_ids[idx]
                conf = float(confs[idx])
                
                x1, y1, x2, y2 = map(int, box)
                bw = x2 - x1
                bh = y2 - y1

                # Discard tiny artifacts
                if bw < 15 or bh < 25:
                    continue

                # 20% crop padding to prevent limb clipping during punches and kicks
                px1 = max(0, int(x1 - 0.20 * bw))
                py1 = max(0, int(y1 - 0.20 * bh))
                px2 = min(w, int(x2 + 0.20 * bw))
                py2 = min(h, int(y2 + 0.20 * bh))

                crop = frame_bgr[py1:py2, px1:px2]
                if crop.size == 0:
                    continue

                crop_h, crop_w = crop.shape[:2]
                crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                
                # Dedicated MediaPipe tracker for this specific person
                tracker = self._get_tracker(pid)
                pose_res = tracker.process(crop_rgb)

                landmarks = np.zeros((33, 4), dtype=np.float32)
                has_pose = False

                if pose_res.pose_landmarks:
                    has_pose = True
                    for i, lm in enumerate(pose_res.pose_landmarks.landmark):
                        global_x = (lm.x * crop_w + px1) / w
                        global_y = (lm.y * crop_h + py1) / h
                        global_z = lm.z * (crop_w / w)
                        landmarks[i] = [global_x, global_y, global_z, lm.visibility]

                    # EMA Smoothing: blends current coordinates with previous frame
                    # 70% current / 30% previous eliminates tracking jitter while preserving fast strikes
                    if pid in self.person_history and self.person_history[pid].has_pose:
                        prev_lms = self.person_history[pid].landmarks
                        landmarks[:, :3] = 0.70 * landmarks[:, :3] + 0.30 * prev_lms[:, :3]
                        landmarks[:, 3] = np.maximum(landmarks[:, 3], prev_lms[:, 3] * 0.90)
                elif pid in self.person_history and self.person_history[pid].has_pose:
                    # Decay landmarks gracefully during brief visual occlusion
                    landmarks = self.person_history[pid].landmarks.copy()
                    landmarks[:, 3] *= 0.80
                    has_pose = True

                person_pose = PersonPose(
                    person_id=pid,
                    bbox=[x1, y1, x2, y2],
                    landmarks=landmarks,
                    confidence=conf,
                    has_pose=has_pose
                )
                detected_persons.append(person_pose)
                self.person_history[pid] = person_pose

        return detected_persons

    def draw_skeleton(
        self,
        frame_bgr: np.ndarray,
        persons: List[PersonPose],
        color_fighting: bool = False
    ) -> np.ndarray:
        """
        Draws MediaPipe body skeleton connections, joint points, and
        clean non-obtrusive action badges for each person on the frame.
        """
        canvas = frame_bgr.copy()
        h, w = canvas.shape[:2]

        # Colors: vivid Red/Orange for fighting, sleek Cyan for normal
        joint_color = (0, 0, 255) if color_fighting else (0, 240, 255)
        bone_color = (30, 60, 255) if color_fighting else (0, 180, 220)
        box_color = (0, 40, 220) if color_fighting else (0, 190, 90)

        for person in persons:
            x1, y1, x2, y2 = person.bbox
            pid = person.person_id
            action_label = person.action or "Normal"

            # Draw slim bounding box
            cv2.rectangle(canvas, (x1, y1), (x2, y2), box_color, 1, cv2.LINE_AA)

            # Compact person action badge above bounding box
            badge_text = f"ID {pid} | {action_label}"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = max(0.35, min(0.48, w / 1200))
            (tw, th), _ = cv2.getTextSize(badge_text, font, font_scale, 1)
            
            by1 = max(4, y1 - th - 8)
            by2 = by1 + th + 6
            bx2 = min(w - 4, x1 + tw + 10)
            
            # Draw semi-transparent pill behind label so video remains visible
            overlay = canvas.copy()
            cv2.rectangle(overlay, (x1, by1), (bx2, by2), (20, 20, 20), -1)
            cv2.addWeighted(overlay, 0.60, canvas, 0.40, 0, canvas)
            cv2.rectangle(canvas, (x1, by1), (bx2, by2), box_color, 1, cv2.LINE_AA)
            cv2.putText(canvas, badge_text, (x1 + 5, by2 - 4), font, font_scale, (255, 255, 255), 1, cv2.LINE_AA)

            if not person.has_pose:
                continue

            lms = person.landmarks

            # Draw Bones / Connections
            for i1, i2 in POSE_CONNECTIONS:
                v1 = lms[i1, 3]
                v2 = lms[i2, 3]
                if v1 > 0.25 and v2 > 0.25:
                    pt1 = (int(lms[i1, 0] * w), int(lms[i1, 1] * h))
                    pt2 = (int(lms[i2, 0] * w), int(lms[i2, 1] * h))
                    cv2.line(canvas, pt1, pt2, bone_color, 2, cv2.LINE_AA)

            # Draw Landmark Points
            for i in range(33):
                v = lms[i, 3]
                if v > 0.25:
                    cx = int(lms[i, 0] * w)
                    cy = int(lms[i, 1] * h)
                    radius = 3 if i in [0, 11, 12, 13, 14, 15, 16] else 2
                    cv2.circle(canvas, (cx, cy), radius, joint_color, -1, cv2.LINE_AA)
                    cv2.circle(canvas, (cx, cy), radius + 1, (255, 255, 255), 1, cv2.LINE_AA)

        return canvas
