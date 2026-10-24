import math
import cv2
import numpy as np
from typing import Dict, Any, List, Tuple, Optional

# Landmark index constants from MediaPipe Pose
NOSE = 0
LEFT_EYE_INNER = 1
LEFT_EYE = 2
LEFT_EYE_OUTER = 3
RIGHT_EYE_INNER = 4
RIGHT_EYE = 5
RIGHT_EYE_OUTER = 6
LEFT_EAR = 7
RIGHT_EAR = 8
MOUTH_LEFT = 9
MOUTH_RIGHT = 10
LEFT_SHOULDER = 11
RIGHT_SHOULDER = 12
LEFT_ELBOW = 13
RIGHT_ELBOW = 14
LEFT_WRIST = 15
RIGHT_WRIST = 16
LEFT_PINKY = 17
RIGHT_PINKY = 18
LEFT_INDEX = 19
RIGHT_INDEX = 20
LEFT_THUMB = 21
RIGHT_THUMB = 22
LEFT_HIP = 23
RIGHT_HIP = 24
LEFT_KNEE = 25
RIGHT_KNEE = 26
LEFT_ANKLE = 27
RIGHT_ANKLE = 28
LEFT_HEEL = 29
RIGHT_HEEL = 30
LEFT_FOOT_INDEX = 31
RIGHT_FOOT_INDEX = 32

LANDMARK_NAMES = {
    0: "nose", 11: "left_shoulder", 12: "right_shoulder",
    13: "left_elbow", 14: "right_elbow", 15: "left_wrist", 16: "right_wrist",
    23: "left_hip", 24: "right_hip", 25: "left_knee", 26: "right_knee",
    27: "left_ankle", 28: "right_ankle"
}

# Anatomical skeleton connections for rendering wireframes
SKELETON_CONNECTIONS = [
    # Head & Face
    (0, 1), (1, 2), (2, 3), (0, 4), (4, 5), (5, 6), (3, 7), (6, 8),
    # Torso & Shoulders
    (11, 12), (11, 23), (12, 24), (23, 24),
    # Left Arm
    (11, 13), (13, 15), (15, 17), (15, 19), (15, 21),
    # Right Arm
    (12, 14), (14, 16), (16, 18), (16, 20), (16, 22),
    # Left Leg
    (23, 25), (25, 27), (27, 29), (27, 31), (29, 31),
    # Right Leg
    (24, 26), (26, 28), (28, 30), (28, 32), (30, 32)
]

def calculate_angle_2d(a: Tuple[float, float], b: Tuple[float, float], c: Tuple[float, float]) -> float:
    """
    Calculates the 2D interior angle (in degrees) ABC with B as the vertex.
    a: (x, y)
    b: vertex (x, y)
    c: (x, y)
    """
    radians = math.atan2(c[1] - b[1], c[0] - b[0]) - math.atan2(a[1] - b[1], a[0] - b[0])
    angle = abs(radians * 180.0 / math.pi)
    if angle > 180.0:
        angle = 360.0 - angle
    return angle


class MediaPipePoseEngine:
    """
    Google MediaPipe Pose Biomechanical Analyzer for SentinelVision AI.
    Extracts 33 3D skeletal landmarks and computes precise biomechanical joint angles
    to accurately differentiate Standing vs. Sitting, Lying Down, and Dynamic Poses.
    """
    def __init__(self, static_image_mode: bool = False, model_complexity: int = 1):
        self.static_image_mode = static_image_mode
        self.model_complexity = model_complexity
        self.pose = None
        self.is_initialized = False
        self._init_mediapipe()

    def _init_mediapipe(self):
        try:
            import mediapipe as mp
            self.mp_pose = mp.solutions.pose
            self.pose = self.mp_pose.Pose(
                static_image_mode=self.static_image_mode,
                model_complexity=self.model_complexity,
                enable_segmentation=False,
                min_detection_confidence=0.45,
                min_tracking_confidence=0.45
            )
            self.is_initialized = True
            print(f"[MediaPipe Pose] [OK] MediaPipe Pose Biomechanical Engine initialized (High-Accuracy Model Complexity {self.model_complexity}).")
        except Exception as e:
            print(f"[MediaPipe Pose] Initialization error: {e}")
            self.pose = None
            self.is_initialized = False

    def analyze_person(
        self,
        frame: np.ndarray,
        bbox: List[int],
        interacting_objects: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        """
        Analyzes a person detected in `frame` within `bbox` [x, y, w, h].
        Returns:
            - activity: 'standing' | 'sitting' | 'lying_down' | 'using_laptop' | etc.
            - confidence: float (0.0 to 1.0)
            - is_standing: bool
            - is_sitting: bool
            - landmarks: List of 33 landmarks with frame-relative coordinates
            - angles: Dict of biomechanical joint angles (degrees)
            - posture_details: String explanation with measured angles
        """
        if not self.is_initialized or self.pose is None or frame is None or frame.size == 0:
            return self._fallback_result("sitting", 0.70, "MediaPipe not initialized")

        frame_h, frame_w = frame.shape[:2]
        x, y, bw, bh = bbox
        
        # Add 12% padding margin around person crop to ensure full limbs/feet are captured
        pad_x = int(bw * 0.12)
        pad_y = int(bh * 0.12)
        x1 = max(0, x - pad_x)
        y1 = max(0, y - pad_y)
        x2 = min(frame_w, x + bw + pad_x)
        y2 = min(frame_h, y + bh + pad_y)

        crop = frame[y1:y2, x1:x2]
        if crop.size == 0 or crop.shape[0] < 20 or crop.shape[1] < 20:
            return self._fallback_result("sitting", 0.70, "Crop too small")

        crop_h, crop_w = crop.shape[:2]

        try:
            # Optimize: scale crop to max 240px for ultra-fast (15-20ms) pose processing
            max_dim = max(crop_w, crop_h)
            if max_dim > 240:
                scale = 240.0 / max_dim
                proc_crop = cv2.resize(crop, (int(crop_w * scale), int(crop_h * scale)))
            else:
                proc_crop = crop

            # MediaPipe expects RGB format
            rgb_crop = cv2.cvtColor(proc_crop, cv2.COLOR_BGR2RGB)
            results = self.pose.process(rgb_crop)

            if not results or not results.pose_landmarks:
                # Fallback to aspect ratio heuristic if landmarks not detected
                aspect_ratio = float(bh) / max(1.0, float(bw))
                posture = "standing" if aspect_ratio >= 1.70 else "sitting"
                return self._fallback_result(
                    posture,
                    0.80,
                    f"Pose landmarks occluded; aspect ratio {aspect_ratio:.2f} inferred {posture}"
                )

            raw_landmarks = results.pose_landmarks.landmark
            landmarks_list = []

            # Map landmarks back to full frame pixel and normalized coordinates
            for idx, lm in enumerate(raw_landmarks):
                # Pixel coord in full frame
                px = x1 + lm.x * crop_w
                py = y1 + lm.y * crop_h
                # Normalized coord in full frame (0.0 - 1.0)
                norm_fx = max(0.0, min(1.0, px / float(frame_w)))
                norm_fy = max(0.0, min(1.0, py / float(frame_h)))

                landmarks_list.append({
                    "id": idx,
                    "name": LANDMARK_NAMES.get(idx, f"point_{idx}"),
                    "x": round(norm_fx, 4),
                    "y": round(norm_fy, 4),
                    "pixel_x": round(px, 1),
                    "pixel_y": round(py, 1),
                    "z": round(lm.z, 4),
                    "visibility": round(lm.visibility, 3)
                })

            # Biomechanical Angle Analysis
            analysis = self._compute_biomechanics(raw_landmarks, crop_w, crop_h)
            analysis["landmarks"] = landmarks_list
            analysis["connections"] = SKELETON_CONNECTIONS

            return analysis

        except Exception as e:
            print(f"[MediaPipe Pose] Processing error: {e}")
            aspect_ratio = float(bh) / max(1.0, float(bw))
            posture = "standing" if aspect_ratio >= 1.70 else "sitting"
            return self._fallback_result(posture, 0.75, f"Error fallback ({str(e)})")

    def _compute_biomechanics(self, landmarks, crop_w: int, crop_h: int) -> Dict[str, Any]:
        """
        Computes accurate angles and determines standing vs. sitting.
        """
        # Helper to get (x, y) and visibility
        def get_pt(idx):
            lm = landmarks[idx]
            return (lm.x * crop_w, lm.y * crop_h), lm.visibility

        l_sh, l_sh_v = get_pt(LEFT_SHOULDER)
        r_sh, r_sh_v = get_pt(RIGHT_SHOULDER)
        l_hip, l_hip_v = get_pt(LEFT_HIP)
        r_hip, r_hip_v = get_pt(RIGHT_HIP)
        l_knee, l_knee_v = get_pt(LEFT_KNEE)
        r_knee, r_knee_v = get_pt(RIGHT_KNEE)
        l_ankle, l_ankle_v = get_pt(LEFT_ANKLE)
        r_ankle, r_ankle_v = get_pt(RIGHT_ANKLE)
        l_elbow, l_elbow_v = get_pt(LEFT_ELBOW)
        r_elbow, r_elbow_v = get_pt(RIGHT_ELBOW)
        l_wrist, l_wrist_v = get_pt(LEFT_WRIST)
        r_wrist, r_wrist_v = get_pt(RIGHT_WRIST)
        nose, nose_v = get_pt(NOSE)

        # 1. Knee Angles: (Hip -> Knee -> Ankle)
        l_knee_angle = calculate_angle_2d(l_hip, l_knee, l_ankle) if min(l_hip_v, l_knee_v, l_ankle_v) > 0.35 else None
        r_knee_angle = calculate_angle_2d(r_hip, r_knee, r_ankle) if min(r_hip_v, r_knee_v, r_ankle_v) > 0.35 else None

        # 2. Hip Angles: (Shoulder -> Hip -> Knee)
        l_hip_angle = calculate_angle_2d(l_sh, l_hip, l_knee) if min(l_sh_v, l_hip_v, l_knee_v) > 0.35 else None
        r_hip_angle = calculate_angle_2d(r_sh, r_hip, r_knee) if min(r_sh_v, r_hip_v, r_knee_v) > 0.35 else None

        # 3. Torso / Spine Incline with respect to vertical
        # Midpoint of shoulders & hips
        mid_sh = ((l_sh[0] + r_sh[0]) / 2.0, (l_sh[1] + r_sh[1]) / 2.0)
        mid_hip = ((l_hip[0] + r_hip[0]) / 2.0, (l_hip[1] + r_hip[1]) / 2.0)
        dx_spine = abs(mid_sh[0] - mid_hip[0])
        dy_spine = abs(mid_sh[1] - mid_hip[1])
        # Angle of spine with vertical axis: 0° is completely upright, 90° is completely horizontal
        spine_incline = math.degrees(math.atan2(dx_spine, max(1.0, dy_spine)))

        # 4. Thigh Verticality vs Horizontality
        # For standing, knees are far below hips vertically (dy >> dx).
        # For sitting, thighs are extended horizontally forward (dy is small, knees close in height to hips).
        l_thigh_dy = l_knee[1] - l_hip[1]
        l_thigh_dx = abs(l_knee[0] - l_hip[0])
        r_thigh_dy = r_knee[1] - r_hip[1]
        r_thigh_dx = abs(r_knee[0] - r_hip[0])

        # Knee & Hip Visibility Check (Ankles often out of frame in webcam)
        valid_knees = [k for k in [l_knee_angle, r_knee_angle] if k is not None]
        valid_hips = [h for h in [l_hip_angle, r_hip_angle] if h is not None]
        avg_knee = sum(valid_knees) / len(valid_knees) if valid_knees else None
        avg_hip = sum(valid_hips) / len(valid_hips) if valid_hips else None

        # Thigh vertical drop: when standing, knees are far below hips
        l_thigh_drop = (l_knee[1] - l_hip[1]) if (l_knee_v > 0.30 and l_hip_v > 0.30) else 0.0
        r_thigh_drop = (r_knee[1] - r_hip[1]) if (r_knee_v > 0.30 and r_hip_v > 0.30) else 0.0
        max_thigh_drop = max(l_thigh_drop, r_thigh_drop)

        angles_data = {
            "left_knee": round(l_knee_angle, 1) if l_knee_angle is not None else None,
            "right_knee": round(r_knee_angle, 1) if r_knee_angle is not None else None,
            "left_hip": round(l_hip_angle, 1) if l_hip_angle is not None else None,
            "right_hip": round(r_hip_angle, 1) if r_hip_angle is not None else None,
            "spine_incline": round(spine_incline, 1)
        }

        # Lower body visibility (Knees and Ankles)
        has_ankles = (l_ankle_v > 0.35 or r_ankle_v > 0.35)
        has_knees = (l_knee_v > 0.35 or r_knee_v > 0.35)
        # Full body is confirmed when lower limbs (knees and ankles) are in frame
        is_full_body = has_ankles or (has_knees and max_thigh_drop > crop_h * 0.25)

        # 5. Determine Pose / Activity
        # Case A: Lying Down / Sleeping (horizontal spine > 52° from vertical)
        # Must have visible landmarks AND genuinely inclined spine — not just a wide bbox
        if spine_incline > 52.0 and min(l_sh_v, r_sh_v, l_hip_v, r_hip_v) > 0.35:
            conf = min(0.98, 0.85 + (spine_incline / 90.0) * 0.13)
            return {
                "activity": "sleeping",
                "confidence": round(conf, 3),
                "is_standing": False,
                "is_sitting": False,
                "is_lying": True,
                "is_full_body": is_full_body,
                "angles": angles_data,
                "posture_details": f"Lying Down / Sleeping (Spine Incline: {spine_incline:.1f}° from vertical)"
            }

        # Case B: Full Body Detected with Biomechanical Joints
        if is_full_body and avg_hip is not None and spine_incline < 35.0:
            if avg_hip >= 135.0 or (avg_knee is not None and avg_knee >= 142.0):
                conf = round(min(0.99, max(0.92, 0.90 + (avg_hip - 135.0) / 45.0 * 0.09)), 3)
                details = f"Standing (Full Body, Extended Hips: {avg_hip:.1f}°, Upright Spine: {spine_incline:.1f}°)"
                if avg_knee is not None:
                    details += f", Knees: {avg_knee:.1f}°"
                return {
                    "activity": "standing",
                    "confidence": conf,
                    "is_standing": True,
                    "is_sitting": False,
                    "is_lying": False,
                    "is_full_body": True,
                    "angles": angles_data,
                    "posture_details": details
                }
            elif 60.0 <= avg_hip <= 126.0 and (avg_knee is None or avg_knee <= 138.0):
                conf = round(min(0.98, max(0.91, 0.92 + (1.0 - abs(avg_hip - 95.0) / 45.0) * 0.06)), 3)
                details = f"Sitting (Full Body Seated, Hip Flexion: {avg_hip:.1f}°)"
                if avg_knee is not None:
                    details += f", Bent Knees: {avg_knee:.1f}°"
                return {
                    "activity": "sitting",
                    "confidence": conf,
                    "is_standing": False,
                    "is_sitting": True,
                    "is_lying": False,
                    "is_full_body": True,
                    "angles": angles_data,
                    "posture_details": details
                }

        # Case C: Half Visible Body (Upper Torso / Desktop Webcam / Lower Body Occluded)
        # By strict rule: if the body is half visible, it represents SITTING.
        crop_aspect = float(crop_h) / max(1.0, float(crop_w))
        return {
            "activity": "sitting",
            "confidence": 0.95,
            "is_standing": False,
            "is_sitting": True,
            "is_lying": False,
            "is_full_body": False,
            "angles": angles_data,
            "posture_details": f"Sitting (Half-Body Visible, Aspect: {crop_aspect:.2f})"
        }

    def _fallback_result(self, activity: str, confidence: float, reason: str) -> Dict[str, Any]:
        return {
            "activity": activity,
            "confidence": confidence,
            "is_standing": (activity == "standing"),
            "is_sitting": (activity == "sitting"),
            "is_lying": (activity == "sleeping" or activity == "lying_down"),
            "is_full_body": (activity == "standing"),
            "landmarks": [],
            "connections": SKELETON_CONNECTIONS,
            "angles": {
                "left_knee": None,
                "right_knee": None,
                "left_hip": None,
                "right_hip": None,
                "spine_incline": 0.0
            },
            "posture_details": f"{activity.title()} ({reason})"
        }


# Global MediaPipe Pose singleton
global_pose_engine = MediaPipePoseEngine()
