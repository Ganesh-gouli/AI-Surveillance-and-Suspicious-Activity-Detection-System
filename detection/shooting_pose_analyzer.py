import os
import math
import cv2
import numpy as np
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass
import mediapipe as mp

# MediaPipe Pose Landmark Indices
# 11: Left Shoulder, 12: Right Shoulder
# 13: Left Elbow,    14: Right Elbow
# 15: Left Wrist,    16: Right Wrist
# 17: Left Pinky,    18: Right Pinky
# 19: Left Index,    20: Right Index
# 23: Left Hip,      24: Right Hip
# 0:  Nose

SKELETON_CONNECTIONS = [
    # Torso
    (11, 12), (11, 23), (12, 24), (23, 24),
    # Left Arm
    (11, 13), (13, 15), (15, 17), (15, 19),
    # Right Arm
    (12, 14), (14, 16), (16, 18), (16, 20),
    # Head/Sightline
    (0, 11), (0, 12)
]


def calculate_angle_2d(p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> float:
    """Calculates angle (in degrees) at vertex p2 formed by lines p1-p2 and p3-p2."""
    v1 = p1[:2] - p2[:2]
    v2 = p3[:2] - p2[:2]
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)
    if norm1 < 1e-5 or norm2 < 1e-5:
        return 0.0
    cosine = np.dot(v1, v2) / (norm1 * norm2)
    cosine = np.clip(cosine, -1.0, 1.0)
    return float(np.degrees(np.arccos(cosine)))


@dataclass
class ShooterPoseProfile:
    person_id: int
    has_pose: bool
    landmarks: Optional[np.ndarray]  # (33, 4): [x, y, z, visibility] normalized to frame
    landmarks_px: Optional[np.ndarray]  # (33, 2): [px_x, px_y] in frame coordinates
    is_aiming: bool
    is_two_handed_grip: bool
    aiming_confidence: float
    aiming_arm: str  # 'right', 'left', 'both', 'none'
    elbow_angle: float
    wrist_distance_ratio: float
    wrist_recoil_velocity: float
    aiming_vector: Tuple[float, float]  # normalized direction vector (dx, dy)


class ShootingPoseAnalyzer:
    """
    High-Speed MediaPipe BlazePose Lite Analyzer for Shooting and Firearm Posture Recognition.
    Analyzes:
    1. Arm extension & combat stance (elbow angle > 120° with elevated wrist at chest/shoulder height).
    2. Two-handed firearm grip (Isosceles / Weaver stance where both wrists converge near the weapon).
    3. Sightline & aiming vector alignment (shoulder -> wrist orientation).
    4. Wrist recoil velocity during gunfire discharge.
    """
    def __init__(
        self,
        min_detection_confidence: float = 0.35,
        min_tracking_confidence: float = 0.35,
        model_complexity: int = 0  # 0: Lite (fastest CPU throughput ~5-8ms), 1: Full
    ):
        self.mp_pose = mp.solutions.pose
        self.pose_detector = self.mp_pose.Pose(
            static_image_mode=True,
            model_complexity=model_complexity,
            smooth_landmarks=True,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )
        self.prev_wrists: Dict[int, np.ndarray] = {}  # pid -> last [x, y] px
        self.recent_recoils: Dict[int, float] = {}

    def reset(self):
        self.prev_wrists.clear()
        self.recent_recoils.clear()

    def close(self):
        try:
            self.pose_detector.close()
        except Exception:
            pass

    def analyze_person_pose(
        self,
        frame_rgb: np.ndarray,
        person_id: int,
        person_bbox: List[int],
        associated_weapon_bbox: Optional[List[int]] = None
    ) -> ShooterPoseProfile:
        """
        Runs MediaPipe BlazePose on the cropped/padded person ROI, transforms landmarks
        back to frame coordinates, and extracts biomechanical shooting features.
        """
        h, w = frame_rgb.shape[:2]
        px, py, pw, ph = person_bbox

        # Add generous padding around person bbox to preserve fully extended arms, hands, and weapons
        pad_x = max(35, int(pw * 0.65))
        pad_y = max(30, int(ph * 0.30))
        rx1 = max(0, px - pad_x)
        ry1 = max(0, py - pad_y)
        rx2 = min(w, px + pw + pad_x)
        ry2 = min(h, py + ph + int(ph * 0.18))

        crop_w = rx2 - rx1
        crop_h = ry2 - ry1

        if pw < 36 or ph < 75 or crop_w < 20 or crop_h < 30:
            return ShooterPoseProfile(
                person_id=person_id,
                has_pose=False,
                landmarks=None,
                landmarks_px=None,
                is_aiming=False,
                is_two_handed_grip=False,
                aiming_confidence=0.0,
                aiming_arm="none",
                elbow_angle=0.0,
                wrist_distance_ratio=1.0,
                wrist_recoil_velocity=0.0,
                aiming_vector=(0.0, 0.0)
            )

        crop_rgb = frame_rgb[ry1:ry2, rx1:rx2]
        # Run BlazePose
        results = self.pose_detector.process(crop_rgb)

        if not results.pose_landmarks:
            return ShooterPoseProfile(
                person_id=person_id,
                has_pose=False,
                landmarks=None,
                landmarks_px=None,
                is_aiming=False,
                is_two_handed_grip=False,
                aiming_confidence=0.0,
                aiming_arm="none",
                elbow_angle=0.0,
                wrist_distance_ratio=1.0,
                wrist_recoil_velocity=0.0,
                aiming_vector=(0.0, 0.0)
            )

        # Extract normalized landmarks and transform to full-frame pixel coordinates
        landmarks = np.zeros((33, 4), dtype=np.float32)
        landmarks_px = np.zeros((33, 2), dtype=np.float32)

        for i, lm in enumerate(results.pose_landmarks.landmark):
            # Coordinates in crop
            cx = lm.x * crop_w + rx1
            cy = lm.y * crop_h + ry1
            landmarks[i] = [cx / w, cy / h, lm.z, lm.visibility]
            landmarks_px[i] = [cx, cy]

        # Landmark references
        l_sh = landmarks_px[11]
        r_sh = landmarks_px[12]
        l_el = landmarks_px[13]
        r_el = landmarks_px[14]
        l_wr = landmarks_px[15]
        r_wr = landmarks_px[16]
        l_hip = landmarks_px[23]
        r_hip = landmarks_px[24]
        nose = landmarks_px[0]

        # Biomechanical measurements
        shoulder_width = max(15.0, np.linalg.norm(r_sh - l_sh))
        torso_height = max(20.0, np.linalg.norm(((l_sh + r_sh) / 2.0) - ((l_hip + r_hip) / 2.0)))

        # 1. Elbow Angles (Extension)
        l_elbow_angle = calculate_angle_2d(l_sh, l_el, l_wr)
        r_elbow_angle = calculate_angle_2d(r_sh, r_el, r_wr)

        # 2. Wrist Elevation (between chest level and elevated aiming over barriers; neither hanging at waist nor high surrender)
        mid_sh = (l_sh + r_sh) / 2.0
        mid_sh_y = float(mid_sh[1])
        mid_hip_y = (l_hip[1] + r_hip[1]) / 2.0
        chest_level_y = mid_sh_y + torso_height * 0.50
        min_aim_y = mid_sh_y - torso_height * 0.38

        # Spatial sanity: person's shoulders must lie inside or immediately near person_bbox
        bx, by, bw, bh = person_bbox
        shoulders_in_bbox = (bx - 15 <= mid_sh[0] <= bx + bw + 15) and (by - 15 <= mid_sh[1] <= by + bh + 15)

        l_wrist_raised = shoulders_in_bbox and (min_aim_y <= l_wr[1] <= chest_level_y)
        r_wrist_raised = shoulders_in_bbox and (min_aim_y <= r_wr[1] <= chest_level_y)

        # Visibility checks for key upper-body landmarks
        l_sh_vis = landmarks[11, 3]
        r_sh_vis = landmarks[12, 3]
        l_wr_vis = landmarks[15, 3]
        r_wr_vis = landmarks[16, 3]
        shoulders_visible = (l_sh_vis >= 0.32 and r_sh_vis >= 0.32)

        # 3. Two-Handed Grip (Wrists closely converged holding firearm at chest/shoulder height with extended arms)
        wrist_dist = float(np.linalg.norm(l_wr - r_wr))
        wrist_ratio = wrist_dist / shoulder_width
        dominant_elbow_cand = max(r_elbow_angle, l_elbow_angle)
        is_two_handed = (
            shoulders_visible
            and (l_wr_vis >= 0.33 and r_wr_vis >= 0.33)
            and (wrist_dist <= 26.0 or (wrist_dist <= 34.0 and wrist_ratio <= 1.35))
            and (l_wrist_raised and r_wrist_raised)
            and (dominant_elbow_cand >= 82.0)
            and (25.0 <= torso_height <= crop_h * 1.6)
        )

        # Check surrender / hands-up victim posture (both hands raised high above head into the air with wide separation)
        is_hands_up_surrender = (l_wr[1] < mid_sh_y - torso_height * 0.35) and (r_wr[1] < mid_sh_y - torso_height * 0.35) and (wrist_dist > 35.0)

        # 4. Aiming Arm & Direction Vector (extended forward horizontally at chest height with visible active wrist)
        is_r_aiming = (
            shoulders_visible
            and (r_wr_vis >= 0.42)
            and (r_elbow_angle > 118.0)
            and r_wrist_raised
            and (float(np.linalg.norm(r_wr - r_sh)) > shoulder_width * 0.78)
            and not is_hands_up_surrender
            and (torso_height <= crop_h * 1.6)
        )
        is_l_aiming = (
            shoulders_visible
            and (l_wr_vis >= 0.42)
            and (l_elbow_angle > 118.0)
            and l_wrist_raised
            and (float(np.linalg.norm(l_wr - l_sh)) > shoulder_width * 0.78)
            and not is_hands_up_surrender
            and (torso_height <= crop_h * 1.6)
        )

        aiming_arm = "none"
        dominant_elbow = 0.0
        aiming_vec = (0.0, 0.0)

        if is_two_handed and not is_hands_up_surrender:
            aiming_arm = "both"
            dominant_elbow = dominant_elbow_cand
            mid_sh = (l_sh + r_sh) / 2.0
            mid_wr = (l_wr + r_wr) / 2.0
            diff = mid_wr - mid_sh
            n = np.linalg.norm(diff)
            aiming_vec = (float(diff[0] / n), float(diff[1] / n)) if n > 0 else (-1.0, 0.0)
        elif is_r_aiming:
            aiming_arm = "right"
            dominant_elbow = r_elbow_angle
            diff = r_wr - r_sh
            n = np.linalg.norm(diff)
            aiming_vec = (float(diff[0] / n), float(diff[1] / n)) if n > 0 else (1.0, 0.0)
        elif is_l_aiming:
            aiming_arm = "left"
            dominant_elbow = l_elbow_angle
            diff = l_wr - l_sh
            n = np.linalg.norm(diff)
            aiming_vec = (float(diff[0] / n), float(diff[1] / n)) if n > 0 else (-1.0, 0.0)

        # 5. Aiming Confidence Score
        aim_score = 0.0
        if not is_hands_up_surrender:
            if is_two_handed and dominant_elbow >= 72.0:
                aim_score += 0.65
                if dominant_elbow > 95.0:
                    aim_score += 0.20
                if dominant_elbow > 115.0:
                    aim_score += 0.14
            elif (is_r_aiming or is_l_aiming) and dominant_elbow >= 118.0:
                if abs(aiming_vec[0]) >= 0.32:
                    aim_score += 0.52
                    ext_bonus = min(0.35, max(0.0, (dominant_elbow - 118.0) / 120.0))
                    aim_score += ext_bonus

        # 6. Weapon Proximity Bonus
        if associated_weapon_bbox is not None:
            wx, wy, ww, wh = associated_weapon_bbox
            wcx = wx + ww / 2.0
            wcy = wy + wh / 2.0
            # Distance from weapon center to closest wrist
            min_w_dist = min(
                np.linalg.norm([wcx - r_wr[0], wcy - r_wr[1]]),
                np.linalg.norm([wcx - l_wr[0], wcy - l_wr[1]])
            )
            if min_w_dist < max(ww, wh) * 1.6:
                aim_score = min(0.99, aim_score + 0.25)

        # 7. Wrist Recoil Velocity Tracking
        active_wrist = r_wr if (aiming_arm in ("right", "both")) else l_wr
        recoil_vel = 0.0
        if person_id in self.prev_wrists:
            prev_w = self.prev_wrists[person_id]
            recoil_vel = float(np.linalg.norm(active_wrist - prev_w))
        self.prev_wrists[person_id] = active_wrist.copy()
        self.recent_recoils[person_id] = recoil_vel

        is_aiming = aim_score >= 0.50

        return ShooterPoseProfile(
            person_id=person_id,
            has_pose=True,
            landmarks=landmarks,
            landmarks_px=landmarks_px,
            is_aiming=is_aiming,
            is_two_handed_grip=is_two_handed,
            aiming_confidence=round(float(min(0.99, aim_score)), 3),
            aiming_arm=aiming_arm,
            elbow_angle=round(dominant_elbow, 1),
            wrist_distance_ratio=round(wrist_ratio, 3),
            wrist_recoil_velocity=round(recoil_vel, 2),
            aiming_vector=aiming_vec
        )

    def draw_shooter_skeleton(
        self,
        frame: np.ndarray,
        pose_profile: ShooterPoseProfile,
        is_shooting: bool,
        is_armed: bool
    ):
        """
        Renders anti-aliased MediaPipe pose skeleton with dynamic color coding:
        - Calm Cyan/Green for normal bystanders
        - Amber/Orange for armed individuals in aiming stance
        - Vivid Red for active shooters / firing
        """
        if not pose_profile.has_pose or pose_profile.landmarks_px is None:
            return

        pts = pose_profile.landmarks_px.astype(int)

        if is_shooting:
            bone_color = (0, 0, 255)       # Red
            joint_color = (50, 50, 255)
            bone_thick = 2
        elif is_armed or pose_profile.is_aiming:
            bone_color = (0, 140, 255)     # Amber
            joint_color = (0, 215, 255)
            bone_thick = 2
        else:
            bone_color = (255, 180, 0)     # Cyan
            joint_color = (255, 220, 100)
            bone_thick = 1

        # Draw bones
        for u, v in SKELETON_CONNECTIONS:
            pt1 = tuple(pts[u])
            pt2 = tuple(pts[v])
            cv2.line(frame, pt1, pt2, bone_color, bone_thick, cv2.LINE_AA)

        # Draw key upper-body joints
        upper_joints = [11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
        for j in upper_joints:
            pt = tuple(pts[j])
            cv2.circle(frame, pt, 3, joint_color, -1, cv2.LINE_AA)

        # If aiming or shooting, render aiming sightline vector from firing wrist
        if (pose_profile.is_aiming or is_shooting) and pose_profile.aiming_arm != "none":
            active_wrist_idx = 16 if pose_profile.aiming_arm in ("right", "both") else 15
            wx, wy = pts[active_wrist_idx]
            dx, dy = pose_profile.aiming_vector
            target_x = int(wx + dx * 45)
            target_y = int(wy + dy * 45)
            # Aiming laser line
            cv2.arrowedLine(frame, (wx, wy), (target_x, target_y), (0, 0, 255) if is_shooting else (0, 165, 255), 2, cv2.LINE_AA, tipLength=0.35)
            # Small crosshair on wrist
            cv2.circle(frame, (wx, wy), 5, (0, 0, 255), 1, cv2.LINE_AA)
