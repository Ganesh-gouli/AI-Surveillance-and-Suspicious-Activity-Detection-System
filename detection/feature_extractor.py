import numpy as np
from typing import List, Dict, Tuple, Optional
from .pose_detector import PersonPose

def calculate_joint_angle_3d(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """
    Calculates angle at joint b between vectors (a - b) and (c - b) in 3D.
    Returns normalized angle in [0, 1] (angle_radians / pi).
    """
    ba = a[:3] - b[:3]
    bc = c[:3] - b[:3]
    norm_ba = np.linalg.norm(ba)
    norm_bc = np.linalg.norm(bc)
    if norm_ba < 1e-5 or norm_bc < 1e-5:
        return 0.5
    cosine = np.dot(ba, bc) / (norm_ba * norm_bc)
    angle = np.arccos(np.clip(cosine, -1.0, 1.0))
    return float(angle / np.pi)


class KinematicPoseFeatureExtractor:
    """
    Extracts normalized body landmarks, velocities, joint angles,
    and multi-person interaction features from PersonPose objects.
    Maintains temporal state for smooth velocity and acceleration calculation.
    """
    def __init__(self, sequence_length: int = 30):
        self.sequence_length = sequence_length
        self.feature_dim = 510  # 250 per person * 2 + 10 interaction features
        
        # Historical state for velocity & acceleration
        self.prev_person_coords: Dict[int, np.ndarray] = {}  # pid -> (33, 3)
        self.prev_person_vel: Dict[int, np.ndarray] = {}     # pid -> (33, 3)
        self.prev_inter_hip_dist: Optional[float] = None
        
        # Temporal sequence buffer
        self.rolling_buffer: List[np.ndarray] = []

    def extract_single_person_features(
        self,
        person: Optional[PersonPose]
    ) -> np.ndarray:
        """
        Extracts 250 kinematic features for one person:
        - 132: Pelvis-normalized landmarks (33 * [x, y, z, vis])
        - 99: Joint velocities (33 * [dx, dy, dz])
        - 6: Joint angles (left/right elbow, knee, shoulder)
        - 12: Key joint accelerations (wrists, elbows: 4 * [ax, ay, az])
        - 1: Presence flag
        """
        if person is None or not person.has_pose:
            return np.zeros(250, dtype=np.float32)

        lms = person.landmarks.copy() # shape (33, 4)
        coords = lms[:, :3]           # shape (33, 3)
        vis = lms[:, 3:4]             # shape (33, 1)

        # 1. Pelvis / Hip Center
        # landmark 23 = left_hip, 24 = right_hip
        hip_center = (coords[23] + coords[24]) / 2.0
        
        # 2. Torso Scale Normalization Factor
        # landmark 11 = left_shoulder, 12 = right_shoulder
        shoulder_center = (coords[11] + coords[12]) / 2.0
        torso_len = np.linalg.norm(shoulder_center - hip_center)
        if torso_len < 0.05:
            # Fallback to shoulder width
            torso_len = np.linalg.norm(coords[11] - coords[12])
        if torso_len < 0.05:
            torso_len = 0.2  # conservative fallback

        # Normalized coordinates relative to pelvis
        norm_coords = (coords - hip_center) / torso_len
        norm_lms = np.hstack([norm_coords, vis]).flatten()  # 33 * 4 = 132

        # 3. Velocities (33 * 3 = 99)
        pid = person.person_id
        if pid in self.prev_person_coords:
            vel = norm_coords - self.prev_person_coords[pid]
            # Clip unphysical jumps due to re-identification or occlusion
            vel = np.clip(vel, -3.0, 3.0)
        else:
            vel = np.zeros_like(norm_coords)

        # 4. Accelerations (4 joints * 3 = 12)
        # 13: l_elbow, 14: r_elbow, 15: l_wrist, 16: r_wrist
        key_joint_indices = [13, 14, 15, 16]
        if pid in self.prev_person_vel:
            acc = vel[key_joint_indices] - self.prev_person_vel[pid][key_joint_indices]
            acc = np.clip(acc, -4.0, 4.0)
        else:
            acc = np.zeros((len(key_joint_indices), 3), dtype=np.float32)

        # Update tracking history
        self.prev_person_coords[pid] = norm_coords
        self.prev_person_vel[pid] = vel

        # 5. Joint Angles (6 angles)
        angles = np.array([
            calculate_joint_angle_3d(norm_coords[11], norm_coords[13], norm_coords[15]), # Left elbow
            calculate_joint_angle_3d(norm_coords[12], norm_coords[14], norm_coords[16]), # Right elbow
            calculate_joint_angle_3d(norm_coords[23], norm_coords[25], norm_coords[27]), # Left knee
            calculate_joint_angle_3d(norm_coords[24], norm_coords[26], norm_coords[28]), # Right knee
            calculate_joint_angle_3d(norm_coords[23], norm_coords[11], norm_coords[13]), # Left shoulder
            calculate_joint_angle_3d(norm_coords[24], norm_coords[12], norm_coords[14]), # Right shoulder
        ], dtype=np.float32)

        # Assemble: 132 + 99 + 6 + 12 + 1 = 250
        presence_flag = np.array([1.0], dtype=np.float32)
        return np.concatenate([
            norm_lms,
            vel.flatten(),
            angles,
            acc.flatten(),
            presence_flag
        ]).astype(np.float32)

    def extract_interaction_features(
        self,
        p1: Optional[PersonPose],
        p2: Optional[PersonPose]
    ) -> np.ndarray:
        """
        Extracts 10 relative interaction features between two people:
        - Inter-hip distance
        - Relative closing velocity
        - Hand-to-torso / hand-to-head proximities
        - Hand-to-hand proximity (grappling)
        - Fast movement co-occurrence
        - Horizontal displacement
        """
        features = np.zeros(10, dtype=np.float32)
        if p1 is None or p2 is None or not p1.has_pose or not p2.has_pose:
            return features

        c1 = p1.landmarks[:, :3]
        c2 = p2.landmarks[:, :3]

        hip1 = (c1[23] + c1[24]) / 2.0
        hip2 = (c2[23] + c2[24]) / 2.0
        torso1 = (c1[11] + c1[12] + c1[23] + c1[24]) / 4.0
        torso2 = (c2[11] + c2[12] + c2[23] + c2[24]) / 4.0
        head1 = c1[0]
        head2 = c2[0]

        # 1. Hip-to-hip distance
        d_hip = float(np.linalg.norm(hip1 - hip2))
        features[0] = d_hip

        # 2. Relative closing velocity
        if self.prev_inter_hip_dist is not None:
            features[1] = float(np.clip(d_hip - self.prev_inter_hip_dist, -2.0, 2.0))
        self.prev_inter_hip_dist = d_hip

        # 3. P1 wrists to P2 torso
        p1_wrists = [c1[15], c1[16]]
        p2_wrists = [c2[15], c2[16]]
        d_w1_t2 = min(np.linalg.norm(w - torso2) for w in p1_wrists)
        features[2] = float(d_w1_t2)

        # 4. P2 wrists to P1 torso
        d_w2_t1 = min(np.linalg.norm(w - torso1) for w in p2_wrists)
        features[3] = float(d_w2_t1)

        # 5. Wrists to Head (punch proximity)
        d_w1_h2 = min(np.linalg.norm(w - head2) for w in p1_wrists)
        d_w2_h1 = min(np.linalg.norm(w - head1) for w in p2_wrists)
        features[4] = float(d_w1_h2)
        features[5] = float(d_w2_h1)

        # 6. Wrist-to-wrist proximity (blocking/grappling)
        d_wrists = min(np.linalg.norm(w1 - w2) for w1 in p1_wrists for w2 in p2_wrists)
        features[6] = float(d_wrists)

        # 7. Relative horizontal displacement
        features[7] = float(hip1[0] - hip2[0])

        # 8. Bounding box IoU / overlap
        b1, b2 = p1.bbox, p2.bbox
        ix1, iy1 = max(b1[0], b2[0]), max(b1[1], b2[1])
        ix2, iy2 = min(b1[2], b2[2]), min(b1[3], b2[3])
        inter_area = max(0, ix2 - ix1) * max(0, iy2 - iy1)
        area1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
        area2 = (b2[2] - b2[0]) * (b2[3] - b2[1])
        iou = inter_area / float(area1 + area2 - inter_area + 1e-6)
        features[8] = float(iou)

        # 9. Multi-person active flag
        features[9] = 1.0

        return features

    def classify_person_action(
        self,
        person: PersonPose,
        opponent: Optional[PersonPose] = None
    ) -> str:
        """
        Classifies fine-grained physical action for a person using MediaPipe 33-pose kinematics:
        - Punching / Striking
        - Kicking
        - Pushing / Grappling
        - Aggressive Movement
        - Defensive Guard
        - Walking
        - Standing / Normal Movement
        """
        if not person.has_pose:
            return "Detected"

        pid = person.person_id
        coords = person.landmarks[:, :3]  # (33, 3)
        
        # Get velocities if available
        vel = self.prev_person_vel.get(pid, np.zeros((33, 3), dtype=np.float32))
        
        # 1. Wrist speeds
        v_wl = float(np.linalg.norm(vel[15]))
        v_wr = float(np.linalg.norm(vel[16]))
        v_wrist_max = max(v_wl, v_wr)

        # 2. Ankle speeds
        v_al = float(np.linalg.norm(vel[27]))
        v_ar = float(np.linalg.norm(vel[28]))
        v_ankle_max = max(v_al, v_ar)

        # 3. Elbow angles (normalized 0..1 in calculate_joint_angle_3d, multiply by 180 deg)
        ang_el = calculate_joint_angle_3d(coords[11], coords[13], coords[15]) * 180.0
        ang_er = calculate_joint_angle_3d(coords[12], coords[14], coords[16]) * 180.0
        arm_extended = (ang_el > 115.0 or ang_er > 115.0)

        # Torso scale
        sh_c = (coords[11] + coords[12]) / 2.0
        hip_c = (coords[23] + coords[24]) / 2.0
        torso_len = max(0.15, float(np.linalg.norm(sh_c - hip_c)))

        # 4. Knee angles & foot elevation (in normalized coords, smaller y is higher up)
        hip_y = (coords[23, 1] + coords[24, 1]) / 2.0
        min_ankle_y = min(coords[27, 1], coords[28, 1])
        ankle_diff_y = abs(coords[27, 1] - coords[28, 1])

        ang_kl = calculate_joint_angle_3d(coords[23], coords[25], coords[27]) * 180.0
        ang_kr = calculate_joint_angle_3d(coords[24], coords[26], coords[28]) * 180.0
        leg_extended = (ang_kl > 125.0 or ang_kr > 125.0)

        # Opponent metrics if present
        d_hips = 999.0
        d_wrist_to_opp_torso = 999.0
        d_wrist_to_opp_head = 999.0
        if opponent is not None and opponent.has_pose:
            opp_c = opponent.landmarks[:, :3]
            opp_hip = (opp_c[23] + opp_c[24]) / 2.0
            my_hip = (coords[23] + coords[24]) / 2.0
            d_hips = float(np.linalg.norm(my_hip - opp_hip))
            
            opp_torso = (opp_c[11] + opp_c[12] + opp_c[23] + opp_c[24]) / 4.0
            opp_head = opp_c[0]
            
            d_wrist_to_opp_torso = min(
                float(np.linalg.norm(coords[15] - opp_torso)),
                float(np.linalg.norm(coords[16] - opp_torso))
            )
            d_wrist_to_opp_head = min(
                float(np.linalg.norm(coords[15] - opp_head)),
                float(np.linalg.norm(coords[16] - opp_head))
            )

        head_pos = coords[0]
        d_wrist_to_head = min(
            float(np.linalg.norm(coords[15] - head_pos)),
            float(np.linalg.norm(coords[16] - head_pos))
        )

        # Action classification hierarchy:
        # A. Kicking: Explosive ankle velocity + one foot distinctly elevated off ground
        # In normal walking, both feet stay near ground (ankle_diff_y is small).
        # In a real kick, one foot is planted while the striking foot is raised toward waist/torso!
        is_kick = (
            v_ankle_max >= 0.24 and
            min_ankle_y < (hip_y + 0.90 * torso_len) and
            ankle_diff_y > (0.28 * torso_len) and
            leg_extended and
            (d_hips < 1.15 or min_ankle_y < (hip_y + 0.50 * torso_len))
        )
        if is_kick:
            return "Kicking"

        # B. Punching / Striking: explosive wrist speed + arm extended outward toward opponent
        # Natural arm swing during walking is gentle (< 0.18) with bent elbows.
        is_punch = (
            v_wrist_max >= 0.24 and
            arm_extended and
            (d_wrist_to_opp_torso < 0.40 or d_wrist_to_opp_head < 0.40 or (d_hips < 0.85 and v_wrist_max >= 0.28))
        )
        if is_punch:
            return "Punching / Striking"

        # C. Pushing / Grappling: close combat grapple with hands clamped on opponent torso
        if d_hips < 0.65 and d_wrist_to_opp_torso < 0.28 and v_wrist_max >= 0.12:
            return "Pushing / Grappling"

        # D. Defensive Guard: hands protecting chin/head in combat range
        if d_wrist_to_head < (0.30 * torso_len) and d_hips < 0.85 and v_wrist_max < 0.14:
            return "Defensive Guard"

        # E. Aggressive Movement / Rush: closing rapidly in combat stance
        torso_speed = float(np.linalg.norm(vel[11] + vel[12]) / 2.0)
        if torso_speed >= 0.16 and d_hips < 0.85 and v_wrist_max >= 0.15:
            return "Aggressive Movement"

        # F. Walking: normal cadence leg movement with low wrist speed
        if 0.04 <= v_ankle_max < 0.22 and v_wrist_max < 0.18:
            return "Walking"

        # G. Standing / Normal Movement
        return "Standing" if v_ankle_max < 0.04 else "Normal Movement"

    def extract_frame_features(self, persons: List[PersonPose]) -> np.ndarray:
        """
        Produces a 510-dimensional feature vector for the frame:
        - Person 1 (250)
        - Person 2 (250)
        - Interaction (10)
        Also assigns fine-grained action labels to each person.
        """
        # Select up to 2 primary actors
        p1 = persons[0] if len(persons) > 0 else None
        p2 = persons[1] if len(persons) > 1 else None

        f_p1 = self.extract_single_person_features(p1)
        f_p2 = self.extract_single_person_features(p2)
        f_inter = self.extract_interaction_features(p1, p2)

        # Classify and assign action for each tracked person
        if p1 is not None and p1.has_pose:
            p1.action = self.classify_person_action(p1, opponent=p2)
        if p2 is not None and p2.has_pose:
            p2.action = self.classify_person_action(p2, opponent=p1)
        for p in persons[2:]:
            if p.has_pose:
                p.action = self.classify_person_action(p, opponent=None)

        return np.concatenate([f_p1, f_p2, f_inter]).astype(np.float32)

    def update_buffer(self, frame_features: np.ndarray) -> Optional[np.ndarray]:
        """
        Adds frame features to rolling buffer.
        If buffer has reached sequence_length (e.g. 30 frames), returns (30, 510) array.
        Otherwise returns None.
        """
        self.rolling_buffer.append(frame_features)
        if len(self.rolling_buffer) > self.sequence_length:
            self.rolling_buffer.pop(0)

        if len(self.rolling_buffer) == self.sequence_length:
            return np.array(self.rolling_buffer, dtype=np.float32)
        return None

    def reset(self):
        """Resets temporal buffers (e.g. at start of new video)."""
        self.rolling_buffer.clear()
        self.prev_person_coords.clear()
        self.prev_person_vel.clear()
        self.prev_inter_hip_dist = None
