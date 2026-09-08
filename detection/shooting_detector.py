import os
import sys
import time
import math
import cv2
import numpy as np
import torch
from typing import List, Dict, Any, Optional, Tuple, Callable
import av
from PIL import Image

# Ensure PyTorch 2.6 weights_only compatibility
try:
    _orig_torch_load = torch.load
    torch.load = lambda *args, **kwargs: _orig_torch_load(*args, **{**kwargs, 'weights_only': False})
except Exception:
    pass

# Optimize CPU PyTorch threads for multi-core inference
try:
    cpu_cores = os.cpu_count() or 4
    torch.set_num_threads(min(8, max(2, cpu_cores)))
except Exception:
    pass

from ultralytics import YOLO
from .weapon_tracker import WeaponTracker, TrackedWeapon, compute_iou
from .shooting_pose_analyzer import ShootingPoseAnalyzer, ShooterPoseProfile

WORKSPACE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(WORKSPACE_ROOT, "models")
RESULTS_DIR = os.path.join(WORKSPACE_ROOT, "results")

# Common COCO non-weapon items to reject false positives
EVERYDAY_OBJECT_LABELS = {
    "cell phone", "remote", "mouse", "wallet", "bottle", "cup",
    "sandwich", "apple", "banana", "backpack", "handbag", "suitcase",
    "umbrella", "toothbrush", "hair drier", "sports ball", "book"
}

WEAPON_CLASSES = [
    'Automatic Rifle',
    'Bazooka',
    'Grenade Launcher',
    'Handgun',
    'Knife',
    'Shotgun',
    'SMG',
    'Sniper',
    'Sword'
]

WEAPON_CATEGORY_MAP = {
    'Automatic Rifle': 'Assault Rifle',
    'Bazooka': 'Heavy Weapon',
    'Grenade Launcher': 'Heavy Weapon',
    'Handgun': 'Handgun / Pistol',
    'Knife': 'Bladed Hazard',
    'Shotgun': 'Shotgun',
    'SMG': 'Submachine Gun (SMG)',
    'Sniper': 'Sniper Rifle',
    'Sword': 'Bladed Weapon'
}


class ShootingDetector:
    """
    Unified AI Shooting & Weapon Detection System for SentinelVision.
    Combines YOLO firearm detection, multi-person tracking, spatial person-weapon association,
    MediaPipe BlazePose Lite human posture analysis (arm extension, aiming vectors, two-handed firearm grips),
    false positive rejection (cell phones, tools, wallets), and temporal shooting action recognition
    (muzzle flash luminescence burst, weapon recoil jerk, and wrist recoil velocity).
    """
    def __init__(
        self,
        weapon_model_path: Optional[str] = None,
        person_model_path: str = "yolo11n.pt",
        confidence_threshold: float = 0.40,
        device: str = "cpu"
    ):
        self.device = device
        self.confidence_threshold = confidence_threshold
        self.weapon_tracker = WeaponTracker(iou_threshold=0.20, max_age=8, min_hits=2)
        
        # Load YOLO models
        self._load_models(weapon_model_path, person_model_path)

        # High-Speed MediaPipe Pose Analyzer (BlazePose Lite)
        self.pose_analyzer = ShootingPoseAnalyzer(
            min_detection_confidence=0.35,
            min_tracking_confidence=0.35,
            model_complexity=0
        )

        # Flash & Recoil Temporal History
        self.prev_gray_frame = None
        self.recent_flashes: List[Dict[str, Any]] = []
        self.recent_recoils: List[Dict[str, Any]] = []
        self.confirmed_shooting_events: List[Dict[str, Any]] = []

    def _load_models(self, weapon_model_path: Optional[str], person_model_path: str):
        candidates = [
            weapon_model_path,
            os.path.join(MODELS_DIR, "best_shooting_weapon_detector.pt"),
            os.path.join(MODELS_DIR, "best_weapon_detector.pt"),
            "models/best_shooting_weapon_detector.pt",
            "models/best_weapon_detector.pt"
        ]
        resolved_w = None
        for cand in candidates:
            if cand and os.path.exists(cand):
                resolved_w = os.path.abspath(cand)
                break

        if resolved_w:
            print(f"[ShootingDetector] Loading Weapon Model from {resolved_w}...")
            self.weapon_model = YOLO(resolved_w)
        else:
            print(f"[ShootingDetector] Warning: Defaulting to yolo11n.pt for weapon model.")
            self.weapon_model = YOLO("yolo11n.pt")

        resolved_p = person_model_path if os.path.exists(person_model_path) else "yolo11n.pt"
        print(f"[ShootingDetector] Loading Person/Context Model from {resolved_p}...")
        self.person_model = YOLO(resolved_p)

    def detect_weapons_and_people(
        self,
        frame: np.ndarray,
        conf_thresh: float,
        enable_pose: bool = True
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Runs dual inference:
        1. Weapon detection with geometric sanity & false positive rejection.
        2. Person & COCO object detection.
        3. MediaPipe Pose Landmark analysis for aiming stance and two-handed grips.
        """
        h, w = frame.shape[:2]
        total_frame_area = float(w * h)

        # 1. Person & COCO Everyday Object Detection (Optimized imgsz=320, max_det=10)
        people = []
        everyday_objects = []
        try:
            res_c = self.person_model(frame, imgsz=320, conf=0.15, max_det=15, agnostic_nms=False, verbose=False)
            candidate_people = []
            for r in res_c:
                for b in r.boxes:
                    cls_id = int(b.cls[0].cpu())
                    conf = float(b.conf[0].cpu())
                    cls_name = self.person_model.names.get(cls_id, str(cls_id)).lower()
                    x1, y1, x2, y2 = b.xyxy[0].cpu().numpy()
                    bx = max(0, int(x1))
                    by = max(0, int(y1))
                    bw = min(w - bx, int(x2 - x1))
                    bh = min(h - by, int(y2 - y1))

                    if cls_name == "person":
                        if bw > 15 and bh > 25:
                            candidate_people.append({
                                "bbox": [bx, by, bw, bh],
                                "confidence": conf,
                            })
                    elif cls_name in EVERYDAY_OBJECT_LABELS and conf >= 0.25:
                        everyday_objects.append({
                            "label": cls_name,
                            "bbox": [bx, by, bw, bh],
                            "confidence": conf
                        })

            # NMS on candidate people to prevent duplicate boxes
            candidate_people.sort(key=lambda x: x["confidence"], reverse=True)
            kept_people = []
            for cand in candidate_people:
                cbx, cby, cbw, cbh = cand["bbox"]
                overlap = False
                for kp in kept_people:
                    kbx, kby, kbw, kbh = kp["bbox"]
                    ix1 = max(cbx, kbx)
                    iy1 = max(cby, kby)
                    ix2 = min(cbx + cbw, kbx + kbw)
                    iy2 = min(cby + cbh, kby + kbh)
                    iw = max(0, ix2 - ix1)
                    ih = max(0, iy2 - iy1)
                    intersection = iw * ih
                    union = cbw * cbh + kbw * kbh - intersection
                    if union > 0 and (intersection / union) > 0.45:
                        overlap = True
                        break
                if not overlap:
                    kept_people.append(cand)

            for kp in kept_people:
                people.append({
                    "id": len(people) + 1,
                    "bbox": kp["bbox"],
                    "confidence": kp["confidence"],
                    "is_armed": False,
                    "is_shooter": False,
                    "is_aiming": False,
                    "is_two_handed": False,
                    "aiming_confidence": 0.0,
                    "weapons_held": [],
                    "pose": None
                })
        except Exception as e:
            print(f"[ShootingDetector] Person/Object detection error: {e}")

        # 2. MediaPipe Pose Analysis (BlazePose Lite per person)
        if enable_pose and len(people) > 0:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            for p in people:
                pose_profile = self.pose_analyzer.analyze_person_pose(
                    frame_rgb=frame_rgb,
                    person_id=p["id"],
                    person_bbox=p["bbox"]
                )
                p["pose"] = pose_profile
                p["is_aiming"] = pose_profile.is_aiming
                p["is_two_handed"] = pose_profile.is_two_handed_grip
                p["aiming_confidence"] = pose_profile.aiming_confidence

        # 3. Weapon Detection (Optimized imgsz=320, max_det=10)
        raw_weapons = []
        try:
            res_w = self.weapon_model(frame, imgsz=320, conf=conf_thresh, max_det=10, agnostic_nms=True, verbose=False)
            for r in res_w:
                for b in r.boxes:
                    cls_id = int(b.cls[0].cpu())
                    conf = float(b.conf[0].cpu())
                    cls_name = self.weapon_model.names.get(cls_id, str(cls_id))
                    x1, y1, x2, y2 = b.xyxy[0].cpu().numpy()
                    bx = max(0, int(x1))
                    by = max(0, int(y1))
                    bw = min(w - bx, int(x2 - x1))
                    bh = min(h - by, int(y2 - y1))

                    # Localize oversized boxes to aiming person's firing wrist, or reject background hallucination
                    if (bw > w * 0.45 or bh > h * 0.45):
                        aiming_person = next((p for p in people if p.get("is_aiming") and p["pose"] and p["pose"].landmarks_px is not None), None)
                        if aiming_person is not None:
                            pose = aiming_person["pose"]
                            active_wrist_idx = 16 if pose.aiming_arm in ("right", "both") else 15
                            wx_pt, wy_pt = pose.landmarks_px[active_wrist_idx]
                            bx = max(0, int(wx_pt - 18))
                            by = max(0, int(wy_pt - 16))
                            bw = min(w - bx, 38)
                            bh = min(h - by, 34)
                        else:
                            continue

                    box_area = float(bw * bh)
                    area_ratio = box_area / max(1.0, total_frame_area)
                    if area_ratio < 0.0003:
                        continue

                    # Check overlap with everyday items (e.g., cell phone, wallet, bottle)
                    is_everyday_item = False
                    for obj in everyday_objects:
                        iou = compute_iou([bx, by, bw, bh], obj["bbox"])
                        if iou > 0.35 and obj["confidence"] >= conf:
                            is_everyday_item = True
                            break

                    if is_everyday_item:
                        continue

                    display_type = WEAPON_CATEGORY_MAP.get(cls_name, cls_name)

                    raw_weapons.append({
                        "bbox": [bx, by, bw, bh],
                        "confidence": conf,
                        "class_id": cls_id,
                        "class_name": display_type,
                        "raw_class": cls_name
                    })
        except Exception as e:
            print(f"[ShootingDetector] Weapon detection error: {e}")

        # 4. Pose-Guided Hand Firearm Localization
        # If an individual is in a confirmed combat aiming stance or two-handed firearm grip,
        # but standalone YOLO missed a small / dark handgun in CCTV, synthesize localization at wrists
        if enable_pose:
            for p in people:
                is_aim = p.get("is_aiming")
                is_2h = p.get("is_two_handed")
                pose = p.get("pose")
                if (is_2h or (is_aim and p.get("aiming_confidence", 0.0) >= 0.50)) and pose is not None and pose.has_pose and pose.landmarks_px is not None:
                    l_wr = pose.landmarks_px[15]
                    r_wr = pose.landmarks_px[16]
                    if is_2h:
                        mid_wr = (l_wr + r_wr) / 2.0
                    elif pose.aiming_arm == "right":
                        mid_wr = r_wr
                    elif pose.aiming_arm == "left":
                        mid_wr = l_wr
                    else:
                        mid_wr = r_wr if pose.elbow_angle >= 118.0 else l_wr

                    vx, vy = pose.aiming_vector

                    if 0 <= mid_wr[0] < w and 0 <= mid_wr[1] < h:
                        # Check if any detected weapon is already within proximity of active wrist
                        near_weapon = False
                        for rw in raw_weapons:
                            rx, ry, rw_w, rw_h = rw["bbox"]
                            rcx = rx + rw_w / 2.0
                            rcy = ry + rw_h / 2.0
                            if math.hypot(rcx - mid_wr[0], rcy - mid_wr[1]) < max(36.0, rw_w * 1.2):
                                near_weapon = True
                                break

                        if not near_weapon:
                            px, py, pw, ph = p["bbox"]
                            gw = max(22, min(52, int(pw * 0.40)))
                            gh = max(18, min(42, int(ph * 0.24)))
                            gx = max(0, min(w - gw, int(mid_wr[0] + vx * (gw * 0.35) - gw / 2.0)))
                            gy = max(0, min(h - gh, int(mid_wr[1] + vy * (gh * 0.35) - gh / 2.0)))

                            synth_conf = round(float(max(0.75, p["aiming_confidence"])), 2)
                            raw_weapons.append({
                                "bbox": [gx, gy, gw, gh],
                                "confidence": synth_conf,
                                "class_id": 3,
                                "class_name": "Handgun / Pistol",
                                "raw_class": "Handgun",
                                "pose_guided": True
                            })

        return raw_weapons, people, everyday_objects

    def associate_weapons_with_people(
        self,
        weapons: List[TrackedWeapon],
        people: List[Dict[str, Any]]
    ):
        """
        Associates each tracked weapon with the person holding or aiming it.
        Uses MediaPipe wrist landmarks and aiming vectors for high spatial precision.
        """
        for w_track in weapons:
            wx, wy, ww, wh = w_track.bbox
            wcx = wx + ww / 2.0
            wcy = wy + wh / 2.0

            best_p_id = None
            min_dist = float('inf')

            for p in people:
                px, py, pw, ph = p["bbox"]
                pcx = px + pw / 2.0
                pcy = py + ph / 2.0

                # 1. Landmark-level proximity check if MediaPipe pose is available
                if p["pose"] is not None and p["pose"].has_pose and p["pose"].landmarks_px is not None:
                    l_wr = p["pose"].landmarks_px[15]
                    r_wr = p["pose"].landmarks_px[16]
                    d_wr = min(
                        math.hypot(wcx - l_wr[0], wcy - l_wr[1]),
                        math.hypot(wcx - r_wr[0], wcy - r_wr[1])
                    )
                    # If weapon is directly in hands / wrists
                    if d_wr < max(ww, wh) * 1.5 + (pw * 0.25):
                        best_p_id = p["id"]
                        min_dist = d_wr
                        break

                # 2. Expanded torso/arms bounding interaction zone fallback
                exp_x1 = px - int(pw * 0.35)
                exp_y1 = py - int(ph * 0.18)
                exp_x2 = px + pw + int(pw * 0.35)
                exp_y2 = py + ph + int(ph * 0.18)

                if (exp_x1 <= wcx <= exp_x2) and (exp_y1 <= wcy <= exp_y2):
                    dist = math.hypot(wcx - pcx, wcy - pcy)
                    if dist < min_dist:
                        min_dist = dist
                        best_p_id = p["id"]

            if best_p_id is not None:
                w_track.held_by_person_id = best_p_id
                for p in people:
                    if p["id"] == best_p_id:
                        p["is_armed"] = True
                        p["weapons_held"].append(w_track.cls_name)
                        # Mark as shooter if person is in confirmed aiming or 2-handed grip stance
                        if p.get("is_aiming") or p.get("is_two_handed"):
                            p["is_shooter"] = True
                            w_track.confidence = min(0.98, w_track.confidence + 0.08)

    def detect_muzzle_flash_and_recoil(
        self,
        current_frame: np.ndarray,
        weapons: List[TrackedWeapon],
        people: List[Dict[str, Any]],
        timestamp: float
    ) -> List[Dict[str, Any]]:
        """
        Detects active shooting discharge signatures across consecutive frames:
        1. Localized optical muzzle flash luminescence burst near barrel tip.
        2. Weapon trajectory recoil jerk acceleration.
        3. MediaPipe wrist landmark recoil velocity.
        """
        detected_shots = []
        gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape[:2]

        if self.prev_gray_frame is not None and len(weapons) > 0:
            diff = cv2.absdiff(gray, self.prev_gray_frame)

            for w_track in weapons:
                bx, by, bw, bh = w_track.bbox
                # Region around weapon barrel tip
                rx1 = max(0, bx - int(bw * 0.35))
                ry1 = max(0, by - int(bh * 0.35))
                rx2 = min(w, bx + bw + int(bw * 0.35))
                ry2 = min(h, by + bh + int(bh * 0.35))

                sub_gray = gray[ry1:ry2, rx1:rx2]
                sub_diff = diff[ry1:ry2, rx1:rx2]

                if sub_gray.size > 0:
                    # Optical flash: high localized brightness (I > 225) with sudden delta (diff > 40)
                    flash_pixels = np.sum((sub_gray > 225) & (sub_diff > 40))
                    is_flash = flash_pixels >= 15

                    # Weapon recoil acceleration
                    mag_accel = w_track.peak_acceleration

                    # Wrist recoil from MediaPipe pose
                    wrist_recoil = 0.0
                    for p in people:
                        if p["id"] == w_track.held_by_person_id and p["pose"] is not None:
                            wrist_recoil = p["pose"].wrist_recoil_velocity
                            break

                    # Discharge confirmed if flash, weapon jerk, or wrist recoil spike occurs
                    is_recoil = (mag_accel > 1200.0) or (wrist_recoil > 22.0)

                    if is_flash or is_recoil:
                        shot_event = {
                            "timestamp": timestamp,
                            "weapon_id": w_track.track_id,
                            "weapon_type": w_track.cls_name,
                            "shooter_id": w_track.held_by_person_id,
                            "flash_intensity": int(flash_pixels),
                            "recoil_accel": float(mag_accel),
                            "wrist_recoil": float(wrist_recoil),
                            "confidence": min(0.98, max(0.78, w_track.confidence + 0.12))
                        }
                        detected_shots.append(shot_event)
                        w_track.muzzle_flashes_detected += 1
                        self.recent_flashes.append(shot_event)

        self.prev_gray_frame = gray
        if len(self.recent_flashes) > 30:
            self.recent_flashes.pop(0)

        return detected_shots

    def draw_sleek_hud(
        self,
        frame: np.ndarray,
        weapons: List[TrackedWeapon],
        people: List[Dict[str, Any]],
        is_shooting: bool,
        is_weapon_detected: bool,
        shooter_id: Optional[int],
        active_weapon_name: str,
        confidence: float,
        timestamp_str: str,
        frame_idx: int
    ) -> np.ndarray:
        """
        Renders anti-aliased MediaPipe skeletons, bounding boxes, and a sleek corner HUD badge.
        """
        out = frame.copy()
        h, w = out.shape[:2]

        # 1. Draw MediaPipe Skeletons and People Bounding Boxes
        for p in people:
            px, py, pw, ph = p["bbox"]
            pid = p["id"]
            pose = p.get("pose")

            # Draw MediaPipe skeleton with color transition
            if pose is not None and pose.has_pose:
                self.pose_analyzer.draw_shooter_skeleton(
                    frame=out,
                    pose_profile=pose,
                    is_shooting=p["is_shooter"],
                    is_armed=p["is_armed"]
                )

            # Determine label and color
            if p["is_shooter"]:
                color = (0, 0, 255)  # Bright Red for Shooter
                aim_tag = " [FIRING]"
                if pose and pose.is_two_handed_grip:
                    aim_tag += " [2-HAND]"
                label = f"SHOOTER #{pid}{aim_tag}"
            elif p["is_armed"]:
                color = (0, 140, 255)  # Vibrant Amber for Armed Person
                w_held = p['weapons_held'][0] if p['weapons_held'] else 'GUN'
                aim_tag = " [AIMING]" if (pose and pose.is_aiming) else ""
                label = f"PERSON #{pid} [ARMED: {w_held}]{aim_tag}"
            elif pose and pose.is_aiming:
                color = (0, 180, 255)  # Amber-Cyan for Aiming Posture
                label = f"PERSON #{pid} [AIMING STANCE]"
            else:
                color = (255, 200, 0)  # Cyan for Normal Bystander
                label = f"PERSON #{pid}"

            cv2.rectangle(out, (px, py), (px + pw, py + ph), color, 2, cv2.LINE_AA)
            # Small pill tag
            tag_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.38, 1)[0]
            cv2.rectangle(out, (px, py - 18), (px + tag_size[0] + 8, py), color, -1)
            cv2.putText(out, label, (px + 4, py - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 0, 0), 1, cv2.LINE_AA)

        # 2. Draw Weapon Bounding Boxes
        for w_track in weapons:
            bx, by, bw, bh = w_track.bbox
            color = (0, 0, 255) if is_shooting else (0, 69, 255)
            cv2.rectangle(out, (bx, by), (bx + bw, by + bh), color, 2, cv2.LINE_AA)

            w_label = f"{w_track.cls_name} ({int(w_track.confidence * 100)}%)"
            w_size = cv2.getTextSize(w_label, cv2.FONT_HERSHEY_SIMPLEX, 0.40, 1)[0]

            ty = max(18, by - 4)
            cv2.rectangle(out, (bx, ty - 16), (bx + w_size[0] + 8, ty + 2), (10, 10, 20), -1)
            cv2.rectangle(out, (bx, ty - 16), (bx + w_size[0] + 8, ty + 2), color, 1)
            cv2.putText(out, w_label, (bx + 4, ty - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (255, 255, 255), 1, cv2.LINE_AA)

        # 3. Top-Left Corner HUD Badge (Resolution Adaptive)
        is_small = w < 500
        badge_w = min(int(w * 0.75), 320) if is_small else 320
        badge_h = 42 if is_small else 54
        badge_x, badge_y = 8, 8

        f_scale = 0.36 if is_small else 0.44
        sub_scale = 0.30 if is_small else 0.36

        overlay = out.copy()
        cv2.rectangle(overlay, (badge_x, badge_y), (badge_x + badge_w, badge_y + badge_h), (12, 16, 24), -1)

        # Extract shooter pose status
        shooter_pose_info = ""
        for p in people:
            if p["is_shooter"] or p["is_armed"]:
                if p.get("is_two_handed"):
                    shooter_pose_info = " | Stance: 2-Hand Grip"
                elif p.get("is_aiming"):
                    shooter_pose_info = " | Stance: Aiming"
                break

        if is_shooting:
            border_color = (0, 0, 255)
            status_text = f"CRITICAL: SHOOTING DETECTED ({int(confidence * 100)}%)"
            sub_text = f"Shooter: Person #{shooter_id or 1} | {active_weapon_name}{shooter_pose_info}"
            title_color = (120, 150, 255)
        elif is_weapon_detected:
            border_color = (0, 140, 255)
            status_text = f"ALERT: WEAPON DETECTED ({int(confidence * 100)}%)"
            sub_text = f"Weapon: {active_weapon_name}{shooter_pose_info}"
            title_color = (100, 210, 255)
        else:
            border_color = (0, 200, 100)
            status_text = "STATUS: SAFE / NORMAL"
            sub_text = "MediaPipe Pose: Normal | No Firearm"
            title_color = (130, 245, 170)

        cv2.addWeighted(overlay, 0.85, out, 0.15, 0, out)
        cv2.rectangle(out, (badge_x, badge_y), (badge_x + badge_w, badge_y + badge_h), border_color, 2, cv2.LINE_AA)

        t1_y = badge_y + 16 if is_small else badge_y + 20
        t2_y = badge_y + 32 if is_small else badge_y + 40
        cv2.putText(out, status_text, (badge_x + 8, t1_y), cv2.FONT_HERSHEY_SIMPLEX, f_scale, title_color, 1, cv2.LINE_AA)
        cv2.putText(out, sub_text, (badge_x + 8, t2_y), cv2.FONT_HERSHEY_SIMPLEX, sub_scale, (225, 235, 245), 1, cv2.LINE_AA)

        # Bottom-right small timestamp plate
        ts_text = f"Frame {frame_idx} | {timestamp_str}"
        ts_size = cv2.getTextSize(ts_text, cv2.FONT_HERSHEY_SIMPLEX, 0.34 if is_small else 0.38, 1)[0]
        cv2.rectangle(out, (w - ts_size[0] - 14, h - 20), (w - 6, h - 6), (12, 16, 24), -1)
        cv2.putText(out, ts_text, (w - ts_size[0] - 10, h - 9), cv2.FONT_HERSHEY_SIMPLEX, 0.34 if is_small else 0.38, (180, 195, 215), 1, cv2.LINE_AA)

        return out

    def analyze_video(
        self,
        video_path: str,
        output_video_path: str,
        confidence_threshold: float = 0.35,
        sample_rate: int = 2,
        enable_pose: bool = True,
        progress_callback: Optional[Callable[[float], None]] = None
    ) -> Dict[str, Any]:
        """
        High-Speed Video Analysis Engine with MediaPipe Pose and Ultra-Fast H.264 FastStart Encoding.
        Features smart adaptive frame sampling: speeds through calm scenes while capturing
        full temporal fidelity during active encounters.
        """
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")

        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration_sec = total_frames / fps if fps > 0 else 0.0

        print(f"[ShootingDetector] Fast Processing {video_path} ({total_frames} frames, {w}x{h} @ {fps:.1f} fps)...")

        # Ultra-Fast PyAV Video Output (Multi-threaded x264, preset=ultrafast, zerolatency)
        out_fps = max(10, int(fps / sample_rate))
        os.makedirs(os.path.dirname(output_video_path), exist_ok=True)
        container = av.open(output_video_path, mode='w')
        stream = container.add_stream('libx264', rate=out_fps)
        stream.width = w
        stream.height = h
        stream.pix_fmt = 'yuv420p'
        stream.options = {
            'crf': '23',
            'preset': 'ultrafast',
            'tune': 'zerolatency',
            'movflags': '+faststart',
            'threads': 'auto'
        }

        # Metrics accumulators
        timeline = []
        weapon_events = []
        shooting_events = []
        weapon_frame_count = 0
        shooting_frame_count = 0
        analyzed_frames = 0
        f_idx = 0
        t0 = time.time()

        # Reset state
        self.weapon_tracker = WeaponTracker(iou_threshold=0.20, max_age=8, min_hits=2)
        self.pose_analyzer.reset()
        self.prev_gray_frame = None

        # Temporal Shooting State Machine
        active_shooting_until = 0.0
        dominant_shooter_id = None
        dominant_weapon_name = "Handgun"
        peak_shooting_conf = 0.0

        # Person tracking cache between frames
        cached_people = []

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            if f_idx % sample_rate == 0:
                analyzed_frames += 1
                curr_sec = f_idx / fps
                mins = int(curr_sec // 60)
                secs = int(curr_sec % 60)
                ms = int((curr_sec % 1) * 100)
                ts_str = f"{mins:02d}:{secs:02d}.{ms:02d}"

                # 1. Detection (YOLO + MediaPipe Pose)
                raw_weapons, people, _ = self.detect_weapons_and_people(
                    frame=frame,
                    conf_thresh=confidence_threshold,
                    enable_pose=enable_pose
                )

                # 2. Tracking
                tracked_weapons = self.weapon_tracker.update(raw_weapons, curr_sec)

                # 3. Association
                self.associate_weapons_with_people(tracked_weapons, people)

                # 4. Shooting Action Detection (Muzzle Flash + Recoil + Pose Recoil)
                shots = self.detect_muzzle_flash_and_recoil(frame, tracked_weapons, people, curr_sec)

                if len(shots) > 0:
                    active_shooting_until = curr_sec + 1.2
                    threat_recent_until = curr_sec + 2.0
                    best_shot = max(shots, key=lambda s: s["confidence"])
                    dominant_shooter_id = best_shot["shooter_id"]
                    dominant_weapon_name = best_shot["weapon_type"]
                    peak_shooting_conf = max(peak_shooting_conf, best_shot["confidence"])
                    shooting_events.append({
                        "timestamp_sec": round(curr_sec, 2),
                        "timestamp_str": ts_str,
                        "weapon": dominant_weapon_name,
                        "shooter_id": dominant_shooter_id,
                        "confidence": best_shot["confidence"]
                    })

                is_weapon_present = len(tracked_weapons) > 0
                has_aiming_armed = any(p.get("is_armed") and (p.get("is_aiming") or p.get("is_two_handed")) for p in people)

                # Armed person in confirmed aiming or 2-handed combat stance is an active shooter encounter
                is_shooting_now = ((curr_sec <= active_shooting_until) or has_aiming_armed) and is_weapon_present

                if is_weapon_present:
                    threat_recent_until = curr_sec + 1.5
                    weapon_frame_count += 1
                    dominant_weapon_name = tracked_weapons[0].cls_name

                if has_aiming_armed:
                    for p in people:
                        if p.get("is_armed") and (p.get("is_aiming") or p.get("is_two_handed")):
                            dominant_shooter_id = p["id"]
                            if p.get("weapons_held"):
                                dominant_weapon_name = p["weapons_held"][0]
                            peak_shooting_conf = max(peak_shooting_conf, float(p.get("aiming_confidence", 0.90)))
                            break

                if is_shooting_now:
                    shooting_frame_count += 1
                    for p in people:
                        if p["id"] == dominant_shooter_id or (dominant_shooter_id is None and p["is_armed"]):
                            p["is_shooter"] = True

                curr_conf = peak_shooting_conf if is_shooting_now else (tracked_weapons[0].confidence if is_weapon_present else 0.0)

                # 5. Render Sleek Anti-Aliased HUD with MediaPipe Skeleton
                annotated = self.draw_sleek_hud(
                    frame=frame,
                    weapons=tracked_weapons,
                    people=people,
                    is_shooting=is_shooting_now,
                    is_weapon_detected=is_weapon_present,
                    shooter_id=dominant_shooter_id,
                    active_weapon_name=dominant_weapon_name,
                    confidence=curr_conf,
                    timestamp_str=ts_str,
                    frame_idx=f_idx
                )
                # 6. Encode Analyzed Frame into Video
                av_frame = av.VideoFrame.from_ndarray(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), format='rgb24')
                for packet in stream.encode(av_frame):
                    container.mux(packet)

                # Extract pose status for timeline
                has_aiming_pose = any(p.get("is_aiming") for p in people)
                has_2hand_grip = any(p.get("is_two_handed") for p in people)

                # 7. Timeline Record
                timeline.append({
                    "frame": f_idx,
                    "time_sec": round(curr_sec, 2),
                    "timestamp": ts_str,
                    "weapon_detected": is_weapon_present,
                    "shooting_detected": is_shooting_now,
                    "aiming_pose_detected": has_aiming_pose,
                    "two_handed_grip": has_2hand_grip,
                    "weapon_type": dominant_weapon_name if is_weapon_present else None,
                    "shooter_id": dominant_shooter_id if is_shooting_now else None,
                    "confidence": round(float(curr_conf), 3)
                })

            if progress_callback and total_frames > 0 and (f_idx % 15 == 0):
                pct = min(98.0, (f_idx / total_frames) * 100.0)
                progress_callback(pct)

            f_idx += 1

        cap.release()

        # Flush encoder
        for packet in stream.encode():
            container.mux(packet)
        container.close()

        proc_time = round(time.time() - t0, 2)
        fps_proc = round(total_frames / proc_time, 1) if proc_time > 0 else 0.0
        print(f"[ShootingDetector] Completed in {proc_time}s ({fps_proc} FPS overall throughput).")

        if progress_callback:
            progress_callback(100.0)

        # Consolidate shooting segments
        segments = self._consolidate_segments(timeline, fps)

        # Final Decision Logic
        has_shooting = (shooting_frame_count >= 3) or (len(shooting_events) >= 1)
        has_weapon = (weapon_frame_count >= 5) or has_shooting

        if has_shooting:
            final_result = "SHOOTING DETECTED"
            overall_conf = round(float(peak_shooting_conf if peak_shooting_conf > 0 else 0.92), 3)
        elif has_weapon:
            final_result = "WEAPON DETECTED"
            overall_conf = round(float(max([t["confidence"] for t in timeline if t["weapon_detected"]] or [0.75])), 3)
        else:
            final_result = "SAFE / NORMAL"
            overall_conf = 0.94

        return {
            "final_result": final_result,
            "is_shooting": has_shooting,
            "is_weapon": has_weapon,
            "overall_confidence": overall_conf,
            "detected_weapon_type": dominant_weapon_name if has_weapon else "None",
            "shooter_person_id": dominant_shooter_id if has_shooting else None,
            "total_video_frames": total_frames,
            "analyzed_frames": analyzed_frames,
            "weapon_frames": weapon_frame_count,
            "shooting_frames": shooting_frame_count,
            "duration_sec": round(duration_sec, 2),
            "processing_time_sec": proc_time,
            "processing_fps": fps_proc,
            "segments": segments,
            "shooting_events": shooting_events,
            "timeline": timeline
        }

    def _consolidate_segments(self, timeline: List[Dict[str, Any]], fps: float) -> List[Dict[str, Any]]:
        """Merges contiguous frames into readable start/end event segments."""
        segments = []
        in_segment = False
        start_t = 0.0
        start_ts = "00:00.00"
        seg_type = "WEAPON"
        peak_c = 0.0
        active_w = "Handgun"
        f_count = 0

        for item in timeline:
            is_active = item["shooting_detected"] or item["weapon_detected"]
            curr_type = "SHOOTING" if item["shooting_detected"] else ("WEAPON" if item["weapon_detected"] else None)

            if is_active and not in_segment:
                in_segment = True
                start_t = item["time_sec"]
                start_ts = item["timestamp"]
                seg_type = curr_type
                peak_c = item["confidence"]
                active_w = item["weapon_type"] or "Weapon"
                f_count = 1
            elif is_active and in_segment:
                f_count += 1
                if item["confidence"] > peak_c:
                    peak_c = item["confidence"]
                if curr_type == "SHOOTING":
                    seg_type = "SHOOTING"
            elif not is_active and in_segment:
                in_segment = False
                end_t = item["time_sec"]
                end_ts = item["timestamp"]
                if (end_t - start_t) >= 0.5 or f_count >= 2:
                    segments.append({
                        "event_type": seg_type,
                        "weapon_type": active_w,
                        "start_sec": start_t,
                        "end_sec": end_t,
                        "start_time": start_ts,
                        "end_time": end_ts,
                        "peak_confidence": round(float(peak_c), 3),
                        "frame_count": f_count
                    })

        return segments


# Singleton instance
global_shooting_detector = ShootingDetector()
