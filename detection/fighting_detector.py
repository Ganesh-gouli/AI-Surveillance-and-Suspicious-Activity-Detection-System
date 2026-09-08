import os
import sys
import time
import cv2
import numpy as np
import torch
from typing import List, Dict, Tuple, Optional, Any

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from .pose_detector import MultiPersonPoseDetector, PersonPose
from .feature_extractor import KinematicPoseFeatureExtractor
from .temporal_model import FightingBiLSTM

class H264VideoWriter:
    """
    Browser-native H.264 Video Writer using PyAV (libx264, yuv420p).
    Generates video files directly playable in Chrome, Edge, Safari, and Firefox HTML5 video players.
    Falls back gracefully to OpenCV VideoWriter if PyAV is unavailable.
    """
    def __init__(self, output_path: str, fps: float, width: int, height: int):
        self.output_path = output_path
        self.width = width
        self.height = height
        self.fps = fps
        self.use_av = False
        self.writer = None
        self.container = None
        self.stream = None

        try:
            import av
            self.container = av.open(output_path, mode='w', options={'movflags': '+faststart'})
            self.stream = self.container.add_stream('libx264', rate=int(round(max(1.0, fps))))
            self.stream.width = width
            self.stream.height = height
            self.stream.pix_fmt = 'yuv420p'
            self.stream.options = {'preset': 'ultrafast', 'crf': '23'}
            self.use_av = True
        except Exception as e:
            print(f"[H264VideoWriter] Notice: PyAV initialization fallback: {e}")
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            self.writer = cv2.VideoWriter(output_path, fourcc, max(1.0, fps), (width, height))

    def write(self, frame_bgr: np.ndarray):
        if self.use_av and self.container is not None:
            import av
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            av_frame = av.VideoFrame.from_ndarray(frame_rgb, format='rgb24')
            for packet in self.stream.encode(av_frame):
                self.container.mux(packet)
        elif self.writer is not None:
            self.writer.write(frame_bgr)

    def release(self):
        if self.use_av and self.container is not None:
            try:
                for packet in self.stream.encode():
                    self.container.mux(packet)
                self.container.close()
            except Exception as e:
                print(f"[H264VideoWriter] Error closing container: {e}")
            finally:
                self.container = None
        elif self.writer is not None:
            self.writer.release()
            self.writer = None

class FightingDetector:
    """
    Production-Grade AI Human Fighting Detector using MediaPipe Pose + BiLSTM.
    Enforces temporal dynamics:
    - Never uses person count as fighting decision.
    - Never uses distance-only heuristics.
    - Requires temporal sequence pattern matching + consecutive frame confirmation.
    """
    def __init__(
        self,
        model_path: Optional[str] = None,
        confidence_threshold: float = 0.70,
        min_consecutive_frames: int = 8,
        smooth_window: int = 7
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.confidence_threshold = confidence_threshold
        self.min_consecutive_frames = min_consecutive_frames
        self.smooth_window = smooth_window

        resolved_model_path = model_path or os.path.join(BASE_DIR, "models", "fighting_mediapipe_bilstm.pt")
        self.model_path = resolved_model_path
        self.model = None
        self.is_loaded = False

        if os.path.exists(resolved_model_path):
            try:
                ckpt = torch.load(resolved_model_path, map_location=self.device, weights_only=False)
                input_dim = ckpt.get("input_dim", 510)
                hidden_dim = ckpt.get("hidden_dim", 128)
                self.model = FightingBiLSTM(input_dim=input_dim, hidden_dim=hidden_dim).to(self.device)
                self.model.load_state_dict(ckpt["model_state_dict"])
                self.model.eval()
                self.is_loaded = True
                print(f"[FightingDetector] Loaded trained PyTorch BiLSTM model from: {resolved_model_path}")
            except Exception as e:
                print(f"[FightingDetector] Error loading model from {resolved_model_path}: {e}")
        else:
            print(f"[FightingDetector] Checkpoint not found at {resolved_model_path}. Operating in uninitialized mode.")

        self.pose_detector = MultiPersonPoseDetector(min_detection_confidence=0.35)
        self.feature_extractor = KinematicPoseFeatureExtractor(sequence_length=30)

        # Real-time state trackers
        self.prob_history: List[float] = []
        self.is_fighting_state = False
        self.consec_high = 0
        self.consec_low = 0
        self.last_alert_time = 0.0
        self.alert_cooldown_sec = 5.0

    @staticmethod
    def consolidate_segments(raw_segments: List[Dict[str, Any]], max_gap_sec: float = 1.5) -> List[Dict[str, Any]]:
        """
        Merges adjacent fighting segments separated by brief pauses (< max_gap_sec),
        producing smooth, continuous incident intervals without flicker.
        """
        if not raw_segments:
            return []
        merged = []
        curr = raw_segments[0].copy()
        curr["actions"] = curr.get("actions", [])
        for nxt in raw_segments[1:]:
            if (nxt["start_sec"] - curr["end_sec"]) <= max_gap_sec:
                curr["end_sec"] = nxt["end_sec"]
                curr["end_time"] = nxt["end_time"]
                curr["peak_conf"] = max(curr["peak_conf"], nxt["peak_conf"])
                curr["frame_count"] += nxt["frame_count"]
                for act in nxt.get("actions", []):
                    if act not in curr["actions"]:
                        curr["actions"].append(act)
            else:
                curr["detected_action"] = ", ".join(curr["actions"]) if curr["actions"] else "Physical Striking"
                merged.append(curr)
                curr = nxt.copy()
                curr["actions"] = curr.get("actions", [])
        curr["detected_action"] = ", ".join(curr["actions"]) if curr["actions"] else "Physical Combat"
        merged.append(curr)

        # Filter verified segments: must have at least 1.2s duration, frame count >= 6,
        # confirmed combat action (Punching, Kicking, Grappling), and peak confidence >= 72%
        combat_actions = {"Punching / Striking", "Kicking", "Pushing / Grappling", "Physical Combat"}
        verified = []
        for s in merged:
            dur = s["end_sec"] - s["start_sec"]
            has_combat = any(act in combat_actions for act in s.get("actions", []))
            if dur >= 1.2 and s["frame_count"] >= 6 and has_combat and s["peak_conf"] >= 72.0:
                verified.append(s)
        return verified

    def detect_frame(self, frame_bgr: np.ndarray, camera_id: str = "CAM-01") -> Dict[str, Any]:
        """
        Runs real-time inference on a single incoming frame (e.g. RTSP stream).
        Applies Hysteresis Schmitt Trigger for smooth detection without flutter.
        """
        if frame_bgr is None or frame_bgr.size == 0:
            return {"is_fighting": False, "confidence": 0.0, "status": "NO_FRAME"}

        persons = self.pose_detector.process_frame(frame_bgr)
        frame_feat = self.feature_extractor.extract_frame_features(persons)
        seq_buffer = self.feature_extractor.update_buffer(frame_feat)

        raw_prob = 0.0
        if seq_buffer is not None and self.is_loaded:
            with torch.no_grad():
                inp = torch.tensor(seq_buffer, dtype=torch.float32).unsqueeze(0).to(self.device)
                pred, _ = self.model(inp)
                raw_prob = float(pred.item())

        # Temporal Smoothing (Moving average)
        self.prob_history.append(raw_prob)
        if len(self.prob_history) > self.smooth_window:
            self.prob_history.pop(0)
        smoothed_prob = float(np.mean(self.prob_history))

        # Dual-Threshold Hysteresis Smoothing (Schmitt Trigger)
        enter_thresh = max(0.65, self.confidence_threshold)
        exit_thresh = max(0.35, self.confidence_threshold - 0.20)

        if not self.is_fighting_state:
            if smoothed_prob >= enter_thresh:
                self.consec_high += 1
                if self.consec_high >= 3:
                    self.is_fighting_state = True
                    self.consec_low = 0
            else:
                self.consec_high = max(0, self.consec_high - 1)
        else:
            if smoothed_prob < exit_thresh:
                self.consec_low += 1
                if self.consec_low >= 4:
                    self.is_fighting_state = False
                    self.consec_high = 0
            else:
                self.consec_low = max(0, self.consec_low - 1)

        is_fighting_confirmed = self.is_fighting_state

        # Determine dominant physical action from detected persons
        person_actions = [p.action for p in persons if p.has_pose and p.action]
        fighting_actions = [a for a in person_actions if a in ["Punching / Striking", "Kicking", "Pushing / Grappling", "Aggressive Movement"]]
        if is_fighting_confirmed:
            dominant_action = fighting_actions[0] if fighting_actions else "Aggressive Striking"
        elif person_actions:
            dominant_action = person_actions[0]
        else:
            dominant_action = "Normal Movement"

        # Alert trigger logic with cooldown
        trigger_alert = False
        current_time = time.time()
        if is_fighting_confirmed and (current_time - self.last_alert_time >= self.alert_cooldown_sec):
            trigger_alert = True
            self.last_alert_time = current_time

        # Render minimal, unobtrusive visual HUD
        annotated_frame = self.render_hud(frame_bgr, persons, smoothed_prob, is_fighting_confirmed, dominant_action=dominant_action)

        return {
            "camera_id": camera_id,
            "is_fighting": is_fighting_confirmed,
            "fighting_confidence": round(smoothed_prob * 100, 1),
            "raw_confidence": round(raw_prob * 100, 1),
            "detected_action": dominant_action,
            "consecutive_frames": self.consec_high if not self.is_fighting_state else (4 - self.consec_low),
            "persons_detected": len(persons),
            "trigger_alert": trigger_alert,
            "annotated_frame": annotated_frame,
            "status_text": "FIGHTING DETECTED" if is_fighting_confirmed else "NORMAL"
        }

    def render_hud(
        self,
        frame_bgr: np.ndarray,
        persons: List[PersonPose],
        prob: float,
        is_fighting: bool,
        dominant_action: str = "Normal Movement"
    ) -> np.ndarray:
        """
        Draws skeletons and a sleek, compact, semi-transparent top-left HUD badge.
        Leaves the entire video clear, visible, and unobscured.
        """
        canvas = self.pose_detector.draw_skeleton(frame_bgr, persons, color_fighting=is_fighting)
        h, w = canvas.shape[:2]

        # Status text and accent color
        if is_fighting:
            status_text = f"FIGHTING DETECTED ({prob*100:.0f}%)"
            status_color = (40, 50, 255)    # Vivid Red/Coral
            border_color = (0, 0, 240)
        else:
            status_text = f"NORMAL ({prob*100:.0f}%)"
            status_color = (60, 220, 100)   # Vivid Emerald
            border_color = (0, 180, 80)

        action_text = f"Action: {dominant_action}"

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale_title = max(0.40, min(0.55, w / 1100))
        font_scale_sub = max(0.35, min(0.45, w / 1300))

        (t1_w, t1_h), _ = cv2.getTextSize(status_text, font, font_scale_title, 1)
        (t2_w, t2_h), _ = cv2.getTextSize(action_text, font, font_scale_sub, 1)

        badge_w = max(t1_w, t2_w) + 24
        badge_h = t1_h + t2_h + 18
        bx1, by1 = 12, 12
        bx2, by2 = bx1 + badge_w, by1 + badge_h

        # Semi-transparent sleek dark badge in top-left corner
        overlay = canvas.copy()
        cv2.rectangle(overlay, (bx1, by1), (bx2, by2), (15, 15, 15), -1)
        cv2.addWeighted(overlay, 0.70, canvas, 0.30, 0, canvas)
        cv2.rectangle(canvas, (bx1, by1), (bx2, by2), border_color, 1, cv2.LINE_AA)

        # Draw Status Line
        y_title = by1 + t1_h + 5
        cv2.putText(canvas, status_text, (bx1 + 10, y_title), font, font_scale_title, status_color, 1, cv2.LINE_AA)

        # Draw Action Line
        y_sub = y_title + t2_h + 6
        cv2.putText(canvas, action_text, (bx1 + 10, y_sub), font, font_scale_sub, (220, 220, 220), 1, cv2.LINE_AA)

        return canvas

    def analyze_video(
        self,
        video_path: str,
        output_video_path: Optional[str] = None,
        confidence_threshold: Optional[float] = None,
        sample_rate: int = 2,
        progress_callback = None
    ) -> Dict[str, Any]:
        """
        Processes an uploaded video file frame-by-frame:
        - Extracts MediaPipe poses & kinematic features with dedicated per-person trackers
        - Classifies physical actions (Punching, Kicking, Pushing, Walking, Standing)
        - Runs PyTorch BiLSTM temporal classifier
        - Applies Schmitt Trigger Hysteresis for smooth, flicker-free fighting detection
        - Consolidates fighting segments separated by brief pauses
        - Writes clean, browser-compatible faststart H.264 video with unobtrusive HUD
        """
        conf_thresh = confidence_threshold or self.confidence_threshold
        t0 = time.time()

        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found at: {video_path}")

        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration_sec = total_frames / fps if fps > 0 else 0.0

        # Output Video Writer setup with resolution capped to 640px for fast web streaming
        proc_w = min(width, 640)
        proc_h = int(height * (proc_w / width)) if width > 0 else height
        proc_w = proc_w - (proc_w % 2)
        proc_h = proc_h - (proc_h % 2)

        writer = None
        if output_video_path:
            os.makedirs(os.path.dirname(os.path.abspath(output_video_path)), exist_ok=True)
            writer = H264VideoWriter(output_video_path, max(1.0, fps / sample_rate), proc_w, proc_h)

        # Fresh state for new video
        self.pose_detector.reset()
        self.feature_extractor.reset()
        prob_history: List[float] = []
        
        # Hysteresis Schmitt Trigger state
        is_fighting_state = False
        consec_high = 0
        consec_low = 0
        enter_thresh = max(0.65, conf_thresh)
        exit_thresh = max(0.35, conf_thresh - 0.20)

        analyzed_frames = 0
        fighting_frames_count = 0
        timeline = []

        # Raw fighting segments tracking
        raw_segments = []
        current_segment = None

        frame_idx = 0
        while cap.isOpened():
            # Fast frame skipping using grab() without decoding compressed video stream
            if frame_idx % sample_rate != 0:
                ret = cap.grab()
                if not ret:
                    break
                frame_idx += 1
                continue

            ret, frame = cap.read()
            if not ret:
                break

            # Resize if needed for high efficiency
            if width > proc_w or height != proc_h:
                frame = cv2.resize(frame, (proc_w, proc_h), interpolation=cv2.INTER_AREA)

            analyzed_frames += 1
            curr_sec = frame_idx / fps

            persons = self.pose_detector.process_frame(frame)
            feat = self.feature_extractor.extract_frame_features(persons)
            seq_buffer = self.feature_extractor.update_buffer(feat)

            raw_p = 0.0
            if seq_buffer is not None and self.is_loaded:
                with torch.no_grad():
                    inp = torch.tensor(seq_buffer, dtype=torch.float32).unsqueeze(0).to(self.device)
                    pred, _ = self.model(inp)
                    raw_p = float(pred.item())

            # Smoothing
            prob_history.append(raw_p)
            if len(prob_history) > self.smooth_window:
                prob_history.pop(0)
            smooth_p = float(np.mean(prob_history))

            # Dual-Threshold Hysteresis Schmitt Trigger: eliminates detection flicker
            if not is_fighting_state:
                if smooth_p >= enter_thresh:
                    consec_high += 1
                    if consec_high >= 3:
                        is_fighting_state = True
                        consec_low = 0
                else:
                    consec_high = max(0, consec_high - 1)
            else:
                if smooth_p < exit_thresh:
                    consec_low += 1
                    if consec_low >= 4:
                        is_fighting_state = False
                        consec_high = 0
                else:
                    consec_low = max(0, consec_low - 1)

            is_fighting = is_fighting_state
            if is_fighting:
                fighting_frames_count += 1

            # Determine dominant physical action from detected persons in current frame
            person_actions = [p.action for p in persons if p.has_pose and p.action]
            fighting_actions = [a for a in person_actions if a in ["Punching / Striking", "Kicking", "Pushing / Grappling", "Aggressive Movement"]]
            if is_fighting:
                dominant_action = fighting_actions[0] if fighting_actions else "Physical Combat"
            elif person_actions:
                dominant_action = person_actions[0]
            else:
                dominant_action = "Normal Movement"

            # Build raw segments
            if is_fighting:
                if current_segment is None:
                    current_segment = {
                        "start_sec": round(curr_sec, 1),
                        "start_time": f"{int(curr_sec//60):02d}:{int(curr_sec%60):02d}",
                        "peak_conf": round(smooth_p * 100, 1),
                        "frame_count": 1,
                        "actions": [dominant_action]
                    }
                else:
                    current_segment["peak_conf"] = max(current_segment["peak_conf"], round(smooth_p * 100, 1))
                    current_segment["frame_count"] += 1
                    if dominant_action not in current_segment["actions"] and dominant_action != "Normal Movement":
                        current_segment["actions"].append(dominant_action)
            else:
                if current_segment is not None:
                    current_segment["end_sec"] = round(curr_sec, 1)
                    current_segment["end_time"] = f"{int(curr_sec//60):02d}:{int(curr_sec%60):02d}"
                    raw_segments.append(current_segment)
                    current_segment = None

            # Timeline data point
            if analyzed_frames % 5 == 0:
                timeline.append({
                    "time_sec": round(curr_sec, 1),
                    "timestamp": f"{int(curr_sec//60):02d}:{int(curr_sec%60):02d}",
                    "fighting_prob": round(smooth_p * 100, 1),
                    "is_fighting": is_fighting,
                    "action": dominant_action,
                    "persons": len(persons)
                })

            # Write annotated frame with clean, non-obtrusive HUD badge
            if writer is not None:
                annotated = self.render_hud(frame, persons, smooth_p, is_fighting, dominant_action=dominant_action)
                writer.write(annotated)

            if progress_callback and total_frames > 0:
                progress_pct = round((frame_idx / total_frames) * 100, 1)
                progress_callback(progress_pct)

            frame_idx += 1

        cap.release()
        if writer is not None:
            writer.release()

        # Close open segment at video end
        if current_segment is not None:
            current_segment["end_sec"] = round(duration_sec, 1)
            current_segment["end_time"] = f"{int(duration_sec//60):02d}:{int(duration_sec%60):02d}"
            raw_segments.append(current_segment)

        # Consolidate candidate segments: merge segments separated by brief pauses (< 1.5s)
        segments = self.consolidate_segments(raw_segments, max_gap_sec=1.5)

        # Video-level decision: requires sustained combat engagement with confirmed action segments
        fighting_pct = (fighting_frames_count / analyzed_frames * 100) if analyzed_frames > 0 else 0.0
        total_fighting_duration = sum(s["end_sec"] - s["start_sec"] for s in segments)
        final_is_fighting = (
            (len(segments) > 0 and fighting_pct >= 10.0 and total_fighting_duration >= 1.5) or
            (fighting_pct >= 20.0 and len(segments) > 0)
        )

        # Calculate overall confidence and primary detected actions
        overall_conf = 0.0
        if len(segments) > 0:
            overall_conf = float(np.mean([s["peak_conf"] for s in segments]))
        elif len(prob_history) > 0:
            overall_conf = float(np.max(prob_history) * 100)

        all_actions = set()
        for s in segments:
            for act in s.get("actions", []):
                all_actions.add(act)
        if final_is_fighting:
            detected_action_summary = ", ".join(all_actions) if all_actions else "Punching / Striking"
        else:
            detected_action_summary = "Normal Activity (Standing / Walking)"

        processing_time = round(time.time() - t0, 2)

        return {
            "final_result": "FIGHTING DETECTED" if final_is_fighting else "NORMAL",
            "is_fighting": final_is_fighting,
            "overall_confidence": round(overall_conf, 1),
            "detected_action": detected_action_summary,
            "analyzed_frames": analyzed_frames,
            "total_video_frames": total_frames,
            "fighting_frames": fighting_frames_count,
            "fighting_percentage": round(fighting_pct, 1),
            "duration_sec": round(duration_sec, 1),
            "processing_time_sec": processing_time,
            "segments": segments,
            "timeline": timeline,
            "output_video_path": output_video_path
        }


# Global singleton instance
global_fighting_detector = FightingDetector()
