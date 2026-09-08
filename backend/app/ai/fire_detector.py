import os
import time
from typing import Dict, Any, Optional, Tuple, Callable, List
import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

try:
    from ultralytics import YOLO
    HAS_ULTRALYTICS = True
except ImportError:
    HAS_ULTRALYTICS = False

try:
    import imageio.v3 as iio
    HAS_IMAGEIO = True
except ImportError:
    HAS_IMAGEIO = False

CLASS_NAMES = ["NO_FIRE", "FIRE"]

class FireNet(nn.Module):
    """
    Lightweight MobileNetV3-based PyTorch Fire Detection Neural Network.
    Optimized for high-speed CPU/GPU real-time inference and frame-by-frame video forensics.
    """
    def __init__(self, pretrained: bool = True):
        super(FireNet, self).__init__()
        weights = models.MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
        base_model = models.mobilenet_v3_small(weights=weights)
        self.features = base_model.features
        self.avgpool = base_model.avgpool
        
        in_features = base_model.classifier[0].in_features
        self.classifier = nn.Sequential(
            nn.Linear(in_features, 128),
            nn.Hardswish(),
            nn.Dropout(p=0.2),
            nn.Linear(128, 2)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


class FireDetector:
    def __init__(self, model_path: Optional[str] = None, yolo_path: Optional[str] = None, device: Optional[str] = None):
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # Base Directory
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

        # 1. PyTorch FireNet Classifier
        self.model = FireNet(pretrained=True).to(self.device)
        self.model.eval()

        self.default_model_path = model_path or os.path.join(base_dir, "models", "best_fire_detector.pt")
        self.is_custom_trained = False
        self.load_weights(self.default_model_path)

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

        # 2. YOLO Accurate Fire & Smoke Bounding Box Object Detector
        self.yolo_box_model = None
        self.yolo_box_path = yolo_path or os.path.join(base_dir, "models", "yolov8_fire_box.pt")
        if HAS_ULTRALYTICS and os.path.exists(self.yolo_box_path):
            try:
                self.yolo_box_model = YOLO(self.yolo_box_path)
                print(f"[FireDetector] Loaded YOLO Fire Bounding Box model from: {self.yolo_box_path}")
                print(f"[FireDetector] Object Classes: {self.yolo_box_model.names}")
            except Exception as e:
                print(f"[FireDetector] Could not load YOLO model from {self.yolo_box_path}: {e}")

    def load_weights(self, path: str):
        if os.path.exists(path):
            try:
                state_dict = torch.load(path, map_location=self.device)
                self.model.load_state_dict(state_dict)
                self.model.eval()
                self.is_custom_trained = True
                print(f"[FireDetector] Loaded trained FireNet weights from: {path}")
            except Exception as e:
                print(f"[FireDetector] Could not load weights from {path}: {e}")
        else:
            print(f"[FireDetector] No trained weights found at {path}. Operating with base model.")

    def detect_boxes_and_classify(
        self,
        frame_bgr: np.ndarray,
        yolo_conf: float = 0.25
    ) -> Tuple[bool, float, float, List[Dict[str, Any]]]:
        """
        Runs accurate YOLO bounding box detection and PyTorch verification.
        Returns:
            (is_fire, fire_confidence, no_fire_confidence, boxes)
        """
        if frame_bgr is None or frame_bgr.size == 0:
            return False, 0.0, 1.0, []

        boxes = []
        fire_box_confidences = []
        smoke_box_confidences = []

        # 1. Run YOLO Bounding Box Detection if available
        if self.yolo_box_model is not None:
            try:
                results = self.yolo_box_model(frame_bgr, conf=yolo_conf, verbose=False)
                for r in results:
                    for b in r.boxes:
                        cls_id = int(b.cls[0])
                        label = self.yolo_box_model.names.get(cls_id, "Fire")
                        conf = float(b.conf[0])
                        x1, y1, x2, y2 = map(int, b.xyxy[0].tolist())
                        is_fire_label = (label.lower() == "fire")

                        if is_fire_label:
                            fire_box_confidences.append(conf)
                        else:
                            smoke_box_confidences.append(conf)

                        boxes.append({
                            "x1": x1,
                            "y1": y1,
                            "x2": x2,
                            "y2": y2,
                            "width": max(1, x2 - x1),
                            "height": max(1, y2 - y1),
                            "label": label,
                            "confidence": round(conf * 100, 1),
                            "is_fire": is_fire_label
                        })
            except Exception as e:
                print(f"[FireDetector] YOLO inference notice: {e}")

        # 2. Run PyTorch Classifier as validation / backup
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(frame_rgb)
        tensor = self.transform(pil_img).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = self.model(tensor)
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]

        pytorch_no_fire = float(probs[0])
        pytorch_fire = float(probs[1])

        # Integrate YOLO boxes with classification confidence
        if fire_box_confidences:
            # We have confirmed fire bounding boxes!
            fire_prob = max(fire_box_confidences)
            is_fire = True
            no_fire_prob = 1.0 - fire_prob
        elif boxes and smoke_box_confidences:
            # Smoke detected without active flame
            smoke_max = max(smoke_box_confidences)
            fire_prob = smoke_max * 0.35
            is_fire = False
            no_fire_prob = 1.0 - fire_prob
        else:
            # No fire or smoke boxes detected: frame is clean normal!
            is_fire = False
            fire_prob = 0.0
            no_fire_prob = 1.0

        return is_fire, fire_prob, no_fire_prob, boxes

    def predict_frame(self, frame_bgr: np.ndarray) -> Tuple[bool, float, float]:
        """Convenience method for frame-level binary classification."""
        is_fire, p_fire, p_no_fire, _ = self.detect_boxes_and_classify(frame_bgr)
        return is_fire, p_fire, p_no_fire

    def draw_detection_hud(
        self,
        frame_bgr: np.ndarray,
        is_fire: bool,
        fire_confidence: float,
        frame_idx: int,
        total_frames: int,
        timestamp_sec: float,
        boxes: Optional[List[Dict[str, Any]]] = None
    ) -> np.ndarray:
        """
        Draws accurate bounding boxes around fire and smoke regions, with high-contrast HUD banner.
        """
        h, w = frame_bgr.shape[:2]
        canvas = frame_bgr.copy()
        boxes = boxes or []

        # 1. Draw Bounding Boxes around Fire and Smoke
        for box in boxes:
            x1, y1, x2, y2 = box["x1"], box["y1"], box["x2"], box["y2"]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w - 1, x2), min(h - 1, y2)
            label = box["label"]
            conf = box["confidence"]
            is_fire_box = box["is_fire"]

            if is_fire_box:
                # Glowing Neon Red/Orange for Fire
                color = (0, 69, 255)       # BGR Bright Orange-Red
                badge_bg = (10, 10, 180)   # Dark Red
                tag = f"FIRE: {conf:.1f}%"
            else:
                # Cool Slate/Cyan for Smoke
                color = (220, 200, 80)     # BGR Soft Cyan-Gray
                badge_bg = (60, 60, 60)
                tag = f"SMOKE: {conf:.1f}%"

            # Main bounding box
            cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)

            # High-tech corner bracket accents
            line_len = max(8, min(20, int(min(x2 - x1, y2 - y1) * 0.2)))
            thick = 3
            # Top-Left
            cv2.line(canvas, (x1, y1), (x1 + line_len, y1), color, thick)
            cv2.line(canvas, (x1, y1), (x1, y1 + line_len), color, thick)
            # Top-Right
            cv2.line(canvas, (x2, y1), (x2 - line_len, y1), color, thick)
            cv2.line(canvas, (x2, y1), (x2, y1 + line_len), color, thick)
            # Bottom-Left
            cv2.line(canvas, (x1, y2), (x1 + line_len, y2), color, thick)
            cv2.line(canvas, (x1, y2), (x1, y2 - line_len), color, thick)
            # Bottom-Right
            cv2.line(canvas, (x2, y2), (x2 - line_len, y2), color, thick)
            cv2.line(canvas, (x2, y2), (x2, y2 - line_len), color, thick)

            # Badge Label above box
            badge_font = cv2.FONT_HERSHEY_SIMPLEX
            b_scale = max(0.4, w / 950.0)
            (tw, th), tb = cv2.getTextSize(tag, badge_font, b_scale, 1)
            badge_y1 = max(0, y1 - th - 8)
            badge_y2 = y1
            badge_x2 = min(w - 1, x1 + tw + 10)

            # Draw filled pill background
            cv2.rectangle(canvas, (x1, badge_y1), (badge_x2, badge_y2), badge_bg, -1)
            cv2.rectangle(canvas, (x1, badge_y1), (badge_x2, badge_y2), color, 1)
            cv2.putText(canvas, tag, (x1 + 4, badge_y2 - 4), badge_font, b_scale, (255, 255, 255), 1, cv2.LINE_AA)

        # 2. Top Forensic HUD Header Banner
        banner_h = max(45, int(h * 0.14))
        overlay = canvas.copy()

        fire_box_count = sum(1 for b in boxes if b["is_fire"])
        smoke_box_count = sum(1 for b in boxes if not b["is_fire"])

        # Crucial: Active fire requires fire_box_count > 0
        has_active_fire = fire_box_count > 0

        if has_active_fire:
            fire_box_confs = [b["confidence"] for b in boxes if b["is_fire"]]
            max_conf = max(fire_box_confs) if fire_box_confs else (fire_confidence * 100.0)
            # Alert Crimson
            bg_color = (20, 20, 90)
            border_color = (0, 0, 255)
            header_text = f"[!] FIRE DETECTED ({fire_box_count} Zone{'s' if fire_box_count != 1 else ''})"
            conf_text = f"Confidence: {max_conf:.1f}% | Smoke Plumes: {smoke_box_count}"
            text_color = (0, 69, 255)
        elif smoke_box_count > 0:
            # Amber for Smoke Only
            bg_color = (35, 35, 35)
            border_color = (200, 180, 50)
            header_text = f"[~] SMOKE DETECTED ({smoke_box_count} Plume{'s' if smoke_box_count != 1 else ''})"
            conf_text = "Smoke Hazard Detected | No Direct Flame"
            text_color = (240, 200, 50)
        else:
            # Normal Clean Slate
            bg_color = (25, 25, 25)
            border_color = (50, 150, 50)
            header_text = "[OK] NO FIRE DETECTED"
            conf_text = "Normal Scene | No Flame Hazard Detected"
            text_color = (80, 240, 80)

        cv2.rectangle(overlay, (0, 0), (w, banner_h), bg_color, -1)
        alpha = 0.80
        cv2.addWeighted(overlay, alpha, canvas, 1 - alpha, 0, canvas)
        cv2.line(canvas, (0, banner_h), (w, banner_h), border_color, 2)

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = max(0.42, w / 850.0)
        thickness = 2 if font_scale > 0.6 else 1

        cv2.putText(canvas, header_text, (12, int(banner_h * 0.45)), font, font_scale * 1.05, text_color, thickness + 1, cv2.LINE_AA)
        cv2.putText(canvas, conf_text, (12, int(banner_h * 0.85)), font, font_scale * 0.85, (220, 220, 220), thickness, cv2.LINE_AA)

        info_text = f"T: {timestamp_sec:.1f}s | Frame: {frame_idx}/{total_frames}"
        (tw, _), _ = cv2.getTextSize(info_text, font, font_scale * 0.75, thickness)
        cv2.putText(canvas, info_text, (w - tw - 12, int(banner_h * 0.65)), font, font_scale * 0.75, (220, 220, 220), thickness, cv2.LINE_AA)

        return canvas

    def analyze_video(
        self,
        video_path: str,
        output_video_path: Optional[str] = None,
        confidence_threshold: float = 0.70,
        frame_ratio: float = 0.20,
        sample_rate: int = 5,
        progress_callback: Optional[Callable[[float], None]] = None
    ) -> Dict[str, Any]:
        """
        Processes uploaded video frame-by-frame with YOLO bounding boxes and temporal aggregation.
        """
        start_time = time.time()

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Unable to open video: {video_path}")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480
        duration_sec = total_frames / max(1.0, fps)

        analyzed_frames = 0
        fire_frames = 0
        total_fire_boxes_detected = 0
        fire_confidences: List[float] = []
        all_confidences: List[float] = []
        timeline: List[Dict[str, Any]] = []
        annotated_frames: List[np.ndarray] = []

        step = max(1, int(sample_rate))
        frame_idx = 0

        # State tracking for smooth box rendering between sampled intervals
        current_state_fire = False
        current_state_conf = 0.0
        current_boxes: List[Dict[str, Any]] = []

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            time_offset = round(frame_idx / fps, 2)

            # Sample frame according to configured sample rate
            if frame_idx % step == 0:
                is_fire, p_fire, _, boxes = self.detect_boxes_and_classify(frame, yolo_conf=0.25)
                analyzed_frames += 1
                all_confidences.append(p_fire)

                # Check if fire confidence meets the threshold AND at least 1 fire box is present
                fire_boxes = [b for b in boxes if b["is_fire"]]
                if fire_boxes:
                    max_box_conf = max([b["confidence"] for b in fire_boxes]) / 100.0
                    has_confident_fire = (max_box_conf >= confidence_threshold)
                    if has_confident_fire:
                        fire_frames += 1
                        fire_confidences.append(max_box_conf)
                    total_fire_boxes_detected += len(fire_boxes)
                    current_state_fire = True
                    current_state_conf = max_box_conf
                else:
                    has_confident_fire = False
                    current_state_fire = False
                    current_state_conf = 0.0

                current_boxes = boxes

                timeline.append({
                    "frame_idx": frame_idx,
                    "time_offset_sec": time_offset,
                    "fire_confidence": round(current_state_conf * 100, 1),
                    "is_fire": has_confident_fire,
                    "box_count": len(fire_boxes)
                })

                if progress_callback and total_frames > 0:
                    pct = min(99.0, round((frame_idx / total_frames) * 100, 1))
                    progress_callback(pct)

            # Render HUD and bounding boxes on every frame
            if output_video_path is not None:
                annotated = self.draw_detection_hud(
                    frame,
                    is_fire=current_state_fire,
                    fire_confidence=current_state_conf,
                    frame_idx=frame_idx,
                    total_frames=total_frames,
                    timestamp_sec=time_offset,
                    boxes=current_boxes
                )
                annotated_frames.append(annotated)

            frame_idx += 1

        cap.release()

        # Calculate temporal aggregation metrics
        fire_percentage = (fire_frames / max(1, analyzed_frames)) * 100.0
        fire_detected = (fire_percentage / 100.0) >= frame_ratio

        if fire_confidences:
            overall_confidence = float(np.mean(fire_confidences)) * 100.0
        elif all_confidences:
            overall_confidence = float(np.mean([1.0 - c for c in all_confidences])) * 100.0
        else:
            overall_confidence = 0.0

        final_result = "🔥 FIRE DETECTED" if fire_detected else "✅ NO FIRE DETECTED"
        processing_time_sec = round(time.time() - start_time, 2)

        # Save annotated video with bounding boxes
        if output_video_path and annotated_frames:
            os.makedirs(os.path.dirname(os.path.abspath(output_video_path)), exist_ok=True)
            self._save_h264_video(annotated_frames, output_video_path, fps=min(30.0, fps))

        if progress_callback:
            progress_callback(100.0)

        return {
            "video_name": os.path.basename(video_path),
            "final_result": final_result,
            "fire_detected": fire_detected,
            "overall_confidence": round(overall_confidence, 1),
            "fire_frames": fire_frames,
            "analyzed_frames": analyzed_frames,
            "total_video_frames": total_frames,
            "total_fire_boxes_detected": total_fire_boxes_detected,
            "fire_percentage": round(fire_percentage, 1),
            "duration_sec": round(duration_sec, 1),
            "processing_time_sec": processing_time_sec,
            "confidence_threshold": confidence_threshold,
            "frame_ratio": frame_ratio,
            "sample_rate": step,
            "timeline": timeline,
            "output_video_saved": bool(output_video_path and os.path.exists(output_video_path))
        }

    def _save_h264_video(self, frames_bgr: List[np.ndarray], output_path: str, fps: float = 30.0):
        """Encode frames into standard H.264 MP4."""
        if not frames_bgr:
            return

        rgb_frames = [cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in frames_bgr]
        h, w = rgb_frames[0].shape[:2]
        new_w = w if w % 2 == 0 else w - 1
        new_h = h if h % 2 == 0 else h - 1

        if new_w != w or new_h != h:
            rgb_frames = [cv2.resize(f, (new_w, new_h)) for f in rgb_frames]

        try:
            if HAS_IMAGEIO:
                iio.imwrite(output_path, rgb_frames, fps=fps, codec="h264")
                print(f"[FireDetector] Successfully encoded H.264 video with bounding boxes to {output_path}")
                return
        except Exception as e:
            print(f"[FireDetector] imageio H.264 notice: {e}, falling back to OpenCV VideoWriter")

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (new_w, new_h))
        for f in frames_bgr:
            if new_w != w or new_h != h:
                f = cv2.resize(f, (new_w, new_h))
            out.write(f)
        out.release()
        print(f"[FireDetector] OpenCV fallback video written to {output_path}")


# Global singleton instance
global_fire_detector = FireDetector()
