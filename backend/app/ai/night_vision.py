import cv2
import numpy as np
from typing import Dict, Any, Tuple, Optional
import base64

class NightVisionEnhancer:
    """
    High-Performance Low-Light & Tactical Night Vision Enhancement Engine.
    Provides LAB-space Adaptive CLAHE, dynamic gamma correction, digital sensor gain,
    ambient Lux estimation, and tactical vision color transforms (Tactical Green NVG,
    CCTV Infrared, Adaptive Low-Light HDR, and Thermal FLIR Heatmap).
    """

    def __init__(self):
        # Pre-configure CLAHE instances for low-latency processing
        self.clahe_standard = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        self.clahe_high_contrast = cv2.createCLAHE(clipLimit=4.5, tileGridSize=(8, 8))
        
        # Precompute gamma lookup tables for common gamma values (gamma < 1.0 brightens shadows)
        self._gamma_tables = {}
        for gamma_val in [0.35, 0.45, 0.55, 0.65, 0.75]:
            inv_gamma = gamma_val
            table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
            self._gamma_tables[gamma_val] = table

        # Sharpening kernel for tactical NVG edge definition
        self.sharpen_kernel = np.array([
            [0, -0.4, 0],
            [-0.4, 2.6, -0.4],
            [0, -0.4, 0]
        ], dtype=np.float32)

    def analyze_luminance(self, frame: np.ndarray) -> Dict[str, Any]:
        """
        Analyzes scene lighting conditions, mean luminance, RMS contrast,
        and estimates ambient Lux (illuminance) level.
        """
        if frame is None or frame.size == 0:
            return {
                "mean_luma": 0.0,
                "estimated_lux": 0.0,
                "is_low_light": True,
                "rms_contrast": 0.0,
                "dynamic_range": 0
            }

        # Convert to grayscale for fast 1-channel luma stats
        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame

        mean_luma = float(np.mean(gray))
        std_luma = float(np.std(gray))
        min_val, max_val, _, _ = cv2.minMaxLoc(gray)

        # Ambient lux estimation mapped to indoor/outdoor surveillance ranges
        # 0-15: Pitch dark / Night without lights
        # 15-40: Deep twilight / Dim low-light room
        # 40-100: Dim indoor lighting
        # 100-300: Standard indoor office
        # >300: Bright ambient light
        estimated_lux = round(max(0.1, (mean_luma / 255.0) ** 1.8 * 400.0), 1)
        is_low_light = mean_luma < 45.0

        return {
            "mean_luma": round(mean_luma, 1),
            "estimated_lux": estimated_lux,
            "is_low_light": is_low_light,
            "rms_contrast": round(std_luma, 1),
            "dynamic_range": int(max_val - min_val)
        }

    def _apply_gamma(self, img: np.ndarray, gamma: float = 0.55) -> np.ndarray:
        """Applies fast non-linear gamma lookup table to brighten midtones & shadows."""
        # Find closest precomputed gamma or compute on the fly
        closest_key = min(self._gamma_tables.keys(), key=lambda k: abs(k - gamma))
        if abs(closest_key - gamma) < 0.06:
            table = self._gamma_tables[closest_key]
        else:
            table = np.array([((i / 255.0) ** gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
        return cv2.LUT(img, table)

    def _apply_ir_illuminator(self, frame: np.ndarray, intensity: float = 0.35) -> np.ndarray:
        """
        Simulates an 850nm forward-facing infrared spotlight beam (radial vignette illumination)
        concentrated towards the center of the surveillance field of view.
        """
        h, w = frame.shape[:2]
        y, x = np.ogrid[:h, :w]
        cy, cx = h / 2.0, w / 2.0
        # Normalized distance from center
        dist_from_center = np.sqrt((x - cx) ** 2 + (y - cy) ** 2) / (np.sqrt(cx ** 2 + cy ** 2) * 0.95)
        # Cosine falloff mask
        mask = np.clip(1.0 - dist_from_center, 0.0, 1.0)
        mask = (mask ** 1.5) * intensity

        # Expand mask to 3 channels
        mask_3ch = np.dstack([mask] * 3).astype(np.float32)
        frame_float = frame.astype(np.float32)
        illuminated = frame_float + (255.0 - frame_float) * mask_3ch
        return np.clip(illuminated, 0, 255).astype(np.uint8)

    def enhance(
        self,
        frame: np.ndarray,
        mode: str = "tactical_green",
        gain: float = 1.0,
        ir_illuminator: bool = False
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Applies requested tactical night vision transformation and enhancement.
        
        Supported modes:
          - 'off': Standard untouched RGB frame.
          - 'tactical_green': Military Gen-3 NVG (phosphor green gradient, CLAHE, edge sharpening).
          - 'low_light': Super-starlight adaptive HDR color recovery (LAB CLAHE + gamma boost).
          - 'infrared': CCTV 850nm monochrome infrared with contrast stretching.
          - 'thermal': FLIR false-color thermal heat signature simulation (Ironbow/Inferno).
        """
        if frame is None or frame.size == 0:
            return frame, {"active": False, "mode": "off"}

        metrics = self.analyze_luminance(frame)
        mode = mode.lower().strip()

        if mode == "off":
            return frame, {
                "active": False,
                "mode": "off",
                "estimated_lux": metrics["estimated_lux"],
                "is_low_light": metrics["is_low_light"],
                "gain": 1.0
            }

        # Apply digital sensor gain multiplier (clamped to [1.0, 4.0])
        effective_gain = max(1.0, min(gain, 4.0))
        working_frame = frame
        if effective_gain > 1.0:
            working_frame = cv2.convertScaleAbs(working_frame, alpha=effective_gain, beta=0)

        # Apply virtual IR illuminator spotlight beam if enabled
        if ir_illuminator:
            working_frame = self._apply_ir_illuminator(working_frame, intensity=0.32)

        enhanced_frame = working_frame

        if mode == "tactical_green":
            # 1. Monochromatic Luma extraction
            gray = cv2.cvtColor(working_frame, cv2.COLOR_BGR2GRAY)
            # 2. Adaptive CLAHE on luminance
            enhanced_gray = self.clahe_high_contrast.apply(gray)
            # 3. Dynamic Gamma correction
            enhanced_gray = self._apply_gamma(enhanced_gray, gamma=0.50)
            # 4. Edge sharpening for night optics crispness
            sharpened = cv2.filter2D(enhanced_gray, -1, self.sharpen_kernel)
            sharpened = cv2.addWeighted(enhanced_gray, 0.6, sharpened, 0.4, 0)
            
            # 5. Tactical PVS-14 Green Phosphor Palette Mapping:
            # Blue: low (~15-20%), Green: peak (~100-115%), Red: subtle amber (~10%)
            g_chan = cv2.convertScaleAbs(sharpened, alpha=1.12, beta=12)
            r_chan = cv2.convertScaleAbs(sharpened, alpha=0.15, beta=0)
            b_chan = cv2.convertScaleAbs(sharpened, alpha=0.22, beta=0)
            enhanced_frame = cv2.merge([b_chan, g_chan, r_chan])

        elif mode == "low_light":
            # Adaptive Full-Color Low-Light HDR (Starlight Sensor Simulation)
            # Convert to LAB color space to isolate Luminance without color degradation
            lab = cv2.cvtColor(working_frame, cv2.COLOR_BGR2LAB)
            l_chan, a_chan, b_chan = cv2.split(lab)
            
            # CLAHE on L channel
            enhanced_l = self.clahe_standard.apply(l_chan)
            # Gamma boost
            enhanced_l = self._apply_gamma(enhanced_l, gamma=0.55)
            
            # Merge back and convert to BGR
            enhanced_lab = cv2.merge([enhanced_l, a_chan, b_chan])
            enhanced_bgr = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
            
            # Slight color saturation restore
            hsv = cv2.cvtColor(enhanced_bgr, cv2.COLOR_BGR2HSV)
            h, s, v = cv2.split(hsv)
            s = cv2.convertScaleAbs(s, alpha=1.25, beta=5)
            enhanced_frame = cv2.cvtColor(cv2.merge([h, s, v]), cv2.COLOR_HSV2BGR)

        elif mode == "infrared":
            # CCTV 850nm / 940nm Monochromatic Infrared
            gray = cv2.cvtColor(working_frame, cv2.COLOR_BGR2GRAY)
            # Heavy contrast stretching & CLAHE
            ir_enhanced = self.clahe_high_contrast.apply(gray)
            ir_enhanced = self._apply_gamma(ir_enhanced, gamma=0.48)
            enhanced_frame = cv2.cvtColor(ir_enhanced, cv2.COLOR_GRAY2BGR)

        elif mode == "thermal":
            # FLIR Thermal Heat Signature Simulation (Ironbow / Inferno Colormap)
            gray = cv2.cvtColor(working_frame, cv2.COLOR_BGR2GRAY)
            clahe_gray = self.clahe_standard.apply(gray)
            # Normalize luma to full range [0, 255]
            normalized = cv2.normalize(clahe_gray, None, 0, 255, cv2.NORM_MINMAX)
            enhanced_frame = cv2.applyColorMap(normalized, cv2.COLORMAP_INFERNO)

        else:
            # Fallback to standard frame
            enhanced_frame = working_frame

        # Sensor Gain in dB
        gain_db = round(20.0 * np.log10(effective_gain), 1) if effective_gain > 1.0 else 0.0

        telemetry = {
            "active": True,
            "mode": mode,
            "estimated_lux": metrics["estimated_lux"],
            "is_low_light": metrics["is_low_light"],
            "mean_luma": metrics["mean_luma"],
            "sensor_gain": round(effective_gain, 2),
            "sensor_gain_db": gain_db,
            "ir_illuminator_active": ir_illuminator,
            "clahe_applied": True
        }

        return enhanced_frame, telemetry

    def enhance_for_ai(
        self,
        frame: np.ndarray,
        mode: str = "off",
        gain: float = 1.0,
        ir_illuminator: bool = False
    ) -> np.ndarray:
        """
        Preprocesses and normalizes low-light webcam frames specifically for YOLOv11
        and MediaPipe detection models.
        
        Even if tactical green or thermal color modes are active for human viewing,
        the AI model performs best when shadows are lifted and contrast is balanced
        in standard 3-channel color or high-contrast illuminated representation.
        """
        if frame is None or frame.size == 0:
            return frame

        metrics = self.analyze_luminance(frame)

        # If scene is already bright and mode is off, keep frame as-is
        if mode == "off" and not metrics["is_low_light"] and gain <= 1.0 and not ir_illuminator:
            return frame

        # Apply digital sensor gain and IR illuminator if active
        working = frame
        effective_gain = max(1.0, min(gain, 4.0))
        if effective_gain > 1.0:
            working = cv2.convertScaleAbs(working, alpha=effective_gain, beta=0)
        if ir_illuminator:
            working = self._apply_ir_illuminator(working, intensity=0.32)

        # For low light or night vision mode, apply LAB CLAHE + Gamma on Luminance channel
        # This keeps RGB color distribution intact for YOLO's learned color weights while
        # boosting dark silhouettes and hidden objects out of pitch black shadows.
        lab = cv2.cvtColor(working, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        enhanced_l = self.clahe_standard.apply(l)
        enhanced_l = self._apply_gamma(enhanced_l, gamma=0.52)
        enhanced_lab = cv2.merge([enhanced_l, a, b])
        ai_frame = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)

        return ai_frame

    def encode_frame_to_base64(self, frame: np.ndarray, quality: int = 75) -> str:
        """Encodes frame to JPEG base64 string for fast API transport."""
        success, encoded_img = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
        if not success:
            return ""
        return "data:image/jpeg;base64," + base64.b64encode(encoded_img).decode("utf-8")

# Global singleton enhancer instance
global_night_vision = NightVisionEnhancer()
