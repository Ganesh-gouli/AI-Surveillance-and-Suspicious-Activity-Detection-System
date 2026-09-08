import pytest
import numpy as np
import cv2
from backend.app.ai.night_vision import NightVisionEnhancer, global_night_vision
from backend.app.ai.inference_pipeline import global_pipeline

def test_analyze_luminance_dark_and_bright():
    enhancer = NightVisionEnhancer()
    
    # 1. Dark frame (simulate pitch dark room / night)
    dark_frame = np.zeros((240, 320, 3), dtype=np.uint8)
    dark_frame[50:150, 100:200] = 15 # very faint illumination
    dark_metrics = enhancer.analyze_luminance(dark_frame)
    
    assert dark_metrics["is_low_light"] is True
    assert dark_metrics["mean_luma"] < 25.0
    assert dark_metrics["estimated_lux"] < 20.0

    # 2. Bright frame (daylight / well-lit indoor)
    bright_frame = np.full((240, 320, 3), 180, dtype=np.uint8)
    bright_metrics = enhancer.analyze_luminance(bright_frame)
    
    assert bright_metrics["is_low_light"] is False
    assert bright_metrics["mean_luma"] > 150.0
    assert bright_metrics["estimated_lux"] > 100.0

def test_night_vision_all_modes():
    enhancer = NightVisionEnhancer()
    test_frame = np.random.randint(10, 60, (240, 320, 3), dtype=np.uint8)
    
    modes = ["tactical_green", "low_light", "infrared", "thermal"]
    for mode in modes:
        enhanced, telemetry = enhancer.enhance(test_frame, mode=mode, gain=1.5, ir_illuminator=True)
        assert enhanced is not None
        assert enhanced.shape == test_frame.shape
        assert enhanced.dtype == np.uint8
        assert telemetry["active"] is True
        assert telemetry["mode"] == mode
        assert telemetry["ir_illuminator_active"] is True
        assert telemetry["sensor_gain"] == 1.5

def test_night_vision_gain_and_ir():
    enhancer = NightVisionEnhancer()
    test_frame = np.full((200, 200, 3), 20, dtype=np.uint8)
    
    enhanced_gain, tel_gain = enhancer.enhance(test_frame, mode="low_light", gain=2.5)
    assert np.mean(enhanced_gain) > np.mean(test_frame)
    assert tel_gain["sensor_gain_db"] > 5.0

    # IR Illuminator should make center brighter than perimeter
    enhanced_ir, _ = enhancer.enhance(test_frame, mode="tactical_green", ir_illuminator=True)
    center_val = np.mean(enhanced_ir[80:120, 80:120])
    corner_val = np.mean(enhanced_ir[0:40, 0:40])
    assert center_val >= corner_val

def test_enhance_for_ai_detection():
    enhancer = NightVisionEnhancer()
    dark_frame = np.full((240, 320, 3), 25, dtype=np.uint8)
    
    ai_frame = enhancer.enhance_for_ai(dark_frame, mode="tactical_green", gain=2.0, ir_illuminator=True)
    assert ai_frame is not None
    assert ai_frame.shape == dark_frame.shape
    # Should have higher contrast and mean than raw dark frame
    assert np.mean(ai_frame) > np.mean(dark_frame)

def test_pipeline_integration_with_night_vision():
    test_frame = np.zeros((240, 320, 3), dtype=np.uint8)
    cv2.circle(test_frame, (160, 120), 40, (80, 80, 80), -1)

    result = global_pipeline.process_frame(
        test_frame,
        camera_id="TEST-NIGHT-CAM",
        night_vision_mode="tactical_green",
        night_vision_gain=1.8,
        ir_illuminator=True
    )

    assert "night_vision" in result
    nv = result["night_vision"]
    assert nv["active"] is True
    assert nv["mode"] == "tactical_green"
    assert nv["sensor_gain"] == 1.8
    assert nv["ir_illuminator"] is True
    assert "estimated_lux" in nv
    assert "people" in result
    assert "fps" in result
