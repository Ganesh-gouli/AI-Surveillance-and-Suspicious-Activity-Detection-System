import os
import sys
import io

# Ensure UTF-8 output on Windows console
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

import cv2
import numpy as np
import base64
from fastapi.testclient import TestClient

from backend.app.main import app
from detection.fighting_detector import FightingDetector

client = TestClient(app)

def run_tests():
    print("=" * 70)
    print("SENTINELVISION AI - COMPREHENSIVE FIGHTING DETECTION SYSTEM TEST")
    print("=" * 70)

    detector = FightingDetector()
    assert detector.is_loaded, "FightingDetector failed to load BiLSTM weights!"
    print("[PASS] BiLSTM Model Checkpoint loaded successfully into FightingDetector.")

    # -------------------------------------------------------------
    # TEST 1: Unseen Normal Video (Multiple People Walking & Sitting)
    # -------------------------------------------------------------
    normal_video_path = os.path.join("dataset", "no_fire", "normal_classroom_cctv.mp4")
    print(f"\n[TEST 1] Testing Unseen Normal Multi-Person Video: {os.path.basename(normal_video_path)}")
    print("Running frame-by-frame MediaPipe Pose + BiLSTM inference with temporal smoothing...")
    
    res_normal = detector.analyze_video(
        video_path=normal_video_path,
        output_video_path=None,
        confidence_threshold=0.70,
        sample_rate=4
    )
    print("--- TEST 1 RESULTS ---")
    print(f"Final Decision:     {res_normal['final_result']}")
    print(f"Is Fighting:        {res_normal['is_fighting']}")
    print(f"Analyzed Frames:    {res_normal['analyzed_frames']}")
    print(f"Fighting Frames:    {res_normal['fighting_frames']}")
    print(f"Fighting Segments:  {len(res_normal['segments'])}")
    print(f"Duration:           {res_normal['duration_sec']}s")
    
    # -------------------------------------------------------------
    # TEST 2: Unseen Fighting Video (Real Altercation & Striking)
    # -------------------------------------------------------------
    fight_video_path = os.path.join("FIGHTING", "Fighting030_x264A.mp4")
    output_annotated_path = os.path.join("results", "fighting_detected_video.mp4")
    print(f"\n[TEST 2] Testing Unseen Real Fighting Video: {os.path.basename(fight_video_path)}")
    print("Running frame-by-frame MediaPipe Pose + BiLSTM inference & generating annotated video...")

    res_fight = detector.analyze_video(
        video_path=fight_video_path,
        output_video_path=output_annotated_path,
        confidence_threshold=0.70,
        sample_rate=2
    )
    print("--- TEST 2 RESULTS ---")
    print(f"Final Decision:     {res_fight['final_result']}")
    print(f"Is Fighting:        {res_fight['is_fighting']}")
    print(f"Overall Confidence: {res_fight['overall_confidence']}%")
    print(f"Analyzed Frames:    {res_fight['analyzed_frames']}")
    print(f"Fighting Frames:    {res_fight['fighting_frames']}")
    print(f"Fighting Segments:  {len(res_fight['segments'])}")
    print(f"Annotated Output:   {output_annotated_path} (Exists: {os.path.exists(output_annotated_path)})")
    if len(res_fight['segments']) > 0:
        print("Detected Fighting Segments:")
        for i, s in enumerate(res_fight['segments']):
            print(f"  Segment {i+1}: {s['start_time']} - {s['end_time']} (Peak Conf: {s['peak_conf']}%)")

    assert res_fight['is_fighting'] == True, "Test 2 Failed: Fighting video should trigger FIGHTING DETECTED!"
    print("\n[PASS] Verified video-level fighting classification: FIGHTING DETECTED.")

    # -------------------------------------------------------------
    # TEST 3: FastAPI Endpoints Verification
    # -------------------------------------------------------------
    print("\n[TEST 3] Testing FastAPI Endpoints for Fighting Detection...")

    # 1. Config endpoint
    r_cfg = client.get("/api/fighting/config")
    print("1. /api/fighting/config status:", r_cfg.status_code, r_cfg.json())
    assert r_cfg.status_code == 200

    # 2. Live frame detection endpoint
    dummy_frame = np.zeros((240, 320, 3), dtype=np.uint8)
    _, buf = cv2.imencode(".jpg", dummy_frame)
    b64 = base64.b64encode(buf).decode("utf-8")

    r_frame = client.post("/api/fighting/detect-frame", json={
        "image_base64": b64,
        "camera_id": "TEST-CAM-01"
    })
    print("2. /api/fighting/detect-frame status:", r_frame.status_code, "is_fighting:", r_frame.json()["is_fighting"])
    assert r_frame.status_code == 200
    assert "annotated_frame" in r_frame.json()

    # 3. Events endpoint
    r_events = client.get("/api/fighting/events")
    print("3. /api/fighting/events status:", r_events.status_code, "events count:", len(r_events.json()["events"]))
    assert r_events.status_code == 200

    # 4. Stream video endpoint
    r_vid = client.get("/api/fighting/video/fighting_detected_video.mp4")
    print("4. /api/fighting/video stream status:", r_vid.status_code, "content-type:", r_vid.headers.get("content-type"))
    assert r_vid.status_code == 200

    print("\n" + "=" * 70)
    print("ALL AI FIGHTING DETECTION INTEGRATION TESTS PASSED SUCCESSFULLY!")
    print("=" * 70)

if __name__ == "__main__":
    run_tests()
