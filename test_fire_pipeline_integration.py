import os
import sys
import io
import cv2
import numpy as np

if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

from backend.app.ai.inference_pipeline import SentinelInferencePipeline

def run_tests():
    print("=" * 65)
    print("TESTING FULL FIRE DETECTION INTEGRATION IN SENTINEL INFERENCE PIPELINE")
    print("=" * 65)

    pipeline = SentinelInferencePipeline()

    # TEST 1: Fire video frame
    fire_video_path = os.path.join("dataset", "fire", "Explosion005_x264A.mp4")
    assert os.path.exists(fire_video_path), f"File missing: {fire_video_path}"
    cap = cv2.VideoCapture(fire_video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 300)
    ret, fire_frame = cap.read()
    cap.release()
    assert ret, "Could not read fire frame"

    print(f"\n[TEST 1] Processing Fire Frame (300) from {os.path.basename(fire_video_path)}...")
    all_alerts = []
    for i in range(4):
        res1 = pipeline.process_frame(fire_frame, camera_id="TEST-CAM-FIRE")
        if res1["active_alerts"]:
            all_alerts.extend(res1["active_alerts"])

    fire_info = res1["fire"]
    print("Fire Output:", fire_info)
    assert fire_info["is_fire"] is True, f"Expected is_fire=True, got {fire_info['is_fire']}"
    assert fire_info["fire_confidence"] >= 60.0, f"Expected fire_confidence >= 60, got {fire_info['fire_confidence']}"
    assert len(fire_info["boxes"]) > 0, "Expected at least 1 fire bounding box"
    assert fire_info["continuous_fire_alert"] is True, "Expected continuous_fire_alert=True after 3 frames"
    print("  -> Fire box:", fire_info["boxes"][0])
    print("  -> Continuous Alert:", fire_info["continuous_fire_alert"])
    print("  -> Total Alerts Generated across sequence:", len(all_alerts))
    fire_alerts = [a for a in all_alerts if a.get("threat_type") == "FIRE_HAZARD"]
    assert len(fire_alerts) > 0, "Expected FIRE_HAZARD alert in active_alerts"
    print("  -> Verified Fire Hazard Alert:", fire_alerts[0]["threat_type"], fire_alerts[0]["severity"])
    print("[TEST 1 PASSED] Fire accurately detected with glowing bounding boxes & alerts!")

    # TEST 2: Normal CCTV video frame
    normal_video_path = os.path.join("dataset", "no_fire", "normal_pedestrian_cctv.mp4")
    assert os.path.exists(normal_video_path), f"File missing: {normal_video_path}"
    cap = cv2.VideoCapture(normal_video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 60)
    ret, normal_frame = cap.read()
    cap.release()
    assert ret, "Could not read normal frame"

    print(f"\n[TEST 2] Processing Normal CCTV Frame from {os.path.basename(normal_video_path)}...")
    pipeline_normal = SentinelInferencePipeline()
    res2 = pipeline_normal.process_frame(normal_frame, camera_id="TEST-CAM-NORMAL")

    fire_info_norm = res2["fire"]
    print("Normal Fire Output:", fire_info_norm)
    assert fire_info_norm["is_fire"] is False, f"False positive detected: {fire_info_norm}"
    assert fire_info_norm["fire_confidence"] == 0.0, f"Expected 0.0 confidence on normal, got {fire_info_norm['fire_confidence']}"
    assert len(fire_info_norm["boxes"]) == 0, f"Expected 0 boxes, got {fire_info_norm['boxes']}"
    print("[TEST 2 PASSED] Zero false positives on clean surveillance scene!")

    # TEST 3: Webcam API endpoint simulation
    from fastapi.testclient import TestClient
    from backend.app.main import app
    import base64

    client = TestClient(app)
    _, buf = cv2.imencode(".jpg", fire_frame)
    b64 = base64.b64encode(buf).decode('utf-8')

    print("\n[TEST 3] Testing /api/detection/frame with Base64 live webcam payload...")
    for _ in range(3):
        r = client.post("/api/detection/frame", json={
            "image_base64": b64,
            "camera_id": "TEST-WEBCAM-LIVE",
            "accuracy_mode": "high"
        })
    assert r.status_code == 200, f"API returned status {r.status_code}"
    api_res = r.json()
    assert "fire" in api_res, "Response missing 'fire' key"
    assert api_res["fire"]["is_fire"] is True, "API did not detect fire"
    print("API 'fire' telemetry:", api_res["fire"])
    print("[TEST 3 PASSED] Live Webcam API payload successfully processes and returns fire hazard telemetry!")

    print("\n" + "=" * 65)
    print("ALL FIRE DETECTION INTEGRATION TESTS PASSED 100% SUCCESSFULLY!")
    print("=" * 65)

if __name__ == "__main__":
    run_tests()
