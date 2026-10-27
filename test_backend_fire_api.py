import sys
import io

if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

import os
import base64
import cv2
import numpy as np
from fastapi.testclient import TestClient
from backend.app.main import app

client = TestClient(app)

def test_api():
    print("=" * 65)
    print("TESTING FASTAPI FIRE DETECTION ENDPOINTS & IMAGE-BASED INFERENCE")
    print("=" * 65)

    base_dir = os.path.dirname(os.path.abspath(__file__))

    # 1. Config endpoint
    r = client.get("/api/fire/config")
    print("\n1. /api/fire/config status:", r.status_code, r.json())
    assert r.status_code == 200

    # 2. Image-Based Fire Detection: Fireplace Image (User's real fireplace)
    fp_path = os.path.join(base_dir, "results", "sample_fireplace.png")
    assert os.path.exists(fp_path), f"Sample fireplace missing at {fp_path}"
    with open(fp_path, "rb") as f:
        r_fire = client.post(
            "/api/fire/analyze-image",
            files={"file": ("sample_fireplace.png", f, "image/png")},
            data={"confidence_threshold": "0.60"}
        )
    print("\n2. /api/fire/analyze-image (Fireplace):", r_fire.status_code)
    assert r_fire.status_code == 200, f"Error: {r_fire.text}"
    fire_json = r_fire.json()
    print("   Status:", fire_json["status"])
    print(f"   Confidence: {fire_json['confidence']}%")
    print(f"   Boxes detected: {fire_json['boxes_count']}")
    assert fire_json["is_fire"] is True, f"Expected is_fire=True, got {fire_json['is_fire']}"
    assert fire_json["status"] == "FIRE DETECTED", f"Expected 'FIRE DETECTED', got {fire_json['status']}"
    assert fire_json["confidence"] >= 60.0, f"Expected confidence >= 60%, got {fire_json['confidence']}"
    assert fire_json["boxes_count"] > 0, "Expected at least 1 flame bounding box"
    assert "annotated_image" in fire_json and fire_json["annotated_image"].startswith("data:image/jpeg;base64,")
    print("   [PASSED] Fireplace correctly detected with accurate bounding boxes & confidence!")

    # 3. Image-Based Fire Detection: Normal Surveillance Room (No Fire)
    norm_path = os.path.join(base_dir, "results", "sample_normal_room.jpg")
    assert os.path.exists(norm_path), f"Sample normal room missing at {norm_path}"
    with open(norm_path, "rb") as f:
        r_norm = client.post(
            "/api/fire/analyze-image",
            files={"file": ("sample_normal_room.jpg", f, "image/jpeg")},
            data={"confidence_threshold": "0.60"}
        )
    print("\n3. /api/fire/analyze-image (Normal Room):", r_norm.status_code)
    assert r_norm.status_code == 200
    norm_json = r_norm.json()
    print("   Status:", norm_json["status"])
    print(f"   Confidence: {norm_json['confidence']}%")
    print(f"   Boxes detected: {norm_json['boxes_count']}")
    assert norm_json["is_fire"] is False
    assert norm_json["status"] == "NO FIRE DETECTED"
    assert norm_json["confidence"] >= 80.0
    assert norm_json["boxes_count"] == 0
    print("   [PASSED] Normal scene classified as NO FIRE with zero false alarms!")

    # 4. Image-Based Fire Detection: Red/Orange Objects (Anti-False-Positive Test)
    red_path = os.path.join(base_dir, "results", "sample_red_object.jpg")
    assert os.path.exists(red_path), f"Sample red object missing at {red_path}"
    with open(red_path, "rb") as f:
        r_red = client.post(
            "/api/fire/analyze-image",
            files={"file": ("sample_red_object.jpg", f, "image/jpeg")},
            data={"confidence_threshold": "0.60"}
        )
    print("\n4. /api/fire/analyze-image (Red/Orange Objects):", r_red.status_code)
    assert r_red.status_code == 200
    red_json = r_red.json()
    print("   Status:", red_json["status"])
    print(f"   Confidence: {red_json['confidence']}%")
    print(f"   Boxes detected: {red_json['boxes_count']}")
    assert red_json["is_fire"] is False
    assert red_json["status"] == "NO FIRE DETECTED"
    assert red_json["boxes_count"] == 0
    print("   [PASSED] Red/orange objects correctly suppressed from false fire alarm!")

    # 5. Image-Based Fire Detection: Dark Scene
    dark_path = os.path.join(base_dir, "results", "sample_dark_scene.jpg")
    assert os.path.exists(dark_path), f"Sample dark scene missing at {dark_path}"
    with open(dark_path, "rb") as f:
        r_dark = client.post(
            "/api/fire/analyze-image",
            files={"file": ("sample_dark_scene.jpg", f, "image/jpeg")},
            data={"confidence_threshold": "0.60"}
        )
    print("\n5. /api/fire/analyze-image (Dark Scene):", r_dark.status_code)
    assert r_dark.status_code == 200
    dark_json = r_dark.json()
    print("   Status:", dark_json["status"])
    print(f"   Confidence: {dark_json['confidence']}%")
    assert dark_json["is_fire"] is False
    assert dark_json["status"] == "NO FIRE DETECTED"
    print("   [PASSED] Dark scene correctly verified with no false fire detection!")

    # 6. Error Handling: Unsupported File Format
    r_bad = client.post(
        "/api/fire/analyze-image",
        files={"file": ("unsupported_document.txt", io.BytesIO(b"Hello world text"), "text/plain")}
    )
    print("\n6. /api/fire/analyze-image (Unsupported .txt file):", r_bad.status_code)
    assert r_bad.status_code == 400
    assert "Unsupported format" in r_bad.json()["detail"]
    print("   [PASSED] Unsupported format rejected with HTTP 400 error and no fake predictions!")

    # 7. Video upload and analysis endpoint
    video_path = os.path.join(base_dir, "dataset", "fire", "Explosion005_x264A.mp4")
    with open(video_path, "rb") as vf:
        r_vid = client.post(
            "/api/fire/analyze-video",
            files={"file": ("Explosion005_x264A.mp4", vf, "video/mp4")},
            data={"confidence_threshold": "0.70", "frame_ratio": "0.20", "sample_rate": "15"}
        )
    print("\n7. /api/fire/analyze-video status:", r_vid.status_code, r_vid.json()["job_id"])
    assert r_vid.status_code == 200
    job_id = r_vid.json()["job_id"]

    # 8. Status check
    r_stat = client.get(f"/api/fire/status/{job_id}")
    print("8. /api/fire/status status:", r_stat.status_code, r_stat.json()["status"])
    assert r_stat.status_code == 200

    print("\n" + "=" * 65)
    print("ALL FASTAPI FIRE DETECTION ENDPOINTS & SCENARIOS PASSED 100%!")
    print("=" * 65)

if __name__ == "__main__":
    test_api()
