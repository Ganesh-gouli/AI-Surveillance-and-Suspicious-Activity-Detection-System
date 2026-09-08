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
    print("=" * 60)
    print("TESTING FASTAPI FIRE DETECTION ENDPOINTS")
    print("=" * 60)

    # 1. Config endpoint
    r = client.get("/api/fire/config")
    print("1. /api/fire/config status:", r.status_code, r.json())
    assert r.status_code == 200

    # 2. Live frame detection endpoint
    test_img = np.zeros((240, 320, 3), dtype=np.uint8)
    _, buf = cv2.imencode(".jpg", test_img)
    b64 = base64.b64encode(buf).decode('utf-8')

    r = client.post("/api/fire/detect-frame", json={
        "image_base64": b64,
        "camera_id": "TEST-CAM"
    })
    print("2. /api/fire/detect-frame status:", r.status_code, r.json())
    assert r.status_code == 200
    assert "is_fire" in r.json()

    # 3. Video upload and analysis endpoint
    base_dir = os.path.dirname(os.path.abspath(__file__))
    video_path = os.path.join(base_dir, "dataset", "fire", "Explosion005_x264A.mp4")
    with open(video_path, "rb") as vf:
        r = client.post(
            "/api/fire/analyze-video",
            files={"file": ("Explosion005_x264A.mp4", vf, "video/mp4")},
            data={"confidence_threshold": "0.70", "frame_ratio": "0.20", "sample_rate": "15"}
        )
    print("3. /api/fire/analyze-video status:", r.status_code, r.json())
    assert r.status_code == 200
    job_id = r.json()["job_id"]

    # 4. Status check
    r = client.get(f"/api/fire/status/{job_id}")
    print("4. /api/fire/status status:", r.status_code, r.json()["status"])
    assert r.status_code == 200

    # 5. Video streaming endpoint
    r = client.get("/api/fire/video/detected_fire_video.mp4")
    print("5. /api/fire/video/detected_fire_video.mp4 status:", r.status_code, "headers:", r.headers.get("content-type"))
    assert r.status_code == 200

    print("\nALL FASTAPI FIRE DETECTION ENDPOINTS VERIFIED SUCCESSFULLY!")

if __name__ == "__main__":
    test_api()
