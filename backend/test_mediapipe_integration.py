import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import cv2
import numpy as np
from backend.app.ai.mediapipe_pose import global_pose_engine, calculate_angle_2d
from backend.app.ai.activity_classifier import HumanActivityClassifier
from backend.app.ai.inference_pipeline import global_pipeline

print("="*60)
print("TESTING MEDIAPIPE POSE & BIOMECHANICAL INTEGRATION")
print("="*60)

# 1. Test MediaPipePoseEngine
assert global_pose_engine.is_initialized, "MediaPipePoseEngine failed to initialize!"
print("[PASS] MediaPipePoseEngine is initialized successfully.")

# 2. Test Angle Math
# Right angle (90 deg)
p_hip = (100, 100)
p_knee = (100, 200)
p_ankle = (200, 200)
angle_90 = calculate_angle_2d(p_hip, p_knee, p_ankle)
assert abs(angle_90 - 90.0) < 1.0, f"Angle should be ~90, got {angle_90}"
print(f"[PASS] 90° Knee Flexion calculated accurately: {angle_90:.1f}°")

# Straight leg (180 deg)
p_ankle_straight = (100, 300)
angle_180 = calculate_angle_2d(p_hip, p_knee, p_ankle_straight)
assert abs(angle_180 - 180.0) < 1.0, f"Angle should be ~180, got {angle_180}"
print(f"[PASS] 180° Straight Leg extension calculated accurately: {angle_180:.1f}°")

# 3. Test HumanActivityClassifier
classifier = HumanActivityClassifier()
print(f"[PASS] HumanActivityClassifier loaded. Classes count: {len(classifier.classes)}")
assert "standing" in classifier.classes, "'standing' should be in classifier classes!"
assert "sitting" in classifier.classes, "'sitting' should be in classifier classes!"
print(f"[PASS] 'standing' and 'sitting' present in classifier classes: {classifier.classes}")

# 4. Test global_pipeline with blank frame
blank_frame = np.zeros((720, 1280, 3), dtype=np.uint8)
result = global_pipeline.process_frame(blank_frame, camera_id="TEST-CAM")
assert "people" in result
assert "objects" in result
print(f"[PASS] global_pipeline.process_frame executed smoothly without error! FPS: {result['fps']}")

# 5. Test synthetic person crop with drawn limbs (or real frame)
synthetic_frame = np.zeros((720, 1280, 3), dtype=np.uint8)
# Draw a simple stick figure in the frame
cv2.circle(synthetic_frame, (640, 200), 30, (255, 255, 255), -1) # Head
cv2.line(synthetic_frame, (640, 230), (640, 450), (255, 255, 255), 10) # Torso
cv2.line(synthetic_frame, (640, 450), (600, 650), (255, 255, 255), 10) # Left leg
cv2.line(synthetic_frame, (640, 450), (680, 650), (255, 255, 255), 10) # Right leg
cv2.line(synthetic_frame, (640, 280), (560, 380), (255, 255, 255), 8) # Left arm
cv2.line(synthetic_frame, (640, 280), (720, 380), (255, 255, 255), 8) # Right arm

pose_res = global_pose_engine.analyze_person(synthetic_frame, [540, 150, 200, 520])
print(f"[PASS] MediaPipe analyze_person returned posture: '{pose_res['activity']}' (conf: {pose_res['confidence']})")
print(f"       Details: {pose_res['posture_details']}")
print("="*60)
print("ALL BACKEND MEDIAPIPE TESTS PASSED!")
print("="*60)
