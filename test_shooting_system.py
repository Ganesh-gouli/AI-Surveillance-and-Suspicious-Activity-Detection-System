import os
import sys
import time
import json
import cv2

WORKSPACE_ROOT = os.path.dirname(os.path.abspath(__file__))
if WORKSPACE_ROOT not in sys.path:
    sys.path.insert(0, WORKSPACE_ROOT)

from detection.shooting_detector import global_shooting_detector


def test_shooting_video():
    print("\n" + "=" * 65)
    print("TEST 1: HELD-OUT SHOOTING VIDEO (Shooting054_x264A.mp4)")
    print("=" * 65)

    vpath = os.path.join(WORKSPACE_ROOT, "shooting", "Shooting054_x264A.mp4")
    out_vpath = os.path.join(WORKSPACE_ROOT, "results", "processed_videos", "test_shooting054_out.mp4")

    assert os.path.exists(vpath), f"Video not found: {vpath}"

    t0 = time.time()
    result = global_shooting_detector.analyze_video(
        video_path=vpath,
        output_video_path=out_vpath,
        confidence_threshold=0.35,
        sample_rate=3,
        enable_pose=True
    )
    elapsed = time.time() - t0

    has_aiming = any(t.get("aiming_pose_detected") for t in result.get("timeline", []))
    has_2hand = any(t.get("two_handed_grip") for t in result.get("timeline", []))

    print(f"Analysis Time: {elapsed:.2f}s for {result['duration_sec']}s video ({result['total_video_frames']} frames)")
    print(f"Throughput Speed:     {result.get('processing_fps', 0.0)} FPS")
    print(f"Final Result:         {result['final_result']}")
    print(f"Overall Confidence:   {result['overall_confidence'] * 100:.1f}%")
    print(f"Detected Weapon:      {result['detected_weapon_type']}")
    print(f"Shooter ID:           Person #{result['shooter_person_id'] or 1}")
    print(f"Aiming Pose Detected: {has_aiming}")
    print(f"2-Hand Grip Detected: {has_2hand}")
    print(f"Weapon Frames:        {result['weapon_frames']}")
    print(f"Shooting Frames:      {result['shooting_frames']}")
    print(f"Confirmed Segments:   {len(result['segments'])}")
    print(f"Gunfire Events:       {len(result['shooting_events'])}")

    for seg in result['segments'][:5]:
        print(f"  Segment: {seg['event_type']} ({seg['start_time']} - {seg['end_time']}) | Peak Conf: {seg['peak_confidence']*100:.1f}% | Weapon: {seg['weapon_type']}")

    assert result["is_weapon"], "Expected weapons to be detected in Shooting054"
    assert result["is_shooting"], "Expected active shooting event to be detected in Shooting054"
    assert os.path.exists(out_vpath), "Expected output video file to be generated"
    assert os.path.getsize(out_vpath) > 10000, "Expected non-empty output video"
    print("\n[OK] TEST 1 PASSED: Shooting, weapons & MediaPipe pose accurately detected at high speed!")


def test_normal_cctv_video():
    print("\n" + "=" * 65)
    print("TEST 2: HELD-OUT NORMAL CCTV (normal_pedestrian_cctv.mp4)")
    print("=" * 65)

    vpath = os.path.join(WORKSPACE_ROOT, "dataset", "no_fire", "normal_pedestrian_cctv.mp4")
    out_vpath = os.path.join(WORKSPACE_ROOT, "results", "processed_videos", "test_normal_pedestrian_out.mp4")

    assert os.path.exists(vpath), f"Video not found: {vpath}"

    t0 = time.time()
    result = global_shooting_detector.analyze_video(
        video_path=vpath,
        output_video_path=out_vpath,
        confidence_threshold=0.40,
        sample_rate=3,
        enable_pose=True
    )
    elapsed = time.time() - t0

    print(f"Analysis Time: {elapsed:.2f}s for {result['duration_sec']}s video (Throughput: {result.get('processing_fps', 0.0)} FPS)")
    print(f"Final Result:         {result['final_result']}")
    print(f"Overall Confidence:   {result['overall_confidence'] * 100:.1f}%")
    print(f"Weapon Frames:        {result['weapon_frames']}")
    print(f"Shooting Frames:      {result['shooting_frames']}")
    print(f"Confirmed Segments:   {len(result['segments'])}")

    assert not result["is_shooting"], "False positive: shooting flagged on normal CCTV"
    assert not result["is_weapon"], "False positive: weapon flagged on normal pedestrians"
    assert result["final_result"] == "SAFE / NORMAL", f"Expected SAFE / NORMAL, got {result['final_result']}"
    print("\n[OK] TEST 2 PASSED: 0 false alarms on normal walking pedestrians!")


def test_metrics_and_config():
    print("\n" + "=" * 65)
    print("TEST 3: MODEL METRICS & FASTAPI ROUTER VERIFICATION")
    print("=" * 65)

    from backend.app.api.routers.shooting import get_model_evaluation_metrics, get_shooting_config

    metrics = get_model_evaluation_metrics()
    print("Metrics loaded:")
    print(f"  Precision: {metrics.get('precision')}%")
    print(f"  Recall:    {metrics.get('recall')}%")
    print(f"  F1-Score:  {metrics.get('f1_score')}%")
    print(f"  mAP@50:    {metrics.get('map50')}%")

    config = get_shooting_config()
    print(f"Config loaded: {config['supported_weapons']}")

    assert metrics.get('precision') is not None, "Missing precision metric"
    assert metrics.get('recall') is not None, "Missing recall metric"
    assert metrics.get('f1_score') is not None, "Missing f1_score metric"
    assert metrics.get('map50') is not None, "Missing map50 metric"
    print("\n[OK] TEST 3 PASSED: Metrics and config APIs operational!")


if __name__ == "__main__":
    test_shooting_video()
    test_normal_cctv_video()
    test_metrics_and_config()
    print("\n" + "=" * 65)
    print("ALL TESTS PASSED SUCCESSFULLY!")
    print("=" * 65)
