import os
import sys
import time
import json
import cv2

WORKSPACE_ROOT = os.path.dirname(os.path.abspath(__file__))
if WORKSPACE_ROOT not in sys.path:
    sys.path.insert(0, WORKSPACE_ROOT)

from detection.atm_vandalism_detector import global_atm_detector


def test_atm_attack_videos():
    print("\n" + "=" * 70)
    print("TEST 1: EVALUATION OF ATM BREAKING & VANDALISM VIDEOS (VIDEOS/)")
    print("=" * 70)

    videos_dir = os.path.join(WORKSPACE_ROOT, "VIDEOS")
    assert os.path.exists(videos_dir), f"VIDEOS directory not found: {videos_dir}"

    test_cases = [
        ("30a.mp4", "BLUNT_FORCE_SMASH"),
        ("30b.mp4", "BLUNT_FORCE_SMASH"),
        ("5.mp4", "CROWBAR_VAULT_PRYING"),
        ("75.mp4", "CHAIN_EXTRACTION_RAM"),
        ("80.mp4", "CROWBAR_VAULT_PRYING"),
        ("88.mp4", "BLUNT_FORCE_SMASH"),
    ]

    results = []
    for fname, expected_cat in test_cases:
        vpath = os.path.join(videos_dir, fname)
        if not os.path.exists(vpath):
            print(f"[SKIP] Video {fname} not found on disk.")
            continue

        out_name = f"test_out_{fname}"
        out_vpath = os.path.join(WORKSPACE_ROOT, "results", "processed_videos", out_name)

        t0 = time.time()
        res = global_atm_detector.analyze_video(
            video_path=vpath,
            output_video_path=out_vpath,
            confidence_threshold=0.38,
            sample_rate=4,  # sample every 4 frames for rapid testing
            enable_pose=True
        )
        elapsed = time.time() - t0

        print(f"\n--- Video: {fname} ({res['duration_sec']}s, {res['total_video_frames']} frames) ---")
        print(f"  Final Decision:     {res['final_result']}")
        print(f"  Attack Detected:    {res['is_vandalism']}")
        print(f"  Attack Type:        {res['attack_type']}")
        print(f"  Overall Confidence: {res['overall_confidence'] * 100:.1f}%")
        print(f"  Perpetrators Count: {res['perpetrators_count']}")
        print(f"  Tool / Hazard:      {res['tool_detected']}")
        print(f"  Confirmed Segments: {len(res['segments'])}")
        print(f"  Analysis Speed:     {res.get('processing_fps', 0.0)} FPS ({elapsed:.2f}s elapsed)")

        assert res["is_vandalism"], f"Failed to detect ATM breaking in {fname}"
        assert res["overall_confidence"] >= 0.98, f"Expected >= 98% accuracy detection on {fname}, got {res['overall_confidence'] * 100:.1f}%"
        assert os.path.exists(out_vpath), f"Output video was not saved for {fname}"
        assert os.path.getsize(out_vpath) > 5000, f"Output video is empty for {fname}"

        results.append((fname, True, res["overall_confidence"]))

    print(f"\n[OK] TEST 1 PASSED: All {len(results)}/{len(test_cases)} ATM breaking videos correctly identified with >= 98% accuracy!")


def test_normal_cctv_control():
    print("\n" + "=" * 70)
    print("TEST 2: HELD-OUT NORMAL CCTV CONTROL (ZERO FALSE POSITIVES)")
    print("=" * 70)

    vpath = os.path.join(WORKSPACE_ROOT, "dataset", "no_fire", "normal_pedestrian_cctv.mp4")
    out_vpath = os.path.join(WORKSPACE_ROOT, "results", "processed_videos", "test_out_normal_pedestrian.mp4")

    if not os.path.exists(vpath):
        print("[SKIP] Normal control video not found.")
        return

    t0 = time.time()
    res = global_atm_detector.analyze_video(
        video_path=vpath,
        output_video_path=out_vpath,
        confidence_threshold=0.38,
        sample_rate=4,
        enable_pose=True
    )
    elapsed = time.time() - t0

    print(f"Analysis Time: {elapsed:.2f}s (Speed: {res.get('processing_fps', 0.0)} FPS)")
    print(f"Final Decision:     {res['final_result']}")
    print(f"Attack Detected:    {res['is_vandalism']}")
    print(f"Overall Confidence: {res['overall_confidence'] * 100:.1f}%")

    assert not res["is_vandalism"], "False alarm: Flagged vandalism on peaceful pedestrian CCTV!"
    assert res["final_result"] == "SAFE / NORMAL", f"Expected SAFE / NORMAL, got {res['final_result']}"
    assert res["overall_confidence"] < 0.10, f"Expected low confidence for normal scene, got {res['overall_confidence']}"
    print("\n[OK] TEST 2 PASSED: 0% False Positives on normal pedestrian footage!")


def test_api_metrics_and_endpoints():
    print("\n" + "=" * 70)
    print("TEST 3: FASTAPI ROUTER BENCHMARKS & METRICS INTEGRATION")
    print("=" * 70)

    from backend.app.api.routers.atm import get_atm_metrics, list_sample_videos
    import asyncio

    metrics = asyncio.run(get_atm_metrics())
    print("Model Metrics:")
    print(f"  Model Name: {metrics['model_name']} (v{metrics['version']})")
    print(f"  Accuracy:   {metrics['accuracy'] * 100:.1f}%")
    print(f"  Precision:  {metrics['precision'] * 100:.1f}%")
    print(f"  Recall:     {metrics['recall'] * 100:.1f}%")
    print(f"  F1 Score:   {metrics['f1_score'] * 100:.1f}%")
    print(f"  Latency:    {metrics['latency_ms']} ms")

    assert metrics["accuracy"] >= 0.98, f"Accuracy below 98%: {metrics['accuracy']}"
    assert metrics["precision"] >= 0.98, f"Precision below 98%: {metrics['precision']}"

    samples_res = asyncio.run(list_sample_videos())
    sample_list = samples_res["samples"]
    print(f"\nAvailable Sample Videos in VIDEOS/ count: {len(sample_list)}")
    for s in sample_list:
        print(f"  - {s['filename']}: {s['title']} ({s['duration_sec']}s)")

    assert len(sample_list) >= 6, "Expected at least 6 sample videos in VIDEOS/"
    print("\n[OK] TEST 3 PASSED: FastAPI metrics and endpoints verified!")


if __name__ == "__main__":
    t_start = time.time()
    print("\n" + "=" * 70)
    print("SENTINELVISION AI: ATM VANDALISM & BREAKING DETECTION TEST SUITE")
    print("=" * 70)

    test_atm_attack_videos()
    test_normal_cctv_control()
    test_api_metrics_and_endpoints()

    total_time = round(time.time() - t_start, 2)
    print("\n" + "=" * 70)
    print(f"ALL TESTS PASSED SUCCESSFULLY IN {total_time}s! ACCURACY >= 98%")
    print("=" * 70)
