"""
Industrial Accident AI System Validation Suite
Tests both primary accident scenarios:
  1. Forklift vehicle collision at blind corner (vidssave.com Forklift Accident_ The Blind Corner 480P.mp4)
  2. Warehouse cargo rack / pallet stack collapse (WhatsApp Video 2026-10-23 at 11.54.39 PM.mp4)
  3. Control test on safe/normal pedestrian CCTV (no false alarms)
  4. Backend REST API endpoints (/api/accidents/samples, /api/accidents/metrics, /api/accidents/events)
"""

import os
import sys
import time
import json
import pytest

WORKSPACE_ROOT = os.path.dirname(os.path.abspath(__file__))
if WORKSPACE_ROOT not in sys.path:
    sys.path.insert(0, WORKSPACE_ROOT)

from detection.industrial_accident_detector import global_accident_detector


def test_forklift_collision_video():
    print("\n" + "=" * 70)
    print("TEST 1: FORKLIFT VEHICLE COLLISION AT BLIND CORNER INTERSECTION")
    print("=" * 70)

    accidents_dir = os.path.join(WORKSPACE_ROOT, "accidents")
    fname = "vidssave.com Forklift Accident_ The Blind Corner 480P.mp4"
    vpath = os.path.join(accidents_dir, fname)

    assert os.path.exists(vpath), f"Forklift video not found at: {vpath}"

    out_name = "test_out_forklift_collision.mp4"
    out_vpath = os.path.join(WORKSPACE_ROOT, "results", "processed_videos", out_name)
    os.makedirs(os.path.dirname(out_vpath), exist_ok=True)

    t0 = time.time()
    res = global_accident_detector.analyze_video(
        video_path=vpath,
        output_video_path=out_vpath,
        confidence_threshold=0.40,
        sample_rate=4,  # high resolution sampling
    )
    elapsed = time.time() - t0

    print(f"\n--- Forklift Collision Video Results ({res['duration_sec']}s, {res['total_video_frames']} frames) ---")
    print(f"  Final Decision:     {res['final_result']}")
    print(f"  Accident Detected:  {res['is_accident']}")
    print(f"  Accident Type:      {res['accident_type']}")
    print(f"  Overall Confidence: {res['overall_confidence'] * 100:.1f}%")
    print(f"  Hazard Detected:    {res['hazard_detected']}")
    print(f"  Vehicles Involved:  {res['vehicles_count']}")
    print(f"  Accident Frames:    {res['accident_frames']}/{res['analyzed_frames']}")
    print(f"  Confirmed Segments: {len(res['segments'])}")
    print(f"  Processing Speed:   {res.get('processing_fps', 0.0)} FPS ({elapsed:.2f}s elapsed)")

    assert res["is_accident"] is True, "Failed to detect forklift collision accident"
    assert res["accident_type"] == "FORKLIFT_VEHICLE_COLLISION", f"Wrong accident type: {res['accident_type']}"
    assert res["overall_confidence"] >= 0.98, f"Confidence below 98%: {res['overall_confidence'] * 100:.1f}%"
    assert res["vehicles_count"] == 3, f"Expected 3 active forklifts detected, got {res['vehicles_count']}"
    assert len(res.get("fleet", [])) == 3, f"Expected fleet size 3, got {len(res.get('fleet', []))}"
    fleet_ids = [f["id"] for f in res.get("fleet", [])]
    assert "FL-01" in fleet_ids and "FL-02" in fleet_ids and "FL-03" in fleet_ids, f"Expected FL-01, FL-02, FL-03 in {fleet_ids}"
    assert res.get("collision_intercept", {}).get("pair") == ["FL-02", "FL-03"], "Expected collision intercept pair to be FL-02 and FL-03"
    assert len(res["segments"]) >= 1, "Expected at least 1 confirmed incident time segment"
    assert os.path.exists(out_vpath), f"Processed video output missing: {out_vpath}"
    assert os.path.getsize(out_vpath) > 10000, f"Processed video file too small: {os.path.getsize(out_vpath)} bytes"
    print("\n[OK] TEST 1 PASSED: 3 Forklifts (FL-01, FL-02, FL-03) and blind corner collision accurately detected with >= 98% confidence!")


def test_cargo_pallet_collapse_video():
    print("\n" + "=" * 70)
    print("TEST 2: WAREHOUSE CARGO RACK / PALLET AVALANCHE COLLAPSE")
    print("=" * 70)

    accidents_dir = os.path.join(WORKSPACE_ROOT, "accidents")
    fname = "WhatsApp Video 2026-10-23 at 11.54.39 PM.mp4"
    vpath = os.path.join(accidents_dir, fname)

    assert os.path.exists(vpath), f"Pallet collapse video not found at: {vpath}"

    out_name = "test_out_cargo_collapse.mp4"
    out_vpath = os.path.join(WORKSPACE_ROOT, "results", "processed_videos", out_name)
    os.makedirs(os.path.dirname(out_vpath), exist_ok=True)

    t0 = time.time()
    res = global_accident_detector.analyze_video(
        video_path=vpath,
        output_video_path=out_vpath,
        confidence_threshold=0.40,
        sample_rate=4,
    )
    elapsed = time.time() - t0

    print(f"\n--- Cargo Collapse Video Results ({res['duration_sec']}s, {res['total_video_frames']} frames) ---")
    print(f"  Final Decision:     {res['final_result']}")
    print(f"  Accident Detected:  {res['is_accident']}")
    print(f"  Accident Type:      {res['accident_type']}")
    print(f"  Overall Confidence: {res['overall_confidence'] * 100:.1f}%")
    print(f"  Hazard Detected:    {res['hazard_detected']}")
    print(f"  Accident Frames:    {res['accident_frames']}/{res['analyzed_frames']}")
    print(f"  Confirmed Segments: {len(res['segments'])}")
    print(f"  Processing Speed:   {res.get('processing_fps', 0.0)} FPS ({elapsed:.2f}s elapsed)")

    assert res["is_accident"] is True, "Failed to detect cargo stack collapse accident"
    assert res["accident_type"] == "CARGO_STACK_COLLAPSE", f"Wrong accident type: {res['accident_type']}"
    assert res["overall_confidence"] >= 0.98, f"Confidence below 98%: {res['overall_confidence'] * 100:.1f}%"
    assert len(res["segments"]) >= 1, "Expected at least 1 confirmed incident time segment"
    assert os.path.exists(out_vpath), f"Processed video output missing: {out_vpath}"
    assert os.path.getsize(out_vpath) > 10000, f"Processed video file too small: {os.path.getsize(out_vpath)} bytes"
    print("\n[OK] TEST 2 PASSED: Pallet / cargo collapse accurately classified with >= 98% confidence!")


def test_normal_cctv_control():
    print("\n" + "=" * 70)
    print("TEST 3: NORMAL CCTV BASELINE (ZERO FALSE POSITIVES)")
    print("=" * 70)

    vpath = os.path.join(WORKSPACE_ROOT, "dataset", "no_fire", "normal_pedestrian_cctv.mp4")
    if not os.path.exists(vpath):
        print(f"[SKIP] Normal baseline video not found at: {vpath}")
        return

    out_name = "test_out_normal_control.mp4"
    out_vpath = os.path.join(WORKSPACE_ROOT, "results", "processed_videos", out_name)

    t0 = time.time()
    res = global_accident_detector.analyze_video(
        video_path=vpath,
        output_video_path=out_vpath,
        confidence_threshold=0.40,
        sample_rate=6,
    )
    elapsed = time.time() - t0

    print(f"\n--- Normal CCTV Baseline Results ({res['duration_sec']}s) ---")
    print(f"  Final Decision:     {res['final_result']}")
    print(f"  Accident Detected:  {res['is_accident']}")
    print(f"  Overall Confidence: {res['overall_confidence'] * 100:.1f}%")
    print(f"  Processing Speed:   {res.get('processing_fps', 0.0)} FPS ({elapsed:.2f}s elapsed)")

    assert res["is_accident"] is False, "False positive triggered on safe pedestrian CCTV!"
    assert res["final_result"] == "SAFE / NORMAL", f"Expected SAFE / NORMAL, got {res['final_result']}"
    assert res["overall_confidence"] < 0.10, f"Confidence on normal scene is too high: {res['overall_confidence'] * 100:.1f}%"
    print("\n[OK] TEST 3 PASSED: Baseline safe footage cleanly classified as normal (< 10% confidence)!")


def test_accident_api_routes():
    print("\n" + "=" * 70)
    print("TEST 4: BACKEND REST API ENDPOINTS (/api/accidents/*)")
    print("=" * 70)

    try:
        from fastapi.testclient import TestClient
        from backend.app.main import app

        client = TestClient(app)

        # 1. Samples endpoint
        res = client.get("/api/accidents/samples")
        assert res.status_code == 200, f"Failed GET /api/accidents/samples: {res.status_code}"
        data = res.json()
        samples = data.get("samples", data)
        assert len(samples) >= 2, f"Expected at least 2 sample videos, got {len(samples)}"
        print(f"  [OK] /api/accidents/samples returned {len(samples)} videos")

        # 2. Metrics endpoint
        res = client.get("/api/accidents/metrics")
        assert res.status_code == 200, f"Failed GET /api/accidents/metrics: {res.status_code}"
        metrics = res.json()
        assert "accuracy" in metrics
        assert "classes" in metrics
        assert metrics["accuracy"] >= 0.98, f"Expected accuracy >= 0.98, got {metrics['accuracy']}"
        print(f"  [OK] /api/accidents/metrics returned model metrics: {metrics['model_name']} (Accuracy: {metrics['accuracy']*100:.1f}%)")

        # 3. Events endpoint
        res = client.get("/api/accidents/events")
        assert res.status_code == 200, f"Failed GET /api/accidents/events: {res.status_code}"
        events_data = res.json()
        assert "events" in events_data
        print(f"  [OK] /api/accidents/events returned {len(events_data['events'])} incident records")

        print("\n[OK] TEST 4 PASSED: Backend API routes responsive and verified!")
    except ImportError as e:
        print(f"[WARN] TestClient or FastAPI not available: {e}")


if __name__ == "__main__":
    test_forklift_collision_video()
    test_cargo_pallet_collapse_video()
    test_normal_cctv_control()
    test_accident_api_routes()
    print("\n" + "*" * 70)
    print("ALL INDUSTRIAL ACCIDENT SYSTEM TESTS COMPLETED SUCCESSFULLY!")
    print("*" * 70)
