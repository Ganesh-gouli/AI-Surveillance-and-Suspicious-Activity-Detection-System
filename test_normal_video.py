import os
import sys
import time

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

from detection.fighting_detector import FightingDetector

def test_normal():
    print("=" * 60)
    print("TESTING NORMAL ACTIVITY (PEOPLE WALKING/TALKING)")
    print("=" * 60)

    detector = FightingDetector()
    test_video = os.path.join(BASE_DIR, "dataset", "no_fire", "normal_pedestrian_cctv.mp4")
    out_video = os.path.join(BASE_DIR, "results", "processed_videos", "test_normal_pedestrian.mp4")

    print(f"Analyzing normal video: {test_video}")
    t0 = time.time()
    result = detector.analyze_video(
        video_path=test_video,
        output_video_path=out_video,
        confidence_threshold=0.70,
        sample_rate=2
    )
    t_elapsed = time.time() - t0

    print("-" * 60)
    print(f"Result: {result['final_result']}")
    print(f"Is Fighting: {result['is_fighting']}")
    print(f"Overall Confidence: {result['overall_confidence']}%")
    print(f"Detected Action: {result.get('detected_action', 'N/A')}")
    print(f"Analyzed Frames: {result['analyzed_frames']}")
    print(f"Fighting Percentage: {result['fighting_percentage']}%")
    print(f"Segments Count: {len(result['segments'])}")
    print(f"Processing time: {t_elapsed:.2f}s")
    print("-" * 60)

    assert not result['is_fighting'], "Normal pedestrians must NOT be classified as fighting!"
    print("✅ NORMAL ACTIVITY TEST PASSED! ZERO FALSE POSITIVES.")

if __name__ == "__main__":
    test_normal()
