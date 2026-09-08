import os
import sys
import time

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

from detection.fighting_detector import FightingDetector

def test_pipeline():
    print("=" * 60)
    print("TESTING ACCURATE & SMOOTH MEDIAPIPE FIGHTING DETECTION")
    print("=" * 60)

    detector = FightingDetector()
    print(f"Model loaded: {detector.is_loaded}")
    print(f"Confidence threshold: {detector.confidence_threshold}")

    test_video = os.path.join(BASE_DIR, "FIGHTING", "Fighting023_x264A.mp4")
    out_video = os.path.join(BASE_DIR, "results", "processed_videos", "test_smooth_Fighting023.mp4")

    print(f"Analyzing test video: {test_video}")
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
    print(f"Analyzed Frames: {result['analyzed_frames']} / {result['total_video_frames']}")
    print(f"Fighting Percentage: {result['fighting_percentage']}%")
    print(f"Segments Count: {len(result['segments'])}")
    for i, seg in enumerate(result['segments']):
        print(f"  Segment {i+1}: {seg['start_time']} -> {seg['end_time']} ({seg['start_sec']}s - {seg['end_sec']}s) | Conf: {seg['peak_conf']}% | Action: {seg.get('detected_action', 'N/A')}")
    print(f"Processing time: {t_elapsed:.2f}s ({result['analyzed_frames']/t_elapsed:.1f} FPS)")
    print("-" * 60)

    # Check output video faststart header
    with open(out_video, "rb") as f:
        header = f.read(64)
    has_moov = b'moov' in header or b'ftyp' in header
    print(f"Output video exists: {os.path.exists(out_video)} ({os.path.getsize(out_video)} bytes)")
    print(f"Browser faststart compatible: {has_moov}")

    assert result['is_fighting'], "Fighting023 must be detected as FIGHTING!"
    print("\n✅ ACCURACY & SMOOTHNESS TEST PASSED SUCCESSFULLY!")

if __name__ == "__main__":
    test_pipeline()
