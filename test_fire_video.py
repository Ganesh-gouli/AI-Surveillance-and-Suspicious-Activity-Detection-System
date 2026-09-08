import os
import sys
import io

# Ensure UTF-8 output on Windows console
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

from backend.app.ai.fire_detector import FireDetector

def run_tests():
    print("=" * 65)
    print("SENTINELVISION AI - GENUINE FIRE DETECTION VIDEO TEST")
    print("=" * 65)

    model_path = os.path.join("models", "best_fire_detector.pt")
    if not os.path.exists(model_path):
        print(f"Error: Model checkpoint not found at {model_path}. Please run train_fire_model.py first.")
        sys.exit(1)

    detector = FireDetector(model_path=model_path)
    os.makedirs("results", exist_ok=True)

    # 1. Test on User's Original Fire Video (Held-out Test Split)
    fire_video_path = os.path.join("dataset", "fire", "Explosion005_x264A.mp4")
    output_annotated_path = os.path.join("results", "detected_fire_video.mp4")

    print(f"\n[TEST 1] Original User Fire Video: {os.path.basename(fire_video_path)}")
    print(f"Path: {fire_video_path}")
    print("Running frame-by-frame PyTorch inference & temporal aggregation...")

    res1 = detector.analyze_video(
        video_path=fire_video_path,
        output_video_path=output_annotated_path,
        confidence_threshold=0.70,
        frame_ratio=0.20,
        sample_rate=5
    )

    print("\n--- TEST 1 RESULTS ---")
    print(f"Video Name:         {res1['video_name']}")
    print(f"Final Result:       {res1['final_result']}")
    print(f"Overall Confidence: {res1['overall_confidence']:.1f}%")
    print(f"Fire Frames:        {res1['fire_frames']}")
    print(f"Analyzed Frames:    {res1['analyzed_frames']} (Total video frames: {res1['total_video_frames']})")
    print(f"Fire Percentage:    {res1['fire_percentage']:.1f}%")
    print(f"Processing Time:    {res1['processing_time_sec']} seconds")
    print(f"Annotated Video:    {output_annotated_path} (Saved: {res1['output_video_saved']})")

    # 2. Generalization Test on Unseen Normal Video
    normal_video_path = os.path.join("dataset", "no_fire", "normal_pedestrian_cctv.mp4")
    print(f"\n[TEST 2] Generalization Test on Normal Video: {os.path.basename(normal_video_path)}")
    print(f"Path: {normal_video_path}")
    print("Running frame-by-frame PyTorch inference & temporal aggregation...")

    res2 = detector.analyze_video(
        video_path=normal_video_path,
        output_video_path=None,
        confidence_threshold=0.70,
        frame_ratio=0.20,
        sample_rate=5
    )

    print("\n--- TEST 2 RESULTS ---")
    print(f"Video Name:         {res2['video_name']}")
    print(f"Final Result:       {res2['final_result']}")
    print(f"Overall Confidence: {res2['overall_confidence']:.1f}%")
    print(f"Fire Frames:        {res2['fire_frames']}")
    print(f"Analyzed Frames:    {res2['analyzed_frames']} (Total video frames: {res2['total_video_frames']})")
    print(f"Fire Percentage:    {res2['fire_percentage']:.1f}%")
    print(f"Processing Time:    {res2['processing_time_sec']} seconds")

    print("\n" + "=" * 65)
    print("AUTOMATED VERIFICATION FINISHED")
    print("=" * 65)

if __name__ == "__main__":
    run_tests()
