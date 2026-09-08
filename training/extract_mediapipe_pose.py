import os
import sys
import glob
import cv2
import numpy as np
from typing import Tuple, List, Dict, Optional
from tqdm import tqdm

# Add root directory to sys.path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from detection.pose_detector import MultiPersonPoseDetector
from detection.feature_extractor import KinematicPoseFeatureExtractor

# Strict Video-Level Dataset Partition (Prevent Data Leakage - Section 11)
VIDEO_SPLITS = {
    "train": {
        "fighting": [
            os.path.join(BASE_DIR, "FIGHTING", "Fighting007_x264A.mp4"),
            os.path.join(BASE_DIR, "FIGHTING", "Fighting020_x264A.mp4"),
            os.path.join(BASE_DIR, "FIGHTING", "Fighting028_x264A.mp4"),
        ],
        "normal": [
            os.path.join(BASE_DIR, "dataset", "no_fire", "normal_store_cctv.mp4"),
            os.path.join(BASE_DIR, "dataset", "no_fire", "normal_worker_cctv.mp4"),
        ]
    },
    "val": {
        "fighting": [
            os.path.join(BASE_DIR, "FIGHTING", "Fighting023_x264A.mp4"),
        ],
        "normal": [
            os.path.join(BASE_DIR, "dataset", "no_fire", "normal_pedestrian_cctv.mp4"),
            os.path.join(BASE_DIR, "dataset", "no_fire", "normal_traffic_cctv.mp4"),
        ]
    },
    "test": {
        "fighting": [
            os.path.join(BASE_DIR, "FIGHTING", "Fighting030_x264A.mp4"),
        ],
        "normal": [
            os.path.join(BASE_DIR, "dataset", "no_fire", "normal_classroom_cctv.mp4"),
        ]
    }
}


def extract_video_features(
    video_path: str,
    detector: MultiPersonPoseDetector,
    extractor: KinematicPoseFeatureExtractor,
    max_frames: int = 500
) -> np.ndarray:
    """
    Reads video frames at ~15 effective FPS, extracts multi-person poses,
    and returns a sequence of per-frame 510-dim kinematic feature vectors.
    """
    if not os.path.exists(video_path):
        print(f"Warning: File not found: {video_path}")
        return np.empty((0, 510), dtype=np.float32)

    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    # Determine frame sampling step to normalize effective temporal rate to ~15 FPS
    if fps >= 50.0:
        step = 4
    elif fps >= 25.0:
        step = 2
    else:
        step = 1

    extractor.reset()
    frame_features_list = []
    
    frame_idx = 0
    sampled_count = 0

    while cap.isOpened() and sampled_count < max_frames:
        ret, frame = cap.read()
        if not ret:
            break

        # Only process every `step` frames
        if frame_idx % step == 0:
            # Skip pure black or blank title screen frames (mean intensity < 10)
            if frame.mean() > 10.0:
                persons = detector.process_frame(frame)
                feat = extractor.extract_frame_features(persons)
                frame_features_list.append(feat)
                sampled_count += 1

        frame_idx += 1

    cap.release()
    if len(frame_features_list) == 0:
        return np.empty((0, 510), dtype=np.float32)

    return np.array(frame_features_list, dtype=np.float32)


def generate_sequences_from_frames(
    frame_features: np.ndarray,
    label: float,
    seq_len: int = 30,
    stride: int = 5
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Slides a window of length seq_len over sequential frame features.
    """
    num_frames = len(frame_features)
    if num_frames < seq_len:
        return np.empty((0, seq_len, 510), dtype=np.float32), np.empty((0, 1), dtype=np.float32)

    seqs = []
    labels = []
    for start in range(0, num_frames - seq_len + 1, stride):
        window = frame_features[start : start + seq_len]
        seqs.append(window)
        labels.append([label])

    return np.array(seqs, dtype=np.float32), np.array(labels, dtype=np.float32)


def process_dataset():
    print("=" * 70)
    print("EXTRACTING MEDIAPIPE POSE & KINEMATIC FEATURES (STRICT SPLIT)")
    print("=" * 70)

    cache_dir = os.path.join(BASE_DIR, "training", "cache")
    out_dir = os.path.join(BASE_DIR, "training")
    os.makedirs(cache_dir, exist_ok=True)

    detector = MultiPersonPoseDetector(min_detection_confidence=0.35)
    extractor = KinematicPoseFeatureExtractor(sequence_length=30)

    split_datasets = {}

    for split_name in ["train", "val", "test"]:
        print(f"\nProcessing Split: [{split_name.upper()}]")
        split_X = []
        split_y = []

        for category, label_val in [("fighting", 1.0), ("normal", 0.0)]:
            video_list = VIDEO_SPLITS[split_name][category]
            print(f"  Category: {category.upper()} (Label: {label_val}) - {len(video_list)} videos")

            for vpath in video_list:
                vname = os.path.basename(vpath)
                cached_file = os.path.join(cache_dir, f"{vname}.npy")

                if os.path.exists(cached_file):
                    print(f"    [Cached] Loading {vname} ...")
                    feats = np.load(cached_file)
                else:
                    print(f"    [Extracting] Processing {vname} ...")
                    feats = extract_video_features(vpath, detector, extractor)
                    np.save(cached_file, feats)
                    print(f"    [Saved] Extracted {len(feats)} frames for {vname}")

                # Create sliding sequences
                seqs, labels = generate_sequences_from_frames(
                    feats,
                    label=label_val,
                    seq_len=30,
                    stride=3 if split_name == "train" else 5
                )
                print(f"      Generated {len(seqs)} sequences of shape (30, 510)")
                if len(seqs) > 0:
                    split_X.append(seqs)
                    split_y.append(labels)

        if len(split_X) > 0:
            X_data = np.concatenate(split_X, axis=0)
            y_data = np.concatenate(split_y, axis=0)
        else:
            X_data = np.empty((0, 30, 510), dtype=np.float32)
            y_data = np.empty((0, 1), dtype=np.float32)

        # Shuffle training set
        if split_name == "train" and len(X_data) > 0:
            perm = np.random.permutation(len(X_data))
            X_data = X_data[perm]
            y_data = y_data[perm]

        np.save(os.path.join(out_dir, f"X_{split_name}.npy"), X_data)
        np.save(os.path.join(out_dir, f"y_{split_name}.npy"), y_data)
        
        fight_count = int(np.sum(y_data == 1.0))
        norm_count = int(np.sum(y_data == 0.0))
        print(f"[{split_name.upper()} COMPLETE] Total: {len(X_data)} sequences | Fighting: {fight_count}, Normal: {norm_count}")

    print("\nFeature extraction completed successfully!")


if __name__ == "__main__":
    process_dataset()
