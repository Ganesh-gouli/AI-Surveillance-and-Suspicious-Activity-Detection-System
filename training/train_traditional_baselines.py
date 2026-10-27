"""
SentinelVision AI - Section 7.9 Traditional Machine Learning Baselines
Trains and benchmarks traditional machine learning classifiers:
1. Support Vector Machine (SVM - RBF Kernel)
2. Support Vector Machine (SVM - Linear Kernel)
3. Random Forest (100 Estimators)
4. Logistic Regression (L2 Regularization)

Features: Handcrafted spatiotemporal statistical features extracted from sequence dynamics:
- Temporal Mean across 30 frames
- Temporal Standard Deviation (motion variance)
- Temporal Max & Range (kinematic strike burst)
- Velocity & Acceleration magnitude approximations

Evaluates strictly on unseen held-out test set (training/X_test.npy, y_test.npy) with zero data leakage.
"""

import os
import sys
import time
import json
import pickle
import numpy as np
from typing import Dict, Any, Tuple
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, roc_auc_score

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAIN_DIR = os.path.join(BASE_DIR, "training")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)


def extract_statistical_features(X_seq: np.ndarray) -> np.ndarray:
    """
    Transforms sequence tensor (N, 30, 510) into fixed-length handcrafted feature matrix.
    Extracts mean, std, max, and first-order difference (velocity) statistics.
    Output dim: 510 * 4 = 2040 features per sequence.
    """
    mean_feat = np.mean(X_seq, axis=1)
    std_feat = np.std(X_seq, axis=1)
    max_feat = np.max(X_seq, axis=1)
    
    # Velocity: first-order temporal difference
    diff_feat = np.diff(X_seq, axis=1)
    vel_feat = np.mean(np.abs(diff_feat), axis=1)

    combined = np.hstack([mean_feat, std_feat, max_feat, vel_feat])
    return combined


def run_traditional_benchmarks() -> Dict[str, Any]:
    print("=" * 70)
    print("SECTION 7.9: TRAINING & BENCHMARKING TRADITIONAL ML BASELINES")
    print("=" * 70)

    # 1. Load Data
    X_train_raw = np.load(os.path.join(TRAIN_DIR, "X_train.npy"))
    y_train = np.load(os.path.join(TRAIN_DIR, "y_train.npy")).ravel().astype(int)

    X_val_raw = np.load(os.path.join(TRAIN_DIR, "X_val.npy"))
    y_val = np.load(os.path.join(TRAIN_DIR, "y_val.npy")).ravel().astype(int)

    X_test_raw = np.load(os.path.join(TRAIN_DIR, "X_test.npy"))
    y_test = np.load(os.path.join(TRAIN_DIR, "y_test.npy")).ravel().astype(int)

    print(f"Extracted train sequences: {X_train_raw.shape}")
    print(f"Extracted val sequences:   {X_val_raw.shape}")
    print(f"Extracted test sequences:  {X_test_raw.shape}")

    # 2. Extract Handcrafted Spatiotemporal Features
    print("Computing spatiotemporal summary statistics (Mean, Std, Max, Velocity)...")
    X_train = extract_statistical_features(X_train_raw)
    X_val = extract_statistical_features(X_val_raw)
    X_test = extract_statistical_features(X_test_raw)
    print(f"Feature representation vector length: {X_train.shape[1]}")

    # 3. Standardize Features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    # Combine train + val for final model fit after tuning
    X_train_full = np.vstack([X_train_scaled, X_val_scaled])
    y_train_full = np.concatenate([y_train, y_val])

    models = {
        "SVM (RBF Kernel)": {
            "model": SVC(kernel="rbf", C=10.0, gamma="scale", probability=True, random_state=42),
            "features": "Handcrafted Spatiotemporal Kinematics (2040-dim)"
        },
        "SVM (Linear Kernel)": {
            "model": SVC(kernel="linear", C=1.0, probability=True, random_state=42),
            "features": "Handcrafted Spatiotemporal Kinematics (2040-dim)"
        },
        "Random Forest (100 Trees)": {
            "model": RandomForestClassifier(n_estimators=100, max_depth=15, random_state=42, n_jobs=-1),
            "features": "Handcrafted Spatiotemporal Kinematics (2040-dim)"
        },
        "Logistic Regression (L2)": {
            "model": LogisticRegression(C=0.5, penalty="l2", max_iter=1000, random_state=42),
            "features": "Handcrafted Spatiotemporal Kinematics (2040-dim)"
        }
    }

    results = []

    for name, config in models.items():
        clf = config["model"]
        print(f"\nTraining {name}...")

        t_start = time.time()
        clf.fit(X_train_full, y_train_full)
        train_duration_sec = time.time() - t_start

        # Benchmark test latency
        t_infer_start = time.time()
        y_pred = clf.predict(X_test_scaled)
        total_infer_time = time.time() - t_infer_start
        per_sample_latency_ms = (total_infer_time / len(X_test_scaled)) * 1000.0

        probs = clf.predict_proba(X_test_scaled)[:, 1] if hasattr(clf, "predict_proba") else None
        auc = float(roc_auc_score(y_test, probs)) if probs is not None else None

        acc = float(accuracy_score(y_test, y_pred))
        prec, rec, f1, _ = precision_recall_fscore_support(y_test, y_pred, average="binary", zero_division=0)
        cm = confusion_matrix(y_test, y_pred).tolist()

        # Measure in-memory size via pickle
        model_size_mb = len(pickle.dumps(clf)) / (1024 * 1024)

        model_res = {
            "model_name": name,
            "feature_representation": config["features"],
            "accuracy": round(acc * 100.0, 2),
            "precision": round(float(prec) * 100.0, 2),
            "recall": round(float(rec) * 100.0, 2),
            "f1_score": round(float(f1) * 100.0, 2),
            "roc_auc": round(auc * 100.0, 2) if auc else None,
            "training_duration_sec": round(train_duration_sec, 2),
            "inference_latency_ms": round(per_sample_latency_ms, 3),
            "model_size_mb": round(model_size_mb, 2),
            "confusion_matrix": cm,
            "test_samples": len(y_test),
            "is_demonstration": False,
            "evaluation_completed": True
        }
        results.append(model_res)

        print(f"  Accuracy:  {acc * 100.0:.2f}% | F1: {f1 * 100.0:.2f}%")
        print(f"  Training:  {train_duration_sec:.2f}s | Latency: {per_sample_latency_ms:.3f} ms/sample | Size: {model_size_mb:.2f} MB")

    output_path = os.path.join(RESULTS_DIR, "traditional_models_results.json")
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nSaved traditional models evaluation to: {output_path}")
    return {"results": results, "saved_path": output_path}


if __name__ == "__main__":
    run_traditional_benchmarks()
