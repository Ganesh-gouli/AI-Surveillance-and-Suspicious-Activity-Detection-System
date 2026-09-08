import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from detection.temporal_model import FightingBiLSTM

def evaluate_test_set():
    print("=" * 70)
    print("EVALUATING ON UNSEEN HELD-OUT TEST VIDEOS (Section 24)")
    print("=" * 70)

    train_dir = os.path.join(BASE_DIR, "training")
    checkpoint_path = os.path.join(BASE_DIR, "models", "fighting_mediapipe_bilstm.pt")
    results_dir = os.path.join(BASE_DIR, "results")

    if not os.path.exists(checkpoint_path):
        print(f"Error: Trained model checkpoint not found at {checkpoint_path}!")
        return

    X_test = np.load(os.path.join(train_dir, "X_test.npy"))
    y_test = np.load(os.path.join(train_dir, "y_test.npy"))

    print(f"Test samples: {X_test.shape} (Fighting: {int(np.sum(y_test==1))}, Normal: {int(np.sum(y_test==0))})")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FightingBiLSTM(input_dim=510, hidden_dim=128, num_layers=2, dropout=0.3).to(device)

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    with torch.no_grad():
        inputs = torch.tensor(X_test, dtype=torch.float32).to(device)
        preds, attn = model(inputs)
        probs = preds.cpu().numpy().flatten()
        y_pred = (probs >= 0.50).astype(int)
        y_true = y_test.flatten().astype(int)

    acc = np.mean(y_pred == y_true)
    prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary', zero_division=0)
    cm = confusion_matrix(y_true, y_pred)

    print("\n" + "=" * 50)
    print("      GENUINE UNSEEN TEST EVALUATION METRICS")
    print("=" * 50)
    print(f"Accuracy:           {acc * 100:.2f}%")
    print(f"FIGHTING Precision: {prec * 100:.2f}%")
    print(f"FIGHTING Recall:    {rec * 100:.2f}%")
    print(f"FIGHTING F1-Score:  {f1 * 100:.2f}%")
    print("=" * 50)

    print("\nClassification Report:")
    target_names = ["NORMAL", "FIGHTING"]
    print(classification_report(y_true, y_pred, target_names=target_names, digits=4, zero_division=0))

    # Confusion Matrix Visualization
    plt.figure(figsize=(6, 5))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title("Confusion Matrix on Unseen Test Videos", fontsize=12, pad=12)
    plt.colorbar()
    tick_marks = np.arange(len(target_names))
    plt.xticks(tick_marks, target_names, fontsize=10)
    plt.yticks(tick_marks, target_names, fontsize=10)

    # Label counts inside cells
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black",
                     fontsize=14, fontweight="bold")

    plt.ylabel('True Class', fontsize=11)
    plt.xlabel('Predicted Class', fontsize=11)
    plt.tight_layout()

    cm_path = os.path.join(results_dir, "confusion_matrix.png")
    plt.savefig(cm_path, dpi=150)
    plt.close()
    print(f"Saved confusion matrix plot to: {cm_path}")

    return {
        "accuracy": float(acc),
        "precision": float(prec),
        "recall": float(rec),
        "f1": float(f1),
        "confusion_matrix": cm.tolist()
    }

if __name__ == "__main__":
    evaluate_test_set()
