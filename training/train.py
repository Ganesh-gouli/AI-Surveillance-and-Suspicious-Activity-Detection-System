import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from detection.temporal_model import FightingBiLSTM

def train_model():
    print("=" * 70)
    print("TRAINING TEMPORAL BiLSTM FIGHTING ACTION CLASSIFIER")
    print("=" * 70)

    train_dir = os.path.join(BASE_DIR, "training")
    models_dir = os.path.join(BASE_DIR, "models")
    results_dir = os.path.join(BASE_DIR, "results")
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    # 1. Load Precomputed Sequences
    X_train = np.load(os.path.join(train_dir, "X_train.npy"))
    y_train = np.load(os.path.join(train_dir, "y_train.npy"))
    X_val = np.load(os.path.join(train_dir, "X_val.npy"))
    y_val = np.load(os.path.join(train_dir, "y_val.npy"))

    print(f"Train dataset: {X_train.shape} | Fighting: {int(np.sum(y_train==1))}, Normal: {int(np.sum(y_train==0))}")
    print(f"Val dataset:   {X_val.shape} | Fighting: {int(np.sum(y_val==1))}, Normal: {int(np.sum(y_val==0))}")

    # Convert to PyTorch Tensors
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using computing device: {device}")

    train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32))
    val_dataset = TensorDataset(torch.tensor(X_val, dtype=torch.float32), torch.tensor(y_val, dtype=torch.float32))

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    # 2. Initialize Model
    model = FightingBiLSTM(input_dim=510, hidden_dim=128, num_layers=2, dropout=0.3).to(device)

    # Class weight calculation
    num_pos = np.sum(y_train == 1.0)
    num_neg = np.sum(y_train == 0.0)
    pos_weight = torch.tensor([num_neg / (num_pos + 1e-5)], dtype=torch.float32).to(device)
    print(f"Class distribution: Pos={num_pos}, Neg={num_neg} (pos_weight={pos_weight.item():.2f})")

    criterion = nn.BCELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

    epochs = 25
    best_val_loss = float('inf')
    best_val_f1 = 0.0
    checkpoint_path = os.path.join(models_dir, "fighting_mediapipe_bilstm.pt")

    history = {
        "train_loss": [], "val_loss": [],
        "train_acc": [], "val_acc": [],
        "val_f1": []
    }

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0

        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            optimizer.zero_grad()
            preds, _ = model(batch_x)
            loss = criterion(preds, batch_y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            running_loss += loss.item() * len(batch_y)
            pred_classes = (preds >= 0.50).float()
            correct_train += (pred_classes == batch_y).sum().item()
            total_train += len(batch_y)

        train_loss = running_loss / total_train
        train_acc = correct_train / total_train

        # Validation Phase
        model.eval()
        val_loss_sum = 0.0
        val_preds_all = []
        val_targets_all = []

        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)

                preds, _ = model(batch_x)
                loss = criterion(preds, batch_y)
                val_loss_sum += loss.item() * len(batch_y)

                val_preds_all.extend(preds.cpu().numpy().flatten())
                val_targets_all.extend(batch_y.cpu().numpy().flatten())

        val_loss = val_loss_sum / len(val_targets_all)
        val_preds_np = np.array(val_preds_all) >= 0.50
        val_targets_np = np.array(val_targets_all) >= 0.50
        val_acc = np.mean(val_preds_np == val_targets_np)

        # F1 calculation
        tp = np.sum(val_preds_np & val_targets_np)
        fp = np.sum(val_preds_np & ~val_targets_np)
        fn = np.sum(~val_preds_np & val_targets_np)
        prec = tp / (tp + fp + 1e-6)
        rec = tp / (tp + fn + 1e-6)
        val_f1 = 2 * (prec * rec) / (prec + rec + 1e-6)

        scheduler.step(val_loss)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)
        history["val_f1"].append(val_f1)

        print(f"Epoch [{epoch:02d}/{epochs:02d}] "
              f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc*100:.1f}% | "
              f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc*100:.1f}% | "
              f"Val Prec: {prec*100:.1f}% | Val Rec: {rec*100:.1f}% | Val F1: {val_f1*100:.1f}%")

        # Save Best Model Checkpoint
        if val_f1 >= best_val_f1 or (val_f1 == best_val_f1 and val_loss < best_val_loss):
            best_val_loss = val_loss
            best_val_f1 = val_f1
            torch.save({
                "model_state_dict": model.state_dict(),
                "input_dim": 510,
                "hidden_dim": 128,
                "sequence_length": 30,
                "epoch": epoch,
                "val_loss": val_loss,
                "val_f1": val_f1,
                "val_acc": val_acc
            }, checkpoint_path)
            print(f"   >>> [SAVED CHECKPOINT] New Best Val F1: {val_f1*100:.1f}% at epoch {epoch}")

    print(f"\nTraining Complete! Best model saved to: {checkpoint_path}")

    # Plot Training Curves
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history["train_loss"], label="Train Loss", color="blue")
    plt.plot(history["val_loss"], label="Val Loss", color="red")
    plt.title("Loss Curve (Temporal BiLSTM)")
    plt.xlabel("Epoch")
    plt.ylabel("BCE Loss")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history["train_acc"], label="Train Acc", color="blue")
    plt.plot(history["val_acc"], label="Val Acc", color="red")
    plt.plot(history["val_f1"], label="Val F1", color="green", linestyle="--")
    plt.title("Accuracy & F1 Score")
    plt.xlabel("Epoch")
    plt.ylabel("Score")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()

    curve_path = os.path.join(results_dir, "training_curves.png")
    plt.tight_layout()
    plt.savefig(curve_path, dpi=150)
    plt.close()
    print(f"Saved training curves plot to: {curve_path}")


if __name__ == "__main__":
    train_model()
