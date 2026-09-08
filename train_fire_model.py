import os
import random
import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

from backend.app.ai.fire_detector import FireNet

# Ensure reproducible split
random.seed(42)
torch.manual_seed(42)

# Strict Video-Level Partitioning to Prevent Data Leakage (Section 9)
TRAIN_VIDEOS = {
    "fire": ["Explosion023_x264A.mp4", "Explosion041_x264A.mp4", "Explosion051_x264A.mp4"],
    "no_fire": ["normal_store_cctv.mp4", "normal_worker_cctv.mp4", "normal_traffic_cctv.mp4"]
}

VAL_VIDEOS = {
    "fire": ["Explosion027_x264A.mp4"],
    "no_fire": ["normal_classroom_cctv.mp4"]
}

TEST_VIDEOS = {
    "fire": ["Explosion005_x264A.mp4"],
    "no_fire": ["normal_pedestrian_cctv.mp4"]
}

def extract_frames_from_video(video_path: str, sample_interval: int = 15, max_frames: int = 200) -> list:
    """Extract frames at regular intervals from a video."""
    frames = []
    if not os.path.exists(video_path):
        print(f"Warning: Video not found at {video_path}")
        return frames

    cap = cv2.VideoCapture(video_path)
    idx = 0
    while cap.isOpened() and len(frames) < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        if idx % sample_interval == 0:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(Image.fromarray(rgb))
        idx += 1
    cap.release()
    return frames

class FireDataset(Dataset):
    def __init__(self, samples: list, transform=None):
        """
        samples: list of tuples (PIL.Image or file_path, label)
        """
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_item, label = self.samples[idx]
        if isinstance(img_item, str):
            img = Image.open(img_item).convert('RGB')
        else:
            img = img_item

        if self.transform:
            img = self.transform(img)

        return img, label

def prepare_splits():
    print("=" * 60)
    print("PREPARING DATASET SPLITS (VIDEO-LEVEL PARTITION)")
    print("=" * 60)

    train_samples = []
    val_samples = []
    test_samples = []

    # 1. Fire training videos
    for vname in TRAIN_VIDEOS["fire"]:
        vpath = os.path.join("dataset", "fire", vname)
        frames = extract_frames_from_video(vpath, sample_interval=12, max_frames=120)
        print(f"Extracted {len(frames)} training frames from fire video: {vname}")
        train_samples.extend([(f, 1) for f in frames])

    # Also add sample images from suspicious uploads/fire and smoke to train set
    smoke_dir = os.path.join("suspicious uploads", "fire and smoke")
    if os.path.exists(smoke_dir):
        smoke_imgs = [os.path.join(smoke_dir, f) for f in os.listdir(smoke_dir) if f.endswith('.jpg')]
        # take 50 images for training
        train_samples.extend([(p, 1) for p in smoke_imgs[:50]])
        print(f"Added {len(smoke_imgs[:50])} still fire/smoke images to train set.")

    # 2. No-fire training videos
    for vname in TRAIN_VIDEOS["no_fire"]:
        vpath = os.path.join("dataset", "no_fire", vname)
        frames = extract_frames_from_video(vpath, sample_interval=15, max_frames=120)
        print(f"Extracted {len(frames)} training frames from normal video: {vname}")
        train_samples.extend([(f, 0) for f in frames])

    # 3. Validation videos
    for vname in VAL_VIDEOS["fire"]:
        vpath = os.path.join("dataset", "fire", vname)
        frames = extract_frames_from_video(vpath, sample_interval=15, max_frames=60)
        print(f"Extracted {len(frames)} validation frames from fire video: {vname}")
        val_samples.extend([(f, 1) for f in frames])

    for vname in VAL_VIDEOS["no_fire"]:
        vpath = os.path.join("dataset", "no_fire", vname)
        frames = extract_frames_from_video(vpath, sample_interval=15, max_frames=60)
        print(f"Extracted {len(frames)} validation frames from normal video: {vname}")
        val_samples.extend([(f, 0) for f in frames])

    # 4. Test videos
    for vname in TEST_VIDEOS["fire"]:
        vpath = os.path.join("dataset", "fire", vname)
        frames = extract_frames_from_video(vpath, sample_interval=15, max_frames=60)
        print(f"Extracted {len(frames)} test frames from fire video: {vname}")
        test_samples.extend([(f, 1) for f in frames])

    for vname in TEST_VIDEOS["no_fire"]:
        vpath = os.path.join("dataset", "no_fire", vname)
        frames = extract_frames_from_video(vpath, sample_interval=15, max_frames=60)
        print(f"Extracted {len(frames)} test frames from normal video: {vname}")
        test_samples.extend([(f, 0) for f in frames])

    random.shuffle(train_samples)
    random.shuffle(val_samples)
    random.shuffle(test_samples)

    print(f"\nFinal Split Summary:")
    print(f"  Training samples:   {len(train_samples)} (Fire: {sum(1 for _, l in train_samples if l == 1)}, Normal: {sum(1 for _, l in train_samples if l == 0)})")
    print(f"  Validation samples: {len(val_samples)} (Fire: {sum(1 for _, l in val_samples if l == 1)}, Normal: {sum(1 for _, l in val_samples if l == 0)})")
    print(f"  Test samples:       {len(test_samples)} (Fire: {sum(1 for _, l in test_samples if l == 1)}, Normal: {sum(1 for _, l in test_samples if l == 0)})")

    return train_samples, val_samples, test_samples

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nUsing compute device: {device}")

    train_samples, val_samples, test_samples = prepare_splits()

    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_dataset = FireDataset(train_samples, transform=train_transform)
    val_dataset = FireDataset(val_samples, transform=val_transform)
    test_dataset = FireDataset(test_samples, transform=val_transform)

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

    model = FireNet(pretrained=True).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    best_val_acc = 0.0
    epochs = 4
    output_dir = "models"
    os.makedirs(output_dir, exist_ok=True)
    model_save_path = os.path.join(output_dir, "best_fire_detector.pt")

    print("\n" + "=" * 60)
    print("STARTING PYTORCH FIRE DETECTION TRAINING")
    print("=" * 60)

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        correct_train = 0
        total_train = 0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * images.size(0)
            preds = torch.argmax(outputs, dim=1)
            correct_train += (preds == labels).sum().item()
            total_train += labels.size(0)

        train_loss = total_loss / max(1, total_train)
        train_acc = (correct_train / max(1, total_train)) * 100.0

        # Evaluate on held-out validation video
        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images)
                preds = torch.argmax(outputs, dim=1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)

        val_acc = (val_correct / max(1, val_total)) * 100.0

        print(f"Epoch [{epoch}/{epochs}] - Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.1f}% | Val Acc: {val_acc:.1f}%")

        if val_acc >= best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), model_save_path)
            print(f"  -> Saved best model checkpoint to {model_save_path} (Val Acc: {val_acc:.1f}%)")

    # Final Genuine Test Evaluation on Held-Out Test Split
    print("\n" + "=" * 60)
    print("EVALUATING ON HELD-OUT TEST SPLIT (Zero Data Leakage)")
    print("=" * 60)
    model.load_state_dict(torch.load(model_save_path, map_location=device))
    model.eval()

    test_correct = 0
    test_total = 0
    y_true = []
    y_pred = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1)
            test_correct += (preds == labels).sum().item()
            test_total += labels.size(0)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())

    test_acc = (test_correct / max(1, test_total)) * 100.0
    print(f"Genuine Held-Out Test Accuracy: {test_acc:.2f}% ({test_correct}/{test_total} frames correct)")
    print("Model training & validation completed successfully.")

if __name__ == "__main__":
    train()
