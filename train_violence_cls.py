import os
import time
from pathlib import Path
from datetime import datetime
import csv

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models

# ---------------- CONFIG ----------------
DATA_DIR = "violence_frames"  # <-- your folder
IMG_SIZE = 224
BATCH_SIZE = 64
EPOCHS = 10
LR = 3e-4
NUM_WORKERS = 0  # windows safe

# IMPORTANT: old models/violence_v1.pt যেন overwrite না হয়
MODEL_DIR = "models"
RUNS_DIR = "runs"

BASE_NAME = "violence_v2"   # <-- new version name (v1 থাকবে intact)
MODEL_OUT = f"{MODEL_DIR}/{BASE_NAME}.pt"

# ---------------------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", device)

train_dir = Path(DATA_DIR) / "train"
val_dir   = Path(DATA_DIR) / "val"

# torchvision ImageFolder sorts class names alphabetically
train_tfms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

val_tfms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

train_ds = datasets.ImageFolder(train_dir, transform=train_tfms)
val_ds   = datasets.ImageFolder(val_dir, transform=val_tfms)

print("Class mapping:", train_ds.class_to_idx)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                          num_workers=NUM_WORKERS, pin_memory=True)
val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                          num_workers=NUM_WORKERS, pin_memory=True)

# -------- Model (MobileNetV3 Large) -------
model = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.DEFAULT)
model.classifier[3] = nn.Linear(model.classifier[3].in_features, 2)
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

scaler = torch.cuda.amp.GradScaler(enabled=(device == "cuda"))

def run_eval():
    model.eval()
    correct, total = 0, 0
    val_loss = 0.0
    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            with torch.cuda.amp.autocast(enabled=(device == "cuda")):
                logits = model(x)
                loss = criterion(logits, y)
            val_loss += loss.item() * x.size(0)
            preds = logits.argmax(1)
            correct += (preds == y).sum().item()
            total += y.size(0)
    return val_loss / max(total, 1), correct / max(total, 1)

def ensure_dirs():
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(RUNS_DIR, exist_ok=True)

def get_run_paths():
    """
    Each run creates unique CSV + plot names, so logs won't overwrite.
    """
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_csv = os.path.join(RUNS_DIR, f"{BASE_NAME}_log_{stamp}.csv")
    loss_png = os.path.join(RUNS_DIR, f"{BASE_NAME}_loss_{stamp}.png")
    acc_png  = os.path.join(RUNS_DIR, f"{BASE_NAME}_acc_{stamp}.png")
    return log_csv, loss_png, acc_png

def write_csv_header(csv_path):
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["epoch", "train_loss", "val_loss", "val_acc", "sec"])

def append_csv_row(csv_path, epoch, train_loss, val_loss, val_acc, sec):
    with open(csv_path, "a", newline="") as f:
        w = csv.writer(f)
        w.writerow([epoch, train_loss, val_loss, val_acc, sec])

def plot_from_csv(csv_path, loss_png, acc_png):
    """
    Reads CSV and saves 2 graphs.
    """
    import matplotlib.pyplot as plt

    epochs, train_loss, val_loss, val_acc = [], [], [], []
    with open(csv_path, "r") as f:
        r = csv.DictReader(f)
        for row in r:
            epochs.append(int(row["epoch"]))
            train_loss.append(float(row["train_loss"]))
            val_loss.append(float(row["val_loss"]))
            val_acc.append(float(row["val_acc"]))

    plt.figure()
    plt.plot(epochs, train_loss, label="Train Loss")
    plt.plot(epochs, val_loss, label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(loss_png, dpi=200)

    plt.figure()
    plt.plot(epochs, val_acc, label="Val Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.tight_layout()
    plt.savefig(acc_png, dpi=200)

    print("📈 Saved plots:")
    print(" -", loss_png)
    print(" -", acc_png)

def save_best(model_path, best_acc_value):
    torch.save({
        "model_state": model.state_dict(),
        "class_to_idx": train_ds.class_to_idx,
        "img_size": IMG_SIZE,
        "best_acc": best_acc_value
    }, model_path)

def main():
    ensure_dirs()

    # Unique log/plot names per run
    log_csv, loss_png, acc_png = get_run_paths()
    write_csv_header(log_csv)
    print("🧾 Logging to:", log_csv)

    # Safety: if same MODEL_OUT exists, don't overwrite—make a backup
    if os.path.exists(MODEL_OUT):
        backup = MODEL_OUT.replace(".pt", f"_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pt")
        os.rename(MODEL_OUT, backup)
        print("⚠️ Existing model found. Renamed old file to:", backup)

    best_acc = 0.0

    for epoch in range(1, EPOCHS + 1):
        t0 = time.time()
        model.train()
        train_loss = 0.0
        seen = 0

        for x, y in train_loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=(device == "cuda")):
                logits = model(x)
                loss = criterion(logits, y)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item() * x.size(0)
            seen += x.size(0)

        train_loss = train_loss / max(seen, 1)
        val_loss, val_acc = run_eval()
        dt = time.time() - t0

        print(f"Epoch {epoch}/{EPOCHS} | "
              f"train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | val_acc={val_acc:.4f} | {dt:.1f}s")

        # log row
        append_csv_row(log_csv, epoch, train_loss, val_loss, val_acc, dt)

        # save best model (to violence_v2.pt)
        if val_acc > best_acc:
            best_acc = val_acc
            save_best(MODEL_OUT, best_acc)
            print("✅ Saved best:", MODEL_OUT)

    print("Done. Best val_acc:", best_acc)

    # Generate plots at the end
    plot_from_csv(log_csv, loss_png, acc_png)

if __name__ == "__main__":
    main()