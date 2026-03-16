import os
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models

from sklearn.metrics import roc_curve, auc, RocCurveDisplay, confusion_matrix, classification_report
import matplotlib.pyplot as plt

# ---------------- CONFIG ----------------
DATA_DIR = "violence_frames"          # your dataset folder
SPLIT = "val"                         # ROC usually on val/test
MODEL_PATH = "models/violence_v2.pt"  # তোমার latest model path
OUT_DIR = "figures"                   # Overleaf-এর figures/ এ রাখলে সুবিধা
IMG_SIZE = 224
BATCH_SIZE = 64
NUM_WORKERS = 0                       # Windows safe

# ---------------- SETUP ----------------
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", device)

os.makedirs(OUT_DIR, exist_ok=True)

val_dir = Path(DATA_DIR) / SPLIT

val_tfms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

ds = datasets.ImageFolder(val_dir, transform=val_tfms)
loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False,
                    num_workers=NUM_WORKERS, pin_memory=True)

print("Class mapping:", ds.class_to_idx)

# ---------------- LOAD MODEL ----------------
ckpt = torch.load(MODEL_PATH, map_location="cpu")
class_to_idx = ckpt.get("class_to_idx", ds.class_to_idx)
img_size = ckpt.get("img_size", IMG_SIZE)

# MobileNetV3 Large same as training
model = models.mobilenet_v3_large(weights=None)
model.classifier[3] = nn.Linear(model.classifier[3].in_features, 2)
model.load_state_dict(ckpt["model_state"])
model = model.to(device)
model.eval()

# ---------------- INFERENCE ----------------
all_probs = []
all_true = []

with torch.no_grad():
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        logits = model(x)
        probs = torch.softmax(logits, dim=1)[:, 1]  # probability of class 1
        all_probs.append(probs.detach().cpu().numpy())
        all_true.append(y.numpy())

y_prob = np.concatenate(all_probs)
y_true = np.concatenate(all_true)

# Ensure "violence" is class 1 (if dataset sorted differently)
# ImageFolder sorts alphabetically -> usually: non_violence=0, violence=1
# If yours is reversed, uncomment the next lines:
# if list(ds.class_to_idx.keys()) == ["violence", "non_violence"]:
#     y_true = 1 - y_true
#     y_prob = 1 - y_prob

# ---------------- ROC + AUC ----------------
fpr, tpr, thresholds = roc_curve(y_true, y_prob)
roc_auc = auc(fpr, tpr)
print("AUC:", roc_auc)

plt.figure()
RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc).plot()
plt.title("ROC Curve (Violence CNN)")
roc_path = os.path.join(OUT_DIR, "roc_violence.png")
plt.savefig(roc_path, dpi=300, bbox_inches="tight")
plt.close()
print("Saved:", roc_path)

# ---------------- OPTIONAL: Confusion Matrix at 0.5 threshold ----------------
y_pred = (y_prob >= 0.5).astype(int)
cm = confusion_matrix(y_true, y_pred)
print("Confusion Matrix:\n", cm)
print(classification_report(y_true, y_pred, target_names=ds.classes))

# Plot confusion matrix nicely
plt.figure()
plt.imshow(cm)
plt.title("Confusion Matrix (threshold=0.5)")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.xticks([0,1], ds.classes, rotation=15)
plt.yticks([0,1], ds.classes)
for i in range(2):
    for j in range(2):
        plt.text(j, i, cm[i, j], ha="center", va="center")
cm_path = os.path.join(OUT_DIR, "cm_violence.png")
plt.savefig(cm_path, dpi=300, bbox_inches="tight")
plt.close()
print("Saved:", cm_path)