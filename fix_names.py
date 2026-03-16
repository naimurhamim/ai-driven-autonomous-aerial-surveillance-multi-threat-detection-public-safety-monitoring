import os
import torch

SRC = os.path.join("models", "fire_smoke_yolo11.pt")
DST = os.path.join("models", "fire_smoke_yolo11_fixed.pt")

if not os.path.exists(SRC):
    raise FileNotFoundError(f"Source model not found: {SRC}")

ckpt = torch.load(SRC, map_location="cpu")

names = {0: "smoke", 1: "fire"}

if isinstance(ckpt, dict):
    ckpt["names"] = names
    model = ckpt.get("model", None)
    if model is not None and hasattr(model, "names"):
        model.names = names

torch.save(ckpt, DST)

print("✅ Source:", SRC)
print("✅ Saved :", DST)
