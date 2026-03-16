import os, time
import cv2

def save_evidence(frame, out_dir="data/captures", prefix="event"):
    os.makedirs(out_dir, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    path = os.path.join(out_dir, f"{prefix}_{ts}.jpg")
    cv2.imwrite(path, frame)
    return path