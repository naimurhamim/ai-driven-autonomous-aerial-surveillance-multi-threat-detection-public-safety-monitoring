import os
import time
import cv2
import torch
import torch.nn.functional as F
from collections import deque
from torchvision import transforms, models
import torch.nn as nn

from utils.telegram_notify import TelegramNotifier
from utils.evidence import save_evidence
from dotenv import load_dotenv

load_dotenv()

# ---------- TELEGRAM ----------
TELEGRAM_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID")
if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
    raise RuntimeError("Missing TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID in environment variables")

notifier = TelegramNotifier(TELEGRAM_TOKEN, TELEGRAM_CHAT_ID)

# ---------- DRONE INFO ----------
DRONE_NAME = "SB-001"
DRONE_LOCATION = "UFTB Zone-A"

# ---------- MODEL ----------
MODEL_PATH = "models/violence_v1.pt"
IMG_SIZE = 224

device = "cuda" if torch.cuda.is_available() else "cpu"

# Build same architecture as training
model = models.mobilenet_v3_large(weights=None)
model.classifier[3] = nn.Linear(model.classifier[3].in_features, 2)

ckpt = torch.load(MODEL_PATH, map_location=device)
model.load_state_dict(ckpt["model_state"])
class_to_idx = ckpt.get("class_to_idx", {"non_violence": 0, "violence": 1})

# Get index for violence class
violence_idx = class_to_idx.get("violence", 1)

model.to(device).eval()

tfm = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# ---------- EVENT LOGIC ----------
WINDOW = 15               # last 15 frames
THRESH = 0.80             # violence prob threshold
MIN_HITS = 9              # need >=9 hits in window
COOLDOWN_SEC = 20         # after an alert, wait 20 seconds
SAMPLE_EVERY_N_FRAMES = 3 # evaluate every 3rd frame (speed + stability)

hits = deque(maxlen=WINDOW)
last_sent = 0
frame_i = 0


def can_send():
    return (time.time() - last_sent) >= COOLDOWN_SEC


def predict_violence_prob(bgr_frame):
    x = tfm(bgr_frame).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(x)
        prob = F.softmax(logits, dim=1)[0, violence_idx].item()
    return prob


def main():
    global last_sent, frame_i

    cap = cv2.VideoCapture(0)  # webcam; video file হলে path দাও

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        frame_i += 1

        # only evaluate every N frames
        if frame_i % SAMPLE_EVERY_N_FRAMES == 0:
            prob = predict_violence_prob(frame)
            is_hit = prob >= THRESH
            hits.append(1 if is_hit else 0)

            # show overlay
            cv2.putText(frame, f"ViolenceProb: {prob:.2f}  Hits:{sum(hits)}/{len(hits)}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0) if is_hit else (0, 0, 255), 2)

            # event trigger
            if len(hits) == WINDOW and sum(hits) >= MIN_HITS and can_send():
                caption = (
                    "🚨 ALERT!\n\n"
                    f"Drone Name: {DRONE_NAME}\n"
                    f"Location: {DRONE_LOCATION}\n\n"
                    "Detection: VIOLENCE\n"
                    "Type: event\n"
                    f"Confidence: {prob*100:.1f}%\n"
                    f"Window: {sum(hits)}/{WINDOW} hits (thr={THRESH})"
                )

                path = save_evidence(frame, prefix="violence")
                notifier.send_photo(path, caption=caption)
                last_sent = time.time()
                hits.clear()  # reset window after alert

        cv2.imshow("Violence Event Detection", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()