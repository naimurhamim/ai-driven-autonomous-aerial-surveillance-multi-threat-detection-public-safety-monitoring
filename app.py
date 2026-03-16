import time
import cv2
import os
from collections import deque

from detectors.yolo_detector import YOLODetector
from utils.telegram_notify import TelegramNotifier
from utils.evidence import save_evidence
from dotenv import load_dotenv

# ---- Violence classifier deps ----
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, models

load_dotenv()

# =========================
# CONFIG
# =========================
TELEGRAM_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID")
if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
    raise RuntimeError("Missing TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID in environment variables")

DRONE_NAME = "SB-001"
DRONE_LOCATION = "UFTB Zone-A"

FIRE_MODEL = "models/fire_v1.pt"
ACCIDENT_MODEL = "models/accident_v1.pt"
WEAPON_MODEL = "models/weapon_v2.1.pt"

# COCO model for person/vehicle verification
VERIFY_MODEL = "yolo11n.pt"

# Violence classifier (frame-based)
VIOLENCE_MODEL = "models/violence_v1.pt"  # ensure exists

# ---------- COOLDOWNS ----------
COOLDOWN_SEC = 10
VIOL_COOLDOWN_SEC = 20

# ---------- Weapon gating policy ----------
# True  => only alert if person exists (fewer false alarms)
# False => allow no-person weapon alerts ONLY when very high conf (balanced)
REQUIRE_PERSON_FOR_WEAPON = True
WEAPON_NO_PERSON_MIN_CONF = 0.80

# ---------- Confirmations ----------
SMOKE_CONFIRM_FRAMES = 2
WEAPON_CONFIRM_FRAMES = 2

# ---------- Violence event settings ----------
VIOL_SAMPLE_EVERY_N_FRAMES = 3
VIOL_WINDOW = 15
VIOL_MIN_HITS = 9
VIOL_THRESH = 0.80

SHOW_NEAR_VIOLENCE = True
NEAR_VIOL_MARGIN = 0.10

# ---------- Video sources ----------
VIDEO_SOURCES = [
    # r"videos\Accident_Detection1.mp4",
    # r"videos\FireSmoke1.mp4",
    # r"videos\Violence1.mp4",
    # r"videos\Violence2.mp4",
]
DEFAULT_WEBCAM_INDEX = 0


# =========================
# INIT
# =========================
notifier = TelegramNotifier(TELEGRAM_TOKEN, TELEGRAM_CHAT_ID)

fire_det = YOLODetector(FIRE_MODEL, conf=0.55)
acc_det = YOLODetector(ACCIDENT_MODEL, conf=0.55)
weapon_det = YOLODetector(WEAPON_MODEL, conf=0.55)
verify_det = YOLODetector(VERIFY_MODEL, conf=0.35)

last_sent = {"fire": 0, "accident": 0, "weapon": 0, "violence": 0}

smoke_hits = 0
weapon_hits = 0
frame_i = 0

PERSON_SET = {"person"}
VEHICLE_SET = {"car", "motorcycle", "bus", "truck"}  # COCO names


def now():
    return time.time()


def should_send(key: str, cooldown: int = COOLDOWN_SEC) -> bool:
    return (now() - last_sent[key]) >= cooldown


def mark_sent(key: str):
    last_sent[key] = now()


def get_detected_classes(results):
    r = results[0]
    if r.boxes is None or len(r.boxes) == 0:
        return []
    names = r.names
    return [names[int(c)] for c in r.boxes.cls.tolist()]


def get_best_detection(results):
    """Returns (best_name, best_conf) from highest-confidence box. If no boxes => (None, None)"""
    r = results[0]
    if r.boxes is None or len(r.boxes) == 0:
        return None, None

    confs = r.boxes.conf.tolist()
    clss = r.boxes.cls.tolist()
    best_i = int(max(range(len(confs)), key=lambda i: confs[i]))
    best_conf = float(confs[best_i])
    best_cls = int(clss[best_i])
    best_name = r.names.get(best_cls, str(best_cls))
    return best_name, best_conf


def has_any(results, target_names: set) -> bool:
    classes = get_detected_classes(results)
    return any(c in target_names for c in classes)


def build_caption(detect_group: str, detected_list: list, best_name=None, best_conf=None, extra_lines: str = ""):
    detected_unique = sorted(set(detected_list))

    if detect_group == "accident":
        if "severe" in detected_unique:
            detection_name, typ = "ACCIDENT", "severe"
        elif "moderate" in detected_unique:
            detection_name, typ = "ACCIDENT", "moderate"
        else:
            detection_name = "ACCIDENT"
            typ = ", ".join(detected_unique) if detected_unique else "unknown"

    elif detect_group == "weapon":
        detection_name = "WEAPON"
        typ = best_name if best_name else (", ".join(detected_unique) if detected_unique else "unknown")

    elif detect_group == "violence":
        detection_name = "VIOLENCE"
        typ = "event"

    else:
        # fire/smoke
        if "fire" in detected_unique and "smoke" in detected_unique:
            detection_name, typ = "FIRE & SMOKE", "both"
        elif "fire" in detected_unique:
            detection_name, typ = "FIRE", "fire"
        elif "smoke" in detected_unique:
            detection_name, typ = "SMOKE", "smoke"
        else:
            detection_name = "FIRE/SMOKE"
            typ = ", ".join(detected_unique) if detected_unique else "unknown"

    conf_line = ""
    if best_conf is not None:
        conf_line = f"Confidence: {best_conf * 100:.1f}%\n"

    caption = (
        f"🚨 ALERT!\n\n"
        f"Drone Name: {DRONE_NAME}\n"
        f"Location: {DRONE_LOCATION}\n\n"
        f"Detection: {detection_name}\n"
        f"Type: {typ}\n"
        f"{conf_line}"
    )
    if extra_lines:
        caption += extra_lines.strip() + "\n"
    return caption


# =========================
# VIOLENCE EVENT CLASSIFIER
# =========================
class ViolenceEventDetector:
    def __init__(self, model_path: str):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.img_size = 224

        # same arch as training
        self.model = models.mobilenet_v3_large(weights=None)
        self.model.classifier[3] = nn.Linear(self.model.classifier[3].in_features, 2)

        ckpt = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(ckpt["model_state"])
        self.class_to_idx = ckpt.get("class_to_idx", {"non_violence": 0, "violence": 1})
        self.violence_idx = self.class_to_idx.get("violence", 1)

        self.model.to(self.device).eval()

        self.tfm = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((self.img_size, self.img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

        self.hits = deque(maxlen=VIOL_WINDOW)

    def predict_prob(self, bgr_frame) -> float:
        x = self.tfm(bgr_frame).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits = self.model(x)
            prob = F.softmax(logits, dim=1)[0, self.violence_idx].item()
        return prob

    def update(self, bgr_frame):
        """Return (triggered:bool, prob:float, hits_sum:int, hits_len:int)"""
        prob = self.predict_prob(bgr_frame)
        hit = 1 if prob >= VIOL_THRESH else 0
        self.hits.append(hit)

        hs = sum(self.hits)
        hl = len(self.hits)

        triggered = (hl == VIOL_WINDOW and hs >= VIOL_MIN_HITS)
        return triggered, prob, hs, hl

    def clear(self):
        self.hits.clear()


viol_ev = ViolenceEventDetector(VIOLENCE_MODEL)


# =========================
# PROCESS ONE SOURCE
# =========================
def process_source(source):
    global smoke_hits, weapon_hits, frame_i

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"❌ Cannot open source: {source}")
        return False  # continue to next source

    has_person = False
    has_vehicle = False

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        frame_i = (frame_i + 1) % 100000

        # We'll show ONLY confirmed detection overlay
        display_frame = frame.copy()
        status_text = ""  # only set when confirmed
        status_color = (0, 0, 255)  # red default

        # -------- Verify (COCO) every 3 frames --------
        if frame_i % 3 == 0:
            verify_res = verify_det.infer(frame)
            has_person = has_any(verify_res, PERSON_SET)
            has_vehicle = has_any(verify_res, VEHICLE_SET)

        # ---------------- FIRE/SMOKE ----------------
        fire_res = fire_det.infer(frame)
        fire_classes = get_detected_classes(fire_res)
        fire_best_name, fire_best_conf = get_best_detection(fire_res)
        fire_caption = None

        if "fire" in fire_classes:
            fire_caption = build_caption("fire", fire_classes, best_name=fire_best_name, best_conf=fire_best_conf)
            smoke_hits = 0
        elif "smoke" in fire_classes:
            smoke_hits += 1
            if smoke_hits >= SMOKE_CONFIRM_FRAMES:
                fire_caption = build_caption("fire", fire_classes, best_name=fire_best_name, best_conf=fire_best_conf)
                smoke_hits = 0
        else:
            smoke_hits = 0

        if fire_caption and should_send("fire"):
            annotated = fire_res[0].plot()
            path = save_evidence(annotated, prefix="fire")
            notifier.send_photo(path, caption=fire_caption)
            mark_sent("fire")

            display_frame = annotated
            status_text = "CONFIRMED: FIRE/SMOKE"
            status_color = (0, 0, 255)

        # ---------------- ACCIDENT ----------------
        acc_res = acc_det.infer(frame)
        acc_classes = get_detected_classes(acc_res)
        acc_best_name, acc_best_conf = get_best_detection(acc_res)

        # accident only if vehicle present (reduces false alarms)
        if acc_classes and has_vehicle and should_send("accident"):
            caption = build_caption("accident", acc_classes, best_name=acc_best_name, best_conf=acc_best_conf)
            annotated = acc_res[0].plot()
            path = save_evidence(annotated, prefix="accident")
            notifier.send_photo(path, caption=caption)
            mark_sent("accident")

            display_frame = annotated
            status_text = "CONFIRMED: ACCIDENT"
            status_color = (0, 0, 255)

        # ---------------- WEAPON ----------------
        weapon_res = weapon_det.infer(frame)
        weapon_classes = get_detected_classes(weapon_res)
        weapon_best_name, weapon_best_conf = get_best_detection(weapon_res)

        weapon_caption = None
        if weapon_classes:
            weapon_hits += 1
            if weapon_hits >= WEAPON_CONFIRM_FRAMES:
                weapon_caption = build_caption("weapon", weapon_classes, best_name=weapon_best_name, best_conf=weapon_best_conf)
                weapon_hits = 0
        else:
            weapon_hits = 0

        if weapon_caption and should_send("weapon"):
            if REQUIRE_PERSON_FOR_WEAPON:
                weapon_ok = has_person
            else:
                weapon_ok = has_person or (weapon_best_conf is not None and weapon_best_conf >= WEAPON_NO_PERSON_MIN_CONF)

            if weapon_ok:
                annotated = weapon_res[0].plot()
                path = save_evidence(annotated, prefix="weapon")
                notifier.send_photo(path, caption=weapon_caption)
                mark_sent("weapon")

                display_frame = annotated
                status_text = "CONFIRMED: WEAPON"
                status_color = (0, 0, 255)

        # ---------------- VIOLENCE (EVENT) ----------------
        # IMPORTANT: We do NOT show violence text always.
        # We'll only show if near/confirmed. And only alert when confirmed window.
        if frame_i % VIOL_SAMPLE_EVERY_N_FRAMES == 0:
            triggered, prob, hs, hl = viol_ev.update(frame)

            near = SHOW_NEAR_VIOLENCE and (prob >= (VIOL_THRESH - NEAR_VIOL_MARGIN) or hs > 0)

            # confirmed send
            if triggered and should_send("violence", cooldown=VIOL_COOLDOWN_SEC):
                extra = f"Window: {hs}/{VIOL_WINDOW} hits (thr={VIOL_THRESH})"
                caption = build_caption("violence", ["violence"], best_conf=prob, extra_lines=extra)

                # Save current display_frame (or raw frame) as evidence
                path = save_evidence(frame, prefix="violence")
                notifier.send_photo(path, caption=caption)
                last_sent["violence"] = now()
                viol_ev.clear()

                # show confirmed on screen
                status_text = "CONFIRMED: VIOLENCE"
                status_color = (0, 0, 255)

            elif near and not status_text:
                # only show if nothing else already confirmed this frame
                status_text = f"CHECKING: VIOLENCE  Prob:{prob:.2f}  Hits:{hs}/{hl}"
                status_color = (0, 165, 255)  # orange-ish

        # ---------------- DISPLAY ----------------
        if status_text:
            cv2.putText(
                display_frame,
                status_text,
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                status_color,
                2
            )

        cv2.imshow("Surveillance", display_frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:
            cap.release()
            cv2.destroyAllWindows()
            return True  # user aborted all

    cap.release()
    cv2.destroyAllWindows()
    return False  # finished this source normally


def main():
    # If list empty, use webcam
    sources = VIDEO_SOURCES[:] if VIDEO_SOURCES else [DEFAULT_WEBCAM_INDEX]

    for src in sources:
        print(f"\n▶ Running source: {src}")
        aborted = process_source(src)
        if aborted:
            print("🛑 Stopped by user (q/esc).")
            break

    print("✅ Done.")


if __name__ == "__main__":
    main()