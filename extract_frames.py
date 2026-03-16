import os
import cv2

# ===== CHANGE THIS =====
VIDEO_ROOT = "violence_split"
FRAME_ROOT = "violence_frames"
FPS = 1   # প্রতি সেকেন্ডে ১টা ফ্রেম
SIZE = 224  # resize 224x224
# =======================

def extract_from_video(video_path, save_dir):
    cap = cv2.VideoCapture(video_path)
    video_name = os.path.splitext(os.path.basename(video_path))[0]

    frame_count = 0
    saved_count = 0
    fps_video = cap.get(cv2.CAP_PROP_FPS)

    interval = int(fps_video // FPS) if fps_video > FPS else 1

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % interval == 0:
            frame = cv2.resize(frame, (SIZE, SIZE))
            save_path = os.path.join(save_dir, f"{video_name}_{saved_count}.jpg")
            cv2.imwrite(save_path, frame)
            saved_count += 1

        frame_count += 1

    cap.release()


def process_split(split):
    for class_name in ["violence", "non_violence"]:
        video_dir = os.path.join(VIDEO_ROOT, split, class_name)
        frame_dir = os.path.join(FRAME_ROOT, split, class_name)

        os.makedirs(frame_dir, exist_ok=True)

        videos = os.listdir(video_dir)

        for video in videos:
            video_path = os.path.join(video_dir, video)
            print(f"Processing: {video_path}")
            extract_from_video(video_path, frame_dir)


def main():
    for split in ["train", "val", "test"]:
        process_split(split)

    print("✅ Frame extraction complete!")


if __name__ == "__main__":
    main()