import os
import shutil
import random

# ======== CHANGE THIS PATH ========
SOURCE_DIR = r"violence_raw"   # তোমার raw dataset path
DEST_DIR   = r"violence_split" # split dataset তৈরি হবে এখানে
# ==================================

SPLITS = {
    "train": 0.8,
    "val": 0.1,
    "test": 0.1
}

random.seed(42)

def split_class(class_name):
    src_class_path = os.path.join(SOURCE_DIR, class_name)
    videos = os.listdir(src_class_path)

    random.shuffle(videos)

    total = len(videos)
    train_end = int(total * SPLITS["train"])
    val_end = train_end + int(total * SPLITS["val"])

    split_data = {
        "train": videos[:train_end],
        "val": videos[train_end:val_end],
        "test": videos[val_end:]
    }

    for split_name, split_videos in split_data.items():
        dest_path = os.path.join(DEST_DIR, split_name, class_name)
        os.makedirs(dest_path, exist_ok=True)

        for video in split_videos:
            src_video = os.path.join(src_class_path, video)
            dest_video = os.path.join(dest_path, video)
            shutil.copy2(src_video, dest_video)

def main():
    os.makedirs(DEST_DIR, exist_ok=True)

    for class_name in ["violence", "non_violence"]:
        split_class(class_name)

    print("✅ Dataset split complete!")

if __name__ == "__main__":
    main()