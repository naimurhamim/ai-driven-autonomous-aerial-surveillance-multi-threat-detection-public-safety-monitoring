import os, glob, shutil, xml.etree.ElementTree as ET
from pathlib import Path

# ====== CHANGE THIS ======
ROOT = r"OD-WeaponDetection-master"  # তোমার আসল path বসাও
OUT  = r"weapon_v2_dataset"                                   # output path বসাও
# =========================

# only these will be treated as WEAPON positives
WEAPON_NAMES = {"knife", "pistol", "gun", "handgun", "pistola", "cuchillo", "weapon"}

IMG_EXTS = (".jpg", ".jpeg", ".png")

def ensure(p): os.makedirs(p, exist_ok=True)

def yolo_line(cls, xmin, ymin, xmax, ymax, w, h):
    xmin, ymin = max(0, xmin), max(0, ymin)
    xmax, ymax = min(w-1, xmax), min(h-1, ymax)
    bw, bh = max(0, xmax-xmin), max(0, ymax-ymin)
    xc, yc = xmin + bw/2, ymin + bh/2
    return f"{cls} {xc/w:.6f} {yc/h:.6f} {bw/w:.6f} {bh/h:.6f}"

def parse_voc(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    size = root.find("size")
    w = int(size.findtext("width"))
    h = int(size.findtext("height"))
    objs = []
    for obj in root.findall("object"):
        name = (obj.findtext("name") or "").strip().lower()
        b = obj.find("bndbox")
        xmin = int(float(b.findtext("xmin")))
        ymin = int(float(b.findtext("ymin")))
        xmax = int(float(b.findtext("xmax")))
        ymax = int(float(b.findtext("ymax")))
        objs.append((name, xmin, ymin, xmax, ymax))
    return w, h, objs

def find_image(img_dir, base):
    for ext in IMG_EXTS:
        p = os.path.join(img_dir, base + ext)
        if os.path.exists(p): return p
    hits = glob.glob(os.path.join(img_dir, base + ".*"))
    hits = [h for h in hits if os.path.splitext(h)[1].lower() in IMG_EXTS]
    return hits[0] if hits else None

def add_pair(img_dir, ann_dir, split):
    # split: train বা val
    out_img_dir = os.path.join(OUT, "images", split)
    out_lbl_dir = os.path.join(OUT, "labels", split)
    ensure(out_img_dir); ensure(out_lbl_dir)

    xmls = glob.glob(os.path.join(ann_dir, "*.xml"))
    for x in xmls:
        base = os.path.splitext(os.path.basename(x))[0]
        img = find_image(img_dir, base)
        if not img:
            continue

        w, h, objs = parse_voc(x)

        lines = []
        for name, xmin, ymin, xmax, ymax in objs:
            # ONLY knife/pistol/gun/weapon will be positive
            if name in WEAPON_NAMES:
                lines.append(yolo_line(0, xmin, ymin, xmax, ymax, w, h))

        # copy image
        out_img = os.path.join(out_img_dir, os.path.basename(img))
        if not os.path.exists(out_img):
            shutil.copy2(img, out_img)

        # write label (empty => negative sample)
        out_lbl = os.path.join(out_lbl_dir, base + ".txt")
        with open(out_lbl, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

def main():
    # 1) Knife_detection (Images + annotations)
    add_pair(
        img_dir=os.path.join(ROOT, "Knife_detection", "Images"),
        ann_dir=os.path.join(ROOT, "Knife_detection", "annotations"),
        split="train"
    )

    # 2) Pistol detection (Weapons + xmls)
    add_pair(
        img_dir=os.path.join(ROOT, "Pistol detection", "Weapons"),
        ann_dir=os.path.join(ROOT, "Pistol detection", "xmls"),
        split="train"
    )

    # 3) Sohas_weapon-Detection (images/annotations + images_test/annotations_test)
    sohas = os.path.join(ROOT, "Weapons and similar handled objects", "Sohas_weapon-Detection")
    # Sohas train
    add_pair(
        img_dir=os.path.join(sohas, "images"),
        ann_dir=os.path.join(sohas, "annotations", "xmls"),
        split="train"
    )

    # Sohas test -> val
    add_pair(
        img_dir=os.path.join(sohas, "images_test"),
        ann_dir=os.path.join(sohas, "annotations_test", "xmls"),
        split="val"
    )

    print("✅ Dataset build done:", OUT)

if __name__ == "__main__":
    main()