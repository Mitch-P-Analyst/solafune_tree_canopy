#!/ucd
#sr/bin/env python3
import os, json, sys
from collections import defaultdict

# ---- CONFIG (edit paths if different) ----
COCO_JSON = os.path.join("data", "train_annotations_coco.json")
ROOT = os.path.join("data", "model-data")  # contains train/ valid/ test/
SPLITS = ["train", "valid", "test"]        # change if you only use some

# -----------------------------------------

def norm_pair_iter(coords, w, h):
    """Yield (x,y) pairs normalised to [0,1] from a flat list of pixel coords."""
    it = iter(coords)
    for x, y in zip(it, it):
        yield (float(x) / w, float(y) / h)

def find_split_for_file(filename):
    """Return 'train'/'valid'/'test' if the image exists in that split, else None."""
    for s in SPLITS:
        p = os.path.join(ROOT, s, "images", filename)
        if os.path.exists(p):
            return s
    return None

def main():
    if not os.path.exists(COCO_JSON):
        sys.exit(f"COCO file not found: {COCO_JSON}")

    with open(COCO_JSON, "r") as f:
        coco = json.load(f)

    # Build image_id -> info and a filename index for quick lookup
    images_by_id = {img["id"]: img for img in coco.get("images", [])}
    # Category remap to contiguous 0..nc-1 in order of ascending id
    cat_ids_sorted = sorted({c["id"] for c in coco.get("categories", [])})
    cat_to_yolo = {cid: i for i, cid in enumerate(cat_ids_sorted)}
    names = [next(c["name"] for c in coco["categories"] if c["id"] == cid) for cid in cat_ids_sorted]

    # Ensure labels dirs exist
    for s in SPLITS:
        os.makedirs(os.path.join(ROOT, s, "labels"), exist_ok=True)

    counts = defaultdict(int)
    skipped_no_split = 0
    skipped_no_seg = 0

    # We will append per-image; keep file handles short-lived.
    for annotation in coco.get("annotations", []):
        img = images_by_id.get(annotation["image_id"])
        if not img:
            continue

        file_name = img["file_name"]
        w, h = float(img["width"]), float(img["height"])

        split = find_split_for_file(file_name)
        if not split:
            skipped_no_split += 1
            continue

        segmentation = annotation.get("segmentation", [])
        # COCO polygons: list of lists (each list is flat [x1,y1,x2,y2,...])
        # If it's RLE (iscrowd=1), we skip unless you add pycocotools to convert.
        if not isinstance(segmentation, list) or len(segmentation) == 0 or not isinstance(segmentation[0], (list, tuple)):
            # Fallback: if you want to convert bbox to a rectangle polygon, uncomment:
            # x, y, bw, bh = ann["bbox"]
            # seg = [[x, y, x+bw, y, x+bw, y+bh, x, y+bh]]
            skipped_no_seg += 1
            continue

        # Concatenate all polygon points for this instance
        pts_norm = []
        for polyon in segmentation:
            # Require at least 3 points (>= 6 numbers)
            if not polyon or len(polyon) < 6:
                continue
            pts_norm.extend([p for xy in norm_pair_iter(polyon, w, h) for p in xy])

        if len(pts_norm) < 6:  # nothing valid after filtering
            skipped_no_seg += 1
            continue

        class_id = cat_to_yolo[annotation["category_id"]]

        stem = os.path.splitext(file_name)[0]
        label_path = os.path.join(ROOT, split, "labels", f"{stem}.txt")
        with open(label_path, "a", encoding="utf-8") as lf:
            lf.write(str(class_id) + " " + " ".join(f"{v:.6f}" for v in pts_norm) + "\n")

        counts[split] += 1

    # Write a minimal data.yaml next to model-data
    data_yaml_path = os.path.join("data.yaml")
    with open(data_yaml_path, "w", encoding="utf-8") as yf:
        yf.write(
f"""# Auto-generated for YOLOv8 segmentation
path: {os.path.abspath(ROOT)}
train: {ROOT}/train/images
val: {ROOT}/valid/images
test: {ROOT}/test/images

names:
  {os.linesep.join(f"- {n}" for n in names)}
"""
        )

    # Summary
    print("Done.")
    print("Labels written per split:", dict(counts))
    print("Skipped (image not found in a split):", skipped_no_split)
    print("Skipped (no polygon segmentation / RLE):", skipped_no_seg)
    print(f"data.yaml written to: {os.path.abspath(data_yaml_path)}")

if __name__ == "__main__":
    main()
