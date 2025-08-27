import json
from pathlib import Path
from ultralytics import YOLO
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent

img_dir = "/content/solafune-canopy-capstone/data/processed/images/predict"

# Load template submission (has the "images" list with file_name/width/height/etc.)
with open(REPO_ROOT / "model-data" / "sample_answer.json", "r") as f:
    submission_json = json.load(f)

# filenames we must predict for, in order
image_files = [img["file_name"] for img in submission_json["images"]]

# class mapping MUST match your training YAML
class_dictionary = {
    0: "individual_tree",
    1: "group_of_trees",   # <-- ensure spelling is correct
}

# load model
model_path = REPO_ROOT / "runs/segment/training_fastNMS12/weights/best.pt"
model = YOLO(model_path)

# predict for each image
for k, image_file in enumerate(image_files):
    image_path = str(img_dir / image_file)  # Path -> str for Ultralytics

    with torch.no_grad():
        results = model.predict(
            source=image_path,
            save=False,
            imgsz=640,
            device="None",     # change to "cpu" if MPS not available
            verbose=False,
            conf=0.2,
            iou=0.6,
            rect=False
        )

    r = results[0]

    # default: no detections
    image_annotations = []

    # If there are no masks, leave annotations empty
    if r.masks is not None and r.boxes is not None:
        # classes/scores as native Python types
        classes = r.boxes.cls.cpu().numpy().astype(int).tolist()   # list[int]
        scores  = r.boxes.conf.cpu().numpy().astype(float).tolist()# list[float]

        # polygons: r.masks.xy is a list of (N,2) np arrays in PIXELS already
        polygons = r.masks.xy or []

        for i, poly in enumerate(polygons):
            if poly.size < 6:
                continue  # need at least 3 points

            # round to pixel ints, flatten, and convert to native list
            seg = poly.round().astype(int).reshape(-1).tolist()

            image_annotations.append({
                "class": class_dictionary[int(classes[i])],
                "confidence_score": float(scores[i]),
                "segmentation": seg,   # flat [x1,y1,x2,y2,...] with Python ints
            })

    # write annotations back into the template for this image
    submission_json["images"][k]["annotations"] = image_annotations

# dump JSON (now only native Python types)
with open(REPO_ROOT / 'exports'/ "submission_b.json", "w") as f:
    json.dump(submission_json, f, indent=4)
