from pathlib import Path
import numpy as np
import json
from PIL import Image
import pandas as pd

# Create path
REPO_ROOT = Path.cwd()

LABELS = REPO_ROOT / 'runs/segment/predict_fastNMS/labels'
IMGS = REPO_ROOT / 'data/images/test'
OUT_JSON = REPO_ROOT / 'exports'/ 'submission.json'




# Image Metadata
def get_meta(IMGS):

    meta_data = {}
    scene_type_default = "Unknown"

    if IMGS.exists():

        for p in Path(IMGS).iterdir():
            if p.suffix.lower() not in {".tif"}:
                continue
            file_name = p.name
            cm_resolution = str(file_name[0:2])

            with Image.open(p) as image:
                width, height = image.size

            meta_data[file_name] = {
                'file_name' : file_name,
                "width" : width,
                'height' : height,
                "cm_resolution" : cm_resolution,
                "scene_type" : scene_type_default

            }
    return meta_data


    
meta = get_meta(IMGS)



images = []

for m in meta.values():
    images.append({
        'file_name' : m['file_name'],
        'width' : m['width'],
        'height' : m['height'],
        'cm_resolution' : m['cm_resolution'],
        'scene_type' : m['scene_type'],
        'annotations' : []
    })
    
images_sorted = sorted(images, key=lambda x: x['file_name'])


def get_annotations(LABELS):
    annotations = []

    for txt_file in Path(LABELS).iterdir():
        image_name = txt_file.name
        # print(f"\n📂 Reading {image_name}")

        with open(txt_file, 'r') as file:
             for line in file:

                # 1. Split the line into tokens
                tokens = line.strip().split()

                # 2. Convert tokens into usable pieces
                class_id = int(tokens[0])                    # first token
                *coords, confidence_level = map(float, tokens[1:])  
                # all middle tokens are coords, last one is confidence
                

                # 3. Group coords into pairs (x,y)
                segmentation = [
                    (coords[i], coords[i+1])
                    for i in range(0, len(coords), 2)
                ]

                # 4. Save everything into dict
                annotations.append({
                    "image": image_name,
                    "class": class_id,
                    "confidence_level": confidence_level,
                    "segmentation": segmentation
                })

    return annotations

    

    # class id
ID_to_Name = {
    0 : 'individual_tree',
    1 : 'group_of_trees'
}

annotations_list = get_annotations(LABELS)

for annot in annotations_list:
    # .txt suffix -> .tif
    annot['image'] = annot['image'][:-4] + '.tif'

    # map class id -> name
    annot['class'] = ID_to_Name.get(annot['class'], 'Unknown')

    # convert confidence_level -> percentage string
    annot['confidence_level'] = f"{annot['confidence_level']:.2f}"
    

def denorm_segmentation(seg, width, height, *, flatten=True, round_int=True):
    """
    seg: list of (x,y) pairs in [0,1]
    returns: flat pixel list [x1,y1,x2,y2,...] (default) or list of pairs
    """
    # If seg is already flat [x1,y1,...], turn into pairs
    if seg and not isinstance(seg[0], (list, tuple)):
        it = iter(seg)
        seg_pairs = list(zip(it, it))
    else:
        seg_pairs = seg

    out = []
    for x, y in seg_pairs:
        X = max(0.0, min(width  - 1, x * width))
        Y = max(0.0, min(height - 1, y * height))
        if round_int:
            X = int(round(X))
            Y = int(round(Y))
        if flatten:
            out.extend([X, Y])
        else:
            out.append((X, Y))
    return out


def append_imgs_annotations(img_list, annot_list):
    # ensure each image has an annotations list
    for img in img_list:
        if "annotations" not in img or img["annotations"] is None:
            img["annotations"] = []

    for image in img_list:
        w, h = image["width"], image["height"]
        fname = image["file_name"]

        # all annots for this image
        matched = (a for a in annot_list if a["image"] == fname)

        for a in matched:
            seg_px = denorm_segmentation(a["segmentation"], w, h, flatten=True, round_int=True)

            image["annotations"].append({
                "class": a["class"],
                "confidence_score": float(a["confidence_level"]),  # or keep as formatted string if you prefer
                "segmentation": seg_px
            })

    return img_list


predict_answer = append_imgs_annotations(images_sorted, annotations_list)

OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
with open(OUT_JSON, "w") as f:
    json.dump({"images": predict_answer}, f, indent=2)
