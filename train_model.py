import ultralytics
from ultralytics import YOLO
import numpy as np
import time

# Load a pretrained model
model = YOLO('yolo11s-seg.pt') # yolo version 11s segementation

# Train the model
results = model.train(
                    data = 'training_configuration_tree_canopy-seg.yaml', # Data Source
                    device='mps',        # your Apple Metal GPU
                    epochs=100,          # number of training epochs
                    imgsz=416,           # image size (default 640)
                    batch=1,             # batch size
                    seed=0,               # Faced numers NMS issues. Stating for comparisons across parameter changes/model changes

                    name='training_fastNMS',
                    conf=0.5,           # filter low-confidence preds early
                    iou=0.5,             # merge more aggressively in NMS
                    max_det=100,         # cap kept detections per image
                    val=True,
                    plots=False,           # don't draw plots every val
                    agnostic_nms=True,     # merge across classes
                    workers=2              # keep small on macOS/MPS
                    # save_json=False,     # skip COCO json export during train
                    # workers=4            # if your CPU can handle it
                    )