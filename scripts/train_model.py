import ultralytics
from ultralytics import YOLO
import numpy as np
import time
import yaml
from pathlib import Path

ROOT = Path(__file__).resolve().parent

# Load model parameters / overrides
with open(ROOT / 'train_model.yaml', 'r') as f:
    overrides = yaml.safe_load(f)

# Load a pretrained model
model = YOLO('yolov8s-seg.pt') # yolo version 11s segementation

# Train the model
results = model.train(**overrides)

# State output location
try:
    print("Saved to:", model.trainer.save_dir)
except Exception:
    pass
