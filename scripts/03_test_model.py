import ultralytics
from ultralytics import YOLO
import numpy as np
import time
import yaml
from pathlib import Path
import torch

# Directories and Paths
REPO_ROOT = Path(__file__).resolve().parent.parent

DATA_CONFIG_PATH = REPO_ROOT / 'configurations' / 'model_data-seg.yaml'

# Load model parameters / overrides
with open(REPO_ROOT / 'configurations' / 'test_model_overrides.yaml', 'r') as f:
    overrides = yaml.safe_load(f)

# # override the path in your dictionary before calling train
overrides['data'] = str(DATA_CONFIG_PATH)


# Load Trained Model Weights
weights = REPO_ROOT / 'runs/segment/training_fastNMS12/weights/best.pt' # Weights from Train Model / Modify where necessary

# Load Trained Model's Weights
model = YOLO(str(weights))  

# Predictions from the model
with torch.inference_mode():
    metrics = model.val(**overrides)


# State output location
try:
    print("Saved to:", model.val.save_dir)
except Exception:
    pass


validation_df = metrics.to_df()
print(validation_df)
print('AP@0.75 (seg):', metrics.seg.map75)