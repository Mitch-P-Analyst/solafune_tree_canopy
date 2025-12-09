# Solafune Tree Canopy Detection Capstone

## Project Overview  

This project involved building a geospatial ML pipeline using Sentinel-2 imagery to detect tree canopies via image segmentation. Hosted by Solafune, I managed data imports of image segmentations, trained a segmentation model, and produced a competition-ready submission for the Solafune **Tree Canopy Detection** challenge.

The pipeline runs on both local environments and Google Colab with minimal path changes, enabling access to GPU acceleration for faster training.

### Motivation

Accurate tree canopy mapping supports urban planning, biodiversity conservation, and climate modelling. Participating in this challenge helped me develop stronger GIS and geospatial machine-learning skills while working on a problem with real-world impact.

This project was also my first application of YOLO-based image segmentation and geospatial data processing, providing valuable experience working with Earth-observation data formats.


### Results
My current submission model, as of **September 24th 2025**, places in the top 10 of 271 competitors on [Solafune leaderboard](https://solafune.com/competitions/26ff758c-7422-4cd1-bfe0-daecfc40db70?modal=%22%22&menu=lb&tab=public), against the assesment criteria of >75% mean IoU on the prediction dataset.

- Best submission score: 0.338 (competition metric)
- Model: YOLO-based tree canopy segmentation on Sentinel-2 imagery
- Runs on: local machine + Google Colab
‍

### Approach
My best submission score of 0.338 used a hold-out validation split to select augmentation strategies and model hyperparameters. I iterated on augmentation across visual parameters such as rotation, hue and saturation, and image scaling, as well as duplicating annotations to increase training exposure and improve the YOLO-based model’s learning environment.

For evaluation, I tracked precision and recall for both classes (“Individual Trees” and “Group of Trees”) to tune confidence thresholds and IoU settings and optimise the competition metric.

### Lessons Learned
- JSON label formatting and format conversion (COCO ↔ YOLO)
- Consistent directory structure for large-scale ML projects across local and digital directories.
- Benefits of augmentation techniques like HSV rotation, colour grading and image scaling.
- Model validation techniques such as hold-out vs k-fold cross-validation.
- Convolutions and convolutional neural networks.


### Tools
- **YOLO segmentation models**  
  - Image segmentation for tree canopy detection.  

- **Python 3.10**  
  - Base language for building the pipeline and running scripts.  
  - Used for the full geospatial + ML pipeline.  

- **PyTorch**  
  - Deep-learning framework used to train segmentation models.  

- **Google Colab**  
  - Notebook environment used for GPU-accelerated training.  

- **JSON / COCO JSON**  
  - Retrieval, storage, and submission of labels.  

- **YAML**  
  - Structured configuration files for model parameters.  

- **GIS toolkits**  
  - Geospatial data handling and processing.  

## Installation  
```bash
git clone <https://github.com/Mitch-P-Analyst/solafune-canopy-capstone.git>
cd solafune-tree-canopy
pip install -r requirements.txt
```
## Repo Directory Structure  
```
├── configurations/                         # YAML configs (data + overrides)
│   ├── model_data-seg.yaml                 # Dataset paths & class names
│   ├── train_model_overrides.yaml          # Training parameters
│   ├── val_model_overrides.yaml            # Validation parameters
│   └── predict_model_overrides.yaml        # Prediction parameters
│
├── data/                                   # Downloaded satellite imagery and mosaics
│   ├── processed/
│   │   ├── images/
│   │   │   ├── predict/                    # Unlabelled (no GT) data for prediction
│   │   │   ├── train/                      # Ground-truth images for training
│   │   │   ├── val/                        # Ground-truth images for validation
│   │   │   └── test/                       # Required by YOLO structure
│   │   ├── labels/
│   │   │   ├── train/                      # Ground-truth labels for training
│   │   │   ├── val/                        # Ground-truth labels for validation
│   │   │   └── test/                       # Required by YOLO structure
│   │   └── JSONs/                          # Converted JSON files
│   ├── raw/
│   │   ├── Solafune_raw_data.md            # Solafune raw data access instructions
│   │   ├── zips/                           # Raw data ZIP files
│   │   └── JSONs/                          # Raw data JSONs
│   └── temp/
│
├── notebooks/
│   ├── 01_data_preparation.ipynb           # Convert JSONs, unzip, split data
│   ├── 02_train_model_colab.ipynb          # Google Colab notebook for model training
│   └── 04_test_model_evaluations.ipynb     # (Optional) In-depth model evaluations
│
├── scripts/
│   ├── 02_train_model.py                   # Train YOLO model
│   ├── 03_val_model.py                     # Validate YOLO model on GT data
│   ├── 05_predict_model.py                 # Create predictions on non-GT data
│   └── 06_export_submission.py             # Convert predictions into Solafune JSON format
│
├── runs/segment/                           # YOLO runs directory (train/val/predict results)
├── exports/                                # JSON submission files
├── README.md                               # This file
├── README.html                             # README in HTML format for digital portfolio
└── requirements.txt                        # Package requirements

```

## Process

### Data
- Data Not Sharable by Solafune Non-Disclosure Agreement.
    - To access data, read [Solafune_raw_data.md](data/raw/Solafune_raw_data.md) which outlines the steps required to reproduce this project using Solafune raw data while complying with competition terms and conditions.

### Files & Run Order


#### 1. Data Preparation
- Notebook:
   - ['notebooks/01_data_preparation.ipynb'](notebooks/01_data_preparation.ipynb)
       - JSON conversion
       - Solafune format -> COCO format
       - COCO format -> YOLO format
       - Unpacking Raw Data
           - Extract ZIP files
               - Training
               - Prediction
       - Data Split Images & Annotations
           - Training
           - Validation


#### 2. Model Training
- You can train on a **Local Device** or on **Google Colab (GPU)**.
   - **Local Device**
       - Script:
           - [`scripts/02_train_model.py`](scripts/02_train_model.py) 
       - Configure hyperparameters
           - [`configurations/train_model_overrides.yaml`](configurations/train_model_overrides.yaml)
       - Choose the pretrained YOLO weights near the top of the script (line ~21):
           ```python
           model = YOLO('yolo11s-seg.pt')  # options: yolo11n-seg.pt, yolo11s-seg.pt, yolo11x-seg.pt, yolov8s-seg.pt
           ```
   - **Google Colab**
       - Notebook:
           - notebooks/02_train_model_colab.ipynb'   
               - [![ Open in Colab ▶ ](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Mitch-P-Analyst/solafune_tree_canopy/blob/main/notebooks/02_train_model_colab.ipynb)
           - Use Colab’s GPU and follow the **in-notebook** instructions to mount Drive, set paths, train model and export model weights.


#### 3. Model Validation
- Script:
   - ['scripts/03_val_model.py'](scripts/03_val_mode.py)
- Configure hyperparameters & trained model weights
   - ['configurations/val_model_overrides.yaml'](configurations/train_model_overrides.yaml)
       - Select Model Weights from trained YOLO model
           ```python
           weights: 'runs/segment/train_Yolo11s_canopy_832_adamW__20251101-0151/weights/best.pt'  # example
           ```
       - Modify validation model parameters YAML file for fine tuning model


#### 4. Metric Testings (Optional)
- Notebook:
   - ['notebooks/04_test_model_evaluations'](notebooks/04_test_model_evaluations.ipynb)
       - Optional Unfinished Notebook file. Containing indepth measures to analyse split data validation.


#### 5. Predictions
- Script:
   - ['scripts/05_predict_model.py'](scripts/05_predict_model.py)
- Configure hyperparameters & trained model weights
   - ['configurations/predict_model_overrides.yaml'](configurations/predict_model_overrides.yaml)
       - Select Model Weights from trained YOLO model
           ```python
           weights: 'runs/segment/train_Yolo11s_canopy_832_adamW__20251101-0151/weights/best.pt'  # example
           ```
       - Modify prediction model parameters YAML file for final model deployment


#### 6. Export
- Script:
   - ['scripts/06_export_submission.py'](scripts/06_export_submission.py)
       - Select Prediction Annotations on **Line 16** from `/runs/segement/` folder for Submission Jile
           ``` python
           # Prediction Annotations
           labels = REPO_ROOT / "runs/segment/pred_train_Yolo11s_canopy_832_adamW__20251101-0151_2025-11-01 10:33" / "labels" # example
           ```


## About

Data science competition hosted by Solafune, tasked to train a machine-learning model capable of detecting tree canopies in multiple urban environments from aerial and satellite imagery.

### Competition page:
https://solafune.com/competitions/26ff758c-7422-4cd1-bfe0-daecfc40db70
