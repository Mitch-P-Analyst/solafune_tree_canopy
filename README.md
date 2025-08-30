# Solafune Tree Canopy Detection Capstone

## Project Overview  
The goal of this project is to design and implement a geospatial pipeline for detecting tree canopy using Sentinel‑2 imagery. Hosted by Solafune, we will manage data imports of image segmentations, train a segmentation model, and produce a competitive-ready submission in the Solafune Tree Canopy Detection challenge.

### Motivation
Accurate tree canopy mapping supports urban planning, biodiversity conservation, and climate modeling. Participating in this challenge develops robust skills while yielding practical impacts while building skillsets in Geospatial Machine Learning. 

### Tools
- YOLO Machine Learning Model
    - image segmentation
- Python 3.10
    - base language for building the pipeline and running scripts
    - Tasks
        - geospatial + ML pipeline
- PyTorch - deep learning framework used to train segmentation models

## Installation  
```bash
git clone <https://github.com/Mitch-P-Analyst/solafune-canopy-capstone.git>
cd solafune-tree-canopy
pip install -r requirements.txt
```
## Repo Directory Structure  
```
├── configurations/                     # YAML configs (data + overrides)
│   ├── model_data-seg.yaml                 # dataset paths & class names
│   ├── train_model_overrides.yaml          # training parameters
│   ├── val_model_overrides.yaml            # validation parameters
│   └── predict_model_overrides.yaml        # prediction parameters
│
├── data/                               # Downloaded satellite imagery and mosaics
│   ├── processed/                       
│   │ ├── images/
│   │ │  ├── predict/                     # Unlabeled no Ground Truth data for prediction 
│   │ │  ├── train/
│   │ │  ├── val/
│   │ │  └── test/
│   │ ├── labels/
│   │ │  ├── train/
│   │ │  ├── val/
│   │ │  └── test/
│   │ └── JSONs/
│   ├── raw/
│   │ ├── zips/
│   │ └── JSONs/
│   └── temp/
│
├── notebooks/                          
│   ├── 01_data_preparation.ipynb           # Convert JSONs, Unzip, Split Data
│   └── 04_test_model_evaluations.ipynb     # **Optional** Indepth model evaluations
│
├── scripts/                                
│   ├── 02_train_model.py                   # Train YOLO Model
│   ├── 03_test_model.py                    # Test/Valdiate YOLO Model on GT Data
│   ├── 05_predict_model.py                 # Create predictions with trained YOLO Model on no GT Data  
│   └── 06_export_submission.py             # Convert prediction outputs into Solafune JSON format
│
├── runs/segments/                          # Relevant model runs
├── exports/                                # JSON Submission files
├── README.md                               # This file
└── requirements.txt                        # Package requirements
```

## Process
- Download Data
<https://drive.google.com/drive/folders/1sB7XVJuFYcJCqzbiHcxKC96WAWCKo3Zj?usp=drive_link>

- Run Notebooks & Scripts in sequence: 

    - 01_data_preparation.ipynb
        - JSON conversion 
            - Solafune format -> COCO format
            - COCO format -> YOLO format
        - Unpacking Raw Data
            - Extract ZIP files
                - Training
                - Prediction
        - Data Split Images & Annotations
            - Training
            - Testing
            - Validation 

    - 02_train_model.py
        - Modify train model parameters YAML file for desired training and naming
            - [Train Model Parameters](configurations/train_model_overrides.yaml)
        - Select YOLO Model Version
            - [Train Model Py File : Row 21](scripts/02_train_model.py)
        
    - 03_test_model.py
        - Modify test model parameters YAML file for fine tuning model
            - [Test Model Parameters](configurations/test_model_overrides.yaml)
        - Select Train Models Weights for validation
            - [Trained Model Segment Outputs | 'Train' Model weights.pt](runs/segment)
                - [Test Model Py File : Row 23](scripts/03_test_model.py)  

    - 04_test_model_evaluations.ipynb | **Optional**
        - Optional Unfinished Notebook file. Containing indepth measures to analyse test split data validation.

    - 05_predict_model.py
        - Modify predict model parameters YAML file for final model predictions
            - [Predict Model Parameter](configurations/predict_model_overrides.yaml)
        - Select Train Models Weights for validation
            - [Trained Model Segment Outputs | 'Train' Model weights.pt](runs/segment)
                - [Test Model Py File : Row 23](scripts/05_predict_model.py) 

    - 06_export_submission.py
        - Select Predict Models Annotations for Submission Jile
            - [Predict Model Segment Outputs | 'Predict' Model labels.txt](runs/segment)
                - [Export Submission Py File : Row 16](scripts/06_export_submission_.py) 

## **Optional** Google Colab 
Relative pathways and constructed Google Colab .ipynb file 

### Required Google Drive Directory Structure
```
├── drive/                               # Downloaded satellite imagery and mosaics
│   ├── MyDrive/                       
│   │   ├── Datasets/
│   │   │  ├── solafune-tree-canopy/                   
│   │   │   │   ├── data/
│   │   │   │   │   ├── processed/
│   │   │   │   │   │   ├── labels/
│   │   │   │   │   │   │   ├── train/
│   │   │   │   │   │   │   ├── test/
│   │   │   │   │   │   │   └── val/
│   │   │   │   │   │   └── images/
│   │   │   │   │   │       ├── train/
│   │   │   │   │   │       ├── test/
│   │   │   │   │   │       └── val/
│   │   │   │   │   │
│   │   │   │   │   ├── temp/
│   │   │   │   │   └──  raw/
│   │   │   │   │      ├── zips/
│   │   │   │   │      └── JSONs/
│   │   │   │   │
│   │   │   │   ├── runs/
│   │   │   │   │   ├── segments/
│   │   │   │   │   └── notebooks/
│   │   │   │   │
│   │   │   │   └──   outputs/
│   │   │   │       └── exports/
```

### Files

- Run Notebook
    - [solafune_tree_canopy.ipynb](https://colab.research.google.com/drive/1KrtNSr8aHL5j8dGBrMzNdlHEesKB712Z?usp=drive_link)


## License
MIT License


