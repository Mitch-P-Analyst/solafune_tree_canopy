# Solafune Tree Canopy Detection Capstone

## Project Overview  
The goal of this capstone is to design and implement a geospatial pipeline for detecting tree canopy—leveraging Sentinel‑2 imagery. We will use the solafune_tools library to manage data ingestion (including cloudless mosaics and optional 5× super-resolution), develop segmentation models, and produce competitive-ready submissions in the Solafune Tree Canopy Detection challenge.

### Motivation
Accurate tree canopy mapping supports urban planning, biodiversity conservation, and climate modeling. Participating in this challenge develops robust applied skills while yielding practical impacts.

### Tools
- YOLO ML


## Directory Structure  
```
├── data/                               # Downloaded satellite imagery and mosaics
│ ├── processed/
│ │ ├── images/
│ │ │  ├── predict/
│ │ │  ├── train/
│ │ │  ├── eval/
│ │ │  └── test/
│ │ ├── labels/
│ │ │  ├── train/
│ │ │  ├── eval/
│ │ │  └── test/
│ │ └── JSONs/
│ ├── raw/
│ │ ├── zips/
│ │ └── JSONs/
│ └── temp/
├── notebooks/                          # Jupyter notebooks for preprocessing
│ └── 01_data_preparation.ipynb
├── scripts/                            # Python scripts
│ ├── 02_train_model.py
│ ├── 03_eval_model.py
│ ├── 04_predict_model.py
│ └── 05_export_submission.py
├── runs / segments /                   # Relevant model runs
├── exports/                            # JSON Submission files
├── README.md                           # This file
└── requirements.txt
```


## Installation  
```bash
git clone <https://github.com/Mitch-P-Analyst/solafune-canopy-capstone.git>
cd solafune-tree-canopy
pip install -r requirements.txt
```

## Steps
- Download Data
<https://drive.google.com/drive/folders/1sB7XVJuFYcJCqzbiHcxKC96WAWCKo3Zj?usp=drive_link>

├── raw data/                            # Python scripts
│ ├── JSONs/
│ │ └── train_annotations.json
│ ├── ZIPs/
│ │ ├── evaluation_images.zip
│ │ └── train_images.zip


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


## License
MIT License


