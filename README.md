# Solafune Tree Canopy Detection Capstone

## Project Overview  
Build a tree canopy detection pipeline using Sentinel‑2 data; ingest and preprocess imagery, train segmentation models, and prepare submissions for the Solafune competition.

## Directory Structure  
```
├── data/                               # Downloaded satellite imagery and mosaics
│ ├── processed
│ │ ├── images
│ │ ├── labels
│ │ └── JSONs
│ ├── raw
│ │ ├── zips
│ │ └── JSONs
├── notebooks/                          # Jupyter notebooks for preprocessing
│ └── 01_data_preparation.ipynb
├── scripts/                            # Python scripts
│ ├── 02_train_model.py
│ ├── 03_predict_model.py
│ └── 04_export_submission.py
├── runs / segments /                   # Relevant model runs
├── exports/                            # JSON Submission files
├── README.md                           # This file
└── requirements.txt
```


## Installation  
```bash
git clone <repo-url>
cd solafune-tree-canopy
pip install -r requirements.txt
pip install solafune_tools[super_resolution]  # for optional SR module
```

## Usage
Data Setup:

``` python
import solafune_tools
solafune_tools.set_data_directory("data/")
```

Run Notebooks: Follow the sequence in notebooks/:

Ingest: Download imagery

Preprocess: Normalize, compute NDVI, tile

Train: Baseline and deep learning models

Evaluate: Assess performance, generate submissions

## License
MIT License


