# Solafune Tree Canopy Detection Capstone

## Project Overview  
The goal of this capstone is to design and implement a geospatial pipeline for detecting tree canopy—leveraging Sentinel‑2 imagery. We will use the solafune_tools library to manage data ingestion (including cloudless mosaics and optional 5× super-resolution), develop segmentation models, and produce competitive-ready submissions in the Solafune Tree Canopy Detection challenge.

### Motivation
Accurate tree canopy mapping supports urban planning, biodiversity conservation, and climate modeling. Participating in this challenge develops robust applied skills while yielding practical impacts.


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


