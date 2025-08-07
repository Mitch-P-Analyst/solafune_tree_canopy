# Solafune Tree Canopy Detection Capstone

## Project Overview  
Build a tree canopy detection pipeline using Sentinel‑2 data; ingest and preprocess imagery, train segmentation models, and prepare submissions for the Solafune competition.

## Directory Structure  

├── data/ # Downloaded satellite imagery and mosaics
├── notebooks/ # Jupyter notebooks for each step
│ ├── 01_data_ingest.ipynb
│ ├── 02_preprocessing.ipynb
│ ├── 03_model_training.ipynb
│ └── 04_evaluation.ipynb
├── src/ # Python scripts/modules
│ ├── ingestion.py
│ ├── preprocessing.py
│ ├── modeling.py
│ └── evaluation.py
├── outputs/ # Model outputs, submission files, logs
├── README.md # This file
└── requirements.txt



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


