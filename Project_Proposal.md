# Solafune‑Inspired Tree Canopy Detection Capstone Using solafune_tools

## Overview & Objective
The goal of this capstone is to design and implement a geospatial pipeline for detecting tree canopy—leveraging Sentinel‑2 imagery. We will use the solafune_tools library to manage data ingestion (including cloudless mosaics and optional 5× super-resolution), develop segmentation models, and produce competitive-ready submissions in the Solafune Tree Canopy Detection challenge.

### Motivation
Accurate tree canopy mapping supports urban planning, biodiversity conservation, and climate modeling. Participating in this challenge develops robust applied skills while yielding practical impacts.

## Tools & Technologies
- Data Ingestion & Preprocessing: solafune_tools, Earth Engine, Geemap
- Geospatial Processing: rasterio, numpy, shapely, GDAL
- Modeling: PyTorch or TensorFlow (U-Net, vision transformers)
- Evaluation & Submission: competition_tools for IoU and formatting compliance

## Workflow
Data Ingestion
- Use solafune_tools to download STAC catalogs and Sentinel‑2 imagery from the Planetary Computer, then create cloud-free mosaics and optionally apply 5× super-resolution.

## Exploration & Visualization

- Visualize mosaics with geemap/Earth Engine to assess canopy cover and cloud issues.

- Preprocessing & Feature Engineering
    - Normalize bands, compute NDVI or other vegetation indices, and slice images into tiles with accompanying label masks.

- Modeling Strategy

- Baseline: NDVI‑based thresholding
    - Advanced: Train segmentation models (U-Net or ViT variants) using super-resolved or standard inputs

- Evaluation
    - Use competition metrics (IoU, etc.) via competition_tools; perform cross-validation and visual inspections.

- Submission & Analysis
    - Prepare predictions in required formats, reflect on failure modes, and refine pipeline.

## Milestones & Timeline
| Week   | Activities                                   |
| ------ | -------------------------------------------- |
| Week 1 | Setup environment; ingest and visualize data |
| Week 2 | Preprocessing; baseline NDVI detection       |
| Week 3 | Segmentation model development and training  |
| Week 4 | Evaluate, fine-tune, error analysis          |
| Week 5 | Final submission; write-up and documentation |


### Deliverables
- Modular Jupyter notebooks or scripts for ingestion, preprocessing, modeling, and evaluation

- Clean, user-friendly README and documentation

- Model artifacts and notebooks with visual outputs

- Final written report summarizing methodology, findings, and future directions





pre plan structure the data

create model
train model

use model on evluatative data

reformat to submit