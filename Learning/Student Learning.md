# Student Learning

## Goal

You are participating in a geospatial machine learning competition to detect tree canopies using remote sensing imagery — likely from satellites like Sentinel-2.

The general goal is to build a model (or use analytical techniques) to identify areas covered by tree canopies in satellite images.

## Packages

### rasterio 
- rasterio is a Python library for reading, writing, and manipulating raster data (i.e. geospatial images — like satellite imagery, aerial photography, elevation data, etc.).

It’s built on top of GDAL (Geospatial Data Abstraction Library) but gives it a Pythonic, easy-to-use interface.

#### Common Use Cases & Functions
| Task                             | Example Function                         |
| -------------------------------- | ---------------------------------------- |
| Open raster image                | `rasterio.open("file.tif")`              |
| Read pixel data (as array)       | `.read()`                                |
| Get metadata (CRS, bounds, etc.) | `.meta`, `.crs`, `.bounds`, `.transform` |
| Write new raster to disk         | `with rasterio.open(..., 'w')`           |
| Rasterize vector geometries      | `rasterio.features.rasterize()`          |
| Reproject rasters                | `rasterio.warp.reproject()`              |
| Visualise rasters (basic)        | `matplotlib.pyplot.imshow()` + `.read()` |

##### Examples
```python
import rasterio 
import matplotlib.pyplot as plt
import numpy as np

# Open a raster image file
with rasterio.open('data/train_images/10cm_train_1.tif') as src: 
    #src is common alias used for raster files. Meaning 'source'

# Learn about raster image file
    print(src.count)         # Number of bands
    print(src.meta)          # Metadata dictionary
    print(src.crs)           # Coordinate Reference System
    print(src.width, src.height)  # Dimensions

    # Or just
    src.meta

# Read raster image file
image = src.read([1,2,3])
```
A satellite image (or any raster) is made up of one or more bands:
- A band is a single 2D array of pixel values.

Each band represents intensity values for a wavelength or colour channel.
- A 3-band image may look like:
    - Band 1 = Red
    - Band 2 = Green
    - Band 3 = Blue

When you `read()` an RGB, the read() function outputs an Numpy Array for each Band Value in the following context- - `(Band Number, Rows (height), Columns (width))`

```python

# Transpose the imaage
image = np.transpose(image, (1,2,0))
```

Why do we do this?
- To visualise the image from the Matrix of Numpy Arrays, matplotlib.pyplot function imshow() expects an input  arrangement of
    - `(Rows (height), Columns, (width), Band Number)`
        - Therefore, we transpose the rasterio.read() output into this arrangement

- Note
    - RGB Bands begin at index 1
    - Numpy Array values begin at index 0



## Data 
📦 What Is the Data?
Since the competition uses satellite imagery, the raw data format is likely:

| Type     | Format                          | Description                                        |
| -------- | ------------------------------- | -------------------------------------------------- |
| Imagery  | `.tif` (GeoTIFF)                | Multiband raster images from Sentinel‑2 or similar |
| Labels   | `.tif` or `.json` or `.geojson` | Masks of where trees are (ground truth)            |
| Metadata | `.json` or `.csv`               | Bounding boxes, timestamps, projection info        |

### Data Summary

- train_images.zip
    - File type :	zip
    - File count:	150
    - Data description: 150 RGB images data for segmentation training


- train_annotations.json
    - File type	:	json
    - File count:	1
    - Data description: Annotation data for segmentation of training images


- evaluation_images.zip
    - File type :	zip
    - File count:	150
    - Data description:	150 RGB images data for segmentation evaluation


- sample_answer.json
    - File type	:	json
    - File count:	1
    - Data description:	Solafune custom dataset format sample for submission and evaluation purposes


### Geospatial Data

When you load images (.tif files) using rasterio.read() or numpy.array() — they return raw numerical arrays, such as a 3D matrix (array of pixel values)
- Example
    
    ```python

    array([[[ 77, 113,  65],   # pixel 1 RGB
            [ 77, 114,  63],   # pixel 2 RGB
            ... 
            ]])
    ``` 
You’ll convert or visualise it using matplotlib or opencv to see the human-readable version.





segmentation vs classification

Object detection 
- intersection over union metric
- non-maximum supression

YOLO Machine Learning Model
- learn formatting 
    - is Singular Looking Object detection model
- use as easy option

COCO format JSON
- Python Package
    - ultralytics COCO
- COCO dataset is the golld standard for object detection tasks

transferlearning
    - like a YOLO model has been trained on 80 things in Cocodataset
    - now trasnfer your knowledge to learn something new that i want to

neutral networks
- Types
    - standard
    - cnn neutral networks
    - Long short-term 
- Weights, biases
- Backpropagration
    - Gradient Descent

pickle


SAHI

real-esrgan satellite imagery


satelite ndvi vegetation 
    false color


- open_CV
    - for images
- Pill / Pillow
    - for AI / machine learning
    - works better with Pytorch
- sckit_image
- scykit_py
- raster



zero-shot-learning ML
