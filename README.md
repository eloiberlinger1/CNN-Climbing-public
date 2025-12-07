# Routeset CNN - Wall and Corner Detection System

**This project aims to implement CNN for the purpose of helping and serving the climbing industry and more specifically Routesetting.** 

## Description

The project for now focuses more on detecting where the wall is in a picture and recognising easy patterns.

- Wall segmentation
- Overhang angle prediction

![Example picture](https://filedn.eu/lqAABNUSgVkfRyevTm4sSpR/bordel/example.png)

  *(07/12/2025 Update: From my point of view, in order to extract and analyse accurate and exploitable data for climbing purposes, a single 2D image is too hard to extract accurate data from. Exploiting 3D photogrammetry / LIDAR scans are the future in this research)*

## Better results have been found using YOLO

![image of results after fine tuning YOLO-seg model](https://filedn.eu/lqAABNUSgVkfRyevTm4sSpR/bordel/yolo_trained_results.png)

To use YOLO, I had to adapt the format of all my annotations and dataset to the YOLO required format.
Example can be found here :
https://colab.research.google.com/drive/1aJrK4Oq5pOiNV55UMiHeFBmcTMs96zxa?usp=sharing


## Installation
```bash
python  -m  venv  venv
source  venv/bin/activate  # Linux/Mac
venv\Scripts\activate  # Windows
# Install dependencies
pip  install  -r  requirements.txt
```

## Usage

### 1. Data Preparation

To prepare a new dataset:

- Resize images to the target format (512x683)
- Create the appropriate folder structure
- Save the resized images
- Annotate all images
  
```bash
python  tools/wall_annotator/annotator.py  dataset1
```
The tool allows you to:

- Annotate walls with polygons
- Save annotations in JSON format
- Automatically generate binary masks

## License

This project is licensed under the GNU GENERAL PUBLIC LICENSE.
A copy of the license should be included in the project root directory (e.g., in a file named LICENSE).