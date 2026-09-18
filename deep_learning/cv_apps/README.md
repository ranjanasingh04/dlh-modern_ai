# Computer Vision Applications

A computer-vision project that prepares the Pascal VOC 2012 dataset for object detection with YOLOv8. The data pipeline selects specified images, retains three target classes, converts Pascal VOC XML annotations into normalized YOLO labels, and validates the resulting dataset structure.

## Project Objective

The goal is to create a reproducible object-detection dataset containing these classes:

| Class ID | Class   |
| -------: | ------- |
|        0 | Person  |
|        1 | Car     |
|        2 | Bicycle |

The resulting dataset can be used to train and evaluate a YOLOv8 object-detection model.

## Key Features

* Downloads and extracts Pascal VOC 2012.
* Reads the required training and validation image names.
* Selects only the requested images.
* Filters annotations to `person`, `car`, and `bicycle`.
* Converts Pascal VOC bounding boxes to normalized YOLO format.
* Creates one label file for every selected image.
* Generates the required YOLOv8 `data.yaml`.
* Supports repeatable dataset generation.
* Uses documented, `pycodestyle`-compliant Python code.
* Preserves Linux-compatible line endings and executable permissions.

## Repository Structure

```text
cv_apps/
├── README.md
├── requirements.txt
├── 0-prep_data.py
├── train_samples.txt
├── val_samples.txt
└── datasets/
    └── detection/
        ├── images/
        │   ├── train/
        │   └── val/
        ├── labels/
        │   ├── train/
        │   └── val/
        └── data.yaml
```

The complete Pascal VOC source dataset is cached outside the repository. Only the selected dataset subset is generated under `datasets/detection/`.

## Technologies

* Python 3.11
* NumPy 2.0.2
* Matplotlib 3.10.0
* OpenCV 4.12.0.88
* PyTorch 2.8.0
* Albumentations 2.0.8
* Ultralytics 8.4.7
* pycodestyle 2.14.0
* Pascal VOC 2012
* YOLOv8

## Environment Setup

Create a dedicated Conda environment:

```bash
conda create --name cv-apps python=3.11 pip -y
conda activate cv-apps
```

Install the dependencies:

```bash
python3 -m pip install -r requirements.txt
```

The `requirements.txt` file contains:

```text
numpy==2.0.2
matplotlib==3.10.0
opencv-python==4.12.0.88
torch==2.8.0
albumentations==2.0.8
ultralytics==8.4.7
pycodestyle==2.14.0
```

## Preparing the Dataset

Place the provided sample-list files in the project directory:

```text
train_samples.txt
val_samples.txt
```

Each file must contain one Pascal VOC image identifier per line:

```text
2007_000027
2007_000032
2007_000033
```

Run the preparation pipeline from the `cv_apps` directory:

```bash
python3 0-prep_data.py
```

The script will:

1. Download Pascal VOC 2012 if it is not already cached.
2. Extract the original images and XML annotations.
3. Read the required training and validation image names.
4. Copy only the selected images.
5. Retain only `person`, `car`, and `bicycle` objects.
6. Convert bounding boxes into YOLO format.
7. Create corresponding label files.
8. Generate the YOLOv8 dataset configuration.

An existing Pascal VOC installation can also be supplied:

```bash
python3 0-prep_data.py --voc-root /path/to/VOCdevkit/VOC2012
```

To disable automatic downloading:

```bash
python3 0-prep_data.py \
    --voc-root /path/to/VOCdevkit/VOC2012 \
    --no-download
```

## Annotation Conversion

Pascal VOC stores bounding boxes as absolute pixel coordinates:

```text
xmin, ymin, xmax, ymax
```

YOLO requires normalized values:

```text
class_id x_center y_center width height
```

The conversion uses:

```text
x_center = ((xmin + xmax) / 2) / image_width
y_center = ((ymin + ymax) / 2) / image_height
width    = (xmax - xmin) / image_width
height   = (ymax - ymin) / image_height
```

All YOLO coordinates are normalized to the range `[0, 1]`.

An example label line is:

```text
0 0.512500 0.461250 0.245000 0.582500
```

Here, `0` represents the `person` class.

## YOLO Configuration

The generated `data.yaml` contains:

```yaml
path: datasets/detection/
train: images/train
val: images/val

nc: 3
names: ["person", "car", "bicycle"]
```

## Dataset Validation

The pipeline ensures that:

* Every selected image has a corresponding label file.
* Training and validation samples remain separate.
* Unrequested images are excluded.
* Objects outside the three target classes are excluded.
* Invalid bounding boxes are skipped.
* Label coordinates are normalized.
* Re-running the pipeline does not leave stale generated files.


The number of images and label files must match within each split.

## Dataset Summary

Update this section after generating the dataset:

| Split      |      Images | Label files | Annotated objects |
| ---------- | ----------: | ----------: | ----------------: |
| Training   | `ADD_COUNT` | `ADD_COUNT` |       `ADD_COUNT` |
| Validation |      `1449` |      `1449` |       `ADD_COUNT` |
| Total      | `ADD_COUNT` | `ADD_COUNT` |       `ADD_COUNT` |


## Running with YOLOv8

After preparing the dataset, a YOLO model can be trained with:

```bash
yolo detect train \
    data=datasets/detection/data.yaml \
    model=yolov8n.pt \
    epochs=50 \
    imgsz=640
```

Training outputs, including evaluation metrics and model weights, are generated under `runs/detect/`.

## Potential Enhancements

Future development may include:

* Training and comparing multiple YOLO model sizes.
* Adding image augmentation with Albumentations.
* Evaluating precision, recall, mAP@50, and mAP@50–95.
* Visualizing predicted and ground-truth bounding boxes.
* Performing confidence-threshold analysis.
* Comparing CPU and GPU inference latency.
* Exporting the trained model to ONNX.
* Creating an interactive detection application.

## What I Learned

This project strengthened my understanding of:

* Object-detection dataset preparation.
* Pascal VOC XML annotations.
* YOLO normalized bounding-box representation.
* Dataset filtering and validation.
* Reproducible data pipelines.
* Defensive file and directory handling.
* Cross-platform development between Windows and Ubuntu.
* Environment and dependency management with Conda.

## Dataset

This project uses the [Pascal VOC 2012 dataset](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/).

The original images and annotations remain subject to the Pascal VOC dataset terms. Large generated dataset files are not intended to be stored directly in this Git repository.

## Author

**Ranjana Singh**

Data & AI learner based in Luxembourg, combining project-management experience with hands-on work in machine learning, deep learning, automation, and computer vision.
