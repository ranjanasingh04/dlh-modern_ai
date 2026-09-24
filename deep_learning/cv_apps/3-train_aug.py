#!/usr/bin/env python3
"""Train an Ultralytics YOLO model with configurable augmentation."""

from ultralytics import YOLO


def train_with_augmentation(data, model_path="yolov8n.pt", epochs=100,
                            imgsz=640, batch=16, augmentation=False,
                            yolo_aug_params=None,
                            albumentations_transforms=None,
                            save=False, plots=False, verbose=False):
    """Train a YOLO model with optional augmentation configuration.

    Args:
        data (str): Path to the YOLO dataset configuration file.
        model_path (str): Model weights or configuration file.
        epochs (int): Number of training epochs.
        imgsz (int or tuple): Training image dimensions.
        batch (int): Number of images in each batch.
        augmentation (bool): Whether default augmentation is enabled.
        yolo_aug_params (dict): Custom native YOLO augmentation values.
        albumentations_transforms (list): Custom Albumentations transforms.
        save (bool): Whether to save model checkpoints.
        plots (bool): Whether to generate training plots.
        verbose (bool): Whether to show detailed training output.

    Returns:
        tuple: The trained YOLO model and complete training results.
    """
    model = YOLO(model_path)

    train_params = {
        "data": data,
        "epochs": epochs,
        "imgsz": imgsz,
        "batch": batch,
        "save": save,
        "plots": plots,
        "verbose": verbose,
    }

    if albumentations_transforms is not None:
        train_params["augmentations"] = albumentations_transforms
    elif yolo_aug_params is not None:
        train_params.update(yolo_aug_params)
    elif not augmentation:
        train_params.update({
            "hsv_h": 0.0,
            "hsv_s": 0.0,
            "hsv_v": 0.0,
            "degrees": 0.0,
            "translate": 0.0,
            "scale": 0.0,
            "shear": 0.0,
            "perspective": 0.0,
            "flipud": 0.0,
            "fliplr": 0.0,
            "bgr": 0.0,
            "mosaic": 0.0,
            "mixup": 0.0,
            "cutmix": 0.0,
            "copy_paste": 0.0,
            "auto_augment": None,
            "erasing": 0.0,
        })

    results = model.train(**train_params)

    return model, results
