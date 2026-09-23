#!/usr/bin/env python3
"""Train an Ultralytics YOLO model with configurable augmentation."""

import os
from typing import Any, Dict, List, Optional, Tuple, Union

from ultralytics import YOLO


def train_with_augmentation(
    data: Optional[str] = None,
    model_path: str = "yolov8n.pt",
    epochs: int = 50,
    imgsz: Union[int, Tuple[int, int]] = 640,
    batch: int = 16,
    augmentation: bool = True,
    yolo_aug_params: Optional[Dict[str, Any]] = None,
    albumentations_transforms: Optional[List[Any]] = None,
    save: bool = True,
    plots: bool = True,
    verbose: bool = True,
    data_yaml: Optional[str] = None,
    model: Optional[str] = None,
    aug: Optional[Any] = None,
    custom_albu: Optional[List[Any]] = None,
):
    """Train a YOLO model with native or custom augmentation.

    Args:
        data: Path to the YOLO dataset YAML file.
        model_path: Path to pretrained weights or a model configuration.
        epochs: Number of training epochs.
        imgsz: Training image size as an integer or height-width tuple.
        batch: Number of images in each training batch.
        augmentation: Whether data augmentation is enabled.
        yolo_aug_params: YOLO-native augmentation parameters.
        albumentations_transforms: Custom Albumentations transforms.
        save: Whether model checkpoints should be saved.
        plots: Whether training and validation plots should be created.
        verbose: Whether detailed training output should be displayed.
        data_yaml: Alternative name for the dataset YAML path.
        model: Alternative name for the model path.
        aug: Alternative native-augmentation configuration.
        custom_albu: Alternative custom Albumentations transform list.

    Returns:
        A tuple containing the trained YOLO model and training results.

    Raises:
        ValueError: If a dataset YAML path is not supplied.
        TypeError: If yolo_aug_params is not a dictionary.
    """
    if data is None:
        data = data_yaml

    if data is None:
        raise ValueError("A dataset YAML path must be provided.")

    if model is not None:
        model_path = model

    if custom_albu is not None:
        albumentations_transforms = custom_albu

    if isinstance(aug, bool):
        augmentation = aug
    elif isinstance(aug, dict):
        yolo_aug_params = aug
    elif aug is not None:
        raise TypeError("aug must be a boolean, dictionary, or None.")

    if (
        yolo_aug_params is not None
        and not isinstance(yolo_aug_params, dict)
    ):
        raise TypeError("yolo_aug_params must be a dictionary or None.")

    training_arguments = {
        "data": data,
        "epochs": epochs,
        "imgsz": imgsz,
        "batch": batch,
        "augment": augmentation,
        "save": save,
        "plots": plots,
        "verbose": verbose,
    }

    if yolo_aug_params:
        training_arguments.update(yolo_aug_params)

    if albumentations_transforms is not None:
        training_arguments["augmentations"] = (
            albumentations_transforms
        )

    if os.name == "nt":
        training_arguments["workers"] = 0

    yolo_model = YOLO(model_path)
    results = yolo_model.train(**training_arguments)

    return yolo_model, results
