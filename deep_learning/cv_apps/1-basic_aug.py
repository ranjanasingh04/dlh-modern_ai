#!/usr/bin/env python3
"""Provide basic Albumentations augmentation for object detection."""

from typing import List, Tuple

import albumentations as A
import numpy as np


def basic_aug(
    image: np.ndarray,
    bboxes: List[List[int]],
    labels: List[int],
) -> Tuple[np.ndarray, np.ndarray, List[int]]:
    """Apply reproducible augmentation to an image and bounding boxes.

    The bounding boxes are expected in Pascal VOC format:

        [x_min, y_min, x_max, y_max]

    Args:
        image: Input image as a NumPy array.
        bboxes: Bounding boxes in Pascal VOC format.
        labels: Class label corresponding to each bounding box.

    Returns:
        A tuple containing the augmented image, augmented bounding boxes
        as a NumPy array, and the corresponding class labels.
    """
    transform = A.Compose(
        [
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.2),
            A.Affine(
                translate_percent={
                    "x": (-0.1, 0.1),
                    "y": (-0.1, 0.1),
                },
                scale=(0.9, 1.1),
                rotate=(-30, 0),
                p=0.5,
            ),
        ],
        bbox_params=A.BboxParams(
            format="pascal_voc",
            label_fields=["labels"],
            clip=True,
            filter_invalid_bboxes=True,
        ),
        seed=42,
    )

    transformed = transform(
        image=image,
        bboxes=bboxes,
        labels=list(labels),
    )

    augmented_boxes = np.asarray(
        transformed["bboxes"],
        dtype=np.float32,
    ).reshape(-1, 4)

    return (
        transformed["image"],
        augmented_boxes,
        list(transformed["labels"]),
    )
