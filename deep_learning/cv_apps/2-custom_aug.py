#!/usr/bin/env python3
"""Provide custom Albumentations augmentation for object detection."""

from typing import List, Tuple

import albumentations as A
import numpy as np


def custom_aug(
    image: np.ndarray,
    bboxes: List[List[int]],
    labels: List[int],
) -> Tuple[np.ndarray, np.ndarray, List[int]]:
    """Apply custom augmentation to an image and bounding boxes.

    Bounding boxes must use Pascal VOC format:

        [x_min, y_min, x_max, y_max]

    Args:
        image: Input image as a NumPy array.
        bboxes: Bounding boxes in Pascal VOC format.
        labels: Class labels corresponding to the bounding boxes.

    Returns:
        A tuple containing the augmented image, augmented bounding
        boxes as a NumPy array, and the corresponding labels.
    """
    transform = A.Compose(
        [
            A.MotionBlur(
                blur_limit=5,
                p=0.9,
            ),
            A.OneOf(
                [
                    A.ElasticTransform(
                        alpha=1,
                        sigma=50,
                        p=0.2,
                    ),
                    A.OpticalDistortion(
                        distort_limit=0.05,
                        p=0.2,
                    ),
                ],
                p=0.9,
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
