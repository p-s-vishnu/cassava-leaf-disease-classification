"""This module includes the augmentation pipeline"""

from albumentations import (
    Compose,
    HorizontalFlip,
    Normalize,
    RandomResizedCrop,
    Resize,
    ShiftScaleRotate,
    Transpose,
    VerticalFlip,
)
from albumentations.pytorch import ToTensorV2

from cassava import config


def get_transforms(*, data):
    """Returns augmentation specific to train/valid data"""
    if data == "train":
        return Compose(
            [
                RandomResizedCrop(size=(config.SIZE, config.SIZE)),
                Transpose(p=0.5),
                HorizontalFlip(p=0.5),
                VerticalFlip(p=0.5),
                ShiftScaleRotate(p=0.5),
                Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2(),
            ]
        )
    if data in ["valid", "test"]:
        return Compose(
            [
                Resize(height=config.SIZE, width=config.SIZE),
                Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2(),
            ]
        )
    raise ValueError(f"Unimplemented data transform: {data}")
