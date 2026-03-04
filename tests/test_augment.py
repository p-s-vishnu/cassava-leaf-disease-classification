import numpy as np
import pytest
import torch

from cassava import config
from cassava.augment import get_transforms


@pytest.fixture
def dummy_image():
    return np.random.randint(0, 255, (600, 600, 3), dtype=np.uint8)


def test_train_transform_returns_tensor(dummy_image):
    transform = get_transforms(data="train")
    result = transform(image=dummy_image)["image"]
    assert isinstance(result, torch.Tensor)


def test_train_transform_output_shape(dummy_image):
    transform = get_transforms(data="train")
    result = transform(image=dummy_image)["image"]
    assert result.shape == (3, config.SIZE, config.SIZE)


def test_valid_transform_output_shape(dummy_image):
    transform = get_transforms(data="valid")
    result = transform(image=dummy_image)["image"]
    assert result.shape == (3, config.SIZE, config.SIZE)


def test_test_transform_output_shape(dummy_image):
    transform = get_transforms(data="test")
    result = transform(image=dummy_image)["image"]
    assert result.shape == (3, config.SIZE, config.SIZE)


def test_transforms_normalize_values(dummy_image):
    transform = get_transforms(data="valid")
    result = transform(image=dummy_image)["image"]
    # After ImageNet normalization, values should not be in [0, 255]
    assert result.max() < 10.0
    assert result.min() > -10.0


def test_valid_transform_is_deterministic(dummy_image):
    valid_t = get_transforms(data="valid")
    # Valid is deterministic (Resize only)
    valid_result1 = valid_t(image=dummy_image)["image"]
    valid_result2 = valid_t(image=dummy_image)["image"]
    assert torch.equal(valid_result1, valid_result2)
