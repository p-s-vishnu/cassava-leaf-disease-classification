import numpy as np
import pytest
import torch

from cassava import config
from cassava.model import BaseClassifier, CassavaClassifier


@pytest.fixture
def model():
    return CassavaClassifier(config.MODEL_NAME, pretrained=False)


def test_model_instantiation(model):
    assert model is not None
    assert isinstance(model, BaseClassifier)


def test_forward_pass_output_shape(model):
    batch_size = 2
    x = torch.randn(batch_size, 3, config.SIZE, config.SIZE)
    model.eval()
    with torch.no_grad():
        output = model(x)
    assert output.shape == (batch_size, config.TARGET_SIZE)


def test_forward_pass_produces_logits(model):
    x = torch.randn(1, 3, config.SIZE, config.SIZE)
    model.eval()
    with torch.no_grad():
        output = model(x)
    assert not torch.isnan(output).any()
    assert not torch.isinf(output).any()


def test_predit_as_json_returns_valid_dict(model):
    image = np.random.randint(0, 255, (config.SIZE, config.SIZE, 3), dtype=np.uint8)
    model.eval()
    result = model.predit_as_json(image)
    assert isinstance(result, dict)
    assert "class_name" in result
    assert "confidence" in result


def test_predit_as_json_class_name_is_valid(model):
    image = np.random.randint(0, 255, (config.SIZE, config.SIZE, 3), dtype=np.uint8)
    model.eval()
    result = model.predit_as_json(image, threshold=0.0)
    valid_names = set(config.LABEL_MAP.values())
    assert result["class_name"] in valid_names


def test_predit_as_json_low_confidence_rejection(model):
    image = np.random.randint(0, 255, (config.SIZE, config.SIZE, 3), dtype=np.uint8)
    model.eval()
    result = model.predit_as_json(image, threshold=1.0)
    assert result["class_name"] == "Not a Cassava leaf!"
    assert result["confidence"] is None
