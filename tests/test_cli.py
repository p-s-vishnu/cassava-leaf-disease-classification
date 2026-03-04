import json
import re

import cv2
import numpy as np
import pandas as pd
import pytest
import torch
from typer.testing import CliRunner

from cassava.cli import app

runner = CliRunner()

_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")


def _strip_ansi(text: str) -> str:
    return _ANSI_RE.sub("", text)


@pytest.fixture
def model_checkpoint(tmp_path):
    """Create a dummy model checkpoint for testing."""
    from cassava import config
    from cassava.model import CassavaClassifier

    m = CassavaClassifier(config.MODEL_NAME, pretrained=False)
    ckpt_path = tmp_path / "test_model.pth"
    torch.save({"model": m.state_dict()}, ckpt_path)
    return ckpt_path


# --- Help tests ---


def test_app_help():
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "train" in result.output
    assert "predict" in result.output


def test_train_help():
    result = runner.invoke(app, ["train", "--help"])
    assert result.exit_code == 0
    output = _strip_ansi(result.output)
    assert "--model" in output
    assert "--epochs" in output
    assert "--lr" in output


def test_predict_help():
    result = runner.invoke(app, ["predict", "--help"])
    assert result.exit_code == 0
    output = _strip_ansi(result.output)
    assert "--checkpoint" in output
    assert "--threshold" in output


def test_evaluate_help():
    result = runner.invoke(app, ["evaluate", "--help"])
    assert result.exit_code == 0
    output = _strip_ansi(result.output)
    assert "--checkpoint" in output
    assert "--data" in output


# --- Train tests ---


def test_train_validates_data_dir(tmp_path):
    """Train should fail gracefully if data dir doesn't have train.csv."""
    result = runner.invoke(app, ["train", "--data-csv", str(tmp_path / "nonexistent.csv")])
    assert result.exit_code != 0 or "not found" in result.output.lower()


# --- Predict tests ---


def test_predict_single_image(tmp_path, model_checkpoint):
    img = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
    img_path = tmp_path / "test.jpg"
    cv2.imwrite(str(img_path), img)

    result = runner.invoke(app, ["predict", str(img_path), "--checkpoint", str(model_checkpoint)])
    assert result.exit_code == 0
    output = json.loads(result.output)
    assert "class_name" in output
    assert "confidence" in output


def test_predict_directory(tmp_path, model_checkpoint):
    img_dir = tmp_path / "imgs"
    img_dir.mkdir()
    for i in range(3):
        img = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
        cv2.imwrite(str(img_dir / f"img_{i}.jpg"), img)

    result = runner.invoke(app, ["predict", str(img_dir), "--checkpoint", str(model_checkpoint)])
    assert result.exit_code == 0
    output = json.loads(result.output)
    assert isinstance(output, list)
    assert len(output) == 3


def test_predict_missing_checkpoint(tmp_path):
    img = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
    img_path = tmp_path / "test.jpg"
    cv2.imwrite(str(img_path), img)

    result = runner.invoke(app, ["predict", str(img_path), "--checkpoint", "nonexistent.pth"])
    assert result.exit_code != 0


# --- Evaluate tests ---


def test_evaluate_computes_metrics(tmp_path, model_checkpoint):
    img_dir = tmp_path / "images"
    img_dir.mkdir()
    rows = []
    for i in range(10):
        img = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
        fname = f"img_{i}.jpg"
        cv2.imwrite(str(img_dir / fname), img)
        rows.append({"image_id": fname, "label": i % 5})

    csv_path = tmp_path / "test_data.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False)

    result = runner.invoke(
        app,
        [
            "evaluate",
            "--checkpoint",
            str(model_checkpoint),
            "--data",
            str(csv_path),
            "--data-dir",
            str(img_dir),
        ],
    )
    assert result.exit_code == 0
    assert "accuracy" in result.output.lower() or "Accuracy" in result.output


def test_evaluate_saves_json(tmp_path, model_checkpoint):
    img_dir = tmp_path / "images"
    img_dir.mkdir()
    rows = []
    for i in range(5):
        img = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
        fname = f"img_{i}.jpg"
        cv2.imwrite(str(img_dir / fname), img)
        rows.append({"image_id": fname, "label": i % 5})

    csv_path = tmp_path / "test_data.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    out_json = tmp_path / "metrics.json"

    result = runner.invoke(
        app,
        [
            "evaluate",
            "--checkpoint",
            str(model_checkpoint),
            "--data",
            str(csv_path),
            "--data-dir",
            str(img_dir),
            "--output",
            str(out_json),
        ],
    )
    assert result.exit_code == 0
    assert out_json.exists()

    metrics = json.loads(out_json.read_text())
    assert "accuracy" in metrics
    assert "per_class" in metrics


def test_evaluate_missing_checkpoint():
    result = runner.invoke(
        app,
        [
            "evaluate",
            "--checkpoint",
            "nonexistent.pth",
            "--data",
            "nonexistent.csv",
        ],
    )
    assert result.exit_code != 0
