"""Cassava leaf disease classifier CLI."""

from pathlib import Path
from typing import Optional

import typer

from cassava import config

app = typer.Typer(
    name="cassava",
    help="Cassava leaf disease classification CLI.",
    add_completion=False,
)


def _apply_overrides(
    model: Optional[str] = None,
    epochs: Optional[int] = None,
    lr: Optional[float] = None,
    batch_size: Optional[int] = None,
    image_size: Optional[int] = None,
    data_dir: Optional[str] = None,
) -> None:
    """Override config.py constants at runtime."""
    if model is not None:
        config.MODEL_NAME = model
    if epochs is not None:
        config.EPOCHS = epochs
    if lr is not None:
        config.LR = lr
    if batch_size is not None:
        config.BATCH_SIZE = batch_size
    if image_size is not None:
        config.SIZE = image_size
    if data_dir is not None:
        config.TRAIN_PATH = data_dir


@app.command()
def train(
    model: Optional[str] = typer.Option(None, help="timm model name"),
    epochs: Optional[int] = typer.Option(None, help="Number of training epochs"),
    lr: Optional[float] = typer.Option(None, help="Learning rate"),
    batch_size: Optional[int] = typer.Option(None, "--batch-size", help="Batch size"),
    image_size: Optional[int] = typer.Option(None, "--image-size", help="Input image size"),
    folds: Optional[str] = typer.Option(None, help="Comma-separated fold indices (e.g. 0,1,2)"),
    data_dir: Optional[str] = typer.Option(None, "--data-dir", help="Path to training images"),
    data_csv: str = typer.Option("data/train.csv", "--data-csv", help="Path to training CSV"),
    output_dir: str = typer.Option("model/", "--output-dir", help="Path to save checkpoints"),
) -> None:
    """Run stratified k-fold training."""
    _apply_overrides(
        model=model,
        epochs=epochs,
        lr=lr,
        batch_size=batch_size,
        image_size=image_size,
        data_dir=data_dir,
    )
    if folds is not None:
        config.TRN_FOLD = [int(f) for f in folds.split(",")]

    typer.echo(f"Training {config.MODEL_NAME} for {config.EPOCHS} epochs...")

    from main import run_training

    try:
        run_training(data_csv=data_csv, output_dir=output_dir)
    except FileNotFoundError as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(code=1)


@app.command()
def predict(
    image_path: str = typer.Argument(..., help="Path to image file or directory"),
    checkpoint: str = typer.Option(..., help="Path to model checkpoint (.pth)"),
    threshold: float = typer.Option(0.5, help="Confidence threshold for rejection"),
    model_name: Optional[str] = typer.Option(None, "--model", help="timm model name"),
) -> None:
    """Classify a cassava leaf image."""
    import json

    import cv2
    import torch

    from cassava.model import CassavaClassifier

    ckpt_path = Path(checkpoint)
    if not ckpt_path.exists():
        typer.echo(f"Error: Checkpoint not found: {checkpoint}", err=True)
        raise typer.Exit(code=1)

    name = model_name or config.MODEL_NAME
    model = CassavaClassifier(name, pretrained=False)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    model.load_state_dict(ckpt["model"])
    model.eval()

    img_path = Path(image_path)
    if img_path.is_file():
        image = cv2.imread(str(img_path))
        if image is None:
            typer.echo(f"Error: Could not read image: {img_path}", err=True)
            raise typer.Exit(code=1)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        result = model.predit_as_json(image, threshold=threshold)
        typer.echo(json.dumps(result, default=str))
    elif img_path.is_dir():
        results = []
        for f in sorted(img_path.iterdir()):
            if f.suffix.lower() in (".jpg", ".jpeg", ".png"):
                image = cv2.imread(str(f))
                if image is None:
                    continue
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                r = model.predit_as_json(image, threshold=threshold)
                r["file"] = f.name
                results.append(r)
        typer.echo(json.dumps(results, default=str))
    else:
        typer.echo(f"Error: Path not found: {image_path}", err=True)
        raise typer.Exit(code=1)


@app.command()
def evaluate(
    checkpoint: str = typer.Option(..., help="Path to model checkpoint (.pth)"),
    data: str = typer.Option(..., help="Path to CSV with image_id and label columns"),
    data_dir: Optional[str] = typer.Option(None, "--data-dir", help="Path to image directory"),
    output: Optional[str] = typer.Option(None, help="Path to save metrics as JSON"),
    model_name: Optional[str] = typer.Option(None, "--model", help="timm model name"),
) -> None:
    """Compute classification metrics on a labeled dataset."""
    import json

    import cv2
    import pandas as pd
    import torch
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

    from cassava.augment import get_transforms
    from cassava.model import CassavaClassifier

    ckpt_path = Path(checkpoint)
    if not ckpt_path.exists():
        typer.echo(f"Error: Checkpoint not found: {checkpoint}", err=True)
        raise typer.Exit(code=1)

    csv_path = Path(data)
    if not csv_path.exists():
        typer.echo(f"Error: CSV not found: {data}", err=True)
        raise typer.Exit(code=1)

    name = model_name or config.MODEL_NAME
    model = CassavaClassifier(name, pretrained=False)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    model.load_state_dict(ckpt["model"])
    model.eval()

    transform = get_transforms(data="test")
    df = pd.read_csv(csv_path)
    img_dir_path = Path(data_dir) if data_dir else Path(config.TRAIN_PATH)

    preds = []
    labels = df["label"].values.tolist()

    for _, row in df.iterrows():
        img_path = img_dir_path / row["image_id"]
        image = cv2.imread(str(img_path))
        if image is None:
            typer.echo(f"Warning: Could not read {img_path}, skipping", err=True)
            continue
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        tensor = transform(image=image)["image"].unsqueeze(0)
        with torch.no_grad():
            logits = model(tensor)
            pred = int(logits.argmax(dim=1).item())
        preds.append(pred)

    acc = accuracy_score(labels, preds)
    target_names = [config.LABEL_MAP[i] for i in range(config.TARGET_SIZE)]
    report = classification_report(labels, preds, target_names=target_names, output_dict=True)
    cm = confusion_matrix(labels, preds)

    typer.echo(f"\nAccuracy: {acc:.4f}\n")
    typer.echo("Per-class metrics:")
    typer.echo(classification_report(labels, preds, target_names=target_names))
    typer.echo("Confusion Matrix:")
    typer.echo(str(cm))

    if output:
        metrics = {
            "accuracy": acc,
            "per_class": {
                k: v
                for k, v in report.items()
                if k not in ("accuracy", "macro avg", "weighted avg")
            },
            "macro_avg": report.get("macro avg"),
            "weighted_avg": report.get("weighted avg"),
            "confusion_matrix": cm.tolist(),
        }
        Path(output).write_text(json.dumps(metrics, indent=2))
        typer.echo(f"\nMetrics saved to {output}")
