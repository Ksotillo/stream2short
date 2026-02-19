"""
yolo_train.py – Train YOLOv8 Nano on our webcam overlay dataset.

Uses transfer learning (frozen backbone) for fast convergence on small dataset.
"""

import logging
from pathlib import Path

log = logging.getLogger(__name__)


def train(
    data_yaml: Path,
    epochs: int = 80,
    imgsz: int = 640,
    batch: int = 16,
    project: str = "runs/webcam_detect",
    name: str = "yolov8n_webcam",
) -> Path:
    """Train YOLOv8n on the webcam dataset."""
    from ultralytics import YOLO

    model = YOLO("yolov8n.pt")

    results = model.train(
        data=str(data_yaml),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        project=project,
        name=name,
        patience=20,
        save=True,
        plots=True,
        augment=True,
        hsv_h=0.015,
        hsv_s=0.3,
        hsv_v=0.3,
        degrees=3.0,
        translate=0.1,
        scale=0.3,
        flipud=0.0,
        fliplr=0.3,
        mosaic=0.5,
        mixup=0.1,
        verbose=True,
    )

    best_path = Path(project) / name / "weights" / "best.pt"
    print(f"\nTraining complete. Best model: {best_path}")
    return best_path


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train YOLOv8n webcam detector")
    parser.add_argument("--data", type=Path, default=Path("yolo_dataset/data.yaml"))
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch", type=int, default=16)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    train(args.data, epochs=args.epochs, batch=args.batch)
