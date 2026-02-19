"""
yolo_detect.py – Webcam detection using trained YOLOv8 model.

Uses the fine-tuned YOLOv8n model to detect webcam overlays directly,
replacing the face-expansion heuristic with learned detection.
"""

import logging
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

log = logging.getLogger(__name__)

_MODEL = None
_MODEL_PATH = None

# Search paths for the trained model
_MODEL_SEARCH = [
    "runs/detect/runs/webcam_detect/yolov8n_v3/weights/best.pt",
    "runs/detect/runs/webcam_detect/yolov8n_webcam2/weights/best.pt",
    "runs/detect/runs/webcam_detect/yolov8n_fast/weights/best.pt",
    "webcam_locator/models/webcam_yolov8n.pt",
    "models/webcam_yolov8n.pt",
]


def _get_model():
    global _MODEL, _MODEL_PATH
    if _MODEL is not None:
        return _MODEL

    for path_str in _MODEL_SEARCH:
        path = Path(path_str)
        if not path.is_absolute():
            path = Path.cwd() / path
        if path.exists():
            _MODEL_PATH = path
            break

    if _MODEL_PATH is None:
        log.info("YOLO webcam model not found; will use heuristic detection")
        return None

    try:
        from ultralytics import YOLO
        _MODEL = YOLO(str(_MODEL_PATH))
        log.info("Loaded YOLO webcam detector from %s", _MODEL_PATH)
        return _MODEL
    except Exception as exc:
        log.warning("Failed to load YOLO model: %s", exc)
        return None


def detect_with_yolo(
    image_bgr: np.ndarray,
    confidence: float = 0.25,
) -> Optional[dict]:
    """Run YOLO webcam detection on a single frame.

    Returns the best detection as a dict with x, y, width, height, confidence,
    or None if no webcam detected.
    """
    model = _get_model()
    if model is None:
        return None

    results = model.predict(image_bgr, conf=confidence, verbose=False)

    if not results or len(results[0].boxes) == 0:
        return None

    # Pick the detection with highest confidence
    boxes = results[0].boxes
    best_idx = boxes.conf.argmax().item()
    best_box = boxes.xyxy[best_idx].cpu().numpy()
    best_conf = float(boxes.conf[best_idx].cpu().numpy())

    x1, y1, x2, y2 = best_box
    x, y, w, h = int(x1), int(y1), int(x2 - x1), int(y2 - y1)

    # Clamp to frame
    frame_h, frame_w = image_bgr.shape[:2]
    x = max(0, min(x, frame_w - 10))
    y = max(0, min(y, frame_h - 10))
    w = max(10, min(w, frame_w - x))
    h = max(10, min(h, frame_h - y))

    return {
        "x": x, "y": y, "width": w, "height": h,
        "confidence": best_conf,
        "source": "yolo",
    }


def is_yolo_available() -> bool:
    """Check if the YOLO model is available."""
    return _get_model() is not None
