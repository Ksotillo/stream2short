"""
faces.py – Face detection using OpenCV DNN (primary) with Haar cascade fallback.
"""

import logging
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

log = logging.getLogger(__name__)

_DNN_NET: Optional[cv2.dnn.Net] = None
_DNN_SEARCHED = False
_HAAR_CASCADE: Optional[cv2.CascadeClassifier] = None

# DNN model files (shipped with opencv-contrib or downloadable)
_DNN_PROTO = "deploy.prototxt"
_DNN_MODEL = "res10_300x300_ssd_iter_140000.caffemodel"
_DNN_CONFIDENCE_THRESHOLD = 0.15


def _find_dnn_model() -> Optional[tuple[str, str]]:
    """Locate the DNN face detector model files."""
    search_dirs = [
        Path(__file__).parent / "models",
        Path.cwd() / "models",
        Path.home() / ".webcam_locator" / "models",
    ]
    for d in search_dirs:
        proto = d / _DNN_PROTO
        model = d / _DNN_MODEL
        if proto.exists() and model.exists():
            return str(proto), str(model)
    return None


def _get_dnn_net() -> Optional[cv2.dnn.Net]:
    global _DNN_NET, _DNN_SEARCHED
    if _DNN_NET is not None:
        return _DNN_NET
    if _DNN_SEARCHED:
        return None

    _DNN_SEARCHED = True
    paths = _find_dnn_model()
    if paths is None:
        log.info("DNN face model not found; will use Haar cascade only")
        return None

    proto, model = paths
    _DNN_NET = cv2.dnn.readNetFromCaffe(proto, model)
    log.info("Loaded DNN face detector from %s", Path(model).parent)
    return _DNN_NET


def _get_haar_cascade() -> cv2.CascadeClassifier:
    global _HAAR_CASCADE
    if _HAAR_CASCADE is None:
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        _HAAR_CASCADE = cv2.CascadeClassifier(cascade_path)
    return _HAAR_CASCADE


def _detect_dnn(image_bgr: np.ndarray, conf_threshold: float = _DNN_CONFIDENCE_THRESHOLD) -> list[dict]:
    """Detect faces using OpenCV DNN SSD model. Returns list of face dicts."""
    net = _get_dnn_net()
    if net is None:
        return []

    h, w = image_bgr.shape[:2]
    blob = cv2.dnn.blobFromImage(
        cv2.resize(image_bgr, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0),
    )
    net.setInput(blob)
    detections = net.forward()

    faces = []
    for i in range(detections.shape[2]):
        confidence = float(detections[0, 0, i, 2])
        if confidence < conf_threshold:
            continue
        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        x1, y1, x2, y2 = box.astype(int)
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(w, x2)
        y2 = min(h, y2)
        if x2 - x1 < 15 or y2 - y1 < 15:
            continue
        faces.append({
            "x": int(x1), "y": int(y1),
            "width": int(x2 - x1), "height": int(y2 - y1),
            "confidence": confidence,
            "method": "dnn",
        })
    return faces


def _detect_haar(image_bgr: np.ndarray) -> list[dict]:
    """Detect faces using Haar cascade. Returns list of face dicts."""
    cascade = _get_haar_cascade()
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    rects = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(20, 20))
    faces = []
    for (x, y, w, h) in rects:
        faces.append({
            "x": int(x), "y": int(y),
            "width": int(w), "height": int(h),
            "confidence": 0.6,
            "method": "haar",
        })
    return faces


def detect_faces(image_bgr: np.ndarray) -> list[dict]:
    """Detect faces using DNN (primary) with Haar cascade fallback.

    Returns a list of face dicts: {x, y, width, height, confidence, method}.
    """
    faces = _detect_dnn(image_bgr)
    if faces:
        return faces

    faces = _detect_haar(image_bgr)
    return faces
