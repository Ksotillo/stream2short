"""webcam_locator_bridge.py – Adapter between Stream2Short and the webcam_locator package.

Provides a drop-in detection strategy that uses the fine-tuned YOLOv8 model
from the webcam_locator package, returning results in the format that
Stream2Short's WebcamRegion expects.

This runs as Strategy 0 inside detect_webcam_region(), before Gemini or OpenCV.
If the YOLO model is not found, this silently returns None so the existing
Gemini → OpenCV fallback chain continues uninterrupted.
"""

import logging
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

log = logging.getLogger(__name__)

# Frames to sample per video. These timestamps match what detect_webcam_region() uses.
_DEFAULT_SAMPLE_TIMES = [3.0, 10.0, 15.0]


def _extract_frames_at_times(
    video_path: str,
    sample_times: list,
) -> list:
    """Extract frames from a video at specific timestamps using cv2.VideoCapture."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        log.warning("webcam_locator_bridge: Cannot open video: %s", video_path)
        return []

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps if fps > 0 else 0

    frames = []
    for ts in sample_times:
        # Clamp timestamp to video duration
        ts_clamped = min(ts, max(0, duration - 0.5)) if duration > 0 else ts
        frame_pos = int(ts_clamped * fps)
        frame_pos = max(0, min(frame_pos, total_frames - 1))

        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_pos)
        ret, frame = cap.read()
        if ret and frame is not None:
            frames.append(frame)

    cap.release()
    return frames


def _derive_position(result: dict, frame_w: int, frame_h: int) -> str:
    """Convert webcam_locator result into a Stream2Short position string.

    Stream2Short uses: 'top-left', 'top-right', 'bottom-left', 'bottom-right', 'full'
    webcam_locator uses: corner field ('top-left', 'top-right', etc.) + type field
    """
    wl_type = result.get("type", "none")
    corner = result.get("corner", "unknown")

    if wl_type == "full_cam":
        return "full"

    # For corner_overlay, the corner field is set precisely
    if wl_type == "corner_overlay" and corner not in ("unknown", ""):
        return corner  # already in the right format: 'top-left', 'top-right', etc.

    # For side_box, top_band, bottom_band, center_box — derive from bbox center
    x = result.get("x", 0)
    y = result.get("y", 0)
    w = result.get("width", 1)
    h = result.get("height", 1)
    cx = x + w // 2
    cy = y + h // 2

    is_right = cx > frame_w // 2
    is_bottom = cy > frame_h // 2

    if is_right and is_bottom:
        return "bottom-right"
    elif is_right:
        return "top-right"
    elif is_bottom:
        return "bottom-left"
    else:
        return "top-left"


def detect_with_locator(
    video_path: str,
    sample_times: list = None,
) -> Optional[dict]:
    """Run webcam_locator YOLOv8 detection on a video file.

    Returns a dict with all fields needed to construct a WebcamRegion, or None
    if the locator is unavailable or no webcam was detected.

    Return dict fields:
        x, y, width, height  – pixel coordinates (frame resolution)
        position             – Stream2Short position string
        gemini_type          – webcam type (side_box, corner_overlay, full_cam, …)
        gemini_confidence    – detection confidence 0–1
        effective_type       – same as gemini_type (for compatibility)
        corner               – raw corner value from classifier
        locator_reason       – human-readable reason string
    """
    if sample_times is None:
        sample_times = _DEFAULT_SAMPLE_TIMES

    # Import webcam_locator lazily so a missing package is a soft failure
    try:
        from webcam_locator.yolo_detect import is_yolo_available
        from webcam_locator.consensus import detect_from_frames
    except ImportError:
        log.debug("webcam_locator package not importable — skipping YOLO strategy")
        return None

    if not is_yolo_available():
        log.info("webcam_locator: YOLO model not found — skipping YOLO strategy")
        return None

    # Extract frames from video
    frames = _extract_frames_at_times(video_path, sample_times)
    if not frames:
        log.warning("webcam_locator: Could not extract any frames from %s", video_path)
        return None

    log.info("webcam_locator: Running YOLO consensus on %d frames", len(frames))

    # Run multi-frame consensus detection (YOLO only, no Gemini to keep it fast)
    result = detect_from_frames(frames, use_gemini=False)

    if not result.get("found"):
        log.info("webcam_locator: No webcam detected (%s)", result.get("reason", ""))
        return None

    frame_h, frame_w = frames[0].shape[:2]
    position = _derive_position(result, frame_w, frame_h)
    wl_type = result.get("type", "unknown")
    confidence = result.get("confidence", 0.0)

    log.info(
        "webcam_locator: ✅ %s @ (%d,%d) %dx%d  pos=%s  conf=%.2f  [%s]",
        wl_type,
        result["x"], result["y"],
        result["width"], result["height"],
        position, confidence,
        result.get("reason", ""),
    )

    return {
        "x": int(result["x"]),
        "y": int(result["y"]),
        "width": int(result["width"]),
        "height": int(result["height"]),
        "position": position,
        "gemini_type": wl_type,
        "gemini_confidence": float(confidence),
        "effective_type": wl_type,
        "corner": result.get("corner", "unknown"),
        "locator_reason": result.get("reason", "YOLO"),
    }
