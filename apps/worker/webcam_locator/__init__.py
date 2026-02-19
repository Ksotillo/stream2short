"""webcam_locator – Detect webcam overlay rectangles in stream frames."""

from webcam_locator.core import detect_webcam_bbox
from webcam_locator.consensus import detect_from_frames

__all__ = ["detect_webcam_bbox", "detect_from_frames"]
