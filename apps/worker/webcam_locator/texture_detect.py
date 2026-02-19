"""
texture_detect.py – Detect webcam overlay rectangles by texture/color analysis.

Instead of expanding from a face, find the overlay directly by detecting
rectangular regions where interior texture differs from surrounding gameplay.

Webcam overlays have:
  - Lower texture complexity (face + room bg) vs gameplay (particles, UI, text)
  - Consistent color distribution vs vibrant gameplay
  - Sharp rectangular border with subtle shadow/color shift

Algorithm:
  1. Block-based Laplacian variance map (texture complexity)
  2. Threshold into low-texture / high-texture binary mask
  3. Find rectangular contours matching webcam size priors
  4. Refine edges with Sobel directional filtering
"""

import logging
from typing import Optional

import cv2
import numpy as np

log = logging.getLogger(__name__)

BLOCK_SIZE = 16
WEBCAM_MIN_AREA_PCT = 0.02
WEBCAM_MAX_AREA_PCT = 0.15
WEBCAM_MIN_AR = 0.8
WEBCAM_MAX_AR = 2.2


def compute_texture_map(gray: np.ndarray, block_size: int = BLOCK_SIZE) -> np.ndarray:
    """Compute block-based texture complexity map using local Laplacian variance.

    Returns a downsampled map where each pixel = texture complexity of a block.
    Lower values = smoother regions (likely webcam), higher = busy (gameplay).
    """
    h, w = gray.shape[:2]
    bh = h // block_size
    bw = w // block_size

    lap = cv2.Laplacian(gray, cv2.CV_64F)

    tex_map = np.zeros((bh, bw), dtype=np.float64)
    for by in range(bh):
        for bx in range(bw):
            y0 = by * block_size
            x0 = bx * block_size
            block = lap[y0:y0 + block_size, x0:x0 + block_size]
            tex_map[by, bx] = block.var()

    return tex_map


def find_texture_rectangles(
    tex_map: np.ndarray,
    frame_w: int, frame_h: int,
    block_size: int = BLOCK_SIZE,
) -> list[dict]:
    """Find rectangular low-texture regions that match webcam size priors."""
    if tex_map.size == 0:
        return []

    # Adaptive threshold: below median = "low texture"
    median_val = np.median(tex_map)
    threshold = median_val * 0.6

    # Binary mask: 1 = low texture (potential webcam)
    mask = (tex_map < threshold).astype(np.uint8) * 255

    # Morphological cleanup
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    frame_area = frame_w * frame_h
    candidates = []

    for cnt in contours:
        rx, ry, rw, rh = cv2.boundingRect(cnt)

        # Convert block coords to pixel coords
        px = rx * block_size
        py = ry * block_size
        pw = rw * block_size
        ph = rh * block_size

        # Clamp to frame
        pw = min(pw, frame_w - px)
        ph = min(ph, frame_h - py)

        area_ratio = (pw * ph) / frame_area
        ar = pw / ph if ph > 0 else 0

        if area_ratio < WEBCAM_MIN_AREA_PCT or area_ratio > WEBCAM_MAX_AREA_PCT:
            continue
        if ar < WEBCAM_MIN_AR or ar > WEBCAM_MAX_AR:
            continue

        # Score by how rectangular the contour is (area vs bounding rect)
        cnt_area = cv2.contourArea(cnt)
        rect_area = rw * rh
        rectangularity = cnt_area / rect_area if rect_area > 0 else 0

        if rectangularity < 0.5:
            continue

        candidates.append({
            "x": px, "y": py, "width": pw, "height": ph,
            "source": "texture_rect",
            "rectangularity": round(rectangularity, 3),
            "area_ratio": round(area_ratio, 4),
        })

    # Sort by area (larger = more likely to be the full webcam overlay)
    candidates.sort(key=lambda c: c["width"] * c["height"], reverse=True)
    return candidates[:5]


def refine_with_sobel(
    candidate: dict,
    image_bgr: np.ndarray,
    search_range: int = 30,
) -> dict:
    """Refine candidate edges using Sobel directional filtering.

    Looks for the sharp border (shadow/color shift) at the webcam overlay edge.
    """
    frame_h, frame_w = image_bgr.shape[:2]
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

    x, y, w, h = candidate["x"], candidate["y"], candidate["width"], candidate["height"]

    # Horizontal Sobel for left/right edges, vertical Sobel for top/bottom
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)

    def _best_vert_edge(col_center: int, row_start: int, row_end: int) -> int:
        c0 = max(0, col_center - search_range)
        c1 = min(frame_w, col_center + search_range)
        r0 = max(0, row_start)
        r1 = min(frame_h, row_end)
        if c1 <= c0 or r1 <= r0:
            return col_center
        strip = np.abs(sobel_x[r0:r1, c0:c1])
        col_energy = strip.mean(axis=0)
        if col_energy.max() < 5.0:
            return col_center
        return int(c0 + np.argmax(col_energy))

    def _best_horiz_edge(row_center: int, col_start: int, col_end: int) -> int:
        r0 = max(0, row_center - search_range)
        r1 = min(frame_h, row_center + search_range)
        c0 = max(0, col_start)
        c1 = min(frame_w, col_end)
        if r1 <= r0 or c1 <= c0:
            return row_center
        strip = np.abs(sobel_y[r0:r1, c0:c1])
        row_energy = strip.mean(axis=1)
        if row_energy.max() < 5.0:
            return row_center
        return int(r0 + np.argmax(row_energy))

    # Only refine edges that aren't at the frame border
    new_x = x if x <= 5 else _best_vert_edge(x, y, y + h)
    new_right = (x + w) if (x + w) >= frame_w - 5 else _best_vert_edge(x + w, y, y + h)
    new_y = y if y <= 5 else _best_horiz_edge(y, x, x + w)
    new_bottom = (y + h) if (y + h) >= frame_h - 5 else _best_horiz_edge(y + h, x, x + w)

    new_w = max(30, new_right - new_x)
    new_h = max(30, new_bottom - new_y)

    result = dict(candidate)
    result.update({
        "x": max(0, new_x), "y": max(0, new_y),
        "width": min(frame_w - max(0, new_x), new_w),
        "height": min(frame_h - max(0, new_y), new_h),
        "source": "texture_rect+sobel",
    })
    return result


def detect_texture_candidates(image_bgr: np.ndarray) -> list[dict]:
    """Full texture-based detection pipeline.

    Returns a list of webcam overlay candidates found by texture analysis.
    """
    frame_h, frame_w = image_bgr.shape[:2]
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    tex_map = compute_texture_map(blurred)
    raw_rects = find_texture_rectangles(tex_map, frame_w, frame_h)

    candidates = []
    for rect in raw_rects:
        candidates.append(rect)
        refined = refine_with_sobel(rect, image_bgr)
        if refined["width"] != rect["width"] or refined["height"] != rect["height"]:
            candidates.append(refined)

    log.debug("Texture detection: %d candidates from %d raw rects", len(candidates), len(raw_rects))
    return candidates


def compute_static_region_mask(frames: list[np.ndarray], threshold: float = 25.0) -> np.ndarray:
    """Compute per-pixel temporal variance across frames.

    Returns a binary mask where 1 = static region (low variance across frames).
    Webcam backgrounds are static while gameplay changes.
    """
    if len(frames) < 3:
        return np.zeros(frames[0].shape[:2], dtype=np.uint8)

    # Convert to grayscale and compute per-pixel std across frames
    grays = [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY).astype(np.float32) for f in frames]
    stack = np.stack(grays, axis=0)
    pixel_std = stack.std(axis=0)

    # Low std = static (webcam bg), high std = dynamic (gameplay)
    static_mask = (pixel_std < threshold).astype(np.uint8) * 255

    # Cleanup
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    static_mask = cv2.morphologyEx(static_mask, cv2.MORPH_CLOSE, kernel, iterations=3)
    static_mask = cv2.morphologyEx(static_mask, cv2.MORPH_OPEN, kernel, iterations=2)

    return static_mask


def find_static_rectangles(
    static_mask: np.ndarray,
    frame_w: int, frame_h: int,
) -> list[dict]:
    """Find rectangular static regions from the temporal variance mask."""
    contours, _ = cv2.findContours(static_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    frame_area = frame_w * frame_h
    candidates = []

    for cnt in contours:
        rx, ry, rw, rh = cv2.boundingRect(cnt)
        area_ratio = (rw * rh) / frame_area
        ar = rw / rh if rh > 0 else 0

        if area_ratio < WEBCAM_MIN_AREA_PCT or area_ratio > WEBCAM_MAX_AREA_PCT:
            continue
        if ar < WEBCAM_MIN_AR or ar > WEBCAM_MAX_AR:
            continue

        cnt_area = cv2.contourArea(cnt)
        rect_area = rw * rh
        rectangularity = cnt_area / rect_area if rect_area > 0 else 0
        if rectangularity < 0.5:
            continue

        candidates.append({
            "x": rx, "y": ry, "width": rw, "height": rh,
            "source": "static_region",
            "rectangularity": round(rectangularity, 3),
            "area_ratio": round(area_ratio, 4),
        })

    candidates.sort(key=lambda c: c["width"] * c["height"], reverse=True)
    return candidates[:3]
