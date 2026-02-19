"""
refine.py – Shrink an oversized bbox inward to find the actual webcam overlay border.

Strategy: for each side of the bbox, scan inward looking for a strong edge
that indicates the boundary between webcam content and gameplay/background.
Uses edge projection profiles within narrow scan bands.
"""

import logging

import cv2
import numpy as np

log = logging.getLogger(__name__)


def _edge_profile_horizontal(edge_map: np.ndarray, y: int, x_start: int, x_end: int, band: int = 3) -> float:
    """Mean edge density along a horizontal line at row y, between x_start and x_end."""
    h, w = edge_map.shape[:2]
    y0 = max(0, y - band)
    y1 = min(h, y + band + 1)
    x0 = max(0, x_start)
    x1 = min(w, x_end)
    if y1 <= y0 or x1 <= x0:
        return 0.0
    return float(edge_map[y0:y1, x0:x1].mean()) / 255.0


def _edge_profile_vertical(edge_map: np.ndarray, x: int, y_start: int, y_end: int, band: int = 3) -> float:
    """Mean edge density along a vertical line at column x, between y_start and y_end."""
    h, w = edge_map.shape[:2]
    x0 = max(0, x - band)
    x1 = min(w, x + band + 1)
    y0 = max(0, y_start)
    y1 = min(h, y_end)
    if x1 <= x0 or y1 <= y0:
        return 0.0
    return float(edge_map[y0:y1, x0:x1].mean()) / 255.0


def refine_bbox_inward(
    bbox: dict,
    image_bgr: np.ndarray,
    max_shrink_pct: float = 0.45,
    edge_threshold: float = 0.08,
) -> dict:
    """Shrink a bbox inward to find the actual webcam overlay border.

    For each side, scans from the current edge inward, looking for a row/column
    with strong edge density (indicating the webcam border). Stops at the first
    strong edge found, or at max_shrink_pct of the bbox dimension.

    Args:
        bbox: dict with x, y, width, height
        image_bgr: the full frame
        max_shrink_pct: maximum fraction of width/height to shrink from each side
        edge_threshold: minimum edge density to consider as a border

    Returns:
        Refined bbox dict (same keys, possibly modified coordinates)
    """
    frame_h, frame_w = image_bgr.shape[:2]
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    edges = cv2.Canny(blurred, 40, 120)

    x, y, w, h = bbox["x"], bbox["y"], bbox["width"], bbox["height"]
    max_dx = int(w * max_shrink_pct)
    max_dy = int(h * max_shrink_pct)

    new_x, new_y, new_r, new_b = x, y, x + w, y + h

    # Scan LEFT edge inward (from x, moving right)
    if x > 3:
        best_col = x
        best_density = 0.0
        for dx in range(0, max_dx, 2):
            col = x + dx
            density = _edge_profile_vertical(edges, col, y, y + h)
            if density > best_density:
                best_density = density
                best_col = col
            if density >= edge_threshold and dx > 5:
                break
        if best_density >= edge_threshold:
            new_x = best_col

    # Scan RIGHT edge inward (from x+w, moving left)
    right = x + w
    if right < frame_w - 3:
        best_col = right
        best_density = 0.0
        for dx in range(0, max_dx, 2):
            col = right - dx
            density = _edge_profile_vertical(edges, col, y, y + h)
            if density > best_density:
                best_density = density
                best_col = col
            if density >= edge_threshold and dx > 5:
                break
        if best_density >= edge_threshold:
            new_r = best_col

    # Scan TOP edge inward (from y, moving down)
    if y > 3:
        best_row = y
        best_density = 0.0
        for dy in range(0, max_dy, 2):
            row = y + dy
            density = _edge_profile_horizontal(edges, row, x, x + w)
            if density > best_density:
                best_density = density
                best_row = row
            if density >= edge_threshold and dy > 5:
                break
        if best_density >= edge_threshold:
            new_y = best_row

    # Scan BOTTOM edge inward (from y+h, moving up)
    bottom = y + h
    if bottom < frame_h - 3:
        best_row = bottom
        best_density = 0.0
        for dy in range(0, max_dy, 2):
            row = bottom - dy
            density = _edge_profile_horizontal(edges, row, x, x + w)
            if density > best_density:
                best_density = density
                best_row = row
            if density >= edge_threshold and dy > 5:
                break
        if best_density >= edge_threshold:
            new_b = best_row

    new_w = max(30, new_r - new_x)
    new_h = max(30, new_b - new_y)

    # Only accept refinement if we actually shrunk (don't expand)
    if new_w >= w and new_h >= h:
        return dict(bbox)

    result = dict(bbox)
    result.update({"x": new_x, "y": new_y, "width": new_w, "height": new_h})
    return result


def refine_two_pass(
    bbox: dict,
    image_bgr: np.ndarray,
    margin_pct: float = 0.3,
) -> dict:
    """Two-pass refinement: crop the Gemini region with margin, then find
    precise webcam borders using high-res edge detection on the crop.

    This works because Gemini is good at rough localization but bad at
    precise boundaries. Edge detection on a small crop is much more
    accurate than on the full frame.
    """
    frame_h, frame_w = image_bgr.shape[:2]
    x, y, w, h = bbox["x"], bbox["y"], bbox["width"], bbox["height"]

    # Expand the crop region with margin for context
    margin_x = int(w * margin_pct)
    margin_y = int(h * margin_pct)
    crop_x = max(0, x - margin_x)
    crop_y = max(0, y - margin_y)
    crop_r = min(frame_w, x + w + margin_x)
    crop_b = min(frame_h, y + h + margin_y)

    crop = image_bgr[crop_y:crop_b, crop_x:crop_r]
    if crop.size < 100:
        return dict(bbox)

    crop_h, crop_w = crop.shape[:2]

    # High-res edge detection on the crop
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    edges = cv2.Canny(blurred, 30, 100)

    # Find strong contours in the crop
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Look for the best rectangular contour that could be the webcam overlay
    best_rect = None
    best_score = 0

    min_area = crop_w * crop_h * 0.08
    max_area = crop_w * crop_h * 0.95

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area or area > max_area:
            continue

        epsilon = 0.02 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)

        if len(approx) < 4 or len(approx) > 8:
            continue

        rx, ry, rw, rh = cv2.boundingRect(approx)
        ar = rw / rh if rh > 0 else 0
        if ar < 0.8 or ar > 2.5:
            continue

        # Score by rectangularity (how well the contour fills its bounding rect)
        rect_area = rw * rh
        if rect_area == 0:
            continue
        rectangularity = area / rect_area

        # Prefer contours that are rectangular and large
        score = rectangularity * (area / (crop_w * crop_h))

        if score > best_score:
            best_score = score
            best_rect = (rx + crop_x, ry + crop_y, rw, rh)

    if best_rect is not None and best_score > 0.05:
        result = dict(bbox)
        result.update({
            "x": best_rect[0], "y": best_rect[1],
            "width": best_rect[2], "height": best_rect[3],
        })
        return result

    # Fallback: use the standard inward refinement on the crop
    # Convert bbox coords to crop-relative, refine, convert back
    local_bbox = {
        "x": x - crop_x, "y": y - crop_y,
        "width": w, "height": h,
    }
    refined_local = refine_bbox_inward(local_bbox, crop, max_shrink_pct=0.40)

    result = dict(bbox)
    result.update({
        "x": refined_local["x"] + crop_x,
        "y": refined_local["y"] + crop_y,
        "width": refined_local["width"],
        "height": refined_local["height"],
    })
    return result
