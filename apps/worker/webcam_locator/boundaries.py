"""
boundaries.py – Find webcam overlay boundaries using edge projection analysis.

Strategy:
1. Compute Canny edge map
2. Project edges onto rows (horizontal lines) and columns (vertical lines)
3. Find peaks = strong boundary lines
4. Generate candidate rectangles from line combinations
5. Also detect if the frame lacks internal boundaries (→ full_cam hint)
"""

import logging
from typing import Optional

import cv2
import numpy as np

log = logging.getLogger(__name__)

# Minimum edge density for a line to be considered a boundary
MIN_PEAK_RATIO = 0.15
# How many pixels to skip from frame edges when looking for internal boundaries
EDGE_MARGIN = 10


def compute_edge_projections(
    edge_map: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute horizontal (row) and vertical (column) edge projections.

    Returns (row_projection, col_projection) normalized to [0, 1].
    """
    h, w = edge_map.shape[:2]
    row_proj = edge_map.astype(np.float64).sum(axis=1) / (w * 255.0)
    col_proj = edge_map.astype(np.float64).sum(axis=0) / (h * 255.0)
    return row_proj, col_proj


def find_boundary_lines(
    projection: np.ndarray,
    min_ratio: float = MIN_PEAK_RATIO,
    min_distance: int = 30,
    margin: int = EDGE_MARGIN,
) -> list[int]:
    """Find strong boundary lines from an edge projection.

    Returns sorted list of positions (row or column indices).
    """
    if len(projection) < 2 * margin:
        return []

    # Only look at interior region (skip frame edges themselves)
    interior = projection[margin:-margin].copy()
    if len(interior) == 0:
        return []

    # Find positions above threshold
    threshold = min_ratio
    peaks_mask = interior > threshold

    # Group consecutive above-threshold positions into clusters
    lines = []
    in_peak = False
    peak_start = 0
    peak_max_val = 0
    peak_max_pos = 0

    for i, above in enumerate(peaks_mask):
        if above:
            if not in_peak:
                in_peak = True
                peak_start = i
                peak_max_val = interior[i]
                peak_max_pos = i
            elif interior[i] > peak_max_val:
                peak_max_val = interior[i]
                peak_max_pos = i
        else:
            if in_peak:
                lines.append(peak_max_pos + margin)
                in_peak = False

    if in_peak:
        lines.append(peak_max_pos + margin)

    # Filter lines that are too close together (keep strongest)
    if len(lines) <= 1:
        return lines

    filtered = [lines[0]]
    for line in lines[1:]:
        if line - filtered[-1] >= min_distance:
            filtered.append(line)
        else:
            # Keep the one with higher projection value
            if projection[line] > projection[filtered[-1]]:
                filtered[-1] = line

    return filtered


def generate_boundary_rectangles(
    h_lines: list[int],
    v_lines: list[int],
    frame_w: int,
    frame_h: int,
    min_area_ratio: float = 0.01,
    max_area_ratio: float = 0.70,
) -> list[dict]:
    """Generate candidate rectangles from boundary line combinations.

    Includes rectangles formed by:
    - Two h_lines + two v_lines (interior rectangle)
    - One h_line + frame edge + two v_lines (edge-anchored)
    - Etc.
    """
    # Add frame edges as potential boundaries
    all_h = sorted(set([0] + h_lines + [frame_h]))
    all_v = sorted(set([0] + v_lines + [frame_w]))

    frame_area = frame_w * frame_h
    candidates = []

    # Cap line counts to avoid combinatorial explosion
    all_h = all_h[:12]
    all_v = all_v[:12]

    for i, top in enumerate(all_h):
        for j, bottom in enumerate(all_h):
            if bottom <= top + 30:
                continue
            for k, left in enumerate(all_v):
                for m, right in enumerate(all_v):
                    if right <= left + 30:
                        continue

                    w = right - left
                    h = bottom - top
                    area_ratio = (w * h) / frame_area

                    if area_ratio < min_area_ratio or area_ratio > max_area_ratio:
                        continue

                    ar = w / h if h > 0 else 0
                    if ar < 0.5 or ar > 3.0:
                        continue

                    candidates.append({
                        "x": left, "y": top, "width": w, "height": h,
                        "source": "boundary_rect",
                        "area_ratio": round(area_ratio, 4),
                    })

                    if len(candidates) >= 200:
                        return candidates

    return candidates


def has_strong_internal_boundaries(
    row_proj: np.ndarray,
    col_proj: np.ndarray,
    threshold: float = 0.12,
    margin_pct: float = 0.10,
) -> bool:
    """Check if the frame has strong internal boundary lines.

    If False, the frame likely has no webcam overlay (could be full_cam).
    """
    h = len(row_proj)
    w = len(col_proj)
    h_margin = int(h * margin_pct)
    w_margin = int(w * margin_pct)

    # Check interior rows and columns for strong boundaries
    if h_margin < h - h_margin:
        interior_h = row_proj[h_margin:h - h_margin]
        if np.any(interior_h > threshold):
            return True

    if w_margin < w - w_margin:
        interior_v = col_proj[w_margin:w - w_margin]
        if np.any(interior_v > threshold):
            return True

    return False


def detect_boundaries(
    image_bgr: np.ndarray,
) -> dict:
    """Full boundary detection pipeline.

    Returns dict with:
        edge_map, row_proj, col_proj, h_lines, v_lines,
        boundary_rects, has_internal_boundaries
    """
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    edge_map = cv2.Canny(blurred, 30, 100)

    frame_h, frame_w = image_bgr.shape[:2]

    row_proj, col_proj = compute_edge_projections(edge_map)
    h_lines = find_boundary_lines(row_proj, min_distance=int(frame_h * 0.05))
    v_lines = find_boundary_lines(col_proj, min_distance=int(frame_w * 0.05))

    boundary_rects = generate_boundary_rectangles(
        h_lines, v_lines, frame_w, frame_h,
    )

    has_bounds = has_strong_internal_boundaries(row_proj, col_proj)

    log.debug(
        "Boundaries: %d h-lines, %d v-lines, %d rects, internal=%s",
        len(h_lines), len(v_lines), len(boundary_rects), has_bounds,
    )

    return {
        "edge_map": edge_map,
        "row_proj": row_proj,
        "col_proj": col_proj,
        "h_lines": h_lines,
        "v_lines": v_lines,
        "boundary_rects": boundary_rects,
        "has_internal_boundaries": has_bounds,
    }
