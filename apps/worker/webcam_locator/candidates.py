"""
candidates.py – Generate webcam bbox candidates from faces, boundaries, and geometry.

Three candidate sources:
1. Face-based: multi-scale expansion + frame-edge snapping
2. Boundary-based: rectangles from edge-projection boundary lines
3. Face-boundary matching: faces matched to boundary rectangles
"""

import logging

import cv2
import numpy as np

from webcam_locator.faces import detect_faces
from webcam_locator.boundaries import detect_boundaries

log = logging.getLogger(__name__)

EXPANSION_SCALES = [4.0, 6.0, 8.0]

# Data-driven priors from labeled dataset:
#   side_box:       area ~3-7% of frame, AR ~1.35-1.78
#   corner_overlay: area ~5-7% of frame, AR ~1.17-1.83
#   full_cam:       area ~44-100% of frame
TYPICAL_WEBCAM_AREA_MIN = 0.02
TYPICAL_WEBCAM_AREA_MAX = 0.15
TYPICAL_WEBCAM_AR_MIN = 1.0
TYPICAL_WEBCAM_AR_MAX = 2.0


def _expand_around_face(
    face: dict,
    frame_w: int, frame_h: int,
    scale: float,
) -> dict:
    """Expand a face bbox by *scale*, clamped to frame."""
    cx = face["x"] + face["width"] // 2
    cy = face["y"] + face["height"] // 2
    new_w = int(face["width"] * scale)
    new_h = int(face["height"] * scale)

    x = max(0, cx - new_w // 2)
    y = max(0, cy - new_h // 2)
    w = min(frame_w - x, new_w)
    h = min(frame_h - y, new_h)

    return {"x": x, "y": y, "width": w, "height": h, "source": "face_expand"}


def _snap_to_frame_edges(
    candidate: dict,
    frame_w: int, frame_h: int,
    snap_threshold_pct: float = 0.08,
) -> dict:
    """If a candidate bbox is near a frame edge, snap it to that edge.

    This is critical for side_box and corner_overlay detection.
    """
    x, y, w, h = candidate["x"], candidate["y"], candidate["width"], candidate["height"]
    tol_x = int(frame_w * snap_threshold_pct)
    tol_y = int(frame_h * snap_threshold_pct)

    snapped = False

    # Snap to left edge
    if x <= tol_x:
        w += x
        x = 0
        snapped = True

    # Snap to top edge
    if y <= tol_y:
        h += y
        y = 0
        snapped = True

    # Snap to right edge
    if (x + w) >= (frame_w - tol_x):
        w = frame_w - x
        snapped = True

    # Snap to bottom edge
    if (y + h) >= (frame_h - tol_y):
        h = frame_h - y
        snapped = True

    result = dict(candidate)
    result.update({"x": x, "y": y, "width": w, "height": h})
    if snapped:
        result["edge_snapped"] = True
    return result


def _match_face_to_boundary_rects(
    face: dict,
    boundary_rects: list[dict],
) -> list[dict]:
    """Find boundary rectangles that contain (or nearly contain) a face."""
    fx = face["x"] + face["width"] // 2
    fy = face["y"] + face["height"] // 2

    matched = []
    for rect in boundary_rects:
        rx, ry, rw, rh = rect["x"], rect["y"], rect["width"], rect["height"]

        # Face center must be inside (or very close to inside) the rectangle
        margin_x = rw * 0.15
        margin_y = rh * 0.15
        if (rx - margin_x <= fx <= rx + rw + margin_x and
                ry - margin_y <= fy <= ry + rh + margin_y):
            # Check face isn't too large for the rectangle
            face_area = face["width"] * face["height"]
            rect_area = rw * rh
            if rect_area > 0 and face_area / rect_area < 0.5:
                cand = dict(rect)
                cand["source"] = "face_boundary_match"
                cand["matched_face"] = face
                matched.append(cand)

    return matched


def _refine_with_edge_projection(
    candidate: dict,
    edge_map: np.ndarray,
    frame_w: int, frame_h: int,
    search_range: int = 50,
) -> dict:
    """Refine candidate edges using local edge projection within a search band."""
    x, y, w, h = candidate["x"], candidate["y"], candidate["width"], candidate["height"]

    def _best_vertical_line(col_center: int, row_start: int, row_end: int) -> int:
        """Find strongest vertical edge near col_center."""
        c0 = max(0, col_center - search_range)
        c1 = min(frame_w, col_center + search_range)
        r0 = max(0, row_start)
        r1 = min(frame_h, row_end)
        if c1 <= c0 or r1 <= r0:
            return col_center
        strip = edge_map[r0:r1, c0:c1]
        col_sums = strip.sum(axis=0).astype(float)
        if col_sums.max() < strip.shape[0] * 255 * 0.08:
            return col_center
        return int(c0 + np.argmax(col_sums))

    def _best_horizontal_line(row_center: int, col_start: int, col_end: int) -> int:
        """Find strongest horizontal edge near row_center."""
        r0 = max(0, row_center - search_range)
        r1 = min(frame_h, row_center + search_range)
        c0 = max(0, col_start)
        c1 = min(frame_w, col_end)
        if r1 <= r0 or c1 <= c0:
            return row_center
        strip = edge_map[r0:r1, c0:c1]
        row_sums = strip.sum(axis=1).astype(float)
        if row_sums.max() < strip.shape[1] * 255 * 0.08:
            return row_center
        return int(r0 + np.argmax(row_sums))

    # Don't refine edges that are at the frame border (they're already correct)
    new_x = x if x <= 5 else _best_vertical_line(x, y, y + h)
    new_right = (x + w) if (x + w) >= frame_w - 5 else _best_vertical_line(x + w, y, y + h)
    new_y = y if y <= 5 else _best_horizontal_line(y, x, x + w)
    new_bottom = (y + h) if (y + h) >= frame_h - 5 else _best_horizontal_line(y + h, x, x + w)

    new_w = max(30, new_right - new_x)
    new_h = max(30, new_bottom - new_y)

    result = dict(candidate)
    result.update({
        "x": max(0, new_x), "y": max(0, new_y),
        "width": min(frame_w - max(0, new_x), new_w),
        "height": min(frame_h - max(0, new_y), new_h),
        "source": candidate.get("source", "") + "+refined",
    })
    return result


def generate_candidates(
    image_bgr: np.ndarray,
) -> tuple[list[dict], list[dict], np.ndarray, dict]:
    """Generate webcam bbox candidates from the input image.

    Returns:
        (candidates, faces, edge_map, boundary_info)
    """
    frame_h, frame_w = image_bgr.shape[:2]

    # Step 1: Detect faces
    faces = detect_faces(image_bgr)
    log.debug("Detected %d faces", len(faces))

    # Step 2: Detect boundaries
    boundary_info = detect_boundaries(image_bgr)
    edge_map = boundary_info["edge_map"]
    boundary_rects = boundary_info["boundary_rects"]

    candidates: list[dict] = []

    # Step 3: Face-based candidates
    for face in faces:
        for scale in EXPANSION_SCALES:
            cand = _expand_around_face(face, frame_w, frame_h, scale)

            # Frame-edge snapped version
            snapped = _snap_to_frame_edges(cand, frame_w, frame_h)
            candidates.append(snapped)

            # Edge-projection refined version
            refined = _refine_with_edge_projection(snapped, edge_map, frame_w, frame_h)
            candidates.append(refined)

        # Face-boundary matched candidates
        matched = _match_face_to_boundary_rects(face, boundary_rects)
        for m in matched:
            candidates.append(m)
            snapped_m = _snap_to_frame_edges(m, frame_w, frame_h)
            candidates.append(snapped_m)

    # Step 4: Boundary rectangles that have plausible webcam dimensions
    for rect in boundary_rects:
        area_ratio = rect.get("area_ratio", 0)
        ar = rect["width"] / rect["height"] if rect["height"] > 0 else 0

        if (TYPICAL_WEBCAM_AREA_MIN <= area_ratio <= TYPICAL_WEBCAM_AREA_MAX and
                TYPICAL_WEBCAM_AR_MIN <= ar <= TYPICAL_WEBCAM_AR_MAX):
            candidates.append(rect)

    # Step 5: Full-frame candidate (for full_cam detection)
    candidates.append({
        "x": 0, "y": 0, "width": frame_w, "height": frame_h,
        "source": "full_frame",
    })

    log.debug("Generated %d total candidates", len(candidates))
    return candidates, faces, edge_map, boundary_info
