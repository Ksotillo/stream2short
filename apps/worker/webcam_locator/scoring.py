"""
scoring.py – Score webcam bbox candidates.

Score = weighted sum of:
  - Face inclusion (face inside with reasonable headroom)
  - Border strength (edges aligned with candidate boundaries)
  - Interior/exterior difference (webcam looks different from gameplay)
  - Gameplay bleed penalty (high-freq texture = gameplay leaking in)
  - Size prior (data-driven: typical webcam occupies 3-10% of frame)
  - Aspect ratio prior (data-driven: typical webcam AR ~1.3-1.8)
  - Full-frame bonus (only for full_frame candidates when no strong boundaries)
"""

import logging

import cv2
import numpy as np

log = logging.getLogger(__name__)

# Tuned weights
W_FACE = 4.0
W_BORDER = 2.5
W_INTERIOR_DIFF = 3.0
W_BLEED = -2.0
W_SIZE = 2.0
W_ASPECT = 1.5
W_FULLFRAME = 5.0


def _face_inclusion_score(candidate: dict, faces: list[dict]) -> float:
    """Score how well the candidate contains a face."""
    if not faces:
        return 0.0

    cx, cy, cw, ch = candidate["x"], candidate["y"], candidate["width"], candidate["height"]
    best = 0.0

    for face in faces:
        fx = face["x"] + face["width"] // 2
        fy = face["y"] + face["height"] // 2

        if not (cx <= fx <= cx + cw and cy <= fy <= cy + ch):
            continue

        face_area = face["width"] * face["height"]
        cand_area = cw * ch
        if cand_area == 0:
            continue
        face_ratio = face_area / cand_area

        # Sweet spot: face is 2-25% of candidate area
        if 0.02 <= face_ratio <= 0.25:
            score = 1.0
        elif face_ratio < 0.02:
            score = 0.4  # face too small relative to candidate
        elif face_ratio <= 0.5:
            score = 0.6
        else:
            score = 0.2  # face too large

        # Headroom bonus
        headroom = (fy - cy) / ch if ch > 0 else 0
        if 0.15 <= headroom <= 0.55:
            score *= 1.0
        else:
            score *= 0.7

        best = max(best, score * face.get("confidence", 0.5))

    return best


def _border_strength_score(candidate: dict, edge_map: np.ndarray) -> float:
    """Score how strongly edges align with candidate boundaries.

    Only scores non-frame-edge borders (frame edges always have strong edges).
    """
    x, y, w, h = candidate["x"], candidate["y"], candidate["width"], candidate["height"]
    fh, fw = edge_map.shape[:2]
    band = 4

    scores = []

    # Top border (skip if at frame edge)
    if y > 5:
        row = edge_map[max(0, y - band):min(fh, y + band), max(0, x):min(fw, x + w)]
        if row.size > 0:
            scores.append(row.mean() / 255.0)

    # Bottom border
    by = y + h
    if by < fh - 5:
        row = edge_map[max(0, by - band):min(fh, by + band), max(0, x):min(fw, x + w)]
        if row.size > 0:
            scores.append(row.mean() / 255.0)

    # Left border
    if x > 5:
        col = edge_map[max(0, y):min(fh, y + h), max(0, x - band):min(fw, x + band)]
        if col.size > 0:
            scores.append(col.mean() / 255.0)

    # Right border
    rx = x + w
    if rx < fw - 5:
        col = edge_map[max(0, y):min(fh, y + h), max(0, rx - band):min(fw, rx + band)]
        if col.size > 0:
            scores.append(col.mean() / 255.0)

    return float(np.mean(scores)) if scores else 0.0


def _interior_exterior_difference(candidate: dict, gray: np.ndarray) -> float:
    """Score how visually different the candidate interior is from the exterior.

    A real webcam overlay should have clearly different texture/brightness
    from the surrounding gameplay.
    """
    x, y, w, h = candidate["x"], candidate["y"], candidate["width"], candidate["height"]
    fh, fw = gray.shape[:2]

    interior = gray[max(0, y):min(fh, y + h), max(0, x):min(fw, x + w)]
    if interior.size < 100:
        return 0.0

    # Build exterior mask (frame minus candidate)
    mask = np.ones((fh, fw), dtype=bool)
    mask[max(0, y):min(fh, y + h), max(0, x):min(fw, x + w)] = False
    exterior = gray[mask]
    if exterior.size < 100:
        return 0.0

    # Compare texture: Laplacian variance
    int_lap = cv2.Laplacian(interior, cv2.CV_64F).var()
    ext_lap_roi = gray.copy()
    ext_lap_roi[max(0, y):min(fh, y + h), max(0, x):min(fw, x + w)] = 0
    # Approximate exterior laplacian as overall minus interior contribution
    full_lap = cv2.Laplacian(gray, cv2.CV_64F).var()
    texture_diff = abs(int_lap - full_lap) / max(full_lap, 1.0)

    # Compare brightness
    bright_diff = abs(float(interior.mean()) - float(exterior.mean())) / 255.0

    # Compare edge density
    int_edges = cv2.Canny(interior, 50, 150)
    int_edge_density = int_edges.mean() / 255.0

    # Score: higher difference = more likely a real overlay
    score = 0.3 * min(1.0, texture_diff * 2.0) + \
            0.3 * min(1.0, bright_diff * 3.0) + \
            0.4 * max(0.0, 1.0 - int_edge_density * 5.0)  # lower internal edges = more webcam-like

    return max(0.0, min(1.0, score))


def _gameplay_bleed_penalty(candidate: dict, gray: np.ndarray) -> float:
    """Penalize candidates with high-frequency texture (gameplay bleed)."""
    x, y, w, h = candidate["x"], candidate["y"], candidate["width"], candidate["height"]
    fh, fw = gray.shape[:2]

    roi = gray[max(0, y):min(fh, y + h), max(0, x):min(fw, x + w)]
    if roi.size < 100:
        return 0.0

    edges = cv2.Canny(roi, 50, 150)
    edge_density = edges.mean() / 255.0

    if edge_density > 0.12:
        return min(1.0, edge_density * 4.0)
    return 0.0


def _size_prior_score(candidate: dict, frame_w: int, frame_h: int) -> float:
    """Data-driven size prior from labeled dataset.

    Typical webcam overlays: 3-10% of frame area.
    Full-frame candidates handled separately.
    """
    area_ratio = (candidate["width"] * candidate["height"]) / (frame_w * frame_h)

    if candidate.get("source") == "full_frame":
        return 0.0  # Scored separately

    # Sweet spot: 3-10% of frame
    if 0.03 <= area_ratio <= 0.10:
        return 1.0
    elif 0.02 <= area_ratio <= 0.15:
        return 0.6
    elif 0.01 <= area_ratio <= 0.25:
        return 0.3
    else:
        return 0.05


def _aspect_ratio_score(candidate: dict) -> float:
    """Data-driven AR prior. Typical webcam: AR 1.3-1.8."""
    w = candidate["width"]
    h = candidate["height"]
    if h == 0:
        return 0.0
    ar = w / h

    if 1.3 <= ar <= 1.85:
        return 1.0
    elif 1.0 <= ar <= 2.0:
        return 0.6
    elif 0.7 <= ar <= 2.5:
        return 0.3
    return 0.1


def _full_frame_score(
    candidate: dict,
    faces: list[dict],
    has_internal_boundaries: bool,
    gray: np.ndarray,
) -> float:
    """Special scoring for the full-frame candidate (full_cam detection).

    High score if: faces exist, no strong internal boundaries, and the
    frame has characteristics of a pure webcam (low gameplay texture).
    """
    if candidate.get("source") != "full_frame":
        return 0.0

    if not faces:
        return 0.0

    score = 0.0

    # No internal boundaries = strong indicator of full_cam
    if not has_internal_boundaries:
        score += 0.4

    # Face is large relative to frame (typical for full_cam)
    frame_area = candidate["width"] * candidate["height"]
    largest_face_area = max((f["width"] * f["height"] for f in faces), default=0)
    face_ratio = largest_face_area / frame_area if frame_area > 0 else 0

    if face_ratio > 0.015:
        score += 0.3
    elif face_ratio > 0.003:
        score += 0.15

    # Low overall edge density = webcam-like frame
    edges = cv2.Canny(gray, 50, 150)
    edge_density = edges.mean() / 255.0
    if edge_density < 0.10:
        score += 0.2
    elif edge_density < 0.15:
        score += 0.1

    # Multiple faces or high-confidence face = more likely full_cam
    if len(faces) >= 1 and max(f.get("confidence", 0) for f in faces) > 0.3:
        score += 0.1

    return score


def score_candidate(
    candidate: dict,
    faces: list[dict],
    edge_map: np.ndarray,
    gray: np.ndarray,
    frame_w: int,
    frame_h: int,
    has_internal_boundaries: bool = True,
) -> float:
    """Compute a composite score for a single candidate bbox."""

    is_full_frame = candidate.get("source") == "full_frame"

    if is_full_frame:
        s_full = _full_frame_score(candidate, faces, has_internal_boundaries, gray)
        total = W_FULLFRAME * s_full
        candidate["_scores"] = {"full_frame": round(s_full, 3), "total": round(total, 3)}
        return total

    s_face = _face_inclusion_score(candidate, faces)
    s_border = _border_strength_score(candidate, edge_map)
    s_diff = _interior_exterior_difference(candidate, gray)
    s_bleed = _gameplay_bleed_penalty(candidate, gray)
    s_size = _size_prior_score(candidate, frame_w, frame_h)
    s_aspect = _aspect_ratio_score(candidate)

    total = (
        W_FACE * s_face
        + W_BORDER * s_border
        + W_INTERIOR_DIFF * s_diff
        + W_BLEED * s_bleed
        + W_SIZE * s_size
        + W_ASPECT * s_aspect
    )

    candidate["_scores"] = {
        "face": round(s_face, 3),
        "border": round(s_border, 3),
        "int_ext_diff": round(s_diff, 3),
        "bleed": round(s_bleed, 3),
        "size": round(s_size, 3),
        "aspect": round(s_aspect, 3),
        "total": round(total, 3),
    }

    return total


def _quick_prefilter_score(c: dict, faces: list[dict], frame_w: int, frame_h: int) -> float:
    """Fast pre-filter: face inclusion + size prior only (no image ops)."""
    s = _face_inclusion_score(c, faces) * W_FACE
    s += _size_prior_score(c, frame_w, frame_h) * W_SIZE
    s += _aspect_ratio_score(c) * W_ASPECT
    return s


def score_candidates(
    candidates: list[dict],
    faces: list[dict],
    edge_map: np.ndarray,
    image_bgr: np.ndarray,
    has_internal_boundaries: bool = True,
    max_full_score: int = 30,
) -> list[dict]:
    """Score all candidates. Pre-filters to top N before expensive scoring."""
    frame_h, frame_w = image_bgr.shape[:2]
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

    # Quick pre-filter to reduce expensive scoring
    if len(candidates) > max_full_score:
        for c in candidates:
            c["_pre_score"] = _quick_prefilter_score(c, faces, frame_w, frame_h)
        candidates.sort(key=lambda c: c.get("_pre_score", 0), reverse=True)
        # Always keep the full_frame candidate
        top = candidates[:max_full_score]
        full_frame = [c for c in candidates[max_full_score:] if c.get("source") == "full_frame"]
        candidates = top + full_frame

    for c in candidates:
        c["score"] = score_candidate(
            c, faces, edge_map, gray, frame_w, frame_h, has_internal_boundaries,
        )

    candidates.sort(key=lambda c: c["score"], reverse=True)
    return candidates
