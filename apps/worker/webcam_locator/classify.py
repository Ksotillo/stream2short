"""
classify.py – Deterministic webcam type classification based on bbox position.

Rules (strict, data-driven):
- full_cam: covers >70% of frame area
- top_band / bottom_band: near top/bottom edge AND wide (>60% of frame)
- corner_overlay: bbox PRECISELY touches two adjacent edges (within 2% = strict)
- side_box: near any edge (within 6% = generous) but not a true corner
- center_box: not near any edge
"""

# Strict threshold for corner_overlay: bbox must actually touch two edges
CORNER_EDGE_PCT = 0.025

# Generous threshold for side_box: bbox just needs to be near an edge
SIDE_EDGE_PCT = 0.06

BAND_WIDTH_RATIO = 0.60
FULL_CAM_AREA_RATIO = 0.70


def classify_bbox(
    x: int, y: int, w: int, h: int,
    frame_w: int, frame_h: int,
) -> tuple[str, str]:
    """Classify a webcam bbox into (type, corner)."""
    area_ratio = (w * h) / (frame_w * frame_h) if (frame_w * frame_h) > 0 else 0
    width_ratio = w / frame_w if frame_w > 0 else 0

    # Full cam
    if area_ratio >= FULL_CAM_AREA_RATIO:
        return "full_cam", "unknown"

    # Band: near horizontal edge AND wide
    strict_tol_y = int(frame_h * CORNER_EDGE_PCT)
    if y <= strict_tol_y and width_ratio > BAND_WIDTH_RATIO:
        return "top_band", "unknown"
    if (y + h) >= (frame_h - strict_tol_y) and width_ratio > BAND_WIDTH_RATIO:
        return "bottom_band", "unknown"

    # Corner overlay: STRICT edge-touch check (2.5%)
    # The bbox must truly be pressed against two adjacent edges
    corner_tol_x = int(frame_w * CORNER_EDGE_PCT)
    corner_tol_y = int(frame_h * CORNER_EDGE_PCT)

    touch_left = x <= corner_tol_x
    touch_right = (x + w) >= (frame_w - corner_tol_x)
    touch_top = y <= corner_tol_y
    touch_bottom = (y + h) >= (frame_h - corner_tol_y)

    if touch_top and touch_left:
        return "corner_overlay", "top-left"
    if touch_top and touch_right:
        return "corner_overlay", "top-right"
    if touch_bottom and touch_left:
        return "corner_overlay", "bottom-left"
    if touch_bottom and touch_right:
        return "corner_overlay", "bottom-right"

    # Side box: generous edge check (6%)
    side_tol_x = int(frame_w * SIDE_EDGE_PCT)
    side_tol_y = int(frame_h * SIDE_EDGE_PCT)

    near_left = x <= side_tol_x
    near_right = (x + w) >= (frame_w - side_tol_x)
    near_top = y <= side_tol_y
    near_bottom = (y + h) >= (frame_h - side_tol_y)

    if near_left or near_right or near_top or near_bottom:
        return "side_box", "unknown"

    return "center_box", "unknown"


def is_true_corner(
    x: int, y: int, w: int, h: int,
    frame_w: int, frame_h: int,
) -> bool:
    """Check if a bbox truly touches two adjacent edges (strict 2.5% threshold)."""
    tol_x = int(frame_w * CORNER_EDGE_PCT)
    tol_y = int(frame_h * CORNER_EDGE_PCT)

    touch_left = x <= tol_x
    touch_right = (x + w) >= (frame_w - tol_x)
    touch_top = y <= tol_y
    touch_bottom = (y + h) >= (frame_h - tol_y)

    return (touch_top and touch_left) or (touch_top and touch_right) or \
           (touch_bottom and touch_left) or (touch_bottom and touch_right)
