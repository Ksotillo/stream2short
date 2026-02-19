"""
consensus.py – Multi-frame consensus for webcam detection.

Run detect_webcam_bbox per-frame, then pick the best result based on:
  - confidence
  - stability (IoU consistency across frames)
  - gameplay-bleed penalty
  - reject weird outliers
"""

import logging
from typing import Optional

import numpy as np

from webcam_locator.core import detect_webcam_bbox

log = logging.getLogger(__name__)


def _iou(a: dict, b: dict) -> float:
    """Compute Intersection-over-Union between two bbox dicts."""
    ax1, ay1 = a["x"], a["y"]
    ax2, ay2 = ax1 + a["width"], ay1 + a["height"]
    bx1, by1 = b["x"], b["y"]
    bx2, by2 = bx1 + b["width"], by1 + b["height"]

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    area_a = a["width"] * a["height"]
    area_b = b["width"] * b["height"]
    union = area_a + area_b - inter

    return inter / union if union > 0 else 0.0


def _median_bbox(detections: list[dict]) -> dict:
    """Compute median bbox across multiple detections."""
    xs = [d["x"] for d in detections]
    ys = [d["y"] for d in detections]
    ws = [d["width"] for d in detections]
    hs = [d["height"] for d in detections]

    return {
        "x": int(np.median(xs)),
        "y": int(np.median(ys)),
        "width": int(np.median(ws)),
        "height": int(np.median(hs)),
    }


def detect_from_frames(
    frames: list[np.ndarray],
    *,
    use_gemini: bool = False,
    debug_dir: Optional[str] = None,
    min_agreement: int = 2,
    iou_threshold: float = 0.3,
) -> dict:
    """Run detection on multiple frames and produce a consensus result.

    Args:
        frames: List of BGR images.
        use_gemini: Pass through to per-frame detection.
        debug_dir: If set, save debug artifacts.
        min_agreement: Minimum frames that must agree for a detection.
        iou_threshold: IoU threshold for grouping detections.

    Returns:
        Consensus result dict (same schema as detect_webcam_bbox).
    """
    if not frames:
        return {
            "found": False, "type": "none", "corner": "unknown",
            "x": 0, "y": 0, "width": 0, "height": 0,
            "confidence": 0.0, "reason": "no frames provided",
        }

    # Run per-frame detection.
    # Gemini optimization: only call Gemini on 2 evenly-spaced frames to
    # save API calls and time (~3-6s per call). The rest use CV-only.
    gemini_frames = set()
    if use_gemini and len(frames) > 0:
        gemini_frames.add(0)
        if len(frames) >= 4:
            gemini_frames.add(len(frames) // 2)
        if len(frames) >= 2:
            gemini_frames.add(len(frames) - 1)

    per_frame: list[dict] = []
    for i, frame in enumerate(frames):
        use_gem_this_frame = use_gemini and (i in gemini_frames)
        result = detect_webcam_bbox(
            frame, use_gemini=use_gem_this_frame, debug_dir=debug_dir, frame_idx=i,
        )
        per_frame.append(result)

    # Separate found vs not-found
    found = [r for r in per_frame if r["found"]]
    log.info("Consensus: %d/%d frames detected webcam", len(found), len(frames))

    # ── Full-cam check (multi-signal) ──
    frame_h, frame_w = frames[0].shape[:2]
    frame_area = frame_w * frame_h

    fullcam_count = sum(1 for r in found if r["type"] == "full_cam")
    large_bbox_count = sum(
        1 for r in found
        if r["width"] * r["height"] > frame_area * 0.35
    )

    # CV-only heuristic: if most detected bboxes are very large AND
    # no strong clustering around a small region, it's likely full_cam.
    # This works even when Gemini doesn't say "full_cam".
    if found:
        median_area_ratio = float(np.median([
            r["width"] * r["height"] / frame_area for r in found
        ]))
    else:
        median_area_ratio = 0

    is_fullcam = False
    fullcam_reason = ""

    if found and fullcam_count >= len(found) * 0.3:
        is_fullcam = True
        fullcam_reason = f"{fullcam_count} full_cam frames"
    elif found and large_bbox_count >= len(found) * 0.4:
        is_fullcam = True
        fullcam_reason = f"{large_bbox_count} large bbox frames"
    elif found and median_area_ratio > 0.30:
        is_fullcam = True
        fullcam_reason = f"median bbox area {median_area_ratio:.0%} of frame"

    if is_fullcam:
        avg_conf = float(np.mean([d["confidence"] for d in found])) if found else 0.5
        return {
            "found": True, "type": "full_cam", "corner": "unknown",
            "x": 0, "y": 0, "width": frame_w, "height": frame_h,
            "confidence": round(min(1.0, avg_conf * 1.2), 3),
            "reason": f"full_cam: {fullcam_reason} ({len(found)} detections)",
        }

    if len(found) < min_agreement:
        return {
            "found": False, "type": "none", "corner": "unknown",
            "x": 0, "y": 0, "width": 0, "height": 0,
            "confidence": 0.0,
            "reason": f"only {len(found)}/{len(frames)} frames detected (need {min_agreement})",
        }

    # Group detections by IoU similarity (greedy clustering)
    clusters: list[list[dict]] = []
    used = [False] * len(found)

    for i, det in enumerate(found):
        if used[i]:
            continue
        cluster = [det]
        used[i] = True
        for j in range(i + 1, len(found)):
            if used[j]:
                continue
            if _iou(det, found[j]) >= iou_threshold:
                cluster.append(found[j])
                used[j] = True
        clusters.append(cluster)

    # Pick largest cluster
    clusters.sort(key=len, reverse=True)
    best_cluster = clusters[0]

    if len(best_cluster) < min_agreement:
        return {
            "found": False, "type": "none", "corner": "unknown",
            "x": 0, "y": 0, "width": 0, "height": 0,
            "confidence": 0.0,
            "reason": f"largest cluster has {len(best_cluster)} detections (need {min_agreement})",
        }

    # Compute consensus bbox (median of cluster)
    med = _median_bbox(best_cluster)

    # ── Post-consensus edge snap: snap to edges where per-frame detections
    # agree, but NEVER snap to more than ONE axis (horiz OR vert) to prevent
    # side_box → corner_overlay misclassification.
    snap_tol_x = int(frame_w * 0.08)
    snap_tol_y = int(frame_h * 0.08)

    n = len(best_cluster)
    near_left = sum(1 for d in best_cluster if d["x"] <= snap_tol_x)
    near_top = sum(1 for d in best_cluster if d["y"] <= snap_tol_y)
    near_right = sum(1 for d in best_cluster if (d["x"] + d["width"]) >= frame_w - snap_tol_x)
    near_bottom = sum(1 for d in best_cluster if (d["y"] + d["height"]) >= frame_h - snap_tol_y)

    majority = n * 0.4

    mx, my, mw, mh = med["x"], med["y"], med["width"], med["height"]

    # Determine which axes have strong per-frame agreement
    snap_h = False  # horizontal snap (left or right)
    snap_v = False  # vertical snap (top or bottom)

    if near_left >= majority and mx <= snap_tol_x:
        mw += mx; mx = 0; snap_h = True
    if near_right >= majority and (mx + mw) >= frame_w - snap_tol_x:
        mw = frame_w - mx; snap_h = True
    if near_top >= majority and my <= snap_tol_y:
        mh += my; my = 0; snap_v = True
    if near_bottom >= majority and (my + mh) >= frame_h - snap_tol_y:
        mh = frame_h - my; snap_v = True

    # Safety: if we snapped to both axes (creating a corner), check if
    # per-frame majority actually says corner_overlay. If not, undo the
    # weaker axis snap to keep it as side_box.
    if snap_h and snap_v:
        from collections import Counter as _Counter
        type_votes_pre = _Counter(d["type"] for d in best_cluster)
        if type_votes_pre.get("corner_overlay", 0) < n * 0.5:
            # Majority of frames don't think it's a corner → undo the
            # axis with less agreement
            h_agreement = max(near_left, near_right)
            v_agreement = max(near_top, near_bottom)
            if h_agreement >= v_agreement:
                # Keep horizontal snap, undo vertical
                my = med["y"]
                mh = med["height"]
            else:
                # Keep vertical snap, undo horizontal
                mx = med["x"]
                mw = med["width"]

    med = {"x": mx, "y": my, "width": mw, "height": mh}

    # Consensus type: majority vote
    from collections import Counter
    type_votes = Counter(d["type"] for d in best_cluster)
    consensus_type = type_votes.most_common(1)[0][0]

    corner_votes = Counter(d["corner"] for d in best_cluster)
    consensus_corner = corner_votes.most_common(1)[0][0]

    # Re-classify using deterministic rules on the snapped median bbox
    from webcam_locator.classify import classify_bbox
    det_type, det_corner = classify_bbox(
        med["x"], med["y"], med["width"], med["height"], frame_w, frame_h,
    )

    # Merge deterministic classification with per-frame vote.
    # The per-frame detections use smaller, per-frame bboxes which often
    # classify more accurately than the (possibly oversized) median bbox.
    vote_ratio = type_votes.most_common(1)[0][1] / len(best_cluster)

    if det_type == consensus_type:
        final_type = det_type
        final_corner = det_corner
    elif det_type == "center_box" and consensus_type != "center_box":
        # Deterministic says center_box but frames disagree → trust frames
        final_type = consensus_type
        final_corner = consensus_corner
    elif consensus_type == "side_box" and det_type == "corner_overlay" and vote_ratio >= 0.4:
        # Per-frame detections strongly say side_box but median bbox
        # touches a corner (common with face expansion overshoot)
        final_type = "side_box"
        final_corner = "unknown"
    elif consensus_type == "side_box" and det_type in ("center_box", "corner_overlay"):
        final_type = "side_box"
        final_corner = "unknown"
    elif vote_ratio >= 0.6 and consensus_type != det_type:
        # Strong per-frame agreement (>60%) overrides deterministic
        final_type = consensus_type
        final_corner = consensus_corner
    else:
        final_type = det_type
        final_corner = det_corner

    # ── Size regularization for side_box and corner_overlay ──
    # Data-driven: side_box/corner webcams are typically 5-8% of frame area,
    # width 17-30% of frame, height 20-30% of frame.
    # If the detected bbox is significantly oversized, shrink it toward
    # the center while keeping the position anchored to the nearest edge.
    if final_type in ("side_box", "corner_overlay"):
        area_ratio = (med["width"] * med["height"]) / (frame_w * frame_h)
        w_ratio = med["width"] / frame_w
        h_ratio = med["height"] / frame_h

        if area_ratio > 0.10 or w_ratio > 0.35 or h_ratio > 0.45:
            # Bbox is oversized. Shrink toward data-driven typical size.
            target_w = min(med["width"], int(frame_w * 0.25))
            target_h = min(med["height"], int(frame_h * 0.28))

            # Keep center, re-derive x/y
            cx = med["x"] + med["width"] // 2
            cy = med["y"] + med["height"] // 2
            new_x = cx - target_w // 2
            new_y = cy - target_h // 2

            # Re-snap to nearest frame edge if it was previously touching
            if med["x"] <= int(frame_w * 0.03):
                new_x = 0
            if med["y"] <= int(frame_h * 0.03):
                new_y = 0
            if (med["x"] + med["width"]) >= frame_w - int(frame_w * 0.03):
                new_x = frame_w - target_w
            if (med["y"] + med["height"]) >= frame_h - int(frame_h * 0.03):
                new_y = frame_h - target_h

            new_x = max(0, min(new_x, frame_w - target_w))
            new_y = max(0, min(new_y, frame_h - target_h))

            med = {"x": new_x, "y": new_y, "width": target_w, "height": target_h}

            # Re-classify after regularization
            final_type, final_corner = classify_bbox(
                med["x"], med["y"], med["width"], med["height"], frame_w, frame_h,
            )

    # Average confidence, boosted by agreement ratio
    avg_conf = np.mean([d["confidence"] for d in best_cluster])
    agreement_boost = len(best_cluster) / len(frames)
    confidence = min(1.0, float(avg_conf) * (0.7 + 0.3 * agreement_boost))

    # Stability score: mean pairwise IoU within cluster
    ious = []
    for i in range(len(best_cluster)):
        for j in range(i + 1, len(best_cluster)):
            ious.append(_iou(best_cluster[i], best_cluster[j]))
    stability = float(np.mean(ious)) if ious else 1.0

    return {
        "found": True,
        "type": final_type,
        "corner": final_corner,
        "x": med["x"],
        "y": med["y"],
        "width": med["width"],
        "height": med["height"],
        "confidence": round(confidence, 3),
        "reason": (
            f"consensus from {len(best_cluster)}/{len(frames)} frames, "
            f"stability={stability:.2f}"
        ),
    }
