"""
core.py – Main single-frame webcam detection API.

Detection strategy (in priority order):
  1. YOLO trained model (if available) – most accurate, directly detects webcam bbox
  2. Gemini Vision API proposal + refinement (if enabled)
  3. Face-based candidate expansion + boundary detection (fallback)

All results are classified using deterministic rules.
"""

import logging
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from webcam_locator.classify import classify_bbox
from webcam_locator.debug import save_debug_artifacts

log = logging.getLogger(__name__)

MIN_SCORE_THRESHOLD = 0.3
GEMINI_SCORE_BONUS = 5.0


def _no_webcam_result(reason: str = "no candidates") -> dict:
    return {
        "found": False,
        "type": "none",
        "corner": "unknown",
        "x": 0, "y": 0, "width": 0, "height": 0,
        "confidence": 0.0,
        "reason": reason,
    }


def detect_webcam_bbox(
    image_bgr: np.ndarray,
    *,
    use_gemini: bool = False,
    debug_dir: Optional[str] = None,
    frame_idx: int = 0,
) -> dict:
    """Detect the webcam overlay bbox in a single frame."""
    frame_h, frame_w = image_bgr.shape[:2]

    # ── Strategy 1: YOLO trained model (highest priority) ──
    try:
        from webcam_locator.yolo_detect import detect_with_yolo, is_yolo_available
        if is_yolo_available():
            yolo_result = detect_with_yolo(image_bgr)
            if yolo_result is not None:
                bx, by, bw, bh = yolo_result["x"], yolo_result["y"], yolo_result["width"], yolo_result["height"]
                bbox_type, corner = classify_bbox(bx, by, bw, bh, frame_w, frame_h)

                result = {
                    "found": True,
                    "type": bbox_type,
                    "corner": corner,
                    "x": bx, "y": by, "width": bw, "height": bh,
                    "confidence": round(yolo_result["confidence"], 3),
                    "reason": f"YOLO detection (conf={yolo_result['confidence']:.2f})",
                }

                if debug_dir:
                    save_debug_artifacts(
                        Path(debug_dir), frame_idx, image_bgr, [], np.array([]), [], result,
                    )
                return result

            # YOLO found nothing — still check for full_cam via other methods
    except ImportError:
        pass

    # ── Strategy 2+3: Gemini + face-based candidates (fallback) ──
    from webcam_locator.candidates import generate_candidates
    from webcam_locator.scoring import score_candidates

    candidates, faces, edge_map, boundary_info = generate_candidates(image_bgr)

    # Gemini proposal
    if use_gemini:
        try:
            from webcam_locator.gemini import gemini_propose
            from webcam_locator.refine import refine_bbox_inward, refine_two_pass

            gemini_result = gemini_propose(image_bgr, frame_w, frame_h)
            if gemini_result is not None:
                candidates.append(gemini_result)

                refined_2p = refine_two_pass(gemini_result, image_bgr)
                if refined_2p["width"] != gemini_result["width"] or refined_2p["height"] != gemini_result["height"]:
                    refined_2p["source"] = "gemini+2pass"
                    refined_2p["gemini_type"] = gemini_result.get("gemini_type")
                    refined_2p["gemini_confidence"] = gemini_result.get("gemini_confidence", 0.7)
                    candidates.append(refined_2p)

                refined = refine_bbox_inward(gemini_result, image_bgr)
                if refined["width"] != gemini_result["width"] or refined["height"] != gemini_result["height"]:
                    refined["source"] = "gemini+refined"
                    refined["gemini_type"] = gemini_result.get("gemini_type")
                    refined["gemini_confidence"] = gemini_result.get("gemini_confidence", 0.7)
                    candidates.append(refined)
        except Exception as exc:
            log.warning("Gemini candidate failed: %s", exc)

    if not candidates:
        result = _no_webcam_result("no candidates generated")
        if debug_dir:
            save_debug_artifacts(Path(debug_dir), frame_idx, image_bgr, faces, edge_map, [], result)
        return result

    has_bounds = boundary_info.get("has_internal_boundaries", True)
    scored = score_candidates(candidates, faces, edge_map, image_bgr, has_bounds)

    for c in scored:
        if c.get("source", "").startswith("gemini"):
            gc = c.get("gemini_confidence", 0.7)
            c["score"] += GEMINI_SCORE_BONUS * gc

    scored.sort(key=lambda c: c["score"], reverse=True)

    best = scored[0]
    score = best["score"]

    if score < MIN_SCORE_THRESHOLD:
        result = _no_webcam_result(f"best score {score:.2f} below threshold")
        if debug_dir:
            save_debug_artifacts(Path(debug_dir), frame_idx, image_bgr, faces, edge_map, scored, result)
        return result

    bx, by, bw, bh = best["x"], best["y"], best["width"], best["height"]

    if best.get("source") == "full_frame":
        bbox_type = "full_cam"
        corner = "unknown"
    elif best.get("source", "").startswith("gemini"):
        det_type, det_corner = classify_bbox(bx, by, bw, bh, frame_w, frame_h)
        gem_type = best.get("gemini_type", "unknown")
        if gem_type == "full_cam" and (bw * bh) / (frame_w * frame_h) > 0.5:
            bbox_type, corner = "full_cam", "unknown"
        else:
            bbox_type, corner = det_type, det_corner
    else:
        bbox_type, corner = classify_bbox(bx, by, bw, bh, frame_w, frame_h)

    max_positive_score = 4.0 + 2.5 + 3.0 + 2.0 + 1.5
    if best.get("source") == "full_frame":
        max_positive_score = 5.0
    elif best.get("source", "").startswith("gemini"):
        max_positive_score += GEMINI_SCORE_BONUS
    confidence = min(1.0, max(0.0, score / max_positive_score))

    result = {
        "found": True,
        "type": bbox_type,
        "corner": corner,
        "x": int(bx), "y": int(by), "width": int(bw), "height": int(bh),
        "confidence": round(confidence, 3),
        "reason": f"best of {len(scored)} candidates (score={score:.2f}, src={best.get('source', '?')})",
    }

    if debug_dir:
        save_debug_artifacts(Path(debug_dir), frame_idx, image_bgr, faces, edge_map, scored, result)

    return result
