"""
debug.py – Save debug artifacts for visual inspection.

When --debug-dir is set, saves per-frame:
  frame_original.jpg, frame_faces.jpg, frame_edges.jpg,
  frame_candidates.jpg, frame_final.jpg, frame_debug.json
"""

import json
import logging
from pathlib import Path

import cv2
import numpy as np

log = logging.getLogger(__name__)

_COLORS = [
    (0, 255, 0),    # green
    (255, 0, 0),    # blue
    (0, 0, 255),    # red
    (255, 255, 0),  # cyan
    (0, 255, 255),  # yellow
    (255, 0, 255),  # magenta
]


def save_debug_artifacts(
    debug_dir: Path,
    frame_idx: int,
    image_bgr: np.ndarray,
    faces: list[dict],
    edge_map: np.ndarray,
    candidates: list[dict],
    result: dict,
) -> None:
    """Write all debug images and JSON for one frame."""
    debug_dir.mkdir(parents=True, exist_ok=True)
    prefix = f"frame_{frame_idx:04d}"

    # 1. Original
    cv2.imwrite(str(debug_dir / f"{prefix}_original.jpg"), image_bgr)

    # 2. Faces
    img_faces = image_bgr.copy()
    for f in faces:
        cv2.rectangle(
            img_faces,
            (f["x"], f["y"]),
            (f["x"] + f["width"], f["y"] + f["height"]),
            (0, 255, 0), 2,
        )
        label = f'{f.get("method", "?")} {f.get("confidence", 0):.2f}'
        cv2.putText(img_faces, label, (f["x"], f["y"] - 5),
                     cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    cv2.imwrite(str(debug_dir / f"{prefix}_faces.jpg"), img_faces)

    # 3. Edge map
    cv2.imwrite(str(debug_dir / f"{prefix}_edges.jpg"), edge_map)

    # 4. Top candidates (up to 6)
    img_cands = image_bgr.copy()
    for i, c in enumerate(candidates[:6]):
        color = _COLORS[i % len(_COLORS)]
        cv2.rectangle(
            img_cands,
            (c["x"], c["y"]),
            (c["x"] + c["width"], c["y"] + c["height"]),
            color, 2,
        )
        score_str = f'#{i + 1} {c.get("score", 0):.2f}'
        cv2.putText(img_cands, score_str, (c["x"], c["y"] - 5),
                     cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    cv2.imwrite(str(debug_dir / f"{prefix}_candidates.jpg"), img_cands)

    # 5. Final result
    img_final = image_bgr.copy()
    if result.get("found"):
        rx, ry = result["x"], result["y"]
        rw, rh = result["width"], result["height"]
        cv2.rectangle(img_final, (rx, ry), (rx + rw, ry + rh), (0, 255, 0), 3)
        label = f'{result["type"]} ({result.get("confidence", 0):.2f})'
        cv2.putText(img_final, label, (rx, ry - 10),
                     cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    else:
        cv2.putText(img_final, "NO WEBCAM", (30, 40),
                     cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
    cv2.imwrite(str(debug_dir / f"{prefix}_final.jpg"), img_final)

    # 6. Debug JSON
    def _sanitize(obj):
        """Make dicts JSON-serializable (strip numpy, face refs, etc.)."""
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, dict):
            return {k: _sanitize(v) for k, v in obj.items() if k != "face"}
        if isinstance(obj, (list, tuple)):
            return [_sanitize(v) for v in obj]
        return obj

    debug_data = {
        "frame_idx": frame_idx,
        "faces": _sanitize(faces),
        "candidate_count": len(candidates),
        "top_candidates": _sanitize(candidates[:6]),
        "result": _sanitize(result),
    }
    with open(debug_dir / f"{prefix}_debug.json", "w") as fh:
        json.dump(debug_data, fh, indent=2, default=str)
