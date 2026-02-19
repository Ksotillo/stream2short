"""
visual_test.py – Generate visual test results for all dataset cases.

For each case, takes the middle frame, draws the detected bbox in yellow
with a type label, and saves to Detection_visual_test_results/.
"""

import json
import logging
import sys
from pathlib import Path

import cv2
import numpy as np

log = logging.getLogger(__name__)

YELLOW = (0, 255, 255)
RED = (0, 0, 255)
GREEN = (0, 255, 0)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)


def run_visual_test(
    dataset_dir: Path,
    output_dir: Path,
    use_gemini: bool = False,
) -> None:
    from webcam_locator.core import detect_webcam_bbox

    cases_dir = dataset_dir / "cases"
    output_dir.mkdir(parents=True, exist_ok=True)

    case_dirs = sorted(d for d in cases_dir.iterdir() if d.is_dir())
    log.info("Visual test: %d cases -> %s", len(case_dirs), output_dir)

    for case_dir in case_dirs:
        case_id = case_dir.name
        frames_dir = case_dir / "frames"
        truth_path = case_dir / "truth.json"
        meta_path = case_dir / "meta.json"

        if not frames_dir.exists():
            continue

        frame_paths = sorted(frames_dir.glob("*.jpg"))
        if not frame_paths:
            continue

        # Pick the middle frame
        mid_idx = len(frame_paths) // 2
        frame_path = frame_paths[mid_idx]

        img = cv2.imread(str(frame_path))
        if img is None:
            continue

        fh, fw = img.shape[:2]

        # Run detection
        result = detect_webcam_bbox(img, use_gemini=use_gemini)

        # Load truth for comparison
        truth_type = "?"
        truth_bbox = None
        if truth_path.exists():
            truth = json.load(open(truth_path))
            if "segments" in truth:
                segs = [s for s in truth["segments"] if s["type"] != "none"]
                if segs:
                    seg = max(segs, key=lambda s: s.get("frame_count", 1))
                    truth_type = seg["type"]
                    # Scale truth coords
                    if meta_path.exists():
                        meta = json.load(open(meta_path))
                        orig = meta.get("frame_size", {})
                        ow, oh = orig.get("width", fw), orig.get("height", fh)
                        sx, sy = fw / ow, fh / oh
                        truth_bbox = {
                            "x": int(seg["x"] * sx), "y": int(seg["y"] * sy),
                            "width": int(seg["width"] * sx), "height": int(seg["height"] * sy),
                        }
                else:
                    truth_type = "none"
            else:
                truth_type = truth.get("type", "?")

        canvas = img.copy()

        # Draw truth bbox in green (dashed effect via thinner line)
        if truth_bbox and truth_type != "none":
            tx, ty = truth_bbox["x"], truth_bbox["y"]
            tw, th = truth_bbox["width"], truth_bbox["height"]
            cv2.rectangle(canvas, (tx, ty), (tx + tw, ty + th), GREEN, 2)
            cv2.putText(canvas, f"TRUTH: {truth_type}", (tx, max(ty - 8, 15)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, GREEN, 2)

        # Draw detection bbox in yellow
        if result["found"]:
            px, py = result["x"], result["y"]
            pw, ph = result["width"], result["height"]
            cv2.rectangle(canvas, (px, py), (px + pw, py + ph), YELLOW, 3)

            label = f"{result['type']} ({result['confidence']:.0%})"
            label_y = py + ph + 25 if py + ph + 30 < fh else py - 10
            # Background for label
            (tw_text, th_text), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            cv2.rectangle(canvas, (px, label_y - th_text - 5), (px + tw_text + 10, label_y + 5), BLACK, -1)
            cv2.putText(canvas, label, (px + 5, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, YELLOW, 2)

            # Type match indicator
            match = result["type"] == truth_type
            status = "MATCH" if match else f"EXPECTED: {truth_type}"
            status_color = GREEN if match else RED
            cv2.putText(canvas, status, (10, fh - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)
        else:
            cv2.putText(canvas, "NO WEBCAM DETECTED", (10, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, RED, 2)
            if truth_type != "none":
                cv2.putText(canvas, f"EXPECTED: {truth_type}", (10, fh - 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, RED, 2)

        # Case ID in top-left
        cv2.putText(canvas, case_id, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, WHITE, 2)

        out_path = output_dir / f"{case_id}.jpg"
        cv2.imwrite(str(out_path), canvas, [cv2.IMWRITE_JPEG_QUALITY, 92])
        log.info("  %s: pred=%s truth=%s", case_id, result.get("type", "none"), truth_type)

    print(f"\nVisual results saved to: {output_dir}")
    print(f"  {len(case_dirs)} case images generated")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate visual detection test results")
    parser.add_argument("--dataset", type=Path, default=Path("dataset"))
    parser.add_argument("--output", type=Path, default=Path("Detection_visual_test_results"))
    parser.add_argument("--use-gemini", type=int, default=0)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
    run_visual_test(args.dataset, args.output, use_gemini=bool(args.use_gemini))
