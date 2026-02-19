"""
relabel.py – Auto-relabel the dataset using YOLO detections, then generate
visual verification images so the user can quickly spot-check.

Usage:
    python -m webcam_locator.relabel --dataset dataset --verify-dir Relabel_verification
"""

import json
import logging
from pathlib import Path

import cv2
import numpy as np

log = logging.getLogger(__name__)


def relabel_dataset(
    dataset_dir: Path,
    verify_dir: Path | None = None,
) -> dict:
    """Re-generate truth.json for all cases using YOLO detections.

    The YOLO model produces accurate bboxes in frame coordinates (1280x720),
    which eliminates any coordinate mismatch from manual labeling at wrong resolution.
    """
    from webcam_locator.core import detect_webcam_bbox
    from webcam_locator.classify import classify_bbox

    cases_dir = dataset_dir / "cases"
    if verify_dir:
        verify_dir.mkdir(parents=True, exist_ok=True)

    case_dirs = sorted(d for d in cases_dir.iterdir() if d.is_dir())
    stats = {"total": 0, "relabeled": 0, "unchanged": 0, "type_changes": []}

    for case_dir in case_dirs:
        case_id = case_dir.name
        frames_dir = case_dir / "frames"
        truth_path = case_dir / "truth.json"

        if not frames_dir.exists():
            continue

        frame_paths = sorted(frames_dir.glob("*.jpg"))
        if not frame_paths:
            continue

        stats["total"] += 1

        # Load old truth for comparison
        old_truth = None
        old_type = "?"
        if truth_path.exists():
            old_truth = json.load(open(truth_path))
            if "segments" in old_truth:
                segs = [s for s in old_truth["segments"] if s["type"] != "none"]
                old_type = max(segs, key=lambda s: s.get("frame_count", 1))["type"] if segs else "none"
            else:
                old_type = old_truth.get("type", "?")

        # Run YOLO detection on middle frame
        mid_frame = frame_paths[len(frame_paths) // 2]
        img = cv2.imread(str(mid_frame))
        if img is None:
            continue

        fh, fw = img.shape[:2]
        result = detect_webcam_bbox(img)

        if result["found"]:
            bx, by, bw, bh = result["x"], result["y"], result["width"], result["height"]
            det_type, det_corner = classify_bbox(bx, by, bw, bh, fw, fh)

            # For full_cam: use full frame as bbox
            if det_type == "full_cam":
                bx, by, bw, bh = 0, 0, fw, fh

            new_truth = {
                "segment_count": 1,
                "has_transitions": False,
                "dominant_type": det_type,
                "segments": [{
                    "type": det_type,
                    "corner": det_corner,
                    "x": bx, "y": by, "width": bw, "height": bh,
                    "frame_start": frame_paths[0].name,
                    "frame_end": frame_paths[-1].name,
                    "time_start_sec": 0,
                    "time_end_sec": 30,
                    "frame_count": len(frame_paths),
                }],
            }
        else:
            new_truth = {
                "segment_count": 0,
                "has_transitions": False,
                "dominant_type": "none",
                "segments": [],
            }

        new_type = new_truth["dominant_type"]

        # Write new truth
        with open(truth_path, "w") as fh_out:
            json.dump(new_truth, fh_out, indent=2)

        if old_type != new_type:
            stats["type_changes"].append(f"{case_id}: {old_type} -> {new_type}")
            stats["relabeled"] += 1
        else:
            stats["unchanged"] += 1

        # Generate verification image
        if verify_dir:
            canvas = img.copy()
            if result["found"]:
                cv2.rectangle(canvas, (bx, by), (bx + bw, by + bh), (0, 255, 255), 3)
                label = f"{new_type} ({result['confidence']:.0%})"
                cv2.putText(canvas, label, (bx, by - 10 if by > 30 else by + bh + 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            else:
                cv2.putText(canvas, "NONE", (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)

            change_text = f"{'CHANGED' if old_type != new_type else 'same'}: {old_type} -> {new_type}"
            color = (0, 0, 255) if old_type != new_type else (0, 255, 0)
            cv2.putText(canvas, f"{case_id}  {change_text}", (10, fh - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            cv2.imwrite(str(verify_dir / f"{case_id}.jpg"), canvas, [cv2.IMWRITE_JPEG_QUALITY, 92])

        log.info("  %s: %s -> %s %s", case_id, old_type, new_type,
                 "(CHANGED)" if old_type != new_type else "")

    print(f"\nRelabeling complete:")
    print(f"  Total cases: {stats['total']}")
    print(f"  Unchanged:   {stats['unchanged']}")
    print(f"  Relabeled:   {stats['relabeled']}")
    if stats["type_changes"]:
        print(f"  Type changes:")
        for tc in stats["type_changes"]:
            print(f"    {tc}")

    return stats


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Auto-relabel dataset using YOLO detections")
    parser.add_argument("--dataset", type=Path, default=Path("dataset"))
    parser.add_argument("--verify-dir", type=Path, default=Path("Relabel_verification"))
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
    relabel_dataset(args.dataset, args.verify_dir)
