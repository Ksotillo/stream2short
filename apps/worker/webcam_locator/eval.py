"""
eval.py – Evaluation harness for webcam detection against labeled dataset.

Computes per-case IoU, type accuracy, and a summary report.

Usage:
    python -m webcam_locator.eval --dataset ./dataset --out eval_report.json
"""

import json
import logging
import sys
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

log = logging.getLogger(__name__)


def compute_iou(pred: dict, truth: dict) -> float:
    """Compute IoU between prediction and ground truth bbox dicts."""
    ax1, ay1 = pred["x"], pred["y"]
    ax2, ay2 = ax1 + pred["width"], ay1 + pred["height"]
    bx1, by1 = truth["x"], truth["y"]
    bx2, by2 = bx1 + truth["width"], by1 + truth["height"]

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    area_a = pred["width"] * pred["height"]
    area_b = truth["width"] * truth["height"]
    union = area_a + area_b - inter

    return inter / union if union > 0 else 0.0


def _load_truth(truth_path: Path) -> Optional[dict]:
    """Load truth.json, handling both segment-based and flat formats."""
    with open(truth_path) as fh:
        data = json.load(fh)

    # Segment-based format: use the dominant segment's bbox
    if "segments" in data:
        segments = data["segments"]
        # Find the largest non-none segment by frame_count
        best_seg = None
        for seg in segments:
            if seg["type"] == "none":
                continue
            if best_seg is None or seg.get("frame_count", 1) > best_seg.get("frame_count", 1):
                best_seg = seg

        if best_seg is None:
            return {"found": False, "type": "none", "corner": "unknown",
                    "x": 0, "y": 0, "width": 0, "height": 0}

        return {
            "found": True,
            "type": best_seg["type"],
            "corner": best_seg.get("corner", "unknown"),
            "x": best_seg["x"],
            "y": best_seg["y"],
            "width": best_seg["width"],
            "height": best_seg["height"],
        }

    # Flat format: use directly
    return data


def evaluate_dataset(
    dataset_dir: Path,
    use_gemini: bool = False,
    debug_dir: Optional[Path] = None,
) -> dict:
    """Run detection on all cases in the dataset and compare against truth.

    Returns a report dict with per-case and summary metrics.
    """
    from webcam_locator.consensus import detect_from_frames

    cases_dir = dataset_dir / "cases"
    if not cases_dir.exists():
        log.error("No cases/ directory found in %s", dataset_dir)
        return {"error": "no cases directory"}

    case_dirs = sorted(d for d in cases_dir.iterdir() if d.is_dir())
    log.info("Evaluating %d cases", len(case_dirs))

    results = []
    for case_dir in case_dirs:
        case_id = case_dir.name
        truth_path = case_dir / "truth.json"
        frames_dir = case_dir / "frames"

        if not truth_path.exists():
            log.warning("Skipping %s (no truth.json)", case_id)
            continue

        truth = _load_truth(truth_path)
        if truth is None:
            log.warning("Skipping %s (invalid truth.json)", case_id)
            continue

        if not frames_dir.exists():
            log.warning("Skipping %s (no frames/)", case_id)
            continue

        # Load frames
        frame_paths = sorted(frames_dir.glob("*.jpg"))
        if not frame_paths:
            log.warning("Skipping %s (no .jpg frames)", case_id)
            continue

        frames = []
        for fp in frame_paths:
            img = cv2.imread(str(fp))
            if img is not None:
                frames.append(img)

        if not frames:
            continue

        # Scale truth coordinates from original video res to frame res
        frame_h, frame_w = frames[0].shape[:2]
        meta_path = case_dir / "meta.json"
        if meta_path.exists() and truth.get("found"):
            meta = json.load(open(meta_path))
            orig_size = meta.get("frame_size")
            if orig_size:
                orig_w, orig_h = orig_size["width"], orig_size["height"]
                # Only scale if truth coords appear to be in original resolution
                # (i.e., truth bbox extends beyond frame dimensions)
                truth_exceeds = (truth["x"] + truth["width"] > frame_w + 5 or
                                 truth["y"] + truth["height"] > frame_h + 5)
                if truth_exceeds and (orig_w != frame_w or orig_h != frame_h):
                    sx = frame_w / orig_w
                    sy = frame_h / orig_h
                    truth["x"] = int(truth["x"] * sx)
                    truth["y"] = int(truth["y"] * sy)
                    truth["width"] = int(truth["width"] * sx)
                    truth["height"] = int(truth["height"] * sy)
                    truth["width"] = min(truth["width"], frame_w - truth["x"])
                    truth["height"] = min(truth["height"], frame_h - truth["y"])

        # Run detection
        case_debug = str(debug_dir / case_id) if debug_dir else None
        pred = detect_from_frames(frames, use_gemini=use_gemini, debug_dir=case_debug)

        # Compute metrics
        iou = 0.0
        type_match = False

        if truth.get("found") and pred.get("found"):
            iou = compute_iou(pred, truth)
            type_match = pred["type"] == truth["type"]
        elif not truth.get("found") and not pred.get("found"):
            iou = 1.0
            type_match = True

        results.append({
            "case_id": case_id,
            "truth_type": truth.get("type", "none"),
            "pred_type": pred.get("type", "none"),
            "truth_found": truth.get("found", False),
            "pred_found": pred.get("found", False),
            "iou": round(iou, 4),
            "type_match": type_match,
            "pred_confidence": pred.get("confidence", 0),
            "pred_reason": pred.get("reason", ""),
        })

        status = "OK" if type_match and iou > 0.3 else "MISS"
        log.info(
            "  %s: %s | IoU=%.3f type=%s->%s conf=%.2f",
            case_id, status, iou, truth.get("type"), pred.get("type"),
            pred.get("confidence", 0),
        )

    # Summary
    if not results:
        return {"error": "no valid cases to evaluate", "cases": []}

    ious = [r["iou"] for r in results]
    type_matches = [r["type_match"] for r in results]
    found_correct = sum(
        1 for r in results
        if r["truth_found"] == r["pred_found"]
    )

    summary = {
        "total_cases": len(results),
        "mean_iou": round(float(np.mean(ious)), 4),
        "median_iou": round(float(np.median(ious)), 4),
        "type_accuracy": round(sum(type_matches) / len(type_matches), 4),
        "found_accuracy": round(found_correct / len(results), 4),
        "iou_above_50": sum(1 for x in ious if x >= 0.5),
        "iou_above_30": sum(1 for x in ious if x >= 0.3),
    }

    return {"summary": summary, "cases": results}


def print_report(report: dict) -> None:
    """Print a human-readable evaluation summary table."""
    if "error" in report:
        print(f"Error: {report['error']}")
        return

    summary = report["summary"]
    cases = report["cases"]

    print("\n" + "=" * 72)
    print("WEBCAM LOCATOR EVALUATION REPORT")
    print("=" * 72)
    print(f"  Total cases:      {summary['total_cases']}")
    print(f"  Mean IoU:         {summary['mean_iou']:.4f}")
    print(f"  Median IoU:       {summary['median_iou']:.4f}")
    print(f"  Type accuracy:    {summary['type_accuracy']:.1%}")
    print(f"  Found accuracy:   {summary['found_accuracy']:.1%}")
    print(f"  IoU >= 0.5:       {summary['iou_above_50']}/{summary['total_cases']}")
    print(f"  IoU >= 0.3:       {summary['iou_above_30']}/{summary['total_cases']}")
    print("=" * 72)

    print(f"\n{'Case':<12} {'Truth':<16} {'Pred':<16} {'IoU':>6} {'Type':>5} {'Conf':>5}")
    print("-" * 72)
    for c in cases:
        match_str = "Y" if c["type_match"] else "N"
        print(
            f"{c['case_id']:<12} {c['truth_type']:<16} {c['pred_type']:<16} "
            f"{c['iou']:>6.3f} {match_str:>5} {c['pred_confidence']:>5.2f}"
        )
    print()


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate webcam locator against labeled dataset")
    parser.add_argument("--dataset", required=True, type=Path, help="Path to dataset root")
    parser.add_argument("--out", type=Path, default=None, help="Save report JSON")
    parser.add_argument("--debug-dir", type=Path, default=None, help="Save debug artifacts")
    parser.add_argument("--use-gemini", type=int, default=0)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")

    report = evaluate_dataset(
        args.dataset,
        use_gemini=bool(args.use_gemini),
        debug_dir=args.debug_dir,
    )

    print_report(report)

    if args.out:
        with open(args.out, "w") as fh:
            json.dump(report, fh, indent=2)
        print(f"Report saved to {args.out}")


if __name__ == "__main__":
    main()
