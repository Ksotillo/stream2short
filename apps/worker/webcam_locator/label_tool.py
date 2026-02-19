"""
label_tool.py – Interactive visual labeling tool for webcam bounding boxes.

Opens each case frame in an OpenCV window. Draw a rectangle by clicking
and dragging. Press a key to set the type. Saves to truth.json in frame
coordinates.

Controls:
  Click + drag    Draw bounding box
  1               Set type: side_box
  2               Set type: corner_overlay
  3               Set type: full_cam
  4               Set type: none (clears bbox)
  s / Enter       Save current label and go to next case
  r               Reset (clear current drawing)
  q               Quit (saves current progress)
  n               Skip to next without saving
  p               Go back to previous case

Usage:
    python -m webcam_locator.label_tool --dataset dataset
"""

import json
import logging
import sys
from pathlib import Path

import cv2
import numpy as np

log = logging.getLogger(__name__)

TYPES = {
    ord("1"): "side_box",
    ord("2"): "corner_overlay",
    ord("3"): "full_cam",
    ord("4"): "none",
}

COLORS = {
    "side_box": (0, 255, 255),       # yellow
    "corner_overlay": (0, 165, 255), # orange
    "full_cam": (0, 255, 0),         # green
    "none": (0, 0, 255),             # red
}

drawing = False
ix, iy = 0, 0
fx, fy = 0, 0
current_bbox = None
current_type = "side_box"


def mouse_callback(event, x, y, flags, param):
    global drawing, ix, iy, fx, fy, current_bbox

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix, iy = x, y
        fx, fy = x, y

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            fx, fy = x, y

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        fx, fy = x, y
        x1 = min(ix, fx)
        y1 = min(iy, fy)
        x2 = max(ix, fx)
        y2 = max(iy, fy)
        if x2 - x1 > 10 and y2 - y1 > 10:
            current_bbox = (x1, y1, x2 - x1, y2 - y1)


def render(img, bbox, label_type, case_id, frame_path, has_existing):
    canvas = img.copy()
    fh, fw = canvas.shape[:2]

    # Draw current/in-progress bbox
    if drawing:
        x1 = min(ix, fx)
        y1 = min(iy, fy)
        x2 = max(ix, fx)
        y2 = max(iy, fy)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), (255, 255, 255), 2)

    if bbox:
        bx, by, bw, bh = bbox
        color = COLORS.get(label_type, (255, 255, 255))
        cv2.rectangle(canvas, (bx, by), (bx + bw, by + bh), color, 3)
        cv2.putText(canvas, label_type, (bx, by - 10 if by > 30 else by + bh + 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    # HUD
    cv2.putText(canvas, f"{case_id}  [{fw}x{fh}]", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    status = "HAS LABEL" if has_existing else "NO LABEL"
    cv2.putText(canvas, status, (fw - 180, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0) if has_existing else (0, 0, 255), 1)

    help_text = "1:side_box  2:corner  3:full_cam  4:none  |  ENTER:save  R:reset  N:skip  Q:quit"
    cv2.putText(canvas, help_text, (10, fh - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

    sel_text = f"Type: {label_type}"
    cv2.putText(canvas, sel_text, (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLORS.get(label_type, (255, 255, 255)), 2)

    if bbox:
        bx, by, bw, bh = bbox
        cv2.putText(canvas, f"({bx},{by}) {bw}x{bh}", (10, 75),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

    return canvas


def load_existing_label(truth_path: Path, frame_w: int, frame_h: int):
    """Load existing truth.json and return (bbox, type) or (None, 'side_box')."""
    if not truth_path.exists():
        return None, "side_box"

    truth = json.load(open(truth_path))
    if "segments" in truth:
        segs = [s for s in truth["segments"] if s["type"] != "none"]
        if not segs:
            return None, "none"
        seg = max(segs, key=lambda s: s.get("frame_count", 1))
        return (seg["x"], seg["y"], seg["width"], seg["height"]), seg["type"]
    elif truth.get("found"):
        return (truth["x"], truth["y"], truth["width"], truth["height"]), truth.get("type", "side_box")
    return None, "none"


def save_label(truth_path: Path, bbox, label_type, frame_paths):
    if label_type == "none" or bbox is None:
        data = {
            "segment_count": 0,
            "has_transitions": False,
            "dominant_type": "none",
            "segments": [],
        }
    else:
        bx, by, bw, bh = bbox
        data = {
            "segment_count": 1,
            "has_transitions": False,
            "dominant_type": label_type,
            "segments": [{
                "type": label_type,
                "corner": "unknown",
                "x": bx, "y": by, "width": bw, "height": bh,
                "frame_start": frame_paths[0].name,
                "frame_end": frame_paths[-1].name,
                "time_start_sec": 0,
                "time_end_sec": 30,
                "frame_count": len(frame_paths),
            }],
        }

    with open(truth_path, "w") as fh:
        json.dump(data, fh, indent=2)


def run_label_tool(dataset_dir: Path, start_case: int = 0):
    global current_bbox, current_type

    cases_dir = dataset_dir / "cases"
    case_dirs = sorted(d for d in cases_dir.iterdir() if d.is_dir())

    if not case_dirs:
        print("No cases found")
        return

    win_name = "Webcam Label Tool"
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(win_name, mouse_callback)

    idx = min(start_case, len(case_dirs) - 1)

    while 0 <= idx < len(case_dirs):
        case_dir = case_dirs[idx]
        case_id = case_dir.name
        frames_dir = case_dir / "frames"
        truth_path = case_dir / "truth.json"

        frame_paths = sorted(frames_dir.glob("*.jpg")) if frames_dir.exists() else []
        if not frame_paths:
            idx += 1
            continue

        mid_frame = frame_paths[len(frame_paths) // 2]
        img = cv2.imread(str(mid_frame))
        if img is None:
            idx += 1
            continue

        fh, fw = img.shape[:2]

        # Load existing label
        existing_bbox, existing_type = load_existing_label(truth_path, fw, fh)
        current_bbox = existing_bbox
        current_type = existing_type
        has_existing = existing_bbox is not None or existing_type == "none"

        print(f"\n[{idx + 1}/{len(case_dirs)}] {case_id} - existing: {existing_type} "
              f"bbox={existing_bbox or 'none'}")

        while True:
            canvas = render(img, current_bbox, current_type, case_id, mid_frame, has_existing)
            cv2.imshow(win_name, canvas)

            key = cv2.waitKey(30) & 0xFF

            if key == 255:
                continue

            if key in TYPES:
                current_type = TYPES[key]
                if current_type == "none":
                    current_bbox = None
                elif current_type == "full_cam":
                    current_bbox = (0, 0, fw, fh)
                print(f"  Type set to: {current_type}")

            elif key == ord("r"):
                current_bbox = None
                current_type = "side_box"
                print("  Reset")

            elif key in (13, ord("s")):  # Enter or S
                save_label(truth_path, current_bbox, current_type, frame_paths)
                print(f"  SAVED: {current_type} bbox={current_bbox}")
                idx += 1
                break

            elif key == ord("n"):
                print("  Skipped")
                idx += 1
                break

            elif key == ord("p"):
                idx = max(0, idx - 1)
                break

            elif key == ord("q"):
                print("\nQuit")
                cv2.destroyAllWindows()
                return

    cv2.destroyAllWindows()
    print(f"\nDone! Labeled all {len(case_dirs)} cases.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Interactive webcam labeling tool")
    parser.add_argument("--dataset", type=Path, default=Path("dataset"))
    parser.add_argument("--start", type=int, default=0, help="Start from case index (0-based)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    run_label_tool(args.dataset, start_case=args.start)
