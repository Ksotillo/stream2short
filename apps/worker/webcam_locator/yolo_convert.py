"""
yolo_convert.py – Convert our segment-based dataset to YOLO format for training.

Creates:
  yolo_dataset/
    images/
      train/   (80% of frames)
      val/     (20% of frames)
    labels/
      train/
      val/
    data.yaml

YOLO label format: class_id x_center y_center width height (all normalized 0-1)
Single class: 0 = webcam
"""

import json
import logging
import random
import shutil
from pathlib import Path

import cv2

log = logging.getLogger(__name__)


def convert_dataset(
    dataset_dir: Path,
    output_dir: Path,
    train_split: float = 0.8,
    seed: int = 42,
) -> None:
    """Convert segment-based dataset to YOLO format."""
    random.seed(seed)
    cases_dir = dataset_dir / "cases"

    # Create directory structure
    for split in ("train", "val"):
        (output_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (output_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

    all_frames = []

    for case_dir in sorted(cases_dir.iterdir()):
        if not case_dir.is_dir():
            continue

        truth_path = case_dir / "truth.json"
        meta_path = case_dir / "meta.json"
        frames_dir = case_dir / "frames"

        if not truth_path.exists() or not frames_dir.exists():
            continue

        truth = json.load(open(truth_path))
        meta = json.load(open(meta_path)) if meta_path.exists() else {}

        # Get original video resolution for coordinate scaling
        orig_size = meta.get("frame_size", {})
        orig_w = orig_size.get("width", 0)
        orig_h = orig_size.get("height", 0)

        # Get the dominant non-none segment
        if "segments" in truth:
            segments = [s for s in truth["segments"] if s["type"] != "none"]
            if not segments:
                continue
            seg = max(segments, key=lambda s: s.get("frame_count", 1))
        elif truth.get("found", False):
            seg = truth
        else:
            continue

        if seg["type"] == "none":
            continue

        # Get all frame paths
        frame_paths = sorted(frames_dir.glob("*.jpg"))
        if not frame_paths:
            continue

        # Get actual frame dimensions
        sample = cv2.imread(str(frame_paths[0]))
        if sample is None:
            continue
        frame_h, frame_w = sample.shape[:2]

        # Only scale if truth coords are in original resolution
        # (i.e., they extend beyond the frame dimensions)
        truth_exceeds = (seg["x"] + seg["width"] > frame_w + 5 or
                         seg["y"] + seg["height"] > frame_h + 5)

        if truth_exceeds and orig_w > 0 and orig_h > 0 and (orig_w != frame_w or orig_h != frame_h):
            sx = frame_w / orig_w
            sy = frame_h / orig_h
        else:
            sx, sy = 1.0, 1.0

        tx = seg["x"] * sx
        ty = seg["y"] * sy
        tw = seg["width"] * sx
        th = seg["height"] * sy

        # Clamp to frame
        tw = min(tw, frame_w - tx)
        th = min(th, frame_h - ty)

        if tw < 10 or th < 10:
            continue

        # Convert to YOLO format (normalized center + size)
        x_center = (tx + tw / 2) / frame_w
        y_center = (ty + th / 2) / frame_h
        w_norm = tw / frame_w
        h_norm = th / frame_h

        # Clamp to [0, 1]
        x_center = max(0.0, min(1.0, x_center))
        y_center = max(0.0, min(1.0, y_center))
        w_norm = max(0.01, min(1.0, w_norm))
        h_norm = max(0.01, min(1.0, h_norm))

        label_line = f"0 {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}\n"

        for fp in frame_paths:
            all_frames.append((fp, label_line, case_dir.name))

    # Shuffle and split
    random.shuffle(all_frames)
    split_idx = int(len(all_frames) * train_split)
    train_frames = all_frames[:split_idx]
    val_frames = all_frames[split_idx:]

    # Write files
    for split_name, split_frames in [("train", train_frames), ("val", val_frames)]:
        for fp, label, case_id in split_frames:
            # Unique filename: case_001_0003.jpg
            out_name = f"{case_id}_{fp.stem}"
            img_dst = output_dir / "images" / split_name / f"{out_name}.jpg"
            lbl_dst = output_dir / "labels" / split_name / f"{out_name}.txt"

            shutil.copy2(fp, img_dst)
            lbl_dst.write_text(label)

    # Write data.yaml
    yaml_content = f"""path: {output_dir.resolve()}
train: images/train
val: images/val

names:
  0: webcam
"""
    (output_dir / "data.yaml").write_text(yaml_content)

    log.info(
        "YOLO dataset: %d train, %d val images from %d total frames",
        len(train_frames), len(val_frames), len(all_frames),
    )
    print(f"YOLO dataset created: {output_dir}")
    print(f"  Train: {len(train_frames)} images")
    print(f"  Val:   {len(val_frames)} images")
    print(f"  Config: {output_dir / 'data.yaml'}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Convert dataset to YOLO format")
    parser.add_argument("--dataset", type=Path, default=Path("dataset"))
    parser.add_argument("--output", type=Path, default=Path("yolo_dataset"))
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    convert_dataset(args.dataset, args.output)
