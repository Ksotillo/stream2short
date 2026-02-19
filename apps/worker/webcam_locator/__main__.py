"""
__main__.py – CLI entry point for webcam_locator.

Usage:
    python -m webcam_locator detect --images img1.jpg img2.jpg
    python -m webcam_locator detect --frames-dir ./clip_frames/
    python -m webcam_locator detect --input ./dataset --out result.json --debug-dir ./debug
    python -m webcam_locator eval --dataset ./dataset --out eval_report.json
    python -m webcam_locator extract-frames --video clip.mp4 --out ./frames --times 3,10,15
"""

import argparse
import json
import logging
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

log = logging.getLogger("webcam_locator")


def cmd_detect(args: argparse.Namespace) -> None:
    """Run webcam detection."""
    from webcam_locator.core import detect_webcam_bbox
    from webcam_locator.consensus import detect_from_frames

    debug_dir = str(args.debug_dir) if args.debug_dir else None
    use_gemini = bool(args.use_gemini)

    results = {}

    # Mode 1: single images
    if args.images:
        for img_path in args.images:
            img = cv2.imread(str(img_path))
            if img is None:
                log.error("Cannot read image: %s", img_path)
                continue
            result = detect_webcam_bbox(img, use_gemini=use_gemini, debug_dir=debug_dir)
            results[str(img_path)] = result
            _print_result(str(img_path), result)

    # Mode 2: frames directory (single case)
    elif args.frames_dir:
        frames_dir = Path(args.frames_dir)
        frames = _load_frames(frames_dir)
        if not frames:
            log.error("No frames found in %s", frames_dir)
            sys.exit(1)
        result = detect_from_frames(frames, use_gemini=use_gemini, debug_dir=debug_dir)
        results["consensus"] = result
        _print_result("consensus", result)

    # Mode 3: dataset (multiple cases)
    elif args.input:
        dataset_dir = Path(args.input)
        cases_dir = dataset_dir / "cases"
        if not cases_dir.exists():
            log.error("No cases/ directory in %s", dataset_dir)
            sys.exit(1)

        case_dirs = sorted(d for d in cases_dir.iterdir() if d.is_dir())
        log.info("Processing %d cases", len(case_dirs))

        for case_dir in case_dirs:
            case_id = case_dir.name
            frames_dir = case_dir / "frames"
            if not frames_dir.exists():
                continue

            frames = _load_frames(frames_dir)
            if not frames:
                continue

            case_debug = str(Path(debug_dir) / case_id) if debug_dir else None
            result = detect_from_frames(frames, use_gemini=use_gemini, debug_dir=case_debug)
            results[case_id] = result
            _print_result(case_id, result)

    else:
        log.error("Provide --images, --frames-dir, or --input")
        sys.exit(1)

    # Save output
    if args.out:
        with open(args.out, "w") as fh:
            json.dump(results, fh, indent=2)
        log.info("Results saved to %s", args.out)


def cmd_eval(args: argparse.Namespace) -> None:
    """Run evaluation."""
    from webcam_locator.eval import evaluate_dataset, print_report

    report = evaluate_dataset(
        Path(args.dataset),
        use_gemini=bool(args.use_gemini),
        debug_dir=Path(args.debug_dir) if args.debug_dir else None,
    )
    print_report(report)

    if args.out:
        with open(args.out, "w") as fh:
            json.dump(report, fh, indent=2)
        log.info("Report saved to %s", args.out)


def cmd_extract_frames(args: argparse.Namespace) -> None:
    """Extract frames from a video file using ffmpeg."""
    video = Path(args.video)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not video.exists():
        log.error("Video not found: %s", video)
        sys.exit(1)

    # Determine timestamps
    timestamps: list[float] = []
    if args.times:
        timestamps = [float(t.strip()) for t in args.times.split(",")]
    elif args.every_sec:
        duration = _get_duration(video)
        if duration is None:
            log.error("Cannot determine video duration")
            sys.exit(1)
        t = args.every_sec
        while t < duration:
            timestamps.append(t)
            t += args.every_sec
            if args.max_frames and len(timestamps) >= args.max_frames:
                break
    else:
        log.error("Provide --times or --every-sec")
        sys.exit(1)

    log.info("Extracting %d frames from %s", len(timestamps), video)
    for idx, ts in enumerate(timestamps, start=1):
        out_path = out_dir / f"{idx:04d}.jpg"
        cmd = [
            "ffmpeg", "-y", "-ss", f"{ts:.3f}", "-i", str(video),
            "-frames:v", "1", "-q:v", "2", str(out_path),
        ]
        try:
            subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            if out_path.exists():
                log.info("  %s @ %.1fs", out_path.name, ts)
        except Exception as exc:
            log.warning("  Failed frame %d at %.1fs: %s", idx, ts, exc)

    log.info("Done. Frames saved to %s", out_dir)


def _load_frames(frames_dir: Path) -> list[np.ndarray]:
    """Load all .jpg frames from a directory."""
    paths = sorted(frames_dir.glob("*.jpg"))
    frames = []
    for p in paths:
        img = cv2.imread(str(p))
        if img is not None:
            frames.append(img)
    return frames


def _get_duration(video_path: Path) -> float | None:
    """Get video duration using ffprobe."""
    try:
        result = subprocess.run(
            ["ffprobe", "-v", "quiet", "-print_format", "json", "-show_format", str(video_path)],
            capture_output=True, text=True, timeout=15,
        )
        info = json.loads(result.stdout)
        return float(info["format"]["duration"])
    except Exception:
        return None


def _print_result(label: str, result: dict) -> None:
    if result.get("found"):
        print(f"  {label}: {result['type']} @ ({result['x']},{result['y']}) "
              f"{result['width']}x{result['height']} conf={result.get('confidence', 0):.2f}")
    else:
        print(f"  {label}: no webcam ({result.get('reason', '')})")


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="webcam_locator",
        description="Webcam overlay detection for Twitch/stream clips",
    )
    sub = parser.add_subparsers(dest="command")

    # detect
    p_detect = sub.add_parser("detect", help="Run webcam detection")
    p_detect.add_argument("--images", nargs="+", type=Path, help="Single image paths")
    p_detect.add_argument("--frames-dir", type=Path, help="Directory of frames (one case)")
    p_detect.add_argument("--input", type=Path, help="Dataset root (multiple cases)")
    p_detect.add_argument("--out", type=Path, help="Output JSON path")
    p_detect.add_argument("--debug-dir", type=Path, help="Debug artifacts directory")
    p_detect.add_argument("--use-gemini", type=int, default=0)

    # eval
    p_eval = sub.add_parser("eval", help="Evaluate against labeled dataset")
    p_eval.add_argument("--dataset", required=True, type=Path)
    p_eval.add_argument("--out", type=Path)
    p_eval.add_argument("--debug-dir", type=Path)
    p_eval.add_argument("--use-gemini", type=int, default=0)

    # extract-frames
    p_extract = sub.add_parser("extract-frames", help="Extract frames from video")
    p_extract.add_argument("--video", required=True, type=Path)
    p_extract.add_argument("--out", required=True, type=Path)
    p_extract.add_argument("--times", type=str, help="Comma-separated timestamps in seconds")
    p_extract.add_argument("--every-sec", type=float, help="Extract every N seconds")
    p_extract.add_argument("--max-frames", type=int, default=0)

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    if args.command == "detect":
        cmd_detect(args)
    elif args.command == "eval":
        cmd_eval(args)
    elif args.command == "extract-frames":
        cmd_extract_frames(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
