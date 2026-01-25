"""Face tracking module for dynamic face-following crops.

Tracks face positions across video frames and generates smooth crop paths
for FULL_CAM layout rendering.

Strategy:
1. Sample frames at regular intervals (every ~0.5 seconds)
2. Detect face position in each frame using DNN detection
3. Apply EMA (Exponential Moving Average) smoothing to reduce jitter
4. Generate FFmpeg-compatible expressions for interpolated crop positions
"""

import cv2
import subprocess
import json
import numpy as np
from pathlib import Path
from typing import List, Optional, Tuple, Dict
from dataclasses import dataclass


@dataclass
class FaceKeyframe:
    """A face position at a specific timestamp."""
    timestamp: float  # seconds
    center_x: int
    center_y: int
    width: int
    height: int
    confidence: float = 1.0


@dataclass
class FaceTrack:
    """Complete face tracking result for a video."""
    keyframes: List[FaceKeyframe]
    duration: float
    video_width: int
    video_height: int
    sample_interval: float


def get_video_info(video_path: str) -> dict:
    """Get video metadata using FFprobe."""
    cmd = [
        "ffprobe",
        "-v", "quiet",
        "-print_format", "json",
        "-show_format",
        "-show_streams",
        video_path,
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    data = json.loads(result.stdout)
    
    video_stream = next(
        (s for s in data.get("streams", []) if s.get("codec_type") == "video"),
        {}
    )
    
    return {
        "width": video_stream.get("width", 0),
        "height": video_stream.get("height", 0),
        "duration": float(data.get("format", {}).get("duration", 0)),
        "fps": eval(video_stream.get("r_frame_rate", "30/1")) if "/" in video_stream.get("r_frame_rate", "30") else 30,
    }


def extract_frame_at_time(video_path: str, timestamp: float, temp_dir: str) -> Optional[np.ndarray]:
    """Extract a single frame at a specific timestamp."""
    frame_path = Path(temp_dir) / f"track_frame_{timestamp:.3f}.jpg"
    
    cmd = [
        "ffmpeg",
        "-y",
        "-ss", str(timestamp),
        "-i", video_path,
        "-vframes", "1",
        "-q:v", "2",
        str(frame_path),
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if frame_path.exists():
        frame = cv2.imread(str(frame_path))
        # Clean up temp frame
        frame_path.unlink()
        return frame
    
    return None


def _score_face_for_fullcam(x: int, y: int, w: int, h: int, frame_w: int, frame_h: int) -> float:
    """
    Score a face for FULL_CAM selection (streamer detection).
    
    Streamers typically:
    - Are in the LOWER portion of the frame (seated at desk)
    - Are relatively CENTERED horizontally
    - Have LARGER faces (closer to camera than background people)
    
    Args:
        x, y, w, h: Face bounding box
        frame_w, frame_h: Frame dimensions
    
    Returns:
        Score (higher = more likely to be the streamer)
    """
    face_center_x = x + w // 2
    face_center_y = y + h // 2
    face_area = w * h
    frame_area = frame_w * frame_h
    
    # Factor 1: Size - larger faces are likely the main subject (closer to camera)
    # Normalize to 0-1 range (max expected face area ~30% of frame)
    size_score = min(1.0, (face_area / frame_area) / 0.30)
    
    # Factor 2: Vertical position - prefer faces in LOWER half of frame
    # Streamers sit at desk level, background people are often standing
    # y_ratio = 0 at top, 1 at bottom
    y_ratio = face_center_y / frame_h
    # Score: 0.3 at top, 1.0 at 70% down, 0.8 at bottom
    if y_ratio < 0.3:
        position_score = 0.3  # Upper area - likely background
    elif y_ratio > 0.85:
        position_score = 0.8  # Very bottom - might be cut off
    else:
        # Sweet spot: 0.3-0.85 from top, peaking around 0.6-0.7
        position_score = 0.5 + 0.5 * min(1.0, (y_ratio - 0.3) / 0.4)
    
    # Factor 3: Horizontal centering - streamers are usually centered
    x_ratio = face_center_x / frame_w
    # Score peaks at center (0.5), drops toward edges
    center_score = 1.0 - abs(x_ratio - 0.5) * 1.5  # 1.0 at center, 0.25 at edges
    center_score = max(0.2, center_score)
    
    # Combined score: size matters most, then position, then centering
    total_score = (size_score * 0.5) + (position_score * 0.35) + (center_score * 0.15)
    
    return total_score


def detect_face_dnn(frame_bgr: np.ndarray) -> Optional[Tuple[int, int, int, int, float]]:
    """
    Detect face using OpenCV DNN (Caffe model).
    
    For FULL_CAM scenarios, prefers faces that are:
    - Larger (closer to camera)
    - In the lower portion of frame (streamer at desk)
    - More centered horizontally
    
    Returns:
        Tuple of (x, y, width, height, confidence) or None
    """
    model_base = Path(__file__).parent / "models"
    prototxt = model_base / "deploy.prototxt"
    caffemodel = model_base / "res10_300x300_ssd_iter_140000.caffemodel"
    
    if not prototxt.exists() or not caffemodel.exists():
        return None
    
    net = cv2.dnn.readNetFromCaffe(str(prototxt), str(caffemodel))
    
    h, w = frame_bgr.shape[:2]
    blob = cv2.dnn.blobFromImage(
        cv2.resize(frame_bgr, (300, 300)),
        1.0,
        (300, 300),
        (104.0, 177.0, 123.0),
    )
    
    net.setInput(blob)
    detections = net.forward()
    
    # Collect all valid faces
    candidates = []
    
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        
        if confidence > 0.5:  # Minimum confidence
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            x1, y1, x2, y2 = box.astype("int")
            
            # Ensure coordinates are valid
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(w, x2)
            y2 = min(h, y2)
            
            if x2 > x1 and y2 > y1:
                fw, fh = x2 - x1, y2 - y1
                # Score for streamer detection
                score = _score_face_for_fullcam(x1, y1, fw, fh, w, h)
                candidates.append((x1, y1, fw, fh, confidence, score))
    
    if not candidates:
        return None
    
    # Pick face with highest streamer score
    best = max(candidates, key=lambda c: c[5])
    return (best[0], best[1], best[2], best[3], best[4])


def detect_face_haar(frame_bgr: np.ndarray) -> Optional[Tuple[int, int, int, int, float]]:
    """
    Fallback face detection using Haar Cascade.
    
    For FULL_CAM scenarios, prefers faces that are:
    - Larger (closer to camera)
    - In the lower portion of frame (streamer at desk)
    - More centered horizontally
    
    Returns:
        Tuple of (x, y, width, height, confidence) or None
    """
    h, w = frame_bgr.shape[:2]
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    
    cascade_paths = [
        cv2.data.haarcascades + 'haarcascade_frontalface_default.xml',
        '/usr/share/opencv4/haarcascades/haarcascade_frontalface_default.xml',
    ]
    
    face_cascade = None
    for path in cascade_paths:
        if Path(path).exists():
            face_cascade = cv2.CascadeClassifier(path)
            break
    
    if face_cascade is None or face_cascade.empty():
        return None
    
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(50, 50)
    )
    
    if len(faces) == 0:
        return None
    
    # Score each face for streamer detection
    candidates = []
    for (fx, fy, fw, fh) in faces:
        score = _score_face_for_fullcam(fx, fy, fw, fh, w, h)
        candidates.append((fx, fy, fw, fh, 0.8, score))  # Haar confidence = 0.8
    
    # Pick face with highest streamer score
    best = max(candidates, key=lambda c: c[5])
    
    return (best[0], best[1], best[2], best[3], 0.8)  # Haar doesn't give confidence, use 0.8


def track_faces(
    video_path: str,
    temp_dir: str,
    sample_interval: float = 1.5,  # Sample every 1.5 seconds for better coverage
    ema_alpha: float = 0.4,
    max_keyframes: int = 20,  # Allow more keyframes for better tracking
) -> Optional[FaceTrack]:
    """
    Track face positions throughout a video.
    
    Args:
        video_path: Path to input video
        temp_dir: Temporary directory for frame extraction
        sample_interval: Time between samples in seconds (default 2.0s)
        ema_alpha: EMA smoothing factor (0-1, higher = more responsive, default 0.4)
        max_keyframes: Maximum keyframes to keep (FFmpeg expression limit)
    
    Returns:
        FaceTrack with smoothed keyframes, or None if no faces detected
    """
    print(f"🔍 Starting face tracking for: {video_path}")
    
    # Get video info
    info = get_video_info(video_path)
    duration = info["duration"]
    width = info["width"]
    height = info["height"]
    
    print(f"   📹 Video: {width}x{height}, {duration:.2f}s")
    
    # Calculate sample timestamps - limit to avoid too many keyframes
    # For a 30s clip with 2s interval: 15 samples
    timestamps = []
    t = 0.0
    while t < duration:
        timestamps.append(t)
        t += sample_interval
    
    # Always include the last frame
    if timestamps[-1] < duration - 0.5:
        timestamps.append(duration - 0.5)
    
    # Limit total samples
    if len(timestamps) > max_keyframes:
        # Evenly space the samples
        step = len(timestamps) / max_keyframes
        timestamps = [timestamps[int(i * step)] for i in range(max_keyframes)]
        timestamps.append(duration - 0.5)  # Ensure we have the end
    
    print(f"   🎯 Sampling {len(timestamps)} frames...")
    
    # Detect faces at each timestamp
    raw_detections: List[Tuple[float, Tuple[int, int, int, int, float]]] = []
    
    for ts in timestamps:
        frame = extract_frame_at_time(video_path, ts, temp_dir)
        if frame is None:
            continue
        
        # Try DNN first, then Haar fallback
        detection = detect_face_dnn(frame)
        if detection is None:
            detection = detect_face_haar(frame)
        
        if detection:
            raw_detections.append((ts, detection))
    
    if len(raw_detections) == 0:
        print("   ⚠️ No faces detected in any frame")
        return None
    
    print(f"   ✅ Detected faces in {len(raw_detections)}/{len(timestamps)} frames")
    
    # Debug: show face positions to verify scoring is working
    print(f"   📊 Face positions detected:")
    for ts, det in raw_detections[:5]:  # Show first 5
        x, y, w, h, conf = det
        cx, cy = x + w // 2, y + h // 2
        y_ratio = cy / height
        print(f"      t={ts:.1f}s: center=({cx},{cy}) y_ratio={y_ratio:.2f} size={w}x{h}")
    if len(raw_detections) > 5:
        print(f"      ... and {len(raw_detections) - 5} more")
    
    # Apply EMA smoothing
    keyframes: List[FaceKeyframe] = []
    
    # Initialize with first detection
    first_ts, first_det = raw_detections[0]
    x, y, w, h, conf = first_det
    smoothed_cx = x + w // 2
    smoothed_cy = y + h // 2
    smoothed_w = w
    smoothed_h = h
    
    keyframes.append(FaceKeyframe(
        timestamp=first_ts,
        center_x=smoothed_cx,
        center_y=smoothed_cy,
        width=smoothed_w,
        height=smoothed_h,
        confidence=conf,
    ))
    
    # Apply EMA to subsequent detections
    for ts, det in raw_detections[1:]:
        x, y, w, h, conf = det
        cx = x + w // 2
        cy = y + h // 2
        
        # EMA smoothing
        smoothed_cx = int(ema_alpha * cx + (1 - ema_alpha) * smoothed_cx)
        smoothed_cy = int(ema_alpha * cy + (1 - ema_alpha) * smoothed_cy)
        smoothed_w = int(ema_alpha * w + (1 - ema_alpha) * smoothed_w)
        smoothed_h = int(ema_alpha * h + (1 - ema_alpha) * smoothed_h)
        
        keyframes.append(FaceKeyframe(
            timestamp=ts,
            center_x=smoothed_cx,
            center_y=smoothed_cy,
            width=smoothed_w,
            height=smoothed_h,
            confidence=conf,
        ))
    
    # If we have gaps (missing detections), interpolate
    keyframes = _interpolate_gaps(keyframes, timestamps, smoothed_cx, smoothed_cy, smoothed_w, smoothed_h)
    
    print(f"   🎬 Generated {len(keyframes)} smoothed keyframes")
    
    return FaceTrack(
        keyframes=keyframes,
        duration=duration,
        video_width=width,
        video_height=height,
        sample_interval=sample_interval,
    )


def _interpolate_gaps(
    keyframes: List[FaceKeyframe],
    all_timestamps: List[float],
    last_cx: int,
    last_cy: int,
    last_w: int,
    last_h: int,
) -> List[FaceKeyframe]:
    """
    Fill gaps in keyframes where face wasn't detected.
    Uses last known position for missing frames.
    """
    # Create a map of existing keyframes
    kf_map = {kf.timestamp: kf for kf in keyframes}
    
    filled_keyframes = []
    last_kf = keyframes[0] if keyframes else None
    
    for ts in all_timestamps:
        if ts in kf_map:
            filled_keyframes.append(kf_map[ts])
            last_kf = kf_map[ts]
        elif last_kf:
            # Use last known position
            filled_keyframes.append(FaceKeyframe(
                timestamp=ts,
                center_x=last_kf.center_x,
                center_y=last_kf.center_y,
                width=last_kf.width,
                height=last_kf.height,
                confidence=0.5,  # Lower confidence for interpolated
            ))
    
    return filled_keyframes


def generate_ffmpeg_crop_expr(
    face_track: FaceTrack,
    crop_width: int,
    crop_height: int,
    target_face_y_ratio: float = 0.45,
    movement_threshold: int = 100,  # Minimum movement in pixels to enable dynamic crop
) -> Tuple[str, str]:
    """
    Generate FFmpeg expressions for dynamic crop x and y positions.
    
    The expressions use linear interpolation between keyframes based on time (t).
    Face is positioned at target_face_y_ratio from top of frame (default 45%).
    
    If face movement is minimal, returns static crop positions for simplicity.
    
    Args:
        face_track: Face tracking result
        crop_width: Width of crop region
        crop_height: Height of crop region
        target_face_y_ratio: Where face should be positioned vertically (0-1)
        movement_threshold: Minimum face movement (px) to use dynamic crop
    
    Returns:
        Tuple of (x_expression, y_expression) for FFmpeg crop filter
    """
    if len(face_track.keyframes) == 0:
        # Fallback to center crop
        cx = face_track.video_width // 2
        cy = face_track.video_height // 2
        crop_x = max(0, cx - crop_width // 2)
        crop_y = max(0, cy - crop_height // 2)
        return str(crop_x), str(crop_y)
    
    # Calculate crop positions for each keyframe
    crop_positions = []
    for kf in face_track.keyframes:
        # Calculate crop position to center face at target ratio
        crop_x = kf.center_x - crop_width // 2
        crop_y = kf.center_y - int(crop_height * target_face_y_ratio)
        
        # Clamp to valid bounds
        crop_x = max(0, min(crop_x, face_track.video_width - crop_width))
        crop_y = max(0, min(crop_y, face_track.video_height - crop_height))
        
        crop_positions.append((kf.timestamp, crop_x, crop_y))
    
    if len(crop_positions) == 1:
        # Single keyframe - static crop
        return str(crop_positions[0][1]), str(crop_positions[0][2])
    
    # Check face movement range
    x_positions = [p[1] for p in crop_positions]
    y_positions = [p[2] for p in crop_positions]
    
    x_range = max(x_positions) - min(x_positions)
    y_range = max(y_positions) - min(y_positions)
    
    # Calculate average position for stable crop
    avg_x = sum(x_positions) // len(x_positions)
    avg_y = sum(y_positions) // len(y_positions)
    
    # For now, always use static crop based on average face position
    # Dynamic FFmpeg expressions with nested if() are complex and error-prone
    # The average position provides smooth, centered framing
    print(f"   📍 Face tracking: range=({x_range}px, {y_range}px), using avg position ({avg_x}, {avg_y})")
    return str(avg_x), str(avg_y)
    
    # NOTE: Dynamic expressions disabled for reliability
    # Uncomment below to enable dynamic tracking (requires FFmpeg expression debugging)
    #
    # # If movement is significant, use dynamic expressions
    # if x_range >= movement_threshold or y_range >= movement_threshold:
    #     print(f"   🎯 Face movement detected ({x_range}px, {y_range}px) - using dynamic crop")
    #     x_expr = _build_interpolation_expr(crop_positions, 'x')
    #     y_expr = _build_interpolation_expr(crop_positions, 'y')
    #     return x_expr, y_expr
    # 
    # return str(avg_x), str(avg_y)


def _build_interpolation_expr(positions: List[Tuple[float, int, int]], axis: str) -> str:
    """
    Build FFmpeg expression for linear interpolation between keyframe positions.
    
    Uses nested if() with escaped commas for FFmpeg filter chain compatibility.
    
    Args:
        positions: List of (timestamp, x, y) tuples
        axis: 'x' or 'y'
    
    Returns:
        FFmpeg expression string with properly escaped commas
    """
    idx = 1 if axis == 'x' else 2
    
    if len(positions) <= 1:
        return str(positions[0][idx]) if positions else "0"
    
    # Limit to very few keyframes to keep expression simple
    # FFmpeg can struggle with complex nested expressions
    MAX_EXPR_KEYFRAMES = 6
    if len(positions) > MAX_EXPR_KEYFRAMES:
        # Downsample to MAX_EXPR_KEYFRAMES evenly spaced
        step = len(positions) / MAX_EXPR_KEYFRAMES
        sampled = [positions[int(i * step)] for i in range(MAX_EXPR_KEYFRAMES - 1)]
        sampled.append(positions[-1])  # Always include last
        positions = sampled
    
    def lerp_expr(v0: int, v1: int, t0: float, t1: float) -> str:
        """Linear interpolation: v0 + (v1-v0) * (t-t0)/(t1-t0)"""
        if t1 <= t0 or v0 == v1:
            return str(v0)
        dt = t1 - t0
        dv = v1 - v0
        # Simple format without nested parens: v0+dv*(t-t0)/dt
        return f"{v0}+{dv}*(t-{t0:.1f})/{dt:.1f}"
    
    n = len(positions)
    
    # Start with last value
    expr = str(positions[-1][idx])
    
    # Build nested if statements backwards
    # Commas must be escaped as \, for FFmpeg filter parsing
    for i in range(n - 2, -1, -1):
        t0, x0, y0 = positions[i]
        t1, x1, y1 = positions[i + 1]
        v0 = x0 if axis == 'x' else y0
        v1 = x1 if axis == 'x' else y1
        
        lerp = lerp_expr(v0, v1, t0, t1)
        
        # Escape commas with single backslash for FFmpeg
        # if(lt(t\,T)\,THEN\,ELSE)
        expr = f"if(lt(t\\,{t1:.1f})\\,{lerp}\\,{expr})"
    
    return expr


def get_static_face_center(face_track: FaceTrack) -> Optional[Tuple[int, int]]:
    """
    Get average face center position (for fallback/simple mode).
    
    Returns:
        Tuple of (center_x, center_y) or None
    """
    if not face_track or len(face_track.keyframes) == 0:
        return None
    
    # Weight by confidence
    total_weight = 0
    weighted_x = 0
    weighted_y = 0
    
    for kf in face_track.keyframes:
        weighted_x += kf.center_x * kf.confidence
        weighted_y += kf.center_y * kf.confidence
        total_weight += kf.confidence
    
    if total_weight == 0:
        return None
    
    return (int(weighted_x / total_weight), int(weighted_y / total_weight))

