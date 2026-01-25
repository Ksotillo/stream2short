"""Face tracking module for dynamic face-following crops.

Tracks face positions across video frames and generates smooth crop paths
for FULL_CAM layout rendering.

Strategy:
1. Sample frames at regular intervals (every ~1.5 seconds)
2. Detect face position using DNN (preferred) or Haar Cascade (fallback)
3. Apply EMA (Exponential Moving Average) smoothing to reduce jitter
4. Generate FFmpeg-compatible crop positions

DNN Model Setup:
- Models are downloaded during Docker build to /app/models/
- deploy.prototxt and res10_300x300_ssd_iter_140000.caffemodel
"""

import cv2
import os
import subprocess
import json
import numpy as np
from pathlib import Path
from typing import List, Optional, Tuple, Dict
from dataclasses import dataclass


# =============================================================================
# GLOBAL CACHED DNN NET (singleton pattern for performance)
# =============================================================================
_DNN_NET = None
_DNN_NET_LOADED = False  # Track if we've attempted loading

DEBUG_FACE_TRACKING = os.environ.get("DEBUG_FACE_TRACKING", "1") == "1"


def _get_dnn_net():
    """
    Load DNN face detection model once and cache it.
    
    Returns:
        cv2.dnn.Net or None if models are missing
    """
    global _DNN_NET, _DNN_NET_LOADED
    
    if _DNN_NET_LOADED:
        return _DNN_NET
    
    _DNN_NET_LOADED = True
    
    model_base = Path(__file__).parent / "models"
    prototxt = model_base / "deploy.prototxt"
    caffemodel = model_base / "res10_300x300_ssd_iter_140000.caffemodel"
    
    if not prototxt.exists() or not caffemodel.exists():
        print(f"   ⚠️ DNN models not found at {model_base}")
        print(f"      Prototxt exists: {prototxt.exists()}")
        print(f"      Caffemodel exists: {caffemodel.exists()}")
        return None
    
    try:
        net = cv2.dnn.readNetFromCaffe(str(prototxt), str(caffemodel))
        # Set optimal backend/target for CPU
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        _DNN_NET = net
        print("   ✅ DNN face detector loaded successfully")
        return _DNN_NET
    except Exception as e:
        print(f"   ❌ Failed to load DNN model: {e}")
        return None


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


def _preprocess_frame_for_detection(frame_bgr: np.ndarray) -> np.ndarray:
    """
    Preprocess frame to improve face detection in challenging lighting.
    
    Applies CLAHE (Contrast Limited Adaptive Histogram Equalization)
    to enhance contrast, especially helpful for blue/dim lighting.
    """
    # Convert to LAB color space
    lab = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2LAB)
    
    # Split channels
    l, a, b = cv2.split(lab)
    
    # Apply CLAHE to L channel (luminance)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_enhanced = clahe.apply(l)
    
    # Merge back
    lab_enhanced = cv2.merge([l_enhanced, a, b])
    
    # Convert back to BGR
    return cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)


def _score_face_for_fullcam(x: int, y: int, w: int, h: int, frame_w: int, frame_h: int) -> float:
    """
    Score a face for FULL_CAM selection (streamer detection).
    
    CRITICAL: This scoring is designed to REJECT background people and SELECT the streamer.
    
    Streamers typically:
    - Are in the LOWER portion of the frame (seated at desk, y_ratio > 0.50)
    - Are on the LEFT or CENTER-LEFT (typical desk setup, x_ratio 0.20-0.50)
    - Have LARGER faces (closer to camera than background people)
    
    Background people typically:
    - Are in the UPPER portion (standing, y_ratio < 0.40)
    - Are on the RIGHT side (walking behind, x_ratio > 0.60)
    - Have SMALLER faces (farther from camera)
    
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
    
    # =========================================================================
    # Factor 1: VERTICAL POSITION (most important)
    # Streamers sit at desk level = LOWER portion of frame
    # =========================================================================
    y_ratio = face_center_y / frame_h
    
    if y_ratio < 0.35:
        # Upper area - VERY LIKELY background person standing
        position_score = 0.10
    elif y_ratio < 0.45:
        # Upper-middle - probably background
        position_score = 0.25
    elif y_ratio < 0.55:
        # Middle - could be either
        position_score = 0.50
    elif y_ratio < 0.70:
        # Lower-middle - likely streamer seated
        position_score = 0.85
    else:
        # Lower area - streamer at desk level
        position_score = 0.95
    
    # =========================================================================
    # Factor 2: HORIZONTAL POSITION
    # Streamers are typically LEFT or CENTER-LEFT (desk setup)
    # Background people often walk behind on the RIGHT
    # =========================================================================
    x_ratio = face_center_x / frame_w
    
    if x_ratio < 0.20:
        # Far left - probably streamer
        horizontal_score = 0.80
    elif x_ratio < 0.40:
        # Left-center - sweet spot for streamer
        horizontal_score = 0.95
    elif x_ratio < 0.55:
        # Center - good for streamer
        horizontal_score = 0.85
    elif x_ratio < 0.70:
        # Right-center - could be either
        horizontal_score = 0.50
    else:
        # Far right - likely background person
        horizontal_score = 0.20
    
    # =========================================================================
    # Factor 3: SIZE
    # Streamers have larger faces (closer to camera)
    # But size should NOT dominate position!
    # =========================================================================
    size_ratio = face_area / frame_area
    
    if size_ratio > 0.10:
        size_score = 1.0  # Very large face
    elif size_ratio > 0.05:
        size_score = 0.85
    elif size_ratio > 0.02:
        size_score = 0.70
    elif size_ratio > 0.01:
        size_score = 0.50
    else:
        size_score = 0.30  # Small face (far away)
    
    # =========================================================================
    # COMBINED SCORE
    # Position is MOST important, then horizontal, then size
    # =========================================================================
    total_score = (position_score * 0.45) + (horizontal_score * 0.35) + (size_score * 0.20)
    
    return total_score


def detect_face_dnn(
    frame_bgr: np.ndarray,
    confidence_threshold: float = 0.35,  # Lower threshold for better detection
    try_preprocessing: bool = True,
) -> Optional[Tuple[int, int, int, int, float]]:
    """
    Detect face using OpenCV DNN (ResNet SSD Caffe model).
    
    This is more robust than Haar Cascade, especially for:
    - Profile/angled faces
    - Variable lighting conditions
    - Faces with glasses/accessories
    
    Args:
        frame_bgr: Input frame in BGR format
        confidence_threshold: Minimum confidence (default 0.35 for better recall)
        try_preprocessing: Whether to try enhanced preprocessing if first pass fails
    
    Returns:
        Tuple of (x, y, width, height, confidence) or None
    """
    net = _get_dnn_net()
    if net is None:
        return None
    
    h, w = frame_bgr.shape[:2]
    
    def _run_detection(input_frame):
        """Run DNN detection on a frame."""
        blob = cv2.dnn.blobFromImage(
            cv2.resize(input_frame, (300, 300)),
            1.0,
            (300, 300),
            (104.0, 177.0, 123.0),
        )
        
        net.setInput(blob)
        detections = net.forward()
        
        candidates = []
        
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            
            if confidence > confidence_threshold:
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
                    y_ratio = (y1 + fh // 2) / h
                    x_ratio = (x1 + fw // 2) / w
                    candidates.append({
                        'bbox': (x1, y1, fw, fh),
                        'confidence': float(confidence),
                        'score': score,
                        'y_ratio': y_ratio,
                        'x_ratio': x_ratio,
                    })
        
        return candidates
    
    # First pass: original frame
    candidates = _run_detection(frame_bgr)
    
    # Second pass: try preprocessing if no good candidates found
    if try_preprocessing and (len(candidates) == 0 or max(c['score'] for c in candidates) < 0.50):
        enhanced = _preprocess_frame_for_detection(frame_bgr)
        enhanced_candidates = _run_detection(enhanced)
        
        # Merge candidates, keeping best score for each rough position
        for ec in enhanced_candidates:
            is_new = True
            for c in candidates:
                # Check if same face (close position)
                if abs(ec['bbox'][0] - c['bbox'][0]) < 50 and abs(ec['bbox'][1] - c['bbox'][1]) < 50:
                    if ec['score'] > c['score']:
                        c.update(ec)
                    is_new = False
                    break
            if is_new:
                candidates.append(ec)
    
    if not candidates:
        return None
    
    # Debug logging
    if DEBUG_FACE_TRACKING:
        candidates_sorted = sorted(candidates, key=lambda c: c['score'], reverse=True)
        print(f"   🔍 DNN candidates ({len(candidates_sorted)}):")
        for i, c in enumerate(candidates_sorted[:3]):
            x, y, fw, fh = c['bbox']
            print(f"      #{i+1}: ({x},{y}) {fw}x{fh} conf={c['confidence']:.2f} score={c['score']:.3f} y={c['y_ratio']:.2f} x={c['x_ratio']:.2f}")
    
    # Pick face with highest streamer score
    best = max(candidates, key=lambda c: c['score'])
    x, y, fw, fh = best['bbox']
    
    return (x, y, fw, fh, best['confidence'])


def detect_face_haar(frame_bgr: np.ndarray) -> Optional[Tuple[int, int, int, int, float]]:
    """
    Fallback face detection using Haar Cascade.
    
    Less robust than DNN but works without model files.
    
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
        y_ratio = (fy + fh // 2) / h
        x_ratio = (fx + fw // 2) / w
        candidates.append({
            'bbox': (fx, fy, fw, fh),
            'confidence': 0.8,
            'score': score,
            'y_ratio': y_ratio,
            'x_ratio': x_ratio,
        })
    
    # Debug logging
    if DEBUG_FACE_TRACKING:
        candidates_sorted = sorted(candidates, key=lambda c: c['score'], reverse=True)
        print(f"   🔍 Haar candidates ({len(candidates_sorted)}):")
        for i, c in enumerate(candidates_sorted[:3]):
            x, y, fw, fh = c['bbox']
            print(f"      #{i+1}: ({x},{y}) {fw}x{fh} score={c['score']:.3f} y={c['y_ratio']:.2f} x={c['x_ratio']:.2f}")
    
    # Pick face with highest streamer score
    best = max(candidates, key=lambda c: c['score'])
    x, y, fw, fh = best['bbox']
    
    return (x, y, fw, fh, 0.8)


def track_faces(
    video_path: str,
    temp_dir: str,
    sample_interval: float = 1.5,
    ema_alpha: float = 0.4,
    max_keyframes: int = 20,
) -> Optional[FaceTrack]:
    """
    Track face positions throughout a video.
    
    Args:
        video_path: Path to input video
        temp_dir: Temporary directory for frame extraction
        sample_interval: Time between samples in seconds
        ema_alpha: EMA smoothing factor (0-1, higher = more responsive)
        max_keyframes: Maximum keyframes to keep
    
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
    
    # Calculate sample timestamps
    timestamps = []
    t = 0.0
    while t < duration:
        timestamps.append(t)
        t += sample_interval
    
    # Always include near the end
    if timestamps[-1] < duration - 0.5:
        timestamps.append(duration - 0.5)
    
    # Limit total samples
    if len(timestamps) > max_keyframes:
        step = len(timestamps) / max_keyframes
        timestamps = [timestamps[int(i * step)] for i in range(max_keyframes)]
        timestamps.append(duration - 0.5)
    
    print(f"   🎯 Sampling {len(timestamps)} frames...")
    
    # Detect faces at each timestamp
    raw_detections: List[Tuple[float, Tuple[int, int, int, int, float]]] = []
    
    for ts in timestamps:
        frame = extract_frame_at_time(video_path, ts, temp_dir)
        if frame is None:
            continue
        
        # Try DNN first (more robust), then Haar fallback
        detection = detect_face_dnn(frame)
        if detection is None:
            detection = detect_face_haar(frame)
        
        if detection:
            raw_detections.append((ts, detection))
    
    if len(raw_detections) == 0:
        print("   ⚠️ No faces detected in any frame")
        return None
    
    print(f"   ✅ Detected faces in {len(raw_detections)}/{len(timestamps)} frames")
    
    # Debug: show face positions
    print(f"   📊 Face positions detected:")
    for ts, det in raw_detections[:5]:
        x, y, w, h, conf = det
        cx, cy = x + w // 2, y + h // 2
        y_ratio = cy / height
        x_ratio = cx / width
        print(f"      t={ts:.1f}s: center=({cx},{cy}) y={y_ratio:.2f} x={x_ratio:.2f} size={w}x{h}")
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
    
    # Interpolate gaps
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
    """Fill gaps in keyframes where face wasn't detected."""
    kf_map = {kf.timestamp: kf for kf in keyframes}
    
    filled_keyframes = []
    last_kf = keyframes[0] if keyframes else None
    
    for ts in all_timestamps:
        if ts in kf_map:
            filled_keyframes.append(kf_map[ts])
            last_kf = kf_map[ts]
        elif last_kf:
            filled_keyframes.append(FaceKeyframe(
                timestamp=ts,
                center_x=last_kf.center_x,
                center_y=last_kf.center_y,
                width=last_kf.width,
                height=last_kf.height,
                confidence=0.5,
            ))
    
    return filled_keyframes


def generate_ffmpeg_crop_expr(
    face_track: FaceTrack,
    crop_width: int,
    crop_height: int,
    target_face_y_ratio: float = 0.45,
    movement_threshold: int = 100,
) -> Tuple[str, str]:
    """
    Generate FFmpeg crop x and y positions.
    
    Returns static positions based on average face position.
    Includes heuristics to detect and reject background people.
    
    Args:
        face_track: Face tracking result
        crop_width: Width of crop region
        crop_height: Height of crop region
        target_face_y_ratio: Where face should be positioned vertically (0-1)
        movement_threshold: Unused (kept for API compatibility)
    
    Returns:
        Tuple of (x_position, y_position) as strings for FFmpeg
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
        crop_x = kf.center_x - crop_width // 2
        crop_y = kf.center_y - int(crop_height * target_face_y_ratio)
        
        # Clamp to valid bounds
        crop_x = max(0, min(crop_x, face_track.video_width - crop_width))
        crop_y = max(0, min(crop_y, face_track.video_height - crop_height))
        
        crop_positions.append((kf.timestamp, crop_x, crop_y))
    
    if len(crop_positions) == 1:
        return str(crop_positions[0][1]), str(crop_positions[0][2])
    
    # Calculate averages
    x_positions = [p[1] for p in crop_positions]
    y_positions = [p[2] for p in crop_positions]
    
    avg_x = sum(x_positions) // len(x_positions)
    avg_y = sum(y_positions) // len(y_positions)
    
    # =========================================================================
    # BACKGROUND DETECTION: Check if detected faces are background people
    # =========================================================================
    avg_face_y = sum(kf.center_y for kf in face_track.keyframes) / len(face_track.keyframes)
    avg_face_x = sum(kf.center_x for kf in face_track.keyframes) / len(face_track.keyframes)
    
    avg_face_y_ratio = avg_face_y / face_track.video_height
    avg_face_x_ratio = avg_face_x / face_track.video_width
    
    # Background indicators:
    # - Faces in upper portion (y_ratio < 0.45) = standing person
    # - Faces on far right (x_ratio > 0.55) = walking behind streamer
    faces_in_upper = avg_face_y_ratio < 0.45
    faces_on_far_right = avg_face_x_ratio > 0.55
    
    if faces_in_upper or faces_on_far_right:
        # Detected faces are likely background people, NOT the streamer!
        # Use TRUE LEFT BIAS: streamer is typically on the LEFT side at their desk
        print(f"   ⚠️ Background faces detected: y_ratio={avg_face_y_ratio:.2f}, x_ratio={avg_face_x_ratio:.2f}")
        print(f"   🎯 Using LEFT-BIASED crop (typical streamer position)")
        
        # TRUE LEFT BIAS: Position crop at 10% from left edge
        # This ensures the streamer on the left side is captured
        default_x = int((face_track.video_width - crop_width) * 0.10)
        default_y = max(0, (face_track.video_height - crop_height) // 2)
        
        # Clamp to valid bounds
        default_x = max(0, min(default_x, face_track.video_width - crop_width))
        default_y = max(0, min(default_y, face_track.video_height - crop_height))
        
        print(f"   📍 Left-biased position: ({default_x}, {default_y})")
        return str(default_x), str(default_y)
    
    # Detected face appears to be the streamer - use tracked position
    x_range = max(x_positions) - min(x_positions)
    y_range = max(y_positions) - min(y_positions)
    
    print(f"   📍 Face tracking: range=({x_range}px, {y_range}px), avg position ({avg_x}, {avg_y})")
    return str(avg_x), str(avg_y)


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
