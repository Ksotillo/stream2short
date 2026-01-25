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
        if _DNN_NET is not None:
            print(f"   🔄 DNN net already loaded (cached)")
        return _DNN_NET
    
    _DNN_NET_LOADED = True
    
    # HARD DIAGNOSTIC: Log exact paths being checked
    model_base = Path(__file__).parent / "models"
    prototxt = model_base / "deploy.prototxt"
    caffemodel = model_base / "res10_300x300_ssd_iter_140000.caffemodel"
    
    print(f"   🔍 DNN MODEL PATHS:")
    print(f"      __file__: {__file__}")
    print(f"      model_base: {model_base}")
    print(f"      model_base.resolve(): {model_base.resolve()}")
    print(f"      prototxt: {prototxt}")
    print(f"      prototxt.exists(): {prototxt.exists()}")
    print(f"      caffemodel: {caffemodel}")
    print(f"      caffemodel.exists(): {caffemodel.exists()}")
    
    if not prototxt.exists() or not caffemodel.exists():
        print(f"   ❌ DNN MODELS NOT FOUND!")
        # Try to list what IS in the models directory
        if model_base.exists():
            print(f"      Contents of {model_base}:")
            for f in model_base.iterdir():
                print(f"         - {f.name}")
        else:
            print(f"      Directory {model_base} does NOT exist!")
        return None
    
    try:
        print(f"   🚀 Loading DNN model with cv2.dnn.readNetFromCaffe...")
        net = cv2.dnn.readNetFromCaffe(str(prototxt), str(caffemodel))
        # Set optimal backend/target for CPU
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        _DNN_NET = net
        print("   ✅ DNN face detector loaded successfully!")
        return _DNN_NET
    except Exception as e:
        import traceback
        print(f"   ❌ EXCEPTION loading DNN model:")
        print(f"      Exception type: {type(e).__name__}")
        print(f"      Exception message: {e}")
        print(f"      Traceback: {traceback.format_exc()}")
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
    confidence_threshold: float = 0.50,  # Raised from 0.35 to reduce false positives
    try_preprocessing: bool = True,
    prev_center: Optional[Tuple[int, int]] = None,  # For temporal association
    max_jump_ratio: float = 0.18,  # Max allowed jump as fraction of frame width
) -> Optional[Tuple[int, int, int, int, float]]:
    """
    Detect face using OpenCV DNN (ResNet SSD Caffe model).
    
    This is more robust than Haar Cascade, especially for:
    - Profile/angled faces
    - Variable lighting conditions
    - Faces with glasses/accessories
    
    Args:
        frame_bgr: Input frame in BGR format
        confidence_threshold: Minimum confidence (default 0.50)
        try_preprocessing: Whether to try enhanced preprocessing if first pass fails
        prev_center: Previous face center (cx, cy) for temporal association
        max_jump_ratio: Max allowed jump as fraction of frame width
    
    Returns:
        Tuple of (x, y, width, height, confidence) or None
    """
    print(f"   🔍 detect_face_dnn() ENTER - conf_thresh={confidence_threshold}, prev_center={prev_center}")
    
    net = _get_dnn_net()
    if net is None:
        print(f"   ❌ detect_face_dnn(): DNN net is None, cannot proceed")
        return None
    
    print(f"   ✅ detect_face_dnn(): DNN net loaded, running detection...")
    
    h, w = frame_bgr.shape[:2]
    frame_area = w * h
    max_jump_px = int(w * max_jump_ratio)
    
    def _run_detection(input_frame):
        """Run DNN detection on a frame with sanity filters."""
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
            
            if confidence >= confidence_threshold:  # >= for clarity
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                x1, y1, x2, y2 = box.astype("int")
                
                # Ensure coordinates are valid
                x1 = max(0, x1)
                y1 = max(0, y1)
                x2 = min(w, x2)
                y2 = min(h, y2)
                
                if x2 > x1 and y2 > y1:
                    fw, fh = x2 - x1, y2 - y1
                    face_area = fw * fh
                    area_ratio = face_area / frame_area
                    aspect_ratio = fw / fh if fh > 0 else 0
                    
                    # ==========================================================
                    # SANITY FILTERS (Task C)
                    # ==========================================================
                    # Reject tiny or huge detections
                    if area_ratio < 0.002 or area_ratio > 0.08:
                        print(f"      [REJECT] area_ratio={area_ratio:.4f} out of [0.002, 0.08]")
                        continue
                    
                    # Reject non-face aspect ratios
                    if aspect_ratio < 0.6 or aspect_ratio > 1.6:
                        print(f"      [REJECT] aspect_ratio={aspect_ratio:.2f} out of [0.6, 1.6]")
                        continue
                    
                    # Score for streamer detection
                    score = _score_face_for_fullcam(x1, y1, fw, fh, w, h)
                    y_ratio = (y1 + fh // 2) / h
                    x_ratio = (x1 + fw // 2) / w
                    cx = x1 + fw // 2
                    cy = y1 + fh // 2
                    
                    candidates.append({
                        'bbox': (x1, y1, fw, fh),
                        'confidence': float(confidence),
                        'score': score,
                        'y_ratio': y_ratio,
                        'x_ratio': x_ratio,
                        'center': (cx, cy),
                    })
        
        return candidates
    
    # First pass: original frame
    candidates = _run_detection(frame_bgr)
    
    # Second pass: try preprocessing if no good candidates found
    if try_preprocessing and (len(candidates) == 0 or max((c['score'] for c in candidates), default=0) < 0.50):
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
        print(f"   ❌ detect_face_dnn(): No valid candidates after filtering")
        return None
    
    # ALWAYS log candidates for diagnostics
    candidates_sorted = sorted(candidates, key=lambda c: c['score'], reverse=True)
    print(f"   📊 detect_face_dnn(): Found {len(candidates_sorted)} valid candidates:")
    for i, c in enumerate(candidates_sorted[:3]):
        x, y, fw, fh = c['bbox']
        print(f"      #{i+1}: ({x},{y}) {fw}x{fh} conf={c['confidence']:.2f} score={c['score']:.3f} y={c['y_ratio']:.2f} x={c['x_ratio']:.2f}")
    if len(candidates_sorted) > 3:
        print(f"      ... and {len(candidates_sorted) - 3} more")
    
    # ==========================================================================
    # TEMPORAL ASSOCIATION (Task B)
    # If we have a previous center, prefer the candidate closest to it
    # ==========================================================================
    if prev_center is not None:
        prev_cx, prev_cy = prev_center
        
        # Find candidate closest to prev_center
        def distance_to_prev(c):
            cx, cy = c['center']
            return ((cx - prev_cx) ** 2 + (cy - prev_cy) ** 2) ** 0.5
        
        candidates_by_dist = sorted(candidates, key=distance_to_prev)
        closest = candidates_by_dist[0]
        dist = distance_to_prev(closest)
        
        if dist <= max_jump_px:
            # Accept closest candidate
            x, y, fw, fh = closest['bbox']
            print(f"   ✅ detect_face_dnn(): Temporal match at ({x},{y}) dist={dist:.0f}px")
            return (x, y, fw, fh, closest['confidence'])
        else:
            # Jump too large - reject all candidates
            print(f"   ⚠️ detect_face_dnn(): Jump {dist:.0f}px > max {max_jump_px}px - holding position")
            return None
    
    # No previous center - pick face with highest streamer score
    best = max(candidates, key=lambda c: c['score'])
    x, y, fw, fh = best['bbox']
    
    print(f"   ✅ detect_face_dnn(): Returning best face at ({x},{y}) {fw}x{fh} conf={best['confidence']:.2f}")
    
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
    
    # ==========================================================================
    # TEMPORAL ASSOCIATION (Task B)
    # Maintain prev_center to avoid jumping between different faces
    # ==========================================================================
    prev_center: Optional[Tuple[int, int]] = None  # (cx, cy)
    anchor_center: Optional[Tuple[int, int]] = None  # First detected face
    
    keyframes: List[FaceKeyframe] = []
    smoothed_cx = 0
    smoothed_cy = 0
    smoothed_w = 0
    smoothed_h = 0
    
    detections_found = 0
    detections_held = 0  # Times we held position due to jump rejection
    
    for i, ts in enumerate(timestamps):
        frame = extract_frame_at_time(video_path, ts, temp_dir)
        if frame is None:
            # Hold last position if no frame
            if prev_center and smoothed_w > 0:
                keyframes.append(FaceKeyframe(
                    timestamp=ts,
                    center_x=smoothed_cx,
                    center_y=smoothed_cy,
                    width=smoothed_w,
                    height=smoothed_h,
                    confidence=0.5,
                ))
                detections_held += 1
            continue
        
        # Try DNN first with temporal association, then Haar fallback
        detection = detect_face_dnn(
            frame,
            prev_center=prev_center if prev_center else anchor_center,
        )
        if detection is None:
            detection = detect_face_haar(frame)
        
        if detection:
            x, y, w, h, conf = detection
            cx = x + w // 2
            cy = y + h // 2
            
            # Set anchor on first detection
            if anchor_center is None:
                anchor_center = (cx, cy)
                print(f"   🎯 Anchor set at ({cx},{cy})")
            
            # Update prev_center
            prev_center = (cx, cy)
            detections_found += 1
            
            # Apply EMA smoothing AFTER association (Task B)
            if smoothed_w == 0:
                # First keyframe - initialize
                smoothed_cx = cx
                smoothed_cy = cy
                smoothed_w = w
                smoothed_h = h
            else:
                # EMA with alpha=0.6 for responsiveness
                smoothed_cx = int(0.6 * cx + 0.4 * smoothed_cx)
                smoothed_cy = int(0.6 * cy + 0.4 * smoothed_cy)
                smoothed_w = int(0.6 * w + 0.4 * smoothed_w)
                smoothed_h = int(0.6 * h + 0.4 * smoothed_h)
            
            keyframes.append(FaceKeyframe(
                timestamp=ts,
                center_x=smoothed_cx,
                center_y=smoothed_cy,
                width=smoothed_w,
                height=smoothed_h,
                confidence=conf,
            ))
        else:
            # No detection or jump rejected - HOLD last position
            if prev_center and smoothed_w > 0:
                keyframes.append(FaceKeyframe(
                    timestamp=ts,
                    center_x=smoothed_cx,
                    center_y=smoothed_cy,
                    width=smoothed_w,
                    height=smoothed_h,
                    confidence=0.5,
                ))
                detections_held += 1
    
    if len(keyframes) == 0:
        print("   ⚠️ No faces detected in any frame")
        return None
    
    print(f"   ✅ Tracking: {detections_found} detections, {detections_held} held positions")
    
    # Debug: show face positions
    print(f"   📊 Final keyframe positions:")
    for kf in keyframes[:5]:
        y_ratio = kf.center_y / height
        x_ratio = kf.center_x / width
        print(f"      t={kf.timestamp:.1f}s: center=({kf.center_x},{kf.center_y}) y={y_ratio:.2f} x={x_ratio:.2f}")
    if len(keyframes) > 5:
        print(f"      ... and {len(keyframes) - 5} more")
    
    print(f"   🎬 Generated {len(keyframes)} keyframes with temporal smoothing")
    
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
    
    IMPORTANT: Coordinates are in SOURCE video dimensions, not scaled.
    FFmpeg pipeline: crop(source) -> scale(output)
    
    Args:
        face_track: Face tracking result
        crop_width: Width of crop region (in source coords, e.g. 607 for 1920x1080 -> 9:16)
        crop_height: Height of crop region (in source coords, e.g. 1080)
        target_face_y_ratio: Where face should be positioned vertically (0-1)
        movement_threshold: Unused (kept for API compatibility)
    
    Returns:
        Tuple of (x_position, y_position) as strings for FFmpeg crop filter
    """
    print(f"   🎯 generate_ffmpeg_crop_expr():")
    print(f"      Video: {face_track.video_width}x{face_track.video_height}")
    print(f"      Crop: {crop_width}x{crop_height}")
    print(f"      Keyframes: {len(face_track.keyframes)}")
    
    if len(face_track.keyframes) == 0:
        # Fallback to center crop
        cx = face_track.video_width // 2
        cy = face_track.video_height // 2
        crop_x = max(0, cx - crop_width // 2)
        crop_y = max(0, cy - crop_height // 2)
        print(f"      No keyframes, using center crop: ({crop_x},{crop_y})")
        return str(crop_x), str(crop_y)
    
    # Calculate crop positions for each keyframe
    crop_positions = []
    for i, kf in enumerate(face_track.keyframes):
        # Center crop on face
        crop_x = kf.center_x - crop_width // 2
        crop_y = kf.center_y - int(crop_height * target_face_y_ratio)
        
        # Clamp to valid bounds
        crop_x = max(0, min(crop_x, face_track.video_width - crop_width))
        crop_y = max(0, min(crop_y, face_track.video_height - crop_height))
        
        crop_positions.append((kf.timestamp, crop_x, crop_y))
        
        # Log first 3 keyframes for debugging (Task D)
        if i < 3:
            print(f"      KF[{i}] t={kf.timestamp:.1f}s: face=({kf.center_x},{kf.center_y}) -> crop=({crop_x},{crop_y})")
    
    if len(crop_positions) == 1:
        return str(crop_positions[0][1]), str(crop_positions[0][2])
    
    # Calculate averages
    x_positions = [p[1] for p in crop_positions]
    y_positions = [p[2] for p in crop_positions]
    
    avg_x = sum(x_positions) // len(x_positions)
    avg_y = sum(y_positions) // len(y_positions)
    
    # =========================================================================
    # BACKGROUND DETECTION (Task 4): Use FACE SIZE, not position
    # Posters/background people have tiny faces
    # =========================================================================
    frame_area = face_track.video_width * face_track.video_height
    
    # Calculate area_ratio and width_ratio for each keyframe
    area_ratios = []
    width_ratios = []
    for kf in face_track.keyframes:
        area = kf.width * kf.height
        area_ratios.append(area / frame_area)
        width_ratios.append(kf.width / face_track.video_width)
    
    # Use median to be robust to outliers
    area_ratios_sorted = sorted(area_ratios)
    width_ratios_sorted = sorted(width_ratios)
    median_idx = len(area_ratios_sorted) // 2
    
    median_area_ratio = area_ratios_sorted[median_idx]
    median_width_ratio = width_ratios_sorted[median_idx]
    
    print(f"   📊 Face sizes: median_area={median_area_ratio:.4f}, median_width={median_width_ratio:.3f}")
    
    # Background/poster indicators based on SIZE:
    # - median area_ratio < 0.008 (0.8% of frame) = tiny/poster face
    # - median width_ratio < 0.06 (6% of frame width) = tiny face
    is_likely_background = (median_area_ratio < 0.008) or (median_width_ratio < 0.06)
    
    if is_likely_background:
        # Detected faces are too small - likely poster/background
        # Use Gemini anchor or left-biased fallback
        print(f"   ⚠️ Tiny faces detected (poster/background): area={median_area_ratio:.4f} width={median_width_ratio:.3f}")
        print(f"   🎯 Using LEFT-BIASED crop (typical streamer position)")
        
        # TRUE LEFT BIAS: Position crop at 10% from left edge
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
    
    # ==========================================================================
    # STABILITY DEBUG: avg crop_x and max delta between consecutive crop_x
    # ==========================================================================
    if len(x_positions) >= 2:
        deltas = [abs(x_positions[i] - x_positions[i-1]) for i in range(1, len(x_positions))]
        max_delta = max(deltas)
        print(f"   📊 STABILITY: avg_crop_x={avg_x}, max_delta={max_delta}px, x_range={x_range}px")
    else:
        print(f"   📊 STABILITY: avg_crop_x={avg_x}, single position")
    
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


# =============================================================================
# GEMINI ANCHOR + OPENCV TRACKER (for when face detection picks background)
# =============================================================================

def _is_face_track_likely_background(face_track: FaceTrack) -> bool:
    """
    Check if the detected face track is likely a background person/poster, not the streamer.
    
    Background indicators (SIZE-BASED, not position):
    - median area_ratio < 0.008 (0.8% of frame) = tiny/poster face
    - median width_ratio < 0.06 (6% of frame width) = tiny face
    """
    if not face_track or len(face_track.keyframes) < 3:
        return False
    
    frame_area = face_track.video_width * face_track.video_height
    
    # Calculate area and width ratios for each keyframe
    area_ratios = []
    width_ratios = []
    for kf in face_track.keyframes:
        area = kf.width * kf.height
        area_ratios.append(area / frame_area)
        width_ratios.append(kf.width / face_track.video_width)
    
    # Use median to be robust to outliers
    area_ratios_sorted = sorted(area_ratios)
    width_ratios_sorted = sorted(width_ratios)
    median_idx = len(area_ratios_sorted) // 2
    
    median_area_ratio = area_ratios_sorted[median_idx]
    median_width_ratio = width_ratios_sorted[median_idx]
    
    # Tiny faces = poster or background person far away
    is_tiny = (median_area_ratio < 0.008) or (median_width_ratio < 0.06)
    
    if is_tiny:
        print(f"   ⚠️ Face track is TINY (poster/background): area={median_area_ratio:.4f}, width={median_width_ratio:.3f}")
        return True
    
    return False


def _get_opencv_tracker():
    """
    Get the best available OpenCV tracker.
    
    Preference order: CSRT > KCF > MOSSE
    CSRT is most accurate but slowest.
    """
    # Try CSRT first (most accurate)
    if hasattr(cv2, 'TrackerCSRT_create'):
        return cv2.TrackerCSRT_create()
    
    # Try legacy API
    if hasattr(cv2, 'legacy') and hasattr(cv2.legacy, 'TrackerCSRT_create'):
        return cv2.legacy.TrackerCSRT_create()
    
    # Fallback to KCF
    if hasattr(cv2, 'TrackerKCF_create'):
        return cv2.TrackerKCF_create()
    
    if hasattr(cv2, 'legacy') and hasattr(cv2.legacy, 'TrackerKCF_create'):
        return cv2.legacy.TrackerKCF_create()
    
    # Last resort: MOSSE (fastest but least accurate)
    if hasattr(cv2, 'TrackerMOSSE_create'):
        return cv2.TrackerMOSSE_create()
    
    if hasattr(cv2, 'legacy') and hasattr(cv2.legacy, 'TrackerMOSSE_create'):
        return cv2.legacy.TrackerMOSSE_create()
    
    print("   ⚠️ No OpenCV tracker available")
    return None


def track_with_gemini_anchor(
    video_path: str,
    temp_dir: str,
    sample_interval: float = 1.5,
    ema_alpha: float = 0.4,
    max_keyframes: int = 20,
) -> Optional[FaceTrack]:
    """
    Track using Gemini anchor + OpenCV tracker.
    
    This is used when face detection picks the wrong person (background).
    
    Flow:
    1. Extract a frame at ~3s (middle of clip)
    2. Ask Gemini to identify the main streamer's bbox
    3. Initialize OpenCV tracker with that bbox
    4. Track across sampled frames
    5. Build keyframes from tracker results
    
    Args:
        video_path: Path to input video
        temp_dir: Temporary directory
        sample_interval: Time between samples
        ema_alpha: EMA smoothing factor
        max_keyframes: Maximum keyframes
    
    Returns:
        FaceTrack or None if tracking fails
    """
    print(f"🎯 Starting Gemini anchor + OpenCV tracker for: {video_path}")
    
    # Get video info
    info = get_video_info(video_path)
    duration = info["duration"]
    width = info["width"]
    height = info["height"]
    
    print(f"   📹 Video: {width}x{height}, {duration:.2f}s")
    
    # ==========================================================================
    # STEP 1: Extract anchor frame and get Gemini bbox
    # ==========================================================================
    anchor_time = min(3.0, duration / 2)  # 3 seconds or middle of clip
    anchor_frame_path = Path(temp_dir) / "gemini_anchor_frame.jpg"
    
    # Extract frame
    cmd = [
        "ffmpeg", "-y",
        "-ss", str(anchor_time),
        "-i", video_path,
        "-vframes", "1",
        "-q:v", "2",
        str(anchor_frame_path),
    ]
    subprocess.run(cmd, capture_output=True, text=True)
    
    if not anchor_frame_path.exists():
        print("   ❌ Failed to extract anchor frame")
        return None
    
    # Get Gemini anchor
    try:
        from gemini_vision import get_fullcam_anchor_bbox_with_gemini
        
        gemini_bbox = get_fullcam_anchor_bbox_with_gemini(
            str(anchor_frame_path), width, height
        )
        
        if gemini_bbox is None:
            print("   ❌ Gemini could not identify streamer")
            # Clean up
            anchor_frame_path.unlink(missing_ok=True)
            return None
        
        anchor_x = gemini_bbox['x']
        anchor_y = gemini_bbox['y']
        anchor_w = gemini_bbox['w']
        anchor_h = gemini_bbox['h']
        
        print(f"   ✅ Gemini anchor: ({anchor_x},{anchor_y}) {anchor_w}x{anchor_h}")
        
    except ImportError as e:
        print(f"   ❌ gemini_vision not available: {e}")
        anchor_frame_path.unlink(missing_ok=True)
        return None
    except Exception as e:
        print(f"   ❌ Gemini error: {e}")
        anchor_frame_path.unlink(missing_ok=True)
        return None
    
    # Clean up anchor frame
    anchor_frame_path.unlink(missing_ok=True)
    
    # ==========================================================================
    # STEP 2: Initialize OpenCV tracker and track across frames
    # ==========================================================================
    
    # Calculate sample timestamps
    timestamps = []
    t = 0.0
    while t < duration:
        timestamps.append(t)
        t += sample_interval
    
    if timestamps[-1] < duration - 0.5:
        timestamps.append(duration - 0.5)
    
    if len(timestamps) > max_keyframes:
        step = len(timestamps) / max_keyframes
        timestamps = [timestamps[int(i * step)] for i in range(max_keyframes)]
        timestamps.append(duration - 0.5)
    
    print(f"   🎯 Tracking across {len(timestamps)} frames...")
    
    # Open video for tracking
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("   ❌ Could not open video for tracking")
        return None
    
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    
    # Get tracker
    tracker = _get_opencv_tracker()
    tracker_initialized = False
    use_redetection = (tracker is None)
    
    if use_redetection:
        print(f"   ⚠️ No tracker available - using anchor-guided re-detection fallback")
    else:
        print(f"   ✅ OpenCV tracker available")
    
    keyframes: List[FaceKeyframe] = []
    track_successes = 0
    track_failures = 0
    redetection_count = 0
    
    # Smoothed values for EMA
    smoothed_cx = anchor_x + anchor_w // 2
    smoothed_cy = anchor_y + anchor_h // 2
    smoothed_w = anchor_w
    smoothed_h = anchor_h
    
    # Previous center for re-detection continuity
    prev_cx = smoothed_cx
    prev_cy = smoothed_cy
    max_jump_px = int(width * 0.18)  # Max allowed jump
    
    for ts in timestamps:
        frame_num = int(ts * fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        ret, frame = cap.read()
        
        if not ret or frame is None:
            continue
        
        bbox = None
        detection_method = "hold"
        
        # =====================================================================
        # METHOD 1: Use OpenCV tracker if available
        # =====================================================================
        if tracker is not None and not use_redetection:
            if not tracker_initialized:
                # Initialize tracker at anchor time
                if abs(ts - anchor_time) < sample_interval:
                    init_bbox = (anchor_x, anchor_y, anchor_w, anchor_h)
                    try:
                        tracker.init(frame, init_bbox)
                        tracker_initialized = True
                        bbox = init_bbox
                        detection_method = "tracker_init"
                        print(f"   🎯 Tracker initialized at t={ts:.1f}s")
                    except Exception as e:
                        print(f"   ⚠️ Tracker init failed: {e}, switching to re-detection")
                        use_redetection = True
            else:
                # Update tracker
                try:
                    success, tracked_bbox = tracker.update(frame)
                    if success:
                        bbox = tuple(int(v) for v in tracked_bbox)
                        track_successes += 1
                        detection_method = "tracker"
                    else:
                        track_failures += 1
                except Exception as e:
                    track_failures += 1
        
        # =====================================================================
        # METHOD 2: Anchor-guided re-detection (fallback when no tracker)
        # =====================================================================
        if bbox is None and use_redetection:
            # Define ROI around last known center (padded by 35% of frame dims)
            roi_pad_x = int(width * 0.35)
            roi_pad_y = int(height * 0.35)
            
            roi_x1 = max(0, prev_cx - roi_pad_x)
            roi_y1 = max(0, prev_cy - roi_pad_y)
            roi_x2 = min(width, prev_cx + roi_pad_x)
            roi_y2 = min(height, prev_cy + roi_pad_y)
            
            roi_w = roi_x2 - roi_x1
            roi_h = roi_y2 - roi_y1
            
            if roi_w > 50 and roi_h > 50:
                # Crop ROI and run detection
                roi_frame = frame[roi_y1:roi_y2, roi_x1:roi_x2]
                
                # Try DNN first
                detection = detect_face_dnn(
                    roi_frame,
                    confidence_threshold=0.50,
                    try_preprocessing=False,  # Skip preprocessing for speed
                    prev_center=None,  # No temporal filtering in ROI
                )
                
                # Fallback to Haar
                if detection is None:
                    detection = detect_face_haar(roi_frame)
                
                if detection is not None:
                    fx, fy, fw, fh, conf = detection
                    
                    # Translate back to full frame coords
                    fx += roi_x1
                    fy += roi_y1
                    
                    # Check jump distance
                    det_cx = fx + fw // 2
                    det_cy = fy + fh // 2
                    jump_dist = ((det_cx - prev_cx) ** 2 + (det_cy - prev_cy) ** 2) ** 0.5
                    
                    if jump_dist <= max_jump_px:
                        bbox = (fx, fy, fw, fh)
                        detection_method = "redetect"
                        redetection_count += 1
                    else:
                        # Jump too large - reject and hold
                        detection_method = "hold_jump"
        
        # =====================================================================
        # FALLBACK: Hold last known position
        # =====================================================================
        if bbox is None:
            bbox = (
                smoothed_cx - smoothed_w // 2,
                smoothed_cy - smoothed_h // 2,
                smoothed_w,
                smoothed_h
            )
            if detection_method == "hold":
                detection_method = "hold_no_detect"
        
        # Extract center
        bx, by, bw, bh = bbox
        cx = bx + bw // 2
        cy = by + bh // 2
        
        # Update prev_center for next frame's re-detection
        prev_cx = cx
        prev_cy = cy
        
        # Apply EMA smoothing
        smoothed_cx = int(ema_alpha * cx + (1 - ema_alpha) * smoothed_cx)
        smoothed_cy = int(ema_alpha * cy + (1 - ema_alpha) * smoothed_cy)
        smoothed_w = int(ema_alpha * bw + (1 - ema_alpha) * smoothed_w)
        smoothed_h = int(ema_alpha * bh + (1 - ema_alpha) * smoothed_h)
        
        keyframes.append(FaceKeyframe(
            timestamp=ts,
            center_x=smoothed_cx,
            center_y=smoothed_cy,
            width=smoothed_w,
            height=smoothed_h,
            confidence=0.9 if detection_method in ["tracker", "redetect"] else 0.7,
        ))
    
    cap.release()
    
    # Log tracking stats
    total_attempts = track_successes + track_failures
    if total_attempts > 0:
        success_rate = track_successes / total_attempts * 100
        print(f"   📊 Tracker success rate: {track_successes}/{total_attempts} ({success_rate:.0f}%)")
    
    if redetection_count > 0:
        print(f"   📊 Re-detection successes: {redetection_count}")
    
    if len(keyframes) == 0:
        print("   ❌ No keyframes generated")
        return None
    
    # =========================================================================
    # PANNING DEBUG (Task 5): Log keyframe positions and verify variation
    # =========================================================================
    print(f"   📊 Keyframe positions (verifying panning):")
    center_xs = [kf.center_x for kf in keyframes]
    
    for i, kf in enumerate(keyframes[:5]):  # Show first 5
        print(f"      t={kf.timestamp:.1f}s: center_x={kf.center_x}, center_y={kf.center_y}")
    if len(keyframes) > 5:
        print(f"      ... and {len(keyframes) - 5} more keyframes")
    
    # Calculate variation
    min_cx = min(center_xs)
    max_cx = max(center_xs)
    cx_range = max_cx - min_cx
    avg_cx = sum(center_xs) / len(center_xs)
    
    # Compute crop_x values for stability check (assuming typical 9:16 crop)
    typical_crop_width = int(height * (9/16))  # ~607 for 1080p
    crop_xs = [max(0, min(cx - typical_crop_width // 2, width - typical_crop_width)) for cx in center_xs]
    
    if len(crop_xs) >= 2:
        deltas = [abs(crop_xs[i] - crop_xs[i-1]) for i in range(1, len(crop_xs))]
        max_delta = max(deltas)
        crop_range = max(crop_xs) - min(crop_xs)
        avg_crop_x = sum(crop_xs) / len(crop_xs)
        print(f"   📊 PANNING STATS: center_x range={cx_range}px, crop_x range={crop_range}px")
        print(f"   📊 PANNING STATS: avg_crop_x={avg_crop_x:.0f}, max_delta={max_delta}px")
        
        if crop_range < 20:
            print(f"   ⚠️ LOW VARIATION: crop_x barely changes (range={crop_range}px)")
    
    avg_x_ratio = avg_cx / width
    print(f"   📍 Final avg x_ratio: {avg_x_ratio:.2f}")
    
    print(f"   ✅ Gemini+Tracker: {len(keyframes)} keyframes")
    
    return FaceTrack(
        keyframes=keyframes,
        duration=duration,
        video_width=width,
        video_height=height,
        sample_interval=sample_interval,
    )


def track_faces_with_fallback(
    video_path: str,
    temp_dir: str,
    sample_interval: float = 1.5,
    ema_alpha: float = 0.4,
    max_keyframes: int = 20,
) -> Optional[FaceTrack]:
    """
    Track faces with automatic fallback to Gemini anchor if face detection picks background.
    
    Flow:
    1. Try standard face tracking (DNN + Haar)
    2. If NO faces found OR detected faces appear to be background → try Gemini anchor + tracker
    3. Return the best result, or None to trigger left-biased fallback
    
    Args:
        video_path: Path to input video
        temp_dir: Temporary directory
        sample_interval: Time between samples
        ema_alpha: EMA smoothing factor
        max_keyframes: Maximum keyframes
    
    Returns:
        FaceTrack or None
    """
    print(f"🔍 track_faces_with_fallback() START")
    
    # First try standard face tracking
    face_track = track_faces(
        video_path=video_path,
        temp_dir=temp_dir,
        sample_interval=sample_interval,
        ema_alpha=ema_alpha,
        max_keyframes=max_keyframes,
    )
    
    # Determine if we need to use Gemini anchor fallback
    need_gemini_fallback = False
    fallback_reason = ""
    
    if face_track is None:
        need_gemini_fallback = True
        fallback_reason = "No faces detected by DNN/Haar"
    elif len(face_track.keyframes) == 0:
        need_gemini_fallback = True
        fallback_reason = "Face track has no keyframes"
    elif _is_face_track_likely_background(face_track):
        need_gemini_fallback = True
        fallback_reason = "Face track appears to be background person"
    
    if need_gemini_fallback:
        print(f"   🔄 {fallback_reason} → trying Gemini anchor...")
        
        gemini_track = track_with_gemini_anchor(
            video_path=video_path,
            temp_dir=temp_dir,
            sample_interval=sample_interval,
            ema_alpha=ema_alpha,
            max_keyframes=max_keyframes,
        )
        
        if gemini_track is not None:
            print(f"   ✅ Gemini anchor tracking succeeded")
            return gemini_track
        else:
            print(f"   ⚠️ Gemini anchor failed → will use left-biased fallback crop")
            # Return None to trigger left-biased fallback in generate_ffmpeg_crop_expr
            return None
    
    print(f"   ✅ Standard face tracking succeeded")
    return face_track
