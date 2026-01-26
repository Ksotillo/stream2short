"""Face tracking module for dynamic face-following crops.

Tracks face positions across video frames and generates smooth crop paths
for FULL_CAM layout rendering with TRUE DYNAMIC PANNING.

Strategy:
1. Sample frames at regular intervals (every ~0.5 seconds for smooth panning)
2. Detect face position using DNN (preferred) or Haar Cascade (fallback)
   - DNN uses lowered confidence threshold (0.35) for profile/partial faces
   - Haar includes profile face detection (both directions)
3. Apply EMA (Exponential Moving Average) smoothing to reduce jitter
4. Generate FFmpeg-compatible crop expressions with time-based interpolation

Detection Features:
- Profile face detection via Haar cascade (catches side views)
- Enhanced search with lower threshold when face is lost but prev_center known
- Temporal association to avoid jumping between faces

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

# =============================================================================
# TRACKING CONSTANTS (tunable)
# =============================================================================
# Time-scaled jump threshold: max_jump = frame_w * MAX_SPEED_RATIO_PER_SEC * dt
MAX_SPEED_RATIO_PER_SEC = 0.22  # Max face movement as ratio of frame width per second
MIN_JUMP_RATIO = 0.18           # Minimum jump threshold (prevents over-rejection)
MAX_JUMP_RATIO = 0.55           # Maximum jump threshold (prevents runaway)

# Lost mode re-acquire
LOST_COUNT_THRESHOLD = 2        # Frames of missed detection before re-acquire attempt
REACQUIRE_MIN_SCORE = 0.70      # Minimum fullcam score to accept re-acquired face

# Edge pressure: relax jump gate when face is near crop boundary
EDGE_MARGIN_RATIO = 0.12        # 12% of crop width defines "near edge"
EDGE_PRESSURE_MULTIPLIER = 1.5  # Multiply max_jump by this when face is near edge

# Candidate selection with distance penalty
DIST_PENALTY = 0.35             # Penalty factor for distance in candidate scoring
MIN_ACCEPT_SCORE = 0.65         # Minimum score for normal tracking acceptance

# Hysteresis: far jumps require higher quality
FAR_JUMP_RATIO = 0.60           # Jump distance ratio that triggers stricter requirements
FAR_JUMP_MIN_SCORE = 0.80       # Minimum score for far jumps
FAR_JUMP_MIN_CONF = 0.85        # Minimum confidence for far jumps


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
    confidence_threshold: float = 0.35,  # Lowered to catch partial/profile faces
    try_preprocessing: bool = True,
    prev_center: Optional[Tuple[int, int]] = None,  # For temporal association
    max_jump_ratio: float = 0.18,  # Max allowed jump as fraction of frame width (used if max_jump_px not set)
    max_jump_px: Optional[int] = None,  # Override: exact pixel threshold (for time-scaled jumps)
) -> Optional[Tuple[int, int, int, int, float]]:
    """
    Detect face using OpenCV DNN (ResNet SSD Caffe model).
    
    This is more robust than Haar Cascade, especially for:
    - Profile/angled faces (lowered confidence threshold helps)
    - Variable lighting conditions
    - Faces with glasses/accessories
    
    Args:
        frame_bgr: Input frame in BGR format
        confidence_threshold: Minimum confidence (default 0.35 for profile face tolerance)
        try_preprocessing: Whether to try enhanced preprocessing if first pass fails
        prev_center: Previous face center (cx, cy) for temporal association
        max_jump_ratio: Max allowed jump as fraction of frame width (used if max_jump_px not set)
        max_jump_px: Override: exact pixel threshold (for time-scaled jumps from track_faces)
    
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
    
    # Use provided max_jump_px or calculate from ratio
    if max_jump_px is None:
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
                    # SANITY FILTERS (Task C) - Relaxed for profile faces
                    # ==========================================================
                    # Reject tiny or huge detections
                    # Lowered min to 0.001 to catch faces at edges/profiles
                    if area_ratio < 0.001 or area_ratio > 0.15:
                        print(f"      [REJECT] area_ratio={area_ratio:.4f} out of [0.001, 0.15]")
                        continue
                    
                    # Reject non-face aspect ratios
                    # Expanded range for profile faces (can be narrower: 0.4) and tilted heads (wider: 2.0)
                    if aspect_ratio < 0.4 or aspect_ratio > 2.0:
                        print(f"      [REJECT] aspect_ratio={aspect_ratio:.2f} out of [0.4, 2.0]")
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
        # ======================================================================
        # ENHANCED SEARCH: If we have prev_center, try harder with lower threshold
        # This helps with profile/partial faces that might be below normal threshold
        # ======================================================================
        if prev_center is not None and confidence_threshold > 0.20:
            print(f"   🔄 No candidates - trying enhanced search around prev_center with lower threshold")
            
            # Try with much lower threshold
            enhanced_result = detect_face_dnn(
                frame_bgr,
                confidence_threshold=0.20,  # Very low threshold
                try_preprocessing=True,
                prev_center=prev_center,
                max_jump_ratio=max_jump_ratio * 1.5,  # Allow slightly larger jumps
            )
            
            if enhanced_result:
                print(f"   ✅ Enhanced search found face!")
                return enhanced_result
        
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
    # TEMPORAL ASSOCIATION with DISTANCE-AWARE SCORING + HYSTERESIS
    # ==========================================================================
    if prev_center is not None:
        prev_cx, prev_cy = prev_center
        
        # Calculate distance for each candidate
        def distance_to_prev(c):
            cx, cy = c['center']
            return ((cx - prev_cx) ** 2 + (cy - prev_cy) ** 2) ** 0.5
        
        # =====================================================================
        # DISTANCE-AWARE SCORING: Rank by combined score
        # combined = score - DIST_PENALTY * (dist / max_jump_px)
        # =====================================================================
        scored_candidates = []
        for c in candidates:
            dist = distance_to_prev(c)
            base_score = c['score']
            dist_penalty = DIST_PENALTY * (dist / max(1, max_jump_px))
            combined_score = base_score - dist_penalty
            scored_candidates.append({
                **c,
                'dist': dist,
                'combined_score': combined_score,
            })
        
        # Sort by combined score (highest first)
        scored_candidates.sort(key=lambda c: c['combined_score'], reverse=True)
        best = scored_candidates[0]
        dist = best['dist']
        
        # Log top candidates with combined scores
        if DEBUG_FACE_TRACKING:
            print(f"   📊 Candidates ranked by combined score (score - {DIST_PENALTY}*dist/max_jump):")
            for i, c in enumerate(scored_candidates[:3]):
                x, y, fw, fh = c['bbox']
                print(f"      #{i+1}: ({x},{y}) score={c['score']:.2f} dist={c['dist']:.0f}px combined={c['combined_score']:.2f}")
        
        # =====================================================================
        # GATE 1: Minimum score requirement
        # =====================================================================
        if best['score'] < MIN_ACCEPT_SCORE:
            x, y, fw, fh = best['bbox']
            print(f"   ⚠️ Jump rejected: score={best['score']:.2f} < MIN_ACCEPT_SCORE={MIN_ACCEPT_SCORE} (dist={dist:.0f}px, conf={best['confidence']:.2f})")
            return None
        
        # =====================================================================
        # GATE 2: Max jump threshold
        # =====================================================================
        if dist > max_jump_px:
            x, y, fw, fh = best['bbox']
            print(f"   ⚠️ Jump rejected: dist={dist:.0f}px > max_jump_px={max_jump_px} (score={best['score']:.2f}, conf={best['confidence']:.2f})")
            return None
        
        # =====================================================================
        # GATE 3: HYSTERESIS - Far jumps require higher quality
        # =====================================================================
        far_jump_threshold = max_jump_px * FAR_JUMP_RATIO
        if dist > far_jump_threshold:
            score_ok = best['score'] >= FAR_JUMP_MIN_SCORE
            conf_ok = best['confidence'] >= FAR_JUMP_MIN_CONF
            
            if not (score_ok or conf_ok):
                x, y, fw, fh = best['bbox']
                print(f"   ⚠️ Jump rejected (HYSTERESIS): dist={dist:.0f}px > {far_jump_threshold:.0f}px requires score>={FAR_JUMP_MIN_SCORE} OR conf>={FAR_JUMP_MIN_CONF}")
                print(f"      Candidate: score={best['score']:.2f}, conf={best['confidence']:.2f} - REJECTED")
                return None
            else:
                print(f"   ✓ Far jump accepted: dist={dist:.0f}px, score={best['score']:.2f}, conf={best['confidence']:.2f}")
        
        # ACCEPT the best candidate
        x, y, fw, fh = best['bbox']
        print(f"   ✅ detect_face_dnn(): Temporal match at ({x},{y}) dist={dist:.0f}px score={best['score']:.2f} combined={best['combined_score']:.2f}")
        return (x, y, fw, fh, best['confidence'])
    
    # No previous center - pick face with highest streamer score
    best = max(candidates, key=lambda c: c['score'])
    x, y, fw, fh = best['bbox']
    
    print(f"   ✅ detect_face_dnn(): Returning best face at ({x},{y}) {fw}x{fh} score={best['score']:.2f} conf={best['confidence']:.2f}")
    
    return (x, y, fw, fh, best['confidence'])


def detect_face_haar(
    frame_bgr: np.ndarray,
    prev_center: Optional[Tuple[int, int]] = None,  # (cx, cy) for proximity selection
) -> Optional[Tuple[int, int, int, int, float]]:
    """
    Fallback face detection using Haar Cascade.
    
    Less robust than DNN but works without model files.
    
    Args:
        frame_bgr: Input frame in BGR format
        prev_center: Optional (cx, cy) to select candidate closest to this point
    
    Returns:
        Tuple of (x, y, width, height, confidence) or None
    """
    h, w = frame_bgr.shape[:2]
    frame_area = w * h
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    
    # MIN SIZE thresholds (same as _detect_best_face_for_fullcam)
    min_area_ratio = 0.01   # 1% of frame
    min_width_ratio = 0.08  # 8% of frame width
    
    # Try multiple cascades for better coverage (frontal + profile)
    cascade_configs = [
        # Frontal face (most common)
        {
            'paths': [
                cv2.data.haarcascades + 'haarcascade_frontalface_default.xml',
                '/usr/share/opencv4/haarcascades/haarcascade_frontalface_default.xml',
            ],
            'name': 'frontal'
        },
        # Profile face (side view) - helps with turned heads
        {
            'paths': [
                cv2.data.haarcascades + 'haarcascade_profileface.xml',
                '/usr/share/opencv4/haarcascades/haarcascade_profileface.xml',
            ],
            'name': 'profile'
        },
    ]
    
    all_faces = []
    
    for config in cascade_configs:
        cascade = None
        for path in config['paths']:
            if Path(path).exists():
                cascade = cv2.CascadeClassifier(path)
                break
        
        if cascade is None or cascade.empty():
            continue
        
        faces = cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(50, 50)
        )
        
        if len(faces) > 0:
            if DEBUG_FACE_TRACKING:
                print(f"      Haar {config['name']}: found {len(faces)} faces")
            all_faces.extend([(f, config['name']) for f in faces])
        
        # Also try flipped image for profile (catches faces looking the other way)
        if config['name'] == 'profile':
            gray_flipped = cv2.flip(gray, 1)  # Horizontal flip
            faces_flipped = cascade.detectMultiScale(
                gray_flipped,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(50, 50)
            )
            if len(faces_flipped) > 0:
                if DEBUG_FACE_TRACKING:
                    print(f"      Haar profile-flipped: found {len(faces_flipped)} faces")
                # Mirror the x coordinates back
                for (fx, fy, fw, fh) in faces_flipped:
                    mirrored_x = w - fx - fw
                    all_faces.append(((mirrored_x, fy, fw, fh), 'profile-flipped'))
    
    if len(all_faces) == 0:
        return None
    
    # Convert to simple list for processing
    faces = [f[0] for f in all_faces]
    
    # Build candidates with MIN SIZE filter
    candidates = []
    for (fx, fy, fw, fh) in faces:
        area_ratio = (fw * fh) / frame_area
        width_ratio = fw / w
        cx = fx + fw // 2
        cy = fy + fh // 2
        
        # Apply MIN SIZE filter
        if area_ratio < min_area_ratio or width_ratio < min_width_ratio:
            if DEBUG_FACE_TRACKING:
                print(f"      [REJECT Haar] {fw}x{fh} area={area_ratio:.4f} width={width_ratio:.3f}")
            continue
        
        # Calculate distance to prev_center if provided
        dist = 0
        if prev_center:
            px, py = prev_center
            dist = ((cx - px) ** 2 + (cy - py) ** 2) ** 0.5
        
        candidates.append({
            'bbox': (fx, fy, fw, fh),
            'center': (cx, cy),
            'confidence': 0.8,
            'area_ratio': area_ratio,
            'width_ratio': width_ratio,
            'dist': dist,
        })
    
    if not candidates:
        if DEBUG_FACE_TRACKING:
            print(f"   ⚠️ Haar: All {len(faces)} faces rejected by MIN SIZE filter")
        return None
    
    # Debug logging
    if DEBUG_FACE_TRACKING:
        print(f"   🔍 Haar candidates after MIN SIZE filter ({len(candidates)}):")
        for i, c in enumerate(candidates[:3]):
            fx, fy, fw, fh = c['bbox']
            cx, cy = c['center']
            print(f"      #{i+1}: {fw}x{fh} center=({cx},{cy}) area={c['area_ratio']:.3%} dist={c['dist']:.0f}px")
    
    # =========================================================================
    # SELECTION: If prev_center provided, choose CLOSEST candidate
    #            Otherwise, choose LARGEST candidate
    # =========================================================================
    if prev_center:
        # Sort by distance to prev_center (closest first)
        candidates_sorted = sorted(candidates, key=lambda c: c['dist'])
        best = candidates_sorted[0]
        fx, fy, fw, fh = best['bbox']
        cx, cy = best['center']
        print(f"   ✅ Haar selected: {fw}x{fh} center=({cx},{cy}) dist={best['dist']:.0f}px prev_center={prev_center} area={best['area_ratio']:.3%} width={best['width_ratio']:.3f}")
    else:
        # No prev_center - choose largest by area
        candidates_sorted = sorted(candidates, key=lambda c: c['area_ratio'], reverse=True)
        best = candidates_sorted[0]
        fx, fy, fw, fh = best['bbox']
        cx, cy = best['center']
        print(f"   ✅ Haar selected (largest): {fw}x{fh} center=({cx},{cy}) area={best['area_ratio']:.3%}")
    
    return (fx, fy, fw, fh, 0.8)


def track_faces(
    video_path: str,
    temp_dir: str,
    sample_interval: float = 0.5,  # Changed from 1.5s to 0.5s for smoother panning
    ema_alpha: float = 0.4,
    max_keyframes: int = 60,  # Increased to accommodate more samples
    initial_center: Optional[Tuple[int, int]] = None,  # (cx, cy) seed from layout detection
    initial_size: Optional[Tuple[int, int]] = None,    # (w, h) expected face size
) -> Optional[FaceTrack]:
    """
    Track face positions throughout a video.
    
    Args:
        video_path: Path to input video
        temp_dir: Temporary directory for frame extraction
        sample_interval: Time between samples in seconds
        ema_alpha: EMA smoothing factor (0-1, higher = more responsive)
        max_keyframes: Maximum keyframes to keep
        initial_center: Optional (cx, cy) seed from layout detection to prevent wrong anchor
        initial_size: Optional (w, h) expected face size for MIN SIZE validation
    
    Returns:
        FaceTrack with smoothed keyframes, or None if no faces detected
    """
    print(f"🔍 Starting face tracking for: {video_path}")
    if initial_center:
        print(f"   🎯 Using layout face_center as tracking seed: {initial_center}")
    
    # Get video info
    info = get_video_info(video_path)
    duration = info["duration"]
    width = info["width"]
    height = info["height"]
    
    print(f"   📹 Video: {width}x{height}, {duration:.2f}s")
    
    # ==========================================================================
    # FULL_CAM HIGH-DENSITY SAMPLING: 45-60 frames for 30s clips (~0.5s spacing)
    # ==========================================================================
    # For FULL_CAM (initial_center provided), use higher density
    if initial_center:
        # Calculate ideal sample count: aim for ~0.5s spacing, min 45 frames
        ideal_samples = max(45, int(duration / 0.5))
        ideal_samples = min(ideal_samples, 60)  # Cap at 60
        effective_interval = duration / ideal_samples
        print(f"   📊 FULL_CAM high-density mode: {ideal_samples} samples (dt≈{effective_interval:.2f}s)")
    else:
        ideal_samples = max_keyframes
        effective_interval = sample_interval
    
    # Calculate sample timestamps
    timestamps = []
    t = 0.0
    actual_interval = duration / max(1, ideal_samples)
    while t < duration:
        timestamps.append(t)
        t += actual_interval
    
    # Always include near the end
    if timestamps[-1] < duration - 0.5:
        timestamps.append(duration - 0.5)
    
    # Limit total samples (safety)
    if len(timestamps) > max_keyframes:
        step = len(timestamps) / max_keyframes
        timestamps = [timestamps[int(i * step)] for i in range(max_keyframes)]
        timestamps.append(duration - 0.5)
    
    avg_dt = duration / max(1, len(timestamps))
    print(f"   🎯 Sampling {len(timestamps)} frames (dt≈{avg_dt:.2f}s)")
    
    # ==========================================================================
    # TEMPORAL ASSOCIATION with SEED from layout detection
    # ==========================================================================
    # If initial_center provided, use it as the seed (prevents wrong anchor)
    if initial_center:
        prev_center: Optional[Tuple[int, int]] = initial_center
        anchor_center: Optional[Tuple[int, int]] = initial_center
        print(f"   🎯 Anchor PRE-SET from layout: {anchor_center}")
    else:
        prev_center = None
        anchor_center = None
    
    # MIN SIZE thresholds for filtering tiny faces (posters/background)
    frame_area = width * height
    min_area_ratio = 0.01   # 1% of frame
    min_width_ratio = 0.08  # 8% of frame width
    
    keyframes: List[FaceKeyframe] = []
    smoothed_cx = 0
    smoothed_cy = 0
    smoothed_w = 0
    smoothed_h = 0
    
    # Initialize smoothed values from initial_size if provided
    if initial_size and initial_center:
        smoothed_cx, smoothed_cy = initial_center
        smoothed_w, smoothed_h = initial_size
    
    detections_found = 0
    detections_held = 0  # Times we held position due to jump rejection
    
    # ==========================================================================
    # TRACKING STATE
    # ==========================================================================
    current_anchor_score = 0.70 if initial_center else 0.0  # Assume layout seed is reliable
    lost_count = 0  # Consecutive frames without accepted detection (for LOST MODE)
    prev_ts = 0.0   # Previous timestamp for dt calculation
    
    # Crop dimensions for edge pressure calculation (9:16 from 1920x1080)
    crop_w = int(height * 9 / 16)  # ~607 for 1080p
    current_crop_x = (width - crop_w) // 2  # Start centered
    
    for i, ts in enumerate(timestamps):
        frame = extract_frame_at_time(video_path, ts, temp_dir)
        if frame is None:
            # Hold last position if no frame
            lost_count += 1
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
            prev_ts = ts
            continue
        
        # =====================================================================
        # TIME-SCALED JUMP THRESHOLD (dt-aware)
        # =====================================================================
        dt = ts - prev_ts if i > 0 else sample_interval
        base_max_jump = width * MAX_SPEED_RATIO_PER_SEC * dt
        max_jump_px = int(max(width * MIN_JUMP_RATIO, min(base_max_jump, width * MAX_JUMP_RATIO)))
        
        # =====================================================================
        # EDGE PRESSURE: Relax jump gate when face is near crop boundary
        # =====================================================================
        edge_pressure_active = False
        if smoothed_w > 0 and smoothed_cx > 0:
            # Calculate where current face center falls in crop window
            face_in_crop_x = smoothed_cx - current_crop_x
            left_margin = crop_w * EDGE_MARGIN_RATIO
            right_margin = crop_w * (1 - EDGE_MARGIN_RATIO)
            
            if face_in_crop_x < left_margin or face_in_crop_x > right_margin:
                max_jump_px = int(max_jump_px * EDGE_PRESSURE_MULTIPLIER)
                edge_pressure_active = True
                if DEBUG_FACE_TRACKING and i < 5:
                    edge_side = "LEFT" if face_in_crop_x < left_margin else "RIGHT"
                    print(f"      Edge pressure: face near {edge_side} edge, max_jump increased to {max_jump_px}px")
        
        # =====================================================================
        # DETECTION: DNN only for global, Haar only for ROI near prev_center
        # =====================================================================
        detection = detect_face_dnn(
            frame,
            prev_center=prev_center if prev_center else anchor_center,
            max_jump_px=max_jump_px,  # Pass time-scaled + edge-pressure adjusted threshold
        )
        
        # HAAR ONLY AS ROI FALLBACK (not for global discovery)
        if detection is None and prev_center is not None:
            detection = detect_face_haar(frame, prev_center=prev_center)
            if detection:
                print(f"      Haar ROI fallback used (near prev_center)")
        
        # =====================================================================
        # LOST MODE: Re-acquire after repeated misses
        # =====================================================================
        detection_accepted = False
        
        if detection:
            x, y, w, h, conf = detection
            cx = x + w // 2
            cy = y + h // 2
            
            detection_score = _score_face_for_fullcam(x, y, w, h, width, height)
            area_ratio = (w * h) / frame_area
            width_ratio = w / width
            
            # Check if this is the first anchor
            if anchor_center is None:
                if area_ratio < min_area_ratio or width_ratio < min_width_ratio:
                    print(f"   ⚠️ Rejecting tiny face as anchor: {w}x{h} area={area_ratio:.4f} width={width_ratio:.3f}")
                    lost_count += 1
                else:
                    anchor_center = (cx, cy)
                    current_anchor_score = detection_score
                    prev_center = (cx, cy)
                    lost_count = 0
                    detection_accepted = True
                    print(f"   🎯 Anchor set at ({cx},{cy}) size={w}x{h} area={area_ratio:.3%} score={detection_score:.2f}")
            else:
                # Check jump distance
                dist_to_prev = ((cx - prev_center[0]) ** 2 + (cy - prev_center[1]) ** 2) ** 0.5
                
                if dist_to_prev <= max_jump_px:
                    # ACCEPT detection - within jump threshold
                    detection_accepted = True
                    lost_count = 0
                    prev_center = (cx, cy)
                    
                    # Update anchor score if this is a better face
                    if detection_score > current_anchor_score:
                        current_anchor_score = detection_score
                        anchor_center = (cx, cy)
                else:
                    # Jump too large - check if it's a much better face for score-based re-acquire
                    if detection_score > current_anchor_score + 0.20:
                        # Better face found far away - accept it (score-based re-acquire)
                        print(f"   🔄 Score-based re-acquire: ({cx},{cy}) score={detection_score:.2f} >> current={current_anchor_score:.2f}")
                        detection_accepted = True
                        lost_count = 0
                        prev_center = (cx, cy)
                        anchor_center = (cx, cy)
                        current_anchor_score = detection_score
                    else:
                        # Reject - too far and not clearly better
                        lost_count += 1
                        if DEBUG_FACE_TRACKING:
                            print(f"      Jump rejected: {dist_to_prev:.0f}px > {max_jump_px}px (dt={dt:.2f}s)")
        else:
            # No detection at all
            lost_count += 1
        
        # =====================================================================
        # LOST MODE RE-ACQUIRE (after repeated misses)
        # =====================================================================
        if not detection_accepted and lost_count >= LOST_COUNT_THRESHOLD and smoothed_w > 0:
            print(f"   🆘 LOST MODE: reacquiring (lost_count={lost_count})")
            
            # Try 1: Global DNN detection (no prev_center constraint)
            global_detection = detect_face_dnn(frame, prev_center=None, confidence_threshold=0.35)
            
            if global_detection:
                gx, gy, gw, gh, gconf = global_detection
                gcx = gx + gw // 2
                gcy = gy + gh // 2
                global_score = _score_face_for_fullcam(gx, gy, gw, gh, width, height)
                
                if global_score >= REACQUIRE_MIN_SCORE:
                    print(f"   ✅ LOST MODE: re-acquired via DNN at center=({gcx},{gcy}) score={global_score:.2f}")
                    detection_accepted = True
                    lost_count = 0
                    prev_center = (gcx, gcy)
                    anchor_center = (gcx, gcy)
                    current_anchor_score = global_score
                    x, y, w, h, conf = global_detection
                    cx, cy = gcx, gcy
                else:
                    print(f"   ⚠️ LOST MODE: DNN found face but score={global_score:.2f} < {REACQUIRE_MIN_SCORE}")
            
            # Try 2: Gemini anchor as last resort
            if not detection_accepted:
                try:
                    import tempfile
                    with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as f:
                        cv2.imwrite(f.name, frame)
                        frame_path_temp = f.name
                    
                    from gemini_vision import get_fullcam_anchor_bbox_with_gemini
                    gemini_result = get_fullcam_anchor_bbox_with_gemini(
                        frame_path=frame_path_temp,
                        video_width=width,
                        video_height=height
                    )
                    
                    os.unlink(frame_path_temp)
                    
                    if gemini_result and 'x' in gemini_result:
                        gx = gemini_result['x']
                        gy = gemini_result['y']
                        gw = gemini_result['w']
                        gh = gemini_result['h']
                        gconf = gemini_result.get('confidence', 0.8)
                        gcx = gx + gw // 2
                        gcy = gy + gh // 2
                        gemini_score = _score_face_for_fullcam(gx, gy, gw, gh, width, height)
                        
                        if gemini_score >= REACQUIRE_MIN_SCORE or gconf >= 0.85:
                            print(f"   ✅ LOST MODE: re-acquired via GEMINI at center=({gcx},{gcy}) score={gemini_score:.2f} conf={gconf:.2f}")
                            detection_accepted = True
                            lost_count = 0
                            prev_center = (gcx, gcy)
                            anchor_center = (gcx, gcy)
                            current_anchor_score = gemini_score
                            x, y, w, h, conf = gx, gy, gw, gh, gconf
                            cx, cy = gcx, gcy
                        else:
                            print(f"   ⚠️ LOST MODE: Gemini score={gemini_score:.2f} < threshold")
                except Exception as e:
                    print(f"   ❌ LOST MODE: Gemini failed: {e}")
            
            if not detection_accepted:
                print(f"   ❌ LOST MODE: re-acquire failed, holding position")
        
        # =====================================================================
        # UPDATE KEYFRAMES
        # =====================================================================
        if detection_accepted:
            detections_found += 1
            
            # Apply EMA smoothing
            if smoothed_w == 0:
                smoothed_cx = cx
                smoothed_cy = cy
                smoothed_w = w
                smoothed_h = h
            else:
                smoothed_cx = int(0.6 * cx + 0.4 * smoothed_cx)
                smoothed_cy = int(0.6 * cy + 0.4 * smoothed_cy)
                smoothed_w = int(0.6 * w + 0.4 * smoothed_w)
                smoothed_h = int(0.6 * h + 0.4 * smoothed_h)
            
            # Update crop_x for edge pressure calculation
            current_crop_x = smoothed_cx - crop_w // 2
            current_crop_x = max(0, min(current_crop_x, width - crop_w))
            
            keyframes.append(FaceKeyframe(
                timestamp=ts,
                center_x=smoothed_cx,
                center_y=smoothed_cy,
                width=smoothed_w,
                height=smoothed_h,
                confidence=conf,
            ))
        else:
            # HOLD last position
            detections_held += 1
            if smoothed_w > 0:
                keyframes.append(FaceKeyframe(
                    timestamp=ts,
                    center_x=smoothed_cx,
                    center_y=smoothed_cy,
                    width=smoothed_w,
                    height=smoothed_h,
                    confidence=0.5,
                ))
        
        prev_ts = ts
    
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
    Generate FFmpeg crop x and y expressions for TRUE DYNAMIC PANNING.
    
    Creates FFmpeg expressions using linear interpolation between keyframes.
    The crop position smoothly follows the face throughout the video.
    
    IMPORTANT: Coordinates are in SOURCE video dimensions, not scaled.
    FFmpeg pipeline: crop(source) -> scale(output)
    
    Args:
        face_track: Face tracking result
        crop_width: Width of crop region (in source coords, e.g. 607 for 1920x1080 -> 9:16)
        crop_height: Height of crop region (in source coords, e.g. 1080)
        target_face_y_ratio: Where face should be positioned vertically (0-1)
        movement_threshold: Minimum x_range to enable panning (avoids jitter on static faces)
    
    Returns:
        Tuple of (x_expression, y_expression) as strings for FFmpeg crop filter
    """
    print(f"   🎯 generate_ffmpeg_crop_expr() - TRUE PANNING MODE:")
    print(f"      Video: {face_track.video_width}x{face_track.video_height}")
    print(f"      Crop: {crop_width}x{crop_height}")
    print(f"      Keyframes: {len(face_track.keyframes)}")
    
    max_x = face_track.video_width - crop_width
    max_y = face_track.video_height - crop_height
    
    if len(face_track.keyframes) == 0:
        # Fallback to center crop
        cx = face_track.video_width // 2
        cy = face_track.video_height // 2
        crop_x = max(0, min(cx - crop_width // 2, max_x))
        crop_y = max(0, min(cy - crop_height // 2, max_y))
        print(f"      No keyframes, using center crop: ({crop_x},{crop_y})")
        return str(crop_x), str(crop_y)
    
    # ==========================================================================
    # CROP SAFETY MARGIN: Ensure face bbox + padding fits inside crop window
    # ==========================================================================
    safety_padding_ratio = 0.10  # 10% of crop width as padding
    safety_padding_x = int(crop_width * safety_padding_ratio)
    
    # Calculate crop positions for each keyframe
    crop_positions = []
    for i, kf in enumerate(face_track.keyframes):
        # Calculate face bbox bounds
        face_left = kf.center_x - kf.width // 2
        face_right = kf.center_x + kf.width // 2
        
        # Ideal crop position (centered on face)
        crop_x = kf.center_x - crop_width // 2
        crop_y = kf.center_y - int(crop_height * target_face_y_ratio)
        
        # =================================================================
        # SAFETY MARGIN: Ensure face bbox + padding is inside crop window
        # =================================================================
        crop_left = crop_x
        crop_right = crop_x + crop_width
        
        # Check if face is too close to left edge of crop
        if face_left - safety_padding_x < crop_left:
            # Shift crop left to include face with padding
            crop_x = face_left - safety_padding_x
            if i < 3:
                print(f"      KF[{i}] safety: shifted LEFT to include face (face_left={face_left}, padding={safety_padding_x})")
        
        # Check if face is too close to right edge of crop
        elif face_right + safety_padding_x > crop_right:
            # Shift crop right to include face with padding
            crop_x = face_right + safety_padding_x - crop_width
            if i < 3:
                print(f"      KF[{i}] safety: shifted RIGHT to include face (face_right={face_right}, padding={safety_padding_x})")
        
        # Clamp to valid bounds
        crop_x = max(0, min(crop_x, max_x))
        crop_y = max(0, min(crop_y, max_y))
        
        crop_positions.append((float(kf.timestamp), int(crop_x), int(crop_y)))
        
        # Log first 5 keyframes for debugging
        if i < 5:
            print(f"      KF[{i}] t={kf.timestamp:.2f}s: face=({kf.center_x},{kf.center_y}) w={kf.width} -> crop=({crop_x},{crop_y})")
    
    if len(crop_positions) > 5:
        print(f"      ... and {len(crop_positions) - 5} more keyframes")
    
    if len(crop_positions) == 1:
        return str(crop_positions[0][1]), str(crop_positions[0][2])
    
    # Calculate stats
    x_positions = [p[1] for p in crop_positions]
    y_positions = [p[2] for p in crop_positions]
    
    avg_x = sum(x_positions) // len(x_positions)
    avg_y = sum(y_positions) // len(y_positions)
    x_range = max(x_positions) - min(x_positions)
    y_range = max(y_positions) - min(y_positions)
    
    # =========================================================================
    # BACKGROUND DETECTION: Use FACE SIZE
    # =========================================================================
    frame_area = face_track.video_width * face_track.video_height
    area_ratios = [(kf.width * kf.height) / frame_area for kf in face_track.keyframes]
    width_ratios = [kf.width / face_track.video_width for kf in face_track.keyframes]
    
    median_area = sorted(area_ratios)[len(area_ratios) // 2]
    median_width = sorted(width_ratios)[len(width_ratios) // 2]
    
    print(f"   📊 Face sizes: median_area={median_area:.4f}, median_width={median_width:.3f}")
    
    if median_area < 0.008 or median_width < 0.06:
        print(f"   ⚠️ Tiny faces detected (poster/background)")
        print(f"   🎯 Using LEFT-BIASED static crop")
        default_x = int(max_x * 0.10)
        default_y = max(0, max_y // 2)
        return str(default_x), str(default_y)
    
    # =========================================================================
    # MOVEMENT CHECK: Only pan if there's meaningful movement
    # =========================================================================
    print(f"   📊 Movement: x_range={x_range}px, y_range={y_range}px")
    
    if x_range < movement_threshold:
        # Not enough movement - use static average to avoid jitter
        print(f"   📍 Low movement ({x_range}px < {movement_threshold}px threshold)")
        print(f"   📍 Using STATIC average: ({avg_x}, {avg_y})")
        return str(avg_x), str(avg_y)
    
    # =========================================================================
    # TRUE DYNAMIC PANNING: Generate FFmpeg interpolation expression
    # =========================================================================
    print(f"   🎬 TRUE PANNING ENABLED: {x_range}px horizontal movement")
    
    # Build FFmpeg expression for x using linear interpolation between keyframes
    # Format: if(lt(t,t1), lerp(x0,x1,t,t0,t1), if(lt(t,t2), lerp(x1,x2,t,t1,t2), ...))
    # Where lerp(a,b,t,t0,t1) = a + (b-a)*(t-t0)/(t1-t0)
    
    def build_lerp_expr(positions: List[Tuple[float, int, int]], coord_idx: int, max_val: int) -> str:
        """
        Build FFmpeg expression for linear interpolation.
        
        Uses FFmpeg's expression syntax with 'between' for time ranges to avoid comma issues.
        FFmpeg parses commas as filter separators, so we use semicolon syntax instead.
        
        Args:
            positions: List of (timestamp, x, y)
            coord_idx: 1 for x, 2 for y
            max_val: Maximum valid value (for clamping) - unused since values are pre-clamped
        """
        if len(positions) <= 1:
            return str(positions[0][coord_idx])
        
        # For FFmpeg, we need to avoid commas in expressions
        # Use a weighted sum approach: sum of (value * weight) where weight is 1 for active segment
        # Expression: v0*between(t,t0,t1) + v1*between(t,t1,t2) + ...
        #
        # But 'between' also uses commas. So let's use a different approach:
        # Use gte (>=) and lt (<) without commas:
        # if(lt(t;t1);lerp1;if(lt(t;t2);lerp2;...))
        # 
        # Actually FFmpeg uses , not ; for function args. The issue is the filter graph parsing.
        # Solution: Escape commas in the expression with backslash when building -vf string
        
        # Build expression without escaping - we'll escape when constructing the filter
        # Start with the last segment's end value (for t >= last timestamp)
        expr = str(positions[-1][coord_idx])
        
        # Build nested if statements from end to start
        for i in range(len(positions) - 2, -1, -1):
            t0, x0_full, y0_full = positions[i]
            t1, x1_full, y1_full = positions[i + 1]
            
            v0 = x0_full if coord_idx == 1 else y0_full
            v1 = x1_full if coord_idx == 1 else y1_full
            
            # Avoid division by zero
            dt = t1 - t0
            if dt < 0.001:
                dt = 0.001
            
            # Linear interpolation: v0 + (v1-v0) * (t-t0) / (t1-t0)
            dv = v1 - v0
            
            if abs(dv) < 1:
                # No movement in this segment - use static value
                lerp = str(v0)
            else:
                # Interpolation expression (values already clamped)
                # Format: (v0 + dv*(t-t0)/dt)
                lerp = f"({v0}+{dv}*(t-{t0:.2f})/{dt:.2f})"
            
            # Wrap in if statement - use COLON instead of comma for FFmpeg expression separator
            # FFmpeg crop filter expressions use : as separator, not ,
            expr = f"if(lt(t,{t1:.2f}),{lerp},{expr})"
        
        return expr
    
    x_expr = build_lerp_expr(crop_positions, 1, max_x)
    y_expr = build_lerp_expr(crop_positions, 2, max_y)
    
    # Log expression length
    print(f"   📐 X expression length: {len(x_expr)} chars")
    print(f"   📐 Y expression length: {len(y_expr)} chars")
    
    # Show first/last crop positions for verification
    print(f"   📍 Panning: t=0s crop_x={crop_positions[0][1]} -> t={crop_positions[-1][0]:.1f}s crop_x={crop_positions[-1][1]}")
    
    # Stability stats
    if len(x_positions) >= 2:
        deltas = [abs(x_positions[i] - x_positions[i-1]) for i in range(1, len(x_positions))]
        max_delta = max(deltas)
        print(f"   📊 STABILITY: max_delta={max_delta}px between consecutive frames")
    
    return x_expr, y_expr


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
    sample_interval: float = 0.5,  # Changed from 1.5s to 0.5s for smoother panning
    ema_alpha: float = 0.4,
    max_keyframes: int = 60,  # Increased to accommodate more samples
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
    sample_interval: float = 0.5,  # Changed from 1.5s to 0.5s for smoother panning
    ema_alpha: float = 0.4,
    max_keyframes: int = 60,  # Increased to accommodate more samples
    initial_center: Optional[Tuple[int, int]] = None,  # (cx, cy) from layout detection
    initial_size: Optional[Tuple[int, int]] = None,    # (w, h) from layout detection
) -> Optional[FaceTrack]:
    """
    Track faces with automatic fallback to Gemini anchor if face detection picks background.
    
    Flow:
    1. Try standard face tracking (DNN + Haar), seeded with initial_center if provided
    2. If NO faces found OR detected faces appear to be background → try Gemini anchor + tracker
    3. Return the best result, or None to trigger left-biased fallback
    
    Args:
        video_path: Path to input video
        temp_dir: Temporary directory
        sample_interval: Time between samples
        ema_alpha: EMA smoothing factor
        max_keyframes: Maximum keyframes
        initial_center: Optional (cx, cy) seed from layout detection
        initial_size: Optional (w, h) from layout detection
    
    Returns:
        FaceTrack or None
    """
    print(f"🔍 track_faces_with_fallback() START")
    if initial_center:
        print(f"   🎯 Seeding with layout face_center: {initial_center}")
    
    # First try standard face tracking (seeded with initial_center if provided)
    face_track = track_faces(
        video_path=video_path,
        temp_dir=temp_dir,
        sample_interval=sample_interval,
        ema_alpha=ema_alpha,
        max_keyframes=max_keyframes,
        initial_center=initial_center,
        initial_size=initial_size,
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
