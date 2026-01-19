"""Webcam/face detection for Stream2Short Worker.

Detects the webcam overlay position in stream recordings and returns
the region coordinates for video compositing.

Strategy:
1. Use Gemini Vision AI to detect IF a webcam exists and WHICH corner
2. Refine the bounding box using multi-frame OpenCV edge/contour detection
3. Use IoU tracking to find stable webcam rectangle across frames
4. Fall back to OpenCV face detection if Gemini unavailable
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Optional, Tuple, Dict, List
from dataclasses import dataclass


@dataclass
class BBoxCandidate:
    """A webcam bounding box candidate with scoring metadata."""
    x: int
    y: int
    width: int
    height: int
    area: int
    ar: float  # aspect ratio
    score: float
    corner: str
    frame_idx: int = 0  # which frame this came from
    
    def to_dict(self) -> Dict[str, int]:
        return {
            'x': self.x,
            'y': self.y,
            'width': self.width,
            'height': self.height,
            'corner': self.corner
        }


class WebcamRegion:
    """Represents a detected webcam region in the video."""
    
    def __init__(self, x: int, y: int, width: int, height: int, position: str):
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.position = position  # 'top-left', 'top-right', 'bottom-left', 'bottom-right'
    
    def __repr__(self):
        return f"WebcamRegion({self.position}: x={self.x}, y={self.y}, w={self.width}, h={self.height})"
    
    def to_ffmpeg_crop(self) -> str:
        """Return FFmpeg crop filter string."""
        return f"crop={self.width}:{self.height}:{self.x}:{self.y}"


def compute_iou(box1: BBoxCandidate, box2: BBoxCandidate) -> float:
    """Compute Intersection over Union between two bounding boxes."""
    x1 = max(box1.x, box2.x)
    y1 = max(box1.y, box2.y)
    x2 = min(box1.x + box1.width, box2.x + box2.width)
    y2 = min(box1.y + box1.height, box2.y + box2.height)
    
    if x2 <= x1 or y2 <= y1:
        return 0.0
    
    intersection = (x2 - x1) * (y2 - y1)
    union = box1.area + box2.area - intersection
    
    return intersection / union if union > 0 else 0.0


# =============================================================================
# DNN FACE DETECTION
# =============================================================================

# Global DNN face detector (lazy loaded)
_dnn_face_net = None
_dnn_face_available = None


def get_dnn_face_detector():
    """
    Get or create the OpenCV DNN face detector.
    
    Uses OpenCV's built-in DNN face detector (ResNet-10 SSD).
    Returns None if model files are not available.
    """
    global _dnn_face_net, _dnn_face_available
    
    if _dnn_face_available is False:
        return None
    
    if _dnn_face_net is not None:
        return _dnn_face_net
    
    # Try to load the DNN face detector
    # OpenCV 4.5+ includes a pre-trained face detector
    try:
        # Method 1: Use OpenCV's FaceDetectorYN (OpenCV 4.5.4+)
        if hasattr(cv2, 'FaceDetectorYN'):
            # YuNet face detector - fast and accurate
            model_path = cv2.data.haarcascades + '../face_detect_yunet/face_detection_yunet_2023mar.onnx'
            if Path(model_path).exists():
                _dnn_face_net = cv2.FaceDetectorYN.create(
                    model_path,
                    "",
                    (320, 320),
                    0.6,  # score threshold
                    0.3,  # NMS threshold
                    5000  # top_k
                )
                _dnn_face_available = True
                print("  ✅ Loaded YuNet face detector")
                return _dnn_face_net
    except Exception as e:
        print(f"  ⚠️ YuNet not available: {e}")
    
    # Method 2: Fall back to Haar Cascade (always available)
    try:
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        if Path(cascade_path).exists():
            _dnn_face_net = cv2.CascadeClassifier(cascade_path)
            if not _dnn_face_net.empty():
                _dnn_face_available = True
                print("  ✅ Loaded Haar Cascade face detector")
                return _dnn_face_net
    except Exception as e:
        print(f"  ⚠️ Haar Cascade not available: {e}")
    
    _dnn_face_available = False
    return None


@dataclass
class FaceDetection:
    """Detected face with bounding box and confidence."""
    x: int
    y: int
    width: int
    height: int
    confidence: float
    
    @property
    def center_x(self) -> int:
        return self.x + self.width // 2
    
    @property
    def center_y(self) -> int:
        return self.y + self.height // 2


def detect_face_dnn(
    frame_bgr: np.ndarray,
    roi_x: int = 0,
    roi_y: int = 0,
    roi_w: int = 0,
    roi_h: int = 0,
    min_confidence: float = 0.5,
) -> Optional[FaceDetection]:
    """
    Detect the best face in a region using DNN or Haar Cascade.
    
    Args:
        frame_bgr: Full video frame (BGR)
        roi_x, roi_y, roi_w, roi_h: Region of interest (0 = use full frame)
        min_confidence: Minimum detection confidence
    
    Returns:
        FaceDetection with full-frame coordinates, or None
    """
    detector = get_dnn_face_detector()
    if detector is None:
        return None
    
    # Extract ROI if specified
    if roi_w > 0 and roi_h > 0:
        roi = frame_bgr[roi_y:roi_y + roi_h, roi_x:roi_x + roi_w].copy()
    else:
        roi = frame_bgr
        roi_x, roi_y = 0, 0
        roi_h, roi_w = frame_bgr.shape[:2]
    
    faces = []
    
    # Detect based on detector type
    if isinstance(detector, cv2.CascadeClassifier):
        # Haar Cascade detection
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        detections = detector.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(40, 40),
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        
        for (fx, fy, fw, fh) in detections:
            # Haar doesn't give confidence, assume 0.8 for detected faces
            faces.append(FaceDetection(
                x=roi_x + fx,
                y=roi_y + fy,
                width=fw,
                height=fh,
                confidence=0.8
            ))
    
    elif hasattr(cv2, 'FaceDetectorYN') and isinstance(detector, cv2.FaceDetectorYN):
        # YuNet detection
        detector.setInputSize((roi_w, roi_h))
        _, detections = detector.detect(roi)
        
        if detections is not None:
            for det in detections:
                fx, fy, fw, fh = int(det[0]), int(det[1]), int(det[2]), int(det[3])
                conf = float(det[-1])
                
                if conf >= min_confidence:
                    faces.append(FaceDetection(
                        x=roi_x + fx,
                        y=roi_y + fy,
                        width=fw,
                        height=fh,
                        confidence=conf
                    ))
    
    if not faces:
        return None
    
    # Return the face with highest confidence (and reasonable size)
    # Filter out very small faces
    valid_faces = [f for f in faces if f.width >= 30 and f.height >= 30]
    if not valid_faces:
        return None
    
    best_face = max(valid_faces, key=lambda f: f.confidence * (f.width * f.height))
    return best_face


def bbox_contains_face(
    bbox: Dict[str, int],
    face: FaceDetection,
    margin_ratio: float = 0.1,
) -> bool:
    """
    Check if a bounding box fully contains a face with required margin.
    
    Args:
        bbox: Dict with x, y, width, height
        face: FaceDetection object
        margin_ratio: Required margin as ratio of face size (0.1 = 10%)
    
    Returns:
        True if bbox contains the face with margin
    """
    margin_x = int(face.width * margin_ratio)
    margin_y = int(face.height * margin_ratio)
    
    # Face bounds with margin
    face_left = face.x - margin_x
    face_right = face.x + face.width + margin_x
    face_top = face.y - margin_y
    face_bottom = face.y + face.height + margin_y
    
    # Bbox bounds
    bbox_left = bbox['x']
    bbox_right = bbox['x'] + bbox['width']
    bbox_top = bbox['y']
    bbox_bottom = bbox['y'] + bbox['height']
    
    return (
        bbox_left <= face_left and
        bbox_right >= face_right and
        bbox_top <= face_top and
        bbox_bottom >= face_bottom
    )


def check_guardrails(
    bbox: Dict[str, int],
    corner: str,
    video_width: int,
    video_height: int,
    center_band_ratio: float = 0.60,  # Stricter: 60% limit
) -> Tuple[bool, str]:
    """
    Check if a bbox violates geometric guardrails (extends too far into center).
    
    Webcam overlays should stay in their corner and not extend into gameplay area.
    
    For 1920 width:
    - top-right/bottom-right: x must be >= 1152 (60%)
    - top-left/bottom-left: x+w must be <= 768 (40%)
    
    Args:
        bbox: Dict with x, y, width, height
        corner: Which corner the webcam should be in
        video_width: Video width
        video_height: Video height
        center_band_ratio: Minimum x for right-side webcams (0.60 = 60%)
    
    Returns:
        Tuple of (passes_guardrail, reason)
    """
    x, y, w, h = bbox['x'], bbox['y'], bbox['width'], bbox['height']
    
    if corner in ['top-right', 'bottom-right']:
        # Right-side webcams: left edge must be >= 60% of width
        min_x = int(video_width * center_band_ratio)
        if x < min_x:
            return False, f"extends too far left (x={x} < {min_x})"
    
    elif corner in ['top-left', 'bottom-left']:
        # Left-side webcams: right edge must be <= 40% of width
        max_right = int(video_width * (1 - center_band_ratio))
        if x + w > max_right:
            return False, f"extends too far right (x+w={x+w} > {max_right})"
    
    return True, "OK"


def refine_webcam_bbox_face_anchor(
    frame_bgr: np.ndarray,
    corner: str,
    video_width: int,
    video_height: int,
    roi_ratio: float = 0.55,
) -> Tuple[Optional[Dict[str, int]], Optional[FaceDetection]]:
    """
    Find webcam bbox anchored around a detected face.
    
    This is more reliable than contour detection because webcams MUST contain faces.
    
    Args:
        frame_bgr: Full video frame (BGR)
        corner: Which corner to search
        video_width: Video width
        video_height: Video height
        roi_ratio: ROI size as fraction of frame
    
    Returns:
        Tuple of (bbox dict or None, FaceDetection or None)
    """
    print(f"  👤 Attempting face-anchored detection in {corner}...")
    
    # Calculate ROI bounds
    roi_w = int(video_width * roi_ratio)
    roi_h = int(video_height * roi_ratio)
    
    if corner == 'top-left':
        roi_x, roi_y = 0, 0
    elif corner == 'top-right':
        roi_x, roi_y = video_width - roi_w, 0
    elif corner == 'bottom-left':
        roi_x, roi_y = 0, video_height - roi_h
    elif corner == 'bottom-right':
        roi_x, roi_y = video_width - roi_w, video_height - roi_h
    else:
        return None, None
    
    # Detect face in ROI
    face = detect_face_dnn(frame_bgr, roi_x, roi_y, roi_w, roi_h)
    
    if face is None:
        print(f"  ⚠️ No face detected in {corner} ROI")
        return None, None
    
    print(f"  👤 Face found: {face.width}x{face.height} at ({face.x},{face.y}), conf={face.confidence:.2f}")
    
    # Generate candidate webcam rectangles around the face
    # Aspect ratios to try (common webcam ratios)
    aspect_ratios = [16/9, 4/3, 1.5]
    
    # Scale factors for rectangle size relative to face
    # (how many times bigger than face the webcam might be)
    scale_factors = [1.8, 2.2, 2.8, 3.5, 4.5]
    
    # Get edge map for scoring
    roi = frame_bgr[roi_y:roi_y + roi_h, roi_x:roi_x + roi_w]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    
    candidates = []
    rejected_face = 0
    rejected_guardrail = 0
    
    for ar in aspect_ratios:
        for scale in scale_factors:
            # Calculate candidate size based on face size
            face_size = max(face.width, face.height)
            
            # Height based on scale, width based on AR
            cand_h = int(face_size * scale)
            cand_w = int(cand_h * ar)
            
            # Don't exceed ROI bounds
            if cand_w > roi_w * 0.95 or cand_h > roi_h * 0.95:
                continue
            
            # Position: center on face, shifted slightly down (face usually in upper third)
            face_rel_x = face.x - roi_x
            face_rel_y = face.y - roi_y
            
            # Center horizontally on face
            cand_x = face_rel_x + face.width // 2 - cand_w // 2
            
            # Position vertically: face should be in upper 40% of webcam
            # So shift the rectangle down from face center
            cand_y = face_rel_y - int(cand_h * 0.25)  # Face at ~30% from top
            
            # Clamp to ROI
            cand_x = max(0, min(cand_x, roi_w - cand_w))
            cand_y = max(0, min(cand_y, roi_h - cand_h))
            
            # Full-frame coordinates
            full_x = roi_x + cand_x
            full_y = roi_y + cand_y
            
            temp_bbox = {
                'x': full_x,
                'y': full_y,
                'width': cand_w,
                'height': cand_h
            }
            
            # CONSTRAINT 1: Must contain the face with margin
            if not bbox_contains_face(temp_bbox, face, margin_ratio=0.05):
                rejected_face += 1
                continue
            
            # CONSTRAINT 2: Guardrails - must not extend into center (gameplay area)
            passes_guardrail, _ = check_guardrails(temp_bbox, corner, video_width, video_height)
            if not passes_guardrail:
                rejected_guardrail += 1
                continue
            
            # Score the candidate
            score = score_webcam_candidate(
                edges, cand_x, cand_y, cand_w, cand_h,
                face_rel_x, face_rel_y, face.width, face.height,
                corner, roi_w, roi_h
            )
            
            candidates.append({
                'x': full_x,
                'y': full_y,
                'width': cand_w,
                'height': cand_h,
                'ar': ar,
                'scale': scale,
                'score': score,
                'corner': corner
            })
    
    print(f"  📊 Candidates: {len(candidates)} valid, {rejected_face} rejected (no face), {rejected_guardrail} rejected (guardrails)")
    
    if not candidates:
        print(f"  ⚠️ No valid webcam candidates found")
        return None, face
    
    # Sort by score and pick best
    candidates.sort(key=lambda c: c['score'], reverse=True)
    best = candidates[0]
    
    print(f"  ✅ Face-anchored bbox: {best['width']}x{best['height']} at ({best['x']},{best['y']}), AR={best['ar']:.2f}, score={best['score']:.3f}")
    
    return best, face


def score_webcam_candidate(
    edges: np.ndarray,
    x: int, y: int, w: int, h: int,
    face_x: int, face_y: int, face_w: int, face_h: int,
    corner: str,
    roi_w: int, roi_h: int,
) -> float:
    """
    Score a webcam candidate rectangle using edge evidence and positioning.
    
    Components:
    1. Border edge strength (webcam overlays often have visible borders)
    2. Face margin (face shouldn't be too close to edges)
    3. Corner alignment (webcam should be near the specified corner)
    4. Area preference (slight preference for larger, but not dominant)
    """
    # 1. Border edge strength (sample pixels along rectangle perimeter)
    border_samples = []
    thickness = 3  # Sample 3 pixels thick
    
    # Top edge
    if y >= 0 and y + thickness < edges.shape[0]:
        border_samples.append(np.mean(edges[y:y+thickness, max(0,x):min(edges.shape[1],x+w)]))
    # Bottom edge
    if y + h - thickness >= 0 and y + h < edges.shape[0]:
        border_samples.append(np.mean(edges[y+h-thickness:y+h, max(0,x):min(edges.shape[1],x+w)]))
    # Left edge
    if x >= 0 and x + thickness < edges.shape[1]:
        border_samples.append(np.mean(edges[max(0,y):min(edges.shape[0],y+h), x:x+thickness]))
    # Right edge
    if x + w - thickness >= 0 and x + w < edges.shape[1]:
        border_samples.append(np.mean(edges[max(0,y):min(edges.shape[0],y+h), x+w-thickness:x+w]))
    
    border_strength = np.mean(border_samples) / 255.0 if border_samples else 0
    
    # 2. Face margin score (face should have breathing room)
    face_center_x = face_x + face_w // 2
    face_center_y = face_y + face_h // 2
    
    # Distance from face to each edge
    left_margin = face_x - x
    right_margin = (x + w) - (face_x + face_w)
    top_margin = face_y - y
    bottom_margin = (y + h) - (face_y + face_h)
    
    # Minimum margin should be at least 10% of face size
    min_desired = max(face_w, face_h) * 0.1
    
    margin_scores = [
        min(1.0, left_margin / max(1, min_desired)),
        min(1.0, right_margin / max(1, min_desired)),
        min(1.0, top_margin / max(1, min_desired)),
        min(1.0, bottom_margin / max(1, min_desired)),
    ]
    face_margin_score = np.mean(margin_scores)
    
    # 3. Corner alignment score
    if corner == 'top-left':
        corner_dist = x + y
    elif corner == 'top-right':
        corner_dist = (roi_w - (x + w)) + y
    elif corner == 'bottom-left':
        corner_dist = x + (roi_h - (y + h))
    elif corner == 'bottom-right':
        corner_dist = (roi_w - (x + w)) + (roi_h - (y + h))
    else:
        corner_dist = x + y
    
    roi_diag = np.sqrt(roi_w**2 + roi_h**2)
    corner_alignment = 1 - (corner_dist / roi_diag)
    
    # 4. Area score (slight preference for larger, normalized)
    area = w * h
    roi_area = roi_w * roi_h
    area_score = min(1.0, (area / roi_area) * 2)  # Cap at 1.0
    
    # Combined score
    score = (
        border_strength * 0.35 +      # Edge evidence is important
        face_margin_score * 0.30 +    # Face should have margin
        corner_alignment * 0.25 +     # Should be in the corner
        area_score * 0.10             # Slight size preference
    )
    
    return score


def refine_webcam_bbox_candidates(
    frame_bgr: np.ndarray,
    corner: str,
    video_width: int,
    video_height: int,
    roi_ratio: float = 0.55,  # Larger ROI (55% instead of 45%)
    top_n: int = 5,
    frame_idx: int = 0,
) -> List[BBoxCandidate]:
    """
    Find top N webcam bounding box candidates using edge detection.
    
    Gemini often returns face-like square bboxes instead of the actual webcam overlay.
    This function analyzes the corner region to find rectangular overlay candidates.
    
    Args:
        frame_bgr: Full video frame (BGR)
        corner: Which corner to search ('top-left', 'top-right', 'bottom-left', 'bottom-right')
        video_width: Full frame width
        video_height: Full frame height
        roi_ratio: What fraction of the frame to use as ROI
        top_n: Number of top candidates to return
        frame_idx: Index of this frame (for tracking)
    
    Returns:
        List of BBoxCandidate objects (up to top_n), sorted by score descending
    """
    # Calculate ROI bounds
    roi_w = int(video_width * roi_ratio)
    roi_h = int(video_height * roi_ratio)
    
    if corner == 'top-left':
        roi_x, roi_y = 0, 0
    elif corner == 'top-right':
        roi_x, roi_y = video_width - roi_w, 0
    elif corner == 'bottom-left':
        roi_x, roi_y = 0, video_height - roi_h
    elif corner == 'bottom-right':
        roi_x, roi_y = video_width - roi_w, video_height - roi_h
    else:
        return []
    
    # Extract ROI
    roi = frame_bgr[roi_y:roi_y + roi_h, roi_x:roi_x + roi_w].copy()
    roi_area = roi_w * roi_h
    
    # Convert to grayscale
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Canny edge detection
    edges = cv2.Canny(blurred, 50, 150)
    
    # Morphological close to connect broken edges
    kernel = np.ones((7, 7), np.uint8)
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)
    
    # Find contours (try both EXTERNAL and TREE to catch nested rectangles)
    contours_ext, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours_tree, _ = cv2.findContours(closed, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    # Combine and deduplicate
    all_contours = list(contours_ext) + list(contours_tree)
    
    if not all_contours:
        return []
    
    # Expected aspect ratios for webcams
    AR_16_9 = 16 / 9  # 1.778
    AR_4_3 = 4 / 3    # 1.333
    
    # Filter and score candidates
    candidates = []
    min_area = roi_area * 0.03  # At least 3% of ROI (lowered for better detection)
    seen_boxes = set()  # Deduplicate similar boxes
    
    for contour in all_contours:
        # Get bounding rectangle
        bx, by, bw, bh = cv2.boundingRect(contour)
        area = bw * bh
        
        # Filter by area
        if area < min_area:
            continue
        
        # Filter by aspect ratio (typical webcam: 1.1 to 2.6)
        ar = bw / bh if bh > 0 else 0
        if ar < 1.1 or ar > 2.6:
            continue
        
        # Deduplicate (boxes within 10px are considered same)
        box_key = (bx // 10, by // 10, bw // 10, bh // 10)
        if box_key in seen_boxes:
            continue
        seen_boxes.add(box_key)
        
        # ============================================================
        # SCORING COMPONENTS
        # ============================================================
        
        # 1. Area score (larger is better, but normalized)
        area_score = min(1.0, (area / roi_area) * 3)  # Cap at 1.0
        
        # 2. Aspect ratio score (closer to 16:9 or 4:3 is better)
        ar_diff_16_9 = abs(ar - AR_16_9)
        ar_diff_4_3 = abs(ar - AR_4_3)
        best_ar_diff = min(ar_diff_16_9, ar_diff_4_3)
        ar_score = max(0, 1 - (best_ar_diff / 0.5))  # Penalize if AR differs by >0.5
        
        # 3. Corner proximity score (closer to corner is better)
        roi_diag = np.sqrt(roi_w**2 + roi_h**2)
        if corner == 'top-left':
            corner_dist = bx + by
        elif corner == 'top-right':
            corner_dist = (roi_w - (bx + bw)) + by
        elif corner == 'bottom-left':
            corner_dist = bx + (roi_h - (by + bh))
        elif corner == 'bottom-right':
            corner_dist = (roi_w - (bx + bw)) + (roi_h - (by + bh))
        else:
            corner_dist = bx + by
        proximity_score = 1 - (corner_dist / roi_diag)
        
        # 4. "Outermost" bias - prefer boxes that extend TO the corner edge
        # This helps select the overlay boundary, not inner content
        if corner == 'top-left':
            # Prefer smaller x+y (touching top-left)
            outermost_score = 1 - ((bx + by) / (roi_w + roi_h))
        elif corner == 'top-right':
            # Prefer larger (x+w) and smaller y
            right_edge_dist = roi_w - (bx + bw)
            outermost_score = 1 - ((right_edge_dist + by) / (roi_w + roi_h))
        elif corner == 'bottom-left':
            # Prefer smaller x and larger (y+h)
            bottom_edge_dist = roi_h - (by + bh)
            outermost_score = 1 - ((bx + bottom_edge_dist) / (roi_w + roi_h))
        elif corner == 'bottom-right':
            # Prefer larger (x+w) and larger (y+h)
            right_edge_dist = roi_w - (bx + bw)
            bottom_edge_dist = roi_h - (by + bh)
            outermost_score = 1 - ((right_edge_dist + bottom_edge_dist) / (roi_w + roi_h))
        else:
            outermost_score = 0.5
        
        # Combined score with weights
        # Outermost bias is important to avoid inner rectangles
        score = (
            area_score * 0.30 +
            ar_score * 0.25 +
            proximity_score * 0.20 +
            outermost_score * 0.25
        )
        
        # Convert to full-frame coordinates
        full_x = roi_x + bx
        full_y = roi_y + by
        
        candidates.append(BBoxCandidate(
            x=full_x,
            y=full_y,
            width=bw,
            height=bh,
            area=area,
            ar=ar,
            score=score,
            corner=corner,
            frame_idx=frame_idx
        ))
    
    # Sort by score (highest first) and return top N
    candidates.sort(key=lambda c: c.score, reverse=True)
    return candidates[:top_n]


def choose_stable_webcam_bbox(
    candidates_per_frame: List[List[BBoxCandidate]],
    corner: str,
    iou_threshold: float = 0.35,
    ar_diff_threshold: float = 0.35,
) -> Optional[Dict[str, int]]:
    """
    Choose the most stable webcam bbox across multiple frames using IoU tracking.
    
    Args:
        candidates_per_frame: List of candidate lists (one per frame)
        corner: Which corner we're looking in
        iou_threshold: Minimum IoU to consider boxes as matching
        ar_diff_threshold: Maximum aspect ratio difference for matching
    
    Returns:
        Dict with x, y, width, height of the chosen bbox, or None
    """
    # Flatten all candidates if we only have one frame
    all_candidates = [c for frame_cands in candidates_per_frame for c in frame_cands]
    
    if not all_candidates:
        print("  ⚠️ No candidates found in any frame")
        return None
    
    if len(candidates_per_frame) == 1 or len(all_candidates) == 1:
        # Single frame or single candidate - just return best
        best = max(all_candidates, key=lambda c: c.score)
        print(f"  📍 Single-frame best: {best.width}x{best.height} at ({best.x},{best.y}), AR={best.ar:.2f}, score={best.score:.3f}")
        return best.to_dict()
    
    # Build tracks across frames using IoU matching
    tracks: List[List[BBoxCandidate]] = []
    
    # Start tracks from first frame's candidates
    for cand in candidates_per_frame[0]:
        tracks.append([cand])
    
    # Match candidates from subsequent frames
    for frame_idx in range(1, len(candidates_per_frame)):
        frame_cands = candidates_per_frame[frame_idx]
        
        # For each track, find best matching candidate in this frame
        for track in tracks:
            last_box = track[-1]
            best_match = None
            best_iou = 0
            
            for cand in frame_cands:
                # Check IoU
                iou = compute_iou(last_box, cand)
                if iou < iou_threshold:
                    continue
                
                # Check aspect ratio similarity
                ar_diff = abs(last_box.ar - cand.ar)
                if ar_diff > ar_diff_threshold:
                    continue
                
                if iou > best_iou:
                    best_iou = iou
                    best_match = cand
            
            if best_match:
                track.append(best_match)
    
    # Score each track
    track_scores = []
    for track in tracks:
        if len(track) < 2:
            # Penalize tracks that don't span multiple frames
            track_score = track[0].score * 0.5
        else:
            # Track score = mean score + consistency bonus - variance penalty
            mean_score = np.mean([c.score for c in track])
            mean_area = np.mean([c.area for c in track])
            
            # Consistency: low variance in area and AR is good
            area_std = np.std([c.area for c in track]) if len(track) > 1 else 0
            ar_std = np.std([c.ar for c in track]) if len(track) > 1 else 0
            
            # Normalize variance penalties
            area_penalty = (area_std / mean_area) * 0.3 if mean_area > 0 else 0
            ar_penalty = ar_std * 0.5
            
            # Bonus for longer tracks (more consistent detection)
            length_bonus = len(track) / len(candidates_per_frame) * 0.2
            
            track_score = mean_score + length_bonus - area_penalty - ar_penalty
        
        track_scores.append((track, track_score))
    
    # Sort tracks by score
    track_scores.sort(key=lambda x: x[1], reverse=True)
    
    if not track_scores:
        print("  ⚠️ No valid tracks found")
        return None
    
    best_track, best_track_score = track_scores[0]
    
    # Log track info
    print(f"  📊 Best track: {len(best_track)} frames, score={best_track_score:.3f}")
    for i, cand in enumerate(best_track):
        print(f"     Frame {cand.frame_idx}: {cand.width}x{cand.height} at ({cand.x},{cand.y}), AR={cand.ar:.2f}")
    
    # Return bbox from middle frame, or average if we want more stability
    if len(best_track) >= 2:
        # Average the boxes for more stability
        avg_x = int(np.mean([c.x for c in best_track]))
        avg_y = int(np.mean([c.y for c in best_track]))
        avg_w = int(np.mean([c.width for c in best_track]))
        avg_h = int(np.mean([c.height for c in best_track]))
        
        print(f"  ✅ Stable bbox (averaged): {avg_w}x{avg_h} at ({avg_x},{avg_y}), AR={avg_w/avg_h:.2f}")
        
        return {
            'x': avg_x,
            'y': avg_y,
            'width': avg_w,
            'height': avg_h,
            'corner': corner
        }
    else:
        # Single candidate - use as is
        cand = best_track[0]
        print(f"  ✅ Best single candidate: {cand.width}x{cand.height} at ({cand.x},{cand.y}), AR={cand.ar:.2f}")
        return cand.to_dict()


def run_multiframe_refinement(
    video_path: str,
    corner: str,
    video_width: int,
    video_height: int,
    timestamps: List[float] = [3.0, 10.0, 15.0],
    top_n_per_frame: int = 5,
) -> Optional[Dict[str, int]]:
    """
    Run webcam bbox refinement across multiple frames and choose stable result.
    
    Args:
        video_path: Path to video file
        corner: Which corner to search
        video_width: Video width
        video_height: Video height
        timestamps: List of timestamps to sample
        top_n_per_frame: Number of candidates to keep per frame
    
    Returns:
        Dict with x, y, width, height of the stable bbox, or None
    """
    print(f"  🔬 Running multi-frame refinement in {corner}...")
    
    candidates_per_frame: List[List[BBoxCandidate]] = []
    
    for i, ts in enumerate(timestamps):
        frame = extract_frame(video_path, ts)
        if frame is None:
            print(f"     Frame {i} ({ts}s): extraction failed")
            continue
        
        cands = refine_webcam_bbox_candidates(
            frame, corner, video_width, video_height,
            top_n=top_n_per_frame, frame_idx=i
        )
        
        if cands:
            print(f"     Frame {i} ({ts}s): {len(cands)} candidates")
            for c in cands[:3]:  # Log top 3
                print(f"       - {c.width}x{c.height} at ({c.x},{c.y}), AR={c.ar:.2f}, score={c.score:.3f}")
            candidates_per_frame.append(cands)
        else:
            print(f"     Frame {i} ({ts}s): no candidates")
    
    if not candidates_per_frame:
        print("  ⚠️ No candidates found in any frame")
        return None
    
    return choose_stable_webcam_bbox(candidates_per_frame, corner)


# Keep the old function name for backward compatibility
def refine_webcam_bbox(
    frame_bgr: np.ndarray,
    corner: str,
    video_width: int,
    video_height: int,
    roi_ratio: float = 0.55,
) -> Optional[Dict[str, int]]:
    """
    Refine webcam bounding box using edge detection (single frame).
    
    This is a compatibility wrapper - for better results use run_multiframe_refinement().
    """
    candidates = refine_webcam_bbox_candidates(
        frame_bgr, corner, video_width, video_height, roi_ratio, top_n=1
    )
    
    if candidates:
        return candidates[0].to_dict()
    return None


def is_gemini_bbox_good(
    bbox: Dict[str, int],
    corner: str,
    video_width: int,
    video_height: int,
    edge_epsilon: int = 12,
) -> Tuple[bool, str]:
    """
    Check if Gemini's bbox is GOOD and should be kept without refinement.
    
    A good bbox:
    - Touches the expected corner edges (within epsilon)
    - Has reasonable size (12-38% of frame width, 10-35% of height)
    - Has reasonable aspect ratio (1.2 to 2.4)
    
    Args:
        bbox: Dict with x, y, width, height
        corner: Expected corner ('top-left', 'top-right', etc.)
        video_width: Frame width
        video_height: Frame height
        edge_epsilon: How close to edge counts as "touching" (pixels)
    
    Returns:
        Tuple of (is_good, reason)
    """
    x = bbox.get('x', 0)
    y = bbox.get('y', 0)
    w = bbox.get('width', 0)
    h = bbox.get('height', 0)
    
    if w <= 0 or h <= 0:
        return False, "invalid dimensions"
    
    ar = w / h
    w_ratio = w / video_width
    h_ratio = h / video_height
    
    # Check aspect ratio (1.2 to 2.4 for typical webcams)
    if ar < 1.2 or ar > 2.4:
        return False, f"bad AR={ar:.2f} (need 1.2-2.4)"
    
    # Check size constraints
    if w_ratio < 0.12:
        return False, f"too narrow ({w_ratio*100:.1f}% < 12%)"
    if w_ratio > 0.38:
        return False, f"too wide ({w_ratio*100:.1f}% > 38%)"
    if h_ratio < 0.10:
        return False, f"too short ({h_ratio*100:.1f}% < 10%)"
    if h_ratio > 0.35:
        return False, f"too tall ({h_ratio*100:.1f}% > 35%)"
    
    # Check corner edge touching
    right_edge = x + w
    bottom_edge = y + h
    
    if corner == 'top-right':
        if y > edge_epsilon:
            return False, f"not touching top (y={y} > {edge_epsilon})"
        if right_edge < video_width - edge_epsilon:
            return False, f"not touching right (right_edge={right_edge} < {video_width - edge_epsilon})"
    
    elif corner == 'top-left':
        if y > edge_epsilon:
            return False, f"not touching top (y={y} > {edge_epsilon})"
        if x > edge_epsilon:
            return False, f"not touching left (x={x} > {edge_epsilon})"
    
    elif corner == 'bottom-right':
        if bottom_edge < video_height - edge_epsilon:
            return False, f"not touching bottom"
        if right_edge < video_width - edge_epsilon:
            return False, f"not touching right"
    
    elif corner == 'bottom-left':
        if bottom_edge < video_height - edge_epsilon:
            return False, f"not touching bottom"
        if x > edge_epsilon:
            return False, f"not touching left"
    
    return True, f"good bbox: {w}x{h} ({w_ratio*100:.0f}%x{h_ratio*100:.0f}%), AR={ar:.2f}"


def is_bbox_suspicious(bbox: Dict[str, int], video_width: int, video_height: int) -> Tuple[bool, str]:
    """
    Check if Gemini's bbox looks suspicious (likely a face crop, not webcam overlay).
    
    Args:
        bbox: Dict with x, y, width, height
        video_width: Full frame width
        video_height: Full frame height
    
    Returns:
        Tuple of (is_suspicious, reason)
    """
    w = bbox.get('width', 0)
    h = bbox.get('height', 0)
    
    if w <= 0 or h <= 0:
        return True, "invalid dimensions"
    
    ar = w / h
    area = w * h
    frame_area = video_width * video_height
    area_ratio = area / frame_area
    
    # Check 1: Near-square aspect ratio (faces are ~1:1, webcams are wider)
    if 0.85 <= ar <= 1.15:
        return True, f"near-square AR={ar:.2f} (likely face crop)"
    
    # Check 2: Too small (less than 3% of frame)
    if area_ratio < 0.03:
        return True, f"too small ({area_ratio*100:.1f}% of frame)"
    
    # Check 3: Inverted aspect ratio (taller than wide)
    if ar < 1.0:
        return True, f"inverted AR={ar:.2f} (taller than wide)"
    
    return False, "looks valid"


def create_fallback_widescreen_bbox(
    corner: str,
    gemini_height: int,
    video_width: int,
    video_height: int,
    target_ar: float = 16/9,
    max_ratio: float = 0.50,
) -> Dict[str, int]:
    """
    Create a fallback widescreen bbox when refinement fails.
    
    Uses the height from Gemini (or a reasonable default) and computes
    width to achieve target aspect ratio (16:9).
    
    Args:
        corner: Which corner to anchor the bbox
        gemini_height: Height from Gemini's bbox (or 0 for default)
        video_width: Full frame width
        video_height: Full frame height
        target_ar: Target aspect ratio (width/height)
        max_ratio: Max fraction of frame for any dimension
    
    Returns:
        Dict with x, y, width, height
    """
    print(f"  🎯 Creating fallback widescreen bbox in {corner}...")
    
    # Use Gemini height or reasonable default (20% of frame height)
    h = gemini_height if gemini_height > 50 else int(video_height * 0.20)
    
    # Compute width for target AR
    w = int(h * target_ar)
    
    # Clamp to max ratio
    max_w = int(video_width * max_ratio)
    max_h = int(video_height * max_ratio)
    
    if w > max_w:
        w = max_w
        h = int(w / target_ar)
    if h > max_h:
        h = max_h
        w = int(h * target_ar)
    
    # Make even
    w = w - (w % 2)
    h = h - (h % 2)
    
    # Anchor to corner
    if corner == 'top-left':
        x, y = 0, 0
    elif corner == 'top-right':
        x, y = video_width - w, 0
    elif corner == 'bottom-left':
        x, y = 0, video_height - h
    elif corner == 'bottom-right':
        x, y = video_width - w, video_height - h
    else:
        x, y = video_width - w, 0  # Default to top-right
    
    print(f"  📐 Fallback bbox: {w}x{h} at ({x},{y}), AR={w/h:.2f}")
    
    return {
        'x': x,
        'y': y,
        'width': w,
        'height': h,
        'corner': corner
    }


def extract_frame(video_path: str, time_sec: float = 5.0) -> Optional[np.ndarray]:
    """
    Extract a single frame from a video at the specified time.
    
    Args:
        video_path: Path to the video file
        time_sec: Time in seconds to extract frame from
        
    Returns:
        Frame as numpy array (BGR) or None if failed
    """
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"⚠️ Could not open video: {video_path}")
        return None
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if fps <= 0:
        fps = 30  # Default fallback
    
    # Calculate target frame number
    target_frame = int(time_sec * fps)
    target_frame = min(target_frame, total_frames - 1)  # Don't exceed video length
    
    # Seek to frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        print(f"⚠️ Could not read frame at {time_sec}s")
        return None
    
    return frame


def detect_faces_in_region(
    frame: np.ndarray,
    region_name: str,
    x_start: int,
    y_start: int,
    width: int,
    height: int,
    face_cascade: cv2.CascadeClassifier,
    padding_ratio: float = 2.0,  # How much padding around face (2.0 = 200% extra - much more zoomed out)
) -> Optional[WebcamRegion]:
    """
    Detect faces in a specific region of the frame.
    
    Returns WebcamRegion with generous padding around face to capture
    the full webcam frame, not just a close-up of the face.
    """
    frame_height, frame_width = frame.shape[:2]
    
    # Crop the region
    region = frame[y_start:y_start+height, x_start:x_start+width]
    
    # Convert to grayscale for face detection
    gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
    
    # Detect faces
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=4,
        minSize=(50, 50),  # Min face size
        flags=cv2.CASCADE_SCALE_IMAGE
    )
    
    if len(faces) > 0:
        # Found at least one face - get the largest one
        largest_face = max(faces, key=lambda f: f[2] * f[3])
        fx, fy, fw, fh = largest_face
        
        # Calculate crop around face with GENEROUS padding
        # Goal: Capture the full webcam frame, not just the face
        # The face coordinates are relative to the region, convert to frame coordinates
        face_center_x = x_start + fx + fw // 2
        face_center_y = y_start + fy + fh // 2
        
        # Calculate crop size with generous padding
        # We want to see shoulders/torso, not just face
        face_size = max(fw, fh)
        
        # Much larger multiplier to zoom out and show more of the person
        crop_height = int(face_size * (1 + padding_ratio * 2))  # Face + lots of padding
        
        # Make the crop wider than tall (16:9 webcam aspect ratio)
        crop_width = int(crop_height * 16 / 9)
        
        # Shift the center down slightly to show more body (face typically in upper third of webcam)
        adjusted_center_y = face_center_y + int(face_size * 0.3)
        
        # Calculate crop coordinates centered on adjusted position
        crop_x = face_center_x - crop_width // 2
        crop_y = adjusted_center_y - crop_height // 2
        
        # Clamp to frame boundaries
        crop_x = max(0, min(crop_x, frame_width - crop_width))
        crop_y = max(0, min(crop_y, frame_height - crop_height))
        crop_width = min(crop_width, frame_width - crop_x)
        crop_height = min(crop_height, frame_height - crop_y)
        
        # Ensure we have reasonable dimensions
        if crop_width < 100 or crop_height < 100:
            print(f"  ⚠️ Crop too small in {region_name}, skipping")
            return None
        
        print(f"  ✅ Face detected in {region_name}: face={fw}x{fh}, crop={crop_width}x{crop_height} (zoomed out)")
        
        return WebcamRegion(
            x=crop_x,
            y=crop_y,
            width=crop_width,
            height=crop_height,
            position=region_name
        )
    
    return None


def detect_webcam_region(
    video_path: str,
    sample_times: list[float] = [3.0, 10.0, 15.0],
    temp_dir: str = "/tmp",
) -> Optional[WebcamRegion]:
    """
    Detect the webcam/facecam region in a video.
    
    First tries Gemini Vision AI for accurate webcam rectangle detection.
    Falls back to OpenCV face detection if Gemini is not configured or fails.
    
    Args:
        video_path: Path to the video file
        sample_times: List of times (in seconds) to sample frames from
        temp_dir: Directory for temporary frame extraction
        
    Returns:
        WebcamRegion if webcam detected, None otherwise
    """
    print(f"🔍 Detecting webcam region in: {video_path}")
    
    # =========================================================================
    # Strategy 1: Try Gemini Vision AI first (more accurate)
    # =========================================================================
    gemini_attempted = False
    gemini_explicitly_no_webcam = False
    
    try:
        from gemini_vision import detect_webcam_with_gemini, extract_frame_for_analysis, save_debug_frame_with_box
        from config import config
        
        if config.GEMINI_API_KEY:
            gemini_attempted = True
            print("  🤖 Attempting Gemini Vision detection...")
            
            # Get video dimensions
            width, height = get_video_dimensions(video_path)
            
            # Try multiple frames for better detection
            timestamps_to_try = [3.0, 8.0, 15.0]
            
            for ts in timestamps_to_try:
                frame_path = f"{temp_dir}/webcam_detect_frame_{int(ts)}.jpg"
                print(f"  📸 Analyzing frame at {ts}s...")
                
                if extract_frame_for_analysis(video_path, frame_path, timestamp=ts):
                    result = detect_webcam_with_gemini(frame_path, width, height)
                    
                    # DON'T DELETE FRAME YET - we need it for refinement!
                    # It will be cleaned up after refinement or at the end
                    
                    if result:
                        if result.get('no_webcam_confirmed'):
                            # Gemini explicitly said there's no webcam - trust it, skip OpenCV
                            print("  ℹ️ Gemini confirmed: NO webcam in this video")
                            gemini_explicitly_no_webcam = True
                            break  # Don't try more frames
                        else:
                            print(f"  ✅ Gemini found webcam: {result}")
                            
                            # Use corner from Gemini if available
                            position = result.get('corner', 'top-left')
                            if position == 'unknown':
                                # Fallback: determine from coordinates
                                if result['x'] > width / 2:
                                    position = 'top-right' if result['y'] < height / 2 else 'bottom-right'
                                elif result['y'] > height / 2:
                                    position = 'bottom-left'
                                else:
                                    position = 'top-left'
                            
                            # Store original Gemini values
                            gemini_x = result['x']
                            gemini_y = result['y']
                            gemini_w = result['width']
                            gemini_h = result['height']
                            gemini_area = gemini_w * gemini_h
                            gemini_ar = gemini_w / gemini_h if gemini_h > 0 else 0
                            
                            cam_x, cam_y, cam_w, cam_h = gemini_x, gemini_y, gemini_w, gemini_h
                            
                            print(f"  📐 Gemini bbox: {gemini_w}x{gemini_h} at ({gemini_x},{gemini_y}), AR={gemini_ar:.2f}")
                            
                            # ============================================================
                            # CHECK 1: Is Gemini bbox GOOD? (should keep without refinement)
                            # ============================================================
                            is_good, good_reason = is_gemini_bbox_good(result, position, width, height)
                            
                            if is_good:
                                print(f"  ✅ Gemini bbox GOOD -> using Gemini: {good_reason}")
                                refinement_method = "gemini-good"
                                detected_face = None
                                # Skip to padding (use smaller padding for good Gemini bbox)
                                # Will be handled below
                            else:
                                print(f"  ⚠️ Gemini bbox not good: {good_reason}")
                                
                                # ============================================================
                                # CHECK 2: Is Gemini bbox SUSPICIOUS? (face crop, etc.)
                                # ============================================================
                                is_suspicious, suspicious_reason = is_bbox_suspicious(result, width, height)
                                
                                if is_suspicious:
                                    print(f"  ⚠️ Gemini bbox suspicious: {suspicious_reason}")
                            
                            # ============================================================
                            # REFINEMENT LOGIC (only if Gemini bbox is NOT good)
                            # ============================================================
                            if refinement_method != "gemini-good":
                                # Load frame for refinement - try disk first, then extract from video
                                frame_for_refine = cv2.imread(frame_path) if frame_path else None
                                
                                if frame_for_refine is None:
                                    print(f"  📹 Frame not on disk, extracting from video...")
                                    frame_for_refine = extract_frame(video_path, ts)
                                
                                detected_face = None
                                refinement_method = None
                                
                                if frame_for_refine is not None:
                                    # --------------------------------------------------------
                                    # Strategy 1: Try contour/overlay-boundary refinement FIRST
                                    # (more conservative than face-anchor)
                                    # --------------------------------------------------------
                                    if is_suspicious:
                                        print(f"  🔬 Trying contour refinement first...")
                                        
                                        contour_bbox = run_multiframe_refinement(
                                            video_path, position, width, height,
                                            timestamps=[3.0, 10.0, 15.0],
                                            top_n_per_frame=5
                                        )
                                        
                                        if contour_bbox:
                                            # Validate contour bbox
                                            passes_guardrail, _ = check_guardrails(contour_bbox, position, width, height)
                                            if passes_guardrail:
                                                print(f"  ✅ Using CONTOUR-refined bbox")
                                                cam_x = contour_bbox['x']
                                                cam_y = contour_bbox['y']
                                                cam_w = contour_bbox['width']
                                                cam_h = contour_bbox['height']
                                                refinement_method = "contour"
                                    
                                    # --------------------------------------------------------
                                    # Strategy 2: Face-anchor as LAST RESORT (with strict gates)
                                    # --------------------------------------------------------
                                    if refinement_method is None and is_suspicious:
                                        print(f"  👤 Trying face-anchor as last resort...")
                                        
                                        face_bbox, detected_face = refine_webcam_bbox_face_anchor(
                                            frame_for_refine, position, width, height
                                        )
                                        
                                        if face_bbox:
                                            face_w = face_bbox['width']
                                            face_h = face_bbox['height']
                                            face_area = face_w * face_h
                                            
                                            # STRICT GATE: Face-anchor cannot grow wildly compared to Gemini
                                            # Reject if: width > gemini_w * 1.25 OR area > gemini_area * 1.4
                                            if face_w > gemini_w * 1.25:
                                                print(f"  ❌ Face-anchor REJECTED: width {face_w} > {gemini_w}*1.25 = {int(gemini_w*1.25)}")
                                            elif face_area > gemini_area * 1.4:
                                                print(f"  ❌ Face-anchor REJECTED: area {face_area} > {gemini_area}*1.4 = {int(gemini_area*1.4)}")
                                            else:
                                                # Face-anchor passed the gate
                                                print(f"  ✅ Face-anchor passed gate, using it")
                                                cam_x = face_bbox['x']
                                                cam_y = face_bbox['y']
                                                cam_w = face_bbox['width']
                                                cam_h = face_bbox['height']
                                                refinement_method = "face-anchor"
                                    
                                    # --------------------------------------------------------
                                    # Fallback: Use Gemini bbox (with validation)
                                    # --------------------------------------------------------
                                    if refinement_method is None:
                                        print(f"  📐 Using Gemini bbox (refinements failed or rejected)")
                                        cam_x, cam_y, cam_w, cam_h = gemini_x, gemini_y, gemini_w, gemini_h
                                        refinement_method = "gemini-fallback"
                                        
                                        # Try to detect face for validation
                                        roi_w = int(width * 0.55)
                                        roi_h = int(height * 0.55)
                                        
                                        if position == 'top-left':
                                            roi_x, roi_y = 0, 0
                                        elif position == 'top-right':
                                            roi_x, roi_y = width - roi_w, 0
                                        elif position == 'bottom-left':
                                            roi_x, roi_y = 0, height - roi_h
                                        else:
                                            roi_x, roi_y = width - roi_w, height - roi_h
                                        
                                        detected_face = detect_face_dnn(frame_for_refine, roi_x, roi_y, roi_w, roi_h)
                                        if detected_face:
                                            print(f"  👤 Face found: {detected_face.width}x{detected_face.height} at ({detected_face.x},{detected_face.y})")
                                else:
                                    print(f"  ⚠️ Could not load frame for refinement!")
                                    refinement_method = "gemini-norefine"
                            
                            # ============================================================
                            # HARD CONSTRAINTS ON FINAL BBOX
                            # 1. Must contain detected face (if any)
                            # 2. Must pass guardrails (not extend into gameplay)
                            # ============================================================
                            final_bbox_check = {
                                'x': cam_x, 'y': cam_y,
                                'width': cam_w, 'height': cam_h
                            }
                            
                            # Constraint 1: Face containment
                            if detected_face:
                                if not bbox_contains_face(final_bbox_check, detected_face, margin_ratio=0.05):
                                    print(f"  ❌ HARD CONSTRAINT FAILED: bbox doesn't contain face!")
                                    print(f"     Face: ({detected_face.x},{detected_face.y}) {detected_face.width}x{detected_face.height}")
                                    print(f"     Bbox: ({cam_x},{cam_y}) {cam_w}x{cam_h}")
                                    
                                    # Force bbox to be face-centered with generous padding
                                    print(f"  🔧 Forcing face-centered bbox...")
                                    
                                    # Calculate generous bbox around face
                                    face_size = max(detected_face.width, detected_face.height)
                                    cam_h = int(face_size * 3.0)  # 3x face size for height
                                    cam_w = int(cam_h * 16 / 9)   # 16:9 aspect ratio
                                    
                                    # Center on face, shifted down slightly
                                    cam_x = detected_face.center_x - cam_w // 2
                                    cam_y = detected_face.center_y - int(cam_h * 0.35)
                                    
                                    refinement_method = "face-forced"
                            
                            # Constraint 2: Guardrails
                            final_bbox_check = {
                                'x': cam_x, 'y': cam_y,
                                'width': cam_w, 'height': cam_h
                            }
                            passes_guardrail, guardrail_reason = check_guardrails(
                                final_bbox_check, position, width, height
                            )
                            
                            if not passes_guardrail:
                                print(f"  ⚠️ Pre-padding bbox fails guardrails: {guardrail_reason}")
                                print(f"  🔧 Clamping to corner band (60%)...")
                                
                                # Clamp to corner band (60% for right, 40% for left)
                                if position in ['top-right', 'bottom-right']:
                                    min_x = int(width * 0.60)
                                    if cam_x < min_x:
                                        # Keep right edge fixed, reduce width from left
                                        old_x = cam_x
                                        old_right = cam_x + cam_w
                                        cam_w = old_right - min_x
                                        cam_x = min_x
                                        print(f"     Clamped: x {old_x} -> {cam_x}, w -> {cam_w}")
                                else:
                                    # Left side: ensure x + w <= 40%
                                    max_right = int(width * 0.40)
                                    if cam_x + cam_w > max_right:
                                        cam_w = max_right - cam_x
                                        print(f"     Clamped: w -> {cam_w}")
                                
                                if refinement_method and "clamped" not in refinement_method:
                                    refinement_method = f"{refinement_method}-clamped"
                            
                            # Clamp to video bounds
                            cam_x = max(0, min(cam_x, width - cam_w))
                            cam_y = max(0, min(cam_y, height - cam_h))
                            cam_w = min(cam_w, width - cam_x)
                            cam_h = min(cam_h, height - cam_y)
                            
                            # ============================================================
                            # DETAILED DEBUG LOGGING
                            # ============================================================
                            print(f"\n  ═══════════════════════════════════════")
                            print(f"  📋 Refinement method: {refinement_method}")
                            print(f"  📐 Gemini bbox: {result['width']}x{result['height']} at ({result['x']},{result['y']})")
                            print(f"  📐 Refined bbox: {cam_w}x{cam_h} at ({cam_x},{cam_y})")
                            if detected_face:
                                print(f"  👤 Face: {detected_face.width}x{detected_face.height} at ({detected_face.x},{detected_face.y})")
                            
                            # ============================================================
                            # APPLY PADDING (relative to refined bbox ONLY)
                            # ============================================================
                            # Use smaller padding (6%) for good Gemini bboxes
                            # Use larger padding (12%) for refined/suspicious bboxes
                            if refinement_method == "gemini-good":
                                padding_ratio = 0.06  # 6% for good Gemini
                            else:
                                padding_ratio = 0.12  # 12% for refined bboxes
                            
                            padding_x = int(cam_w * padding_ratio)
                            padding_y = int(cam_h * padding_ratio)
                            
                            # Expand region
                            new_x = cam_x - padding_x
                            new_y = cam_y - padding_y
                            new_w = cam_w + (padding_x * 2)
                            new_h = cam_h + (padding_y * 2)
                            
                            print(f"  📐 After padding (+{int(padding_ratio*100)}%): {new_w}x{new_h} at ({new_x},{new_y})")
                            
                            # ============================================================
                            # CLAMP TO VIDEO BOUNDS (DO NOT shift position for good/refined bbox)
                            # For good Gemini or refined methods: only clamp, don't reposition!
                            # ============================================================
                            is_authoritative = refinement_method in ["gemini-good", "face-anchor", "contour", "face-forced"]
                            
                            if 'right' in position:
                                # Right-side: if bbox extends past right edge, REDUCE WIDTH (not shift x)
                                if new_x + new_w > width:
                                    overflow = (new_x + new_w) - width
                                    new_w = new_w - overflow
                                    print(f"     Right overflow: reduced w by {overflow} -> {new_w}")
                                
                                # If x is negative, only reduce width (don't shift right)
                                if new_x < 0:
                                    if is_authoritative:
                                        # For good/refined: reduce width from left side
                                        new_w = new_w + new_x
                                        new_x = 0
                                        print(f"     Left overflow (authoritative): x=0, w={new_w}")
                                    else:
                                        # For fallback: allow shifting
                                        new_w = new_w + new_x
                                        new_x = 0
                            else:
                                # Left-side: if x is negative, just clamp to 0
                                if new_x < 0:
                                    new_x = 0
                                # If extends past right, reduce width
                                if new_x + new_w > width:
                                    new_w = width - new_x
                            
                            # Vertical clamping
                            if new_y < 0:
                                new_h = new_h + new_y
                                new_y = 0
                            if new_y + new_h > height:
                                new_h = height - new_y
                            
                            print(f"  📐 After frame clamp: {new_w}x{new_h} at ({new_x},{new_y})")
                            
                            # ============================================================
                            # GAME BLEED GUARDRAIL (stricter 60% for right-side)
                            # For top-right: left edge must be >= 60% of width
                            # For top-left: right edge must be <= 40% of width
                            # ============================================================
                            if 'right' in position:
                                min_x = int(width * 0.60)  # 1152 for 1920 width
                                if new_x < min_x:
                                    # REDUCE width from LEFT, keep right edge fixed
                                    old_w = new_w
                                    old_right_edge = new_x + new_w
                                    new_w = old_right_edge - min_x
                                    new_x = min_x
                                    print(f"  ⚠️ Game bleed guardrail: x {new_x - (old_w - new_w)} -> {new_x}, w {old_w} -> {new_w}")
                            else:
                                max_right = int(width * 0.40)  # 768 for 1920 width
                                if new_x + new_w > max_right:
                                    new_w = max_right - new_x
                                    print(f"  ⚠️ Game bleed guardrail: w -> {new_w}")
                            
                            # Ensure minimum dimensions
                            new_w = max(100, new_w)
                            new_h = max(100, new_h)
                            
                            print(f"  📐 FINAL bbox: {new_w}x{new_h} at ({new_x},{new_y})")
                            print(f"  ═══════════════════════════════════════\n")
                            
                            # Save debug frame showing where webcam was detected
                            debug_frame_path = f"{temp_dir}/debug_webcam_detection.jpg"
                            save_debug_frame_with_box(
                                frame_path, debug_frame_path,
                                new_x, new_y, new_w, new_h
                            )
                            
                            # Clean up the frame file now that we're done
                            try:
                                import os
                                if frame_path and os.path.exists(frame_path):
                                    os.remove(frame_path)
                            except:
                                pass
                            
                            return WebcamRegion(
                                x=new_x,
                                y=new_y,
                                width=new_w,
                                height=new_h,
                                position=position
                            )
                else:
                    print(f"  ⚠️ Could not extract frame at {ts}s")
            
            if not gemini_explicitly_no_webcam:
                print("  ⚠️ Gemini didn't find webcam, falling back to face detection")
    except ImportError:
        print("  ⚠️ Gemini module not available")
    except Exception as e:
        print(f"  ⚠️ Gemini detection failed: {e}")
    
    # If Gemini explicitly said "no webcam", trust it and skip OpenCV
    # (OpenCV has too many false positives with game graphics)
    if gemini_explicitly_no_webcam:
        print("  📹 Skipping OpenCV (Gemini confirmed no webcam) → using simple crop")
        return None
    
    # =========================================================================
    # Strategy 2: Fall back to OpenCV face detection
    # Only if Gemini wasn't configured or had an API error (not "no webcam")
    # =========================================================================
    print("  🔍 Using OpenCV face detection...")
    
    # Load OpenCV's pre-trained face detector
    # Try multiple paths for the cascade file
    cascade_paths = [
        cv2.data.haarcascades + 'haarcascade_frontalface_default.xml',
        '/usr/share/opencv4/haarcascades/haarcascade_frontalface_default.xml',
        '/usr/local/share/opencv4/haarcascades/haarcascade_frontalface_default.xml',
    ]
    
    face_cascade = None
    for path in cascade_paths:
        if Path(path).exists():
            face_cascade = cv2.CascadeClassifier(path)
            break
    
    if face_cascade is None or face_cascade.empty():
        print("⚠️ Could not load face cascade classifier")
        return None
    
    # Track detections per region
    region_detections: dict[str, int] = {
        'top-left': 0,
        'top-right': 0,
        'bottom-left': 0,
        'bottom-right': 0,
    }
    
    detected_regions: dict[str, WebcamRegion] = {}
    
    for sample_time in sample_times:
        frame = extract_frame(video_path, sample_time)
        if frame is None:
            continue
        
        frame_height, frame_width = frame.shape[:2]
        
        # Define corner regions (approximately 30% of each dimension)
        region_width = int(frame_width * 0.35)
        region_height = int(frame_height * 0.40)
        
        print(f"  📸 Sampling frame at {sample_time}s ({frame_width}x{frame_height})")
        
        # Check each corner region
        corners = {
            'top-left': (0, 0),
            'top-right': (frame_width - region_width, 0),
            'bottom-left': (0, frame_height - region_height),
            'bottom-right': (frame_width - region_width, frame_height - region_height),
        }
        
        for region_name, (x_start, y_start) in corners.items():
            result = detect_faces_in_region(
                frame=frame,
                region_name=region_name,
                x_start=x_start,
                y_start=y_start,
                width=region_width,
                height=region_height,
                face_cascade=face_cascade,
            )
            
            if result:
                region_detections[region_name] += 1
                detected_regions[region_name] = result
    
    # Find region with most consistent detections
    if not any(region_detections.values()):
        print("❌ No webcam/face detected in any corner")
        return None
    
    best_region = max(region_detections, key=region_detections.get)
    detection_count = region_detections[best_region]
    
    if detection_count == 0:
        print("❌ No consistent face detections")
        return None
    
    result = detected_regions[best_region]
    print(f"✅ Webcam detected: {result}")
    return result


def get_video_dimensions(video_path: str) -> Tuple[int, int]:
    """Get video width and height."""
    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    return width, height


if __name__ == "__main__":
    # Test with a video file
    import sys
    if len(sys.argv) > 1:
        video_path = sys.argv[1]
        result = detect_webcam_region(video_path)
        print(f"\nResult: {result}")

