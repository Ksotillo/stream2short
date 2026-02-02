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
import json
import numpy as np
import os
from pathlib import Path
from typing import Optional, Tuple, Dict, List, Literal
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
    
    def __init__(self, x: int, y: int, width: int, height: int, position: str,
                 gemini_type: str = 'unknown', gemini_confidence: float = 0.0):
        self.x = int(x)
        self.y = int(y)
        self.width = int(width)
        self.height = int(height)
        self.position = position  # 'top-left', 'top-right', 'bottom-left', 'bottom-right', 'full'
        self.gemini_type = gemini_type  # 'corner_overlay', 'top_band', 'bottom_band', 'center_box', 'full_cam', 'none'
        self.gemini_confidence = float(gemini_confidence)
    
    def __repr__(self):
        return f"WebcamRegion({self.position}: x={self.x}, y={self.y}, w={self.width}, h={self.height}, type={self.gemini_type})"
    
    def to_ffmpeg_crop(self) -> str:
        """Return FFmpeg crop filter string."""
        return f"crop={self.width}:{self.height}:{self.x}:{self.y}"
    
    def to_dict(self) -> Dict[str, any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'x': _to_python_scalar(self.x),
            'y': _to_python_scalar(self.y),
            'width': _to_python_scalar(self.width),
            'height': _to_python_scalar(self.height),
            'position': self.position,
            'gemini_type': self.gemini_type,
            'gemini_confidence': _to_python_scalar(self.gemini_confidence),
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'WebcamRegion':
        """Create from dictionary."""
        return cls(
            x=data['x'],
            y=data['y'],
            width=data['width'],
            height=data['height'],
            position=data['position'],
            gemini_type=data.get('gemini_type', 'unknown'),
            gemini_confidence=data.get('gemini_confidence', 0.0),
        )


@dataclass
class FaceCenter:
    """Face center point for face-centered cropping."""
    x: int
    y: int
    width: int
    height: int
    
    @property
    def center_x(self) -> int:
        return self.x + self.width // 2
    
    @property
    def center_y(self) -> int:
        return self.y + self.height // 2
    
    def to_dict(self) -> Dict:
        return {'x': self.x, 'y': self.y, 'width': self.width, 'height': self.height}
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'FaceCenter':
        return cls(x=data['x'], y=data['y'], width=data['width'], height=data['height'])


def _to_python_scalar(val):
    """Convert numpy scalar types to Python native types for JSON serialization."""
    if val is None:
        return None
    if hasattr(val, 'item'):  # numpy scalar
        return val.item()
    if isinstance(val, (int, float, str, bool)):
        return val
    # Try to convert to appropriate type
    try:
        if isinstance(val, (np.integer,)):
            return int(val)
        if isinstance(val, (np.floating,)):
            return float(val)
    except:
        pass
    return val


@dataclass
class LayoutInfo:
    """
    Layout classification result with all detection info.
    
    Layouts:
    - FULL_CAM: Entire clip is webcam (no gameplay). Use full-frame crop.
    - TOP_BAND: Wide webcam band at top with gameplay below. Source is already stacked.
    - SPLIT: Traditional webcam + gameplay split layout (corner overlay).
    - NO_WEBCAM: No webcam detected. Use simple center crop.
    """
    layout: str  # 'FULL_CAM', 'TOP_BAND', 'SPLIT', 'NO_WEBCAM'
    webcam_region: Optional[WebcamRegion]
    face_center: Optional[FaceCenter]
    reason: str
    bbox_area_ratio: float = 0.0
    gemini_type: str = 'unknown'  # Type from Gemini detection
    confidence: float = 0.0  # Detection confidence
    
    def to_dict(self) -> Dict:
        return {
            'layout': self.layout,
            'webcam_region': self.webcam_region.to_dict() if self.webcam_region else None,
            'face_center': self.face_center.to_dict() if self.face_center else None,
            'reason': self.reason,
            'bbox_area_ratio': _to_python_scalar(self.bbox_area_ratio),
            'gemini_type': self.gemini_type,
            'confidence': _to_python_scalar(self.confidence),
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'LayoutInfo':
        return cls(
            layout=data['layout'],
            webcam_region=WebcamRegion.from_dict(data['webcam_region']) if data.get('webcam_region') else None,
            face_center=FaceCenter.from_dict(data['face_center']) if data.get('face_center') else None,
            reason=data['reason'],
            bbox_area_ratio=data.get('bbox_area_ratio', 0.0),
            gemini_type=data.get('gemini_type', 'unknown'),
            confidence=data.get('confidence', 0.0),
        )


def classify_layout(
    frame_bgr: np.ndarray,
    webcam_bbox: Optional[Dict[str, int]],
    corner: Optional[str],
    video_width: int,
    video_height: int,
    gemini_type: str = 'unknown',
    gemini_confidence: float = 0.0,
) -> Tuple[str, str, float]:
    """
    Classify the video layout based on webcam bbox and frame analysis.
    
    Supports layouts:
    - FULL_CAM: Entire clip is webcam (>70% or Gemini type is full_cam)
    - TOP_BAND: Wide webcam band at top (width >= 55%, height 18-60%, near top)
    - SPLIT: Traditional corner overlay (area 3-30%, in a corner zone)
    - NO_WEBCAM: No webcam detected
    
    Args:
        frame_bgr: BGR frame from video
        webcam_bbox: Detected webcam bbox {'x', 'y', 'width', 'height'} or None
        corner: Detected corner ('top-left', 'top-right', etc.) or None
        video_width: Full frame width
        video_height: Full frame height
        gemini_type: Type from Gemini detection (corner_overlay, top_band, etc.)
        gemini_confidence: Confidence from Gemini detection
    
    Returns:
        Tuple of (layout, reason, bbox_area_ratio)
        layout: 'FULL_CAM', 'TOP_BAND', 'SPLIT', or 'NO_WEBCAM'
    """
    if webcam_bbox is None:
        return 'NO_WEBCAM', 'No webcam detected', 0.0
    
    # Special case: 'full' position means the entire frame is the camera (FULL_CAM)
    if corner == 'full':
        return 'FULL_CAM', 'Full frame is camera (no overlay detected, face found)', 1.0
    
    # Extract bbox dimensions
    bx = int(webcam_bbox.get('x', 0))
    by = int(webcam_bbox.get('y', 0))
    bw = int(webcam_bbox.get('width', 0))
    bh = int(webcam_bbox.get('height', 0))
    
    if bw <= 0 or bh <= 0:
        return 'NO_WEBCAM', 'Invalid webcam dimensions', 0.0
    
    # Compute key ratios
    frame_area = video_width * video_height
    bbox_area = bw * bh
    bbox_area_ratio = bbox_area / frame_area
    width_ratio = bw / video_width
    height_ratio = bh / video_height
    outside_area_ratio = 1.0 - bbox_area_ratio
    
    # Edge touching tolerance
    tol_x = int(video_width * 0.03)  # 3% of width
    tol_y = int(video_height * 0.03)  # 3% of height
    
    # Check if bbox touches edges
    touches_top = by <= tol_y
    touches_bottom = (by + bh) >= (video_height - tol_y)
    touches_left = bx <= tol_x
    touches_right = (bx + bw) >= (video_width - tol_x)
    
    # Check if in corner zone
    is_in_corner = (
        (touches_top and touches_left) or
        (touches_top and touches_right) or
        (touches_bottom and touches_left) or
        (touches_bottom and touches_right)
    )
    
    # Check if near top (y within top 25% of frame)
    y_center = by + bh / 2
    is_near_top = y_center < video_height * 0.40  # Center in top 40%
    is_near_bottom = y_center > video_height * 0.60  # Center in bottom 40%
    
    print(f"  📊 Layout classifier:")
    print(f"     Gemini type: {gemini_type}, confidence: {gemini_confidence:.2f}")
    print(f"     bbox: {bw}x{bh} at ({bx},{by})")
    print(f"     width_ratio: {width_ratio:.2%}, height_ratio: {height_ratio:.2%}")
    print(f"     bbox_area_ratio: {bbox_area_ratio:.2%}")
    print(f"     touches: top={touches_top}, bottom={touches_bottom}, left={touches_left}, right={touches_right}")
    print(f"     is_in_corner: {is_in_corner}, is_near_top: {is_near_top}")
    
    # ==========================================================================
    # RULE 1: FULL_CAM - Gemini explicitly says full_cam OR very large bbox
    # ==========================================================================
    if gemini_type == 'full_cam':
        reason = f"FULL_CAM: Gemini detected full_cam type (conf={gemini_confidence:.2f})"
        print(f"     ✅ {reason}")
        return 'FULL_CAM', reason, bbox_area_ratio
    
    if bbox_area_ratio >= 0.70:
        reason = f"FULL_CAM: bbox_area_ratio={bbox_area_ratio:.2%} >= 70%"
        print(f"     ✅ {reason}")
        return 'FULL_CAM', reason, bbox_area_ratio
    
    # ==========================================================================
    # RULE 2: Large webcam (>= 60%) with low outside complexity -> FULL_CAM
    # ==========================================================================
    if bbox_area_ratio >= 0.55:
        edge_density_outside = _compute_edge_density_outside(
            frame_bgr, bx, by, bw, bh, video_width, video_height
        )
        print(f"     edge_density_outside: {edge_density_outside:.4f}")
        
        if edge_density_outside < 0.05:
            reason = f"FULL_CAM: bbox_area_ratio={bbox_area_ratio:.2%}, low outside complexity ({edge_density_outside:.4f})"
            print(f"     ✅ {reason}")
            return 'FULL_CAM', reason, bbox_area_ratio
    
    # ==========================================================================
    # RULE 3: TOP_BAND - Wide webcam near top (Gemini type or heuristics)
    # Conditions: width >= 55%, height 18-60%, webcam is in top portion
    # ==========================================================================
    is_top_band_candidate = (
        (gemini_type == 'top_band') or
        (width_ratio >= 0.55 and 0.15 <= height_ratio <= 0.60 and is_near_top and touches_top)
    )
    
    if is_top_band_candidate:
        # Additional validation: top band should span most of width
        spans_width = width_ratio >= 0.55
        reasonable_height = 0.15 <= height_ratio <= 0.60
        
        if spans_width and reasonable_height and is_near_top:
            reason = f"TOP_BAND: width={width_ratio:.2%}, height={height_ratio:.2%}, near_top={is_near_top}, gemini_type={gemini_type}"
            print(f"     ✅ {reason}")
            return 'TOP_BAND', reason, bbox_area_ratio
    
    # ==========================================================================
    # RULE 4: BOTTOM_BAND (similar to TOP_BAND but at bottom) - treat as TOP_BAND for rendering
    # ==========================================================================
    if gemini_type == 'bottom_band' or (width_ratio >= 0.55 and 0.15 <= height_ratio <= 0.60 and is_near_bottom and touches_bottom):
        reason = f"TOP_BAND (bottom variant): width={width_ratio:.2%}, height={height_ratio:.2%}"
        print(f"     ✅ {reason}")
        return 'TOP_BAND', reason, bbox_area_ratio
    
    # ==========================================================================
    # RULE 5: CENTER_BOX - Large centered webcam (treat as TOP_BAND for rendering)
    # ==========================================================================
    if gemini_type == 'center_box':
        if bbox_area_ratio >= 0.20:  # Reasonably large
            reason = f"TOP_BAND (center_box): area={bbox_area_ratio:.2%}, gemini_type={gemini_type}"
            print(f"     ✅ {reason}")
            return 'TOP_BAND', reason, bbox_area_ratio
    
    # ==========================================================================
    # RULE 6: SPLIT - Classic corner overlay
    # Conditions: in corner zone, area 3-35%
    # ==========================================================================
    is_corner_overlay = (
        (gemini_type == 'corner_overlay') or
        (is_in_corner and 0.03 <= bbox_area_ratio <= 0.35)
    )
    
    if is_corner_overlay:
        reason = f"SPLIT: corner overlay, area={bbox_area_ratio:.2%}, corner={corner}, gemini_type={gemini_type}"
        print(f"     📐 {reason}")
        return 'SPLIT', reason, bbox_area_ratio
    
    # ==========================================================================
    # FALLBACK: If we have a webcam but it doesn't fit other categories
    # ==========================================================================
    if bbox_area_ratio >= 0.10:
        # Larger webcams that don't fit other categories -> TOP_BAND
        reason = f"TOP_BAND (fallback): area={bbox_area_ratio:.2%} >= 10%"
        print(f"     📐 {reason}")
        return 'TOP_BAND', reason, bbox_area_ratio
    
    # Small webcam not in corner -> treat as SPLIT anyway
    reason = f"SPLIT (fallback): area={bbox_area_ratio:.2%}"
    print(f"     📐 {reason}")
    return 'SPLIT', reason, bbox_area_ratio


def _compute_edge_density_outside(
    frame_bgr: np.ndarray,
    bx: int, by: int, bw: int, bh: int,
    video_width: int, video_height: int,
) -> float:
    """
    Compute edge density in the area OUTSIDE the webcam bbox.
    
    Low edge density suggests the outside area is empty/simple (supports FULL_CAM).
    High edge density suggests game UI/content exists (supports SPLIT).
    
    Returns:
        Edge density as a ratio (0.0 to 1.0)
    """
    # Convert to grayscale
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    
    # Apply Canny edge detection
    edges = cv2.Canny(gray, 50, 150)
    
    # Create mask for OUTSIDE the bbox
    mask = np.ones_like(edges, dtype=np.uint8) * 255
    # Zero out the webcam bbox area
    mask[by:by+bh, bx:bx+bw] = 0
    
    # Count edge pixels outside bbox
    edges_outside = cv2.bitwise_and(edges, edges, mask=mask)
    edge_count = np.count_nonzero(edges_outside)
    
    # Count total pixels outside bbox
    outside_pixels = (video_width * video_height) - (bw * bh)
    
    if outside_pixels <= 0:
        return 0.0
    
    return edge_count / outside_pixels


def detect_face_in_frame(
    frame_bgr: np.ndarray,
    roi_x: int = 0,
    roi_y: int = 0,
    roi_w: int = 0,
    roi_h: int = 0,
) -> Optional[FaceCenter]:
    """
    Detect face in frame (or ROI) and return center point.
    
    Used for face-centered cropping in FULL_CAM mode.
    
    Args:
        frame_bgr: BGR frame
        roi_x, roi_y, roi_w, roi_h: Region of interest (0 = full frame)
    
    Returns:
        FaceCenter or None if no face detected
    """
    if roi_w == 0 or roi_h == 0:
        roi_w = frame_bgr.shape[1]
        roi_h = frame_bgr.shape[0]
    
    # Try DNN face detection first (more accurate)
    face = detect_face_dnn(frame_bgr, roi_x, roi_y, roi_w, roi_h)
    if face:
        return FaceCenter(
            x=face.x,
            y=face.y,
            width=face.width,
            height=face.height
        )
    
    # Fallback to Haar Cascade
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
    
    # Detect faces
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(50, 50)
    )
    
    if len(faces) == 0:
        return None
    
    # Return largest face
    largest = max(faces, key=lambda f: f[2] * f[3])
    x, y, w, h = largest
    
    return FaceCenter(x=x, y=y, width=w, height=h)


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


def compute_iou_dict(box1: Dict[str, int], box2: Dict[str, int]) -> float:
    """
    Compute Intersection over Union between two dict bounding boxes.
    
    Args:
        box1: Dict with x, y, width, height
        box2: Dict with x, y, width, height
        
    Returns:
        IoU value (0.0 to 1.0)
    """
    x1 = max(box1['x'], box2['x'])
    y1 = max(box1['y'], box2['y'])
    x2 = min(box1['x'] + box1['width'], box2['x'] + box2['width'])
    y2 = min(box1['y'] + box1['height'], box2['y'] + box2['height'])
    
    if x2 <= x1 or y2 <= y1:
        return 0.0
    
    intersection = (x2 - x1) * (y2 - y1)
    area1 = box1['width'] * box1['height']
    area2 = box2['width'] * box2['height']
    union = area1 + area2 - intersection
    
    return intersection / union if union > 0 else 0.0


def is_hud_like_box(
    bbox: Dict[str, int],
    corner: str,
    video_width: int,
    video_height: int,
    frame_bgr: np.ndarray = None,
) -> Tuple[bool, str]:
    """
    Check if a bbox looks like a HUD/minimap element rather than a webcam.
    
    HUD indicators (reject if ALL are true):
    1. In typical HUD zone (e.g., top-left stream often has minimap mid-left)
    2. No face detected inside bbox (light DNN pass)
    3. Near-square aspect ratio (minimaps are often square)
    4. High edge density / UI-like appearance
    
    Args:
        bbox: Dict with x, y, width, height
        corner: Expected corner
        video_width: Video width
        video_height: Video height
        frame_bgr: Optional frame for face detection and edge analysis
    
    Returns:
        Tuple of (is_hud_like, reason)
    """
    x, y, w, h = bbox['x'], bbox['y'], bbox['width'], bbox['height']
    
    # Check 1: Position in typical HUD zones
    # For top-left streams, minimap is often mid-left (x < 40%W, y > 20%H)
    x_ratio = (x + w/2) / video_width
    y_ratio = (y + h/2) / video_height
    
    is_in_hud_zone = False
    hud_zone_reason = ""
    
    if corner == 'top-left':
        # Minimap is typically below the webcam, mid-left area
        # HUD zone: x < 40%W AND y > 22%H (below typical webcam)
        if x_ratio < 0.40 and y_ratio > 0.22:
            is_in_hud_zone = True
            hud_zone_reason = f"mid-left HUD zone (x_ratio={x_ratio:.2f}, y_ratio={y_ratio:.2f})"
    elif corner == 'top-right':
        # Similar check for top-right streams
        if x_ratio > 0.60 and y_ratio > 0.22:
            is_in_hud_zone = True
            hud_zone_reason = f"mid-right HUD zone (x_ratio={x_ratio:.2f}, y_ratio={y_ratio:.2f})"
    
    if not is_in_hud_zone:
        return False, "not in HUD zone"
    
    # Check 2: Aspect ratio - minimaps are often near-square
    ar = w / h if h > 0 else 0
    is_near_square = 0.80 <= ar <= 1.25
    
    # Check 3: Try face detection inside bbox (if we have a frame)
    has_face = False
    if frame_bgr is not None:
        try:
            face = detect_face_dnn(frame_bgr, x, y, w, h)
            has_face = face is not None
        except:
            pass
    
    # Check 4: Edge density (HUD elements have high edge density from UI lines)
    has_high_edge_density = False
    if frame_bgr is not None:
        try:
            roi = frame_bgr[y:y+h, x:x+w]
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 50, 150)
            edge_density = np.count_nonzero(edges) / (w * h)
            # HUD elements typically have edge density > 15%
            has_high_edge_density = edge_density > 0.15
        except:
            pass
    
    # Reject as HUD if: in HUD zone AND near-square AND no face
    if is_in_hud_zone and is_near_square and not has_face:
        return True, f"HUD-like: {hud_zone_reason}, near-square AR={ar:.2f}, no face detected"
    
    # Also reject if in HUD zone with very high edge density and no face
    if is_in_hud_zone and has_high_edge_density and not has_face:
        return True, f"HUD-like: {hud_zone_reason}, high edge density, no face detected"
    
    return False, "passed HUD checks"


def check_corner_proximity(
    bbox: Dict[str, int],
    corner: str,
    video_width: int,
    video_height: int,
    max_inset_ratio: float = 0.06,
) -> Tuple[bool, str]:
    """
    Check if a bbox is properly positioned near its expected corner.
    
    Webcam overlays should be close to their corner edges. This prevents
    selecting HUD elements like minimaps that are elsewhere on screen.
    
    STRICT Y CONSTRAINTS added to prevent minimap false positives:
    - top-left/top-right: y must be <= 18% of frame height (or <= 22% with looser check)
    - bottom-left/bottom-right: y must be >= 55% of frame height
    
    Args:
        bbox: Dict with x, y, width, height
        corner: Expected corner ('top-left', 'top-right', etc.)
        video_width: Video width
        video_height: Video height
        max_inset_ratio: Max distance from edge as ratio (0.06 = 6%)
        
    Returns:
        Tuple of (is_near_corner, reason)
    """
    x, y, w, h = bbox['x'], bbox['y'], bbox['width'], bbox['height']
    
    # Calculate max inset in pixels (6% of smaller dimension)
    max_inset = int(max_inset_ratio * min(video_width, video_height))
    
    # STRICT Y constraints to prevent minimap drift
    max_y_top = int(video_height * 0.22)  # Top webcams must start within top 22%
    min_y_bottom = int(video_height * 0.55)  # Bottom webcams must start at 55%+
    
    right_edge = x + w
    bottom_edge = y + h
    
    if corner == 'top-left':
        # STRICT: y must be near top (within 22% of frame height)
        if y > max_y_top:
            return False, f"top-left: y={y} > max_y_top={max_y_top} (not near top edge)"
        # x should be near left edge (allow some offset)
        if x > max_inset * 2:
            return False, f"top-left: x={x} > {max_inset * 2} (not near left edge)"
        # Also check: at least one edge should be very close to corner
        if x > max_inset and y > max_inset:
            return False, f"top-left: neither edge touches corner (x={x}, y={y})"
    
    elif corner == 'top-right':
        # STRICT: y must be near top
        if y > max_y_top:
            return False, f"top-right: y={y} > max_y_top={max_y_top} (not near top edge)"
        # Right edge should be near video width
        if right_edge < (video_width - max_inset) and y > max_inset:
            return False, f"top-right: neither edge near corner (right={right_edge}, y={y})"
        if y > max_inset * 3:
            return False, f"top-right: too far from top (y={y})"
    
    elif corner == 'bottom-left':
        if x > max_inset and bottom_edge < (video_height - max_inset):
            return False, f"bottom-left: neither edge near corner"
        if bottom_edge < (video_height - max_inset * 3):
            return False, f"bottom-left: too far from bottom"
    
    elif corner == 'bottom-right':
        if right_edge < (video_width - max_inset) and bottom_edge < (video_height - max_inset):
            return False, f"bottom-right: neither edge near corner"
        if bottom_edge < (video_height - max_inset * 3):
            return False, f"bottom-right: too far from bottom"
    
    return True, "near expected corner"


def is_true_corner_overlay(
    bbox: Dict[str, int],
    frame_width: int,
    frame_height: int,
    corner: str,
    edge_threshold_ratio: float = 0.02,
) -> Tuple[bool, str]:
    """
    Check if a bbox is truly attached to its corner edges.
    
    A "true corner" overlay has edges touching or very close to (within 2%) 
    the frame edges. Mid-right/mid-left overlays are NOT true corners even 
    if Gemini says "corner_overlay".
    
    Args:
        bbox: Dict with x, y, width, height
        frame_width: Video frame width
        frame_height: Video frame height
        corner: Expected corner ('top-left', 'top-right', etc.)
        edge_threshold_ratio: Max gap ratio to consider "touching" (0.02 = 2%)
        
    Returns:
        Tuple of (is_true_corner, debug_reason)
    """
    x, y, w, h = bbox['x'], bbox['y'], bbox['width'], bbox['height']
    
    # Calculate thresholds in pixels (2% of dimension)
    threshold_x = int(frame_width * edge_threshold_ratio)
    threshold_y = int(frame_height * edge_threshold_ratio)
    
    # Calculate gaps to each edge
    gap_left = x
    gap_right = frame_width - (x + w)
    gap_top = y
    gap_bottom = frame_height - (y + h)
    
    debug_info = f"gaps: L={gap_left}, R={gap_right}, T={gap_top}, B={gap_bottom}, thresh_x={threshold_x}, thresh_y={threshold_y}"
    
    if corner == 'top-right':
        # True top-right: right edge touches right AND top edge touches top
        touches_right = gap_right <= threshold_x
        touches_top = gap_top <= threshold_y
        is_true = touches_right and touches_top
        reason = f"top-right: touches_right={touches_right} (gap={gap_right}), touches_top={touches_top} (gap={gap_top})"
        return is_true, f"{reason} | {debug_info}"
    
    elif corner == 'top-left':
        touches_left = gap_left <= threshold_x
        touches_top = gap_top <= threshold_y
        is_true = touches_left and touches_top
        reason = f"top-left: touches_left={touches_left} (gap={gap_left}), touches_top={touches_top} (gap={gap_top})"
        return is_true, f"{reason} | {debug_info}"
    
    elif corner == 'bottom-right':
        touches_right = gap_right <= threshold_x
        touches_bottom = gap_bottom <= threshold_y
        is_true = touches_right and touches_bottom
        reason = f"bottom-right: touches_right={touches_right}, touches_bottom={touches_bottom}"
        return is_true, f"{reason} | {debug_info}"
    
    elif corner == 'bottom-left':
        touches_left = gap_left <= threshold_x
        touches_bottom = gap_bottom <= threshold_y
        is_true = touches_left and touches_bottom
        reason = f"bottom-left: touches_left={touches_left}, touches_bottom={touches_bottom}"
        return is_true, f"{reason} | {debug_info}"
    
    # Unknown corner
    return False, f"unknown corner: {corner}"


def tighten_webcam_bbox_contours(
    frame_bgr: np.ndarray,
    initial_bbox: Dict[str, int],
    frame_width: int,
    frame_height: int,
    face_center: Optional[Tuple[int, int]] = None,
    expand_ratio: float = 0.15,
    debug: bool = False,
) -> Optional[Dict[str, int]]:
    """
    Tighten a webcam bbox using contour/edge detection to find the actual webcam rectangle.
    
    This is useful when Gemini returns a rough bbox that includes some game UI.
    We search for rectangular contours within an expanded search window.
    
    Args:
        frame_bgr: BGR frame image
        initial_bbox: Initial bbox from Gemini
        frame_width: Frame width
        frame_height: Frame height
        face_center: Optional (x, y) of detected face center (for validation)
        expand_ratio: How much to expand search window (0.15 = 15%)
        debug: Enable debug logging
        
    Returns:
        Tightened bbox dict or None if no good candidate found
    """
    x, y, w, h = initial_bbox['x'], initial_bbox['y'], initial_bbox['width'], initial_bbox['height']
    
    # Expand search window
    expand_x = int(w * expand_ratio)
    expand_y = int(h * expand_ratio)
    
    search_x = max(0, x - expand_x)
    search_y = max(0, y - expand_y)
    search_w = min(frame_width - search_x, w + 2 * expand_x)
    search_h = min(frame_height - search_y, h + 2 * expand_y)
    
    # Extract search region
    roi = frame_bgr[search_y:search_y+search_h, search_x:search_x+search_w]
    if roi.size == 0:
        if debug:
            print(f"  ⚠️ Empty ROI for contour search")
        return None
    
    # Convert to grayscale and detect edges
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 30, 100)
    
    # Morphological closing to connect edge fragments
    kernel = np.ones((5, 5), np.uint8)
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)
    
    # Find contours
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        if debug:
            print(f"  ⚠️ No contours found in search region")
        return None
    
    # Score candidates
    candidates = []
    min_area = w * h * 0.3  # At least 30% of initial bbox area
    max_area = search_w * search_h * 0.95  # Not the entire search region
    
    for contour in contours:
        # Get bounding rectangle
        cx, cy, cw, ch = cv2.boundingRect(contour)
        area = cw * ch
        
        # Filter by area
        if area < min_area or area > max_area:
            continue
        
        # Filter by aspect ratio (webcams are typically 1.2 to 3.0)
        ar = cw / ch if ch > 0 else 0
        if ar < 1.2 or ar > 3.0:
            continue
        
        # Convert to full frame coordinates
        full_x = search_x + cx
        full_y = search_y + cy
        
        # Rectangularity score (contour area vs bounding rect area)
        contour_area = cv2.contourArea(contour)
        rectangularity = contour_area / area if area > 0 else 0
        
        # Score: prefer rectangular, centered on initial, containing face
        center_dist = abs((full_x + cw/2) - (x + w/2)) + abs((full_y + ch/2) - (y + h/2))
        center_score = 1.0 / (1.0 + center_dist / 100)
        
        # Face containment bonus
        face_score = 0.0
        if face_center:
            fx, fy = face_center
            if full_x <= fx <= full_x + cw and full_y <= fy <= full_y + ch:
                face_score = 0.3  # Bonus for containing face
        
        score = rectangularity * 0.4 + center_score * 0.3 + face_score + (area / max_area) * 0.1
        
        candidates.append({
            'x': full_x, 'y': full_y, 'width': cw, 'height': ch,
            'score': score, 'ar': ar, 'rect': rectangularity
        })
    
    if not candidates:
        if debug:
            print(f"  ⚠️ No valid contour candidates")
        return None
    
    # Sort by score and return best
    candidates.sort(key=lambda c: c['score'], reverse=True)
    best = candidates[0]
    
    if debug:
        print(f"  📐 Contour refinement: {len(candidates)} candidates, best score={best['score']:.3f}")
        print(f"     Best: {best['width']}x{best['height']} at ({best['x']},{best['y']}), AR={best['ar']:.2f}")
    
    return {
        'x': best['x'], 'y': best['y'],
        'width': best['width'], 'height': best['height']
    }


def refine_side_box_with_face_edges(
    frame_bgr: np.ndarray,
    initial_bbox: Dict[str, int],
    face_bbox: Optional[Dict[str, int]],
    frame_width: int,
    frame_height: int,
    debug: bool = False,
) -> Optional[Dict[str, int]]:
    """
    Refine a side_box webcam bbox using face-anchored edge projection.
    
    This is the key algorithm for inset/floating webcams that don't touch edges.
    Uses the face position to define a search ROI, then finds the true webcam
    rectangle by analyzing edge projections (column/row sums of edge map).
    
    Algorithm:
    1. Define ROI centered on face (or initial bbox if no face)
    2. Compute edge map (Canny) within ROI
    3. Calculate column sums (vertical edges) and row sums (horizontal edges)
    4. Find strongest border lines left/right/top/bottom of face center
    5. Validate the resulting rectangle
    
    Args:
        frame_bgr: BGR frame image
        initial_bbox: Initial bbox from Gemini (used as fallback and ROI hint)
        face_bbox: Detected face bbox (x, y, width, height) - critical for accuracy
        frame_width: Full frame width
        frame_height: Full frame height
        debug: Enable debug logging
        
    Returns:
        Refined bbox dict or None if refinement failed
    """
    bx, by, bw, bh = initial_bbox['x'], initial_bbox['y'], initial_bbox['width'], initial_bbox['height']
    
    # Get face center (use initial bbox center if no face detected)
    if face_bbox:
        fx = face_bbox['x'] + face_bbox['width'] // 2
        fy = face_bbox['y'] + face_bbox['height'] // 2
        face_w = face_bbox['width']
        face_h = face_bbox['height']
    else:
        # Fallback: assume face is centered in initial bbox, upper portion
        fx = bx + bw // 2
        fy = by + int(bh * 0.35)  # Face usually in upper third
        face_w = int(bw * 0.25)
        face_h = int(bh * 0.25)
    
    if debug:
        print(f"  🔬 Face-anchored edge refinement:")
        print(f"     Face center: ({fx}, {fy}), initial bbox: {bw}x{bh} at ({bx},{by})")
    
    # Define ROI around face - expand to capture full webcam box
    # Use larger multipliers to ensure we capture the full webcam frame
    roi_w = min(frame_width, max(int(bw * 2.2), int(frame_width * 0.5)))
    roi_h = min(frame_height, max(int(bh * 2.0), int(frame_height * 0.5)))
    
    # Center ROI on face
    roi_x = max(0, fx - roi_w // 2)
    roi_y = max(0, fy - roi_h // 2)
    
    # Clamp ROI to frame
    roi_x = min(roi_x, frame_width - roi_w)
    roi_y = min(roi_y, frame_height - roi_h)
    roi_w = min(roi_w, frame_width - roi_x)
    roi_h = min(roi_h, frame_height - roi_y)
    
    if roi_w < 100 or roi_h < 100:
        if debug:
            print(f"     ⚠️ ROI too small: {roi_w}x{roi_h}")
        return None
    
    # Extract ROI
    roi = frame_bgr[roi_y:roi_y+roi_h, roi_x:roi_x+roi_w]
    
    # Convert to grayscale and compute edges
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    
    # Edge projections
    col_sum = edges.sum(axis=0).astype(float)  # Vertical edges (left/right borders)
    row_sum = edges.sum(axis=1).astype(float)  # Horizontal edges (top/bottom borders)
    
    # Smooth with moving average (window ~15-21)
    window_size = min(21, roi_w // 10, roi_h // 10)
    if window_size > 3:
        kernel = np.ones(window_size) / window_size
        col_sum = np.convolve(col_sum, kernel, mode='same')
        row_sum = np.convolve(row_sum, kernel, mode='same')
    
    # Face position in ROI coordinates
    face_x_roi = fx - roi_x
    face_y_roi = fy - roi_y
    
    # Find borders using peak detection
    # Left border: strongest peak LEFT of face center
    left_region = col_sum[:face_x_roi]
    if len(left_region) > 10:
        # Find peaks in left region
        left_peaks = []
        for i in range(5, len(left_region) - 5):
            if col_sum[i] > col_sum[i-5] and col_sum[i] > col_sum[i+5]:
                left_peaks.append((i, col_sum[i]))
        if left_peaks:
            # Take the strongest peak
            left_peaks.sort(key=lambda x: x[1], reverse=True)
            left_border_roi = left_peaks[0][0]
        else:
            # Fallback: use 10% from left
            left_border_roi = int(face_x_roi * 0.15)
    else:
        left_border_roi = 0
    
    # Right border: strongest peak RIGHT of face center
    right_region = col_sum[face_x_roi:]
    if len(right_region) > 10:
        right_peaks = []
        for i in range(5, len(right_region) - 5):
            idx = face_x_roi + i
            if col_sum[idx] > col_sum[idx-5] and col_sum[idx] > col_sum[idx+5]:
                right_peaks.append((idx, col_sum[idx]))
        if right_peaks:
            right_peaks.sort(key=lambda x: x[1], reverse=True)
            right_border_roi = right_peaks[0][0]
        else:
            # Fallback: use 85% towards right
            right_border_roi = face_x_roi + int(len(right_region) * 0.85)
    else:
        right_border_roi = roi_w - 1
    
    # Top border: strongest peak ABOVE face center
    top_region = row_sum[:face_y_roi]
    if len(top_region) > 10:
        top_peaks = []
        for i in range(5, len(top_region) - 5):
            if row_sum[i] > row_sum[i-5] and row_sum[i] > row_sum[i+5]:
                top_peaks.append((i, row_sum[i]))
        if top_peaks:
            top_peaks.sort(key=lambda x: x[1], reverse=True)
            top_border_roi = top_peaks[0][0]
        else:
            top_border_roi = int(face_y_roi * 0.15)
    else:
        top_border_roi = 0
    
    # Bottom border: strongest peak BELOW face center
    bottom_region = row_sum[face_y_roi:]
    if len(bottom_region) > 10:
        bottom_peaks = []
        for i in range(5, len(bottom_region) - 5):
            idx = face_y_roi + i
            if row_sum[idx] > row_sum[idx-5] and row_sum[idx] > row_sum[idx+5]:
                bottom_peaks.append((idx, row_sum[idx]))
        if bottom_peaks:
            bottom_peaks.sort(key=lambda x: x[1], reverse=True)
            bottom_border_roi = bottom_peaks[0][0]
        else:
            bottom_border_roi = face_y_roi + int(len(bottom_region) * 0.85)
    else:
        bottom_border_roi = roi_h - 1
    
    # Convert back to full frame coordinates
    ref_x = roi_x + left_border_roi
    ref_y = roi_y + top_border_roi
    ref_w = right_border_roi - left_border_roi
    ref_h = bottom_border_roi - top_border_roi
    
    if debug:
        print(f"     ROI: {roi_w}x{roi_h} at ({roi_x},{roi_y})")
        print(f"     Borders (ROI): L={left_border_roi}, R={right_border_roi}, T={top_border_roi}, B={bottom_border_roi}")
        print(f"     Raw refined: {ref_w}x{ref_h} at ({ref_x},{ref_y})")
    
    # Validation
    # 1. Minimum size: width >= 15% frame_w, height >= 12% frame_h
    min_width = int(frame_width * 0.15)
    min_height = int(frame_height * 0.12)
    
    if ref_w < min_width or ref_h < min_height:
        if debug:
            print(f"     ⚠️ Refined bbox too small: {ref_w}x{ref_h} (min: {min_width}x{min_height})")
        # Try fallback: face-anchored expansion with target AR
        return _face_anchored_fallback(fx, fy, face_w, face_h, frame_width, frame_height, debug)
    
    # 2. Aspect ratio check: 1.2 <= AR <= 3.5
    ar = ref_w / ref_h if ref_h > 0 else 0
    if ar < 1.2 or ar > 3.5:
        if debug:
            print(f"     ⚠️ Bad aspect ratio: {ar:.2f} (expected 1.2-3.5)")
        # Try fallback
        return _face_anchored_fallback(fx, fy, face_w, face_h, frame_width, frame_height, debug)
    
    # 3. Must contain face center
    if not (ref_x <= fx <= ref_x + ref_w and ref_y <= fy <= ref_y + ref_h):
        if debug:
            print(f"     ⚠️ Refined bbox doesn't contain face center")
        return _face_anchored_fallback(fx, fy, face_w, face_h, frame_width, frame_height, debug)
    
    # 4. Clamp to frame
    ref_x = max(0, min(ref_x, frame_width - ref_w))
    ref_y = max(0, min(ref_y, frame_height - ref_h))
    ref_w = min(ref_w, frame_width - ref_x)
    ref_h = min(ref_h, frame_height - ref_y)
    
    if debug:
        print(f"     ✅ Refined bbox: {ref_w}x{ref_h} at ({ref_x},{ref_y}), AR={ar:.2f}")
    
    return {
        'x': ref_x, 'y': ref_y,
        'width': ref_w, 'height': ref_h
    }


def _face_anchored_fallback(
    fx: int, fy: int, 
    face_w: int, face_h: int,
    frame_width: int, frame_height: int,
    debug: bool = False
) -> Optional[Dict[str, int]]:
    """
    Fallback when edge refinement fails: create a box around face with target AR.
    
    Uses 16:9-ish aspect ratio (1.78) which is common for webcams.
    """
    # Target AR ~1.78 (16:9)
    target_ar = 1.78
    
    # Base size on face (face is typically ~20-35% of webcam height)
    box_h = int(face_h * 3.5)  # Face ~28% of height
    box_w = int(box_h * target_ar)
    
    # Ensure minimum size
    box_w = max(box_w, int(frame_width * 0.20))
    box_h = max(box_h, int(frame_height * 0.15))
    
    # Recalculate with AR
    box_w = int(box_h * target_ar)
    
    # Center on face (face usually in upper portion)
    box_x = fx - box_w // 2
    box_y = fy - int(box_h * 0.30)  # Face at ~30% from top
    
    # Clamp to frame
    box_x = max(0, min(box_x, frame_width - box_w))
    box_y = max(0, min(box_y, frame_height - box_h))
    box_w = min(box_w, frame_width - box_x)
    box_h = min(box_h, frame_height - box_y)
    
    if debug:
        print(f"     📐 Fallback face-anchored: {box_w}x{box_h} at ({box_x},{box_y})")
    
    return {
        'x': box_x, 'y': box_y,
        'width': box_w, 'height': box_h
    }


def refine_side_box_multiframe(
    video_path: str,
    initial_bbox: Dict[str, int],
    frame_width: int,
    frame_height: int,
    timestamps: List[float] = [3.0, 10.0, 15.0],
    debug: bool = False,
) -> Optional[Dict[str, int]]:
    """
    Run face-anchored edge refinement on multiple frames and take median.
    
    This provides more stable results by averaging out frame-specific noise.
    
    Args:
        video_path: Path to video file
        initial_bbox: Initial bbox from Gemini
        frame_width: Frame width
        frame_height: Frame height
        timestamps: List of timestamps to sample
        debug: Enable debug logging
        
    Returns:
        Median-refined bbox or None if all frames failed
    """
    refined_boxes = []
    
    for ts in timestamps:
        frame = extract_frame(video_path, ts)
        if frame is None:
            continue
        
        # Detect face in frame
        roi_x = max(0, initial_bbox['x'] - 50)
        roi_y = max(0, initial_bbox['y'] - 50)
        roi_w = min(frame_width - roi_x, initial_bbox['width'] + 100)
        roi_h = min(frame_height - roi_y, initial_bbox['height'] + 100)
        
        face = detect_face_dnn(frame, roi_x, roi_y, roi_w, roi_h)
        face_bbox = None
        if face:
            face_bbox = {
                'x': face.x, 'y': face.y,
                'width': face.width, 'height': face.height
            }
        
        # Run refinement
        refined = refine_side_box_with_face_edges(
            frame, initial_bbox, face_bbox,
            frame_width, frame_height,
            debug=debug and ts == timestamps[0]  # Only debug first frame
        )
        
        if refined:
            refined_boxes.append(refined)
            if debug:
                print(f"     Frame {ts}s: {refined['width']}x{refined['height']} at ({refined['x']},{refined['y']})")
    
    if not refined_boxes:
        if debug:
            print(f"  ⚠️ Multi-frame refinement failed on all {len(timestamps)} frames")
        return None
    
    # Take median of x, y, width, height
    xs = [b['x'] for b in refined_boxes]
    ys = [b['y'] for b in refined_boxes]
    ws = [b['width'] for b in refined_boxes]
    hs = [b['height'] for b in refined_boxes]
    
    median_bbox = {
        'x': int(np.median(xs)),
        'y': int(np.median(ys)),
        'width': int(np.median(ws)),
        'height': int(np.median(hs))
    }
    
    if debug:
        print(f"  ✅ Multi-frame median ({len(refined_boxes)}/{len(timestamps)} frames): " +
              f"{median_bbox['width']}x{median_bbox['height']} at ({median_bbox['x']},{median_bbox['y']})")
    
    return median_bbox


def convert_numpy_to_python(obj):
    """
    Recursively convert numpy types to Python native types for JSON serialization.
    
    Args:
        obj: Object to convert (can be dict, list, numpy type, or primitive)
        
    Returns:
        Object with all numpy types converted to Python types
    """
    if isinstance(obj, dict):
        return {k: convert_numpy_to_python(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_to_python(item) for item in obj]
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.bool_,)):
        return bool(obj)
    else:
        return obj


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


def _score_face_for_fullcam(face: FaceDetection, frame_w: int, frame_h: int) -> float:
    """
    Score a face for FULL_CAM streamer selection.
    
    CRITICAL: This scoring is designed to REJECT background people and SELECT the streamer.
    
    Streamers typically:
    - Are in the LOWER portion of the frame (seated at desk, y_ratio > 0.50)
    - Are on the LEFT or CENTER-LEFT (typical desk setup, x_ratio 0.20-0.50)
    - Have LARGER faces (closer to camera than background people)
    
    Background people typically:
    - Are in the UPPER portion (standing, y_ratio < 0.40)
    - Are on the RIGHT side (walking behind, x_ratio > 0.60)
    - Have SMALLER faces (farther from camera)
    """
    face_area = face.width * face.height
    frame_area = frame_w * frame_h
    
    # =========================================================================
    # Factor 1: VERTICAL POSITION (most important)
    # Streamers sit at desk level = LOWER portion of frame
    # =========================================================================
    y_ratio = face.center_y / frame_h
    
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
    x_ratio = face.center_x / frame_w
    
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
    return (position_score * 0.45) + (horizontal_score * 0.35) + (size_score * 0.20)


def _detect_best_face_for_fullcam(
    frame_bgr: np.ndarray,
    frame_path: Optional[str] = None,  # For Gemini escalation
) -> Tuple[Optional[FaceDetection], str, float]:
    """
    Detect the best face for FULL_CAM mode (the main streamer, not background people).
    
    IMPORTANT: NO HAAR CASCADE for global seed selection!
    - DNN only for initial detection
    - Gemini escalation if DNN fails or produces unreliable result
    
    Returns:
        Tuple of (face, detector_used, fullcam_score)
        - face: FaceDetection or None
        - detector_used: 'DNN', 'GEMINI', or 'NONE'
        - fullcam_score: 0-1 reliability score for the seed
    """
    h, w = frame_bgr.shape[:2]
    frame_area = w * h
    
    print(f"  🔍 _detect_best_face_for_fullcam() ENTER - frame {w}x{h}")
    print(f"  ⚠️ NOTE: Haar CASCADE DISABLED for global seed (DNN + Gemini only)")
    
    # ==========================================================================
    # TRY DNN DETECTOR ONLY (no Haar for global seed)
    # ==========================================================================
    dnn_face = None
    dnn_score = 0.0
    
    print(f"  🚀 Trying DNN detector...")
    try:
        from face_tracking import detect_face_dnn
        
        dnn_result = detect_face_dnn(frame_bgr, confidence_threshold=0.35, try_preprocessing=True)
        
        if dnn_result is not None:
            fx, fy, fw, fh, conf = dnn_result
            print(f"  ✅ DNN RETURNED FACE: ({fx},{fy}) {fw}x{fh} conf={conf:.2f}")
            
            # Create FaceDetection and compute fullcam score
            dnn_face = FaceDetection(x=fx, y=fy, width=fw, height=fh, confidence=conf)
            dnn_score = _score_face_for_fullcam(dnn_face, w, h)
            
            y_ratio = dnn_face.center_y / h
            x_ratio = dnn_face.center_x / w
            print(f"  📊 DNN face fullcam_score={dnn_score:.3f} y_ratio={y_ratio:.2f} x_ratio={x_ratio:.2f}")
        else:
            print(f"  ⚠️ DNN returned None (no faces detected)")
            
    except ImportError as e:
        print(f"  ❌ ImportError: face_tracking module not available: {e}")
    except Exception as e:
        import traceback
        print(f"  ❌ DNN detection EXCEPTION: {type(e).__name__}: {e}")
        print(f"     Traceback: {traceback.format_exc()}")
    
    # ==========================================================================
    # SEED RELIABILITY CHECK
    # Reject DNN seed if:
    # - Score < 0.70 (unreliable)
    # - y_ratio < 0.45 (too high in frame - likely background)
    # ==========================================================================
    RELIABILITY_THRESHOLD = 0.70
    Y_RATIO_MIN = 0.45
    
    dnn_is_reliable = False
    if dnn_face is not None:
        y_ratio = dnn_face.center_y / h
        
        if dnn_score >= RELIABILITY_THRESHOLD and y_ratio >= Y_RATIO_MIN:
            dnn_is_reliable = True
            print(f"  ✅ DNN seed is RELIABLE (score={dnn_score:.2f} >= {RELIABILITY_THRESHOLD}, y={y_ratio:.2f} >= {Y_RATIO_MIN})")
        else:
            reasons = []
            if dnn_score < RELIABILITY_THRESHOLD:
                reasons.append(f"score={dnn_score:.2f} < {RELIABILITY_THRESHOLD}")
            if y_ratio < Y_RATIO_MIN:
                reasons.append(f"y_ratio={y_ratio:.2f} < {Y_RATIO_MIN} (too high in frame)")
            print(f"  ⚠️ DNN seed UNRELIABLE: {', '.join(reasons)}")
    
    # ==========================================================================
    # GEMINI ESCALATION if DNN failed or unreliable
    # ==========================================================================
    if not dnn_is_reliable and frame_path is not None:
        print(f"  🤖 Escalating to GEMINI anchor detection...")
        try:
            from gemini_vision import get_fullcam_anchor_bbox_with_gemini
            
            gemini_result = get_fullcam_anchor_bbox_with_gemini(
                frame_path=frame_path,
                video_width=w,
                video_height=h
            )
            
            if gemini_result and 'x' in gemini_result:
                gx = gemini_result['x']
                gy = gemini_result['y']
                gw = gemini_result['w']
                gh = gemini_result['h']
                gconf = gemini_result.get('confidence', 0.8)
                
                gemini_face = FaceDetection(x=gx, y=gy, width=gw, height=gh, confidence=gconf)
                gemini_score = _score_face_for_fullcam(gemini_face, w, h)
                
                gy_ratio = gemini_face.center_y / h
                gx_ratio = gemini_face.center_x / w
                
                print(f"  ✅ GEMINI RETURNED FACE: ({gx},{gy}) {gw}x{gh}")
                print(f"  📊 Gemini face fullcam_score={gemini_score:.3f} y_ratio={gy_ratio:.2f} x_ratio={gx_ratio:.2f}")
                
                # Use Gemini if it's better than DNN OR if DNN was unreliable
                use_gemini = False
                if dnn_face is None:
                    use_gemini = True
                    print(f"  🎯 Using GEMINI (DNN found nothing)")
                elif gemini_score > dnn_score + 0.10:
                    use_gemini = True
                    print(f"  🎯 Using GEMINI (score {gemini_score:.2f} > DNN {dnn_score:.2f} + 0.10)")
                elif not dnn_is_reliable and gemini_score >= RELIABILITY_THRESHOLD:
                    use_gemini = True
                    print(f"  🎯 Using GEMINI (DNN unreliable, Gemini score {gemini_score:.2f} >= {RELIABILITY_THRESHOLD})")
                
                if use_gemini:
                    return (gemini_face, 'GEMINI', gemini_score)
            else:
                print(f"  ⚠️ Gemini returned no valid bbox")
                
        except Exception as e:
            print(f"  ❌ Gemini escalation failed: {e}")
    
    # ==========================================================================
    # RETURN BEST AVAILABLE
    # ==========================================================================
    if dnn_face is not None:
        print(f"  🎯 Using DNN seed (detector=DNN, score={dnn_score:.3f})")
        return (dnn_face, 'DNN', dnn_score)
    
    print(f"  ❌ _detect_best_face_for_fullcam(): No reliable face found")
    return (None, 'NONE', 0.0)


def _detect_best_face_multiframe(
    video_path: str,
    timestamps: List[float] = None,
) -> Tuple[Optional[FaceDetection], str, float]:
    """
    Multi-frame bootstrap for FULL_CAM seed selection.
    
    Samples multiple frames and aggregates face detections to find the most
    consistent, reliable streamer face.
    
    Args:
        video_path: Path to video file
        timestamps: List of timestamps to sample (default: [1, 3, 5, 7] seconds)
    
    Returns:
        Tuple of (best_face, detector_used, score)
    """
    if timestamps is None:
        timestamps = [1.0, 3.0, 5.0, 7.0]
    
    print(f"  🎬 Multi-frame bootstrap: sampling {len(timestamps)} frames")
    
    # Collect face detections from each frame
    all_detections = []  # List of (face, detector, score, timestamp)
    
    for ts in timestamps:
        frame = extract_frame(video_path, ts)
        if frame is None:
            print(f"     t={ts}s: Could not extract frame")
            continue
        
        h, w = frame.shape[:2]
        
        # Save frame for potential Gemini escalation
        frame_path_temp = None
        try:
            import tempfile
            with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as f:
                cv2.imwrite(f.name, frame)
                frame_path_temp = f.name
        except:
            pass
        
        face, detector, score = _detect_best_face_for_fullcam(frame, frame_path_temp)
        
        # Clean up
        if frame_path_temp:
            try:
                os.unlink(frame_path_temp)
            except:
                pass
        
        if face:
            y_ratio = face.center_y / h
            x_ratio = face.center_x / w
            print(f"     t={ts}s: {detector} face at ({face.center_x},{face.center_y}) score={score:.2f} y={y_ratio:.2f}")
            all_detections.append((face, detector, score, ts, w, h))
        else:
            print(f"     t={ts}s: No face detected")
    
    if not all_detections:
        print(f"  ❌ Multi-frame: No faces found in any frame")
        return (None, 'NONE', 0.0)
    
    # ==========================================================================
    # AGGREGATION: Group faces by proximity and find most consistent
    # ==========================================================================
    # Simple approach: use the face with highest score that appears in multiple frames
    # More sophisticated: cluster by center position
    
    # For now, prefer faces that:
    # 1. Have high score (>= 0.70)
    # 2. Appear in multiple frames with consistent position
    
    # Group by approximate center (within 15% of frame width)
    face_groups = []  # List of lists
    
    for det in all_detections:
        face, detector, score, ts, w, h = det
        cx, cy = face.center_x, face.center_y
        
        # Find existing group within distance threshold
        added = False
        proximity_threshold = w * 0.15
        
        for group in face_groups:
            ref_face = group[0][0]
            ref_cx, ref_cy = ref_face.center_x, ref_face.center_y
            dist = ((cx - ref_cx) ** 2 + (cy - ref_cy) ** 2) ** 0.5
            
            if dist < proximity_threshold:
                group.append(det)
                added = True
                break
        
        if not added:
            face_groups.append([det])
    
    print(f"  📊 Multi-frame: Found {len(face_groups)} distinct face groups")
    
    # Score each group by: count * avg_score
    best_group = None
    best_group_score = 0
    
    for i, group in enumerate(face_groups):
        count = len(group)
        avg_score = sum(d[2] for d in group) / count
        group_score = count * avg_score
        
        # Get representative face (highest individual score in group)
        best_in_group = max(group, key=lambda d: d[2])
        face = best_in_group[0]
        h = best_in_group[5]
        w = best_in_group[4]
        y_ratio = face.center_y / h
        x_ratio = face.center_x / w
        
        print(f"     Group {i+1}: {count} detections, avg_score={avg_score:.2f}, group_score={group_score:.2f}")
        print(f"            Best: ({face.center_x},{face.center_y}) y={y_ratio:.2f} x={x_ratio:.2f}")
        
        if group_score > best_group_score:
            best_group_score = group_score
            best_group = group
    
    if best_group:
        # Return the best detection from the winning group
        best_det = max(best_group, key=lambda d: d[2])
        face, detector, score, ts, w, h = best_det
        print(f"  ✅ Multi-frame winner: {detector} at t={ts}s, score={score:.2f}")
        return (face, detector, score)
    
    return (None, 'NONE', 0.0)


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
) -> Tuple[bool, str]:
    """
    Check if Gemini's bbox is GOOD and should be kept without refinement.
    
    A good bbox:
    - Is positioned in the correct corner region (not strict edge touching)
    - Has reasonable size (12-38% of frame width, 10-35% of height)
    - Has reasonable aspect ratio (1.2 to 2.4)
    
    Args:
        bbox: Dict with x, y, width, height
        corner: Expected corner ('top-left', 'top-right', etc.)
        video_width: Frame width
        video_height: Frame height
    
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
    right_edge = x + w
    bottom_edge = y + h
    
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
    
    # ============================================================
    # LOOSER CORNER POSITION RULES (not strict edge touching)
    # ============================================================
    # For top-right: y near top, bbox in right portion of screen
    # For top-left: y near top, bbox in left portion of screen
    # These are looser than "must touch edge within 12px"
    
    if corner == 'top-right':
        # y must be near top (within 20px or 3% of height)
        max_y = max(20, int(video_height * 0.03))
        if y > max_y:
            return False, f"not near top (y={y} > {max_y})"
        
        # bbox must start in right 45% of screen (x >= 55%)
        min_x = int(video_width * 0.55)
        if x < min_x:
            return False, f"too far left (x={x} < {min_x})"
        
        # right edge must reach at least 80% of screen width
        min_right = int(video_width * 0.80)
        if right_edge < min_right:
            return False, f"right edge too far from edge (right={right_edge} < {min_right})"
    
    elif corner == 'top-left':
        max_y = max(20, int(video_height * 0.03))
        if y > max_y:
            return False, f"not near top (y={y} > {max_y})"
        
        # bbox must end in left 45% of screen (x+w <= 45%)
        max_right = int(video_width * 0.45)
        if right_edge > max_right:
            return False, f"extends too far right (right={right_edge} > {max_right})"
        
        # x must start within left 20%
        max_x = int(video_width * 0.20)
        if x > max_x:
            return False, f"too far from left edge (x={x} > {max_x})"
    
    elif corner == 'bottom-right':
        # bottom edge must be near bottom (within 20px or 3% of height)
        min_bottom = video_height - max(20, int(video_height * 0.03))
        if bottom_edge < min_bottom:
            return False, f"not near bottom"
        
        min_x = int(video_width * 0.55)
        if x < min_x:
            return False, f"too far left (x={x} < {min_x})"
        
        min_right = int(video_width * 0.80)
        if right_edge < min_right:
            return False, f"right edge too far from edge"
    
    elif corner == 'bottom-left':
        min_bottom = video_height - max(20, int(video_height * 0.03))
        if bottom_edge < min_bottom:
            return False, f"not near bottom"
        
        max_right = int(video_width * 0.45)
        if right_edge > max_right:
            return False, f"extends too far right"
        
        max_x = int(video_width * 0.20)
        if x > max_x:
            return False, f"too far from left edge"
    
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
        from gemini_vision import detect_webcam_with_gemini, extract_frame_for_analysis, save_debug_frame_with_box, save_debug_frame_multi_box
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
                            
                            # NEW: Extract gemini type and confidence
                            gemini_type = result.get('type', 'corner_overlay')
                            gemini_confidence = result.get('confidence', 0.5)
                            
                            # NEW: Get effective_type (may differ from gemini_type for mid-right overlays)
                            effective_type = result.get('effective_type', gemini_type)
                            
                            # Determine if this is a TRUE corner overlay (for guardrail decisions)
                            is_true_corner = (effective_type == 'corner_overlay')
                            if effective_type == 'side_box':
                                is_true_corner = False
                                print(f"  ⚠️ SIDE_BOX detected: corner guardrails will be DISABLED")
                            
                            cam_x, cam_y, cam_w, cam_h = gemini_x, gemini_y, gemini_w, gemini_h
                            
                            # Initialize variables that must always be defined
                            refinement_method = "gemini"  # Default: use Gemini as-is
                            detected_face = None
                            is_suspicious = False
                            
                            print(f"  📐 Gemini detection: type={gemini_type}, effective={effective_type}, conf={gemini_confidence:.2f}")
                            print(f"     is_true_corner={is_true_corner}")
                            
                            print(f"  📐 Gemini bbox: {gemini_w}x{gemini_h} at ({gemini_x},{gemini_y}), AR={gemini_ar:.2f}")
                            
                            # ============================================================
                            # CHECK 1: Is Gemini bbox GOOD? (should keep without refinement)
                            # ============================================================
                            is_good, good_reason = is_gemini_bbox_good(result, position, width, height)
                            
                            if is_good:
                                print(f"  ✅ Gemini bbox GOOD -> using Gemini: {good_reason}")
                                refinement_method = "gemini-good"
                                # Skip refinement, use smaller padding
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
                            # Store Gemini bbox for IoU gate checking later
                            gemini_bbox_for_iou = {
                                'x': gemini_x, 'y': gemini_y,
                                'width': gemini_w, 'height': gemini_h
                            }
                            
                            if refinement_method != "gemini-good":
                                # Only attempt refinement if bbox is SUSPICIOUS
                                # If not good but also not suspicious, just use Gemini with normal padding
                                if not is_suspicious:
                                    print(f"  📐 Gemini bbox not suspicious, using as-is (with normal padding)")
                                    refinement_method = "gemini"
                                else:
                                    # Load frame for refinement - try disk first, then extract from video
                                    frame_for_refine = cv2.imread(frame_path) if frame_path else None
                                    
                                    if frame_for_refine is None:
                                        print(f"  📹 Frame not on disk, extracting from video...")
                                        frame_for_refine = extract_frame(video_path, ts)
                                    
                                    if frame_for_refine is not None:
                                        # --------------------------------------------------------
                                        # Strategy 1: Try FACE-ANCHOR FIRST (for suspicious/square bboxes)
                                        # Face-anchor is more reliable when Gemini returns a face crop
                                        # --------------------------------------------------------
                                        print(f"  👤 Trying face-anchor FIRST (Gemini bbox suspicious: {suspicious_reason})...")
                                        
                                        face_bbox, detected_face = refine_webcam_bbox_face_anchor(
                                            frame_for_refine, position, width, height
                                        )
                                        
                                        if face_bbox:
                                            face_w = face_bbox['width']
                                            face_h = face_bbox['height']
                                            face_area = face_w * face_h
                                            
                                            # Gate 1: Corner proximity check
                                            near_corner, corner_reason = check_corner_proximity(
                                                face_bbox, position, width, height
                                            )
                                            if not near_corner:
                                                print(f"  ❌ Face-anchor REJECTED: {corner_reason}")
                                            else:
                                                # Gate 2: IoU overlap with Gemini bbox
                                                iou = compute_iou_dict(face_bbox, gemini_bbox_for_iou)
                                                if iou < 0.05:
                                                    # Check if at least some overlap (not necessarily IoU)
                                                    has_overlap = (
                                                        face_bbox['x'] < gemini_x + gemini_w and
                                                        face_bbox['x'] + face_w > gemini_x and
                                                        face_bbox['y'] < gemini_y + gemini_h and
                                                        face_bbox['y'] + face_h > gemini_y
                                                    )
                                                    if not has_overlap:
                                                        print(f"  ❌ Face-anchor REJECTED: no overlap with Gemini (IoU={iou:.3f})")
                                                    else:
                                                        print(f"  ⚠️ Low IoU ({iou:.3f}) but boxes overlap, allowing...")
                                                        # Accept with overlap
                                                        print(f"  ✅ Face-anchor accepted (corner + overlap)")
                                                        cam_x = face_bbox['x']
                                                        cam_y = face_bbox['y']
                                                        cam_w = face_bbox['width']
                                                        cam_h = face_bbox['height']
                                                        refinement_method = "face-anchor"
                                                else:
                                                    # Good IoU
                                                    print(f"  ✅ Face-anchor accepted (IoU={iou:.3f})")
                                                    cam_x = face_bbox['x']
                                                    cam_y = face_bbox['y']
                                                    cam_w = face_bbox['width']
                                                    cam_h = face_bbox['height']
                                                    refinement_method = "face-anchor"
                                        
                                        # --------------------------------------------------------
                                        # Strategy 2: Try contour refinement if face-anchor failed
                                        # --------------------------------------------------------
                                        if refinement_method == "gemini":
                                            print(f"  🔬 Face-anchor failed, trying contour refinement...")
                                            
                                            contour_bbox = run_multiframe_refinement(
                                                video_path, position, width, height,
                                                timestamps=[3.0, 10.0, 15.0],
                                                top_n_per_frame=5
                                            )
                                            
                                            if contour_bbox:
                                                # Gate 1: Existing guardrails
                                                passes_guardrail, guardrail_reason = check_guardrails(
                                                    contour_bbox, position, width, height
                                                )
                                                
                                                # Gate 2: Corner proximity
                                                near_corner, corner_reason = check_corner_proximity(
                                                    contour_bbox, position, width, height
                                                )
                                                
                                                # Gate 3: IoU overlap with Gemini
                                                iou = compute_iou_dict(contour_bbox, gemini_bbox_for_iou)
                                                has_overlap = iou >= 0.05 or (
                                                    contour_bbox['x'] < gemini_x + gemini_w and
                                                    contour_bbox['x'] + contour_bbox['width'] > gemini_x and
                                                    contour_bbox['y'] < gemini_y + gemini_h and
                                                    contour_bbox['y'] + contour_bbox['height'] > gemini_y
                                                )
                                                
                                                if not passes_guardrail:
                                                    print(f"  ❌ Contour REJECTED: guardrail failed ({guardrail_reason})")
                                                elif not near_corner:
                                                    print(f"  ❌ Contour REJECTED: {corner_reason}")
                                                elif not has_overlap:
                                                    print(f"  ❌ Contour REJECTED: no overlap with Gemini (IoU={iou:.3f})")
                                                else:
                                                    print(f"  ✅ Contour accepted (guardrail + corner + IoU={iou:.3f})")
                                                    cam_x = contour_bbox['x']
                                                    cam_y = contour_bbox['y']
                                                    cam_w = contour_bbox['width']
                                                    cam_h = contour_bbox['height']
                                                    refinement_method = "contour"
                                        
                                        # --------------------------------------------------------
                                        # Fallback: Use Gemini bbox (with validation)
                                        # --------------------------------------------------------
                                        if refinement_method == "gemini":
                                            print(f"  📐 Using Gemini bbox (all refinements failed or rejected)")
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
                            # FACE-ANCHORED EDGE REFINEMENT for SIDE_BOX overlays
                            # For inset/floating webcams, use edge projection to find
                            # the true webcam rectangle. Uses multi-frame median for stability.
                            # ============================================================
                            if not is_true_corner or effective_type == 'side_box':
                                print(f"  🔬 Attempting FACE-ANCHORED EDGE refinement for side_box overlay...")
                                print(f"     Initial bbox: {cam_w}x{cam_h} at ({cam_x},{cam_y})")
                                
                                # Use multi-frame refinement for stability
                                edge_refined = refine_side_box_multiframe(
                                    video_path,
                                    {'x': cam_x, 'y': cam_y, 'width': cam_w, 'height': cam_h},
                                    width, height,
                                    timestamps=[3.0, 10.0, 15.0],
                                    debug=True
                                )
                                
                                if edge_refined:
                                    # Validate: must contain face if we have one
                                    valid = True
                                    if detected_face:
                                        if not bbox_contains_face(edge_refined, detected_face, margin_ratio=0.05):
                                            print(f"  ⚠️ Edge-refined bbox doesn't contain face, keeping original")
                                            valid = False
                                    
                                    if valid:
                                        print(f"  ✅ Edge refinement accepted: {edge_refined['width']}x{edge_refined['height']} at ({edge_refined['x']},{edge_refined['y']})")
                                        cam_x = edge_refined['x']
                                        cam_y = edge_refined['y']
                                        cam_w = edge_refined['width']
                                        cam_h = edge_refined['height']
                                        refinement_method = f"{refinement_method}+edge"
                                else:
                                    print(f"  ⚠️ Edge refinement failed, keeping current bbox")
                            
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
                            # Padding strategy by type:
                            # - good Gemini corner: 6% (already touching edges)
                            # - side_box (edge-refined): 4-5% (minimal - edge refinement is precise)
                            # - other refined: 8-10% (moderate)
                            if refinement_method == "gemini-good":
                                padding_ratio = 0.06  # 6% for good Gemini corners
                            elif not is_true_corner or effective_type == 'side_box':
                                # SMALLER padding for side_box - refinement is more precise
                                # and padding is what causes game bleed
                                padding_ratio = 0.04  # 4% for side_box
                                print(f"  ℹ️ Using minimal padding (4%) for side_box to prevent bleed")
                            else:
                                padding_ratio = 0.08  # 8% for other refined bboxes
                            
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
                            # GAME BLEED GUARDRAIL (ONLY for TRUE corner overlays!)
                            # For mid-right/side_box overlays, this guardrail would clip
                            # the actual webcam, so we SKIP it.
                            # ============================================================
                            if is_true_corner:
                                if 'right' in position:
                                    min_x = int(width * 0.60)  # 1152 for 1920 width
                                    if new_x < min_x:
                                        # REDUCE width from LEFT, keep right edge fixed
                                        old_w = new_w
                                        old_right_edge = new_x + new_w
                                        new_w = old_right_edge - min_x
                                        new_x = min_x
                                        print(f"  ⚠️ Game bleed guardrail (corner): x {new_x - (old_w - new_w)} -> {new_x}, w {old_w} -> {new_w}")
                                else:
                                    max_right = int(width * 0.40)  # 768 for 1920 width
                                    if new_x + new_w > max_right:
                                        new_w = max_right - new_x
                                        print(f"  ⚠️ Game bleed guardrail (corner): w -> {new_w}")
                            else:
                                print(f"  ℹ️ Skipping game bleed guardrail (side_box overlay, is_true_corner=False)")
                            
                            # Ensure minimum dimensions
                            new_w = max(100, new_w)
                            new_h = max(100, new_h)
                            
                            print(f"  📐 FINAL bbox: {new_w}x{new_h} at ({new_x},{new_y})")
                            print(f"  ═══════════════════════════════════════\n")
                            
                            # ============================================================
                            # SAVE DEBUG IMAGES (initial + refined)
                            # ============================================================
                            # 1. Initial Gemini bbox (RED) - what Gemini detected
                            debug_initial_path = f"{temp_dir}/debug_webcam_detect_initial.jpg"
                            save_debug_frame_with_box(
                                frame_path, debug_initial_path,
                                gemini_x, gemini_y, gemini_w, gemini_h,
                                label="GEMINI", color=(0, 0, 255)  # Red
                            )
                            
                            # 2. Final refined bbox (GREEN) - what we'll actually use
                            debug_refined_path = f"{temp_dir}/debug_webcam_detect_refined.jpg"
                            save_debug_frame_with_box(
                                frame_path, debug_refined_path,
                                new_x, new_y, new_w, new_h,
                                label=f"FINAL ({refinement_method})", color=(0, 255, 0)  # Green
                            )
                            
                            # 3. Combined view with all boxes
                            debug_combined_path = f"{temp_dir}/debug_webcam_detection.jpg"
                            debug_boxes = [
                                {'x': gemini_x, 'y': gemini_y, 'w': gemini_w, 'h': gemini_h, 
                                 'label': 'Gemini', 'color': (0, 0, 255)},  # Red
                            ]
                            
                            # Add pre-padding box if different from Gemini
                            if (cam_x != gemini_x or cam_y != gemini_y or 
                                cam_w != gemini_w or cam_h != gemini_h):
                                debug_boxes.append({
                                    'x': cam_x, 'y': cam_y, 'w': cam_w, 'h': cam_h,
                                    'label': 'Refined', 'color': (0, 255, 255)  # Yellow
                                })
                            
                            # Final bbox
                            debug_boxes.append({
                                'x': new_x, 'y': new_y, 'w': new_w, 'h': new_h,
                                'label': 'Final', 'color': (0, 255, 0)  # Green
                            })
                            
                            # Face if detected
                            if detected_face:
                                debug_boxes.append({
                                    'x': detected_face.x, 'y': detected_face.y,
                                    'w': detected_face.width, 'h': detected_face.height,
                                    'label': 'Face', 'color': (255, 0, 0)  # Blue
                                })
                            
                            save_debug_frame_multi_box(frame_path, debug_combined_path, debug_boxes)
                            
                            # Clean up the frame file now that we're done
                            try:
                                if frame_path and os.path.exists(frame_path):
                                    os.remove(frame_path)
                            except:
                                pass
                            
                            return WebcamRegion(
                                x=new_x,
                                y=new_y,
                                width=new_w,
                                height=new_h,
                                position=position,
                                gemini_type=gemini_type,
                                gemini_confidence=gemini_confidence,
                            )
                else:
                    print(f"  ⚠️ Could not extract frame at {ts}s")
            
            if not gemini_explicitly_no_webcam:
                print("  ⚠️ Gemini didn't find webcam, falling back to face detection")
    except ImportError:
        print("  ⚠️ Gemini module not available")
    except Exception as e:
        print(f"  ⚠️ Gemini detection failed: {e}")
    
    # If Gemini explicitly said "no webcam overlay", TREAT AS FULL_CAM by default
    # (The entire frame might BE the webcam/camera - no overlay, just the person)
    # This is the key insight: if there's no game overlay, it's FULL_CAM even if face is small/angled
    if gemini_explicitly_no_webcam:
        print("  🔍 Gemini says no webcam overlay → checking for FULL_CAM...")
        print("  💡 IMPORTANT: If no overlay, treat as FULL_CAM unless we detect a game")
        
        # Extract a frame to check for face in full frame
        frame = extract_frame(video_path, 5.0)
        if frame is None:
            frame = extract_frame(video_path, 3.0)
        
        if frame is not None:
            height, width = frame.shape[:2]
            
            # Save frame for Gemini escalation
            import tempfile
            frame_path_temp = None
            try:
                with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as f:
                    cv2.imwrite(f.name, frame)
                    frame_path_temp = f.name
            except Exception as e:
                print(f"  ⚠️ Could not save temp frame for Gemini: {e}")
            
            # Detect best face using DNN + Gemini escalation (NO HAAR for global seed)
            face, detector_used, seed_score = _detect_best_face_for_fullcam(frame, frame_path_temp)
            
            # Clean up temp frame
            if frame_path_temp:
                try:
                    os.unlink(frame_path_temp)
                except:
                    pass
            
            # Calculate face metrics for logging
            face_ratio = 0.0
            face_center_x = 0
            face_center_y = 0
            is_centered_x = False
            is_in_frame = False
            
            if face:
                face_area = face.width * face.height
                frame_area = width * height
                face_ratio = face_area / frame_area
                
                face_center_x = face.center_x
                face_center_y = face.center_y
                
                is_centered_x = 0.10 * width < face_center_x < 0.90 * width
                is_in_frame = 0.05 * height < face_center_y < 0.95 * height
                
                print(f"  👤 Face detected for FULL_CAM (detector={detector_used}, seed_score={seed_score:.2f}):")
                print(f"     Face size: {face.width}x{face.height} ({face_ratio:.1%} of frame)")
                print(f"     Face center: ({face_center_x}, {face_center_y})")
                print(f"     In-frame: X={is_centered_x}, Y={is_in_frame}")
            
            # ==========================================================================
            # CRITICAL FIX: If Gemini says "no webcam overlay", classify as FULL_CAM
            # even if face is small/angled - we'll use Gemini anchor for tracking
            # ==========================================================================
            # FULL_CAM criteria (RELAXED):
            # Option A: Face detected AND meets basic criteria
            # Option B: No face but Gemini confirmed no overlay → still FULL_CAM
            #           (Gemini anchor + tracker will handle this later)
            
            classify_as_fullcam = False
            fullcam_reason = ""
            
            if face and is_centered_x and is_in_frame:
                # Face detected and in frame - FULL_CAM regardless of size
                # (we removed the face_ratio >= 0.015 requirement because angled faces can be small)
                classify_as_fullcam = True
                fullcam_reason = f"Face detected at ({face_center_x},{face_center_y}), size {face_ratio:.1%}"
            elif face:
                # Face detected but maybe at edge - still FULL_CAM, face tracking will handle
                classify_as_fullcam = True
                fullcam_reason = f"Face at edge ({face_center_x},{face_center_y}), will use tracking"
            else:
                # No face detected but Gemini says no overlay
                # This is still FULL_CAM - Gemini anchor will find the person
                classify_as_fullcam = True
                fullcam_reason = "No face by OpenCV, but Gemini confirms no overlay - using Gemini anchor"
            
            if classify_as_fullcam:
                print(f"  ✅ FULL_CAM detected: {fullcam_reason}")
                
                # Return a full-frame WebcamRegion to trigger FULL_CAM layout
                inset = int(min(width, height) * 0.01)
                return WebcamRegion(
                    x=inset,
                    y=inset,
                    width=width - 2 * inset,
                    height=height - 2 * inset,
                    position='full',  # Special marker for FULL_CAM
                    gemini_type='full_cam',
                    gemini_confidence=0.7,
                )
        else:
            # Could not extract frame - still treat as FULL_CAM since Gemini says no overlay
            # Face tracking with Gemini anchor will handle finding the person
            print(f"  ⚠️ Could not extract frame, but Gemini says no overlay → FULL_CAM")
            return WebcamRegion(
                x=0,
                y=0,
                width=1920,  # Will be resized anyway
                height=1080,
                position='full',
                gemini_type='full_cam',
                gemini_confidence=0.5,
            )
    
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


# =============================================================================
# LAYOUT DETECTION WITH CACHING
# =============================================================================

_CACHE_FILENAME = "layout_detection_cache.json"

# Cache version - increment when detection algorithm changes significantly
# This ensures old cached bboxes are invalidated when logic changes
_CACHE_VERSION = 4  # v4: Face-anchored edge refinement for side_box, reduced padding, multi-frame median


def _get_cache_path(temp_dir: str) -> str:
    """Get the path to the cache file in the temp directory."""
    return os.path.join(temp_dir, _CACHE_FILENAME)


def _load_cached_layout(temp_dir: str) -> Optional[LayoutInfo]:
    """
    Load cached layout detection result if available.
    
    Validates cache version to ensure compatibility with current detection logic.
    
    Args:
        temp_dir: Job temp directory
        
    Returns:
        LayoutInfo if cache exists, valid, and version matches, None otherwise
    """
    cache_path = _get_cache_path(temp_dir)
    
    if not os.path.exists(cache_path):
        return None
    
    try:
        with open(cache_path, 'r') as f:
            data = json.load(f)
        
        # Check cache version
        cached_version = data.get('_cache_version', 1)  # Default to v1 for old caches
        if cached_version != _CACHE_VERSION:
            print(f"  ⚠️ Cache version mismatch (cached: v{cached_version}, current: v{_CACHE_VERSION})")
            print(f"     Invalidating cached result - will re-detect with new algorithm")
            # Delete stale cache
            try:
                os.remove(cache_path)
            except:
                pass
            return None
        
        layout_info = LayoutInfo.from_dict(data)
        print(f"  📦 Loaded cached layout (v{_CACHE_VERSION}): {layout_info.layout}")
        return layout_info
    except Exception as e:
        print(f"  ⚠️ Failed to load cache: {e}")
        # Delete corrupted cache
        try:
            os.remove(cache_path)
        except:
            pass
        return None


def _save_layout_cache(temp_dir: str, layout_info: LayoutInfo) -> None:
    """
    Save layout detection result to cache.
    
    Uses atomic write (write to temp file, then rename) to avoid corrupted JSON.
    Converts numpy types to Python types for JSON serialization.
    Includes cache version for compatibility checking.
    
    Args:
        temp_dir: Job temp directory
        layout_info: Detection result to cache
    """
    cache_path = _get_cache_path(temp_dir)
    temp_path = cache_path + ".tmp"
    
    try:
        # Convert to dict and recursively convert numpy types
        data = layout_info.to_dict()
        data = convert_numpy_to_python(data)
        
        # Add cache version
        data['_cache_version'] = _CACHE_VERSION
        
        print(f"  🔄 Saving cache (v{_CACHE_VERSION})...")
        
        # Write to temp file first (atomic)
        with open(temp_path, 'w') as f:
            json.dump(data, f, indent=2)
        
        # Atomic rename
        os.replace(temp_path, cache_path)
        print(f"  💾 Saved layout cache: {cache_path}")
    except Exception as e:
        print(f"  ⚠️ Failed to save cache: {e}")
        # Clean up temp file if it exists
        if os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except:
                pass


def detect_layout_with_cache(
    video_path: str,
    temp_dir: str,
    sample_times: list[float] = [3.0, 10.0, 15.0],
    force_refresh: bool = False,
) -> LayoutInfo:
    """
    Detect video layout with caching support.
    
    This is the main entry point for layout detection. It:
    1. Checks for cached result (for performance when rendering both versions)
    2. Runs webcam detection if no cache
    3. Classifies layout (FULL_CAM, SPLIT, NO_WEBCAM)
    4. Caches the result
    
    Args:
        video_path: Path to video file
        temp_dir: Job temp directory for caching
        sample_times: Frame sampling times
        force_refresh: If True, ignore cache and re-detect
        
    Returns:
        LayoutInfo with layout classification and webcam/face info
    """
    print(f"\n🎯 Detecting layout for: {video_path}")
    
    # Check cache first (unless force refresh)
    if not force_refresh:
        cached = _load_cached_layout(temp_dir)
        if cached:
            return cached
    
    # Get video dimensions
    width, height = get_video_dimensions(video_path)
    
    # Run webcam detection
    webcam_region = detect_webcam_region(video_path, sample_times, temp_dir)
    
    # If no webcam, return early
    if webcam_region is None:
        layout_info = LayoutInfo(
            layout='NO_WEBCAM',
            webcam_region=None,
            face_center=None,
            reason='No webcam detected by Gemini or OpenCV',
            bbox_area_ratio=0.0,
        )
        _save_layout_cache(temp_dir, layout_info)
        print(f"  📹 Layout: NO_WEBCAM")
        return layout_info
    
    # Extract a frame for layout classification
    frame = extract_frame(video_path, 5.0)  # Use 5s mark for classification
    
    if frame is None:
        # Fallback: assume SPLIT if we can't extract frame
        layout_info = LayoutInfo(
            layout='SPLIT',
            webcam_region=webcam_region,
            face_center=None,
            reason='Frame extraction failed, defaulting to SPLIT',
            bbox_area_ratio=0.0,
        )
        _save_layout_cache(temp_dir, layout_info)
        return layout_info
    
    # Create bbox dict for classifier
    webcam_bbox = {
        'x': webcam_region.x,
        'y': webcam_region.y,
        'width': webcam_region.width,
        'height': webcam_region.height,
    }
    
    # Get gemini type and confidence from webcam_region
    gemini_type = getattr(webcam_region, 'gemini_type', 'unknown')
    gemini_confidence = getattr(webcam_region, 'gemini_confidence', 0.0)
    
    # Classify layout
    layout, reason, bbox_area_ratio = classify_layout(
        frame_bgr=frame,
        webcam_bbox=webcam_bbox,
        corner=webcam_region.position,
        video_width=width,
        video_height=height,
        gemini_type=gemini_type,
        gemini_confidence=gemini_confidence,
    )
    
    # Detect face for face-centered cropping (especially for FULL_CAM)
    face_center = None
    seed_detector = None
    seed_score = 0.0
    
    if layout == 'FULL_CAM':
        # For FULL_CAM, use _detect_best_face_for_fullcam which has DNN + Gemini (NO HAAR)
        print(f"  🔍 FULL_CAM: Using _detect_best_face_for_fullcam (DNN + Gemini, NO HAAR for seed)")
        
        # Save frame for Gemini escalation
        frame_path_temp = None
        try:
            import tempfile
            with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as f:
                cv2.imwrite(f.name, frame)
                frame_path_temp = f.name
        except Exception as e:
            print(f"  ⚠️ Could not save temp frame for Gemini: {e}")
        
        best_face, seed_detector, seed_score = _detect_best_face_for_fullcam(frame, frame_path_temp)
        
        # Clean up temp frame
        if frame_path_temp:
            try:
                os.unlink(frame_path_temp)
            except:
                pass
        
        if best_face:
            face_center = FaceCenter(
                x=best_face.x,
                y=best_face.y,
                width=best_face.width,
                height=best_face.height
            )
            y_ratio = face_center.center_y / height
            x_ratio = face_center.center_x / width
            print(f"  👤 FULL_CAM face_center SET: ({face_center.center_x}, {face_center.center_y})")
            print(f"     Detector: {seed_detector}, Score: {seed_score:.2f}, y_ratio: {y_ratio:.2f}, x_ratio: {x_ratio:.2f}")
        else:
            print(f"  ⚠️ FULL_CAM: No face found, face tracking will use Gemini anchor")
        
        # For FULL_CAM, set webcam_region to full frame (or slightly inset)
        # This ensures the render uses the full frame, not just the detected bbox
        inset = int(min(width, height) * 0.02)  # 2% inset to avoid edge artifacts
        webcam_region = WebcamRegion(
            x=inset,
            y=inset,
            width=width - 2 * inset,
            height=height - 2 * inset,
            position='full',
            gemini_type='full_cam',
            gemini_confidence=gemini_confidence,
        )
        print(f"  📐 FULL_CAM effective region: {webcam_region}")
    else:
        # For SPLIT, detect face in webcam region for validation
        face_center = detect_face_in_frame(
            frame,
            webcam_region.x,
            webcam_region.y,
            webcam_region.width,
            webcam_region.height,
        )
    
    # Build result
    layout_info = LayoutInfo(
        layout=layout,
        webcam_region=webcam_region,
        face_center=face_center,
        reason=reason,
        bbox_area_ratio=bbox_area_ratio,
        gemini_type=gemini_type,
        confidence=gemini_confidence,
    )
    
    # Cache result
    _save_layout_cache(temp_dir, layout_info)
    
    # Log summary
    print(f"\n  ═══════════════════════════════════════")
    print(f"  📊 LAYOUT DETECTION SUMMARY")
    print(f"  ═══════════════════════════════════════")
    print(f"  Layout: {layout}")
    print(f"  Gemini type: {gemini_type}, confidence: {gemini_confidence:.2f}")
    print(f"  Reason: {reason}")
    print(f"  bbox_area_ratio: {bbox_area_ratio:.2%}")
    print(f"  Webcam region: {webcam_region}")
    print(f"  Face center: {face_center}")
    if layout == 'FULL_CAM' and face_center:
        print(f"  Face-centered crop: YES (face at {face_center.center_x}, {face_center.center_y})")
    if layout == 'TOP_BAND':
        print(f"  TOP_BAND detected - source stacked layout, will use simple vertical crop")
    print(f"  ═══════════════════════════════════════\n")
    
    return layout_info


if __name__ == "__main__":
    # Test with a video file
    import sys
    if len(sys.argv) > 1:
        video_path = sys.argv[1]
        result = detect_webcam_region(video_path)
        print(f"\nResult: {result}")

